"""
data_module.py - 数据管理模块 (增强版 v3.1)
修复 Tushare API 限流问题 + 保留所有原有功能 + 修复列索引错误 + 过滤ST股票 + 加入聪明动量因子

主要改进:
✅ 修复 KeyError: "['position', 'amount'] not in index"
✅ get_price_data 返回 amount 列
✅ load_data_from_tushare 正确分离价格列和因子列
✅ 智能限流控制 (自适应等待)
✅ 批量请求优化 (减少API调用次数)
✅ 新增: 自动过滤 ST/S*ST/*ST 股票
✅ 新增: 聪明动量因子 (smart_mom_20d) - 收益率/波动率，更稳健的动量指标
"""

import pandas as pd
import numpy as np
import os
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from collections import deque

# Tushare导入
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("⚠️  Tushare未安装: pip install tushare")

# 导入资金流因子计算器
from money_flow_factors import MoneyFlowFactorCalculator, integrate_money_flow_to_stockranker


# ========== 第1部分：基础工具类 ==========

class DataCache:
    """数据缓存管理类"""

    def __init__(self, cache_dir='./data_cache'):
        """初始化缓存管理器"""
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            print(f"✓ 创建缓存目录: {cache_dir}")

    def _get_cache_key(self, prefix, start_date, end_date, **kwargs):
        """生成缓存key"""
        key_str = f"{prefix}_{start_date}_{end_date}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def save_to_cache(self, data, cache_name):
        """保存数据到缓存 (Pickle)"""
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"✗ 缓存保存失败: {e}")
            return False

    def load_from_cache(self, cache_name):
        """从缓存加载数据 (Pickle)"""
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                return data
            except Exception as e:
                print(f"✗ 缓存加载失败: {e}")
        return None

    def save_to_csv(self, df, filename):
        """保存DataFrame到CSV"""
        csv_path = os.path.join(self.cache_dir, f"{filename}.csv")
        try:
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"✓ CSV已保存: {filename}")
            return True
        except Exception as e:
            print(f"✗ CSV保存失败: {e}")
            return False

    def load_from_csv(self, filename):
        """从CSV加载DataFrame"""
        csv_path = os.path.join(self.cache_dir, f"{filename}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, encoding='utf-8-sig')
                print(f"✓ 从CSV加载: {filename}")
                return df
            except Exception as e:
                print(f"✗ CSV加载失败: {e}")
        return None

    def list_cache_files(self):
        """列出所有缓存文件"""
        if not os.path.exists(self.cache_dir):
            return []
        files = []
        for f in os.listdir(self.cache_dir):
            if f.endswith('.pkl') or f.endswith('.csv'):
                file_path = os.path.join(self.cache_dir, f)
                file_size = os.path.getsize(file_path) / 1024  # KB
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                files.append({
                    'name': f,
                    'size_kb': f"{file_size:.2f}",
                    'modified': file_time.strftime('%Y-%m-%d %H:%M:%S')
                })
        return files

    def clear_cache(self):
        """清空缓存"""
        if os.path.exists(self.cache_dir):
            count = 0
            for f in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, f)
                try:
                    os.remove(file_path)
                    count += 1
                except Exception as e:
                    print(f"✗ 删除文件失败 {f}: {e}")
            print(f"✓ 已清空缓存 ({count}个文件)")


class RateLimiter:
    """访问频率控制器 - 每分钟800次访问限制后暂停等待"""

    def __init__(self, max_calls=800, time_window=60):
        """
        初始化限流器
        Args:
            max_calls: 时间窗口内最大调用次数 (默认800/分钟)
            time_window: 时间窗口(秒)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.call_times = deque()  # 记录调用时间戳
        self.total_calls = 0
        self.total_waits = 0

    def wait_if_needed(self):
        """等待直到可以继续调用API - 确保不超过频率限制"""
        now = time.time()

        # 移除时间窗口外的记录
        while self.call_times and now - self.call_times[0] > self.time_window:
            self.call_times.popleft()

        # 如果达到限制，等待到最早的调用超出时间窗口
        while len(self.call_times) >= self.max_calls:
            sleep_time = self.time_window - (now - self.call_times[0]) + 0.1
            if sleep_time > 0:
                self.total_waits += 1
                print(f"⏳ 触发访问限制，等待 {sleep_time:.1f} 秒...")
                time.sleep(sleep_time)
                now = time.time()
                # 清理过期记录
                while self.call_times and now - self.call_times[0] > self.time_window:
                    self.call_times.popleft()
            else:
                break

        # 记录本次调用
        self.call_times.append(now)
        self.total_calls += 1

        # 基础延迟(避免瞬时高峰)
        time.sleep(0.05)

    def get_stats(self):
        """获取统计信息"""
        return {
            'total_calls': self.total_calls,
            'total_waits': self.total_waits,
            'current_window_calls': len(self.call_times)
        }


# ========== 第2部分：Tushare数据源类 ==========

class TushareDataSource:
    """Tushare数据源管理类 - 优化限流版本"""

    def __init__(self, cache_manager=None, token=None, rate_limiter=None):
        """初始化Tushare数据源"""
        self.cache = cache_manager

        if not TUSHARE_AVAILABLE:
            raise ImportError("请先安装Tushare: pip install tushare")

        if token and TUSHARE_AVAILABLE:
            import tushare as ts_module
            ts_module.set_token(token)

        try:
            if TUSHARE_AVAILABLE:
                import tushare as ts_module
                self.pro = ts_module.pro_api()
                print("✓ Tushare API初始化成功")
            else:
                self.pro = None
                print("⚠️ Tushare未安装")
        except Exception as e:
            print(f"✗ Tushare初始化失败: {e}")
            print("请设置token: ts.set_token('你的token')")
            self.pro = None

        # 初始化限流器
        self.rate_limiter = rate_limiter or RateLimiter(max_calls=800, time_window=60)
        print(f"✓ 限流器已启用: {self.rate_limiter.max_calls}次/分钟")

    def get_stock_list(self, date=None, min_days_listed=180):
        """
        获取股票列表 (修复版 - 增加上市日期过滤和ST过滤)
        """
        if self.pro is None:
            return []

        try:
            print("使用Tushare获取股票列表...")
            self.rate_limiter.wait_if_needed()

            # 确保 fields 中包含 'name' 以便过滤 ST
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )

            # ========== 关键修复 Issue A: 过滤上市日期 ==========
            if date:
                backtest_start = pd.to_datetime(date)
                latest_list_date = backtest_start - timedelta(days=min_days_listed)
                df['list_date'] = pd.to_datetime(df['list_date'], format='%Y%m%d', errors='coerce')

                original_count = len(df)
                df = df[df['list_date'] <= latest_list_date].copy()
                filtered_count = original_count - len(df)

                print(f"  📅 上市日期过滤: 回测开始 {date}, 过滤新股 {filtered_count} 只")

            # ========== 关键修复: 过滤 ST 股票 ==========
            if 'name' in df.columns:
                original_count = len(df)
                df = df[~df['name'].astype(str).str.contains('ST', case=False, na=False)].copy()
                st_filtered = original_count - len(df)
                print(f"  🗑️ ST股票过滤: 剔除 {st_filtered} 只风险警示股")

            # 过滤特殊板块
            original_count = len(df)
            df = df[~df['symbol'].astype(str).str.startswith(('688', '300', '8', '4', '92'))].copy()
            special_filtered = original_count - len(df)

            if special_filtered > 0:
                print(f"  🚫 特殊板块过滤: {special_filtered} 只 (科创板/创业板/北交所)")

            stock_codes = df['ts_code'].tolist()
            print(f"✓ 最终获取 {len(stock_codes)} 只符合条件的股票")

            return stock_codes

        except Exception as e:
            print(f"✗ 获取股票列表失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_price_data(self, ts_code, start_date, end_date, list_date=None, max_retries=3):
        """获取单只股票的日线数据 (带限流和重试)"""
        if self.pro is None:
            return None

        cache_name = f"price_{ts_code}_v2.5_{start_date.replace('-', '')}_{end_date.replace('-', '')}"

        if self.cache:
            cached_data = self.cache.load_from_cache(cache_name)
            if cached_data is not None:
                return cached_data

        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                df = self.pro.daily(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    fields='trade_date,open,high,low,close,vol,amount'
                )

                if df is None or len(df) == 0:
                    return None

                df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df['instrument'] = ts_code
                df = df.sort_values('date').reset_index(drop=True)

                if list_date is not None:
                    list_date_dt = pd.to_datetime(list_date, format='%Y%m%d', errors='coerce')
                    if pd.notna(list_date_dt):
                        df = df[df['date'] >= list_date_dt].copy()

                # ✅ 修复: 添加 amount 列
                result = df[['date', 'instrument', 'open', 'close', 'high', 'low', 'volume', 'amount']]

                if self.cache:
                    self.cache.save_to_cache(result, cache_name)

                return result

            except Exception as e:
                error_msg = str(e)
                if "每分钟最多访问" in error_msg or "抱歉" in error_msg:
                    wait_time = 5 * (attempt + 1)
                    print(f"    ⏳ {ts_code}: 触发限流，等待{wait_time}秒后重试... ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"    ✗ {ts_code}: {e}")
                    break

        return None

    def get_index_daily(self, ts_code='000001.SH', start_date=None, end_date=None):
        """获取指数日线数据 (用于择时)"""
        if self.pro is None:
            return None

        cache_name = f"index_{ts_code}_{start_date}_{end_date}"
        if self.cache:
            cached_data = self.cache.load_from_cache(cache_name)
            if cached_data is not None:
                return cached_data

        try:
            print(f"  📊 获取指数数据: {ts_code}...")
            self.rate_limiter.wait_if_needed()

            df = self.pro.index_daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', '') if start_date else None,
                end_date=end_date.replace('-', '') if end_date else None,
                fields='trade_date,close,open,high,low,vol'
            )

            if df is None or len(df) == 0:
                return None

            df = df.rename(columns={'trade_date': 'date', 'vol': 'volume'})
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values('date').reset_index(drop=True)
            df['date'] = df['date'].astype(str)

            if self.cache:
                self.cache.save_to_cache(df, cache_name)

            return df

        except Exception as e:
            print(f"  ⚠️  获取指数数据失败: {e}")
            return None

    def get_daily_basic(self, ts_code, start_date, end_date):
        """获取每日指标数据(PE/PB/PS等)"""
        if self.pro is None:
            return None

        try:
            self.rate_limiter.wait_if_needed()
            df = self.pro.daily_basic(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                fields='trade_date,pe,pb,ps,total_mv'
            )

            if df is None or len(df) == 0:
                return None

            df = df.rename(columns={'trade_date': 'date'})
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['instrument'] = ts_code
            return df

        except Exception as e:
            print(f"✗ 获取 {ts_code} 基本面数据失败: {e}")
            return None

    def get_financial_indicators(self, ts_code, start_date, end_date, max_retries=3):
        """获取财务指标数据(ROE, ROA, 毛利率, 净利率, 资产负债率)"""
        if self.pro is None:
            return None

        cache_name = f"financial_{ts_code}_{start_date}_{end_date}"
        if self.cache:
            cached_data = self.cache.load_from_cache(cache_name)
            if cached_data is not None:
                return cached_data

        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait_if_needed()
                df = self.pro.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date.replace('-', ''),
                    end_date=end_date.replace('-', ''),
                    fields='ts_code,ann_date,end_date,roe,roa,grossprofit_margin,netprofit_margin,debt_to_assets'
                )

                if df is None or len(df) == 0:
                    return None

                df = df.rename(columns={
                    'ann_date': 'date',
                    'grossprofit_margin': 'gross_margin',
                    'netprofit_margin': 'net_margin',
                    'debt_to_assets': 'debt_ratio'
                })

                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df['instrument'] = ts_code
                df = df.sort_values('date').reset_index(drop=True)

                if self.cache:
                    self.cache.save_to_cache(df, cache_name)
                return df

            except Exception as e:
                error_msg = str(e)
                if "每分钟最多访问" in error_msg or "抱歉" in error_msg:
                    wait_time = 5 * (attempt + 1)
                    print(f"    ⏳ {ts_code}: 触发限流，等待{wait_time}秒... ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"✗ 获取 {ts_code} 财务指标失败: {e}")
                    break
        return None

    def get_industry_data(self, instruments, use_cache=True):
        """获取股票行业数据"""
        if self.pro is None:
            return pd.DataFrame({'instrument': instruments, 'industry': 'Unknown'})

        cache_name = "industry_data_all_v2.5"
        if use_cache and self.cache:
            cached_data = self.cache.load_from_cache(cache_name)
            if cached_data is not None:
                cached_data = cached_data[cached_data['instrument'].isin(instruments)]
                if len(cached_data) > 0:
                    print(f"  ✓ 从缓存加载行业数据")
                    return cached_data

        try:
            print(f"  📊 获取 {len(instruments)} 只股票的行业数据...")
            self.rate_limiter.wait_if_needed()

            stock_basic = self.pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')

            if use_cache and self.cache:
                stock_basic_cache = stock_basic.rename(columns={'ts_code': 'instrument'})
                self.cache.save_to_cache(stock_basic_cache[['instrument', 'industry']], cache_name)

            stock_basic = stock_basic[stock_basic['ts_code'].isin(instruments)]
            stock_basic = stock_basic.rename(columns={'ts_code': 'instrument'}).copy()
            stock_basic['industry'] = stock_basic['industry'].fillna('其他')
            result = stock_basic[['instrument', 'industry']]

            missing = set(instruments) - set(result['instrument'])
            if missing:
                missing_df = pd.DataFrame({'instrument': list(missing), 'industry': '其他'})
                result = pd.concat([result, missing_df], ignore_index=True)

            print(f"  ✓ 行业数据获取完成, 行业数: {result['industry'].nunique()}个")
            return result

        except Exception as e:
            print(f"  ⚠️  获取行业数据失败: {e}")
            return pd.DataFrame({'instrument': instruments, 'industry': 'Unknown'})

    def merge_financial_data_to_daily(self, price_df, financial_df):
        """将季度财务数据合并到日线数据 (Merge Asof)"""
        if financial_df is None or len(financial_df) == 0:
            print("  ⚠️  财务数据为空，跳过合并")
            return price_df

        price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
        financial_df['date'] = pd.to_datetime(financial_df['date'], errors='coerce')

        financial_df = financial_df.dropna(subset=['date', 'instrument'])
        price_df = price_df.dropna(subset=['date', 'instrument'])

        if len(financial_df) == 0:
            return price_df

        result_list = []
        success_count = 0

        print("  🔗 合并财务数据到日线...")
        instruments = price_df['instrument'].unique()

        for idx, instrument in enumerate(instruments):
            if (idx + 1) % 500 == 0:
                print(f"     进度: {idx + 1}/{len(instruments)}")

            price_subset = price_df[price_df['instrument'] == instrument].copy()
            financial_subset = financial_df[financial_df['instrument'] == instrument].copy()

            if len(financial_subset) == 0:
                result_list.append(price_subset)
                continue

            financial_subset = financial_subset.dropna(subset=['date'])

            try:
                merged = pd.merge_asof(
                    price_subset.sort_values('date'),
                    financial_subset.sort_values('date')[
                        ['date', 'roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']
                    ],
                    on='date',
                    direction='backward'
                )
                result_list.append(merged)
                success_count += 1
            except Exception:
                result_list.append(price_subset)

        if len(result_list) == 0:
            return price_df

        result_df = pd.concat(result_list, ignore_index=True)
        print(f"  ✓ 合并完成: 成功 {success_count} 只")
        return result_df

    def print_rate_limit_stats(self):
        """打印限流统计信息"""
        stats = self.rate_limiter.get_stats()
        print(f"\n📊 API调用统计:")
        print(f"  - 总调用次数: {stats['total_calls']}")
        print(f"  - 触发限流次数: {stats['total_waits']}")


# ========== 第3部分：StockRanker 多因子评分模型 ==========

class StockRankerModel:
    """StockRanker 多因子评分模型 (内存优化版)"""

    def __init__(self, custom_weights=None, use_fundamental=True, use_money_flow=True, money_flow_style='balanced'):
        self.use_fundamental = use_fundamental
        self.use_money_flow = use_money_flow
        self.money_flow_style = money_flow_style
        
        # 初始化资金流计算器
        if self.use_money_flow:
            from money_flow_factors import MoneyFlowFactorCalculator
            self.money_flow_calculator = MoneyFlowFactorCalculator(
                use_full_tick_data=False,
                keep_only_essential=True  # ✅ 关键：仅保留核心因子
            )
            
            # 获取推荐的资金流因子权重
            money_flow_weights = self.money_flow_calculator.get_recommended_weights(money_flow_style)
        else:
            money_flow_weights = {}
        
        if custom_weights:
            self.factor_weights = custom_weights
        else:
            # 基础因子权重（根据是否启用资金流调整）
            base_weights = {}
            
            if use_fundamental and use_money_flow:
                # 基本面 + 资金流模式（推荐）
                base_weights = {
                    # 估值因子（权重从25%降到15%）
                    'pe_ratio': -0.06, 'pb_ratio': -0.06, 'ps_ratio': -0.03,
                    
                    # 波动率（权重从15%降到10%）
                    'volatility_20d': -0.05, 'volatility_60d': -0.05,
                    
                    # 成交量（权重从15%降到10%）
                    'money_flow_20d': 0.05, 'volume_ratio': 0.05,
                    
                    # 动量（权重从15%降到12%）
                    'return_20d': 0.06, 'return_60d': 0.06,
                    # 聪明动量（高权重）
                    'smart_mom_20d': 0.15,
                    
                    # 基本面（权重从30%降到25%）
                    'roe': 0.08, 'roa': 0.04,
                    'gross_margin': 0.04, 'net_margin': 0.04,
                    'debt_ratio': -0.05,
                }
                # 资金流权重（28%，从money_flow_weights获取）
                base_weights.update(money_flow_weights)
                
            elif use_fundamental:
                # 仅基本面模式（原有权重）
                base_weights = {
                    'pe_ratio': -0.10, 'pb_ratio': -0.10, 'ps_ratio': -0.05,
                    'volatility_20d': -0.08, 'volatility_60d': -0.07,
                    'money_flow_20d': 0.08, 'volume_ratio': 0.07,
                    'return_20d': 0.08, 'return_60d': 0.07,
                    'smart_mom_20d': 0.15,  # 聪明动量高权重
                    'roe': 0.10, 'roa': 0.05,
                    'gross_margin': 0.05, 'net_margin': 0.05,
                    'debt_ratio': -0.05
                }
                
            elif use_money_flow:
                # 技术 + 资金流模式
                base_weights = {
                    'pe_ratio': -0.10, 'pb_ratio': -0.10, 'ps_ratio': -0.08,
                    'volatility_20d': -0.08, 'volatility_60d': -0.07,
                    'money_flow_20d': 0.06, 'volume_ratio': 0.06,
                    'return_20d': 0.08, 'return_60d': 0.07,
                    'smart_mom_20d': 0.15,  # 聪明动量高权重
                }
                base_weights.update(money_flow_weights)
                
            else:
                # 仅技术因子模式（原有权重）
                base_weights = {
                    'pe_ratio': -0.15, 'pb_ratio': -0.15, 'ps_ratio': -0.10,
                    'volatility_20d': -0.10, 'volatility_60d': -0.10,
                    'money_flow_20d': 0.10, 'volume_ratio': 0.10,
                    'return_20d': 0.10, 'return_60d': 0.10,
                    'smart_mom_20d': 0.15  # 聪明动量高权重
                }
            
            self.factor_weights = base_weights

        print(f"\n📊 StockRanker 模型初始化")
        print(f"   基本面: {'✓' if use_fundamental else '✗'}")
        print(f"   资金流: {'✓' if use_money_flow else '✗'}")
        if use_money_flow:
            print(f"   资金流风格: {money_flow_style}")
            print(f"   因子数量: {len(self.factor_weights)} 个")

    def calculate_valuation_factors(self, df):
        df['pe_ratio'] = df['close'] / df.groupby('instrument')['close'].transform('mean')
        df['pb_ratio'] = df['close'] / (df.groupby('instrument')['close'].transform('mean') * 0.8)
        df['ps_ratio'] = df['close'] / (df.groupby('instrument')['close'].transform('mean') * 1.2)
        return df

    def calculate_volatility_factors(self, df):
        df['volatility_20d'] = df.groupby('instrument')['close'].rolling(20).std().reset_index(0, drop=True)
        df['volatility_60d'] = df.groupby('instrument')['close'].rolling(60).std().reset_index(0, drop=True)
        return df

    def calculate_money_flow_factors(self, df):
        df['money_flow_20d'] = (df['volume'] * df['close']).rolling(20).mean()
        df['volume_ma5'] = df.groupby('instrument')['volume'].rolling(5).mean().reset_index(0, drop=True)
        df['volume_ma20'] = df.groupby('instrument')['volume'].rolling(20).mean().reset_index(0, drop=True)
        df['volume_ratio'] = df['volume_ma5'] / (df['volume_ma20'] + 1e-6)
        return df

    def calculate_momentum_factors(self, df):
        df['return_20d'] = df.groupby('instrument')['close'].pct_change(20)
        df['return_60d'] = df.groupby('instrument')['close'].pct_change(60)
        
        # ✅ 聪明动量因子 (Sharpe-like Momentum): 收益/波动率，惩罚剧烈波动的上涨
        volatility_20d = df.groupby('instrument')['close'].rolling(20).std().reset_index(0, drop=True)
        df['smart_mom_20d'] = df['return_20d'] / (volatility_20d + 1e-6)
        return df

    def process_fundamental_factors(self, df):
        if not self.use_fundamental: return df
        fundamental_cols = ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']
        for col in fundamental_cols:
            if col in df.columns:
                median_val = df.groupby('instrument')[col].transform('median')
                df[col] = df[col].fillna(median_val)
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)
        return df

    def calculate_all_factors(self, price_data):
        print("\n⚙️  计算StockRanker多因子...")
        df = price_data.copy()
        
        # 原有因子计算
        df = self.calculate_valuation_factors(df)
        df = self.calculate_volatility_factors(df)
        df = self.calculate_money_flow_factors(df)
        df = self.calculate_momentum_factors(df)
        
        if self.use_fundamental:
            df = self.process_fundamental_factors(df)
        
        # ✅ 资金流因子计算（内存优化）
        if self.use_money_flow:
            print("\n💰 计算资金流因子...")
            df = self.money_flow_calculator.calculate_simplified_money_flow(df)
            
            # 打印摘要（包含内存占用）
            self.money_flow_calculator.print_factor_summary(df)
        
        return df

    def normalize_factors(self, df):
        for factor in self.factor_weights.keys():
            if factor in df.columns:
                df[f'{factor}_norm'] = df.groupby('date')[factor].rank(pct=True)
        return df

    def calculate_position_score(self, df):
        print("\n📊 计算综合评分...")
        
        # ✅ 关键优化：避免一次性标准化所有因子
        # 分批标准化，立即计算贡献
        
        df['position'] = 0.0
        
        for factor, weight in self.factor_weights.items():
            if factor in df.columns:
                # 直接标准化并累加，不保留 _norm 列
                factor_rank = df.groupby('date')[factor].rank(pct=True).fillna(0.5)
                df['position'] += factor_rank * weight
                
                # 立即删除临时变量
                del factor_rank
        
        # 归一化到0-1
        min_score = df.groupby('date')['position'].transform('min')
        max_score = df.groupby('date')['position'].transform('max')
        df['position'] = (df['position'] - min_score) / (max_score - min_score + 1e-6)
        
        # 清理
        del min_score, max_score
        
        print("✓ 评分计算完成")
        return df


# ========== 第4部分：简单因子计算函数 ==========

def calculate_simple_factors(price_data):
    """计算简单技术因子(兼容旧版本)"""
    df = price_data.copy()

    # 动量因子
    df['return_5d'] = df.groupby('instrument')['close'].pct_change(5)
    df['return_20d'] = df.groupby('instrument')['close'].pct_change(20)

    # 波动率与成交量
    df['volatility_20d'] = df.groupby('instrument')['close'].rolling(20).std().reset_index(0, drop=True)
    df['volume_ma5'] = df.groupby('instrument')['volume'].rolling(5).mean().reset_index(0, drop=True)
    df['volume_ma20'] = df.groupby('instrument')['volume'].rolling(20).mean().reset_index(0, drop=True)
    df['volume_ratio'] = df['volume_ma5'] / (df['volume_ma20'] + 1e-6)

    # RSI
    delta = df.groupby('instrument')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 简单评分
    for col in ['return_20d', 'volume_ratio', 'rsi']:
        if col in df.columns:
            df[f'{col}_norm'] = df.groupby('date')[col].rank(pct=True)

    df['position'] = 0
    weights = {'return_20d_norm': 0.4, 'volume_ratio_norm': 0.3, 'rsi_norm': 0.3}
    for factor, weight in weights.items():
        if factor in df.columns:
            df['position'] += df[factor].fillna(0.5) * weight

    return df


# ========== 第5部分：主数据加载函数 ==========

def load_data_from_tushare(
    start_date, end_date, max_stocks=50, use_cache=True,
    cache_manager=None, use_stockranker=True,
    custom_weights=None, tushare_token=None,
    use_fundamental=True, min_days_listed=180,
    use_money_flow=True, money_flow_style='balanced'  # ✅ 新增参数
):
    """
    从Tushare加载数据并计算因子 (内存优化版 v2.6)
    
    新增参数:
        use_money_flow: 是否启用资金流因子
        money_flow_style: 'conservative' | 'balanced' | 'aggressive'
    """
    
    print("\n" + "=" * 80)
    print("📦 数据加载模块 (内存优化版 v2.6)")
    print("=" * 80)

    # 生成缓存Key
    model_suffix = "stockranker" if use_stockranker else "simple"
    if use_fundamental: model_suffix += "_fundamental"
    if use_money_flow: model_suffix += "_moneyflow"  # ✅ 添加资金流标识
    cache_key = f"factor_data_ts_v2.6_{start_date}_{end_date}_{max_stocks}_{model_suffix}_{min_days_listed}"
    price_cache_key = f"price_data_ts_v2.6_{start_date}_{end_date}_{max_stocks}_{min_days_listed}"

    # 1. 尝试从缓存加载
    if use_cache and cache_manager:
        print("\n🔍 检查缓存...")
        factor_data = cache_manager.load_from_csv(cache_key)
        price_data = cache_manager.load_from_csv(price_cache_key)
        if factor_data is not None and price_data is not None:
            print("✓ 使用缓存数据")
            return factor_data, price_data

    # 2. 初始化数据源
    rate_limiter = RateLimiter(max_calls=800, time_window=60)
    data_source = TushareDataSource(
        cache_manager=cache_manager if use_cache else None,
        token=tushare_token,
        rate_limiter=rate_limiter
    )

    # 3. 获取股票列表
    stock_list = data_source.get_stock_list(date=start_date, min_days_listed=min_days_listed)
    if not stock_list: return None, None
    stock_list = stock_list[:max_stocks]

    # 获取股票上市信息
    if data_source.pro:
        stock_info_df = data_source.pro.stock_basic(exchange='', list_status='L', fields='ts_code,list_date')
    else:
        stock_info_df = pd.DataFrame(columns=['ts_code', 'list_date'])
    stock_info_dict = dict(zip(stock_info_df['ts_code'], stock_info_df['list_date']))

    # 4. 获取价格数据
    all_price_data = []
    print(f"\n📊 获取 {len(stock_list)} 只股票的历史数据...")

    start_time = time.time()
    for i, ts_code in enumerate(stock_list):
        if (i + 1) % 10 == 0: print(f"  进度: {i + 1}/{len(stock_list)}")

        list_date = stock_info_dict.get(ts_code)
        df = data_source.get_price_data(ts_code, start_date, end_date, list_date=list_date)
        if df is not None: all_price_data.append(df)

    if not all_price_data: return None, None
    price_df = pd.concat(all_price_data, ignore_index=True)

    # 5. 获取并合并基本面数据
    if use_stockranker and use_fundamental:
        print(f"\n📈 获取基本面财务数据...")
        all_financial_data = []
        for i, ts_code in enumerate(stock_list):
            if (i + 1) % 10 == 0: print(f"  进度: {i + 1}/{len(stock_list)}")
            f_df = data_source.get_financial_indicators(ts_code, start_date, end_date)
            if f_df is not None: all_financial_data.append(f_df)

        if all_financial_data:
            financial_df = pd.concat(all_financial_data, ignore_index=True)
            price_df = data_source.merge_financial_data_to_daily(price_df, financial_df)
        else:
            print("⚠️  未获取到基本面数据,将不使用基本面因子")
            use_fundamental = False

    price_df['date'] = price_df['date'].astype(str)

    # 6. 计算因子
    if use_stockranker:
        model = StockRankerModel(
            custom_weights=custom_weights,
            use_fundamental=use_fundamental,
            use_money_flow=use_money_flow,        # ✅ 传入参数
            money_flow_style=money_flow_style     # ✅ 传入参数
        )
        factor_df = model.calculate_all_factors(price_df)
        factor_df = model.calculate_position_score(factor_df)
    else:
        print("\n⚙️  计算简单技术因子...")
        factor_df = calculate_simple_factors(price_df)

    factor_df = factor_df.dropna(subset=['position'])

    # 7. 整理输出列
    essential_columns = ['date', 'instrument', 'position']
    price_only_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']

    # 自动识别所有因子列 (排除必须列和价格列)
    all_columns = factor_df.columns.tolist()
    factor_columns = [col for col in all_columns if col not in essential_columns + price_only_columns]

    result_factor = factor_df[essential_columns + factor_columns].copy()

    # ✅ 修复: 价格数据不包含 position 列
    # price_df 包含 'amount' (由 get_price_data 修复提供) 但不包含 'position'
    price_columns_to_keep = ['date', 'instrument'] + price_only_columns

    if use_fundamental:
        for col in ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']:
            if col in price_df.columns: price_columns_to_keep.append(col)

    # 过滤掉 price_df 中不存在的列
    price_columns_to_keep = [col for col in price_columns_to_keep if col in price_df.columns]

    result_price = price_df[price_columns_to_keep].copy()

    # 8. 获取并合并行业数据
    print("\n📊 获取行业数据...")
    industry_data = data_source.get_industry_data(stock_list, use_cache=use_cache)
    if industry_data is not None and not industry_data.empty:
        result_factor = result_factor.merge(industry_data, on='instrument', how='left')
        result_factor['industry'] = result_factor['industry'].fillna('其他')
    else:
        result_factor['industry'] = 'Unknown'

    # 9. 保存缓存
    if use_cache and cache_manager:
        print("\n💾 保存到缓存...")
        cache_manager.save_to_csv(result_factor, cache_key)
        cache_manager.save_to_csv(result_price, price_cache_key)

    data_source.print_rate_limit_stats()
    print("✓ 数据准备完成")
    return result_factor, result_price


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("数据模块加载完成。请在主程序中导入使用。")