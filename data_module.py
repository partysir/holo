"""
data_module.py - 数据管理模块 (Tushare版 + 基本面因子)
新增功能: ROE, ROA, 毛利率, 净利率, 资产负债率

data_module.py - 数据管理模块 (Tushare版)
负责: 数据缓存、数据获取、StockRanker多因子计算

使用前准备:
1. 注册Tushare账号: https://tushare.pro/register
2. 获取token: https://tushare.pro/user/token
3. 安装: pip install tushare pandas numpy
4. 设置token: 在main.py中添加 ts.set_token('你的token')
"""

import pandas as pd
import numpy as np
import os
import pickle
import hashlib
import time
from datetime import datetime

# Tushare导入
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("⚠️  Tushare未安装: pip install tushare")


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
        """保存数据到缓存"""
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ 数据已缓存: {cache_name}")
            return True
        except Exception as e:
            print(f"✗ 缓存保存失败: {e}")
            return False

    def load_from_cache(self, cache_name):
        """从缓存加载数据"""
        cache_path = os.path.join(self.cache_dir, f"{cache_name}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"✓ 从缓存加载: {cache_name}")
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

class TushareDataSource:
    """Tushare数据源管理类 - 扩展基本面数据"""

    def __init__(self, cache_manager=None, token=None):
        """初始化Tushare数据源"""
        self.cache = cache_manager

        if not TUSHARE_AVAILABLE:
            raise ImportError("请先安装Tushare: pip install tushare")

        if token:
            ts.set_token(token)

        try:
            self.pro = ts.pro_api()
            print("✓ Tushare API初始化成功")
        except Exception as e:
            print(f"✗ Tushare初始化失败: {e}")
            print("请设置token: ts.set_token('你的token')")
            self.pro = None

    def get_stock_list(self, date=None):
        """获取股票列表"""
        if self.pro is None:
            return []

        try:
            print("使用Tushare获取股票列表...")
            df = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market'
            )

            # 过滤科创板(688)、创业板(300)、北交所(8/4开头)
            df = df[~df['symbol'].str.startswith(('688', '300', '8', '4'))]

            stock_codes = df['ts_code'].tolist()
            print(f"✓ 获取到 {len(stock_codes)} 只股票")

            return stock_codes

        except Exception as e:
            print(f"✗ 获取股票列表失败: {e}")
            return []

    def get_price_data(self, ts_code, start_date, end_date):
        """获取单只股票的日线数据(带缓存)"""
        if self.pro is None:
            return None

        cache_name = f"price_{ts_code}_{start_date}_{end_date}"
        if self.cache:
            cached_data = self.cache.load_from_cache(cache_name)
            if cached_data is not None:
                return cached_data

        try:
            df = self.pro.daily(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                fields='trade_date,open,high,low,close,vol,amount'
            )

            if df is None or len(df) == 0:
                return None

            df = df.rename(columns={
                'trade_date': 'date',
                'vol': 'volume'
            })

            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['instrument'] = ts_code
            df = df.sort_values('date').reset_index(drop=True)

            result = df[['date', 'instrument', 'open', 'close', 'high', 'low', 'volume']]

            if self.cache:
                self.cache.save_to_cache(result, cache_name)

            time.sleep(0.31)
            return result

        except Exception as e:
            print(f"✗ 获取 {ts_code} 数据失败: {e}")
            time.sleep(1)
            return None

    def get_daily_basic(self, ts_code, start_date, end_date):
        """获取每日指标数据(PE/PB/PS等)"""
        if self.pro is None:
            return None

        try:
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

            time.sleep(0.31)
            return df

        except Exception as e:
            print(f"✗ 获取 {ts_code} 基本面数据失败: {e}")
            return None

    # ========== 新增: 行业数据获取 ==========

    def get_industry_data(self, instruments, use_cache=True):
        """
        获取股票行业数据（新增方法 - 使用 stock_basic）

        Args:
            instruments: 股票代码列表
            use_cache: 是否使用缓存

        Returns:
            DataFrame: [instrument, industry]
        """
        if self.pro is None:
            return pd.DataFrame({
                'instrument': instruments,
                'industry': 'Unknown'
            })

        # 检查缓存
        cache_name = f"industry_data_all"
        if use_cache and self.cache:
            cached_data = self.cache.load_from_cache(cache_name)
            if cached_data is not None:
                cached_data = cached_data[cached_data['instrument'].isin(instruments)]
                if len(cached_data) > 0:
                    print(f"  ✓ 从缓存加载行业数据")
                    return cached_data

        try:
            print(f"  📊 获取 {len(instruments)} 只股票的行业数据...")

            # ✅ 使用 stock_basic 获取申万行业（一次调用获取所有）
            stock_basic = self.pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,name,industry'
            )

            # 保存完整数据到缓存
            if use_cache and self.cache:
                stock_basic_cache = stock_basic.rename(columns={'ts_code': 'instrument'})
                self.cache.save_to_cache(stock_basic_cache[['instrument', 'industry']], cache_name)

            # 过滤目标股票
            stock_basic = stock_basic[stock_basic['ts_code'].isin(instruments)]
            stock_basic = stock_basic.rename(columns={'ts_code': 'instrument'})
            stock_basic['industry'] = stock_basic['industry'].fillna('其他')

            result = stock_basic[['instrument', 'industry']]

            # 补充未匹配的股票
            missing = set(instruments) - set(result['instrument'])
            if missing:
                print(f"  ⚠️  {len(missing)} 只股票未找到行业，标记为'其他'")
                missing_df = pd.DataFrame({
                    'instrument': list(missing),
                    'industry': '其他'
                })
                result = pd.concat([result, missing_df], ignore_index=True)

            print(f"  ✓ 行业数据获取完成")
            print(f"     覆盖率: {(len(result) - len(missing)) / len(instruments) * 100:.1f}%")
            print(f"     行业数: {result['industry'].nunique()}个")

            # 显示行业分布
            top_industries = result['industry'].value_counts().head(5)
            print(f"     TOP5行业:")
            for ind, cnt in top_industries.items():
                print(f"       - {ind}: {cnt}只")

            return result

        except Exception as e:
            print(f"  ⚠️  获取行业数据失败: {e}")
            return pd.DataFrame({
                'instrument': instruments,
                'industry': 'Unknown'
            })
    # ========== 新增: 基本面财务数据获取 ==========

    def get_financial_indicators(self, ts_code, start_date, end_date):
        """
        获取财务指标数据(ROE, ROA, 毛利率, 净利率, 资产负债率)
        :param ts_code: 股票代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :return: DataFrame
        """
        if self.pro is None:
            return None

        # 检查缓存
        cache_name = f"financial_{ts_code}_{start_date}_{end_date}"
        if self.cache:
            cached_data = self.cache.load_from_cache(cache_name)
            if cached_data is not None:
                return cached_data

        try:
            # 获取财务指标数据
            df = self.pro.fina_indicator(
                ts_code=ts_code,
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                fields='ts_code,ann_date,end_date,roe,roa,grossprofit_margin,netprofit_margin,debt_to_assets'
            )

            if df is None or len(df) == 0:
                return None

            # 数据预处理
            df = df.rename(columns={
                'ann_date': 'date',  # 使用公告日期
                'grossprofit_margin': 'gross_margin',
                'netprofit_margin': 'net_margin',
                'debt_to_assets': 'debt_ratio'
            })

            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df['instrument'] = ts_code

            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)

            # 保存到缓存
            if self.cache:
                self.cache.save_to_cache(df, cache_name)

            time.sleep(0.31)  # API限流
            return df

        except Exception as e:
            print(f"✗ 获取 {ts_code} 财务指标失败: {e}")
            time.sleep(1)
            return None

    def merge_financial_data_to_daily(self, price_df, financial_df):
        """
        将季度财务数据合并到日线数据（修复版）
        使用前向填充方法:每个交易日使用最近公告的财务数据

        :param price_df: 日线价格数据
        :param financial_df: 财务指标数据
        :return: 合并后的DataFrame
        """
        if financial_df is None or len(financial_df) == 0:
            print("  ⚠️  财务数据为空，跳过合并")
            return price_df

        # 确保日期格式一致
        price_df['date'] = pd.to_datetime(price_df['date'], errors='coerce')
        financial_df['date'] = pd.to_datetime(financial_df['date'], errors='coerce')

        financial_df = financial_df.dropna(subset=['date', 'instrument'])
        price_df = price_df.dropna(subset=['date', 'instrument'])

        if len(financial_df) == 0:
            return price_df
        # ========== 关键修复：清理空值 ==========
        print("  🔍 清理数据空值...")

        # 1. 清理财务数据中的空值
        original_len = len(financial_df)
        financial_df = financial_df.dropna(subset=['date', 'instrument'])
        cleaned_len = len(financial_df)

        if original_len > cleaned_len:
            print(f"     财务数据: 移除 {original_len - cleaned_len} 条空值记录")

        # 2. 清理价格数据中的空值
        price_df = price_df.dropna(subset=['date', 'instrument'])

        if len(financial_df) == 0:
            print("  ⚠️  财务数据清理后为空，跳过合并")
            return price_df

        # 对每只股票单独处理
        result_list = []
        success_count = 0
        fail_count = 0

        print("  🔗 合并财务数据到日线...")
        instruments = price_df['instrument'].unique()

        for idx, instrument in enumerate(instruments):
            if (idx + 1) % 500 == 0:
                print(f"     进度: {idx + 1}/{len(instruments)}")

            price_subset = price_df[price_df['instrument'] == instrument].copy()
            financial_subset = financial_df[financial_df['instrument'] == instrument].copy()

            if len(financial_subset) == 0:
                # 没有财务数据，直接使用价格数据
                result_list.append(price_subset)
                continue

            # 再次确保当前股票的财务数据没有空值
            financial_subset = financial_subset.dropna(subset=['date'])

            if len(financial_subset) == 0:
                result_list.append(price_subset)
                continue

            try:
                # 使用merge_asof进行前向填充合并
                merged = pd.merge_asof(
                    price_subset.sort_values('date'),
                    financial_subset.sort_values('date')[
                        ['date', 'roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']
                    ],
                    on='date',
                    direction='backward'  # 向后查找最近的财务数据
                )
                result_list.append(merged)
                success_count += 1

            except Exception as e:
                # 合并失败，使用原始价格数据
                print(f"     ⚠️  {instrument} 合并失败: {e}")
                result_list.append(price_subset)
                fail_count += 1

        if len(result_list) == 0:
            print("  ⚠️  没有可合并的数据")
            return price_df

        result_df = pd.concat(result_list, ignore_index=True)

        # 统计信息
        print(f"  ✓ 合并完成:")
        print(f"     成功: {success_count} 只")
        if fail_count > 0:
            print(f"     失败: {fail_count} 只")

        # 统计基本面数据覆盖率
        fundamental_cols = ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']
        available_cols = [col for col in fundamental_cols if col in result_df.columns]

        if available_cols:
            has_data = result_df[available_cols].notna().any(axis=1)
            coverage = (has_data.sum() / len(result_df)) * 100
            print(f"     覆盖率: {coverage:.1f}%")

        return result_df

class StockRankerModel:
    """
    StockRanker 多因子评分模型 (扩展版)
    整合: 估值、波动率、资金流、动量、基本面因子
    """

    def __init__(self, custom_weights=None, use_fundamental=True):
        """
        初始化模型
        :param custom_weights: 自定义因子权重字典
        :param use_fundamental: 是否使用基本面因子
        """
        self.use_fundamental = use_fundamental

        if custom_weights:
            self.factor_weights = custom_weights
        else:
            # 默认权重配置 (包含基本面因子)
            if use_fundamental:
                self.factor_weights = {
                    # 估值因子 (25%) - 越低越好
                    'pe_ratio': -0.10,
                    'pb_ratio': -0.10,
                    'ps_ratio': -0.05,

                    # 波动率因子 (15%) - 越低越好
                    'volatility_20d': -0.08,
                    'volatility_60d': -0.07,

                    # 资金流因子 (15%) - 越高越好
                    'money_flow_20d': 0.08,
                    'volume_ratio': 0.07,

                    # 动量因子 (15%) - 越高越好
                    'return_20d': 0.08,
                    'return_60d': 0.07,

                    # 基本面因子 (30%) - 新增
                    'roe': 0.10,  # ROE越高越好
                    'roa': 0.05,  # ROA越高越好
                    'gross_margin': 0.05,  # 毛利率越高越好
                    'net_margin': 0.05,  # 净利率越高越好
                    'debt_ratio': -0.05  # 资产负债率越低越好
                }
            else:
                # 不使用基本面因子的原始配置
                self.factor_weights = {
                    'pe_ratio': -0.15,
                    'pb_ratio': -0.15,
                    'ps_ratio': -0.10,
                    'volatility_20d': -0.10,
                    'volatility_60d': -0.10,
                    'money_flow_20d': 0.10,
                    'volume_ratio': 0.10,
                    'return_20d': 0.10,
                    'return_60d': 0.10
                }

        print("\n" + "=" * 60)
        print("📊 StockRanker 多因子评分模型")
        if use_fundamental:
            print("    ✨ 基本面因子已启用")
        print("=" * 60)
        self._print_weights()

    def _print_weights(self):
        """打印因子权重配置"""
        print("\n因子权重配置:")

        # 估值因子
        print("  ├─ 估值因子 (25%)" if self.use_fundamental else "  ├─ 估值因子 (40%)")
        print(f"  │   ├─ PE市盈率: {self.factor_weights.get('pe_ratio', 0):.2%}")
        print(f"  │   ├─ PB市净率: {self.factor_weights.get('pb_ratio', 0):.2%}")
        print(f"  │   └─ PS市销率: {self.factor_weights.get('ps_ratio', 0):.2%}")

        # 波动率因子
        print("  ├─ 波动率因子 (15%)" if self.use_fundamental else "  ├─ 波动率因子 (20%)")
        print(f"  │   ├─ 20日波动率: {self.factor_weights.get('volatility_20d', 0):.2%}")
        print(f"  │   └─ 60日波动率: {self.factor_weights.get('volatility_60d', 0):.2%}")

        # 资金流因子
        print("  ├─ 资金流因子 (15%)" if self.use_fundamental else "  ├─ 资金流因子 (20%)")
        print(f"  │   ├─ 20日资金流: {self.factor_weights.get('money_flow_20d', 0):.2%}")
        print(f"  │   └─ 量比: {self.factor_weights.get('volume_ratio', 0):.2%}")

        # 动量因子
        print("  ├─ 动量因子 (15%)" if self.use_fundamental else "  └─ 动量因子 (20%)")
        print(f"  │   ├─ 20日收益率: {self.factor_weights.get('return_20d', 0):.2%}")
        print(f"  │   └─ 60日收益率: {self.factor_weights.get('return_60d', 0):.2%}")

        # 基本面因子 (新增)
        if self.use_fundamental:
            print("  └─ 基本面因子 (30%) ✨新增")
            print(f"      ├─ ROE(净资产收益率): {self.factor_weights.get('roe', 0):.2%}")
            print(f"      ├─ ROA(总资产收益率): {self.factor_weights.get('roa', 0):.2%}")
            print(f"      ├─ 毛利率: {self.factor_weights.get('gross_margin', 0):.2%}")
            print(f"      ├─ 净利率: {self.factor_weights.get('net_margin', 0):.2%}")
            print(f"      └─ 资产负债率: {self.factor_weights.get('debt_ratio', 0):.2%}")

    def calculate_valuation_factors(self, df):
        """计算估值因子(简化版 - 使用技术指标估算)"""
        df['pe_ratio'] = df['close'] / df.groupby('instrument')['close'].transform('mean')
        df['pb_ratio'] = df['close'] / (df.groupby('instrument')['close'].transform('mean') * 0.8)
        df['ps_ratio'] = df['close'] / (df.groupby('instrument')['close'].transform('mean') * 1.2)
        return df

    def calculate_volatility_factors(self, df):
        """计算波动率因子"""
        df['volatility_20d'] = df.groupby('instrument')['close'].rolling(20).std().reset_index(0, drop=True)
        df['volatility_60d'] = df.groupby('instrument')['close'].rolling(60).std().reset_index(0, drop=True)
        return df

    def calculate_money_flow_factors(self, df):
        """计算资金流因子"""
        df['money_flow_20d'] = (df['volume'] * df['close']).rolling(20).mean()
        df['volume_ma5'] = df.groupby('instrument')['volume'].rolling(5).mean().reset_index(0, drop=True)
        df['volume_ma20'] = df.groupby('instrument')['volume'].rolling(20).mean().reset_index(0, drop=True)
        df['volume_ratio'] = df['volume_ma5'] / (df['volume_ma20'] + 1e-6)
        return df

    def calculate_momentum_factors(self, df):
        """计算动量因子"""
        df['return_20d'] = df.groupby('instrument')['close'].pct_change(20)
        df['return_60d'] = df.groupby('instrument')['close'].pct_change(60)
        return df

    def process_fundamental_factors(self, df):
        """
        处理基本面因子
        基本面数据已通过merge_financial_data_to_daily合并到df中
        这里只需确保数据质量和处理异常值
        """
        if not self.use_fundamental:
            return df

        fundamental_cols = ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']

        for col in fundamental_cols:
            if col in df.columns:
                # 处理异常值:使用中位数填充缺失值
                median_val = df.groupby('instrument')[col].transform('median')
                df[col] = df[col].fillna(median_val)

                # 限制极端值(使用1%和99%分位数)
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)

        return df

    def calculate_all_factors(self, price_data):
        """计算所有因子"""
        print("\n⚙️  计算StockRanker多因子...")
        df = price_data.copy()

        print("  ├─ 估值因子...")
        df = self.calculate_valuation_factors(df)

        print("  ├─ 波动率因子...")
        df = self.calculate_volatility_factors(df)

        print("  ├─ 资金流因子...")
        df = self.calculate_money_flow_factors(df)

        print("  ├─ 动量因子...")
        df = self.calculate_momentum_factors(df)

        if self.use_fundamental:
            print("  └─ 基本面因子处理...")
            df = self.process_fundamental_factors(df)

        print("✓ 因子计算完成")
        return df

    def normalize_factors(self, df):
        """标准化因子(按日期排序百分位)"""
        for factor in self.factor_weights.keys():
            if factor in df.columns:
                # 使用rank进行标准化,处理缺失值
                df[f'{factor}_norm'] = df.groupby('date')[factor].rank(pct=True)
        return df

    def calculate_position_score(self, df):
        """计算综合评分"""
        print("\n📊 计算综合评分...")
        df = self.normalize_factors(df)

        # 加权求和
        df['position'] = 0
        for factor, weight in self.factor_weights.items():
            norm_factor = f'{factor}_norm'
            if norm_factor in df.columns:
                df['position'] += df[norm_factor].fillna(0.5) * weight

        # 归一化到0-1区间
        min_score = df.groupby('date')['position'].transform('min')
        max_score = df.groupby('date')['position'].transform('max')
        df['position'] = (df['position'] - min_score) / (max_score - min_score + 1e-6)

        print("✓ 评分计算完成")
        return df


def load_data_from_tushare(start_date, end_date, max_stocks=50, use_cache=True,
                           cache_manager=None, use_stockranker=True,
                           custom_weights=None, tushare_token=None,
                           use_fundamental=True):
    """
    从Tushare加载数据并计算因子 (扩展版 - 支持基本面因子 + 行业数据)

    :param start_date: 开始日期
    :param end_date: 结束日期
    :param max_stocks: 最大股票数
    :param use_cache: 是否使用缓存
    :param cache_manager: 缓存管理器
    :param use_stockranker: 是否使用StockRanker模型
    :param custom_weights: 自定义因子权重
    :param tushare_token: Tushare token
    :param use_fundamental: 是否使用基本面因子
    """
    print("\n" + "=" * 80)
    print("📦 数据加载模块 (Tushare版 + 基本面 + 行业)")
    print("=" * 80)

    model_type = "StockRanker多因子" if use_stockranker else "简单技术因子"
    if use_stockranker and use_fundamental:
        model_type += " + 基本面"
    print(f"因子模型: {model_type}")

    # 生成缓存文件名
    model_suffix = "stockranker" if use_stockranker else "simple"
    if use_fundamental:
        model_suffix += "_fundamental"

    cache_key = f"factor_data_ts_{start_date}_{end_date}_{max_stocks}_{model_suffix}"
    price_cache_key = f"price_data_ts_{start_date}_{end_date}_{max_stocks}"

    # 尝试从缓存加载
    if use_cache and cache_manager:
        print("\n🔍 检查缓存...")
        factor_data = cache_manager.load_from_csv(cache_key)
        price_data = cache_manager.load_from_csv(price_cache_key)

        if factor_data is not None and price_data is not None:
            print("✓ 使用缓存数据")
            print(f"  - 因子数据: {len(factor_data)} 条")
            print(f"  - 价格数据: {len(price_data)} 条")
            return factor_data, price_data
        else:
            print("✗ 缓存未找到,开始从Tushare获取...")

    # 初始化Tushare数据源
    data_source = TushareDataSource(
        cache_manager=cache_manager if use_cache else None,
        token=tushare_token
    )

    # 获取股票列表
    stock_list = data_source.get_stock_list()
    if not stock_list:
        print("✗ 无法获取股票列表!")
        return None, None

    stock_list = stock_list[:max_stocks]

    # ========== 获取价格数据 ==========
    all_price_data = []
    success_count = 0

    print(f"\n📊 获取 {len(stock_list)} 只股票的历史数据...")
    print("进度: ", end='')

    for i, ts_code in enumerate(stock_list):
        if (i + 1) % 5 == 0:
            print(f"{i + 1}/{len(stock_list)} ", end='', flush=True)

        df = data_source.get_price_data(ts_code, start_date, end_date)
        if df is not None and len(df) > 0:
            all_price_data.append(df)
            success_count += 1

    print(f"\n✓ 成功获取 {success_count}/{len(stock_list)} 只股票的价格数据")

    if len(all_price_data) == 0:
        print("✗ 未获取到任何数据!")
        return None, None

    # 合并价格数据
    price_df = pd.concat(all_price_data, ignore_index=True)

    # ========== 获取基本面数据 ==========
    if use_stockranker and use_fundamental:
        print(f"\n📈 获取基本面财务数据...")
        all_financial_data = []
        financial_success = 0

        print("进度: ", end='')
        for i, ts_code in enumerate(stock_list):
            if (i + 1) % 5 == 0:
                print(f"{i + 1}/{len(stock_list)} ", end='', flush=True)

            financial_df = data_source.get_financial_indicators(ts_code, start_date, end_date)
            if financial_df is not None and len(financial_df) > 0:
                all_financial_data.append(financial_df)
                financial_success += 1

        print(f"\n✓ 成功获取 {financial_success}/{len(stock_list)} 只股票的财务数据")

        if len(all_financial_data) > 0:
            financial_df = pd.concat(all_financial_data, ignore_index=True)
            print("\n🔗 合并基本面数据到日线数据...")
            price_df = data_source.merge_financial_data_to_daily(price_df, financial_df)
            print("✓ 基本面数据合并完成")

            fundamental_cols = ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']
            available_cols = [col for col in fundamental_cols if col in price_df.columns]
            if available_cols:
                coverage = (price_df[available_cols].notna().any(axis=1).sum() / len(price_df)) * 100
                print(f"  基本面数据覆盖率: {coverage:.1f}%")
        else:
            print("⚠️  未获取到基本面数据,将不使用基本面因子")
            use_fundamental = False

    price_df['date'] = price_df['date'].astype(str)

    # ========== 选择因子计算方法 ==========
    if use_stockranker:
        model = StockRankerModel(
            custom_weights=custom_weights,
            use_fundamental=use_fundamental
        )
        factor_df = model.calculate_all_factors(price_df)
        factor_df = model.calculate_position_score(factor_df)
    else:
        print("\n⚙️  计算简单技术因子...")
        factor_df = calculate_simple_factors(price_df)

    factor_df = factor_df.dropna(subset=['position'])

    result_factor = factor_df[['date', 'instrument', 'position']].copy()
    result_price = price_df.copy()

    # ========== ✅ 关键添加：获取并合并行业数据 ==========
    print("\n📊 获取行业数据...")
    industry_data = data_source.get_industry_data(stock_list, use_cache=use_cache)

    if industry_data is not None and len(industry_data) > 0:
        # 合并行业数据到因子数据
        result_factor = result_factor.merge(
            industry_data,
            on='instrument',
            how='left'
        )
        result_factor['industry'] = result_factor['industry'].fillna('其他')
        print(f"  ✓ 行业数据已合并到因子数据")
    else:
        result_factor['industry'] = 'Unknown'
        print(f"  ⚠️  未能获取行业数据，使用默认值")

    # 保存到缓存
    if use_cache and cache_manager:
        print("\n💾 保存到缓存...")
        cache_manager.save_to_csv(result_factor, cache_key)
        cache_manager.save_to_csv(result_price, price_cache_key)

    print(f"\n✓ 数据准备完成:")
    print(f"  - 因子数据: {len(result_factor)} 条")
    print(f"  - 价格数据: {len(result_price)} 条")
    print(f"  - 股票数量: {result_factor['instrument'].nunique()} 只")
    print(f"  - 交易日数: {result_factor['date'].nunique()} 天")
    print(f"  - 行业数量: {result_factor['industry'].nunique()} 个")  # ✅ 添加行业统计

    if use_fundamental and use_stockranker:
        print(f"  - 基本面因子: 已启用 (ROE/ROA/毛利率/净利率/资产负债率)")

    return result_factor, result_price

def calculate_simple_factors(price_data):
    """计算简单技术因子(兼容旧版本)"""
    df = price_data.copy()

    # 动量因子
    df['return_5d'] = df.groupby('instrument')['close'].pct_change(5)
    df['return_10d'] = df.groupby('instrument')['close'].pct_change(10)
    df['return_20d'] = df.groupby('instrument')['close'].pct_change(20)

    # 波动率因子
    df['volatility_20d'] = df.groupby('instrument')['close'].rolling(20).std().reset_index(0, drop=True)

    # 成交量因子
    df['volume_ma5'] = df.groupby('instrument')['volume'].rolling(5).mean().reset_index(0, drop=True)
    df['volume_ma20'] = df.groupby('instrument')['volume'].rolling(20).mean().reset_index(0, drop=True)
    df['volume_ratio'] = df['volume_ma5'] / (df['volume_ma20'] + 1e-6)

    # RSI因子
    delta = df.groupby('instrument')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-6)
    df['rsi'] = 100 - (100 / (1 + rs))

    # 标准化各因子
    for col in ['return_20d', 'volume_ratio', 'rsi']:
        if col in df.columns:
            df[f'{col}_norm'] = df.groupby('date')[col].rank(pct=True)

    # 综合评分
    weights = {
        'return_20d_norm': 0.4,
        'volume_ratio_norm': 0.3,
        'rsi_norm': 0.3
    }

    df['position'] = 0
    for factor, weight in weights.items():
        if factor in df.columns:
            df['position'] += df[factor].fillna(0.5) * weight

    df['position'] = df['position'] - (df['rsi_norm'].fillna(0.5) - 0.5) * 0.2

    return df
