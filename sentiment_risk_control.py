# -*- coding: utf-8 -*-
"""
sentiment_risk_control.py - 舆情风控/增强模块 (v1.0)

🎯 核心功能：
1. ⚠️  一票否决：检测严重负面舆情（立案调查、违规处罚等）
2. 📈 加分提权：捕捉正面题材（政策支持、行业热点等）
3. 🔍 智能过滤：区分噪音与真实信号

数据源：
- Tushare news (财经新闻)
- Tushare cctv_news (新闻联播 - 政策风向标)
- Tushare fina_audit (财务审计)
- Tushare disclosure (公告预警)

使用方式：
```python
from sentiment_risk_control import SentimentRiskController

# 初始化
controller = SentimentRiskController(tushare_token=YOUR_TOKEN)

# 对选股结果进行风控增强
filtered_stocks = controller.apply_sentiment_filter(
    selected_stocks=top_stocks_df,
    factor_data=factor_data,
    price_data=price_data
)
```

版本：v1.0
日期：2025-12-17
作者：Claude
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
from typing import List, Dict, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

try:
    import tushare as ts

    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("⚠️  Tushare未安装: pip install tushare")


# ============================================================================
# 第一部分：舆情数据采集器
# ============================================================================

class SentimentDataCollector:
    """舆情数据采集器"""

    def __init__(self, token: Optional[str] = None, cache_manager=None):
        """初始化采集器"""
        if not TUSHARE_AVAILABLE:
            raise ImportError("请先安装Tushare: pip install tushare")

        if token:
            ts.set_token(token)  # type: ignore

        try:
            self.pro = ts.pro_api()  # type: ignore
            print("✓ Tushare API初始化成功")
        except Exception as e:
            print(f"✗ Tushare初始化失败: {e}")
            self.pro = None

        self.cache = cache_manager
        self.request_count = 0
        self.last_request_time = time.time()

    def _rate_limit(self, wait_time: float = 0.5):
        """访问频率控制"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < wait_time:
            time.sleep(wait_time - elapsed)

        self.last_request_time = time.time()
        self.request_count += 1

        # 每100次请求暂停5秒
        if self.request_count % 100 == 0:
            print(f"  ⏳ API调用{self.request_count}次，暂停5秒...")
            time.sleep(5)

    def get_news(self, start_date: str, end_date: str, src: Optional[str] = None) -> pd.DataFrame:
        """
        获取财经新闻

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            src: 新闻来源 (sina/ths/wallstreet等)

        Returns:
            DataFrame: 新闻数据
        """
        if self.pro is None:
            return pd.DataFrame()

        try:
            self._rate_limit()

            df = self.pro.news(
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', ''),
                src=src
            )

            if df is not None and len(df) > 0:
                df['date'] = pd.to_datetime(df['datetime'], errors='coerce').dt.date
                df['date'] = df['date'].astype(str)
                print(f"  ✓ 获取财经新闻: {len(df)} 条")
                return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

            return pd.DataFrame()

        except Exception as e:
            print(f"  ⚠️  获取新闻失败: {e}")
            return pd.DataFrame()

    def get_cctv_news(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取新闻联播内容 (政策风向标)

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            DataFrame: 新闻联播数据
        """
        if self.pro is None:
            return pd.DataFrame()

        try:
            self._rate_limit()

            df = self.pro.cctv_news(
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )

            if df is not None and len(df) > 0:
                print(f"  ✓ 获取新闻联播: {len(df)} 条")
                return df

            return pd.DataFrame()

        except Exception as e:
            print(f"  ⚠️  获取新闻联播失败: {e}")
            return pd.DataFrame()

    def get_financial_audit(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取财务审计/立案调查信息 (批量查询优化版)

        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            DataFrame: 审计数据
        """
        if self.pro is None:
            return pd.DataFrame()

        try:
            self._rate_limit(wait_time=1.0)  # 这个接口限制更严格

            # 批量查询整个市场的审计信息
            df = self.pro.fina_audit(
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )

            if df is not None and len(df) > 0:
                # 只保留指定股票的审计信息
                df = df[df['ts_code'].isin(ts_codes)]
                
                if len(df) > 0:
                    df['ann_date'] = pd.to_datetime(df['ann_date'], format='%Y%m%d', errors='coerce')
                    df['date'] = df['ann_date'].dt.strftime('%Y-%m-%d')  # type: ignore
                    print(f"  ✓ 获取财务审计: {len(df)} 条 (批量查询)")
                    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
                else:
                    print(f"  ℹ️  指定期间内无相关股票的财务审计信息")
                    return pd.DataFrame()
            else:
                print(f"  ℹ️  指定期间内无财务审计信息")
                return pd.DataFrame()

        except Exception as e:
            if "必填参数" in str(e):
                # 如果是必填参数错误，尝试按股票逐个查询
                print(f"  ⚠️  批量查询财务审计失败，尝试逐个查询...")
                all_data = []
                
                for ts_code in ts_codes[:10]:  # 限制查询数量避免超限
                    try:
                        self._rate_limit(wait_time=1.0)
                        stock_df = self.pro.fina_audit(
                            ts_code=ts_code,
                            start_date=start_date.replace('-', ''),
                            end_date=end_date.replace('-', '')
                        )
                        
                        if stock_df is not None and len(stock_df) > 0:
                            all_data.append(stock_df)
                            
                    except Exception:
                        continue  # 静默失败
                
                if all_data:
                    result = pd.concat(all_data, ignore_index=True)
                    result['ann_date'] = pd.to_datetime(result['ann_date'], format='%Y%m%d', errors='coerce')
                    result['date'] = result['ann_date'].dt.strftime('%Y-%m-%d')  # type: ignore
                    result = result[result['ts_code'].isin(ts_codes)]  # 再次过滤
                    print(f"  ✓ 逐个查询获取财务审计: {len(result)} 条")
                    return result if isinstance(result, pd.DataFrame) else pd.DataFrame()
                else:
                    print(f"  ℹ️  逐个查询也未获取到财务审计信息")
                    return pd.DataFrame()
            else:
                print(f"  ⚠️  获取财务审计失败: {e}")
                return pd.DataFrame()

    def get_disclosure_info(self, ts_codes: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取公告预警信息 (立案调查、违规处罚等) (批量查询优化版)

        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'

        Returns:
            DataFrame: 公告数据
        """
        if self.pro is None:
            return pd.DataFrame()

        try:
            self._rate_limit(wait_time=1.0)

            # 批量查询整个市场的公告信息
            df = self.pro.disclosure_date(
                start_date=start_date.replace('-', ''),
                end_date=end_date.replace('-', '')
            )

            if df is not None and len(df) > 0:
                # 只保留指定股票的公告信息
                df = df[df['ts_code'].isin(ts_codes)]
                
                if len(df) > 0:
                    if 'actual_date' in df.columns:
                        df['actual_date'] = pd.to_datetime(df['actual_date'], format='%Y%m%d', errors='coerce')
                        df['date'] = df['actual_date'].dt.strftime('%Y-%m-%d')  # type: ignore
                    print(f"  ✓ 获取公告信息: {len(df)} 条 (批量查询)")
                    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

            return pd.DataFrame()

        except Exception as e:
            print(f"  ⚠️  获取公告信息失败: {e}")
            return pd.DataFrame()

    def get_news_batch(self, start_date: str, end_date: str, src: Optional[str] = None) -> pd.DataFrame:
        """
        批量获取财经新闻 (按时间段批量查询)

        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            src: 新闻来源 (sina/ths/wallstreet等)

        Returns:
            DataFrame: 新闻数据
        """
        if self.pro is None:
            return pd.DataFrame()

        try:
            # 优先获取最近几天的新闻，因为越近越重要
            all_news = []
            current_date = pd.to_datetime(end_date)  # 从结束日期开始
            start_dt = pd.to_datetime(start_date)
            
            # 限制查询天数，避免超出API限制
            days_processed = 0
            max_days = 3  # 最多查询3天的新闻数据，优先最近的
            
            while current_date >= start_dt and days_processed < max_days:
                date_str = current_date.strftime('%Y-%m-%d')
                date_str_no_dash = current_date.strftime('%Y%m%d')
                
                try:
                    self._rate_limit(2.0)  # 增加等待时间
                    
                    df = self.pro.news(
                        start_date=date_str_no_dash,
                        end_date=date_str_no_dash,
                        src=src
                    )
                    
                    if df is not None and len(df) > 0:
                        df['date'] = date_str
                        all_news.append(df)
                        print(f"    ✓ 获取{date_str}新闻: {len(df)}条")
                    else:
                        print(f"    ℹ️  {date_str}无新闻数据")
                        
                except Exception as e:
                    if "最多访问该接口" in str(e):
                        print(f"    ⚠️  {date_str}新闻获取受限: {str(e).split('。')[0]}")
                        # 遇到限制时暂停更长时间
                        time.sleep(5)
                        break  # 遇到限制立即停止
                    else:
                        print(f"    ⚠️  获取{date_str}新闻失败: {e}")
                
                current_date -= timedelta(days=1)  # 向前推一天
                days_processed += 1
            
            if all_news:
                result = pd.concat(all_news, ignore_index=True)
                if 'datetime' in result.columns:
                    result['datetime'] = pd.to_datetime(result['datetime'], errors='coerce')
                print(f"  ✓ 批量获取财经新闻: {len(result)} 条 (优先最近{max_days}天)")
                return result if isinstance(result, pd.DataFrame) else pd.DataFrame()
            else:
                print(f"  ℹ️  未获取到新闻数据 (优先最近{max_days}天)")
                return pd.DataFrame()

        except Exception as e:
            print(f"  ⚠️  批量获取新闻失败: {e}")
            return pd.DataFrame()


# ============================================================================
# 第二部分：舆情规则引擎
# ============================================================================

class SentimentRuleEngine:
    """舆情规则引擎 - 定义一票否决和加分规则"""

    # 一票否决关键词 (严重负面)
    VETO_KEYWORDS = {
        'critical': [
            '立案调查', '证监会调查', '涉嫌违规', '欺诈发行',
            '财务造假', '内幕交易', 'ST', '*ST', '退市风险',
            '重大违法', '暂停上市', '终止上市', '破产重整'
        ],
        'high_risk': [
            '业绩爆雷', '业绩大幅下滑', '商誉减值', '债务违约',
            '控股股东质押', '资金链断裂', '高管辞职', '董事长辞职'
        ]
    }

    # 加分关键词 (正面题材)
    BOOST_KEYWORDS = {
        'policy_support': {
            'keywords': [
                '新质生产力', '低空经济', '人工智能', '数字经济',
                '国企改革', '一带一路', '碳中和', '新能源',
                '半导体', '自主可控', '国产替代', '科技创新'
            ],
            'boost_score': 0.10  # 加10%评分
        },
        'hot_concept': {
            'keywords': [
                '业绩预增', '中报预喜', '重大订单', '战略合作',
                '股权激励', '回购增持', '并购重组', '资产注入'
            ],
            'boost_score': 0.05  # 加5%评分
        },
        'cctv_mention': {
            'keywords': [
                # 这个会在CCTV新闻中匹配行业关键词
                '制造业', '科技', '创新', '产业升级', '高质量发展'
            ],
            'boost_score': 0.08  # 新闻联播提及加8%
        }
    }

    def __init__(self):
        """初始化规则引擎"""
        print("\n🎯 舆情规则引擎初始化")
        print(f"  - 一票否决关键词: {len(self.VETO_KEYWORDS['critical']) + len(self.VETO_KEYWORDS['high_risk'])} 个")
        print(f"  - 加分关键词组: {len(self.BOOST_KEYWORDS)} 组")

    def check_veto_triggers(self, text: str) -> Tuple[bool, str]:
        """
        检查是否触发一票否决

        Returns:
            (is_veto, reason)
        """
        if pd.isna(text) or not isinstance(text, str):
            return False, ""

        text = text.lower()

        # Critical级别：直接否决
        for keyword in self.VETO_KEYWORDS['critical']:
            if keyword.lower() in text:
                return True, f"Critical风险: {keyword}"

        # High Risk级别：计数触发
        high_risk_count = sum(1 for kw in self.VETO_KEYWORDS['high_risk']
                              if kw.lower() in text)
        if high_risk_count >= 2:  # 同时出现2个以上高风险词
            return True, f"高风险预警 ({high_risk_count}个负面词)"

        return False, ""

    def calculate_boost_score(self, text: str, source: str = 'news') -> Tuple[float, List[str]]:
        """
        计算加分值

        Returns:
            (boost_score, matched_keywords)
        """
        if pd.isna(text) or not isinstance(text, str):
            return 0.0, []

        text = text.lower()
        total_boost = 0.0
        matched = []

        for category, config in self.BOOST_KEYWORDS.items():
            # CCTV新闻特殊处理
            if category == 'cctv_mention' and source != 'cctv':
                continue

            for keyword in config['keywords']:
                if keyword.lower() in text:
                    total_boost += config['boost_score']
                    matched.append(f"{keyword}(+{config['boost_score']:.1%})")
                    break  # 每个类别只加分一次

        return min(total_boost, 0.20), matched  # 最多加20%


# ============================================================================
# 第三部分：舆情分析器
# ============================================================================

class SentimentAnalyzer:
    """舆情分析器 - 整合数据采集和规则判断"""

    def __init__(self, collector: SentimentDataCollector, rule_engine: SentimentRuleEngine):
        """初始化分析器"""
        self.collector = collector
        self.rules = rule_engine

    def analyze_stock_sentiment(self, ts_code: str, start_date: str, end_date: str,
                              cached_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict:
        """
        分析单只股票的舆情 (使用缓存数据)

        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            cached_data: 缓存的数据字典，包含audit_df, disclosure_df, news_df

        Returns:
            {
                'ts_code': str,
                'is_veto': bool,
                'veto_reason': str,
                'boost_score': float,
                'boost_reasons': List[str],
                'news_count': int,
                'audit_issues': int
            }
        """
        result = {
            'ts_code': ts_code,
            'is_veto': False,
            'veto_reason': '',
            'boost_score': 0.0,
            'boost_reasons': [],
            'news_count': 0,
            'audit_issues': 0
        }

        # 使用缓存数据进行分析
        if cached_data:
            audit_df = cached_data.get('audit_df', pd.DataFrame())
            disclosure_df = cached_data.get('disclosure_df', pd.DataFrame())
            news_df = cached_data.get('news_df', pd.DataFrame())
        else:
            # 如果没有缓存数据，则单独查询
            audit_df = self.collector.get_financial_audit([ts_code], start_date, end_date)
            disclosure_df = self.collector.get_disclosure_info([ts_code], start_date, end_date)
            news_df = self.collector.get_news_batch(start_date, end_date)

        # 1. 检查财务审计 (最高优先级)
        if not audit_df.empty:
            stock_audit = audit_df[audit_df['ts_code'] == ts_code]
            result['audit_issues'] = len(stock_audit)
            # 如果有审计问题，默认一票否决
            if len(stock_audit) > 0:
                result['is_veto'] = True
                result['veto_reason'] = f"财务审计异常 ({len(stock_audit)}条)"
                return result

        # 2. 检查公告信息
        if not disclosure_df.empty:
            stock_disclosure = disclosure_df[disclosure_df['ts_code'] == ts_code]
            # 检查是否有关键负面词
            critical_keywords_found = []
            high_risk_keywords_found = []
            
            for _, row in stock_disclosure.iterrows():
                title = str(row.get('title', ''))
                if pd.isna(title):
                    continue
                    
                # 检查Critical级别关键词
                for keyword in self.rules.VETO_KEYWORDS['critical']:
                    if keyword.lower() in title.lower():
                        critical_keywords_found.append(keyword)
                
                # 检查High Risk级别关键词
                for keyword in self.rules.VETO_KEYWORDS['high_risk']:
                    if keyword.lower() in title.lower():
                        high_risk_keywords_found.append(keyword)
            
            # Critical级别：直接否决
            if critical_keywords_found:
                result['is_veto'] = True
                result['veto_reason'] = f"严重风险: {critical_keywords_found[0]}"
                return result
            
            # High Risk级别：多个关键词触发否决
            if len(high_risk_keywords_found) >= 2:
                result['is_veto'] = True
                result['veto_reason'] = f"高风险预警 ({len(high_risk_keywords_found)}个负面词)"
                return result

        # 3. 分析新闻
        if not news_df.empty:
            # 筛选该股票的新闻
            if 'ts_code' in news_df.columns:
                stock_news = news_df[news_df['ts_code'] == ts_code]
            else:
                # 如果没有ts_code列，假设所有新闻都是相关的
                stock_news = news_df
                
            result['news_count'] = len(stock_news)
            
            # 计算加分
            boost_scores = []
            for _, row in stock_news.iterrows():
                title = str(row.get('title', ''))
                content = str(row.get('content', ''))
                full_text = f"{title} {content}"
                
                if pd.isna(title) and pd.isna(content):
                    continue
                
                boost_score, matched_keywords = self.rules.calculate_boost_score(full_text, source='news')
                if boost_score > 0:
                    boost_scores.append((boost_score, matched_keywords))
            
            # 累加加分项（设置上限）
            total_boost = 0.0
            all_matched = []
            for boost_score, matched_keywords in boost_scores:
                total_boost += boost_score
                all_matched.extend(matched_keywords)
            
            # 设置最大加分限制
            result['boost_score'] = min(total_boost, 0.20)  # 最多加20%
            result['boost_reasons'] = all_matched[:10]  # 最多记录10个匹配词

        return result

    def batch_analyze_sentiment(self, ts_codes: List[str], start_date: str, end_date: str) -> Dict[str, Dict]:
        """
        批量分析多只股票的舆情

        Args:
            ts_codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            Dict[ts_code, sentiment_analysis_result]
        """
        print(f"  📊 批量分析 {len(ts_codes)} 只股票舆情...")
        
        # 1. 批量获取所有数据
        print("    [1/3] 批量获取财务审计数据...")
        audit_df = self.collector.get_financial_audit(ts_codes, start_date, end_date)
        
        print("    [2/3] 批量获取公告信息...")
        disclosure_df = self.collector.get_disclosure_info(ts_codes, start_date, end_date)
        
        print("    [3/3] 批量获取新闻数据...")
        news_df = self.collector.get_news_batch(start_date, end_date)
        
        # 2. 构建缓存数据字典
        cached_data = {
            'audit_df': audit_df,
            'disclosure_df': disclosure_df,
            'news_df': news_df
        }
        
        # 3. 逐个分析每只股票
        results = {}
        for i, ts_code in enumerate(ts_codes):
            if (i + 1) % 50 == 0:
                print(f"      进度: {i + 1}/{len(ts_codes)}")
            
            results[ts_code] = self.analyze_stock_sentiment(ts_code, start_date, end_date, cached_data)
        
        # 4. 统计分析结果
        veto_count = sum(1 for r in results.values() if r['is_veto'])
        boost_count = sum(1 for r in results.values() if r['boost_score'] > 0)
        
        if veto_count > 0:
            print(f"    🚫 发现 {veto_count} 只风险股票")
        if boost_count > 0:
            print(f"    📈 发现 {boost_count} 只加分股票")
        
        return results

    def analyze_market_sentiment(self, start_date: str, end_date: str) -> Dict:
        """
        分析市场整体舆情 (新闻联播、热点题材)

        Returns:
            {
                'hot_themes': List[str],
                'policy_support_keywords': List[str],
                'market_mood': str  # 'positive', 'neutral', 'negative'
            }
        """
        result = {
            'hot_themes': [],
            'policy_support_keywords': [],
            'market_mood': 'neutral'
        }

        # 获取新闻联播
        cctv_df = self.collector.get_cctv_news(start_date, end_date)

        if not cctv_df.empty:
            all_text = ' '.join(cctv_df['title'].dropna().tolist())

            # 提取政策支持关键词
            for keyword in self.rules.BOOST_KEYWORDS['policy_support']['keywords']:
                if keyword.lower() in all_text.lower():
                    result['policy_support_keywords'].append(keyword)

            # 简单情绪判断
            positive_count = sum(1 for kw in result['policy_support_keywords'])
            if positive_count >= 3:
                result['market_mood'] = 'positive'

        return result


# ============================================================================
# 第四部分：主控制器
# ============================================================================

class SentimentRiskController:
    """舆情风控/增强主控制器"""

    def __init__(self, tushare_token: Optional[str] = None, cache_manager=None,
                 lookback_days: int = 30):
        """
        初始化控制器

        Args:
            tushare_token: Tushare Token
            cache_manager: 缓存管理器
            lookback_days: 舆情回溯天数 (默认30天)
        """
        print("\n" + "=" * 80)
        print("🛡️  舆情风控/增强模块初始化")
        print("=" * 80)

        self.lookback_days = lookback_days

        # 初始化组件
        self.collector = SentimentDataCollector(token=tushare_token, cache_manager=cache_manager)
        self.rules = SentimentRuleEngine()
        self.analyzer = SentimentAnalyzer(self.collector, self.rules)

        print(f"\n✓ 初始化完成 (回溯期: {lookback_days}天)")

    def apply_sentiment_filter(self, selected_stocks: pd.DataFrame,
                               factor_data: pd.DataFrame,
                               price_data: pd.DataFrame,
                               enable_veto: bool = True,
                               enable_boost: bool = True) -> pd.DataFrame:
        """
        对选股结果应用舆情过滤和增强

        Args:
            selected_stocks: 选股结果 (必须包含 'instrument' 列)
            factor_data: 因子数据 (用于获取日期)
            price_data: 价格数据
            enable_veto: 是否启用一票否决
            enable_boost: 是否启用加分增强

        Returns:
            DataFrame: 过滤后的股票列表
        """
        print("\n" + "=" * 80)
        print("🔍 执行舆情风控/增强")
        print("=" * 80)

        if selected_stocks.empty:
            print("  ⚠️  输入为空，跳过舆情分析")
            return selected_stocks

        # 确定分析时间范围
        latest_date = factor_data['date'].max()
        end_date = str(latest_date).split(' ')[0]
        start_date = (pd.to_datetime(end_date) - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')

        print(f"  📅 分析期间: {start_date} ~ {end_date}")
        print(f"  📊 待分析股票: {len(selected_stocks)} 只")

        # 获取股票列表
        stock_list = selected_stocks['instrument'].unique().tolist()

        # 1. 市场整体舆情分析
        print("\n  [1/3] 分析市场整体舆情...")
        market_sentiment = self.analyzer.analyze_market_sentiment(start_date, end_date)

        if market_sentiment['policy_support_keywords']:
            print(f"    ✓ 政策热点: {', '.join(market_sentiment['policy_support_keywords'][:5])}")

        # 2. 批量个股舆情分析 (高效方式)
        print(f"\n  [2/3] 批量个股舆情分析...")
        sentiment_results = self.analyzer.batch_analyze_sentiment(stock_list, start_date, end_date)

        # 3. 应用过滤规则
        print(f"\n  [3/3] 应用过滤规则...")

        result = selected_stocks.copy()

        # 一票否决
        if enable_veto:
            veto_list = []
            for ts_code, sentiment in sentiment_results.items():
                if sentiment['is_veto']:
                    veto_list.append({
                        'instrument': ts_code,
                        'reason': sentiment['veto_reason']
                    })

            if veto_list:
                veto_codes = [item['instrument'] for item in veto_list]
                original_count = len(result)
                result = result[~result['instrument'].isin(veto_codes)]
                filtered_count = original_count - len(result)

                print(f"\n  🚫 一票否决: {filtered_count} 只")
                for item in veto_list[:5]:  # 只打印前5个
                    print(f"     • {item['instrument']}: {item['reason']}")
                if len(veto_list) > 5:
                    print(f"     ... 还有 {len(veto_list) - 5} 只")
            else:
                print(f"\n  ✅ 一票否决检查: 未发现风险股票")

        # 加分增强
        if enable_boost:
            # 确保result中有position或ml_score列
            score_col = 'ml_score' if 'ml_score' in result.columns else 'position'

            if score_col in result.columns:
                boost_count = 0
                boost_examples = []  # 记录加分示例
                
                for ts_code, sentiment in sentiment_results.items():
                    if sentiment['boost_score'] > 0.01:  # 至少要有1%的加分才记录
                        # 更新result中对应股票的评分
                        mask = result['instrument'] == ts_code
                        if mask.any():
                            old_score = result.loc[mask, score_col].values[0]
                            new_score = old_score * (1 + sentiment['boost_score'])
                            result.loc[mask, score_col] = new_score
                            boost_count += 1
                            
                            # 记录示例（最多记录5个）
                            if len(boost_examples) < 5:
                                boost_examples.append({
                                    'code': ts_code,
                                    'boost': sentiment['boost_score'],
                                    'reasons': sentiment['boost_reasons'][:3]  # 最多3个原因
                                })

                if boost_count > 0:
                    print(f"\n  📈 加分增强: {boost_count} 只")
                    # 显示加分示例
                    for example in boost_examples:
                        reasons_str = ', '.join(example['reasons']) if example['reasons'] else '题材加分'
                        print(f"     • {example['code']}: +{example['boost']:.1%} ({reasons_str})")
                    if boost_count > len(boost_examples):
                        print(f"     ... 还有 {boost_count - len(boost_examples)} 只")
                else:
                    print(f"\n  ℹ️  加分增强: 未发现可加分股票")
            else:
                print(f"\n  ⚠️  加分增强: 未找到评分列({score_col})")

        # 重新排序
        if 'ml_score' in result.columns:
            result = result.sort_values('ml_score', ascending=False)  # type: ignore
        elif 'position' in result.columns:
            result = result.sort_values('position', ascending=False)  # type: ignore

        print(f"\n  ✅ 舆情风控完成: {len(selected_stocks)} → {len(result)} 只")

        return result.reset_index(drop=True)  # type: ignore

    def generate_sentiment_report(self, selected_stocks: pd.DataFrame,
                                  filtered_stocks: pd.DataFrame) -> Dict:
        """
        生成舆情分析报告

        Returns:
            {
                'original_count': int,
                'filtered_count': int,
                'veto_count': int,
                'boost_count': int,
                'summary': str
            }
        """
        report = {
            'original_count': len(selected_stocks),
            'filtered_count': len(filtered_stocks),
            'veto_count': len(selected_stocks) - len(filtered_stocks),
            'boost_count': 0,
            'summary': ''
        }

        # 生成摘要
        summary_lines = [
            f"原始选股: {report['original_count']} 只",
            f"一票否决: {report['veto_count']} 只",
            f"最终通过: {report['filtered_count']} 只",
        ]

        report['summary'] = '\n'.join(summary_lines)

        return report


# ============================================================================
# 第五部分：便捷函数
# ============================================================================

def apply_sentiment_control(selected_stocks: pd.DataFrame,
                            factor_data: pd.DataFrame,
                            price_data: pd.DataFrame,
                            tushare_token: Optional[str] = None,
                            cache_manager=None,
                            enable_veto: bool = True,
                            enable_boost: bool = True,
                            lookback_days: int = 30) -> pd.DataFrame:
    """
    便捷函数：一键应用舆情风控

    使用示例:
    ```python
    from sentiment_risk_control import apply_sentiment_control

    filtered = apply_sentiment_control(
        selected_stocks=top_stocks,
        factor_data=factor_data,
        price_data=price_data,
        tushare_token=YOUR_TOKEN
    )
    ```
    """
    controller = SentimentRiskController(
        tushare_token=tushare_token,
        cache_manager=cache_manager,
        lookback_days=lookback_days
    )

    return controller.apply_sentiment_filter(
        selected_stocks=selected_stocks,
        factor_data=factor_data,
        price_data=price_data,
        enable_veto=enable_veto,
        enable_boost=enable_boost
    )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("舆情风控模块 - 独立测试")
    print("=" * 80)

    # 模拟数据
    test_stocks = pd.DataFrame({
        'instrument': ['000001.SZ', '600000.SH', '000002.SZ'],
        'position': [0.95, 0.92, 0.88],
        'date': ['2024-01-15'] * 3
    })

    test_factor_data = pd.DataFrame({
        'date': ['2024-01-15'] * 3,
        'instrument': ['000001.SZ', '600000.SH', '000002.SZ'],
        'position': [0.95, 0.92, 0.88]
    })

    test_price_data = pd.DataFrame({
        'date': ['2024-01-15'] * 3,
        'instrument': ['000001.SZ', '600000.SH', '000002.SZ'],
        'close': [10.0, 15.0, 20.0]
    })

    print("\n✓ 模块加载成功！")
    print("\n使用方法:")
    print("```python")
    print("from sentiment_risk_control import SentimentRiskController")
    print("")
    print("controller = SentimentRiskController(tushare_token=YOUR_TOKEN)")
    print("filtered = controller.apply_sentiment_filter(")
    print("    selected_stocks=top_stocks,")
    print("    factor_data=factor_data,")
    print("    price_data=price_data")
    print(")")
    print("```")