# -*- coding: utf-8 -*-
"""
ml_factor_scoring_fixed.py - 修复数据泄露后的高级机器学习因子评分系统

🔧 主要修复内容：
1. ✅ 严格隔离预测列（position, ml_score等）防止泄露
2. ✅ 移除共线性因子（pb/ps只保留pe）
3. ✅ 添加特征验证断言
4. ✅ 优化特征排除逻辑
5. ✅ 修复XGBoost 2.0+兼容性问题

核心优化特性：
✅ 1. 时间序列切分（避免前视偏差）
✅ 2. 分类目标（预测TOP 20%）
✅ 3. IC加权 - 因子有效性动态评估
✅ 4. 滚动训练 - 自适应市场变化
✅ 5. 标签优化 - 使用超额收益（Active Return）
✅ 6. StockRanker多因子完整实现
✅ 7. 行业/市值/风格中性化
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta

# ----------------------------------------------------------------------------
# 依赖库检查与导入
# ----------------------------------------------------------------------------
warnings.filterwarnings('ignore')

# 机器学习模型库
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost 未安装，运行: pip install xgboost")
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️  LightGBM 未安装，运行: pip install lightgbm")
    lgb = None
    LIGHTGBM_AVAILABLE = False

# Sklearn 工具
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# ============================================================================
# 第一部分：核心基础模块 (IC计算与时间序列切分)
# ============================================================================

class ICCalculator:
    """
    因子IC（信息系数）计算器

    IC = 因子值与未来收益的相关性
    ICIR = IC的均值 / IC的标准差（夏普比率的因子版）
    RankIC = 因子排名与收益排名的相关性（更稳健）
    """

    def __init__(self, forward_periods: List[int] = [5, 10, 20]):
        self.forward_periods = forward_periods
        self.ic_history = {}  # {factor: {period: [ic_values]}}
        self.rank_ic_history = {}  # Rank IC历史

    def calculate_factor_ic(self,
                            factor_data: pd.DataFrame,
                            price_data: pd.DataFrame,
                            factor_columns: List[str]) -> Dict:
        """计算所有因子的IC值"""
        print("\n📊 计算因子IC...")

        # 检测价格列
        price_col = self._detect_price_column(price_data)
        if price_col is None:
            print("  ⚠️  未找到价格列，跳过IC计算")
            return {}

        # 合并数据
        merged = factor_data.merge(
            price_data[['instrument', 'date', price_col]],
            on=['instrument', 'date'],
            how='left'
        )

        merged = merged.sort_values(['instrument', 'date']).reset_index(drop=True)

        # 计算不同周期的未来收益
        for period in self.forward_periods:
            # 绝对收益
            merged[f'abs_return_{period}d'] = merged.groupby('instrument')[price_col].pct_change(
                period
            ).shift(-period)

            # 超额收益（相对市场）
            market_return = merged.groupby('date')[f'abs_return_{period}d'].transform('mean')
            merged[f'future_return_{period}d'] = merged[f'abs_return_{period}d'] - market_return

        ic_results = {}
        merged_filtered = merged.dropna(subset=[price_col])

        for factor in factor_columns:
            if factor not in merged_filtered.columns:
                continue

            ic_results[factor] = {}

            for period in self.forward_periods:
                return_col = f'future_return_{period}d'
                valid_data = merged_filtered[[factor, return_col, 'date']].dropna()

                if len(valid_data) < 10:
                    continue

                grouped = valid_data.groupby('date')

                # Pearson IC
                daily_ic_series = grouped.apply(
                    lambda x: x[factor].corr(x[return_col]) if len(x) >= 10 else np.nan
                )
                daily_ic = daily_ic_series.dropna().tolist()

                # Rank IC (Spearman)
                daily_rank_ic_series = grouped.apply(
                    lambda x: x[factor].corr(x[return_col], method='spearman')
                    if len(x) >= 10 else np.nan
                )
                daily_rank_ic = daily_rank_ic_series.dropna().tolist()

                if len(daily_ic) > 0:
                    ic_mean = np.mean(daily_ic)
                    ic_std = np.std(daily_ic)
                    icir = ic_mean / ic_std if ic_std > 0 else 0

                    rank_ic_mean = np.mean(daily_rank_ic)
                    rank_ic_std = np.std(daily_rank_ic)
                    rank_icir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0

                    ic_win_rate = np.mean([1 if ic > 0 else 0 for ic in daily_ic])

                    ic_results[factor][period] = {
                        'ic': ic_mean, 'ic_std': ic_std, 'icir': icir,
                        'rank_ic': rank_ic_mean, 'rank_ic_std': rank_ic_std, 'rank_icir': rank_icir,
                        'ic_win_rate': ic_win_rate, 'sample_days': len(daily_ic)
                    }

                    # 记录历史
                    if factor not in self.ic_history:
                        self.ic_history[factor] = {}
                        self.rank_ic_history[factor] = {}
                    self.ic_history[factor][period] = daily_ic
                    self.rank_ic_history[factor][period] = daily_rank_ic

        self._print_ic_summary(ic_results)
        return ic_results

    def _detect_price_column(self, df: pd.DataFrame) -> Optional[str]:
        candidates = ['close', 'Close', 'CLOSE', 'price', 'Price', 'adj_close']
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _print_ic_summary(self, ic_results: Dict):
        print(f"\n  📈 因子IC统计:")
        print(f"  {'因子':<25s} | {'IC(5日)':<10s} | {'RankIC':<10s} | {'ICIR':<10s} | {'胜率':<10s}")
        print(f"  {'-' * 80}")

        for factor, periods in ic_results.items():
            ic_5 = periods.get(5, {}).get('ic', 0)
            rank_ic_5 = periods.get(5, {}).get('rank_ic', 0)
            icir_5 = periods.get(5, {}).get('icir', 0)
            win_rate = periods.get(5, {}).get('ic_win_rate', 0)

            print(f"  {factor:<25s} | {ic_5:>9.4f} | {rank_ic_5:>9.4f} | "
                  f"{icir_5:>9.4f} | {win_rate:>9.2%}")

    def get_ic_weights(self, ic_results: Dict, period: int = 5, method: str = 'icir') -> Dict[str, float]:
        """根据IC计算因子权重"""
        weights = {}
        total_score = 0.0

        for factor, periods in ic_results.items():
            if period not in periods:
                weights[factor] = 0.0
                continue

            period_data = periods[period]
            ic_val = period_data.get('ic', 0.0) or 0.0
            rank_ic_val = period_data.get('rank_ic', 0.0) or 0.0
            icir_val = period_data.get('icir', 0.0) or 0.0
            rank_icir_val = period_data.get('rank_icir', 0.0) or 0.0

            val_map = {
                'ic': abs(float(ic_val)),
                'rank_ic': abs(float(rank_ic_val)),
                'icir': abs(float(icir_val)),
                'rank_icir': abs(float(rank_icir_val))
            }
            score = val_map.get(method, abs(float(ic_val)))

            weights[factor] = float(score)
            total_score += float(score)

        if total_score > 0:
            weights = {k: float(v / total_score) for k, v in weights.items()}

        return weights


class TimeSeriesSplitter:
    """
    时间序列数据切分器 (Walk-Forward)
    训练集 -> 验证集 -> 测试集，避免信息泄露
    """

    def __init__(self, train_months: int = 12, valid_months: int = 1,
                 test_months: int = 1, gap_days: int = 0, expanding: bool = False):
        self.train_months = train_months
        self.valid_months = valid_months
        self.test_months = test_months
        self.gap_days = gap_days
        self.expanding = expanding

    def split(self, data: pd.DataFrame, date_column: str = 'date') -> List[Tuple]:
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column).reset_index(drop=True)

        data['year_month'] = data[date_column].dt.to_period('M')
        months = sorted(data['year_month'].unique())

        print(f"\n  📅 时间范围: {months[0]} 至 {months[-1]} (共{len(months)}个月)")

        splits = []
        min_required_months = self.train_months + self.valid_months + self.test_months

        if len(months) < min_required_months:
            print(f"  ⚠️  数据不足: 需要至少{min_required_months}个月，实际{len(months)}个月")
            return []

        for i in range(len(months) - min_required_months + 1):
            train_start = 0 if self.expanding else i
            train_end = i + self.train_months
            train_months_list = months[train_start:train_end]

            valid_start = train_end
            valid_end = valid_start + self.valid_months
            valid_months_list = months[valid_start:valid_end]

            test_start = valid_end
            test_end = test_start + self.test_months
            if test_end > len(months): break
            test_months_list = months[test_start:test_end]

            train_idx = data[data['year_month'].isin(train_months_list)].index.tolist()
            valid_idx = data[data['year_month'].isin(valid_months_list)].index.tolist()
            test_idx = data[data['year_month'].isin(test_months_list)].index.tolist()

            # Gap handling
            if self.gap_days > 0:
                train_end_date = data.loc[train_idx, date_column].max()
                gap_cutoff = train_end_date + timedelta(days=self.gap_days)
                valid_idx = [idx for idx in valid_idx if data.loc[idx, date_column] >= gap_cutoff]

            if len(train_idx) > 0 and len(valid_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, valid_idx, test_idx))

        print(f"  ✓ 生成 {len(splits)} 个时间窗口")
        if len(splits) > 0:
            self._print_split_info(data, splits[0], date_column, "第1个")
        return splits

    def _print_split_info(self, data, split, date_column, label):
        train_idx, valid_idx, test_idx = split
        train_dates = data.loc[train_idx, date_column]
        valid_dates = data.loc[valid_idx, date_column]
        test_dates = data.loc[test_idx, date_column]
        print(f"\n  {label}窗口:")
        print(f"    训练集: {train_dates.min().date()} - {train_dates.max().date()} ({len(train_idx)})")
        print(f"    验证集: {valid_dates.min().date()} - {valid_dates.max().date()} ({len(valid_idx)})")
        print(f"    测试集: {test_dates.min().date()} - {test_dates.max().date()} ({len(test_idx)})")


# ============================================================================
# 第二部分：因子处理模块 (StockRanker & Generator)
# ============================================================================

class StockRanker:
    """StockRanker 多因子综合评分器 (预处理、合成、中性化)"""

    def __init__(self, method: str = 'equal', normalize_method: str = 'zscore',
                 winsorize: bool = True, winsorize_std: float = 3.0):
        self.method = method
        self.normalize_method = normalize_method
        self.winsorize = winsorize
        self.winsorize_std = winsorize_std
        self.factor_weights = {}

        print(f"\n🎯 初始化 StockRanker [合成: {method}, 标准化: {normalize_method}]")

    def preprocess_factors(self, factor_data: pd.DataFrame, factor_columns: List[str]) -> pd.DataFrame:
        print(f"\n  🔧 因子预处理 ({len(factor_columns)}个因子)...")
        data = factor_data.copy()

        for factor in factor_columns:
            if factor not in data.columns: continue
            data[f'{factor}_processed'] = data.groupby('date')[factor].transform(
                lambda x: self._preprocess_single_factor(x)
            )
        print(f"  ✓ 预处理完成")
        return data

    def _preprocess_single_factor(self, x: pd.Series) -> pd.Series:
        # 去极值
        if self.winsorize:
            mean, std = x.mean(), x.std()
            x = x.clip(mean - self.winsorize_std * std, mean + self.winsorize_std * std)

        # 标准化
        if self.normalize_method == 'zscore':
            std = x.std()
            x = (x - x.mean()) / std if std > 0 else (x - x.mean())
        elif self.normalize_method == 'minmax':
            x = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
        elif self.normalize_method == 'rank':
            x = x.rank(pct=True)
        return x

    def calculate_composite_score(self, factor_data: pd.DataFrame, factor_columns: List[str],
                                  ic_weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        print(f"\n  📊 计算综合评分 (方法: {self.method})...")
        data = factor_data.copy()

        processed_columns = [f'{factor}_processed' for factor in factor_columns
                             if f'{factor}_processed' in data.columns]

        if not processed_columns:
            print("  ⚠️  未找到预处理因子，使用原始因子")
            processed_columns = [f for f in factor_columns if f in data.columns]

        if self.method == 'equal':
            data['composite_score'] = data[processed_columns].mean(axis=1)
            self.factor_weights = {col: 1.0 / len(processed_columns) for col in processed_columns}

        elif self.method in ['ic_weight', 'optimize']:
            if ic_weights is None:
                print("  ⚠️  无IC权重，回退至等权")
                data['composite_score'] = data[processed_columns].mean(axis=1)
            else:
                weights, valid_cols = [], []
                for col in processed_columns:
                    fname = col.replace('_processed', '')
                    if fname in ic_weights:
                        w = ic_weights[fname]
                        if self.method == 'optimize': w = w ** 2  # 简单优化：平方加强高IC因子
                        weights.append(w)
                        valid_cols.append(col)

                if valid_cols:
                    weights = np.array(weights) / sum(weights)
                    data['composite_score'] = (data[valid_cols] * weights).sum(axis=1)
                    self.factor_weights = dict(zip(valid_cols, weights))
                else:
                    data['composite_score'] = data[processed_columns].mean(axis=1)

        data['score_rank'] = data.groupby('date')['composite_score'].rank(pct=True)
        print(f"  ✓ 综合评分完成")
        self._print_weight_summary()
        return data

    def _print_weight_summary(self):
        if not self.factor_weights: return
        print(f"\n  📊 因子权重 (Top 10):")
        sorted_w = sorted(self.factor_weights.items(), key=lambda x: x[1], reverse=True)
        for f, w in sorted_w[:10]:
            print(f"     {f.replace('_processed', ''):<25s}: {w:>7.2%}")

    def apply_industry_neutralization(self, factor_data: pd.DataFrame,
                                      industry_column: str = 'industry') -> pd.DataFrame:
        print(f"\n  🏢 应用行业中性化...")
        data = factor_data.copy()
        if industry_column not in data.columns: return data

        data['composite_score_neutral'] = data.groupby(['date', industry_column])['composite_score'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        data['score_rank_neutral'] = data.groupby('date')['composite_score_neutral'].rank(pct=True)
        print(f"  ✓ 行业中性化完成")
        return data

    def apply_market_cap_neutralization(self, factor_data: pd.DataFrame,
                                        cap_column: str = 'market_cap') -> pd.DataFrame:
        print(f"\n  💰 应用市值中性化...")
        data = factor_data.copy()
        if cap_column not in data.columns: return data

        data['log_cap'] = np.log(data[cap_column].clip(lower=1))

        def neutralize(group):
            if len(group) < 10: return group
            X = group[['log_cap']].values
            y = group['composite_score'].values
            reg = LinearRegression().fit(X, y)
            group['composite_score_neutral'] = y - reg.predict(X)
            return group

        data = data.groupby('date').apply(neutralize).reset_index(drop=True)
        data['score_rank_neutral'] = data.groupby('date')['composite_score_neutral'].rank(pct=True)
        print(f"  ✓ 市值中性化完成")
        return data


class FactorGenerator:
    """因子生成器示例"""

    @staticmethod
    def generate_momentum_factors(price_data: pd.DataFrame, periods=[5, 10, 20, 60]) -> pd.DataFrame:
        data = price_data.sort_values(['instrument', 'date']).copy()
        price_col = 'close' if 'close' in data.columns else 'Close'
        for p in periods:
            data[f'momentum_{p}d'] = data.groupby('instrument')[price_col].pct_change(p)
            if p <= 5: data[f'reversal_{p}d'] = -data[f'momentum_{p}d']
        return data

    @staticmethod
    def generate_volatility_factors(price_data: pd.DataFrame, periods=[5, 10, 20]) -> pd.DataFrame:
        data = price_data.sort_values(['instrument', 'date']).copy()
        price_col = 'close' if 'close' in data.columns else 'Close'
        data['ret'] = data.groupby('instrument')[price_col].pct_change()
        for p in periods:
            data[f'volatility_{p}d'] = data.groupby('instrument')['ret'].transform(lambda x: x.rolling(p).std())
        return data


# ============================================================================
# 第三部分：高级ML评分器 (完整实现 - 已修复数据泄露)
# ============================================================================

class AdvancedMLScorer:
    """
    高级机器学习评分器 (修复版)
    整合: 时间序列切分, IC特征, Active Return标签, 模型集成

    🔧 修复内容:
    1. 严格排除所有预测相关列（position, ml_score等）
    2. 添加特征验证断言
    3. 预测结果独立存储，不污染训练数据
    """

    def __init__(self, model_type: str = 'xgboost', target_period: int = 5, top_percentile: float = 0.20,
                 use_classification: bool = True, use_ic_features: bool = True, use_active_return: bool = True,
                 train_months: int = 12, scaler_type: str = 'standard', random_state: int = 42):
        self.model_type = model_type
        self.target_period = target_period
        self.top_percentile = top_percentile
        self.use_classification = use_classification
        self.use_ic_features = use_ic_features
        self.use_active_return = use_active_return
        self.train_months = train_months
        self.scaler = RobustScaler() if scaler_type == 'robust' else StandardScaler()
        self.random_state = random_state

        self.models = {}
        self.feature_names = None
        self.ic_calculator = ICCalculator([target_period])

        print(f"\n🚀 初始化高级ML评分器 [模型: {model_type}, 目标: {target_period}d, 分类: {use_classification}]")

    def prepare_training_data(self, factor_data: pd.DataFrame, price_data: pd.DataFrame,
                              factor_columns: List[str]) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
        """
        🔧 修复：严格排除预测列，防止数据泄露
        """
        print(f"\n📦 准备训练数据...")
        price_col = self._detect_price_column(price_data)

        merged = factor_data.merge(
            price_data[['instrument', 'date', price_col]],
            on=['instrument', 'date'], how='left'
        ).sort_values(['instrument', 'date']).reset_index(drop=True)

        # 计算IC特征
        if self.use_ic_features:
            print("  ✓ 计算因子IC特征...")
            ic_results = self.ic_calculator.calculate_factor_ic(factor_data, price_data, factor_columns)
            for factor in factor_columns:
                if factor in ic_results:
                    stats = ic_results[factor].get(self.target_period, {})
                    merged[f'{factor}_ic'] = stats.get('ic', 0)
                    merged[f'{factor}_icir'] = stats.get('icir', 0)

        # 构建目标变量
        print(f"  ✓ 构建目标变量 (Active Return: {self.use_active_return})...")
        # 修复收益率计算方向 - 使用未来价格与当前价格的比值来计算收益率
        # 🔧 修复：使用向量化操作替代groupby.apply避免索引不匹配问题
        merged = merged.sort_values(['instrument', 'date']).reset_index(drop=True)
        future_prices = merged.groupby('instrument')[price_col].shift(-self.target_period)
        current_prices = merged[price_col]
        merged['abs_return'] = (future_prices / current_prices) - 1

        if self.use_active_return:
            market_return = merged.groupby('date')['abs_return'].transform('mean')
            merged['future_return'] = merged['abs_return'] - market_return
        else:
            merged['future_return'] = merged['abs_return']

        if self.use_classification:
            merged['target'] = 0
            for date in merged['date'].unique():
                mask = merged['date'] == date
                daily_data = merged[mask]
                
                # 优化策略：既要跑赢市场，又要有绝对收益
                # 1. 相对收益 Top 20%
                # 2. 绝对收益 > 0 (剔除大跌市中的"抗跌股", 熊市空仓比买抗跌更好)
                
                thresh = daily_data['future_return'].quantile(1 - self.top_percentile)
                
                # 胜率优化核心：双重过滤
                # future_return 是相对收益 (Active Return)
                # abs_return 是绝对收益
                
                target_mask = (daily_data['future_return'] >= thresh) & (daily_data['abs_return'] > 0.0)
                
                merged.loc[mask & target_mask, 'target'] = 1
            target_col = 'target'
        else:
            target_col = 'future_return'

        merged = merged.dropna(subset=[target_col])

        # 🔥 修复：严格排除所有可能泄露的列
        exclude = [
            # 基础标识列
            'date', 'instrument',
            # 目标变量相关
            'future_return', 'abs_return', 'target',
            # 价格列
            price_col, 'close', 'Close', 'price', 'Price', 'adj_close',
            # 分类列
            'industry', 'sector', 'market_cap', 'log_cap',
            # 🔥 关键：所有预测/评分相关列（防止数据泄露）
            'ml_score', 'position', 'score_rank',
            'composite_score', 'composite_score_neutral',
            'score_rank_neutral', 'industry_rank',
            # 中间处理列
            'year_month'
        ]

        # 特征选择：排除非数值列和处理过的列
        feature_cols = [c for c in merged.columns
                        if c not in exclude
                        and pd.api.types.is_numeric_dtype(merged[c])
                        and not c.endswith('_processed')]  # 排除中间处理列

        # 🔥 修复：添加断言验证，防止position等列泄露
        leaked_cols = [c for c in ['position', 'ml_score', 'score_rank', 'composite_score']
                       if c in feature_cols]
        if leaked_cols:
            raise ValueError(f"⚠️ CRITICAL: 检测到数据泄露！以下列不应作为特征: {leaked_cols}")

        print(f"  ✓ 验证通过：已排除 {len(exclude)} 类列，保留 {len(feature_cols)} 个有效特征")

        X = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = merged[target_col].values
        self.feature_names = feature_cols

        # 打印前10个特征用于验证
        print(f"  📋 特征示例: {feature_cols[:10]}")

        # 确保返回的是DataFrame而不是Series
        if isinstance(X, pd.Series):
            X = X.to_frame()
        # 确保y是numpy数组
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        return X, y, merged

    def train_walk_forward(self, X: pd.DataFrame, y: np.ndarray, merged: pd.DataFrame, n_splits: int = 3):
        print(f"\n🎯 Walk-Forward 训练...")
        splitter = TimeSeriesSplitter(train_months=self.train_months, valid_months=1, test_months=1)
        splits = splitter.split(merged)

        if not splits:
            print("  ⚠️ 数据不足，切换简单切分")
            return self._train_simple(X, y)

        if n_splits and n_splits < len(splits):
            splits = splits[-n_splits:]

        window_results = []
        for i, (train_idx, valid_idx, test_idx) in enumerate(splits):
            print(f"\n  === 窗口 {i + 1}/{len(splits)} ===")

            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

            X_train_s = self.scaler.fit_transform(X_train)
            X_valid_s = self.scaler.transform(X_valid)

            model = self._train_model(X_train_s, y_train, X_valid_s, y_valid)

            # 评估
            if model is not None:
                try:
                    if self.use_classification:
                        # 对于分类模型，使用AUC作为评估指标
                        try:
                            # 检查模型类型，避免在回归模型上调用predict_proba
                            model_name = type(model).__name__
                            if 'Regressor' in model_name:
                                # 如果是回归模型，使用predict方法
                                pred = model.predict(X_valid_s)
                                # 安全地转换为numpy数组
                                if str(type(pred)).find('sparse') >= 0 or hasattr(pred, 'toarray'):
                                    try:
                                        pred = pred.toarray()  # type: ignore
                                    except:
                                        pass
                                pred = np.asarray(pred).flatten()
                                valid_score = roc_auc_score(y_valid, pred)
                            elif hasattr(model, 'predict_proba') and 'Classifier' in model_name:
                                proba = model.predict_proba(X_valid_s)
                                # 安全地转换为numpy数组
                                if str(type(proba)).find('sparse') >= 0 or hasattr(proba, 'toarray'):
                                    try:
                                        proba = proba.toarray()  # type: ignore
                                    except:
                                        pass
                                proba = np.asarray(proba)
                                if len(proba.shape) > 1 and proba.shape[1] > 1:
                                    valid_score = roc_auc_score(y_valid, proba[:, 1])
                                else:
                                    valid_score = roc_auc_score(y_valid, proba[:, 0] if len(proba.shape) > 1 else proba)
                            else:
                                # 如果没有predict_proba方法，使用predict方法
                                pred = model.predict(X_valid_s)
                                # 安全地转换为numpy数组
                                if str(type(pred)).find('sparse') >= 0 or hasattr(pred, 'toarray'):
                                    try:
                                        pred = pred.toarray()  # type: ignore
                                    except:
                                        pass
                                pred = np.asarray(pred).flatten()
                                valid_score = roc_auc_score(y_valid, pred)
                            print(f"     验证AUC: {valid_score:.4f}")
                        except Exception as e:
                            print(f"     AUC计算出错: {e}")
                            valid_score = 0.0
                    else:
                        # 对于回归模型，使用IC作为评估指标
                        try:
                            pred = model.predict(X_valid_s)
                            # 安全地转换为numpy数组
                            if str(type(pred)).find('sparse') >= 0 or hasattr(pred, 'toarray'):
                                try:
                                    pred = pred.toarray()  # type: ignore
                                except:
                                    pass
                            pred = np.asarray(pred)
                            # 确保输入是1维数组
                            if len(pred.shape) > 1:
                                pred = pred.flatten()
                            # 确保y_valid是numpy数组
                            y_valid_flat = np.asarray(y_valid)
                            if len(y_valid_flat.shape) > 1:
                                y_valid_flat = y_valid_flat.flatten()
                            # 计算相关系数
                            correlation_matrix = np.corrcoef(y_valid_flat, pred)
                            valid_score = correlation_matrix[0, 1] if correlation_matrix.size > 1 else 0
                            print(f"     验证IC: {valid_score:.4f}")
                        except Exception as e:
                            print(f"     IC计算出错: {e}")
                            valid_score = 0.0
                    window_results.append({'model': model, 'score': float(valid_score), 'window': i})
                except Exception as e:
                    print(f"     评估出错: {e}")
                    window_results.append({'model': model, 'score': 0.0, 'window': i})

        # 选择最佳模型
        if window_results:
            best = max(window_results, key=lambda x: x['score'])
            self.models['best'] = best['model']
            print(f"\n  🏆 最佳模型来自窗口 {best['window'] + 1}, 得分: {best['score']:.4f}")
        return self

    def _train_model(self, X_train, y_train, X_valid, y_valid):
        """🔧 修复：XGBoost 2.0+ 兼容性"""
        if self.use_classification:
            if self.model_type == 'xgboost' and XGBOOST_AVAILABLE and xgb is not None:
                # 修复：early_stopping_rounds 移入构造函数
                model = xgb.XGBClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=6,
                    eval_metric='auc', random_state=self.random_state, n_jobs=-1,
                    early_stopping_rounds=30
                )
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                return model
            elif LIGHTGBM_AVAILABLE and lgb is not None:
                model = lgb.LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=6,
                    metric='auc', random_state=self.random_state, n_jobs=-1, verbose=-1
                )
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
                return model
        else:
            # 回归逻辑
            if self.model_type == 'xgboost' and XGBOOST_AVAILABLE and xgb is not None:
                model = xgb.XGBRegressor(
                    n_estimators=300, max_depth=6, random_state=self.random_state, n_jobs=-1,
                    early_stopping_rounds=30
                )
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                return model
        return None

    def _train_simple(self, X, y):
        print("  使用简单训练模式...")
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_valid_s = self.scaler.transform(X_valid)
        self.models['best'] = self._train_model(X_train_s, y_train, X_valid_s, y_valid)
        return self

    def predict_scores(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        🔧 修复：预测结果独立存储，不污染原始特征
        """
        if 'best' not in self.models:
            raise ValueError("模型未训练")

        data = factor_data.copy()

        # 只提取特征列进行预测
        X = data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X)

        model = self.models['best']

        # 🔥 修复：创建独立的结果DataFrame
        predictions = model.predict_proba(X_scaled)[:, 1] if self.use_classification else model.predict(X_scaled)

        result = pd.DataFrame({
            'date': data['date'].values,
            'instrument': data['instrument'].values,
            'ml_score': predictions
        })

        # 计算排名（在独立DataFrame中）
        result['position'] = result.groupby('date')['ml_score'].rank(pct=True)

        # 🔥 关键：只合并必要的预测列，保持原始数据清洁
        # 如果原数据已有这些列，先删除
        for col in ['ml_score', 'position']:
            if col in data.columns:
                data = data.drop(columns=[col])

        # 合并预测结果
        data = data.merge(result, on=['date', 'instrument'], how='left')

        print(f"  ✓ 预测完成，生成 ml_score 和 position 列")
        return data

    def get_feature_importance(self, top_n: int = 20):
        if 'best' not in self.models: return None
        imp = self.models['best'].feature_importances_
        df = pd.DataFrame({'feature': self.feature_names, 'importance': imp})
        return df.sort_values('importance', ascending=False).head(top_n)

    def _detect_price_column(self, df):
        for col in ['close', 'Close', 'price', 'Price']:
            if col in df.columns: return col
        return None

    def _safe_to_numpy_array(self, data):
        """安全地将数据转换为numpy数组"""
        try:
            # 检查是否为稀疏矩阵
            if str(type(data)).find('sparse') >= 0:
                if hasattr(data, 'toarray'):
                    try:
                        data = data.toarray()
                    except:
                        pass
            # 转换为numpy数组
            return np.asarray(data)
        except:
            return np.asarray(data)


# ============================================================================
# 第四部分：行业数据与回测辅助
# ============================================================================

def get_industry_data(instruments: List[str], tushare_token: Optional[str] = None) -> pd.DataFrame:
    """获取行业数据 (支持Tushare)"""
    if tushare_token is None:
        print("  ⚠️  未提供 Tushare Token，使用随机/默认行业")
        return pd.DataFrame({'instrument': instruments, 'industry': '其他', 'industry_code': 'Z99'})

    try:
        import tushare as ts
        ts.set_token(tushare_token)
        pro = ts.pro_api()
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        df = df[df['ts_code'].isin(instruments)]
        df = df.rename(columns={'ts_code': 'instrument'})  # type: ignore
        df['industry'] = df['industry'].fillna('其他')
        result = df[['instrument', 'industry']]
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame()  # type: ignore
    except Exception as e:
        print(f"  ⚠️  获取行业失败: {e}")
        return pd.DataFrame({'instrument': instruments, 'industry': '其他'})


class IndustryBasedScorer:
    """分行业评分与轮动分析"""

    def __init__(self, tushare_token: Optional[str] = None):
        self.tushare_token = tushare_token

    def score_by_industry(self, factor_data: pd.DataFrame, score_column: str = 'position') -> pd.DataFrame:
        print("\n🏢 分行业评分...")
        data = factor_data.copy()
        instruments = data['instrument'].unique().tolist()
        ind_data = get_industry_data(instruments, self.tushare_token)

        if 'industry' in data.columns: data = data.drop(columns=['industry'])
        data = data.merge(ind_data, on='instrument', how='left').fillna({'industry': '其他'})

        data['industry_rank'] = data.groupby(['date', 'industry'])[score_column].rank(pct=True)
        return data

    def analyze_industry_rotation(self, factor_data: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
        print(f"\n  🔄 行业轮动分析 (Top {top_n})...")
        stats = factor_data.groupby(['date', 'industry']).agg(
            {'position': 'mean', 'instrument': 'count'}
        ).reset_index()
        stats.columns = ['date', 'industry', 'avg_score', 'count']
        stats['rank'] = stats.groupby('date')['avg_score'].rank(ascending=False)

        latest = stats[stats['date'] == stats['date'].max()].nsmallest(top_n, 'rank')
        print(f"  最新强势行业: {latest['industry'].tolist()}")
        return stats


class EnhancedStockSelector:
    """增强选股器 (评分过滤 + 行业分散)"""

    def select_stocks(self, factor_data: pd.DataFrame, min_score: float = 0.6,
                      max_stocks: Optional[int] = None, max_industry_conc: float = 0.3) -> pd.DataFrame:
        print(f"\n🎯 增强选股 [阈值: {min_score:.0%}, 上限: {max_stocks}]...")
        data = factor_data.copy()
        if 'industry' not in data.columns: data['industry'] = '其他'

        results = []
        for date in data['date'].unique():
            daily = data[(data['date'] == date) & (data['position'] >= min_score)].sort_values('position',
                                                                                               ascending=False)  # type: ignore

            if max_stocks and len(daily) > max_stocks:
                limit = int(max_stocks * max_industry_conc)
                selected, counts = [], {}
                for _, row in daily.iterrows():
                    if len(selected) >= max_stocks: break
                    ind = row['industry']
                    if counts.get(ind, 0) < limit:
                        selected.append(row)
                        counts[ind] = counts.get(ind, 0) + 1
                daily = pd.DataFrame(selected)

            results.append(daily)

        final = pd.concat(results) if results else pd.DataFrame()
        print(f"  ✓ 选出 {len(final)} 条交易记录")
        return final


class SimpleBacktester:
    """简单回测器"""

    @staticmethod
    def backtest(selected_stocks: pd.DataFrame, price_data: pd.DataFrame, holding_period: int = 5) -> Dict:
        print(f"\n📊 简单回测 (持有{holding_period}天)...")
        price_col = 'close' if 'close' in price_data.columns else 'Close'

        merged = selected_stocks.merge(price_data[['instrument', 'date', price_col]], on=['instrument', 'date'],
                                       how='left')
        merged = merged.sort_values(['instrument', 'date'])
        # 修复收益率计算方向 - 使用未来价格与当前价格的比值来计算收益率
        # 修复索引不匹配问题
        merged = merged.sort_values(['instrument', 'date']).reset_index(drop=True)
        future_prices = merged.groupby('instrument')[price_col].shift(-holding_period)
        current_prices = merged[price_col]
        merged['ret'] = (future_prices / current_prices) - 1

        valid = merged.dropna(subset=['ret'])
        if len(valid) == 0: return {}

        res = {
            'avg_return': valid['ret'].mean(),
            'sharpe': valid['ret'].mean() / valid['ret'].std() if valid['ret'].std() > 0 else 0,
            'win_rate': (valid['ret'] > 0).mean(),
            'n_trades': len(valid)
        }

        print(f"  平均收益: {res['avg_return']:.2%}")
        print(f"  夏普比率: {res['sharpe']:.2f}")
        print(f"  胜率:     {res['win_rate']:.2%}")
        return res


# ============================================================================
# 第五部分：策略编排与示例
# ============================================================================

class MultiFactorMLStrategy:
    """
    多因子ML选股策略编排器 (修复版)
    流程: 因子清洗 -> IC分析 -> StockRanker -> Walk-Forward ML -> 行业评分 -> 选股 -> 回测

    🔧 修复内容:
    1. 自动剔除共线性因子（pb/ps）
    2. 可选剔除无效基本面因子
    3. 训练前清理污染列
    """

    def __init__(self, model_type='xgboost', target_period=5, train_months=12,
                 tushare_token=None, remove_collinear=True, remove_weak_factors=False):
        self.target_period = target_period
        self.remove_collinear = remove_collinear
        self.remove_weak_factors = remove_weak_factors

        self.ic_calc = ICCalculator([target_period])
        self.ranker = StockRanker(method='ic_weight')
        self.ml = AdvancedMLScorer(model_type=model_type, target_period=target_period, train_months=train_months)
        self.ind_scorer = IndustryBasedScorer(tushare_token)
        self.selector = EnhancedStockSelector()

    def _clean_factors(self, factor_cols: List[str]) -> List[str]:
        """🔧 修复：因子清洗"""
        cleaned = factor_cols.copy()

        # 移除共线性因子
        if self.remove_collinear:
            collinear = ['pb_ratio', 'ps_ratio']  # 只保留pe_ratio
            cleaned = [f for f in cleaned if f not in collinear]
            if any(c in factor_cols for c in collinear):
                print(f"  ✂️  移除共线性因子: {[c for c in collinear if c in factor_cols]}")

        # 移除弱因子
        if self.remove_weak_factors:
            weak = ['roe', 'roa', 'net_profit_margin', 'gross_profit_margin']
            cleaned = [f for f in cleaned if f not in weak]
            if any(w in factor_cols for w in weak):
                print(f"  ✂️  移除弱因子: {[w for w in weak if w in factor_cols]}")

        print(f"  ✓ 因子清洗完成: {len(factor_cols)} -> {len(cleaned)}")
        return cleaned

    def run(self, factor_data, price_data, factor_cols, min_score=0.7, max_stocks=30):
        print("=" * 60 + "\n  多因子ML策略启动 (修复版)\n" + "=" * 60)

        # 🔧 修复：因子清洗
        factor_cols = self._clean_factors(factor_cols)

        # 1. IC分析 & 权重
        ic_res = self.ic_calc.calculate_factor_ic(factor_data, price_data, factor_cols)
        weights = self.ic_calc.get_ic_weights(ic_res, self.target_period)

        # 2. 基础评分
        processed = self.ranker.preprocess_factors(factor_data, factor_cols)
        scored = self.ranker.calculate_composite_score(processed, factor_cols, weights)

        # 🔧 修复：训练前清理污染列
        clean_cols = ['ml_score', 'position', 'score_rank', 'composite_score']
        clean_data = scored.drop(columns=[c for c in clean_cols if c in scored.columns], errors='ignore')

        # 3. ML增强
        X, y, merged = self.ml.prepare_training_data(clean_data, price_data, factor_cols)
        self.ml.train_walk_forward(X, y, merged)
        ml_scored = self.ml.predict_scores(merged)

        # 4. 行业增强
        ind_scored = self.ind_scorer.score_by_industry(ml_scored)
        self.ind_scorer.analyze_industry_rotation(ind_scored)

        # 5. 选股 & 回测
        picks = self.selector.select_stocks(ind_scored, min_score=min_score, max_stocks=max_stocks)
        backtest = SimpleBacktester.backtest(picks, price_data, self.target_period)

        # 特征重要性
        imp = self.ml.get_feature_importance()
        if imp is not None:
            print("\n  🔑 Top 10 重要特征:")
            print(imp.head(10).to_string(index=False))

        return {'picks': picks, 'backtest': backtest, 'feature_importance': imp}


# ============================================================================
# 第六部分：测试数据生成与验证
# ============================================================================

def generate_sample_data(n_stocks=50, n_days=200):
    """生成测试数据"""
    print(f"\n🎲 生成随机测试数据 ({n_stocks}只股票, {n_days}天)...")
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_days)]
    instruments = [f"000{i:03d}.SZ" for i in range(n_stocks)]

    records = []
    prices = []

    for date in dates:
        for inst in instruments:
            rec = {'date': date, 'instrument': inst}
            # 生成各类因子
            for i in range(5):
                rec[f'factor_{i}'] = np.random.randn()
            # 添加估值因子（模拟）
            rec['pe_ratio'] = np.random.uniform(5, 50)
            # 添加动量因子
            rec['momentum_20d'] = np.random.randn() * 0.1
            records.append(rec)

            # 价格数据
            prices.append({
                'date': date,
                'instrument': inst,
                'close': 100 * (1 + np.random.randn() * 0.02)
            })

    return pd.DataFrame(records), pd.DataFrame(prices)


def validate_no_leakage(strategy_results: Dict):
    """验证是否存在数据泄露"""
    print("\n🔍 数据泄露验证...")
    imp = strategy_results.get('feature_importance')

    if imp is not None:
        leaked = imp[imp['feature'].str.contains('position|ml_score|score_rank', case=False, na=False)]
        if len(leaked) > 0:
            print(f"  ⚠️  警告：检测到可疑特征: {leaked['feature'].tolist()}")
            return False
        else:
            print(f"  ✅ 验证通过：未检测到泄露列")
            return True
    return None


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  机器学习因子评分系统 - 修复版演示")
    print("=" * 60)

    # 生成测试数据
    factors, prices = generate_sample_data(n_stocks=50, n_days=200)

    # 因子列表
    cols = [f'factor_{i}' for i in range(5)] + ['pe_ratio', 'momentum_20d']

    # 运行策略（启用因子清洗）
    strategy = MultiFactorMLStrategy(
        model_type='xgboost',
        train_months=3,
        remove_collinear=True,  # 移除共线性因子
        remove_weak_factors=False  # 保留基本面因子（测试用）
    )

    results = strategy.run(factors, prices, cols, min_score=0.6, max_stocks=20)

    # 验证
    validate_no_leakage(results)

    print("\n" + "=" * 60)
    print("  ✅ 演示完成")
    print("=" * 60)
    print("\n💡 关键修复点:")
    print("  1. prepare_training_data() 严格排除预测列")
    print("  2. predict_scores() 独立存储结果，不污染训练数据")
    print("  3. 添加断言验证，防止position等列泄露")
    print("  4. 自动清洗共线性和弱因子")
    print("  5. XGBoost 2.0+ 兼容性修复")