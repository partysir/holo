# -*- coding: utf-8 -*-
"""
ml_factor_scoring_complete.py - 完整的高级机器学习因子评分系统

整合内容：
1. 核心基础模块 (IC计算, 时间序列切分)
2. 因子处理模块 (StockRanker, 因子生成)
3. 高级ML评分器 (Walk-Forward训练, XGBoost/LightGBM集成)
4. 行业与回测模块 (行业中性化, 选股, 简单回测)
5. 策略编排与示例 (MultiFactorMLStrategy)

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
        total_score = 0

        for factor, periods in ic_results.items():
            if period not in periods:
                weights[factor] = 0
                continue

            val_map = {
                'ic': abs(periods[period].get('ic', 0)),
                'rank_ic': abs(periods[period].get('rank_ic', 0)),
                'icir': abs(periods[period].get('icir', 0)),
                'rank_icir': abs(periods[period].get('rank_icir', 0))
            }
            score = val_map.get(method, abs(periods[period].get('ic', 0)))

            weights[factor] = score
            total_score += score

        if total_score > 0:
            weights = {k: v / total_score for k, v in weights.items()}

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
# 第三部分：高级ML评分器 (完整实现)
# ============================================================================

class AdvancedMLScorer:
    """
    高级机器学习评分器
    整合: 时间序列切分, IC特征, Active Return标签, 模型集成
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
        merged['abs_return'] = merged.groupby('instrument')[price_col].pct_change(self.target_period).shift(
            -self.target_period)

        if self.use_active_return:
            market_return = merged.groupby('date')['abs_return'].transform('mean')
            merged['future_return'] = merged['abs_return'] - market_return
        else:
            merged['future_return'] = merged['abs_return']

        if self.use_classification:
            merged['target'] = 0
            for date in merged['date'].unique():
                mask = merged['date'] == date
                rets = merged.loc[mask, 'future_return']
                if len(rets) > 5:
                    thresh = rets.quantile(1 - self.top_percentile)
                    merged.loc[mask & (merged['future_return'] >= thresh), 'target'] = 1
            target_col = 'target'
        else:
            target_col = 'future_return'

        merged = merged.dropna(subset=[target_col])

        # 特征选择
        exclude = ['date', 'instrument', 'future_return', 'abs_return', 'target', price_col,
                   'industry', 'ml_score', 'position', 'composite_score']
        feature_cols = [c for c in merged.columns if c not in exclude and pd.api.types.is_numeric_dtype(merged[c])]

        X = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = merged[target_col].values
        self.feature_names = feature_cols

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
                if self.use_classification:
                    valid_score = roc_auc_score(y_valid, model.predict_proba(X_valid_s)[:, 1])
                    print(f"     验证AUC: {valid_score:.4f}")
                else:
                    valid_score = np.corrcoef(y_valid, model.predict(X_valid_s))[0, 1]
                    print(f"     验证IC: {valid_score:.4f}")
                window_results.append({'model': model, 'score': valid_score, 'window': i})

        # 选择最佳模型
        if window_results:
            best = max(window_results, key=lambda x: x['score'])
            self.models['best'] = best['model']
            print(f"\n  🏆 最佳模型来自窗口 {best['window'] + 1}, 得分: {best['score']:.4f}")
        return self

    def _train_model(self, X_train, y_train, X_valid, y_valid):
        """通用模型训练入口 (修复 XGBoost 2.0+ 兼容性)"""
        if self.use_classification:
            if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                # 修复：early_stopping_rounds 移入构造函数
                model = xgb.XGBClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=6,
                    eval_metric='auc', random_state=self.random_state, n_jobs=-1,
                    early_stopping_rounds=30
                )
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
                return model
            elif LIGHTGBM_AVAILABLE:
                model = lgb.LGBMClassifier(
                    n_estimators=300, learning_rate=0.05, max_depth=6,
                    metric='auc', random_state=self.random_state, n_jobs=-1, verbose=-1
                )
                model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
                return model
        else:
            # 回归逻辑
            if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
                # 修复：early_stopping_rounds 移入构造函数
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
        if 'best' not in self.models: raise ValueError("模型未训练")
        data = factor_data.copy()
        X = data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X)

        model = self.models['best']
        if self.use_classification:
            data['ml_score'] = model.predict_proba(X_scaled)[:, 1]
        else:
            data['ml_score'] = model.predict(X_scaled)

        data['position'] = data.groupby('date')['ml_score'].rank(pct=True)
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
        df = df.rename(columns={'ts_code': 'instrument'})
        df['industry'] = df['industry'].fillna('其他')
        return df[['instrument', 'industry']]
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
                                                                                               ascending=False)

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
        merged['ret'] = merged.groupby('instrument')[price_col].pct_change(holding_period).shift(-holding_period)

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
    多因子ML选股策略编排器
    流程: 因子IC -> StockRanker -> Walk-Forward ML -> 行业评分 -> 选股 -> 回测
    """

    def __init__(self, model_type='xgboost', target_period=5, train_months=12, tushare_token=None):
        self.target_period = target_period
        self.ic_calc = ICCalculator([target_period])
        self.ranker = StockRanker(method='ic_weight')
        self.ml = AdvancedMLScorer(model_type=model_type, target_period=target_period, train_months=train_months)
        self.ind_scorer = IndustryBasedScorer(tushare_token)
        self.selector = EnhancedStockSelector()

    def run(self, factor_data, price_data, factor_cols, min_score=0.7, max_stocks=30):
        print("=" * 60 + "\n  多因子ML策略启动\n" + "=" * 60)

        # 1. IC分析 & 权重
        ic_res = self.ic_calc.calculate_factor_ic(factor_data, price_data, factor_cols)
        weights = self.ic_calc.get_ic_weights(ic_res, self.target_period)

        # 2. 基础评分
        processed = self.ranker.preprocess_factors(factor_data, factor_cols)
        scored = self.ranker.calculate_composite_score(processed, factor_cols, weights)

        # 3. ML增强
        X, y, merged = self.ml.prepare_training_data(scored, price_data, factor_cols)
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
            print("\n  🔑 Top 5 重要特征:")
            print(imp.head(5))

        return {'picks': picks, 'backtest': backtest, 'feature_importance': imp}


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
            for i in range(5): rec[f'factor_{i}'] = np.random.randn()
            records.append(rec)
            prices.append({'date': date, 'instrument': inst, 'close': 100 * (1 + np.random.randn() * 0.1)})

    return pd.DataFrame(records), pd.DataFrame(prices)


if __name__ == '__main__':
    # 示例运行
    factors, prices = generate_sample_data()
    cols = [f'factor_{i}' for i in range(5)]

    strategy = MultiFactorMLStrategy(model_type='xgboost', train_months=3)
    results = strategy.run(factors, prices, cols)

    print("\n✅ 演示完成")