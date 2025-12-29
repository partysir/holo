"""
ml_factor_scoring_fixed.py - 滚动训练修复版 v2.8 (消除评分重复)

修复内容：
1. ✅ 消除与StockRanker的评分重复问题
2. ✅ ML模型使用原始因子，而非预计算的position
3. ✅ 清晰的评分流程：原始因子 → ML预测 → 最终评分
4. ✅ 保留所有原有功能（滚动窗口、数据隔离等）
"""

import pandas as pd
import numpy as np
import warnings
import traceback
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    lgb = None
    LIGHTGBM_AVAILABLE = False

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression


# ============================================================================
# 核心组件: 数据隔离与正交化
# ============================================================================

class PurgingEmbargoSplitter:
    """数据隔离切分器 (用于训练集内部的验证集划分)"""
    def __init__(self, n_splits=5, embargo_days=5):
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, data, date_column='date'):
        # 简单的时序切分
        n_samples = len(data)
        indices = np.arange(n_samples)

        # 至少要有足够的数据
        if n_samples < 100:
            return []

        fold_size = n_samples // (self.n_splits + 1)
        splits = []

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            valid_end = fold_size * (i + 2)

            if i == self.n_splits - 1:
                valid_end = n_samples # 最后一个fold取完

            train_idx = indices[:train_end]
            valid_idx = indices[train_end:valid_end]

            # Embargo: 删除训练集末尾靠近验证集的部分
            if self.embargo_days > 0 and len(train_idx) > 0:
                cutoff = len(train_idx) - max(1, int(len(train_idx)*0.05))
                train_idx = train_idx[:cutoff]

            splits.append((train_idx, valid_idx))

        return splits


class FeatureOrthogonalizer:
    """特征正交化"""
    def __init__(self, neutralize_market=True, neutralize_industry=True):
        self.neutralize_market = neutralize_market
        self.neutralize_industry = neutralize_industry

    def fit_transform(self, factor_data, factor_columns, price_data=None):
        factor_data = factor_data.copy()

        # 1. 市场中性化
        if self.neutralize_market and price_data is not None:
            try:
                factor_data = self._neutralize_market(factor_data, factor_columns, price_data)
            except Exception:
                pass

        # 2. 行业中性化
        if self.neutralize_industry and 'industry' in factor_data.columns:
            try:
                factor_data = self._neutralize_industry(factor_data, factor_columns)
            except Exception:
                pass

        return factor_data

    def _neutralize_market(self, factor_data, factor_columns, price_data):
        price_col = self._detect_price_column(price_data)
        if not price_col: return factor_data

        # 清理同名列
        if price_col in factor_data.columns:
            factor_data = factor_data.drop(columns=[price_col])

        try:
            merged = factor_data.merge(
                price_data[['instrument', 'date', price_col]],
                on=['instrument', 'date'],
                how='left'
            )
        except:
            return factor_data

        # 计算收益率
        merged['daily_return'] = merged.groupby('instrument')[price_col].pct_change().fillna(0)

        for factor in factor_columns:
            if factor not in merged.columns: continue

            valid_mask = merged[factor].notna() & np.isfinite(merged[factor])
            if valid_mask.sum() < 50: continue

            X = merged.loc[valid_mask, 'daily_return'].values.reshape(-1, 1)
            y = merged.loc[valid_mask, factor].values

            try:
                model = LinearRegression().fit(X, y)
                merged.loc[valid_mask, factor] = y - model.predict(X)
            except:
                pass

        return merged.drop(columns=['daily_return'], errors='ignore')

    def _neutralize_industry(self, factor_data, factor_columns):
        if factor_data['industry'].nunique() <= 1: return factor_data

        industry_dummies = pd.get_dummies(factor_data['industry'], prefix='ind', drop_first=True)

        for factor in factor_columns:
            if factor not in factor_data.columns: continue

            valid_idx = factor_data[factor].notna() & np.isfinite(factor_data[factor])
            if valid_idx.sum() < 50: continue

            X = industry_dummies.loc[valid_idx]
            y = factor_data.loc[valid_idx, factor].values

            try:
                model = LinearRegression().fit(X, y)
                factor_data.loc[valid_idx, factor] = y - model.predict(X)
            except:
                pass

        return factor_data

    def _detect_price_column(self, df):
        if df is None: return None
        for col in ['close', 'Close', 'CLOSE', 'price', 'Price', 'adj_close', 'trade_price']:
            if col in df.columns: return col
        return None


class EnsembleVotingScorer:
    """集成投票评分器"""
    def __init__(self, voting_strategy='average'):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, X_valid, y_valid):
        X_train = np.nan_to_num(X_train)
        X_valid = np.nan_to_num(X_valid)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        if XGBOOST_AVAILABLE:
            self.xgb_model = xgb.XGBClassifier(
                objective='binary:logistic', eval_metric='auc',
                max_depth=4, learning_rate=0.05, n_estimators=100, n_jobs=-1, verbosity=0
            )
            try:
                self.xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_valid_scaled, y_valid)], verbose=False)
            except:
                self.xgb_model.fit(X_train_scaled, y_train)

        if LIGHTGBM_AVAILABLE:
            self.lgb_model = lgb.LGBMClassifier(
                objective='binary', metric='auc',
                max_depth=4, learning_rate=0.05, n_estimators=100, n_jobs=-1, verbose=-1
            )
            try:
                self.lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_valid_scaled, y_valid)])
            except:
                self.lgb_model.fit(X_train_scaled, y_train)
        return self

    def predict_proba(self, X):
        X = np.nan_to_num(X)
        X_scaled = self.scaler.transform(X)
        preds = []
        if self.xgb_model: preds.append(self.xgb_model.predict_proba(X_scaled)[:, 1])
        if self.lgb_model: preds.append(self.lgb_model.predict_proba(X_scaled)[:, 1])

        if not preds: return np.zeros(len(X))
        return np.mean(preds, axis=0)


# ============================================================================
# UltraMLScorer (核心类) - v2.8 修复评分重复
# ============================================================================

class UltraMLScorer:
    """超级ML评分器 - 滚动训练版 (v2.8 消除评分重复)"""
    def __init__(self, target_period=5, top_percentile=0.2, embargo_days=5,
                 neutralize_market=True, neutralize_industry=True,
                 voting_strategy='average', train_months=12, **kwargs):

        self.target_period = target_period
        self.top_percentile = top_percentile
        self.embargo_days = embargo_days
        self.train_months = train_months

        self.orthogonalizer = FeatureOrthogonalizer(neutralize_market, neutralize_industry)
        self.ensemble = None
        self.feature_names = None

        print(f"\n🚀 UltraMLScorer v2.8 (修复评分重复)")
        print(f"  ✅ 预测周期: {target_period}天")
        print(f"  ✅ 滚动训练窗口: {train_months}个月")
        print(f"  ✅ 使用原始因子训练，避免position泄露")

    def _identify_factor_columns(self, factor_data):
        """
        🔧 关键修复：智能识别原始因子列
        排除：日期、代码、价格、已有评分
        """
        exclude_patterns = [
            'date', 'instrument', 'industry',  # 元数据
            'open', 'high', 'low', 'close', 'volume', 'amount',  # 价格数据
            'position', 'ml_score',  # 已有评分（避免泄露）
            '_norm', '_rank', '_score'  # 中间变量
        ]

        factor_columns = []
        for col in factor_data.columns:
            # 检查是否为数值列
            if not pd.api.types.is_numeric_dtype(factor_data[col]):
                continue

            # 检查是否在排除列表中
            if any(pattern in col.lower() for pattern in exclude_patterns):
                continue

            factor_columns.append(col)

        return factor_columns

    def prepare_batch_data(self, factor_data, price_data, factor_columns, is_inference=False):
        """准备单个批次的数据（保持原逻辑）"""
        # 1. 价格列处理
        price_col = self._detect_price_column(price_data)
        if not price_col:
            price_col = self._detect_price_column(factor_data)
            price_data = factor_data.copy()

        if not price_col:
            price_col = 'close' if 'close' in factor_data.columns else 'open'
            if price_col not in factor_data.columns:
                factor_data['close'] = 100.0
                price_col = 'close'
            price_data = factor_data.copy()

        # 2. 合并（清理已有的price_col避免重复）
        if price_col in factor_data.columns:
            factor_data_clean = factor_data.drop(columns=[price_col])
        else:
            factor_data_clean = factor_data

        try:
            merged = factor_data_clean.merge(
                price_data[['instrument', 'date', price_col]],
                on=['instrument', 'date'],
                how='left'
            )
        except:
            merged = factor_data.copy()
            if price_col not in merged.columns: merged[price_col] = 100.0

        # 保存价格列用于后续恢复
        temp_price = merged[price_col].copy()

        # 3. 正交化
        merged = self.orthogonalizer.fit_transform(merged, factor_columns, price_data)

        # 恢复价格列
        if price_col not in merged.columns:
            merged[price_col] = temp_price

        # 4. 构建目标 (仅训练模式需要)
        if not is_inference:
            merged['fwd_ret'] = merged.groupby('instrument')[price_col].pct_change(self.target_period).shift(-self.target_period)
            merged['fwd_ret'] = merged['fwd_ret'].fillna(0)

            merged['target'] = 0
            for date in merged['date'].unique():
                mask = merged['date'] == date
                if mask.sum() > 5:
                    rets = merged.loc[mask, 'fwd_ret']
                    thresh = rets.quantile(1 - self.top_percentile)
                    merged.loc[mask & (merged['fwd_ret'] >= thresh), 'target'] = 1

            merged = merged.dropna(subset=['target'])

        # 5. 提取特征
        X = merged[factor_columns].replace([np.inf, -np.inf], np.nan).fillna(0)

        if is_inference:
            return X, merged
        else:
            y = merged['target'].values
            return X, y, merged

    def train_model(self, X, y):
        """训练单个模型（保持原逻辑）"""
        splitter = PurgingEmbargoSplitter(n_splits=3, embargo_days=self.embargo_days)
        splits = splitter.split(X)

        if not splits:
            return EnsembleVotingScorer().train(X, y, X, y)

        best_score = -1
        best_model = None

        for train_idx, valid_idx in splits:
            if len(train_idx) < 10 or len(valid_idx) < 10: continue

            X_train, y_train = X.iloc[train_idx], y[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y[valid_idx]

            if y_train.sum() < 2 or y_valid.sum() < 2: continue

            model = EnsembleVotingScorer().train(X_train, y_train, X_valid, y_valid)

            try:
                score = roc_auc_score(y_valid, model.predict_proba(X_valid))
            except:
                score = 0.5

            if score > best_score:
                best_score = score
                best_model = model

        return best_model if best_model else EnsembleVotingScorer().train(X, y, X, y)

    def predict(self, factor_data, price_data=None):
        """
        执行滚动预测 (Rolling Prediction) - v2.8 修复版

        ✅ 关键修复：使用原始因子列，避免使用预计算的position
        """
        print(f"\n🎯 开始滚动窗口预测 (v2.8 - 修复评分重复)...")

        # 1. 准备基础数据
        factor_data = factor_data.sort_values('date').copy()
        if price_data is not None:
            price_data = price_data.sort_values('date').copy()

        # ✅ 关键修复：智能识别原始因子列
        factor_columns = self._identify_factor_columns(factor_data)

        if len(factor_columns) == 0:
            print("  ⚠️  未找到有效因子列，使用降级策略")
            # 降级：简单标准化平均
            numeric_cols = factor_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [c for c in numeric_cols if c not in ['date', 'position', 'ml_score']]
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                factor_data['ml_score'] = scaler.fit_transform(factor_data[numeric_cols].fillna(0)).mean(axis=1)
                factor_data['position'] = factor_data.groupby('date')['ml_score'].rank(pct=True)
            return factor_data

        self.feature_names = factor_columns
        print(f"  📊 使用 {len(factor_columns)} 个原始因子: {factor_columns[:5]}...")

        # 2. 生成时间切片 (按月)
        dates = sorted(factor_data['date'].unique())
        if not dates: return factor_data

        date_objs = pd.to_datetime(dates)
        start_date = date_objs[0]
        end_date = date_objs[-1]

        current_date = start_date
        results = []

        train_window = relativedelta(months=self.train_months)
        step_delta = relativedelta(months=1)

        first_train_end = start_date + train_window

        print(f"  📅 数据范围: {start_date.date()} -> {end_date.date()}")
        print(f"  ❄️ 冷启动期: {start_date.date()} -> {first_train_end.date()}")

        while current_date <= end_date:
            next_date = current_date + step_delta

            mask_pred = (pd.to_datetime(factor_data['date']) >= current_date) & \
                        (pd.to_datetime(factor_data['date']) < next_date)
            batch_data = factor_data[mask_pred].copy()

            if batch_data.empty:
                current_date = next_date
                continue

            train_start = current_date - train_window

            if train_start < start_date:
                # === 冷启动模式 ===
                X = batch_data[factor_columns].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                batch_data['ml_score'] = X_scaled.mean(axis=1)

            else:
                # === 滚动训练模式 ===
                mask_train = (pd.to_datetime(factor_data['date']) >= train_start) & \
                             (pd.to_datetime(factor_data['date']) < current_date)

                train_factors = factor_data[mask_train]
                train_prices = price_data[price_data['date'].isin(train_factors['date'])] if price_data is not None else None

                try:
                    X_train, y_train, _ = self.prepare_batch_data(train_factors, train_prices, factor_columns, is_inference=False)
                    model = self.train_model(X_train, y_train)
                    self.ensemble = model

                    X_pred, _ = self.prepare_batch_data(batch_data, price_data, factor_columns, is_inference=True)
                    batch_data['ml_score'] = model.predict_proba(X_pred)

                except Exception as e:
                    print(f"    ⚠️ 训练失败 ({e})，降级为规则打分")
                    X = batch_data[factor_columns].fillna(0)
                    scaler = StandardScaler()
                    batch_data['ml_score'] = scaler.fit_transform(X).mean(axis=1)

            results.append(batch_data)
            current_date = next_date

        # 3. 合并结果
        if not results:
            return factor_data

        final_result = pd.concat(results)
        final_result['position'] = final_result.groupby('date')['ml_score'].rank(pct=True)

        print(f"  ✅ 滚动预测完成，生成了 {len(final_result)} 条评分")
        return final_result

    def _detect_price_column(self, df):
        if df is None: return None
        for col in ['close', 'Close', 'CLOSE', 'price', 'Price', 'adj_close', 'trade_price']:
            if col in df.columns: return col
        return None

    def get_trained_model(self):
        if self.ensemble:
            if self.ensemble.xgb_model: return self.ensemble.xgb_model
            if self.ensemble.lgb_model: return self.ensemble.lgb_model
        return None

    def get_feature_names(self):
        return self.feature_names


# ============================================================================
# 适配器类：AdvancedMLScorer
# ============================================================================

class AdvancedMLScorer(UltraMLScorer):
    """
    [适配器] 兼容 main-2.py 的调用
    """
    def __init__(self, model_type='xgboost', target_period=5, top_percentile=0.2,
                 use_classification=True, use_ic_features=False, train_months=12, **kwargs):
        super().__init__(
            target_period=target_period,
            top_percentile=top_percentile,
            embargo_days=5,
            neutralize_market=True,
            neutralize_industry=True,
            voting_strategy='average',
            train_months=train_months
        )

    def predict_scores(self, factor_data, price_data, factor_columns=None):
        return self.predict(factor_data, price_data)


# ============================================================================
# 辅助类
# ============================================================================

class ICCalculator:
    pass

class IndustryBasedScorer:
    def __init__(self, tushare_token=None): pass
    def score_by_industry(self, factor_data, cols): return factor_data

class EnhancedStockSelector:
    def __init__(self): pass
    def select_stocks(self, factor_data, min_score=0.6, **kwargs):
        if 'ml_score' in factor_data.columns:
            return factor_data[factor_data['ml_score'] >= min_score]
        return factor_data