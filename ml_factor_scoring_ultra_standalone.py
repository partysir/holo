import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# 检查模型库
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


# ============================================================================
# 1. 数据隔离器 (Purged Walk-Forward)
# ============================================================================
class PurgingEmbargoSplitter:
    """数据隔离切分器"""

    def __init__(self, train_months=12, valid_months=1, embargo_days=5):
        self.train_months = train_months
        self.valid_months = valid_months
        self.embargo_days = embargo_days

    def split(self, data, date_column='date'):
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column)

        data['year_month'] = data[date_column].dt.to_period('M')
        unique_months = sorted(data['year_month'].unique())

        splits = []

        for i in range(len(unique_months) - self.train_months - self.valid_months + 1):
            train_start = unique_months[i]
            train_end = unique_months[i + self.train_months - 1]
            valid_start = unique_months[i + self.train_months]
            valid_end = unique_months[i + self.train_months + self.valid_months - 1]

            train_idx = data[data['year_month'].between(train_start, train_end)].index
            valid_idx = data[data['year_month'].between(valid_start, valid_end)].index

            # Purging: 剔除训练集尾部
            if self.embargo_days > 0 and len(train_idx) > 0:
                train_dates = data.loc[train_idx, date_column]
                train_cutoff = train_dates.max() - pd.Timedelta(days=self.embargo_days)
                train_idx = train_idx[data.loc[train_idx, date_column] <= train_cutoff]

            if len(train_idx) > 100 and len(valid_idx) > 0:
                splits.append((train_idx, valid_idx))

        return splits


# ============================================================================
# 2. 特征正交化器 (截面版)
# ============================================================================
class FeatureOrthogonalizer:
    """特征正交化 - 截面回归版"""

    def __init__(self, neutralize_market=True, neutralize_industry=True):
        self.neutralize_market = neutralize_market
        self.neutralize_industry = neutralize_industry

    def fit_transform(self, data, factor_columns):
        """逐日截面回归正交化"""
        if not self.neutralize_market and not self.neutralize_industry:
            return data

        print(f"  🔧 执行正交化 (市场={self.neutralize_market}, 行业={self.neutralize_industry})...")
        data = data.copy()

        # 1. 准备市场因子 (全市场均值)
        price_col = self._detect_price_column(data)
        has_market_col = False

        if self.neutralize_market:
            if price_col:
                # 临时计算收益率
                data['_ret'] = data.groupby('instrument')[price_col].pct_change()
                data['_mkt'] = data.groupby('date')['_ret'].transform('mean').fillna(0)
                has_market_col = True
            else:
                print("  ⚠️  警告: 未找到价格列，跳过市场中性化")

        # 2. 准备行业因子
        has_industry_col = False
        if self.neutralize_industry and 'industry' in data.columns:
            data['industry'] = data['industry'].fillna('Other')
            has_industry_col = True

        valid_factors = [f for f in factor_columns if f in data.columns]

        # 3. 准备GroupBy需要的列列表
        group_cols = list(valid_factors)
        if has_market_col: group_cols.append('_mkt')
        if has_industry_col: group_cols.append('industry')

        # 定义单日处理函数
        def neutralize_day(df_day):
            if len(df_day) < 10: return df_day[valid_factors]

            X_list = []
            if has_market_col:
                X_list.append(df_day[['_mkt']].values)

            if has_industry_col:
                ind = pd.get_dummies(df_day['industry'], drop_first=True).values
                if ind.shape[1] > 0:
                    X_list.append(ind)

            if not X_list: return df_day[valid_factors]

            try:
                X = np.hstack(X_list)
                y = df_day[valid_factors].values

                # 线性回归求残差: e = y - X*beta
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                res = y - X @ beta
                return pd.DataFrame(res, index=df_day.index, columns=valid_factors)
            except Exception:
                return df_day[valid_factors]

        # 4. 执行 GroupBy Apply
        ortho = data.groupby('date')[group_cols].apply(neutralize_day)

        # 处理多级索引问题
        if isinstance(ortho, pd.DataFrame):
            if 'date' in ortho.index.names:
                try:
                    ortho = ortho.reset_index(level='date', drop=True)
                except IndexError:
                    pass

            # 使用 update 原地更新
            data.update(ortho)

        # 清理
        data = data.drop(columns=['_ret', '_mkt'], errors='ignore')
        return data

    def _detect_price_column(self, df):
        for col in ['close', 'Close', 'price', 'Price', 'CLOSE']:
            if col in df.columns: return col
        for col in df.columns:
            if 'close' in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                return col
        return None


# ============================================================================
# 3. 集成投票器
# ============================================================================
class EnsembleVotingScorer:
    """集成投票器"""

    def __init__(self, voting_strategy='strict'):
        self.voting_strategy = voting_strategy
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, X_valid, y_valid):
        X_train = X_train.fillna(0)
        X_valid = X_valid.fillna(0)

        X_train_s = self.scaler.fit_transform(X_train)
        X_valid_s = self.scaler.transform(X_valid)

        if XGBOOST_AVAILABLE:
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                eval_metric='logloss', verbosity=0, use_label_encoder=False
            )
            try:
                self.xgb_model.fit(X_train_s, y_train, eval_set=[(X_valid_s, y_valid)], verbose=False)
            except:
                self.xgb_model.fit(X_train_s, y_train)

        if LIGHTGBM_AVAILABLE:
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, num_leaves=20,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1
            )
            try:
                self.lgb_model.fit(X_train_s, y_train, eval_set=[(X_valid_s, y_valid)],
                                   callbacks=[lgb.early_stopping(20, verbose=False)])
            except:
                self.lgb_model.fit(X_train_s, y_train)
        return self

    def predict_proba(self, X):
        X = X.fillna(0)
        X_s = self.scaler.transform(X)
        preds = []
        if self.xgb_model: preds.append(self.xgb_model.predict_proba(X_s)[:, 1])
        if self.lgb_model: preds.append(self.lgb_model.predict_proba(X_s)[:, 1])

        if not preds: return np.zeros(len(X))

        p_avg = np.mean(preds, axis=0)

        if self.voting_strategy == 'strict' and len(preds) == 2:
            consensus = (preds[0] > 0.5) & (preds[1] > 0.5)
            return np.where(consensus, p_avg * 1.2, p_avg * 0.8)

        return p_avg


# ============================================================================
# 4. UltraMLScorer (主类 - API兼容版)
# ============================================================================
class UltraMLScorer:
    """超级ML评分器 - API兼容版 (胜率增强版)"""

    def __init__(self,
                 target_period=5,
                 top_percentile=0.20,
                 embargo_days=5,
                 neutralize_market=False,  # ✅ 关键修改：默认关闭市场中性化，保留Beta
                 neutralize_industry=True,
                 voting_strategy='average',
                 train_months=12,
                 random_state=42):

        self.target_period = target_period
        self.top_percentile = top_percentile
        self.embargo_days = embargo_days
        self.train_months = train_months
        self.voting_strategy = voting_strategy

        self.orthogonalizer = FeatureOrthogonalizer(neutralize_market, neutralize_industry)
        self.ensemble = None
        self.feature_names = None
        self.scaler = StandardScaler()

        print(f"\n🚀 初始化UltraMLScorer (胜率增强版):")
        print(f"  Target=Top{top_percentile:.0%}+AbsRet>0, MktNeut={neutralize_market}")

    def prepare_data(self, factor_data, price_data, factor_columns):
        """准备数据：合并 + 正交化 + 标签生成"""
        print(f"\n📦 准备训练数据...")

        price_col = self.orthogonalizer._detect_price_column(price_data)
        if not price_col:
            raise ValueError("未在 price_data 中找到价格列")

        merged = factor_data.copy()
        if price_col in merged.columns:
            print(f"  ✓ 价格列 '{price_col}' 已存在，跳过合并")
        else:
            merged = merged.merge(price_data[['instrument', 'date', price_col]], on=['instrument', 'date'], how='left')

        merged = merged.sort_values(['instrument', 'date'])

        # 排除包含 'ml_score', 'position' 等列作为特征
        actual_features = [c for c in factor_columns if
                           c in merged.columns and c not in ['ml_score', 'position', 'target']]

        # 特征正交化
        merged = self.orthogonalizer.fit_transform(merged, actual_features)

        # ✅ 生成Target (关键修改：加入绝对收益限制)
        price_col = self.orthogonalizer._detect_price_column(merged)
        merged['fwd_ret'] = merged.groupby('instrument')[price_col].pct_change(self.target_period).shift(
            -self.target_period)

        merged['target'] = 0

        def get_label_strict(x):
            if len(x) < 5: return pd.Series(0, index=x.index)
            # 1. 相对排名 Top 20%
            thresh = float(x.quantile(1 - self.top_percentile))
            is_top = (x >= thresh)

            # 2. ✅ 绝对收益 > 0.5% (过滤掉下跌趋势中的"相对好")
            is_profit = (x > 0.005)

            return (is_top & is_profit).astype(int)

        merged['target'] = merged.groupby('date')['fwd_ret'].apply(get_label_strict).reset_index(level=0, drop=True)

        valid_data = merged.dropna(subset=['target', 'fwd_ret'] + actual_features)
        print(f"  ✓ 有效样本: {len(valid_data)} (正样本比例: {valid_data['target'].mean():.1%})")

        self.feature_names = actual_features
        return valid_data[actual_features], valid_data['target'], valid_data

    def train(self, X, y, merged, verbose=False):
        """滚动训练"""
        print(f"\n🎯 滚动训练 (Train={self.train_months}m)...")

        splitter = PurgingEmbargoSplitter(self.train_months, embargo_days=self.embargo_days)
        splits = splitter.split(merged)

        if not splits:
            print("  ⚠️ 数据不足，无法切分，使用全量训练")
            model = EnsembleVotingScorer(self.voting_strategy)
            model.train(X, y, X, y)
            self.ensemble = model
            return self

        for i, (tr_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            if y_tr.sum() < 10 or y_val.sum() < 5: continue

            model = EnsembleVotingScorer(self.voting_strategy)
            model.train(X_tr, y_tr, X_val, y_val)
            self.ensemble = model  # 保留最新的模型

            if verbose: print(f"  Window {i + 1}: Done")

        print("  ✓ 训练完成")
        return self

    def predict(self, factor_data, price_data=None):
        """预测"""
        if price_data is not None:
            # 回测时确保特征一致
            factor_data = self.orthogonalizer.fit_transform(factor_data, self.feature_names)

        print(f"\n🔮 执行预测...")
        X = factor_data[self.feature_names].fillna(0)

        if self.ensemble is None:
            raise ValueError("模型未训练")

        preds = self.ensemble.predict_proba(X)

        result = factor_data.copy()
        result['ml_score'] = preds
        return result


__all__ = ['UltraMLScorer']