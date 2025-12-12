"""
ml_factor_scoring_ultra.py - 超级优化版（提高实盘胜率）

🎯 四大核心优化：
✅ 1. 数据隔离 (Purging & Embargoing) - 消除信息泄露
✅ 2. 特征正交化 (Feature Orthogonalization) - 提取纯Alpha
✅ 3. 模型集成 (Ensemble Voting) - 降低过拟合
✅ 4. 精准目标 (Precision@K Focus) - 优化Top选股
"""

import pandas as pd
import numpy as np
import warnings
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
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression


# ============================================================================
# 优化1: 数据隔离器 (Purging & Embargoing)
# ============================================================================

class PurgingEmbargoSplitter:
    """
    数据隔离切分器

    核心思想：
    - Purging: 删除训练集末尾与验证集有重叠的样本
    - Embargo: 在训练集和验证集之间加入Gap（隔离期）

    示例：
        预测5日收益，需要5日Gap
        Train: [月1-12] -> Gap: [12月末5天] -> Valid: [月13]
    """

    def __init__(self, train_months=12, valid_months=1, test_months=1,
                 embargo_days=5):
        """
        Args:
            embargo_days: 隔离期（天）- 应该 >= 预测周期
        """
        self.train_months = train_months
        self.valid_months = valid_months
        self.test_months = test_months
        self.embargo_days = embargo_days

    def split(self, data, date_column='date'):
        """时间序列切分 + 数据隔离"""
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column)

        data['year_month'] = data[date_column].dt.to_period('M')
        months = sorted(data['year_month'].unique())

        splits = []

        for i in range(len(months) - self.train_months - self.valid_months - self.test_months + 1):
            train_end = i + self.train_months
            valid_end = train_end + self.valid_months
            test_end = valid_end + self.test_months

            train_months_list = months[i:train_end]
            valid_months_list = months[train_end:valid_end]
            test_months_list = months[valid_end:test_end]

            # 初始索引
            train_idx = data[data['year_month'].isin(train_months_list)].index
            valid_idx = data[data['year_month'].isin(valid_months_list)].index
            test_idx = data[data['year_month'].isin(test_months_list)].index

            # ✅ 关键优化：Purging + Embargo
            if self.embargo_days > 0 and len(train_idx) > 0 and len(valid_idx) > 0:
                # 获取训练集最后一天
                train_last_date = data.loc[train_idx, date_column].max()

                # 删除训练集中会与验证集重叠的样本
                # 即删除训练集最后 embargo_days 天的数据
                embargo_cutoff = train_last_date - pd.Timedelta(days=self.embargo_days)
                train_idx = train_idx[data.loc[train_idx, date_column] <= embargo_cutoff]

            if len(train_idx) > 0 and len(valid_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, valid_idx, test_idx))

        return splits


# ============================================================================
# 优化2: 特征正交化器 (Feature Orthogonalization)
# ============================================================================

class FeatureOrthogonalizer:
    """
    特征正交化 - 提取纯Alpha

    方法：
    1. 市场中性化：残差 = 因子 - β_market × 市场收益
    2. 行业中性化：残差 = 因子 - Σ(β_industry × 行业哑变量)
    """

    def __init__(self, neutralize_market=True, neutralize_industry=True):
        self.neutralize_market = neutralize_market
        self.neutralize_industry = neutralize_industry
        self.market_models = {}  # {factor: LinearRegression}
        self.industry_models = {}

    def fit_transform(self, factor_data, factor_columns, price_data=None):
        """
        拟合并转换因子

        Args:
            factor_data: 因子数据（必须包含date, instrument）
            factor_columns: 需要中性化的因子列表
            price_data: 价格数据（用于计算市场收益）
        """
        print("\n🔧 特征正交化...")

        factor_data = factor_data.copy()

        # ===== 1. 市场中性化 =====
        if self.neutralize_market and price_data is not None:
            print("  ✓ 市场中性化...")
            factor_data = self._neutralize_market(
                factor_data, factor_columns, price_data
            )

        # ===== 2. 行业中性化 =====
        if self.neutralize_industry and 'industry' in factor_data.columns:
            print("  ✓ 行业中性化...")
            factor_data = self._neutralize_industry(
                factor_data, factor_columns
            )

        print("  ✓ 正交化完成")
        return factor_data

    def _neutralize_market(self, factor_data, factor_columns, price_data):
        """市场中性化"""
        # 计算市场收益（每日平均收益）
        price_col = self._detect_price_column(price_data)
        if price_col is None:
            return factor_data

        merged = factor_data.merge(
            price_data[['instrument', 'date', price_col]],
            on=['instrument', 'date'],
            how='left'
        )

        # 每日市场收益
        merged['daily_return'] = merged.groupby('instrument')[price_col].pct_change()
        market_return = merged.groupby('date')['daily_return'].transform('mean')

        # 对每个因子回归
        for factor in factor_columns:
            if factor not in merged.columns:
                continue

            # 过滤有效数据
            valid = merged[[factor, 'daily_return']].dropna()
            if len(valid) < 100:
                continue

            # 回归：factor = α + β × market_return + ε
            X = valid['daily_return'].values.reshape(-1, 1)
            y = valid[factor].values

            model = LinearRegression()
            model.fit(X, y)

            # 残差 = 因子 - 预测值
            merged.loc[valid.index, factor] = y - model.predict(X)

            self.market_models[factor] = model

        factor_data = merged.drop(columns=[price_col, 'daily_return'], errors='ignore')
        return factor_data

    def _neutralize_industry(self, factor_data, factor_columns):
        """行业中性化"""
        # 创建行业哑变量
        industry_dummies = pd.get_dummies(
            factor_data['industry'],
            prefix='ind',
            drop_first=True  # 避免完全共线性
        )

        for factor in factor_columns:
            if factor not in factor_data.columns:
                continue

            # 过滤有效数据
            valid_idx = factor_data[factor].notna()
            if valid_idx.sum() < 100:
                continue

            X = industry_dummies.loc[valid_idx]
            y = factor_data.loc[valid_idx, factor].values

            # 回归：factor = Σ(β_i × industry_i) + ε
            model = LinearRegression()
            model.fit(X, y)

            # 残差
            factor_data.loc[valid_idx, factor] = y - model.predict(X)

            self.industry_models[factor] = model

        return factor_data

    def _detect_price_column(self, df):
        candidates = ['close', 'Close', 'CLOSE', 'price', 'Price']
        for col in candidates:
            if col in df.columns:
                return col
        return None


# ============================================================================
# 优化3: 集成投票器 (Ensemble Voting)
# ============================================================================

class EnsembleVotingScorer:
    """
    集成投票评分器

    策略：
    - 同时训练 XGBoost 和 LightGBM
    - 预测时取概率均值
    - 可选：只有两个模型都看多（概率>0.5）时才给高分
    """

    def __init__(self, voting_strategy='average', strict_threshold=0.6):
        """
        Args:
            voting_strategy: 'average' | 'strict'
                - average: 简单平均两个模型的预测
                - strict: 只有两个模型都看多时才给高分
            strict_threshold: strict模式下的阈值
        """
        self.voting_strategy = voting_strategy
        self.strict_threshold = strict_threshold
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()

    def train(self, X_train, y_train, X_valid, y_valid, verbose=False):
        """训练两个模型"""
        print(f"\n🤝 集成训练 ({self.voting_strategy})...")

        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        # ===== XGBoost =====
        if XGBOOST_AVAILABLE:
            print("  ✓ 训练 XGBoost...")
            self.xgb_model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='auc',
                max_depth=5,  # 降低复杂度
                learning_rate=0.03,
                n_estimators=300,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                n_jobs=-1
            )

            try:
                self.xgb_model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_valid_scaled, y_valid)],
                    early_stopping_rounds=30,
                    verbose=verbose
                )
            except:
                self.xgb_model.fit(X_train_scaled, y_train)

        # ===== LightGBM =====
        if LIGHTGBM_AVAILABLE:
            print("  ✓ 训练 LightGBM...")
            self.lgb_model = lgb.LGBMClassifier(
                objective='binary',
                metric='auc',
                max_depth=5,
                learning_rate=0.03,
                n_estimators=300,
                subsample=0.7,
                colsample_bytree=0.7,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            try:
                self.lgb_model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_valid_scaled, y_valid)],
                    callbacks=[lgb.early_stopping(30, verbose=verbose)]
                )
            except:
                self.lgb_model.fit(X_train_scaled, y_train)

        # 评估
        y_pred = self.predict_proba(X_valid)
        auc = roc_auc_score(y_valid, y_pred)
        print(f"  ✓ 集成验证AUC: {auc:.4f}")

        return self

    def predict_proba(self, X):
        """集成预测"""
        X_scaled = self.scaler.transform(X)

        predictions = []

        if self.xgb_model is not None:
            pred_xgb = self.xgb_model.predict_proba(X_scaled)[:, 1]
            predictions.append(pred_xgb)

        if self.lgb_model is not None:
            pred_lgb = self.lgb_model.predict_proba(X_scaled)[:, 1]
            predictions.append(pred_lgb)

        if len(predictions) == 0:
            raise ValueError("没有可用模型")

        # ===== 投票策略 =====
        if self.voting_strategy == 'average':
            # 简单平均
            return np.mean(predictions, axis=0)

        elif self.voting_strategy == 'strict':
            # 严格模式：两个都看多才给高分
            avg_pred = np.mean(predictions, axis=0)

            # 只有当两个模型都超过阈值时，才保留原始分数
            if len(predictions) == 2:
                both_bullish = (predictions[0] > self.strict_threshold) & \
                               (predictions[1] > self.strict_threshold)
                return np.where(both_bullish, avg_pred, avg_pred * 0.5)
            else:
                return avg_pred


# ============================================================================
# 优化4: Precision@K 评估器
# ============================================================================

class PrecisionAtKEvaluator:
    """
    Precision@K 评估器

    关注指标：
    - Precision@20%: Top 20% 中有多少是真正的赢家
    - Recall@20%: 真正的赢家有多少被选中
    """

    @staticmethod
    def precision_at_k(y_true, y_pred_proba, k=0.2):
        """
        计算 Precision@K

        Args:
            y_true: 真实标签
            y_pred_proba: 预测概率
            k: Top K比例

        Returns:
            precision, recall
        """
        n = len(y_true)
        top_k = int(n * k)

        # 选出Top K
        top_k_idx = np.argsort(y_pred_proba)[-top_k:]

        y_true_top = y_true[top_k_idx]

        precision = y_true_top.sum() / len(y_true_top) if len(y_true_top) > 0 else 0
        recall = y_true_top.sum() / y_true.sum() if y_true.sum() > 0 else 0

        return precision, recall

    @staticmethod
    def evaluate_model(model, X_valid, y_valid, k=0.2):
        """完整评估"""
        y_pred_proba = model.predict_proba(X_valid)

        auc = roc_auc_score(y_valid, y_pred_proba)
        prec, rec = PrecisionAtKEvaluator.precision_at_k(y_valid, y_pred_proba, k)

        print(f"    AUC: {auc:.4f}")
        print(f"    Precision@{int(k*100)}%: {prec:.4f}")
        print(f"    Recall@{int(k*100)}%: {rec:.4f}")

        return {'auc': auc, 'precision': prec, 'recall': rec}


# ============================================================================
# 超级ML评分器 (整合四大优化)
# ============================================================================

class UltraMLScorer:
    """
    超级ML评分器

    整合：
    1. ✅ Purging & Embargo
    2. ✅ Feature Orthogonalization
    3. ✅ Ensemble Voting
    4. ✅ Precision@K Focus
    """

    def __init__(self,
                 target_period=5,
                 top_percentile=0.20,
                 embargo_days=5,
                 neutralize_market=True,
                 neutralize_industry=True,
                 voting_strategy='average',
                 train_months=12,
                 random_state=42):

        self.target_period = target_period
        self.top_percentile = top_percentile
        self.embargo_days = embargo_days
        self.neutralize_market = neutralize_market
        self.neutralize_industry = neutralize_industry
        self.voting_strategy = voting_strategy
        self.train_months = train_months
        self.random_state = random_state

        self.orthogonalizer = FeatureOrthogonalizer(
            neutralize_market, neutralize_industry
        )
        self.ensemble = EnsembleVotingScorer(voting_strategy)
        self.feature_names = None

        print(f"\n🚀 超级ML评分器")
        print(f"  ✅ 数据隔离: {embargo_days}天Gap")
        print(f"  ✅ 特征正交: 市场={neutralize_market}, 行业={neutralize_industry}")
        print(f"  ✅ 集成投票: {voting_strategy}")
        print(f"  ✅ 目标优化: Precision@{int(top_percentile*100)}%")

    def prepare_data(self, factor_data, price_data, factor_columns):
        """准备数据 + 特征正交化"""
        print(f"\n📦 准备训练数据...")

        # 检测价格列
        price_col = self._detect_price_column(price_data)
        if price_col is None:
            raise ValueError("未找到价格列")

        # 合并
        merged = factor_data.merge(
            price_data[['instrument', 'date', price_col]],
            on=['instrument', 'date'],
            how='left'
        )
        merged = merged.sort_values(['instrument', 'date'])

        # ===== 特征正交化 =====
        merged = self.orthogonalizer.fit_transform(
            merged, factor_columns, price_data
        )

        # ===== 计算超额收益目标 =====
        merged['abs_return'] = merged.groupby('instrument')[price_col].pct_change(
            self.target_period
        ).shift(-self.target_period)

        market_return = merged.groupby('date')['abs_return'].transform('mean')
        merged['future_return'] = merged['abs_return'] - market_return

        # 分类目标
        merged['target'] = 0
        for date in merged['date'].unique():
            date_mask = merged['date'] == date
            returns = merged.loc[date_mask, 'future_return']
            if len(returns) > 5:
                threshold = returns.quantile(1 - self.top_percentile)
                merged.loc[date_mask & (merged['future_return'] >= threshold), 'target'] = 1

        # 过滤
        merged = merged.dropna(subset=['target'])

        print(f"  ✓ 有效样本: {len(merged)}")
        print(f"  ✓ 正样本比例: {merged['target'].mean():.2%}")

        # 构建特征
        exclude = [
            'date', 'instrument', 'future_return', 'abs_return',
            'target', price_col, 'industry', 'year_month'
        ]
        feature_cols = [c for c in merged.columns
                       if c not in exclude and pd.api.types.is_numeric_dtype(merged[c])]

        X = merged[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        y = merged['target'].values

        self.feature_names = feature_cols

        return X, y, merged

    def train(self, X, y, merged, verbose=True):
        """Walk-Forward训练 + Purging"""
        print(f"\n🎯 Walk-Forward训练 (Purging={self.embargo_days}天)...")

        # 使用优化的切分器
        splitter = PurgingEmbargoSplitter(
            train_months=self.train_months,
            valid_months=1,
            test_months=1,
            embargo_days=self.embargo_days
        )

        splits = splitter.split(merged, date_column='date')

        if len(splits) == 0:
            print("  ⚠️  数据不足")
            return self

        print(f"  ✓ 生成 {len(splits)} 个窗口")

        best_model = None
        best_score = -np.inf

        for i, (train_idx, valid_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            X_valid = X.iloc[valid_idx]
            y_valid = y[valid_idx]

            print(f"\n  窗口 {i+1}/{len(splits)}")

            # 训练集成模型
            ensemble = EnsembleVotingScorer(self.voting_strategy)
            ensemble.train(X_train, y_train, X_valid, y_valid, verbose=False)

            # ===== Precision@K 评估 =====
            metrics = PrecisionAtKEvaluator.evaluate_model(
                ensemble, X_valid, y_valid, self.top_percentile
            )

            # 使用 Precision@K 作为选择标准（而非AUC）
            score = metrics['precision']

            if score > best_score:
                best_score = score
                best_model = ensemble

        self.ensemble = best_model
        print(f"\n  ✓ 最佳模型 Precision@{int(self.top_percentile*100)}%: {best_score:.4f}")

        return self

    def predict(self, factor_data, price_data=None):
        """预测评分"""
        if price_data is not None:
            # 重新训练
            X, y, merged = self.prepare_data(
                factor_data, price_data, self.feature_names
            )
            self.train(X, y, merged, verbose=False)
            factor_data = merged.copy()

        print(f"\n🎯 预测评分...")

        valid_features = [c for c in self.feature_names if c in factor_data.columns]
        X = factor_data[valid_features].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        predicted_proba = self.ensemble.predict_proba(X)
        factor_data['ml_score'] = predicted_proba
        factor_data['position'] = factor_data.groupby('date')['ml_score'].rank(pct=True)

        print(f"  ✓ 预测完成")
        print(f"  ✓ Top 20% 平均分: {factor_data[factor_data['position']>=0.8]['ml_score'].mean():.4f}")

        return factor_data

    def _detect_price_column(self, df):
        candidates = ['close', 'Close', 'CLOSE', 'price', 'Price']
        for col in candidates:
            if col in df.columns:
                return col
        return None


# ============================================================================
# 使用示例
# ============================================================================

def demo_ultra_scorer():
    """演示超级评分器"""
    print("="*80)
    print("超级ML评分器 - 四大优化演示")
    print("="*80)

    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=400, freq='D')
    instruments = [f'STOCK_{i:03d}' for i in range(100)]

    data = []
    for date in dates:
        for inst in instruments:
            data.append({
                'date': date,
                'instrument': inst,
                'close': 100 + np.random.randn() * 10,
                'factor1': np.random.randn(),
                'factor2': np.random.randn(),
                'factor3': np.random.randn(),
                'industry': np.random.choice(['科技', '金融', '消费', '医药'])
            })

    df = pd.DataFrame(data)

    factor_cols = ['factor1', 'factor2', 'factor3']

    # 初始化超级评分器
    scorer = UltraMLScorer(
        target_period=5,
        top_percentile=0.20,
        embargo_days=5,
        neutralize_market=True,
        neutralize_industry=True,
        voting_strategy='average',
        train_months=6
    )

    # 准备数据
    X, y, merged = scorer.prepare_data(
        df, df, factor_cols
    )

    # 训练
    scorer.train(X, y, merged)

    # 预测
    result = scorer.predict(df.tail(500), df)

    print("\n" + "="*80)
    print("✅ 演示完成！")
    print("="*80)


if __name__ == '__main__':
    demo_ultra_scorer()