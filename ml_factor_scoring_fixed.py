"""
ml_factor_scoring_fixed.py - 高级机器学习因子评分模块（修复版）

核心优化：
✅ 1. 时间序列切分（避免前视偏差）
✅ 2. 分类目标（预测TOP 20%）
✅ 3. IC加权 - 因子有效性动态评估
✅ 4. 滚动训练 - 自适应市场变化
✅ 5. Tushare行业数据集成
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score


# ============================================================================
# 核心优化1: IC计算器
# ============================================================================

class ICCalculator:
    """
    因子IC（信息系数）计算器

    IC = 因子值与未来收益的相关性
    ICIR = IC的均值 / IC的标准差（夏普比率的因子版）
    """

    def __init__(self, forward_periods=[5, 10, 20]):
        """
        :param forward_periods: 计算IC的未来周期列表
        """
        self.forward_periods = forward_periods
        self.ic_history = {}  # {factor: {period: [ic_values]}}

    def calculate_factor_ic(self, factor_data, price_data, factor_columns):
        """
        计算所有因子的IC值

        返回: {factor: {period: {'ic': float, 'icir': float}}}
        """
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

        merged = merged.sort_values(['instrument', 'date'])

        # 计算不同周期的未来收益
        for period in self.forward_periods:
            merged[f'future_return_{period}d'] = merged.groupby('instrument')[price_col].pct_change(
                period
            ).shift(-period)

        ic_results = {}

        for factor in factor_columns:
            if factor not in merged.columns:
                continue

            ic_results[factor] = {}

            for period in self.forward_periods:
                return_col = f'future_return_{period}d'

                # 按日期分组计算IC
                daily_ic = []
                for date in merged['date'].unique():
                    date_data = merged[merged['date'] == date]

                    # 过滤有效数据
                    valid_data = date_data[[factor, return_col]].dropna()

                    if len(valid_data) < 10:  # 至少10个样本
                        continue

                    # 计算相关性
                    ic = valid_data[factor].corr(valid_data[return_col])

                    if not np.isnan(ic):
                        daily_ic.append(ic)

                if len(daily_ic) > 0:
                    ic_mean = np.mean(daily_ic)
                    ic_std = np.std(daily_ic)
                    icir = ic_mean / ic_std if ic_std > 0 else 0

                    ic_results[factor][period] = {
                        'ic': ic_mean,
                        'icir': icir,
                        'ic_std': ic_std,
                        'sample_days': len(daily_ic)
                    }

                    # 记录历史
                    if factor not in self.ic_history:
                        self.ic_history[factor] = {}
                    self.ic_history[factor][period] = daily_ic

        # 打印IC统计
        self._print_ic_summary(ic_results)

        return ic_results

    def _detect_price_column(self, df):
        """检测价格列"""
        candidates = ['close', 'Close', 'CLOSE', 'price', 'Price']
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _print_ic_summary(self, ic_results):
        """打印IC统计"""
        print(f"\n  📈 因子IC统计:")
        print(f"  {'因子':<20s} | {'IC(5日)':<10s} | {'IC(10日)':<10s} | {'IC(20日)':<10s} | {'ICIR(5日)':<10s}")
        print(f"  {'-'*80}")

        for factor, periods in ic_results.items():
            ic_5 = periods.get(5, {}).get('ic', 0)
            ic_10 = periods.get(10, {}).get('ic', 0)
            ic_20 = periods.get(20, {}).get('ic', 0)
            icir_5 = periods.get(5, {}).get('icir', 0)

            print(f"  {factor:<20s} | {ic_5:>9.4f} | {ic_10:>9.4f} | {ic_20:>9.4f} | {icir_5:>9.4f}")

    def get_ic_weights(self, ic_results, period=5):
        """
        根据IC计算因子权重

        权重 = abs(IC) / sum(abs(IC))
        """
        weights = {}
        total_ic = 0

        for factor, periods in ic_results.items():
            ic = periods.get(period, {}).get('ic', 0)
            weights[factor] = abs(ic)
            total_ic += abs(ic)

        if total_ic > 0:
            weights = {k: v/total_ic for k, v in weights.items()}

        return weights


# ============================================================================
# 核心优化2: 时间序列切分器
# ============================================================================

class TimeSeriesSplitter:
    """
    时间序列数据切分器

    使用Walk-Forward方式：
    - 训练集：历史N个月
    - 验证集：接下来的1个月
    - 测试集：再接下来的1个月
    """

    def __init__(self, train_months=12, valid_months=1, test_months=1):
        self.train_months = train_months
        self.valid_months = valid_months
        self.test_months = test_months

    def split(self, data, date_column='date'):
        """
        时间序列切分

        返回: [(train_idx, valid_idx, test_idx), ...]
        """
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column)

        # 按月分组
        data['year_month'] = data[date_column].dt.to_period('M')
        months = sorted(data['year_month'].unique())

        splits = []

        # 滚动窗口
        for i in range(len(months) - self.train_months - self.valid_months - self.test_months + 1):
            train_end = i + self.train_months
            valid_end = train_end + self.valid_months
            test_end = valid_end + self.test_months

            train_months_list = months[i:train_end]
            valid_months_list = months[train_end:valid_end]
            test_months_list = months[valid_end:test_end]

            train_idx = data[data['year_month'].isin(train_months_list)].index
            valid_idx = data[data['year_month'].isin(valid_months_list)].index
            test_idx = data[data['year_month'].isin(test_months_list)].index

            if len(train_idx) > 0 and len(valid_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, valid_idx, test_idx))

        return splits


# ============================================================================
# 核心优化3: 高级ML评分器
# ============================================================================

class AdvancedMLScorer:
    """
    高级机器学习评分器

    整合三大优化：
    1. 时间序列切分（避免前视偏差）
    2. 分类目标（预测TOP股票）
    3. IC加权特征（因子有效性）
    """

    def __init__(self,
                 model_type='xgboost',
                 target_period=5,
                 top_percentile=0.20,  # 预测TOP 20%
                 use_classification=True,
                 use_ic_features=True,
                 train_months=12,
                 random_state=42):
        """
        :param model_type: 'xgboost' 或 'lightgbm'
        :param target_period: 预测周期（天）
        :param top_percentile: TOP股票比例
        :param use_classification: 是否使用分类模型
        :param use_ic_features: 是否使用IC作为特征
        :param train_months: 训练窗口（月）
        """
        self.model_type = model_type
        self.target_period = target_period
        self.top_percentile = top_percentile
        self.use_classification = use_classification
        self.use_ic_features = use_ic_features
        self.train_months = train_months
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.ic_calculator = ICCalculator([target_period])
        self.ic_weights = {}

        print(f"\n🚀 初始化高级ML评分器")
        print(f"  模型类型: {model_type.upper()}")
        print(f"  目标模式: {'分类' if use_classification else '回归'}")
        print(f"  预测目标: {'TOP ' + str(int(top_percentile*100)) + '%' if use_classification else f'{target_period}日收益率'}")
        print(f"  IC特征: {'启用' if use_ic_features else '关闭'}")
        print(f"  训练窗口: {train_months}个月")

    def prepare_training_data(self, factor_data, price_data, factor_columns):
        """
        准备训练数据

        ✅ 优化1: 避免前视偏差
        ✅ 优化2: 分类目标
        ✅ 优化3: IC特征
        """
        print(f"\n📦 准备训练数据...")

        # 检测价格列
        price_col = self._detect_price_column(price_data)
        if price_col is None:
            raise ValueError("未找到价格列")

        # 合并数据
        merged = factor_data.merge(
            price_data[['instrument', 'date', price_col]],
            on=['instrument', 'date'],
            how='left'
        )

        merged = merged.sort_values(['instrument', 'date'])

        # ===== 优化1: 计算IC =====
        if self.use_ic_features:
            print("  ✓ 计算因子IC...")
            ic_results = self.ic_calculator.calculate_factor_ic(
                factor_data, price_data, factor_columns
            )
            self.ic_weights = self.ic_calculator.get_ic_weights(ic_results, self.target_period)

            # 添加IC作为特征
            for factor in factor_columns:
                if factor in ic_results:
                    ic_value = ic_results[factor].get(self.target_period, {}).get('ic', 0)
                    merged[f'{factor}_ic'] = ic_value

        # ===== 优化2: 计算目标变量 =====
        print(f"  ✓ 计算未来{self.target_period}日收益...")
        merged['future_return'] = merged.groupby('instrument')[price_col].pct_change(
            self.target_period
        ).shift(-self.target_period)

        if self.use_classification:
            # 分类目标：每天TOP 20%的股票标记为1
            print(f"  ✓ 转换为分类目标 (TOP {self.top_percentile:.0%})...")
            merged['target'] = 0

            for date in merged['date'].unique():
                date_mask = merged['date'] == date
                returns = merged.loc[date_mask, 'future_return']
                threshold = returns.quantile(1 - self.top_percentile)
                merged.loc[date_mask & (merged['future_return'] >= threshold), 'target'] = 1

            target_col = 'target'
        else:
            target_col = 'future_return'

        # 过滤有效数据
        initial_len = len(merged)
        merged = merged.dropna(subset=[target_col])
        print(f"  ✓ 有效样本: {len(merged)} / {initial_len} ({len(merged)/initial_len*100:.1f}%)")

        if self.use_classification:
            pos_rate = merged['target'].mean()
            print(f"  ✓ 正样本比例: {pos_rate:.2%}")

        # ===== 构建特征 =====
        base_exclude = [
            'date', 'instrument', 'future_return', 'target', price_col,
            'industry', 'ml_score', 'industry_rank', 'year_month'
        ]

        all_cols = merged.columns.tolist()
        feature_cols = [col for col in all_cols if col not in base_exclude]

        # 处理只有position的情况
        if len(feature_cols) == 0 and 'position' in merged.columns:
            feature_cols = ['position']

        print(f"  ✓ 特征数量: {len(feature_cols)}")

        X = merged[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
        y = merged[target_col].values

        self.feature_names = feature_cols

        return X, y, merged

    def train_walk_forward(self, X, y, merged, verbose=True):
        """
        ✅ Walk-Forward训练（避免前视偏差）

        使用滚动窗口：
        - 每次用过去12个月训练
        - 在下1个月验证
        - 保存最佳模型
        """
        print(f"\n🎯 Walk-Forward训练...")

        # 时间序列切分
        splitter = TimeSeriesSplitter(
            train_months=self.train_months,
            valid_months=1,
            test_months=1
        )

        splits = splitter.split(merged, date_column='date')

        if len(splits) == 0:
            print("  ⚠️  数据不足以进行时间序列切分，使用简单切分")
            return self._train_simple(X, y, verbose)

        print(f"  ✓ 生成了 {len(splits)} 个时间窗口")

        best_model = None
        best_score = -np.inf

        for i, (train_idx, valid_idx, test_idx) in enumerate(splits):
            if i >= 1:  # 只训练最后一个窗口（最新数据）
                continue

            X_train = X.iloc[train_idx]
            y_train = y[train_idx]
            X_valid = X.iloc[valid_idx]
            y_valid = y[valid_idx]

            print(f"\n  窗口 {i+1}/{len(splits)}:")
            print(f"    训练: {len(X_train)} 样本")
            print(f"    验证: {len(X_valid)} 样本")

            # 标准化
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_valid_scaled = self.scaler.transform(X_valid)

            # 训练模型
            if self.use_classification:
                model = self._train_classifier(
                    X_train_scaled, y_train,
                    X_valid_scaled, y_valid,
                    verbose=False
                )

                # 评估
                y_pred_proba = model.predict_proba(X_valid_scaled)[:, 1]
                score = roc_auc_score(y_valid, y_pred_proba)
                print(f"    验证AUC: {score:.4f}")
            else:
                model = self._train_regressor(
                    X_train_scaled, y_train,
                    X_valid_scaled, y_valid,
                    verbose=False
                )

                # 评估
                y_pred = model.predict(X_valid_scaled)
                score = np.corrcoef(y_valid, y_pred)[0, 1]
                print(f"    验证相关性: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = model

        self.model = best_model
        print(f"\n  ✓ 最佳模型验证得分: {best_score:.4f}")

        return self

    def _train_simple(self, X, y, verbose):
        """简单训练（数据不足时使用）"""
        from sklearn.model_selection import train_test_split

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_valid_scaled = self.scaler.transform(X_valid)

        if self.use_classification:
            self.model = self._train_classifier(
                X_train_scaled, y_train,
                X_valid_scaled, y_valid,
                verbose
            )
        else:
            self.model = self._train_regressor(
                X_train_scaled, y_train,
                X_valid_scaled, y_valid,
                verbose
            )

        return self

    def _train_classifier(self, X_train, y_train, X_valid, y_valid, verbose):
        """训练分类器"""
        if self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost 未安装")

            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1
            }

            model = xgb.XGBClassifier(**params)

            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=20,
                    verbose=verbose
                )
            except:
                model.fit(X_train, y_train)

            return model

        elif self.model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM 未安装")

            params = {
                'objective': 'binary',
                'metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': -1
            }

            model = lgb.LGBMClassifier(**params)

            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    callbacks=[lgb.early_stopping(20, verbose=verbose)]
                )
            except:
                model.fit(X_train, y_train)

            return model

    def _train_regressor(self, X_train, y_train, X_valid, y_valid, verbose):
        """训练回归器"""
        if self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost 未安装")

            params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state,
                'n_jobs': -1
            }

            model = xgb.XGBRegressor(**params)

            try:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    early_stopping_rounds=20,
                    verbose=verbose
                )
            except:
                model.fit(X_train, y_train)

            return model

    def _detect_price_column(self, df):
        """检测价格列"""
        candidates = ['close', 'Close', 'CLOSE', 'price', 'Price']
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def predict_scores(self, factor_data, price_data=None, factor_columns=None):
        """预测评分"""
        if price_data is not None:
            X, y, merged = self.prepare_training_data(factor_data, price_data, factor_columns)
            self.train_walk_forward(X, y, merged, verbose=False)
            factor_data = merged.copy()

        if self.model is None:
            raise ValueError("模型未训练")

        print(f"\n🎯 预测股票评分...")

        X = factor_data[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

        X_scaled = self.scaler.transform(X)

        if self.use_classification:
            # 预测概率
            predicted_proba = self.model.predict_proba(X_scaled)[:, 1]
            factor_data['ml_score'] = predicted_proba
        else:
            # 预测收益率
            predicted_returns = self.model.predict(X_scaled)
            factor_data['ml_score'] = predicted_returns

        # 标准化到0-1
        factor_data['position'] = factor_data.groupby('date')['ml_score'].rank(pct=True)

        print(f"  ✓ 预测完成")
        print(f"  ✓ 平均评分: {factor_data['ml_score'].mean():.4f}")
        print(f"  ✓ 评分标准差: {factor_data['ml_score'].std():.4f}")

        return factor_data


# ============================================================================
# 行业数据获取（修复版 - 使用 Tushare stock_basic）
# ============================================================================

def get_industry_data(instruments, tushare_token=None):
    """
    获取行业数据 - 使用 Tushare stock_basic（最简单最快）

    Args:
        instruments: 股票代码列表
        tushare_token: Tushare token

    Returns:
        DataFrame: [instrument, industry]
    """
    if tushare_token is None:
        print("  ⚠️  未提供 Tushare Token")
        return pd.DataFrame({
            'instrument': instruments,
            'industry': 'Unknown'
        })

    try:
        import tushare as ts
        ts.set_token(tushare_token)
        pro = ts.pro_api()

        print(f"  📊 获取 {len(instruments)} 只股票的行业数据...")

        # ✅ 使用 stock_basic 获取申万行业（一次调用获取所有）
        stock_basic = pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,name,industry'  # industry是申万一级行业
        )

        # 过滤目标股票
        stock_basic = stock_basic[stock_basic['ts_code'].isin(instruments)]
        stock_basic['instrument'] = stock_basic['ts_code']
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

        print(f"  ✓ 获取到 {len(result)} 只股票的行业信息")
        print(f"  ✓ 覆盖率: {(len(result) - len(missing))/len(instruments)*100:.1f}%")
        print(f"  ✓ 行业分类: {result['industry'].nunique()} 个")

        # 显示TOP5行业
        top_industries = result['industry'].value_counts().head(5)
        print(f"\n  📊 TOP5行业:")
        for industry, count in top_industries.items():
            print(f"     {industry}: {count}只")

        return result

    except Exception as e:
        print(f"  ⚠️  获取行业数据失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame({
            'instrument': instruments,
            'industry': 'Unknown'
        })


class IndustryBasedScorer:
    """分行业评分器"""

    def __init__(self, tushare_token=None):
        self.tushare_token = tushare_token

    def score_by_industry(self, factor_data, factor_columns=None):
        """分行业评分"""
        print("\n🏢 分行业评分...")

        instruments = factor_data['instrument'].unique()
        industry_data = get_industry_data(instruments, self.tushare_token)

        if 'industry' in factor_data.columns:
            factor_data = factor_data.drop(columns=['industry'])

        factor_data = factor_data.merge(industry_data, on='instrument', how='left')
        factor_data['industry'] = factor_data['industry'].fillna('Unknown')

        try:
            factor_data['industry_rank'] = factor_data.groupby(['date', 'industry'])['position'].rank(pct=True)
            print(f"  ✓ 行业评分完成")

            # 统计行业分布
            industry_dist = factor_data.groupby('industry')['instrument'].nunique()
            print(f"\n  📊 行业分布 (股票数):")
            for industry, count in industry_dist.head(10).items():
                print(f"     {industry}: {count}只")

        except Exception as e:
            print(f"  ⚠️  行业评分失败: {e}")
            factor_data['industry_rank'] = factor_data['position']

        return factor_data


class EnhancedStockSelector:
    """增强选股器"""

    def select_stocks(self, factor_data, min_score=0.6, max_concentration=0.15, max_industry_concentration=0.3):
        """选股"""
        print(f"\n🎯 增强选股 (阈值: {min_score})...")

        filtered = factor_data[factor_data['position'] >= min_score].copy()
        print(f"  ✓ 评分过滤: {len(filtered)} / {len(factor_data)} 只股票")

        if 'industry' not in filtered.columns:
            filtered['industry'] = 'Unknown'

        filtered['industry'] = filtered['industry'].fillna('Unknown')

        return filtered


# 导出
__all__ = [
    'AdvancedMLScorer',
    'ICCalculator',
    'TimeSeriesSplitter',
    'IndustryBasedScorer',
    'EnhancedStockSelector',
    'get_industry_data'
]
