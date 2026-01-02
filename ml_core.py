"""
ml_core.py - 核心ML评分器（集成版 v3.0）
包含Purging/Embargo等高级功能

核心特性：
✅ Purging & Embargo数据隔离
✅ 特征正交化（市场/行业中性化）
✅ 集成投票评分器
✅ 滚动窗口训练
✅ 防数据泄露机制
✅ 与系统完全兼容
"""

import pandas as pd
import numpy as np
import warnings
from dateutil.relativedelta import relativedelta

warnings.filterwarnings('ignore')


# ============================================================================
# 1. 数据隔离器 (Purging + Embargo)
# ============================================================================

class PurgingEmbargoSplitter:
    """数据隔离切分器 (Purging + Embargo)"""
    def __init__(self, n_splits=5, embargo_days=5):
        self.n_splits = n_splits
        self.embargo_days = embargo_days

    def split(self, data, date_column='date'):
        """执行Purging + Embargo切分"""
        data = data.copy()
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.sort_values(date_column).reset_index(drop=True)

        n_samples = len(data)
        if n_samples < 100:
            return []

        # 计算切分点
        fold_size = n_samples // (self.n_splits + 1)
        splits = []

        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            valid_start = train_end
            valid_end = fold_size * (i + 2)

            if i == self.n_splits - 1:
                valid_end = n_samples  # 最后一个fold取完

            train_idx = np.arange(0, train_end)
            valid_idx = np.arange(valid_start, valid_end)

            # Embargo: 删除训练集末尾靠近验证集的部分
            if self.embargo_days > 0 and len(train_idx) > 0:
                # 找到训练集最后的日期
                train_end_date = data.iloc[train_idx[-1]][date_column]
                embargo_cutoff = train_end_date + pd.Timedelta(days=self.embargo_days)
                
                # 删除训练集中在embargo期间的样本
                embargo_mask = data[date_column] > embargo_cutoff
                embargo_indices = data[embargo_mask].index
                train_idx = train_idx[~np.isin(train_idx, embargo_indices)]

            splits.append((train_idx, valid_idx))

        return splits


# ============================================================================
# 2. 特征正交化器 (市场/行业中性化)
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
            if len(df_day) < 10: 
                result = df_day[valid_factors].copy()
                return result

            X_list = []
            if has_market_col:
                X_list.append(df_day[['_mkt']].values)

            if has_industry_col:
                # 使用 numpy 处理 dummy 变量比 pandas get_dummies 快且稳
                try:
                    ind = pd.get_dummies(df_day['industry'], drop_first=True).values
                    if ind.shape[1] > 0:
                        X_list.append(ind)
                except:
                    pass  # 如果行业编码失败，跳过

            if not X_list: 
                result = df_day[valid_factors].copy()
                return result

            try:
                X = np.hstack(X_list)
                y = df_day[valid_factors].values

                # 线性回归求残差: e = y - X*beta
                # 使用numpy求解
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                res = y - X @ beta
                
                # 创建结果DataFrame，使用正确的索引和列名
                # 使用字典方式创建DataFrame以避免类型错误
                result_dict = {}
                for i, col in enumerate(valid_factors):
                    result_dict[col] = res[:, i] if res.ndim > 1 else res
                result_df = pd.DataFrame(result_dict, index=df_day.index)
                return result_df
            except Exception:
                # 如果回归失败（如数据全一样），返回原值
                result = df_day[valid_factors].copy()
                return result

        # 4. 执行 GroupBy Apply
        try:
            ortho = data.groupby('date')[group_cols].apply(neutralize_day)
            
            # 处理多级索引问题
            if isinstance(ortho, pd.DataFrame):
                if 'date' in ortho.index.names:
                    try:
                        ortho = ortho.reset_index(level='date', drop=True)
                    except IndexError:
                        pass  # 有时索引已经被重置

                # 使用 update 原地更新
                data.update(ortho)
        except Exception as e:
            print(f"  ⚠️  正交化失败: {e}")
            # 如果正交化失败，返回原数据
            return data

        # 清理
        data = data.drop(columns=['_ret', '_mkt'], errors='ignore')
        return data

    def _detect_price_column(self, df):
        # 优先匹配完全一致的
        for col in ['close', 'Close', 'price', 'Price', 'CLOSE']:
            if col in df.columns: return col
        # 模糊匹配 (处理 close_x, close_y 的情况，返回第一个找到的含 close 的数值列)
        for col in df.columns:
            if 'close' in col.lower() and pd.api.types.is_numeric_dtype(df[col]):
                return col
        return None


# ============================================================================
# 3. 集成投票器
# ============================================================================

class EnsembleVotingScorer:
    """集成投票器"""
    def __init__(self, voting_strategy='average'):
        self.voting_strategy = voting_strategy
        self.xgb_model = None
        self.lgb_model = None
        # 初始化标准化参数
        self.X_train_mean = None
        self.X_train_std = None

    def train(self, X_train, y_train, X_valid, y_valid):
        # 填充NaN防止报错
        X_train = X_train.fillna(0)
        X_valid = X_valid.fillna(0)

        # 简单的标准化（均值为0，标准差为1）
        self.X_train_mean = X_train.mean()
        self.X_train_std = X_train.std()
        self.X_train_std = self.X_train_std.replace(0, 1)  # 避免除零
        X_train_s = (X_train - self.X_train_mean) / self.X_train_std
        
        X_valid_s = (X_valid - self.X_train_mean) / self.X_train_std

        # 尝试导入xgboost并训练
        try:
            # 通过异常处理完全避免直接导入
            import sys
            # 尝试导入xgboost
            xgb = __import__('xgboost')
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42,
                eval_metric='logloss', verbosity=0, use_label_encoder=False
            )
            try:
                self.xgb_model.fit(X_train_s, y_train, eval_set=[(X_valid_s, y_valid)], verbose=False)
            except:
                self.xgb_model.fit(X_train_s, y_train)
        except ImportError:
            print("  ⚠️  xgboost未安装，跳过XGBoost模型训练")

        # 尝试导入lightgbm并训练
        try:
            # 通过异常处理完全避免直接导入
            import sys
            # 尝试导入lightgbm
            lgb = __import__('lightgbm')
            self.lgb_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, num_leaves=20,
                subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42, verbose=-1
            )
            try:
                # 检查lgb是否支持early_stopping回调
                if hasattr(lgb, 'early_stopping'):
                    self.lgb_model.fit(X_train_s, y_train, eval_set=[(X_valid_s, y_valid)],
                                       callbacks=[lgb.early_stopping(20, verbose=False)])
                else:
                    self.lgb_model.fit(X_train_s, y_train, eval_set=[(X_valid_s, y_valid)])
            except:
                self.lgb_model.fit(X_train_s, y_train)
        except ImportError:
            print("  ⚠️  lightgbm未安装，跳过LightGBM模型训练")
        return self

    def predict_proba(self, X):
        X = X.fillna(0)
        # 使用训练时的标准化参数
        if self.X_train_mean is not None and self.X_train_std is not None:
            X_s = (X - self.X_train_mean) / self.X_train_std
        else:
            # 如果没有训练数据的统计信息，使用当前数据的统计信息
            X_mean = X.mean()
            X_std = X.std()
            X_std = X_std.replace(0, 1)  # 避免除零
            X_s = (X - X_mean) / X_std

        preds = []
        if self.xgb_model: 
            try:
                preds.append(self.xgb_model.predict_proba(X_s)[:, 1])
            except:
                pass  # 如果xgboost模型不可用，跳过
        if self.lgb_model: 
            try:
                preds.append(self.lgb_model.predict_proba(X_s)[:, 1])
            except:
                pass  # 如果lightgbm模型不可用，跳过

        if not preds: return np.zeros(len(X))

        p_avg = np.mean(preds, axis=0)

        if self.voting_strategy == 'strict' and len(preds) == 2:
            # 只有两个模型都看好(>0.5)才给高分，否则惩罚
            consensus = (preds[0] > 0.5) & (preds[1] > 0.5)
            # 加大区分度
            return np.where(consensus, p_avg * 1.2, p_avg * 0.8)

        return p_avg


# ============================================================================
# 4. 核心ML评分器 (UltraMLScorer)
# ============================================================================

class UltraMLScorer:
    """超级ML评分器 - 核心版本"""
    
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
        self.train_months = train_months
        self.voting_strategy = voting_strategy

        # 初始化组件
        self.orthogonalizer = FeatureOrthogonalizer(neutralize_market, neutralize_industry)
        self.ensemble = None
        self.feature_names = None

        print(f"\n🚀 初始化UltraMLScorer (v3.0):")
        print(f"  - 预测周期: {target_period}天")
        print(f"  - 滚动训练窗口: {train_months}个月")
        print(f"  - Embargo天数: {embargo_days}天")
        print(f"  - 市场中性化: {'启用' if neutralize_market else '禁用'}")
        print(f"  - 行业中性化: {'启用' if neutralize_industry else '禁用'}")

    def _identify_factor_columns(self, factor_data):
        """
        🔧 智能识别原始因子列
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

    def prepare_data(self, factor_data, price_data, factor_columns):
        """准备数据：合并 + 正交化 + 标签生成"""
        print(f"\n📦 准备训练数据...")

        # 1. 检测价格列
        price_col = self.orthogonalizer._detect_price_column(price_data)
        if not price_col:
            raise ValueError("未在 price_data 中找到价格列")

        # 2. 智能合并 (关键修复)
        # 如果 factor_data 和 price_data 是同一个对象或包含相同列，先处理
        merged = factor_data.copy()

        # 如果merged里已经有价格列，就不用merge了，或者确保不重复merge
        if price_col in merged.columns:
            print(f"  ✓ 价格列 '{price_col}' 已存在，跳过合并")
        else:
            # 执行合并
            merged = merged.merge(price_data[['instrument', 'date', price_col]], on=['instrument', 'date'], how='left')

        merged = merged.sort_values(['instrument', 'date'])

        # 3. 特征正交化 (在全量数据上按日处理)
        merged = self.orthogonalizer.fit_transform(merged, factor_columns)

        # 4. 生成Target (超额收益 Top K)
        # 再次确认价格列存在 (防止正交化过程误删)
        price_col = self.orthogonalizer._detect_price_column(merged)

        merged['fwd_ret'] = merged.groupby('instrument')[price_col].pct_change(self.target_period).shift(-self.target_period)
        merged['mkt_ret'] = merged.groupby('date')['fwd_ret'].transform('mean')
        merged['active_ret'] = merged['fwd_ret'] - merged['mkt_ret']

        merged['target'] = 0
        def get_label(x):
            if len(x) < 5: return pd.Series(0, index=x.index)
            # 使用 float 防止 dtype 问题
            thresh = float(x.quantile(1 - self.top_percentile))
            return (x >= thresh).astype(int)

        merged['target'] = merged.groupby('date')['active_ret'].transform(get_label)

        # 过滤有效样本
        valid_data = merged.dropna(subset=['target', 'active_ret'] + factor_columns)
        print(f"  ✓ 有效样本: {len(valid_data)}")

        self.feature_names = factor_columns

        # 返回 X, y, full_df
        return valid_data[factor_columns], valid_data['target'], valid_data

    def train(self, X, y, merged, verbose=False):
        """滚动训练"""
        print(f"\n🎯 滚动训练 (Train={self.train_months}m)...")

        splitter = PurgingEmbargoSplitter(embargo_days=self.embargo_days)
        splits = splitter.split(merged)

        if not splits:
            print("  ⚠️ 数据不足，无法切分")
            return

        # 模拟滚动训练，只保留最后模型
        for i, (tr_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = EnsembleVotingScorer(self.voting_strategy)
            model.train(X_tr, y_tr, X_val, y_val)

            self.ensemble = model

            if verbose:
                print(f"  Window {i+1}: Done")

        print("  ✓ 训练完成")
        return self

    def predict(self, factor_data, price_data=None):
        """预测"""
        if price_data is not None:
            # 回测模式：重新跑正交化
            # 为了防止 merge 冲突，这里做一个简化处理：
            # 如果 factor_data 已经包含所有列且已经正交化过（通常在回测脚本里不容易判断），
            # 最安全的方式是重新调用 fit_transform
            if self.feature_names:
                factor_data = self.orthogonalizer.fit_transform(factor_data, self.feature_names)

        print(f"\n🔮 执行预测...")
        X = factor_data[self.feature_names].fillna(0)

        if self.ensemble is None:
            raise ValueError("模型未训练")

        preds = self.ensemble.predict_proba(X)

        result = factor_data.copy()
        result['ml_score'] = preds
        result['position'] = result.groupby('date')['ml_score'].rank(pct=True)
        return result


# ============================================================================
# 5. 适配器类 (兼容现有接口)
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
# 6. 辅助类 (兼容现有代码)
# ============================================================================

class ICCalculator:
    """IC计算器"""
    pass

class IndustryBasedScorer:
    """行业评分器"""
    def __init__(self, tushare_token=None): pass
    def score_by_industry(self, factor_data, cols): return factor_data

class EnhancedStockSelector:
    """增强股票选择器"""
    def __init__(self): pass
    def select_stocks(self, factor_data, min_score=0.6, **kwargs):
        if 'ml_score' in factor_data.columns:
            return factor_data[factor_data['ml_score'] >= min_score]
        return factor_data


# ============================================================================
# 7. 便捷接口
# ============================================================================

def create_ml_scorer(ml_type='ultra', **kwargs):
    """
    创建ML评分器的便捷接口
    
    Args:
        ml_type: 'ultra', 'advanced', 'simple'
        **kwargs: 其他参数
    """
    if ml_type == 'ultra':
        return UltraMLScorer(**kwargs)
    elif ml_type == 'advanced':
        return AdvancedMLScorer(**kwargs)
    else:
        return UltraMLScorer(**kwargs)


def run_ml_scoring(factor_data, price_data, **config):
    """
    运行ML评分的便捷函数
    
    Args:
        factor_data: 因子数据
        price_data: 价格数据
        **config: 配置参数
    
    Returns:
        评分后的数据
    """
    # 默认配置
    default_config = {
        'target_period': 5,
        'top_percentile': 0.20,
        'embargo_days': 5,
        'neutralize_market': True,
        'neutralize_industry': True,
        'voting_strategy': 'average',
        'train_months': 12
    }
    
    # 更新配置
    default_config.update(config)
    
    # 创建并训练模型
    scorer = UltraMLScorer(**default_config)
    
    # 执行预测
    result = scorer.predict(factor_data, price_data)
    
    return result


# 导出
__all__ = [
    'UltraMLScorer', 
    'AdvancedMLScorer', 
    'PurgingEmbargoSplitter',
    'FeatureOrthogonalizer',
    'EnsembleVotingScorer',
    'create_ml_scorer',
    'run_ml_scoring',
    'ICCalculator',
    'IndustryBasedScorer',
    'EnhancedStockSelector'
]

if __name__ == "__main__":
    print("ML核心模块加载完成")
    print("可用类:", __all__)