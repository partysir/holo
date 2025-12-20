"""
ML因子评分模块 v2.0 - 简化实盘版

核心功能：
1. Walk-Forward训练
2. 因子重要性分析
3. 分类/回归双模式

精简内容：
- 移除复杂的IC计算
- 移除行业评分
- 专注核心预测功能

作者：AI量化团队
日期：2025-12-19
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_squared_error

# LightGBM可选依赖
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM未安装，将使用RandomForest")


class AdvancedMLScorer:
    """
    高级ML评分器
    """
    
    def __init__(self,
                 model_type='lightgbm',
                 target_period=5,
                 top_percentile=0.2,
                 use_classification=True,
                 use_ic_features=False,
                 use_active_return=True,
                 train_months=12):
        """
        Args:
            model_type: 'lightgbm' 或 'random_forest'
            target_period: 预测周期（天）
            top_percentile: 分类目标：前X%为正样本
            use_classification: True=分类，False=回归
            use_ic_features: 是否使用IC加权特征（简化版忽略）
            use_active_return: 是否使用主动收益（忽略）
            train_months: 训练窗口（月）
        """
        self.model_type = model_type
        self.target_period = target_period
        self.top_percentile = top_percentile
        self.use_classification = use_classification
        self.train_months = train_months
        
        self.model = None
        self.feature_names = None
        self.feature_importance_df = None
    
    def prepare_training_data(self, factor_data, price_data, factor_columns):
        """
        准备训练数据
        
        Returns:
            X, y, merged_df
        """
        print("  [prepare_training_data] 开始准备数据...")
        
        # 确保price_data包含必要列
        if 'close' not in price_data.columns:
            raise ValueError("price_data必须包含'close'列")
        
        # 合并因子和价格
        merged = factor_data.merge(
            price_data[['date', 'instrument', 'close']],
            on=['date', 'instrument'],
            how='left'
        )
        
        # 按股票和日期排序
        merged = merged.sort_values(['instrument', 'date']).reset_index(drop=True)
        
        # 计算未来收益（使用向量化操作避免索引问题）
        future_prices = merged.groupby('instrument')['close'].shift(-self.target_period)
        current_prices = merged['close']
        merged['abs_return'] = (future_prices / current_prices) - 1
        
        # 构建目标变量
        if self.use_classification:
            merged['target'] = 0
            for date in merged['date'].unique():
                mask = merged['date'] == date
                rets = merged.loc[mask, 'abs_return']
                if len(rets) > 5:
                    thresh = rets.quantile(1 - self.top_percentile)
                    merged.loc[mask & (merged['abs_return'] >= thresh), 'target'] = 1
            target_col = 'target'
        else:
            target_col = 'abs_return'
        
        # 删除目标变量为空的行
        merged = merged.dropna(subset=[target_col])
        
        # 排除不需要的列
        exclude_cols = [
            'date', 'instrument', 'close', 'abs_return', 'target',
            'ml_score', 'position', 'score_rank', 'composite_score'
        ]
        
        # 确定特征列
        feature_cols = [c for c in merged.columns 
                       if c not in exclude_cols and pd.api.types.is_numeric_dtype(merged[c])]
        
        # 选择特征和目标
        X = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = merged[target_col].values
        
        self.feature_names = feature_cols
        
        print(f"  [prepare_training_data] 数据准备完成")
        print(f"    • 特征数量: {len(feature_cols)}")
        print(f"    • 样本数量: {len(X)}")
        print(f"    • 目标变量: {target_col}")
        
        return X, y, merged
    
    def train_walk_forward(self, X, y, merged_df, n_splits=3):
        """
        Walk-Forward训练
        """
        print("  [train_walk_forward] 开始训练...")
        
        # 按时间排序
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        # 创建时间序列分割器
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        models = []
        scores = []
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"    训练折叠 {i+1}/{n_splits}...")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 训练模型
            if self.use_classification:
                if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42
                    )
                else:
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=42
                    )
            else:
                if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                    model = lgb.LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=6,
                        random_state=42
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=6,
                        random_state=42
                    )
            
            # 训练
            model.fit(X_train, y_train)
            models.append(model)
            
            # 评估
            if self.use_classification:
                y_pred = model.predict_proba(X_test)[:, 1]
                score = roc_auc_score(y_test, y_pred)
            else:
                y_pred = model.predict(X_test)
                score = -mean_squared_error(y_test, y_pred)
            
            scores.append(score)
            print(f"      得分: {score:.4f}")
        
        # 选择最佳模型（最后一个）
        self.model = models[-1]
        
        print(f"  [train_walk_forward] 训练完成")
        print(f"    • 平均得分: {np.mean(scores):.4f}")
        print(f"    • 最佳得分: {np.max(scores):.4f}")
        
        return self
    
    def predict_scores(self, factor_data):
        """
        预测评分
        """
        print("  [predict_scores] 开始预测...")
        
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 准备特征数据
        X = factor_data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 预测
        if self.use_classification:
            predictions = self.model.predict_proba(X)[:, 1]
        else:
            predictions = self.model.predict(X)
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            'date': factor_data['date'].values,
            'instrument': factor_data['instrument'].values,
            'ml_score': predictions
        })
        
        # 计算排名百分位
        result['position'] = result.groupby('date')['ml_score'].rank(pct=True)
        
        # 合并回原数据
        for col in ['ml_score', 'position']:
            if col in factor_data.columns:
                factor_data = factor_data.drop(columns=[col])
        
        factor_data = factor_data.merge(result, on=['date', 'instrument'], how='left')
        
        print(f"  [predict_scores] 预测完成")
        print(f"    • 预测样本数: {len(predictions)}")
        
        return factor_data
    
    def get_feature_importance(self, top_n=10):
        """
        获取特征重要性
        """
        if self.model is None:
            return None
        
        importance = self.model.feature_importances_
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        return df.sort_values('importance', ascending=False).head(top_n)


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    np.random.seed(42)
    
    # 生成因子数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    instruments = [f'STOCK_{i:03d}' for i in range(50)]
    
    factor_data_list = []
    for date in dates:
        for instrument in instruments:
            factor_data_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'instrument': instrument,
                'factor1': np.random.randn(),
                'factor2': np.random.randn(),
                'factor3': np.random.randn(),
                'factor4': np.random.randn(),
                'factor5': np.random.randn()
            })
    
    factor_data = pd.DataFrame(factor_data_list)
    
    # 生成价格数据
    price_data_list = []
    for date in dates:
        for instrument in instruments:
            # 模拟价格走势
            price = 100 + np.cumsum(np.random.randn(1) * 0.02)[0]
            price_data_list.append({
                'date': date.strftime('%Y-%m-%d'),
                'instrument': instrument,
                'close': max(1, price)  # 确保价格为正
            })
    
    price_data = pd.DataFrame(price_data_list)
    
    print("生成测试数据完成")
    print(f"因子数据: {len(factor_data)} 行")
    print(f"价格数据: {len(price_data)} 行")
    
    # 测试ML评分器
    scorer = AdvancedMLScorer(
        model_type='random_forest',
        target_period=5,
        use_classification=True
    )
    
    # 准备训练数据
    factor_columns = [f'factor{i}' for i in range(1, 6)]
    X, y, merged_df = scorer.prepare_training_data(factor_data, price_data, factor_columns)
    
    # Walk-Forward训练
    scorer.train_walk_forward(X, y, merged_df, n_splits=3)
    
    # 预测评分
    scored_data = scorer.predict_scores(factor_data)
    
    # 特征重要性
    importance = scorer.get_feature_importance()
    print("\n特征重要性:")
    print(importance)
    
    print("\n测试完成!")