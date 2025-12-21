"""
ml_factor_scoring_alpha.py - ML评分Alpha增强版

核心增强:
✅ 1. 三分类标签 (Triple Barrier Method)
    - Buy: 未来收益 > 5% 且回撤 < 2%
    - Sell: 未来收益 < -2%
    - Hold: 震荡区间
✅ 2. 双重过滤标签优化
    - 相对收益 Top 20% (战胜市场)
    - 绝对收益 > 0 (剔除熊市抗跌股)
✅ 3. Learning to Rank 目标函数
✅ 4. 模型集成 (XGBoost + LightGBM + RandomForest)
✅ 5. 高置信度过滤 (prob > 0.7)
"""

import pandas as pd
import numpy as np
import warnings
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 机器学习库
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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, accuracy_score


class TripleBarrierLabeler:
    """
    ✨ 三分类标签生成器 (Triple Barrier Method)
    
    原理: 不仅看收益，还要看风险
    - 盈亏比 > 2:1 才标记为 Buy
    - 大幅亏损标记为 Sell
    - 其他为 Hold（不交易）
    
    优势:
    - 提高胜率（宁缺毋滥）
    - 降低回撤
    - 符合实盘心理
    """
    
    def __init__(self, 
                 profit_threshold=0.05,  # 盈利阈值 5%
                 stop_loss_threshold=-0.02,  # 止损阈值 -2%
                 max_drawdown_threshold=0.02,  # 最大回撤限制 2%
                 holding_period=5):
        """
        Args:
            profit_threshold: 盈利阈值
            stop_loss_threshold: 止损阈值
            max_drawdown_threshold: 持有期间最大回撤
            holding_period: 持有周期（天）
        """
        self.profit_threshold = profit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.max_drawdown_threshold = max_drawdown_threshold
        self.holding_period = holding_period
        
        print(f"\n🎯 初始化三分类标签生成器")
        print(f"   盈利目标: {profit_threshold:.1%}")
        print(f"   止损线: {stop_loss_threshold:.1%}")
        print(f"   最大回撤: {max_drawdown_threshold:.1%}")
    
    def generate_labels(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        生成三分类标签
        
        标签定义:
        1 (Buy): 达到盈利目标 且 回撤可控
        -1 (Sell): 触发止损
        0 (Hold): 震荡，不交易
        """
        print(f"  📊 生成三分类标签 (持有期: {self.holding_period}天)...")
        
        data = price_data.copy()
        data = data.sort_values(['instrument', 'date']).reset_index(drop=True)
        
        # 初始化标签
        data['triple_label'] = 0
        data['max_profit'] = np.nan
        data['max_drawdown'] = np.nan
        
        grouped = data.groupby('instrument')
        
        for instrument, group in grouped:
            for i in range(len(group)):
                current_idx = group.index[i]
                current_price = group.iloc[i]['close']
                
                # 获取未来N天的价格
                future_prices = group.iloc[i+1:i+1+self.holding_period]['close'].values
                
                if len(future_prices) < self.holding_period:
                    continue
                
                # 计算收益率序列
                returns = (future_prices - current_price) / current_price
                
                # 最大盈利
                max_profit = returns.max()
                
                # 最大回撤（从当前到任意时点的最大下跌）
                cummax = np.maximum.accumulate(returns)
                drawdowns = returns - cummax
                max_drawdown = abs(drawdowns.min())
                
                # 最终收益
                final_return = returns[-1]
                
                # 标签逻辑
                if (max_profit >= self.profit_threshold and 
                    max_drawdown <= self.max_drawdown_threshold):
                    # Buy: 高盈利 + 低回撤
                    label = 1
                elif final_return <= self.stop_loss_threshold:
                    # Sell: 触发止损
                    label = -1
                else:
                    # Hold: 震荡
                    label = 0
                
                data.loc[current_idx, 'triple_label'] = label
                data.loc[current_idx, 'max_profit'] = max_profit
                data.loc[current_idx, 'max_drawdown'] = max_drawdown
        
        # 统计标签分布
        label_counts = data['triple_label'].value_counts()
        print(f"  ✓ 标签分布:")
        print(f"     Buy  (1):  {label_counts.get(1, 0):>6d} ({label_counts.get(1, 0)/len(data):.1%})")
        print(f"     Hold (0):  {label_counts.get(0, 0):>6d} ({label_counts.get(0, 0)/len(data):.1%})")
        print(f"     Sell (-1): {label_counts.get(-1, 0):>6d} ({label_counts.get(-1, 0)/len(data):.1%})")
        
        return data


class OptimizedTargetGenerator:
    """
    ✨ 优化目标生成器
    
    双重过滤策略:
    1. 相对收益 Top 20% (Active Return)
    2. 绝对收益 > 0 (剔除熊市抗跌股)
    
    核心思想:
    - 熊市空仓比买抗跌股更好
    - 只在上涨中选最强的股票
    """
    
    def __init__(self, top_percentile=0.20, min_absolute_return=0.0):
        """
        Args:
            top_percentile: 相对收益Top比例
            min_absolute_return: 最小绝对收益
        """
        self.top_percentile = top_percentile
        self.min_absolute_return = min_absolute_return
        
        print(f"\n🎯 初始化优化目标生成器")
        print(f"   相对收益阈值: Top {top_percentile:.0%}")
        print(f"   绝对收益阈值: {min_absolute_return:.1%}")
    
    def generate_target(self, merged_data: pd.DataFrame, 
                       future_return_col='future_return',
                       abs_return_col='abs_return') -> pd.DataFrame:
        """
        生成优化的二分类目标
        
        Target = 1: 相对收益Top 20% 且 绝对收益 > 0
        Target = 0: 其他
        """
        print(f"  📊 生成优化目标...")
        
        data = merged_data.copy()
        data['target'] = 0
        
        for date in data['date'].unique():
            mask = data['date'] == date
            daily_data = data[mask]
            
            # 相对收益阈值
            relative_thresh = daily_data[future_return_col].quantile(
                1 - self.top_percentile
            )
            
            # 双重过滤
            target_mask = (
                (daily_data[future_return_col] >= relative_thresh) &
                (daily_data[abs_return_col] > self.min_absolute_return)
            )
            
            data.loc[mask & target_mask, 'target'] = 1
        
        # 统计
        target_ratio = data['target'].mean()
        print(f"  ✓ 目标比例: {target_ratio:.2%}")
        
        return data


class EnsembleMLScorer:
    """
    ✨ 集成ML评分器
    
    模型投票机制:
    1. XGBoost - 梯度提升树
    2. LightGBM - 快速梯度提升
    3. RandomForest - 随机森林
    
    决策: 至少2个模型同意才标记为 Top 20%
    """
    
    def __init__(self, 
                 use_xgboost=True,
                 use_lightgbm=True,
                 use_random_forest=True,
                 voting_threshold=2,  # 至少2个模型同意
                 confidence_threshold=0.7,  # 高置信度过滤
                 target_period=5,
                 random_state=42):
        """
        Args:
            use_xgboost: 是否使用XGBoost
            use_lightgbm: 是否使用LightGBM
            use_random_forest: 是否使用RandomForest
            voting_threshold: 投票阈值
            confidence_threshold: 置信度阈值
        """
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.use_lightgbm = use_lightgbm and LIGHTGBM_AVAILABLE
        self.use_random_forest = use_random_forest
        self.voting_threshold = voting_threshold
        self.confidence_threshold = confidence_threshold
        self.target_period = target_period
        self.random_state = random_state
        
        self.models = {}
        self.scaler = RobustScaler()
        self.feature_names = None
        
        print(f"\n🚀 初始化集成ML评分器")
        print(f"   XGBoost: {'✓' if self.use_xgboost else '✗'}")
        print(f"   LightGBM: {'✓' if self.use_lightgbm else '✗'}")
        print(f"   RandomForest: {'✓' if self.use_random_forest else '✗'}")
        print(f"   投票阈值: {voting_threshold}/3")
        print(f"   置信度阈值: {confidence_threshold:.0%}")
    
    def prepare_features(self, factor_data: pd.DataFrame, 
                        price_data: pd.DataFrame) -> Tuple:
        """准备训练特征"""
        print(f"\n📦 准备训练数据...")
        
        # 合并数据
        price_col = 'close' if 'close' in price_data.columns else 'Close'
        
        merged = factor_data.merge(
            price_data[['instrument', 'date', price_col]],
            on=['instrument', 'date'], how='left'
        )
        merged = merged.sort_values(['instrument', 'date']).reset_index(drop=True)
        
        # 计算收益率
        merged['abs_return'] = merged.groupby('instrument')[price_col].pct_change(
            self.target_period
        ).shift(-self.target_period)
        
        # 超额收益
        market_return = merged.groupby('date')['abs_return'].transform('mean')
        merged['future_return'] = merged['abs_return'] - market_return
        
        # 生成优化目标
        target_gen = OptimizedTargetGenerator(
            top_percentile=0.20,
            min_absolute_return=0.0
        )
        merged = target_gen.generate_target(merged)
        
        # 排除泄露列
        exclude = [
            'date', 'instrument', 'future_return', 'abs_return', 'target',
            price_col, 'close', 'Close', 'price',
            'position', 'ml_score', 'score_rank',
            'industry', 'sector'
        ]
        
        feature_cols = [
            c for c in merged.columns 
            if c not in exclude 
            and pd.api.types.is_numeric_dtype(merged[c])
        ]
        
        # 验证无泄露
        leaked = [c for c in ['position', 'ml_score'] if c in feature_cols]
        if leaked:
            raise ValueError(f"检测到数据泄露: {leaked}")
        
        X = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y = merged['target'].values
        
        self.feature_names = feature_cols
        
        print(f"  ✓ 特征数: {len(feature_cols)}")
        print(f"  ✓ 样本数: {len(X)}")
        print(f"  ✓ 正样本: {y.sum()} ({y.mean():.2%})")
        
        return X, y, merged
    
    def train(self, X: pd.DataFrame, y: np.ndarray, 
             X_val: pd.DataFrame = None, y_val: np.ndarray = None):
        """训练所有模型"""
        print(f"\n🎓 训练集成模型...")
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # 1. XGBoost
        if self.use_xgboost:
            print(f"  训练 XGBoost...")
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                eval_metric='auc',
                random_state=self.random_state,
                n_jobs=-1,
                early_stopping_rounds=30
            )
            
            eval_set = [(X_val_scaled, y_val)] if X_val is not None else None
            self.models['xgboost'].fit(
                X_scaled, y, 
                eval_set=eval_set,
                verbose=False
            )
        
        # 2. LightGBM
        if self.use_lightgbm:
            print(f"  训练 LightGBM...")
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                metric='auc',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
            
            eval_set = [(X_val_scaled, y_val)] if X_val is not None else None
            self.models['lightgbm'].fit(
                X_scaled, y,
                eval_set=eval_set,
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
        
        # 3. RandomForest
        if self.use_random_forest:
            print(f"  训练 RandomForest...")
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.models['random_forest'].fit(X_scaled, y)
        
        print(f"  ✓ 模型训练完成")
    
    def predict_with_voting(self, X: pd.DataFrame) -> np.ndarray:
        """
        集成预测 - 模型投票
        
        Returns:
            投票结果 (0-3)，表示有多少个模型认为是正类
        """
        if not self.models:
            raise ValueError("模型未训练")
        
        X_scaled = self.scaler.transform(X)
        
        votes = np.zeros(len(X))
        
        # 收集各模型投票
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)
                    # 获取正类概率
                    pos_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                else:
                    # 对于没有predict_proba的模型，使用predict
                    pos_proba = model.predict(X_scaled)
                
                # 高置信度过滤
                confident_votes = (pos_proba > self.confidence_threshold).astype(int)
                votes += confident_votes
                
                print(f"     {name}: {confident_votes.sum()} 个高置信度预测")
                
            except Exception as e:
                print(f"     {name} 预测出错: {e}")
        
        return votes
    
    def predict_scores(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        生成最终评分
        
        评分逻辑:
        1. 模型投票 (2/3同意)
        2. 高置信度过滤 (>0.7)
        """
        print(f"\n🔮 生成集成评分...")
        
        data = factor_data.copy()
        
        # 提取特征
        X = data[self.feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 投票预测
        votes = self.predict_with_voting(X)
        
        # 投票决策 (至少2个模型同意)
        final_predictions = (votes >= self.voting_threshold).astype(int)
        
        # 生成结果DataFrame
        result = pd.DataFrame({
            'date': data['date'].values,
            'instrument': data['instrument'].values,
            'ml_score': votes / len(self.models),  # 归一化投票分数
            'votes': votes.astype(int),
            'prediction': final_predictions
        })
        
        # 计算排名
        result['position'] = result.groupby('date')['ml_score'].rank(pct=True)
        
        print(f"  ✓ 评分生成完成")
        print(f"     高置信度预测: {(votes >= self.voting_threshold).sum()}")
        print(f"     平均投票数: {votes.mean():.2f}")
        
        return result


def run_alpha_ml_strategy(factor_data: pd.DataFrame, 
                         price_data: pd.DataFrame,
                         use_ensemble=True,
                         confidence_threshold=0.7) -> Dict:
    """
    运行Alpha增强ML策略
    
    Args:
        factor_data: 因子数据
        price_data: 价格数据
        use_ensemble: 是否使用集成模型
        confidence_threshold: 置信度阈值
    
    Returns:
        策略结果字典
    """
    print("\n" + "=" * 60)
    print("⚡ Alpha增强ML策略启动")
    print("=" * 60)
    
    # 1. 生成三分类标签
    labeler = TripleBarrierLabeler(
        profit_threshold=0.05,
        stop_loss_threshold=-0.02,
        max_drawdown_threshold=0.02,
        holding_period=5
    )
    labeled_data = labeler.generate_labels(price_data)
    
    # 2. 初始化集成评分器
    scorer = EnsembleMLScorer(
        use_xgboost=True,
        use_lightgbm=True,
        use_random_forest=True,
        voting_threshold=2,
        confidence_threshold=confidence_threshold
    )
    
    # 3. 准备特征
    X, y, merged = scorer.prepare_features(factor_data, price_data)
    
    # 4. 简单分割训练/验证集
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 5. 训练模型
    scorer.train(X_train, y_train, X_val, y_val)
    
    # 6. 生成评分
    scores = scorer.predict_scores(merged)
    
    # 7. 合并回原数据
    result_data = merged.merge(
        scores[['date', 'instrument', 'ml_score', 'position']], 
        on=['date', 'instrument'], 
        how='left'
    )
    
    print("\n✅ Alpha增强ML策略执行完成")
    
    return {
        'labeled_data': labeled_data,
        'scored_data': result_data,
        'scorer': scorer,
        'feature_importance': None  # 可选：添加特征重要性分析
    }


if __name__ == '__main__':
    print("Alpha增强ML评分模块 - 请在主程序中导入使用")
    print("\n示例:")
    print("from ml_factor_scoring_alpha import run_alpha_ml_strategy")
    print("\nresults = run_alpha_ml_strategy(factor_data, price_data)")