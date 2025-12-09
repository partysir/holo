"""
ml_factor_scoring_fixed.py - 修复版机器学习因子评分模块

修复内容：
1. ✅ 修复IndustryBasedScorer行业数据获取
2. ✅ 修复EnhancedStockSelector行业列访问
3. ✅ 改进错误处理和提示信息
4. ✅ 修复特征列检测逻辑 - 处理只有position列的情况
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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================================
# 工具函数
# ============================================================================

def detect_price_column(df):
    """智能检测价格列"""
    price_candidates = [
        'close', 'Close', 'CLOSE',
        'close_price', 'closing_price', 
        'price', 'Price'
    ]
    
    for col in price_candidates:
        if col in df.columns:
            print(f"  ✓ 检测到价格列: {col}")
            return col
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"  ⚠️  未找到标准价格列，可用数值列: {numeric_cols}")
    return None


def get_industry_data(instruments, tushare_token=None):
    """获取股票行业信息"""
    print("\n🏢 获取行业数据...")
    
    if tushare_token is None:
        print("  ⚠️  未提供 Tushare Token，使用默认行业分类")
        return pd.DataFrame({
            'instrument': instruments,
            'industry': 'Unknown'
        })
    
    try:
        import tushare as ts
        ts.set_token(tushare_token)
        pro = ts.pro_api()
        
        stock_basic = pro.stock_basic(
            exchange='',
            list_status='L',
            fields='ts_code,industry'
        )
        
        stock_basic['instrument'] = stock_basic['ts_code']
        
        industry_data = stock_basic[
            stock_basic['instrument'].isin(instruments)
        ][['instrument', 'industry']]
        
        industry_data['industry'] = industry_data['industry'].fillna('Unknown')
        
        print(f"  ✓ 获取了 {len(industry_data)} 只股票的行业信息")
        industry_count = industry_data['industry'].nunique()
        print(f"  ✓ 涵盖 {industry_count} 个行业")
        
        return industry_data
        
    except Exception as e:
        print(f"  ⚠️  获取行业数据失败: {e}")
        print(f"  使用默认行业分类")
        return pd.DataFrame({
            'instrument': instruments,
            'industry': 'Unknown'
        })


# ============================================================================
# 核心类: MLFactorScorer
# ============================================================================

class MLFactorScorer:
    """机器学习因子评分器"""
    
    def __init__(self, model_type='xgboost', target_period=5, random_state=42):
        self.model_type = model_type
        self.target_period = target_period
        self.random_state = random_state
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        
        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost 未安装")
        if model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM 未安装")
    
    def prepare_training_data(self, factor_data, price_data):
        """准备训练数据"""
        print(f"\n🤖 准备训练数据 (目标周期: {self.target_period}日)...")
        
        price_col = detect_price_column(price_data)
        if price_col is None:
            raise ValueError("未找到价格列，无法准备训练数据")
        
        merged = factor_data.merge(
            price_data[['instrument', 'date', price_col]],
            on=['instrument', 'date'],
            how='left'
        )
        
        print(f"  ✓ 合并数据: {len(merged)} 条记录")
        
        merged = merged.sort_values(['instrument', 'date'])
        merged['future_return'] = merged.groupby('instrument')[price_col].pct_change(
            self.target_period
        ).shift(-self.target_period)
        
        initial_len = len(merged)
        merged = merged.dropna(subset=['future_return'])
        
        print(f"  ✓ 计算未来{self.target_period}日收益率")
        print(f"  ✓ 有效样本: {len(merged)} / {initial_len} ({len(merged)/initial_len*100:.1f}%)")
        
        # ✨ 关键修复：更智能的特征列检测
        # 基础排除列（必须排除的）
        base_exclude = [
            'date', 'instrument', 'future_return', price_col,
            'industry', 'ml_score', 'industry_rank'
        ]
        
        # 检查是否有足够的特征列
        all_numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        potential_features = [col for col in all_numeric_cols if col not in base_exclude]
        
        print(f"  ✓ 检测到 {len(potential_features)} 个潜在特征列")
        
        # ✨ 如果只有position列，将其作为特征（不排除）
        if len(potential_features) == 0:
            print("  ⚠️  警告：没有检测到常规特征列")
            # 尝试使用position作为特征
            if 'position' in merged.columns:
                print("  ✓ 使用 'position' 作为唯一特征列")
                feature_cols = ['position']
            else:
                raise ValueError("没有任何可用的特征列用于训练")
        elif len(potential_features) == 1 and potential_features[0] == 'position':
            # 如果唯一的特征就是position，直接使用
            feature_cols = ['position']
            print("  ✓ 使用 'position' 作为唯一特征列")
        else:
            # 正常情况：排除position，使用其他技术因子
            feature_cols = [col for col in potential_features if col != 'position']
            if len(feature_cols) == 0:
                # 如果排除position后没有其他特征，还是使用position
                feature_cols = ['position']
                print("  ✓ 使用 'position' 作为唯一特征列")
            else:
                print(f"  ✓ 使用 {len(feature_cols)} 个特征列（已排除position）")
        
        X = merged[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        
        if X.isnull().all().all():
            raise ValueError("特征数据全为NaN，无法进行训练")
        
        X = X.fillna(X.median())
        y = merged['future_return'].values
        
        if len(y) == 0 or np.isnan(y).all():
            raise ValueError("目标值无效，无法进行训练")
        
        self.feature_names = feature_cols
        
        print(f"  ✓ 最终特征数量: {len(feature_cols)}")
        if len(feature_cols) > 0:
            print(f"  ✓ 特征列表: {', '.join(feature_cols[:10])}{'...' if len(feature_cols) > 10 else ''}")
        
        return X, y, merged
    
    def train(self, X, y, test_size=0.2, verbose=True):
        """训练模型"""
        print(f"\n🚀 训练 {self.model_type.upper()} 模型...")
        
        if len(X) == 0 or len(y) == 0:
            raise ValueError("训练数据为空")
        
        if len(X) != len(y):
            raise ValueError(f"特征矩阵和目标标签长度不匹配: {len(X)} vs {len(y)}")
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        y = np.array(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            shuffle=True
        )
        
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("数据集划分后训练集或测试集为空")
        
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        
        try:
            X_train_values = X_train.values
            X_test_values = X_test.values
            
            if np.isnan(X_train_values).any() or np.isinf(X_train_values).any():
                print("  ⚠️  训练数据中存在NaN或无穷值，进行清理...")
                X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            if np.isnan(X_test_values).any() or np.isinf(X_test_values).any():
                print("  ⚠️  测试数据中存在NaN或无穷值，进行清理...")
                X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        except Exception as e:
            print(f"  ⚠️  数据检查失败: {e}")
        
        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("数据清理后训练集或测试集为空")
        
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        except Exception as e:
            print(f"  ⚠️  标准化失败: {e}")
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        if self.model_type == 'xgboost':
            self.model = self._train_xgboost(
                X_train_scaled, y_train, 
                X_test_scaled, y_test,
                verbose
            )
        elif self.model_type == 'lightgbm':
            self.model = self._train_lightgbm(
                X_train_scaled, y_train,
                X_test_scaled, y_test,
                verbose
            )
        
        try:
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            train_corr = np.corrcoef(np.array(y_train), np.array(train_pred))[0, 1] if len(y_train) > 1 else 0
            test_corr = np.corrcoef(np.array(y_test), np.array(test_pred))[0, 1] if len(y_test) > 1 else 0
            
            print(f"\n  📊 模型评估:")
            print(f"     训练集相关性: {train_corr:.4f}")
            print(f"     测试集相关性: {test_corr:.4f}")
        except Exception as e:
            print(f"  ⚠️  模型评估失败: {e}")
        
        self._extract_feature_importance()
        return self
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val, verbose):
        """训练 XGBoost 模型"""
        if not XGBOOST_AVAILABLE or xgb is None:
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
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=verbose
            )
        except TypeError:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=verbose
            )
        
        return model
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val, verbose):
        """训练 LightGBM 模型"""
        if not LIGHTGBM_AVAILABLE or lgb is None:
            raise ImportError("LightGBM 未安装")
            
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        
        try:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(20, verbose=verbose)]
            )
        except Exception:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)]
            )

        return model
    
    def _extract_feature_importance(self):
        """提取特征重要性"""
        if self.model is None:
            return
        
        importance = self.model.feature_importances_
        importance_sum = np.sum(importance)
        if importance_sum > 0:
            importance = importance / importance_sum
        
        if self.feature_names is not None:
            self.feature_importance = dict(zip(list(self.feature_names), list(importance)))
        
        if self.feature_importance:
            sorted_importance = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            print(f"\n  🎯 特征重要性 TOP5:")
            for i, (feature, score) in enumerate(sorted_importance[:5], 1):
                print(f"     {i}. {feature}: {score:.4f}")
    
    def predict_scores(self, factor_data, price_data=None):
        """预测评分"""
        if price_data is not None:
            X, y, merged = self.prepare_training_data(factor_data, price_data)
            self.train(X, y, verbose=False)
            factor_data = merged.copy()
        
        if self.model is None:
            raise ValueError("模型未训练，请提供 price_data 或先调用 train()")
        
        print(f"\n🎯 预测股票评分...")
        
        X = factor_data[self.feature_names].copy()
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        X_scaled = self.scaler.transform(X)
        predicted_returns = self.model.predict(X_scaled)
        
        factor_data['ml_score'] = predicted_returns
        factor_data['position'] = factor_data.groupby('date')['ml_score'].rank(pct=True)
        
        print(f"  ✓ 预测完成")
        pred_returns_array = np.array(predicted_returns)
        print(f"  ✓ 平均预测收益: {pred_returns_array.mean():.4f}")
        print(f"  ✓ 预测收益标准差: {pred_returns_array.std():.4f}")
        
        return factor_data
    
    def get_feature_importance(self):
        """获取特征重要性字典"""
        if self.feature_importance is None:
            return {}
        return self.feature_importance
    
    def dynamic_weight_adjustment(self, factor_data, factor_columns):
        """动态权重调整"""
        weights = {}
        for col in factor_columns:
            if col in factor_data.columns:
                std = factor_data[col].std()
                weights[col] = 1.0 / (std + 1e-6) if std > 0 else 1.0
        
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
            
        return weights


# ============================================================================
# 分行业评分器（修复版）
# ============================================================================

class IndustryBasedScorer:
    """分行业评分器"""
    
    def __init__(self, tushare_token=None):
        self.tushare_token = tushare_token
        self.industry_data = None
    
    def score_by_industry(self, factor_data, factor_columns=None):
        """分行业评分"""
        print("\n🏢 分行业评分...")
        
        # 1. 获取行业数据
        instruments = factor_data['instrument'].unique()
        self.industry_data = get_industry_data(instruments, self.tushare_token)
        
        # 2. 合并行业数据
        if self.industry_data is not None and len(self.industry_data) > 0:
            if 'industry' in factor_data.columns:
                factor_data = factor_data.drop(columns=['industry'])
            
            factor_data = factor_data.merge(
                self.industry_data,
                on='instrument',
                how='left'
            )
            
            factor_data['industry'] = factor_data['industry'].fillna('Unknown')
            
            print(f"  ✓ 成功合并行业数据")
            print(f"  ✓ 涵盖行业数: {factor_data['industry'].nunique()}")
        
        # 3. 确保有industry列
        if 'industry' not in factor_data.columns:
            print("  ⚠️  未找到行业数据，添加默认行业")
            factor_data['industry'] = 'Unknown'
        
        # 4. 按行业分组进行排名
        try:
            factor_data['industry_rank'] = factor_data.groupby(['date', 'industry'])['position'].rank(pct=True)
            print(f"  ✓ 行业评分完成")
        except Exception as e:
            print(f"  ⚠️  行业排名失败: {e}")
            factor_data['industry_rank'] = factor_data['position']
        
        return factor_data


# ============================================================================
# 增强选股器（修复版）
# ============================================================================

class EnhancedStockSelector:
    """增强选股器"""
    
    def __init__(self):
        pass
    
    def select_stocks(self, factor_data, min_score=0.6, max_concentration=0.15, max_industry_concentration=0.3):
        """选股"""
        print(f"\n🎯 增强选股 (阈值: {min_score})...")
        
        # 1. 过滤低分股票
        filtered = factor_data[factor_data['position'] >= min_score].copy()
        print(f"  ✓ 评分过滤: {len(filtered)} / {len(factor_data)} 只股票")
        
        # 2. 确保有行业列
        if 'industry' not in filtered.columns:
            print("  ⚠️  缺少行业信息，添加默认行业")
            filtered['industry'] = 'Unknown'
        
        # 3. 填充缺失的行业值
        filtered['industry'] = filtered['industry'].fillna('Unknown')
        
        # 4. 按行业分组，控制行业集中度
        max_stocks_per_industry = max(1, int(len(filtered) * max_industry_concentration))
        
        selected_stocks = []
        industry_selected = {}
        
        # 按评分排序
        filtered = filtered.sort_values('position', ascending=False)
        
        for idx, row in filtered.iterrows():
            industry = row['industry']
            
            if pd.isna(industry):
                industry = 'Unknown'
            
            if industry not in industry_selected:
                industry_selected[industry] = 0
            
            if industry_selected[industry] < max_stocks_per_industry:
                selected_stocks.append(idx)
                industry_selected[industry] += 1
        
        # 5. 返回选中的股票
        if len(selected_stocks) == 0:
            print("  ⚠️  没有选中任何股票，返回原数据")
            return filtered
        
        result = factor_data.loc[selected_stocks].copy()
        print(f"  ✓ 行业分散: 最终选择 {len(result)} 只股票")
        
        if 'industry' in result.columns:
            print(f"  ✓ 涉及行业: {result['industry'].nunique()} 个")
        
        return result


# 导出类和函数
__all__ = [
    'MLFactorScorer',
    'IndustryBasedScorer',
    'EnhancedStockSelector'
]