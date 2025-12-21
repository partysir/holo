"""
data_module_alpha_enhanced.py - Alpha增强因子工程模块

核心增强:
✅ 1. Alpha101 风格因子: 量价相关性、乖离率、低波因子
✅ 2. 微观结构因子: 日内强度、成交占比
✅ 3. 因子正交化: 去除多重共线性
✅ 4. 高级动量因子: 多周期复合动量

使用方法:
    from data_module_alpha_enhanced import EnhancedFactorGenerator
    
    generator = EnhancedFactorGenerator()
    enhanced_df = generator.generate_all_factors(price_data)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Optional


class EnhancedFactorGenerator:
    """
    Alpha增强因子生成器
    
    集成:
    - Alpha101 经典因子
    - 微观结构因子
    - 多周期动量复合
    - 因子正交化
    """
    
    def __init__(self, enable_orthogonalization=True, debug=False):
        """
        Args:
            enable_orthogonalization: 是否启用因子正交化
            debug: 是否输出调试信息
        """
        self.enable_orthogonalization = enable_orthogonalization
        self.debug = debug
        
        print(f"\n🚀 初始化Alpha增强因子生成器")
        print(f"   因子正交化: {'启用' if enable_orthogonalization else '禁用'}")
    
    def calculate_alpha101_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✨ Alpha101 风格因子
        
        这些非线性因子能捕捉更多市场异象:
        1. 量价相关性 (Smart Money)
        2. 乖离率 (Bias) - 均值回归
        3. 低波因子 (Low Volatility Anomaly)
        """
        print("  📊 计算Alpha101因子...")
        
        data = df.copy()
        grouped = data.groupby('instrument')
        
        # 1. 日内强度 (Intraday Strength)
        # 衡量主力资金买入/卖出压力
        # (收盘-开盘) / (最高-最低) * 成交量占比
        data['intraday_strength'] = (
            (data['close'] - data['open']) / 
            ((data['high'] - data['low']) + 1e-6)
        )
        
        # 2. 量价相关性 (Volume-Price Correlation)
        # 量价齐升通常比缩量上涨更可靠
        def rolling_corr_price_volume(x):
            """10日滚动相关性"""
            if len(x) < 10:
                return pd.Series(np.nan, index=x.index)
            return x['close'].rolling(10).corr(x['volume'])
        
        # 简化版：使用价格变化与成交量变化的相关性
        data['price_chg'] = grouped['close'].pct_change()
        data['volume_chg'] = grouped['volume'].pct_change()
        
        # 10日滚动相关性
        data['vol_price_corr'] = grouped.apply(
            lambda x: x['price_chg'].rolling(10).corr(x['volume_chg'])
        ).reset_index(level=0, drop=True)
        
        # 3. 乖离率 (Bias Rate)
        # 价格偏离均线程度，捕捉超买超卖
        data['ma_20'] = grouped['close'].transform(lambda x: x.rolling(20).mean())
        data['bias_20'] = (data['close'] - data['ma_20']) / (data['ma_20'] + 1e-6)
        
        # 4. 低波因子 (Low Volatility Preference)
        # 低波动股票长期表现优于高波动股票
        data['volatility_20'] = grouped['close'].transform(
            lambda x: x.rolling(20).std()
        )
        data['low_vol_score'] = 1.0 / (data['volatility_20'] + 1e-6)
        
        # 5. Alpha006 简化版: -1 * Correlation(Open, Volume, 10)
        # 开盘价与成交量负相关表示机构逆向操作
        data['open_chg'] = grouped['open'].pct_change()
        data['alpha006'] = -1 * grouped.apply(
            lambda x: x['open_chg'].rolling(10).corr(x['volume_chg'])
        ).reset_index(level=0, drop=True)
        
        if self.debug:
            print(f"     新增5个Alpha101因子")
        
        return data
    
    def calculate_microstructure_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✨ 微观结构因子
        
        捕捉日内高频交易的降频信号:
        1. 买卖压力不平衡
        2. 成交额占比
        3. 价格跳跃
        """
        print("  🔬 计算微观结构因子...")
        
        data = df.copy()
        grouped = data.groupby('instrument')
        
        # 1. 买卖压力 (Buy/Sell Pressure)
        # 使用影线长度衡量
        data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
        data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
        data['shadow_ratio'] = (
            (data['upper_shadow'] - data['lower_shadow']) / 
            ((data['high'] - data['low']) + 1e-6)
        )
        
        # 2. 成交额占比 (Amount Ratio)
        # 该股票成交额相对于总成交额的占比变化
        if 'amount' in data.columns:
            data['amount_ma5'] = grouped['amount'].transform(
                lambda x: x.rolling(5).mean()
            )
            data['amount_ma20'] = grouped['amount'].transform(
                lambda x: x.rolling(20).mean()
            )
            data['amount_ratio'] = data['amount_ma5'] / (data['amount_ma20'] + 1e-6)
        else:
            # 如果没有amount列，用volume替代
            data['volume_ma5'] = grouped['volume'].transform(
                lambda x: x.rolling(5).mean()
            )
            data['volume_ma20'] = grouped['volume'].transform(
                lambda x: x.rolling(20).mean()
            )
            data['amount_ratio'] = data['volume_ma5'] / (data['volume_ma20'] + 1e-6)
        
        # 3. 价格跳跃 (Price Jump)
        # 开盘价相对于前一日收盘价的跳空
        data['price_jump'] = grouped.apply(
            lambda x: (x['open'] - x['close'].shift(1)) / (x['close'].shift(1) + 1e-6)
        ).reset_index(level=0, drop=True)
        
        if self.debug:
            print(f"     新增4个微观结构因子")
        
        return data
    
    def calculate_composite_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✨ 多周期复合动量
        
        不同周期的动量有不同含义:
        - 短期(5-10日): 反转效应
        - 中期(20-60日): 趋势延续
        - 长期(120-250日): 价值回归
        """
        print("  📈 计算复合动量因子...")
        
        data = df.copy()
        grouped = data.groupby('instrument')
        
        # 多周期动量
        periods = [5, 10, 20, 60, 120]
        
        for p in periods:
            data[f'momentum_{p}d'] = grouped['close'].pct_change(p)
        
        # 复合动量: 加权平均
        # 短期权重小，长期权重大（捕捉趋势）
        if all(f'momentum_{p}d' in data.columns for p in periods):
            weights = np.array([0.1, 0.15, 0.25, 0.3, 0.2])  # 权重和为1
            
            momentum_cols = [f'momentum_{p}d' for p in periods]
            data['composite_momentum'] = (
                data[momentum_cols].fillna(0) * weights
            ).sum(axis=1)
        
        # 动量加速度 (Momentum Acceleration)
        # 动量的变化率
        data['momentum_accel'] = grouped['momentum_20d'].diff()
        
        if self.debug:
            print(f"     新增{len(periods)+2}个动量因子")
        
        return data
    
    def calculate_volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ✨ 波动率因子簇
        
        波动率的多维度刻画:
        1. 历史波动率 (多周期)
        2. 波动率偏度 (Volatility Skew)
        3. 上行/下行波动率
        """
        print("  📊 计算波动率因子...")
        
        data = df.copy()
        grouped = data.groupby('instrument')
        
        # 计算收益率
        data['returns'] = grouped['close'].pct_change()
        
        # 1. 多周期历史波动率
        for period in [5, 10, 20, 60]:
            data[f'volatility_{period}d'] = grouped['returns'].transform(
                lambda x: x.rolling(period).std() * np.sqrt(252)  # 年化
            )
        
        # 2. 上行/下行波动率 (Upside/Downside Volatility)
        # 分别计算正收益和负收益的波动率
        data['upside_vol'] = grouped['returns'].transform(
            lambda x: x[x > 0].rolling(20, min_periods=5).std()
        )
        data['downside_vol'] = grouped['returns'].transform(
            lambda x: x[x < 0].rolling(20, min_periods=5).std()
        )
        
        # 波动率偏度
        data['vol_skew'] = (data['upside_vol'] - data['downside_vol']) / (
            data['upside_vol'] + data['downside_vol'] + 1e-6
        )
        
        if self.debug:
            print(f"     新增7个波动率因子")
        
        return data
    
    def orthogonalize_factors(self, df: pd.DataFrame, 
                              factor_columns: list) -> pd.DataFrame:
        """
        ✨ 因子正交化 (Orthogonalization)
        
        目的: 去除因子之间的多重共线性
        方法: 对每个因子，去除其他因子的线性影响
        
        优势:
        - 让XGBoost学到更纯粹的信息
        - 提升模型稳定性
        - 减少过拟合
        """
        if not self.enable_orthogonalization:
            return df
        
        print("  🔧 因子正交化...")
        
        data = df.copy()
        
        # 按日期分组正交化
        orthogonalized_data = []
        
        for date in data['date'].unique():
            date_mask = data['date'] == date
            daily_data = data[date_mask].copy()
            
            if len(daily_data) < 10:  # 样本太少跳过
                orthogonalized_data.append(daily_data)
                continue
            
            # 提取因子数据
            X = daily_data[factor_columns].fillna(0)
            
            # 标准化
            X_mean = X.mean()
            X_std = X.std()
            X_normalized = (X - X_mean) / (X_std + 1e-6)
            
            # 正交化：对每个因子，减去其他因子的投影
            X_ortho = X_normalized.copy()
            
            for i, factor in enumerate(factor_columns):
                # 其他因子
                other_factors = [f for f in factor_columns if f != factor]
                
                if len(other_factors) == 0:
                    continue
                
                # 回归
                y = X_normalized[factor].values.reshape(-1, 1)
                X_others = X_normalized[other_factors].values
                
                try:
                    # 去除其他因子的影响
                    reg = LinearRegression()
                    reg.fit(X_others, y)
                    predicted = reg.predict(X_others)
                    residual = y - predicted
                    
                    X_ortho[factor] = residual.flatten()
                except:
                    pass  # 回归失败保持原值
            
            # 重新标准化
            X_ortho = (X_ortho - X_ortho.mean()) / (X_ortho.std() + 1e-6)
            
            # 更新数据
            for factor in factor_columns:
                daily_data[f'{factor}_ortho'] = X_ortho[factor].values
            
            orthogonalized_data.append(daily_data)
        
        result = pd.concat(orthogonalized_data, ignore_index=True)
        
        if self.debug:
            print(f"     正交化完成，新增{len(factor_columns)}个正交因子")
        
        return result
    
    def generate_all_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        生成所有Alpha增强因子
        
        Args:
            price_data: 价格数据，需包含 open, high, low, close, volume
        
        Returns:
            增强后的数据框，包含所有新因子
        """
        print("\n" + "=" * 60)
        print("🔬 Alpha增强因子工程")
        print("=" * 60)
        
        df = price_data.copy()
        
        # 1. Alpha101 因子
        df = self.calculate_alpha101_factors(df)
        
        # 2. 微观结构因子
        df = self.calculate_microstructure_factors(df)
        
        # 3. 复合动量因子
        df = self.calculate_composite_momentum(df)
        
        # 4. 波动率因子
        df = self.calculate_volatility_factors(df)
        
        # 5. 识别所有新生成的因子
        new_factor_columns = [
            'intraday_strength', 'vol_price_corr', 'bias_20', 'low_vol_score', 'alpha006',
            'shadow_ratio', 'amount_ratio', 'price_jump',
            'composite_momentum', 'momentum_accel',
            'vol_skew'
        ]
        
        # 添加多周期动量和波动率
        new_factor_columns += [f'momentum_{p}d' for p in [5, 10, 20, 60, 120]]
        new_factor_columns += [f'volatility_{p}d' for p in [5, 10, 20, 60]]
        new_factor_columns += ['upside_vol', 'downside_vol']
        
        # 过滤出实际存在的因子
        existing_factors = [f for f in new_factor_columns if f in df.columns]
        
        # 6. 因子正交化（可选）
        if self.enable_orthogonalization and len(existing_factors) > 0:
            df = self.orthogonalize_factors(df, existing_factors)
        
        print("\n✅ 因子生成完成")
        print(f"   总计新增: {len(existing_factors)} 个原始因子")
        if self.enable_orthogonalization:
            print(f"   正交化后: {len(existing_factors)} 个正交因子")
        
        return df