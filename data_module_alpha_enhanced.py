"""
data_module_alpha_enhanced.py - Alpha增强版数据模块

核心功能：
✅ 集成Alpha101风格因子计算
✅ 量价相关性、乖离率和波动率倒数等Alpha因子
✅ 高效的因子计算与缓存机制
✅ 与现有数据流程兼容
"""

import pandas as pd
import numpy as np
import os
import pickle
import hashlib
import time
from datetime import datetime, timedelta
from collections import deque

# Tushare导入
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("⚠️  Tushare未安装: pip install tushare")


class AlphaFactorCalculator:
    """Alpha因子计算器 - 集成Alpha101风格因子"""
    
    def __init__(self, alpha_factors_enabled=True, custom_alpha_weights=None):
        self.alpha_factors_enabled = alpha_factors_enabled
        self.custom_alpha_weights = custom_alpha_weights or {}
        # 默认Alpha因子权重
        self.default_alpha_weights = {
            # 量价相关性因子
            'alpha_price_volume_corr_5d': 0.05,
            'alpha_price_volume_corr_10d': 0.05,
            'alpha_price_volume_corr_20d': 0.03,
            # 乖离率因子
            'alpha_bias_5d': 0.04,
            'alpha_bias_10d': 0.04,
            'alpha_bias_20d': 0.03,
            # 波动率倒数因子
            'alpha_volatility_inverse_5d': 0.03,
            'alpha_volatility_inverse_10d': 0.03,
            'alpha_volatility_inverse_20d': 0.02,
            # 价格动量因子
            'alpha_price_momentum_5d': 0.04,
            'alpha_price_momentum_10d': 0.04,
            'alpha_price_momentum_20d': 0.03,
        }
        self.alpha_weights = {**self.default_alpha_weights, **self.custom_alpha_weights}
        print(f"📊 Alpha因子计算器初始化: {len(self.alpha_weights)} 个因子")
    
    def calculate_alpha_factors(self, df):
        """计算Alpha101风格因子"""
        if not self.alpha_factors_enabled:
            return df
        
        print("\n🔍 计算Alpha101风格因子...")
        start_time = time.time()
        
        # 1. 量价相关性因子
        df = self._calculate_price_volume_correlation_factors(df)
        # 2. 乖离率因子
        df = self._calculate_bias_factors(df)
        # 3. 波动率倒数因子
        df = self._calculate_volatility_inverse_factors(df)
        # 4. 价格动量因子
        df = self._calculate_price_momentum_factors(df)
        
        elapsed = time.time() - start_time
        print(f"✓ Alpha因子计算完成 (耗时: {elapsed:.2f}秒)")
        return df
    
    def _calculate_price_volume_correlation_factors(self, df):
        """计算量价相关性因子"""
        df = df.copy()
        for period in [5, 10, 20]:
            # 计算价格和成交量的相关性
            corr_col = f'alpha_price_volume_corr_{period}d'
            df[corr_col] = df.groupby('instrument').apply(
                lambda x: x[['close', 'volume']].rolling(window=period).corr().iloc[::2].iloc[:, 1] if len(x) >= period else pd.Series([np.nan] * len(x))
            ).reset_index(level=0, drop=True).values
        return df
    
    def _calculate_bias_factors(self, df):
        """计算乖离率因子"""
        df = df.copy()
        for period in [5, 10, 20]:
            # 计算价格相对于均线的乖离率
            ma_col = f'ma_{period}d'
            bias_col = f'alpha_bias_{period}d'
            df[ma_col] = df.groupby('instrument')['close'].rolling(window=period).mean().reset_index(0, drop=True)
            df[bias_col] = (df['close'] - df[ma_col]) / df[ma_col]
        return df
    
    def _calculate_volatility_inverse_factors(self, df):
        """计算波动率倒数因子（波动率越小，因子值越大）"""
        df = df.copy()
        for period in [5, 10, 20]:
            # 计算波动率倒数（作为稳定性因子）
            vol_col = f'volatility_{period}d'
            inv_vol_col = f'alpha_volatility_inverse_{period}d'
            df[vol_col] = df.groupby('instrument')['close'].rolling(window=period).std().reset_index(0, drop=True)
            df[inv_vol_col] = 1.0 / (df[vol_col] + 1e-8)  # 避免除零
        return df
    
    def _calculate_price_momentum_factors(self, df):
        """计算价格动量因子"""
        df = df.copy()
        for period in [5, 10, 20]:
            # 计算价格动量
            momentum_col = f'alpha_price_momentum_{period}d'
            df[momentum_col] = df.groupby('instrument')['close'].pct_change(period)
        return df

    def get_alpha_factor_names(self):
        """获取所有Alpha因子名称"""
        return list(self.alpha_weights.keys())


class EnhancedStockRankerModel:
    """增强版StockRanker模型 - 集成Alpha因子"""
    
    def __init__(self, custom_weights=None, use_fundamental=True, use_money_flow=True, 
                 use_alpha_factors=True, money_flow_style='balanced', alpha_weights=None):
        self.use_fundamental = use_fundamental
        self.use_money_flow = use_money_flow
        self.use_alpha_factors = use_alpha_factors
        
        # 初始化Alpha因子计算器
        if self.use_alpha_factors:
            self.alpha_calculator = AlphaFactorCalculator(alpha_factors_enabled=True, custom_alpha_weights=alpha_weights)
        else:
            self.alpha_calculator = None
        
        # 初始化资金流计算器
        if self.use_money_flow:
            from money_flow_factors import MoneyFlowFactorCalculator
            self.money_flow_calculator = MoneyFlowFactorCalculator(
                use_full_tick_data=False,
                keep_only_essential=True
            )
            money_flow_weights = self.money_flow_calculator.get_recommended_weights(money_flow_style)
        else:
            money_flow_weights = {}
        
        if custom_weights:
            self.factor_weights = custom_weights
        else:
            # 基础因子权重
            base_weights = {}
            
            if use_fundamental and use_money_flow and use_alpha_factors:
                # 基本面 + 资金流 + Alpha模式
                base_weights = {
                    # 估值因子
                    'pe_ratio': -0.04, 'pb_ratio': -0.04, 'ps_ratio': -0.02,
                    # 波动率
                    'volatility_20d': -0.03, 'volatility_60d': -0.02,
                    # 成交量
                    'money_flow_20d': 0.03, 'volume_ratio': 0.02,
                    # 动量
                    'return_20d': 0.04, 'return_60d': 0.03,
                    # 基本面
                    'roe': 0.05, 'roa': 0.02,
                    'gross_margin': 0.02, 'net_margin': 0.02,
                    'debt_ratio': -0.03,
                }
                # 资金流权重
                base_weights.update(money_flow_weights)
                # Alpha权重
                if self.use_alpha_factors and self.alpha_calculator:
                    base_weights.update(self.alpha_calculator.alpha_weights)
                    
            elif use_fundamental and use_money_flow:
                # 基本面 + 资金流模式
                base_weights = {
                    'pe_ratio': -0.06, 'pb_ratio': -0.06, 'ps_ratio': -0.03,
                    'volatility_20d': -0.05, 'volatility_60d': -0.05,
                    'money_flow_20d': 0.05, 'volume_ratio': 0.05,
                    'return_20d': 0.06, 'return_60d': 0.06,
                    'roe': 0.08, 'roa': 0.04,
                    'gross_margin': 0.04, 'net_margin': 0.04,
                    'debt_ratio': -0.05,
                }
                base_weights.update(money_flow_weights)
            else:
                # 传统模式
                base_weights = {
                    'pe_ratio': -0.10, 'pb_ratio': -0.10, 'ps_ratio': -0.08,
                    'volatility_20d': -0.08, 'volatility_60d': -0.07,
                    'money_flow_20d': 0.06, 'volume_ratio': 0.06,
                    'return_20d': 0.08, 'return_60d': 0.07,
                }
                base_weights.update(money_flow_weights)
            
            self.factor_weights = base_weights

        print(f"\n📊 增强版StockRanker模型初始化")
        print(f"   基本面: {'✓' if use_fundamental else '✗'}")
        print(f"   资金流: {'✓' if use_money_flow else '✗'}")
        print(f"   Alpha因子: {'✓' if use_alpha_factors else '✗'}")
        if use_money_flow:
            print(f"   资金流风格: {money_flow_style}")
        if use_alpha_factors:
            print(f"   Alpha因子数: {len(self.alpha_calculator.get_alpha_factor_names()) if self.alpha_calculator else 0} 个")
        print(f"   总因子数: {len(self.factor_weights)} 个")

    def calculate_valuation_factors(self, df):
        """计算估值因子"""
        df = df.copy()
        df['pe_ratio'] = df['close'] / df.groupby('instrument')['close'].transform('mean')
        df['pb_ratio'] = df['close'] / (df.groupby('instrument')['close'].transform('mean') * 0.8)
        df['ps_ratio'] = df['close'] / (df.groupby('instrument')['close'].transform('mean') * 1.2)
        return df

    def calculate_volatility_factors(self, df):
        """计算波动率因子"""
        df = df.copy()
        df['volatility_20d'] = df.groupby('instrument')['close'].rolling(20).std().reset_index(0, drop=True)
        df['volatility_60d'] = df.groupby('instrument')['close'].rolling(60).std().reset_index(0, drop=True)
        return df

    def calculate_money_flow_factors(self, df):
        """计算资金流因子"""
        df = df.copy()
        df['money_flow_20d'] = (df['volume'] * df['close']).rolling(20).mean()
        df['volume_ma5'] = df.groupby('instrument')['volume'].rolling(5).mean().reset_index(0, drop=True)
        df['volume_ma20'] = df.groupby('instrument')['volume'].rolling(20).mean().reset_index(0, drop=True)
        df['volume_ratio'] = df['volume_ma5'] / (df['volume_ma20'] + 1e-6)
        return df

    def calculate_momentum_factors(self, df):
        """计算动量因子"""
        df = df.copy()
        df['return_20d'] = df.groupby('instrument')['close'].pct_change(20)
        df['return_60d'] = df.groupby('instrument')['close'].pct_change(60)
        return df

    def process_fundamental_factors(self, df):
        """处理基本面因子"""
        if not self.use_fundamental:
            return df
        fundamental_cols = ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']
        for col in fundamental_cols:
            if col in df.columns:
                median_val = df.groupby('instrument')[col].transform('median')
                df[col] = df[col].fillna(median_val)
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)
        return df

    def calculate_all_factors(self, price_data):
        """计算所有因子（包括Alpha因子）"""
        print("\n⚙️  计算增强版多因子...")
        df = price_data.copy()
        
        # 原有因子计算
        df = self.calculate_valuation_factors(df)
        df = self.calculate_volatility_factors(df)
        df = self.calculate_money_flow_factors(df)
        df = self.calculate_momentum_factors(df)
        if self.use_fundamental:
            df = self.process_fundamental_factors(df)
        if self.use_money_flow:
            print("\n💰 计算资金流因子...")
            df = self.money_flow_calculator.calculate_simplified_money_flow(df)
            self.money_flow_calculator.print_factor_summary(df)
        if self.use_alpha_factors and self.alpha_calculator:
            print("\n🔍 计算Alpha因子...")
            df = self.alpha_calculator.calculate_alpha_factors(df)
        
        return df

    def normalize_factors(self, df):
        """标准化因子"""
        for factor in self.factor_weights.keys():
            if factor in df.columns:
                df[f'{factor}_norm'] = df.groupby('date')[factor].rank(pct=True)
        return df

    def calculate_position_score(self, df):
        """计算综合评分"""
        print("\n📊 计算增强版综合评分...")
        df = df.copy()
        df['position'] = 0.0

        for factor, weight in self.factor_weights.items():
            if factor in df.columns:
                # 直接标准化并累加，不保留 _norm 列
                factor_rank = df.groupby('date')[factor].rank(pct=True).fillna(0.5)
                df['position'] += factor_rank * weight
                # 立即删除临时变量
                del factor_rank

        # 归一化到0-1
        min_score = df.groupby('date')['position'].transform('min')
        max_score = df.groupby('date')['position'].transform('max')
        df['position'] = (df['position'] - min_score) / (max_score - min_score + 1e-6)

        # 清理
        del min_score, max_score

        print("✓ 评分计算完成")
        return df


# 导出函数
def calculate_alpha_factors_enhanced(price_data, use_alpha_factors=True, alpha_weights=None):
    """增强版Alpha因子计算函数"""
    alpha_calculator = AlphaFactorCalculator(alpha_factors_enabled=use_alpha_factors, custom_alpha_weights=alpha_weights)
    return alpha_calculator.calculate_alpha_factors(price_data)


def get_enhanced_factor_model(use_fundamental=True, use_money_flow=True, use_alpha_factors=True, alpha_weights=None):
    """获取增强版因子模型"""
    return EnhancedStockRankerModel(
        use_fundamental=use_fundamental,
        use_money_flow=use_money_flow,
        use_alpha_factors=use_alpha_factors,
        alpha_weights=alpha_weights
    )


if __name__ == "__main__":
    print("Alpha增强版数据模块加载完成")
    print("可用功能:")
    print("- AlphaFactorCalculator: Alpha因子计算器")
    print("- EnhancedStockRankerModel: 增强版StockRanker模型")
    print("- calculate_alpha_factors_enhanced: 增强版Alpha因子计算函数")
    print("- get_enhanced_factor_model: 获取增强版因子模型")