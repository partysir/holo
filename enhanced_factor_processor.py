"""
enhanced_factor_processor.py - 增强型因子处理核心
功能：整合基础因子与Alpha增强因子，执行标准化、中性化和清洗
"""

import pandas as pd
import numpy as np
from data_module_alpha_enhanced import AlphaFactorCalculator

class EnhancedFactorProcessor:
    def __init__(self, neutralize_industry=True, neutralize_market=True, 
                 use_alpha_factors=True, alpha_weights=None):
        self.neutralize_industry = neutralize_industry
        self.neutralize_market = neutralize_market
        self.use_alpha_factors = use_alpha_factors
        self.alpha_weights = alpha_weights or {}
        
        # 初始化Alpha因子计算器
        if self.use_alpha_factors:
            self.alpha_calculator = AlphaFactorCalculator(
                alpha_factors_enabled=True, 
                custom_alpha_weights=self.alpha_weights
            )
        else:
            self.alpha_calculator = None

    def process_factors(self, factor_data, price_data):
        """
        处理因子的主入口
        Args:
            factor_data: 包含基础因子数据的DataFrame
            price_data: 包含价格数据的DataFrame
        """
        print("\n⚙️ 开始因子增强处理流程...")
        
        df = factor_data.copy()
        
        # 1. 生成增强Alpha因子 (调用 data_module_alpha_enhanced)
        if self.use_alpha_factors and self.alpha_calculator:
            try:
                print("🔍 应用Alpha因子增强...")
                df = self.alpha_calculator.calculate_alpha_factors(df)
            except Exception as e:
                print(f"⚠️ Alpha因子生成部分失败: {e}")
                # 如果生成失败，至少保证不报错，继续使用原有数据
        
        # 2. 自动识别因子列 (数值型且非元数据)
        exclude_cols = [
            'date', 'instrument', 'industry', 'open', 'high', 'low', 'close', 
            'volume', 'amount', 'is_st', 'list_days', 'position', 'ml_score'
        ]
        
        current_numeric_cols = [
            c for c in df.columns 
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        
        print(f"   检测到 {len(current_numeric_cols)} 个潜在因子列")
        
        # 3. 缺失值处理 (使用中位数填充)
        df[current_numeric_cols] = df[current_numeric_cols].fillna(df[current_numeric_cols].median())
        
        # 4. 去极值 (3倍标准差)
        df = self._clip_extremes(df, current_numeric_cols)
        
        # 5. 标准化 (Z-Score)
        df = self._standardize(df, current_numeric_cols)
        
        # 6. 行业中性化 (可选)
        if self.neutralize_industry and 'industry' in df.columns:
            df = self._neutralize_industry(df, current_numeric_cols)
        
        # 7. 市场中性化 (可选)
        if self.neutralize_market:
            df = self._neutralize_market(df, current_numeric_cols)
            
        return df

    def _clip_extremes(self, df, cols):
        """MAD去极值或3-sigma"""
        print("   执行去极值处理...")
        for col in cols:
            median = df[col].median()
            mad = (df[col] - median).abs().median()
            upper = median + 3 * 1.4826 * mad
            lower = median - 3 * 1.4826 * mad
            df[col] = df[col].clip(lower, upper)
        return df

    def _standardize(self, df, cols):
        """Z-Score标准化"""
        print("   执行标准化处理...")
        # 按日期分组标准化更好，防止时间序列偏差
        def zscore_group(group):
            return (group - group.mean()) / (group.std() + 1e-6)
        
        # 这里的性能优化：直接对所有因子列按日期分组apply比较慢
        # 实盘时通常只有一天数据，直接整体做
        if df['date'].nunique() == 1:
            df[cols] = (df[cols] - df[cols].mean()) / (df[cols].std() + 1e-6)
        else:
            # 历史数据处理
            grouped = df.groupby('date')[cols].transform(zscore_group)
            df[cols] = grouped
            
        return df

    def _neutralize_industry(self, df, cols):
        """
        行业中性化：减去行业均值
        """
        print("   执行行业中性化...")
        if 'industry' not in df.columns:
            print("   ⚠️ 未找到行业列，跳过行业中性化")
            return df
            
        # 计算行业均值
        industry_means = df.groupby(['date', 'industry'])[cols].transform('mean')
        df[cols] = df[cols] - industry_means
        return df

    def _neutralize_market(self, df, cols):
        """
        市场中性化：减去市场均值
        """
        print("   执行市场中性化...")
        # 计算市场均值（按日期）
        market_means = df.groupby('date')[cols].transform('mean')
        df[cols] = df[cols] - market_means
        return df

# 便捷函数
def create_enhanced_factor_processor(neutralize_industry=True, neutralize_market=True, 
                                   use_alpha_factors=True, alpha_weights=None):
    """
    创建增强版因子处理器
    
    Args:
        neutralize_industry: 是否行业中性化
        neutralize_market: 是否市场中性化
        use_alpha_factors: 是否使用Alpha因子
        alpha_weights: Alpha因子权重
        
    Returns:
        EnhancedFactorProcessor实例
    """
    return EnhancedFactorProcessor(
        neutralize_industry=neutralize_industry,
        neutralize_market=neutralize_market,
        use_alpha_factors=use_alpha_factors,
        alpha_weights=alpha_weights
    )


def process_with_alpha_factors(factor_data, price_data, neutralize_industry=True, neutralize_market=True):
    """
    使用Alpha因子处理数据的便捷函数
    
    Args:
        factor_data: 因子数据
        price_data: 价格数据
        neutralize_industry: 是否行业中性化
        neutralize_market: 是否市场中性化
        
    Returns:
        处理后的因子数据
    """
    processor = create_enhanced_factor_processor(
        neutralize_industry=neutralize_industry,
        neutralize_market=neutralize_market,
        use_alpha_factors=True
    )
    
    return processor.process_factors(factor_data, price_data)


# 导出类
__all__ = ['EnhancedFactorProcessor', 'create_enhanced_factor_processor', 'process_with_alpha_factors']