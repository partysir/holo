"""
enhanced_factor_processor.py - 增强因子处理器（修复版）

修复内容：
1. ✅ 修复因子列过滤逻辑，保留处理后的因子
2. ✅ 改进数值列检测方法
3. ✅ 优化缺失值处理
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class EnhancedFactorProcessor:
    """增强因子处理器"""
    
    def __init__(self, neutralize_industry=True, neutralize_market=True):
        self.neutralize_industry = neutralize_industry
        self.neutralize_market = neutralize_market
        self.scaler = StandardScaler()

    def process_factors(self, factor_data, factor_columns):
        """处理因子（修复版）"""
        print(f"    开始因子处理...")

        # 过滤有效因子
        valid_factors = []
        for col in factor_columns:
            if col in factor_data.columns:
                if pd.api.types.is_numeric_dtype(factor_data[col]):
                    valid_factors.append(col)

        print(f"      有效因子列: {len(valid_factors)} 个")

        if len(valid_factors) == 0:
            return factor_data

        # ===== 行业中性化 =====
        # 只有当数据中已经存在industry列时才进行行业中性化
        if self.neutralize_industry and 'industry' in factor_data.columns:
            # 检查industry列是否有效
            unique_industries = factor_data['industry'].nunique()
            if unique_industries > 1:  # 至少有2个不同行业
                print(f"      ✓ 检测到有效行业数据: {unique_industries}个行业")
                print(f"      执行行业中性化...")
                factor_data = self._neutralize_by_industry(factor_data, valid_factors)
            else:
                print(f"      ⚠️  行业数据不足（只有{unique_industries}个行业），跳过行业中性化")
        elif self.neutralize_industry:
            print(f"      ⚠️  已启用行业中性化，但缺少行业数据")

        # ===== 市场中性化 =====
        if self.neutralize_market:
            print(f"      执行市场中性化...")
            factor_data = self._neutralize_by_market(factor_data, valid_factors)

        print(f"    因子处理完成: {len(valid_factors)} 个因子")

        return factor_data
    
    def _neutralize_by_market(self, data, factor_columns):
        """
        市场中性化
        
        Args:
            data: 数据DataFrame
            factor_columns: 因子列名列表
            
        Returns:
            中性化后的数据
        """
        neutralized_data = data.copy()
        
        for col in factor_columns:
            if col not in neutralized_data.columns:
                continue
            
            try:
                # 按日期去均值（市场中性化）
                neutralized_data[col] = neutralized_data.groupby('date')[col].transform(
                    lambda x: x - x.mean() if len(x) > 0 else x
                )
            except Exception as e:
                print(f"      ⚠️  市场中性化失败 ({col}): {e}")
                continue
        
        print("      市场中性化完成")
        return neutralized_data
    
    def _winsorize_mad(self, series, n=3):
        """
        使用MAD方法去极值
        
        Args:
            series: 数据序列
            n: MAD倍数
            
        Returns:
            去极值后的序列
        """
        # 移除缺失值
        valid_series = series.dropna()
        if len(valid_series) == 0:
            return series
        
        median = valid_series.median()
        mad = np.median(np.abs(valid_series - median))
        
        if mad == 0:
            return series
        
        threshold = n * mad * 1.4826  # 转换为标准差单位
        
        lower_bound = median - threshold
        upper_bound = median + threshold
        
        return series.clip(lower_bound, upper_bound)
    
    def _standardize_zscore(self, series):
        """
        Z-Score标准化
        
        Args:
            series: 数据序列
            
        Returns:
            标准化后的序列
        """
        # 移除缺失值
        valid_series = series.dropna()
        if len(valid_series) == 0:
            return series
        
        mean = valid_series.mean()
        std = valid_series.std()
        
        if std == 0 or pd.isna(std):
            return series - mean
        else:
            return (series - mean) / std
    
    def _fill_missing_values(self, data, column):
        """
        填充缺失值
        
        Args:
            data: 数据DataFrame
            column: 列名
            
        Returns:
            填充后的序列
        """
        series = data[column].copy()
        
        # 如果没有缺失值，直接返回
        if not series.isna().any():
            return series
        
        # 如果没有行业列，使用全局中位数填充
        if 'industry' not in data.columns:
            median_value = series.median()
            if pd.isna(median_value):
                median_value = 0
            return series.fillna(median_value)
        
        # 按行业中位数填充
        filled_series = series.copy()
        for industry in data['industry'].unique():
            if pd.isna(industry):
                continue
            mask = data['industry'] == industry
            industry_median = series[mask].median()
            if pd.isna(industry_median):
                industry_median = series.median()
            if pd.isna(industry_median):
                industry_median = 0
            filled_series[mask & series.isna()] = industry_median
        
        # 如果还有缺失值，用全局中位数填充
        if filled_series.isna().any():
            global_median = series.median()
            if pd.isna(global_median):
                global_median = 0
            filled_series = filled_series.fillna(global_median)
        
        return filled_series
    
    def _neutralize_by_industry(self, data, factor_columns):
        """
        行业中性化
        
        Args:
            data: 数据DataFrame
            factor_columns: 因子列名列表
            
        Returns:
            中性化后的数据
        """
        neutralized_data = data.copy()
        
        for col in factor_columns:
            if col not in neutralized_data.columns:
                continue
            
            try:
                # 按行业去均值
                neutralized_data[col] = neutralized_data.groupby('industry')[col].transform(
                    lambda x: x - x.mean() if len(x) > 0 else x
                )
            except Exception as e:
                print(f"      ⚠️  行业中性化失败 ({col}): {e}")
                continue
        
        print("      行业中性化完成")
        return neutralized_data
    
    def calculate_factor_metrics(self, factor_data, factor_columns, forward_period=5):
        """
        计算因子有效性指标
        
        Args:
            factor_data: 因子数据
            factor_columns: 因子列名列表
            forward_period: 前瞻期
            
        Returns:
            因子有效性指标字典
        """
        metrics = {}
        
        # 如果没有收盘价数据，无法计算IC
        if 'close' not in factor_data.columns:
            print("      ⚠️  缺少价格数据，无法计算因子有效性指标")
            return metrics
        
        # 计算未来收益
        factor_data = factor_data.sort_values(['instrument', 'date'])
        factor_data['future_return'] = factor_data.groupby('instrument')['close'].pct_change(forward_period).shift(-forward_period)
        
        for col in factor_columns:
            if col not in factor_data.columns:
                continue
            
            try:
                # 计算IC值
                valid_data = factor_data[[col, 'future_return']].dropna()
                if len(valid_data) > 10:  # 至少需要10个样本
                    ic = valid_data[col].corr(valid_data['future_return'])
                    if not pd.isna(ic):
                        metrics[col] = {
                            'ic': ic,
                            'abs_ic': abs(ic)
                        }
            except Exception as e:
                print(f"      ⚠️  计算 {col} 的IC失败: {e}")
                continue
        
        if metrics:
            print(f"      ✓ 因子有效性计算完成: {len(metrics)} 个因子")
            # 显示前3个因子的IC
            sorted_metrics = sorted(metrics.items(), key=lambda x: x[1]['abs_ic'], reverse=True)
            for i, (factor, metric) in enumerate(sorted_metrics[:3]):
                print(f"        {i+1}. {factor}: IC={metric['ic']:.4f}")
        
        return metrics


# 导出类
__all__ = ['EnhancedFactorProcessor']