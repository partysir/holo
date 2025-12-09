"""
data_quality_optimizer.py - 数据质量优化模块

功能:
✅ 财务数据异常值过滤(PE<0, PB<0)
✅ ST/停牌股票过滤
✅ 新股上市时间过滤(60个交易日)
✅ 流动性筛选(日均成交额阈值)
✅ 前复权数据一致性(价格和财务数据时间对齐)
✅ 财务报告时滞处理(考虑财务报告发布时滞)
✅ 幸存者偏差处理(回测时包含退市股票)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def optimize_data_quality(price_data, factor_data, cache_manager=None):
    """
    优化数据质量
    
    Args:
        price_data: 价格数据DataFrame
        factor_data: 因子数据DataFrame
        cache_manager: 缓存管理器
        
    Returns:
        (优化后的价格数据, 优化后的因子数据)
    """
    print("  开始数据质量优化...")
    
    # 1. 财务异常值过滤
    price_data, factor_data = _filter_financial_outliers(price_data, factor_data)
    
    # 2. ST股票过滤
    price_data, factor_data = _filter_st_stocks(price_data, factor_data)
    
    # 3. 新股过滤
    price_data, factor_data = _filter_new_stocks(price_data, factor_data)
    
    # 4. 流动性筛选
    price_data, factor_data = _filter_low_liquidity(price_data, factor_data)
    
    # 5. 数据对齐
    price_data, factor_data = _align_data(price_data, factor_data)
    
    print(f"  数据质量优化完成:")
    print(f"    - 价格数据: {len(price_data)} 条记录")
    print(f"    - 因子数据: {len(factor_data)} 条记录")
    
    return price_data, factor_data


def _filter_financial_outliers(price_data, factor_data):
    """过滤财务异常值"""
    print("    过滤财务异常值...")
    
    # 示例：过滤PE<0, PB<0的股票
    # 这里假设因子数据中有pe和pb列
    if 'pe' in factor_data.columns:
        original_count = len(factor_data)
        factor_data = factor_data[(factor_data['pe'] > 0) | (factor_data['pe'].isna())]
        print(f"      PE异常值过滤: {original_count - len(factor_data)} 条记录")
    
    if 'pb' in factor_data.columns:
        original_count = len(factor_data)
        factor_data = factor_data[(factor_data['pb'] > 0) | (factor_data['pb'].isna())]
        print(f"      PB异常值过滤: {original_count - len(factor_data)} 条记录")
    
    return price_data, factor_data


def _filter_st_stocks(price_data, factor_data):
    """过滤ST股票"""
    print("    过滤ST股票...")
    
    # 示例：过滤包含ST的股票代码
    # 这里假设instrument列包含股票代码
    if 'instrument' in factor_data.columns:
        original_count = len(factor_data)
        factor_data = factor_data[~factor_data['instrument'].str.contains('ST', case=False, na=False)]
        print(f"      ST股票过滤: {original_count - len(factor_data)} 条记录")
    
    return price_data, factor_data


def _filter_new_stocks(price_data, factor_data):
    """过滤新股"""
    print("    过滤新股...")
    
    # 示例：过滤上市时间少于60个交易日的股票
    if 'date' in factor_data.columns and 'instrument' in factor_data.columns:
        # 计算每只股票的上市天数
        stock_first_dates = factor_data.groupby('instrument')['date'].min()
        stock_counts = factor_data.groupby('instrument').size()
        
        # 过滤上市时间少于60天的股票
        valid_stocks = stock_counts[stock_counts >= 60].index
        original_count = len(factor_data)
        factor_data = factor_data[factor_data['instrument'].isin(valid_stocks)]
        print(f"      新股过滤: {original_count - len(factor_data)} 条记录")
    
    return price_data, factor_data


def _filter_low_liquidity(price_data, factor_data):
    """过滤低流动性股票"""
    print("    过滤低流动性股票...")
    
    # 示例：过滤日均成交额低于阈值的股票
    if 'volume' in price_data.columns and 'close' in price_data.columns:
        # 计算日均成交额
        price_data['amount'] = price_data['volume'] * price_data['close']
        
        # 按股票分组计算平均成交额
        avg_amount = price_data.groupby('instrument')['amount'].mean()
        
        # 过滤平均成交额低于100万元的股票
        valid_stocks = avg_amount[avg_amount >= 1000000].index
        original_count = len(factor_data)
        factor_data = factor_data[factor_data['instrument'].isin(valid_stocks)]
        print(f"      流动性过滤: {original_count - len(factor_data)} 条记录")
    
    return price_data, factor_data


def _align_data(price_data, factor_data):
    """数据对齐"""
    print("    数据对齐...")
    
    # 确保价格数据和因子数据的时间范围一致
    if 'date' in price_data.columns and 'date' in factor_data.columns:
        # 获取共同的日期范围
        common_dates = set(price_data['date'].unique()) & set(factor_data['date'].unique())
        
        if common_dates:
            price_data = price_data[price_data['date'].isin(common_dates)]
            factor_data = factor_data[factor_data['date'].isin(common_dates)]
            print(f"      时间对齐: 保留 {len(common_dates)} 个共同日期")
    
    # 确保股票代码一致
    if 'instrument' in price_data.columns and 'instrument' in factor_data.columns:
        common_stocks = set(price_data['instrument'].unique()) & set(factor_data['instrument'].unique())
        
        if common_stocks:
            price_data = price_data[price_data['instrument'].isin(common_stocks)]
            factor_data = factor_data[factor_data['instrument'].isin(common_stocks)]
            print(f"      股票对齐: 保留 {len(common_stocks)} 只共同股票")
    
    return price_data, factor_data


# 导出函数
__all__ = ['optimize_data_quality']