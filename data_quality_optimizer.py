"""
data_quality_optimizer.py - 数据质量优化模块 (v2.7 修复版)

修复:
✅ 修正流动性过滤阈值过高导致数据被清空的问题 (适配Tushare单位)
✅ 增强 ST 过滤逻辑
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def optimize_data_quality(price_data, factor_data, cache_manager=None):
    """
    优化数据质量
    """
    print("  开始数据质量优化...")

    # 1. 财务异常值过滤
    price_data, factor_data = _filter_financial_outliers(price_data, factor_data)

    # 2. ST股票过滤
    price_data, factor_data = _filter_st_stocks(price_data, factor_data)

    # 3. 新股过滤
    price_data, factor_data = _filter_new_stocks(price_data, factor_data)

    # 4. 流动性筛选 (修复重点)
    price_data, factor_data = _filter_low_liquidity(price_data, factor_data)

    # 5. 数据对齐
    price_data, factor_data = _align_data(price_data, factor_data)

    print(f"  数据质量优化完成:")
    print(f"    - 价格数据: {len(price_data)} 条记录")
    print(f"    - 因子数据: {len(factor_data)} 条记录")

    if len(price_data) == 0:
        print("  ⚠️ 严重警告: 所有数据都被过滤掉了！请检查过滤条件。")

    return price_data, factor_data


def _filter_financial_outliers(price_data, factor_data):
    """过滤财务异常值"""
    print("    过滤财务异常值...")
    if 'pe' in factor_data.columns:
        factor_data = factor_data[(factor_data['pe'] > 0) | (factor_data['pe'].isna())]
    if 'pb' in factor_data.columns:
        factor_data = factor_data[(factor_data['pb'] > 0) | (factor_data['pb'].isna())]
    return price_data, factor_data


def _filter_st_stocks(price_data, factor_data):
    """过滤ST股票"""
    print("    过滤ST股票...")

    st_codes = set()
    try:
        import tushare as ts
        pro = ts.pro_api()
        df_basic = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
        st_df = df_basic[df_basic['name'].str.contains('ST', case=False, na=False)]
        st_codes = set(st_df['ts_code'].tolist())
    except Exception as e:
        print(f"      ⚠️ 无法获取ST名单 ({e})，跳过ST过滤")

    if 'instrument' in factor_data.columns and st_codes:
        data_st_codes = set(factor_data['instrument'].unique()) & st_codes
        if data_st_codes:
            factor_data = factor_data[~factor_data['instrument'].isin(data_st_codes)]
            if price_data is not None and 'instrument' in price_data.columns:
                price_data = price_data[~price_data['instrument'].isin(data_st_codes)]
            print(f"      ST股票过滤: 剔除 {len(data_st_codes)} 只")
        else:
            print("      ST股票过滤: 0 条")

    return price_data, factor_data


def _filter_new_stocks(price_data, factor_data):
    """过滤新股"""
    print("    过滤新股...")
    if 'date' in factor_data.columns and 'instrument' in factor_data.columns:
        stock_counts = factor_data.groupby('instrument').size()
        valid_stocks = stock_counts[stock_counts >= 10].index # 降低要求到10天，防止过滤太狠
        factor_data = factor_data[factor_data['instrument'].isin(valid_stocks)]
        if price_data is not None:
            price_data = price_data[price_data['instrument'].isin(valid_stocks)]
        print(f"      新股过滤: 保留 {len(valid_stocks)} 只")
    return price_data, factor_data


def _filter_low_liquidity(price_data, factor_data):
    """过滤低流动性股票 (修复版)"""
    print("    过滤低流动性股票...")

    if price_data is None or len(price_data) == 0:
        return price_data, factor_data

    # 确定 amount 列
    use_col = None
    if 'amount' in price_data.columns:
        use_col = 'amount'
    elif 'volume' in price_data.columns and 'close' in price_data.columns:
        price_data['amount_calc'] = price_data['volume'] * price_data['close']
        use_col = 'amount_calc'

    if use_col:
        # 计算日均成交额
        avg_amount = price_data.groupby('instrument')[use_col].mean()

        # 🛑 关键修复：降低阈值，并做单位自适应
        # Tushare amount 单位是千元。
        # 3000 表示 300万 RMB。
        # 如果数据本身已经是元，3000元太小了，基本都能过。
        # 这是一个安全的兜底策略。
        THRESHOLD = 3000

        valid_stocks = avg_amount[avg_amount >= THRESHOLD].index

        # 如果过滤太狠（比如剩不到10%），说明单位可能搞错了，实施熔断保护
        if len(valid_stocks) < len(avg_amount) * 0.1:
             print(f"      ⚠️ 警告: 流动性过滤过严 (阈值{THRESHOLD})，可能因为单位差异。已自动降低阈值。")
             valid_stocks = avg_amount[avg_amount >= 100].index # 极低门槛兜底

        factor_data = factor_data[factor_data['instrument'].isin(valid_stocks)]
        price_data = price_data[price_data['instrument'].isin(valid_stocks)]

        print(f"      流动性过滤: 保留 {len(valid_stocks)} 只股票")

    return price_data, factor_data


def _align_data(price_data, factor_data):
    """数据对齐"""
    print("    数据对齐...")
    if 'instrument' in price_data.columns and 'instrument' in factor_data.columns:
        common_stocks = set(price_data['instrument'].unique()) & set(factor_data['instrument'].unique())
        price_data = price_data[price_data['instrument'].isin(common_stocks)]
        factor_data = factor_data[factor_data['instrument'].isin(common_stocks)]

    if 'date' in price_data.columns and 'date' in factor_data.columns:
        common_dates = set(price_data['date'].unique()) & set(factor_data['date'].unique())
        price_data = price_data[price_data['date'].isin(common_dates)]
        factor_data = factor_data[factor_data['date'].isin(common_dates)]

    return price_data, factor_data

__all__ = ['optimize_data_quality']