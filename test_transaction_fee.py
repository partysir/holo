#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试交易费用计算
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_transaction_fee_calculation():
    """测试交易费用计算"""
    print("🧪 测试交易费用计算")
    print("=" * 50)
    
    # 模拟交易数据（包含大额和小额交易）
    trade_data = [
        {
            '日期': '2023-01-01',
            '股票': '000001.SZ',
            '买卖操作': '买入',
            '数量': 1000,
            '成交价': 10.0,
            '成交金额': 10000.0,
            '平仓盈亏': 0.0,
            '交易费用': 0.0  # 待计算
        },
        {
            '日期': '2023-01-10',
            '股票': '000001.SZ',
            '买卖操作': '卖出',
            '数量': 1000,
            '成交价': 11.0,
            '成交金额': 11000.0,
            '平仓盈亏': 1000.0,  # (11-10)*1000
            '交易费用': 0.0  # 待计算
        },
        {
            '日期': '2023-01-15',
            '股票': '000002.SZ',
            '买卖操作': '买入',
            '数量': 100,
            '成交价': 5.0,
            '成交金额': 500.0,
            '平仓盈亏': 0.0,
            '交易费用': 0.0  # 待计算（应为5元最低收费）
        },
        {
            '日期': '2023-01-20',
            '股票': '000002.SZ',
            '买卖操作': '卖出',
            '数量': 100,
            '成交价': 5.5,
            '成交金额': 550.0,
            '平仓盈亏': 50.0,  # (5.5-5)*100
            '交易费用': 0.0  # 待计算（应为5元最低收费）
        }
    ]
    
    df = pd.DataFrame(trade_data)
    
    # 国信证券费率设置
    TRANSACTION_FEE_RATE = 0.00025  # 万2.5
    STAMP_DUTY_RATE = 0.001         # 千分之一印花税
    MIN_TRANSACTION_FEE = 5.0       # 最低收费5元
    
    # 存储每笔交易的买入信息，用于计算卖出时的总费用
    buy_info = {}
    
    # 计算交易费用
    for i, row in df.iterrows():
        if row['买卖操作'] == '买入':
            # 买入时只需计算手续费，最低5元
            fee = max(row['成交金额'] * TRANSACTION_FEE_RATE, MIN_TRANSACTION_FEE)
            df.at[i, '交易费用'] = float(fee)
            # 保存买入信息
            buy_info[row['股票']] = row['成交金额']
        elif row['买卖操作'] == '卖出':
            # 卖出时需要计算手续费和印花税，最低5元
            # 获取对应的买入金额
            buy_amount = buy_info.get(row['股票'], row['成交金额'])
            
            # 买入时的手续费
            buy_fee = max(buy_amount * TRANSACTION_FEE_RATE, MIN_TRANSACTION_FEE)
            # 卖出时的手续费+印花税
            sell_fee = max(row['成交金额'] * (TRANSACTION_FEE_RATE + STAMP_DUTY_RATE), MIN_TRANSACTION_FEE)
            total_fee = buy_fee + sell_fee
            df.at[i, '交易费用'] = float(total_fee)
    
    print("交易记录:")
    print(df.to_string(index=False))
    
    # 计算总盈亏
    total_pnl = df['平仓盈亏'].sum()
    total_fees = df['交易费用'].sum()
    net_pnl = total_pnl - total_fees
    
    print(f"\n📊 统计摘要:")
    print(f"  平仓盈亏总和: ¥{total_pnl:,.2f}")
    print(f"  交易费用总和: ¥{total_fees:,.2f}")
    print(f"  净盈亏: ¥{net_pnl:,.2f}")
    
    # 验证最低收费规则
    print(f"\n🔍 验证最低收费规则:")
    for _, row in df.iterrows():
        if row['买卖操作'] == '买入':
            calculated_fee = max(row['成交金额'] * TRANSACTION_FEE_RATE, MIN_TRANSACTION_FEE)
            print(f"  {row['股票']} {row['买卖操作']} - 成交金额: ¥{row['成交金额']:,.2f}, 计算费用: ¥{calculated_fee:.2f}, 实际费用: ¥{row['交易费用']:.2f}")
        elif row['买卖操作'] == '卖出':
            buy_amount = buy_info.get(row['股票'], row['成交金额'])
            buy_fee = max(buy_amount * TRANSACTION_FEE_RATE, MIN_TRANSACTION_FEE)
            sell_fee = max(row['成交金额'] * (TRANSACTION_FEE_RATE + STAMP_DUTY_RATE), MIN_TRANSACTION_FEE)
            total_calculated_fee = buy_fee + sell_fee
            print(f"  {row['股票']} {row['买卖操作']} - 成交金额: ¥{row['成交金额']:,.2f}, 计算费用: ¥{total_calculated_fee:.2f}, 实际费用: ¥{row['交易费用']:.2f}")
    
    return df

if __name__ == "__main__":
    test_transaction_fee_calculation()