#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试持仓显示功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_holdings_display():
    """测试持仓显示功能"""
    print("🧪 测试持仓显示功能")
    print("=" * 50)
    
    # 模拟持仓数据
    holdings_data = [
        {
            'stock': '000001.SZ',
            'entry_date': '2023-01-01',
            'holding_days': 100,
            'shares': 1000,
            'cost': 10.0,
            'current_price': 11.0,
            'current_value': 11000.0,
            'pnl': 1000.0,
            'pnl_rate': 0.1,
            'score': 0.85,
            'position_ratio': 0.4
        },
        {
            'stock': '000002.SZ',
            'entry_date': '2023-01-15',
            'holding_days': 85,
            'shares': 500,
            'cost': 20.0,
            'current_price': 19.0,
            'current_value': 9500.0,
            'pnl': -500.0,
            'pnl_rate': -0.05,
            'score': 0.75,
            'position_ratio': 0.35
        },
        {
            'stock': '000003.SZ',
            'entry_date': '2023-02-01',
            'holding_days': 70,
            'shares': 800,
            'cost': 15.0,
            'current_price': 15.5,
            'current_value': 12400.0,
            'pnl': 400.0,
            'pnl_rate': 0.0333,
            'score': 0.90,
            'position_ratio': 0.25
        }
    ]
    
    df = pd.DataFrame(holdings_data)
    
    # 显示持仓信息
    print(f"\n📅 今日日期: 2023-04-10")
    print("=" * 130)

    # 账户概览
    total_value = df['current_value'].sum()
    total_pnl = df['pnl'].sum()
    total_cost = total_value - total_pnl
    total_pnl_rate = total_pnl / total_cost if total_cost > 0 else 0

    print(f"\n📊 账户概览:")
    print(f"  总资产: ¥{total_value + 10000:,.0f}")  # 假设有10000现金
    print(f"  持仓市值: ¥{total_value:,.0f}")
    print(f"  持仓成本: ¥{total_cost:,.0f}")
    print(f"  浮动盈亏: ¥{total_pnl:+,.0f} ({total_pnl_rate:+.2%})")
    print(f"  持仓数量: {len(df)} 只")
    print(f"  平均评分: {df['score'].mean():.4f}")

    # 盈亏统计
    profit_count = (df['pnl'] > 0).sum()
    loss_count = (df['pnl'] < 0).sum()
    flat_count = (df['pnl'] == 0).sum()

    print(f"\n📈 盈亏分布:")
    print(f"  盈利: {profit_count} 只 ({profit_count / len(df) * 100:.1f}%)")
    print(f"  亏损: {loss_count} 只 ({loss_count / len(df) * 100:.1f}%)")
    print(f"  持平: {flat_count} 只")

    # 详细持仓列表
    print(f"\n{'=' * 130}")
    header = f"{'排名':4s} {'股票代码':12s} {'买入日期':12s} {'持仓股数':>8s} "
    header += f"{'持仓占比':>8s} {'成本价':>8s} {'现价':>8s} {'浮动盈亏':>10s} {'收益率':>8s} "
    header += f"{'评分':>8s}"
    print(header)
    print(f"{'=' * 130}")

    for idx, row in df.iterrows():
        rank = idx + 1

        if row['pnl'] > 0:
            pnl_color = "+"
        elif row['pnl'] < 0:
            pnl_color = ""
        else:
            pnl_color = " "

        line = f"{rank:3d}  {row['stock']:12s} {row['entry_date']:12s} {row['shares']:8.0f} "
        line += f"{row['position_ratio']:7.2%} {row['cost']:8.2f} {row['current_price']:8.2f} "
        line += f"{pnl_color}¥{row['pnl']:9,.0f} {pnl_color}{row['pnl_rate']:7.2%} "
        line += f"{row['score']:7.4f}"
        print(line)

    print(f"{'=' * 130}\n")

if __name__ == "__main__":
    test_holdings_display()