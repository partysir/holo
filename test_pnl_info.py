#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试总盈亏信息传递
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def test_pnl_info():
    """测试总盈亏信息传递"""
    print("🧪 测试总盈亏信息传递")
    print("=" * 50)
    
    # 模拟交易历史数据（包含盈利和亏损）
    trade_data = [
        {
            '日期': '2023-01-01',
            '股票': '000001.SZ',
            '买卖操作': '买入',
            '数量': 1000,
            '成交价': 10.0,
            '成交金额': 10000.0,
            '平仓盈亏': 0.0,
            '交易费用': 5.0
        },
        {
            '日期': '2023-01-10',
            '股票': '000001.SZ',
            '买卖操作': '卖出',
            '数量': 1000,
            '成交价': 11.0,
            '成交金额': 11000.0,
            '平仓盈亏': 1000.0,  # 盈利
            '交易费用': 16.25
        },
        {
            '日期': '2023-01-15',
            '股票': '000002.SZ',
            '买卖操作': '买入',
            '数量': 100,
            '成交价': 5.0,
            '成交金额': 500.0,
            '平仓盈亏': 0.0,
            '交易费用': 5.0
        },
        {
            '日期': '2023-01-20',
            '股票': '000002.SZ',
            '买卖操作': '卖出',
            '数量': 100,
            '成交价': 4.5,
            '成交金额': 450.0,
            '平仓盈亏': -50.0,  # 亏损
            '交易费用': 5.0
        }
    ]
    
    df = pd.DataFrame(trade_data)
    
    # 按照用户要求的方式计算盈亏
    sell_trades = df[df['买卖操作'] == '卖出']
    profit_trades = sell_trades[sell_trades['平仓盈亏'] > 0]
    loss_trades = sell_trades[sell_trades['平仓盈亏'] < 0]
    
    # 总盈利（只算正的盈亏部分）
    total_profit = profit_trades['平仓盈亏'].sum()
    # 总亏损（只算负的盈亏部分）
    total_loss = loss_trades['平仓盈亏'].sum()
    # 净盈亏 = 总盈利 + 总亏损
    net_pnl = total_profit + total_loss
    # 交易费用总和
    total_fees = df['交易费用'].sum()
    # 扣除费用后的净盈亏
    net_pnl_after_fees = net_pnl - total_fees
    
    # 模拟context对象
    context = {
        'initial_capital': 1000000.0,
        'final_value': 1000968.75,
        'total_return': 0.00096875,
        'win_rate': 0.5,
        'pnl_info': {
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_pnl': net_pnl,
            'total_fees': total_fees,
            'net_pnl_after_fees': net_pnl_after_fees,
            'trade_count': len(df),
            'buy_count': len(df[df['买卖操作'] == '买入']),
            'sell_count': len(sell_trades),
            'profit_trades': len(profit_trades),
            'loss_trades': len(loss_trades)
        }
    }
    
    # 显示回测结果
    print("\n📊 回测结果:")
    print(f"  最终资产: ¥{context['final_value']:,.2f}")
    print(f"  总收益率: {context['total_return']:+.2%}")
    print(f"  胜率: {context['win_rate']:.2%}")
    
    # 显示总盈亏信息
    if 'pnl_info' in context:
        pnl_info = context['pnl_info']
        print(f"\n💰 交易绩效摘要:")
        print(f"  总交易次数: {pnl_info['trade_count']}")
        print(f"  买入次数: {pnl_info['buy_count']}")
        print(f"  卖出次数: {pnl_info['sell_count']}")
        print(f"  盈利次数: {pnl_info['profit_trades']}")
        print(f"  亏损次数: {pnl_info['loss_trades']}")
        print(f"  总盈利 (正盈亏部分): ¥{pnl_info['total_profit']:,.2f}")
        print(f"  总亏损 (负盈亏部分): ¥{pnl_info['total_loss']:,.2f}")
        print(f"  净盈亏 (总盈利 + 总亏损): ¥{pnl_info['net_pnl']:,.2f}")
        print(f"  交易费用总和: ¥{pnl_info['total_fees']:,.2f}")
        print(f"  扣除费用后净盈亏: ¥{pnl_info['net_pnl_after_fees']:,.2f}")
        if context['initial_capital'] > 0:
            net_return = pnl_info['net_pnl_after_fees'] / context['initial_capital']
            print(f"  净收益率: {net_return:+.2%}")
    
    return context

if __name__ == "__main__":
    test_pnl_info()