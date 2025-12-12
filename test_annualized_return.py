#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试年化收益率计算
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_annualized_return():
    """测试年化收益率计算"""
    print("🧪 测试年化收益率计算")
    print("=" * 50)
    
    # 模拟每日资产价值数据（100个交易日）
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # 模拟资产价值（初始100万，逐渐增长）
    initial_capital = 1000000.0
    values = [initial_capital]
    
    # 模拟每日收益率（平均0.1%）
    for i in range(1, 100):
        daily_return = np.random.normal(0.001, 0.02)  # 平均0.1%，标准差2%
        new_value = values[-1] * (1 + daily_return)
        values.append(new_value)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'portfolio_value': values
    })
    
    # 计算总收益率
    final_value = df['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    # 计算年化收益率
    trading_days = len(df)
    years = trading_days / 252
    if years > 0 and total_return > -1:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0
    
    print(f"初始资金: ¥{initial_capital:,.2f}")
    print(f"最终资产: ¥{final_value:,.2f}")
    print(f"交易天数: {trading_days} 天 ({years:.2f}年)")
    print(f"总收益率: {total_return:+.2%}")
    print(f"年化收益率: {annualized_return:+.2%}")
    
    # 验证计算是否正确
    print(f"\n🔍 验证计算:")
    print(f"  (1 + 总收益率)^(1/年数) - 1 = (1 + {total_return:.4f})^(1/{years:.2f}) - 1 = {annualized_return:.4f}")
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'trading_days': trading_days
    }

if __name__ == "__main__":
    test_annualized_return()