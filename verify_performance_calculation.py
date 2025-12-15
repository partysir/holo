#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证绩效计算是否正确
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def verify_performance_calculation():
    """验证绩效计算逻辑"""
    print("🔍 验证绩效计算逻辑")
    print("=" * 60)
    
    # 模拟真实情况：初始资金100万，经过一段时间增长到几千万
    initial_capital = 1000000  # 100万
    final_value = 872858522.19  # 8.7亿（来自用户提供的错误数据）
    
    # 模拟每日记录（简化）
    dates = pd.date_range('2023-01-01', '2025-12-12', freq='D')
    dates = [d.strftime('%Y-%m-%d') for d in dates]
    
    # 模拟组合价值增长过程
    # 假设是指数增长
    growth_rate = (final_value / initial_capital) ** (1 / len(dates)) - 1
    portfolio_values = [initial_capital * (1 + growth_rate) ** i for i in range(len(dates))]
    
    # 创建模拟数据
    daily_records = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values
    })
    
    # 计算每日收益率
    daily_records['return'] = daily_records['portfolio_value'].pct_change()
    daily_records['return'].iloc[0] = (daily_records['portfolio_value'].iloc[0] - initial_capital) / initial_capital
    
    print(f"初始资金: ¥{initial_capital:,.2f}")
    print(f"最终资产: ¥{final_value:,.2f}")
    print(f"交易天数: {len(dates)} 天")
    print()
    
    # 错误的计算方式（使用第一天的组合价值作为基准）
    first_day_value = daily_records['portfolio_value'].iloc[0]
    wrong_total_return = (final_value - first_day_value) / first_day_value
    
    # 正确的计算方式（使用初始资金作为基准）
    correct_total_return = (final_value - initial_capital) / initial_capital
    
    print(f"❌ 错误计算方式: ({final_value:,.2f} - {first_day_value:,.2f}) / {first_day_value:,.2f} = {wrong_total_return:+.2%}")
    print(f"✅ 正确计算方式: ({final_value:,.2f} - {initial_capital:,.2f}) / {initial_capital:,.2f} = {correct_total_return:+.2%}")
    
    # 年化收益率计算
    years = len(dates) / 365
    annualized_return = 0
    if years > 0 and correct_total_return > -1:
        annualized_return = (1 + correct_total_return) ** (1 / years) - 1
        print(f"年化收益率: {annualized_return:+.2%}")
    
    # 分析问题根源
    print("\n" + "=" * 60)
    print("问题分析:")
    if abs(wrong_total_return - correct_total_return) > 0.01:
        print("❌ 存在收益率计算错误!")
        print(f"   差异: {abs(wrong_total_return - correct_total_return):.2%}")
        if wrong_total_return > correct_total_return:
            print("   原因: 使用了错误的基准值进行计算")
        else:
            print("   原因: 基准值设置不正确")
    else:
        print("✅ 收益率计算正确")
    
    print("\n修复建议:")
    print("1. 确保使用初始资金作为收益率计算的基准")
    print("2. 检查daily_records中是否正确记录了初始资金")
    print("3. 验证context中是否包含了正确的capital_base字段")
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'wrong_return': wrong_total_return,
        'correct_return': correct_total_return,
        'annualized_return': annualized_return
    }

if __name__ == "__main__":
    result = verify_performance_calculation()
    print(f"\n📊 验证结果:")
    print(f"   初始资金: ¥{result['initial_capital']:,.2f}")
    print(f"   最终资产: ¥{result['final_value']:,.2f}")
    print(f"   错误收益率: {result['wrong_return']:+.2%}")
    print(f"   正确收益率: {result['correct_return']:+.2%}")
    print(f"   年化收益率: {result['annualized_return']:+.2%}")