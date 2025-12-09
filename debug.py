"""
回测逻辑验证和修复工具

用于诊断和修复资产规模异常问题
"""

import pandas as pd
import numpy as np


def diagnose_backtest_data(context, daily_records_df, trade_records_df):
    """诊断回测数据的准确性"""
    
    print("\n" + "=" * 100)
    print("🔍 回测数据诊断报告")
    print("=" * 100)
    
    # 1. 基础信息检查
    print("\n【1. 基础信息】")
    print(f"  交易记录数: {len(trade_records_df)}")
    print(f"  交易天数: {len(daily_records_df)}")
    print(f"  初始资金: ¥{daily_records_df['portfolio_value'].iloc[0]:,.2f}")
    print(f"  最终资产: ¥{daily_records_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"  总收益率: {(daily_records_df['portfolio_value'].iloc[-1] / daily_records_df['portfolio_value'].iloc[0] - 1) * 100:.2f}%")
    
    # 2. 资产变化异常检查
    print("\n【2. 资产变化异常检查】")
    daily_returns = daily_records_df['portfolio_value'].pct_change()
    
    # 单日涨幅超过50%的异常
    extreme_gains = daily_returns[daily_returns > 0.5]
    if len(extreme_gains) > 0:
        print(f"  ⚠️  发现 {len(extreme_gains)} 天单日涨幅超过50%")
        print(f"  最大单日涨幅: {daily_returns.max() * 100:.2f}%")
        print(f"  异常日期示例:")
        for date, ret in extreme_gains.head(5).items():
            idx = daily_records_df[daily_records_df.index == date].index[0]
            print(f"    {daily_records_df.loc[idx, 'date']}: +{ret*100:.2f}% "
                  f"(¥{daily_records_df.loc[idx-1, 'portfolio_value']:,.0f} → "
                  f"¥{daily_records_df.loc[idx, 'portfolio_value']:,.0f})")
    
    # 3. 持仓股数检查
    print("\n【3. 持仓股数检查】")
    current_positions = context.get('positions', {})
    
    if current_positions:
        print(f"  当前持仓数: {len(current_positions)} 只")
        
        abnormal_positions = []
        for stock, info in current_positions.items():
            shares = info['shares']
            
            # 检查股数是否异常（超过1亿股）
            if shares > 100_000_000:
                abnormal_positions.append((stock, shares))
        
        if abnormal_positions:
            print(f"\n  ⚠️  发现 {len(abnormal_positions)} 只股票持仓异常（>1亿股）:")
            for stock, shares in abnormal_positions[:5]:
                print(f"    {stock}: {shares:,.0f} 股 ({shares/100_000_000:.2f}亿股)")
        else:
            print(f"  ✓ 持仓股数正常")
    
    # 4. 交易金额检查
    print("\n【4. 交易金额检查】")
    
    buy_trades = trade_records_df[trade_records_df['action'] == 'buy']
    sell_trades = trade_records_df[trade_records_df['action'] == 'sell']
    
    if len(buy_trades) > 0:
        buy_trades['amount'] = buy_trades['shares'] * buy_trades['price']
        
        # 检查单笔买入金额
        large_buys = buy_trades[buy_trades['amount'] > 10_000_000_000]  # 超过100亿
        
        if len(large_buys) > 0:
            print(f"  ⚠️  发现 {len(large_buys)} 笔买入金额超过100亿:")
            for _, trade in large_buys.head(5).iterrows():
                print(f"    {trade['date']} | {trade['stock']} | "
                      f"¥{trade['amount']:,.0f} ({trade['shares']:,.0f}股 @ ¥{trade['price']:.2f})")
    
    # 5. 资金使用率检查
    print("\n【5. 资金使用率检查】")
    
    # 计算每日资金使用率
    daily_records_df['position_ratio'] = (
        daily_records_df['portfolio_value'] - daily_records_df['cash']
    ) / daily_records_df['portfolio_value']
    
    avg_position_ratio = daily_records_df['position_ratio'].mean()
    max_position_ratio = daily_records_df['position_ratio'].max()
    
    print(f"  平均仓位: {avg_position_ratio * 100:.2f}%")
    print(f"  最高仓位: {max_position_ratio * 100:.2f}%")
    
    # 检查是否有超仓情况
    over_position = daily_records_df[daily_records_df['position_ratio'] > 1.0]
    if len(over_position) > 0:
        print(f"  ⚠️  发现 {len(over_position)} 天仓位超过100%（可能使用了杠杆或计算错误）")
    
    # 6. 给出诊断结论
    print("\n" + "=" * 100)
    print("【诊断结论】")
    print("=" * 100)
    
    issues = []
    
    if daily_records_df['portfolio_value'].iloc[-1] > 1_000_000_000:  # 超过10亿
        issues.append("❌ 资产规模异常：最终资产超过10亿元，不符合100万初始资金的合理范围")
    
    if len(extreme_gains) > 10:
        issues.append("❌ 收益率异常：存在多次单日涨幅超过50%的情况")
    
    if abnormal_positions:
        issues.append("❌ 持仓数量异常：存在持仓超过1亿股的股票")
    
    if len(over_position) > 0:
        issues.append("❌ 仓位计算错误：存在仓位超过100%的情况")
    
    if issues:
        print("\n⚠️  发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
        
        print("\n💡 可能的原因:")
        print("  1. 买入时使用全部资金而非分仓买入")
        print("  2. 股数计算时使用了错误的资金金额")
        print("  3. 卖出后资金累加错误导致资产膨胀")
        print("  4. 价格数据使用了复权价格但当作实际价格计算")
        print("  5. 没有考虑市场容量限制（无法买入如此大量股票）")
    else:
        print("\n✓ 回测数据基本正常")
    
    print()
    
    return issues


def verify_position_calculation(trade_records_df, initial_cash=1_000_000):
    """验证持仓计算逻辑是否正确"""
    
    print("\n" + "=" * 100)
    print("🔧 持仓计算逻辑验证")
    print("=" * 100)
    
    cash = initial_cash
    positions = {}
    
    trade_records_df = trade_records_df.sort_values('date')
    
    print(f"\n初始资金: ¥{cash:,.2f}")
    print("\n前10笔交易验证:")
    print("-" * 100)
    
    for idx, (_, trade) in enumerate(trade_records_df.head(10).iterrows()):
        stock = trade['stock']
        action = trade['action']
        shares = trade['shares']
        price = trade['price']
        amount = shares * price
        
        if action == 'buy':
            # 验证是否有足够现金
            if amount > cash:
                print(f"\n⚠️  第{idx+1}笔交易: {trade['date']} 买入 {stock}")
                print(f"   需要资金: ¥{amount:,.2f}")
                print(f"   可用现金: ¥{cash:,.2f}")
                print(f"   ❌ 资金不足！这笔交易在实际中无法执行")
            else:
                cash -= amount
                positions[stock] = {'shares': shares, 'cost': price}
                print(f"{idx+1}. {trade['date']} 买入 {stock}: {shares:,.0f}股 @ ¥{price:.2f} = ¥{amount:,.2f}")
                print(f"   剩余现金: ¥{cash:,.2f}")
        
        elif action == 'sell':
            if stock in positions:
                cash += amount
                profit = (price - positions[stock]['cost']) * shares
                del positions[stock]
                print(f"{idx+1}. {trade['date']} 卖出 {stock}: {shares:,.0f}股 @ ¥{price:.2f} = ¥{amount:,.2f}")
                print(f"   盈亏: ¥{profit:+,.2f}")
                print(f"   剩余现金: ¥{cash:,.2f}")
    
    # 计算当前资产
    position_value = sum(info['shares'] * info['cost'] for info in positions.values())
    total_value = cash + position_value
    
    print("\n" + "-" * 100)
    print(f"验证结果（前10笔交易后）:")
    print(f"  现金: ¥{cash:,.2f}")
    print(f"  持仓市值: ¥{position_value:,.2f}")
    print(f"  总资产: ¥{total_value:,.2f}")
    print(f"  收益率: {(total_value/initial_cash - 1)*100:+.2f}%")
    
    print()


def suggest_fixes():
    """给出修复建议"""
    
    print("\n" + "=" * 100)
    print("💡 修复建议")
    print("=" * 100)
    
    print("""
1. 【检查买入逻辑】
   应该使用分仓买入，而非全仓买入：
   
   ❌ 错误示例：
   shares = cash / price  # 用全部现金买入
   
   ✓ 正确示例：
   max_stocks = 10  # 最多持有10只股票
   position_size = cash / max_stocks  # 每只股票分配10%资金
   shares = position_size / price
   
2. 【添加资金检查】
   每次买入前检查是否有足够现金：
   
   amount = shares * price
   if amount > cash:
       shares = int(cash / price)  # 调整为可买入的最大股数
       amount = shares * price

3. 【添加市场容量限制】
   单只股票持仓不应超过其流通盘的一定比例（如5%）：
   
   max_shares = stock_float * 0.05  # 最多持有流通盘的5%
   shares = min(shares, max_shares)

4. 【添加交易成本】
   买卖都要扣除手续费和印花税：
   
   commission_rate = 0.0003  # 万三手续费
   stamp_tax = 0.001  # 千一印花税（仅卖出）
   
   buy_cost = amount * (1 + commission_rate)
   sell_amount = amount * (1 - commission_rate - stamp_tax)

5. 【验证价格数据】
   确认使用的是实际交易价格，而非复权价格：
   
   # 如果使用复权价格，需要转换回实际价格
   # 或者统一使用复权价格计算收益，但要标注清楚

6. 【添加调试日志】
   在关键步骤输出调试信息：
   
   print(f"买入前: 现金={cash}, 拟买入={amount}")
   print(f"买入后: 现金={cash}, 持仓={positions}")
""")


# 使用示例
def run_diagnosis(context, daily_records, trade_records):
    """运行完整诊断"""
    
    # 1. 诊断数据
    issues = diagnose_backtest_data(context, daily_records, trade_records)
    
    # 2. 验证计算逻辑
    verify_position_calculation(trade_records)
    
    # 3. 给出修复建议
    if issues:
        suggest_fixes()
    
    return issues