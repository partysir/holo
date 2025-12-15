"""
holdings_monitor.py - 修复版 v2.0

修复内容:
✅ 修复盈亏重复计算问题
✅ 修复交易费用未正确扣除
✅ 添加盈亏合理性检查
✅ 改进收益率计算逻辑
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def generate_daily_holdings_report(context, factor_data, price_data,
                                   output_dir='./reports',
                                   print_to_console=True,
                                   save_to_csv=True):
    """生成每日持仓监控报告（修复版）"""
    print("\n" + "=" * 100)
    print("📊 生成每日持仓监控报告 v2.0")
    print("="  * 100)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trade_records = context.get('trade_records', pd.DataFrame())
    daily_records = context.get('daily_records', pd.DataFrame())
    
    if trade_records.empty or daily_records.empty:
        print("⚠️  没有交易记录")
        return None, None

    # 获取初始资金
    initial_capital = context.get('initial_capital', 10_000_000)
    print(f"📍 初始资金: ¥{initial_capital:,.0f}")

    # 重建每日持仓状态
    daily_holdings, trade_history = rebuild_daily_holdings_fixed(
        trade_records, daily_records, factor_data, price_data, initial_capital
    )

    # 验证盈亏合理性
    validate_pnl_reasonableness(trade_history, initial_capital)

    if print_to_console and not daily_holdings.empty:
        print_daily_holdings_to_console(daily_holdings)

    pnl_info = None
    if save_to_csv:
        pnl_info = save_holdings_to_csv_fixed(
            daily_holdings, trade_history, output_dir, initial_capital
        )

    print("\n✓ 持仓报告生成完成")
    return daily_holdings, pnl_info


def rebuild_daily_holdings_fixed(trade_records, daily_records, factor_data, 
                                 price_data, initial_capital):
    """重建每日持仓状态（修复版 - 避免重复统计）"""
    
    # 统一日期格式
    trade_records = trade_records.copy()
    trade_records['date'] = trade_records['date'].astype(str)
    daily_records['date'] = daily_records['date'].astype(str)
    factor_data['date'] = factor_data['date'].astype(str)
    price_data['date'] = price_data['date'].astype(str)

    # 识别评分列
    score_column = identify_score_column(factor_data)
    print(f"✓ 使用评分列: {score_column}")

    # 费率设置
    TRANSACTION_FEE_RATE = 0.00025
    STAMP_DUTY_RATE = 0.001
    MIN_TRANSACTION_FEE = 5.0

    all_holdings = []
    trade_history = []  # 只用于记录交易，不重复记录盈亏
    current_positions = {}

    trades_df = trade_records.sort_values('date')
    dates = sorted(daily_records['date'].unique())

    print(f"  处理 {len(dates)} 个交易日...")

    for idx, date in enumerate(dates):
        if (idx + 1) % 50 == 0:
            print(f"    进度: {idx + 1}/{len(dates)}")
        
        date_str = str(date)
        daily_trades = trades_df[trades_df['date'] == date_str]

        # ===== 处理卖出交易 =====
        for _, trade in daily_trades.iterrows():
            stock = trade['stock']
            action = trade['action']
            shares = trade['shares']
            price = trade['price']

            if action == 'sell' and stock in current_positions:
                entry_info = current_positions[stock]
                entry_price = entry_info['cost']
                entry_date = entry_info['entry_date']
                
                # 计算盈亏（不含费用）
                holding_days = (pd.to_datetime(date_str) - pd.to_datetime(entry_date)).days
                gross_pnl = (price - entry_price) * shares
                
                # 计算交易费用
                buy_amount = entry_price * shares
                sell_amount = price * shares
                buy_fee = max(buy_amount * TRANSACTION_FEE_RATE, MIN_TRANSACTION_FEE)
                sell_fee = max(sell_amount * (TRANSACTION_FEE_RATE + STAMP_DUTY_RATE), 
                              MIN_TRANSACTION_FEE)
                total_fee = buy_fee + sell_fee
                
                # 净盈亏 = 毛盈亏 - 交易费用
                net_pnl = gross_pnl - total_fee
                net_pnl_rate = net_pnl / (entry_price * shares) if entry_price > 0 else 0
                
                # ✅ 关键修复：只在 trade_history 中记录一次
                trade_history.append({
                    'date': date_str,
                    'stock': stock,
                    'action': '卖出',
                    'shares': shares,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'entry_date': entry_date,
                    'holding_days': holding_days,
                    'gross_pnl': gross_pnl,
                    'fees': total_fee,
                    'net_pnl': net_pnl,
                    'net_pnl_rate': net_pnl_rate
                })
                
                # 删除持仓
                del current_positions[stock]

        # ===== 处理买入交易 =====
        for _, trade in daily_trades.iterrows():
            stock = trade['stock']
            action = trade['action']
            shares = trade['shares']
            price = trade['price']

            if action == 'buy':
                current_positions[stock] = {
                    'shares': shares,
                    'cost': price,
                    'entry_date': date_str
                }
                
                # 计算买入费用
                buy_amount = price * shares
                buy_fee = max(buy_amount * TRANSACTION_FEE_RATE, MIN_TRANSACTION_FEE)
                
                trade_history.append({
                    'date': date_str,
                    'stock': stock,
                    'action': '买入',
                    'shares': shares,
                    'entry_price': price,
                    'exit_price': None,
                    'entry_date': date_str,
                    'holding_days': 0,
                    'gross_pnl': 0,
                    'fees': buy_fee,
                    'net_pnl': -buy_fee,  # 买入时费用是负收益
                    'net_pnl_rate': -buy_fee / buy_amount if buy_amount > 0 else 0
                })

        # ===== 记录当日持仓状态（用于监控，不用于盈亏统计）=====
        for stock, info in current_positions.items():
            price_row = price_data[
                (price_data['instrument'] == stock) &
                (price_data['date'] == date_str)
            ]

            if len(price_row) == 0:
                continue

            current_price = price_row['close'].iloc[0]
            score = get_stock_score(factor_data, stock, date_str, score_column)

            shares = info['shares']
            cost = info['cost']
            current_value = shares * current_price
            unrealized_pnl = (current_price - cost) * shares
            unrealized_pnl_rate = (current_price - cost) / cost if cost > 0 else 0

            holding_days = (pd.to_datetime(date_str) - pd.to_datetime(info['entry_date'])).days

            # 判断是否是当日买入
            daily_buy = daily_trades[
                (daily_trades['stock'] == stock) & 
                (daily_trades['action'] == 'buy')
            ]
            is_new_buy = len(daily_buy) > 0

            all_holdings.append({
                'date': date_str,
                'stock': stock,
                'action': 'buy' if is_new_buy else 'hold',
                'shares': shares,
                'cost': cost,
                'current_price': current_price,
                'current_value': current_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_rate': unrealized_pnl_rate,
                'score': score,
                'holding_days': holding_days
            })

    return pd.DataFrame(all_holdings), pd.DataFrame(trade_history)


def validate_pnl_reasonableness(trade_history_df, initial_capital):
    """验证盈亏合理性"""
    print("\n" + "="*80)
    print("🔍 盈亏合理性检查")
    print("="*80)
    
    if trade_history_df.empty:
        return
    
    sells = trade_history_df[trade_history_df['action'] == '卖出']
    
    if len(sells) == 0:
        print("  ℹ️  暂无卖出交易")
        return
    
    # 检查单笔盈亏
    max_profit = sells['net_pnl'].max()
    max_loss = sells['net_pnl'].min()
    
    print(f"  单笔最大盈利: ¥{max_profit:,.2f}")
    print(f"  单笔最大亏损: ¥{max_loss:,.2f}")
    
    # 合理性阈值：单笔盈亏不应超过初始资金的50%
    threshold = initial_capital * 0.5
    
    abnormal_profit = sells[sells['net_pnl'] > threshold]
    abnormal_loss = sells[sells['net_pnl'] < -threshold]
    
    if len(abnormal_profit) > 0:
        print(f"\n  ⚠️  发现 {len(abnormal_profit)} 笔异常盈利（>50%初始资金）:")
        for _, row in abnormal_profit.head(3).iterrows():
            print(f"     {row['date']} | {row['stock']} | "
                  f"¥{row['net_pnl']:,.0f} ({row['net_pnl_rate']:+.2%})")
    
    if len(abnormal_loss) > 0:
        print(f"\n  ⚠️  发现 {len(abnormal_loss)} 笔异常亏损（>50%初始资金）:")
        for _, row in abnormal_loss.head(3).iterrows():
            print(f"     {row['date']} | {row['stock']} | "
                  f"¥{row['net_pnl']:,.0f} ({row['net_pnl_rate']:+.2%})")
    
    if len(abnormal_profit) == 0 and len(abnormal_loss) == 0:
        print("  ✓ 所有交易盈亏在合理范围内")


def save_holdings_to_csv_fixed(holdings_df, trade_history_df, output_dir, initial_capital):
    """保存持仓数据到CSV（修复版 - 避免重复统计）"""
    
    # 1. 保存持仓监控数据
    if not holdings_df.empty:
        holdings_export = holdings_df.rename(columns={
            'date': '日期',
            'stock': '股票',
            'shares': '持仓数量',
            'cost': '持仓均价',
            'current_price': '当前价格',
            'current_value': '持仓市值',
            'unrealized_pnl': '浮动盈亏',
            'unrealized_pnl_rate': '浮动收益率',
            'score': '评分',
            'holding_days': '持有天数'
        })
        
        holdings_path = os.path.join(output_dir, 'daily_holdings_monitor.csv')
        holdings_export.to_csv(holdings_path, index=False, encoding='utf-8-sig')
        print(f"\n💾 持仓监控数据已保存: {holdings_path}")

    # 2. 保存交易历史（不重复）- 只使用 trade_history_df
    if not trade_history_df.empty:
        trade_export = trade_history_df.copy()
        
        # 确保列名正确
        if 'date' in trade_export.columns:
            trade_export = trade_export.rename(columns={
                'date': '日期',
                'stock': '股票',
                'action': '操作',
                'shares': '数量',
                'entry_price': '买入价',
                'exit_price': '卖出价',
                'entry_date': '买入日期',
                'holding_days': '持有天数',
                'gross_pnl': '毛盈亏',
                'fees': '交易费用',
                'net_pnl': '净盈亏',
                'net_pnl_rate': '收益率'
            })
        
        trade_path = os.path.join(output_dir, 'trade_history_fixed.csv')
        trade_export.to_csv(trade_path, index=False, encoding='utf-8-sig')
        print(f"💾 交易历史已保存: {trade_path}")
        
        # ✅ 修复：正确计算总盈亏（只从交易历史统计一次）
        print("\n" + "─" * 80)
        print("📊 交易统计（修复版 - 避免重复统计）")
        print("─" * 80)
        
        sells = trade_export[trade_export['操作'] == '卖出']
        buys = trade_export[trade_export['操作'] == '买入']
        
        if len(sells) > 0:
            # 计算卖出交易的盈亏
            profit_trades = sells[sells['净盈亏'] > 0]
            loss_trades = sells[sells['净盈亏'] < 0]
            
            total_profit = profit_trades['净盈亏'].sum()
            total_loss = loss_trades['净盈亏'].sum()
            net_pnl_from_sells = total_profit + total_loss
            
            # 计算买入交易的费用（如果已经包含在净盈亏中，就不用再算）
            # 由于我们在 rebuild_daily_holdings_fixed 中，买入时的 net_pnl 已经是 -fee
            # 所以这里不需要再单独计算买入费用
            
            # 总净盈亏 = 所有卖出的净盈亏之和
            total_net_pnl = net_pnl_from_sells
            
            # 计算正确的收益率
            correct_return_rate = total_net_pnl / initial_capital
            
            print(f"  总交易次数: {len(trade_export)}")
            print(f"  买入次数: {len(buys)}")
            print(f"  卖出次数: {len(sells)}")
            print(f"\n  盈利次数: {len(profit_trades)} ({len(profit_trades)/len(sells)*100:.1f}%)")
            print(f"  亏损次数: {len(loss_trades)} ({len(loss_trades)/len(sells)*100:.1f}%)")
            print(f"\n  卖出总盈利: ¥{total_profit:,.2f}")
            print(f"  卖出总亏损: ¥{total_loss:,.2f}")
            print(f"  净盈亏: ¥{total_net_pnl:,.2f}")
            print(f"\n  ✅ 正确收益率: {correct_return_rate:+.2%}")
            print(f"     (基于初始资金 ¥{initial_capital:,.0f})")
            
            return {
                'total_trades': len(trade_export),
                'buy_count': len(buys),
                'sell_count': len(sells),
                'profit_trades': len(profit_trades),
                'loss_trades': len(loss_trades),
                'total_profit': total_profit,
                'total_loss': total_loss,
                'total_net_pnl': total_net_pnl,
                'correct_return_rate': correct_return_rate,
                'initial_capital': initial_capital
            }
    
    return None


def identify_score_column(factor_data):
    """识别评分列"""
    possible_names = ['position', 'score', 'factor_score', 'rank']
    
    for col in possible_names:
        if col in factor_data.columns:
            return col
    
    return None


def get_stock_score(factor_data, stock, date_str, score_column):
    """获取股票评分"""
    if score_column is None:
        return 0.5
    
    score_row = factor_data[
        (factor_data['instrument'] == stock) &
        (factor_data['date'] == date_str)
    ]
    
    if len(score_row) > 0:
        score = score_row[score_column].iloc[0]
        if pd.isna(score) or not np.isfinite(score):
            return 0.5
        return float(score)
    
    return 0.5


def print_daily_holdings_to_console(holdings_df, max_days=3):
    """简化的持仓打印"""
    if holdings_df.empty:
        return
    
    dates = sorted(holdings_df['date'].unique())
    recent_dates = dates[-max_days:]
    
    print("\n" + "="*100)
    print(f"📈 最近{len(recent_dates)}日持仓概览")
    print("="*100)
    
    for date in recent_dates:
        day_holdings = holdings_df[holdings_df['date'] == date]
        
        total_value = day_holdings['current_value'].sum()
        total_pnl = day_holdings['unrealized_pnl'].sum()
        
        print(f"\n{date} | 持仓{len(day_holdings)}只 | "
              f"市值¥{total_value:,.0f} | 浮盈¥{total_pnl:+,.0f}")