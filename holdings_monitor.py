"""
holdings_monitor.py - 每日持仓监控报告（完全修复版）

修复内容：
✅ 修复卖出记录缺失问题 - 在卖出当天也记录持仓状态
✅ 修复评分显示问题 - 增强评分列匹配和调试输出
✅ 添加策略表现分析 - 识别长期持有亏损的问题
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def generate_daily_holdings_report(context, factor_data, price_data,
                                   output_dir='./reports',
                                   print_to_console=True,
                                   save_to_csv=True):
    """生成每日持仓监控报告"""
    print("\n" + "=" * 100)
    print("📊 生成每日持仓监控报告")
    print("=" * 100)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 提取交易记录
    trade_records = context.get('trade_records', pd.DataFrame())
    daily_records = context.get('daily_records', pd.DataFrame())
    
    if trade_records.empty or daily_records.empty:
        print("⚠️  没有交易记录")
        return None

    # 🔍 调试：检查 factor_data 的列名
    print(f"\n🔍 factor_data 列名: {factor_data.columns.tolist()}")
    print(f"🔍 factor_data 样本数据:")
    print(factor_data.head(2))

    # 重建每日持仓状态
    daily_holdings, trade_history = rebuild_daily_holdings(
        trade_records, daily_records, factor_data, price_data
    )

    # 终端输出
    if print_to_console and not daily_holdings.empty:
        print_daily_holdings_to_console(daily_holdings)
        
        # 新增：策略表现分析
        analyze_strategy_performance(daily_holdings, trade_history)

    # 保存CSV
    if save_to_csv:
        save_holdings_to_csv(daily_holdings, trade_history, output_dir)

    print("\n✓ 持仓报告生成完成")
    return daily_holdings


def rebuild_daily_holdings(trade_records, daily_records, factor_data, price_data):
    """重建每日持仓状态和完整交易历史（修复卖出记录缺失）"""
    all_holdings = []
    trade_history = []
    current_positions = {}
    
    # 确保日期格式一致
    trade_records = trade_records.copy()
    daily_records = daily_records.copy()
    factor_data = factor_data.copy()
    price_data = price_data.copy()
    
    # 统一日期格式为字符串
    trade_records['date'] = trade_records['date'].astype(str)
    daily_records['date'] = daily_records['date'].astype(str)
    factor_data['date'] = factor_data['date'].astype(str)
    price_data['date'] = price_data['date'].astype(str)

    # 🔍 自动识别评分列
    score_column = identify_score_column(factor_data)
    print(f"\n✓ 识别到评分列: {score_column}")

    trades_df = trade_records.sort_values('date')
    dates = sorted(daily_records['date'].unique())

    print(f"  处理 {len(dates)} 个交易日...")

    for idx, date in enumerate(dates):
        if (idx + 1) % 50 == 0:
            print(f"    进度: {idx + 1}/{len(dates)}")
        
        date_str = str(date)

        # 处理当日交易
        daily_trades = trades_df[trades_df['date'] == date_str]

        # ✅ 修复1：先记录卖出前的持仓状态，再执行卖出
        for _, trade in daily_trades.iterrows():
            stock = trade['stock']
            action = trade['action']
            shares = trade['shares']
            price = trade['price']
            reason = trade.get('reason', 'unknown')

            if action == 'sell' and stock in current_positions:
                # 📌 关键修复：卖出前先记录持仓状态到 all_holdings
                entry_info = current_positions[stock]
                entry_date = entry_info['entry_date']
                entry_price = entry_info['cost']
                
                # 计算持有天数和盈亏
                holding_days = (pd.to_datetime(date_str) - pd.to_datetime(entry_date)).days
                pnl = (price - entry_price) * shares
                pnl_rate = (price - entry_price) / entry_price
                
                # 获取卖出时的评分
                score = get_stock_score(factor_data, stock, date_str, score_column)
                
                # 记录卖出当天的持仓状态（action='sell'）
                all_holdings.append({
                    'date': date_str,
                    'stock': stock,
                    'action': 'sell',
                    'shares': shares,
                    'price': price,
                    'cost': entry_price,
                    'entry_date': entry_date,
                    'current_value': shares * price,
                    'pnl': pnl,
                    'pnl_rate': pnl_rate,
                    'score': score,
                    'holding_days': holding_days,
                    'reason': reason
                })
                
                # 记录到交易历史
                trade_history.append({
                    'date': date_str,
                    'stock': stock,
                    'action': '卖出',
                    'shares': shares,
                    'price': price,
                    'amount': shares * price,
                    'reason': reason,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'holding_days': holding_days,
                    'pnl': pnl,
                    'pnl_rate': pnl_rate
                })
                
                # 然后删除持仓
                del current_positions[stock]

        # 处理买入
        for _, trade in daily_trades.iterrows():
            stock = trade['stock']
            action = trade['action']
            shares = trade['shares']
            price = trade['price']
            reason = trade.get('reason', 'unknown')

            if action == 'buy':
                current_positions[stock] = {
                    'shares': shares,
                    'cost': price,
                    'entry_date': date_str,
                    'entry_reason': reason
                }
                
                # 记录买入交易
                trade_history.append({
                    'date': date_str,
                    'stock': stock,
                    'action': '买入',
                    'shares': shares,
                    'price': price,
                    'amount': shares * price,
                    'reason': reason,
                    'holding_days': 0,
                    'pnl': 0,
                    'pnl_rate': 0
                })

        # 记录当日持仓状态（hold）
        for stock, info in current_positions.items():
            # 获取当前价格
            price_row = price_data[
                (price_data['instrument'] == stock) &
                (price_data['date'] == date_str)
            ]

            if len(price_row) == 0:
                continue

            current_price = price_row['close'].iloc[0]

            # ✅ 修复2：使用改进的评分获取函数
            score = get_stock_score(factor_data, stock, date_str, score_column)

            # 计算盈亏
            shares = info['shares']
            cost = info['cost']
            current_value = shares * current_price
            cost_value = shares * cost
            pnl = current_value - cost_value
            pnl_rate = (current_price - cost) / cost if cost > 0 else 0

            # 判断当日操作
            daily_trade = daily_trades[daily_trades['stock'] == stock]
            if len(daily_trade) > 0 and daily_trade['action'].iloc[0] == 'buy':
                action = 'buy'
                reason = daily_trade['reason'].iloc[0] if 'reason' in daily_trade.columns else 'unknown'
            else:
                action = 'hold'
                reason = 'holding'

            # 持有天数
            holding_days = (pd.to_datetime(date_str) - pd.to_datetime(info['entry_date'])).days

            all_holdings.append({
                'date': date_str,
                'stock': stock,
                'action': action,
                'shares': shares,
                'price': current_price,
                'cost': cost,
                'entry_date': info['entry_date'],
                'current_value': current_value,
                'pnl': pnl,
                'pnl_rate': pnl_rate,
                'score': score,
                'holding_days': holding_days,
                'reason': reason
            })

    return pd.DataFrame(all_holdings), pd.DataFrame(trade_history)


def identify_score_column(factor_data):
    """✅ 自动识别评分列"""
    possible_names = ['position', 'score', 'factor_score', 'rank', 'signal']
    
    for col in possible_names:
        if col in factor_data.columns:
            return col
    
    # 如果都找不到，尝试找数值列
    numeric_cols = factor_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['date', 'instrument']:
            print(f"⚠️  未找到标准评分列，使用 '{col}' 作为评分")
            return col
    
    print("⚠️  警告：未找到任何评分列，将使用默认值0.5")
    return None


def get_stock_score(factor_data, stock, date_str, score_column):
    """✅ 改进的评分获取函数"""
    if score_column is None:
        return 0.5
    
    score_row = factor_data[
        (factor_data['instrument'] == stock) &
        (factor_data['date'] == date_str)
    ]
    
    if len(score_row) > 0:
        score = score_row[score_column].iloc[0]
        # 处理可能的异常值
        if pd.isna(score) or not np.isfinite(score):
            return 0.5
        return float(score)
    
    return 0.5


def analyze_strategy_performance(holdings_df, trade_history_df):
    """✅ 新增：策略表现分析"""
    print("\n" + "=" * 100)
    print("🔍 策略表现分析")
    print("=" * 100)
    
    if holdings_df.empty:
        return
    
    # 1. 长期持有分析
    long_holdings = holdings_df[holdings_df['holding_days'] > 20].copy()
    if not long_holdings.empty:
        loss_long = long_holdings[long_holdings['pnl'] < 0]
        
        print("\n📌 长期持有（>20天）分析:")
        print(f"   总数: {len(long_holdings)} 只")
        print(f"   亏损: {len(loss_long)} 只 ({len(loss_long)/len(long_holdings)*100:.1f}%)")
        
        if not loss_long.empty:
            print(f"\n   ⚠️  长期持有亏损股票:")
            for _, row in loss_long.nlargest(5, 'holding_days').iterrows():
                print(f"      {row['stock']:12s} | 持有{row['holding_days']:3d}天 | "
                      f"亏损{row['pnl_rate']:+.2%} | 评分{row['score']:.4f}")
    
    # 2. 快速亏损分析
    if not trade_history_df.empty:
        sell_trades = trade_history_df[trade_history_df['action'] == '卖出']
        if not sell_trades.empty:
            quick_loss = sell_trades[
                (sell_trades['holding_days'] < 10) & 
                (sell_trades['pnl_rate'] < -0.05)
            ]
            
            if not quick_loss.empty:
                print(f"\n📌 快速亏损（<10天且亏损>5%）分析:")
                print(f"   发生次数: {len(quick_loss)} 次")
                print(f"   平均亏损: {quick_loss['pnl_rate'].mean():.2%}")
                
                print(f"\n   ⚠️  快速亏损案例:")
                for _, row in quick_loss.nlargest(5, 'pnl_rate', keep='first').iterrows():
                    print(f"      {row['date']} | {row['stock']:12s} | "
                          f"持有{row['holding_days']}天 | 亏损{row['pnl_rate']:.2%}")
    
    # 3. 评分有效性分析
    if 'score' in holdings_df.columns:
        print(f"\n📌 评分有效性分析:")
        
        # 评分分布
        high_score = holdings_df[holdings_df['score'] > 0.7]
        low_score = holdings_df[holdings_df['score'] < 0.3]
        
        if not high_score.empty:
            high_score_profit_rate = (high_score['pnl'] > 0).sum() / len(high_score)
            print(f"   高评分(>0.7): {len(high_score)}条 | 盈利率 {high_score_profit_rate:.1%}")
        
        if not low_score.empty:
            low_score_profit_rate = (low_score['pnl'] > 0).sum() / len(low_score)
            print(f"   低评分(<0.3): {len(low_score)}条 | 盈利率 {low_score_profit_rate:.1%}")
        
        # 如果评分都是0.5，给出警告
        unique_scores = holdings_df['score'].nunique()
        if unique_scores == 1 and holdings_df['score'].iloc[0] == 0.5:
            print(f"\n   ⚠️  警告：所有评分都是0.5，可能存在以下问题:")
            print(f"      1. factor_data 中评分列名不是 'position'")
            print(f"      2. factor_data 与 price_data 的股票代码或日期不匹配")
            print(f"      3. factor_data 缺失或为空")
    
    print()


def print_daily_holdings_to_console(holdings_df, max_days_to_print=5):
    """美化输出到终端"""
    if len(holdings_df) == 0:
        print("\n⚠️  没有持仓数据")
        return

    dates = sorted(holdings_df['date'].unique())
    recent_dates = dates[-max_days_to_print:]

    print("\n" + "=" * 100)
    print(f"📈 最近 {len(recent_dates)} 个交易日持仓详情")
    print("=" * 100)

    for date in recent_dates:
        date_holdings = holdings_df[holdings_df['date'] == date].copy()

        if len(date_holdings) == 0:
            continue

        date_holdings = date_holdings.sort_values('score', ascending=False)

        buys = date_holdings[date_holdings['action'] == 'buy']
        sells = date_holdings[date_holdings['action'] == 'sell']
        holds = date_holdings[date_holdings['action'] == 'hold']

        total_value = date_holdings['current_value'].sum()
        total_pnl = date_holdings['pnl'].sum()
        total_cost = total_value - total_pnl
        total_pnl_rate = total_pnl / total_cost if total_cost > 0 else 0

        print(f"\n{'─' * 100}")
        print(f"📅 {date} | 持仓 {len(date_holdings)}只 | "
              f"买入 {len(buys)}只 | 卖出 {len(sells)}只 | "
              f"总市值 ¥{total_value:,.0f} | "
              f"浮动盈亏 ¥{total_pnl:+,.0f} ({total_pnl_rate:+.2%})")
        print(f"{'─' * 100}")

        if len(buys) > 0:
            print(f"\n  🔵 买入 ({len(buys)}只):")
            for _, row in buys.iterrows():
                reason_text = f"[{row['reason']}]" if row['reason'] != 'unknown' else ""
                print(f"     {row['stock']:12s} | "
                      f"价格: ¥{row['price']:7.2f} | "
                      f"数量: {row['shares']:6,.0f}股 | "
                      f"金额: ¥{row['current_value']:9,.0f} | "
                      f"评分: {row['score']:.4f} {reason_text}")

        if len(sells) > 0:
            print(f"\n  🔴 卖出 ({len(sells)}只):")
            for _, row in sells.iterrows():
                reason_icon = "💰" if row['pnl'] > 0 else "📉"
                reason_text = f"[{row['reason']}]" if row['reason'] != 'unknown' else ""
                print(f"     {row['stock']:12s} | "
                      f"买入: {row['entry_date']} | "
                      f"卖出: ¥{row['price']:7.2f} | "
                      f"成本: ¥{row['cost']:7.2f} | "
                      f"盈亏: {reason_icon}¥{row['pnl']:+9,.0f} ({row['pnl_rate']:+.2%}) | "
                      f"持有: {row['holding_days']}天 {reason_text}")

        if len(holds) > 6:
            print(f"\n  ⚪ 持仓中 ({len(holds)}只，显示评分最高3只和最低3只):")
            top_3 = holds.head(3)
            for _, row in top_3.iterrows():
                pnl_icon = "📈" if row['pnl'] > 0 else "📉"
                print(f"     {row['stock']:12s} | "
                      f"买入: {row['entry_date']} | "
                      f"现价: ¥{row['price']:7.2f} | "
                      f"成本: ¥{row['cost']:7.2f} | "
                      f"浮盈: {pnl_icon}¥{row['pnl']:+9,.0f} ({row['pnl_rate']:+.2%}) | "
                      f"评分: {row['score']:.4f} ⭐ | "
                      f"持有: {row['holding_days']}天")
            
            if len(holds) > 6:
                print(f"     ... 省略 {len(holds) - 6} 只中间评分股票 ...")
            
            bottom_3 = holds.tail(3)
            for _, row in bottom_3.iterrows():
                pnl_icon = "📈" if row['pnl'] > 0 else "📉"
                print(f"     {row['stock']:12s} | "
                      f"买入: {row['entry_date']} | "
                      f"现价: ¥{row['price']:7.2f} | "
                      f"成本: ¥{row['cost']:7.2f} | "
                      f"浮盈: {pnl_icon}¥{row['pnl']:+9,.0f} ({row['pnl_rate']:+.2%}) | "
                      f"评分: {row['score']:.4f} ⚠️  | "
                      f"持有: {row['holding_days']}天")
        elif len(holds) > 0:
            print(f"\n  ⚪ 持仓中 ({len(holds)}只):")
            for _, row in holds.iterrows():
                pnl_icon = "📈" if row['pnl'] > 0 else "📉"
                print(f"     {row['stock']:12s} | "
                      f"买入: {row['entry_date']} | "
                      f"现价: ¥{row['price']:7.2f} | "
                      f"成本: ¥{row['cost']:7.2f} | "
                      f"浮盈: {pnl_icon}¥{row['pnl']:+9,.0f} ({row['pnl_rate']:+.2%}) | "
                      f"评分: {row['score']:.4f} | "
                      f"持有: {row['holding_days']}天")

    print("\n" + "=" * 100)


def save_holdings_to_csv(holdings_df, trade_history_df, output_dir):
    """保存持仓数据到CSV"""
    if len(holdings_df) == 0:
        print("\n⚠️  没有数据可保存")
        return

    # 1. 保存完整持仓历史（包含卖出记录）
    full_path = os.path.join(output_dir, 'daily_holdings_detail.csv')
    holdings_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 完整持仓历史已保存: {full_path}")
    
    # 验证卖出记录
    sell_count = (holdings_df['action'] == 'sell').sum()
    print(f"   ✓ 包含 {sell_count} 条卖出记录")

    # 2. 保存交易历史
    if not trade_history_df.empty:
        trade_path = os.path.join(output_dir, 'trade_history_detail.csv')
        trade_history_df.to_csv(trade_path, index=False, encoding='utf-8-sig')
        print(f"💾 交易历史明细已保存: {trade_path}")
        
        # 打印交易统计
        print("\n" + "─" * 80)
        print("📊 交易统计摘要")
        print("─" * 80)
        
        buy_trades = trade_history_df[trade_history_df['action'] == '买入']
        sell_trades = trade_history_df[trade_history_df['action'] == '卖出']
        
        print(f"  总交易次数: {len(trade_history_df)}")
        print(f"  买入次数: {len(buy_trades)}")
        print(f"  卖出次数: {len(sell_trades)}")
        
        if len(sell_trades) > 0:
            profit_trades = sell_trades[sell_trades['pnl'] > 0]
            loss_trades = sell_trades[sell_trades['pnl'] < 0]
            
            print(f"\n  盈利次数: {len(profit_trades)} ({len(profit_trades)/len(sell_trades)*100:.1f}%)")
            print(f"  亏损次数: {len(loss_trades)} ({len(loss_trades)/len(sell_trades)*100:.1f}%)")
            print(f"  总盈亏: ¥{sell_trades['pnl'].sum():,.0f}")
            print(f"  平均盈亏: ¥{sell_trades['pnl'].mean():,.0f}")
            print(f"  平均收益率: {sell_trades['pnl_rate'].mean():+.2%}")
            print(f"  平均持有天数: {sell_trades['holding_days'].mean():.1f}天")

    # 3. 生成每日汇总统计
    daily_summary = holdings_df.groupby('date').agg({
        'stock': 'count',
        'current_value': 'sum',
        'pnl': 'sum',
        'score': 'mean'
    }).reset_index()

    daily_summary.columns = ['date', 'position_count', 'total_value', 'total_pnl', 'avg_score']
    daily_summary['pnl_rate'] = daily_summary['total_pnl'] / (daily_summary['total_value'] - daily_summary['total_pnl'])

    summary_path = os.path.join(output_dir, 'daily_holdings_summary.csv')
    daily_summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"💾 每日汇总统计已保存: {summary_path}")

    # 4. 生成股票持仓统计
    stock_stats = holdings_df.groupby('stock').agg({
        'date': 'count',
        'pnl': 'last',
        'pnl_rate': 'last',
        'score': 'mean',
        'holding_days': 'max'
    }).reset_index()

    stock_stats.columns = ['stock', 'total_days', 'final_pnl', 'final_pnl_rate', 'avg_score', 'max_holding_days']
    stock_stats = stock_stats.sort_values('final_pnl', ascending=False)

    stock_path = os.path.join(output_dir, 'stock_holding_stats.csv')
    stock_stats.to_csv(stock_path, index=False, encoding='utf-8-sig')
    print(f"💾 股票持仓统计已保存: {stock_path}")

    # 打印持仓统计摘要
    print("\n" + "─" * 80)
    print("📊 持仓统计摘要")
    print("─" * 80)
    print(f"  总交易日数: {len(daily_summary)}")
    print(f"  涉及股票数: {len(stock_stats)}")
    print(f"  平均持仓数: {daily_summary['position_count'].mean():.1f} 只")
    print(f"  平均评分: {daily_summary['avg_score'].mean():.4f}")
    print(f"  最大浮盈: ¥{daily_summary['total_pnl'].max():,.0f}")
    print(f"  最大浮亏: ¥{daily_summary['total_pnl'].min():,.0f}")

    if len(stock_stats) > 0:
        print(f"\n  📈 盈利TOP3:")
        for idx, row in stock_stats.head(3).iterrows():
            print(f"     {row['stock']:12s} | ¥{row['final_pnl']:+10,.0f} ({row['final_pnl_rate']:+.2%}) | "
                  f"持有{row['max_holding_days']}天 | 均分{row['avg_score']:.4f}")

        print(f"\n  📉 亏损TOP3:")
        for idx, row in stock_stats.tail(3).iterrows():
            print(f"     {row['stock']:12s} | ¥{row['final_pnl']:+10,.0f} ({row['final_pnl_rate']:+.2%}) | "
                  f"持有{row['max_holding_days']}天 | 均分{row['avg_score']:.4f}")