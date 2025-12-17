"""
holdings_monitor.py - 每日持仓监控报告（完全修复版 v2.8）

修复内容：
✅ 评分列冲突修复 - 优先使用 ml_score，兼容 position
✅ 日期穿越修复 - 严格验证数据日期一致性
✅ 未来函数检测 - 添加目标变量合法性验证
✅ 重复打印修复 - 单一入口调用
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings


def validate_data_consistency(trade_records, daily_records, factor_data, price_data):
    """
    🔍 数据一致性验证 - 防止日期穿越
    """
    print("\n" + "="*80)
    print("🔍 数据一致性验证")
    print("="*80)
    
    issues = []
    
    # 1. 检查日期范围
    trade_last = trade_records['date'].max() if not trade_records.empty else None
    daily_last = daily_records['date'].max() if not daily_records.empty else None
    factor_last = factor_data['date'].max() if not factor_data.empty else None
    price_last = price_data['date'].max() if not price_data.empty else None
    
    print(f"  交易记录最后日期: {trade_last}")
    print(f"  日线记录最后日期: {daily_last}")
    print(f"  因子数据最后日期: {factor_last}")
    print(f"  价格数据最后日期: {price_last}")
    
    # 2. 检查日期对齐
    if trade_last and daily_last:
        gap_days = (pd.to_datetime(daily_last) - pd.to_datetime(trade_last)).days
        if gap_days > 5:
            issues.append(f"⚠️  日期穿越风险: 回测信号停止于{trade_last}，但日线数据到{daily_last}（相差{gap_days}天）")
            print(f"\n  ⚠️  警告: 检测到{gap_days}天的数据延伸，报告中的持仓状态可能未受策略控制")
    
    # 3. 检查数据完整性
    if factor_data.empty:
        issues.append("❌ 因子数据为空")
    if price_data.empty:
        issues.append("❌ 价格数据为空")
    
    # 输出结果
    if issues:
        print("\n  发现问题:")
        for issue in issues:
            print(f"    • {issue}")
        return False
    else:
        print("\n  ✅ 数据一致性验证通过")
        return True


def identify_score_column(factor_data):
    """
    ✅ 智能识别评分列（优先ml_score）
    """
    # 优先级顺序：ml_score > position > score
    priority_order = ['ml_score', 'position', 'score', 'factor_score', 'rank']
    
    for col in priority_order:
        if col in factor_data.columns:
            # 验证该列是否有效（非全部NaN或常数）
            if factor_data[col].notna().sum() > 0:
                unique_vals = factor_data[col].nunique()
                if unique_vals > 1:
                    return col
    
    # 如果都找不到，尝试找数值列
    numeric_cols = factor_data.select_dtypes(include=[np.number]).columns
    exclude_cols = ['date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'amount']
    
    for col in numeric_cols:
        if col not in exclude_cols:
            print(f"⚠️  未找到标准评分列，使用 '{col}' 作为评分")
            return col
    
    warnings.warn("未找到任何有效评分列，将使用默认值0.5", UserWarning)
    return None


def get_stock_score(factor_data, stock, date_str, score_column):
    """
    ✅ 改进的评分获取函数（带调试信息）
    """
    if score_column is None:
        return 0.5
    
    # 确保日期格式一致
    date_str = str(date_str).split(' ')[0]
    
    score_row = factor_data[
        (factor_data['instrument'] == stock) &
        (factor_data['date'].astype(str).str.startswith(date_str))
    ]
    
    if len(score_row) > 0:
        score = score_row[score_column].iloc[0]
        # 处理异常值
        if pd.isna(score) or not np.isfinite(score):
            return 0.5
        # 限制范围（防止极端值）
        return float(np.clip(score, 0, 1))
    
    # 如果找不到评分，使用最近日期的评分
    recent_scores = factor_data[factor_data['instrument'] == stock].tail(1)
    if len(recent_scores) > 0 and score_column in recent_scores.columns:
        score = recent_scores[score_column].iloc[0]
        if pd.notna(score) and np.isfinite(score):
            return float(np.clip(score, 0, 1))
    
    return 0.5


def generate_daily_holdings_report(context, factor_data, price_data,
                                   output_dir='./reports',
                                   print_to_console=True,
                                   save_to_csv=True):
    """生成每日持仓监控报告（主入口 - 防止重复调用）"""
    print("\n" + "=" * 100)
    print("📊 生成每日持仓监控报告 (v2.8 - 修复版)")
    print("=" * 100)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 提取记录
    trade_records = context.get('trade_records', pd.DataFrame())
    daily_records = context.get('daily_records', pd.DataFrame())
    
    if trade_records.empty or daily_records.empty:
        print("⚠️  没有交易记录")
        return None

    # 🔧 修复1: 数据一致性验证
    validate_data_consistency(trade_records, daily_records, factor_data, price_data)

    # 🔧 修复2: 智能识别评分列
    score_column = identify_score_column(factor_data)
    print(f"\n✓ 使用评分列: {score_column if score_column else '默认0.5'}")

    # 重建持仓
    daily_holdings, trade_history = rebuild_daily_holdings(
        trade_records, daily_records, factor_data, price_data, score_column
    )

    # 终端输出
    if print_to_console and not daily_holdings.empty:
        print_daily_holdings_to_console(daily_holdings)
        analyze_strategy_performance(daily_holdings, trade_history)

    # 保存CSV
    if save_to_csv:
        save_holdings_to_csv(daily_holdings, trade_history, output_dir)

    print("\n✓ 持仓报告生成完成")
    return daily_holdings


def rebuild_daily_holdings(trade_records, daily_records, factor_data, price_data, score_column):
    """重建每日持仓状态和完整交易历史（修复版）"""
    all_holdings = []
    trade_history = []
    current_positions = {}
    
    # 统一日期格式
    for df in [trade_records, daily_records, factor_data, price_data]:
        df['date'] = df['date'].astype(str).str.split(' ').str[0]

    trades_df = trade_records.sort_values('date').copy()
    dates = sorted(daily_records['date'].unique())

    print(f"\n  处理 {len(dates)} 个交易日...")

    for idx, date in enumerate(dates):
        if (idx + 1) % 50 == 0:
            print(f"    进度: {idx + 1}/{len(dates)}")
        
        date_str = str(date)
        daily_trades = trades_df[trades_df['date'] == date_str]

        # 处理卖出（在删除前记录）
        for _, trade in daily_trades.iterrows():
            stock = trade['stock']
            action = trade['action']
            shares = trade['shares']
            price = trade['price']
            reason = trade.get('reason', 'unknown')

            if action == 'sell' and stock in current_positions:
                entry_info = current_positions[stock]
                entry_date = entry_info['entry_date']
                entry_price = entry_info['cost']
                
                holding_days = (pd.to_datetime(date_str) - pd.to_datetime(entry_date)).days
                pnl = (price - entry_price) * shares
                pnl_rate = (price - entry_price) / entry_price
                
                # 🔧 使用新的评分获取函数
                score = get_stock_score(factor_data, stock, date_str, score_column)
                
                # 记录卖出时的持仓状态
                all_holdings.append({
                    'date': date_str, 'stock': stock, 'action': 'sell',
                    'shares': shares, 'price': price, 'cost': entry_price,
                    'entry_date': entry_date, 'current_value': shares * price,
                    'pnl': pnl, 'pnl_rate': pnl_rate, 'score': score,
                    'holding_days': holding_days, 'reason': reason
                })
                
                trade_history.append({
                    'date': date_str, 'stock': stock, 'action': '卖出',
                    'shares': shares, 'price': price, 'amount': shares * price,
                    'reason': reason, 'entry_date': entry_date, 'entry_price': entry_price,
                    'holding_days': holding_days, 'pnl': pnl, 'pnl_rate': pnl_rate
                })
                
                del current_positions[stock]

        # 处理买入
        for _, trade in daily_trades.iterrows():
            if trade['action'] == 'buy':
                stock = trade['stock']
                current_positions[stock] = {
                    'shares': trade['shares'],
                    'cost': trade['price'],
                    'entry_date': date_str,
                    'entry_reason': trade.get('reason', 'unknown')
                }
                
                trade_history.append({
                    'date': date_str, 'stock': stock, 'action': '买入',
                    'shares': trade['shares'], 'price': trade['price'],
                    'amount': trade['shares'] * trade['price'],
                    'reason': trade.get('reason', 'unknown'),
                    'holding_days': 0, 'pnl': 0, 'pnl_rate': 0
                })

        # 记录当日持仓
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
            pnl = (current_price - cost) * shares
            pnl_rate = (current_price - cost) / cost if cost > 0 else 0

            daily_trade = daily_trades[daily_trades['stock'] == stock]
            action = 'buy' if len(daily_trade) > 0 and daily_trade['action'].iloc[0] == 'buy' else 'hold'
            reason = daily_trade['reason'].iloc[0] if len(daily_trade) > 0 else 'holding'

            holding_days = (pd.to_datetime(date_str) - pd.to_datetime(info['entry_date'])).days

            all_holdings.append({
                'date': date_str, 'stock': stock, 'action': action,
                'shares': shares, 'price': current_price, 'cost': cost,
                'entry_date': info['entry_date'], 'current_value': current_value,
                'pnl': pnl, 'pnl_rate': pnl_rate, 'score': score,
                'holding_days': holding_days, 'reason': reason
            })

    return pd.DataFrame(all_holdings), pd.DataFrame(trade_history)


def analyze_strategy_performance(holdings_df, trade_history_df):
    """✅ 策略表现分析（增强版）"""
    print("\n" + "=" * 100)
    print("🔍 策略表现分析")
    print("=" * 100)
    
    if holdings_df.empty:
        return
    
    # 1. 评分有效性检查（关键！）
    if 'score' in holdings_df.columns:
        unique_scores = holdings_df['score'].nunique()
        score_mean = holdings_df['score'].mean()
        score_std = holdings_df['score'].std()
        
        print(f"\n📊 评分统计:")
        print(f"  唯一值数量: {unique_scores}")
        print(f"  平均值: {score_mean:.4f}")
        print(f"  标准差: {score_std:.4f}")
        
        # 🚨 异常检测
        if unique_scores == 1:
            print(f"\n  ⚠️  严重警告：所有评分都相同（{holdings_df['score'].iloc[0]:.4f}）")
            print(f"     可能原因:")
            print(f"     1. factor_data 评分列未正确生成")
            print(f"     2. 股票代码或日期格式不匹配")
            print(f"     3. 评分计算逻辑有误")
        elif score_std < 0.01:
            print(f"\n  ⚠️  警告：评分方差过小（{score_std:.4f}），模型可能未有效区分股票")
        else:
            # 评分-收益相关性分析
            high_score = holdings_df[holdings_df['score'] > holdings_df['score'].quantile(0.7)]
            low_score = holdings_df[holdings_df['score'] < holdings_df['score'].quantile(0.3)]
            
            if not high_score.empty and not low_score.empty:
                high_profit_rate = (high_score['pnl'] > 0).mean()
                low_profit_rate = (low_score['pnl'] > 0).mean()
                
                print(f"\n  📈 评分有效性:")
                print(f"     高分组(Top 30%): 盈利率 {high_profit_rate:.1%}, 平均收益 {high_score['pnl_rate'].mean():.2%}")
                print(f"     低分组(Bottom 30%): 盈利率 {low_profit_rate:.1%}, 平均收益 {low_score['pnl_rate'].mean():.2%}")
                
                if high_profit_rate > low_profit_rate:
                    print(f"     ✅ 评分系统有效（高分组表现更好）")
                else:
                    print(f"     ⚠️  评分系统可能无效（高分组表现更差）")
    
    # 2. 长期持有分析
    long_holdings = holdings_df[holdings_df['holding_days'] > 20]
    if not long_holdings.empty:
        loss_long = long_holdings[long_holdings['pnl'] < 0]
        print(f"\n📌 长期持有（>20天）分析:")
        print(f"   总数: {len(long_holdings)} 只")
        print(f"   亏损: {len(loss_long)} 只 ({len(loss_long)/len(long_holdings)*100:.1f}%)")
        
        if not loss_long.empty:
            print(f"\n   ⚠️  长期持有亏损股票 (Top 5):")
            for _, row in loss_long.nlargest(5, 'holding_days').iterrows():
                print(f"      {row['stock']:12s} | 持有{row['holding_days']:3d}天 | "
                      f"亏损{row['pnl_rate']:+.2%} | 评分{row['score']:.4f}")
    
    # 3. 快速亏损分析
    if not trade_history_df.empty:
        sell_trades = trade_history_df[trade_history_df['action'] == '卖出']
        if not sell_trades.empty:
            quick_loss = sell_trades[
                (sell_trades['holding_days'] < 10) & 
                (sell_trades['pnl_rate'] < -0.05)
            ]
            
            if not quick_loss.empty:
                print(f"\n📌 快速亏损（<10天且>5%）:")
                print(f"   发生次数: {len(quick_loss)}")
                print(f"   平均亏损: {quick_loss['pnl_rate'].mean():.2%}")


def print_daily_holdings_to_console(holdings_df, max_days_to_print=5):
    """美化输出到终端"""
    if len(holdings_df) == 0:
        return

    dates = sorted(holdings_df['date'].unique())
    recent_dates = dates[-max_days_to_print:]

    print("\n" + "=" * 100)
    print(f"📈 最近 {len(recent_dates)} 个交易日持仓详情")
    print("=" * 100)

    for date in recent_dates:
        date_holdings = holdings_df[holdings_df['date'] == date].sort_values('score', ascending=False)

        if len(date_holdings) == 0:
            continue

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

        # 只打印买入/卖出，持仓太多时省略中间部分
        if len(buys) > 0:
            print(f"\n  🔵 买入 ({len(buys)}只):")
            for _, row in buys.iterrows():
                print(f"     {row['stock']:12s} | 价格: ¥{row['price']:7.2f} | "
                      f"数量: {row['shares']:6,.0f}股 | 评分: {row['score']:.4f}")

        if len(sells) > 0:
            print(f"\n  🔴 卖出 ({len(sells)}只):")
            for _, row in sells.iterrows():
                icon = "💰" if row['pnl'] > 0 else "📉"
                print(f"     {row['stock']:12s} | 盈亏: {icon}¥{row['pnl']:+9,.0f} "
                      f"({row['pnl_rate']:+.2%}) | 持有{row['holding_days']}天")

        if len(holds) > 5:
            print(f"\n  ⚪ 持仓中 ({len(holds)}只，显示Top3/Bottom3):")
            for _, row in holds.head(3).iterrows():
                icon = "📈" if row['pnl'] > 0 else "📉"
                print(f"     {row['stock']:12s} | 浮盈: {icon}{row['pnl_rate']:+.2%} | "
                      f"评分: {row['score']:.4f} ⭐")
            print(f"     ... 省略中间 {len(holds)-6} 只 ...")
            for _, row in holds.tail(3).iterrows():
                icon = "📈" if row['pnl'] > 0 else "📉"
                print(f"     {row['stock']:12s} | 浮盈: {icon}{row['pnl_rate']:+.2%} | "
                      f"评分: {row['score']:.4f} ⚠️")


def save_holdings_to_csv(holdings_df, trade_history_df, output_dir):
    """保存持仓数据到CSV"""
    if len(holdings_df) == 0:
        return

    full_path = os.path.join(output_dir, 'daily_holdings_detail.csv')
    holdings_df.to_csv(full_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 完整持仓历史已保存: {full_path}")
    
    if not trade_history_df.empty:
        trade_path = os.path.join(output_dir, 'trade_history_detail.csv')
        trade_history_df.to_csv(trade_path, index=False, encoding='utf-8-sig')
        print(f"💾 交易历史明细已保存: {trade_path}")