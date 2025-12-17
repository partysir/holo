"""
show_today_holdings.py - 今日持仓可视化面板（v2.8 修复版）

修复内容：
✅ 评分列智能识别（优先ml_score）
✅ 防止重复打印
✅ 日期一致性验证
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings

# 配置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def identify_score_column_safe(factor_data):
    """🔧 安全的评分列识别（与monitoring模块一致）"""
    priority_order = ['ml_score', 'position', 'score', 'factor_score']
    
    for col in priority_order:
        if col in factor_data.columns:
            if factor_data[col].notna().sum() > 0 and factor_data[col].nunique() > 1:
                return col
    
    warnings.warn("未找到有效评分列，使用默认值0.5", UserWarning)
    return None


def get_stock_score_safe(factor_data, stock, date_str, score_column):
    """🔧 安全的评分获取函数"""
    if score_column is None:
        return 0.5
    
    date_str = str(date_str).split(' ')[0]
    
    score_row = factor_data[
        (factor_data['instrument'] == stock) &
        (factor_data['date'].astype(str).str.startswith(date_str))
    ]
    
    if len(score_row) > 0:
        score = score_row[score_column].iloc[0]
        if pd.notna(score) and np.isfinite(score):
            return float(np.clip(score, 0, 1))
    
    # Fallback：使用最近的评分
    recent = factor_data[factor_data['instrument'] == stock].tail(1)
    if len(recent) > 0 and score_column in recent.columns:
        score = recent[score_column].iloc[0]
        if pd.notna(score) and np.isfinite(score):
            return float(np.clip(score, 0, 1))
    
    return 0.5


def get_today_holdings(context, factor_data, price_data):
    """获取今日持仓详情（修复版）"""
    trade_records = context.get('trade_records', pd.DataFrame())
    daily_records = context.get('daily_records', pd.DataFrame())
    
    if trade_records.empty or daily_records.empty:
        return None

    # 统一日期格式
    for df in [trade_records, daily_records, factor_data, price_data]:
        df['date'] = df['date'].astype(str).str.split(' ').str[0]

    last_date = str(daily_records['date'].max())
    
    # 🔧 识别评分列
    score_column = identify_score_column_safe(factor_data)
    print(f"  ✓ 使用评分列: {score_column if score_column else '默认0.5'}")

    # 重建持仓
    current_positions = {}
    trades_df = trade_records.sort_values('date')

    for _, trade in trades_df.iterrows():
        stock = trade['stock']
        action = trade['action']

        if action == 'buy':
            current_positions[stock] = {
                'shares': trade['shares'],
                'cost': trade['price'],
                'entry_date': trade['date']
            }
        elif action == 'sell' and stock in current_positions:
            del current_positions[stock]

    if not current_positions:
        return None

    # 构建持仓详情
    holdings = []
    for stock, info in current_positions.items():
        price_row = price_data[
            (price_data['instrument'] == stock) &
            (price_data['date'] == last_date)
        ]

        if len(price_row) == 0:
            continue

        current_price = price_row['close'].iloc[0]
        
        # 🔧 使用安全的评分获取
        score = get_stock_score_safe(factor_data, stock, last_date, score_column)

        shares = info['shares']
        cost = info['cost']
        current_value = shares * current_price
        pnl = (current_price - cost) * shares
        pnl_rate = (current_price - cost) / cost if cost > 0 else 0
        holding_days = (pd.to_datetime(last_date) - pd.to_datetime(info['entry_date'])).days

        holdings.append({
            'stock': stock, 'entry_date': info['entry_date'],
            'holding_days': holding_days, 'shares': shares,
            'cost': cost, 'current_price': current_price,
            'current_value': current_value, 'pnl': pnl,
            'pnl_rate': pnl_rate, 'score': score
        })

    df = pd.DataFrame(holdings)
    if not df.empty and 'score' in df.columns:
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return df


def print_today_holdings_console(holdings_df, context):
    """终端输出今日持仓（防止重复打印）"""
    if holdings_df is None or len(holdings_df) == 0:
        return

    daily_records = context.get('daily_records', pd.DataFrame())
    if daily_records.empty:
        return
    
    last_record = daily_records.iloc[-1]
    
    print(f"\n📅 今日日期: {last_record['date']}")
    print("=" * 120)

    # 账户概览
    total_value = holdings_df['current_value'].sum()
    total_pnl = holdings_df['pnl'].sum()
    total_cost = total_value - total_pnl
    total_pnl_rate = total_pnl / total_cost if total_cost > 0 else 0

    print(f"\n📊 账户概览:")
    print(f"  总资产: ¥{last_record.get('portfolio_value', total_value):,.0f}")
    print(f"  持仓市值: ¥{total_value:,.0f}")
    print(f"  浮动盈亏: ¥{total_pnl:+,.0f} ({total_pnl_rate:+.2%})")
    print(f"  持仓数量: {len(holdings_df)} 只")
    print(f"  平均评分: {holdings_df['score'].mean():.4f}")

    # 盈亏统计
    profit_count = (holdings_df['pnl'] > 0).sum()
    print(f"\n📈 盈亏分布: 盈利 {profit_count} 只 ({profit_count / len(holdings_df) * 100:.1f}%)")

    # 详细列表（简化版）
    print(f"\n{'排名':4s} {'代码':12s} {'买入日':12s} {'持有':4s} {'成本':>8s} "
          f"{'现价':>8s} {'盈亏':>10s} {'收益率':>8s} {'评分':>8s}")
    print("=" * 90)

    for idx, row in holdings_df.head(10).iterrows():
        status = "📈" if row['pnl'] > 0 else "📉" if row['pnl'] < 0 else "⚪"
        print(f"{idx+1:3d}  {row['stock']:12s} {row['entry_date']:12s} "
              f"{row['holding_days']:3d}天 {row['cost']:8.2f} {row['current_price']:8.2f} "
              f"{row['pnl']:+9,.0f} {row['pnl_rate']:+7.2%} {row['score']:7.4f} {status}")

    if len(holdings_df) > 10:
        print(f"\n... 省略 {len(holdings_df)-10} 只股票 ...")


def plot_today_holdings_dashboard(holdings_df, context, output_dir='./reports'):
    """生成可视化面板（防止重复生成）"""
    if holdings_df is None or len(holdings_df) == 0:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(18, 10))

    # 1. 收益率排名
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in holdings_df['pnl_rate']]
    ax1.barh(range(len(holdings_df)), holdings_df['pnl_rate'] * 100, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(holdings_df)))
    ax1.set_yticklabels(holdings_df['stock'], fontsize=8)
    ax1.set_xlabel('收益率 (%)')
    ax1.set_title('📊 持仓收益率')
    ax1.grid(axis='x', alpha=0.3)

    # 2. 评分排名
    ax2 = plt.subplot(2, 3, 2)
    colors2 = ['#f39c12' if x >= 0.7 else '#3498db' for x in holdings_df['score']]
    ax2.barh(range(len(holdings_df)), holdings_df['score'], color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(holdings_df)))
    ax2.set_yticklabels(holdings_df['stock'], fontsize=8)
    ax2.set_xlabel('评分')
    ax2.set_title('⭐ 因子评分')
    ax2.set_xlim(0, 1)

    # 3. 市值分布
    ax3 = plt.subplot(2, 3, 3)
    top_n = min(8, len(holdings_df))
    values = list(holdings_df.head(top_n)['current_value'])
    labels = list(holdings_df.head(top_n)['stock'])
    if len(holdings_df) > top_n:
        values.append(holdings_df.iloc[top_n:]['current_value'].sum())
        labels.append('其他')
    
    ax3.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax3.set_title('💰 持仓市值分布')

    # 4-6. 其他图表...
    ax4 = plt.subplot(2, 3, 4)
    ax4.bar(holdings_df['stock'], holdings_df['holding_days'], alpha=0.7)
    ax4.set_xlabel('股票代码')
    ax4.set_ylabel('持有天数')
    ax4.set_title('📅 持有天数')
    ax4.tick_params(axis='x', rotation=45, labelsize=8)

    ax5 = plt.subplot(2, 3, 5)
    colors5 = ['#2ecc71' if x > 0 else '#e74c3c' for x in holdings_df['pnl']]
    ax5.bar(holdings_df['stock'], holdings_df['pnl'], color=colors5, alpha=0.7)
    ax5.set_xlabel('股票代码')
    ax5.set_ylabel('盈亏金额 (元)')
    ax5.set_title('💰 盈亏分布')
    ax5.tick_params(axis='x', rotation=45, labelsize=8)

    ax6 = plt.subplot(2, 3, 6)
    stats_text = f"""
    持仓统计:
    总市值: ¥{holdings_df['current_value'].sum():,.0f}
    浮动盈亏: ¥{holdings_df['pnl'].sum():+,.0f}
    平均收益率: {holdings_df['pnl_rate'].mean():+.2%}
    平均评分: {holdings_df['score'].mean():.4f}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    ax6.set_title('📈 综合统计')
    ax6.axis('off')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'today_holdings_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  💾 持仓面板已保存: {output_path}")
    plt.close()


def show_today_holdings_dashboard(context, factor_data, price_data, output_dir='./reports'):
    """
    🔧 主函数（单一入口，防止重复调用）
    """
    print("\n" + "=" * 120)
    print("🎯 生成今日持仓仪表板 (v2.8)")
    print("=" * 120)

    # 获取持仓
    holdings_df = get_today_holdings(context, factor_data, price_data)

    if holdings_df is None or len(holdings_df) == 0:
        print("\n⚠️  今日无持仓数据")
        return None

    # 🔧 关键修复：只调用一次输出函数
    print_today_holdings_console(holdings_df, context)
    plot_today_holdings_dashboard(holdings_df, context, output_dir)

    # 保存CSV
    output_path = os.path.join(output_dir, 'today_holdings.csv')
    holdings_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"  💾 持仓明细已保存: {output_path}")

    print("\n✅ 今日持仓分析完成")
    return holdings_df