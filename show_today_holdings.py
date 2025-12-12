"""
show_today_holdings.py - 今日持仓可视化面板（修复版）

修复内容：
✅ 修复缺失列的检查和处理
✅ 改进图表兼容性
✅ 增强错误处理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def get_today_holdings(context, factor_data, price_data):
    """获取今日持仓详情"""
    trade_records = context.get('trade_records', pd.DataFrame())
    daily_records = context.get('daily_records', pd.DataFrame())
    
    if trade_records.empty or daily_records.empty:
        print("\n⚠️  没有交易记录")
        return None

    # 统一日期格式
    trade_records = trade_records.copy()
    daily_records = daily_records.copy()
    factor_data = factor_data.copy()
    price_data = price_data.copy()
    
    trade_records['date'] = trade_records['date'].astype(str)
    daily_records['date'] = daily_records['date'].astype(str)
    factor_data['date'] = factor_data['date'].astype(str)
    price_data['date'] = price_data['date'].astype(str)

    last_date = str(daily_records['date'].max())
    print(f"\n📅 分析日期: {last_date}")

    # 重建当前持仓
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
        print("\n⚠️  今日无持仓")
        return None

    # 构建今日持仓详情
    holdings = []
    for stock, info in current_positions.items():
        price_row = price_data[
            (price_data['instrument'] == stock) &
            (price_data['date'] == last_date)
        ]

        if len(price_row) == 0:
            continue

        current_price = price_row['close'].iloc[0]

        # 获取最新评分
        score_row = factor_data[
            (factor_data['instrument'] == stock) &
            (factor_data['date'] == last_date)
        ]
        score = score_row['position'].iloc[0] if len(score_row) > 0 else 0.5

        shares = info['shares']
        cost = info['cost']
        current_value = shares * current_price
        cost_value = shares * cost
        pnl = current_value - cost_value
        pnl_rate = (current_price - cost) / cost if cost > 0 else 0

        holding_days = (pd.to_datetime(last_date) - pd.to_datetime(info['entry_date'])).days

        holdings.append({
            'stock': stock,
            'entry_date': info['entry_date'],
            'holding_days': holding_days,
            'shares': shares,
            'cost': cost,
            'current_price': current_price,
            'current_value': current_value,
            'pnl': pnl,
            'pnl_rate': pnl_rate,
            'score': score
        })

    df = pd.DataFrame(holdings)
    if not df.empty and 'score' in df.columns:
        df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # 计算总持仓市值
    if not df.empty:
        total_value = df['current_value'].sum()
        # 添加持仓占比列
        df['position_ratio'] = df['current_value'] / total_value if total_value > 0 else 0
    
    return df


def print_today_holdings_console(holdings_df, context):
    """终端输出今日持仓"""
    if holdings_df is None or len(holdings_df) == 0:
        print("\n⚠️  今日无持仓")
        return

    daily_records = context.get('daily_records', pd.DataFrame())
    if daily_records.empty:
        print("\n⚠️  没有日线记录")
        return
    
    last_record = daily_records.iloc[-1]
    
    print(f"\n📅 今日日期: {last_record['date']}")
    print("=" * 130)

    # 账户概览
    total_value = holdings_df['current_value'].sum()
    total_pnl = holdings_df['pnl'].sum()
    total_cost = total_value - total_pnl
    total_pnl_rate = total_pnl / total_cost if total_cost > 0 else 0

    print(f"\n📊 账户概览:")
    print(f"  总资产: ¥{last_record.get('portfolio_value', total_value):,.0f}")
    print(f"  持仓市值: ¥{total_value:,.0f}")
    print(f"  持仓成本: ¥{total_cost:,.0f}")
    print(f"  浮动盈亏: ¥{total_pnl:+,.0f} ({total_pnl_rate:+.2%})")
    print(f"  持仓数量: {len(holdings_df)} 只")
    if 'score' in holdings_df.columns:
        print(f"  平均评分: {holdings_df['score'].mean():.4f}")

    # 盈亏统计
    profit_count = (holdings_df['pnl'] > 0).sum()
    loss_count = (holdings_df['pnl'] < 0).sum()
    flat_count = (holdings_df['pnl'] == 0).sum()

    print(f"\n📈 盈亏分布:")
    print(f"  盈利: {profit_count} 只 ({profit_count / len(holdings_df) * 100:.1f}%)")
    print(f"  亏损: {loss_count} 只 ({loss_count / len(holdings_df) * 100:.1f}%)")
    print(f"  持平: {flat_count} 只")

    # 详细持仓列表
    print(f"\n{'=' * 130}")
    header = f"{'排名':4s} {'股票代码':12s} {'买入日期':12s} {'持仓股数':>8s} "
    header += f"{'持仓占比':>8s} {'成本价':>8s} {'现价':>8s} {'浮动盈亏':>10s} {'收益率':>8s} "
    if 'score' in holdings_df.columns:
        header += f"{'评分':>8s}"
    print(header)
    print(f"{'=' * 130}")

    for idx, row in holdings_df.iterrows():
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
        if 'score' in holdings_df.columns:
            line += f"{row['score']:7.4f}"
        print(line)

    print(f"{'=' * 130}\n")

    # 关键持仓提示
    print("💡 关键持仓提示:")

    if len(holdings_df) > 0:
        if 'score' in holdings_df.columns:
            best_stock = holdings_df.iloc[0]
            print(f"  🏆 评分最高: {best_stock['stock']} (评分: {best_stock['score']:.4f}, "
                  f"收益: {best_stock['pnl_rate']:+.2%})")

        max_profit_stock = holdings_df.loc[holdings_df['pnl'].idxmax()]
        print(f"  💰 盈利最多: {max_profit_stock['stock']} (盈亏: ¥{max_profit_stock['pnl']:+,.0f}, "
              f"收益: {max_profit_stock['pnl_rate']:+.2%})")

        if holdings_df['pnl'].min() < 0:
            max_loss_stock = holdings_df.loc[holdings_df['pnl'].idxmin()]
            print(f"  📉 亏损最多: {max_loss_stock['stock']} (盈亏: ¥{max_loss_stock['pnl']:+,.0f}, "
                  f"收益: {max_loss_stock['pnl_rate']:+.2%})")

        longest_stock = holdings_df.loc[holdings_df['holding_days'].idxmax()]
        print(f"  🕐 持有最久: {longest_stock['stock']} (持有: {longest_stock['holding_days']}天, "
              f"收益: {longest_stock['pnl_rate']:+.2%})")

    print()


def plot_today_holdings_dashboard(holdings_df, context, output_dir='./reports'):
    """生成今日持仓可视化面板"""
    if holdings_df is None or len(holdings_df) == 0:
        print("\n⚠️  无持仓数据，跳过图表生成")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(20, 12))

    # 1. 持仓收益率排名
    ax1 = plt.subplot(2, 3, 1)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' if x < 0 else '#95a5a6'
              for x in holdings_df['pnl_rate']]
    ax1.barh(range(len(holdings_df)), holdings_df['pnl_rate'] * 100, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(holdings_df)))
    ax1.set_yticklabels(holdings_df['stock'], fontsize=8)
    ax1.set_xlabel('收益率 (%)', fontsize=10)
    ax1.set_title('📊 持仓收益率排名', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)

    # 2. 评分排名
    ax2 = plt.subplot(2, 3, 2)
    if 'score' in holdings_df.columns:
        colors2 = ['#f39c12' if x >= 0.8 else '#3498db' if x >= 0.6 else '#95a5a6'
                   for x in holdings_df['score']]
        ax2.barh(range(len(holdings_df)), holdings_df['score'], color=colors2, alpha=0.7)
        ax2.set_yticks(range(len(holdings_df)))
        ax2.set_yticklabels(holdings_df['stock'], fontsize=8)
        ax2.set_xlabel('评分', fontsize=10)
        ax2.set_title('⭐ 因子评分排名', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.grid(axis='x', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, '无评分数据', ha='center', va='center', fontsize=12)
        ax2.set_title('⭐ 因子评分排名', fontsize=12, fontweight='bold')
        ax2.axis('off')

    # 3. 持仓市值占比
    ax3 = plt.subplot(2, 3, 3)
    top_n = min(8, len(holdings_df))
    top_holdings = holdings_df.head(top_n)
    other_value = holdings_df.iloc[top_n:]['current_value'].sum() if len(holdings_df) > top_n else 0

    values = list(top_holdings['current_value'])
    labels = list(top_holdings['stock'])
    if other_value > 0:
        values.append(other_value)
        labels.append('其他')

    colors3 = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
               '#1abc9c', '#e67e22', '#34495e', '#95a5a6'][:len(values)]
    
    ax3.pie(values, labels=labels, autopct='%1.1f%%',
            colors=colors3, startangle=90)
    ax3.set_title('💰 持仓市值分布', fontsize=12, fontweight='bold')

    # 4. 持有天数分布
    ax4 = plt.subplot(2, 3, 4)
    colors4 = ['#3498db' if x < 10 else '#f39c12' if x < 30 else '#e74c3c'
               for x in holdings_df['holding_days']]
    ax4.bar(holdings_df['stock'], holdings_df['holding_days'],
            color=colors4, alpha=0.7)
    ax4.set_xlabel('股票代码', fontsize=10)
    ax4.set_ylabel('持有天数', fontsize=10)
    ax4.set_title('📅 持有天数分布', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45, labelsize=8)
    ax4.grid(axis='y', alpha=0.3)

    # 5. 盈亏金额分布
    ax5 = plt.subplot(2, 3, 5)
    colors5 = ['#2ecc71' if x > 0 else '#e74c3c' for x in holdings_df['pnl']]
    ax5.bar(holdings_df['stock'], holdings_df['pnl'], color=colors5, alpha=0.7)
    ax5.set_xlabel('股票代码', fontsize=10)
    ax5.set_ylabel('盈亏金额 (元)', fontsize=10)
    ax5.set_title('💰 盈亏金额分布', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45, labelsize=8)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax5.grid(axis='y', alpha=0.3)

    # 6. 综合统计
    ax6 = plt.subplot(2, 3, 6)
    
    total_value = holdings_df['current_value'].sum()
    total_pnl = holdings_df['pnl'].sum()
    profit_count = (holdings_df['pnl'] > 0).sum()
    loss_count = (holdings_df['pnl'] < 0).sum()
    
    stats_text = f"""
    持仓统计:
    总市值: ¥{total_value:,.0f}
    浮动盈亏: ¥{total_pnl:+,.0f}
    盈利股票: {profit_count} 只
    亏损股票: {loss_count} 只
    
    平均收益率: {holdings_df['pnl_rate'].mean():+.2%}
    最高收益率: {holdings_df['pnl_rate'].max():+.2%}
    最低收益率: {holdings_df['pnl_rate'].min():+.2%}
    """
    
    if 'score' in holdings_df.columns:
        stats_text += f"""
        
        评分统计:
        平均评分: {holdings_df['score'].mean():.4f}
        最高评分: {holdings_df['score'].max():.4f}
        最低评分: {holdings_df['score'].min():.4f}
        """
    
    ax6.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax6.set_title('📈 综合统计', fontsize=12, fontweight='bold')
    ax6.axis('off')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'today_holdings_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 今日持仓面板已保存: {output_path}")
    plt.close()


def save_today_holdings_csv(holdings_df, output_dir='./reports'):
    """保存今日持仓到CSV"""
    if holdings_df is None or len(holdings_df) == 0:
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'today_holdings.csv')
    holdings_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 今日持仓明细已保存: {output_path}")


def show_today_holdings_dashboard(context, factor_data, price_data,
                                  output_dir='./reports'):
    """主函数：显示今日持仓完整仪表板"""
    print("\n" + "=" * 120)
    print("🎯 生成今日持仓仪表板")
    print("=" * 120)

    holdings_df = get_today_holdings(context, factor_data, price_data)

    if holdings_df is None or len(holdings_df) == 0:
        print("\n⚠️  今日无持仓数据")
        return None

    print_today_holdings_console(holdings_df, context)
    plot_today_holdings_dashboard(holdings_df, context, output_dir)
    save_today_holdings_csv(holdings_df, output_dir)

    print("\n" + "=" * 120)
    print("✅ 今日持仓分析完成！")
    print("=" * 120)
    print(f"\n📁 输出文件:")
    print(f"  • {output_dir}/today_holdings_dashboard.png  - 持仓可视化面板")
    print(f"  • {output_dir}/today_holdings.csv             - 持仓明细CSV")
    print()

    return holdings_df