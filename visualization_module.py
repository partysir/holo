"""
visualization_module.py - 可视化模块（修复版）

修复内容:
1. 风险指标计算错误
2. 持仓明细显示买入时间
3. 最大回撤计算修复
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import os

# 配置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def generate_performance_report(context, output_dir='./reports'):
    """
    生成绩效报告（修复版）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    daily_records = context['daily_records']
    trade_records = context['trade_records']

    # 基础信息
    start_date = daily_records['date'].iloc[0]
    end_date = daily_records['date'].iloc[-1]
    trading_days = len(daily_records)
    total_trades = len(trade_records)

    # 收益指标
    initial_capital = daily_records['portfolio_value'].iloc[0]
    final_value = daily_records['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # ✨ 修复：年化收益率计算
    years = trading_days / 252
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0

    # ✨ 修复：最大回撤计算
    cummax = daily_records['portfolio_value'].cummax()
    drawdown = (daily_records['portfolio_value'] - cummax) / cummax
    max_drawdown = drawdown.min()

    # ✨ 修复：日收益率和波动率计算
    daily_returns = daily_records['portfolio_value'].pct_change().dropna()

    # 过滤异常值（防止除零或极端波动）
    daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
    daily_returns = daily_returns[np.abs(daily_returns) < 1]  # 过滤掉单日涨跌超过100%的异常值

    if len(daily_returns) > 1:
        volatility_daily = daily_returns.std()
        annualized_volatility = volatility_daily * np.sqrt(252)

        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03
        excess_return = annualized_return - risk_free_rate
        sharpe_ratio = excess_return / annualized_volatility if annualized_volatility > 0 else 0
    else:
        annualized_volatility = 0
        sharpe_ratio = 0

    # 交易指标
    sell_trades = trade_records[trade_records['action'] == 'sell']

    if len(sell_trades) > 0:
        win_rate = (sell_trades['pnl'] > 0).sum() / len(sell_trades)
        avg_holding_days = sell_trades['holding_days'].mean()

        profit_trades = sell_trades[sell_trades['pnl'] > 0]
        loss_trades = sell_trades[sell_trades['pnl'] < 0]

        avg_profit = profit_trades['pnl'].mean() if len(profit_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0
    else:
        win_rate = 0
        avg_holding_days = 0
        avg_profit = 0
        avg_loss = 0
        profit_loss_ratio = 0

    # 当前持仓（显示买入时间）
    positions = context.get('positions', {})

    # 生成报告
    report_path = os.path.join(output_dir, 'performance_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("📊 策略绩效报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("【回测基本信息】\n")
        f.write(f"回测开始日期: {start_date}\n")
        f.write(f"回测结束日期: {end_date}\n")
        f.write(f"回测交易天数: {trading_days} 天\n")
        f.write(f"总交易次数:   {total_trades} 次\n")
        f.write("=" * 80 + "\n\n")

        f.write("【收益指标】\n")
        f.write(f"初始资金:     ¥{initial_capital:,.2f}\n")
        f.write(f"最终资产:     ¥{final_value:,.2f}\n")
        f.write(f"总收益:       ¥{final_value - initial_capital:,.2f}\n")
        f.write(f"总收益率:     {total_return:+.2%}\n")
        f.write(f"年化收益率:   {annualized_return:+.2%}\n\n")

        f.write("【风险指标】\n")
        f.write(f"最大回撤:     {max_drawdown:.2%}\n")
        f.write(f"年化波动率:   {annualized_volatility:.2%}\n")
        f.write(f"夏普比率:     {sharpe_ratio:.4f}\n\n")

        f.write("【交易指标】\n")
        f.write(f"总交易次数:   {len(sell_trades)}\n")
        f.write(f"胜率:         {win_rate:.2%}\n")
        f.write(f"平均持仓天数: {avg_holding_days:.1f} 天\n")
        f.write(f"平均盈利:     ¥{avg_profit:,.2f}\n")
        f.write(f"平均亏损:     ¥{avg_loss:,.2f}\n")
        f.write(f"盈亏比:       {profit_loss_ratio:.2f}\n\n")

        f.write("【当前持仓】\n")
        f.write(f"持仓数量:     {len(positions)} 只\n")
        f.write("持仓明细:\n")

        if positions:
            # 按买入时间排序
            sorted_positions = sorted(positions.items(),
                                    key=lambda x: x[1]['entry_date'])

            for stock, info in sorted_positions:
                holding_days = (pd.to_datetime(end_date) -
                              pd.to_datetime(info['entry_date'])).days
                f.write(f"  {stock}: {info['shares']} 股 @ ¥{info['cost']:.2f} "
                       f"(买入: {info['entry_date']}, 持有{holding_days}天)\n")
        else:
            f.write("  暂无持仓\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ 绩效报告已保存: {report_path}")

    # 打印到终端
    print("\n" + "=" * 80)
    print("📊 策略绩效报告")
    print("=" * 80 + "\n")

    print("【收益指标】")
    print(f"  总收益率:     {total_return:+.2%}")
    print(f"  年化收益率:   {annualized_return:+.2%}")

    print("\n【风险指标】")
    print(f"  最大回撤:     {max_drawdown:.2%}")
    print(f"  年化波动率:   {annualized_volatility:.2%}")
    print(f"  夏普比率:     {sharpe_ratio:.4f}")

    print("\n【交易指标】")
    print(f"  胜率:         {win_rate:.2%}")
    print(f"  平均持仓天数: {avg_holding_days:.1f} 天")
    print(f"  盈亏比:       {profit_loss_ratio:.2f}")

    print("\n【当前持仓】")
    print(f"  持仓数量:     {len(positions)} 只")

    if positions:
        sorted_positions = sorted(positions.items(),
                                key=lambda x: x[1]['entry_date'])
        print("  持仓明细:")
        for stock, info in sorted_positions:
            holding_days = (pd.to_datetime(end_date) -
                          pd.to_datetime(info['entry_date'])).days
            print(f"    {stock}: {info['shares']} 股 @ ¥{info['cost']:.2f} "
                  f"(买入: {info['entry_date']}, 持有{holding_days}天)")

    print()


def plot_monitoring_results(context, output_dir='./reports'):
    """生成监控面板"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    daily_records = context['daily_records']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. 资产曲线
    ax1 = axes[0, 0]
    ax1.plot(range(len(daily_records)), daily_records['portfolio_value'],
            linewidth=2, color='#2E86AB')
    ax1.set_title('资产曲线', fontsize=14, fontweight='bold')
    ax1.set_xlabel('交易日')
    ax1.set_ylabel('资产 (元)')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')

    # 2. 回撤曲线
    ax2 = axes[0, 1]
    cummax = daily_records['portfolio_value'].cummax()
    drawdown = (daily_records['portfolio_value'] - cummax) / cummax * 100
    ax2.fill_between(range(len(drawdown)), drawdown, 0,
                     color='#A23B72', alpha=0.5)
    ax2.set_title('回撤曲线', fontsize=14, fontweight='bold')
    ax2.set_xlabel('交易日')
    ax2.set_ylabel('回撤 (%)')
    ax2.grid(True, alpha=0.3)

    # 3. 持仓数量
    ax3 = axes[1, 0]
    ax3.plot(range(len(daily_records)), daily_records['position_count'],
            linewidth=2, color='#F18F01', marker='o', markersize=2)
    ax3.set_title('持仓数量', fontsize=14, fontweight='bold')
    ax3.set_xlabel('交易日')
    ax3.set_ylabel('持仓股票数')
    ax3.grid(True, alpha=0.3)

    # 4. 现金余额
    ax4 = axes[1, 1]
    ax4.plot(range(len(daily_records)), daily_records['cash'],
            linewidth=2, color='#06A77D')
    ax4.set_title('现金余额', fontsize=14, fontweight='bold')
    ax4.set_xlabel('交易日')
    ax4.set_ylabel('现金 (元)')
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'monitoring_dashboard.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 监控面板已保存: {output_path}")


def plot_top_stocks_evolution(context, output_dir='./reports'):
    """生成TOP股票分析图"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trade_records = context['trade_records']
    sell_trades = trade_records[trade_records['action'] == 'sell']

    if len(sell_trades) == 0:
        print("⚠️  无卖出交易，跳过TOP股票分析")
        return

    # 按盈亏排序
    top_profits = sell_trades.nlargest(10, 'pnl')
    top_losses = sell_trades.nsmallest(10, 'pnl')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 盈利TOP10
    ax1 = axes[0]
    colors1 = ['#2ecc71' for _ in range(len(top_profits))]
    bars1 = ax1.barh(range(len(top_profits)), top_profits['pnl']/1000,
                     color=colors1, alpha=0.7)
    ax1.set_yticks(range(len(top_profits)))
    ax1.set_yticklabels(top_profits['stock'], fontsize=9)
    ax1.set_xlabel('盈利 (千元)', fontsize=11)
    ax1.set_title('📈 盈利TOP10', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # 亏损TOP10
    ax2 = axes[1]
    colors2 = ['#e74c3c' for _ in range(len(top_losses))]
    bars2 = ax2.barh(range(len(top_losses)), top_losses['pnl']/1000,
                     color=colors2, alpha=0.7)
    ax2.set_yticks(range(len(top_losses)))
    ax2.set_yticklabels(top_losses['stock'], fontsize=9)
    ax2.set_xlabel('亏损 (千元)', fontsize=11)
    ax2.set_title('📉 亏损TOP10', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'top_stocks_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ TOP股票分析已保存: {output_path}")