"""
visualization_module.py - 可视化模块（修复收益率计算版）

修复内容:
1. ✅ 修复总收益率计算错误（使用正确的初始资金）
2. ✅ 修复年化收益率计算
3. ✅ 风险指标计算优化
4. ✅ 持仓明细显示买入时间
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

    # ✨ 修复：使用正确的初始资金
    if 'capital_base' in context:
        initial_capital = context['capital_base']
    else:
        # 从第一天的收益率反推初始资金
        first_record = daily_records.iloc[0]
        if first_record['return'] != 0:
            initial_capital = first_record['portfolio_value'] / (1 + first_record['return'])
        else:
            initial_capital = first_record['portfolio_value']
    
    final_value = daily_records['portfolio_value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital

    # ✨ 修复：年化收益率计算（按照年化收益率计算规范）
    years = trading_days / 365  # 使用365天而非252天
    if years > 0 and total_return > -1:  # 确保本金未完全亏损
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = 0

    # ✨ 修复：最大回撤计算（基于初始资金）
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

    # 交易指标（基于trade_history_detail.csv中的数据）
    sell_trades = trade_records[trade_records['action'] == 'sell']
    
    # 初始化指标
    total_profit = 0
    total_loss = 0
    net_pnl = 0
    total_fees = 0
    win_rate = 0
    avg_holding_days = 0
    avg_profit = 0
    avg_loss = 0
    profit_loss_ratio = 0
    
    if len(sell_trades) > 0:
        # 按照用户要求的方式计算盈亏（与holdings_monitor.py保持一致）
        profit_trades = sell_trades[sell_trades['pnl'] > 0]
        loss_trades = sell_trades[sell_trades['pnl'] < 0]
        
        # 总盈利（只算正的盈亏部分）
        total_profit = profit_trades['pnl'].sum()
        # 总亏损（只算负的盈亏部分）
        total_loss = loss_trades['pnl'].sum()
        # 净盈亏 = 总盈利 + 总亏损
        net_pnl = total_profit + total_loss
        # 交易费用总和（这里需要从context或其他地方获取交易费用信息）
        # 由于trade_records中没有费用信息，我们需要从其他途径获取
        total_fees = sell_trades['fee'].sum() if 'fee' in sell_trades.columns else 0
        
        win_rate = len(profit_trades) / len(sell_trades) if len(sell_trades) > 0 else 0
        avg_holding_days = sell_trades['holding_days'].mean() if 'holding_days' in sell_trades.columns else 0
        
        avg_profit = profit_trades['pnl'].mean() if len(profit_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
        profit_loss_ratio = abs(avg_profit / avg_loss) if avg_loss != 0 else 0

    # 当前持仓
    positions = context.get('positions', {})

    # 生成报告
    report_path = os.path.join(output_dir, 'performance_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("📊 策略绩效报告（修复版）\n")
        f.write("=" * 80 + "\n\n")

        f.write("【回测基本信息】\n")
        f.write(f"回测开始日期: {start_date}\n")
        f.write(f"回测结束日期: {end_date}\n")
        f.write(f"回测交易天数: {trading_days} 天 ({years:.2f}年)\n")
        f.write(f"总交易次数: {total_trades} 次\n")
        f.write("=" * 80 + "\n\n")

        f.write("【收益指标】\n")
        f.write(f"初始资金: ¥{initial_capital:,.2f}\n")
        f.write(f"最终资产: ¥{final_value:,.2f}\n")
        f.write(f"总收益: ¥{final_value - initial_capital:,.2f}\n")
        f.write(f"总收益率: {total_return:+.2%}\n")
        f.write(f"年化收益率: {annualized_return:+.2%}\n")
        f.write(f"总盈利 (正盈亏部分): ¥{total_profit:,.2f}\n")
        f.write(f"总亏损 (负盈亏部分): ¥{total_loss:,.2f}\n")
        f.write(f"净盈亏 (总盈利 + 总亏损): ¥{net_pnl:,.2f}\n")
        f.write(f"交易费用总和: ¥{total_fees:,.2f}\n")
        f.write(f"扣除费用后净盈亏: ¥{net_pnl - total_fees:,.2f}\n")
        f.write(f"净收益率: {(net_pnl - total_fees) / initial_capital if initial_capital > 0 else 0:+.2%}\n\n")

        f.write("【风险指标】\n")
        f.write(f"最大回撤: {max_drawdown:.2%}\n")
        f.write(f"年化波动率: {annualized_volatility:.2%}\n")
        f.write(f"夏普比率: {sharpe_ratio:.4f}\n\n")

        f.write("【交易指标】\n")
        f.write(f"总交易次数: {len(sell_trades)}\n")
        f.write(f"胜率: {win_rate:.2%}\n")
        f.write(f"平均持仓天数: {avg_holding_days:.1f} 天\n")
        f.write(f"平均盈利: ¥{avg_profit:,.2f}\n")
        f.write(f"平均亏损: ¥{avg_loss:,.2f}\n")
        f.write(f"盈亏比: {profit_loss_ratio:.2f}\n\n")

        f.write("【当前持仓】\n")
        f.write(f"持仓数量: {len(positions)} 只\n")
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
    print(f"  总收益率: {total_return:+.2%}")
    print(f"  年化收益率: {annualized_return:+.2%}")
    print(f"  总盈利 (正盈亏部分): ¥{total_profit:,.2f}")
    print(f"  总亏损 (负盈亏部分): ¥{total_loss:,.2f}")
    print(f"  净盈亏 (总盈利 + 总亏损): ¥{net_pnl:,.2f}")
    print(f"  交易费用总和: ¥{total_fees:,.2f}")
    print(f"  扣除费用后净盈亏: ¥{net_pnl - total_fees:,.2f}")
    print(f"  净收益率: {(net_pnl - total_fees) / initial_capital if initial_capital > 0 else 0:+.2%}")

    print("\n【风险指标】")
    print(f"  最大回撤: {max_drawdown:.2%}")
    print(f"  年化波动率: {annualized_volatility:.2%}")
    print(f"  夏普比率: {sharpe_ratio:.4f}")

    print("\n【交易指标】")
    print(f"  胜率: {win_rate:.2%}")
    print(f"  平均持仓天数: {avg_holding_days:.1f} 天")
    print(f"  盈亏比: {profit_loss_ratio:.2f}")

    print("\n【当前持仓】")
    print(f"  持仓数量: {len(positions)} 只")

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
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_pnl': net_pnl,
        'total_fees': total_fees,
        'net_return': (net_pnl - total_fees) / initial_capital if initial_capital > 0 else 0
    }


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
        print("⚠️ 无卖出交易，跳过TOP股票分析")
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


# ========== 额外的诊断工具 ==========

def diagnose_return_calculation(context):
    """
    诊断收益率计算是否正确
    """
    print("\n" + "=" * 80)
    print("🔍 收益率计算诊断")
    print("=" * 80)
    
    daily_records = context['daily_records']
    
    # 获取初始资金
    if 'capital_base' in context:
        capital_base = context['capital_base']
        print(f"✓ 从context获取初始资金: ¥{capital_base:,.2f}")
    else:
        first_record = daily_records.iloc[0]
        if first_record['return'] != 0:
            capital_base = first_record['portfolio_value'] / (1 + first_record['return'])
        else:
            capital_base = first_record['portfolio_value']
        print(f"⚠️ 从第一天记录反推初始资金: ¥{capital_base:,.2f}")
    
    # 第一天和最后一天的数据
    first_day = daily_records.iloc[0]
    last_day = daily_records.iloc[-1]
    
    print(f"\n第一天 ({first_day['date']}):")
    print(f"  组合价值: ¥{first_day['portfolio_value']:,.2f}")
    print(f"  记录的收益率: {first_day['return']:.2%}")
    
    print(f"\n最后一天 ({last_day['date']}):")
    print(f"  组合价值: ¥{last_day['portfolio_value']:,.2f}")
    print(f"  记录的收益率: {last_day['return']:.2%}")
    
    # 计算总收益率
    total_return_correct = (last_day['portfolio_value'] - capital_base) / capital_base
    total_return_wrong = (last_day['portfolio_value'] - first_day['portfolio_value']) / first_day['portfolio_value']
    
    print(f"\n收益率计算:")
    print(f"  ✅ 正确方法: ({last_day['portfolio_value']:,.0f} - {capital_base:,.0f}) / {capital_base:,.0f} = {total_return_correct:+.2%}")
    print(f"  ❌ 错误方法: ({last_day['portfolio_value']:,.0f} - {first_day['portfolio_value']:,.0f}) / {first_day['portfolio_value']:,.0f} = {total_return_wrong:+.2%}")
    
    if abs(total_return_correct - total_return_wrong) > 0.01:
        print(f"\n⚠️ 检测到收益率计算差异: {abs(total_return_correct - total_return_wrong):.2%}")
    else:
        print(f"\n✓ 收益率计算一致")
    
    print("=" * 80 + "\n")