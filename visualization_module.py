"""
visualization_module_patch.py - 报告模块修复补丁 v3.0
用于修复交易成本和持仓天数显示

使用方法：
1. 如果已有 visualization_module.py，将此补丁内容合并进去
2. 如果没有，可以直接使用此文件替代相关报告生成功能

版本：v3.0
日期：2025-12-29
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def generate_performance_report(context, output_dir='./reports'):
    """
    ✅ 修复版绩效报告生成器 - 正确显示交易成本和持仓天数

    Parameters:
    -----------
    context : dict
        回测上下文，包含 daily_records, trade_records, total_cost 等字段
    output_dir : str
        报告输出目录
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    report_path = os.path.join(output_dir, 'performance_report.txt')

    # 提取数据
    df_daily = context.get('daily_records', pd.DataFrame())
    df_trades = context.get('trade_records', pd.DataFrame())
    final_value = context.get('final_value', 0)
    total_return = context.get('total_return', 0)
    win_rate = context.get('win_rate', 0)
    total_realized_pnl = context.get('total_realized_pnl', 0)
    total_cost = context.get('total_cost', 0)  # ✅ 新增
    avg_holding_days = context.get('avg_holding_days', 0)  # ✅ 新增
    initial_capital = context.get('initial_capital', 1000000)
    positions = context.get('positions', {})

    # 计算指标
    if not df_daily.empty and 'return' in df_daily.columns:
        daily_returns = df_daily['return'].pct_change().dropna()

        # 年化收益率
        trading_days = len(df_daily)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 波动率
        volatility = daily_returns.std() * np.sqrt(252)

        # 夏普比率
        sharpe = annualized_return / volatility if volatility > 0 else 0

        # 最大回撤
        cumulative = (1 + df_daily['return']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
    else:
        annualized_return = 0
        volatility = 0
        sharpe = 0
        max_drawdown = 0

    # 盈亏统计
    if not df_trades.empty and 'action' in df_trades.columns:
        sell_trades = df_trades[df_trades['action'] == 'sell']
        if not sell_trades.empty and 'pnl' in sell_trades.columns:
            positive_pnl = sell_trades[sell_trades['pnl'] > 0]['pnl'].sum()
            negative_pnl = sell_trades[sell_trades['pnl'] < 0]['pnl'].sum()
            profit_loss_ratio = abs(positive_pnl / negative_pnl) if negative_pnl < 0 else 0
        else:
            positive_pnl = 0
            negative_pnl = 0
            profit_loss_ratio = 0
    else:
        positive_pnl = 0
        negative_pnl = 0
        profit_loss_ratio = 0

    # 生成报告
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("📊 策略绩效报告")
    report_lines.append("=" * 80)
    report_lines.append("")

    # 收益指标
    report_lines.append("【收益指标】")
    report_lines.append(f"  总收益率: {total_return:.2%}")
    report_lines.append(f"  年化收益率: {annualized_return:.2%}")
    report_lines.append(f"  总盈利 (正盈亏部分): ¥{positive_pnl:,.2f}")
    report_lines.append(f"  总亏损 (负盈亏部分): ¥{negative_pnl:,.2f}")
    report_lines.append(f"  净盈亏 (总盈利 + 总亏损): ¥{total_realized_pnl:,.2f}")
    report_lines.append(f"  交易费用总和: ¥{total_cost:,.2f}")  # ✅ 修复显示
    report_lines.append(f"  扣除费用后净盈亏: ¥{total_realized_pnl:,.2f}")
    report_lines.append(f"  净收益率: {(total_realized_pnl / initial_capital):.2%}")
    report_lines.append("")

    # 风险指标
    report_lines.append("【风险指标】")
    report_lines.append(f"  最大回撤: {max_drawdown:.2%}")
    report_lines.append(f"  年化波动率: {volatility:.2%}")
    report_lines.append(f"  夏普比率: {sharpe:.4f}")
    report_lines.append("")

    # 交易指标
    report_lines.append("【交易指标】")
    report_lines.append(f"  胜率: {win_rate:.2%}")
    report_lines.append(f"  平均持仓天数: {avg_holding_days:.1f} 天")  # ✅ 修复显示
    report_lines.append(f"  盈亏比: {profit_loss_ratio:.2f}")
    report_lines.append("")

    # 当前持仓
    report_lines.append("【当前持仓】")
    report_lines.append(f"  持仓数量: {len(positions)} 只")

    if positions:
        report_lines.append("  持仓明细:")
        for stock, info in positions.items():
            shares = info['shares']
            cost = info['cost']
            entry_date = info['entry_date']

            # 计算持仓天数
            today = datetime.now()
            entry_dt = pd.to_datetime(entry_date)
            holding_days = (today - entry_dt).days

            report_lines.append(f"    {stock}: {shares:.0f} 股 @ ¥{cost:.2f} (买入: {entry_date}, 持有{holding_days}天)")

    report_lines.append("")

    # 写入文件
    report_content = "\n".join(report_lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    # 同时打印到控制台
    print("\n" + report_content)
    print(f"✓ 绩效报告已保存: {report_path}")

    return report_path


def print_trading_summary(context):
    """
    ✅ 打印交易汇总信息
    """
    df_trades = context.get('trade_records', pd.DataFrame())

    if df_trades.empty:
        print("\n⚠️ 全程无交易记录")
        return

    print("\n" + "=" * 80)
    print("📊 交易汇总")
    print("=" * 80)

    # 按操作类型统计
    buy_count = len(df_trades[df_trades['action'] == 'buy'])
    sell_count = len(df_trades[df_trades['action'] == 'sell'])

    print(f"\n交易次数: 买入 {buy_count} 次, 卖出 {sell_count} 次, 总计 {len(df_trades)} 次")

    # 卖出原因统计
    if 'reason' in df_trades.columns:
        sell_trades = df_trades[df_trades['action'] == 'sell']
        if not sell_trades.empty:
            print("\n卖出原因分布:")
            reason_counts = sell_trades['reason'].value_counts()
            for reason, count in reason_counts.items():
                pct = count / len(sell_trades) * 100
                print(f"  {reason}: {count} 次 ({pct:.1f}%)")

    # 盈亏分布
    sell_trades = df_trades[df_trades['action'] == 'sell']
    if not sell_trades.empty and 'pnl' in sell_trades.columns:
        wins = sell_trades[sell_trades['pnl'] > 0]
        losses = sell_trades[sell_trades['pnl'] < 0]

        print(f"\n盈亏分布:")
        print(f"  盈利交易: {len(wins)} 次, 平均盈利 ¥{wins['pnl'].mean():,.2f}")
        print(f"  亏损交易: {len(losses)} 次, 平均亏损 ¥{losses['pnl'].mean():,.2f}")

        if not wins.empty:
            print(f"  最大单笔盈利: ¥{wins['pnl'].max():,.2f}")
        if not losses.empty:
            print(f"  最大单笔亏损: ¥{losses['pnl'].min():,.2f}")

    print("=" * 80)


def validate_context_fields(context):
    """
    ✅ 验证 context 字典是否包含所有必需字段
    """
    required_fields = [
        'daily_records', 'trade_records', 'final_value', 'total_return',
        'win_rate', 'positions', 'total_realized_pnl', 'initial_capital',
        'total_cost', 'avg_holding_days'  # ✅ 新增必需字段
    ]

    missing_fields = [field for field in required_fields if field not in context]

    if missing_fields:
        print(f"⚠️  警告: context 缺少以下字段: {missing_fields}")
        print("   -> 部分报告功能可能不可用")
        return False

    return True


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("""
    本文件为报告模块修复补丁
    
    使用方法：
    1. 将 generate_performance_report() 函数集成到你的 visualization_module.py
    2. 在 main.py 中调用时，确保 context 包含 total_cost 和 avg_holding_days
    
    修复要点：
    - 正确显示交易成本（从 context['total_cost'] 读取）
    - 正确显示平均持仓天数（从 context['avg_holding_days'] 读取）
    - 增强了盈亏统计和交易汇总功能
    """)