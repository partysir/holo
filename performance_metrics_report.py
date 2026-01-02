"""
performance_metrics_report.py - 完整收益指标报告模块

功能:
1. ✅ 收益率统计（总收益、年化收益、月度收益）
2. ✅ 胜率分析（总胜率、多头胜率、单笔统计）
3. ✅ 盈亏统计（盈利笔数、亏损笔数、盈亏比）
4. ✅ 风险指标（夏普、最大回撤、波动率）
5. ✅ 交易统计（总交易次数、换手率、持仓周期）
6. ✅ 对比基准（超额收益、信息比率）

版本: v1.0
日期: 2025-12-30
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


def calculate_performance_metrics(context, benchmark_data=None):
    """
    计算完整的绩效指标

    参数:
        context: 回测上下文，包含daily_records和trade_records
        benchmark_data: 基准数据（可选）

    返回:
        dict: 包含所有绩效指标的字典
    """
    metrics = {}

    # 提取数据
    daily_records = context.get('daily_records', pd.DataFrame())
    trade_records = context.get('trade_records', pd.DataFrame())

    if daily_records.empty:
        print("  ⚠️  没有日度记录数据")
        return metrics

    # ========== 1. 基础信息 ==========
    metrics['start_date'] = daily_records['date'].min()
    metrics['end_date'] = daily_records['date'].max()
    metrics['trading_days'] = len(daily_records)
    metrics['initial_capital'] = daily_records['portfolio_value'].iloc[0]
    metrics['final_capital'] = daily_records['portfolio_value'].iloc[-1]

    # ========== 2. 收益率指标 ==========

    # 总收益率
    metrics['total_return'] = (metrics['final_capital'] / metrics['initial_capital'] - 1)

    # 年化收益率
    years = metrics['trading_days'] / 252
    if years > 0:
        metrics['annualized_return'] = (1 + metrics['total_return']) ** (1 / years) - 1
    else:
        metrics['annualized_return'] = 0

    # 日收益率序列
    daily_returns = daily_records['portfolio_value'].pct_change().fillna(0)

    # 月度收益率
    if 'date' in daily_records.columns:
        daily_records['year_month'] = pd.to_datetime(daily_records['date']).dt.to_period('M')
        monthly_returns = daily_records.groupby('year_month')['portfolio_value'].last().pct_change()

        metrics['avg_monthly_return'] = monthly_returns.mean()
        metrics['best_month'] = monthly_returns.max()
        metrics['worst_month'] = monthly_returns.min()
        metrics['positive_months'] = (monthly_returns > 0).sum()
        metrics['total_months'] = len(monthly_returns)
        metrics['monthly_win_rate'] = metrics['positive_months'] / metrics['total_months'] if metrics[
                                                                                                  'total_months'] > 0 else 0

    # ========== 3. 风险指标 ==========

    # 波动率（年化）
    metrics['volatility'] = daily_returns.std() * np.sqrt(252)

    # 夏普比率
    risk_free_rate = 0.03  # 假设无风险利率3%
    if metrics['volatility'] > 0:
        metrics['sharpe_ratio'] = (metrics['annualized_return'] - risk_free_rate) / metrics['volatility']
    else:
        metrics['sharpe_ratio'] = 0

    # 最大回撤
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    metrics['max_drawdown'] = drawdown.min()
    metrics['max_drawdown_duration'] = _calculate_max_dd_duration(drawdown)

    # Calmar比率
    if abs(metrics['max_drawdown']) > 0:
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown'])
    else:
        metrics['calmar_ratio'] = 0

    # Sortino比率（下行波动率）
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(252)
    if downside_std > 0:
        metrics['sortino_ratio'] = (metrics['annualized_return'] - risk_free_rate) / downside_std
    else:
        metrics['sortino_ratio'] = 0

    # ========== 4. 交易统计 ==========

    if not trade_records.empty:
        # 总交易次数
        metrics['total_trades'] = len(trade_records)
        metrics['buy_trades'] = len(trade_records[trade_records['action'] == 'buy'])
        metrics['sell_trades'] = len(trade_records[trade_records['action'] == 'sell'])

        # 配对交易分析（买入-卖出配对）
        paired_trades = _pair_trades(trade_records)

        if paired_trades:
            metrics['completed_trades'] = len(paired_trades)

            # 盈利/亏损交易
            winning_trades = [t for t in paired_trades if t['profit'] > 0]
            losing_trades = [t for t in paired_trades if t['profit'] < 0]

            metrics['winning_trades'] = len(winning_trades)
            metrics['losing_trades'] = len(losing_trades)

            # 胜率
            metrics['win_rate'] = len(winning_trades) / len(paired_trades) if len(paired_trades) > 0 else 0

            # 平均盈利/亏损
            metrics['avg_win'] = np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0
            metrics['avg_loss'] = np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0

            # 最大盈利/亏损
            all_profits = [t['profit'] for t in paired_trades]
            metrics['max_win'] = max(all_profits) if all_profits else 0
            metrics['max_loss'] = min(all_profits) if all_profits else 0

            # 盈亏比
            if abs(metrics['avg_loss']) > 0:
                metrics['profit_loss_ratio'] = metrics['avg_win'] / abs(metrics['avg_loss'])
            else:
                metrics['profit_loss_ratio'] = 0

            # 平均持仓天数
            holding_periods = [t['holding_days'] for t in paired_trades]
            metrics['avg_holding_days'] = np.mean(holding_periods) if holding_periods else 0
            metrics['max_holding_days'] = max(holding_periods) if holding_periods else 0
            metrics['min_holding_days'] = min(holding_periods) if holding_periods else 0
        else:
            metrics['completed_trades'] = 0
            metrics['winning_trades'] = 0
            metrics['losing_trades'] = 0
            metrics['win_rate'] = 0

    # 换手率
    if 'turnover' in daily_records.columns:
        metrics['avg_turnover'] = daily_records['turnover'].mean()

    # ========== 5. 基准对比 ==========

    if benchmark_data is not None and not benchmark_data.empty:
        try:
            # 对齐日期
            benchmark_data['date'] = pd.to_datetime(benchmark_data['date'])
            daily_records['date'] = pd.to_datetime(daily_records['date'])

            merged = daily_records.merge(benchmark_data[['date', 'close']], on='date', how='left',
                                         suffixes=('', '_bench'))

            if 'close_bench' in merged.columns:
                # 基准收益
                bench_initial = merged['close_bench'].iloc[0]
                bench_final = merged['close_bench'].iloc[-1]
                metrics['benchmark_return'] = (bench_final / bench_initial - 1) if bench_initial > 0 else 0

                # 超额收益
                metrics['excess_return'] = metrics['total_return'] - metrics['benchmark_return']

                # 基准年化收益
                if years > 0:
                    metrics['benchmark_annualized'] = (1 + metrics['benchmark_return']) ** (1 / years) - 1
                    metrics['excess_annualized'] = metrics['annualized_return'] - metrics['benchmark_annualized']

                # 信息比率
                bench_returns = merged['close_bench'].pct_change().fillna(0)
                excess_returns = daily_returns - bench_returns
                tracking_error = excess_returns.std() * np.sqrt(252)

                if tracking_error > 0:
                    metrics['information_ratio'] = metrics['excess_annualized'] / tracking_error
                else:
                    metrics['information_ratio'] = 0

        except Exception as e:
            print(f"  ⚠️  基准对比计算失败: {e}")

    return metrics


def _pair_trades(trade_records):
    """
    配对买入-卖出交易

    返回: list of dict, 每个dict包含配对交易的信息
    """
    paired = []

    # 按股票分组
    for stock in trade_records['stock'].unique():
        stock_trades = trade_records[trade_records['stock'] == stock].sort_values('date')

        position = 0
        buy_price = 0
        buy_date = None
        buy_shares = 0

        for _, trade in stock_trades.iterrows():
            if trade['action'] == 'buy':
                if position == 0:
                    # 开仓
                    buy_price = trade['price']
                    buy_date = trade['date']
                    buy_shares = trade['shares']
                    position = trade['shares']
                else:
                    # 加仓（简化处理：取平均价格）
                    total_value = buy_price * position + trade['price'] * trade['shares']
                    position += trade['shares']
                    buy_price = total_value / position if position > 0 else 0

            elif trade['action'] == 'sell' and position > 0:
                # 平仓
                sell_price = trade['price']
                sell_date = trade['date']
                sell_shares = min(trade['shares'], position)

                # 计算盈利
                profit_pct = (sell_price / buy_price - 1) if buy_price > 0 else 0
                profit_amount = (sell_price - buy_price) * sell_shares

                # 持仓天数
                if buy_date and sell_date:
                    holding_days = (pd.to_datetime(sell_date) - pd.to_datetime(buy_date)).days
                else:
                    holding_days = 0

                paired.append({
                    'stock': stock,
                    'buy_date': buy_date,
                    'sell_date': sell_date,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'shares': sell_shares,
                    'profit': profit_pct,
                    'profit_amount': profit_amount,
                    'holding_days': holding_days
                })

                position -= sell_shares
                if position <= 0:
                    position = 0
                    buy_price = 0
                    buy_date = None

    return paired


def _calculate_max_dd_duration(drawdown_series):
    """计算最大回撤持续时间（天数）"""
    is_dd = drawdown_series < 0
    dd_periods = (is_dd != is_dd.shift()).cumsum()
    dd_lengths = drawdown_series.groupby(dd_periods).apply(lambda x: len(x) if (x < 0).any() else 0)
    return dd_lengths.max() if len(dd_lengths) > 0 else 0


def generate_metrics_report(metrics, output_path=None):
    """
    生成格式化的指标报告

    参数:
        metrics: 指标字典
        output_path: 输出文件路径（可选）
    """
    lines = []

    lines.append("=" * 80)
    lines.append("               完整收益指标报告")
    lines.append("=" * 80)
    lines.append("")

    # ========== 1. 基础信息 ==========
    lines.append("【基础信息】")
    lines.append("-" * 80)
    lines.append(f"  回测起始日期:     {metrics.get('start_date', 'N/A')}")
    lines.append(f"  回测结束日期:     {metrics.get('end_date', 'N/A')}")
    lines.append(f"  交易日数:         {metrics.get('trading_days', 0)} 天")
    lines.append(f"  初始资金:         ¥{metrics.get('initial_capital', 0):,.2f}")
    lines.append(f"  最终资金:         ¥{metrics.get('final_capital', 0):,.2f}")
    lines.append("")

    # ========== 2. 收益指标 ==========
    lines.append("【收益指标】")
    lines.append("-" * 80)
    lines.append(f"  总收益率:         {metrics.get('total_return', 0):.2%}")
    lines.append(f"  年化收益率:       {metrics.get('annualized_return', 0):.2%}")
    lines.append(f"  平均月度收益:     {metrics.get('avg_monthly_return', 0):.2%}")
    lines.append(f"  最佳月份:         {metrics.get('best_month', 0):.2%}")
    lines.append(f"  最差月份:         {metrics.get('worst_month', 0):.2%}")

    if 'monthly_win_rate' in metrics:
        lines.append(
            f"  月度胜率:         {metrics['monthly_win_rate']:.2%} ({metrics.get('positive_months', 0)}/{metrics.get('total_months', 0)})")

    lines.append("")

    # ========== 3. 风险指标 ==========
    lines.append("【风险指标】")
    lines.append("-" * 80)
    lines.append(f"  年化波动率:       {metrics.get('volatility', 0):.2%}")
    lines.append(f"  最大回撤:         {metrics.get('max_drawdown', 0):.2%}")
    lines.append(f"  最大回撤持续:     {metrics.get('max_drawdown_duration', 0)} 天")
    lines.append(f"  夏普比率:         {metrics.get('sharpe_ratio', 0):.3f}")
    lines.append(f"  Sortino比率:      {metrics.get('sortino_ratio', 0):.3f}")
    lines.append(f"  Calmar比率:       {metrics.get('calmar_ratio', 0):.3f}")
    lines.append("")

    # ========== 4. 交易统计 ==========
    lines.append("【交易统计】")
    lines.append("-" * 80)
    lines.append(f"  总交易次数:       {metrics.get('total_trades', 0)} 笔")
    lines.append(f"  买入次数:         {metrics.get('buy_trades', 0)} 笔")
    lines.append(f"  卖出次数:         {metrics.get('sell_trades', 0)} 笔")
    lines.append(f"  完整交易周期:     {metrics.get('completed_trades', 0)} 笔")
    lines.append("")

    # ========== 5. 胜率与盈亏 ==========
    if 'win_rate' in metrics:
        lines.append("【胜率与盈亏分析】")
        lines.append("-" * 80)
        lines.append(f"  总胜率:           {metrics['win_rate']:.2%}")
        lines.append(f"  盈利交易:         {metrics.get('winning_trades', 0)} 笔")
        lines.append(f"  亏损交易:         {metrics.get('losing_trades', 0)} 笔")
        lines.append("")
        lines.append(f"  平均盈利:         {metrics.get('avg_win', 0):.2%}")
        lines.append(f"  平均亏损:         {metrics.get('avg_loss', 0):.2%}")
        lines.append(f"  盈亏比:           {metrics.get('profit_loss_ratio', 0):.2f}")
        lines.append("")
        lines.append(f"  最大单笔盈利:     {metrics.get('max_win', 0):.2%}")
        lines.append(f"  最大单笔亏损:     {metrics.get('max_loss', 0):.2%}")
        lines.append("")
        lines.append(f"  平均持仓天数:     {metrics.get('avg_holding_days', 0):.1f} 天")
        lines.append(f"  最长持仓:         {metrics.get('max_holding_days', 0)} 天")
        lines.append(f"  最短持仓:         {metrics.get('min_holding_days', 0)} 天")
        lines.append("")

    # ========== 6. 基准对比 ==========
    if 'benchmark_return' in metrics:
        lines.append("【基准对比】")
        lines.append("-" * 80)
        lines.append(f"  基准总收益:       {metrics['benchmark_return']:.2%}")
        lines.append(f"  基准年化收益:     {metrics.get('benchmark_annualized', 0):.2%}")
        lines.append(f"  超额收益:         {metrics.get('excess_return', 0):.2%}")
        lines.append(f"  年化超额收益:     {metrics.get('excess_annualized', 0):.2%}")
        lines.append(f"  信息比率:         {metrics.get('information_ratio', 0):.3f}")
        lines.append("")

    # ========== 7. 综合评级 ==========
    lines.append("【综合评级】")
    lines.append("-" * 80)

    rating = _calculate_rating(metrics)
    lines.append(f"  策略评级:         {rating['grade']} ({rating['score']:.1f}/100)")
    lines.append(f"  评级说明:         {rating['comment']}")
    lines.append("")

    lines.append("=" * 80)
    lines.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # 输出
    report_text = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n✓ 指标报告已保存: {output_path}")

    return report_text


def _calculate_rating(metrics):
    """
    计算策略综合评级

    评分维度:
    - 收益性 (30分): 年化收益率
    - 稳定性 (25分): 夏普比率
    - 风险控制 (25分): 最大回撤
    - 胜率 (20分): 交易胜率
    """
    score = 0

    # 1. 收益性 (30分)
    ann_ret = metrics.get('annualized_return', 0)
    if ann_ret > 0.30:
        score += 30
    elif ann_ret > 0.20:
        score += 25
    elif ann_ret > 0.15:
        score += 20
    elif ann_ret > 0.10:
        score += 15
    elif ann_ret > 0.05:
        score += 10
    elif ann_ret > 0:
        score += 5

    # 2. 稳定性 (25分)
    sharpe = metrics.get('sharpe_ratio', 0)
    if sharpe > 2.0:
        score += 25
    elif sharpe > 1.5:
        score += 20
    elif sharpe > 1.0:
        score += 15
    elif sharpe > 0.5:
        score += 10
    elif sharpe > 0:
        score += 5

    # 3. 风险控制 (25分)
    max_dd = abs(metrics.get('max_drawdown', 0))
    if max_dd < 0.10:
        score += 25
    elif max_dd < 0.15:
        score += 20
    elif max_dd < 0.20:
        score += 15
    elif max_dd < 0.30:
        score += 10
    elif max_dd < 0.40:
        score += 5

    # 4. 胜率 (20分)
    win_rate = metrics.get('win_rate', 0)
    if win_rate > 0.60:
        score += 20
    elif win_rate > 0.55:
        score += 15
    elif win_rate > 0.50:
        score += 12
    elif win_rate > 0.45:
        score += 8
    elif win_rate > 0.40:
        score += 5

    # 评级
    if score >= 85:
        grade = "A+ (优秀)"
        comment = "策略表现卓越，收益、风险、胜率均衡优秀"
    elif score >= 75:
        grade = "A (良好)"
        comment = "策略表现良好，具备较强的盈利能力和风险控制"
    elif score >= 65:
        grade = "B+ (中上)"
        comment = "策略表现中上，在某些方面有优势"
    elif score >= 55:
        grade = "B (中等)"
        comment = "策略表现中等，有改进空间"
    elif score >= 45:
        grade = "C (一般)"
        comment = "策略表现一般，需要优化"
    else:
        grade = "D (较差)"
        comment = "策略表现较差，建议重新设计"

    return {
        'score': score,
        'grade': grade,
        'comment': comment
    }


def generate_detailed_trades_report(trade_records, output_path=None):
    """
    生成详细的交易明细报告

    参数:
        trade_records: 交易记录DataFrame
        output_path: 输出文件路径（可选）
    """
    if trade_records.empty:
        print("  ⚠️  没有交易记录")
        return None

    # 配对交易
    paired_trades = _pair_trades(trade_records)

    if not paired_trades:
        print("  ⚠️  没有完整的交易周期")
        return None

    # 转为DataFrame
    df = pd.DataFrame(paired_trades)

    # 排序（按收益率降序）
    df = df.sort_values('profit', ascending=False)

    lines = []

    lines.append("=" * 100)
    lines.append("                   详细交易明细报告")
    lines.append("=" * 100)
    lines.append("")

    lines.append(f"总交易周期数: {len(df)}")
    lines.append(f"盈利交易: {len(df[df['profit'] > 0])} 笔")
    lines.append(f"亏损交易: {len(df[df['profit'] < 0])} 笔")
    lines.append("")

    lines.append("-" * 100)
    lines.append(
        f"{'代码':<12} {'买入日期':<12} {'卖出日期':<12} {'买价':<8} {'卖价':<8} {'收益率':<10} {'持仓天数':<10}")
    lines.append("-" * 100)

    for _, row in df.iterrows():
        profit_str = f"{row['profit']:+.2%}"
        lines.append(
            f"{row['stock']:<12} "
            f"{str(row['buy_date']):<12} "
            f"{str(row['sell_date']):<12} "
            f"{row['buy_price']:<8.2f} "
            f"{row['sell_price']:<8.2f} "
            f"{profit_str:<10} "
            f"{row['holding_days']:<10.0f}"
        )

    lines.append("-" * 100)
    lines.append("")

    # 统计
    lines.append("【统计摘要】")
    lines.append(f"  平均收益率:   {df['profit'].mean():.2%}")
    lines.append(f"  最大收益:     {df['profit'].max():.2%}")
    lines.append(f"  最大亏损:     {df['profit'].min():.2%}")
    lines.append(f"  平均持仓:     {df['holding_days'].mean():.1f} 天")
    lines.append("")

    lines.append("=" * 100)

    report_text = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ 交易明细已保存: {output_path}")

    return report_text


# ========== 便捷函数 ==========

def generate_full_performance_report(context, benchmark_data=None, output_dir='./reports'):
    """
    生成完整的绩效报告（主函数）

    参数:
        context: 回测上下文
        benchmark_data: 基准数据（可选）
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("📊 生成完整收益指标报告")
    print("=" * 80)

    # 1. 计算指标
    print("\n[1/3] 计算绩效指标...")
    metrics = calculate_performance_metrics(context, benchmark_data)

    # 2. 生成指标报告
    print("\n[2/3] 生成指标报告...")
    metrics_path = os.path.join(output_dir, 'performance_metrics.txt')
    report_text = generate_metrics_report(metrics, output_path=metrics_path)

    # 打印到控制台
    print("\n" + report_text)

    # 3. 生成交易明细
    print("\n[3/3] 生成交易明细...")
    trade_records = context.get('trade_records', pd.DataFrame())
    if not trade_records.empty:
        trades_path = os.path.join(output_dir, 'detailed_trades.txt')
        generate_detailed_trades_report(trade_records, output_path=trades_path)

    # 4. 保存为CSV（方便Excel分析）
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['值']
    csv_path = os.path.join(output_dir, 'performance_metrics.csv')
    metrics_df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"✓ 指标CSV已保存: {csv_path}")

    print("\n" + "=" * 80)
    print("✅ 收益指标报告生成完成！")
    print("=" * 80)

    return metrics