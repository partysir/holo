"""
strategy_comparison.py - 策略对比测试

对比不同配置的策略表现:
1. 调仓周期：1日 vs 5日 vs 10日
2. 仓位方法：等权 vs 评分加权 vs 评分平方
3. 评分衰减：有 vs 无
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 配置matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from data_module import DataCache
from data_module_incremental import load_data_with_incremental_update
from enhanced_strategy import run_enhanced_strategy


def run_strategy_comparison():
    """运行策略对比测试"""
    print("\n" + "=" * 80)
    print("📊 策略对比测试系统")
    print("=" * 80)

    # 1. 加载数据
    print("\n【步骤1/3】加载数据")

    START_DATE = "2023-01-01"
    END_DATE = "2025-12-05"

    # ✨ 配置项
    USE_SAMPLING = False  # 是否使用抽样（False=使用全部股票）
    SAMPLE_SIZE = 3950    # 股票数量
    FORCE_FULL_UPDATE = True  # ✨ 强制全量更新（确保使用新的SAMPLE_SIZE）

    print(f"\n  配置:")
    print(f"    使用抽样: {'是' if USE_SAMPLING else '否'}")
    print(f"    股票数量: {SAMPLE_SIZE}")
    print(f"    强制更新: {'是' if FORCE_FULL_UPDATE else '否（使用缓存）'}")

    if not USE_SAMPLING and SAMPLE_SIZE > 2000:
        print(f"\n  ⚠️  注意: 使用{SAMPLE_SIZE}只股票，首次加载需要较长时间（约3-5分钟）")
        response = input("  是否继续？(y/n): ").lower()
        if response != 'y':
            print("  已取消")
            return

    cache_manager = DataCache(cache_dir='./data_cache')

    factor_data, price_data = load_data_with_incremental_update(
        START_DATE,
        END_DATE,
        max_stocks=SAMPLE_SIZE,  # ✨ 使用max_stocks参数
        cache_manager=cache_manager,
        use_stockranker=True,
        tushare_token="2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211",
        use_fundamental=True,
        use_sampling=USE_SAMPLING,  # ✨ 可配置
        sample_size=SAMPLE_SIZE,
        max_workers=10,
        force_full_update=FORCE_FULL_UPDATE  # ✨ 强制更新
    )

    if factor_data is None or price_data is None:
        print("❌ 数据加载失败")
        return

    print(f"✓ 数据加载完成")

    # 2. 定义策略配置
    strategies = {
        # 基准策略
        '基准-每日调仓': {
            'rebalance_days': 1,
            'position_method': 'equal',
            'score_decay_rate': 1.0,  # 不衰减
        },

        # 调仓周期对比
        '5日调仓-等权': {
            'rebalance_days': 5,
            'position_method': 'equal',
            'score_decay_rate': 1.0,
        },
        '10日调仓-等权': {
            'rebalance_days': 10,
            'position_method': 'equal',
            'score_decay_rate': 1.0,
        },

        # 仓位方法对比
        '每日调仓-评分加权': {
            'rebalance_days': 1,
            'position_method': 'score_weighted',
            'score_decay_rate': 1.0,
        },
        '每日调仓-评分平方': {
            'rebalance_days': 1,
            'position_method': 'score_squared',
            'score_decay_rate': 1.0,
        },

        # 评分衰减对比
        '每日调仓-评分衰减': {
            'rebalance_days': 1,
            'position_method': 'score_weighted',
            'score_decay_rate': 0.98,  # 每天衰减2%
        },

        # 组合策略
        '5日调仓-评分加权-衰减': {
            'rebalance_days': 5,
            'position_method': 'score_weighted',
            'score_decay_rate': 0.98,
        },
    }

    # 3. 运行所有策略
    print("\n【步骤2/3】运行策略对比")

    results = {}
    for name, config in strategies.items():
        print(f"\n{'─'*80}")
        print(f"运行策略: {name}")
        print(f"  调仓周期: {config['rebalance_days']}天")
        print(f"  仓位方法: {config['position_method']}")
        print(f"  评分衰减: {config['score_decay_rate']:.2%}/天")
        print(f"{'─'*80}")

        try:
            context = run_enhanced_strategy(
                factor_data=factor_data,
                price_data=price_data,
                start_date=START_DATE,
                end_date=END_DATE,
                capital_base=1000000,
                position_size=10,
                rebalance_days=config['rebalance_days'],
                position_method=config['position_method'],
                score_decay_rate=config['score_decay_rate'],
                buy_cost=0.0003,
                sell_cost=0.0003,
                tax_ratio=0.0005,
                stop_loss=-0.15,
                score_threshold=0.15,
                force_replace_days=45,
                silent=True
            )

            # 计算指标
            daily_records = context['daily_records']
            trade_records = context['trade_records']

            total_return = context['total_return']
            win_rate = context['win_rate']

            # 最大回撤
            cummax = daily_records['portfolio_value'].cummax()
            drawdown = (daily_records['portfolio_value'] - cummax) / cummax
            max_drawdown = drawdown.min()

            # 夏普比率
            daily_returns = daily_records['portfolio_value'].pct_change().dropna()
            daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
            daily_returns = daily_returns[np.abs(daily_returns) < 1]

            if len(daily_returns) > 1:
                volatility = daily_returns.std()
                years = len(daily_records) / 252
                annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
                sharpe = (annualized_return - 0.03) / (volatility * np.sqrt(252)) if volatility > 0 else 0
            else:
                sharpe = 0
                annualized_return = 0

            # 交易次数
            sell_trades = trade_records[trade_records['action'] == 'sell']
            trade_count = len(sell_trades)
            avg_holding_days = sell_trades['holding_days'].mean() if len(sell_trades) > 0 else 0

            results[name] = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'trade_count': trade_count,
                'avg_holding_days': avg_holding_days,
                'daily_records': daily_records
            }

            print(f"\n  ✓ 完成")
            print(f"     总收益: {total_return:+.2%}")
            print(f"     夏普: {sharpe:.4f}")
            print(f"     回撤: {max_drawdown:.2%}")
            print(f"     胜率: {win_rate:.2%}")
            print(f"     交易: {trade_count}次")

        except Exception as e:
            print(f"  ❌ 失败: {e}")
            import traceback
            traceback.print_exc()

    # 4. 生成对比报告
    print("\n【步骤3/3】生成对比报告")

    generate_comparison_report(results)
    plot_comparison_charts(results)

    print("\n" + "=" * 80)
    print("✅ 策略对比完成！")
    print("=" * 80)


def generate_comparison_report(results):
    """生成对比报告"""
    print("\n" + "=" * 80)
    print("📊 策略表现对比")
    print("=" * 80)

    # 创建对比表格
    comparison_df = pd.DataFrame({
        name: {
            '总收益率': f"{data['total_return']:+.2%}",
            '年化收益': f"{data['annualized_return']:+.2%}",
            '夏普比率': f"{data['sharpe']:.4f}",
            '最大回撤': f"{data['max_drawdown']:.2%}",
            '胜率': f"{data['win_rate']:.2%}",
            '交易次数': f"{data['trade_count']}",
            '平均持有': f"{data['avg_holding_days']:.1f}天",
        }
        for name, data in results.items()
    }).T

    print("\n" + comparison_df.to_string())

    # 找出最优策略
    print("\n" + "=" * 80)
    print("🏆 最优策略")
    print("=" * 80)

    best_return = max(results.items(), key=lambda x: x[1]['total_return'])
    print(f"\n  最高收益: {best_return[0]} ({best_return[1]['total_return']:+.2%})")

    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe'])
    print(f"  最高夏普: {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.4f})")

    best_drawdown = min(results.items(), key=lambda x: x[1]['max_drawdown'])
    print(f"  最小回撤: {best_drawdown[0]} ({best_drawdown[1]['max_drawdown']:.2%})")

    best_winrate = max(results.items(), key=lambda x: x[1]['win_rate'])
    print(f"  最高胜率: {best_winrate[0]} ({best_winrate[1]['win_rate']:.2%})")

    # 保存到文件
    import os
    os.makedirs('./reports', exist_ok=True)

    with open('./reports/strategy_comparison.txt', 'w', encoding='utf-8') as f:
        f.write("策略对比报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(comparison_df.to_string())
        f.write("\n\n最优策略:\n")
        f.write(f"  最高收益: {best_return[0]} ({best_return[1]['total_return']:+.2%})\n")
        f.write(f"  最高夏普: {best_sharpe[0]} ({best_sharpe[1]['sharpe']:.4f})\n")
        f.write(f"  最小回撤: {best_drawdown[0]} ({best_drawdown[1]['max_drawdown']:.2%})\n")
        f.write(f"  最高胜率: {best_winrate[0]} ({best_winrate[1]['win_rate']:.2%})\n")

    print("\n✓ 报告已保存: ./reports/strategy_comparison.txt")


def plot_comparison_charts(results):
    """绘制对比图表"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. 净值曲线对比
    ax1 = axes[0, 0]
    for name, data in results.items():
        records = data['daily_records']
        ax1.plot(range(len(records)), records['portfolio_value'],
                label=name, linewidth=2, alpha=0.8)
    ax1.set_title('净值曲线对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('交易日')
    ax1.set_ylabel('组合净值 (元)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. 收益率对比
    ax2 = axes[0, 1]
    names = list(results.keys())
    returns = [data['total_return'] * 100 for data in results.values()]
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in returns]
    bars = ax2.barh(range(len(names)), returns, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('总收益率 (%)')
    ax2.set_title('收益率对比', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # 3. 风险指标对比
    ax3 = axes[1, 0]
    sharpes = [data['sharpe'] for data in results.values()]
    drawdowns = [abs(data['max_drawdown']) * 100 for data in results.values()]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax3.bar(x - width/2, sharpes, width, label='夏普比率', alpha=0.7)
    bars2 = ax3.bar(x + width/2, drawdowns, width, label='最大回撤(%)', alpha=0.7)

    ax3.set_xlabel('策略')
    ax3.set_title('风险指标对比', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4. 交易特征对比
    ax4 = axes[1, 1]
    winrates = [data['win_rate'] * 100 for data in results.values()]
    trade_counts = [data['trade_count'] for data in results.values()]

    # 归一化到0-100
    max_trades = max(trade_counts) if trade_counts else 1
    normalized_trades = [t / max_trades * 100 for t in trade_counts]

    x = np.arange(len(names))
    bars1 = ax4.bar(x - width/2, winrates, width, label='胜率(%)', alpha=0.7)
    bars2 = ax4.bar(x + width/2, normalized_trades, width, label='交易频率(归一化)', alpha=0.7)

    ax4.set_xlabel('策略')
    ax4.set_title('交易特征对比', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = './reports/strategy_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ 图表已保存: {output_path}")


if __name__ == "__main__":
    try:
        run_strategy_comparison()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()