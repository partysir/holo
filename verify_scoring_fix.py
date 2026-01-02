"""
verify_scoring_fix.py - 验证评分修复效果

用途:
1. 对比修复前后的评分差异
2. 验证评分融合正确性
3. 检查回测和实盘一致性
"""

import pandas as pd
import numpy as np


def verify_score_columns(factor_data, mode='backtest'):
    """验证评分列的存在性"""
    print(f"\n{'=' * 70}")
    print(f"🔍 验证{mode}模式的评分列")
    print('=' * 70)

    required_cols = ['date', 'instrument', 'position']
    optional_cols = ['stockranker_score', 'ml_score']

    # 检查必需列
    missing_required = [c for c in required_cols if c not in factor_data.columns]
    if missing_required:
        print(f"❌ 缺少必需列: {missing_required}")
        return False
    else:
        print(f"✅ 必需列存在: {required_cols}")

    # 检查可选列
    existing_optional = [c for c in optional_cols if c in factor_data.columns]
    if existing_optional:
        print(f"✅ 评分明细列: {existing_optional}")
    else:
        print(f"⚠️  未找到评分明细列（可能是纯模式）")

    # 检查position的合理性
    pos_min = factor_data['position'].min()
    pos_max = factor_data['position'].max()

    if pos_min < 0 or pos_max > 1:
        print(f"❌ position超出范围: [{pos_min}, {pos_max}]")
        return False
    else:
        print(f"✅ position范围正常: [{pos_min:.4f}, {pos_max:.4f}]")

    return True


def verify_score_consistency(factor_data):
    """验证评分一致性"""
    print(f"\n{'=' * 70}")
    print("🔍 验证评分一致性")
    print('=' * 70)

    if 'stockranker_score' not in factor_data.columns or 'ml_score' not in factor_data.columns:
        print("⚠️  缺少评分明细列，跳过一致性检查")
        return True

    # 随机选择一天
    sample_date = factor_data['date'].sample(1).iloc[0]
    sample_data = factor_data[factor_data['date'] == sample_date].copy()

    print(f"抽样日期: {sample_date}")
    print(f"股票数量: {len(sample_data)}")

    # 检查评分相关性
    corr_sr_ml = sample_data[['stockranker_score', 'ml_score']].corr().iloc[0, 1]
    corr_sr_pos = sample_data[['stockranker_score', 'position']].corr().iloc[0, 1]
    corr_ml_pos = sample_data[['ml_score', 'position']].corr().iloc[0, 1]

    print(f"\n相关性分析:")
    print(f"  stockranker_score vs ml_score:  {corr_sr_ml:.4f}")
    print(f"  stockranker_score vs position:  {corr_sr_pos:.4f}")
    print(f"  ml_score vs position:           {corr_ml_pos:.4f}")

    # 判断
    if abs(corr_sr_pos - 1.0) < 0.01:
        print("\n⚠️  警告: position几乎完全等于stockranker_score (可能未融合)")
    elif abs(corr_ml_pos - 1.0) < 0.01:
        print("\n⚠️  警告: position几乎完全等于ml_score (可能未融合)")
    else:
        print("\n✅ 评分融合正常")

    # 打印Top 5对比
    print(f"\nTop 5股票评分对比:")
    print(f"{'排名':<4} {'代码':<12} {'position':<10} {'SR评分':<10} {'ML评分':<10}")
    print("-" * 60)

    top_5 = sample_data.nlargest(5, 'position')
    for i, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i:<4} {row['instrument']:<12} {row['position']:<10.4f} "
              f"{row['stockranker_score']:<10.4f} {row['ml_score']:<10.4f}")

    return True


def verify_backtest_live_consistency(backtest_data, live_data):
    """验证回测和实盘的一致性"""
    print(f"\n{'=' * 70}")
    print("🔍 验证回测-实盘一致性")
    print('=' * 70)

    # 找到共同日期
    common_dates = set(backtest_data['date']) & set(live_data['date'])

    if not common_dates:
        print("⚠️  没有共同日期，无法对比")
        return False

    sample_date = sorted(common_dates)[-1]  # 最新日期
    print(f"对比日期: {sample_date}")

    bt_data = backtest_data[backtest_data['date'] == sample_date].sort_values('position', ascending=False).head(10)
    live_data_filtered = live_data[live_data['date'] == sample_date].sort_values('position', ascending=False).head(10)

    bt_stocks = set(bt_data['instrument'])
    live_stocks = set(live_data_filtered['instrument'])

    overlap = bt_stocks & live_stocks
    overlap_rate = len(overlap) / 10

    print(f"\nTop 10股票重合度:")
    print(f"  回测选中: {bt_stocks}")
    print(f"  实盘选中: {live_stocks}")
    print(f"  重合数量: {len(overlap)}/10")
    print(f"  重合率: {overlap_rate:.0%}")

    if overlap_rate >= 0.8:
        print("\n✅ 回测-实盘一致性良好")
        return True
    elif overlap_rate >= 0.5:
        print("\n⚠️  回测-实盘一致性一般")
        return True
    else:
        print("\n❌ 回测-实盘不一致！")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("🔧 评分修复验证工具")
    print("=" * 70)

    # 提示用户
    print("\n请确保已运行:")
    print("  1. main_fixed.py (回测)")
    print("  2. main_live_trading_fixed.py (实盘)")

    # 尝试加载回测数据
    try:
        import glob
        backtest_files = sorted(glob.glob('./reports/*/score_comparison.csv'))
        if backtest_files:
            latest_backtest = backtest_files[-1]
            backtest_data = pd.read_csv(latest_backtest)
            print(f"\n✅ 已加载回测数据: {latest_backtest}")
            verify_score_columns(backtest_data, mode='backtest')
            verify_score_consistency(backtest_data)
        else:
            print("\n⚠️  未找到回测数据")
            backtest_data = None
    except Exception as e:
        print(f"\n❌ 加载回测数据失败: {e}")
        backtest_data = None

    # 尝试加载实盘数据
    try:
        import glob
        live_files = sorted(glob.glob('./live_trading/signals_*.csv'))
        if live_files:
            latest_live = live_files[-1]
            live_data = pd.read_csv(latest_live)
            print(f"\n✅ 已加载实盘数据: {latest_live}")
            verify_score_columns(live_data, mode='live')
        else:
            print("\n⚠️  未找到实盘数据")
            live_data = None
    except Exception as e:
        print(f"\n❌ 加载实盘数据失败: {e}")
        live_data = None

    # 对比回测和实盘
    if backtest_data is not None and live_data is not None:
        verify_backtest_live_consistency(backtest_data, live_data)

    print("\n" + "=" * 70)
    print("✅ 验证完成")
    print("=" * 70)


if __name__ == "__main__":
    main()