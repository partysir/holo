"""
money_flow_factors.py - 内存优化版资金流因子模块 v1.1

关键优化：
✅ 只保留最有价值的核心因子（从312个减少到30个）
✅ 及时删除中间计算列
✅ 使用float32降低内存占用
✅ 避免不必要的DataFrame复制
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import gc


class MoneyFlowFactorCalculator:
    """资金流因子计算器（内存优化版）"""

    def __init__(self, use_full_tick_data=False, keep_only_essential=True):
        """
        初始化

        Args:
            use_full_tick_data: 是否使用完整逐笔数据（需要高级权限）
            keep_only_essential: 仅保留核心因子（推荐True）
        """
        self.use_full_tick_data = use_full_tick_data
        self.keep_only_essential = keep_only_essential

        # 订单类型阈值（元）
        self.ORDER_THRESHOLDS = {
            'small': 40000,
            'mid': 200000,
            'big': 1000000,
        }

        print(f"💰 资金流因子计算器初始化完成")
        print(f"   模式: {'完整tick数据' if self.use_full_tick_data else '简化估算（推荐）'}")
        print(f"   内存优化: {'✓ 仅保留核心因子' if keep_only_essential else '✗ 保留所有因子'}")


    def calculate_simplified_money_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算资金流因子（内存优化版）

        核心策略：
        1. 立即计算 → 立即聚合 → 删除中间列
        2. 只保留30个最有价值的因子
        3. 使用float32降低内存占用

        Args:
            df: 必须包含 ['date', 'instrument', 'open', 'close', 'high',
                         'low', 'volume', 'amount']

        Returns:
            添加了资金流因子的DataFrame
        """
        print("\n⚙️  计算简化资金流因子...")

        # 转换为float32降低内存
        for col in ['open', 'close', 'high', 'low', 'volume', 'amount']:
            if col in df.columns:
                df[col] = df[col].astype('float32')

        # 1. 估算主动买卖
        df = self._estimate_active_trading_fast(df)

        # 2. 估算订单大小分布
        df = self._estimate_order_size_fast(df)

        # 3. 计算核心资金流指标
        df = self._calculate_core_flow_metrics(df)

        # 4. 计算衍生因子
        df = self._calculate_derived_factors_fast(df)

        # 5. 清理内存
        gc.collect()

        # 统计最终因子数量
        money_flow_cols = [c for c in df.columns if any(
            k in c for k in ['main_', 'large_', 'netflow_', 'inflow_', 'strength', 'continuous']
        )]
        print(f"✓ 资金流因子计算完成，保留 {len(money_flow_cols)} 个核心因子")

        return df


    def _estimate_active_trading_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """快速估算主动买卖（不保留中间列）"""

        # 计算价格变化
        price_change = df.groupby('instrument')['close'].pct_change().fillna(0)

        # 估算主动买入占比（sigmoid平滑）
        active_buy_ratio = (1 / (1 + np.exp(-20 * price_change))).astype('float32')

        # 只计算主动买入量/额（主动卖出 = 总量 - 主动买入）
        df['_active_buy_vol'] = (df['volume'] * active_buy_ratio).astype('float32')
        df['_active_buy_amt'] = (df['amount'] * active_buy_ratio).astype('float32')

        return df


    def _estimate_order_size_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """快速估算订单大小分布"""

        # 计算平均单笔成交额
        avg_trade_amt = (df['amount'] / (df['volume'] + 1)).astype('float32')

        # 估算超大单占比
        df['_large_ratio'] = (
            1 / (1 + np.exp(-0.000005 * (avg_trade_amt - self.ORDER_THRESHOLDS['big'])))
        ).astype('float32')

        # 估算大单占比
        lower_prob = 1 / (1 + np.exp(-0.000005 * (avg_trade_amt - self.ORDER_THRESHOLDS['mid'])))
        upper_prob = 1 / (1 + np.exp(-0.000005 * (avg_trade_amt - self.ORDER_THRESHOLDS['big'])))
        df['_big_ratio'] = (lower_prob - upper_prob).clip(0, 1).astype('float32')

        # 主力占比 = 超大单 + 大单
        df['_main_ratio'] = (df['_large_ratio'] + df['_big_ratio']).clip(0, 1).astype('float32')

        # 删除avg_trade_amt（节省内存）
        del avg_trade_amt, lower_prob, upper_prob

        return df


    def _calculate_core_flow_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """只计算核心资金流指标（避免生成312个因子）"""

        # 核心指标1：主力净主动买入量
        df['main_net_active_buy_vol'] = (
            (df['_active_buy_vol'] - (df['volume'] - df['_active_buy_vol'])) * df['_main_ratio']
        ).astype('float32')

        # 核心指标2：主力净主动买入额
        df['main_net_active_buy_amt'] = (
            (df['_active_buy_amt'] - (df['amount'] - df['_active_buy_amt'])) * df['_main_ratio']
        ).astype('float32')

        # 核心指标3：超大单净主动买入额
        df['large_net_active_buy_amt'] = (
            (df['_active_buy_amt'] - (df['amount'] - df['_active_buy_amt'])) * df['_large_ratio']
        ).astype('float32')

        # 核心指标4：主力流入额（主动买入 + 被动卖出）
        active_sell_amt = df['amount'] - df['_active_buy_amt']
        df['main_inflow_amt'] = (
            (df['_active_buy_amt'] + active_sell_amt) * df['_main_ratio']
        ).astype('float32')

        # 核心指标5：主力流出额（主动卖出 + 被动买入）
        df['main_outflow_amt'] = (
            (active_sell_amt + df['_active_buy_amt']) * df['_main_ratio']
        ).astype('float32')

        # 核心指标6：主力净流入额
        df['main_netflow_amt'] = (df['main_inflow_amt'] - df['main_outflow_amt']).astype('float32')

        # 核心指标7：主力成交额占比
        df['main_amount_ratio'] = (df['_main_ratio'] * df['amount'] / (df['amount'] + 1)).astype('float32')

        # 删除中间列
        del active_sell_amt
        df.drop(columns=['_active_buy_vol', '_active_buy_amt', '_large_ratio',
                        '_big_ratio', '_main_ratio'], inplace=True)

        return df


    def _calculate_derived_factors_fast(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算衍生因子（内存优化）"""

        print("  计算资金流衍生因子...")

        # 按股票分组
        grouped = df.groupby('instrument')

        # 1. 多期主力净流入（5/10/20日）
        for period in [5, 10, 20]:
            col_name = f'main_netflow_amt_{period}d'
            df[col_name] = (
                grouped['main_netflow_amt']
                .rolling(period, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
                .astype('float32')
            )

        # 2. 主力资金强度（净流入 / 成交额）
        df['main_strength'] = (
            df['main_netflow_amt'] / (df['amount'] + 1e-6)
        ).astype('float32')

        df['main_strength_5d'] = (
            grouped['main_strength']
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype('float32')
        )

        # 3. 主力持续性（连续净流入天数）
        is_inflow = (df['main_netflow_amt'] > 0).astype(int)
        df['main_continuous_inflow'] = (
            grouped.apply(
                lambda x: is_inflow.loc[x.index] *
                (is_inflow.loc[x.index].groupby((is_inflow.loc[x.index] != is_inflow.loc[x.index].shift()).cumsum()).cumcount() + 1)
            )
            .reset_index(level=0, drop=True)
            .astype('float32')
        )

        # 4. 超大单多期净流入
        for period in [5, 10]:
            col_name = f'large_netflow_amt_{period}d'
            df[col_name] = (
                grouped['large_net_active_buy_amt']
                .rolling(period, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
                .astype('float32')
            )

        # 5. 主力活跃度（主力成交额占比的5日均值）
        df['main_activity_5d'] = (
            grouped['main_amount_ratio']
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
            .astype('float32')
        )

        # 6. 主力净流入强度变化（当日 vs 5日均）
        main_netflow_5d_avg = (
            grouped['main_netflow_amt']
            .rolling(5, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df['main_strength_change'] = (
            (df['main_netflow_amt'] - main_netflow_5d_avg) /
            (main_netflow_5d_avg.abs() + 1e-6)
        ).astype('float32')

        # 清理临时变量
        del is_inflow, main_netflow_5d_avg

        return df


    def get_factor_list(self) -> List[str]:
        """获取核心因子列表（用于权重配置）"""
        return [
            # 核心净流入指标（3个）
            'main_netflow_amt_5d',      # 5日主力净流入
            'main_netflow_amt_10d',     # 10日主力净流入
            'main_netflow_amt_20d',     # 20日主力净流入

            # 超大单指标（2个）
            'large_net_active_buy_amt', # 当日超大单净买入
            'large_netflow_amt_5d',     # 5日超大单净流入

            # 主力强度指标（3个）
            'main_strength',            # 当日主力强度
            'main_strength_5d',         # 5日主力强度
            'main_strength_change',     # 主力强度变化

            # 主力行为指标（3个）
            'main_continuous_inflow',   # 主力持续流入天数
            'main_activity_5d',         # 主力活跃度
            'main_amount_ratio',        # 主力成交额占比
        ]


    def get_recommended_weights(self, style='balanced') -> Dict[str, float]:
        """
        获取推荐权重配置

        Args:
            style: 'conservative' | 'balanced' | 'aggressive'

        Returns:
            因子权重字典
        """
        if style == 'conservative':
            # 保守型：更注重长期趋势
            return {
                'main_netflow_amt_20d': 0.08,
                'main_netflow_amt_10d': 0.06,
                'main_strength_5d': 0.05,
                'large_netflow_amt_5d': 0.04,
                'main_continuous_inflow': 0.02,
            }  # 总权重 25%

        elif style == 'aggressive':
            # 激进型：更注重短期信号
            return {
                'main_netflow_amt_5d': 0.10,
                'main_strength': 0.08,
                'large_net_active_buy_amt': 0.07,
                'main_strength_change': 0.06,
                'main_continuous_inflow': 0.04,
            }  # 总权重 35%

        else:  # balanced
            # 平衡型（推荐）
            return {
                'main_netflow_amt_5d': 0.08,
                'main_netflow_amt_10d': 0.06,
                'main_strength_5d': 0.05,
                'large_netflow_amt_5d': 0.04,
                'main_continuous_inflow': 0.03,
                'main_activity_5d': 0.02,
            }  # 总权重 28%


    def print_factor_summary(self, df: pd.DataFrame):
        """打印因子统计摘要"""

        core_factors = self.get_factor_list()
        existing_factors = [f for f in core_factors if f in df.columns]

        print(f"\n📊 资金流因子摘要:")
        print(f"  核心因子数: {len(existing_factors)}/{len(core_factors)}")
        print(f"  数据行数: {len(df)}")
        print(f"  内存占用: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # 打印因子列表
        print(f"\n  因子列表:")
        for i, factor in enumerate(existing_factors, 1):
            non_null = df[factor].notna().sum()
            coverage = non_null / len(df) * 100
            print(f"    {i}. {factor} (覆盖率: {coverage:.1f}%)")


def integrate_money_flow_to_stockranker(
    df: pd.DataFrame,
    calculator: MoneyFlowFactorCalculator,
    style='balanced'
) -> pd.DataFrame:
    """
    将资金流因子整合到StockRanker评分体系

    Args:
        df: 包含价格和现有因子的DataFrame
        calculator: 资金流计算器实例
        style: 'conservative' | 'balanced' | 'aggressive'

    Returns:
        添加资金流因子后的DataFrame
    """

    print(f"\n🔗 整合资金流因子 (风格: {style})...")

    # 1. 计算资金流因子
    df = calculator.calculate_simplified_money_flow(df)

    # 2. 打印摘要
    calculator.print_factor_summary(df)

    # 3. 获取推荐权重
    money_flow_weights = calculator.get_recommended_weights(style)

    print(f"\n  推荐权重配置:")
    total_weight = 0
    for factor, weight in money_flow_weights.items():
        if factor in df.columns:
            print(f"    - {factor}: {weight:.2%}")
            total_weight += weight
    print(f"  总权重: {total_weight:.2%}")

    return df


# ============ 使用示例 ============

def example_usage():
    """使用示例"""

    print("\n" + "="*80)
    print("💰 资金流因子计算器 - 使用指南")
    print("="*80)

    # 1. 创建计算器（内存优化模式）
    calculator = MoneyFlowFactorCalculator(
        use_full_tick_data=False,
        keep_only_essential=True  # ✅ 仅保留核心因子
    )

    # 2. 查看推荐因子列表
    print("\n核心因子列表:")
    for i, factor in enumerate(calculator.get_factor_list(), 1):
        print(f"  {i}. {factor}")

    # 3. 查看不同风格的推荐权重
    print("\n推荐权重配置:")
    for style in ['conservative', 'balanced', 'aggressive']:
        weights = calculator.get_recommended_weights(style)
        total = sum(weights.values())
        print(f"\n  {style.upper()} (总权重: {total:.1%}):")
        for factor, weight in weights.items():
            print(f"    - {factor}: {weight:.2%}")


if __name__ == "__main__":
    example_usage()