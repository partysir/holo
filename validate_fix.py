"""
validate_fix.py - 修复效果验证脚本
用于验证策略修复后的关键指标是否改善

版本：v3.0
日期：2025-12-29
"""

import pandas as pd
import numpy as np
from datetime import datetime


class FixValidator:
    """修复效果验证器"""

    def __init__(self, context):
        """
        Parameters:
        -----------
        context : dict
            回测引擎返回的上下文
        """
        self.context = context
        self.issues = []
        self.warnings = []
        self.success = []

    def validate_all(self):
        """执行所有验证检查"""
        print("\n" + "=" * 80)
        print("🔍 修复效果验证报告")
        print("=" * 80)

        self.check_trading_cost()
        self.check_holding_days()
        self.check_rebalance_frequency()
        self.check_risk_metrics()
        self.check_context_fields()

        # 打印结果
        print("\n" + "-" * 80)
        print(f"✅ 通过检查: {len(self.success)} 项")
        for msg in self.success:
            print(f"  ✓ {msg}")

        if self.warnings:
            print(f"\n⚠️  警告: {len(self.warnings)} 项")
            for msg in self.warnings:
                print(f"  ⚠  {msg}")

        if self.issues:
            print(f"\n❌ 问题: {len(self.issues)} 项")
            for msg in self.issues:
                print(f"  ✗ {msg}")
        else:
            print("\n🎉 所有关键问题已修复！")

        print("=" * 80)

        return len(self.issues) == 0

    def check_trading_cost(self):
        """检查1: 交易成本是否正确统计"""
        total_cost = self.context.get('total_cost', 0)
        df_trades = self.context.get('trade_records', pd.DataFrame())

        if total_cost > 0:
            self.success.append(f"交易成本已正确统计: ¥{total_cost:,.2f}")

            # 验证合理性
            if not df_trades.empty:
                total_amount = df_trades['amount'].sum()
                cost_ratio = total_cost / total_amount if total_amount > 0 else 0

                if 0.0003 <= cost_ratio <= 0.002:  # 万3 到 千2 之间
                    self.success.append(f"交易成本率合理: {cost_ratio:.4%}")
                else:
                    self.warnings.append(f"交易成本率异常: {cost_ratio:.4%}")
        else:
            if df_trades.empty:
                self.warnings.append("无交易记录，交易成本为0属正常")
            else:
                self.issues.append("有交易但成本为0，可能未正确计算")

    def check_holding_days(self):
        """检查2: 平均持仓天数是否合理"""
        avg_days = self.context.get('avg_holding_days', 0)
        df_trades = self.context.get('trade_records', pd.DataFrame())

        if avg_days > 0:
            if avg_days >= 5:
                self.success.append(f"平均持仓天数合理: {avg_days:.1f} 天")
            elif avg_days >= 3:
                self.warnings.append(f"平均持仓天数较短: {avg_days:.1f} 天，建议延长调仓周期")
            else:
                self.issues.append(f"平均持仓天数过短: {avg_days:.1f} 天，策略过于频繁交易")
        else:
            if not df_trades.empty:
                self.issues.append("有交易但持仓天数为0，可能未正确计算")

    def check_rebalance_frequency(self):
        """检查3: 调仓频率是否合理"""
        df_trades = self.context.get('trade_records', pd.DataFrame())
        df_daily = self.context.get('daily_records', pd.DataFrame())

        if not df_trades.empty and not df_daily.empty:
            trading_days = len(df_daily)
            trade_count = len(df_trades)
            trades_per_day = trade_count / trading_days

            if trades_per_day <= 0.5:  # 平均每2天不到1笔交易
                self.success.append(f"交易频率合理: 平均每天 {trades_per_day:.2f} 笔")
            elif trades_per_day <= 1.0:
                self.warnings.append(f"交易频率较高: 平均每天 {trades_per_day:.2f} 笔")
            else:
                self.issues.append(f"交易频率过高: 平均每天 {trades_per_day:.2f} 笔")

    def check_risk_metrics(self):
        """检查4: 风险指标是否改善"""
        total_return = self.context.get('total_return', 0)
        win_rate = self.context.get('win_rate', 0)

        # 收益率检查
        if total_return > 0:
            self.success.append(f"总收益为正: {total_return:.2%}")
        elif total_return > -0.20:
            self.warnings.append(f"总收益为负但在可接受范围: {total_return:.2%}")
        else:
            self.issues.append(f"总收益严重为负: {total_return:.2%}")

        # 胜率检查
        if win_rate >= 0.45:
            self.success.append(f"胜率良好: {win_rate:.2%}")
        elif win_rate >= 0.35:
            self.warnings.append(f"胜率一般: {win_rate:.2%}")
        else:
            self.issues.append(f"胜率过低: {win_rate:.2%}")

        # 计算最大回撤
        df_daily = self.context.get('daily_records', pd.DataFrame())
        if not df_daily.empty and 'return' in df_daily.columns:
            cumulative = (1 + df_daily['return']).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            if max_drawdown > -0.25:
                self.success.append(f"最大回撤可控: {max_drawdown:.2%}")
            elif max_drawdown > -0.40:
                self.warnings.append(f"最大回撤较大: {max_drawdown:.2%}")
            else:
                self.issues.append(f"最大回撤过大: {max_drawdown:.2%}")

    def check_context_fields(self):
        """检查5: 必需字段是否完整"""
        required_fields = {
            'daily_records': '每日记录',
            'trade_records': '交易记录',
            'total_cost': '交易成本',
            'avg_holding_days': '平均持仓天数',
            'final_value': '最终市值',
            'total_return': '总收益率',
            'win_rate': '胜率',
            'positions': '当前持仓'
        }

        missing = []
        for field, desc in required_fields.items():
            if field not in self.context:
                missing.append(f"{desc}({field})")

        if not missing:
            self.success.append("所有必需字段完整")
        else:
            self.issues.append(f"缺少字段: {', '.join(missing)}")


def quick_validate(context):
    """快速验证接口"""
    validator = FixValidator(context)
    return validator.validate_all()


def compare_before_after(context_before, context_after):
    """
    对比修复前后的效果

    Parameters:
    -----------
    context_before : dict
        修复前的回测结果
    context_after : dict
        修复后的回测结果
    """
    print("\n" + "=" * 80)
    print("📊 修复前后对比")
    print("=" * 80)

    metrics = {
        '总收益率': ('total_return', lambda x: f"{x:.2%}"),
        '胜率': ('win_rate', lambda x: f"{x:.2%}"),
        '平均持仓天数': ('avg_holding_days', lambda x: f"{x:.1f}天"),
        '交易成本': ('total_cost', lambda x: f"¥{x:,.2f}"),
    }

    print(f"\n{'指标':<20} | {'修复前':<20} | {'修复后':<20} | {'变化'}")
    print("-" * 80)

    for metric_name, (field, formatter) in metrics.items():
        before_val = context_before.get(field, 0)
        after_val = context_after.get(field, 0)

        before_str = formatter(before_val) if before_val else "N/A"
        after_str = formatter(after_val) if after_val else "N/A"

        # 计算变化
        if before_val and after_val and before_val != 0:
            change = (after_val - before_val) / abs(before_val)
            change_str = f"{change:+.1%}"
            if change > 0:
                change_str = "✅ " + change_str
            elif change < -0.05:
                change_str = "⚠️  " + change_str
        else:
            change_str = "-"

        print(f"{metric_name:<20} | {before_str:<20} | {after_str:<20} | {change_str}")

    print("=" * 80)


# ========== 使用示例 ==========
if __name__ == "__main__":
    print("""
    修复验证脚本使用方法：

    # 方法1: 快速验证（在 main.py 中）
    from validate_fix import quick_validate

    context = run_factor_based_strategy_v2(...)
    quick_validate(context)

    # 方法2: 对比修复前后
    from validate_fix import compare_before_after

    context_before = {...}  # 修复前的结果
    context_after = run_factor_based_strategy_v2(...)  # 修复后的结果
    compare_before_after(context_before, context_after)

    验证项目：
    ✓ 交易成本是否正确统计
    ✓ 平均持仓天数是否合理（≥5天）
    ✓ 调仓频率是否降低
    ✓ 风险指标是否改善
    ✓ 必需字段是否完整
    """)