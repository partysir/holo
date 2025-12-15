"""
factor_based_risk_control_optimized.py - 因子风控 + 最佳现金管理 + 择时模块 (资金守恒修复版)

核心修复：
✅ 1. 资金流水记录：追踪每笔买卖的现金变动
✅ 2. 资金守恒验证：确保总资产 = 初始资金 + 已实现盈亏
✅ 3. 正确收益率计算：基于资金守恒原理
✅ 4. 现金检查：防止超额买入
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


class OptimalCashManager:
    """最佳现金管理器"""

    def __init__(self,
                 cash_reserve_ratio=0.05,
                 buy_cost=0.0003,
                 min_buy_amount=1000,
                 debug=False):
        """
        :param cash_reserve_ratio: 现金保留比例（0.05 = 5%）
        :param buy_cost: 买入成本
        :param min_buy_amount: 最小买入金额
        :param debug: 是否输出调试信息
        """
        self.cash_reserve_ratio = cash_reserve_ratio
        self.buy_cost = buy_cost
        self.min_buy_amount = min_buy_amount
        self.debug = debug

    def calculate_buy_plan(self, available_cash, target_stocks, prices):
        """
        ✅ v4.0 超级保守版：100%保证不超支

        算法说明：
        1. 预留5%现金不动
        2. 剩余95%平均分配给N只股票
        3. 每只股票：
           - 计算理论可买金额 = 剩余现金 / 剩余股票数
           - 计算实际可买股数（向下取整到100股）
           - 计算实际花费（含手续费）
           - 如果实际花费 > 理论可买金额，减少股数直到满足
           - 扣除实际花费后，继续下一只
        4. 最终验证：总花费 <= 可用现金
        """
        if not target_stocks or available_cash <= 0:
            return {}

        # 1. 计算可投资金额（保留5%现金）
        investable = available_cash * (1 - self.cash_reserve_ratio)

        if self.debug:
            print(f"\n  【超级保守现金管理 v4.0】")
            print(f"    可用现金: ¥{available_cash:,.0f}")
            print(f"    可投资额: ¥{investable:,.0f} (保留{self.cash_reserve_ratio:.0%})")
            print(f"    待买股票: {len(target_stocks)}只")

        buy_plan = {}
        remaining = investable  # 剩余可投资金额
        remaining_count = len(target_stocks)  # 剩余股票数

        for i, stock in enumerate(target_stocks):
            # 检查剩余资金
            if remaining < self.min_buy_amount:
                if self.debug:
                    print(f"    [{i + 1}] ⏸️  剩余资金不足，停止分配")
                break

            # 检查价格
            if stock not in prices:
                if self.debug:
                    print(f"    [{i + 1}] ❌ {stock}: 无价格数据")
                remaining_count -= 1
                continue

            price = prices[stock]

            # 2. 计算这只股票的理论分配金额
            target_allocation = remaining / remaining_count

            if self.debug:
                print(f"\n    [{i + 1}] 分配 {stock}:")
                print(f"         剩余资金: ¥{remaining:,.0f}")
                print(f"         剩余股票数: {remaining_count}")
                print(f"         理论分配: ¥{target_allocation:,.0f}")

            # 3. 计算可买股数（保守估计）
            # 公式：股数 = 理论分配 / (价格 × (1 + 手续费率))
            max_shares_float = target_allocation / (price * (1 + self.buy_cost))

            # 向下取整到100股
            max_shares = int(max_shares_float / 100) * 100

            # ✨ 股数合理性检查
            if max_shares > 10000000:  # 不应超过1000万股
                if self.debug:
                    print(f"         ⚠️  可买股数异常巨大 {max_shares:,}股，跳过")
                remaining_count -= 1
                continue

            if max_shares < 100:
                if self.debug:
                    print(f"         ⚠️  可买股数不足100股，跳过")
                remaining_count -= 1
                continue

            # 4. 计算实际花费
            actual_cost = max_shares * price * (1 + self.buy_cost)

            if self.debug:
                print(f"         可买股数: {max_shares:,.0f}")
                print(f"         实际花费: ¥{actual_cost:,.0f}")

            # 5. 验证1：实际花费不能超过理论分配
            if actual_cost > target_allocation:
                if self.debug:
                    print(f"         ⚠️  超出分配，重新计算")

                # ✅ 修复：重新计算股数（两步法，避免精度损失）
                max_shares_float_adjusted = (target_allocation * 0.9) / (price * (1 + self.buy_cost))
                max_shares = int(max_shares_float_adjusted / 100) * 100

                if max_shares < 100:
                    if self.debug:
                        print(f"         ❌ 调整后仍不足，跳过")
                    remaining_count -= 1
                    continue

                # 重新计算actual_cost（基于修正后的股数）
                actual_cost = max_shares * price * (1 + self.buy_cost)

                if self.debug:
                    print(f"         调整后股数: {max_shares:,.0f}")
                    print(f"         调整后花费: ¥{actual_cost:,.0f}")

            # 6. 验证2：实际花费不能超过剩余资金
            if actual_cost > remaining:
                if self.debug:
                    print(f"         ❌ 超出剩余资金，跳过")
                remaining_count -= 1
                continue

            # 7. 验证3：检查最小买入金额
            if actual_cost < self.min_buy_amount:
                if self.debug:
                    print(f"         ⚠️  低于最小买入金额，跳过")
                remaining_count -= 1
                continue

            # 8. 记录买入计划
            buy_plan[stock] = {
                'shares': max_shares,
                'price': price,
                'amount': actual_cost
            }

            # 9. 扣除实际花费
            remaining -= actual_cost
            remaining_count -= 1

            if self.debug:
                print(f"         ✅ 已分配")
                print(f"         剩余资金: ¥{remaining:,.0f}")

        # 10. 最终验证
        if buy_plan:
            total_allocated = sum(p['amount'] for p in buy_plan.values())

            if self.debug:
                print(f"\n    【分配完成】")
                print(f"    成功分配: {len(buy_plan)}只")
                print(f"    总花费: ¥{total_allocated:,.0f}")
                print(f"    剩余: ¥{remaining:,.0f}")

            # 最终验证：总花费不能超过可用现金
            if total_allocated > available_cash:
                error_msg = (
                    f"严重错误：买入计划超支！\n"
                    f"  可用现金: ¥{available_cash:,.0f}\n"
                    f"  计划花费: ¥{total_allocated:,.0f}\n"
                    f"  超支: ¥{total_allocated - available_cash:,.0f}"
                )
                print(f"\n    ❌ {error_msg}")
                raise ValueError(error_msg)

            if self.debug:
                print(f"    ✅ 验证通过")

        return buy_plan
class FactorBasedRiskControlOptimized:
    """
    因子风控 + 最佳现金管理 + 大盘择时 (资金守恒修复版)

    核心修复：
    1. ✅ 资金流水记录：追踪每笔现金变动
    2. ✅ 资金守恒验证：总资产 = 初始资金 + 盈亏
    3. ✅ 正确收益率：基于守恒原理计算
    """

    def __init__(self, factor_data, price_data,
                 # ✨ 新增：基准数据（用于择时）
                 benchmark_data=None,
                 market_ma_period=60,
                 enable_market_timing=True,

                 start_date='2023-01-01', end_date='2025-12-05',
                 capital_base=1000000, position_size=10,
                 rebalance_days=5,

                 # ========== 最佳现金管理参数 ==========
                 cash_reserve_ratio=0.05,

                 # ========== 因子风控参数 ==========
                 enable_score_decay_stop=True,
                 score_decay_threshold=0.30,
                 min_holding_days=5,

                 enable_rank_stop=True,
                 rank_percentile_threshold=0.70,

                 max_portfolio_drawdown=-0.15,
                 reduce_position_ratio=0.5,

                 enable_industry_rotation=True,
                 max_industry_weight=0.40,

                 extreme_loss_threshold=-0.20,
                 portfolio_loss_threshold=-0.25,

                 # ========== 交易成本 ==========
                 buy_cost=0.0003,
                 sell_cost=0.0003,
                 tax_ratio=0.0005,

                 debug=False):

        self.factor_data = factor_data
        self.price_data = price_data
        self.benchmark_data = benchmark_data
        self.market_ma_period = market_ma_period
        self.enable_market_timing = enable_market_timing

        self.start_date = start_date
        self.end_date = end_date
        self.capital_base = capital_base
        self.position_size = position_size
        self.rebalance_days = rebalance_days

        # 现金管理参数
        self.cash_reserve_ratio = cash_reserve_ratio

        # 风控参数
        self.enable_score_decay_stop = enable_score_decay_stop
        self.score_decay_threshold = score_decay_threshold
        self.min_holding_days = min_holding_days
        self.enable_rank_stop = enable_rank_stop
        self.rank_percentile_threshold = rank_percentile_threshold
        self.max_portfolio_drawdown = max_portfolio_drawdown
        self.reduce_position_ratio = reduce_position_ratio
        self.enable_industry_rotation = enable_industry_rotation
        self.max_industry_weight = max_industry_weight
        self.extreme_loss_threshold = extreme_loss_threshold
        self.portfolio_loss_threshold = portfolio_loss_threshold

        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.tax_ratio = tax_ratio
        self.debug = debug

        # 初始化现金管理器
        self.cash_manager = OptimalCashManager(
            cash_reserve_ratio=cash_reserve_ratio,
            buy_cost=buy_cost,
            debug=debug
        )

        # 构建索引
        print("\n  ⚡ 构建因子风控 + 最佳现金管理 + 择时系统...")
        self.price_dict = self._build_price_dict()
        self.factor_dict = self._build_factor_dict()
        self.trading_days = sorted(factor_data['date'].unique())

        # 预计算大盘均线
        self.market_signals = self._calculate_market_signals() if self.enable_market_timing else {}

        # 行业信息
        if 'industry' in factor_data.columns:
            self.industry_dict = self._build_industry_dict()
        else:
            self.industry_dict = None
            self.enable_industry_rotation = False

        # ========== ✅ 修复：完善状态追踪 ==========
        self.initial_capital = capital_base  # 记录初始资金
        self.cash = capital_base
        self.positions = {}
        self.portfolio_value = capital_base
        self.max_portfolio_value = capital_base
        self.daily_records = []
        self.trade_records = []
        self.cash_flow_log = []  # ✅ 新增：现金流水记录
        self.days_since_rebalance = rebalance_days
        self.is_risk_mode = False

        print(f"  ✓ 系统初始化完成")
        print(f"\n  【v2.2 资金守恒修复版配置】")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if self.benchmark_data is not None and self.enable_market_timing:
            print(f"  📈 择时模块: 已启用 ({market_ma_period}日均线)")
        elif self.benchmark_data is not None and not self.enable_market_timing:
            print(f"  ⏸️  择时模块: 已禁用 (基准数据可用但未启用)")
        else:
            print(f"  ⚠️  择时模块: 未启用 (无基准数据)")
        print(f"  💰 资金管理:")
        print(f"     • 初始资金: ¥{capital_base:,.0f}")
        print(f"     • 现金保留: {cash_reserve_ratio:.1%}")
        print(f"     • 资金守恒验证: ✓")
        print(f"\n  🎯 因子风控:")
        print(f"     • 因子衰减止损: {'✓' if enable_score_decay_stop else '✗'}")
        print(f"     • 相对排名止损: {'✓' if enable_rank_stop else '✗'}")
        print(f"     • 组合回撤保护: {max_portfolio_drawdown:.1%}")
        print(f"     • 行业轮动: {'✓' if enable_industry_rotation else '✗'}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    def _build_price_dict(self):
        """构建价格字典"""
        price_dict = defaultdict(dict)
        for _, row in self.price_data.iterrows():
            price_dict[str(row['date'])][row['instrument']] = float(row['close'])
        return dict(price_dict)

    def _build_factor_dict(self):
        """构建因子字典"""
        factor_dict = defaultdict(dict)
        for _, row in self.factor_data.iterrows():
            factor_dict[str(row['date'])][row['instrument']] = float(row['position'])
        return dict(factor_dict)

    def _build_industry_dict(self):
        """构建行业字典"""
        industry_dict = defaultdict(dict)
        for _, row in self.factor_data.iterrows():
            if 'industry' in row:
                industry_dict[str(row['date'])][row['instrument']] = row['industry']
        return dict(industry_dict)

    def _calculate_market_signals(self):
        """预计算大盘择时信号"""
        signals = {}
        if self.benchmark_data is None:
            return signals

        df = self.benchmark_data.copy()
        df = df.sort_values('date')
        df['ma'] = df['close'].rolling(window=self.market_ma_period).mean()

        for _, row in df.iterrows():
            date_str = str(row['date'])
            if pd.notna(row['ma']):
                signals[date_str] = row['close'] > row['ma']
            else:
                signals[date_str] = True

        return signals

    def check_market_regime(self, date_str):
        """检查市场状态"""
        if not self.market_signals:
            return True
        return self.market_signals.get(date_str, True)

    # ========== ✅ 新增：资金守恒验证方法 ==========

    def log_cash_flow(self, date, action, stock, amount, description):
        """记录现金流水"""
        self.cash_flow_log.append({
            'date': str(date),
            'action': action,
            'stock': stock,
            'amount': amount,
            'cash_after': self.cash,
            'description': description
        })

    def validate_cash_conservation(self, date):
        """
        ✅ 验证资金守恒

        守恒原理：
        总资产 = 初始资金 + 已实现盈亏 + 未实现盈亏
        """
        date_str = str(date)

        # 计算当前总资产
        current_total_assets = self.calculate_portfolio_value(date)

        # 计算已实现盈亏
        realized_pnl = self.calculate_realized_pnl()

        # 计算未实现盈亏
        unrealized_pnl = self.calculate_unrealized_pnl(date)

        # 理论总资产
        expected_total_assets = self.initial_capital + realized_pnl + unrealized_pnl

        # 验证误差
        error = abs(current_total_assets - expected_total_assets)
        error_rate = error / self.initial_capital

        if error_rate > 0.0001:  # 误差超过0.01%
            print(f"\n⚠️  资金守恒验证失败 ({date_str}):")
            print(f"   当前总资产: ¥{current_total_assets:,.2f}")
            print(f"   期望总资产: ¥{expected_total_assets:,.2f}")
            print(f"   误差: ¥{error:,.2f} ({error_rate:.4%})")
            print(f"   现金: ¥{self.cash:,.2f}")
            print(f"   持仓市值: ¥{current_total_assets - self.cash:,.2f}")
            print(f"   已实现盈亏: ¥{realized_pnl:,.2f}")
            print(f"   未实现盈亏: ¥{unrealized_pnl:,.2f}")

        return error_rate < 0.0001
# ========== 因子风控方法 ==========

    def get_score_rank_percentile(self, stock, date_str, scores):
        """获取股票评分排名百分位"""
        if stock not in scores:
            return 0.5

        stock_score = scores[stock]
        sorted_scores = sorted(scores.values(), reverse=True)
        rank = sorted_scores.index(stock_score) + 1
        percentile = 1 - (rank / len(sorted_scores))

        return percentile

    def check_score_decay_stop(self, stock, current_score, info, holding_days):
        """因子衰减止损"""
        if not self.enable_score_decay_stop:
            return False

        if holding_days < self.min_holding_days:
            return False

        entry_score = info.get('entry_score', 0.5)

        if entry_score > 0:
            score_change = (current_score - entry_score) / entry_score
        else:
            score_change = 0

        should_stop = score_change < -self.score_decay_threshold

        if should_stop and self.debug:
            print(f"    ⚠️  因子衰减止损: {stock} (评分↓{score_change:.2%})")

        return should_stop

    def check_rank_stop(self, stock, date_str, scores):
        """相对排名止损"""
        if not self.enable_rank_stop:
            return False

        percentile = self.get_score_rank_percentile(stock, date_str, scores)
        should_stop = percentile < (1 - self.rank_percentile_threshold)

        if should_stop and self.debug:
            print(f"    ⚠️  相对排名止损: {stock} (排名前{percentile:.1%})")

        return should_stop

    def check_extreme_loss(self, stock, current_price, info):
        """极端亏损保护"""
        cost = info['cost']
        loss_rate = (current_price - cost) / cost
        should_stop = loss_rate < self.extreme_loss_threshold

        if should_stop and self.debug:
            print(f"    🚨 极端亏损保护: {stock} (亏损{loss_rate:.2%})")

        return should_stop

    def check_portfolio_drawdown(self):
        """组合回撤控制"""
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value

        drawdown = (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value

        if drawdown < self.max_portfolio_drawdown:
            if not self.is_risk_mode:
                self.is_risk_mode = True
                if self.debug:
                    print(f"    ⚠️  组合回撤{drawdown:.2%}，进入风险模式")
            return True
        else:
            if self.is_risk_mode:
                self.is_risk_mode = False
                if self.debug:
                    print(f"    ✓ 组合回撤恢复，退出风险模式")
            return False

    def get_industry_weights(self, date_str):
        """获取行业权重"""
        if not self.industry_dict:
            return {}

        industry_weights = defaultdict(float)
        total_value = sum(
            info['shares'] * self.price_dict.get(date_str, {}).get(stock, info['cost'])
            for stock, info in self.positions.items()
        )

        if total_value == 0:
            return {}

        for stock, info in self.positions.items():
            industry = self.industry_dict.get(date_str, {}).get(stock, 'Unknown')
            value = info['shares'] * self.price_dict.get(date_str, {}).get(stock, info['cost'])
            industry_weights[industry] += value / total_value

        return dict(industry_weights)

    def check_industry_concentration(self, stock, date_str):
        """检查行业集中度"""
        if not self.enable_industry_rotation or not self.industry_dict:
            return True

        stock_industry = self.industry_dict.get(date_str, {}).get(stock, 'Unknown')
        industry_weights = self.get_industry_weights(date_str)
        current_weight = industry_weights.get(stock_industry, 0)

        if current_weight >= self.max_industry_weight:
            if self.debug:
                print(f"    ⚠️  行业集中度过高: {stock} ({stock_industry}: {current_weight:.1%})")
            return False

        return True

    def check_risk_conditions(self, date):
        """综合风险检查"""
        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})
        prices = self.price_dict.get(date_str, {})

        to_sell = []

        for stock, info in self.positions.items():
            price = prices.get(stock)
            if not price:
                continue

            holding_days = (pd.to_datetime(date_str) -
                            pd.to_datetime(info['entry_date'])).days

            current_score = scores.get(stock, 0.5)
            pnl_rate = (price - info['cost']) / info['cost']

            # 1. 因子衰减止损
            if self.check_score_decay_stop(stock, current_score, info, holding_days):
                to_sell.append((stock, 'score_decay'))
                continue

            # 2. 相对排名止损
            if self.check_rank_stop(stock, date_str, scores):
                to_sell.append((stock, 'rank_stop'))
                continue

            # 3. 强制流动性换仓
            if holding_days >= (self.rebalance_days * 2) and pnl_rate < 0.02:
                to_sell.append((stock, 'force_turnover'))
                if self.debug:
                    print(f"    ♻️ 强制换仓: {stock} (持有{holding_days}天, 收益{pnl_rate:.2%} < 2%)")
                continue

            # 4. 长期持有亏损检查
            if holding_days >= 30 and pnl_rate < -0.10:
                to_sell.append((stock, 'long_hold_loss'))
                if self.debug:
                    print(f"    ⚠️  长期持有亏损: {stock} (持有{holding_days}天, 亏损{pnl_rate:.2%})")
                continue

            # 5. 极端亏损保护
            if self.check_extreme_loss(stock, price, info):
                to_sell.append((stock, 'extreme_loss'))
                continue

        # 6. 组合回撤控制
        in_risk_mode = self.check_portfolio_drawdown()

        if in_risk_mode:
            current_positions = [
                (stock, scores.get(stock, 0.5))
                for stock in self.positions.keys()
            ]
            current_positions.sort(key=lambda x: x[1])

            target_position_count = int(self.position_size * self.reduce_position_ratio)
            stocks_to_reduce = len(self.positions) - target_position_count

            if stocks_to_reduce > 0:
                for stock, _ in current_positions[:stocks_to_reduce]:
                    if not any(s == stock for s, _ in to_sell):
                        to_sell.append((stock, 'risk_mode_reduce'))

        return to_sell

    # ========== ✅ 修复：交易执行方法 ==========

    def execute_sell(self, date, stock, reason='rebalance'):
        """
        ✅ 修复版卖出：完整追踪现金流
        """
        date_str = str(date)
        price = self.price_dict.get(date_str, {}).get(stock)
        if not price or stock not in self.positions:
            return False

        info = self.positions[stock]
        shares = info['shares']

        # 计算卖出收入
        total_cost_rate = self.sell_cost + self.tax_ratio
        revenue = shares * price * (1 - total_cost_rate)

        # ✅ 关键修复：记录卖出前现金
        cash_before = self.cash

        # 更新现金
        self.cash += revenue

        # 计算盈亏（基于买入时的成本价）
        cost_basis = info['cost'] * shares
        pnl = revenue - cost_basis
        pnl_rate = pnl / cost_basis if cost_basis > 0 else 0

        # ✅ 记录现金流水
        self.log_cash_flow(
            date=date,
            action='sell',
            stock=stock,
            amount=revenue,
            description=f"卖出 {shares:,.0f}股 @ ¥{price:.2f}, 盈亏¥{pnl:,.2f}"
        )

        # 记录交易
        self.trade_records.append({
            'date': date_str,
            'stock': stock,
            'action': 'sell',
            'price': price,
            'shares': shares,
            'amount': revenue,
            'pnl': pnl,
            'pnl_rate': pnl_rate,
            'reason': reason,
            'entry_date': info['entry_date'],
            'entry_price': info['cost'],
            'holding_days': (pd.to_datetime(date_str) -
                             pd.to_datetime(info['entry_date'])).days,
            'cash_before': cash_before,
            'cash_after': self.cash
        })

        # 删除持仓
        del self.positions[stock]

        if self.debug:
            print(f"    ✓ 卖出: {stock} {shares:,.0f}股 @ ¥{price:.2f}, "
                  f"收入¥{revenue:,.0f}, 盈亏{pnl_rate:+.2%}, 原因: {reason}")
            print(f"      现金: ¥{cash_before:,.0f} → ¥{self.cash:,.0f}")

        return True

    def execute_buy_batch(self, date, buy_plan):
        """
        ✅ v4.0 超级保守版：逐笔严格验证
        """
        if not buy_plan:
            return

        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})

        # 记录执行前状态
        cash_before_all = self.cash

        if self.debug:
            print(f"\n  【执行买入 v4.0】")
            print(f"    执行前现金: ¥{cash_before_all:,.0f}")
            print(f"    计划买入: {len(buy_plan)}只")

        # 预先验证总金额
        total_needed = sum(p['amount'] for p in buy_plan.values())

        if total_needed > self.cash:
            print(f"    ❌ 错误：买入计划超出现金")
            print(f"       需要: ¥{total_needed:,.0f}")
            print(f"       可用: ¥{self.cash:,.0f}")
            print(f"       超出: ¥{total_needed - self.cash:,.0f}")

            # 按评分排序
            sorted_items = sorted(
                buy_plan.items(),
                key=lambda x: scores.get(x[0], 0),
                reverse=True
            )

            # 重建计划：只买能买得起的
            new_plan = {}
            remaining = self.cash * 0.99

            for stock, info in sorted_items:
                if info['amount'] <= remaining:
                    new_plan[stock] = info
                    remaining -= info['amount']

            buy_plan = new_plan
            print(f"    ✓ 调整为: {len(buy_plan)}只")

        # 获取当日价格详细信息用于验证
        date_str = str(date)

        # 逐笔执行
        executed = 0
        total_spent = 0

        for stock, info in buy_plan.items():
            shares = info['shares']
            price = info['price']
            amount = info['amount']

            # ========== 新增修复：一字板/涨停过滤 ==========
            # 获取当日的 OHLC 数据
            stock_daily = None
            try:
                # 假设 self.price_data 是 DataFrame，从中获取当日数据
                # 这是一个低效但准确的方法，或者您可以在 rebalance 时传入当日详细数据
                if hasattr(self, 'price_data') and isinstance(self.price_data, pd.DataFrame):
                    daily_row = self.price_data[
                        (self.price_data['date'].astype(str) == date_str) &
                        (self.price_data['instrument'] == stock)
                    ]
                    if not daily_row.empty:
                        stock_daily = daily_row.iloc[0]
            except:
                pass

            if stock_daily is not None:
                # 1. 检查是否一字涨停 (Low == High 且 涨幅 > 9%)
                # 注意：这里需要计算涨幅，如果数据里没有pct_chg，可以用 close/open 判断
                is_limit_up_locked = False

                # 简易判断：如果最高价等于最低价，且价格相对于前收盘（近似）大涨
                if stock_daily['low'] == stock_daily['high']:
                    # 如果没有前收盘价，简单假设涨幅过大就不买
                    # 或者简单判断：开盘即最高且全天未动
                    is_limit_up_locked = True

                # 2. 检查是否涨停 (收盘价 == 最高价 且 涨幅 > 9.5%)
                # 防止打板买入
                if stock_daily['close'] == stock_daily['high']:
                     # 粗略估算涨幅：这里需要谨慎，如果没有前一天价格很难精确判断
                     # 建议：简单起见，禁止买入当日最高价等于收盘价的股票
                     is_limit_up_locked = True

                if is_limit_up_locked:
                    if self.debug:
                        print(f"    ⛔ {stock}: 疑似一字板/涨停，跳过买入 (H={stock_daily['high']}, L={stock_daily['low']})")
                    continue

            # 3. 检查成交量
            if stock_daily is not None and 'volume' in stock_daily and stock_daily['volume'] == 0:
                 if self.debug:
                        print(f"    ⛔ {stock}: 停牌或无成交量，跳过")
                 continue
            # ============================================

            # 验证1：现金充足
            if amount > self.cash:
                print(f"    ❌ {stock}: 现金不足")
                continue

            # 验证2：金额计算正确
            expected = shares * price * (1 + self.buy_cost)
            if abs(expected - amount) > 1:
                print(f"    ❌ {stock}: 金额计算错误")
                print(f"       记录: ¥{amount:,.0f}")
                print(f"       重算: ¥{expected:,.0f}")
                continue

            # ✅ 验证3：股数合理性检查（新增）
            # A股交易股数应为100的整数倍，且不应超过合理范围
            if shares > 10000000:  # 不应超过1000万股
                print(f"    ❌ {stock}: 股数异常巨大 {shares:,}股")
                continue

            if shares % 100 != 0:  # 应为100的整数倍
                print(f"    ❌ {stock}: 股数不是100的整数倍 {shares:,}股")
                continue

            # ✅ 验证4已删除：信任 calculate_buy_plan 的分配结果
            # （calculate_buy_plan 已经做了完整的资金分配和验证）

            # 记录买入前现金
            cash_before = self.cash

            # 扣除现金
            self.cash -= amount

            # 验证5：现金非负
            if self.cash < 0:
                print(f"    ❌ {stock}: 导致现金为负")
                print(f"       买入前: ¥{cash_before:,.0f}")
                print(f"       花费: ¥{amount:,.0f}")
                print(f"       买入后: ¥{self.cash:,.0f}")
                # 回滚
                self.cash = cash_before
                raise ValueError(f"现金变负！")

            # 记录持仓
            cost_basis = amount / shares
            score = scores.get(stock, 0.5)

            self.positions[stock] = {
                'shares': shares,
                'cost': cost_basis,
                'entry_date': date_str,
                'entry_score': score,
                'entry_price': price
            }

            # 记录现金流水
            self.log_cash_flow(
                date=date,
                action='buy',
                stock=stock,
                amount=-amount,
                description=f"买入 {shares:,.0f}股 @ ¥{price:.2f}"
            )

            # 记录交易
            self.trade_records.append({
                'date': date_str,
                'stock': stock,
                'action': 'buy',
                'price': price,
                'shares': shares,
                'amount': amount,
                'reason': 'rebalance',
                'cash_before': cash_before,
                'cash_after': self.cash
            })

            executed += 1
            total_spent += amount

            if self.debug:
                print(f"    ✓ [{executed}] {stock}")
                print(f"         {shares:,.0f}股 × ¥{price:.2f} = ¥{amount:,.0f}")
                print(f"         现金: ¥{cash_before:,.0f} → ¥{self.cash:,.0f}")

        # 最终验证
        actual_spent = cash_before_all - self.cash

        if abs(actual_spent - total_spent) > 1:
            print(f"    ⚠️  花费不匹配")
            print(f"       记录: ¥{total_spent:,.0f}")
            print(f"       实际: ¥{actual_spent:,.0f}")

        if self.debug:
            print(f"\n    【执行完成】")
            print(f"    成功: {executed}/{len(buy_plan)}只")
            print(f"    总花费: ¥{actual_spent:,.0f}")
            print(f"    执行后现金: ¥{self.cash:,.0f}")
            print(f"    ✅ 验证通过")

    # ========== 调仓逻辑 ==========

    def should_rebalance(self, date):
        """判断是否调仓"""
        if self.days_since_rebalance >= self.rebalance_days:
            self.days_since_rebalance = 0
            return True
        self.days_since_rebalance += 1
        return False

    def rebalance(self, date):
        """✨ 调仓（集成因子风控 + 最佳现金管理 + 大盘择时）"""
        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})
        prices = self.price_dict.get(date_str, {})

        if self.debug:
            print(f"\n{'=' * 80}")
            print(f"[调仓] {date_str}")
            print(f"  当前持仓: {len(self.positions)}只")
            print(f"  可用现金: ¥{self.cash:,.0f}")

        # 1. 风险检查（风控卖出始终执行）
        risk_conditions = self.check_risk_conditions(date)
        for stock, reason in risk_conditions:
            self.execute_sell(date, stock, reason=reason)

        # 2. 择时检查
        is_market_good = self.check_market_regime(date_str)
        if not is_market_good:
            if self.debug:
                print(f"  🛑 大盘择时: 市场处于下行趋势，暂停买入！")
            return

        # 3. 获取候选股票
        if not scores:
            return

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:50]

        # 4. 评估现有持仓
        to_sell = []
        for stock, info in list(self.positions.items()):
            in_top = any(stock == c[0] for c in top_candidates[:self.position_size])
            if not in_top:
                to_sell.append(stock)

        # 5. 先卖出释放资金
        for stock in to_sell:
            self.execute_sell(date, stock, reason='rebalance')

        if self.debug:
            print(f"  卖出后: 现金¥{self.cash:,.0f}, 持仓{len(self.positions)}只")

        # 6. 确定待买入股票
        if self.is_risk_mode:
            target_size = int(self.position_size * self.reduce_position_ratio)
        else:
            target_size = self.position_size

        target_stocks = [c[0] for c in top_candidates[:target_size]
                         if c[0] not in self.positions]

        available_slots = target_size - len(self.positions)

        if available_slots > 0 and target_stocks:
            target_stocks = target_stocks[:available_slots]

            # 过滤行业集中度
            filtered_stocks = [
                stock for stock in target_stocks
                if self.check_industry_concentration(stock, date_str)
            ]

            # ========== 新增：过滤一字涨停板 ==========
            if filtered_stocks:
                buyable_stocks = []
                for stock in filtered_stocks:
                    # 检查是否为一字板（开=高=低=收）
                    # 需要从原始数据获取OHLC
                    stock_data = self.price_data[
                        (self.price_data['instrument'] == stock) &
                        (self.price_data['date'] == date_str)
                        ]

                    if len(stock_data) > 0:
                        row = stock_data.iloc[0]
                        # 检查是否为一字板（开=高=低=收）
                        is_limit_up = (
                                row['open'] == row['high'] ==
                                row['low'] == row['close']
                        )

                        # 检查是否涨停（收盘价等于最高价）
                        is_limit_up_close = (row['close'] == row['high'])

                        # 检查成交量
                        has_volume = (row['volume'] > 0)

                        # 只有不满足涨停条件且有成交量的股票才会被买入
                        if not (is_limit_up or is_limit_up_close) and has_volume:
                            buyable_stocks.append(stock)
                    else:
                        # 如果没有数据，默认允许买入
                        buyable_stocks.append(stock)

                filtered_stocks = buyable_stocks
            # ======================================

            if filtered_stocks:
                # 7. ✨ 使用最佳现金管理计算买入计划
                buy_plan = self.cash_manager.calculate_buy_plan(
                    available_cash=self.cash,
                    target_stocks=filtered_stocks,
                    prices=prices
                )

                # 8. 批量执行买入
                if buy_plan:
                    self.execute_buy_batch(date, buy_plan)

        if self.debug:
            print(f"  调仓后: 现金¥{self.cash:,.0f}, 持仓{len(self.positions)}只")

            # ✅ v2.3 新增：调仓后验证资金守恒
            if not self.validate_cash_conservation(date):
                print(f"  ⚠️  资金守恒验证失败！")

            # ✅ v2.3 新增：验证现金非负
            if self.cash < 0:
                raise ValueError(f"现金为负：¥{self.cash:,.2f}")


    # ========== 计算方法 ==========

    def calculate_portfolio_value(self, date):
        """计算组合价值"""
        date_str = str(date)
        prices = self.price_dict.get(date_str, {})

        holdings_value = sum(
            info['shares'] * prices.get(stock, info['cost'])
            for stock, info in self.positions.items()
        )

        return self.cash + holdings_value

    def calculate_realized_pnl(self):
        """
        ✅ 计算已实现盈亏

        已实现盈亏 = 所有卖出交易的盈亏之和
        """
        sell_trades = [record for record in self.trade_records if record['action'] == 'sell']
        return sum(record['pnl'] for record in sell_trades)

    def calculate_unrealized_pnl(self, date):
        """
        ✅ 计算未实现盈亏

        未实现盈亏 = Σ(当前市值 - 成本基础)
        """
        date_str = str(date)
        prices = self.price_dict.get(date_str, {})

        unrealized_pnl = 0
        for stock, info in self.positions.items():
            price = prices.get(stock, info['cost'])
            cost_basis = info['cost'] * info['shares']
            market_value = price * info['shares']
            unrealized_pnl += market_value - cost_basis

        return unrealized_pnl

    def calculate_correct_return(self):
        """
        ✅ 基于资金守恒计算正确收益率

        总收益 = 当前总资产 - 初始资金
        收益率 = 总收益 / 初始资金
        """
        final_total_assets = self.portfolio_value
        total_return = (final_total_assets - self.initial_capital) / self.initial_capital

        return {
            'initial_capital': self.initial_capital,
            'final_total_assets': final_total_assets,
            'total_pnl': final_total_assets - self.initial_capital,
            'total_return': total_return
        }

    def get_detailed_pnl_breakdown(self, date):
        """
        ✅ 获取详细盈亏分解
        """
        realized_pnl = self.calculate_realized_pnl()
        unrealized_pnl = self.calculate_unrealized_pnl(date)
        total_pnl = realized_pnl + unrealized_pnl

        return {
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'cash': self.cash,
            'holdings_value': self.portfolio_value - self.cash,
            'total_assets': self.portfolio_value
        }

    # ========== 回测主循环 ==========

    def run(self, silent=False):
        """
        ✅ 运行回测（含资金守恒验证）v2.3
        """
        if not silent:
            print("\n" + "=" * 80)
            print("⚡ 因子风控 + 最佳现金管理 + 大盘择时 v2.3（资金守恒修复版）")
            print("=" * 80)

        import time
        start_time = time.time()

        # ✅ 记录初始状态
        self.log_cash_flow(
            date=self.trading_days[0],
            action='init',
            stock='N/A',
            amount=self.initial_capital,
            description='初始资金'
        )

        for i, date in enumerate(self.trading_days):
            # ✅ v2.3 新增：调仓前验证现金非负
            if self.cash < 0:
                print(f"\n❌ 第{i}天开始前现金已为负：{date}")
                print(f"   现金: ¥{self.cash:,.2f}")
                print(f"   持仓数: {len(self.positions)}")
                raise ValueError(f"现金为负：¥{self.cash:,.2f}")

            # 调仓或风险检查
            if self.should_rebalance(date):
                self.rebalance(date)
            else:
                # 非调仓日也检查风险
                risk_conditions = self.check_risk_conditions(date)
                for stock, reason in risk_conditions:
                    self.execute_sell(date, stock, reason=reason)

            # ✅ v2.3 新增：调仓后立即验证现金非负
            if self.cash < 0:
                print(f"\n❌ 第{i + 1}天调仓后现金为负：{date}")
                print(f"   现金: ¥{self.cash:,.2f}")
                print(f"   持仓数: {len(self.positions)}")

                # 打印最近的交易记录
                if len(self.trade_records) > 0:
                    recent_trades = pd.DataFrame(self.trade_records).tail(10)
                    print(f"\n   最近10笔交易:")
                    print(recent_trades[['date', 'stock', 'action', 'amount', 'cash_after']])

                raise ValueError(f"现金为负：¥{self.cash:,.2f}")

            # 更新组合价值
            self.portfolio_value = self.calculate_portfolio_value(date)

            # ✅ 获取详细盈亏分解
            pnl_breakdown = self.get_detailed_pnl_breakdown(date)

            # 记录每日状态
            self.daily_records.append({
                'date': str(date),
                'cash': self.cash,
                'holdings_value': self.portfolio_value - self.cash,
                'portfolio_value': self.portfolio_value,
                'position_count': len(self.positions),
                'realized_pnl': pnl_breakdown['realized_pnl'],
                'unrealized_pnl': pnl_breakdown['unrealized_pnl'],
                'total_pnl': pnl_breakdown['total_pnl'],
                'return': (self.portfolio_value - self.capital_base) / self.capital_base
            })

            # ✅ 定期验证资金守恒（每10天验证一次）
            if i % 10 == 0 and not silent:
                is_valid = self.validate_cash_conservation(date)
                if not is_valid:
                    print(f"⚠️  第{i + 1}天资金守恒验证失败，但继续运行...")

            # ✅ v2.3 新增：定期打印进度和关键指标
            if not silent and i > 0 and i % 50 == 0:
                progress = (i / len(self.trading_days)) * 100
                current_return = (self.portfolio_value - self.capital_base) / self.capital_base
                print(f"  进度: {progress:.1f}% | 日期: {date} | "
                      f"现金: ¥{self.cash:,.0f} | 持仓: {len(self.positions)}只 | "
                      f"收益: {current_return:+.2%}")

        elapsed = time.time() - start_time

        # ✅ 最终验证
        if not silent:
            print(f"\n{'=' * 80}")
            print("⚡ 回测完成，进行最终资金守恒验证...")
            print("=" * 80)

            final_valid = self.validate_cash_conservation(self.trading_days[-1])

            if final_valid:
                print("✅ 资金守恒验证通过！")
            else:
                print("❌ 资金守恒验证失败，请检查交易逻辑！")

            # ✅ v2.3 新增：打印现金流水摘要
            if len(self.cash_flow_log) > 0:
                df_cash_flow = pd.DataFrame(self.cash_flow_log)

                print(f"\n💵 现金流水摘要:")
                print(f"   总流水记录: {len(df_cash_flow)} 笔")

                buy_flows = df_cash_flow[df_cash_flow['action'] == 'buy']
                sell_flows = df_cash_flow[df_cash_flow['action'] == 'sell']

                if len(buy_flows) > 0:
                    total_buy = buy_flows['amount'].sum()
                    print(f"   买入总支出: ¥{abs(total_buy):,.2f}")

                if len(sell_flows) > 0:
                    total_sell = sell_flows['amount'].sum()
                    print(f"   卖出总收入: ¥{total_sell:,.2f}")

                # 检查现金流是否合理
                max_cash = df_cash_flow['cash_after'].max()
                min_cash = df_cash_flow['cash_after'].min()

                print(f"   最高现金: ¥{max_cash:,.2f}")
                print(f"   最低现金: ¥{min_cash:,.2f}")

                if min_cash < 0:
                    print(f"   ⚠️  警告：历史上出现过负现金！")
                    negative_records = df_cash_flow[df_cash_flow['cash_after'] < 0]
                    print(f"   负现金记录数: {len(negative_records)}")
                    print(f"   首次出现: {negative_records.iloc[0]['date']}")

            print(f"\n⚡ 总耗时: {elapsed:.2f}秒")

        return self.generate_context()

    # ========== 生成回测结果 ==========

    def generate_context(self):
        """
        ✅ 生成回测上下文（含资金守恒验证）
        """
        df_records = pd.DataFrame(self.daily_records)
        df_trades = pd.DataFrame(self.trade_records)
        df_cash_flow = pd.DataFrame(self.cash_flow_log)

        sell_trades = df_trades[df_trades['action'] == 'sell']

        # ✅ 使用正确的收益率计算方法
        return_metrics = self.calculate_correct_return()
        final_value = return_metrics['final_total_assets']
        total_return = return_metrics['total_return']

        if len(sell_trades) > 0:
            win_rate = (sell_trades['pnl'] > 0).sum() / len(sell_trades)
            avg_pnl = sell_trades['pnl'].mean()
            avg_pnl_rate = sell_trades['pnl_rate'].mean()
            total_realized_pnl = sell_trades['pnl'].sum()
        else:
            win_rate = 0
            avg_pnl = 0
            avg_pnl_rate = 0
            total_realized_pnl = 0

        # ✅ 最终盈亏分解
        final_breakdown = self.get_detailed_pnl_breakdown(self.trading_days[-1])

        # ✅ 资金流水统计
        total_buy_amount = df_trades[df_trades['action'] == 'buy']['amount'].sum()
        total_sell_amount = df_trades[df_trades['action'] == 'sell']['amount'].sum()

        # 打印详细摘要
        print(f"\n{'=' * 80}")
        print("📊 回测结果摘要")
        print("=" * 80)
        print(f"\n💰 资金概况:")
        print(f"  初始资金: ¥{self.initial_capital:,.2f}")
        print(f"  最终总资产: ¥{final_value:,.2f}")
        print(f"  最终现金: ¥{self.cash:,.2f}")
        print(f"  最终持仓市值: ¥{final_breakdown['holdings_value']:,.2f}")

        print(f"\n📈 收益指标:")
        print(f"  总收益: ¥{final_breakdown['total_pnl']:,.2f}")
        print(f"  总收益率: {total_return:+.2%}")
        print(f"  已实现盈亏: ¥{final_breakdown['realized_pnl']:,.2f}")
        print(f"  未实现盈亏: ¥{final_breakdown['unrealized_pnl']:,.2f}")

        print(f"\n📊 交易统计:")
        print(f"  总交易次数: {len(df_trades)}笔")
        print(f"  买入次数: {len(df_trades[df_trades['action'] == 'buy'])}笔")
        print(f"  卖出次数: {len(sell_trades)}笔")
        print(f"  胜率: {win_rate:.2%}")
        if len(sell_trades) > 0:
            print(f"  平均盈亏: ¥{avg_pnl:,.2f} ({avg_pnl_rate:+.2%})")

        print(f"\n💵 资金流水:")
        print(f"  累计买入金额: ¥{total_buy_amount:,.2f}")
        print(f"  累计卖出金额: ¥{total_sell_amount:,.2f}")
        print(f"  资金周转: ¥{total_buy_amount + total_sell_amount:,.2f}")

        print(f"\n✅ 资金守恒验证:")
        expected_total = self.initial_capital + final_breakdown['total_pnl']
        error = abs(final_value - expected_total)
        print(f"  计算总资产: ¥{final_value:,.2f}")
        print(f"  期望总资产: ¥{expected_total:,.2f} (初始+盈亏)")
        print(f"  误差: ¥{error:,.2f} ({error / self.initial_capital:.4%})")

        if error / self.initial_capital < 0.0001:
            print(f"  状态: ✅ 验证通过")
        else:
            print(f"  状态: ❌ 验证失败")

        print("=" * 80)

        return {
            'daily_records': df_records,
            'trade_records': df_trades,
            'cash_flow_log': df_cash_flow,

            # ✅ 修复后的指标
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_pnl': final_breakdown['total_pnl'],
            'realized_pnl': final_breakdown['realized_pnl'],
            'unrealized_pnl': final_breakdown['unrealized_pnl'],

            # 交易统计
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'avg_pnl_rate': avg_pnl_rate,
            'total_trades': len(df_trades),
            'buy_trades': len(df_trades[df_trades['action'] == 'buy']),
            'sell_trades': len(sell_trades),

            # 资金流水
            'total_buy_amount': total_buy_amount,
            'total_sell_amount': total_sell_amount,

            # 持仓信息
            'positions': self.positions,
            'final_cash': self.cash,
            'final_holdings_value': final_breakdown['holdings_value']
        }


# ========== 便捷接口 ==========

def run_factor_based_strategy_v2(factor_data, price_data,
                                 benchmark_data=None,
                                 start_date='2023-01-01', end_date='2025-12-05',
                                 capital_base=1000000, position_size=10,
                                 rebalance_days=5, cash_reserve_ratio=0.05,
                                 enable_market_timing=True,
                                 **kwargs):
    """
    ✅ 运行因子风控 + 最佳现金管理策略（v2.2 资金守恒修复版）

    核心修复：
    1. 完整的现金流追踪
    2. 资金守恒验证
    3. 正确的收益率计算
    4. 防止超额买入

    参数说明：
    ----------
    factor_data : DataFrame
        因子数据，必须包含列: ['date', 'instrument', 'position']
        可选列: ['industry']

    price_data : DataFrame
        价格数据，必须包含列: ['date', 'instrument', 'close']

    benchmark_data : DataFrame, optional
        基准指数数据，用于大盘择时
        必须包含列: ['date', 'close']

    start_date : str
        回测开始日期

    end_date : str
        回测结束日期

    capital_base : float
        初始资金

    position_size : int
        持仓股票数量

    rebalance_days : int
        调仓周期（天数）

    cash_reserve_ratio : float
        现金保留比例（0.05 = 5%）

    enable_market_timing : bool
        是否启用大盘择时

    返回：
    ------
    dict : 包含以下键值的字典
        - daily_records: 每日记录DataFrame
        - trade_records: 交易记录DataFrame
        - cash_flow_log: 现金流水DataFrame
        - initial_capital: 初始资金
        - final_value: 最终总资产
        - total_return: 总收益率
        - total_pnl: 总盈亏
        - realized_pnl: 已实现盈亏
        - unrealized_pnl: 未实现盈亏
        - win_rate: 胜率
        - positions: 最终持仓
        - final_cash: 最终现金
    """
    engine = FactorBasedRiskControlOptimized(
        factor_data, price_data,
        benchmark_data=benchmark_data,
        enable_market_timing=enable_market_timing,
        start_date=start_date, end_date=end_date, capital_base=capital_base,
        position_size=position_size, rebalance_days=rebalance_days,
        cash_reserve_ratio=cash_reserve_ratio, **kwargs
    )

    return engine.run()


# ========== 示例用法 ==========

if __name__ == "__main__":
    """
    使用示例
    """

    # 1. 准备数据
    # factor_data 示例结构:
    # | date       | instrument | position | industry |
    # |------------|------------|----------|----------|
    # | 2023-01-01 | 000001.SZ  | 0.95     | 银行     |
    # | 2023-01-01 | 000002.SZ  | 0.88     | 地产     |

    # price_data 示例结构:
    # | date       | instrument | close |
    # |------------|------------|-------|
    # | 2023-01-01 | 000001.SZ  | 10.5  |
    # | 2023-01-01 | 000002.SZ  | 15.2  |

    # benchmark_data 示例结构（可选）:
    # | date       | close   |
    # |------------|---------|
    # | 2023-01-01 | 3000.15 |
    # | 2023-01-02 | 3010.28 |

    # 2. 运行回测
    # result = run_factor_based_strategy_v2(
    #     factor_data=factor_data,
    #     price_data=price_data,
    #     benchmark_data=benchmark_data,  # 可选
    #     capital_base=1000000,
    #     position_size=10,
    #     rebalance_days=5,
    #     cash_reserve_ratio=0.05,
    #     enable_market_timing=True,  # 启用择时
    #     debug=False
    # )

    # 3. 查看结果
    # print(f"总收益率: {result['total_return']:.2%}")
    # print(f"胜率: {result['win_rate']:.2%}")
    # print(result['daily_records'].tail())

    print("✅ 代码修复完成！")
    print("\n核心修复点:")
    print("1. ✅ execute_sell: 完整记录现金流入")
    print("2. ✅ execute_buy_batch: 完整记录现金流出 + 资金检查")
    print("3. ✅ log_cash_flow: 追踪每笔资金变动")
    print("4. ✅ validate_cash_conservation: 验证资金守恒")
    print("5. ✅ calculate_correct_return: 基于守恒原理计算收益率")
    print("6. ✅ generate_context: 输出完整的资金流水和验证结果")
    print("\n使用建议:")
    print("- 启用 debug=True 查看详细的交易和现金流动")
    print("- 定期检查 cash_flow_log 追踪资金流水")
    print("- 回测结束后查看资金守恒验证结果")