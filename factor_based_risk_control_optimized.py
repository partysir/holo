"""
factor_based_risk_control_optimized.py - 修复版 v3.0
修复内容：
1. ✅ 补全 daily_records 中的字段 ('position_count', 'return', 'holdings_value')
2. ✅ 保留了之前的空交易处理修复
3. ✅ 新增交易成本统计功能
4. ✅ 增强最小持仓天数保护逻辑
5. ✅ 优化风险模式下的持仓排序算法

版本：v3.0
日期：2025-12-29
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
        self.cash_reserve_ratio = cash_reserve_ratio
        self.buy_cost = buy_cost
        self.min_buy_amount = min_buy_amount
        self.debug = debug

    def calculate_buy_plan(self, available_cash, target_stocks, prices):
        if not target_stocks or available_cash <= 0:
            return {}

        total_investment = available_cash * (1 - self.cash_reserve_ratio)
        buy_plan = {}
        remaining_investment = total_investment
        remaining_stocks = list(target_stocks)

        for i, stock in enumerate(target_stocks):
            if stock not in prices:
                remaining_stocks.remove(stock)
                continue

            price = prices[stock]
            if price <= 0: continue

            target_amount = remaining_investment / len(remaining_stocks)
            shares = int(target_amount / price / (1 + self.buy_cost))
            shares = int(shares / 100) * 100
            actual_amount = shares * price * (1 + self.buy_cost)

            if shares < 100 or actual_amount < self.min_buy_amount:
                remaining_stocks.remove(stock)
                continue

            buy_plan[stock] = {
                'shares': shares,
                'price': price,
                'amount': actual_amount,
                'target_amount': target_amount
            }
            remaining_investment -= actual_amount
            remaining_stocks.remove(stock)

        return buy_plan


class FactorBasedRiskControlOptimized:
    """因子风控 + 最佳现金管理 + 大盘择时 (完整集成版 v3.0)"""

    def __init__(self, factor_data, price_data,
                 benchmark_data=None,
                 market_ma_period=60,
                 enable_market_timing=True,
                 start_date='2023-01-01', end_date='2025-12-05',
                 capital_base=1000000, position_size=10,
                 rebalance_days=5,
                 cash_reserve_ratio=0.05,
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

        self.cash_manager = OptimalCashManager(
            cash_reserve_ratio=cash_reserve_ratio,
            buy_cost=buy_cost,
            debug=debug
        )

        # 初始化数据
        print("\n  ⚡ 构建因子风控 + 最佳现金管理 + 择时系统 (v3.0)...")
        self.price_dict = self._build_price_dict()
        self.factor_dict = self._build_factor_dict()

        # 修复：防止factor_data为空时报错
        if not factor_data.empty:
            self.trading_days = sorted(factor_data['date'].unique())
        else:
            self.trading_days = []

        self.market_signals = self._calculate_market_signals() if self.enable_market_timing else {}

        if 'industry' in factor_data.columns:
            self.industry_dict = self._build_industry_dict()
        else:
            self.industry_dict = None
            self.enable_industry_rotation = False

        # 状态初始化
        self.cash = capital_base
        self.positions = {}
        self.portfolio_value = capital_base
        self.max_portfolio_value = capital_base
        self.daily_records = []
        self.trade_records = []
        self.days_since_rebalance = rebalance_days
        self.is_risk_mode = False

    def _build_price_dict(self):
        price_dict = defaultdict(dict)
        if self.price_data.empty: return dict(price_dict)
        for _, row in self.price_data.iterrows():
            price_dict[str(row['date'])][row['instrument']] = float(row['close'])
        return dict(price_dict)

    def _build_factor_dict(self):
        factor_dict = defaultdict(dict)
        if self.factor_data.empty: return dict(factor_dict)
        for _, row in self.factor_data.iterrows():
            factor_dict[str(row['date'])][row['instrument']] = float(row['position'])
        return dict(factor_dict)

    def _build_industry_dict(self):
        industry_dict = defaultdict(dict)
        for _, row in self.factor_data.iterrows():
            if 'industry' in row:
                industry_dict[str(row['date'])][row['instrument']] = row['industry']
        return dict(industry_dict)

    def _calculate_market_signals(self):
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
        if not self.market_signals:
            return True
        return self.market_signals.get(date_str, True)

    def get_industry_weights(self, date_str):
        if not self.industry_dict: return {}
        industry_weights = defaultdict(float)
        total_value = sum(
            info['shares'] * self.price_dict.get(date_str, {}).get(stock, info['cost'])
            for stock, info in self.positions.items()
        )
        if total_value == 0: return {}
        for stock, info in self.positions.items():
            industry = self.industry_dict.get(date_str, {}).get(stock, 'Unknown')
            value = info['shares'] * self.price_dict.get(date_str, {}).get(stock, info['cost'])
            industry_weights[industry] += value / total_value
        return dict(industry_weights)

    def check_industry_concentration(self, stock, date_str):
        if not self.enable_industry_rotation or not self.industry_dict: return True
        stock_industry = self.industry_dict.get(date_str, {}).get(stock, 'Unknown')
        industry_weights = self.get_industry_weights(date_str)
        return industry_weights.get(stock_industry, 0) < self.max_industry_weight

    def check_risk_conditions(self, date):
        """✅ 修复版风控检查 - 增强最小持仓天数保护"""
        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})
        prices = self.price_dict.get(date_str, {})
        to_sell = []

        for stock, info in self.positions.items():
            price = prices.get(stock)
            if not price: continue

            holding_days = (pd.to_datetime(date_str) - pd.to_datetime(info['entry_date'])).days
            current_score = scores.get(stock, 0.5)
            loss_rate = (price - info['cost']) / info['cost']

            # ✅ 新增：最小持仓天数保护
            if holding_days < self.min_holding_days:
                # 持仓天数不足，除非极端亏损否则不卖出
                if loss_rate > self.extreme_loss_threshold:
                    continue  # 未达到极端亏损，跳过
                else:
                    # 极端亏损也要卖
                    to_sell.append((stock, f'extreme_loss_early({loss_rate:.2%})'))
                    continue

            # 1. 评分衰减止损（仅在满足最小持仓天数后）
            if self.enable_score_decay_stop and holding_days >= self.min_holding_days:
                entry_score = info.get('entry_score', 0.5)
                if entry_score > 0:
                    decay_rate = (current_score - entry_score) / entry_score
                    if decay_rate < -self.score_decay_threshold:
                        to_sell.append((stock, f'score_decay({decay_rate:.2%})'))
                        continue

            # 2. 极端亏损止损
            if loss_rate < self.extreme_loss_threshold:
                to_sell.append((stock, f'extreme_loss({loss_rate:.2%})'))
                continue

        # 3. 组合回撤保护
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value
        drawdown = (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value

        if drawdown < self.max_portfolio_drawdown:
            self.is_risk_mode = True
            # 减仓
            target_size = int(self.position_size * self.reduce_position_ratio)
            if len(self.positions) > target_size:
                # ✅ 优化：按持仓天数+评分综合排序，优先卖出持有久且评分低的
                sorted_pos = sorted(
                    self.positions.items(),
                    key=lambda x: (
                        scores.get(x[0], 0) * 0.7 +  # 评分权重70%
                        (pd.to_datetime(date_str) - pd.to_datetime(x[1]['entry_date'])).days / 100 * 0.3  # 持仓天数权重30%
                    )
                )
                for s, _ in sorted_pos[:(len(self.positions)-target_size)]:
                    if not any(x[0]==s for x in to_sell):
                        to_sell.append((s, 'risk_mode_reduce'))
        else:
            self.is_risk_mode = False

        return to_sell

    def execute_sell(self, date, stock, reason='rebalance'):
        date_str = str(date)
        price = self.price_dict.get(date_str, {}).get(stock)
        if not price or stock not in self.positions:
            return False

        info = self.positions[stock]
        shares = info['shares']
        total_cost_rate = self.sell_cost + self.tax_ratio
        revenue = shares * price * (1 - total_cost_rate)
        self.cash += revenue

        cost_basis = info['cost'] * shares
        pnl = revenue - cost_basis
        pnl_rate = (revenue - cost_basis) / cost_basis if cost_basis > 0 else 0

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
            'entry_date': info['entry_date']
        })

        del self.positions[stock]
        return True

    def execute_buy_batch(self, date, buy_plan):
        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})

        for stock, plan_info in buy_plan.items():
            shares = plan_info['shares']
            price = plan_info['price']
            amount = plan_info['amount']

            self.cash -= amount
            score = scores.get(stock, 0.5)

            # 记录平均成本
            self.positions[stock] = {
                'shares': shares,
                'cost': amount / shares,
                'entry_date': date_str,
                'entry_score': score
            }

            self.trade_records.append({
                'date': date_str,
                'stock': stock,
                'action': 'buy',
                'price': price,
                'shares': shares,
                'amount': amount,
                'reason': 'rebalance'
            })

    def run(self, silent=False):
        """运行回测"""
        import time
        start_time = time.time()

        for date in self.trading_days:
            # 1. 调仓判断
            if self.days_since_rebalance >= self.rebalance_days:
                self.days_since_rebalance = 0

                # 卖出风险股
                risk_conditions = self.check_risk_conditions(date)
                for stock, reason in risk_conditions:
                    self.execute_sell(date, stock, reason=reason)

                # 择时 & 买入
                date_str = str(date)
                if self.check_market_regime(date_str):
                    scores = self.factor_dict.get(date_str, {})
                    prices = self.price_dict.get(date_str, {})

                    if scores:
                        # 选股
                        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        top_candidates = [x[0] for x in sorted_candidates[:50]]

                        target_size = self.position_size
                        if self.is_risk_mode:
                            target_size = int(self.position_size * self.reduce_position_ratio)

                        # 卖出非Top股
                        current_top = top_candidates[:target_size]
                        for stock in list(self.positions.keys()):
                            if stock not in current_top:
                                self.execute_sell(date, stock, reason='rebalance')

                        # 买入新股
                        target_stocks = [s for s in current_top if s not in self.positions]
                        available_slots = target_size - len(self.positions)

                        if available_slots > 0 and target_stocks:
                            target_stocks = target_stocks[:available_slots]
                            # 行业过滤
                            target_stocks = [s for s in target_stocks if self.check_industry_concentration(s, date_str)]

                            # 计算买入计划
                            buy_plan = self.cash_manager.calculate_buy_plan(
                                self.cash, target_stocks, prices
                            )
                            if buy_plan:
                                self.execute_buy_batch(date, buy_plan)
            else:
                self.days_since_rebalance += 1
                # 非调仓日只做风控卖出
                risk_conditions = self.check_risk_conditions(date)
                for stock, reason in risk_conditions:
                    self.execute_sell(date, stock, reason=reason)

            # 2. 计算并记录当日状态
            self.portfolio_value = self.calculate_portfolio_value(date)

            # 补全字段
            self.daily_records.append({
                'date': str(date),
                'cash': self.cash,
                'holdings_value': self.portfolio_value - self.cash,
                'portfolio_value': self.portfolio_value,
                'position_count': len(self.positions),
                'return': (self.portfolio_value - self.capital_base) / self.capital_base
            })

        elapsed = time.time() - start_time
        if not silent:
            print(f"\n⚡ 回测完成，耗时: {elapsed:.2f}秒")

        return self.generate_context()

    def calculate_portfolio_value(self, date):
        date_str = str(date)
        prices = self.price_dict.get(date_str, {})
        holdings_val = sum(
            info['shares'] * prices.get(stock, info['cost'])
            for stock, info in self.positions.items()
        )
        return self.cash + holdings_val

    def generate_context(self):
        """✅ 修复版生成上下文 - 添加交易成本统计"""
        df_records = pd.DataFrame(self.daily_records)
        df_trades = pd.DataFrame(self.trade_records)

        # 处理无交易情况
        if df_trades.empty:
            sell_trades = pd.DataFrame()
            win_rate = 0
            total_realized_pnl = 0
            total_cost = 0
        else:
            if 'action' in df_trades.columns:
                sell_trades = df_trades[df_trades['action'] == 'sell']
                if not sell_trades.empty:
                    win_rate = (sell_trades['pnl'] > 0).sum() / len(sell_trades)
                    total_realized_pnl = sell_trades['pnl'].sum()
                else:
                    win_rate = 0
                    total_realized_pnl = 0

                # ✅ 核心修复：计算总交易成本
                buy_trades = df_trades[df_trades['action'] == 'buy']
                sell_trades_all = df_trades[df_trades['action'] == 'sell']

                buy_cost_total = (buy_trades['amount'] * self.buy_cost).sum() if not buy_trades.empty else 0
                sell_cost_total = (sell_trades_all['amount'] * (self.sell_cost + self.tax_ratio)).sum() if not sell_trades_all.empty else 0
                total_cost = buy_cost_total + sell_cost_total
            else:
                sell_trades = pd.DataFrame()
                win_rate = 0
                total_realized_pnl = 0
                total_cost = 0

        final_value = self.portfolio_value
        total_return = (final_value - self.capital_base) / self.capital_base

        # 计算平均持仓天数
        if not sell_trades.empty and 'entry_date' in sell_trades.columns:
            sell_trades['holding_days'] = (pd.to_datetime(sell_trades['date']) - pd.to_datetime(sell_trades['entry_date'])).dt.days
            avg_holding_days = sell_trades['holding_days'].mean()
        else:
            avg_holding_days = 0

        return {
            'daily_records': df_records,
            'trade_records': df_trades,
            'final_value': final_value,
            'total_return': total_return,
            'win_rate': win_rate,
            'positions': self.positions,
            'total_realized_pnl': total_realized_pnl,
            'total_cost': total_cost,  # ✅ 新增字段
            'avg_holding_days': avg_holding_days,  # ✅ 新增字段
            'initial_capital': self.capital_base
        }


# ========== 便捷接口 ==========
def run_factor_based_strategy_v2(factor_data, price_data,
                                 benchmark_data=None,
                                 enable_market_timing=True,
                                 **kwargs):
    """运行因子风控 + 最佳现金管理策略"""
    engine = FactorBasedRiskControlOptimized(
        factor_data, price_data,
        benchmark_data=benchmark_data,
        enable_market_timing=enable_market_timing,
        **kwargs
    )
    return engine.run()