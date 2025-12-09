"""
ultimate_fast_system.py - 策略优化版

优化内容:
1. 降低换仓频率：提高评分阈值，增加交易成本考虑
2. 动态持仓管理：长期持仓定期重新评估
3. 仓位控制：根据市场环境动态调整仓位
4. 分档止损：短期宽松，长期严格
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
import time


class OptimizedBacktest:
    """优化版回测引擎"""

    def __init__(self, factor_data, price_data, start_date, end_date,
                 capital_base=1000000, position_size=10,
                 stop_loss=-0.15, take_profit=None,
                 score_threshold=0.15,  # ✨ 提高到15%，降低换仓频率
                 max_rebalance_per_day=1,
                 force_replace_days=45,  # ✨ 缩短到45天
                 transaction_cost=0.0015,  # ✨ 交易成本0.15%
                 min_holding_days=7,  # ✨ 最少持有7天
                 dynamic_stop_loss=True):  # ✨ 动态止损
        """
        优化版初始化

        新增参数:
        :param transaction_cost: 交易成本（买入+卖出）
        :param min_holding_days: 最少持有天数（避免频繁交易）
        :param dynamic_stop_loss: 是否启用动态止损
        """
        self.start_date = start_date
        self.end_date = end_date
        self.capital_base = capital_base
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.score_threshold = score_threshold
        self.max_rebalance_per_day = max_rebalance_per_day
        self.force_replace_days = force_replace_days
        self.transaction_cost = transaction_cost
        self.min_holding_days = min_holding_days
        self.dynamic_stop_loss = dynamic_stop_loss

        # 构建字典索引
        print("  ⚡ Building fast index...")
        start_time = time.time()

        self.price_dict = self._build_price_dict(price_data)
        self.factor_dict = self._build_factor_dict(factor_data)
        self.trading_days = sorted(factor_data['date'].unique())

        print(f"  ✓ Index built in {time.time() - start_time:.2f}s")
        print(f"     Trading days: {len(self.trading_days)}")
        print(f"     Stocks: {len(set(factor_data['instrument']))}")

        # 状态变量
        self.cash = capital_base
        self.positions = {}
        self.portfolio_value = capital_base
        self.daily_records = []
        self.trade_records = []

        # ✨ 市场环境判断
        self.market_ma20 = None
        self.market_volatility = None

    def _build_price_dict(self, price_data):
        """构建价格快速索引"""
        price_dict = defaultdict(dict)
        for _, row in price_data.iterrows():
            date = str(row['date'])
            stock = row['instrument']
            price_dict[date][stock] = float(row['close'])
        return dict(price_dict)

    def _build_factor_dict(self, factor_data):
        """构建因子快速索引"""
        factor_dict = defaultdict(dict)
        for _, row in factor_data.iterrows():
            date = str(row['date'])
            stock = row['instrument']
            factor_dict[date][stock] = float(row['position'])
        return dict(factor_dict)

    def get_dynamic_stop_loss(self, holding_days):
        """
        ✨ 动态止损：持有时间越长，止损越严格

        策略：
        - 0-15天：-18%（宽松，给股票表现时间）
        - 16-30天：-15%（标准）
        - 31-45天：-12%（收紧）
        - 46天+：-10%（严格，长期表现不佳应退出）
        """
        if not self.dynamic_stop_loss:
            return self.stop_loss

        if holding_days <= 15:
            return -0.18
        elif holding_days <= 30:
            return -0.15
        elif holding_days <= 45:
            return -0.12
        else:
            return -0.10

    def calculate_market_condition(self, date):
        """
        ✨ 计算市场环境

        使用组合净值作为市场代理
        """
        date_str = str(date)

        # 计算20日移动平均
        if len(self.daily_records) >= 20:
            recent_values = [r['portfolio_value'] for r in self.daily_records[-20:]]
            self.market_ma20 = np.mean(recent_values)

            # 计算波动率
            returns = np.diff(recent_values) / recent_values[:-1]
            self.market_volatility = np.std(returns)
        else:
            self.market_ma20 = self.portfolio_value
            self.market_volatility = 0

    def get_position_adjustment_factor(self):
        """
        ✨ 根据市场环境调整仓位

        市场好（组合在20日均线上）：满仓
        市场差（组合在20日均线下且波动大）：降低仓位
        """
        if self.market_ma20 is None:
            return 1.0

        # 当前净值相对均线的位置
        position_ratio = self.portfolio_value / self.market_ma20

        # 高波动降低仓位
        if self.market_volatility > 0.03:  # 日波动率>3%
            if position_ratio < 0.95:  # 跌破均线5%
                return 0.7  # 降低到70%仓位

        return 1.0  # 正常满仓

    def get_top_stocks_fast(self, date, top_n=50):
        """快速获取高分股票"""
        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})
        if not scores:
            return []

        stocks = list(scores.keys())
        score_values = np.array([scores[s] for s in stocks])
        top_indices = np.argsort(score_values)[-top_n:][::-1]

        return [stocks[i] for i in top_indices]

    def check_stop_loss_take_profit_batch(self, date):
        """批量检查止损止盈（使用动态止损）"""
        date_str = str(date)
        daily_prices = self.price_dict.get(date_str, {})

        to_sell = []
        for stock, info in self.positions.items():
            price = daily_prices.get(stock)
            if not price:
                continue

            pnl_rate = (price - info['cost']) / info['cost']
            holding_days = (pd.to_datetime(date_str) - pd.to_datetime(info['entry_date'])).days

            # ✨ 使用动态止损
            dynamic_stop = self.get_dynamic_stop_loss(holding_days)

            if pnl_rate <= dynamic_stop:
                to_sell.append((stock, 'stop_loss', pnl_rate))
            elif self.take_profit and pnl_rate >= self.take_profit:
                to_sell.append((stock, 'take_profit', pnl_rate))

        return to_sell

    def execute_trade_fast(self, date, stock, action, shares=None, reason='rebalance'):
        """快速执行交易（含交易成本）"""
        date_str = str(date)
        price = self.price_dict.get(date_str, {}).get(stock)
        if not price:
            return False

        if action == 'buy':
            # ✨ 考虑交易成本
            cost_with_fee = shares * price * (1 + self.transaction_cost)
            if cost_with_fee > self.cash:
                return False

            self.cash -= cost_with_fee
            score = self.factor_dict.get(date_str, {}).get(stock, 0.5)

            self.positions[stock] = {
                'shares': shares,
                'cost': price,  # 记录原始价格
                'entry_date': date_str,
                'entry_score': score
            }

            self.trade_records.append({
                'date': date_str,
                'stock': stock,
                'action': 'buy',
                'price': price,
                'shares': shares,
                'amount': cost_with_fee,
                'reason': reason
            })
            return True

        elif action == 'sell':
            if stock not in self.positions:
                return False

            info = self.positions[stock]
            shares = info['shares']

            # ✨ 考虑交易成本
            revenue_after_fee = shares * price * (1 - self.transaction_cost)
            self.cash += revenue_after_fee

            pnl = revenue_after_fee - (shares * info['cost'] * (1 + self.transaction_cost))
            pnl_rate = (price - info['cost']) / info['cost']

            self.trade_records.append({
                'date': date_str,
                'stock': stock,
                'action': 'sell',
                'price': price,
                'shares': shares,
                'amount': revenue_after_fee,
                'pnl': pnl,
                'pnl_rate': pnl_rate,
                'reason': reason,
                'entry_date': info['entry_date'],
                'holding_days': (pd.to_datetime(date_str) - pd.to_datetime(info['entry_date'])).days
            })

            del self.positions[stock]
            return True

        return False

    def rebalance_fast(self, date):
        """
        优化版再平衡

        改进：
        1. 提高换仓阈值（15%）
        2. 考虑交易成本
        3. 最少持有天数限制
        4. 动态仓位调整
        5. 长期持仓强制评估
        """
        date_str = str(date)

        # 1. 止损止盈（使用动态止损）
        to_sell = self.check_stop_loss_take_profit_batch(date)
        for stock, reason, _ in to_sell:
            self.execute_trade_fast(date, stock, 'sell', reason=reason)

        # 2. 获取候选股票
        candidates = self.get_top_stocks_fast(date, top_n=50)
        if not candidates:
            return

        # 3. ✨ 计算目标仓位数（根据市场环境）
        position_factor = self.get_position_adjustment_factor()
        target_positions = int(self.position_size * position_factor)

        # 4. 填充空仓
        available_slots = target_positions - len(self.positions)
        if available_slots > 0:
            for stock in candidates:
                if stock not in self.positions and available_slots > 0:
                    price = self.price_dict.get(date_str, {}).get(stock)
                    if price and self.cash > 0:
                        shares = int(self.cash / target_positions / price / (1 + self.transaction_cost) * 0.95)
                        if shares > 0:
                            success = self.execute_trade_fast(date, stock, 'buy', shares, 'fill')
                            if success:
                                available_slots -= 1

        # 5. ✨ 优化换仓逻辑
        if len(self.positions) >= target_positions:
            scores = self.factor_dict.get(date_str, {})
            prices = self.price_dict.get(date_str, {})

            stock_evaluation = []
            for stock, info in self.positions.items():
                score = scores.get(stock, 0)
                price = prices.get(stock, info['cost'])
                pnl_rate = (price - info['cost']) / info['cost']
                holding_days = (pd.to_datetime(date_str) - pd.to_datetime(info['entry_date'])).days

                # ✨ 综合评价
                # 1. 评分下降
                score_decline = info['entry_score'] - score

                # 2. 长期持仓且表现差
                long_and_poor = (holding_days >= self.force_replace_days and
                               (pnl_rate < 0 or score < 0.5))

                # 3. 持有时间短于最小天数（不换）
                too_short = holding_days < self.min_holding_days

                priority = score
                if long_and_poor:
                    priority -= 1.0  # 最高优先级替换
                elif score_decline > 0.2:
                    priority -= 0.5  # 评分大幅下降

                if too_short:
                    priority += 10.0  # 保护最近买入的

                stock_evaluation.append({
                    'stock': stock,
                    'score': score,
                    'pnl_rate': pnl_rate,
                    'holding_days': holding_days,
                    'priority': priority,
                    'too_short': too_short,
                    'long_and_poor': long_and_poor
                })

            # 按优先级排序
            stock_evaluation.sort(key=lambda x: x['priority'])

            # 每日最多换N只
            rebalance_count = 0
            for worst in stock_evaluation:
                if rebalance_count >= self.max_rebalance_per_day:
                    break

                # 跳过持有时间过短的
                if worst['too_short']:
                    continue

                worst_stock = worst['stock']
                worst_score = worst['score']

                # 寻找替换候选
                for candidate in candidates:
                    if candidate not in self.positions:
                        candidate_score = scores.get(candidate, 0)

                        # ✨ 提高换仓门槛
                        score_improvement = candidate_score - worst_score

                        # 考虑交易成本后的收益改善
                        expected_improvement = score_improvement - (self.transaction_cost * 2)

                        should_replace = False

                        # 强制换仓（长期表现差）
                        if worst['long_and_poor'] and score_improvement > 0.05:
                            should_replace = True

                        # 正常换仓（评分差异大于阈值+交易成本）
                        elif expected_improvement > self.score_threshold:
                            should_replace = True

                        if should_replace:
                            # 卖出
                            sell_success = self.execute_trade_fast(
                                date, worst_stock, 'sell',
                                reason='force_replace' if worst['long_and_poor'] else 'rebalance'
                            )

                            if sell_success:
                                # 买入
                                price = prices.get(candidate)
                                if price and self.cash > 0:
                                    shares = int(self.cash / price / (1 + self.transaction_cost) * 0.95)
                                    if shares > 0:
                                        buy_success = self.execute_trade_fast(
                                            date, candidate, 'buy', shares, 'rebalance'
                                        )
                                        if buy_success:
                                            rebalance_count += 1

                            break

    def calculate_portfolio_value_batch(self, date):
        """批量计算组合价值"""
        date_str = str(date)
        prices = self.price_dict.get(date_str, {})

        holdings_value = sum(
            info['shares'] * prices.get(stock, 0)
            for stock, info in self.positions.items()
        )

        return self.cash + holdings_value

    def run(self, silent=False):
        """运行回测"""
        if not silent:
            print("\n" + "=" * 80)
            print("⚡ Optimized Backtest Engine")
            print("=" * 80)
            print(f"  持仓数量: {self.position_size} 只")
            print(f"  换仓阈值: {self.score_threshold:.1%} (提高降低换仓频率)")
            print(f"  交易成本: {self.transaction_cost:.2%}")
            print(f"  最少持有: {self.min_holding_days} 天")
            print(f"  动态止损: {'启用' if self.dynamic_stop_loss else '禁用'}")
            print(f"  强制换仓: {self.force_replace_days} 天")

        total_days = len(self.trading_days)
        start_time = time.time()

        for day_idx, date in enumerate(self.trading_days):
            # 计算市场环境
            self.calculate_market_condition(date)

            # 再平衡
            self.rebalance_fast(date)

            # 计算组合价值
            self.portfolio_value = self.calculate_portfolio_value_batch(date)

            # 记录
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
            print(f"\n⚡ Backtest completed in {elapsed:.2f}s")
            print(f"   Average: {elapsed / total_days * 1000:.1f}ms/day")

        return self.generate_context()

    def generate_context(self):
        """生成回测上下文"""
        df_records = pd.DataFrame(self.daily_records)
        df_trades = pd.DataFrame(self.trade_records)

        sell_trades = df_trades[df_trades['action'] == 'sell']

        final_value = self.portfolio_value
        total_return = (final_value - self.capital_base) / self.capital_base

        if len(sell_trades) > 0:
            win_rate = (sell_trades['pnl'] > 0).sum() / len(sell_trades)
        else:
            win_rate = 0

        return {
            'daily_records': df_records,
            'trade_records': df_trades,
            'final_value': final_value,
            'total_return': total_return,
            'win_rate': win_rate,
            'positions': self.positions
        }


# ========== 便捷接口 ==========

def run_ultimate_fast_backtest(factor_data, price_data, start_date, end_date,
                               capital_base=1000000, position_size=10,
                               stop_loss=-0.15, take_profit=None,
                               score_threshold=0.15, silent=False,
                               max_rebalance_per_day=1, force_replace_days=45,
                               transaction_cost=0.0015, min_holding_days=7,
                               dynamic_stop_loss=True):
    """
    运行优化版回测
    """
    engine = OptimizedBacktest(
        factor_data, price_data, start_date, end_date,
        capital_base, position_size, stop_loss, take_profit, score_threshold,
        max_rebalance_per_day, force_replace_days, transaction_cost,
        min_holding_days, dynamic_stop_loss
    )

    return engine.run(silent=silent)