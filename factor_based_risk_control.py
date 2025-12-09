"""
factor_based_risk_control.py - 基于因子的风险控制

核心理念：
✅ 相信多因子选股能力
✅ 不添加额外技术指标
✅ 用因子本身做风险控制
✅ 统计套利 + 风控平衡

优化方向：
1. 因子衰减止损 - 评分大幅下降才止损
2. 相对排名止损 - 跌出前N%才止损
3. 行业轮动保护 - 行业整体走弱才止损
4. 组合层面风控 - 而非个股止损
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


class FactorBasedRiskControl:
    """
    基于因子的风险控制策略

    核心原则：
    1. 信任因子评分系统
    2. 避免过度交易
    3. 用因子信号做风控
    4. 保持统计套利优势
    """

    def __init__(self, factor_data, price_data,
                 start_date='2023-01-01', end_date='2025-12-05',
                 capital_base=1000000, position_size=10,
                 rebalance_days=5,
                 position_method='equal',

                 # ========== 基于因子的风控参数 ==========

                 # 1. 因子衰减止损
                 enable_score_decay_stop=True,  # 启用评分衰减止损
                 score_decay_threshold=0.30,  # 评分下降30%止损
                 min_holding_days=5,  # 最少持有5天（避免过早止损）

                 # 2. 相对排名止损
                 enable_rank_stop=True,  # 启用排名止损
                 rank_percentile_threshold=0.70,  # 跌出前70%止损

                 # 3. 组合层面风控
                 max_portfolio_drawdown=-0.15,  # 组合回撤-15%降仓
                 reduce_position_ratio=0.5,  # 降仓到50%

                 # 4. 行业风控
                 enable_industry_rotation=True,  # 启用行业轮动
                 max_industry_weight=0.40,  # 单行业最大40%

                 # 5. 极端情况保护
                 extreme_loss_threshold=-0.20,  # 单股极端亏损-20%
                 portfolio_loss_threshold=-0.25,  # 组合极端亏损-25%

                 # 其他参数
                 buy_cost=0.0003,
                 sell_cost=0.0003,
                 tax_ratio=0.0005,
                 debug=False):

        self.factor_data = factor_data
        self.price_data = price_data
        self.start_date = start_date
        self.end_date = end_date
        self.capital_base = capital_base
        self.position_size = position_size
        self.rebalance_days = rebalance_days
        self.position_method = position_method

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

        # 构建索引
        print("\n  ⚡ 构建因子风控系统...")
        self.price_dict = self._build_price_dict()
        self.factor_dict = self._build_factor_dict()
        self.trading_days = sorted(factor_data['date'].unique())

        # 行业信息
        if 'industry' in factor_data.columns:
            self.industry_dict = self._build_industry_dict()
        else:
            self.industry_dict = None
            self.enable_industry_rotation = False

        # 状态
        self.cash = capital_base
        self.positions = {}
        self.portfolio_value = capital_base
        self.max_portfolio_value = capital_base  # 记录历史最高净值
        self.daily_records = []
        self.trade_records = []
        self.days_since_rebalance = 0
        self.is_risk_mode = False  # 风险模式标志

        print(f"  ✓ 因子风控系统初始化完成")
        print(f"\n  【风控配置 - 基于因子】")
        print(f"    ✓ 因子衰减止损: {'启用' if enable_score_decay_stop else '关闭'}")
        if enable_score_decay_stop:
            print(f"      评分下降>{score_decay_threshold:.0%} 且 持有>{min_holding_days}天")
        print(f"    ✓ 相对排名止损: {'启用' if enable_rank_stop else '关闭'}")
        if enable_rank_stop:
            print(f"      跌出前{rank_percentile_threshold:.0%}")
        print(f"    ✓ 组合回撤保护: {max_portfolio_drawdown:.1%}")
        print(f"    ✓ 行业轮动: {'启用' if enable_industry_rotation else '关闭'}")
        print(f"    ✓ 极端亏损保护: 单股{extreme_loss_threshold:.0%} | 组合{portfolio_loss_threshold:.0%}")
        print(f"\n  【核心理念】")
        print(f"    • 相信多因子选股能力")
        print(f"    • 用因子信号做风控")
        print(f"    • 避免过度交易")
        print(f"    • 保持统计套利优势")

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

    def get_score_rank_percentile(self, stock, date_str, scores):
        """
        获取股票在当日的评分排名百分位

        返回：0-1之间的值，1表示最高分
        """
        if stock not in scores:
            return 0.5

        stock_score = scores[stock]
        sorted_scores = sorted(scores.values(), reverse=True)

        rank = sorted_scores.index(stock_score) + 1
        percentile = 1 - (rank / len(sorted_scores))

        return percentile

    def check_score_decay_stop(self, stock, current_score, info, holding_days):
        """
        ✅ 因子衰减止损

        逻辑：
        1. 评分相对买入时大幅下降
        2. 持有时间足够长（避免过早判断）
        3. 说明因子认为股票变差了
        """
        if not self.enable_score_decay_stop:
            return False

        # 必须持有足够长时间
        if holding_days < self.min_holding_days:
            return False

        entry_score = info.get('entry_score', 0.5)

        # 评分下降幅度
        if entry_score > 0:
            score_change = (current_score - entry_score) / entry_score
        else:
            score_change = 0

        # 评分大幅下降
        should_stop = score_change < -self.score_decay_threshold

        if should_stop and self.debug:
            print(f"    因子衰减止损: {stock}")
            print(f"      买入评分: {entry_score:.4f}")
            print(f"      当前评分: {current_score:.4f}")
            print(f"      下降幅度: {score_change:.2%}")

        return should_stop

    def check_rank_stop(self, stock, date_str, scores):
        """
        ✅ 相对排名止损

        逻辑：
        1. 股票在当前评分体系中排名靠后
        2. 说明相对其他股票变差了
        3. 应该换成排名更高的股票
        """
        if not self.enable_rank_stop:
            return False

        percentile = self.get_score_rank_percentile(stock, date_str, scores)

        # 跌出前N%
        should_stop = percentile < (1 - self.rank_percentile_threshold)

        if should_stop and self.debug:
            print(f"    相对排名止损: {stock}")
            print(f"      当前排名: 前{percentile:.1%}")
            print(f"      阈值: 前{self.rank_percentile_threshold:.1%}")

        return should_stop

    def check_extreme_loss(self, stock, current_price, info):
        """
        ✅ 极端亏损保护

        逻辑：
        虽然相信因子，但也要防止极端情况
        单股亏损超过-20%，可能是黑天鹅事件
        """
        cost = info['cost']
        loss_rate = (current_price - cost) / cost

        should_stop = loss_rate < self.extreme_loss_threshold

        if should_stop and self.debug:
            print(f"    极端亏损保护: {stock}")
            print(f"      亏损幅度: {loss_rate:.2%}")

        return should_stop

    def check_portfolio_drawdown(self):
        """
        ✅ 组合层面回撤控制

        逻辑：
        1. 组合净值回撤超过阈值
        2. 可能市场环境不利
        3. 降低仓位，控制风险
        """
        if self.portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = self.portfolio_value

        drawdown = (self.portfolio_value - self.max_portfolio_value) / self.max_portfolio_value

        # 回撤超过阈值，进入风险模式
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
        """
        获取当前持仓的行业权重
        """
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
        """
        检查行业集中度

        避免单一行业过度集中
        """
        if not self.enable_industry_rotation or not self.industry_dict:
            return True

        stock_industry = self.industry_dict.get(date_str, {}).get(stock, 'Unknown')
        industry_weights = self.get_industry_weights(date_str)

        current_weight = industry_weights.get(stock_industry, 0)

        # 该行业权重已经过高
        if current_weight >= self.max_industry_weight:
            if self.debug:
                print(f"    行业集中度过高: {stock} ({stock_industry}: {current_weight:.1%})")
            return False

        return True

    def check_risk_conditions(self, date):
        """
        ✅ 综合风险检查（基于因子）
        """
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

            # 1. 因子衰减止损
            if self.check_score_decay_stop(stock, current_score, info, holding_days):
                to_sell.append((stock, 'score_decay'))
                continue

            # 2. 相对排名止损
            if self.check_rank_stop(stock, date_str, scores):
                to_sell.append((stock, 'rank_stop'))
                continue

            # 3. 极端亏损保护
            if self.check_extreme_loss(stock, price, info):
                to_sell.append((stock, 'extreme_loss'))
                continue

        # 4. 组合回撤控制
        in_risk_mode = self.check_portfolio_drawdown()

        if in_risk_mode:
            # 风险模式：卖出评分最低的股票，降低仓位
            current_positions = [
                (stock, scores.get(stock, 0.5))
                for stock in self.positions.keys()
            ]
            current_positions.sort(key=lambda x: x[1])

            # 降仓到目标比例
            target_position_count = int(self.position_size * self.reduce_position_ratio)
            stocks_to_reduce = len(self.positions) - target_position_count

            if stocks_to_reduce > 0:
                for stock, _ in current_positions[:stocks_to_reduce]:
                    if (stock, 'score_decay') not in to_sell and \
                            (stock, 'rank_stop') not in to_sell and \
                            (stock, 'extreme_loss') not in to_sell:
                        to_sell.append((stock, 'risk_mode_reduce'))

        return to_sell

    def should_rebalance(self, date):
        """判断是否调仓"""
        if self.days_since_rebalance >= self.rebalance_days:
            self.days_since_rebalance = 0
            return True
        self.days_since_rebalance += 1
        return False

    def execute_trade(self, date, stock, action, weight=None, reason='rebalance'):
        """执行交易"""
        date_str = str(date)
        price = self.price_dict.get(date_str, {}).get(stock)
        if not price:
            return False

        if action == 'buy':
            # 检查行业集中度
            if not self.check_industry_concentration(stock, date_str):
                return False

            if weight is not None:
                target_value = self.cash * weight
                shares = int(target_value / price / (1 + self.buy_cost))
            else:
                return False

            shares = int(shares / 100) * 100

            if shares < 100:
                return False

            cost_total = shares * price * (1 + self.buy_cost)

            if cost_total > self.cash:
                shares = int(self.cash / price / (1 + self.buy_cost))
                shares = int(shares / 100) * 100
                if shares < 100:
                    return False
                cost_total = shares * price * (1 + self.buy_cost)

            self.cash -= cost_total
            score = self.factor_dict.get(date_str, {}).get(stock, 0.5)

            self.positions[stock] = {
                'shares': shares,
                'cost': price,
                'entry_date': date_str,
                'entry_score': score
            }

            self.trade_records.append({
                'date': date_str,
                'stock': stock,
                'action': 'buy',
                'price': price,
                'shares': shares,
                'amount': cost_total,
                'reason': reason
            })

            return True

        elif action == 'sell':
            if stock not in self.positions:
                return False

            info = self.positions[stock]
            shares = info['shares']

            total_cost_rate = self.sell_cost + self.tax_ratio
            revenue = shares * price * (1 - total_cost_rate)
            self.cash += revenue

            pnl = revenue - (shares * info['cost'] * (1 + self.buy_cost))
            pnl_rate = (price - info['cost']) / info['cost']

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
                'holding_days': (pd.to_datetime(date_str) -
                                 pd.to_datetime(info['entry_date'])).days
            })

            del self.positions[stock]
            return True

        return False

    def rebalance(self, date):
        """调仓"""
        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})

        if self.debug:
            print(f"\n[调仓] {date_str}")

        # 1. 风险检查（基于因子）
        risk_conditions = self.check_risk_conditions(date)
        for stock, reason in risk_conditions:
            self.execute_trade(date, stock, 'sell', reason=reason)

        # 2. 获取候选股票
        if not scores:
            return

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = sorted_candidates[:50]

        # 3. 评估现有持仓
        to_sell = []
        for stock, info in list(self.positions.items()):
            current_score = scores.get(stock, 0)
            in_top = any(stock == c[0] for c in top_candidates[:self.position_size])

            # 只有不在top中才考虑卖出
            if not in_top:
                to_sell.append(stock)

        for stock in to_sell:
            self.execute_trade(date, stock, 'sell', reason='rebalance')

        # 4. 买入新股票
        # 风险模式下减少仓位
        if self.is_risk_mode:
            target_size = int(self.position_size * self.reduce_position_ratio)
        else:
            target_size = self.position_size

        target_stocks = [c[0] for c in top_candidates[:target_size]
                         if c[0] not in self.positions]

        available_slots = target_size - len(self.positions)

        if available_slots > 0 and target_stocks:
            target_stocks = target_stocks[:available_slots]
            weight = 1.0 / len(target_stocks)

            for stock in target_stocks:
                self.execute_trade(date, stock, 'buy', weight=weight, reason='rebalance')

    def calculate_portfolio_value(self, date):
        """计算组合价值"""
        date_str = str(date)
        prices = self.price_dict.get(date_str, {})

        holdings_value = sum(
            info['shares'] * prices.get(stock, info['cost'])
            for stock, info in self.positions.items()
        )

        return self.cash + holdings_value

    def run(self, silent=False):
        """运行回测"""
        if not silent:
            print("\n" + "=" * 80)
            print("⚡ 基于因子的风险控制策略")
            print("=" * 80)

        import time
        start_time = time.time()

        for date in self.trading_days:
            if self.should_rebalance(date):
                self.rebalance(date)
            else:
                # 非调仓日也检查风险
                risk_conditions = self.check_risk_conditions(date)
                for stock, reason in risk_conditions:
                    self.execute_trade(date, stock, 'sell', reason=reason)

            self.portfolio_value = self.calculate_portfolio_value(date)

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


def run_factor_based_strategy(factor_data, price_data, start_date, end_date,
                              capital_base=1000000, position_size=10,
                              rebalance_days=5, **kwargs):
    """运行基于因子的风控策略"""
    engine = FactorBasedRiskControl(
        factor_data, price_data,
        start_date, end_date, capital_base, position_size,
        rebalance_days, **kwargs
    )

    return engine.run()