"""
enhanced_strategy.py - 增强版策略系统（资金计算修复版）

关键修复：
✅ 修复仓位计算错误 - 使用可用现金而非总资产
✅ 添加资金充足性检查
✅ 添加调试日志选项
✅ 修复买入金额计算逻辑
✅ 确保资金动态更新
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import time


class EnhancedStrategy:
    """增强版策略引擎（修复版）"""

    def __init__(self, factor_data, price_data, benchmark_data=None,
                 start_date='2023-01-01', end_date='2025-12-05',
                 capital_base=1000000, position_size=10,
                 rebalance_days=1,
                 position_method='equal',  # ✨ 默认改为等权
                 buy_cost=0.0003, sell_cost=0.0003, tax_ratio=0.0005,
                 stop_loss=-0.15, score_threshold=0.15,
                 score_decay_rate=1.0,  # ✨ 默认不衰减
                 force_replace_days=45,
                 debug=False):  # ✨ 新增调试选项
        """
        初始化增强版策略

        :param position_method: 仓位分配方法
            - 'equal': 等权重（推荐，更稳健）
            - 'score_weighted': 评分加权
            - 'score_squared': 评分平方加权
        :param score_decay_rate: 每日评分衰减率（1.0=不衰减）
        :param debug: 是否输出调试信息
        """
        self.factor_data = factor_data
        self.price_data = price_data
        self.benchmark_data = benchmark_data
        self.start_date = start_date
        self.end_date = end_date
        self.capital_base = capital_base
        self.position_size = position_size
        self.rebalance_days = rebalance_days
        self.position_method = position_method
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.tax_ratio = tax_ratio
        self.stop_loss = stop_loss
        self.score_threshold = score_threshold
        self.score_decay_rate = score_decay_rate
        self.force_replace_days = force_replace_days
        self.debug = debug

        # 构建索引
        print(f"\n  ⚡ 构建快速索引...")
        self.price_dict = self._build_price_dict()
        self.factor_dict = self._build_factor_dict()
        self.trading_days = sorted(factor_data['date'].unique())

        # 状态
        self.cash = capital_base
        self.positions = {}
        self.portfolio_value = capital_base
        self.daily_records = []
        self.trade_records = []
        self.days_since_rebalance = 0

        print(f"  ✓ 初始化完成")
        print(f"     交易日: {len(self.trading_days)}")
        print(f"     股票数: {len(set(factor_data['instrument']))}")
        print(f"     调仓周期: {rebalance_days}天")
        print(f"     仓位方法: {position_method}")

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

    def calculate_position_weights(self, stocks_scores):
        """
        ✨ 计算仓位权重

        :param stocks_scores: [(stock, score), ...]
        :return: {stock: weight, ...}
        """
        if self.position_method == 'equal':
            # 等权重
            weight = 1.0 / len(stocks_scores)
            return {stock: weight for stock, score in stocks_scores}

        elif self.position_method == 'score_weighted':
            # 评分加权
            total_score = sum(score for _, score in stocks_scores)
            if total_score == 0:
                weight = 1.0 / len(stocks_scores)
                return {stock: weight for stock, score in stocks_scores}
            return {stock: score / total_score for stock, score in stocks_scores}

        elif self.position_method == 'score_squared':
            # 评分平方加权
            squared_scores = [(stock, score ** 2) for stock, score in stocks_scores]
            total = sum(s for _, s in squared_scores)
            if total == 0:
                weight = 1.0 / len(stocks_scores)
                return {stock: weight for stock, score in stocks_scores}
            return {stock: s / total for stock, s in squared_scores}

        else:
            # 默认等权
            weight = 1.0 / len(stocks_scores)
            return {stock: weight for stock, score in stocks_scores}

    def update_position_scores(self, date):
        """更新持仓评分（带衰减）"""
        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})

        for stock, info in self.positions.items():
            if stock in scores:
                latest_score = scores[stock]
                holding_days = (pd.to_datetime(date_str) -
                              pd.to_datetime(info['entry_date'])).days
                decay_factor = self.score_decay_rate ** holding_days
                info['current_score'] = latest_score * decay_factor
            else:
                info['current_score'] = info.get('entry_score', 0.5)

    def should_rebalance(self, date):
        """判断是否应该调仓"""
        if self.days_since_rebalance >= self.rebalance_days:
            self.days_since_rebalance = 0
            return True
        self.days_since_rebalance += 1
        return False

    def check_stop_loss(self, date):
        """检查止损"""
        date_str = str(date)
        prices = self.price_dict.get(date_str, {})

        to_sell = []
        for stock, info in self.positions.items():
            price = prices.get(stock)
            if not price:
                continue

            pnl_rate = (price - info['cost']) / info['cost']
            if pnl_rate <= self.stop_loss:
                to_sell.append(stock)

        return to_sell

    def execute_trade(self, date, stock, action, shares=None, weight=None, reason='rebalance'):
        """
        执行交易（修复版）

        ✅ 关键修复：使用可用现金而非总资产计算买入金额
        ✅ 修复买入金额计算逻辑
        """
        date_str = str(date)
        price = self.price_dict.get(date_str, {}).get(stock)
        if not price:
            return False

        if action == 'buy':
            # ========== 关键修复：使用可用现金 ==========
            target_value = 0  # 初始化变量
            if weight is not None:
                # ✅ 修复：正确计算可买入金额
                target_value = self.cash * weight
                # 修复买入金额计算逻辑
                available_value = target_value / (1 + self.buy_cost)
                shares = int(available_value / price) if available_value is not None else 0
            else:
                shares = 0

            if self.debug:
                print(f"  [BUY] {date_str} {stock}:")
                print(f"    可用现金: ¥{self.cash:,.0f}")
                print(f"    目标金额: ¥{target_value:,.0f} (权重{weight:.1%})")
                print(f"    计算股数: {shares:,.0f}股")

            # ✨ A股整百股限制
            shares = int(shares / 100) * 100 if shares is not None else 0

            if shares < 100:  # ✨ 至少100股
                if self.debug:
                    print(f"    ❌ 股数不足100股，放弃买入")
                return False

            cost_total = shares * price * (1 + self.buy_cost)

            # ✅ 资金充足性检查
            if cost_total > self.cash:
                if self.debug:
                    print(f"    ⚠️  资金不足: 需要¥{cost_total:,.0f} > 可用¥{self.cash:,.0f}")

                # 按可用资金买入
                available_value = self.cash / (1 + self.buy_cost)
                shares = int(available_value / price)
                shares = int(shares / 100) * 100

                if shares < 100:
                    if self.debug:
                        print(f"    ❌ 调整后仍不足100股，放弃买入")
                    return False

                cost_total = shares * price * (1 + self.buy_cost)

                if self.debug:
                    print(f"    ✓ 调整为: {shares:,.0f}股，¥{cost_total:,.0f}")

            # 执行买入
            self.cash -= cost_total
            score = self.factor_dict.get(date_str, {}).get(stock, 0.5)

            self.positions[stock] = {
                'shares': shares,
                'cost': price,
                'entry_date': date_str,
                'entry_score': score,
                'current_score': score
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

            if self.debug:
                print(f"    ✓ 买入成功: {shares:,.0f}股 @ ¥{price:.2f}")
                print(f"    剩余现金: ¥{self.cash:,.0f}")

            return True

        elif action == 'sell':
            if stock not in self.positions:
                return False

            info = self.positions[stock]
            shares = info['shares']

            # 卖出成本 = 佣金 + 印花税
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

            if self.debug:
                print(f"  [SELL] {date_str} {stock}:")
                print(f"    卖出: {shares:,.0f}股 @ ¥{price:.2f}")
                print(f"    盈亏: ¥{pnl:+,.0f} ({pnl_rate:+.2%})")
                print(f"    回笼现金: ¥{revenue:,.0f}")
                print(f"    当前现金: ¥{self.cash:,.0f}")

            del self.positions[stock]
            return True

        return False

    def rebalance(self, date):
        """
        调仓（修复版）

        ✅ 确保先卖出再买入，避免资金不足
        ✅ 新增：绝对分值过滤，提高确定性
        """
        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})
        prices = self.price_dict.get(date_str, {})

        if self.debug:
            print(f"\n{'='*60}")
            print(f"[调仓] {date_str}")

        # 1. 止损检查
        stop_loss_stocks = self.check_stop_loss(date)
        for stock in stop_loss_stocks:
            self.execute_trade(date, stock, 'sell', reason='stop_loss')

        # 2. 获取候选股票
        if not scores:
            return

        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # ✅ 新增：绝对分值过滤
        # 如果排第一的股票评分都很低(例如<0.6)，说明由于 strict 模式所有模型都不看好
        # 这时候宁愿空仓
        valid_candidates = [c for c in sorted_candidates if c[1] > 0.6] 
        
        # 如果符合条件的太少，就取前几名，但这样能过滤掉垃圾时间
        if not valid_candidates:
             if self.debug: print("  ⚠️ 候选股票评分均过低，放弃买入")
             top_candidates = [] # 空列表，不买入
        else:
             top_candidates = valid_candidates[:50]

        # 3. 评估现有持仓
        to_sell = []
        for stock, info in self.positions.items():
            in_top = any(stock == c[0] for c in top_candidates[:self.position_size])
            current_score = info.get('current_score', info.get('entry_score', 0.5))
            score_decline = info['entry_score'] - current_score
            holding_days = (pd.to_datetime(date_str) -
                          pd.to_datetime(info['entry_date'])).days
            long_and_poor = (holding_days >= self.force_replace_days and
                           current_score < 0.5)

            if not in_top or score_decline > self.score_threshold or long_and_poor:
                to_sell.append(stock)

        # ✅ 关键：先卖出，释放资金
        if self.debug and to_sell:
            print(f"\n  卖出 {len(to_sell)} 只:")
        for stock in to_sell:
            self.execute_trade(date, stock, 'sell', reason='rebalance')

        # 4. 买入新股票
        target_stocks = [c[0] for c in top_candidates[:self.position_size]
                        if c[0] not in self.positions]

        available_slots = self.position_size - len(self.positions)

        if available_slots > 0 and target_stocks:
            target_stocks = target_stocks[:available_slots]
            target_scores = [(s, scores[s]) for s in target_stocks]

            # 计算仓位权重
            weights = self.calculate_position_weights(target_scores)

            if self.debug:
                print(f"\n  买入 {len(target_stocks)} 只:")
                print(f"  可用现金: ¥{self.cash:,.0f}")

            # 买入 - 按权重动态分配资金
            total_weight = sum(weights.values())
            if total_weight > 0:
                # 重新归一化权重
                normalized_weights = {stock: weight/total_weight for stock, weight in weights.items()}

                # 按照归一化权重买入
                for stock, weight in normalized_weights.items():
                    if stock in prices:
                        self.execute_trade(date, stock, 'buy', weight=weight, reason='rebalance')

    def calculate_portfolio_value(self, date):
        """计算组合价值"""
        date_str = str(date)
        prices = self.price_dict.get(date_str, {})

        holdings_value = sum(
            info['shares'] * prices.get(stock, info['cost'])
            for stock, info in self.positions.items()
        )

        total_value = self.cash + holdings_value

        # ✅ 合理性检查
        if total_value > self.capital_base * 1000:  # 超过初始资金1000倍
            print(f"\n⚠️  警告：资产规模异常！")
            print(f"  日期: {date_str}")
            print(f"  现金: ¥{self.cash:,.0f}")
            print(f"  持仓市值: ¥{holdings_value:,.0f}")
            print(f"  总资产: ¥{total_value:,.0f}")
            print(f"  持仓数: {len(self.positions)}")

            # 显示异常持仓
            for stock, info in list(self.positions.items())[:3]:
                price = prices.get(stock, info['cost'])
                value = info['shares'] * price
                print(f"    {stock}: {info['shares']:,.0f}股 × ¥{price:.2f} = ¥{value:,.0f}")

        return total_value

    def run(self, silent=False):
        """运行回测"""
        if not silent:
            print("\n" + "=" * 80)
            print("⚡ 增强版策略引擎（资金计算修复版）")
            print("=" * 80)
            print(f"  持仓数量: {self.position_size} 只")
            print(f"  调仓周期: {self.rebalance_days} 天")
            print(f"  仓位方法: {self.position_method}")
            print(f"  交易成本: 买{self.buy_cost:.2%} + 卖{self.sell_cost:.2%} + 税{self.tax_ratio:.2%}")

        start_time = time.time()
        total_days = len(self.trading_days)

        for day_idx, date in enumerate(self.trading_days):
            # 判断是否调仓日
            if self.should_rebalance(date):
                self.rebalance(date)
            else:
                # 非调仓日也检查止损
                stop_loss_stocks = self.check_stop_loss(date)
                for stock in stop_loss_stocks:
                    self.execute_trade(date, stock, 'sell', reason='stop_loss')

            # 计算组合价值
            self.portfolio_value = self.calculate_portfolio_value(date)

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
            print(f"\n⚡ 回测完成，耗时: {elapsed:.2f}秒")
            print(f"   平均: {elapsed / total_days * 1000:.1f}毫秒/天")

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
            'positions': self.positions,
            'config': {
                'rebalance_days': self.rebalance_days,
                'position_method': self.position_method,
                'score_decay_rate': self.score_decay_rate
            }
        }


# ========== 便捷接口 ==========
def run_enhanced_strategy(factor_data, price_data, start_date, end_date,
                         capital_base=1000000, position_size=10,
                         rebalance_days=5, position_method='equal',
                         buy_cost=0.0003, sell_cost=0.0003, tax_ratio=0.0005,
                         stop_loss=-0.15, score_threshold=0.15,
                         score_decay_rate=1.0, force_replace_days=45,
                         cash_reserve_ratio=0.0,  # 添加现金储备比例参数
                         silent=False, debug=False):
    """运行增强版策略"""
    engine = EnhancedStrategy(
        factor_data, price_data, None,
        start_date, end_date, capital_base, position_size,
        rebalance_days, position_method,
        buy_cost, sell_cost, tax_ratio,
        stop_loss, score_threshold, score_decay_rate, force_replace_days,
        debug
    )
    
    # 如果提供了现金储备比例，则相应调整初始资金
    if cash_reserve_ratio > 0:
        adjusted_capital = int(capital_base * (1 - cash_reserve_ratio))
        engine.capital_base = adjusted_capital
        engine.cash = float(adjusted_capital)
        print(f"  💰 现金储备比例: {cash_reserve_ratio:.1%} (实际投资资金: ¥{adjusted_capital:,.0f})")

    return engine.run(silent=silent)
