"""
factor_based_risk_control_optimized.py - 因子风控 + 最佳现金管理 + 择时模块 (修复版)

核心改进：
✅ 1. 择时模块：大盘均线择时，规避系统性风险
✅ 2. 因子风控：用因子本身做风险控制
✅ 3. 最佳现金管理：动态等权 + 现金保留
✅ 4. 修复调仓逻辑：首日立即调仓 + 强制换仓机制
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import statsmodels.api as sm


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
        ✨ 核心算法：计算最优买入方案（动态等权）

        算法：
        1. 总投资额 = 可用现金 × (1 - 保留比例)
        2. 对每只股票：
           - 单只目标金额 = 剩余投资额 / 剩余股票数
           - 买入后：剩余投资额 -= 实际花费

        :return: {stock: shares}
        """
        if not target_stocks or available_cash <= 0:
            return {}

        # 1️⃣ 计算总投资金额
        total_investment = available_cash * (1 - self.cash_reserve_ratio)

        if self.debug:
            print(f"\n  【最佳现金管理】")
            print(f"    可用现金: ¥{available_cash:,.0f}")
            print(f"    保留比例: {self.cash_reserve_ratio:.1%}")
            print(f"    总投资额: ¥{total_investment:,.0f}")
            print(f"    待买入: {len(target_stocks)}只")

        # 2️⃣ 动态等权买入
        buy_plan = {}
        remaining_investment = total_investment
        remaining_stocks = list(target_stocks)

        for i, stock in enumerate(target_stocks):
            if stock not in prices:
                if self.debug:
                    print(f"    [{i + 1}] ❌ {stock}: 无价格")
                remaining_stocks.remove(stock)
                continue

            price = prices[stock]

            # ✨ 从剩余投资额中等分
            target_amount = remaining_investment / len(remaining_stocks)

            # 计算股数（考虑买入成本）
            shares = int(target_amount / price / (1 + self.buy_cost))

            # A股整百股
            shares = int(shares / 100) * 100

            # 检查最小买入
            actual_amount = shares * price * (1 + self.buy_cost)

            if shares < 100 or actual_amount < self.min_buy_amount:
                if self.debug:
                    print(f"    [{i + 1}] ⚠️  {stock}: 金额不足 (¥{actual_amount:,.0f})")
                remaining_stocks.remove(stock)
                continue

            # 记录买入计划
            buy_plan[stock] = {
                'shares': shares,
                'price': price,
                'amount': actual_amount,
                'target_amount': target_amount
            }

            # ✅ 关键：从剩余投资额中扣除实际花费
            remaining_investment -= actual_amount
            remaining_stocks.remove(stock)

            if self.debug:
                print(f"    [{i + 1}] ✓ {stock}: {shares:,.0f}股 @ ¥{price:.2f} = ¥{actual_amount:,.0f}")

        # 3️⃣ 统计
        if self.debug and buy_plan:
            total_used = sum(info['amount'] for info in buy_plan.values())
            utilization = total_used / available_cash
            avg_amount = total_used / len(buy_plan)

            print(f"\n    【买入计划汇总】")
            print(f"    成功: {len(buy_plan)}/{len(target_stocks)}只")
            print(f"    花费: ¥{total_used:,.0f}")
            print(f"    剩余: ¥{available_cash - total_used:,.0f}")
            print(f"    利用率: {utilization:.2%}")
            print(f"    平均单只: ¥{avg_amount:,.0f}")

        return buy_plan


class FactorBasedRiskControlOptimized:
    """
    因子风控 + 最佳现金管理 + 大盘择时 (完整集成版)

    核心改进：
    1. ✅ 因子风控：评分衰减、排名止损、行业轮动
    2. ✅ 最佳现金管理：动态等权 + 5%现金保留
    3. ✅ 择时模块：大盘均线择时，规避系统性风险
    """

    def __init__(self, factor_data, price_data,
                 # ✨ 新增：基准数据（用于择时）
                 benchmark_data=None,
                 market_ma_period=60, # 60日均线择时

                 start_date='2023-01-01', end_date='2025-12-05',
                 capital_base=1000000, position_size=10,
                 rebalance_days=5,

                 # ========== 最佳现金管理参数 ==========
                 cash_reserve_ratio=0.05,  # 保留5%现金

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
        self.benchmark_data = benchmark_data # 指数数据
        self.market_ma_period = market_ma_period

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
        self.market_signals = self._calculate_market_signals()

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
        self.max_portfolio_value = capital_base
        self.daily_records = []
        self.trade_records = []
        # ✅ 修改：初始化为 rebalance_days，确保第一天就触发调仓
        self.days_since_rebalance = rebalance_days
        self.is_risk_mode = False

        print(f"  ✓ 系统初始化完成")
        print(f"\n  【v2.2 完整集成版配置】")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        if self.benchmark_data is not None:
            print(f"  📈 择时模块: 已启用 ({market_ma_period}日均线)")
        else:
            print(f"  ⚠️  择时模块: 未启用 (无基准数据)")
        print(f"  💰 最佳现金管理:")
        print(f"     • 现金保留: {cash_reserve_ratio:.1%}")
        print(f"     • 资金利用率目标: {1 - cash_reserve_ratio:.1%}")
        print(f"     • 仓位分配: 动态等权")
        print(f"\n  🎯 因子风控:")
        print(f"     • 因子衰减止损: {'✓' if enable_score_decay_stop else '✗'} (评分↓{score_decay_threshold:.0%})")
        print(f"     • 相对排名止损: {'✓' if enable_rank_stop else '✗'} (跌出前{rank_percentile_threshold:.0%})")
        print(f"     • 组合回撤保护: {max_portfolio_drawdown:.1%}")
        print(f"     • 行业轮动: {'✓' if enable_industry_rotation else '✗'}")
        print(f"     • 极端亏损保护: 单股{extreme_loss_threshold:.0%} | 组合{portfolio_loss_threshold:.0%}")
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
        """
        升级版：使用 RSRS (阻力支撑相对强度) 进行大盘择时
        """
        signals = {}
        if self.benchmark_data is None:
            return signals
        
        df = self.benchmark_data.copy().sort_values('date')
        
        # RSRS 参数
        N = 18  # 回归周期
        M = 600 # 均值周期
        
        rsrs_values = []
        
        # 滚动计算 RSRS 斜率
        highs = df['high'].values
        lows = df['low'].values
        
        for i in range(len(df)):
            if i < N:
                rsrs_values.append(0)
                continue
                
            y = highs[i-N:i]
            x = lows[i-N:i]
            x = sm.add_constant(x)
            
            model = sm.OLS(y, x)
            results = model.fit()
            beta = results.params[1] # 斜率
            rsrs_values.append(beta)
            
        df['rsrs'] = rsrs_values
        
        # 标准化 RSRS (RSRS_Z)
        df['rsrs_mean'] = df['rsrs'].rolling(window=M).mean()
        df['rsrs_std'] = df['rsrs'].rolling(window=M).std()
        df['rsrs_z'] = (df['rsrs'] - df['rsrs_mean']) / df['rsrs_std']
        
        # 信号生成: RSRS_Z > 0.7 买入, RSRS_Z < -0.7 卖出/风控
        # 平滑处理：结合右侧趋势
        for i, row in df.iterrows():
            date_str = str(row['date'])
            z_score = row['rsrs_z']
            
            # 激进择时：RSRS分值大于0.7看多，小于-0.7看空，中间震荡
            if pd.isna(z_score):
                signals[date_str] = True
            else:
                signals[date_str] = z_score > -0.7 # 只要不是极弱势，都允许做多
                
        return signals

    def check_market_regime(self, date_str):
        """
        检查市场状态
        返回: True(市场健康/看多), False(市场风险/看空)
        """
        if not self.market_signals:
            return True
        return self.market_signals.get(date_str, True)

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

            # ========== ✅ 新增：强制流动性换仓 ==========
            # 如果持有超过 2 个调仓周期（比如10天）且收益微薄或亏损，强制卖出
            # 这能强制策略“动起来”，避免死拿僵尸股
            if holding_days >= (self.rebalance_days * 2) and pnl_rate < 0.02:
                to_sell.append((stock, 'force_turnover'))
                if self.debug:
                    print(f"    ♻️ 强制换仓: {stock} (持有{holding_days}天, 收益{pnl_rate:.2%} < 2%)")
                continue
            # ==========================================

            # 3. 长期持有亏损检查
            if holding_days >= 30 and pnl_rate < -0.10:
                to_sell.append((stock, 'long_hold_loss'))
                if self.debug:
                    print(f"    ⚠️  长期持有亏损: {stock} (持有{holding_days}天, 亏损{pnl_rate:.2%})")
                continue

            # 4. 极端亏损保护
            if self.check_extreme_loss(stock, price, info):
                to_sell.append((stock, 'extreme_loss'))
                continue

        # 5. 组合回撤控制
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

    # ========== 交易执行方法 ==========

    def execute_sell(self, date, stock, reason='rebalance'):
        """执行卖出"""
        date_str = str(date)
        price = self.price_dict.get(date_str, {}).get(stock)
        if not price or stock not in self.positions:
            return False

        info = self.positions[stock]
        shares = info['shares']

        total_cost_rate = self.sell_cost + self.tax_ratio
        revenue = shares * price * (1 - total_cost_rate)
        self.cash += revenue

        # 修复盈亏计算的一致性问题
        # 使用买入时记录的成本（已包含交易费用）直接计算盈亏
        cost_basis = info['cost'] * shares  # 成本基础 = 买入价格 × 股数
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
            'entry_date': info['entry_date'],
            'holding_days': (pd.to_datetime(date_str) -
                             pd.to_datetime(info['entry_date'])).days
        })

        del self.positions[stock]

        if self.debug:
            print(f"    ✓ 卖出: {stock} {shares:,.0f}股 @ ¥{price:.2f}, 盈亏{pnl_rate:+.2%}, 原因: {reason}")

        return True

    def execute_buy_batch(self, date, buy_plan):
        """✨ 批量执行买入（使用最优买入计划）"""
        date_str = str(date)
        scores = self.factor_dict.get(date_str, {})

        for stock, plan_info in buy_plan.items():
            shares = plan_info['shares']
            price = plan_info['price']
            amount = plan_info['amount']

            # 执行买入
            self.cash -= amount
            score = scores.get(stock, 0.5)

            # 修复：记录包含交易成本的基础成本价
            cost_basis = amount / shares  # 包含交易成本的实际成本价

            self.positions[stock] = {
                'shares': shares,
                'cost': cost_basis,  # 使用包含交易成本的成本价
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

            if self.debug:
                print(f"    ✓ 买入: {stock} {shares:,.0f}股 @ ¥{price:.2f} = ¥{amount:,.0f}")

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

        # 1. 风险检查 (风控卖出始终执行，不受择时影响)
        risk_conditions = self.check_risk_conditions(date)
        for stock, reason in risk_conditions:
            self.execute_sell(date, stock, reason=reason)

        # 2. 择时检查：如果大盘不好，只卖不买
        is_market_good = self.check_market_regime(date_str)
        if not is_market_good:
            if self.debug:
                print(f"  🛑 大盘择时: 市场处于下行趋势 (价格 < MA{self.market_ma_period})，暂停买入！")

            # 在熊市中，可以选择只进行卖出操作，不再进行后续的买入逻辑
            # 这里直接退出函数，不再执行买入
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
        """计算已实现盈亏"""
        sell_trades = [record for record in self.trade_records if record['action'] == 'sell']
        return sum(record['pnl'] for record in sell_trades)

    def calculate_unrealized_pnl(self, date):
        """计算未实现盈亏"""
        date_str = str(date)
        prices = self.price_dict.get(date_str, {})

        unrealized_pnl = 0
        for stock, info in self.positions.items():
            price = prices.get(stock, info['cost'])
            cost_basis = info['cost'] * info['shares']
            market_value = price * info['shares']
            unrealized_pnl += market_value - cost_basis

        return unrealized_pnl

    def run(self, silent=False):
        """运行回测"""
        if not silent:
            print("\n" + "=" * 80)
            print("⚡ 因子风控 + 最佳现金管理 + 大盘择时 v2.1")
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
                    self.execute_sell(date, stock, reason=reason)

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

        # 计算总盈亏明细
        total_realized_pnl = sell_trades['pnl'].sum() if len(sell_trades) > 0 else 0

        return {
            'daily_records': df_records,
            'trade_records': df_trades,
            'final_value': final_value,
            'total_return': total_return,
            'win_rate': win_rate,
            'positions': self.positions,
            'total_realized_pnl': total_realized_pnl
        }


# ========== 便捷接口 ==========
def run_factor_based_strategy_v2(factor_data, price_data,
                                 # 新增：基准数据
                                 benchmark_data=None,
                                 # 原有参数
                                 start_date='2023-01-01', end_date='2025-12-05',
                                 capital_base=1000000, position_size=10,
                                 rebalance_days=5, cash_reserve_ratio=0.05,
                                 **kwargs):
    """运行因子风控 + 最佳现金管理策略（v2.1 含择时）"""
    engine = FactorBasedRiskControlOptimized(
        factor_data, price_data,
        benchmark_data=benchmark_data, # 传入基准数据
        start_date=start_date, end_date=end_date, capital_base=capital_base,
        position_size=position_size, rebalance_days=rebalance_days,
        cash_reserve_ratio=cash_reserve_ratio, **kwargs
    )

    return engine.run()