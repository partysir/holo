"""
test_position_calculation.py - 测试仓位计算逻辑

独立测试买入逻辑，找出问题根源
"""


def test_buy_logic():
    """测试买入逻辑"""
    print("=" * 80)
    print("测试仓位计算逻辑")
    print("=" * 80)

    # 模拟初始状态
    initial_cash = 1000000  # 100万
    cash = initial_cash
    positions = {}

    # 模拟第1次调仓：买入3只股票
    print("\n【第1次调仓】买入3只股票")
    print(f"可用现金: ¥{cash:,.0f}")

    stocks_to_buy = [
        ('600000.SH', 10.00),
        ('000001.SZ', 15.00),
        ('600036.SH', 20.00)
    ]

    # 方法1：错误的方法（会导致资产膨胀）
    print("\n方法1（错误）：每只都用 总现金 * 33.3%")
    cash_wrong = cash
    for stock, price in stocks_to_buy:
        weight = 1.0 / len(stocks_to_buy)  # 33.3%
        target_value = cash * weight  # ❌ 错误：每次都用初始现金
        shares = int(target_value / price / 1.0003)
        shares = int(shares / 100) * 100
        cost = shares * price * 1.0003
        cash_wrong -= cost

        print(f"  {stock}: {shares:,}股 @ ¥{price:.2f} = ¥{cost:,.0f}")

    print(f"  剩余现金: ¥{cash_wrong:,.0f}")
    print(f"  总支出: ¥{cash - cash_wrong:,.0f}")
    print(f"  ❌ 问题：总支出 = {(cash - cash_wrong) / cash:.1%} > 100%（不合理！）")

    # 方法2：正确的方法
    print("\n方法2（正确）：每只都用 当前现金 * 33.3%")
    cash_correct = cash
    for i, (stock, price) in enumerate(stocks_to_buy):
        weight = 1.0 / len(stocks_to_buy)
        target_value = cash_correct * weight  # ✅ 正确：用当前现金
        shares = int(target_value / price / 1.0003)
        shares = int(shares / 100) * 100
        cost = shares * price * 1.0003
        cash_correct -= cost

        print(f"  {stock}: {shares:,}股 @ ¥{price:.2f} = ¥{cost:,.0f}, 剩余¥{cash_correct:,.0f}")

    print(f"  最终现金: ¥{cash_correct:,.0f}")
    print(f"  总支出: ¥{cash - cash_correct:,.0f}")
    print(f"  ✅ 正确：总支出 = {(cash - cash_correct) / cash:.1%} < 100%")

    # 模拟多次调仓
    print("\n" + "=" * 80)
    print("【多次调仓模拟】")
    print("=" * 80)

    cash = initial_cash

    for round_num in range(1, 4):
        print(f"\n第{round_num}次调仓:")
        print(f"  调仓前: ¥{cash:,.0f}")

        # 假设每次赚10%
        cash = cash * 1.1
        print(f"  盈利后: ¥{cash:,.0f}")

        # 卖出所有持仓（假设）
        # 买入3只新股票（错误方法）
        initial_round_cash = cash
        for stock, price in stocks_to_buy:
            weight = 1.0 / len(stocks_to_buy)
            target_value = initial_round_cash * weight  # ❌ 用初始现金
            shares = int(target_value / price / 1.0003)
            shares = int(shares / 100) * 100
            cost = shares * price * 1.0003
            cash -= cost

        print(f"  调仓后: ¥{cash:,.0f}")
        print(f"  现金变化: {(cash - initial_round_cash) / initial_round_cash:.1%}")

        if cash < 0:
            print(f"  🚨 现金为负！")
            break

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    print("""
如果每次买入时都用 `初始现金 * 权重`，会导致：
1. 第1只股票：用 100% * 33% = 33%
2. 第2只股票：用 100% * 33% = 33%
3. 第3只股票：用 100% * 33% = 33%
总计：99% > 实际可用（因为第1只已经扣掉33%）

正确做法：
1. 第1只：用 100% * 33% = 33%，剩余67%
2. 第2只：用 67% * 50% = 33.5%，剩余33.5%（从剩余2只中分）
3. 第3只：用 33.5% * 100% = 33.5%

更简单的正确做法：
1. 计算每只目标金额：总现金 / 股票数 = 每只33.3%
2. 按顺序买入，每次从当前现金扣除
""")


def test_actual_case():
    """测试实际案例"""
    print("\n" + "=" * 80)
    print("【实际案例分析】")
    print("=" * 80)

    # 您的实际持仓
    holdings = [
        ('600200.SH', 8221600, 0.98),  # 820万股
        ('000002.SZ', 1347700, 4.97),  # 134万股
        ('301030.SZ', 217000, 15.50),  # 21.7万股
    ]

    print("\n实际持仓分析:")
    total_value = 0
    for stock, shares, price in holdings:
        value = shares * price
        total_value += value
        print(f"  {stock}: {shares:,}股 @ ¥{price:.2f} = ¥{value:,.0f}")

    print(f"\n  总市值: ¥{total_value:,.0f}")

    # 推算买入时的现金
    if total_value > 1000000:
        print(f"\n  ❌ 问题：市值{total_value:,.0f} >> 初始资金100万")
        print(f"  可能原因：")
        print(f"    1. 每次买入都用了全部现金（没有逐步扣除）")
        print(f"    2. 或者多次累积盈利后的正常增长")

        # 验证是否是正常盈利
        implied_return = (total_value / 1000000) - 1
        print(f"\n  如果是正常盈利，意味着总收益率: {implied_return:+.1%}")

        if implied_return > 10:  # 1000%
            print(f"  ❌ 收益率{implied_return:.0%}不合理，应该是仓位计算错误")


if __name__ == "__main__":
    test_buy_logic()
    test_actual_case()

    print("\n" + "=" * 80)
    print("🔧 修复建议")
    print("=" * 80)
    print("""
在 factor_based_risk_control.py 的 rebalance() 方法中：

# 当前可能的错误代码：
for stock in target_stocks:
    weight = 1.0 / len(target_stocks)
    self.execute_trade(date, stock, 'buy', weight=weight)
    # ❌ 问题：execute_trade 内部用的是 self.cash * weight
    #         但self.cash在第一次买入后应该已经减少了
    #         如果没有减少，就会重复使用全部现金

# 修复方案：确保 execute_trade 中的 self.cash -= cost_total 生效
# 并在 rebalance 中添加调试日志：

if self.debug:
    print(f"买入前现金: ¥{self.cash:,.0f}")

for stock in target_stocks:
    weight = 1.0 / len(target_stocks)
    success = self.execute_trade(date, stock, 'buy', weight=weight)
    if self.debug and success:
        print(f"  买入后现金: ¥{self.cash:,.0f}")
""")