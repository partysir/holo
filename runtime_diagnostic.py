"""
runtime_diagnostic.py - 运行时诊断工具

在回测运行时实时检测异常
"""

import pandas as pd
import numpy as np


class RuntimeDiagnostic:
    """运行时诊断器"""

    def __init__(self, initial_capital=1000000, max_shares_per_stock=1000000):
        """
        :param initial_capital: 初始资金
        :param max_shares_per_stock: 单只股票最大持仓（股）
        """
        self.initial_capital = initial_capital
        self.max_shares_per_stock = max_shares_per_stock
        self.alerts = []
        self.trade_count = 0

    def check_trade(self, date, stock, action, shares, price, cash_before, cash_after):
        """
        检查单笔交易

        :param date: 交易日期
        :param stock: 股票代码
        :param action: 'buy' 或 'sell'
        :param shares: 股数
        :param price: 价格
        :param cash_before: 交易前现金
        :param cash_after: 交易后现金
        """
        self.trade_count += 1

        # 1. 检查股数
        if shares > self.max_shares_per_stock:
            alert = {
                'date': date,
                'type': 'EXCESSIVE_SHARES',
                'severity': 'HIGH',
                'stock': stock,
                'shares': shares,
                'limit': self.max_shares_per_stock,
                'message': f'股数异常：{shares:,}股 > 上限{self.max_shares_per_stock:,}股'
            }
            self.alerts.append(alert)
            print(f"\n🚨 异常警报 #{len(self.alerts)}")
            print(f"   {alert['message']}")
            print(f"   日期: {date}, 股票: {stock}, 操作: {action}")

        # 2. 检查现金
        if action == 'buy':
            expected_cash = cash_before - (shares * price * 1.0015)  # 含手续费

            if abs(cash_after - expected_cash) > 1000:  # 容差1000元
                alert = {
                    'date': date,
                    'type': 'CASH_MISMATCH',
                    'severity': 'MEDIUM',
                    'stock': stock,
                    'expected': expected_cash,
                    'actual': cash_after,
                    'diff': cash_after - expected_cash,
                    'message': f'现金计算错误：差异¥{abs(cash_after - expected_cash):,.0f}'
                }
                self.alerts.append(alert)
                print(f"\n⚠️  警告 #{len(self.alerts)}")
                print(f"   {alert['message']}")

        # 3. 检查现金为负
        if cash_after < 0:
            alert = {
                'date': date,
                'type': 'NEGATIVE_CASH',
                'severity': 'CRITICAL',
                'cash': cash_after,
                'message': f'现金为负：¥{cash_after:,.0f}'
            }
            self.alerts.append(alert)
            print(f"\n🚨🚨 严重错误 #{len(self.alerts)}")
            print(f"   {alert['message']}")
            print(f"   这不应该发生！检查买入逻辑！")

        # 4. 检查资产膨胀
        if self.trade_count % 100 == 0:  # 每100笔交易检查一次
            if cash_after > self.initial_capital * 100:  # 现金超过初始100倍
                alert = {
                    'date': date,
                    'type': 'ASSET_INFLATION',
                    'severity': 'HIGH',
                    'cash': cash_after,
                    'multiple': cash_after / self.initial_capital,
                    'message': f'资产异常膨胀：现金是初始资金的{cash_after / self.initial_capital:.0f}倍'
                }
                self.alerts.append(alert)
                print(f"\n🚨 资产膨胀警报 #{len(self.alerts)}")
                print(f"   {alert['message']}")

    def check_portfolio(self, date, cash, positions, portfolio_value):
        """
        检查组合状态

        :param date: 日期
        :param cash: 现金
        :param positions: 持仓 {stock: {'shares': ..., 'cost': ...}}
        :param portfolio_value: 组合总价值
        """
        # 1. 检查持仓股数
        for stock, info in positions.items():
            if info['shares'] > self.max_shares_per_stock:
                alert = {
                    'date': date,
                    'type': 'POSITION_EXCESSIVE',
                    'severity': 'HIGH',
                    'stock': stock,
                    'shares': info['shares'],
                    'message': f'持仓异常：{stock} {info["shares"]:,}股'
                }
                self.alerts.append(alert)
                print(f"\n🚨 持仓异常 #{len(self.alerts)}")
                print(f"   {alert['message']}")

        # 2. 检查资产膨胀
        if portfolio_value > self.initial_capital * 100:
            alert = {
                'date': date,
                'type': 'PORTFOLIO_INFLATION',
                'severity': 'CRITICAL',
                'portfolio_value': portfolio_value,
                'multiple': portfolio_value / self.initial_capital,
                'message': f'组合价值异常：¥{portfolio_value:,.0f} (初始的{portfolio_value / self.initial_capital:.0f}倍)'
            }
            self.alerts.append(alert)
            print(f"\n🚨🚨 组合膨胀 #{len(self.alerts)}")
            print(f"   {alert['message']}")
            print(f"   检查是否每次买入都用了全部资金！")

    def get_summary(self):
        """获取诊断摘要"""
        if len(self.alerts) == 0:
            return "✅ 未发现异常"

        summary = f"\n{'=' * 80}\n"
        summary += f"🚨 诊断摘要：发现 {len(self.alerts)} 个问题\n"
        summary += f"{'=' * 80}\n"

        # 按严重程度分类
        critical = [a for a in self.alerts if a['severity'] == 'CRITICAL']
        high = [a for a in self.alerts if a['severity'] == 'HIGH']
        medium = [a for a in self.alerts if a['severity'] == 'MEDIUM']

        if critical:
            summary += f"\n🚨 严重错误 ({len(critical)}个):\n"
            for a in critical[:5]:
                summary += f"   - {a['date']}: {a['message']}\n"

        if high:
            summary += f"\n⚠️  高优先级 ({len(high)}个):\n"
            for a in high[:5]:
                summary += f"   - {a['date']}: {a['message']}\n"

        if medium:
            summary += f"\n💡 中优先级 ({len(medium)}个):\n"
            for a in medium[:3]:
                summary += f"   - {a['message']}\n"

        return summary


def integrate_diagnostic_into_strategy():
    """
    集成诊断器到策略

    使用方法：
    1. 在 factor_based_risk_control.py 开头导入：
       from runtime_diagnostic import RuntimeDiagnostic

    2. 在 __init__ 中初始化：
       self.diagnostic = RuntimeDiagnostic(capital_base)

    3. 在 execute_trade 中买入后添加：
       if action == 'buy':
           self.diagnostic.check_trade(
               date_str, stock, 'buy', shares, price,
               cash_before, self.cash
           )

    4. 在 run 方法结束前添加：
       print(self.diagnostic.get_summary())
    """

    code_snippet = '''
# 在 FactorBasedRiskControl.__init__ 中添加：
self.diagnostic = RuntimeDiagnostic(capital_base)

# 在 execute_trade 中添加（buy分支）：
if action == 'buy':
    cash_before = self.cash
    # ... 原有买入代码 ...
    self.cash -= cost_total

    # 添加诊断
    self.diagnostic.check_trade(
        date_str, stock, 'buy', shares, price,
        cash_before, self.cash
    )

# 在 run 方法结束前添加：
print(self.diagnostic.get_summary())
'''

    print(code_snippet)


if __name__ == "__main__":
    print("运行时诊断工具")
    print("\n使用方法：")
    integrate_diagnostic_into_strategy()