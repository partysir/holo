"""
诊断脚本 - 分析异常交易记录

使用方法：
1. 从你的回测结果中导出 trade_records.csv
2. 运行此脚本分析问题
"""

import pandas as pd
import numpy as np


def diagnose_trade_records(trade_records_path='./reports/2025-12-14/trade_history_fixed.csv'):
    """诊断交易记录"""

    print("=" * 80)
    print("🔍 交易记录诊断分析")
    print("=" * 80)

    # 读取交易记录
    df = pd.read_csv(trade_records_path, encoding='utf-8-sig')

    # 基本统计
    print(f"\n📊 基本信息:")
    print(f"  总记录数: {len(df)}")
    print(f"  日期范围: {df['日期'].min()} ~ {df['日期'].max()}")

    # 分析买入交易
    buys = df[df['操作'] == '买入']
    sells = df[df['操作'] == '卖出']

    print(f"\n💰 买入交易分析:")
    print(f"  买入次数: {len(buys)}")

    if len(buys) > 0:
        # 检查异常大额买入
        buys_sorted = buys.sort_values('净盈亏', ascending=False)

        print(f"\n  单笔买入费用统计:")
        print(f"    平均: ¥{buys['净盈亏'].mean():,.2f}")
        print(f"    最大: ¥{buys['净盈亏'].max():,.2f}")
        print(f"    最小: ¥{buys['净盈亏'].min():,.2f}")

        # 找出异常交易（费用超过10万的）
        abnormal_buys = buys[abs(buys['净盈亏']) > 100000]

        if len(abnormal_buys) > 0:
            print(f"\n  ⚠️  发现 {len(abnormal_buys)} 笔异常高额买入:")
            print(abnormal_buys[['日期', '股票', '数量', '买入价', '交易费用', '净盈亏']].head(10))

    print(f"\n📈 卖出交易分析:")
    print(f"  卖出次数: {len(sells)}")

    if len(sells) > 0:
        print(f"\n  盈亏统计:")
        print(f"    总净盈亏: ¥{sells['净盈亏'].sum():,.2f}")
        print(f"    平均盈亏: ¥{sells['净盈亏'].mean():,.2f}")
        print(f"    最大盈利: ¥{sells['净盈亏'].max():,.2f}")
        print(f"    最大亏损: ¥{sells['净盈亏'].min():,.2f}")

        # 检查异常盈亏（单笔超过50万）
        abnormal_profits = sells[sells['净盈亏'] > 500000]
        abnormal_losses = sells[sells['净盈亏'] < -500000]

        if len(abnormal_profits) > 0:
            print(f"\n  ⚠️  发现 {len(abnormal_profits)} 笔异常高盈利:")
            top_profits = abnormal_profits.nlargest(5, '净盈亏')
            print(top_profits[['日期', '股票', '数量', '买入价', '卖出价', '净盈亏', '收益率']])

            # 分析异常盈利的原因
            print(f"\n  🔍 异常盈利分析:")
            for idx, row in top_profits.iterrows():
                shares = row['数量']
                buy_price = row['买入价']
                sell_price = row['卖出价']
                expected_shares = 1000000 * 0.1 / buy_price  # 假设10%仓位，100万本金

                print(f"\n    {row['股票']} ({row['日期']}):")
                print(f"      买入股数: {shares:,.0f} 股")
                print(f"      预期股数: {expected_shares:,.0f} 股 (假设10%仓位)")
                print(f"      股数倍数: {shares / expected_shares:.1f}x")
                print(f"      买入价: ¥{buy_price:.2f}")
                print(f"      卖出价: ¥{sell_price:.2f}")
                print(f"      总成本: ¥{shares * buy_price:,.0f}")
                print(f"      总收入: ¥{shares * sell_price:,.0f}")

                if shares > expected_shares * 2:
                    print(f"      ❌ 股数异常：超过预期 {shares / expected_shares:.1f} 倍")

        if len(abnormal_losses) > 0:
            print(f"\n  ⚠️  发现 {len(abnormal_losses)} 笔异常高亏损:")
            print(abnormal_losses[['日期', '股票', '数量', '买入价', '卖出价', '净盈亏', '收益率']].head(5))

    # 检查股数是否合理
    print(f"\n🔢 股数合理性检查:")

    if '数量' in df.columns:
        max_shares = df['数量'].max()
        avg_shares = df['数量'].mean()

        print(f"  最大股数: {max_shares:,.0f}")
        print(f"  平均股数: {avg_shares:,.0f}")

        # 假设初始资金100万，单只10%仓位
        # 对于10元的股票，最多买10万/10 = 1万股
        # 对于1元的股票，最多买10万/1 = 10万股
        # 所以正常情况下，股数应该在 1000 - 100,000 范围内

        abnormal_high_shares = df[df['数量'] > 1000000]  # 超过100万股

        if len(abnormal_high_shares) > 0:
            print(f"\n  ⚠️  发现 {len(abnormal_high_shares)} 笔异常高股数（>100万股）:")
            print(abnormal_high_shares[['日期', '股票', '操作', '数量', '买入价']].head(10))

            # 分析第一笔异常交易
            if len(abnormal_high_shares) > 0:
                first_abnormal = abnormal_high_shares.iloc[0]
                print(f"\n  📍 第一笔异常交易详情:")
                print(f"    日期: {first_abnormal['日期']}")
                print(f"    股票: {first_abnormal['股票']}")
                print(f"    操作: {first_abnormal['操作']}")
                print(f"    股数: {first_abnormal['数量']:,.0f}")

                if first_abnormal['操作'] == '买入':
                    print(f"    价格: ¥{first_abnormal['买入价']:.2f}")
                    total_cost = first_abnormal['数量'] * first_abnormal['买入价']
                    print(f"    总成本: ¥{total_cost:,.0f}")
                    print(f"    ❌ 这笔交易需要 {total_cost / 1000000:.1f} 百万资金！")

    print(f"\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    # 运行诊断
    try:
        diagnose_trade_records()
    except FileNotFoundError:
        print("❌ 文件未找到，请检查路径")
        print("默认路径: ./reports/2025-12-14/trade_history_fixed.csv")
    except Exception as e:
        print(f"❌ 诊断失败: {e}")
        import traceback

        traceback.print_exc()