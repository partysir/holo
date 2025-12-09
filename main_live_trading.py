"""
main_live_trading.py - 实盘交易版

配置:
- 5日调仓-等权（最高胜率 53.24%）
- 每日检查但不一定交易
- 生成持仓建议CSV
- 支持国信证券接口
"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

import tushare as ts

TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"
ts.set_token(TUSHARE_TOKEN)

from data_module import DataCache
from data_module_incremental import load_data_with_incremental_update
from enhanced_strategy import run_enhanced_strategy


# ========== 实盘配置 ==========
class LiveTradingConfig:
    """实盘交易配置"""

    # 策略参数（根据回测最优结果）
    REBALANCE_DAYS = 5  # ✨ 5日调仓
    POSITION_METHOD = 'equal'  # ✨ 等权
    POSITION_SIZE = 10  # 持仓10只

    # 风控参数
    STOP_LOSS = -0.15  # 止损-15%
    SCORE_THRESHOLD = 0.15  # 换仓阈值
    FORCE_REPLACE_DAYS = 45  # 强制评估周期

    # 交易成本
    BUY_COST = 0.0003
    SELL_COST = 0.0003
    TAX_RATIO = 0.0005

    # 数据配置
    USE_SAMPLING = False
    SAMPLE_SIZE = 3950

    # 实盘控制
    ENABLE_AUTO_TRADE = False  # ✨ 是否启用自动交易（默认关闭，仅生成建议）

    # 国信证券配置
    GUOSEN_CONFIG = {
        'broker': 'guosen',  # 券商代码
        'account': '',  # 资金账号
        'password': '',  # 交易密码
        'comm_password': '',  # 通讯密码
        'ip': '',  # 交易服务器IP
        'port': 0,  # 端口
    }


def check_trading_day():
    """检查是否是交易日"""
    try:
        pro = ts.pro_api()
        today = datetime.now().strftime('%Y%m%d')

        cal = pro.trade_cal(
            exchange='SSE',
            start_date=today,
            end_date=today
        )

        if len(cal) == 0:
            return False

        return cal.iloc[0]['is_open'] == 1
    except Exception as e:
        print(f"⚠️  交易日检查失败: {e}")
        return True  # 默认假设为交易日


def load_historical_state():
    """加载历史状态"""
    state_file = './live_trading_state.json'

    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass

    return {
        'last_rebalance_date': None,
        'positions': {},
        'rebalance_history': []
    }


def save_current_state(state):
    """保存当前状态"""
    with open('./live_trading_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def should_rebalance(state):
    """判断是否应该调仓"""
    last_date = state.get('last_rebalance_date')

    if last_date is None:
        return True, "首次运行"

    last_dt = datetime.strptime(last_date, '%Y-%m-%d')
    today = datetime.now()

    # 计算交易日差距
    days_diff = (today - last_dt).days

    if days_diff >= LiveTradingConfig.REBALANCE_DAYS:
        return True, f"距上次调仓{days_diff}天"

    return False, f"距上次调仓仅{days_diff}天"


def get_today_signals(factor_data, price_data):
    """
    获取今日交易信号

    :return: DataFrame with columns: stock, score, target_weight, current_price
    """
    today = datetime.now().strftime('%Y-%m-%d')

    # 获取今日因子数据
    today_factors = factor_data[factor_data['date'] == today]

    if len(today_factors) == 0:
        # 如果没有今天的数据，使用最新一天
        latest_date = factor_data['date'].max()
        today_factors = factor_data[factor_data['date'] == latest_date]
        print(f"  ℹ️  使用最新数据: {latest_date}")

    # 按评分排序，取前N只
    top_stocks = today_factors.nlargest(LiveTradingConfig.POSITION_SIZE, 'position')

    # 等权分配
    weight = 1.0 / len(top_stocks)

    # 获取价格
    today_prices = price_data[price_data['date'] == today_factors['date'].iloc[0]]

    signals = []
    for _, row in top_stocks.iterrows():
        stock = row['instrument']
        score = row['position']

        price_row = today_prices[today_prices['instrument'] == stock]
        price = price_row['close'].iloc[0] if len(price_row) > 0 else None

        signals.append({
            'stock': stock,
            'score': score,
            'target_weight': weight,
            'current_price': price
        })

    return pd.DataFrame(signals)


def compare_with_current_positions(signals, current_positions):
    """
    对比目标持仓和当前持仓

    :param signals: 目标持仓DataFrame
    :param current_positions: 当前持仓dict {stock: shares}
    :return: (to_buy, to_sell)
    """
    target_stocks = set(signals['stock'])
    current_stocks = set(current_positions.keys())

    # 需要卖出的
    to_sell = list(current_stocks - target_stocks)

    # 需要买入的
    to_buy = signals[~signals['stock'].isin(current_stocks)]

    return to_buy, to_sell


def generate_trading_orders(signals, current_positions, available_cash, total_value):
    """
    生成交易订单

    :return: DataFrame with columns: stock, action, shares, price, amount
    """
    orders = []

    target_stocks = set(signals['stock'])
    current_stocks = set(current_positions.keys())

    # 1. 卖出不在目标中的股票
    for stock in (current_stocks - target_stocks):
        shares = current_positions[stock]
        orders.append({
            'stock': stock,
            'action': 'sell',
            'shares': shares,
            'price': 0,  # 市价
            'amount': 0,
            'reason': '不在目标持仓'
        })

    # 2. 买入新股票
    to_buy = signals[~signals['stock'].isin(current_stocks)]

    for _, row in to_buy.iterrows():
        target_amount = total_value * row['target_weight']
        price = row['current_price']

        if price and price > 0:
            shares = int(target_amount / price / 100) * 100  # 整百股

            if shares >= 100:
                orders.append({
                    'stock': row['stock'],
                    'action': 'buy',
                    'shares': shares,
                    'price': price,
                    'amount': shares * price,
                    'reason': f"评分: {row['score']:.4f}"
                })

    return pd.DataFrame(orders)


def save_trading_orders(orders_df, output_dir='./live_trading'):
    """保存交易订单到CSV"""
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.now().strftime('%Y%m%d')

    # 保存详细订单
    orders_path = os.path.join(output_dir, f'trading_orders_{today}.csv')
    orders_df.to_csv(orders_path, index=False, encoding='utf-8-sig')

    print(f"\n💾 交易订单已保存: {orders_path}")

    # 生成简化版（用于手工交易）
    simple_orders = []

    for _, order in orders_df.iterrows():
        if order['action'] == 'buy':
            simple_orders.append(f"买入 {order['stock']} {order['shares']}股")
        elif order['action'] == 'sell':
            simple_orders.append(f"卖出 {order['stock']} {order['shares']}股")

    simple_path = os.path.join(output_dir, f'trading_instructions_{today}.txt')
    with open(simple_path, 'w', encoding='utf-8') as f:
        f.write(f"交易日期: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"调仓周期: {LiveTradingConfig.REBALANCE_DAYS}日\n")
        f.write("=" * 60 + "\n\n")
        f.write("交易指令:\n\n")
        for i, instruction in enumerate(simple_orders, 1):
            f.write(f"{i}. {instruction}\n")

    print(f"💾 交易指令已保存: {simple_path}")

    return orders_path


def execute_orders_guosen(orders_df, config):
    """
    ✨ 通过国信证券API执行订单

    注意: 需要安装 easytrader 库
    pip install easytrader
    """
    if not LiveTradingConfig.ENABLE_AUTO_TRADE:
        print("\n⚠️  自动交易未启用，仅生成订单文件")
        return

    try:
        import easytrader

        # 初始化交易接口
        user = easytrader.use('guosen')
        user.prepare(
            user=config['account'],
            password=config['password'],
            comm_password=config['comm_password']
        )

        print("\n🔗 已连接国信证券")

        # 执行订单
        for _, order in orders_df.iterrows():
            stock = order['stock']
            action = order['action']
            shares = order['shares']

            try:
                if action == 'buy':
                    result = user.buy(stock, price=0, amount=shares)  # 市价单
                    print(f"  ✓ 买入 {stock} {shares}股")

                elif action == 'sell':
                    result = user.sell(stock, price=0, amount=shares)
                    print(f"  ✓ 卖出 {stock} {shares}股")

            except Exception as e:
                print(f"  ❌ 订单失败 {stock}: {e}")

        print("\n✅ 订单执行完成")

    except ImportError:
        print("\n❌ 未安装 easytrader 库")
        print("   安装命令: pip install easytrader")
    except Exception as e:
        print(f"\n❌ 交易执行失败: {e}")


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🤖 实盘交易系统")
    print("=" * 80)
    print(f"  策略: 5日调仓-等权（胜率53.24%）")
    print(f"  模式: {'自动交易' if LiveTradingConfig.ENABLE_AUTO_TRADE else '仅生成建议'}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 检查交易日
    print("\n【步骤1/5】检查交易日")

    if not check_trading_day():
        print("  ℹ️  今天不是交易日")
        return

    print("  ✓ 确认为交易日")

    # 2. 加载历史状态
    print("\n【步骤2/5】加载历史状态")

    state = load_historical_state()

    if state['last_rebalance_date']:
        print(f"  上次调仓: {state['last_rebalance_date']}")
        print(f"  当前持仓: {len(state['positions'])} 只")
    else:
        print("  首次运行")

    # 3. 判断是否需要调仓
    print("\n【步骤3/5】判断调仓时机")

    need_rebalance, reason = should_rebalance(state)
    print(f"  是否调仓: {'是' if need_rebalance else '否'} ({reason})")

    if not need_rebalance:
        print("\n  今日无需调仓")
        return

    # 4. 加载数据
    print("\n【步骤4/5】加载最新数据")

    START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    cache_manager = DataCache(cache_dir='./data_cache')

    factor_data, price_data = load_data_with_incremental_update(
        START_DATE,
        END_DATE,
        max_stocks=LiveTradingConfig.SAMPLE_SIZE,
        cache_manager=cache_manager,
        use_stockranker=True,
        tushare_token=TUSHARE_TOKEN,
        use_fundamental=True,
        use_sampling=LiveTradingConfig.USE_SAMPLING,
        sample_size=LiveTradingConfig.SAMPLE_SIZE,
        max_workers=10,
        force_full_update=False
    )

    if factor_data is None or price_data is None:
        print("  ❌ 数据加载失败")
        return

    print(f"  ✓ 数据加载完成")

    # 5. 生成交易信号
    print("\n【步骤5/5】生成交易信号")

    # 获取今日信号
    signals = get_today_signals(factor_data, price_data)

    print(f"\n  目标持仓 ({len(signals)} 只):")
    for i, row in signals.iterrows():
        print(f"    {i + 1:2d}. {row['stock']:12s} | 评分: {row['score']:.4f} | "
              f"权重: {row['target_weight']:.1%} | 价格: ¥{row['current_price']:.2f}")

    # 对比当前持仓
    current_positions = state.get('positions', {})
    to_buy, to_sell = compare_with_current_positions(signals, current_positions)

    print(f"\n  需要调整:")
    print(f"    卖出: {len(to_sell)} 只")
    print(f"    买入: {len(to_buy)} 只")

    # 生成订单
    # 假设初始资金100万，实际应从券商账户获取
    available_cash = 1000000
    total_value = 1000000

    orders = generate_trading_orders(signals, current_positions, available_cash, total_value)

    if len(orders) > 0:
        print(f"\n  交易订单 ({len(orders)} 条):")
        for _, order in orders.iterrows():
            action_icon = "🔵" if order['action'] == 'buy' else "🔴"
            print(f"    {action_icon} {order['action']:4s} {order['stock']:12s} "
                  f"{order['shares']:6.0f}股 @ ¥{order['price']:.2f}")

        # 保存订单
        save_trading_orders(orders)

        # 询问是否执行
        if LiveTradingConfig.ENABLE_AUTO_TRADE:
            response = input("\n  是否执行交易？(y/n): ").lower()
            if response == 'y':
                execute_orders_guosen(orders, LiveTradingConfig.GUOSEN_CONFIG)
            else:
                print("  已取消自动交易")

        # 更新状态
        state['last_rebalance_date'] = datetime.now().strftime('%Y-%m-%d')
        state['positions'] = {row['stock']: row['shares']
                              for _, row in signals.iterrows()}
        state['rebalance_history'].append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'orders_count': len(orders)
        })

        save_current_state(state)

    else:
        print("\n  无需交易")

    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()