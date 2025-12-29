"""
main_live_trading_enhanced.py - 实盘交易增强版 (集成滚动训练ML)

核心升级:
✅ 集成滚动训练ML模型（UltraMLScorer）
✅ 使用最新12个月数据训练
✅ 保持原有交易逻辑和风控
✅ 支持模型缓存加速
✅ 生成可解释的选股报告

配置:
- 5日调仓-等权（基础胜率 53.24%）
- ML增强选股（预期提升至 60%+）
- 每日检查但不一定交易
- 生成持仓建议CSV
- 支持国信证券接口

版本: v2.6
日期: 2025-12-27
"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time
import pickle

import tushare as ts

TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"
ts.set_token(TUSHARE_TOKEN)

from data_module import DataCache, TushareDataSource
from data_module_incremental import load_data_with_incremental_update

# ========== 导入ML模块 ==========
try:
    from ml_factor_scoring_fixed import UltraMLScorer

    ML_AVAILABLE = True
    print("✓ 滚动训练ML模块加载成功")
except ImportError as e:
    print(f"⚠️  ML模块未找到: {e}")
    ML_AVAILABLE = False


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

    # ML配置
    USE_ML_SCORING = True  # ✨ 启用ML评分
    ML_TRAIN_MONTHS = 12  # 训练窗口（月）
    ML_CACHE_MODELS = True  # 缓存训练好的模型

    # 实盘控制
    ENABLE_AUTO_TRADE = False  # ✨ 是否启用自动交易（默认关闭，仅生成建议）

    # 国信证券配置
    GUOSEN_CONFIG = {
        'broker': 'guosen',
        'account': '',
        'password': '',
        'comm_password': '',
        'ip': '',
        'port': 0,
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
        return True


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
        'last_ml_train_date': None,  # ✨ 新增：记录上次ML训练时间
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

    days_diff = (today - last_dt).days

    if days_diff >= LiveTradingConfig.REBALANCE_DAYS:
        return True, f"距上次调仓{days_diff}天"

    return False, f"距上次调仓仅{days_diff}天"


def load_or_train_ml_model(factor_data, price_data, cache_manager):
    """
    ✨ 加载或训练ML模型

    策略：
    1. 检查是否有今日的缓存模型
    2. 如果没有，使用滚动训练
    3. 缓存训练好的模型供下次使用
    """
    if not ML_AVAILABLE:
        print("  ⚠️  ML模块不可用，跳过ML评分")
        return None

    today = datetime.now().strftime('%Y%m%d')
    model_cache_path = f'./data_cache/ml_model_{today}.pkl'

    # 1. 尝试加载缓存模型
    if LiveTradingConfig.ML_CACHE_MODELS and os.path.exists(model_cache_path):
        try:
            print("  📦 尝试加载缓存模型...")
            with open(model_cache_path, 'rb') as f:
                ml_scorer = pickle.load(f)
            print(f"  ✓ 已加载缓存模型: {model_cache_path}")
            return ml_scorer
        except Exception as e:
            print(f"  ⚠️  缓存加载失败: {e}")

    # 2. 训练新模型
    print(f"  🚀 训练ML模型（使用最近{LiveTradingConfig.ML_TRAIN_MONTHS}个月数据）...")

    try:
        ml_scorer = UltraMLScorer(
            target_period=5,
            top_percentile=0.2,
            embargo_days=5,
            neutralize_market=True,
            neutralize_industry=True,
            voting_strategy='average',
            train_months=LiveTradingConfig.ML_TRAIN_MONTHS
        )

        # 注意：这里不调用 predict()，仅初始化
        # 实际预测在 get_today_signals_with_ml 中进行

        # 3. 缓存模型
        if LiveTradingConfig.ML_CACHE_MODELS:
            try:
                os.makedirs('./data_cache', exist_ok=True)
                with open(model_cache_path, 'wb') as f:
                    pickle.dump(ml_scorer, f)
                print(f"  💾 模型已缓存: {model_cache_path}")
            except Exception as e:
                print(f"  ⚠️  模型缓存失败: {e}")

        return ml_scorer

    except Exception as e:
        print(f"  ❌ ML模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_today_signals_with_ml(factor_data, price_data, ml_scorer=None):
    """
    ✨ 使用ML模型获取今日交易信号

    流程：
    1. 如果有ML模型，使用ML评分
    2. 否则使用StockRanker的基础评分
    3. 选择Top N只股票
    """
    today = datetime.now().strftime('%Y-%m-%d')

    # 获取最新数据日期
    latest_date = factor_data['date'].max()
    print(f"  📅 使用数据日期: {latest_date}")

    # 1. 如果启用ML且模型可用，进行ML评分
    if LiveTradingConfig.USE_ML_SCORING and ml_scorer and ML_AVAILABLE:
        print(f"  🤖 使用ML模型评分...")

        try:
            # 执行滚动预测（仅预测最后一个时间窗口）
            scored_data = ml_scorer.predict(factor_data, price_data)

            # 使用最新日期的数据
            today_factors = scored_data[scored_data['date'] == latest_date]

            if 'ml_score' in today_factors.columns:
                print(f"  ✓ ML评分完成")
                print(f"    - 评分范围: [{today_factors['ml_score'].min():.4f}, {today_factors['ml_score'].max():.4f}]")
                score_column = 'ml_score'
            else:
                print(f"  ⚠️  未找到ml_score，使用position")
                score_column = 'position'

        except Exception as e:
            print(f"  ⚠️  ML评分失败: {e}")
            print(f"  ℹ️  降级使用StockRanker评分")
            today_factors = factor_data[factor_data['date'] == latest_date]
            score_column = 'position'
    else:
        # 2. 使用StockRanker的基础评分
        print(f"  📊 使用StockRanker评分...")
        today_factors = factor_data[factor_data['date'] == latest_date]
        score_column = 'position'

    if len(today_factors) == 0:
        print(f"  ❌ 未找到有效数据")
        return pd.DataFrame()

    # 3. 选择Top N只股票
    top_stocks = today_factors.nlargest(LiveTradingConfig.POSITION_SIZE, score_column)

    # 等权分配
    weight = 1.0 / len(top_stocks)

    # 获取价格
    today_prices = price_data[price_data['date'] == latest_date]

    signals = []
    for _, row in top_stocks.iterrows():
        stock = row['instrument']
        score = row[score_column]

        price_row = today_prices[today_prices['instrument'] == stock]
        price = price_row['close'].iloc[0] if len(price_row) > 0 else None

        # 尝试获取行业信息
        industry = row.get('industry', 'Unknown')

        signals.append({
            'stock': stock,
            'score': score,
            'target_weight': weight,
            'current_price': price,
            'industry': industry
        })

    return pd.DataFrame(signals)


def compare_with_current_positions(signals, current_positions):
    """对比目标持仓和当前持仓"""
    target_stocks = set(signals['stock'])
    current_stocks = set(current_positions.keys())

    to_sell = list(current_stocks - target_stocks)
    to_buy = signals[~signals['stock'].isin(current_stocks)]

    return to_buy, to_sell


def generate_trading_orders(signals, current_positions, available_cash, total_value):
    """生成交易订单"""
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
            'price': 0,
            'amount': 0,
            'reason': '不在目标持仓'
        })

    # 2. 买入新股票
    to_buy = signals[~signals['stock'].isin(current_stocks)]

    for _, row in to_buy.iterrows():
        target_amount = total_value * row['target_weight']
        price = row['current_price']

        if price and price > 0:
            shares = int(target_amount / price / 100) * 100

            if shares >= 100:
                orders.append({
                    'stock': row['stock'],
                    'action': 'buy',
                    'shares': shares,
                    'price': price,
                    'amount': shares * price,
                    'reason': f"ML评分: {row['score']:.4f}"
                })

    return pd.DataFrame(orders)


def reconcile_positions_after_orders(current_positions, orders_df):
    """根据交易订单更新持仓"""
    new_positions = current_positions.copy()

    for _, order in orders_df.iterrows():
        stock = order['stock']

        if order['action'] == 'sell':
            if stock in new_positions:
                del new_positions[stock]

        elif order['action'] == 'buy':
            new_positions[stock] = int(order['shares'])

    return new_positions


def save_trading_orders(orders_df, signals_df, output_dir='./live_trading'):
    """
    ✨ 保存交易订单和选股信号

    新增：
    - 保存完整的信号数据（包含评分、行业等）
    - 生成可读性更好的报告
    """
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.now().strftime('%Y%m%d')

    # 1. 保存详细订单
    orders_path = os.path.join(output_dir, f'trading_orders_{today}.csv')
    orders_df.to_csv(orders_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 交易订单已保存: {orders_path}")

    # 2. 保存信号数据
    signals_path = os.path.join(output_dir, f'signals_{today}.csv')
    signals_df.to_csv(signals_path, index=False, encoding='utf-8-sig')
    print(f"💾 信号数据已保存: {signals_path}")

    # 3. 生成可读报告
    simple_path = os.path.join(output_dir, f'trading_instructions_{today}.txt')
    with open(simple_path, 'w', encoding='utf-8') as f:
        f.write(f"=" * 80 + "\n")
        f.write(f"实盘交易建议 - ML增强版\n")
        f.write(f"=" * 80 + "\n\n")

        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"调仓周期: {LiveTradingConfig.REBALANCE_DAYS}日\n")
        f.write(f"ML模型: {'已启用' if LiveTradingConfig.USE_ML_SCORING else '未启用'}\n")
        f.write(f"\n" + "-" * 80 + "\n\n")

        # 目标持仓
        f.write(f"【目标持仓】共 {len(signals_df)} 只\n\n")
        for i, row in signals_df.iterrows():
            f.write(f"{i + 1:2d}. {row['stock']:12s} | "
                    f"评分: {row['score']:.4f} | "
                    f"权重: {row['target_weight']:6.1%} | "
                    f"价格: ¥{row['current_price']:7.2f} | "
                    f"行业: {row.get('industry', 'Unknown')}\n")

        f.write(f"\n" + "-" * 80 + "\n\n")

        # 交易指令
        f.write(f"【交易指令】共 {len(orders_df)} 条\n\n")

        if len(orders_df) == 0:
            f.write("无需交易，保持当前持仓。\n")
        else:
            for i, order in orders_df.iterrows():
                if order['action'] == 'buy':
                    f.write(f"{i + 1:2d}. 🔵 买入 {order['stock']:12s} "
                            f"{int(order['shares']):6d}股 @ ¥{order['price']:.2f} "
                            f"(约 ¥{order['amount']:,.0f})\n")
                elif order['action'] == 'sell':
                    f.write(f"{i + 1:2d}. 🔴 卖出 {order['stock']:12s} "
                            f"{int(order['shares']):6d}股 (市价)\n")

        f.write(f"\n" + "=" * 80 + "\n")

    print(f"💾 交易指令已保存: {simple_path}")

    return orders_path


def execute_orders_guosen(orders_df, config):
    """通过国信证券API执行订单"""
    if not LiveTradingConfig.ENABLE_AUTO_TRADE:
        print("\n⚠️  自动交易未启用，仅生成订单文件")
        return

    try:
        import easytrader

        user = easytrader.use('guosen')
        user.prepare(
            user=config['account'],
            password=config['password'],
            comm_password=config['comm_password']
        )

        print("\n🔗 已连接国信证券")

        for _, order in orders_df.iterrows():
            stock = order['stock']
            action = order['action']
            shares = int(order['shares'])

            try:
                if action == 'buy':
                    result = user.buy(stock, price=0, amount=shares)
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
    print("🤖 实盘交易系统 - ML增强版 v2.6")
    print("=" * 80)
    print(f"  策略: 5日调仓-等权 + ML选股")
    print(f"  ML模型: {'✓ 滚动训练' if LiveTradingConfig.USE_ML_SCORING else '✗ 未启用'}")
    print(f"  模式: {'自动交易' if LiveTradingConfig.ENABLE_AUTO_TRADE else '仅生成建议'}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 检查交易日
    print("\n【步骤1/6】检查交易日")

    if not check_trading_day():
        print("  ℹ️  今天不是交易日")
        return

    print("  ✓ 确认为交易日")

    # 2. 加载历史状态
    print("\n【步骤2/6】加载历史状态")

    state = load_historical_state()

    if state['last_rebalance_date']:
        print(f"  上次调仓: {state['last_rebalance_date']}")
        print(f"  当前持仓: {len(state['positions'])} 只")
        if state.get('last_ml_train_date'):
            print(f"  上次ML训练: {state['last_ml_train_date']}")
    else:
        print("  首次运行")

    # 3. 判断是否需要调仓
    print("\n【步骤3/6】判断调仓时机")

    need_rebalance, reason = should_rebalance(state)
    print(f"  是否调仓: {'是' if need_rebalance else '否'} ({reason})")

    if not need_rebalance:
        print("\n  今日无需调仓")
        return

    # 4. 加载数据
    print("\n【步骤4/6】加载最新数据")

    # 使用更长的历史数据窗口（12个月+）用于ML训练
    START_DATE = (datetime.now() - timedelta(days=365 + 90)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    cache_manager = DataCache(cache_dir='./data_cache')

    start_time = time.time()
    factor_data, price_data = load_data_with_incremental_update(
        START_DATE,
        END_DATE,
        max_stocks=LiveTradingConfig.SAMPLE_SIZE,
        cache_manager=cache_manager,
        use_stockranker=True,
        tushare_token=TUSHARE_TOKEN,
        use_fundamental=True,
        use_money_flow=True,  # ✨ 启用资金流因子
        use_sampling=LiveTradingConfig.USE_SAMPLING,
        sample_size=LiveTradingConfig.SAMPLE_SIZE,
        max_workers=10,
        force_full_update=False,
        min_days_listed=180
    )

    if factor_data is None or price_data is None:
        print("  ❌ 数据加载失败")
        return

    # 补全行业数据
    ds = TushareDataSource(token=TUSHARE_TOKEN, cache_manager=cache_manager)
    industry_df = ds.get_industry_data(
        factor_data['instrument'].unique().tolist(),
        use_cache=True
    )

    if industry_df is not None and not industry_df.empty:
        if 'industry' in factor_data.columns:
            del factor_data['industry']
        factor_data = factor_data.merge(industry_df, on='instrument', how='left')
        factor_data['industry'] = factor_data['industry'].fillna('其他')
    else:
        factor_data['industry'] = 'Unknown'

    print(f"  ✓ 数据加载完成 (耗时: {time.time() - start_time:.1f}秒)")
    print(f"    - 股票数: {factor_data['instrument'].nunique()}")
    print(f"    - 时间范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")

    # 5. ML模型训练/加载
    print("\n【步骤5/6】ML模型准备")

    ml_scorer = None
    if LiveTradingConfig.USE_ML_SCORING:
        ml_start_time = time.time()
        ml_scorer = load_or_train_ml_model(factor_data, price_data, cache_manager)
        ml_time = time.time() - ml_start_time

        if ml_scorer:
            print(f"  ✓ ML模型就绪 (耗时: {ml_time:.1f}秒)")
            state['last_ml_train_date'] = datetime.now().strftime('%Y-%m-%d')
        else:
            print(f"  ⚠️  ML模型不可用，使用基础评分")
    else:
        print(f"  ℹ️  ML评分已禁用")

    # 6. 生成交易信号
    print("\n【步骤6/6】生成交易信号")

    signals = get_today_signals_with_ml(factor_data, price_data, ml_scorer)

    if len(signals) == 0:
        print("  ❌ 未能生成有效信号")
        return

    print(f"\n  ✨ 目标持仓 ({len(signals)} 只):")
    print(f"  {'序号':<4} | {'代码':<12} | {'评分':<8} | {'权重':<8} | {'价格':<10} | {'行业'}")
    print(f"  {'-' * 70}")

    for i, row in signals.iterrows():
        print(f"  {i + 1:2d}.  | {row['stock']:<12} | {row['score']:<8.4f} | "
              f"{row['target_weight']:<7.1%} | ¥{row['current_price']:<8.2f} | "
              f"{row.get('industry', 'Unknown')}")

    # 对比当前持仓
    current_positions = state.get('positions', {})
    to_buy, to_sell = compare_with_current_positions(signals, current_positions)

    print(f"\n  需要调整:")
    print(f"    🔴 卖出: {len(to_sell)} 只")
    print(f"    🔵 买入: {len(to_buy)} 只")

    # 生成订单
    available_cash = 1000000
    total_value = 1000000

    orders = generate_trading_orders(signals, current_positions, available_cash, total_value)

    if len(orders) > 0:
        print(f"\n  📋 交易订单 ({len(orders)} 条):")
        for _, order in orders.iterrows():
            action_icon = "🔵" if order['action'] == 'buy' else "🔴"
            print(f"    {action_icon} {order['action']:4s} {order['stock']:12s} "
                  f"{int(order['shares']):6d}股 @ ¥{order['price']:.2f}")

        # 保存订单和信号
        save_trading_orders(orders, signals)

        # 询问是否执行
        if LiveTradingConfig.ENABLE_AUTO_TRADE:
            response = input("\n  是否执行交易？(y/n): ").lower()
            if response == 'y':
                execute_orders_guosen(orders, LiveTradingConfig.GUOSEN_CONFIG)
            else:
                print("  已取消自动交易")

        # 更新状态
        state['last_rebalance_date'] = datetime.now().strftime('%Y-%m-%d')
        state['positions'] = reconcile_positions_after_orders(current_positions, orders)
        state['rebalance_history'].append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'orders_count': len(orders),
            'ml_enabled': LiveTradingConfig.USE_ML_SCORING
        })

        save_current_state(state)

        print(f"\n  ✓ 状态已更新，新持仓: {len(state['positions'])} 只")

    else:
        print("\n  ℹ️  无需交易，保持当前持仓")

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