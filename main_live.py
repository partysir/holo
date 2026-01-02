"""
main_live.py - 实盘交易主入口 (重构版 v3.0)
集成滚动训练ML、Alpha因子增强、舆情风控

核心特性:
✅ 使用重构后的模块结构
✅ 集成Alpha因子计算库
✅ 集成Purging/Embargo ML模型
✅ 集成舆情风控模块
✅ 优化实盘交易流程

配置:
- 5日调仓-等权（基础胜率 53.24%）
- ML增强选股（预期提升至 60%+）
- 每日检查但不一定交易
- 支持多券商API接口
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

# 从重构后的配置文件导入
from config import TUSHARE_TOKEN, LiveTradingConfig, BrokerConfig, get_live_trading_params
from data_module import DataCache, TushareDataSource
from data_module_incremental import load_data_with_incremental_update
from data_module_alpha_enhanced import AlphaFactorCalculator
from enhanced_factor_processor import EnhancedFactorProcessor
from ml_core import UltraMLScorer
from sentiment_risk_control import SentimentRiskController

ts.set_token(TUSHARE_TOKEN)

# ========== 实盘配置 ==========
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
        'last_ml_train_date': None,  # 记录上次ML训练时间
        'positions': {},
        'rebalance_history': [],
        'risk_events': []  # 记录风险事件
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
    加载或训练ML模型

    策略：
    1. 检查是否有今日的缓存模型
    2. 如果没有，使用滚动训练
    3. 缓存训练好的模型供下次使用
    """
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

        # 使用重构后的数据处理流程
        factor_processor = EnhancedFactorProcessor(
            neutralize_industry=True, 
            neutralize_market=True, 
            use_alpha_factors=True
        )
        processed_factors = factor_processor.process_factors(factor_data, price_data)

        # 训练模型
        factor_columns = [col for col in processed_factors.columns if col not in ['instrument', 'date', 'industry', 'close']]
        X, y, merged_data = ml_scorer.prepare_data(processed_factors, price_data, factor_columns)
        ml_scorer.train(X, y, merged_data)

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
    使用ML模型获取今日交易信号

    流程：
    1. 如果有ML模型，使用ML评分
    2. 否则使用StockRanker的基础评分
    3. 选择Top N只股票
    """
    today = datetime.now().strftime('%Y-%m-%d')

    # 获取最新数据日期
    latest_date = factor_data['date'].max()
    print(f"  📅 使用数据日期: {latest_date}")

    # 使用重构后的数据处理流程
    factor_processor = EnhancedFactorProcessor(
        neutralize_industry=True, 
        neutralize_market=True, 
        use_alpha_factors=True
    )
    processed_factors = factor_processor.process_factors(factor_data, price_data)

    # 1. 如果启用ML且模型可用，进行ML评分
    if LiveTradingConfig.USE_ML_SCORING and ml_scorer:
        print(f"  🤖 使用ML模型评分...")

        try:
            # 执行预测
            scored_data = ml_scorer.predict(processed_factors, price_data)

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
            today_factors = processed_factors[processed_factors['date'] == latest_date]
            score_column = 'position'
    else:
        # 2. 使用StockRanker的基础评分
        print(f"  📊 使用StockRanker评分...")
        today_factors = processed_factors[processed_factors['date'] == latest_date]
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
    保存交易订单和选股信号

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
                    f"评分: {row['score']:8.4f} | "
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


def execute_orders_broker(orders_df, broker_type='guosen'):
    """通过指定券商API执行订单"""
    if not LiveTradingConfig.ENABLE_AUTO_TRADE:
        print("\n⚠️  自动交易未启用，仅生成订单文件")
        return

    try:
        import easytrader

        if broker_type == 'guosen':
            user = easytrader.use('guosen')
            config = BrokerConfig.GUOSEN
        elif broker_type == 'gf':
            user = easytrader.use('gf')
            config = BrokerConfig.GUANGFA
        elif broker_type == 'ht':
            user = easytrader.use('ht')
            config = BrokerConfig.HUATAI
        elif broker_type == 'yh':
            user = easytrader.use('yh')
            config = BrokerConfig.YINHE
        elif broker_type == 'yjb':
            user = easytrader.use('yjb')
            config = BrokerConfig.YJB
        else:
            print(f"\n❌ 不支持的券商类型: {broker_type}")
            return

        # 从环境变量获取账户信息（更安全）
        config['account'] = os.getenv('BROKER_ACCOUNT', config.get('account', ''))
        config['password'] = os.getenv('BROKER_PASSWORD', config.get('password', ''))
        config['comm_password'] = os.getenv('BROKER_COMM_PASSWORD', config.get('comm_password', ''))

        user.prepare(
            user=config['account'],
            password=config['password'],
            comm_password=config['comm_password']
        )

        print(f"\n🔗 已连接{broker_type}证券")

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
    print("🤖 实盘交易系统 - 重构版 v3.0")
    print("=" * 80)
    print(f"  策略: 5日调仓-等权 + ML+Alpha选股")
    print(f"  ML模型: {'✓ 滚动训练' if LiveTradingConfig.USE_ML_SCORING else '✗ 未启用'}")
    print(f"  Alpha因子: {'✓ 启用' if True else '✗ 未启用'}")
    print(f"  模式: {'自动交易' if LiveTradingConfig.ENABLE_AUTO_TRADE else '仅生成建议'}")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 检查交易日
    print("\n【步骤1/7】检查交易日")
    if not check_trading_day():
        print("  ℹ️  今天不是交易日")
        return

    print("  ✓ 确认为交易日")

    # 2. 加载历史状态
    print("\n【步骤2/7】加载历史状态")
    state = load_historical_state()

    if state['last_rebalance_date']:
        print(f"  上次调仓: {state['last_rebalance_date']}")
        print(f"  当前持仓: {len(state['positions'])} 只")
        if state.get('last_ml_train_date'):
            print(f"  上次ML训练: {state['last_ml_train_date']}")
    else:
        print("  首次运行")

    # 3. 判断是否需要调仓
    print("\n【步骤3/7】判断调仓时机")
    need_rebalance, reason = should_rebalance(state)
    print(f"  是否调仓: {'是' if need_rebalance else '否'} ({reason})")

    if not need_rebalance:
        print("\n  今日无需调仓")
        return

    # 4. 舆情风控检查
    print("\n【步骤4/7】舆情风控检查")
    try:
        risk_controller = SentimentRiskController()
        market_sentiment = risk_controller.get_market_sentiment()
        print(f"  市场情绪: {market_sentiment['overall_sentiment']:.2f} ({market_sentiment['confidence']:.2f})")
        print(f"  风险等级: {market_sentiment['risk_level']}")

        if market_sentiment['risk_level'] == 'HIGH':
            print("  ⚠️  市场风险等级高，暂停交易")
            return
        elif market_sentiment['risk_level'] == 'MEDIUM':
            print("  ⚠️  市场风险等级中等，谨慎操作")
    except Exception as e:
        print(f"  ⚠️  舆情风控检查失败: {e}")

    # 5. 加载数据
    print("\n【步骤5/7】加载最新数据")
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
        use_money_flow=True,  # 启用资金流因子
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

    # 6. ML模型训练/加载
    print("\n【步骤6/7】ML模型准备")
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

    # 7. 生成交易信号
    print("\n【步骤7/7】生成交易信号")
    signals = get_today_signals_with_ml(factor_data, price_data, ml_scorer)

    if len(signals) == 0:
        print("  ❌ 未能生成有效信号")
        return

    print(f"\n  ✨ 目标持仓 ({len(signals)} 只):")
    print("  序号 | 代码         | 评分     | 权重   | 价格     | 行业")
    print("  --------------------------------------------------------------")

    for i, row in signals.iterrows():
        idx = i + 1
        stock = row['stock']
        score = row['score']
        weight = row['target_weight']
        price = row['current_price']
        industry = row.get('industry', 'Unknown')
        output_str = "  {:2d}.  | {:12s} | {:8.4f} | {:>6.1%} | ¥{:7.2f} | {}".format(idx, stock, score, weight, price, industry)
        print(output_str)

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

        # 执行交易
        if LiveTradingConfig.ENABLE_AUTO_TRADE:
            response = input("\n  是否执行交易？(y/n): ").lower()
            if response == 'y':
                execute_orders_broker(orders, LiveTradingConfig.BROKER)
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