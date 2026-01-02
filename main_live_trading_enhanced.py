"""
main_live_trading_enhanced.py - 完整增强版实盘交易系统 v3.1

核心功能:
1. 评分融合 (StockRanker + ML)
2. 今日股票推荐（Top 10）
3. 详细推荐报告
4. 持仓分析
5. 风险提示

修复内容 (v3.1):
- ✅ 修复技术指标计算：使用 price_data 而非 factor_data
- ✅ 修复5日回报率计算逻辑
- ✅ 增加数据验证和异常值检测
- ✅ 增强日志输出

版本: v3.1
日期: 2025-12-30
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import tushare as ts

# ========== 配置 ==========
TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"
ts.set_token(TUSHARE_TOKEN)

from data_module import DataCache
from data_module_incremental import load_data_with_incremental_update
from score_fusion_module import ScoreFusionEngine

# ML模块
ML_AVAILABLE = False
try:
    from ml_factor_scoring_fixed_v29 import UltraMLScorer as AdvancedMLScorer
    ML_AVAILABLE = True
    print("✅ ML module available (v2.9)")
except ImportError:
    try:
        from ml_factor_scoring_fixed import UltraMLScorer as AdvancedMLScorer
        ML_AVAILABLE = True
        print("✅ ML module available")
    except ImportError:
        print("⚠️  ML module not available")


# ========== 配置类 ==========
class LiveTradingConfig:
    """实盘交易配置"""

    REBALANCE_DAYS = 5
    POSITION_SIZE = 10
    RECOMMEND_TOP_N = 10  # 推荐前10只
    RECOMMEND_MIN_SCORE = 0.6  # 最低推荐分数

    # 评分融合
    USE_ML = True
    FUSION_METHOD = 'weighted'
    FUSION_ALPHA = 0.4
    FUSION_BETA = 0.6

    # 数据配置
    SAMPLE_SIZE = 3950

    # 实盘控制
    ENABLE_AUTO_TRADE = False


# ========== 评分处理 ==========

def fix_stockranker_scoring(factor_data):
    if 'position' in factor_data.columns:
        factor_data['stockranker_score'] = factor_data['position']
        factor_data.drop(columns=['position'], inplace=True)
    return factor_data


def fix_ml_scoring(factor_data, price_data):
    if not LiveTradingConfig.USE_ML or not ML_AVAILABLE:
        return factor_data

    print("\nRunning ML scoring...")

    try:
        ml_scorer = AdvancedMLScorer(
            target_period=5,
            top_percentile=0.2,
            train_months=12
        )

        temp_sr = None
        if 'stockranker_score' in factor_data.columns:
            temp_sr = factor_data['stockranker_score'].copy()

        factor_data = ml_scorer.predict(factor_data, price_data)

        if temp_sr is not None:
            factor_data['stockranker_score'] = temp_sr

        if 'position' in factor_data.columns and 'ml_score' not in factor_data.columns:
            factor_data['ml_score'] = factor_data['position']
            factor_data.drop(columns=['position'], inplace=True)

        print("ML scoring completed")

    except Exception as e:
        print(f"ML scoring failed: {e}")
        import traceback
        traceback.print_exc()

    return factor_data


def fuse_scores(factor_data):
    print("\nFusing scores...")

    fusion_engine = ScoreFusionEngine(
        fusion_method=LiveTradingConfig.FUSION_METHOD,
        alpha=LiveTradingConfig.FUSION_ALPHA,
        beta=LiveTradingConfig.FUSION_BETA
    )

    has_ml = 'ml_score' in factor_data.columns
    factor_data = fusion_engine.fuse_scores(factor_data, has_ml=has_ml)

    return factor_data


# ========== 推荐生成 ==========

def generate_recommendations(factor_data, price_data, state):
    """生成今日推荐 - v3.1 增强版"""
    print("\n" + "="*80)
    print("Generating Today's Stock Recommendations")
    print("="*80)

    # ✅ 数据验证
    print(f"\n📊 数据概况:")
    print(f"  Factor Data: {len(factor_data)} 行, {len(factor_data['instrument'].unique())} 只股票")
    print(f"  Price Data:  {len(price_data)} 行, {len(price_data['instrument'].unique())} 只股票")
    print(f"  最新日期:    {price_data['date'].max()}")

    # ✅ 检查ML评分范围
    if 'ml_score' in factor_data.columns:
        ml_min, ml_max = factor_data['ml_score'].min(), factor_data['ml_score'].max()
        print(f"  ML Score范围: [{ml_min:.4f}, {ml_max:.4f}]")

        if ml_max > 1.5:
            print(f"  ⚠️ 警告: ML评分超过正常范围，可能需要检查归一化")

    today = datetime.now().strftime('%Y-%m-%d')

    # 获取今日数据
    today_factors = factor_data[factor_data['date'] == today]

    if len(today_factors) == 0:
        latest_date = factor_data['date'].max()
        today_factors = factor_data[factor_data['date'] == latest_date]
        print(f"Using latest data: {latest_date}")
        today = latest_date

    # 筛选高分股票
    high_score = today_factors[
        today_factors['position'] >= LiveTradingConfig.RECOMMEND_MIN_SCORE
    ].copy()

    # 排序
    high_score = high_score.sort_values('position', ascending=False)

    # 取Top N
    recommendations = high_score.head(LiveTradingConfig.RECOMMEND_TOP_N).copy()

    # 获取价格
    today_prices = price_data[price_data['date'] == today]

    # 增强信息
    recommendations = enhance_recommendations(recommendations, today_prices, price_data, factor_data, state)

    print(f"\n✅ Generated {len(recommendations)} recommendations")

    return recommendations


def enhance_recommendations(recommendations, today_prices, price_data, factor_data, state):
    """
    增强推荐信息 - v3.1 修复版

    ✅ 关键修复：使用 price_data 计算技术指标，而非 factor_data
    """

    # 1. 添加当前价格（从 today_prices）
    recommendations = recommendations.merge(
        today_prices[['instrument', 'close', 'volume', 'amount']],
        on='instrument',
        how='left',
        suffixes=('', '_price')
    )

    # 2. 计算技术指标（✅ 使用 price_data 而非 factor_data）
    print("  📊 计算技术指标...")

    for idx, row in recommendations.iterrows():
        stock = row['instrument']

        # ✅ 从 price_data 获取历史价格
        hist = price_data[
            price_data['instrument'] == stock
        ].sort_values('date').tail(30)  # 取最近30天

        if len(hist) >= 6:
            # 计算5日回报率
            try:
                price_5d_ago = hist['close'].iloc[-6]
                price_now = hist['close'].iloc[-1]

                if price_5d_ago > 0:
                    returns_5d = (price_now / price_5d_ago - 1)
                else:
                    returns_5d = 0

                recommendations.at[idx, 'return_5d'] = returns_5d

            except Exception as e:
                recommendations.at[idx, 'return_5d'] = 0
                print(f"    ⚠️ {stock}: 5日回报率计算失败 ({e})")
        else:
            recommendations.at[idx, 'return_5d'] = 0
            print(f"    ⚠️ {stock}: 历史数据不足 ({len(hist)} 天)")

        if len(hist) >= 20:
            # 计算20日波动率
            try:
                vol = hist['close'].pct_change().std()
                recommendations.at[idx, 'volatility_20d'] = vol
            except:
                recommendations.at[idx, 'volatility_20d'] = 0
        else:
            recommendations.at[idx, 'volatility_20d'] = 0

    # 3. 推荐等级
    recommendations['recommend_level'] = pd.cut(
        recommendations['position'],
        bins=[0, 0.7, 0.8, 0.9, 1.0],
        labels=['Hold', 'Accumulate', 'Buy', 'Strong Buy']
    )

    # 4. 持仓状态
    current_positions = state.get('positions', {})
    recommendations['in_portfolio'] = recommendations['instrument'].apply(
        lambda x: 'Yes' if x in current_positions else 'No'
    )

    # 5. 风险等级
    recommendations['risk_level'] = recommendations.apply(
        lambda row: classify_risk(row), axis=1
    )

    # ✅ 数据质量报告
    print(f"\n  📋 技术指标质量:")
    valid_returns = (recommendations['return_5d'] != 0).sum()
    valid_vol = (recommendations['volatility_20d'] != 0).sum()
    print(f"    有效5日回报率: {valid_returns}/{len(recommendations)}")
    print(f"    有效波动率: {valid_vol}/{len(recommendations)}")

    return recommendations


def classify_risk(row):
    """分类风险"""
    vol = row.get('volatility_20d', 0)

    if vol < 0.02:
        return 'Low'
    elif vol < 0.04:
        return 'Medium'
    else:
        return 'High'


# ========== 报告生成 ==========

def generate_report(recommendations, output_dir='./live_trading'):
    """生成推荐报告 (Top 10)"""
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.now().strftime('%Y%m%d')
    report_path = os.path.join(output_dir, f'stock_recommendations_{today}.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        # 标题
        f.write("="*90 + "\n")
        f.write("          TODAY'S TOP 10 STOCK RECOMMENDATIONS\n")
        f.write("="*90 + "\n\n")

        # 信息
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Strategy: {LiveTradingConfig.FUSION_METHOD}")
        if LiveTradingConfig.FUSION_METHOD == 'weighted':
            f.write(f" (SR:{LiveTradingConfig.FUSION_ALPHA:.0%} + ML:{LiveTradingConfig.FUSION_BETA:.0%})")
        f.write("\n\n")

        # 摘要
        f.write("-"*90 + "\n")
        f.write("SUMMARY\n")
        f.write("-"*90 + "\n\n")

        level_counts = recommendations['recommend_level'].value_counts()
        f.write("Recommendation Levels: ")
        f.write(" | ".join([f"{level}({count})" for level, count in level_counts.items()]))
        f.write("\n")

        if 'industry' in recommendations.columns:
            f.write("\nTop Industries:\n")
            industry_counts = recommendations['industry'].value_counts().head(5)
            for industry, count in industry_counts.items():
                f.write(f"  {industry}: {count}\n")

        # 详细列表
        f.write("\n\n" + "-"*90 + "\n")
        f.write(f"{'#':<3} {'Code':<12} {'Level':<13} {'Score':<8} {'SR':<8} {'ML':<8} {'Price':<10} {'5D%':<11} {'Risk':<8}\n")
        f.write("-"*90 + "\n")

        for i, (_, row) in enumerate(recommendations.iterrows(), 1):
            level = str(row.get('recommend_level', 'N/A'))[:12]
            score = row.get('position', 0)
            sr_score = row.get('stockranker_score', 0)
            ml_score = row.get('ml_score', 0)
            price = row.get('close', 0)
            ret5d = row.get('return_5d', 0)
            risk = row.get('risk_level', 'N/A')

            # 趋势图标
            if ret5d > 0.03:
                icon = "+++"
            elif ret5d > 0:
                icon = "+"
            elif ret5d < -0.03:
                icon = "---"
            elif ret5d < 0:
                icon = "-"
            else:
                icon = "="

            # 星级
            if score >= 0.9:
                stars = "***"
            elif score >= 0.8:
                stars = "**"
            elif score >= 0.7:
                stars = "*"
            else:
                stars = ""

            f.write(f"{i:<3} {row['instrument']:<12} {level:<13} {score:<8.4f} {sr_score:<8.4f} "
                   f"{ml_score:<8.4f} ${price:<9.2f} {icon}{ret5d:<10.2%} {risk:<8} {stars}\n")

        f.write("-"*90 + "\n")

        # 重点推荐
        f.write("\n\n*** FOCUS ON TOP 3 ***\n\n")
        for i, (_, row) in enumerate(recommendations.head(3).iterrows(), 1):
            f.write(f"{i}. {row['instrument']} - {row.get('recommend_level', 'N/A')}\n")
            f.write(f"   Final Score: {row['position']:.4f}\n")

            if pd.notna(row.get('stockranker_score')):
                f.write(f"   - Multi-Factor: {row['stockranker_score']:.4f}\n")

            if pd.notna(row.get('ml_score')):
                f.write(f"   - ML Prediction: {row['ml_score']:.4f}\n")

            if pd.notna(row.get('close')):
                f.write(f"   - Current Price: ${row['close']:.2f}\n")

            if pd.notna(row.get('return_5d')):
                ret = row['return_5d']
                if ret > 0.03:
                    trend = "Strong Uptrend"
                elif ret > 0:
                    trend = "Uptrend"
                elif ret > -0.01:
                    trend = "Sideways"
                else:
                    trend = "Downtrend"
                f.write(f"   - 5-Day Momentum: {ret:+.2%} ({trend})\n")

            if 'industry' in row and pd.notna(row['industry']):
                f.write(f"   - Sector: {row['industry']}\n")

            f.write("\n")

        # 风险提示
        f.write("\n" + "="*90 + "\n")
        f.write("RISK DISCLAIMER\n")
        f.write("="*90 + "\n\n")
        f.write("1. This report is generated by quantitative models for reference only\n")
        f.write("2. Not financial advice - please do your own research\n")
        f.write("3. Stock market involves significant risks\n")
        f.write("4. Past performance does not guarantee future results\n")
        f.write("5. Invest according to your risk tolerance\n")
        f.write("6. Diversification is recommended\n\n")

        f.write("This system combines multi-factor analysis with machine learning\n")
        f.write("to identify stocks with strong potential. However, all investments\n")
        f.write("carry risk and should be made with caution.\n")

    print(f"\nReport saved: {report_path}")

    return report_path


def print_recommendations(recommendations):
    """打印推荐到终端 (Top 10)"""
    print("\n" + "="*90)
    print("          TODAY'S TOP 10 STOCK RECOMMENDATIONS")
    print("="*90)

    print(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Strategy: {LiveTradingConfig.FUSION_METHOD} (SR:{LiveTradingConfig.FUSION_ALPHA:.0%} + ML:{LiveTradingConfig.FUSION_BETA:.0%})")

    # 等级分布
    if 'recommend_level' in recommendations.columns:
        level_counts = recommendations['recommend_level'].value_counts()
        print(f"\nRecommendation Levels: ", end="")
        print(" | ".join([f"{level}({count})" for level, count in level_counts.items()]))

    # 详细列表
    print("\n" + "-"*90)
    print(f"{'#':<3} {'Code':<12} {'Level':<12} {'Score':<8} {'SR':<8} {'ML':<8} {'Price':<10} {'5D%':<10} {'Risk':<8}")
    print("-"*90)

    for i, (_, row) in enumerate(recommendations.iterrows(), 1):
        level = str(row.get('recommend_level', 'N/A'))[:11]
        score = row.get('position', 0)
        sr_score = row.get('stockranker_score', 0)
        ml_score = row.get('ml_score', 0)
        price = row.get('close', 0)
        ret5d = row.get('return_5d', 0)
        risk = row.get('risk_level', 'N/A')

        # 图标
        if ret5d > 0.03:
            icon = "+++"
        elif ret5d > 0:
            icon = "+"
        elif ret5d < -0.03:
            icon = "---"
        elif ret5d < 0:
            icon = "-"
        else:
            icon = "="

        # 星级（根据评分）
        if score >= 0.9:
            stars = "***"
        elif score >= 0.8:
            stars = "**"
        elif score >= 0.7:
            stars = "*"
        else:
            stars = ""

        print(f"{i:<3} {row['instrument']:<12} {level:<12} {score:<8.4f} {sr_score:<8.4f} "
              f"{ml_score:<8.4f} ${price:<9.2f} {icon}{ret5d:<9.2%} {risk:<8} {stars}")

    print("-"*90)

    # 重点推荐 (Top 3)
    print("\n*** FOCUS ON TOP 3 ***\n")
    for i, (_, row) in enumerate(recommendations.head(3).iterrows(), 1):
        print(f"{i}. {row['instrument']} - {row.get('recommend_level', 'N/A')}")
        print(f"   Final Score: {row['position']:.4f}")

        if pd.notna(row.get('stockranker_score')):
            print(f"   - Multi-Factor: {row['stockranker_score']:.4f}")

        if pd.notna(row.get('ml_score')):
            print(f"   - ML Prediction: {row['ml_score']:.4f}")

        if pd.notna(row.get('close')):
            print(f"   - Current Price: ${row['close']:.2f}")

        if pd.notna(row.get('return_5d')):
            momentum = "Strong Uptrend" if row['return_5d'] > 0.03 else "Uptrend" if row['return_5d'] > 0 else "Downtrend"
            print(f"   - 5-Day Momentum: {row['return_5d']:+.2%} ({momentum})")

        if 'industry' in row and pd.notna(row['industry']):
            print(f"   - Sector: {row['industry']}")

        print()


def save_recommendations_csv(recommendations, output_dir='./live_trading'):
    """保存CSV"""
    os.makedirs(output_dir, exist_ok=True)

    today = datetime.now().strftime('%Y%m%d')
    csv_path = os.path.join(output_dir, f'stock_recommendations_{today}.csv')

    cols = [
        'instrument', 'position', 'recommend_level', 'close',
        'return_5d', 'volatility_20d', 'risk_level', 'in_portfolio'
    ]

    if 'stockranker_score' in recommendations.columns:
        cols.insert(2, 'stockranker_score')

    if 'ml_score' in recommendations.columns:
        cols.insert(3, 'ml_score')

    if 'industry' in recommendations.columns:
        cols.append('industry')

    cols = [c for c in cols if c in recommendations.columns]

    recommendations[cols].to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(f"CSV saved: {csv_path}")

    return csv_path


# ========== 交易功能 ==========

def check_trading_day():
    try:
        pro = ts.pro_api()
        today = datetime.now().strftime('%Y%m%d')
        cal = pro.trade_cal(exchange='SSE', start_date=today, end_date=today)
        if len(cal) == 0:
            return False
        return cal.iloc[0]['is_open'] == 1
    except:
        return True


def load_state():
    state_file = './live_trading_state.json'
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'last_rebalance_date': None, 'positions': {}}


def save_state(state):
    with open('./live_trading_state.json', 'w') as f:
        json.dump(state, f, indent=2)


def should_rebalance(state):
    last_date = state.get('last_rebalance_date')
    if not last_date:
        return True, "First run"

    last_dt = datetime.strptime(last_date, '%Y-%m-%d')
    days = (datetime.now() - last_dt).days

    if days >= LiveTradingConfig.REBALANCE_DAYS:
        return True, f"{days} days since last rebalance"

    return False, f"Only {days} days"


def get_trading_signals(recommendations):
    """从推荐提取交易信号"""
    top = recommendations.head(LiveTradingConfig.POSITION_SIZE).copy()
    weight = 1.0 / len(top)
    top['target_weight'] = weight
    top['current_price'] = top['close']

    return top[['instrument', 'position', 'target_weight', 'current_price']]


def generate_orders(signals, positions, cash, total):
    """生成订单"""
    orders = []
    target = set(signals['instrument'])
    current = set(positions.keys())

    # 卖出
    for stock in (current - target):
        orders.append({
            'stock': stock,
            'action': 'sell',
            'shares': positions[stock],
            'price': 0
        })

    # 买入
    for _, row in signals[~signals['instrument'].isin(current)].iterrows():
        amt = total * row['target_weight']
        price = row['current_price']

        if price > 0:
            shares = int(amt / price / 100) * 100
            if shares >= 100:
                orders.append({
                    'stock': row['instrument'],
                    'action': 'buy',
                    'shares': shares,
                    'price': price
                })

    return pd.DataFrame(orders)


def save_orders(orders, signals, output_dir='./live_trading'):
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime('%Y%m%d')

    orders.to_csv(
        os.path.join(output_dir, f'trading_orders_{today}.csv'),
        index=False, encoding='utf-8-sig'
    )

    signals.to_csv(
        os.path.join(output_dir, f'signals_{today}.csv'),
        index=False, encoding='utf-8-sig'
    )


# ========== 主函数 ==========

def main():
    print("\n" + "=" * 80)
    print("LIVE TRADING SYSTEM v3.1 (Enhanced)")
    print("=" * 80)
    print(f"Strategy: {LiveTradingConfig.REBALANCE_DAYS}-day rebalance")
    print(f"Scoring: {LiveTradingConfig.FUSION_METHOD}")
    print(f"Position Size: {LiveTradingConfig.POSITION_SIZE}")
    print(f"Recommendations: Top {LiveTradingConfig.RECOMMEND_TOP_N}")

    # 1. Check trading day
    print("\n[Step 1/7] Check Trading Day")
    if not check_trading_day():
        print("Not a trading day")
        return
    print("Confirmed")

    # 2. Load state
    print("\n[Step 2/7] Load State")
    state = load_state()
    if state['last_rebalance_date']:
        print(f"Last rebalance: {state['last_rebalance_date']}")
    else:
        print("First run")

    # 3. Load data
    print("\n[Step 3/7] Load Data")
    START = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    END = datetime.now().strftime('%Y-%m-%d')

    cache = DataCache(cache_dir='./data_cache')

    factor_data, price_data = load_data_with_incremental_update(
        START, END,
        max_stocks=LiveTradingConfig.SAMPLE_SIZE,
        cache_manager=cache,
        use_stockranker=True,
        tushare_token=TUSHARE_TOKEN,
        use_fundamental=True,
        max_workers=10
    )

    if factor_data is None:
        print("Data loading failed")
        return

    print("Data loaded")

    # 4. Score processing
    print("\n[Step 4/7] Score Processing")
    factor_data = fix_stockranker_scoring(factor_data)
    factor_data = fix_ml_scoring(factor_data, price_data)
    factor_data = fuse_scores(factor_data)
    print("Scoring completed")

    # 5. Generate recommendations (KEY FEATURE)
    print("\n[Step 5/7] Generate Recommendations")
    recommendations = generate_recommendations(factor_data, price_data, state)

    # Print to console
    print_recommendations(recommendations)

    # Save report
    generate_report(recommendations)
    save_recommendations_csv(recommendations)

    # 6. Check rebalance
    print("\n[Step 6/7] Check Rebalance")
    need_rebal, reason = should_rebalance(state)
    print(f"Rebalance: {need_rebal} ({reason})")

    if not need_rebal:
        print("\nNo rebalance needed today")
        return

    # 7. Generate trading orders
    print("\n[Step 7/7] Generate Orders")
    signals = get_trading_signals(recommendations)
    positions = state.get('positions', {})

    orders = generate_orders(signals, positions, 1000000, 1000000)

    if len(orders) > 0:
        print(f"\nOrders: {len(orders)}")
        for _, order in orders.iterrows():
            print(f"  {order['action'].upper()} {order['stock']} {int(order['shares'])} shares")

        save_orders(orders, signals)

        # Update state
        state['last_rebalance_date'] = datetime.now().strftime('%Y-%m-%d')
        for _, order in orders.iterrows():
            if order['action'] == 'sell':
                if order['stock'] in state['positions']:
                    del state['positions'][order['stock']]
            elif order['action'] == 'buy':
                state['positions'][order['stock']] = int(order['shares'])

        save_state(state)
        print("\nState updated")
    else:
        print("\nNo orders needed")

    print("\n" + "=" * 80)
    print("COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()