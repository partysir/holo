#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合因子评分选股回测系统 v2.3 - 前视偏差修复版
"""

import time
import traceback
from datetime import datetime

import warnings

warnings.filterwarnings('ignore')

import time
import random
import os
import traceback

import tushare as ts
import pandas as pd
import numpy as np

# ========== 导入配置 ==========
from config import (
    TUSHARE_TOKEN,  # 修复了此处的换行错误
    StrategyConfig,
    BacktestConfig,
    RiskControlConfig,
    TradingCostConfig,
    DataConfig,
    FactorConfig,
    MLConfig,
    OutputConfig,
    get_strategy_params,
    validate_configs,
    print_config_comparison
)

ts.set_token(TUSHARE_TOKEN)

# 导入数据模块
from data_module import DataCache, TushareDataSource
from data_module_incremental import load_data_with_incremental_update

# ========== 导入高级ML模块 ==========
ML_AVAILABLE = False
try:
    from ml_factor_scoring_fixed import UltraMLScorer

    ML_AVAILABLE = True
    print("✓ 高级ML模块加载成功")
except ImportError as e:
    print(f"⚠️  高级ML模块未找到: {e}")
    ML_AVAILABLE = False

# ========== 导入策略引擎 ==========
try:
    from factor_based_risk_control_optimized import run_factor_based_strategy_v2

    print("✓ v2.1优化版策略引擎加载成功")
    STRATEGY_VERSION = "v2.0"
except ImportError:
    print("⚠️  v2.0优化版未找到，使用v1.0")
    from factor_based_risk_control import run_factor_based_strategy

    STRATEGY_VERSION = "v1.0"

from visualization_module import (
    plot_monitoring_results,
    plot_top_stocks_evolution,
    generate_performance_report
)
from show_today_holdings import show_today_holdings_dashboard
from holdings_monitor import generate_daily_holdings_report
from date_organized_reports import generate_date_organized_reports


def print_banner():
    """打印启动横幅"""
    print("\n" + "=" * 80)
    print("    综合因子评分选股回测系统 v2.3 - 前视偏差修复版")
    print("=" * 80)
    print("\n🎯 核心修复:")
    print("  ✅ Issue A - 上市日期过滤：只选择回测开始前已上市的股票")
    print("  ✅ Issue B - 历史数据清洗：过滤上市前的价格数据")
    print("  ✅ 新增参数 - min_days_listed：控制最短上市时间（默认180天）")
    print()


def print_trading_plan(context, price_data, factor_data):
    """
    🖨️ 打印清晰的交易计划和持仓监控
    """
    """
    添加到 main.py 的末尾，在 print_trading_plan(context, price_data, factor_data) 之后
    """

    # ============ 诊断代码：检查股数异常 ============
    print("\n" + "=" * 80)
    print("🔍 股数异常诊断")
    print("=" * 80)

    if context is not None:
        import pandas as pd

        # 1. 检查交易记录
        df_trades = pd.DataFrame(context['trade_records'])

        if not df_trades.empty:
            buy_trades = df_trades[df_trades['action'] == 'buy'].copy()

            print(f"\n📊 买入交易统计:")
            print(f"  总买入次数: {len(buy_trades)}")

            if len(buy_trades) > 0:
                print(f"\n前10笔买入交易:")
                print(buy_trades[['date', 'stock', 'shares', 'price', 'amount', 'cash_before', 'cash_after']].head(
                    10).to_string())

                # 检查第一笔买入
                first_buy = buy_trades.iloc[0]
                print(f"\n🔎 第一笔买入详细分析:")
                print(f"  日期: {first_buy['date']}")
                print(f"  股票: {first_buy['stock']}")
                print(f"  股数: {first_buy['shares']:,.0f}")
                print(f"  价格: ¥{first_buy['price']:.2f}")
                print(f"  金额: ¥{first_buy['amount']:,.2f}")
                print(f"  买入前现金: ¥{first_buy['cash_before']:,.2f}")
                print(f"  买入后现金: ¥{first_buy['cash_after']:,.2f}")

                # 验证计算
                expected_cost = first_buy['shares'] * first_buy['price'] * 1.0003
                actual_spent = first_buy['cash_before'] - first_buy['cash_after']

                print(f"\n验证:")
                print(
                    f"  计算金额: {first_buy['shares']:,.0f} × ¥{first_buy['price']:.2f} × 1.0003 = ¥{expected_cost:,.2f}")
                print(f"  记录金额: ¥{first_buy['amount']:,.2f}")
                print(f"  实际花费: ¥{actual_spent:,.2f}")
                print(f"  金额误差: ¥{abs(expected_cost - first_buy['amount']):,.2f}")

                # 检查股数分布
                print(f"\n📈 股数分布统计:")
                print(f"  最小股数: {buy_trades['shares'].min():,.0f}")
                print(f"  最大股数: {buy_trades['shares'].max():,.0f}")
                print(f"  平均股数: {buy_trades['shares'].mean():,.0f}")
                print(f"  中位数股数: {buy_trades['shares'].median():,.0f}")

                # 找出异常大的股数
                abnormal_trades = buy_trades[buy_trades['shares'] > 100000].copy()
                if len(abnormal_trades) > 0:
                    print(f"\n⚠️ 发现 {len(abnormal_trades)} 笔股数异常交易 (>100,000股):")
                    print(abnormal_trades[['date', 'stock', 'shares', 'price', 'amount']].to_string())

        # 2. 检查最终持仓
        positions = context.get('positions', {})
        print(f"\n💼 最终持仓检查:")
        print(f"  持仓数量: {len(positions)}")

        if positions:
            print(f"\n持仓详情:")
            for stock, info in positions.items():
                print(f"  {stock}: {info['shares']:,.0f} 股 @ ¥{info['cost']:.2f}")

                # 检查是否异常
                if info['shares'] > 100000:
                    print(f"    ⚠️ 股数异常！超过10万股")
                    print(f"    买入日期: {info['entry_date']}")

                    # 查找这只股票的所有买入记录
                    stock_buys = buy_trades[buy_trades['stock'] == stock].copy()
                    if len(stock_buys) > 0:
                        print(f"    该股票的所有买入记录:")
                        print(stock_buys[['date', 'shares', 'price', 'amount']].to_string())

        # 3. 检查现金流
        print(f"\n💵 现金流检查:")
        initial_cash = context.get('initial_capital', 1000000)
        final_cash = context.get('final_cash', 0)
        print(f"  初始资金: ¥{initial_cash:,.2f}")
        print(f"  最终现金: ¥{final_cash:,.2f}")

        total_buy = buy_trades['amount'].sum() if not buy_trades.empty else 0
        sell_trades = df_trades[df_trades['action'] == 'sell'].copy()
        total_sell = sell_trades['amount'].sum() if not sell_trades.empty else 0

        print(f"  累计买入: ¥{total_buy:,.2f}")
        print(f"  累计卖出: ¥{total_sell:,.2f}")
        print(f"  净流出: ¥{total_buy - total_sell:,.2f}")

        expected_cash = initial_cash - total_buy + total_sell
        print(f"  期望现金: ¥{expected_cash:,.2f}")
        print(f"  现金误差: ¥{abs(expected_cash - final_cash):,.2f}")

    print("\n" + "=" * 80)
    if context is None:
        return

    print("\n" + "#" * 80)
    print("📋 步骤9: 交易指令与持仓监控 (最终报告)")
    print("#" * 80 + "\n")

    df_trades = context.get('trade_records', pd.DataFrame())
    if df_trades.empty:
        print("⚠️ 全程无交易记录")
        return

    last_date = df_trades['date'].max()
    today_trades = df_trades[df_trades['date'] == last_date].copy()

    print(f"📅 信号日期: {last_date}")

    # 打印调仓指令
    print(f"\n📢 【今日调仓指令】 共 {len(today_trades)} 笔")
    if len(today_trades) == 0:
        print("   ✅ 今日无操作，继续持仓。")
    else:
        print("-" * 75)
        print(f"{'方向':<6} | {'代码':<10} | {'价格':<8} | {'股数':<8} | {'金额':<10} | {'原因'}")
        print("-" * 75)

        for _, row in today_trades.iterrows():
            action = "🔵买入" if row['action'] == 'buy' else "🔴卖出"
            price_val = row['price'] if pd.notnull(row['price']) else 0
            shares_val = row['shares'] if pd.notnull(row['shares']) else 0
            amount_val = row['amount'] if pd.notnull(row['amount']) else 0

            print(
                f"{action:<6} | {row['stock']:<10} | {price_val:<8.2f} | {shares_val:<8.0f} | ¥{amount_val:<9.0f} | {row.get('reason', '')}")
        print("-" * 75)

    # 打印当前持仓详情
    positions = context.get('positions', {})
    final_value = context.get('final_value', 0)
    total_return = context.get('total_return', 0)

    if not positions:
        print("\n💼 【当前持仓】 空仓")
    else:
        print(f"\n💼 【当前持仓监控】 共 {len(positions)} 只")
        print("-" * 125)
        print(
            f"{'代码':<10} | {'买入日期':<12} | {'持仓股数':<8} | {'持仓占比':<8} | {'成本价':<8} | {'现价':<8} | {'浮动盈亏':<10} | {'收益率':<8} | {'评分'}")
        print("-" * 125)

        total_mv = 0
        total_pnl = 0

        # 获取最后一天的数据用于展示
        try:
            last_scores = \
            factor_data[factor_data['date'] == str(last_date)][['instrument', 'position']].set_index('instrument')[
                'position'].to_dict()
            last_prices = \
            price_data[price_data['date'] == str(last_date)][['instrument', 'close']].set_index('instrument')[
                'close'].to_dict()
        except Exception:
            last_scores = {}
            last_prices = {}

        for code, info in positions.items():
            shares = info['shares']
            cost = info['cost']
            entry_date = info['entry_date']  # 买入日期
            current_price = last_prices.get(code, cost)  # 如果没有现价，暂用成本价代替
            score = last_scores.get(code, 0.0)

            mv = shares * current_price
            pnl = (current_price - cost) * shares
            pnl_rate = (current_price - cost) / cost if cost != 0 else 0

            # 计算持仓占比（假设我们有总资产信息）
            position_ratio = mv / final_value if final_value > 0 else 0

            total_mv += mv
            total_pnl += pnl

            pnl_str = f"¥{pnl:+,.0f}"
            rate_str = f"{pnl_rate:+.2%}"
            ratio_str = f"{position_ratio:.2%}"

            print(
                f"{code:<10} | {entry_date:<12} | {shares:<8.0f} | {ratio_str:<8} | {cost:<8.2f} | {current_price:<8.2f} | {pnl_str:<10} | {rate_str:<8} | {score:.4f}")

        print("-" * 125)
        cash = final_value - total_mv
        print(f"💰 账户概览: 持仓市值 ¥{total_mv:,.0f} | 可用现金 ¥{cash:,.0f} | 总资产 ¥{final_value:,.0f}")
        print(f"📈 累计收益: {total_return:+.2%}")
        print("\n")


def main():
    """主函数"""
    print_banner()

    # ============ 显示配置 ============
    print("【当前配置】")
    print(f"  策略版本: {StrategyConfig.STRATEGY_VERSION}")
    print(f"  回测区间: {BacktestConfig.START_DATE} ~ {BacktestConfig.END_DATE}")
    print(f"  初始资金: ¥{BacktestConfig.CAPITAL_BASE:,}")
    print(f"  持仓数量: {BacktestConfig.POSITION_SIZE} 只")

    # 从配置获取参数
    START_DATE = BacktestConfig.START_DATE
    END_DATE = BacktestConfig.END_DATE
    CAPITAL_BASE = BacktestConfig.CAPITAL_BASE
    POSITION_SIZE = BacktestConfig.POSITION_SIZE
    REBALANCE_DAYS = BacktestConfig.REBALANCE_DAYS

    # 减少股票数量以节省内存
    USE_SAMPLING = True  # 启用采样
    SAMPLE_SIZE = 4000  # 减少到4000只股票进行测试
    if not USE_SAMPLING and SAMPLE_SIZE < 5000:
        SAMPLE_SIZE = 5000

    # ========== 关键新增：最短上市时间参数 ==========
    MIN_DAYS_LISTED = 60  # 要求股票至少上市60天（2个月）
    print(f"\n🔒 前视偏差防护:")
    print(f"  - 最短上市时间: {MIN_DAYS_LISTED} 天")
    print(f"  - 效果: 剔除在 {START_DATE} 前 {MIN_DAYS_LISTED} 天内上市的次新股")

    # ============ 初始化 ============
    cache_manager = DataCache(cache_dir=DataConfig.CACHE_DIR)

    # 步骤0: 获取大盘指数
    benchmark_data = None
    try:
        if StrategyConfig.ENABLE_MARKET_TIMING:
            print("\n" + "=" * 80)
            print("📈 步骤0: 获取大盘指数数据 (用于择时)")
            print("=" * 80)
            ds_temp = TushareDataSource(cache_manager=cache_manager, token=TUSHARE_TOKEN)
            benchmark_data = ds_temp.get_index_daily(ts_code='000001.SH', start_date=START_DATE, end_date=END_DATE)
            if benchmark_data is not None:
                print(f"  ✓ 获取上证指数数据: {len(benchmark_data)} 条")
        else:
            print("\n" + "=" * 80)
            print("⏭️  步骤0: 大盘择时已禁用")
            print("=" * 80)
            print("  ℹ️  跳过大盘指数数据获取")
    except Exception as e:
        print(f"  ⚠️  获取指数失败: {e}")

    # ============ 步骤1: 数据加载（修复版） ============
    try:
        data_start_time = time.time()
        print("\n" + "=" * 80)
        print("📦 步骤1: 数据加载 (v2.3 - 修复前视偏差)")
        print("=" * 80)

        # ========== 使用增量更新模块加载数据 ==========
        # 注意：load_data_with_incremental_update 需要在内部支持 min_days_listed 参数
        factor_data, price_data = load_data_with_incremental_update(
            START_DATE,
            END_DATE,
            max_stocks=SAMPLE_SIZE,
            cache_manager=cache_manager,
            use_stockranker=FactorConfig.USE_STOCKRANKER,
            custom_weights=FactorConfig.CUSTOM_WEIGHTS,
            tushare_token=TUSHARE_TOKEN,
            use_fundamental=FactorConfig.USE_FUNDAMENTAL,
            force_full_update=DataConfig.FORCE_FULL_UPDATE,
            use_sampling=USE_SAMPLING,
            sample_size=SAMPLE_SIZE,
            max_workers=DataConfig.MAX_WORKERS,
            min_days_listed=MIN_DAYS_LISTED,  # ✅ 关键参数：传递给数据加载器
            use_money_flow=FactorConfig.USE_MONEY_FLOW  # ✅ 启用资金流因子
        )

        if factor_data is None or price_data is None:
            print("\n❌ 数据获取失败")
            return

        if factor_data.empty or price_data.empty:
            print("\n❌ 获取到的数据为空，请检查日期范围或Token")
            return

        print(f"  ✓ 数据加载耗时: {time.time() - data_start_time:.1f} 秒")

        # ========== 验证：检查是否还有新股 ==========
        print("\n🔍 数据质量验证:")
        unique_stocks = factor_data['instrument'].unique()
        print(f"  - 股票池大小: {len(unique_stocks)} 只")

        # 检查是否有新股代码（920北交所、689科创板部分等，视需求过滤）
        # 这里仅作提示，不强制删除，因为 data_module 应该已经处理了 min_days_listed
        new_stock_codes = [s for s in unique_stocks if s.startswith(('920', '8', '4'))]  # 示例：检查北交所等
        if new_stock_codes:
            print(f"  ℹ️  提示：包含 {len(new_stock_codes)} 只北交所/新三板代码")

        print(f"  ✅ 数据加载完成，已应用上市时间过滤 (min_days_listed={MIN_DAYS_LISTED})")

    except Exception as e:
        print(f"\n❌ 数据加载异常: {e}")
        traceback.print_exc()
        return

    # ============ 步骤1.5: 补全行业数据 ============
    print("\n" + "=" * 80)
    print("🏭 步骤1.5: 补全行业数据 (用于中性化)")
    print("=" * 80)

    try:
        ds = TushareDataSource(token=TUSHARE_TOKEN, cache_manager=cache_manager)
        unique_stocks = factor_data['instrument'].unique().tolist()
        industry_df = ds.get_industry_data(unique_stocks, use_cache=True)

        if industry_df is not None and not industry_df.empty:
            if 'industry' in factor_data.columns:
                del factor_data['industry']
            factor_data = factor_data.merge(industry_df, on='instrument', how='left')
            factor_data['industry'] = factor_data['industry'].fillna('其他')
            print(f"  ✓ 成功合并行业数据: 覆盖 {factor_data['industry'].nunique()} 个行业")
        else:
            print("  ⚠️  未获取到行业数据，使用默认值")
            factor_data['industry'] = 'Unknown'

    except Exception as e:
        print(f"  ⚠️  补全行业数据失败: {e}")
        if 'industry' not in factor_data.columns:
            factor_data['industry'] = 'Unknown'

    # ============ 步骤2: 数据质量优化 ============
    try:
        print("\n" + "=" * 80)
        print("🔍 步骤2: 数据质量优化")
        print("=" * 80)
        from data_quality_optimizer import optimize_data_quality
        price_data, factor_data = optimize_data_quality(price_data, factor_data, cache_manager=cache_manager)
    except Exception as e:
        print(f"\n⚠️  数据质量优化警告: {e}")

    # ============ 步骤3: 因子增强处理 ============
    try:
        print("\n" + "=" * 80)
        print("🎯 步骤3: 因子增强处理")
        print("=" * 80)

        from enhanced_factor_processor import EnhancedFactorProcessor

        factor_processor = EnhancedFactorProcessor(
            neutralize_industry=True,  # 现在已有行业数据，可以安全开启
            neutralize_market=False
        )

        exclude_columns = ['date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'amount', 'industry']
        factor_columns = [col for col in factor_data.columns if col not in exclude_columns]

        # 确保只处理数值列
        factor_columns = [c for c in factor_columns if pd.api.types.is_numeric_dtype(factor_data[c])]

        print(f"  检测到 {len(factor_columns)} 个有效因子列")

        if len(factor_columns) > 0:
            factor_data = factor_processor.process_factors(factor_data, factor_columns)

    except Exception as e:
        print(f"\n⚠️  因子增强处理警告: {e}")
        traceback.print_exc()

    # ============ 步骤4: ML因子评分 (修复集成版) ============
    if MLConfig.USE_ADVANCED_ML and ML_AVAILABLE:
        try:
            print("\n" + "=" * 80)
            print("🚀 步骤4: 高级ML因子评分 (Ultra Mode)")
            print("   ✨ 启用 Strict Voting (双重确认) 以提升胜率")
            print("=" * 80)

            # ✅ 实例化超级评分器
            ml_scorer = UltraMLScorer(
                target_period=MLConfig.ML_TARGET_PERIOD,
                top_percentile=MLConfig.ML_TOP_PERCENTILE,
                train_months=MLConfig.ML_TRAIN_MONTHS,
                # ✅ 关键：使用 'strict' 策略，只有多个模型共识才给高分
                voting_strategy='strict',
                # ✅ 关键：启用特征正交化，提取纯Alpha
                neutralize_market=True,
                neutralize_industry=True
            )

            # ✅ 调用 predict 方法 (注意方法名差异)
            # UltraMLScorer 会自动处理训练和预测
            factor_data = ml_scorer.predict(factor_data, price_data)

        except Exception as e:
            print(f"⚠️  ML评分失败: {e}")
            import traceback
            traceback.print_exc()

    # ========== 步骤7: 运行回测引擎 ==========
    context = None
    try:
        print("\n" + "=" * 80)
        print(f"🚀 步骤7: {STRATEGY_VERSION} 回测引擎 (含择时)")
        print("=" * 80)

        strategy_params = get_strategy_params()
        # 添加调仓周期参数
        strategy_params['rebalance_days'] = REBALANCE_DAYS

        # 运行回测
        context = run_factor_based_strategy_v2(
            factor_data=factor_data,
            price_data=price_data,
            benchmark_data=benchmark_data,
            **strategy_params
        )

    except Exception as e:
        print(f"\n❌ 回测执行异常: {e}")
        traceback.print_exc()
        return

    # ============ 步骤8: 生成报告 ============
    try:
        print(f"\n{'=' * 80}")
        print("📊 步骤8: 生成分析报告")
        print(f"{'=' * 80}\n")

        # 生成按日期组织的文件夹
        date_folder = generate_date_organized_reports(
            context=context,
            factor_data=factor_data,
            price_data=price_data,
            base_dir=OutputConfig.REPORTS_DIR
        )

        # 生成持仓面板
        show_today_holdings_dashboard(
            context=context,
            factor_data=factor_data,
            price_data=price_data,
            output_dir=date_folder
        )

        # 生成详细的持仓和交易报告，并获取总盈亏信息
        print("\n" + "=" * 80)
        print("📋 生成详细持仓和交易报告")
        print("=" * 80)

        from holdings_monitor import generate_daily_holdings_report

        daily_holdings, pnl_info = generate_daily_holdings_report(
            context=context,
            factor_data=factor_data,
            price_data=price_data,
            output_dir=date_folder,
            print_to_console=True,
            save_to_csv=True
        )

        # 获取绩效报告信息（包含年化收益率等指标）
        from visualization_module import generate_performance_report
        performance_info = generate_performance_report(context, output_dir=date_folder)

        # 显示总盈亏信息
        if pnl_info:
            print("\n" + "=" * 80)
            print("💰 交易绩效摘要")
            print("=" * 80)
            print(f"  总交易次数: {pnl_info['trade_count']}")
            print(f"  买入次数: {pnl_info['buy_count']}")
            print(f"  卖出次数: {pnl_info['sell_count']}")
            print(f"  盈利次数: {pnl_info['profit_trades']}")
            print(f"  亏损次数: {pnl_info['loss_trades']}")
            print(f"  总盈利 (正盈亏部分): ¥{pnl_info['total_profit']:,.2f}")
            print(f"  总亏损 (负盈亏部分): ¥{pnl_info['total_loss']:,.2f}")
            print(f"  净盈亏 (总盈利 + 总亏损): ¥{pnl_info['net_pnl']:,.2f}")
            print(f"  交易费用总和: ¥{pnl_info['total_fees']:,.2f}")
            print(f"  扣除费用后净盈亏: ¥{pnl_info['net_pnl_after_fees']:,.2f}")

            # ✅ 修复：使用正确的初始资金计算净收益率
            if 'initial_capital' in context and context['initial_capital'] > 0:
                net_return = pnl_info['net_pnl_after_fees'] / context['initial_capital']
                print(f"  净收益率: {net_return:+.2%}")

            if pnl_info and 'correct_return_rate' in pnl_info:
                print(f"\n📈 正确的绩效指标:")
                print(f"  总净盈亏: ¥{pnl_info['total_net_pnl']:,.2f}")
                print(f"  正确收益率: {pnl_info['correct_return_rate']:+.2%}")
                print(f"  (基于初始资金: ¥{pnl_info['initial_capital']:,.0f})")

        # 显示年化收益率等绩效指标
        if performance_info:
            print(f"\n📈 绩效指标:")
            print(f"  总收益率: {performance_info['total_return']:+.2%}")
            print(f"  年化收益率: {performance_info['annualized_return']:+.2%}")
            print(f"  最大回撤: {performance_info['max_drawdown']:.2%}")
            print(f"  夏普比率: {performance_info['sharpe_ratio']:.4f}")

    except Exception as e:
        print(f"⚠️  报告生成警告: {e}")

    # ============ 步骤9: 打印交易计划 (启用) ============
    # 启用之前注释掉的代码，确保用户能看到结果
    print_trading_plan(context, price_data, factor_data)

    print("\n" + "=" * 80)
    print("✅ 任务全部完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        traceback.print_exc()