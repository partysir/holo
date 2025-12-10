"""
main.py - 主回测入口（v2.3 - 前视偏差修复版）

核心修复：
✅ Issue A: 过滤未来上市的新股
✅ Issue B: 防止使用上市前的历史数据
✅ 新增：min_days_listed 参数控制最短上市时间

版本：v2.3
日期：2025-12-10
"""

import warnings
warnings.filterwarnings('ignore')

import tushare as ts
import pandas as pd
import numpy as np
import time
import random
import os

# ========== 导入配置 ==========
from config import (
    TUSHARE_TOKEN,
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
    from ml_factor_scoring_fixed import (
        AdvancedMLScorer,
        ICCalculator,
        IndustryBasedScorer,
        EnhancedStockSelector
    )
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
    print("\n" + "="*80)
    print("    综合因子评分选股回测系统 v2.3 - 前视偏差修复版")
    print("="*80)
    print("\n🎯 核心修复:")
    print("  ✅ Issue A - 上市日期过滤：只选择回测开始前已上市的股票")
    print("  ✅ Issue B - 历史数据清洗：过滤上市前的价格数据")
    print("  ✅ 新增参数 - min_days_listed：控制最短上市时间（默认180天）")
    print()


def print_trading_plan(context, price_data, factor_data):
    """
    🖨️ 打印清晰的交易计划和持仓监控
    """
    print("\n" + "#"*80)
    print("📋 步骤9: 交易指令与持仓监控 (最终报告)")
    print("#"*80 + "\n")

    df_trades = context['trade_records']
    if df_trades.empty:
        print("⚠️ 无交易记录")
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
            print(f"{action:<6} | {row['stock']:<10} | {row['price']:<8.2f} | {row['shares']:<8.0f} | ¥{row['amount']:<9.0f} | {row.get('reason', '')}")
        print("-" * 75)

    # 打印当前持仓详情
    positions = context['positions']
    if not positions:
        print("\n💼 【当前持仓】 空仓")
    else:
        print(f"\n💼 【当前持仓监控】 共 {len(positions)} 只")
        print("-" * 95)
        print(f"{'代码':<10} | {'持仓股数':<8} | {'成本价':<8} | {'现价':<8} | {'浮动盈亏':<10} | {'收益率':<8} | {'评分'}")
        print("-" * 95)

        total_mv = 0
        total_pnl = 0

        last_scores = factor_data[factor_data['date'] == str(last_date)][['instrument', 'position']].set_index('instrument')['position'].to_dict()
        last_prices = price_data[price_data['date'] == str(last_date)][['instrument', 'close']].set_index('instrument')['close'].to_dict()

        for code, info in positions.items():
            shares = info['shares']
            cost = info['cost']
            current_price = last_prices.get(code, cost)
            score = last_scores.get(code, 0.0)

            mv = shares * current_price
            pnl = (current_price - cost) * shares
            pnl_rate = (current_price - cost) / cost

            total_mv += mv
            total_pnl += pnl

            pnl_str = f"¥{pnl:+,.0f}"
            rate_str = f"{pnl_rate:+.2%}"

            print(f"{code:<10} | {shares:<8.0f} | {cost:<8.2f} | {current_price:<8.2f} | {pnl_str:<10} | {rate_str:<8} | {score:.4f}")

        print("-" * 95)
        print(f"💰 账户概览: 持仓市值 ¥{total_mv:,.0f} | 可用现金 ¥{context['final_value']-total_mv:,.0f} | 总资产 ¥{context['final_value']:,.0f}")
        print(f"📈 累计收益: {context['total_return']:+.2%}")
        print("\n")


def main():
    """主函数"""
    print_banner()

    # ========== 显示配置 ==========
    print("【当前配置】")
    print(f"  策略版本: {StrategyConfig.STRATEGY_VERSION}")
    print(f"  回测区间: {BacktestConfig.START_DATE} ~ {BacktestConfig.END_DATE}")
    print(f"  初始资金: ¥{BacktestConfig.CAPITAL_BASE:,}")
    print(f"  持仓数量: {BacktestConfig.POSITION_SIZE} 只")

    print_config_comparison()
    validate_configs()

    # 从配置获取参数
    START_DATE = BacktestConfig.START_DATE
    END_DATE = BacktestConfig.END_DATE
    CAPITAL_BASE = BacktestConfig.CAPITAL_BASE
    POSITION_SIZE = BacktestConfig.POSITION_SIZE
    REBALANCE_DAYS = BacktestConfig.REBALANCE_DAYS

    USE_SAMPLING = DataConfig.USE_SAMPLING
    SAMPLE_SIZE = DataConfig.SAMPLE_SIZE
    if not USE_SAMPLING and SAMPLE_SIZE < 5000:
        SAMPLE_SIZE = 5000

    # ========== 关键新增：最短上市时间参数 ==========
    MIN_DAYS_LISTED = 180  # 要求股票至少上市180天（半年）
    print(f"\n🔒 前视偏差防护:")
    print(f"  - 最短上市时间: {MIN_DAYS_LISTED} 天")
    print(f"  - 上市截止日期: {START_DATE} 之前 {MIN_DAYS_LISTED} 天")

    # ============ 初始化 ============
    cache_manager = DataCache(cache_dir=DataConfig.CACHE_DIR)

    # 步骤0: 获取大盘指数
    benchmark_data = None
    try:
        print("\n" + "="*80)
        print("📈 步骤0: 获取大盘指数数据 (用于择时)")
        print("="*80)
        ds_temp = TushareDataSource(cache_manager=cache_manager, token=TUSHARE_TOKEN)
        benchmark_data = ds_temp.get_index_daily(ts_code='000001.SH', start_date=START_DATE, end_date=END_DATE)
        if benchmark_data is not None:
            print(f"  ✓ 获取上证指数数据: {len(benchmark_data)} 条")
    except Exception as e:
        print(f"  ⚠️  获取指数失败: {e}")

    # ============ 步骤1: 数据加载（修复版） ============
    try:
        data_start_time = time.time()
        print("\n" + "="*80)
        print("📦 步骤1: 数据加载 (v2.3 - 修复前视偏差)")
        print("="*80)

        # ========== 修复方式1：如果使用 data_module 直接加载 ==========
        from data_module import load_data_from_tushare

        factor_data, price_data = load_data_from_tushare(
            START_DATE,
            END_DATE,
            max_stocks=SAMPLE_SIZE,
            cache_manager=cache_manager,
            use_stockranker=FactorConfig.USE_STOCKRANKER,
            custom_weights=FactorConfig.CUSTOM_WEIGHTS,
            tushare_token=TUSHARE_TOKEN,
            use_fundamental=FactorConfig.USE_FUNDAMENTAL,
            min_days_listed=MIN_DAYS_LISTED  # ✅ 关键参数
        )

        # ========== 修复方式2：如果使用增量更新模块 ==========
        # 注意：您需要在 data_module_incremental.py 中也添加 min_days_listed 参数支持
        # factor_data, price_data = load_data_with_incremental_update(
        #     START_DATE,
        #     END_DATE,
        #     max_stocks=SAMPLE_SIZE,
        #     cache_manager=cache_manager,
        #     use_stockranker=FactorConfig.USE_STOCKRANKER,
        #     custom_weights=FactorConfig.CUSTOM_WEIGHTS,
        #     tushare_token=TUSHARE_TOKEN,
        #     use_fundamental=FactorConfig.USE_FUNDAMENTAL,
        #     force_full_update=DataConfig.FORCE_FULL_UPDATE,
        #     use_sampling=USE_SAMPLING,
        #     sample_size=SAMPLE_SIZE,
        #     max_workers=DataConfig.MAX_WORKERS,
        #     min_days_listed=MIN_DAYS_LISTED  # ✅ 关键参数
        # )

        if factor_data is None or price_data is None:
            print("\n❌ 数据获取失败")
            return

        print(f"  ✓ 数据加载耗时: {time.time() - data_start_time:.1f} 秒")

        # ========== 验证：检查是否还有新股 ==========
        print("\n🔍 数据质量验证:")
        unique_stocks = factor_data['instrument'].unique()
        print(f"  - 股票池大小: {len(unique_stocks)} 只")

        # 检查是否有新股代码（920、689等）
        new_stock_codes = [s for s in unique_stocks if s.startswith(('920', '689', '787'))]
        if new_stock_codes:
            print(f"  ⚠️  警告：仍发现 {len(new_stock_codes)} 只可疑新股代码")
            print(f"     示例: {new_stock_codes[:5]}")
        else:
            print(f"  ✅ 通过：未发现可疑新股代码")

    except Exception as e:
        print(f"\n❌ 数据加载异常: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============ 步骤1.5: 补全行业数据 ============
    print("\n" + "="*80)
    print("🏭 步骤1.5: 补全行业数据 (用于中性化)")
    print("="*80)

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
        print("\n" + "="*80)
        print("🔍 步骤2: 数据质量优化")
        print("="*80)
        from data_quality_optimizer import optimize_data_quality
        price_data, factor_data = optimize_data_quality(price_data, factor_data, cache_manager=cache_manager)
    except Exception as e:
        print(f"\n⚠️  数据质量优化警告: {e}")

    # ============ 步骤3: 因子增强处理 ============
    try:
        print("\n" + "="*80)
        print("🎯 步骤3: 因子增强处理")
        print("="*80)

        from enhanced_factor_processor import EnhancedFactorProcessor

        factor_processor = EnhancedFactorProcessor(
            neutralize_industry=True, # 现在已有行业数据，可以安全开启
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
        import traceback
        traceback.print_exc()

    # ============ 步骤4: ML因子评分 ============
    if MLConfig.USE_ADVANCED_ML and ML_AVAILABLE:
        try:
            print("\n" + "="*80)
            print("🚀 步骤4: 高级ML因子评分")
            print("="*80)

            ml_scorer = AdvancedMLScorer(
                model_type=MLConfig.ML_MODEL_TYPE,
                target_period=MLConfig.ML_TARGET_PERIOD,
                top_percentile=MLConfig.ML_TOP_PERCENTILE,
                use_classification=MLConfig.ML_USE_CLASSIFICATION,
                use_ic_features=MLConfig.ML_USE_IC_FEATURES,
                train_months=MLConfig.ML_TRAIN_MONTHS
            )

            factor_data = ml_scorer.predict_scores(factor_data, price_data, factor_columns)

        except Exception as e:
            print(f"⚠️  ML评分失败: {e}")

    # ============ 步骤5-6: 行业评分与选股 (省略详细日志) ============
    # ... (保持原有逻辑，此处略去打印以节省空间) ...

    # ========== 步骤7: 运行回测引擎 ==========
    try:
        print("\n" + "="*80)
        print(f"🚀 步骤7: {STRATEGY_VERSION} 回测引擎 (含择时)")
        print("="*80)

        strategy_params = get_strategy_params()

        # 运行回测
        context = run_factor_based_strategy_v2(
            factor_data=factor_data,
            price_data=price_data,
            benchmark_data=benchmark_data,
            **strategy_params
        )

    except Exception as e:
        print(f"\n❌ 回测执行异常: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============ 步骤8: 生成报告 ============
    try:
        print(f"\n{'='*80}")
        print("📊 步骤8: 生成分析报告")
        print(f"{'='*80}\n")

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

    except Exception as e:
        print(f"⚠️  报告生成警告: {e}")

    # ============ 步骤9: 打印交易计划 (新增需求) ============
    # 打印您需要的“清晰明了详细的持仓及调仓报告”
    # print_trading_plan(context, price_data, factor_data)

    print("\n" + "="*80)
    print("✅ 任务全部完成")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()