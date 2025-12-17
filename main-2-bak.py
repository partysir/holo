"""
main.py - 主回测入口（v2.6 - 实盘精选版 Top5）

核心更新：
✅ 实盘清单优化: 仅输出评分最高的 Top 5 股票，便于聚焦
✅ 全流程保留: 包含前视偏差修复、Walk-Forward 全窗口训练、XGBoost 兼容性修复

版本：v2.6
日期：2025-12-15
"""

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

# ========== 导入高级ML模块 (适配 ml_factor_scoring_fixed.py) ==========
ML_AVAILABLE = False
try:
    # 注意：确保目录下有 ml_factor_scoring_fixed.py 文件
    from ml_factor_scoring_fixed import (
        AdvancedMLScorer,
        ICCalculator,
        IndustryBasedScorer,
        EnhancedStockSelector
    )

    ML_AVAILABLE = True
    print("✓ 高级ML模块加载成功 (ml_factor_scoring_fixed)")
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
    print("    综合因子评分选股回测系统 v2.6 - 实盘精选版 (Top 5)")
    print("=" * 80)
    print("\n🎯 核心特性:")
    print("  ✅ 全历史窗口滚动训练 (Robust Walk-Forward)")
    print("  ✅ 实盘 Top 5 精选推荐")
    print("  ✅ 前视偏差严格防护")
    print()


def print_trading_plan(context, price_data, factor_data):
    """
    🖨️ 打印清晰的交易计划和持仓监控
    """
    if context is None:
        return

    print("\n" + "#" * 80)
    print("📋 步骤9: 交易指令与持仓监控 (回测模拟结果)")
    print("#" * 80 + "\n")

    df_trades = context.get('trade_records', pd.DataFrame())
    if df_trades.empty:
        print("⚠️ 全程无交易记录")
        return

    last_date = df_trades['date'].max()
    today_trades = df_trades[df_trades['date'] == last_date].copy()

    print(f"📅 回测最后信号日期: {last_date}")

    # 打印调仓指令
    print(f"\n📢 【模拟调仓指令】 共 {len(today_trades)} 笔")
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
        print("-" * 95)
        print(
            f"{'代码':<10} | {'持仓股数':<8} | {'成本价':<8} | {'现价':<8} | {'浮动盈亏':<10} | {'收益率':<8} | {'评分'}")
        print("-" * 95)

        total_mv = 0
        total_pnl = 0

        # 获取最后一天的数据用于展示
        try:
            # 兼容处理：检查评分列名是 'position' 还是 'ml_score'
            score_col = 'position' if 'position' in factor_data.columns else 'ml_score'

            # 确保日期格式一致
            last_date_str = str(last_date).split(' ')[0]
            if isinstance(factor_data['date'].iloc[0], str):
                mask_factor = factor_data['date'].str.startswith(last_date_str)
            else:
                mask_factor = factor_data['date'] == pd.Timestamp(last_date_str)

            last_scores = factor_data[mask_factor][['instrument', score_col]].set_index('instrument')[
                score_col].to_dict()

            if isinstance(price_data['date'].iloc[0], str):
                mask_price = price_data['date'].str.startswith(last_date_str)
            else:
                mask_price = price_data['date'] == pd.Timestamp(last_date_str)

            last_prices = price_data[mask_price][['instrument', 'close']].set_index('instrument')['close'].to_dict()
        except Exception as e:
            # print(f"DEBUG: 获取最后一日数据失败 {e}")
            last_scores = {}
            last_prices = {}

        for code, info in positions.items():
            shares = info['shares']
            cost = info['cost']
            current_price = last_prices.get(code, cost)  # 如果没有现价，暂用成本价代替
            score = last_scores.get(code, 0.0)

            mv = shares * current_price
            pnl = (current_price - cost) * shares
            pnl_rate = (current_price - cost) / cost if cost != 0 else 0

            total_mv += mv
            total_pnl += pnl

            pnl_str = f"¥{pnl:+,.0f}"
            rate_str = f"{pnl_rate:+.2%}"

            print(
                f"{code:<10} | {shares:<8.0f} | {cost:<8.2f} | {current_price:<8.2f} | {pnl_str:<10} | {rate_str:<8} | {score:.4f}")

        print("-" * 95)
        cash = final_value - total_mv
        print(f"💰 账户概览: 持仓市值 ¥{total_mv:,.0f} | 可用现金 ¥{cash:,.0f} | 总资产 ¥{final_value:,.0f}")
        print(f"📈 累计收益: {total_return:+.2%}")
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
    print(f"  - 效果: 剔除在 {START_DATE} 前 {MIN_DAYS_LISTED} 天内上市的次新股")

    # ============ 初始化 ============
    cache_manager = DataCache(cache_dir=DataConfig.CACHE_DIR)

    # 步骤0: 获取大盘指数
    benchmark_data = None
    try:
        print("\n" + "=" * 80)
        print("📈 步骤0: 获取大盘指数数据 (用于择时)")
        print("=" * 80)
        ds_temp = TushareDataSource(cache_manager=cache_manager, token=TUSHARE_TOKEN)
        benchmark_data = ds_temp.get_index_daily(ts_code='000001.SH', start_date=START_DATE, end_date=END_DATE)
        if benchmark_data is not None:
            print(f"  ✓ 获取上证指数数据: {len(benchmark_data)} 条")
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
            min_days_listed=MIN_DAYS_LISTED  # ✅ 关键参数：传递给数据加载器
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

    # ============ 步骤4: ML因子评分 (✅ 修复并适配新API) ============
    if MLConfig.USE_ADVANCED_ML and ML_AVAILABLE:
        try:
            print("\n" + "=" * 80)
            print("🚀 步骤4: 高级ML因子评分 (Walk-Forward 训练模式)")
            print("=" * 80)

            # 1. 初始化评分器
            ml_scorer = AdvancedMLScorer(
                model_type=MLConfig.ML_MODEL_TYPE,
                target_period=MLConfig.ML_TARGET_PERIOD,
                top_percentile=MLConfig.ML_TOP_PERCENTILE,
                use_classification=MLConfig.ML_USE_CLASSIFICATION,
                use_ic_features=MLConfig.ML_USE_IC_FEATURES,
                use_active_return=True,  # 开启超额收益目标
                train_months=MLConfig.ML_TRAIN_MONTHS
            )

            # 2. 准备训练数据 (计算IC特征、标注标签、处理缺失值)
            print("   [1/3] 准备训练数据...")
            X, y, merged_df = ml_scorer.prepare_training_data(
                factor_data,
                price_data,
                factor_columns
            )

            # 3. 执行 Walk-Forward 滚动训练
            print("   [2/3] 执行 Walk-Forward 滚动训练 (全历史窗口)...")
            # ✅ 修改：n_splits=None 表示训练所有可用的历史窗口，最稳健
            ml_scorer.train_walk_forward(X, y, merged_df, n_splits=None)

            # 4. 预测评分
            print("   [3/3] 全量数据预测评分...")
            # 覆盖原始 factor_data，因为 ml_scorer 返回的 dataframe 包含了 'position', 'ml_score' 等新列
            # 同时也包含了计算出来的 IC 特征
            factor_data = ml_scorer.predict_scores(merged_df)

            # 打印特征重要性
            importance = ml_scorer.get_feature_importance(top_n=10)
            if importance is not None:
                print("\n📊 TOP 10 关键因子:")
                for idx, row in importance.iterrows():
                    print(f"   {row['feature']:<25}: {row['importance']:.4f}")

        except Exception as e:
            print(f"⚠️  ML评分流程失败: {e}")
            traceback.print_exc()
            # 如果 ML 失败，factor_data 保持原样，后续流程可能会因为缺少 score 列而报错
            # 这里做一个简单的容错：如果缺少 position 列，用等权合成
            if 'position' not in factor_data.columns and len(factor_columns) > 0:
                print("   ⚠️ 启用备用评分方案：因子等权平均")
                factor_data['position'] = factor_data[factor_columns].mean(axis=1).rank(pct=True)

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

    except Exception as e:
        print(f"⚠️  报告生成警告: {e}")
        traceback.print_exc()

    # ============ 步骤9: 打印交易计划 (启用) ============
    print_trading_plan(context, price_data, factor_data)

    # ========== 【新增】实盘建仓专用清单 (Top 5) ==========
    print("\n" + "=" * 80)
    print("🚀 实盘建仓推荐清单 (最新日期 Top 5)")
    print("=" * 80)

    # 1. 获取最新一个交易日的数据
    latest_date = factor_data['date'].max()
    print(f"📅 数据截止日期: {latest_date}")

    # 2. 筛选当天的股票并按评分排序
    # 注意：确保这里使用的是经过 ML 预测后的 factor_data
    latest_stocks = factor_data[factor_data['date'] == latest_date].copy()

    # 兼容字段名
    score_col = 'position' if 'position' in latest_stocks.columns else 'ml_score'

    if score_col in latest_stocks.columns:
        # 过滤停牌或一字板（如果有价格数据辅助判断更好，这里主要按分数排）
        # ✅ 修改：这里改成了 Top 5
        target_stocks = latest_stocks.sort_values(by=score_col, ascending=False).head(5)

        print(f"{'排名':<6} | {'代码':<10} | {'行业':<10} | {'ML评分':<10}")
        print("-" * 50)

        for idx, (i, row) in enumerate(target_stocks.iterrows(), 1):
            stock = row['instrument']
            industry = row.get('industry', '未知')
            score = row[score_col]
            print(f"{idx:<6} | {stock:<10} | {industry:<10} | {score:.4f}")

        print("-" * 50)
        print("💡 实盘操作建议：")
        print("1. 此清单为全市场评分最高的 5 只股票。")
        print("2. 建议开盘后观察，若未停牌且未涨停，可直接买入。")
        print("3. 如遇不可买入情况，请顺延至第 6 名（需自行查看数据）。")
    else:
        print("❌ 无法生成推荐清单：未找到评分字段")

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