"""
main_enhanced.py - 整合完整收益指标的增强版主程序

新增功能:
✅ 完整的收益指标报告
✅ 胜率、盈亏比分析
✅ 详细交易明细
✅ 综合评级系统

版本: v3.1
"""

import warnings
warnings.filterwarnings('ignore')

import time
import os
import sys
import pandas as pd
import numpy as np
import tushare as ts

# ========== 导入配置 ==========
from config import (
    TUSHARE_TOKEN, StrategyConfig, BacktestConfig, DataConfig,
    FactorConfig, MLConfig, OutputConfig, RiskControlConfig,
    TradingCostConfig, get_strategy_params
)

ts.set_token(TUSHARE_TOKEN)

# ========== 导入核心模块 ==========
from data_module import DataCache, TushareDataSource
from data_module_incremental import load_data_with_incremental_update
from score_fusion_module import ScoreFusionEngine

# ✅ 新增：导入收益指标报告模块
from performance_metrics_report import generate_full_performance_report

# ========== ML模块 ==========
ML_AVAILABLE = False
if MLConfig.USE_ADVANCED_ML:
    try:
        import sklearn
        import xgboost
        from ml_factor_scoring_fixed import UltraMLScorer as AdvancedMLScorer
        ML_AVAILABLE = True
        print("✓ ML模块加载成功")
    except ImportError as e:
        print(f"⚠️  ML模块不可用: {e}")
        ML_AVAILABLE = False

# ========== 策略引擎 ==========
from factor_based_risk_control_optimized import run_factor_based_strategy_v2
STRATEGY_VERSION = "v3.1 (Enhanced with Performance Metrics)"

# ========== 报告模块 ==========
try:
    from show_today_holdings import show_today_holdings_dashboard
    from holdings_monitor import generate_daily_holdings_report
    from date_organized_reports import generate_date_organized_reports
    from visualization_module_patch import generate_performance_report
except ImportError:
    def show_today_holdings_dashboard(*args, **kwargs): pass
    def generate_daily_holdings_report(*args, **kwargs): pass
    def generate_date_organized_reports(*args, **kwargs):
        os.makedirs(OutputConfig.REPORTS_DIR, exist_ok=True)
        return OutputConfig.REPORTS_DIR
    def generate_performance_report(context, output_dir='./reports'):
        print("✓ 简化版报告")


# ==============================================================================
# 核心修复函数
# ==============================================================================

def fix_stockranker_scoring(factor_data):
    """修复StockRanker评分"""
    print("\n🔧 修复StockRanker评分流程...")

    if 'position' in factor_data.columns:
        factor_data['stockranker_score'] = factor_data['position']
        factor_data.drop(columns=['position'], inplace=True)

        print(f"  ✓ 已保存StockRanker评分为 stockranker_score")
        print(f"  ✓ 范围: [{factor_data['stockranker_score'].min():.4f}, "
              f"{factor_data['stockranker_score'].max():.4f}]")
    else:
        print("  ⚠️  未找到StockRanker的position列")

    return factor_data


def fix_ml_scoring(factor_data, price_data):
    """修复ML评分"""
    if not ML_AVAILABLE:
        print("\n⚠️  跳过ML评分 (模块不可用)")
        return factor_data

    print("\n🤖 运行ML评分 (修复版)...")

    try:
        ml_scorer = AdvancedMLScorer(
            target_period=MLConfig.ML_TARGET_PERIOD,
            top_percentile=MLConfig.ML_TOP_PERCENTILE,
            train_months=MLConfig.ML_TRAIN_MONTHS
        )

        temp_sr_score = None
        if 'stockranker_score' in factor_data.columns:
            temp_sr_score = factor_data['stockranker_score'].copy()

        factor_data = ml_scorer.predict(factor_data, price_data)

        if temp_sr_score is not None:
            factor_data['stockranker_score'] = temp_sr_score

        if 'position' in factor_data.columns and 'ml_score' not in factor_data.columns:
            factor_data['ml_score'] = factor_data['position']
            factor_data.drop(columns=['position'], inplace=True)

        print(f"  ✓ ML评分完成")
        if 'ml_score' in factor_data.columns:
            print(f"  ✓ ml_score范围: [{factor_data['ml_score'].min():.4f}, "
                  f"{factor_data['ml_score'].max():.4f}]")

    except Exception as e:
        print(f"  ❌ ML评分失败: {e}")
        import traceback
        traceback.print_exc()

    return factor_data


# ==============================================================================
# 主函数
# ==============================================================================

def main():
    print("\n" + "="*80)
    print("    多因子+ML选股回测系统 v3.1 (Enhanced)")
    print("="*80)
    print("\n🎯 新增功能:")
    print("  ✅ 完整收益指标报告")
    print("  ✅ 胜率、盈亏比分析")
    print("  ✅ 详细交易明细")
    print("  ✅ 综合策略评级")
    print()

    # 配置摘要
    print("【配置】")
    print(f"  回测区间: {BacktestConfig.START_DATE} ~ {BacktestConfig.END_DATE}")
    print(f"  初始资金: ¥{BacktestConfig.CAPITAL_BASE:,}")
    print(f"  持仓数量: {BacktestConfig.POSITION_SIZE} 只")
    print(f"  调仓周期: {BacktestConfig.REBALANCE_DAYS} 天")
    print()

    cache_manager = DataCache(cache_dir=DataConfig.CACHE_DIR)

    # ========== 步骤0: 大盘指数 ==========
    benchmark_data = None
    if StrategyConfig.ENABLE_MARKET_TIMING:
        try:
            print("📈 步骤0: 获取大盘指数")
            ds_temp = TushareDataSource(cache_manager=cache_manager, token=TUSHARE_TOKEN)
            benchmark_data = ds_temp.get_index_daily(
                ts_code='000001.SH',
                start_date=BacktestConfig.START_DATE,
                end_date=BacktestConfig.END_DATE
            )
            print(f"  ✓ 上证指数: {len(benchmark_data) if benchmark_data is not None else 0} 条")
        except Exception as e:
            print(f"  ⚠️  获取失败: {e}")

    # ========== 步骤1: 数据加载 ==========
    try:
        print("\n📦 步骤1: 加载数据")
        start_time = time.time()

        factor_data, price_data = load_data_with_incremental_update(
            BacktestConfig.START_DATE,
            BacktestConfig.END_DATE,
            max_stocks=DataConfig.MAX_STOCKS,
            cache_manager=cache_manager,
            use_stockranker=FactorConfig.USE_STOCKRANKER,
            custom_weights=FactorConfig.CUSTOM_WEIGHTS,
            tushare_token=TUSHARE_TOKEN,
            use_fundamental=FactorConfig.USE_FUNDAMENTAL,
            force_full_update=DataConfig.FORCE_FULL_UPDATE,
            use_sampling=DataConfig.USE_SAMPLING,
            sample_size=DataConfig.SAMPLE_SIZE,
            max_workers=DataConfig.MAX_WORKERS,
            min_days_listed=180,
            use_money_flow=FactorConfig.USE_MONEY_FLOW
        )

        if factor_data is None or factor_data.empty:
            print("\n❌ 数据加载失败")
            return

        print(f"  ✓ 耗时: {time.time() - start_time:.1f}秒")
        print(f"  ✓ 股票数: {factor_data['instrument'].nunique()}")

    except Exception as e:
        print(f"\n❌ 数据加载异常: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 步骤2: 数据质量优化 ==========
    try:
        print("\n🔍 步骤2: 数据质量优化")
        from data_quality_optimizer import optimize_data_quality
        price_data, factor_data = optimize_data_quality(
            price_data, factor_data, cache_manager=cache_manager
        )
    except Exception as e:
        print(f"  ⚠️  优化警告: {e}")

    # ========== 步骤3-5: 评分流程 ==========
    factor_data = fix_stockranker_scoring(factor_data)
    factor_data = fix_ml_scoring(factor_data, price_data)

    print("\n" + "="*80)
    print("🔗 步骤5: 评分融合")
    print("="*80)

    fusion_engine = ScoreFusionEngine(
        fusion_method='weighted',
        alpha=0.4,
        beta=0.6
    )

    factor_data = fusion_engine.fuse_scores(
        factor_data,
        has_ml=ML_AVAILABLE and 'ml_score' in factor_data.columns
    )

    print(f"\n✅ 评分融合完成!")

    # ========== 步骤6: 运行回测 ==========
    context = None
    try:
        print("\n" + "="*80)
        print(f"🚀 步骤6: 回测引擎 ({STRATEGY_VERSION})")
        print("="*80)

        strategy_params = get_strategy_params()

        context = run_factor_based_strategy_v2(
            factor_data=factor_data,
            price_data=price_data,
            benchmark_data=benchmark_data,
            **strategy_params
        )

    except Exception as e:
        print(f"\n❌ 回测异常: {e}")
        import traceback
        traceback.print_exc()
        return

    # ========== 步骤7: 生成报告目录 ==========
    try:
        print(f"\n{'='*80}")
        print("📊 步骤7: 生成报告")
        print(f"{'='*80}\n")

        date_folder = generate_date_organized_reports(
            context=context,
            factor_data=factor_data,
            price_data=price_data,
            base_dir=OutputConfig.REPORTS_DIR
        )

        print(f"📂 报告目录: {os.path.abspath(date_folder)}")

        # 保存评分对比
        if all(c in factor_data.columns for c in ['stockranker_score', 'ml_score', 'position']):
            score_comparison = factor_data[[
                'date', 'instrument',
                'stockranker_score', 'ml_score', 'position'
            ]].copy()

            comparison_path = os.path.join(date_folder, 'score_comparison.csv')
            score_comparison.to_csv(comparison_path, index=False, encoding='utf-8-sig')
            print(f"✓ 评分对比已保存: {comparison_path}")

        # 生成绩效报告
        generate_performance_report(context, output_dir=date_folder)

    except Exception as e:
        print(f"⚠️  报告生成出错: {e}")
        import traceback
        traceback.print_exc()

    # ========== ✅ 步骤8: 完整收益指标报告 ==========
    if context:
        try:
            print("\n" + "="*80)
            print("📈 步骤8: 生成完整收益指标报告")
            print("="*80)

            metrics = generate_full_performance_report(
                context=context,
                benchmark_data=benchmark_data,
                output_dir=date_folder
            )

        except Exception as e:
            print(f"⚠️  收益指标报告生成失败: {e}")
            import traceback
            traceback.print_exc()

    # ========== 步骤9: 交易计划 ==========
    if context:
        print("\n" + "#"*80)
        print("📋 步骤9: 今日持仓与交易指令")
        print("#"*80 + "\n")

        df_trades = context.get('trade_records', pd.DataFrame())
        if not df_trades.empty:
            last_date = df_trades['date'].max()
            today_trades = df_trades[df_trades['date'] == last_date]

            print(f"📅 信号日期: {last_date}")
            print(f"📢 今日操作: {len(today_trades)} 笔\n")

            if len(today_trades) > 0:
                print("-" * 75)
                print(f"{'方向':<6} | {'代码':<10} | {'价格':<8} | {'股数':<8} | {'原因'}")
                print("-" * 75)

                for _, row in today_trades.iterrows():
                    action = "🔵买入" if row['action'] == 'buy' else "🔴卖出"
                    print(f"{action:<6} | {row['stock']:<10} | "
                          f"{row['price']:<8.2f} | {row['shares']:<8.0f} | "
                          f"{row.get('reason', '')}")
                print("-" * 75)

    print("\n" + "="*80)
    print("✅ 全部完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()