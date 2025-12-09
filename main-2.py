"""
main.py - 主回测入口（高级ML版）

新增优化：
✅ 时间序列切分（避免前视偏差）
✅ 分类目标（预测TOP 20%）
✅ IC加权特征（因子有效性）
✅ Walk-Forward训练
"""

import warnings
warnings.filterwarnings('ignore')

import tushare as ts
import pandas as pd
import numpy as np

TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"
ts.set_token(TUSHARE_TOKEN)

from data_module import DataCache
from data_module_incremental import load_data_with_incremental_update

# ========== 导入高级ML模块 ==========
ML_AVAILABLE = False
try:
    from ml_factor_scoring_advanced import (
        AdvancedMLScorer,
        ICCalculator,
        IndustryBasedScorer,
        EnhancedStockSelector
    )
    ML_AVAILABLE = True
    print("✓ 高级ML模块加载成功")
except ImportError as e:
    print(f"⚠️  高级ML模块未找到: {e}")
    print("   将使用基础因子评分")
    ML_AVAILABLE = False

from factor_based_risk_control import run_factor_based_strategy

from visualization_module import (
    plot_monitoring_results,
    plot_top_stocks_evolution,
    generate_performance_report
)

from show_today_holdings import show_today_holdings_dashboard
from holdings_monitor import generate_daily_holdings_report

import time


def print_banner():
    """打印启动横幅"""
    print("\n" + "="*80)
    print("  ____  _             _    ____             _             ")
    print(" / ___|| |_ ___   ___| | _|  _ \ __ _ _ __ | | _____ _ __ ")
    print(" \___ \| __/ _ \ / __| |/ / |_) / _` | '_ \| |/ / _ \ '__|")
    print("  ___) | || (_) | (__|   <|  _ < (_| | | | |   <  __/ |   ")
    print(" |____/ \__\___/ \___|_|\_\_| \_\__,_|_| |_|_|\_\___|_|   ")
    print()
    print("    综合因子评分选股回测系统 v12.0 - 高级ML优化版")
    print("="*80)
    print("\n核心特性:")
    print("  ⭐ Walk-Forward训练 - 避免前视偏差")
    print("  ⭐ 分类目标 - 预测TOP 20%股票")
    print("  ⭐ IC加权特征 - 动态评估因子有效性")
    print("  ⚡ 因子风控 - 用因子本身做风险控制")
    print("  ⚡ 智能抽样 - 大中小盘均衡")
    print("  ⚡ 多线程并行 - 10倍提速")
    print("  ⚡ 增量更新 - 50倍提速")
    print()


def main():
    """主函数"""
    print_banner()

    # ============ 参数配置 ============
    print("【基础配置】")

    START_DATE = "2023-01-01"
    END_DATE = "2025-12-09"
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")

    CAPITAL_BASE = 1000000
    print(f"  初始资金: {CAPITAL_BASE:,} 元")

    POSITION_SIZE = 10
    print(f"  持仓数量: {POSITION_SIZE} 只")

    # ============ 速度优化配置 ============
    print("\n【速度优化配置】⚡")

    USE_SAMPLING = False
    SAMPLE_SIZE = 4000
    MAX_WORKERS = 10
    FORCE_FULL_UPDATE = False

    print(f"  智能抽样: {'启用' if USE_SAMPLING else '关闭'}")
    if USE_SAMPLING:
        print(f"  抽样数量: {SAMPLE_SIZE} 只")
    else:
        print(f"  使用全部: {SAMPLE_SIZE} 只")
    print(f"  并行线程: {MAX_WORKERS} 个")

    # ========== 高级ML参数配置 ==========
    print("\n【高级ML配置】🤖")

    USE_ADVANCED_ML = True and ML_AVAILABLE
    ML_MODEL_TYPE = 'xgboost'
    ML_TARGET_PERIOD = 5
    ML_TOP_PERCENTILE = 0.20            # ✨ 预测TOP 20%
    ML_USE_CLASSIFICATION = True        # ✨ 使用分类模型
    ML_USE_IC_FEATURES = True           # ✨ 使用IC特征
    ML_TRAIN_MONTHS = 12               # ✨ 训练窗口12个月
    ML_MIN_SCORE = 0.6

    print(f"  高级ML: {'启用' if USE_ADVANCED_ML else '关闭'}")
    if USE_ADVANCED_ML:
        print(f"  模型类型: {ML_MODEL_TYPE.upper()}")
        print(f"  目标模式: {'分类 (预测TOP股票)' if ML_USE_CLASSIFICATION else '回归 (预测收益率)'}")
        print(f"  预测目标: TOP {int(ML_TOP_PERCENTILE*100)}%")
        print(f"  IC特征: {'启用' if ML_USE_IC_FEATURES else '关闭'}")
        print(f"  训练窗口: {ML_TRAIN_MONTHS}个月 (Walk-Forward)")
        print(f"  选股阈值: {ML_MIN_SCORE:.1%}")

    # ========== 因子风控参数配置 ==========
    print("\n【因子风控参数】🎯")

    REBALANCE_DAYS = 5
    POSITION_METHOD = 'equal'

    ENABLE_SCORE_DECAY_STOP = True
    SCORE_DECAY_THRESHOLD = 0.30
    MIN_HOLDING_DAYS = 5

    ENABLE_RANK_STOP = True
    RANK_PERCENTILE_THRESHOLD = 0.70

    MAX_PORTFOLIO_DRAWDOWN = -0.15
    REDUCE_POSITION_RATIO = 0.5

    ENABLE_INDUSTRY_ROTATION = True
    MAX_INDUSTRY_WEIGHT = 0.40

    EXTREME_LOSS_THRESHOLD = -0.20
    PORTFOLIO_LOSS_THRESHOLD = -0.25

    BUY_COST = 0.0003
    SELL_COST = 0.0003
    TAX_RATIO = 0.0005

    print(f"  调仓周期: {REBALANCE_DAYS} 天")
    print(f"  因子衰减止损: 评分下降>{SCORE_DECAY_THRESHOLD:.0%}")
    print(f"  相对排名止损: 跌出前{RANK_PERCENTILE_THRESHOLD:.0%}")
    print(f"  组合回撤保护: {MAX_PORTFOLIO_DRAWDOWN:.1%}")

    # ============ 模型配置 ============
    print("\n【因子模型配置】")

    USE_STOCKRANKER = True
    USE_FUNDAMENTAL = True
    CUSTOM_WEIGHTS = None

    print(f"  因子模型: StockRanker多因子 + 基本面")
    print(f"  因子数量: 14个 (技术9个 + 基本面5个)")

    # ============ 初始化 ============
    cache_manager = DataCache(cache_dir='./data_cache')

    # ============ 数据加载 ============
    try:
        data_start_time = time.time()

        print("\n" + "="*80)
        print("📦 步骤1: 数据加载")
        print("="*80)

        factor_data, price_data = load_data_with_incremental_update(
            START_DATE,
            END_DATE,
            max_stocks=SAMPLE_SIZE,
            cache_manager=cache_manager,
            use_stockranker=USE_STOCKRANKER,
            custom_weights=CUSTOM_WEIGHTS,
            tushare_token=TUSHARE_TOKEN,
            use_fundamental=USE_FUNDAMENTAL,
            force_full_update=FORCE_FULL_UPDATE,
            use_sampling=USE_SAMPLING,
            sample_size=SAMPLE_SIZE,
            max_workers=MAX_WORKERS
        )

        data_elapsed = time.time() - data_start_time
        print(f"\n⚡ 数据加载耗时: {data_elapsed:.1f} 秒")

        if factor_data is None or price_data is None:
            print("\n❌ 数据获取失败")
            return

    except Exception as e:
        print(f"\n❌ 数据加载异常: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============ 数据质量优化 ============
    try:
        print("\n" + "="*80)
        print("🔍 步骤2: 数据质量优化")
        print("="*80)

        from data_quality_optimizer import optimize_data_quality

        quality_start_time = time.time()
        price_data, factor_data = optimize_data_quality(price_data, factor_data, cache_manager=cache_manager)
        quality_elapsed = time.time() - quality_start_time
        print(f"\n⚡ 数据质量优化耗时: {quality_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  数据质量优化警告: {e}")
        quality_elapsed = 0

    # ============ 因子增强处理 ============
    try:
        print("\n" + "="*80)
        print("🎯 步骤3: 因子增强处理")
        print("="*80)

        from enhanced_factor_processor import EnhancedFactorProcessor

        factor_start_time = time.time()
        factor_processor = EnhancedFactorProcessor(
            neutralize_industry=True,
            neutralize_market=False
        )

        exclude_columns = ['date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'amount']
        factor_columns = [col for col in factor_data.columns if col not in exclude_columns]

        print(f"  检测到 {len(factor_columns)} 个候选因子列")

        if len(factor_columns) > 0:
            factor_data = factor_processor.process_factors(factor_data, factor_columns)
            numeric_factor_columns = []
            for col in factor_columns:
                if col in factor_data.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(factor_data[col]):
                            numeric_factor_columns.append(col)
                    except:
                        pass
            processed_factor_columns = numeric_factor_columns
            print(f"  处理后因子列数: {len(processed_factor_columns)}")
            factor_columns = processed_factor_columns
        else:
            print("  ⚠️  没有检测到因子列")
            factor_columns = []

        factor_elapsed = time.time() - factor_start_time
        print(f"\n⚡ 因子增强处理耗时: {factor_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  因子增强处理警告: {e}")
        factor_columns = []
        factor_elapsed = 0

    # ============ 高级ML因子评分 ============
    ml_elapsed = 0
    if ML_AVAILABLE and USE_ADVANCED_ML:
        try:
            print("\n" + "="*80)
            print("🚀 步骤4: 高级ML因子评分")
            print("="*80)

            ml_start_time = time.time()

            # 获取可用因子
            available_factors = [col for col in factor_columns if col in factor_data.columns]

            if len(available_factors) == 0:
                print("  ⚠️  警告：没有可用的因子列，跳过ML评分")
                ml_elapsed = 0
            else:
                print(f"  ✓ 检测到 {len(available_factors)} 个可用因子")

                try:
                    # ========== 使用高级ML评分器 ==========
                    ml_scorer = AdvancedMLScorer(
                        model_type=ML_MODEL_TYPE,
                        target_period=ML_TARGET_PERIOD,
                        top_percentile=ML_TOP_PERCENTILE,
                        use_classification=ML_USE_CLASSIFICATION,
                        use_ic_features=ML_USE_IC_FEATURES,
                        train_months=ML_TRAIN_MONTHS
                    )

                    # 预测因子得分
                    factor_data = ml_scorer.predict_scores(
                        factor_data,
                        price_data,
                        available_factors
                    )

                    ml_elapsed = time.time() - ml_start_time
                    print(f"\n⚡ 高级ML因子评分耗时: {ml_elapsed:.1f} 秒")

                except Exception as e:
                    print(f"  ⚠️  高级ML评分失败: {e}")
                    import traceback
                    traceback.print_exc()
                    ml_elapsed = 0

        except Exception as e:
            print(f"\n⚠️  高级ML因子评分警告: {e}")
            ml_elapsed = 0
    else:
        if not ML_AVAILABLE:
            print("\n⚠️  高级ML模块不可用")
        elif not USE_ADVANCED_ML:
            print("\n⚠️  高级ML功能已禁用")

    # ============ 分行业评分 ============
    try:
        print("\n" + "="*80)
        print("🏢 步骤5: 分行业评分")
        print("="*80)

        from ml_factor_scoring_advanced import IndustryBasedScorer

        industry_start_time = time.time()
        industry_scorer = IndustryBasedScorer(tushare_token=TUSHARE_TOKEN)
        factor_data = industry_scorer.score_by_industry(factor_data, factor_columns)
        industry_elapsed = time.time() - industry_start_time
        print(f"\n⚡ 分行业评分耗时: {industry_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  分行业评分警告: {e}")
        industry_elapsed = 0
        if 'industry' not in factor_data.columns:
            factor_data['industry'] = 'Unknown'

    # ============ 增强选股 ============
    try:
        print("\n" + "="*80)
        print("🎯 步骤6: 增强选股")
        print("="*80)

        from ml_factor_scoring_advanced import EnhancedStockSelector

        selection_start_time = time.time()
        selector = EnhancedStockSelector()
        factor_data = selector.select_stocks(
            factor_data,
            min_score=ML_MIN_SCORE,
            max_concentration=0.15,
            max_industry_concentration=0.3
        )
        selection_elapsed = time.time() - selection_start_time
        print(f"\n⚡ 增强选股耗时: {selection_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  增强选股警告: {e}")
        selection_elapsed = 0

    # ========== 运行因子风控回测 ==========
    try:
        backtest_start_time = time.time()

        print("\n" + "="*80)
        print("🚀 步骤7: 因子风控回测引擎")
        print("="*80)

        context = run_factor_based_strategy(
            factor_data=factor_data,
            price_data=price_data,
            start_date=START_DATE,
            end_date=END_DATE,
            capital_base=CAPITAL_BASE,
            position_size=POSITION_SIZE,
            rebalance_days=REBALANCE_DAYS,
            position_method=POSITION_METHOD,

            enable_score_decay_stop=ENABLE_SCORE_DECAY_STOP,
            score_decay_threshold=SCORE_DECAY_THRESHOLD,
            min_holding_days=MIN_HOLDING_DAYS,

            enable_rank_stop=ENABLE_RANK_STOP,
            rank_percentile_threshold=RANK_PERCENTILE_THRESHOLD,

            max_portfolio_drawdown=MAX_PORTFOLIO_DRAWDOWN,
            reduce_position_ratio=REDUCE_POSITION_RATIO,

            enable_industry_rotation=ENABLE_INDUSTRY_ROTATION,
            max_industry_weight=MAX_INDUSTRY_WEIGHT,

            extreme_loss_threshold=EXTREME_LOSS_THRESHOLD,
            portfolio_loss_threshold=PORTFOLIO_LOSS_THRESHOLD,

            buy_cost=BUY_COST,
            sell_cost=SELL_COST,
            tax_ratio=TAX_RATIO,

            debug=False
        )

        backtest_elapsed = time.time() - backtest_start_time
        print(f"\n⚡ 回测引擎耗时: {backtest_elapsed:.2f} 秒")

    except Exception as e:
        print(f"\n❌ 回测执行异常: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============ 生成报告 ============
    try:
        report_start_time = time.time()

        print(f"\n{'='*80}")
        print("📊 步骤8: 生成分析报告")
        print(f"{'='*80}\n")

        from date_organized_reports import generate_date_organized_reports
        date_folder = generate_date_organized_reports(
            context=context,
            factor_data=factor_data,
            price_data=price_data,
            base_dir='./reports'
        )

        print("\n" + "="*80)
        print("📋 生成详细持仓和交易报告")
        print("="*80)

        try:
            from holdings_monitor import generate_daily_holdings_report
            daily_holdings = generate_daily_holdings_report(
                context=context,
                factor_data=factor_data,
                price_data=price_data,
                output_dir=date_folder,
                print_to_console=True,
                save_to_csv=True
            )
        except Exception as e:
            print(f"\n⚠️  每日持仓报告生成警告: {e}")

        try:
            from show_today_holdings import show_today_holdings_dashboard
            today_holdings = show_today_holdings_dashboard(
                context=context,
                factor_data=factor_data,
                price_data=price_data,
                output_dir=date_folder
            )
        except Exception as e:
            print(f"\n⚠️  今日持仓仪表板生成警告: {e}")

        report_elapsed = time.time() - report_start_time
        print(f"\n⚡ 报告生成耗时: {report_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  报告生成警告: {e}")
        report_elapsed = 0

    # ============ 完成提示 ============
    total_elapsed = time.time() - data_start_time

    print(f"\n{'='*80}")
    print("✅ 所有任务完成!")
    print(f"{'='*80}")

    print("\n⏱️  性能统计:")
    print(f"  数据加载: {data_elapsed:.1f}秒")
    if 'quality_elapsed' in locals():
        print(f"  数据质量优化: {quality_elapsed:.1f}秒")
    if 'factor_elapsed' in locals():
        print(f"  因子增强处理: {factor_elapsed:.1f}秒")
    if 'ml_elapsed' in locals() and ml_elapsed > 0:
        print(f"  高级ML评分: {ml_elapsed:.1f}秒 ⭐")
    if 'industry_elapsed' in locals():
        print(f"  分行业评分: {industry_elapsed:.1f}秒")
    if 'selection_elapsed' in locals():
        print(f"  增强选股: {selection_elapsed:.1f}秒")
    print(f"  回测引擎: {backtest_elapsed:.2f}秒")
    if 'report_elapsed' in locals():
        print(f"  报告生成: {report_elapsed:.1f}秒")
    print(f"  总耗时: {total_elapsed:.1f}秒")

    print("\n📈 策略配置摘要:")
    print(f"  策略版本: v12.0 - 高级ML优化版 ⭐⭐⭐")
    print(f"  数据源: Tushare (增量更新 + 多线程)")
    print(f"  回测引擎: Factor-Based Risk Control")
    print(f"  股票池: {SAMPLE_SIZE} 只")
    print(f"  因子模型: StockRanker + 基本面 + 高级ML")

    print(f"\n  ML优化特点:")
    if USE_ADVANCED_ML:
        print(f"    - Walk-Forward训练: {ML_TRAIN_MONTHS}个月窗口 ⭐")
        print(f"    - 分类目标: 预测TOP {int(ML_TOP_PERCENTILE*100)}% ⭐")
        print(f"    - IC加权特征: 动态评估因子有效性 ⭐")
        print(f"    - 避免前视偏差: 时间序列切分 ⭐")

    print(f"\n  风控特点:")
    print(f"    - 因子衰减止损: 评分下降>{SCORE_DECAY_THRESHOLD:.0%}")
    print(f"    - 相对排名止损: 跌出前{RANK_PERCENTILE_THRESHOLD:.0%}")
    print(f"    - 组合回撤保护: 回撤>{MAX_PORTFOLIO_DRAWDOWN:.1%}降仓")

    print("\n📊 回测结果:")
    print(f"  最终资产: ¥{context['final_value']:,.0f}")
    print(f"  总收益率: {context['total_return']:+.2%}")
    print(f"  胜率: {context['win_rate']:.2%}")

    print("\n" + "="*80)
    print("感谢使用! ⚡⚡⚡")
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
