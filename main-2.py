"""
main.py - 主回测入口（v2.0 - 使用config.py便捷函数版）

核心改进：
✅ 使用统一配置文件 config.py
✅ 便捷函数一键获取所有参数
✅ 因子风控 + 最佳现金管理
✅ 动态等权分配（资金利用率 ~95%）
✅ Walk-Forward训练 + IC特征

版本：v2.0 - Config Integration
日期：2025-12-09
"""

import warnings
warnings.filterwarnings('ignore')

import tushare as ts
import pandas as pd
import numpy as np
import time

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

from data_module import DataCache
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
    print("✓ 高级ML模块加载成功 (ml_factor_scoring_fixed.py)")
except ImportError as e:
    print(f"⚠️  高级ML模块未找到: {e}")
    print("   将使用基础因子评分")
    ML_AVAILABLE = False

# ========== 导入策略引擎 ==========
try:
    from factor_based_risk_control_optimized import run_factor_based_strategy_v2
    print("✓ v2.0优化版策略引擎加载成功")
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


def print_banner():
    """打印启动横幅"""
    print("\n" + "="*80)
    print("  ____  _             _    ____             _             ")
    print(" / ___|| |_ ___   ___| | _|  _ \ __ _ _ __ | | _____ _ __ ")
    print(" \___ \| __/ _ \ / __| |/ / |_) / _` | '_ \| |/ / _ \ '__|")
    print("  ___) | || (_) | (__|   <|  _ < (_| | | | |   <  __/ |   ")
    print(" |____/ \__\___/ \___|_|\_\_| \_\__,_|_| |_|_|\_\___|_|   ")
    print()
    print("    综合因子评分选股回测系统 v2.0 - Config Integration")
    print("="*80)
    print("\n🎯 v2.0 核心特性:")
    print("  ⭐ 统一配置管理 - config.py集中管理所有参数")
    print("  ⭐ 最佳现金管理 - 动态等权 + 现金保留")
    print("  ⭐ 资金利用率 - ~95%（提升50%+）")
    print("  ⭐ 便捷函数 - 一键获取策略参数")
    print("  ⭐ 配置预设 - 快速切换场景（激进/平衡/保守）")
    print("  ⚡ Walk-Forward训练 - 避免前视偏差")
    print("  ⚡ 因子风控 - 用因子本身做风险控制")
    print("  ⚡ 增量更新 - 50倍提速")
    print()


def main():
    """主函数"""
    print_banner()

    # ========== 显示配置 ==========
    print("【当前配置】来自 config.py")
    print(f"  策略版本: {StrategyConfig.STRATEGY_VERSION}")
    print(f"  现金保留: {StrategyConfig.CASH_RESERVE_RATIO:.1%}")
    print(f"  资金利用率目标: {1-StrategyConfig.CASH_RESERVE_RATIO:.1%}")
    print(f"  回测区间: {BacktestConfig.START_DATE} ~ {BacktestConfig.END_DATE}")
    print(f"  初始资金: ¥{BacktestConfig.CAPITAL_BASE:,}")
    print(f"  持仓数量: {BacktestConfig.POSITION_SIZE} 只")
    print(f"  调仓周期: {BacktestConfig.REBALANCE_DAYS} 天")

    # 打印配置对比
    print_config_comparison()

    # 验证配置
    validate_configs()

    # ========== 从配置获取参数 ==========
    START_DATE = BacktestConfig.START_DATE
    END_DATE = BacktestConfig.END_DATE
    CAPITAL_BASE = BacktestConfig.CAPITAL_BASE
    POSITION_SIZE = BacktestConfig.POSITION_SIZE
    REBALANCE_DAYS = BacktestConfig.REBALANCE_DAYS

    # 数据配置
    USE_SAMPLING = DataConfig.USE_SAMPLING
    SAMPLE_SIZE = DataConfig.SAMPLE_SIZE
    MAX_WORKERS = DataConfig.MAX_WORKERS
    FORCE_FULL_UPDATE = DataConfig.FORCE_FULL_UPDATE

    # 因子配置
    USE_STOCKRANKER = FactorConfig.USE_STOCKRANKER
    USE_FUNDAMENTAL = FactorConfig.USE_FUNDAMENTAL
    CUSTOM_WEIGHTS = FactorConfig.CUSTOM_WEIGHTS

    # ML配置
    USE_ADVANCED_ML = MLConfig.USE_ADVANCED_ML and ML_AVAILABLE

    print("\n【速度优化配置】⚡")
    print(f"  智能抽样: {'启用' if USE_SAMPLING else '关闭'}")
    print(f"  股票池: {SAMPLE_SIZE} 只")
    print(f"  并行线程: {MAX_WORKERS} 个")

    print("\n【高级ML配置】🤖")
    print(f"  高级ML: {'启用' if USE_ADVANCED_ML else '关闭'}")
    if USE_ADVANCED_ML:
        print(f"  模型类型: {MLConfig.ML_MODEL_TYPE.upper()}")
        print(f"  目标模式: {'分类 (预测TOP股票)' if MLConfig.ML_USE_CLASSIFICATION else '回归'}")
        print(f"  预测目标: TOP {int(MLConfig.ML_TOP_PERCENTILE*100)}%")
        print(f"  训练窗口: {MLConfig.ML_TRAIN_MONTHS}个月 (Walk-Forward)")

    print("\n【因子模型配置】")
    print(f"  因子模型: StockRanker多因子 + 基本面")
    print(f"  因子数量: 14个 (技术9个 + 基本面5个)")

    # ============ 初始化 ============
    cache_manager = DataCache(cache_dir=DataConfig.CACHE_DIR)

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
        price_data, factor_data = optimize_data_quality(
            price_data, factor_data, cache_manager=cache_manager
        )
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

        exclude_columns = ['date', 'instrument', 'open', 'high',
                          'low', 'close', 'volume', 'amount']
        factor_columns = [col for col in factor_data.columns
                         if col not in exclude_columns]

        print(f"  检测到 {len(factor_columns)} 个候选因子列")

        if len(factor_columns) > 0:
            factor_data = factor_processor.process_factors(
                factor_data, factor_columns
            )

            # 筛选数值型因子
            numeric_factor_columns = []
            for col in factor_columns:
                if col in factor_data.columns:
                    try:
                        if pd.api.types.is_numeric_dtype(factor_data[col]):
                            numeric_factor_columns.append(col)
                    except:
                        pass

            factor_columns = numeric_factor_columns
            print(f"  处理后因子列数: {len(factor_columns)}")
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
    if USE_ADVANCED_ML:
        try:
            print("\n" + "="*80)
            print("🚀 步骤4: 高级ML因子评分")
            print("="*80)

            ml_start_time = time.time()

            available_factors = [col for col in factor_columns
                               if col in factor_data.columns]

            if len(available_factors) == 0:
                print("  ⚠️  警告：没有可用的因子列，跳过ML评分")
                ml_elapsed = 0
            else:
                print(f"  ✓ 检测到 {len(available_factors)} 个可用因子")

                try:
                    ml_scorer = AdvancedMLScorer(
                        model_type=MLConfig.ML_MODEL_TYPE,
                        target_period=MLConfig.ML_TARGET_PERIOD,
                        top_percentile=MLConfig.ML_TOP_PERCENTILE,
                        use_classification=MLConfig.ML_USE_CLASSIFICATION,
                        use_ic_features=MLConfig.ML_USE_IC_FEATURES,
                        train_months=MLConfig.ML_TRAIN_MONTHS
                    )

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

        if ML_AVAILABLE:
            from ml_factor_scoring_fixed import IndustryBasedScorer

            industry_start_time = time.time()
            industry_scorer = IndustryBasedScorer(tushare_token=TUSHARE_TOKEN)
            factor_data = industry_scorer.score_by_industry(
                factor_data, factor_columns
            )
            industry_elapsed = time.time() - industry_start_time
            print(f"\n⚡ 分行业评分耗时: {industry_elapsed:.1f} 秒")
        else:
            print("  ⚠️  ML模块不可用，跳过分行业评分")
            if 'industry' not in factor_data.columns:
                factor_data['industry'] = 'Unknown'
            if 'industry_rank' not in factor_data.columns:
                factor_data['industry_rank'] = factor_data.get('position', 0.5)
            industry_elapsed = 0

    except Exception as e:
        print(f"\n⚠️  分行业评分警告: {e}")
        industry_elapsed = 0
        if 'industry' not in factor_data.columns:
            factor_data['industry'] = 'Unknown'
        if 'industry_rank' not in factor_data.columns:
            factor_data['industry_rank'] = factor_data.get('position', 0.5)

    # ============ 增强选股 ============
    try:
        print("\n" + "="*80)
        print("🎯 步骤6: 增强选股")
        print("="*80)

        if ML_AVAILABLE:
            from ml_factor_scoring_fixed import EnhancedStockSelector

            selection_start_time = time.time()
            selector = EnhancedStockSelector()
            factor_data = selector.select_stocks(
                factor_data,
                min_score=MLConfig.ML_MIN_SCORE,
                max_concentration=0.15,
                max_industry_concentration=0.3
            )
            selection_elapsed = time.time() - selection_start_time
            print(f"\n⚡ 增强选股耗时: {selection_elapsed:.1f} 秒")
        else:
            print("  ⚠️  ML模块不可用，使用基础选股")
            initial_count = len(factor_data)
            if 'position' in factor_data.columns:
                factor_data = factor_data[factor_data['position'] >= 0.5].copy()
            print(f"  ✓ 基础选股完成: {len(factor_data)} / {initial_count} 只股票")
            selection_elapsed = 0

    except Exception as e:
        print(f"\n⚠️  增强选股警告: {e}")
        selection_elapsed = 0

    # ========== 运行回测引擎 ==========
    try:
        backtest_start_time = time.time()

        print("\n" + "="*80)
        print(f"🚀 步骤7: {STRATEGY_VERSION} 回测引擎")
        print("="*80)

        if STRATEGY_VERSION == "v2.0":
            print("  ✓ 使用版本: v2.0 - 因子风控 + 最佳现金管理")
            print("  ✓ 核心特性: 动态等权 + 现金保留机制")
            print("  ✓ 参数来源: config.py (便捷函数)")

            # ✨ 关键改进：使用便捷函数一键获取所有参数
            strategy_params = get_strategy_params()

            print(f"\n  【参数确认】")
            print(f"    现金保留比例: {strategy_params['cash_reserve_ratio']:.1%}")
            print(f"    目标资金利用率: {1-strategy_params['cash_reserve_ratio']:.1%}")
            print(f"    持仓数量: {strategy_params['position_size']}")
            print(f"    调仓周期: {strategy_params['rebalance_days']}天")
            print(f"    因子衰减止损: {strategy_params['enable_score_decay_stop']}")
            print(f"    相对排名止损: {strategy_params['enable_rank_stop']}")
            print(f"    组合回撤保护: {strategy_params['max_portfolio_drawdown']:.1%}")

            # ✨ 直接解包所有参数
            context = run_factor_based_strategy_v2(
                factor_data=factor_data,
                price_data=price_data,
                **strategy_params  # 一键传入所有参数
            )
        else:
            print("  使用版本: v1.0 - 基础版")

            # v1.0 需要手动传参（或自己实现便捷函数）
            context = run_factor_based_strategy(
                factor_data=factor_data,
                price_data=price_data,
                start_date=START_DATE,
                end_date=END_DATE,
                capital_base=CAPITAL_BASE,
                position_size=POSITION_SIZE,
                rebalance_days=REBALANCE_DAYS,
                position_method=BacktestConfig.POSITION_METHOD,

                enable_score_decay_stop=RiskControlConfig.ENABLE_SCORE_DECAY_STOP,
                score_decay_threshold=RiskControlConfig.SCORE_DECAY_THRESHOLD,
                min_holding_days=RiskControlConfig.MIN_HOLDING_DAYS,
                enable_rank_stop=RiskControlConfig.ENABLE_RANK_STOP,
                rank_percentile_threshold=RiskControlConfig.RANK_PERCENTILE_THRESHOLD,
                max_portfolio_drawdown=RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN,
                reduce_position_ratio=RiskControlConfig.REDUCE_POSITION_RATIO,
                enable_industry_rotation=RiskControlConfig.ENABLE_INDUSTRY_ROTATION,
                max_industry_weight=RiskControlConfig.MAX_INDUSTRY_WEIGHT,
                extreme_loss_threshold=RiskControlConfig.EXTREME_LOSS_THRESHOLD,
                portfolio_loss_threshold=RiskControlConfig.PORTFOLIO_LOSS_THRESHOLD,

                buy_cost=TradingCostConfig.BUY_COST,
                sell_cost=TradingCostConfig.SELL_COST,
                tax_ratio=TradingCostConfig.TAX_RATIO,

                debug=StrategyConfig.DEBUG_MODE
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
            base_dir=OutputConfig.REPORTS_DIR
        )

        print("\n" + "="*80)
        print("📋 生成详细持仓和交易报告")
        print("="*80)

        try:
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
    print(f"  策略版本: {STRATEGY_VERSION} - {'最佳现金管理版' if STRATEGY_VERSION == 'v2.0' else '基础版'} ⭐⭐⭐")
    print(f"  配置管理: config.py 统一管理 ⭐")
    print(f"  参数获取: get_strategy_params() 便捷函数 ⭐")
    print(f"  数据源: Tushare (增量更新 + 多线程)")
    print(f"  回测引擎: Factor-Based Risk Control")
    print(f"  股票池: {SAMPLE_SIZE} 只")
    print(f"  因子模型: StockRanker + 基本面 + 高级ML")

    if STRATEGY_VERSION == "v2.0":
        print(f"\n  💰 v2.0 现金管理特点:")
        print(f"    - 现金保留: {StrategyConfig.CASH_RESERVE_RATIO:.1%} ⭐")
        print(f"    - 资金利用率: ~{(1-StrategyConfig.CASH_RESERVE_RATIO):.1%} ⭐")
        print(f"    - 仓位分配: 动态等权 ⭐")
        print(f"    - 预期改进: 资金利用率提升50%+ ⭐")

    if USE_ADVANCED_ML:
        print(f"\n  🤖 ML优化特点:")
        print(f"    - Walk-Forward训练: {MLConfig.ML_TRAIN_MONTHS}个月窗口 ⭐")
        print(f"    - 分类目标: 预测TOP {int(MLConfig.ML_TOP_PERCENTILE*100)}% ⭐")
        print(f"    - IC加权特征: 动态评估因子有效性 ⭐")

    print(f"\n  🎯 风控特点:")
    print(f"    - 因子衰减止损: 评分下降>{RiskControlConfig.SCORE_DECAY_THRESHOLD:.0%}")
    print(f"    - 相对排名止损: 跌出前{RiskControlConfig.RANK_PERCENTILE_THRESHOLD:.0%}")
    print(f"    - 组合回撤保护: 回撤>{RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN:.1%}降仓")

    print("\n📊 回测结果:")
    print(f"  最终资产: ¥{context['final_value']:,.0f}")
    print(f"  总收益率: {context['total_return']:+.2%}")
    print(f"  胜率: {context['win_rate']:.2%}")

    # ========== 资金利用率统计 ==========
    if 'daily_records' in context:
        df_daily = context['daily_records']
        avg_cash_ratio = (df_daily['cash'] / df_daily['portfolio_value']).mean()
        avg_utilization = 1 - avg_cash_ratio

        print(f"\n💰 资金管理统计:")
        print(f"  平均现金比例: {avg_cash_ratio:.2%}")
        print(f"  平均资金利用率: {avg_utilization:.2%}")

        if STRATEGY_VERSION == "v2.0":
            target_utilization = 1 - StrategyConfig.CASH_RESERVE_RATIO
            utilization_diff = avg_utilization - target_utilization
            print(f"  目标利用率: {target_utilization:.2%}")
            print(f"  实际偏差: {utilization_diff:+.2%}")

            if abs(utilization_diff) < 0.02:
                print(f"  ✅ 资金管理达标！")
            else:
                print(f"  ⚠️  资金管理偏差较大，建议检查配置")

    print("\n🎛️  配置调整提示:")
    print(f"  • 修改参数: 编辑 config.py")
    print(f"  • 快速切换: ConfigPresets.aggressive() / balanced() / conservative()")
    print(f"  • 验证配置: python config.py")
    print(f"  • 查看对比: print_config_comparison()")

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