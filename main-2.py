"""
main.py - 主回测入口 (完整修复版 v3.0.1)

核心修复：
✅ Issue A: 过滤未来上市的新股
✅ Issue B: 防止使用上市前的历史数据
✅ Issue C: 消除StockRanker与ML的评分重复问题
✅ 依赖修复: 自动处理 sklearn/plotly 缺失情况
✅ 新增: 交易成本和持仓天数正确显示
✅ 新增: 修复效果验证功能
✅ 修复: RiskControlConfig 导入错误

版本：v3.0.1
日期：2025-12-29
"""

import warnings
warnings.filterwarnings('ignore')

import time
import os
import sys
import traceback
import pandas as pd
import numpy as np
import tushare as ts

# ========== 1. 导入配置 ==========
try:
    from config import (
        TUSHARE_TOKEN, StrategyConfig, BacktestConfig, DataConfig,
        FactorConfig, MLConfig, OutputConfig, RiskControlConfig,
        TradingCostConfig, get_strategy_params
    )
except ImportError as e:
    print(f"❌ 错误: 缺少配置类: {e}")
    print("请确保 config.py 包含所有必需的配置类。")
    sys.exit(1)

# 设置 Token
if TUSHARE_TOKEN:
    ts.set_token(TUSHARE_TOKEN)
else:
    print("⚠️  警告: Config 中未设置 Tushare Token")

# ========== 2. 导入核心模块 ==========
try:
    from data_module import DataCache, TushareDataSource
    from data_module_incremental import load_data_with_incremental_update
    from money_flow_factors import MoneyFlowFactorCalculator
except ImportError as e:
    print(f"❌ 关键模块缺失: {e}")
    print("请确保 data_module.py, data_module_incremental.py, money_flow_factors.py 都在当前目录。")
    sys.exit(1)

# ========== 3. ML 模块检测与导入 ==========
ML_AVAILABLE = False
if MLConfig.USE_ADVANCED_ML:
    try:
        import sklearn
        import xgboost
        from ml_factor_scoring_fixed import UltraMLScorer as AdvancedMLScorer
        ML_AVAILABLE = True
        print("✓ 高级ML模块 (Scikit-learn/XGBoost) 加载成功")
    except ImportError as e:
        print(f"⚠️  ML模块不可用: {e}")
        print("   -> 将降级使用 StockRanker 基础评分")
        ML_AVAILABLE = False

# ========== 4. 策略引擎导入 ==========
try:
    from factor_based_risk_control_optimized import run_factor_based_strategy_v2
    STRATEGY_VERSION = "v3.0 (Optimized + Fixed)"
    print("✓ 策略引擎 (Optimized v3) 加载成功")
except ImportError:
    print("⚠️  无法加载优化版策略，尝试使用基础版...")
    try:
        from factor_based_risk_control import run_factor_based_strategy as run_factor_based_strategy_v2
        STRATEGY_VERSION = "v1.0 (Basic)"
    except ImportError:
        print("❌ 错误: 无法加载任何策略引擎。")
        sys.exit(1)

# ========== 5. 可视化模块导入 ==========
try:
    import plotly
    import video_visualization
    VISUALIZATION_AVAILABLE = True
    print("✓ 可视化模块 (Plotly) 加载成功")
except ImportError:
    print("⚠️  可视化模块不可用 (缺少 plotly 库)")
    VISUALIZATION_AVAILABLE = False
    # 创建哑对象，防止后续代码报错
    class DummyVis:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    video_visualization = DummyVis()

# ========== 6. 验证模块导入 ==========
VALIDATE_AVAILABLE = False
try:
    from validate_fix import quick_validate, FixValidator
    VALIDATE_AVAILABLE = True
    print("✓ 验证模块加载成功")
except ImportError:
    print("⚠️  验证模块不可用 (validate_fix.py 不存在)")
    VALIDATE_AVAILABLE = False

# 导入报告生成工具
try:
    from show_today_holdings import show_today_holdings_dashboard
    from holdings_monitor import generate_daily_holdings_report
    from date_organized_reports import generate_date_organized_reports
except ImportError:
    # 定义空函数以防报错
    def show_today_holdings_dashboard(*args, **kwargs): pass
    def generate_daily_holdings_report(*args, **kwargs): pass
    def generate_date_organized_reports(*args, **kwargs):
        if not os.path.exists(OutputConfig.REPORTS_DIR):
            os.makedirs(OutputConfig.REPORTS_DIR)
        return OutputConfig.REPORTS_DIR

# 尝试导入修复后的报告模块
try:
    from visualization_module_patch import generate_performance_report
    print("✓ 使用修复版报告模块")
except ImportError:
    try:
        from visualization_module import generate_performance_report
        print("✓ 使用原版报告模块")
    except ImportError:
        # 如果都没有，使用内置简化版
        def generate_performance_report(context, output_dir='./reports'):
            """简化版报告生成器"""
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            report_path = os.path.join(output_dir, 'performance_report.txt')

            total_return = context.get('total_return', 0)
            win_rate = context.get('win_rate', 0)
            final_value = context.get('final_value', 0)
            total_cost = context.get('total_cost', 0)
            avg_holding_days = context.get('avg_holding_days', 0)

            report = f"""
================================================================================
📊 策略绩效报告 (简化版)
================================================================================

【收益指标】
  总收益率: {total_return:.2%}
  最终市值: ¥{final_value:,.2f}
  交易费用: ¥{total_cost:,.2f}

【交易指标】
  胜率: {win_rate:.2%}
  平均持仓天数: {avg_holding_days:.1f} 天

================================================================================
"""
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(report)
            print(f"✓ 绩效报告已保存: {report_path}")

# ==============================================================================
# 辅助函数
# ==============================================================================

def print_banner():
    """打印启动横幅"""
    print("\n" + "="*80)
    print("    综合因子评分选股回测系统 v3.0 - 完整修复版")
    print("="*80)
    print("\n🎯 核心修复:")
    print("  ✅ 依赖自动检测 (sklearn, plotly)")
    print("  ✅ 完整的增量数据更新逻辑")
    print("  ✅ 评分流程优化: 原始因子 -> ML -> 最终 Position")
    print("  ✅ 交易成本正确统计和显示")
    print("  ✅ 平均持仓天数正确计算")
    print("  ✅ 调仓频率优化 (5天→10天)")
    print("  ✅ 风控参数优化")
    print()

def print_config_summary():
    """打印关键配置摘要"""
    print("【当前配置摘要】")
    print(f"  策略版本: {StrategyConfig.STRATEGY_VERSION}")
    print(f"  回测区间: {BacktestConfig.START_DATE} ~ {BacktestConfig.END_DATE}")
    print(f"  初始资金: ¥{BacktestConfig.CAPITAL_BASE:,}")
    print(f"  持仓数量: {BacktestConfig.POSITION_SIZE} 只")
    print(f"  调仓周期: {BacktestConfig.REBALANCE_DAYS} 天  {'✅' if BacktestConfig.REBALANCE_DAYS >= 10 else '⚠️  建议≥10天'}")
    print(f"  最小持仓: {RiskControlConfig.MIN_HOLDING_DAYS} 天")
    print(f"  极端止损: {RiskControlConfig.EXTREME_LOSS_THRESHOLD:.1%}")
    print(f"  评分衰减阈值: {RiskControlConfig.SCORE_DECAY_THRESHOLD:.1%}")
    print(f"  交易成本: 买入{TradingCostConfig.BUY_COST:.2%} + 卖出{TradingCostConfig.SELL_COST:.2%} + 印花税{TradingCostConfig.TAX_RATIO:.2%}")
    print()

def print_trading_plan(context, price_data, factor_data):
    """✅ 增强版交易计划和持仓监控"""
    if context is None:
        return

    print("\n" + "#"*80)
    print("📋 步骤9: 交易指令与持仓监控 (最终报告)")
    print("#"*80 + "\n")

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

            print(f"{action:<6} | {row['stock']:<10} | {price_val:<8.2f} | {shares_val:<8.0f} | ¥{amount_val:<9.0f} | {row.get('reason', '')}")
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
        print(f"{'代码':<10} | {'买入日期':<12} | {'持仓股数':<8} | {'持仓占比':<8} | {'成本价':<8} | {'现价':<8} | {'浮动盈亏':<10} | {'收益率':<8} | {'评分'}")
        print("-" * 125)

        total_mv = 0

        try:
            last_scores = factor_data[factor_data['date'] == str(last_date)][['instrument', 'position']].set_index('instrument')['position'].to_dict()
            last_prices = price_data[price_data['date'] == str(last_date)][['instrument', 'close']].set_index('instrument')['close'].to_dict()
        except Exception:
            last_scores = {}
            last_prices = {}

        for code, info in positions.items():
            shares = info['shares']
            cost = info['cost']
            entry_date = info['entry_date']
            current_price = last_prices.get(code, cost)
            score = last_scores.get(code, 0.0)

            mv = shares * current_price
            pnl = (current_price - cost) * shares
            pnl_rate = (current_price - cost) / cost if cost != 0 else 0
            position_ratio = mv / final_value if final_value > 0 else 0

            total_mv += mv
            pnl_str = f"¥{pnl:+,.0f}"
            rate_str = f"{pnl_rate:+.2%}"
            ratio_str = f"{position_ratio:.2%}"

            print(f"{code:<10} | {entry_date:<12} | {shares:<8.0f} | {ratio_str:<8} | {cost:<8.2f} | {current_price:<8.2f} | {pnl_str:<10} | {rate_str:<8} | {score:.4f}")

        print("-" * 125)
        cash = final_value - total_mv
        print(f"💰 账户概览: 持仓市值 ¥{total_mv:,.0f} | 可用现金 ¥{cash:,.0f} | 总资产 ¥{final_value:,.0f}")
        print(f"📈 累计收益: {total_return:+.2%}")
        print("\n")

# ==============================================================================
# 主函数
# ==============================================================================

def main():
    print_banner()

    # ============ 显示配置 ============
    print_config_summary()

    cache_manager = DataCache(cache_dir=DataConfig.CACHE_DIR)

    # 步骤0: 获取大盘指数
    benchmark_data = None
    if StrategyConfig.ENABLE_MARKET_TIMING:
        try:
            print("\n📈 步骤0: 获取大盘指数数据 (用于择时)")
            ds_temp = TushareDataSource(cache_manager=cache_manager, token=TUSHARE_TOKEN)
            benchmark_data = ds_temp.get_index_daily(ts_code='000001.SH', start_date=BacktestConfig.START_DATE, end_date=BacktestConfig.END_DATE)
            print(f"  ✓ 获取上证指数数据: {len(benchmark_data) if benchmark_data is not None else 0} 条")
        except Exception as e:
            print(f"  ⚠️  获取指数失败: {e}")

    # ============ 步骤1: 数据加载 ============
    try:
        data_start_time = time.time()
        print("\n📦 步骤1: 数据加载 (增量模式)")

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

        if factor_data is None or price_data is None or factor_data.empty:
            print("\n❌ 数据获取失败或为空。请检查网络或Token。")
            return

        print(f"  ✓ 数据加载耗时: {time.time() - data_start_time:.1f} 秒")
        print(f"  ✓ 股票池大小: {factor_data['instrument'].nunique()} 只")

    except Exception as e:
        print(f"\n❌ 数据加载异常: {e}")
        traceback.print_exc()
        return

    # ============ 步骤2: 数据质量优化 ============
    try:
        print("\n🔍 步骤2: 数据质量优化")
        from data_quality_optimizer import optimize_data_quality
        price_data, factor_data = optimize_data_quality(price_data, factor_data, cache_manager=cache_manager)
    except Exception as e:
        print(f"\n⚠️  数据质量优化警告: {e}")

    # ============ 步骤4: ML因子评分 ============
    if MLConfig.USE_ADVANCED_ML and ML_AVAILABLE:
        try:
            print("\n🚀 步骤4: ML因子评分 (使用原始因子训练)")

            ml_scorer = AdvancedMLScorer(
                target_period=MLConfig.ML_TARGET_PERIOD,
                top_percentile=MLConfig.ML_TOP_PERCENTILE,
                train_months=MLConfig.ML_TRAIN_MONTHS
            )

            # ML模型自动识别原始因子并生成 'ml_score'
            factor_data = ml_scorer.predict(factor_data, price_data)

            print(f"\n✅ ML评分完成:")
            if 'ml_score' in factor_data.columns:
                print(f"  - ml_score 范围: [{factor_data['ml_score'].min():.4f}, {factor_data['ml_score'].max():.4f}]")

        except Exception as e:
            print(f"⚠️  ML评分失败: {e}")
            print(f"  ℹ️  回退使用 StockRanker 基础评分")

    # ========== 步骤7: 运行回测引擎 ==========
    context = None
    try:
        print("\n" + "="*80)
        print(f"🚀 步骤7: {STRATEGY_VERSION} 回测引擎")
        print("="*80)

        strategy_params = get_strategy_params()

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
        print(f"\n{'='*80}")
        print("📊 步骤8: 生成分析报告")
        print(f"{'='*80}\n")

        date_folder = generate_date_organized_reports(
            context=context,
            factor_data=factor_data,
            price_data=price_data,
            base_dir=OutputConfig.REPORTS_DIR
        )

        abs_report_path = os.path.abspath(date_folder)
        print(f"📂 报告文件夹: {abs_report_path}")

        # 生成可视化
        if VISUALIZATION_AVAILABLE:
            print("-" * 40)
            print("🎬 正在生成可视化图表...")
            try:
                video_visualization.plot_equity_curve_interactive(context, output_dir=date_folder)

                if 'ml_score' in factor_data.columns:
                    video_visualization.plot_score_timeline(factor_data, top_n=5, output_dir=date_folder)
                    video_visualization.plot_holdings_heatmap(context, factor_data, output_dir=date_folder)

                print("✅ 图表生成完毕")
            except Exception as e:
                print(f"❌ 图表生成部分失败: {e}")

        # 保存交易记录
        if context and 'trade_records' in context:
            trades_df = context['trade_records']
            if not trades_df.empty:
                trades_path = os.path.join(date_folder, "trades.csv")
                trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
                print(f"✅ 交易记录已保存: {len(trades_df)} 条")

        # 生成文字版绩效报告
        generate_performance_report(context, output_dir=date_folder)

        print(f"\n✨ 全部报告生成完毕！")

    except Exception as e:
        print(f"⚠️  报告生成出错: {e}")
        traceback.print_exc()
        # 不阻断流程，继续打印交易计划

    # ============ 步骤9: 打印交易计划 ============
    print_trading_plan(context, price_data, factor_data)

    # ============ 步骤10: 修复效果验证 ============
    if VALIDATE_AVAILABLE and context:
        try:
            print("\n" + "="*80)
            print("🔍 步骤10: 修复效果验证")
            print("="*80)
            quick_validate(context)
        except Exception as e:
            print(f"⚠️  验证过程出错: {e}")

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
        traceback.print_exc()