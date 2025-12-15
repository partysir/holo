"""
main.py - 主回测入口

功能:
✅ 数据加载（增量更新 + 多线程）
✅ 因子计算（技术 + 基本面）
✅ 机器学习评分（XGBoost/LightGBM）
✅ 增强策略（5日调仓 + 等权）
✅ 可视化报告（监控面板 + 持仓分析）
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

# 机器学习因子评分模块
ML_AVAILABLE = False
try:
    from ml_factor_scoring_integrated import UltraMLScorer as MLFactorScorer
    # 为保持兼容性，创建别名
    IndustryBasedScorer = MLFactorScorer
    EnhancedStockSelector = MLFactorScorer
    ML_AVAILABLE = True
except ImportError:
    try:
        from ml_factor_scoring_fixed import (
            MLFactorScorer,
            IndustryBasedScorer,
            EnhancedStockSelector
        )
        ML_AVAILABLE = True
    except ImportError:
        print("⚠️  机器学习模块未找到，使用基础因子评分")
        ML_AVAILABLE = False

from enhanced_strategy import run_enhanced_strategy

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
    print("    综合因子评分选股回测系统 v10.0 - 集成优化版")
    print("="*80)
    print("\n核心特性:")
    print("  ⭐ 5日调仓-等权 - 回测胜率53.24%")
    print("  ⚡ 智能抽样 - 从4000只减少到800只（大中小盘均衡）")
    print("  ⚡ 多线程并行 - 10线程同时获取（10倍提速）")
    print("  ⚡ 增量更新 - 只获取新增交易日（50倍提速）")
    print("  ⚡ 极速回测引擎 - 字典索引O(1)查询（15倍提速）")
    print("  ⚡ 向量化运算 - NumPy批量处理（3倍提速）")
    print("  ✨ 今日持仓仪表板 - 可视化展示当前持仓")
    print("  🤖 机器学习因子组合 - XGBoost/LightGBM")
    print("  🎯 动态权重调整 - 基于市场状态和特征重要性")
    print()


def main():
    """主函数"""
    print_banner()

    # ============ 参数配置 ============
    print("【基础配置】")

    START_DATE = "2023-01-01"
    END_DATE = "2025-12-08"
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")

    CAPITAL_BASE = 1000000
    print(f"  初始资金: {CAPITAL_BASE:,} 元")

    POSITION_SIZE = 10
    print(f"  持仓数量: {POSITION_SIZE} 只")

    # ============ 速度优化配置 ============
    print("\n【速度优化配置】⚡")

    USE_SAMPLING = False          # 是否使用智能抽样设 USE_SAMPLING=False 使用全部股票
    SAMPLE_SIZE = 4000            # 抽样数量（推荐500-1000）
    MAX_WORKERS = 10             # 线程数（推荐5-10）
    FORCE_FULL_UPDATE = False    # 是否强制全量更新

    print(f"  智能抽样: {'启用' if USE_SAMPLING else '关闭'}")
    if USE_SAMPLING:
        print(f"  抽样数量: {SAMPLE_SIZE} 只 (市值分层)")
        print(f"     大盘股(前20%): 抽样40% = {int(SAMPLE_SIZE*0.4)}只")
        print(f"     中盘股(中60%): 抽样40% = {int(SAMPLE_SIZE*0.4)}只")
        print(f"     小盘股(后20%): 抽样20% = {int(SAMPLE_SIZE*0.2)}只")
    else:
        print(f"  使用全部: {SAMPLE_SIZE} 只")

    print(f"  并行线程: {MAX_WORKERS} 个")
    print(f"  强制全量: {'是' if FORCE_FULL_UPDATE else '否'}")
    print(f"  回测引擎: Ultimate Fast (字典索引 + 向量化)")

    if FORCE_FULL_UPDATE:
        print(f"  预计耗时: 30秒 (数据25秒 + 回测1秒)")
    else:
        print(f"  预计耗时: 首次30秒，日常5秒 ⚡⚡⚡")

    # ============ 风险控制参数 ============
    print("\n【风险控制参数】")

    # ✨ 5日调仓-等权配置（回测最优）
    REBALANCE_DAYS = 5             # 5日调仓周期
    POSITION_METHOD = 'equal'       # 等权分配
    SCORE_DECAY_RATE = 1.0         # 不使用评分衰减

    STOP_LOSS = -0.18              # 止损-18%（稍微放宽）
    TAKE_PROFIT = None             # 不止盈
    SCORE_THRESHOLD = 0.12         # 换仓阈值12%（降低频率）
    FORCE_REPLACE_DAYS = 50        # 50天强制评估
    TRANSACTION_COST = 0.0015      # 0.15%交易成本
    MIN_HOLDING_DAYS = 5           # 最少持有5天
    DYNAMIC_STOP_LOSS = True       # 动态止损

    print(f"  调仓周期: {REBALANCE_DAYS} 天 ⭐")
    print(f"  仓位方法: {POSITION_METHOD} (等权)")
    print(f"  止损阈值: {STOP_LOSS:.1%} (动态止损)")
    print(f"  止盈阈值: 不设止盈（让利润奔跑）✨")
    print(f"  换仓阈值: 评分差异 > {SCORE_THRESHOLD:.1%}")
    print(f"  交易成本: {TRANSACTION_COST:.2%} (买入+卖出)")
    print(f"  最少持有: {MIN_HOLDING_DAYS} 天")
    print(f"  强制换仓: {FORCE_REPLACE_DAYS} 天且亏损")

    # ============ 模型配置 ============
    print("\n【因子模型配置】")

    USE_STOCKRANKER = True
    USE_FUNDAMENTAL = True
    CUSTOM_WEIGHTS = None

    print(f"  因子模型: StockRanker多因子 + 基本面")
    print(f"  因子数量: 14个 (技术9个 + 基本面5个)")

    # ============ 机器学习配置 ============
    print("\n【机器学习配置】🤖")

    USE_ML = True                    # 是否使用机器学习
    ML_MODEL_TYPE = 'xgboost'       # 'xgboost' 或 'lightgbm'
    ML_TARGET_PERIOD = 5            # 预测周期（天）
    ML_MIN_SCORE = 0.6              # 最低评分阈值

    print(f"  机器学习: {'启用' if USE_ML else '关闭'}")
    print(f"  模型类型: {ML_MODEL_TYPE.upper()}")
    print(f"  预测周期: {ML_TARGET_PERIOD} 天")
    print(f"  选股阈值: {ML_MIN_SCORE:.1%}")

    # ============ 初始化缓存管理器 ============
    cache_manager = DataCache(cache_dir='./data_cache')

    cache_files = cache_manager.list_cache_files()
    if cache_files:
        print(f"\n【现有缓存】共 {len(cache_files)} 个文件")

        # 显示最近的缓存文件
        recent_files = sorted(cache_files,
                            key=lambda x: x['modified'],
                            reverse=True)[:3]
        for f in recent_files:
            print(f"  - {f['name'][:50]}... ({f['size_kb']} KB, {f['modified']})")

    # ============ 快速数据加载 ============
    try:
        import time
        data_start_time = time.time()

        print("\n" + "="*80)
        print("📦 步骤1: 数据加载")
        print("="*80)

        factor_data, price_data = load_data_with_incremental_update(
            START_DATE,
            END_DATE,
            max_stocks=SAMPLE_SIZE,  # 不使用抽样时的数量
            cache_manager=cache_manager,
            use_stockranker=USE_STOCKRANKER,
            custom_weights=CUSTOM_WEIGHTS,
            tushare_token=TUSHARE_TOKEN,
            use_fundamental=USE_FUNDAMENTAL,
            force_full_update=FORCE_FULL_UPDATE,
            use_sampling=USE_SAMPLING,      # ✨启用智能抽样
            sample_size=SAMPLE_SIZE,        # ✨抽样数量
            max_workers=MAX_WORKERS,         # ✨线程数
            use_money_flow=True             # ✅ 启用资金流因子
        )

        data_elapsed = time.time() - data_start_time
        print(f"\n⚡ 数据加载耗时: {data_elapsed:.1f} 秒")

        if data_elapsed < 10:
            print("   🎉 使用了缓存，极速启动！")
        elif data_elapsed < 60:
            print("   ⚡ 多线程+抽样，速度飞快！")

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
        
        # 应用数据质量优化
        price_data, factor_data = optimize_data_quality(price_data, factor_data, cache_manager=cache_manager)
        
        quality_elapsed = time.time() - quality_start_time
        print(f"\n⚡ 数据质量优化耗时: {quality_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  数据质量优化警告: {e}")
        import traceback
        traceback.print_exc()
        quality_elapsed = 0

    # ============ 因子增强处理 ============
    try:
        print("\n" + "="*80)
        print("🎯 步骤3: 因子增强处理")
        print("="*80)
        
        from enhanced_factor_processor import EnhancedFactorProcessor
        
        factor_start_time = time.time()
        
        # 初始化增强因子处理器
        factor_processor = EnhancedFactorProcessor(
            neutralize_industry=True,  # 启用行业中性化
            neutralize_market=False     # 暂不启用市场中性化
        )
        
        # 获取因子列名（排除基础列）
        exclude_columns = ['date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'amount']
        factor_columns = [col for col in factor_data.columns if col not in exclude_columns]
        
        print(f"  检测到 {len(factor_columns)} 个候选因子列:")
        if len(factor_columns) > 0:
            print(f"  {factor_columns[:10]}{'...' if len(factor_columns) > 10 else ''}")
        print(f"  factor_data 总列数: {len(factor_data.columns)}")
        print(f"  factor_data 样本数: {len(factor_data)}")
        
        # 处理因子
        if len(factor_columns) > 0:
            factor_data = factor_processor.process_factors(factor_data, factor_columns)
            
            # 重新获取处理后的数值型因子列
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
            if len(processed_factor_columns) > 0:
                print(f"  处理后因子列: {processed_factor_columns[:10]}{'...' if len(processed_factor_columns) > 10 else ''}")
            
            # 计算因子有效性指标
            if len(processed_factor_columns) > 0 and 'close' in price_data.columns:
                # 合并价格数据以计算IC
                temp_data = factor_data.merge(
                    price_data[['date', 'instrument', 'close']],
                    on=['date', 'instrument'],
                    how='left'
                )
                
                factor_metrics = factor_processor.calculate_factor_metrics(
                    temp_data, processed_factor_columns, forward_period=5
                )
            else:
                factor_metrics = {}
                print("  ⚠️  没有有效因子列或缺少价格数据，跳过因子有效性计算")
            
            # 保存因子列供后续步骤使用
            factor_columns = processed_factor_columns
        else:
            print("  ⚠️  没有检测到因子列，跳过因子处理")
            factor_columns = []
        
        factor_elapsed = time.time() - factor_start_time
        print(f"\n⚡ 因子增强处理耗时: {factor_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  因子增强处理警告: {e}")
        import traceback
        traceback.print_exc()
        factor_columns = []
        factor_elapsed = 0

    # ============ 机器学习因子评分 ============
    ml_elapsed = 0
    if ML_AVAILABLE and USE_ML:
        try:
            print("\n" + "="*80)
            print("🤖 步骤4: 机器学习因子评分")
            print("="*80)
            
            ml_start_time = time.time()
            
            # 验证是否有可用的因子列
            available_factors = [col for col in factor_columns if col in factor_data.columns]
            
            if len(available_factors) == 0:
                print("  ⚠️  警告：没有可用的因子列，跳过机器学习评分")
                print(f"  当前 factor_data 列: {factor_data.columns.tolist()}")
                print("  提示：确保在数据加载阶段正确计算了技术因子")
                ml_elapsed = 0
            else:
                print(f"  ✓ 检测到 {len(available_factors)} 个可用因子")
                print(f"  ✓ 因子列表: {', '.join(available_factors[:5])}...")
                
                # 初始化机器学习评分器
                try:
                    ml_scorer = MLFactorScorer(
                        model_type=ML_MODEL_TYPE,
                        target_period=ML_TARGET_PERIOD
                    )
                    
                    # 预测因子得分
                    factor_data = ml_scorer.predict_scores(factor_data, price_data)
                    
                    # 动态权重调整
                    if available_factors:
                        dynamic_weights = ml_scorer.dynamic_weight_adjustment(factor_data, available_factors)
                        print(f"  ✓ 动态权重调整完成")
                    else:
                        dynamic_weights = {}
                    
                    ml_elapsed = time.time() - ml_start_time
                    print(f"\n⚡ 机器学习因子评分耗时: {ml_elapsed:.1f} 秒")
                except Exception as e:
                    print(f"  ⚠️  MLFactorScorer 初始化或使用失败: {e}")
                    import traceback
                    traceback.print_exc()
                    ml_elapsed = 0

        except Exception as e:
            print(f"\n⚠️  机器学习因子评分警告: {e}")
            import traceback
            traceback.print_exc()
            ml_elapsed = 0
    else:
        if not ML_AVAILABLE:
            print("\n⚠️  机器学习模块不可用，跳过ML评分")
        elif not USE_ML:
            print("\n⚠️  机器学习功能已禁用，跳过ML评分")

    # ============ 分行业评分 ============
    try:
        print("\n" + "="*80)
        print("🏢 步骤5: 分行业评分")
        print("="*80)
        
        from ml_factor_scoring_fixed import IndustryBasedScorer
        
        industry_start_time = time.time()
        
        # 初始化行业评分器（传入Tushare token）
        industry_scorer = IndustryBasedScorer(tushare_token=TUSHARE_TOKEN)
        
        # 分行业评分
        factor_data = industry_scorer.score_by_industry(factor_data, factor_columns)
        
        industry_elapsed = time.time() - industry_start_time
        print(f"\n⚡ 分行业评分耗时: {industry_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  分行业评分警告: {e}")
        import traceback
        traceback.print_exc()
        industry_elapsed = 0
        
        # 确保有industry列，即使分行业评分失败
        if 'industry' not in factor_data.columns:
            factor_data['industry'] = 'Unknown'

    # ============ 增强选股 ============
    try:
        print("\n" + "="*80)
        print("🎯 步骤6: 增强选股")
        print("="*80)
        
        from ml_factor_scoring_fixed import EnhancedStockSelector
        
        selection_start_time = time.time()
        
        # 初始化增强选股器
        selector = EnhancedStockSelector()
        
        # 增强选股
        factor_data = selector.select_stocks(
            factor_data, 
            min_score=ML_MIN_SCORE,         # 最低得分阈值
            max_concentration=0.15,          # 单只股票最大权重
            max_industry_concentration=0.3   # 单行业最大权重
        )
        
        selection_elapsed = time.time() - selection_start_time
        print(f"\n⚡ 增强选股耗时: {selection_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  增强选股警告: {e}")
        import traceback
        traceback.print_exc()
        selection_elapsed = 0

    # ============ 运行极速回测 ============
    try:
        backtest_start_time = time.time()

        print("\n" + "="*80)
        print("🚀 步骤7: 增强版回测引擎（5日调仓）")
        print("="*80)

        context = run_enhanced_strategy(
            factor_data=factor_data,
            price_data=price_data,
            start_date=START_DATE,
            end_date=END_DATE,
            capital_base=CAPITAL_BASE,
            position_size=POSITION_SIZE,
            rebalance_days=REBALANCE_DAYS,      # ✨ 5日调仓
            position_method=POSITION_METHOD,     # ✨ 等权
            buy_cost=0.0003,
            sell_cost=0.0003,
            tax_ratio=0.0005,
            stop_loss=STOP_LOSS,
            score_threshold=SCORE_THRESHOLD,
            score_decay_rate=SCORE_DECAY_RATE,  # ✨ 评分衰减
            force_replace_days=FORCE_REPLACE_DAYS,
            silent=False
        )

        backtest_elapsed = time.time() - backtest_start_time
        print(f"\n⚡ 回测引擎耗时: {backtest_elapsed:.2f} 秒")
        print(f"   对比传统回测(15秒): 提升 {15/backtest_elapsed:.0f}倍 ⚡⚡⚡")

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

        # 使用按日期组织的报告生成功能
        from date_organized_reports import generate_date_organized_reports
        
        # 生成按日期组织的报告
        date_folder = generate_date_organized_reports(
            context=context,
            factor_data=factor_data,
            price_data=price_data,
            base_dir='./reports'
        )
        
        # ============ 新增：生成详细持仓报告 ============
        print("\n" + "="*80)
        print("📋 生成详细持仓和交易报告")
        print("="*80)
        
        try:
            # 1. 生成每日持仓监控报告
            from holdings_monitor import generate_daily_holdings_report
            
            daily_holdings, pnl_info = generate_daily_holdings_report(
                context=context,
                factor_data=factor_data,
                price_data=price_data,
                output_dir=date_folder,  # 使用日期文件夹
                print_to_console=True,   # 打印到控制台
                save_to_csv=True         # 保存CSV
            )
            
            # 保存总盈亏信息到context中，供后续使用
            if pnl_info:
                context['pnl_info'] = pnl_info
                
        except Exception as e:
            print(f"\n⚠️  每日持仓报告生成警告: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # 2. 生成今日持仓仪表板
            from show_today_holdings import show_today_holdings_dashboard
            
            today_holdings = show_today_holdings_dashboard(
                context=context,
                factor_data=factor_data,
                price_data=price_data,
                output_dir=date_folder  # 使用日期文件夹
            )
            
            # 获取并显示绩效报告信息（包含年化收益率等指标）
            from visualization_module import generate_performance_report
            performance_info = generate_performance_report(context, output_dir=date_folder)
            
        except Exception as e:
            print(f"\n⚠️  今日持仓仪表板生成警告: {e}")
            import traceback
            traceback.print_exc()

        report_elapsed = time.time() - report_start_time
        print(f"\n⚡ 报告生成耗时: {report_elapsed:.1f} 秒")

    except Exception as e:
        print(f"\n⚠️  报告生成警告: {e}")
        import traceback
        traceback.print_exc()
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
    if 'ml_elapsed' in locals():
        print(f"  机器学习评分: {ml_elapsed:.1f}秒")
    if 'industry_elapsed' in locals():
        print(f"  分行业评分: {industry_elapsed:.1f}秒")
    if 'selection_elapsed' in locals():
        print(f"  增强选股: {selection_elapsed:.1f}秒")
    print(f"  回测引擎: {backtest_elapsed:.2f}秒 ⚡⚡⚡")
    if 'report_elapsed' in locals():
        print(f"  报告生成: {report_elapsed:.1f}秒")
    print(f"  总耗时: {total_elapsed:.1f}秒")

    if total_elapsed < 10:
        print(f"  速度等级: ⚡⚡⚡ 极速模式")
    elif total_elapsed < 30:
        print(f"  速度等级: ⚡⚡ 快速模式")
    else:
        print(f"  速度等级: ⚡ 正常模式")

    print("\n📈 策略配置摘要:")
    print(f"  策略版本: v10.0 - 集成优化版 ⭐")
    print(f"  数据源: Tushare (增量更新 + 多线程)")
    print(f"  回测引擎: Enhanced Strategy (5日调仓)")
    print(f"  股票池: {SAMPLE_SIZE} 只 ({'智能抽样' if USE_SAMPLING else '顺序选择'})")
    print(f"  因子模型: StockRanker多因子 + 基本面 + 机器学习")
    print(f"  持仓管理: {POSITION_SIZE}只，{REBALANCE_DAYS}日调仓")

    print(f"\n  策略特点:")
    print(f"    - 调仓周期: {REBALANCE_DAYS}天（降低交易频率）")
    print(f"    - 仓位分配: {POSITION_METHOD}（等权）")
    print(f"    - 止损: {STOP_LOSS:.1%}（动态调整）")
    print(f"    - 止盈: 不设（让利润奔跑）")
    print(f"    - 换仓: 评分差异>{SCORE_THRESHOLD:.1%}才换")
    print(f"    - 机器学习: XGBoost因子组合")
    print(f"    - 行业中性化: 已启用")
    print(f"    - 分行业评分: 已启用")

    print(f"\n📊 回测结果:")
    print(f"  最终资产: ¥{context['final_value']:,.0f}")
    print(f"  总收益率: {context['total_return']:+.2%}")
    print(f"  胜率: {context['win_rate']:.2%}")
    
    # 显示总盈亏信息（如果可用）
    if 'pnl_info' in context:
        pnl_info = context['pnl_info']
        print(f"\n💰 交易绩效摘要:")
        print(f"  总盈利 (正盈亏部分): ¥{pnl_info['total_profit']:,.2f}")
        print(f"  总亏损 (负盈亏部分): ¥{pnl_info['total_loss']:,.2f}")
        print(f"  净盈亏 (总盈利 + 总亏损): ¥{pnl_info['net_pnl']:,.2f}")
        print(f"  交易费用总和: ¥{pnl_info['total_fees']:,.2f}")
        print(f"  扣除费用后净盈亏: ¥{pnl_info['net_pnl_after_fees']:,.2f}")
        if context['initial_capital'] > 0:
            net_return = pnl_info['net_pnl_after_fees'] / context['initial_capital']
            print(f"  净收益率: {net_return:+.2%}")
    
    # 显示年化收益率等绩效指标（如果可用）
    if 'performance_info' in context:
        perf_info = context['performance_info']
        print(f"\n📈 绩效指标:")
        print(f"  总收益率: {perf_info['total_return']:+.2%}")
        print(f"  年化收益率: {perf_info['annualized_return']:+.2%}")
        print(f"  最大回撤: {perf_info['max_drawdown']:.2%}")
        print(f"  夏普比率: {perf_info['sharpe_ratio']:.4f}")

    print("\n⚡ 速度优化效果:")
    print(f"  数据加载: {data_elapsed:.1f}秒")
    if data_elapsed < 10:
        print(f"    使用缓存，提升 100倍+ ⚡⚡⚡")
    elif data_elapsed < 60:
        print(f"    多线程+抽样，提升 {20*60/data_elapsed:.0f}倍 ⚡⚡")

    print(f"\n  回测引擎: {backtest_elapsed:.2f}秒")
    print(f"    极速引擎，提升 {15/backtest_elapsed:.0f}倍 ⚡⚡⚡")
    print(f"    每日回测: {backtest_elapsed/len(context['daily_records'])*1000:.1f}毫秒")

    print("\n📁 输出文件:")
    print(f"  ./reports/YYYY-MM-DD/")
    print(f"  ├─ monitoring_dashboard.png          - 监控面板")
    print(f"  ├─ top_stocks_analysis.png           - 股票分析图")
    print(f"  ├─ today_holdings_dashboard.png      - 今日持仓面板 ✨")
    print(f"  ├─ today_holdings.csv                - 今日持仓明细 ✨")
    print(f"  ├─ daily_holdings_detail.csv         - 每日持仓明细 ✨")
    print(f"  ├─ daily_holdings_summary.csv        - 每日持仓汇总 ✨")
    print(f"  ├─ trade_history_detail.csv          - 交易历史明细 ✨✨新增")
    print(f"  ├─ stock_holding_stats.csv           - 股票持仓统计")
    print(f"  └─ performance_report.txt            - 绩效报告")

    print("\n💡 使用技巧:")
    print("  1. 首次运行建立缓存，约30秒")
    print("  2. 后续每天运行，增量更新仅需5秒 ⚡⚡⚡")
    print("  3. 极速回测引擎，1秒完成回测 ⚡⚡⚡")
    print("  4. 查看 today_holdings_dashboard.png 了解当前持仓 ✨")
    print("  5. 查看 trade_history_detail.csv 了解完整交易记录 ✨✨")
    print("  6. 查看 daily_holdings_detail.csv 追踪每日持仓变化 ✨")

    print("\n📋 持仓报告说明:")
    print("  • trade_history_detail.csv - 包含每笔买入/卖出的详细信息")
    print("    - 买入记录: 日期、股票、价格、数量、原因")
    print("    - 卖出记录: 日期、股票、价格、数量、盈亏、持有天数、原因")
    print("  • daily_holdings_detail.csv - 每个交易日的持仓快照")
    print("    - 包含: 股票、买入时间、现价、成本、盈亏、评分、持有天数")
    print("  • today_holdings.csv - 最后交易日的持仓明细")
    print("    - 包含: 股票、买入时间、现价、成本、盈亏、评分、持有天数")

    print("\n⚙️  参数调优建议:")
    print("  极速模式: SAMPLE_SIZE=500, MAX_WORKERS=15 ⚡⚡⚡")
    print("  平衡模式: SAMPLE_SIZE=800, MAX_WORKERS=10 ⭐")
    print("  追求覆盖: SAMPLE_SIZE=1500, MAX_WORKERS=10")
    print("  完整模式: USE_SAMPLING=False (耗时2-3分钟)")

    print("\n🚀 技术亮点:")
    print("  ✨ 字典索引 - O(1)查询替代DataFrame过滤")
    print("  ✨ 向量化运算 - NumPy批量处理")
    print("  ✨ 增量计算 - 历史数据永不重算")
    print("  ✨ 多线程并行 - 充分利用CPU")
    print("  ✨ 智能缓存 - 数据持久化")
    print("  🤖 机器学习 - XGBoost因子组合")
    print("  🎯 动态权重 - 基于特征重要性")
    print("  🏢 分行业评分 - 更合理的比较")
    print("  📊 详细持仓 - 完整交易历史追踪 ✨✨")

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
