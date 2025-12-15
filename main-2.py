#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合因子评分选股回测系统 v2.4 - 完整修复版

核心修复：
1. ✅ 数据质量检查（过滤一字板、无效代码）
2. ✅ 统一 ML 模块（使用 ml_factor_scoring_unified）
3. ✅ 防止前视偏差（上市日期过滤）
4. ✅ 流动性约束（限制单只股票买入量）
5. ✅ 资金守恒验证（确保现金不为负）
"""

import time
import traceback
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

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
)

ts.set_token(TUSHARE_TOKEN)

# 导入数据模块
from data_module import DataCache, TushareDataSource
from data_module_incremental import load_data_with_incremental_update

# ========== 导入整合版 ML 模块 ==========
ML_AVAILABLE = False
try:
    from ml_factor_scoring_integrated import UltraMLScorer

    ML_AVAILABLE = True
    print("✓ 整合版 ML 模块加载成功")
except ImportError:
    try:
        from ml_factor_scoring_unified import UltraMLScorer

        ML_AVAILABLE = True
        print("✓ 统一修复版 ML 模块加载成功")
    except ImportError:
        try:
            from ml_factor_scoring_fixed import UltraMLScorer

            ML_AVAILABLE = True
            print("✓ 固定版 ML 模块加载成功")
        except ImportError as e:
            print(f"⚠️  ML 模块未找到: {e}")
            ML_AVAILABLE = False

# ========== 导入数据质量检查工具 ==========
try:
    from data_quality_checker import (
        DataQualityChecker,
        filter_unbuyable_stocks,
        fix_invalid_codes
    )

    DATA_QUALITY_AVAILABLE = True
    print("✓ 数据质量检查工具加载成功")
except ImportError:
    DATA_QUALITY_AVAILABLE = False
    print("⚠️  数据质量检查工具未找到（可选）")

# ========== 导入策略引擎 ==========
try:
    from factor_based_risk_control_optimized import run_factor_based_strategy_v2

    print("✓ v2.2 优化版策略引擎加载成功")
    STRATEGY_VERSION = "v2.2"
except ImportError:
    print("⚠️  v2.2 优化版未找到")
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
    print("    综合因子评分选股回测系统 v2.4 - 完整修复版")
    print("=" * 80)
    print("\n🎯 核心修复:")
    print("  ✅ 数据质量检查 - 过滤一字板、无效代码")
    print("  ✅ 统一 ML 模块 - 使用 ml_factor_scoring_unified")
    print("  ✅ 防止前视偏差 - 上市日期过滤 + 数据清洗")
    print("  ✅ 流动性约束 - 限制单只股票最大买入量")
    print("  ✅ 资金守恒验证 - 确保现金计算正确")
    print()


def run_data_quality_check(price_data, factor_data, trade_records=None):
    """
    🔍 运行数据质量检查

    Returns:
        dict: 检查结果
    """
    if not DATA_QUALITY_AVAILABLE:
        print("  ⏭️  跳过数据质量检查（工具未安装）")
        return None

    print("\n" + "=" * 80)
    print("🔍 数据质量全面检查")
    print("=" * 80)

    checker = DataQualityChecker()
    results = checker.run_full_check(price_data, trade_records)

    return results


def apply_data_fixes(price_data, factor_data):
    """
    🛠️ 应用数据修复

    Returns:
        tuple: (clean_price_data, clean_factor_data)
    """
    if not DATA_QUALITY_AVAILABLE:
        print("  ⏭️  跳过数据修复（工具未安装）")
        return price_data, factor_data

    print("\n" + "=" * 80)
    print("🛠️ 应用数据修复")
    print("=" * 80)

    # 1. 修复无效代码
    print("\n1️⃣ 修复股票代码...")
    code_mapping = {
        '302132.SZ': '300114.SZ',  # 中航电测
    }
    price_data = fix_invalid_codes(price_data, code_mapping)
    factor_data = fix_invalid_codes(factor_data, code_mapping)

    # 2. 过滤无法买入的股票
    print("\n2️⃣ 过滤一字涨停板和无成交量数据...")
    price_data = filter_unbuyable_stocks(price_data)

    # 3. 同步 factor_data
    print("\n3️⃣ 同步因子数据...")
    valid_combinations = set(
        zip(price_data['date'].astype(str), price_data['instrument'])
    )

    factor_data['date'] = factor_data['date'].astype(str)
    original_len = len(factor_data)

    factor_data = factor_data[
        factor_data.apply(
            lambda x: (x['date'], x['instrument']) in valid_combinations,
            axis=1
        )
    ]

    filtered_len = original_len - len(factor_data)
    print(f"  ✓ 因子数据同步完成")
    print(f"    原始: {original_len:,} 行")
    print(f"    保留: {len(factor_data):,} 行")
    print(f"    移除: {filtered_len:,} 行 ({filtered_len / original_len * 100:.2f}%)")

    return price_data, factor_data


def diagnose_abnormal_trades(context):
    """
    🔍 诊断异常交易（集成到主程序）
    """
    if context is None:
        return

    print("\n" + "=" * 80)
    print("🔍 交易异常诊断")
    print("=" * 80)

    df_trades = pd.DataFrame(context.get('trade_records', []))

    if df_trades.empty:
        print("  ℹ️  无交易记录")
        return

    buy_trades = df_trades[df_trades['action'] == 'buy'].copy()

    if len(buy_trades) == 0:
        print("  ℹ️  无买入记录")
        return

    # 1. 检查股数分布
    print(f"\n📊 买入股数统计:")
    print(f"  总买入次数: {len(buy_trades)}")
    print(f"  最小股数: {buy_trades['shares'].min():,.0f}")
    print(f"  最大股数: {buy_trades['shares'].max():,.0f}")
    print(f"  平均股数: {buy_trades['shares'].mean():,.0f}")
    print(f"  中位数: {buy_trades['shares'].median():,.0f}")

    # 2. 找出异常交易
    abnormal = buy_trades[buy_trades['shares'] > 100000].copy()

    if len(abnormal) > 0:
        print(f"\n⚠️  发现 {len(abnormal)} 笔异常大额交易 (>100,000股):")
        print(abnormal[['date', 'stock', 'shares', 'price', 'amount']].head(10).to_string())

        # 3. 分析第一笔异常交易
        first_abnormal = abnormal.iloc[0]
        print(f"\n🔎 第一笔异常交易详情:")
        print(f"  日期: {first_abnormal['date']}")
        print(f"  股票: {first_abnormal['stock']}")
        print(f"  股数: {first_abnormal['shares']:,.0f}")
        print(f"  价格: ¥{first_abnormal['price']:.2f}")
        print(f"  金额: ¥{first_abnormal['amount']:,.2f}")

        # 验证是否为一字板
        print(f"\n  ⚠️  建议检查:")
        print(f"    1. 该股票当日是否为一字涨停")
        print(f"    2. 股票代码是否正确")
        print(f"    3. 数据是否经过复权处理")
    else:
        print(f"\n✅ 未发现异常大额交易")


def print_trading_plan(context, price_data, factor_data):
    """
    🖨️ 打印交易计划和持仓监控
    """
    if context is None:
        return

    print("\n" + "#" * 80)
    print("📋 步骤9: 交易指令与持仓监控")
    print("#" * 80 + "\n")

    df_trades = context.get('trade_records', pd.DataFrame())
    if df_trades.empty:
        print("⚠️ 无交易记录")
        return

    last_date = df_trades['date'].max()
    today_trades = df_trades[df_trades['date'] == last_date].copy()

    print(f"📅 信号日期: {last_date}")

    # 打印调仓指令
    print(f"\n📢 【今日调仓指令】 共 {len(today_trades)} 笔")
    if len(today_trades) == 0:
        print("   ✅ 今日无操作")
    else:
        print("-" * 75)
        print(f"{'方向':<6} | {'代码':<10} | {'价格':<8} | {'股数':<8} | {'金额':<10} | {'原因'}")
        print("-" * 75)

        for _, row in today_trades.iterrows():
            action = "🔵买入" if row['action'] == 'buy' else "🔴卖出"
            price = row.get('price', 0)
            shares = row.get('shares', 0)
            amount = row.get('amount', 0)
            reason = row.get('reason', '')

            print(f"{action:<6} | {row['stock']:<10} | {price:<8.2f} | "
                  f"{shares:<8.0f} | ¥{amount:<9.0f} | {reason}")
        print("-" * 75)

    # 打印持仓
    positions = context.get('positions', {})
    final_value = context.get('final_value', 0)

    if not positions:
        print("\n💼 【当前持仓】 空仓")
    else:
        print(f"\n💼 【当前持仓】 共 {len(positions)} 只")
        print("-" * 100)
        print(f"{'代码':<10} | {'买入日期':<12} | {'股数':<8} | "
              f"{'成本':<8} | {'现价':<8} | {'浮盈':<10} | {'收益率':<8}")
        print("-" * 100)

        # 获取最新价格和评分
        try:
            last_scores = (
                factor_data[factor_data['date'] == str(last_date)]
                [['instrument', 'position']]
                .set_index('instrument')['position']
                .to_dict()
            )
            last_prices = (
                price_data[price_data['date'] == str(last_date)]
                [['instrument', 'close']]
                .set_index('instrument')['close']
                .to_dict()
            )
        except:
            last_scores = {}
            last_prices = {}

        total_mv = 0
        total_pnl = 0

        for code, info in positions.items():
            shares = info['shares']
            cost = info['cost']
            entry_date = info['entry_date']
            current_price = last_prices.get(code, cost)

            mv = shares * current_price
            pnl = (current_price - cost) * shares
            pnl_rate = (current_price - cost) / cost if cost > 0 else 0

            total_mv += mv
            total_pnl += pnl

            print(f"{code:<10} | {entry_date:<12} | {shares:<8.0f} | "
                  f"{cost:<8.2f} | {current_price:<8.2f} | "
                  f"¥{pnl:+9.0f} | {pnl_rate:+7.2%}")

        print("-" * 100)
        cash = final_value - total_mv
        print(f"💰 账户: 持仓市值 ¥{total_mv:,.0f} | "
              f"现金 ¥{cash:,.0f} | 总资产 ¥{final_value:,.0f}")


def main():
    """主函数"""
    print_banner()

    # ============ 配置参数 ============
    START_DATE = BacktestConfig.START_DATE
    END_DATE = BacktestConfig.END_DATE
    CAPITAL_BASE = BacktestConfig.CAPITAL_BASE
    POSITION_SIZE = BacktestConfig.POSITION_SIZE
    REBALANCE_DAYS = BacktestConfig.REBALANCE_DAYS

    # 采样参数
    USE_SAMPLING = True
    SAMPLE_SIZE = 4000

    # ✅ 关键：最短上市时间
    MIN_DAYS_LISTED = 180  # 6个月

    print("【当前配置】")
    print(f"  策略版本: {STRATEGY_VERSION}")
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")
    print(f"  初始资金: ¥{CAPITAL_BASE:,}")
    print(f"  持仓数量: {POSITION_SIZE} 只")
    print(f"  调仓周期: {REBALANCE_DAYS} 天")
    print(f"\n🔒 防前视偏差:")
    print(f"  - 最短上市时间: {MIN_DAYS_LISTED} 天")
    print(f"  - 数据质量检查: {'✓' if DATA_QUALITY_AVAILABLE else '✗'}")
    print(f"  - ML 模块: {'✓' if ML_AVAILABLE else '✗'}")

    # ============ 初始化 ============
    cache_manager = DataCache(cache_dir=DataConfig.CACHE_DIR)

    # ============ 步骤0: 获取基准数据 ============
    benchmark_data = None
    try:
        if StrategyConfig.ENABLE_MARKET_TIMING:
            print("\n" + "=" * 80)
            print("📈 步骤0: 获取大盘指数（择时）")
            print("=" * 80)

            ds_temp = TushareDataSource(
                cache_manager=cache_manager,
                token=TUSHARE_TOKEN
            )
            benchmark_data = ds_temp.get_index_daily(
                ts_code='000001.SH',
                start_date=START_DATE,
                end_date=END_DATE
            )

            if benchmark_data is not None:
                print(f"  ✓ 获取上证指数: {len(benchmark_data)} 条")
        else:
            print("\n⏭️  大盘择时已禁用")
    except Exception as e:
        print(f"  ⚠️  获取指数失败: {e}")

    # ============ 步骤1: 数据加载 ============
    try:
        print("\n" + "=" * 80)
        print("📦 步骤1: 数据加载 (v2.4 修复版)")
        print("=" * 80)

        data_start = time.time()

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
            min_days_listed=MIN_DAYS_LISTED,
            use_money_flow=FactorConfig.USE_MONEY_FLOW
        )

        if factor_data is None or price_data is None:
            print("\n❌ 数据获取失败")
            return

        if factor_data.empty or price_data.empty:
            print("\n❌ 数据为空")
            return

        print(f"  ✓ 数据加载耗时: {time.time() - data_start:.1f} 秒")
        print(f"  ✓ 股票数: {factor_data['instrument'].nunique()}")
        print(f"  ✓ 交易日: {factor_data['date'].nunique()}")

    except Exception as e:
        print(f"\n❌ 数据加载异常: {e}")
        traceback.print_exc()
        return

    # ============ 步骤1.5: 数据质量检查与修复 ============
    quality_results = run_data_quality_check(price_data, factor_data)
    price_data, factor_data = apply_data_fixes(price_data, factor_data)

    # ============ 步骤1.6: 补全行业数据 ============
    print("\n" + "=" * 80)
    print("🏭 步骤1.6: 补全行业数据")
    print("=" * 80)

    try:
        ds = TushareDataSource(token=TUSHARE_TOKEN, cache_manager=cache_manager)
        unique_stocks = factor_data['instrument'].unique().tolist()
        industry_df = ds.get_industry_data(unique_stocks, use_cache=True)

        if industry_df is not None and not industry_df.empty:
            if 'industry' in factor_data.columns:
                del factor_data['industry']

            factor_data = factor_data.merge(
                industry_df,
                on='instrument',
                how='left'
            )
            factor_data['industry'] = factor_data['industry'].fillna('其他')

            print(f"  ✓ 行业数据: {factor_data['industry'].nunique()} 个")
        else:
            factor_data['industry'] = 'Unknown'
    except Exception as e:
        print(f"  ⚠️  补全行业失败: {e}")
        factor_data['industry'] = 'Unknown'

    # ============ 步骤2: 数据质量优化 ============
    try:
        print("\n" + "=" * 80)
        print("🔍 步骤2: 数据质量优化")
        print("=" * 80)

        from data_quality_optimizer import optimize_data_quality
        price_data, factor_data = optimize_data_quality(
            price_data,
            factor_data,
            cache_manager=cache_manager
        )
    except Exception as e:
        print(f"  ⚠️  优化警告: {e}")

    # ============ 步骤3: 因子增强 ============
    try:
        print("\n" + "=" * 80)
        print("🎯 步骤3: 因子增强处理")
        print("=" * 80)

        from enhanced_factor_processor import EnhancedFactorProcessor

        factor_processor = EnhancedFactorProcessor(
            neutralize_industry=True,
            neutralize_market=False
        )

        exclude_columns = [
            'date', 'instrument', 'open', 'high', 'low',
            'close', 'volume', 'amount', 'industry'
        ]

        factor_columns = [
            col for col in factor_data.columns
            if col not in exclude_columns
               and pd.api.types.is_numeric_dtype(factor_data[col])
        ]

        print(f"  检测到 {len(factor_columns)} 个因子")

        if len(factor_columns) > 0:
            factor_data = factor_processor.process_factors(
                factor_data,
                factor_columns
            )
    except Exception as e:
        print(f"  ⚠️  因子增强警告: {e}")
        traceback.print_exc()

    # ============ 步骤4: ML 评分 ============
    if MLConfig.USE_ADVANCED_ML and ML_AVAILABLE:
        try:
            print("\n" + "=" * 80)
            print("🚀 步骤4: ML 因子评分 (Unified)")
            print("=" * 80)

            ml_scorer = UltraMLScorer(
                target_period=MLConfig.ML_TARGET_PERIOD,
                top_percentile=MLConfig.ML_TOP_PERCENTILE,
                train_months=MLConfig.ML_TRAIN_MONTHS,
                voting_strategy='strict',
                neutralize_market=True,
                neutralize_industry=True
                # debug=False  # 可设为 True 查看详细日志 (整合版暂不支持)
            )

            factor_data = ml_scorer.predict(factor_data, price_data)

        except Exception as e:
            print(f"  ⚠️  ML 评分失败: {e}")
            traceback.print_exc()

    # ============ 步骤7: 回测执行 ============
    context = None
    try:
        print("\n" + "=" * 80)
        print(f"🚀 步骤7: {STRATEGY_VERSION} 回测引擎")
        print("=" * 80)

        strategy_params = get_strategy_params()
        strategy_params['rebalance_days'] = REBALANCE_DAYS

        context = run_factor_based_strategy_v2(
            factor_data=factor_data,
            price_data=price_data,
            benchmark_data=benchmark_data,
            **strategy_params
        )

    except Exception as e:
        print(f"\n❌ 回测异常: {e}")
        traceback.print_exc()
        return

    # ============ 步骤8: 生成报告 ============
    try:
        print(f"\n{'=' * 80}")
        print("📊 步骤8: 生成报告")
        print(f"{'=' * 80}\n")

        # 生成按日期组织的报告
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

        # 生成详细报告
        daily_holdings, pnl_info = generate_daily_holdings_report(
            context=context,
            factor_data=factor_data,
            price_data=price_data,
            output_dir=date_folder,
            print_to_console=True,
            save_to_csv=True
        )

        # 显示绩效
        performance_info = generate_performance_report(
            context,
            output_dir=date_folder
        )

        if pnl_info:
            print("\n" + "=" * 80)
            print("💰 交易绩效摘要")
            print("=" * 80)
            print(f"  总交易: {pnl_info['trade_count']}")
            print(f"  盈利次数: {pnl_info['profit_trades']}")
            print(f"  亏损次数: {pnl_info['loss_trades']}")
            print(f"  净盈亏: ¥{pnl_info['net_pnl']:,.2f}")

            if 'initial_capital' in context:
                net_return = pnl_info['net_pnl_after_fees'] / context['initial_capital']
                print(f"  净收益率: {net_return:+.2%}")

        if performance_info:
            print(f"\n📈 绩效指标:")
            print(f"  总收益率: {performance_info['total_return']:+.2%}")
            print(f"  年化收益率: {performance_info['annualized_return']:+.2%}")
            print(f"  最大回撤: {performance_info['max_drawdown']:.2%}")
            print(f"  夏普比率: {performance_info['sharpe_ratio']:.4f}")

    except Exception as e:
        print(f"  ⚠️  报告生成警告: {e}")

    # ============ 步骤9: 交易诊断 ============
    diagnose_abnormal_trades(context)
    print_trading_plan(context, price_data, factor_data)

    print("\n" + "=" * 80)
    print("✅ 任务完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        traceback.print_exc()