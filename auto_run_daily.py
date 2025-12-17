"""
auto_run_daily.py - 全自动每日运行脚本
功能：
1. 自动将 END_DATE 设置为今天
2. 执行增量数据更新
3. 运行策略生成最新信号
4. 输出《今日调仓指令》和《当前持仓明细》
"""

import warnings

warnings.filterwarnings('ignore')

import tushare as ts
import pandas as pd
import numpy as np
import time
import datetime
import sys
import os

# ========== 1. 动态设置日期为今天 ==========
today = datetime.datetime.now().strftime('%Y%m%d')
print(f"\n📅 启动自动运行程序，当前日期: {today}")

# 导入配置
from config import (
    TUSHARE_TOKEN,
    StrategyConfig,
    BacktestConfig,
    DataConfig,
    FactorConfig,
    MLConfig,
    OutputConfig,
    get_strategy_params
)

# 强制覆盖配置中的结束日期为今天
BacktestConfig.END_DATE = today
# 建议：实盘运行时，开始日期往推1-2年即可，不需要跑太久，提高速度
# BacktestConfig.START_DATE = (datetime.datetime.now() - datetime.timedelta(days=730)).strftime('%Y%m%d')

ts.set_token(TUSHARE_TOKEN)

from data_module import DataCache
from data_module_incremental import load_data_with_incremental_update
from show_today_holdings import show_today_holdings_dashboard

# 尝试导入策略引擎
try:
    from factor_based_risk_control_optimized import run_factor_based_strategy_v2
except ImportError:
    from factor_based_risk_control import run_factor_based_strategy

# 尝试导入ML模块
ML_AVAILABLE = False
try:
    from ml_factor_scoring_fixed import AdvancedMLScorer, EnhancedStockSelector

    ML_AVAILABLE = True
except ImportError:
    pass


def run_daily_task():
    print(f"\n{'=' * 60}")
    print(f"🚀 开始执行每日策略更新任务 [{today}]")
    print(f"{'=' * 60}")

    # 1. 初始化
    cache_manager = DataCache(cache_dir=DataConfig.CACHE_DIR)

    # 2. 获取数据 (自动增量更新)
    print("\n📦正在检查并更新数据...")

    # 确保实盘时必须使用足够大的股票池
    sample_size = DataConfig.SAMPLE_SIZE
    if sample_size < 4000:
        sample_size = 5000

    factor_data, price_data = load_data_with_incremental_update(
        BacktestConfig.START_DATE,
        BacktestConfig.END_DATE,
        max_stocks=sample_size,
        cache_manager=cache_manager,
        use_stockranker=FactorConfig.USE_STOCKRANKER,
        custom_weights=FactorConfig.CUSTOM_WEIGHTS,
        tushare_token=TUSHARE_TOKEN,
        use_fundamental=FactorConfig.USE_FUNDAMENTAL,
        force_full_update=False,  # 增量更新
        use_sampling=False,  # 实盘必须关闭抽样，跑全市场
        sample_size=sample_size,
        max_workers=DataConfig.MAX_WORKERS
    )

    if factor_data is None:
        print("❌ 数据获取失败，程序终止")
        return

    # 3. 数据处理与评分 (简化版流程)
    print("\n⚙️ 正在处理因子与评分...")
    from enhanced_factor_processor import EnhancedFactorProcessor
    processor = EnhancedFactorProcessor(neutralize_industry=True)

    # 筛选因子列
    exclude = ['date', 'instrument', 'open', 'high', 'low', 'close', 'volume', 'amount', 'industry']
    cols = [c for c in factor_data.columns if c not in exclude and pd.api.types.is_numeric_dtype(factor_data[c])]

    if cols:
        factor_data = processor.process_factors(factor_data, cols)

    # ML 评分 (如果启用)
    if MLConfig.USE_ADVANCED_ML and ML_AVAILABLE:
        print("🤖 执行ML模型预测...")
        scorer = AdvancedMLScorer(
            model_type=MLConfig.ML_MODEL_TYPE,
            use_classification=MLConfig.ML_USE_CLASSIFICATION,
            train_months=MLConfig.ML_TRAIN_MONTHS
        )
        factor_data = scorer.predict_scores(factor_data, price_data, cols)

    # 4. 运行策略回测引擎 (计算到今天的最新仓位)
    print("\n📈 计算最新持仓状态...")
    strategy_params = get_strategy_params()

    # 运行策略
    context = run_factor_based_strategy_v2(
        factor_data=factor_data,
        price_data=price_data,
        **strategy_params
    )

    # 5. 生成今日报告
    print(f"\n{'=' * 60}")
    print(f"📢 {today} 策略信号生成完毕")
    print(f"{'=' * 60}\n")

    # 调用现有的仪表盘功能，并保存到 reports/today
    today_report_dir = os.path.join(OutputConfig.REPORTS_DIR, f"daily_run_{today}")
    if not os.path.exists(today_report_dir):
        os.makedirs(today_report_dir)

    # 生成持仓仪表盘
    show_today_holdings_dashboard(
        context=context,
        factor_data=factor_data,
        price_data=price_data,
        output_dir=today_report_dir
    )

    # 6. 提取并打印具体的调仓指令
    print_action_plan(context, price_data)


def print_action_plan(context, price_data):
    """
    专门打印今日（或最近一个交易日）的调仓指令
    """
    df_history = pd.DataFrame(context['history'])
    if df_history.empty:
        print("无历史交易记录")
        return

    last_date = df_history['date'].max()
    print(f"\n📝 【调仓指令单】 信号日期: {last_date}")

    # 获取最近一天的交易记录
    actions = df_history[df_history['date'] == last_date]

    if actions.empty:
        print("✅ 今日无调仓操作，继续持有现有组合。")
    else:
        print(f"⚠️ 发现 {len(actions)} 笔调仓指令，请执行：")
        print("-" * 50)
        print(f"{'方向':<6} | {'代码':<10} | {'价格':<8} | {'股数':<8} | {'金额':<10}")
        print("-" * 50)

        for _, row in actions.iterrows():
            direction = "买入" if row['action'] == 'buy' else "卖出"
            print(
                f"{direction:<6} | {row['instrument']:<10} | {row['price']:<8.2f} | {row['shares']:<8} | {row['cost']:<10.0f}")
        print("-" * 50)

    # 打印当前持仓摘要
    positions = context['positions']
    if positions:
        print(f"\n💼 【当前持仓】 共 {len(positions)} 只")
        total_mv = 0
        for code, pos in positions.items():
            # 获取最新价格
            last_price = 0
            stock_price = price_data[price_data['instrument'] == code]
            if not stock_price.empty:
                last_price = stock_price.iloc[-1]['close']

            mv = pos['shares'] * last_price
            total_mv += mv
            print(f"   - {code}: {pos['shares']}股 (市值: ¥{mv:,.0f})")

        cash = context['cash']
        print(f"\n💰 账户概览:")
        print(f"   股票市值: ¥{total_mv:,.0f}")
        print(f"   可用现金: ¥{cash:,.0f}")
        print(f"   总资产:   ¥{total_mv + cash:,.0f}")


if __name__ == "__main__":
    try:
        run_daily_task()

        # 保持窗口打开 60秒，方便查看结果（如果是双击运行）
        print("\n✅ 任务完成。窗口将在60秒后关闭...")
        time.sleep(60)

    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback

        traceback.print_exc()
        input("按回车键退出...")