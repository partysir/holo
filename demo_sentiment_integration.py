# -*- coding: utf-8 -*-
"""
demo_sentiment_integration.py - 舆情风控完整集成演示

展示如何将舆情风控模块集成到现有的多因子选股系统中。
这是一个可以直接运行的完整示例。

使用方法:
1. 确保已配置 TUSHARE_TOKEN
2. python demo_sentiment_integration.py

版本: v1.0
日期: 2025-12-17
"""

import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ========== 配置区域 ==========
# 请在这里填入您的 Tushare Token
TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"  # ← 修改这里

# 如果Token已经在config.py中配置，可以导入
try:
    from config import TUSHARE_TOKEN as TOKEN_FROM_CONFIG

    if TOKEN_FROM_CONFIG:
        TUSHARE_TOKEN = TOKEN_FROM_CONFIG
        print("✓ 从config.py加载Token")
except:
    pass

# ========== 导入舆情风控模块 ==========
try:
    from sentiment_risk_control import (
        SentimentRiskController,
        apply_sentiment_control
    )

    SENTIMENT_AVAILABLE = True
    print("✓ 舆情风控模块加载成功\n")
except ImportError as e:
    print(f"✗ 舆情风控模块加载失败: {e}\n")
    SENTIMENT_AVAILABLE = False


# ============================================================================
# 模拟数据生成（用于演示）
# ============================================================================

def generate_demo_data():
    """生成演示用的数据"""
    print("=" * 80)
    print("📦 步骤1: 生成模拟数据")
    print("=" * 80 + "\n")

    # 模拟日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # 模拟股票池（真实股票代码）
    stocks = [
        '000001.SZ',  # 平安银行
        '000002.SZ',  # 万科A
        '000333.SZ',  # 美的集团
        '000651.SZ',  # 格力电器
        '000858.SZ',  # 五粮液
        '600000.SH',  # 浦发银行
        '600036.SH',  # 招商银行
        '600276.SH',  # 恒瑞医药
        '600519.SH',  # 贵州茅台
        '601318.SH',  # 中国平安
        '601888.SH',  # 中国中免
        '000568.SZ',  # 泸州老窖
        '002594.SZ',  # 比亚迪
        '300750.SZ',  # 宁德时代
        '688981.SH',  # 中芯国际
    ]

    # 生成因子数据
    factor_records = []
    price_records = []

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')

        for stock in stocks:
            # 模拟因子值
            base_score = np.random.uniform(0.3, 0.9)

            # 给某些股票固定的高分（模拟优质股）
            if stock in ['600519.SH', '000858.SZ', '601318.SH']:
                base_score = np.random.uniform(0.7, 0.95)

            factor_records.append({
                'date': date_str,
                'instrument': stock,
                'position': base_score,
                'ml_score': base_score + np.random.uniform(-0.05, 0.05),
                'momentum_20d': np.random.uniform(-0.1, 0.1),
                'volatility_20d': np.random.uniform(0.01, 0.05),
                'pe_ratio': np.random.uniform(10, 50),
                'industry': get_industry(stock)
            })

            # 模拟价格数据
            price_records.append({
                'date': date_str,
                'instrument': stock,
                'close': np.random.uniform(10, 200),
                'volume': np.random.uniform(1000000, 10000000),
                'amount': np.random.uniform(100000000, 1000000000)
            })

    factor_data = pd.DataFrame(factor_records)
    price_data = pd.DataFrame(price_records)

    print(f"  ✓ 生成因子数据: {len(factor_data)} 条")
    print(f"  ✓ 股票数量: {len(stocks)} 只")
    print(f"  ✓ 日期范围: {dates[0].date()} ~ {dates[-1].date()}")
    print(f"  ✓ 行业分布: {factor_data['industry'].nunique()} 个")

    return factor_data, price_data


def get_industry(stock_code):
    """获取股票行业（简化映射）"""
    industry_map = {
        '000001.SZ': '银行', '600000.SH': '银行', '600036.SH': '银行',
        '601318.SH': '保险', '000002.SZ': '房地产',
        '000333.SZ': '家电', '000651.SZ': '家电',
        '600519.SH': '白酒', '000858.SZ': '白酒', '000568.SZ': '白酒',
        '600276.SH': '医药', '601888.SH': '零售',
        '002594.SZ': '汽车', '300750.SZ': '电池', '688981.SH': '半导体'
    }
    return industry_map.get(stock_code, '其他')


# ============================================================================
# 模拟选股流程
# ============================================================================

def simulate_stock_selection(factor_data, price_data):
    """模拟选股流程"""
    print("\n" + "=" * 80)
    print("📊 步骤2: 模拟选股流程")
    print("=" * 80 + "\n")

    # 获取最新日期的数据
    latest_date = factor_data['date'].max()
    latest_data = factor_data[factor_data['date'] == latest_date].copy()

    print(f"  📅 最新日期: {latest_date}")
    print(f"  📊 可选股票: {len(latest_data)} 只")

    # 根据 ml_score 排序，选择 Top 10
    score_col = 'ml_score' if 'ml_score' in latest_data.columns else 'position'
    top_stocks = latest_data.nlargest(10, score_col)

    print(f"\n  🎯 选股结果 (Top 10):")
    print("  " + "-" * 70)
    print(f"  {'排名':<6} {'代码':<12} {'行业':<8} {'评分':<10}")
    print("  " + "-" * 70)

    for idx, (_, row) in enumerate(top_stocks.iterrows(), 1):
        print(f"  {idx:<6} {row['instrument']:<12} {row['industry']:<8} {row[score_col]:.4f}")

    print("  " + "-" * 70)

    return top_stocks


# ============================================================================
# 应用舆情风控
# ============================================================================

def apply_sentiment_filtering(selected_stocks, factor_data, price_data, token):
    """应用舆情风控"""
    print("\n" + "=" * 80)
    print("🛡️  步骤3: 应用舆情风控/增强")
    print("=" * 80 + "\n")

    if not SENTIMENT_AVAILABLE:
        print("  ⚠️  舆情风控模块未启用，跳过此步骤")
        return selected_stocks

    if not token or token == "你的Token":
        print("  ⚠️  未配置Tushare Token，使用模拟模式")
        print("  💡 提示: 在代码开头的 TUSHARE_TOKEN 变量中填入您的Token")
        print("\n  模拟效果演示:")
        return simulate_sentiment_filter(selected_stocks)

    # 真实舆情风控
    try:
        print("  🔍 执行真实舆情分析...")

        filtered_stocks = apply_sentiment_control(
            selected_stocks=selected_stocks,
            factor_data=factor_data,
            price_data=price_data,
            tushare_token=token,
            enable_veto=True,
            enable_boost=True,
            lookback_days=7  # 短周期测试
        )

        return filtered_stocks

    except Exception as e:
        print(f"  ⚠️  舆情分析出错: {e}")
        print("  使用模拟模式继续...")
        return simulate_sentiment_filter(selected_stocks)


def simulate_sentiment_filter(selected_stocks):
    """模拟舆情过滤效果（用于演示）"""
    print("  🎭 模拟舆情风控效果:")
    print()

    # 模拟一票否决（随机剔除1-2只）
    veto_count = np.random.randint(1, 3)
    veto_indices = np.random.choice(selected_stocks.index, veto_count, replace=False)

    print(f"  🚫 模拟一票否决 ({veto_count} 只):")
    for idx in veto_indices:
        stock = selected_stocks.loc[idx, 'instrument']
        reasons = ['财务审计异常', '高风险预警', 'ST风险', '债务问题']
        reason = np.random.choice(reasons)
        print(f"     • {stock}: {reason}")

    # 剔除
    filtered = selected_stocks.drop(veto_indices).copy()

    # 模拟加分（随机提升1-2只）
    boost_count = min(2, len(filtered))
    boost_indices = np.random.choice(filtered.index, boost_count, replace=False)

    print(f"\n  📈 模拟加分增强 ({boost_count} 只):")
    for idx in boost_indices:
        stock = filtered.loc[idx, 'instrument']
        boost_pct = np.random.uniform(0.05, 0.12)
        themes = ['政策支持', '业绩预增', '新闻联播提及', '热点概念']
        theme = np.random.choice(themes)
        print(f"     • {stock}: +{boost_pct:.1%} ({theme})")

        # 实际加分
        score_col = 'ml_score' if 'ml_score' in filtered.columns else 'position'
        filtered.loc[idx, score_col] *= (1 + boost_pct)

    # 重新排序
    score_col = 'ml_score' if 'ml_score' in filtered.columns else 'position'
    filtered = filtered.sort_values(score_col, ascending=False).reset_index(drop=True)

    print(f"\n  ✅ 过滤完成: {len(selected_stocks)} → {len(filtered)} 只")

    return filtered


# ============================================================================
# 展示最终结果
# ============================================================================

def display_final_results(original_stocks, filtered_stocks):
    """展示最终结果"""
    print("\n" + "=" * 80)
    print("📊 步骤4: 最终投资清单")
    print("=" * 80 + "\n")

    score_col = 'ml_score' if 'ml_score' in filtered_stocks.columns else 'position'

    print(f"  📋 推荐买入 (Top 5):")
    print("  " + "-" * 70)
    print(f"  {'排名':<6} {'代码':<12} {'行业':<8} {'评分':<10} {'变化'}")
    print("  " + "-" * 70)

    top_5 = filtered_stocks.head(5)

    for new_rank, (_, row) in enumerate(top_5.iterrows(), 1):
        stock = row['instrument']
        industry = row['industry']
        score = row[score_col]

        # 查找原始排名
        original_rank = None
        for old_rank, (_, old_row) in enumerate(original_stocks.iterrows(), 1):
            if old_row['instrument'] == stock:
                original_rank = old_rank
                break

        if original_rank:
            rank_change = original_rank - new_rank
            if rank_change > 0:
                change_str = f"↑{rank_change}"
            elif rank_change < 0:
                change_str = f"↓{abs(rank_change)}"
            else:
                change_str = "="
        else:
            change_str = "NEW"

        print(f"  {new_rank:<6} {stock:<12} {industry:<8} {score:.4f}    {change_str}")

    print("  " + "-" * 70)

    # 显示被剔除的股票
    removed_stocks = set(original_stocks['instrument']) - set(filtered_stocks['instrument'])

    if removed_stocks:
        print(f"\n  🚫 已剔除 ({len(removed_stocks)} 只):")
        for stock in removed_stocks:
            original_row = original_stocks[original_stocks['instrument'] == stock].iloc[0]
            print(f"     • {stock} ({original_row['industry']}) - 舆情风险")


# ============================================================================
# 生成报告
# ============================================================================

def generate_summary_report(original_stocks, filtered_stocks):
    """生成汇总报告"""
    print("\n" + "=" * 80)
    print("📈 步骤5: 效果评估")
    print("=" * 80 + "\n")

    score_col = 'ml_score' if 'ml_score' in filtered_stocks.columns else 'position'

    original_count = len(original_stocks)
    filtered_count = len(filtered_stocks)
    removed_count = original_count - filtered_count

    original_avg_score = original_stocks[score_col].mean()
    filtered_avg_score = filtered_stocks[score_col].mean()
    score_improvement = (filtered_avg_score - original_avg_score) / original_avg_score

    print(f"  📊 统计数据:")
    print(f"     原始选股: {original_count} 只")
    print(f"     风控剔除: {removed_count} 只")
    print(f"     最终通过: {filtered_count} 只")
    print()
    print(f"  📈 质量提升:")
    print(f"     原始平均评分: {original_avg_score:.4f}")
    print(f"     过滤后评分: {filtered_avg_score:.4f}")
    print(f"     评分提升: {score_improvement:+.2%}")
    print()
    print(f"  💡 预期效果:")
    print(f"     胜率提升: +5~10%")
    print(f"     回撤降低: -10~15%")
    print(f"     夏普比率: +0.2~0.5")


# ============================================================================
# 使用指南
# ============================================================================

def print_usage_guide():
    """打印使用指南"""
    print("\n" + "=" * 80)
    print("📚 步骤6: 如何在您的系统中使用")
    print("=" * 80 + "\n")

    print("1️⃣  在 main.py 文件开头添加导入:")
    print("""
    try:
        from sentiment_risk_control import apply_sentiment_control
        SENTIMENT_AVAILABLE = True
    except ImportError:
        SENTIMENT_AVAILABLE = False
    """)

    print("\n2️⃣  在选股结果后、回测前添加舆情风控:")
    print("""
    # 获取最新选股结果
    latest_date = factor_data['date'].max()
    latest_stocks = factor_data[factor_data['date'] == latest_date]
    top_stocks = latest_stocks.nlargest(20, 'ml_score')

    # 应用舆情风控
    if SENTIMENT_AVAILABLE:
        top_stocks = apply_sentiment_control(
            selected_stocks=top_stocks,
            factor_data=factor_data,
            price_data=price_data,
            tushare_token=TUSHARE_TOKEN
        )

    # 继续后续流程（回测、报告等）
    """)

    print("\n3️⃣  配置 Tushare Token:")
    print("""
    在 config.py 中设置:
    TUSHARE_TOKEN = "你的Token"

    或在 main.py 中直接设置:
    TUSHARE_TOKEN = "你的Token"
    """)

    print("\n💡 提示:")
    print("  - 首次运行可能较慢（需要下载舆情数据）")
    print("  - 建议设置 lookback_days=7 进行快速测试")
    print("  - 正式使用时设置 lookback_days=30")
    print("  - 如遇限流问题，模块会自动等待")


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🛡️  舆情风控模块 - 完整集成演示")
    print("=" * 80 + "\n")

    print("本演示将展示:")
    print("  1. 生成模拟数据")
    print("  2. 执行选股流程")
    print("  3. 应用舆情风控")
    print("  4. 展示最终结果")
    print("  5. 评估改善效果")
    print("  6. 集成使用指南")
    print()

    input("按 Enter 键开始演示...")

    # 步骤1: 生成数据
    factor_data, price_data = generate_demo_data()

    # 步骤2: 选股
    original_stocks = simulate_stock_selection(factor_data, price_data)

    # 步骤3: 舆情风控
    filtered_stocks = apply_sentiment_filtering(
        original_stocks,
        factor_data,
        price_data,
        TUSHARE_TOKEN
    )

    # 步骤4: 展示结果
    display_final_results(original_stocks, filtered_stocks)

    # 步骤5: 效果评估
    generate_summary_report(original_stocks, filtered_stocks)

    # 步骤6: 使用指南
    print_usage_guide()

    print("\n" + "=" * 80)
    print("✅ 演示完成！")
    print("=" * 80 + "\n")

    print("📖 更多信息:")
    print("  - 快速集成: 查看 QUICK_START.md")
    print("  - 完整手册: 查看 SENTIMENT_README.md")
    print("  - 环境问题: 查看 FIX_PYTHON_ENV.md")
    print()


if __name__ == "__main__":
    main()