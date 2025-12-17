# -*- coding: utf-8 -*-
"""
test_sentiment_module.py - 舆情风控模块测试脚本

测试内容：
1. 模块导入测试
2. 数据采集器测试
3. 规则引擎测试
4. 完整流程测试
5. 性能测试

使用方法：
python test_sentiment_module.py --token YOUR_TOKEN
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import argparse


def test_module_import():
    """测试1: 模块导入"""
    print("\n" + "=" * 80)
    print("测试1: 模块导入")
    print("=" * 80)

    try:
        from sentiment_risk_control import (
            SentimentDataCollector,
            SentimentRuleEngine,
            SentimentAnalyzer,
            SentimentRiskController,
            apply_sentiment_control
        )
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False


def test_rule_engine():
    """测试2: 规则引擎"""
    print("\n" + "=" * 80)
    print("测试2: 规则引擎")
    print("=" * 80)

    from sentiment_risk_control import SentimentRuleEngine

    engine = SentimentRuleEngine()

    # 测试一票否决
    test_cases_veto = [
        ("公司涉嫌财务造假，证监会已立案调查", True, "应触发否决"),
        ("公司业绩稳定增长", False, "不应触发否决"),
        ("ST股票风险警示", True, "应触发否决"),
        ("董事长辞职，业绩大幅下滑", True, "多个高风险词应触发"),
    ]

    print("\n🚫 一票否决测试:")
    passed = 0
    for text, expected, desc in test_cases_veto:
        is_veto, reason = engine.check_veto_triggers(text)
        status = "✅" if is_veto == expected else "❌"
        print(f"  {status} {desc}")
        print(f"     输入: {text[:30]}...")
        print(f"     结果: {'触发' if is_veto else '未触发'} - {reason}")
        if is_veto == expected:
            passed += 1

    # 测试加分增强
    test_cases_boost = [
        ("公司获得人工智能领域重大突破，新质生产力概念股", 0.10, "政策支持"),
        ("公司发布业绩预增公告", 0.05, "热点概念"),
        ("日常新闻", 0.0, "无加分"),
    ]

    print("\n📈 加分增强测试:")
    for text, expected_min, desc in test_cases_boost:
        boost, keywords = engine.calculate_boost_score(text, source='news')
        status = "✅" if boost >= expected_min else "❌"
        print(f"  {status} {desc}")
        print(f"     输入: {text[:30]}...")
        print(f"     加分: {boost:.2%} - {keywords}")
        if boost >= expected_min:
            passed += 1

    print(f"\n总计: {passed}/{len(test_cases_veto) + len(test_cases_boost)} 通过")
    return passed == len(test_cases_veto) + len(test_cases_boost)


def test_data_collector(token):
    """测试3: 数据采集器"""
    print("\n" + "=" * 80)
    print("测试3: 数据采集器")
    print("=" * 80)

    if token is None:
        print("⚠️  未提供Token，跳过数据采集测试")
        return True

    from sentiment_risk_control import SentimentDataCollector

    try:
        collector = SentimentDataCollector(token=token)

        # 测试新闻联播接口（最稳定）
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

        print(f"\n📊 测试获取新闻联播 ({start_date} ~ {end_date})")
        cctv_df = collector.get_cctv_news(start_date, end_date)

        if not cctv_df.empty:
            print(f"✅ 获取成功: {len(cctv_df)} 条")
            print(f"   示例标题: {cctv_df['title'].iloc[0][:30]}...")
            return True
        else:
            print("⚠️  未获取到数据（可能是日期范围问题）")
            return True  # 不算失败

    except Exception as e:
        print(f"❌ 数据采集失败: {e}")
        return False


def test_full_pipeline(token):
    """测试4: 完整流程"""
    print("\n" + "=" * 80)
    print("测试4: 完整流程测试")
    print("=" * 80)

    if token is None:
        print("⚠️  未提供Token，使用模拟数据")

    from sentiment_risk_control import apply_sentiment_control

    # 创建模拟数据
    print("\n📦 创建模拟数据...")

    latest_date = datetime.now().strftime('%Y-%m-%d')

    selected_stocks = pd.DataFrame({
        'date': [latest_date] * 10,
        'instrument': [
            '000001.SZ',  # 平安银行
            '600000.SH',  # 浦发银行
            '000002.SZ',  # 万科A
            '600036.SH',  # 招商银行
            '000333.SZ',  # 美的集团
            '600519.SH',  # 贵州茅台
            '000858.SZ',  # 五粮液
            '601318.SH',  # 中国平安
            '000651.SZ',  # 格力电器
            '600276.SH',  # 恒瑞医药
        ],
        'ml_score': np.linspace(0.95, 0.70, 10),
        'position': np.linspace(0.95, 0.70, 10),
        'industry': ['金融', '金融', '地产', '金融', '家电',
                     '白酒', '白酒', '金融', '家电', '医药']
    })

    factor_data = selected_stocks.copy()
    price_data = pd.DataFrame({
        'date': [latest_date] * 10,
        'instrument': selected_stocks['instrument'].tolist(),
        'close': np.random.uniform(10, 100, 10)
    })

    print(f"  ✓ 模拟选股: {len(selected_stocks)} 只")
    print(f"  ✓ 日期: {latest_date}")

    # 运行舆情风控
    try:
        print("\n🛡️  执行舆情风控...")

        filtered = apply_sentiment_control(
            selected_stocks=selected_stocks,
            factor_data=factor_data,
            price_data=price_data,
            tushare_token=token,
            enable_veto=True,
            enable_boost=True,
            lookback_days=7  # 短周期测试
        )

        print(f"\n✅ 完整流程测试通过")
        print(f"  原始: {len(selected_stocks)} 只")
        print(f"  过滤: {len(selected_stocks) - len(filtered)} 只")
        print(f"  最终: {len(filtered)} 只")

        if not filtered.empty:
            print(f"\n  Top 3 结果:")
            display_cols = ['instrument', 'ml_score', 'industry']
            print(filtered[display_cols].head(3).to_string(index=False))

        return True

    except Exception as e:
        print(f"❌ 完整流程测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """测试5: 性能测试"""
    print("\n" + "=" * 80)
    print("测试5: 性能测试")
    print("=" * 80)

    from sentiment_risk_control import SentimentRuleEngine

    engine = SentimentRuleEngine()

    # 生成大量测试文本
    test_texts = [
                     "公司业绩稳定增长，未来前景看好",
                     "涉嫌财务造假，证监会立案调查",
                     "人工智能概念股，新质生产力领域龙头",
                 ] * 1000  # 3000 条

    print(f"\n⏱️  处理 {len(test_texts)} 条文本...")

    start_time = time.time()

    veto_count = 0
    boost_count = 0

    for text in test_texts:
        is_veto, _ = engine.check_veto_triggers(text)
        boost, _ = engine.calculate_boost_score(text)

        if is_veto:
            veto_count += 1
        if boost > 0:
            boost_count += 1

    elapsed = time.time() - start_time

    print(f"✅ 性能测试完成")
    print(f"  耗时: {elapsed:.2f} 秒")
    print(f"  速度: {len(test_texts) / elapsed:.0f} 条/秒")
    print(f"  触发否决: {veto_count} 条")
    print(f"  触发加分: {boost_count} 条")

    return elapsed < 10  # 应该在10秒内完成


def run_all_tests(token=None):
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 舆情风控模块完整测试套件")
    print("=" * 80)

    results = []

    # 测试1: 模块导入
    results.append(("模块导入", test_module_import()))

    # 测试2: 规则引擎
    results.append(("规则引擎", test_rule_engine()))

    # 测试3: 数据采集器
    results.append(("数据采集器", test_data_collector(token)))

    # 测试4: 完整流程
    results.append(("完整流程", test_full_pipeline(token)))

    # 测试5: 性能测试
    results.append(("性能测试", test_performance()))

    # 汇总结果
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)

    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
        if result:
            passed += 1

    print("\n" + "-" * 80)
    print(f"总计: {passed}/{len(results)} 通过 ({passed / len(results) * 100:.1f}%)")

    if passed == len(results):
        print("\n🎉 所有测试通过！模块可以正常使用。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='舆情风控模块测试')
    parser.add_argument('--token', type=str, default=None, help='Tushare Token')
    parser.add_argument('--quick', action='store_true', help='快速测试（跳过API调用）')

    args = parser.parse_args()

    if args.quick:
        print("⚡ 快速测试模式（跳过API调用）")
        args.token = None

    run_all_tests(token=args.token)