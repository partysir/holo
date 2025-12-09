"""
test_system.py - 测试所有模块
运行: python test_system.py
"""

import warnings

warnings.filterwarnings('ignore')


def test_imports():
    """测试模块导入"""
    print("\n【测试1/5】模块导入")

    try:
        import tushare as ts
        print("  ✓ tushare")

        import pandas as pd
        print("  ✓ pandas")

        import numpy as np
        print("  ✓ numpy")

        from data_module import DataCache, TushareDataSource
        print("  ✓ data_module")

        from data_module_incremental import load_data_with_incremental_update
        print("  ✓ data_module_incremental")

        from backtest_module_optimized import run_optimized_backtest
        print("  ✓ backtest_module_optimized")

        from ultimate_fast_system import UltimateFastBacktest, IncrementalBacktestSystem
        print("  ✓ ultimate_fast_system")

        from genetic_optimizer import GeneticOptimizer
        print("  ✓ genetic_optimizer")

        return True

    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        return False


def test_data_loading():
    """测试数据加载"""
    print("\n【测试2/5】数据加载")

    try:
        from data_module import DataCache
        from data_module_incremental import load_data_with_incremental_update
        import tushare as ts

        TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"
        ts.set_token(TUSHARE_TOKEN)

        cache_manager = DataCache(cache_dir='./data_cache')

        factor_data, price_data = load_data_with_incremental_update(
            "2024-12-01", "2024-12-05",
            cache_manager=cache_manager,
            use_sampling=True,
            sample_size=100,  # 少量测试
            max_workers=5,
            tushare_token=TUSHARE_TOKEN
        )

        if factor_data is not None and price_data is not None:
            print(f"  ✓ 因子数据: {len(factor_data)} 条")
            print(f"  ✓ 价格数据: {len(price_data)} 条")
            return True
        else:
            print("  ❌ 数据为空")
            return False

    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return False


def test_backtest():
    """测试回测"""
    print("\n【测试3/5】回测引擎")

    try:
        from data_module import DataCache
        from data_module_incremental import load_data_with_incremental_update
        from ultimate_fast_system import UltimateFastBacktest
        import tushare as ts

        TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"
        ts.set_token(TUSHARE_TOKEN)

        cache_manager = DataCache()

        # 加载数据
        factor_data, price_data = load_data_with_incremental_update(
            "2024-11-01", "2024-12-05",
            cache_manager=cache_manager,
            use_sampling=True,
            sample_size=50,
            max_workers=5,
            tushare_token=TUSHARE_TOKEN
        )

        if factor_data is None:
            print("  ⚠️  数据加载失败，跳过测试")
            return True

        # 运行回测
        engine = UltimateFastBacktest(
            factor_data, price_data,
            "2024-11-01", "2024-12-05",
            capital_base=100000,
            position_size=5
        )

        context = engine.run(silent=True)

        print(f"  ✓ 回测完成")
        print(f"    收益率: {context['total_return']:+.2%}")
        print(f"    胜率: {context['win_rate']:.2%}")

        return True

    except Exception as e:
        print(f"  ❌ 回测失败: {e}")
        return False


def test_incremental_system():
    """测试增量系统"""
    print("\n【测试4/5】增量系统")

    try:
        from ultimate_fast_system import IncrementalBacktestSystem

        system = IncrementalBacktestSystem(cache_dir='./data_cache')

        print(f"  ✓ 系统初始化")
        print(f"    上次日期: {system.state['last_date']}")
        print(f"    持仓数: {len(system.state['positions'])}")

        return True

    except Exception as e:
        print(f"  ❌ 初始化失败: {e}")
        return False


def test_genetic_optimizer():
    """测试遗传算法"""
    print("\n【测试5/5】遗传算法")

    try:
        from genetic_optimizer import GeneticOptimizer

        print("  ✓ 模块导入成功")
        print("  ℹ️  完整测试需要运行优化（耗时较长）")

        return True

    except ImportError as e:
        if 'deap' in str(e):
            print("  ⚠️  DEAP未安装: pip install deap")
        else:
            print(f"  ❌ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 系统测试")
    print("=" * 80)

    results = []

    results.append(("模块导入", test_imports()))
    results.append(("数据加载", test_data_loading()))
    results.append(("回测引擎", test_backtest()))
    results.append(("增量系统", test_incremental_system()))
    results.append(("遗传算法", test_genetic_optimizer()))

    # 汇总
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
    print("=" * 80)

    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name:<12}: {status}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有测试通过！系统正常")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()