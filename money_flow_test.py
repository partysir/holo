"""
修改main.py ,main-2.py调用示例 - 内存优化版

修改要点：
1. 传入 use_money_flow 和 money_flow_style 参数
2. 控制最大股票数（避免内存爆炸）
3. 使用采样模式加速测试
"""

from config import *
from data_module_incremental import load_data_with_incremental_update
from data_module import DataCache

# ========== 方案1：保守测试（推荐首次使用）==========

def test_money_flow_conservative():
    """保守测试：少量股票 + 短时间"""
    
    print("\n" + "="*80)
    print("📊 资金流因子测试 - 保守模式")
    print("="*80)
    
    # 初始化缓存
    cache = DataCache(cache_dir='./data_cache')
    
    # 测试参数（缩小规模）
    test_config = {
        'start_date': '2024-01-01',      # 仅测试1年
        'end_date': '2024-12-31',
        'max_stocks': 100,                # 仅100只股票
        'use_sampling': True,
        'sample_size': 100,
        'max_workers': 4,
        'min_days_listed': 180,
        
        # ✅ 资金流配置
        'use_money_flow': True,
        'money_flow_style': 'balanced',   # 'conservative' | 'balanced' | 'aggressive'
        
        # 其他配置
        'use_stockranker': True,
        'use_fundamental': True,
        'cache_manager': cache,
        'tushare_token': TUSHARE_TOKEN,
        'force_full_update': False,
    }
    
    print("\n测试配置:")
    print(f"  时间范围: {test_config['start_date']} 至 {test_config['end_date']}")
    print(f"  股票数量: {test_config['sample_size']}")
    print(f"  资金流因子: {'✓' if test_config['use_money_flow'] else '✗'}")
    print(f"  资金流风格: {test_config['money_flow_style']}")
    
    # 加载数据
    try:
        factor_data, price_data = load_data_with_incremental_update(**test_config)
        
        if factor_data is not None:
            print("\n✅ 数据加载成功！")
            print(f"  因子数据: {len(factor_data)} 行")
            print(f"  价格数据: {len(price_data)} 行")
            print(f"  因子列数: {len(factor_data.columns)} 个")
            print(f"  内存占用: {factor_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            
            # 查看资金流因子
            money_flow_cols = [c for c in factor_data.columns if 'main_' in c or 'large_' in c]
            print(f"\n  资金流因子列表 ({len(money_flow_cols)} 个):")
            for col in money_flow_cols[:10]:
                print(f"    - {col}")
            if len(money_flow_cols) > 10:
                print(f"    ... 还有 {len(money_flow_cols) - 10} 个")
            
            return factor_data, price_data
        else:
            print("\n❌ 数据加载失败")
            return None, None
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ========== 方案2：完整回测（确认无内存问题后使用）==========

def run_full_backtest_with_money_flow():
    """完整回测（包含资金流因子）"""
    
    print("\n" + "="*80)
    print("📊 完整回测 - 资金流增强版")
    print("="*80)
    
    cache = DataCache(cache_dir='./data_cache')
    
    # 完整配置
    backtest_config = {
        'start_date': BacktestConfig.START_DATE,
        'end_date': BacktestConfig.END_DATE,
        'max_stocks': DataConfig.MAX_STOCKS,
        'use_sampling': DataConfig.USE_SAMPLING,
        'sample_size': DataConfig.SAMPLE_SIZE,
        'max_workers': DataConfig.MAX_WORKERS,
        'min_days_listed': 180,
        
        # ✅ 资金流配置
        'use_money_flow': FactorConfig.USE_MONEY_FLOW,
        'money_flow_style': 'balanced',  # 从config读取
        
        # 其他配置
        'use_stockranker': FactorConfig.USE_STOCKRANKER,
        'use_fundamental': FactorConfig.USE_FUNDAMENTAL,
        'cache_manager': cache,
        'tushare_token': TUSHARE_TOKEN,
        'force_full_update': DataConfig.FORCE_FULL_UPDATE,
    }
    
    # 加载数据
    factor_data, price_data = load_data_with_incremental_update(**backtest_config)
    
    if factor_data is None or price_data is None:
        print("\n❌ 数据加载失败！")
        return
    
    print("\n✅ 数据准备完成，开始回测...")
    
    # 导入回测模块
    from backtest_engine import BacktestEngine
    
    # 创建回测引擎
    engine = BacktestEngine(
        factor_data=factor_data,
        price_data=price_data,
        **get_strategy_params()
    )
    
    # 运行回测
    context = engine.run()
    
    # 生成报告
    print("\n📊 生成回测报告...")
    from visualization_module import (
        plot_monitoring_results,
        plot_top_stocks_evolution,
        generate_performance_report
    )
    
    plot_monitoring_results(context)
    plot_top_stocks_evolution(context)
    generate_performance_report(context)
    
    print("\n✅ 回测完成！")

# ========== 方案3：对比测试（资金流 vs 无资金流）==========

def compare_with_without_money_flow():
    """对比测试：评估资金流因子的增益"""
    
    print("\n" + "="*80)
    print("📊 对比测试：资金流增益分析")
    print("="*80)
    
    cache = DataCache(cache_dir='./data_cache')
    
    test_config_base = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'max_stocks': 100,
        'use_sampling': True,
        'sample_size': 100,
        'max_workers': 4,
        'min_days_listed': 180,
        'use_stockranker': True,
        'use_fundamental': True,
        'cache_manager': cache,
        'tushare_token': TUSHARE_TOKEN,
        'force_full_update': False,
    }
    
    results = {}
    
    # 测试1：不使用资金流
    print("\n【测试1】不使用资金流因子...")
    config_no_mf = {**test_config_base, 'use_money_flow': False}
    try:
        factor_data_no_mf, price_data_no_mf = load_data_with_incremental_update(**config_no_mf)
        if factor_data_no_mf is not None:
            results['no_money_flow'] = {
                'factor_count': len(factor_data_no_mf.columns),
                'memory_mb': factor_data_no_mf.memory_usage(deep=True).sum() / 1024**2,
            }
            print(f"  ✓ 因子数: {results['no_money_flow']['factor_count']}")
            print(f"  ✓ 内存: {results['no_money_flow']['memory_mb']:.1f} MB")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 测试2：使用资金流（保守）
    print("\n【测试2】使用资金流因子（保守风格）...")
    config_mf_conservative = {**test_config_base, 'use_money_flow': True, 'money_flow_style': 'conservative'}
    try:
        factor_data_mf_c, price_data_mf_c = load_data_with_incremental_update(**config_mf_conservative)
        if factor_data_mf_c is not None:
            results['money_flow_conservative'] = {
                'factor_count': len(factor_data_mf_c.columns),
                'memory_mb': factor_data_mf_c.memory_usage(deep=True).sum() / 1024**2,
            }
            print(f"  ✓ 因子数: {results['money_flow_conservative']['factor_count']}")
            print(f"  ✓ 内存: {results['money_flow_conservative']['memory_mb']:.1f} MB")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 测试3：使用资金流（激进）
    print("\n【测试3】使用资金流因子（激进风格）...")
    config_mf_aggressive = {**test_config_base, 'use_money_flow': True, 'money_flow_style': 'aggressive'}
    try:
        factor_data_mf_a, price_data_mf_a = load_data_with_incremental_update(**config_mf_aggressive)
        if factor_data_mf_a is not None:
            results['money_flow_aggressive'] = {
                'factor_count': len(factor_data_mf_a.columns),
                'memory_mb': factor_data_mf_a.memory_usage(deep=True).sum() / 1024**2,
            }
            print(f"  ✓ 因子数: {results['money_flow_aggressive']['factor_count']}")
            print(f"  ✓ 内存: {results['money_flow_aggressive']['memory_mb']:.1f} MB")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
    
    # 汇总对比
    print("\n" + "="*80)
    print("📊 对比结果汇总")
    print("="*80)
    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  因子数: {stats['factor_count']}")
        print(f"  内存占用: {stats['memory_mb']:.1f} MB")

# ========== 主函数 ==========

if __name__ == "__main__":
    
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = 'test'  # 默认测试模式
    
    if mode == 'test':
        # 保守测试（推荐首次使用）
        print("🧪 运行模式: 保守测试")
        test_money_flow_conservative()
    
    elif mode == 'full':
        # 完整回测
        print("🚀 运行模式: 完整回测")
        run_full_backtest_with_money_flow()
    
    elif mode == 'compare':
        # 对比测试
        print("📊 运行模式: 对比测试")
        compare_with_without_money_flow()
    
    else:
        print(f"❌ 未知模式: {mode}")
        print("\n可用模式:")
        print("  python money_flow_test.py test      # 保守测试（推荐）")
        print("  python money_flow_test.py full      # 完整回测")
        print("  python money_flow_test.py compare   # 对比测试")