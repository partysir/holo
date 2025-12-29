"""
data_module_incremental.py - 增量数据加载模块 (完整修复版 v2.7)
适配 main-2.py v2.8
"""
import pandas as pd
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# 导入必要的类和函数 (确保 data_module.py 在同一目录下)
try:
    from data_module import (
        TushareDataSource,
        StockRankerModel,
        calculate_simple_factors,
        RateLimiter
    )
except ImportError:
    print("❌ 错误：未找到 data_module.py。")
    raise

def load_data_with_incremental_update(
    start_date,
    end_date,
    max_stocks=50,
    cache_manager=None,
    use_stockranker=True,
    custom_weights=None,
    tushare_token=None,
    use_fundamental=True,
    force_full_update=False,
    use_sampling=True,
    sample_size=100,
    max_workers=4,
    min_days_listed=180,
    use_money_flow=True,
    money_flow_style='balanced'
):
    """
    增量更新数据加载函数 (完整实现 v2.7)
    """
    
    print("\n" + "=" * 80)
    print("📦 增量更新数据加载 (v2.7 适配版)")
    print("=" * 80)

    # 显示前视偏差防护配置
    print(f"\n🔒 前视偏差防护:")
    print(f"  - 最短上市时间: {min_days_listed} 天")
    print(f"  - 回测开始日期: {start_date}")

    # 计算最晚上市日期
    backtest_start = pd.to_datetime(start_date)
    latest_list_date = backtest_start - timedelta(days=min_days_listed)
    print(f"  - 要求上市于: {latest_list_date.strftime('%Y-%m-%d')} 之前")

    # 生成缓存键 (统一版本号为 v2.7，避免缓存冲突)
    model_suffix = "stockranker" if use_stockranker else "simple"
    if use_fundamental:
        model_suffix += "_fundamental"
    if use_money_flow:
        model_suffix += "_moneyflow"
        
    cache_key = f"factor_data_incr_v2.7_{start_date}_{end_date}_{max_stocks}_{model_suffix}_{min_days_listed}"
    price_cache_key = f"price_data_incr_v2.7_{start_date}_{end_date}_{max_stocks}_{min_days_listed}"

    # 1. 尝试从缓存加载
    if not force_full_update and cache_manager:
        print("\n🔍 检查缓存...")
        factor_data = cache_manager.load_from_csv(cache_key)
        price_data = cache_manager.load_from_csv(price_cache_key)

        if factor_data is not None and price_data is not None:
            print("✓ 使用缓存数据")
            print(f"  - 因子数据: {len(factor_data)} 条")
            print(f"  - 价格数据: {len(price_data)} 条")
            return factor_data, price_data
        else:
            print("✗ 缓存未找到或已过期，开始更新...")

    # 2. 初始化数据源
    rate_limiter = RateLimiter(max_calls=800, time_window=60)
    data_source = TushareDataSource(
        cache_manager=cache_manager,
        token=tushare_token,
        rate_limiter=rate_limiter
    )

    # 3. 获取股票列表
    print("\n📋 获取股票列表...")
    stock_list = data_source.get_stock_list(
        date=start_date,
        min_days_listed=min_days_listed
    )

    if not stock_list:
        print("✗ 无法获取股票列表!")
        return None, None

    # 采样处理
    if use_sampling:
        original_count = len(stock_list)
        stock_list = stock_list[:sample_size]
        print(f"  📊 采样模式: {len(stock_list)}/{original_count} 只股票")
    else:
        stock_list = stock_list[:max_stocks]
        print(f"  📊 完整模式: {len(stock_list)} 只股票")

    # 4. 获取股票上市日期信息 (用于过滤历史数据)
    print("\n📅 获取股票上市日期信息...")
    stock_info_df = data_source.pro.stock_basic(
        exchange='',
        list_status='L',
        fields='ts_code,list_date'
    )
    stock_info_dict = dict(zip(stock_info_df['ts_code'], stock_info_df['list_date']))

    # 5. 多线程获取价格数据
    print(f"\n📊 使用 {max_workers} 个线程并行获取数据...")
    all_price_data = []
    success_count = 0

    def fetch_price_data(ts_code):
        try:
            list_date = stock_info_dict.get(ts_code)
            # 调用 TushareDataSource 的方法
            df = data_source.get_price_data(
                ts_code,
                start_date,
                end_date,
                list_date=list_date
            )
            return ts_code, df
        except Exception as e:
            return ts_code, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_price_data, code): code for code in stock_list}

        for i, future in enumerate(as_completed(futures), 1):
            ts_code, df = future.result()
            if df is not None and not df.empty:
                all_price_data.append(df)
                success_count += 1
            if i % 50 == 0:
                print(f"  进度: {i}/{len(stock_list)} (成功: {success_count})")

    if not all_price_data:
        print("✗ 未获取到任何数据!")
        return None, None

    # 合并价格数据
    price_df = pd.concat(all_price_data, ignore_index=True)
    
    # 6. 获取并合并基本面数据
    if use_stockranker and use_fundamental:
        print(f"\n📈 获取基本面财务数据...")
        all_financial_data = []
        financial_success = 0
        
        def fetch_financial(ts_code):
            return data_source.get_financial_indicators(ts_code, start_date, end_date)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_financial, code) for code in price_df['instrument'].unique()]
            for future in as_completed(futures):
                res = future.result()
                if res is not None and not res.empty:
                    all_financial_data.append(res)
                    financial_success += 1
        
        print(f"  ✓ 获取到 {financial_success} 只股票的财务数据")
        if all_financial_data:
            financial_df = pd.concat(all_financial_data, ignore_index=True)
            price_df = data_source.merge_financial_data_to_daily(price_df, financial_df)

    price_df['date'] = price_df['date'].astype(str)

    # 7. 计算因子
    print("\n⚙️  计算因子...")
    if use_stockranker:
        model = StockRankerModel(
            custom_weights=custom_weights,
            use_fundamental=use_fundamental,
            use_money_flow=use_money_flow,
            money_flow_style=money_flow_style
        )
        factor_df = model.calculate_all_factors(price_df)
        factor_df = model.calculate_position_score(factor_df)
    else:
        factor_df = calculate_simple_factors(price_df)

    # 8. 清理与列筛选
    if 'position' in factor_df.columns:
        factor_df = factor_df[factor_df['position'].notna()]

    essential_columns = ['date', 'instrument', 'position']
    price_only_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
    
    # 智能识别所有因子列 (排除必须列和价格列)
    all_cols = factor_df.columns.tolist()
    factor_cols = [c for c in all_cols if c not in essential_columns + price_only_columns]
    
    result_factor = factor_df[essential_columns + factor_cols].copy()
    
    # 价格数据保留列
    price_cols_keep = essential_columns + price_only_columns
    if use_fundamental:
        for c in ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']:
            if c in price_df.columns: price_cols_keep.append(c)
    
    price_cols_keep = [c for c in price_cols_keep if c in price_df.columns]
    result_price = price_df[price_cols_keep].copy()

    # 9. 补全行业数据
    print("\n🏭 补全行业数据...")
    industry_data = data_source.get_industry_data(price_df['instrument'].unique().tolist(), use_cache=True)
    if industry_data is not None:
        result_factor = result_factor.merge(industry_data, on='instrument', how='left')
        result_factor['industry'] = result_factor['industry'].fillna('其他')

    # 10. 保存缓存
    if cache_manager:
        print("\n💾 保存到缓存...")
        cache_manager.save_to_csv(result_factor, cache_key)
        cache_manager.save_to_csv(result_price, price_cache_key)

    print("✓ 数据准备完成")
    return result_factor, result_price