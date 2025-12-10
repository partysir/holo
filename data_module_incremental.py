"""
data_module_incremental.py - 增量更新模块修复版 v2.3

关键修复：
✅ 添加 min_days_listed 参数传递
✅ 在获取股票列表时过滤新股
✅ 在获取价格数据时过滤上市前数据
"""

import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 导入修复后的数据模块
from data_module import (
    DataCache,
    TushareDataSource,
    StockRankerModel,
    calculate_simple_factors
)


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
    min_days_listed=180  # ✅ 关键新增参数
):
    """
    增量更新数据加载函数 (修复版 v2.3)

    新增参数:
        min_days_listed: 股票最少上市天数，默认180天
    """

    print("\n" + "=" * 80)
    print("📦 增量更新数据加载 (v2.3 - 修复前视偏差)")
    print("=" * 80)

    # 显示前视偏差防护配置
    print(f"\n🔒 前视偏差防护:")
    print(f"  - 最短上市时间: {min_days_listed} 天")
    print(f"  - 回测开始日期: {start_date}")

    # 计算最晚上市日期
    backtest_start = pd.to_datetime(start_date)
    latest_list_date = backtest_start - timedelta(days=min_days_listed)
    print(f"  - 要求上市于: {latest_list_date.strftime('%Y-%m-%d')} 之前")

    model_type = "StockRanker多因子" if use_stockranker else "简单技术因子"
    if use_stockranker and use_fundamental:
        model_type += " + 基本面"
    print(f"  - 因子模型: {model_type}")

    # 生成缓存键（包含版本号和min_days_listed）
    model_suffix = "stockranker" if use_stockranker else "simple"
    if use_fundamental:
        model_suffix += "_fundamental"

    cache_key = f"factor_data_incr_v2.3_{start_date}_{end_date}_{max_stocks}_{model_suffix}_{min_days_listed}"
    price_cache_key = f"price_data_incr_v2.3_{start_date}_{end_date}_{max_stocks}_{min_days_listed}"

    # 尝试从缓存加载
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
            print("✗ 缓存未找到，开始增量更新...")

    # 初始化数据源
    from data_module import RateLimiter
    rate_limiter = RateLimiter(max_calls=800, time_window=60)
    data_source = TushareDataSource(
        cache_manager=cache_manager,
        token=tushare_token,
        rate_limiter=rate_limiter
    )

    # ========== 关键修复1：获取股票列表时传入日期 ==========
    print("\n📋 获取股票列表...")
    stock_list = data_source.get_stock_list(
        date=start_date,              # ✅ 传入回测开始日期
        min_days_listed=min_days_listed  # ✅ 传入最短上市天数
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

    # ========== 关键修复2：获取股票上市日期信息 ==========
    print("\n📅 获取股票上市日期信息...")
    stock_info_df = data_source.pro.stock_basic(
        exchange='',
        list_status='L',
        fields='ts_code,list_date'
    )
    stock_info_dict = dict(zip(stock_info_df['ts_code'], stock_info_df['list_date']))
    print(f"  ✓ 获取到 {len(stock_info_dict)} 只股票的上市日期")

    # ========== 多线程获取价格数据 ==========
    print(f"\n📊 使用 {max_workers} 个线程并行获取数据...")
    all_price_data = []
    success_count = 0
    failed_stocks = []

    def fetch_price_data(ts_code):
        """获取单只股票数据（带上市日期过滤）"""
        try:
            list_date = stock_info_dict.get(ts_code)  # ✅ 获取上市日期
            df = data_source.get_price_data(
                ts_code,
                start_date,
                end_date,
                list_date=list_date  # ✅ 传入上市日期进行过滤
            )
            return ts_code, df
        except Exception as e:
            print(f"  ✗ {ts_code} 失败: {e}")
            return ts_code, None

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_price_data, code): code for code in stock_list}

        for i, future in enumerate(as_completed(futures), 1):
            ts_code, df = future.result()

            if df is not None and len(df) > 0:
                all_price_data.append(df)
                success_count += 1
            else:
                failed_stocks.append(ts_code)

            # 进度显示
            if i % 50 == 0:
                print(f"  进度: {i}/{len(stock_list)} (成功: {success_count})")

    print(f"\n✓ 价格数据获取完成:")
    print(f"  - 成功: {success_count}/{len(stock_list)} 只")
    if failed_stocks:
        print(f"  - 失败: {len(failed_stocks)} 只")
        print(f"    示例: {failed_stocks[:5]}")

    if len(all_price_data) == 0:
        print("✗ 未获取到任何数据!")
        return None, None

    # 合并价格数据
    price_df = pd.concat(all_price_data, ignore_index=True)
    print(f"  - 总记录数: {len(price_df)} 条")

    # ========== 验证：检查是否还有新股 ==========
    print("\n🔍 数据质量验证:")
    unique_stocks = price_df['instrument'].unique()
    print(f"  - 实际股票数: {len(unique_stocks)} 只")

    # 检查北交所新股（920开头）和科创板新股（689开头）
    new_stock_codes = [s for s in unique_stocks if s.startswith(('920', '689', '787'))]
    if new_stock_codes:
        print(f"  ⚠️  警告：仍发现 {len(new_stock_codes)} 只可疑新股代码")
        print(f"     示例: {new_stock_codes[:5]}")
        print(f"  ⚠️  建议：增大 min_days_listed 参数或检查 get_stock_list 过滤逻辑")
    else:
        print(f"  ✅ 通过：未发现可疑新股代码")

    # ========== 获取基本面数据 ==========
    if use_stockranker and use_fundamental:
        print(f"\n📈 获取基本面财务数据 (并行模式)...")
        all_financial_data = []
        financial_success = 0

        def fetch_financial_data(ts_code):
            """获取单只股票财务数据"""
            try:
                df = data_source.get_financial_indicators(ts_code, start_date, end_date)
                return ts_code, df
            except Exception as e:
                return ts_code, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_financial_data, code): code
                      for code in unique_stocks}

            for i, future in enumerate(as_completed(futures), 1):
                ts_code, df = future.result()

                if df is not None and len(df) > 0:
                    all_financial_data.append(df)
                    financial_success += 1

                if i % 50 == 0:
                    print(f"  进度: {i}/{len(unique_stocks)} (成功: {financial_success})")

        print(f"✓ 财务数据获取完成: {financial_success}/{len(unique_stocks)} 只")

        if len(all_financial_data) > 0:
            financial_df = pd.concat(all_financial_data, ignore_index=True)
            print("\n🔗 合并基本面数据到日线数据...")
            price_df = data_source.merge_financial_data_to_daily(price_df, financial_df)
        else:
            print("⚠️  未获取到基本面数据，将不使用基本面因子")
            use_fundamental = False

    # 确保日期格式一致
    price_df['date'] = price_df['date'].astype(str)

    # ========== 因子计算 ==========
    if use_stockranker:
        model = StockRankerModel(
            custom_weights=custom_weights,
            use_fundamental=use_fundamental
        )
        factor_df = model.calculate_all_factors(price_df)
        factor_df = model.calculate_position_score(factor_df)
    else:
        print("\n⚙️  计算简单技术因子...")
        factor_df = calculate_simple_factors(price_df)

    factor_df = factor_df.dropna(subset=['position'])

    # ========== 关键修复：保留所有因子列 ==========
    essential_columns = ['date', 'instrument', 'position']
    price_only_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']

    all_columns = factor_df.columns.tolist()
    factor_columns = [col for col in all_columns
                     if col not in essential_columns + price_only_columns]

    print(f"\n📊 因子列识别:")
    print(f"  - 必须列: {essential_columns}")
    print(f"  - 识别到的因子列: {len(factor_columns)} 个")
    if len(factor_columns) <= 10:
        print(f"    {factor_columns}")
    else:
        print(f"    前10个: {factor_columns[:10]}")
        print(f"    ... 还有 {len(factor_columns)-10} 个")

    # 保留因子列
    columns_to_keep = essential_columns + factor_columns
    result_factor = factor_df[columns_to_keep].copy()

    # 保留价格列
    price_columns_to_keep = essential_columns + price_only_columns
    if use_fundamental:
        fundamental_cols = ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']
        for col in fundamental_cols:
            if col in price_df.columns:
                price_columns_to_keep.append(col)

    result_price = price_df[price_columns_to_keep].copy()

    # ========== 获取行业数据 ==========
    print("\n📊 获取行业数据...")
    industry_data = data_source.get_industry_data(unique_stocks.tolist(), use_cache=True)

    if industry_data is not None and len(industry_data) > 0:
        result_factor = result_factor.merge(industry_data, on='instrument', how='left')
        result_factor['industry'] = result_factor['industry'].fillna('其他')
        print(f"  ✓ 行业数据已合并")
    else:
        result_factor['industry'] = 'Unknown'
        print(f"  ⚠️  未获取到行业数据")

    # ========== 保存到缓存 ==========
    if cache_manager:
        print("\n💾 保存到缓存...")
        cache_manager.save_to_csv(result_factor, cache_key)
        cache_manager.save_to_csv(result_price, price_cache_key)

    # ========== 最终统计 ==========
    print(f"\n✓ 数据准备完成:")
    print(f"  - 因子数据: {len(result_factor)} 条")
    print(f"  - 价格数据: {len(result_price)} 条")
    print(f"  - 股票数量: {result_factor['instrument'].nunique()} 只")
    print(f"  - 交易日数: {result_factor['date'].nunique()} 天")
    print(f"  - 因子列数: {len(factor_columns)} 个")  # ✅ 显示因子数量
    print(f"  - 行业数量: {result_factor['industry'].nunique()} 个")

    if use_fundamental and use_stockranker:
        print(f"  - 基本面因子: 已启用")

    return result_factor, result_price