"""
data_module_incremental.py - 修复版
✅ 修复：保留所有因子列，不只是position

关键修复：
- 第366行：保留所有因子列供机器学习使用
- 增加因子列统计输出
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# ============ API限流器 ============

class ThreadSafeRateLimiter:
    """线程安全的API限流器"""

    def __init__(self, max_calls_per_minute=800):
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = threading.Lock()

    def acquire(self):
        """获取调用许可"""
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]

            if len(self.calls) >= self.max_calls:
                sleep_time = 60 - (now - self.calls[0]) + 0.1
                time.sleep(sleep_time)
                self.calls = []

            self.calls.append(time.time())


# ============ 智能股票抽样器 ============

class SmartStockSampler:
    """智能股票抽样器 - 按市值分层抽样"""

    def __init__(self, data_source):
        self.data_source = data_source

    def get_stratified_sample(self, stock_list, sample_size=800):
        print(f"\n  🎯 智能抽样: 从 {len(stock_list)} 只中选择 {sample_size} 只...")

        if len(stock_list) <= sample_size:
            print(f"  ℹ️  股票数量不足，使用全部 {len(stock_list)} 只")
            return stock_list

        try:
            pro = self.data_source.pro
            stock_info = pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,name,total_mv'
            )

            stock_info = stock_info[stock_info['ts_code'].isin(stock_list)]
            stock_info = stock_info.dropna(subset=['total_mv'])

            if len(stock_info) == 0:
                print(f"  ⚠️  无法获取市值信息，使用随机抽样")
                import random
                return random.sample(stock_list, sample_size)

            stock_info = stock_info.sort_values('total_mv', ascending=False)
            total_count = len(stock_info)

            large_cap = stock_info.head(int(total_count * 0.2))
            mid_cap = stock_info.iloc[int(total_count * 0.2):int(total_count * 0.8)]
            small_cap = stock_info.tail(int(total_count * 0.2))

            n_large = int(sample_size * 0.4)
            n_mid = int(sample_size * 0.4)
            n_small = sample_size - n_large - n_mid

            sampled = pd.concat([
                large_cap.sample(n=min(n_large, len(large_cap)), random_state=42),
                mid_cap.sample(n=min(n_mid, len(mid_cap)), random_state=42),
                small_cap.sample(n=min(n_small, len(small_cap)), random_state=42)
            ])

            selected = sampled['ts_code'].tolist()

            print(f"  ✓ 抽样完成: 大盘 {n_large}只 | 中盘 {n_mid}只 | 小盘 {n_small}只")
            return selected

        except Exception as e:
            print(f"  ⚠️  智能抽样失败: {e}")
            print(f"  使用随机抽样...")
            import random
            return random.sample(stock_list, min(sample_size, len(stock_list)))


# ============ 并行数据获取器 ============

class ParallelDataFetcher:
    """多线程并行数据获取器"""

    def __init__(self, data_source, max_workers=10, rate_limiter=None):
        self.data_source = data_source
        self.max_workers = max_workers
        self.rate_limiter = rate_limiter or ThreadSafeRateLimiter(max_calls_per_minute=800)

    def fetch_price_data_parallel(self, stock_list, start_date, end_date):
        print(f"\n  🚀 多线程获取价格数据 ({self.max_workers}线程)...")

        all_data = []
        success_count = 0
        fail_count = 0

        def fetch_one(ts_code):
            try:
                self.rate_limiter.acquire()
                df = self.data_source.get_price_data(ts_code, start_date, end_date)
                if df is not None and len(df) > 0:
                    return ('success', df)
                return ('fail', None)
            except Exception as e:
                return ('fail', None)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fetch_one, stock): stock for stock in stock_list}

            for i, future in enumerate(as_completed(futures)):
                status, data = future.result()

                if status == 'success':
                    all_data.append(data)
                    success_count += 1
                else:
                    fail_count += 1

                if (i + 1) % 100 == 0 or i == len(stock_list) - 1:
                    progress = (i + 1) / len(stock_list) * 100
                    print(f"    进度: {i + 1}/{len(stock_list)} ({progress:.1f}%) | "
                          f"成功: {success_count} | 失败: {fail_count}")

        print(f"  ✓ 成功获取 {success_count}/{len(stock_list)} 只股票")

        if len(all_data) == 0:
            return None

        return pd.concat(all_data, ignore_index=True)

    def fetch_financial_data_parallel(self, stock_list, start_date, end_date):
        print(f"\n  🚀 多线程获取基本面数据 ({self.max_workers}线程)...")

        all_data = []
        success_count = 0

        def fetch_one(ts_code):
            try:
                self.rate_limiter.acquire()
                df = self.data_source.get_financial_indicators(ts_code, start_date, end_date)
                if df is not None and len(df) > 0:
                    return ('success', df)
                return ('fail', None)
            except Exception as e:
                return ('fail', None)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(fetch_one, stock): stock for stock in stock_list}

            for i, future in enumerate(as_completed(futures)):
                status, data = future.result()

                if status == 'success':
                    all_data.append(data)
                    success_count += 1

                if (i + 1) % 100 == 0 or i == len(stock_list) - 1:
                    progress = (i + 1) / len(stock_list) * 100
                    print(f"    进度: {i + 1}/{len(stock_list)} ({progress:.1f}%) | 成功: {success_count}")

        print(f"  ✓ 成功获取 {success_count}/{len(stock_list)} 只股票")

        if len(all_data) == 0:
            return None

        return pd.concat(all_data, ignore_index=True)


# ============ 增量数据管理器 ============

class IncrementalDataManager:
    """增量数据管理器"""

    def __init__(self, cache_manager, data_source):
        self.cache = cache_manager
        self.data_source = data_source

        print("\n" + "=" * 80)
        print("⚡ 增量数据更新系统")
        print("=" * 80)

    def get_cache_date_range(self, cache_name):
        cached_data = self.cache.load_from_csv(cache_name)
        if cached_data is None:
            return None

        if 'date' in cached_data.columns:
            dates = pd.to_datetime(cached_data['date'])
            return dates.min(), dates.max()

        return None

    def should_use_incremental_update(self, cache_name, target_end_date):
        date_range = self.get_cache_date_range(cache_name)

        if date_range is None:
            print("  📦 未发现缓存，将执行全量获取")
            return False, None

        cache_start, cache_end = date_range
        target_end = pd.to_datetime(target_end_date)

        days_diff = (target_end - cache_end).days

        if days_diff <= 0:
            print(f"  ✓ 缓存已是最新 (截止 {cache_end.strftime('%Y-%m-%d')})")
            return False, cache_end

        elif days_diff <= 30:
            print(f"  ⚡ 增量更新模式: 需更新 {days_diff} 天")
            print(f"     缓存日期: {cache_end.strftime('%Y-%m-%d')}")
            print(f"     目标日期: {target_end_date}")
            return True, cache_end

        else:
            print(f"  ⚠️  缓存过旧 ({days_diff} 天)，将执行全量获取")
            return False, None


# ============ 主数据加载函数（修复版）============

def load_data_with_incremental_update(start_date, end_date, max_stocks=800,
                                     cache_manager=None, use_stockranker=True,
                                     custom_weights=None, tushare_token=None,
                                     use_fundamental=True, force_full_update=False,
                                     use_sampling=True, sample_size=800, max_workers=10):
    """
    使用增量更新 + 多线程 + 智能抽样加载数据

    ✅ 修复：保留所有因子列，不只是position
    """
    print("\n" + "=" * 80)
    print("📦 数据加载模块 (增量更新 + 多线程 + 智能抽样)")
    print("=" * 80)

    from data_module import TushareDataSource, StockRankerModel

    data_source = TushareDataSource(
        cache_manager=cache_manager,
        token=tushare_token
    )

    model_suffix = "stockranker" if use_stockranker else "simple"
    if use_fundamental:
        model_suffix += "_fundamental"
    if use_sampling:
        model_suffix += f"_sample{sample_size}"

    price_cache_key = f"price_data_fast_{start_date}_{end_date}_{sample_size if use_sampling else max_stocks}"
    factor_cache_key = f"factor_data_fast_{start_date}_{end_date}_{sample_size if use_sampling else max_stocks}_{model_suffix}"
    financial_cache_key = f"financial_data_fast_{start_date}_{end_date}_{sample_size if use_sampling else max_stocks}"

    incremental_mgr = IncrementalDataManager(cache_manager, data_source)

    use_incremental = False
    cache_end_date = None

    if not force_full_update and cache_manager:
        use_incremental, cache_end_date = incremental_mgr.should_use_incremental_update(
            price_cache_key, end_date
        )

    if force_full_update:
        print("  🔄 强制全量更新模式")

    print("\n  📋 获取股票列表...")
    full_stock_list = data_source.get_stock_list()
    if not full_stock_list:
        print("✗ 无法获取股票列表!")
        return None, None

    print(f"  ✓ 获取到 {len(full_stock_list)} 只股票")

    if use_sampling and len(full_stock_list) > sample_size:
        sampler = SmartStockSampler(data_source)
        stock_list = sampler.get_stratified_sample(full_stock_list, sample_size)
    else:
        stock_list = full_stock_list[:max_stocks]
        if not use_sampling:
            print(f"  ℹ️  不使用抽样，使用前 {len(stock_list)} 只股票")

    rate_limiter = ThreadSafeRateLimiter(max_calls_per_minute=800)
    fetcher = ParallelDataFetcher(data_source, max_workers=max_workers, rate_limiter=rate_limiter)

    if use_incremental and cache_end_date:
        print("\n" + "=" * 80)
        print("⚡ 增量更新模式")
        print("=" * 80)

        print("\n  📂 加载历史数据...")
        old_price_data = cache_manager.load_from_csv(price_cache_key)
        old_financial_data = cache_manager.load_from_csv(financial_cache_key)

        incremental_start = (cache_end_date + timedelta(days=1)).strftime('%Y-%m-%d')
        incremental_end = end_date

        new_price_data = fetcher.fetch_price_data_parallel(
            stock_list, incremental_start, incremental_end
        )

        if new_price_data is not None and len(new_price_data) > 0:
            old_price_data['date'] = old_price_data['date'].astype(str)
            new_price_data['date'] = new_price_data['date'].astype(str)

            existing_dates = set(old_price_data['date'].unique())
            new_price_data_unique = new_price_data[~new_price_data['date'].isin(existing_dates)]

            price_df = pd.concat([old_price_data, new_price_data_unique], ignore_index=True)
            price_df = price_df.sort_values(['instrument', 'date']).reset_index(drop=True)

            print(f"  ✓ 数据合并完成:")
            print(f"     历史数据: {len(old_price_data)} 条")
            print(f"     新增数据: {len(new_price_data_unique)} 条")
            print(f"     合并总计: {len(price_df)} 条")
        else:
            print("  ⚠️  未获取到新数据，使用缓存数据")
            price_df = old_price_data

        financial_df = old_financial_data
        if use_fundamental:
            cache_quarter = pd.Period(cache_end_date, freq='Q')
            target_quarter = pd.Period(end_date, freq='Q')

            if target_quarter > cache_quarter:
                print(f"\n  📈 跨季度，更新基本面数据...")
                new_financial = fetcher.fetch_financial_data_parallel(
                    stock_list, incremental_start, incremental_end
                )

                if new_financial is not None and len(new_financial) > 0:
                    if old_financial_data is not None:
                        financial_df = pd.concat([old_financial_data, new_financial], ignore_index=True)
                        financial_df = financial_df.drop_duplicates(subset=['instrument', 'date'], keep='last')
                    else:
                        financial_df = new_financial
                    print(f"  ✓ 基本面数据已更新")
            else:
                print(f"  ℹ️  未跨季度，基本面数据无需更新")

    else:
        print("\n" + "=" * 80)
        print("📥 全量获取模式")
        print("=" * 80)

        price_df = fetcher.fetch_price_data_parallel(stock_list, start_date, end_date)

        if price_df is None or len(price_df) == 0:
            print("✗ 未获取到任何数据!")
            return None, None

        financial_df = None
        if use_fundamental:
            financial_df = fetcher.fetch_financial_data_parallel(stock_list, start_date, end_date)

    if use_fundamental and financial_df is not None:
        financial_df = financial_df.dropna(subset=['date', 'instrument'])

        if len(financial_df) > 0:
            print("\n  🔗 合并基本面数据到日线...")
            price_df = data_source.merge_financial_data_to_daily(price_df, financial_df)
            print("  ✓ 基本面数据合并完成")

            fundamental_cols = ['roe', 'roa', 'gross_margin', 'net_margin', 'debt_ratio']
            available_cols = [col for col in fundamental_cols if col in price_df.columns]
            if available_cols:
                coverage = (price_df[available_cols].notna().any(axis=1).sum() / len(price_df)) * 100
                print(f"     覆盖率: {coverage:.1f}%")

    price_df['date'] = price_df['date'].astype(str)

    if use_stockranker:
        model = StockRankerModel(
            custom_weights=custom_weights,
            use_fundamental=use_fundamental
        )
        factor_df = model.calculate_all_factors(price_df)
        factor_df = model.calculate_position_score(factor_df)
    else:
        from data_module import calculate_simple_factors
        factor_df = calculate_simple_factors(price_df)

    factor_df = factor_df.dropna(subset=['position'])

    # ✅ 关键修复：保留所有因子列，不只是position
    # 排除价格列和一些冗余列
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pre_close', 
                    'change', 'pct_chg', 'turnover_rate']
    
    # 保留所有非排除的列
    keep_cols = [col for col in factor_df.columns if col not in exclude_cols]
    result_factor = factor_df[keep_cols].copy()
    
    result_price = price_df.copy()

    if cache_manager:
        print("\n  💾 保存到缓存...")
        cache_manager.save_to_csv(result_price, price_cache_key)
        cache_manager.save_to_csv(result_factor, factor_cache_key)
        if financial_df is not None:
            cache_manager.save_to_csv(financial_df, financial_cache_key)

    # ✅ 统计因子列信息
    factor_columns = [col for col in result_factor.columns 
                      if col not in ['date', 'instrument', 'position']]
    
    print(f"\n✓ 数据准备完成:")
    print(f"  - 因子数据: {len(result_factor)} 条")
    print(f"  - 价格数据: {len(result_price)} 条")
    print(f"  - 股票数量: {result_factor['instrument'].nunique()} 只")
    print(f"  - 交易日数: {result_factor['date'].nunique()} 天")
    print(f"  - 因子列数: {len(factor_columns)} 个")
    
    if len(factor_columns) > 0:
        print(f"  - 因子列表: {', '.join(factor_columns[:10])}{'...' if len(factor_columns) > 10 else ''}")

    if use_incremental and cache_end_date:
        days_added = (pd.to_datetime(end_date) - cache_end_date).days
        print(f"  - 新增天数: {days_added} 天 ⚡")

    if use_sampling:
        print(f"  - 抽样方式: 市值分层抽样")

    return result_factor, result_price