"""
data_quality_checker.py - 数据质量检查与修复工具

功能：
1. 检测股票代码是否有效
2. 识别一字涨停/跌停
3. 检测异常持仓量
4. 验证价格连续性
"""

import pandas as pd
import numpy as np


class DataQualityChecker:
    """数据质量检查器"""

    def __init__(self):
        # A股有效代码段
        self.valid_code_patterns = {
            '沪市主板': (r'^60[0-9]{4}\.SH$', '600000-603999'),
            '科创板': (r'^68[8-9]{1}[0-9]{3}\.SH$', '688000-689999'),
            '深市主板': (r'^00[0-2]{1}[0-9]{3}\.SZ$', '000000-002999'),
            '创业板': (r'^30[0-1]{1}[0-9]{3}\.SZ$', '300000-301999'),
            '北交所': (r'^[4|8]{1}[0-9]{5}\.BJ$', '400000-899999')
        }

    def check_stock_code(self, code):
        """检查股票代码有效性"""
        import re
        for board, (pattern, range_desc) in self.valid_code_patterns.items():
            if re.match(pattern, code):
                return True, board
        return False, None

    def detect_limit_up(self, df):
        """
        检测一字涨停板

        判断标准：
        1. 开盘价 = 最高价 = 最低价 = 收盘价
        2. 涨幅 >= 9.9% (主板) 或 >= 19.9% (创业板/科创板)
        """
        df = df.copy()

        # 计算涨幅（需要前一日收盘价）
        df['prev_close'] = df.groupby('instrument')['close'].shift(1)
        df['pct_chg'] = (df['close'] - df['prev_close']) / df['prev_close'] * 100

        # 判断板块（根据代码）
        def get_limit_threshold(code):
            if code.startswith('688') or code.startswith('30'):
                return 19.9  # 科创板/创业板 20cm
            return 9.9  # 主板 10cm

        df['limit_threshold'] = df['instrument'].apply(get_limit_threshold)

        # 一字板条件
        df['is_limit_up'] = (
                (df['open'] == df['high']) &
                (df['high'] == df['low']) &
                (df['low'] == df['close']) &
                (df['pct_chg'] >= df['limit_threshold'])
        )

        return df

    def detect_abnormal_position(self, trade_records, price_data):
        """
        检测异常持仓量

        异常标准：
        1. 单只股票持仓 > 流通股本的10%
        2. 单日买入量 > 当日成交量的20%
        """
        abnormal_trades = []

        # 处理中英文列名
        action_col = 'action' if 'action' in trade_records.columns else '操作'
        date_col = 'date' if 'date' in trade_records.columns else '日期'
        stock_col = 'stock' if 'stock' in trade_records.columns else '股票'
        shares_col = 'shares' if 'shares' in trade_records.columns else '数量'
        
        # 确保买入标识的一致性
        buy_action = 'buy' if action_col == 'action' else '买入'

        for idx, trade in trade_records.iterrows():
            if trade[action_col] != buy_action:
                continue

            date = trade[date_col]
            stock = trade[stock_col]
            shares = trade[shares_col]

            # 获取当日成交量
            day_data = price_data[
                (price_data['date'] == date) &
                (price_data['instrument'] == stock)
                ]

            if len(day_data) == 0:
                continue

            volume = day_data.iloc[0]['volume']

            # 检查是否超过当日成交量的20%
            if shares > volume * 0.2:
                abnormal_trades.append({
                    'date': date,
                    'stock': stock,
                    'shares': shares,
                    'volume': volume,
                    'ratio': shares / volume,
                    'reason': '买入量超过当日成交量20%'
                })

        return pd.DataFrame(abnormal_trades)

    def check_price_continuity(self, df):
        """
        检查价格连续性

        异常标准：
        1. 单日涨跌幅 > 30%（排除复权因素）
        2. 连续多日无成交量
        """
        df = df.copy()
        df = df.sort_values(['instrument', 'date'])

        # 计算涨跌幅
        df['prev_close'] = df.groupby('instrument')['close'].shift(1)
        df['pct_chg'] = (df['close'] - df['prev_close']) / df['prev_close'] * 100

        # 标记异常
        df['abnormal_chg'] = abs(df['pct_chg']) > 30
        df['no_volume'] = df['volume'] == 0

        abnormal = df[df['abnormal_chg'] | df['no_volume']].copy()

        return abnormal[['date', 'instrument', 'close', 'pct_chg', 'volume']]

    def run_full_check(self, price_data, trade_records=None):
        """运行完整检查"""
        print("\n" + "=" * 80)
        print("🔍 数据质量全面检查")
        print("=" * 80)

        results = {}

        # 1. 股票代码检查
        print("\n📋 检查股票代码有效性...")
        unique_codes = price_data['instrument'].unique()
        invalid_codes = []

        for code in unique_codes:
            is_valid, board = self.check_stock_code(code)
            if not is_valid:
                invalid_codes.append(code)

        if invalid_codes:
            print(f"  ❌ 发现 {len(invalid_codes)} 个无效代码:")
            for code in invalid_codes[:10]:  # 只显示前10个
                print(f"     {code}")
        else:
            print(f"  ✅ 所有代码有效 ({len(unique_codes)} 只)")

        results['invalid_codes'] = invalid_codes

        # 2. 一字涨停检查
        print("\n📈 检查一字涨停板...")
        df_with_limits = self.detect_limit_up(price_data)
        limit_up_days = df_with_limits[df_with_limits['is_limit_up']]

        print(f"  ⚠️  发现 {len(limit_up_days)} 个一字涨停交易日")
        if len(limit_up_days) > 0:
            print(f"  涉及股票: {limit_up_days['instrument'].nunique()} 只")
            print("\n  样例:")
            sample = limit_up_days[['date', 'instrument', 'close', 'pct_chg']].head(5)
            print(sample.to_string(index=False))

        results['limit_up_days'] = limit_up_days

        # 3. 异常持仓检查
        if trade_records is not None:
            print("\n💰 检查异常持仓量...")
            try:
                abnormal_pos = self.detect_abnormal_position(trade_records, price_data)

                if len(abnormal_pos) > 0:
                    print(f"  ❌ 发现 {len(abnormal_pos)} 笔异常交易:")
                    print(abnormal_pos.to_string(index=False))
                else:
                    print("  ✅ 未发现异常持仓")

                results['abnormal_positions'] = abnormal_pos
            except Exception as e:
                print(f"  ⚠️  异常持仓检查失败: {e}")
                results['abnormal_positions'] = pd.DataFrame()

        # 4. 价格连续性检查
        print("\n📊 检查价格连续性...")
        abnormal_prices = self.check_price_continuity(price_data)

        if len(abnormal_prices) > 0:
            print(f"  ⚠️  发现 {len(abnormal_prices)} 个异常价格点:")
            print(abnormal_prices.head(10).to_string(index=False))
        else:
            print("  ✅ 价格连续性正常")

        results['abnormal_prices'] = abnormal_prices

        print("\n" + "=" * 80)
        print("✅ 检查完成")
        print("=" * 80)

        return results


def fix_invalid_codes(price_data, mapping=None):
    """
    修复无效股票代码

    Args:
        price_data: 价格数据
        mapping: 代码映射字典 {'错误代码': '正确代码'}
    """
    if mapping is None:
        # 常见错误映射
        mapping = {
            '302132.SZ': '300114.SZ',  # 中航电测
        }

    df = price_data.copy()

    for wrong, correct in mapping.items():
        if wrong in df['instrument'].values:
            print(f"  修复: {wrong} → {correct}")
            df.loc[df['instrument'] == wrong, 'instrument'] = correct

    return df


def filter_unbuyable_stocks(price_data):
    """
    过滤无法买入的股票

    策略：
    1. 移除一字涨停日的数据
    2. 移除成交量为0的数据
    """
    checker = DataQualityChecker()

    print("\n🚫 过滤无法买入的股票...")
    original_len = len(price_data)

    # 检测一字涨停
    df = checker.detect_limit_up(price_data)

    # 过滤
    df = df[
        (~df['is_limit_up']) &  # 非一字涨停
        (df['volume'] > 0)  # 有成交量
        ].copy()

    filtered_len = original_len - len(df)

    print(f"  原始数据: {original_len:,} 行")
    print(f"  过滤后: {len(df):,} 行")
    print(f"  移除: {filtered_len:,} 行 ({filtered_len / original_len * 100:.2f}%)")

    # 清理临时列
    df = df.drop(columns=['prev_close', 'pct_chg', 'limit_threshold', 'is_limit_up'],
                 errors='ignore')

    return df


# ========== 使用示例 ==========

if __name__ == "__main__":
    print("数据质量检查工具已加载")
    print("\n使用方法:")
    print("1. checker = DataQualityChecker()")
    print("2. results = checker.run_full_check(price_data, trade_records)")
    print("3. clean_data = filter_unbuyable_stocks(price_data)")