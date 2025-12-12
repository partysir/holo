"""
optimizer_comparison.py - 优化效果对比测试 (最终修复版)

核心修复：
1. 数据切片：从按行切片(tail)改为按日期切片，确保能计算未来收益
2. 评估器：智能识别价格列，防止重复Merge导致的KeyError
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ============================================================================
# 1. 模拟真实市场数据生成器
# ============================================================================

class MarketDataSimulator:
    """生成接近真实市场的模拟数据"""

    def __init__(self, n_stocks=200, n_days=500, random_state=42):
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.random_state = random_state
        np.random.seed(random_state)

    def generate(self):
        print("📊 生成模拟市场数据...")
        dates = pd.date_range('2023-01-01', periods=self.n_days, freq='D')
        instruments = [f'STOCK_{i:03d}' for i in range(self.n_stocks)]
        industries = ['科技', '金融', '消费', '医药', '能源']

        # 1. 分配行业
        stock_industry = {inst: np.random.choice(industries) for inst in instruments}

        # 2. 生成价格 (随机游走 + 趋势 + 行业效应)
        price_matrix = np.zeros((self.n_days, self.n_stocks))
        mkt_trend = np.cumsum(np.random.randn(self.n_days) * 0.01) + 0.0005

        for i, inst in enumerate(instruments):
            alpha = np.random.randn() * 0.0002
            beta = 0.5 + np.random.rand() * 1.0
            noise = np.random.randn(self.n_days) * 0.02
            # 价格生成
            ret = alpha + beta * mkt_trend + noise
            price_matrix[:, i] = 100 * np.exp(np.cumsum(ret))

        # 3. 构造DataFrame并计算因子
        data_list = []
        for t, date in enumerate(dates):
            # 为了速度，我们只在最后200天生成完整因子，或者每隔几天
            # 这里为了模拟完整性，生成所有数据，但使用向量化思维简化逻辑
            pass

        # 为简化生成过程，直接构建长表
        df_list = []
        for i, inst in enumerate(instruments):
            prices = price_matrix[:, i]
            df = pd.DataFrame({
                'date': dates,
                'instrument': inst,
                'industry': stock_industry[inst],
                'close': prices
            })

            # 计算因子
            df['factor_momentum'] = df['close'].pct_change(20).fillna(0)
            df['factor_reversal'] = -df['close'].pct_change(5).fillna(0)
            df['factor_volatility'] = -df['close'].pct_change().rolling(20).std().fillna(0)
            df['factor_noise'] = np.random.randn(len(df))

            df_list.append(df)

        final_df = pd.concat(df_list, ignore_index=True)
        final_df = final_df.sort_values(['date', 'instrument'])

        print(f"  ✓ 生成 {len(final_df)} 条数据 ({self.n_stocks}只股票 x {self.n_days}天)")
        return final_df


# ============================================================================
# 2. 回测评估器 (智能修复版)
# ============================================================================

class BacktestEvaluator:
    """回测评估器 (修复Merge冲突)"""

    @staticmethod
    def calculate_ic(predictions, actuals):
        df = pd.DataFrame({'pred': predictions, 'actual': actuals}).dropna()
        if len(df) < 10: return np.nan
        return df['pred'].corr(df['actual'])

    @staticmethod
    def evaluate_portfolio(factor_data, price_data, score_col='ml_score',
                          holding_period=5, top_pct=0.2):
        print("\n📈 回测评估...")

        # === 智能合并逻辑 ===
        # 1. 检测 factor_data 是否已有价格
        has_price = False
        price_col_name = 'close'
        for col in factor_data.columns:
            if col.lower() in ['close', 'price']:
                has_price = True
                price_col_name = col
                break

        if has_price:
            print(f"  ✓ 数据中已包含价格列 '{price_col_name}'，跳过合并")
            merged = factor_data.copy()
        else:
            # 需要从 price_data 合并
            print("  Combinig price data...")
            # 找到 price_data 里的价格列名
            p_col = 'close'
            for col in price_data.columns:
                if col.lower() in ['close', 'price']:
                    p_col = col
                    break

            merged = factor_data.merge(
                price_data[['instrument', 'date', p_col]],
                on=['instrument', 'date'],
                how='left'
            )
            price_col_name = p_col

        merged = merged.sort_values(['instrument', 'date'])

        # === 收益率计算 ===
        # 确保是数值
        merged[price_col_name] = pd.to_numeric(merged[price_col_name], errors='coerce')

        # 计算未来收益 (Shift 负数)
        merged['future_return'] = merged.groupby('instrument')[price_col_name].pct_change(holding_period).shift(-holding_period)

        # 计算超额收益
        market_ret = merged.groupby('date')['future_return'].transform('mean')
        merged['excess_return'] = merged['future_return'] - market_ret

        # 过滤有效行 (必须有分数，且有未来收益)
        valid = merged.dropna(subset=[score_col, 'excess_return'])

        if len(valid) == 0:
            print("  ⚠️ 无有效回测数据 (可能是由于处于最后几个交易日，无法计算未来收益)")
            return None

        # === 指标计算 ===

        # 1. IC
        daily_ic = []
        for date, group in valid.groupby('date'):
            if len(group) > 10:
                ic = group[score_col].corr(group['excess_return'])
                if not np.isnan(ic): daily_ic.append(ic)

        ic_mean = np.mean(daily_ic) if daily_ic else 0
        ic_ir = ic_mean / np.std(daily_ic) if (daily_ic and np.std(daily_ic) > 0) else 0

        # 2. Precision@K
        # 取 Top N%
        threshold = valid[score_col].quantile(1 - top_pct)
        top_picks = valid[valid[score_col] >= threshold]

        # 胜率: 超额收益 > 0
        win_rate = (top_picks['excess_return'] > 0).mean()

        # 精确度: 全局 Top K
        top_k_global = int(len(valid) * top_pct)
        if top_k_global > 0:
            best_preds = valid.nlargest(top_k_global, score_col)
            precision = (best_preds['excess_return'] > 0).mean()
        else:
            precision = 0

        avg_ret = top_picks['excess_return'].mean()

        results = {
            'ic': ic_mean,
            'ir': ic_ir,
            'win_rate': win_rate,
            'precision_at_k': precision,
            'avg_excess_return': avg_ret
        }

        print(f"  IC: {ic_mean:.4f} | IR: {ic_ir:.4f}")
        print(f"  Win Rate (Top {int(top_pct*100)}%): {win_rate:.2%}")
        print(f"  Avg Excess Ret: {avg_ret:.4%}")

        return results


# ============================================================================
# 3. 对比测试主逻辑
# ============================================================================

def compare_optimizations():
    print("="*80)
    print("🎯 ML因子评分优化效果对比测试")
    print("="*80)

    # 1. 准备数据
    sim = MarketDataSimulator(n_stocks=200, n_days=500)
    data = sim.generate()
    factor_cols = ['factor_momentum', 'factor_reversal', 'factor_volatility', 'factor_noise']

    # 划分训练集和测试集 (按时间切分，模拟实盘)
    # 取前80%做训练，后20%做测试
    # 关键修正：不能用 data.tail(200) 这种按行切分，必须按日期切分
    split_date = data['date'].min() + (data['date'].max() - data['date'].min()) * 0.8

    train_data = data[data['date'] <= split_date].copy()
    test_data = data[data['date'] > split_date].copy()

    print(f"\n📅 数据切分:")
    print(f"  训练集: {len(train_data)} 行 ({train_data['date'].min().date()} - {train_data['date'].max().date()})")
    print(f"  测试集: {len(test_data)} 行 ({test_data['date'].min().date()} - {test_data['date'].max().date()})")

    results = {}

    # --- 测试1: 基础版 ---
    print("\n" + "="*60)
    print("📊 测试1: 基础版 (ml_factor_scoring_fixed)")
    print("="*60)
    try:
        from ml_factor_scoring_fixed import AdvancedMLScorer
        scorer_basic = AdvancedMLScorer(
            model_type='xgboost', target_period=5, top_percentile=0.2,
            use_ic_features=False, train_months=6
        )

        X, y, merged = scorer_basic.prepare_training_data(train_data, train_data, factor_cols)
        scorer_basic.train_walk_forward(X, y, merged, verbose=False)

        # 预测
        res_basic = scorer_basic.predict_scores(test_data, data, factor_cols) # 传入data作为price_data源

        results['basic'] = BacktestEvaluator.evaluate_portfolio(
            res_basic, data, 'ml_score', holding_period=5
        )
    except ImportError:
        print("  ⚠️ 模块未找到，跳过")
        results['basic'] = None
    except Exception as e:
        print(f"  ⚠️ 测试失败: {e}")
        results['basic'] = None

    # --- 测试2: 超级优化版 ---
    print("\n" + "="*60)
    print("📊 测试2: 超级优化版 (UltraMLScorer)")
    print("="*60)
    try:
        from ml_factor_scoring_ultra_standalone import UltraMLScorer

        scorer_ultra = UltraMLScorer(
            target_period=5,
            top_percentile=0.20,
            embargo_days=5,
            neutralize_market=True,
            neutralize_industry=True,
            voting_strategy='average',
            train_months=6
        )

        # 训练
        X, y, merged = scorer_ultra.prepare_data(train_data, train_data, factor_cols)
        scorer_ultra.train(X, y, merged, verbose=False)

        # 预测
        # 注意：在预测时，我们只传入 test_data
        # UltraMLScorer 会自动处理 test_data 的正交化
        res_ultra = scorer_ultra.predict(test_data, test_data) # 传入test_data本身作为price_source

        results['ultra'] = BacktestEvaluator.evaluate_portfolio(
            res_ultra, data, 'ml_score', holding_period=5
        )

    except Exception as e:
        print(f"  ⚠️ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['ultra'] = None

    # --- 汇总结果 ---
    print("\n" + "="*80)
    print("🏆 最终对比结果")
    print("="*80)

    rows = []
    for name, res in results.items():
        if res:
            rows.append({
                'Version': name,
                'IC': res['ic'],
                'IR': res['ir'],
                'WinRate': res['win_rate'],
                'Precision': res['precision_at_k'],
                'AvgRet': res['avg_excess_return']
            })
        else:
            rows.append({'Version': name, 'IC': 0, 'IR': 0})

    df_res = pd.DataFrame(rows)
    print(df_res.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

if __name__ == '__main__':
    compare_optimizations()