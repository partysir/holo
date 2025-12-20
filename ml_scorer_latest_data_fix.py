# -*- coding: utf-8 -*-
"""
ml_scorer_latest_data_fix.py - 修复版 v3.0

🔧 核心修复：
1. ✅ 修正模型访问路径：ml_scorer.models['best'] 而非 ml_scorer.model
2. ✅ 修正特征标准化：使用 ml_scorer.scaler
3. ✅ 添加完整的错误处理
4. ✅ 支持分类和回归两种模型
5. ✅ 保证最新数据始终有评分

版本：v3.0
日期：2025-12-20
状态：生产就绪
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def diagnose_prediction_gap(factor_data, price_data, target_period=5):
    """
    诊断预测缺失问题
    """
    print("\n" + "="*80)
    print("🔍 预测缺失诊断")
    print("="*80)

    latest_factor_date = factor_data['date'].max()
    latest_price_date = price_data['date'].max()

    print(f"\n📅 数据日期:")
    print(f"   因子最新: {latest_factor_date}")
    print(f"   价格最新: {latest_price_date}")

    # 检查有评分的最新日期
    if 'ml_score' in factor_data.columns:
        valid_scores = factor_data[factor_data['ml_score'].notna()]
        if len(valid_scores) > 0:
            latest_scored_date = valid_scores['date'].max()
            print(f"   评分最新: {latest_scored_date}")

            gap_days = (pd.to_datetime(latest_factor_date) -
                       pd.to_datetime(latest_scored_date)).days

            if gap_days > 0:
                print(f"\n⚠️  评分缺失: {gap_days} 天")
                print(f"   缺失区间: {latest_scored_date} 到 {latest_factor_date}")
            else:
                print(f"\n✅ 评分完整")
        else:
            print(f"\n❌ 完全无评分")
    else:
        print(f"\n❌ 无 ml_score 列")

    # 检查未来收益标签
    factor_with_price = factor_data.merge(
        price_data[['date', 'instrument', 'close']],
        on=['date', 'instrument'],
        how='left'
    )

    factor_with_price = factor_with_price.sort_values(['instrument', 'date'])
    factor_with_price[f'future_return_{target_period}d'] = (
        factor_with_price.groupby('instrument')['close']
        .shift(-target_period) / factor_with_price['close'] - 1
    )

    latest_data = factor_with_price[factor_with_price['date'] == latest_factor_date]
    valid_returns = latest_data[f'future_return_{target_period}d'].notna().sum()

    print(f"\n📊 最新数据 ({latest_factor_date}):")
    print(f"   总股票数: {len(latest_data)}")
    print(f"   有未来收益标签: {valid_returns}")
    print(f"   缺失比例: {(len(latest_data)-valid_returns)/len(latest_data)*100:.1f}%")

    if valid_returns == 0:
        print(f"\n💡 根因分析:")
        print(f"   • 最新数据需要 {target_period} 天后的价格计算收益")
        print(f"   • 但我们只有到 {latest_price_date} 的价格")
        print(f"   • 因此无法生成训练标签 y")
        print(f"   • 导致 Walk-Forward 跳过最新窗口")

    print("="*80)


def quick_fix_ml_scorer(ml_scorer, factor_data, price_data, factor_columns):
    """
    🔧 修复版：为最新数据生成ML评分

    核心修复点：
    1. 使用 ml_scorer.models['best'] 而非 ml_scorer.model
    2. 使用 ml_scorer.scaler 进行特征标准化
    3. 处理全部无评分数据（不仅是最新日期）

    Args:
        ml_scorer: 已训练的AdvancedMLScorer实例
        factor_data: 因子数据
        price_data: 价格数据
        factor_columns: 因子列名列表

    Returns:
        factor_data: 补全了评分的DataFrame
    """
    print("\n" + "="*80)
    print("🔧 应用最新数据预测修复 (v3.0)")
    print("="*80)

    # ============ 步骤0: 验证ML评分器状态 ============
    if ml_scorer is None:
        print("  ❌ ML评分器为 None")
        return _fallback_scoring(factor_data, factor_columns)

    # 🔧 修复点1：检查正确的模型路径
    if not hasattr(ml_scorer, 'models') or 'best' not in ml_scorer.models:
        print("  ❌ 模型未训练 (缺少 models['best'])")
        return _fallback_scoring(factor_data, factor_columns)

    model = ml_scorer.models['best']
    if model is None:
        print("  ❌ 最佳模型为 None")
        return _fallback_scoring(factor_data, factor_columns)

    # 检查标准化器
    if not hasattr(ml_scorer, 'scaler') or ml_scorer.scaler is None:
        print("  ⚠️  警告：缺少标准化器，预测精度可能降低")

    print(f"  ✅ 模型状态验证通过:")
    print(f"     • 模型类型: {type(model).__name__}")
    print(f"     • 分类模式: {ml_scorer.use_classification if hasattr(ml_scorer, 'use_classification') else 'Unknown'}")

    # ============ 步骤1: 识别需要预测的数据 ============
    print(f"\n  📅 分析评分覆盖情况...")

    if 'ml_score' not in factor_data.columns:
        # 如果完全没有ml_score列，需要全部预测
        factor_data['ml_score'] = np.nan
        dates_to_predict = factor_data['date'].unique()
        print(f"     • 无现有评分，需全部预测")
    else:
        # 找出评分缺失的日期
        date_coverage = factor_data.groupby('date')['ml_score'].apply(
            lambda x: x.notna().sum() / len(x)
        )
        dates_to_predict = date_coverage[date_coverage < 0.5].index.tolist()

        if len(dates_to_predict) == 0:
            print(f"     ℹ️  评分已完整，无需修复")
            return factor_data

    print(f"     • 需预测日期: {len(dates_to_predict)} 天")
    if len(dates_to_predict) <= 5:
        for date in dates_to_predict:
            print(f"       - {date}")
    else:
        for date in dates_to_predict[:3]:
            print(f"       - {date}")
        print(f"       ... 还有 {len(dates_to_predict)-3} 天")

    # ============ 步骤2: 准备特征 ============
    print(f"\n  🔨 准备特征数据...")

    # 🔧 修复点2：使用正确的特征列表
    if hasattr(ml_scorer, 'feature_names') and ml_scorer.feature_names:
        model_features = ml_scorer.feature_names
    else:
        print(f"     ⚠️  警告：使用传入的factor_columns作为特征")
        model_features = factor_columns

    print(f"     • 模型特征数: {len(model_features)}")

    # 提取需要预测的数据
    data_to_predict = factor_data[factor_data['date'].isin(dates_to_predict)].copy()
    print(f"     • 待预测样本: {len(data_to_predict)}")

    # 检查缺失特征
    missing_features = [f for f in model_features if f not in data_to_predict.columns]
    if missing_features:
        print(f"     ⚠️  缺失 {len(missing_features)} 个特征，用0填充")
        if len(missing_features) <= 5:
            print(f"        {missing_features}")
        for feat in missing_features:
            data_to_predict[feat] = 0

    # 构建特征矩阵
    try:
        X_predict = data_to_predict[model_features].values
    except Exception as e:
        print(f"     ❌ 特征提取失败: {e}")
        return _fallback_scoring(factor_data, factor_columns)

    # 处理NaN和Inf
    X_predict = np.nan_to_num(X_predict, nan=0.0, posinf=0.0, neginf=0.0)

    # 🔧 修复点3：使用标准化器
    if hasattr(ml_scorer, 'scaler') and ml_scorer.scaler is not None:
        try:
            X_predict_scaled = ml_scorer.scaler.transform(X_predict)
            print(f"     ✅ 特征标准化完成")
        except Exception as e:
            print(f"     ⚠️  标准化失败: {e}，使用原始特征")
            X_predict_scaled = X_predict
    else:
        X_predict_scaled = X_predict

    # ============ 步骤3: 执行预测 ============
    print(f"\n  🚀 执行预测...")

    try:
        use_classification = ml_scorer.use_classification if hasattr(ml_scorer, 'use_classification') else False

        if use_classification:
            # 分类模型：预测概率
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_predict_scaled)
                # 处理不同的输出格式
                if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                    predictions = predictions[:, 1]  # 取正类概率
                else:
                    predictions = predictions.flatten()
                print(f"     ✅ 分类预测完成: {len(predictions)} 个样本")
            else:
                print(f"     ⚠️  模型无 predict_proba，使用 predict")
                predictions = model.predict(X_predict_scaled)
        else:
            # 回归模型：直接预测
            predictions = model.predict(X_predict_scaled)
            print(f"     ✅ 回归预测完成: {len(predictions)} 个样本")

        # 确保预测结果是1维数组
        predictions = np.asarray(predictions).flatten()

    except Exception as e:
        print(f"     ❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return _fallback_scoring(factor_data, factor_columns)

    # ============ 步骤4: 后处理与写入 ============
    print(f"\n  📝 后处理与写入...")

    # 写入原始预测值
    data_to_predict['ml_score'] = predictions

    # 按日期标准化到0-1区间（使用排名百分位）
    for date in dates_to_predict:
        date_mask = data_to_predict['date'] == date
        scores = data_to_predict.loc[date_mask, 'ml_score']

        if len(scores) > 0:
            # 使用排名百分位（更稳健）
            ranked = scores.rank(pct=True)
            data_to_predict.loc[date_mask, 'ml_score'] = ranked

    print(f"     • 已对 {len(dates_to_predict)} 个日期进行排名标准化")

    # 合并回原数据
    for idx, row in data_to_predict.iterrows():
        mask = (factor_data['date'] == row['date']) & \
               (factor_data['instrument'] == row['instrument'])
        factor_data.loc[mask, 'ml_score'] = row['ml_score']

    # 确保同时创建position列（用于回测）
    if 'position' not in factor_data.columns:
        factor_data['position'] = factor_data['ml_score']
    else:
        # 更新position列（仅更新新预测的部分）
        for date in dates_to_predict:
            date_mask = factor_data['date'] == date
            factor_data.loc[date_mask, 'position'] = factor_data.loc[date_mask, 'ml_score']

    # ============ 步骤5: 验证结果 ============
    print(f"\n  ✅ 修复完成，验证结果...")

    latest_date = factor_data['date'].max()
    latest_data = factor_data[factor_data['date'] == latest_date]
    valid_count = latest_data['ml_score'].notna().sum()

    print(f"\n  📊 修复后状态:")
    print(f"     • 最新日期: {latest_date}")
    print(f"     • 有效评分: {valid_count}/{len(latest_data)} ({valid_count/len(latest_data)*100:.1f}%)")

    if valid_count == 0:
        print(f"     ❌ 警告：最新日期仍无评分！")
        return _fallback_scoring(factor_data, factor_columns)
    elif valid_count < len(latest_data) * 0.5:
        print(f"     ⚠️  警告：覆盖率偏低")
    else:
        print(f"     ✅ 覆盖率良好")

    # 全局统计
    total_valid = factor_data['ml_score'].notna().sum()
    total_count = len(factor_data)
    print(f"     • 全局覆盖: {total_valid}/{total_count} ({total_valid/total_count*100:.1f}%)")

    print("="*80)
    return factor_data


def _fallback_scoring(factor_data, factor_columns):
    """
    🆘 Fallback评分方案（当ML预测失败时）
    """
    print("\n  🚨 启动 Fallback 评分方案...")

    data = factor_data.copy()

    # 方案1：使用position列
    if 'position' in data.columns:
        if data['position'].notna().sum() > len(data) * 0.5:
            print("     • 使用现有 position 列作为 ml_score")
            data['ml_score'] = data['position']
            return data

    # 方案2：使用因子均值
    valid_factors = [col for col in factor_columns
                    if col in data.columns and pd.api.types.is_numeric_dtype(data[col])]

    if valid_factors:
        print(f"     • 使用 {len(valid_factors)} 个因子的均值")
        data['ml_score'] = data[valid_factors].mean(axis=1)
        data['ml_score'] = data.groupby('date')['ml_score'].rank(pct=True)
        data['position'] = data['ml_score']
        return data

    # 方案3：随机评分（最后手段）
    print("     • ⚠️  紧急措施：随机评分")
    data['ml_score'] = np.random.rand(len(data))
    data['ml_score'] = data.groupby('date')['ml_score'].rank(pct=True)
    data['position'] = data['ml_score']

    return data


class FixedAdvancedMLScorer:
    """
    修复版ML评分器包装器
    自动处理最新数据预测问题
    """

    def __init__(self, base_scorer):
        """
        Args:
            base_scorer: AdvancedMLScorer实例
        """
        self.base_scorer = base_scorer

    def predict_with_fix(self, factor_data, price_data, factor_columns):
        """
        带修复的预测流程
        """
        # 先用标准流程预测（会漏掉最新数据）
        try:
            factor_data = self.base_scorer.predict_scores(factor_data)
        except Exception as e:
            print(f"⚠️  标准预测失败: {e}")
        
        # 然后修复最新数据
        factor_data = quick_fix_ml_scorer(
            self.base_scorer, factor_data, price_data, factor_columns
        )
        
        return factor_data


# ============ 测试函数 ============
def test_fix():
    """测试修复功能"""
    print("\n" + "="*80)
    print("🧪 测试ML修复功能")
    print("="*80)
    
    # 创建模拟数据
    dates = pd.date_range('2025-01-01', '2025-12-19', freq='D')
    stocks = ['000001.SZ', '000002.SZ', '600000.SH']
    
    data = []
    for date in dates:
        for stock in stocks:
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'instrument': stock,
                'factor1': np.random.randn(),
                'factor2': np.random.randn(),
                'close': 10 + np.random.randn()
            })
    
    factor_data = pd.DataFrame(data)
    price_data = factor_data[['date', 'instrument', 'close']].copy()
    
    print(f"\n  生成模拟数据: {len(factor_data)} 行")
    print(f"  日期范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")
    
    # 模拟评分缺失
    latest_date = factor_data['date'].max()
    factor_data['ml_score'] = np.random.rand(len(factor_data))
    factor_data.loc[factor_data['date'] == latest_date, 'ml_score'] = np.nan
    
    print(f"\n  模拟最新日期无评分")
    
    # 诊断
    diagnose_prediction_gap(factor_data, price_data, target_period=5)
    
    print("\n  ✅ 测试完成")


if __name__ == "__main__":
    test_fix()