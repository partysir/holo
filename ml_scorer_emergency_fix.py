# -*- coding: utf-8 -*-
"""
ml_scorer_emergency_fix.py - ML评分器紧急修复补丁

🚑 紧急修复内容：
1. ✅ 捕获并处理 quick_fix_ml_scorer 失败的情况
2. ✅ 提供 fallback 机制：使用原始 predict_scores
3. ✅ 确保 ml_score 列始终存在
4. ✅ 添加详细的错误日志和诊断信息

使用方法：
    在 main.py 中替换原有的修复调用：

    # 替换这部分代码：
    if ML_FIX_AVAILABLE:
        factor_data = quick_fix_ml_scorer(...)

    # 改为：
    if ML_FIX_AVAILABLE:
        factor_data = safe_apply_ml_fix(...)
"""

import pandas as pd
import numpy as np
import traceback
from typing import List, Optional


def diagnose_ml_scorer_state(ml_scorer) -> dict:
    """
    🔍 诊断 ML 评分器状态

    Returns:
        dict: 诊断信息
    """
    diagnosis = {
        'has_models': False,
        'has_best_model': False,
        'has_feature_names': False,
        'has_scaler': False,
        'feature_count': 0,
        'model_type': None
    }

    try:
        if hasattr(ml_scorer, 'models'):
            diagnosis['has_models'] = len(ml_scorer.models) > 0
            diagnosis['has_best_model'] = 'best' in ml_scorer.models
            if diagnosis['has_best_model']:
                diagnosis['model_type'] = type(ml_scorer.models['best']).__name__

        if hasattr(ml_scorer, 'feature_names'):
            diagnosis['has_feature_names'] = ml_scorer.feature_names is not None
            diagnosis['feature_count'] = len(ml_scorer.feature_names) if ml_scorer.feature_names else 0

        if hasattr(ml_scorer, 'scaler'):
            diagnosis['has_scaler'] = ml_scorer.scaler is not None

    except Exception as e:
        print(f"   ⚠️  诊断过程出错: {e}")

    return diagnosis


def print_diagnosis_report(diagnosis: dict):
    """打印诊断报告"""
    print("\n   📋 ML评分器状态诊断:")
    print(f"      • 模型容器: {'✓' if diagnosis['has_models'] else '✗'}")
    print(f"      • 最佳模型: {'✓' if diagnosis['has_best_model'] else '✗'} ({diagnosis['model_type'] or 'N/A'})")
    print(f"      • 特征列表: {'✓' if diagnosis['has_feature_names'] else '✗'} ({diagnosis['feature_count']} 个)")
    print(f"      • 标准化器: {'✓' if diagnosis['has_scaler'] else '✗'}")


def fallback_predict_scores(ml_scorer, factor_data: pd.DataFrame,
                            factor_columns: List[str]) -> pd.DataFrame:
    """
    🆘 Fallback 预测方法（当修复补丁失败时）

    使用原始的 predict_scores 方法，但添加错误处理
    """
    print("   🔄 使用 Fallback 预测方法...")

    try:
        # 检查 ml_scorer 状态
        if not hasattr(ml_scorer, 'models') or 'best' not in ml_scorer.models:
            raise ValueError("ML评分器未训练或缺少最佳模型")

        if not hasattr(ml_scorer, 'feature_names') or not ml_scorer.feature_names:
            raise ValueError("ML评分器缺少特征列表")

        # 调用原始预测方法
        result = ml_scorer.predict_scores(factor_data)

        # 验证结果
        if 'ml_score' not in result.columns:
            raise ValueError("预测结果缺少 ml_score 列")

        # 统计有效评分
        valid_count = result['ml_score'].notna().sum()
        total_count = len(result)
        print(f"   ✓ Fallback 预测成功: {valid_count}/{total_count} ({valid_count / total_count:.1%})")

        return result

    except Exception as e:
        print(f"   ❌ Fallback 预测失败: {e}")
        traceback.print_exc()

        # 最终 fallback：使用 position 列作为 ml_score
        print("   🚨 启动紧急备用方案：使用 position 作为 ml_score")
        result = factor_data.copy()

        if 'position' in result.columns:
            result['ml_score'] = result['position']
            print("   ✓ 已将 position 列复制为 ml_score")
        else:
            # 如果连 position 都没有，使用因子均值
            print("   ⚠️  position 列也不存在，使用因子均值")
            valid_factors = [col for col in factor_columns if col in result.columns]
            if valid_factors:
                result['ml_score'] = result[valid_factors].mean(axis=1)
                result['ml_score'] = result.groupby('date')['ml_score'].rank(pct=True)
                result['position'] = result['ml_score']
            else:
                # 最坏情况：随机评分（仅用于防止崩溃）
                print("   🚨 紧急措施：生成随机评分（请立即检查数据）")
                result['ml_score'] = np.random.rand(len(result))
                result['position'] = result.groupby('date')['ml_score'].rank(pct=True)

        return result


def safe_apply_ml_fix(ml_scorer, factor_data: pd.DataFrame,
                      price_data: pd.DataFrame, factor_columns: List[str],
                      ML_FIX_AVAILABLE: bool = True) -> pd.DataFrame:
    """
    🛡️ 安全应用 ML 修复（带多重保障）

    Args:
        ml_scorer: ML评分器实例
        factor_data: 因子数据
        price_data: 价格数据
        factor_columns: 因子列表
        ML_FIX_AVAILABLE: 修复补丁是否可用

    Returns:
        pd.DataFrame: 包含 ml_score 的数据（保证不为空）
    """
    print("   [3/5] 应用最新数据预测修复（安全模式）...")

    # 诊断 ML 评分器状态
    diagnosis = diagnose_ml_scorer_state(ml_scorer)
    print_diagnosis_report(diagnosis)

    # 如果评分器状态异常，直接使用 fallback
    if not diagnosis['has_best_model'] or not diagnosis['has_feature_names']:
        print("   ⚠️  ML评分器状态异常，跳过修复补丁")
        return fallback_predict_scores(ml_scorer, factor_data, factor_columns)

    # 尝试使用修复补丁
    if ML_FIX_AVAILABLE:
        try:
            from ml_scorer_latest_data_fix import quick_fix_ml_scorer

            print("   🔧 尝试应用修复补丁...")
            result = quick_fix_ml_scorer(
                ml_scorer=ml_scorer,
                factor_data=factor_data,
                price_data=price_data,
                factor_columns=factor_columns
            )

            # 验证修复结果
            if result is None:
                raise ValueError("修复补丁返回 None")

            if 'ml_score' not in result.columns:
                raise ValueError("修复结果缺少 ml_score 列")

            # 检查最新日期的评分
            latest_date = result['date'].max()
            latest_scores = result[result['date'] == latest_date]
            valid_scores = latest_scores['ml_score'].notna().sum()

            if valid_scores == 0:
                raise ValueError(f"最新日期 ({latest_date}) 无有效评分")

            print(f"   ✅ 修复补丁应用成功:")
            print(f"      • 最新日期: {latest_date}")
            print(f"      • 有效评分: {valid_scores}/{len(latest_scores)} ({valid_scores / len(latest_scores):.1%})")

            return result

        except Exception as e:
            print(f"   ⚠️  修复补丁失败: {e}")
            print("   🔄 切换到 Fallback 方法...")
            return fallback_predict_scores(ml_scorer, factor_data, factor_columns)

    else:
        # 修复补丁不可用，直接使用 fallback
        print("   ℹ️  修复补丁未加载，使用 Fallback 方法")
        return fallback_predict_scores(ml_scorer, factor_data, factor_columns)


def validate_ml_score_coverage(factor_data: pd.DataFrame,
                               min_coverage: float = 0.5) -> bool:
    """
    ✅ 验证 ml_score 覆盖率

    Args:
        factor_data: 因子数据
        min_coverage: 最小覆盖率阈值

    Returns:
        bool: 是否通过验证
    """
    if 'ml_score' not in factor_data.columns:
        print("   ❌ 验证失败：缺少 ml_score 列")
        return False

    # 按日期统计覆盖率
    daily_coverage = factor_data.groupby('date').apply(
        lambda x: x['ml_score'].notna().sum() / len(x)
    )

    # 最新日期的覆盖率
    latest_date = factor_data['date'].max()
    latest_coverage = daily_coverage.iloc[-1] if not daily_coverage.empty else 0

    # 总体覆盖率
    total_coverage = factor_data['ml_score'].notna().sum() / len(factor_data)

    print(f"\n   📊 ML评分覆盖率验证:")
    print(f"      • 最新日期 ({latest_date}): {latest_coverage:.1%}")
    print(f"      • 总体覆盖率: {total_coverage:.1%}")
    print(f"      • 低覆盖日期数: {(daily_coverage < min_coverage).sum()}/{len(daily_coverage)}")

    if latest_coverage < min_coverage:
        print(f"   ⚠️  警告：最新日期覆盖率低于阈值 {min_coverage:.1%}")
        return False

    print(f"   ✅ 验证通过：覆盖率符合要求")
    return True


def emergency_repair_ml_score(factor_data: pd.DataFrame,
                              factor_columns: List[str]) -> pd.DataFrame:
    """
    🚑 紧急修复：当所有方法都失败时

    使用简单的因子均值作为评分
    """
    print("\n   🚨 启动紧急修复...")

    data = factor_data.copy()

    # 选择有效的数值因子
    valid_factors = []
    for col in factor_columns:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            valid_factors.append(col)

    if not valid_factors:
        print("   ❌ 无有效因子，无法生成评分")
        data['ml_score'] = np.random.rand(len(data))
        data['position'] = data.groupby('date')['ml_score'].rank(pct=True)
        return data

    print(f"   ℹ️  使用 {len(valid_factors)} 个因子计算等权评分")

    # 计算等权评分
    data['ml_score'] = data[valid_factors].mean(axis=1)
    data['position'] = data.groupby('date')['ml_score'].rank(pct=True)

    # 验证
    valid_count = data['ml_score'].notna().sum()
    print(f"   ✓ 紧急修复完成: {valid_count}/{len(data)} ({valid_count / len(data):.1%})")

    return data


# ============ 使用示例 ============

def example_usage():
    """
    使用示例（在 main.py 中替换相应代码）
    """
    print("""
    # ========== 在 main.py 的步骤4中替换 ==========

    # 原代码（第310-330行左右）:
    if ML_FIX_AVAILABLE:
        factor_data = quick_fix_ml_scorer(
            ml_scorer=ml_scorer,
            factor_data=factor_data,
            price_data=price_data,
            factor_columns=factor_columns
        )

    # 替换为：
    if ML_FIX_AVAILABLE:
        from ml_scorer_emergency_fix import safe_apply_ml_fix, validate_ml_score_coverage

        factor_data = safe_apply_ml_fix(
            ml_scorer=ml_scorer,
            factor_data=factor_data,
            price_data=price_data,
            factor_columns=factor_columns,
            ML_FIX_AVAILABLE=ML_FIX_AVAILABLE
        )

        # 可选：验证覆盖率
        validate_ml_score_coverage(factor_data, min_coverage=0.5)
    """)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ML评分器紧急修复补丁 - 使用指南")
    print("=" * 80)
    example_usage()
    print("\n" + "=" * 80)
    print("💡 关键特性:")
    print("  1. ✅ 多重保障机制（修复补丁 → Fallback → 紧急备用）")
    print("  2. ✅ 详细的状态诊断")
    print("  3. ✅ 自动错误恢复")
    print("  4. ✅ 确保 ml_score 列始终存在")
    print("  5. ✅ 覆盖率验证")
    print("=" * 80)