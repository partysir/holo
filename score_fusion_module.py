"""
score_fusion_module.py - 评分融合模块 (修复核心逻辑)

核心思想：
1. StockRanker生成 stockranker_score
2. ML生成 ml_score
3. 融合为最终的 position

作者: Claude
日期: 2025-01-02
"""

import pandas as pd
import numpy as np


class ScoreFusionEngine:
    """评分融合引擎 - 解决评分混乱问题"""

    def __init__(self, fusion_method='weighted', alpha=0.4, beta=0.6):
        """
        Args:
            fusion_method: 融合方法
                - 'weighted': 加权融合 (推荐)
                - 'rank_average': 排名平均
                - 'ml_only': 仅用ML (当前main-2.py的行为)
                - 'stockranker_only': 仅用StockRanker
            alpha: StockRanker权重
            beta: ML权重 (alpha + beta = 1.0)
        """
        self.fusion_method = fusion_method
        self.alpha = alpha
        self.beta = beta

        if abs(alpha + beta - 1.0) > 0.01:
            print(f"⚠️  警告: alpha({alpha}) + beta({beta}) != 1.0，自动归一化")
            total = alpha + beta
            self.alpha = alpha / total
            self.beta = beta / total

        print(f"\n🔗 评分融合引擎初始化")
        print(f"  方法: {fusion_method}")
        if fusion_method == 'weighted':
            print(f"  权重: StockRanker({self.alpha:.1%}) + ML({self.beta:.1%})")

    def fuse_scores(self, factor_data, has_ml=True):
        """
        融合StockRanker和ML评分

        Args:
            factor_data: DataFrame，必须包含以下列之一：
                - stockranker_score: StockRanker的原始评分
                - ml_score: ML模型的预测评分
            has_ml: 是否有ML评分

        Returns:
            factor_data: 添加了最终position列
        """
        print("\n🔗 开始评分融合...")

        # 检查必需的列
        required_cols = []
        if 'stockranker_score' in factor_data.columns:
            required_cols.append('stockranker_score')
        if has_ml and 'ml_score' in factor_data.columns:
            required_cols.append('ml_score')

        if not required_cols:
            print("❌ 错误: 没有找到任何评分列 (stockranker_score 或 ml_score)")
            # 降级: 使用简单的因子平均
            return self._fallback_scoring(factor_data)

        # 根据方法融合
        if self.fusion_method == 'weighted':
            factor_data = self._weighted_fusion(factor_data, has_ml)

        elif self.fusion_method == 'rank_average':
            factor_data = self._rank_average_fusion(factor_data, has_ml)

        elif self.fusion_method == 'ml_only':
            if 'ml_score' in factor_data.columns:
                factor_data['position'] = factor_data.groupby('date')['ml_score'].rank(pct=True)
            else:
                print("⚠️  ml_only模式但缺少ml_score，降级使用stockranker_score")
                factor_data['position'] = factor_data.groupby('date')['stockranker_score'].rank(pct=True)

        elif self.fusion_method == 'stockranker_only':
            if 'stockranker_score' in factor_data.columns:
                factor_data['position'] = factor_data.groupby('date')['stockranker_score'].rank(pct=True)
            else:
                print("⚠️  stockranker_only模式但缺少stockranker_score，降级使用ml_score")
                factor_data['position'] = factor_data.groupby('date')['ml_score'].rank(pct=True)

        # 验证结果
        self._validate_position(factor_data)

        print(f"✅ 评分融合完成")
        print(f"  最终position范围: [{factor_data['position'].min():.4f}, {factor_data['position'].max():.4f}]")

        return factor_data

    def _weighted_fusion(self, df, has_ml):
        """加权融合"""
        if has_ml and 'ml_score' in df.columns and 'stockranker_score' in df.columns:
            # 两种评分都有 - 加权融合
            df['fused_score'] = (
                    self.alpha * df['stockranker_score'] +
                    self.beta * df['ml_score']
            )
            print(f"  使用加权融合: {self.alpha:.1%} × StockRanker + {self.beta:.1%} × ML")

        elif 'ml_score' in df.columns:
            # 只有ML
            df['fused_score'] = df['ml_score']
            print(f"  仅使用ML评分")

        elif 'stockranker_score' in df.columns:
            # 只有StockRanker
            df['fused_score'] = df['stockranker_score']
            print(f"  仅使用StockRanker评分")

        else:
            raise ValueError("无可用评分")

        # 转为排名百分位
        df['position'] = df.groupby('date')['fused_score'].rank(pct=True)

        return df

    def _rank_average_fusion(self, df, has_ml):
        """排名平均融合"""
        if has_ml and 'ml_score' in df.columns and 'stockranker_score' in df.columns:
            # 计算各自的排名
            df['rank_sr'] = df.groupby('date')['stockranker_score'].rank(pct=True)
            df['rank_ml'] = df.groupby('date')['ml_score'].rank(pct=True)

            # 排名平均
            df['position'] = (df['rank_sr'] + df['rank_ml']) / 2

            # 清理
            df.drop(columns=['rank_sr', 'rank_ml'], inplace=True)

            print(f"  使用排名平均融合")

        elif 'ml_score' in df.columns:
            df['position'] = df.groupby('date')['ml_score'].rank(pct=True)
            print(f"  仅使用ML排名")

        elif 'stockranker_score' in df.columns:
            df['position'] = df.groupby('date')['stockranker_score'].rank(pct=True)
            print(f"  仅使用StockRanker排名")

        return df

    def _fallback_scoring(self, df):
        """降级评分 - 当没有任何评分时"""
        print("⚠️  降级: 使用简单因子平均")

        # 找数值列（排除元数据）
        exclude_cols = ['date', 'instrument', 'industry', 'open', 'high',
                        'low', 'close', 'volume', 'amount']
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in exclude_cols]

        if len(numeric_cols) > 0:
            # 标准化后平均
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()

            df['fallback_score'] = scaler.fit_transform(
                df[numeric_cols].fillna(0)
            ).mean(axis=1)

            df['position'] = df.groupby('date')['fallback_score'].rank(pct=True)
            df.drop(columns=['fallback_score'], inplace=True)
        else:
            # 彻底失败 - 随机评分
            df['position'] = 0.5

        return df

    def _validate_position(self, df):
        """验证position列的正确性"""
        if 'position' not in df.columns:
            raise ValueError("❌ 融合失败: 未生成position列")

        # 检查范围
        if df['position'].min() < 0 or df['position'].max() > 1:
            print(f"⚠️  警告: position超出[0,1]范围: [{df['position'].min()}, {df['position'].max()}]")

        # 检查缺失值
        null_count = df['position'].isna().sum()
        if null_count > 0:
            print(f"⚠️  警告: position有{null_count}个缺失值")


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""

    # 模拟数据
    factor_data = pd.DataFrame({
        'date': ['2024-01-01'] * 10,
        'instrument': [f'00000{i}.SZ' for i in range(10)],
        'stockranker_score': np.random.rand(10),
        'ml_score': np.random.rand(10)
    })

    # 创建融合引擎
    fusion_engine = ScoreFusionEngine(
        fusion_method='weighted',
        alpha=0.4,  # StockRanker权重40%
        beta=0.6  # ML权重60%
    )

    # 融合评分
    result = fusion_engine.fuse_scores(factor_data, has_ml=True)

    print("\n融合结果:")
    print(result[['instrument', 'stockranker_score', 'ml_score', 'position']].head())


if __name__ == "__main__":
    example_usage()