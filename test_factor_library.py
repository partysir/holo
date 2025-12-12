from advanced_modules.factors.advanced_factor_library import AdvancedFactorLibrary

# 使用你的真实数据
from data_module_incremental import load_data_with_incremental_update

# 加载数据
factor_data, price_data = load_data_with_incremental_update(
    '2023-01-01',
    '2025-12-01',
    max_stocks=50
)

# 初始化因子库
factor_lib = AdvancedFactorLibrary()

# 计算因子
factor_values = factor_lib.calculate_all_factors(price_data)

# 评估因子
eval_df = factor_lib.evaluate_factors(factor_values, price_data)

# 自动剔除失效因子
factor_lib.auto_prune_factors()

# 因子组合
combined_score = factor_lib.combine_factors(
    factor_values,
    method='ic_weight',
    eval_df=eval_df
)

print("✅ 因子库测试成功!")