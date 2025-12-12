"""
RL模型离线训练脚本
单独运行，训练好后保存模型
"""

import torch
import random
import numpy as np
from advanced_modules.rl.deep_rl_trading_engine import RLStrategyIntegrator
from data_module_incremental import load_data_with_incremental_update

# 加载数据
print("加载训练数据...")
factor_data, price_data = load_data_with_incremental_update(
    '2023-01-01',
    '2025-12-10',
    max_stocks=100
)

# 创建集成器
print("初始化RL集成器...")
integrator = RLStrategyIntegrator(factor_data, price_data)

# 训练
print("开始训练...")
results = integrator.train_agent(episodes=1000)

print(f"\n训练完成!")
print(f"  最终收益: {results['final_return']:+.2%}")

# 保存模型
if hasattr(integrator.agent, 'policy_net'):
    torch.save(
        integrator.agent.policy_net.state_dict(),
        'rl_policy_model.pth'
    )
    print("  ✅ 模型已保存: rl_policy_model.pth")