"""
genetic_optimizer.py - 遗传算法参数优化系统

功能:
1. 自动优化策略参数
2. 多目标优化（收益、夏普、回撤、胜率）
3. 自动更新config.py
4. 生成优化报告

使用方法:
python genetic_optimizer.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import time
import json
import os
import re
import random

# 导入回测模块
from data_module import DataCache
from data_module_incremental import load_data_with_incremental_update
from ultimate_fast_system import run_ultimate_fast_backtest


# ========== 遗传算法配置 ==========
class GAConfig:
    """遗传算法配置"""
    POPULATION_SIZE = 20      # 种群大小
    GENERATIONS = 30          # 迭代代数
    CROSSOVER_RATE = 0.8     # 交叉概率
    MUTATION_RATE = 0.2      # 变异概率
    ELITISM_RATE = 0.1       # 精英保留率

    # 适应度权重 (收益, 夏普, 回撤, 胜率)
    FITNESS_WEIGHTS = (0.3, 0.3, 0.25, 0.15)


# ========== 参数定义 ==========
PARAM_BOUNDS = {
    'SCORE_THRESHOLD': (0.08, 0.25),      # 换仓阈值
    'STOP_LOSS': (-0.25, -0.08),          # 止损
    'FORCE_REPLACE_DAYS': (30, 70),       # 强制换仓天数
    'MIN_HOLDING_DAYS': (3, 15),          # 最少持有天数
    'TRANSACTION_COST': (0.0010, 0.0020), # 交易成本
}

PARAM_TYPES = {
    'SCORE_THRESHOLD': 'float',
    'STOP_LOSS': 'float',
    'FORCE_REPLACE_DAYS': 'int',
    'MIN_HOLDING_DAYS': 'int',
    'TRANSACTION_COST': 'float',
}


# ========== 适应度函数 ==========
def backtest_fitness(params_array, factor_data, price_data, start_date, end_date):
    """
    回测适应度函数

    :param params_array: 参数数组
    :return: 适应度值（越大越好）
    """
    # 解析参数
    score_threshold = params_array[0]
    stop_loss = params_array[1]
    force_replace_days = int(params_array[2])
    min_holding_days = int(params_array[3])
    transaction_cost = params_array[4]

    try:
        # 运行回测（静默模式）
        context = run_ultimate_fast_backtest(
            factor_data=factor_data,
            price_data=price_data,
            start_date=start_date,
            end_date=end_date,
            capital_base=1000000,
            position_size=10,
            stop_loss=stop_loss,
            take_profit=None,
            score_threshold=score_threshold,
            max_rebalance_per_day=1,
            force_replace_days=force_replace_days,
            transaction_cost=transaction_cost,
            min_holding_days=min_holding_days,
            dynamic_stop_loss=True,
            silent=True  # 静默模式
        )

        # 计算指标
        total_return = context['total_return']
        win_rate = context['win_rate']

        daily_records = context['daily_records']

        # 最大回撤
        cummax = daily_records['portfolio_value'].cummax()
        drawdown = (daily_records['portfolio_value'] - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        # 夏普比率
        daily_returns = daily_records['portfolio_value'].pct_change().dropna()
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan).dropna()
        daily_returns = daily_returns[np.abs(daily_returns) < 1]

        if len(daily_returns) > 1:
            volatility = daily_returns.std()
            sharpe = (total_return / (len(daily_records)/252) - 0.03) / (volatility * np.sqrt(252)) if volatility > 0 else 0
        else:
            sharpe = 0

        # ========== 综合适应度 ==========
        # 归一化各指标
        return_score = min(total_return / 3.0, 1.0)  # 300%收益为满分
        sharpe_score = min(sharpe / 3.0, 1.0)        # 夏普3为满分
        drawdown_score = max(0, 1 - max_drawdown / 0.4)  # 40%回撤为0分
        winrate_score = min(win_rate / 0.6, 1.0)     # 60%胜率为满分

        # 加权综合
        weights = GAConfig.FITNESS_WEIGHTS
        fitness = (return_score * weights[0] +
                  sharpe_score * weights[1] +
                  drawdown_score * weights[2] +
                  winrate_score * weights[3])

        return fitness, {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trades': len(context['trade_records'][context['trade_records']['action']=='sell'])
        }

    except Exception as e:
        print(f"  ⚠️  回测失败: {e}")
        return 0, {}


# ========== 遗传算法实现 ==========
class GeneticOptimizer:
    """遗传算法优化器"""

    def __init__(self, factor_data, price_data, start_date, end_date):
        self.factor_data = factor_data
        self.price_data = price_data
        self.start_date = start_date
        self.end_date = end_date

        # 参数边界
        self.bounds = np.array([PARAM_BOUNDS[k] for k in sorted(PARAM_BOUNDS.keys())])
        self.param_names = sorted(PARAM_BOUNDS.keys())
        self.n_params = len(self.param_names)

        # 记录
        self.best_fitness_history = []
        self.best_params_history = []
        self.best_details_history = []

    def initialize_population(self):
        """初始化种群"""
        population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (GAConfig.POPULATION_SIZE, self.n_params)
        )
        return population

    def evaluate_population(self, population):
        """评估种群"""
        fitness_values = []
        details_list = []

        print(f"\n  评估种群 ({len(population)} 个体)...")

        for i, individual in enumerate(population):
            fitness, details = backtest_fitness(
                individual, self.factor_data, self.price_data,
                self.start_date, self.end_date
            )
            fitness_values.append(fitness)
            details_list.append(details)

            if (i + 1) % 5 == 0:
                print(f"    进度: {i+1}/{len(population)}")

        return np.array(fitness_values), details_list

    def tournament_selection(self, population, fitness, tournament_size=3):
        """锦标赛选择"""
        selected = []
        for _ in range(GAConfig.POPULATION_SIZE):
            contestants_idx = np.random.choice(
                GAConfig.POPULATION_SIZE,
                tournament_size,
                replace=False
            )
            winner_idx = contestants_idx[np.argmax(fitness[contestants_idx])]
            selected.append(population[winner_idx].copy())
        return np.array(selected)

    def crossover(self, parent1, parent2):
        """混合交叉"""
        if random.random() < GAConfig.CROSSOVER_RATE:
            alpha = 0.5
            child1 = alpha * parent1 + (1 - alpha) * parent2
            child2 = (1 - alpha) * parent1 + alpha * parent2

            # 边界处理
            child1 = np.clip(child1, self.bounds[:, 0], self.bounds[:, 1])
            child2 = np.clip(child2, self.bounds[:, 0], self.bounds[:, 1])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, individual):
        """高斯变异"""
        mutant = individual.copy()
        for i in range(self.n_params):
            if random.random() < GAConfig.MUTATION_RATE:
                sigma = (self.bounds[i, 1] - self.bounds[i, 0]) * 0.1
                mutant[i] += np.random.normal(0, sigma)
                mutant[i] = np.clip(mutant[i], self.bounds[i, 0], self.bounds[i, 1])
        return mutant

    def optimize(self):
        """执行优化"""
        print("\n" + "=" * 80)
        print("🧬 遗传算法参数优化")
        print("=" * 80)
        print(f"  种群大小: {GAConfig.POPULATION_SIZE}")
        print(f"  迭代代数: {GAConfig.GENERATIONS}")
        print(f"  优化参数: {', '.join(self.param_names)}")

        start_time = time.time()

        # 初始化种群
        population = self.initialize_population()
        fitness, details = self.evaluate_population(population)

        for generation in range(GAConfig.GENERATIONS):
            print(f"\n{'='*80}")
            print(f"第 {generation + 1}/{GAConfig.GENERATIONS} 代")
            print(f"{'='*80}")

            # 精英保留
            elite_count = int(GAConfig.POPULATION_SIZE * GAConfig.ELITISM_RATE)
            elite_indices = np.argsort(fitness)[-elite_count:]
            elites = population[elite_indices].copy()

            # 选择
            selected = self.tournament_selection(population, fitness)

            # 交叉和变异
            offspring = []
            for i in range(0, GAConfig.POPULATION_SIZE - elite_count, 2):
                parent1 = selected[i]
                parent2 = selected[min(i + 1, GAConfig.POPULATION_SIZE - 1)]

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                offspring.extend([child1, child2])

            # 组合新种群
            offspring = np.array(offspring[:GAConfig.POPULATION_SIZE - elite_count])
            population = np.vstack([elites, offspring])

            # 评估
            fitness, details = self.evaluate_population(population)

            # 记录
            best_idx = np.argmax(fitness)
            self.best_fitness_history.append(fitness[best_idx])
            self.best_params_history.append(population[best_idx].copy())
            self.best_details_history.append(details[best_idx])

            # 显示最佳个体
            print(f"\n  📊 当代最佳:")
            print(f"     适应度: {fitness[best_idx]:.4f}")
            for j, name in enumerate(self.param_names):
                print(f"     {name}: {population[best_idx][j]:.4f}")

            if details[best_idx]:
                d = details[best_idx]
                print(f"\n     收益率: {d.get('total_return', 0):.2%}")
                print(f"     夏普比率: {d.get('sharpe', 0):.4f}")
                print(f"     最大回撤: {d.get('max_drawdown', 0):.2%}")
                print(f"     胜率: {d.get('win_rate', 0):.2%}")
                print(f"     交易次数: {d.get('trades', 0)}")

        elapsed = time.time() - start_time

        print(f"\n{'='*80}")
        print(f"✅ 优化完成！耗时: {elapsed/60:.1f} 分钟")
        print(f"{'='*80}")

        # 返回最优结果
        best_gen_idx = np.argmax(self.best_fitness_history)
        return {
            'best_params': self.best_params_history[best_gen_idx],
            'best_fitness': self.best_fitness_history[best_gen_idx],
            'best_details': self.best_details_history[best_gen_idx],
            'fitness_history': self.best_fitness_history,
            'param_names': self.param_names
        }


# ========== 更新config.py ==========
def update_config_file(best_params, param_names, backup=True):
    """更新config.py文件"""
    config_path = 'config.py'

    if not os.path.exists(config_path):
        print(f"⚠️  未找到 {config_path}")
        return

    # 备份
    if backup:
        backup_path = f"config.py.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"\n✓ 已备份到: {backup_path}")

    # 创建参数映射
    param_mapping = {name: best_params[i] for i, name in enumerate(param_names)}

    # 读取文件
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 更新参数
    updates_made = []
    for i, line in enumerate(lines):
        for param_name, param_value in param_mapping.items():
            pattern = rf'^\s*{param_name}\s*=\s*(.+?)(?:\s*#.*)?$'
            if re.match(pattern, line):
                old_value = line.strip()

                # 格式化新值
                indent = len(line) - len(line.lstrip())
                comment_match = re.search(r'#.*$', line)
                comment = comment_match.group(0) if comment_match else ""

                if PARAM_TYPES[param_name] == 'int':
                    new_line = f"{' ' * indent}{param_name} = {int(param_value)}  {comment}\n"
                else:
                    new_line = f"{' ' * indent}{param_name} = {param_value:.4f}  {comment}\n"

                lines[i] = new_line
                updates_made.append(f"{param_name}: {old_value.split('=')[1].split('#')[0].strip()} -> {param_value:.4f}")

    # 写回文件
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    # 输出更新信息
    print("\n" + "=" * 80)
    print("✓ config.py 已更新")
    print("=" * 80)
    for update in updates_made:
        print(f"  {update}")
    print()


# ========== 保存优化报告 ==========
def save_optimization_report(result, output_path='./reports/optimization_report.txt'):
    """保存优化报告"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("🧬 遗传算法优化报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"优化时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"迭代代数: {GAConfig.GENERATIONS}\n")
        f.write(f"种群大小: {GAConfig.POPULATION_SIZE}\n\n")

        f.write("最优参数:\n")
        f.write("-" * 80 + "\n")
        for i, name in enumerate(result['param_names']):
            value = result['best_params'][i]
            if PARAM_TYPES[name] == 'int':
                f.write(f"  {name:30s} = {int(value)}\n")
            else:
                f.write(f"  {name:30s} = {value:.4f}\n")

        f.write("\n回测表现:\n")
        f.write("-" * 80 + "\n")
        details = result['best_details']
        f.write(f"  总收益率:   {details.get('total_return', 0):+.2%}\n")
        f.write(f"  夏普比率:   {details.get('sharpe', 0):.4f}\n")
        f.write(f"  最大回撤:   {details.get('max_drawdown', 0):.2%}\n")
        f.write(f"  胜率:       {details.get('win_rate', 0):.2%}\n")
        f.write(f"  交易次数:   {details.get('trades', 0)}\n")

        f.write("\n适应度历史:\n")
        f.write("-" * 80 + "\n")
        for gen, fitness in enumerate(result['fitness_history'], 1):
            f.write(f"  第{gen:2d}代: {fitness:.4f}\n")

    print(f"✓ 优化报告已保存: {output_path}")


# ========== 主函数 ==========
def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🧬 遗传算法参数优化系统")
    print("=" * 80)

    # 1. 加载数据
    print("\n【步骤1/4】加载数据")

    START_DATE = "2023-01-01"
    END_DATE = "2025-12-07"
    SAMPLE_SIZE = 3923

    cache_manager = DataCache(cache_dir='./data_cache')

    factor_data, price_data = load_data_with_incremental_update(
        START_DATE,
        END_DATE,
        cache_manager=cache_manager,
        use_stockranker=True,
        tushare_token="2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211",
        use_fundamental=True,
        use_sampling=True,
        sample_size=SAMPLE_SIZE,
        max_workers=10,
        force_full_update=False
    )

    if factor_data is None or price_data is None:
        print("❌ 数据加载失败")
        return

    print(f"✓ 数据加载完成")

    # 2. 运行遗传算法
    print("\n【步骤2/4】运行遗传算法")

    optimizer = GeneticOptimizer(factor_data, price_data, START_DATE, END_DATE)
    result = optimizer.optimize()

    # 3. 保存报告
    print("\n【步骤3/4】保存报告")
    save_optimization_report(result)

    # 4. 询问是否更新config
    print("\n【步骤4/4】更新配置文件")
    print("\n最优参数:")
    for i, name in enumerate(result['param_names']):
        value = result['best_params'][i]
        if PARAM_TYPES[name] == 'int':
            print(f"  {name}: {int(value)}")
        else:
            print(f"  {name}: {value:.4f}")

    print("\n回测表现:")
    details = result['best_details']
    print(f"  总收益率:   {details.get('total_return', 0):+.2%}")
    print(f"  夏普比率:   {details.get('sharpe', 0):.4f}")
    print(f"  最大回撤:   {details.get('max_drawdown', 0):.2%}")
    print(f"  胜率:       {details.get('win_rate', 0):.2%}")

    response = input("\n是否要将这些参数更新到 config.py？(y/n): ").lower()

    if response == 'y':
        update_config_file(result['best_params'], result['param_names'], backup=True)
        print("\n✅ 完成！配置已更新")
        print("💡 建议: 运行 python main.py 验证新参数")
    else:
        print("\n✗ 已取消更新")

    print("\n" + "=" * 80)
    print("🎉 优化流程完成！")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()