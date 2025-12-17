"""
config.py - 统一配置文件 v2.0
集中管理所有参数，方便调整

版本：v2.0 - 整合最佳现金管理
日期：2025-12-09
新增：现金管理参数、策略版本选择、资金利用率配置
"""

import os
from datetime import datetime, timedelta

# ========== Tushare配置 ==========
TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"

# ========== 策略版本配置 ==========
class StrategyConfig:
    """策略版本配置"""

    # 策略版本
    STRATEGY_VERSION = "v2.0"  # "v1.0" 或 "v2.0"

    # v2.0 新增：最佳现金管理参数
    CASH_RESERVE_RATIO = 0.05  # 现金保留比例（5%）

    """
    现金保留比例建议：
    - 激进型（高胜率策略）: 0.02 (98%仓位)
    - 平衡型（推荐）: 0.05 (95%仓位)
    - 保守型（波动市场）: 0.10 (90%仓位)
    """

    # 调试模式
    DEBUG_MODE = False  # 是否显示详细交易日志


# ========== 回测参数 ==========
class BacktestConfig:
    """回测配置"""

    # 日期范围
    START_DATE = "2023-01-01"
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    # 资金配置
    CAPITAL_BASE = 1000000  # 初始资金

    # 持仓配置
    POSITION_SIZE = 10  # 持仓数量
    REBALANCE_DAYS = 5  # 调仓周期（天）
    POSITION_METHOD = 'equal'  # 仓位分配方法（v1.0使用）

    # 风险控制（通用）
    STOP_LOSS = -0.15  # 止损阈值
    TAKE_PROFIT = None  # 止盈阈值（None=不止盈）
    SCORE_THRESHOLD = 0.10  # 换仓阈值（v1.0使用）
    SCORE_DECAY_RATE = 1.0  # 评分衰减率（v1.0使用）
    FORCE_REPLACE_DAYS = 20  # 强制换仓天数（v1.0使用）

    # 打印控制
    PRINT_INTERVAL = 5  # 每N天打印一次


# ========== 因子风控参数 ==========
class RiskControlConfig:
    """因子风控配置（适用于v1.0和v2.0）"""

    # 1. 因子衰减止损
    ENABLE_SCORE_DECAY_STOP = True  # 启用因子衰减止损
    SCORE_DECAY_THRESHOLD = 0.30  # 评分下降30%止损
    MIN_HOLDING_DAYS = 5  # 最少持有5天

    # 2. 相对排名止损
    ENABLE_RANK_STOP = True  # 启用相对排名止损
    RANK_PERCENTILE_THRESHOLD = 0.80  # 跌出前70%止损

    # 3. 组合回撤保护
    MAX_PORTFOLIO_DRAWDOWN = -0.15  # 组合回撤-15%降仓
    REDUCE_POSITION_RATIO = 0.5  # 降仓到50%

    # 4. 行业轮动控制
    ENABLE_INDUSTRY_ROTATION = True  # 启用行业轮动
    MAX_INDUSTRY_WEIGHT = 0.40  # 单行业最大40%

    # 5. 极端亏损保护
    EXTREME_LOSS_THRESHOLD = -0.10  # 单股极端亏损-20%
    PORTFOLIO_LOSS_THRESHOLD = -0.25  # 组合极端亏损-25%


# ========== 交易成本配置 ==========
class TradingCostConfig:
    """交易成本配置"""

    BUY_COST = 0.0003  # 买入佣金（万3）
    SELL_COST = 0.0003  # 卖出佣金（万3）
    TAX_RATIO = 0.0005  # 印花税（千分之一，仅卖出）

    # 滑点（可选）
    SLIPPAGE = 0.0  # 滑点比例


# ========== 数据配置 ==========
class DataConfig:
    """数据配置"""

    # 缓存目录
    CACHE_DIR = './data_cache'

    # 股票选择
    USE_SAMPLING = False  # 是否使用智能抽样
    SAMPLE_SIZE = 4000  # 抽样数量
    MAX_STOCKS = 5000  # 不抽样时的最大股票数

    # 性能优化
    MAX_WORKERS = 10  # 并行线程数
    FORCE_FULL_UPDATE = False  # 是否强制全量更新


# ========== 因子配置 ==========
class FactorConfig:
    """因子配置"""

    # 模型选择
    USE_STOCKRANKER = True  # 使用StockRanker模型
    USE_FUNDAMENTAL = True  # 使用基本面因子

    # 自定义权重（None=使用默认）
    CUSTOM_WEIGHTS = None

    # IC调整
    ENABLE_IC_ADJUSTMENT = True  # 启用IC动态调权
    IC_ADJUSTMENT_DECAY = 0.7  # IC调权衰减系数


# ========== 高级ML配置 ==========
class MLConfig:
    """高级ML配置"""

    # ML开关
    USE_ADVANCED_ML = True  # 启用高级ML评分

    # 模型参数
    ML_MODEL_TYPE = 'xgboost'  # 模型类型：'xgboost', 'lightgbm', 'random_forest'
    ML_TARGET_PERIOD = 5  # 预测周期（天）
    ML_TOP_PERCENTILE = 0.20  # 预测TOP 20%

    # 训练参数
    ML_USE_CLASSIFICATION = True  # 使用分类模型（预测TOP股票）
    ML_USE_IC_FEATURES = True  # 使用IC加权特征
    ML_TRAIN_MONTHS = 12  # Walk-Forward训练窗口（月）

    # 选股参数
    ML_MIN_SCORE = 0.6  # 最低评分阈值

    """机器学习配置"""
    USE_ADVANCED_ML = True
    ML_MODEL_TYPE = 'xgboost'  # 'xgboost' 或 'lightgbm'
    ML_TARGET_PERIOD = 5
    ML_TOP_PERCENTILE = 0.20
    ML_USE_CLASSIFICATION = True
    ML_USE_IC_FEATURES = True
    ML_TRAIN_MONTHS = 12
    
    # ========== 新增配置 ==========
    ML_MODEL_DIR = './models'        # 模型保存目录
    ML_AUTO_SAVE = True              # 是否自动保存模型
    ML_PREDICT_ONLY = True          # 是否仅预测模式
    ML_FORCE_RETRAIN = False         # 是否强制重新训练

# ========== 遗传算法配置 ==========
class GeneticConfig:
    """遗传算法配置"""

    # 种群参数
    GENERATIONS = 30  # 迭代代数
    POPULATION_SIZE = 50  # 种群大小

    # 遗传操作
    CROSSOVER_PROB = 0.7  # 交叉概率
    MUTATION_PROB = 0.3  # 变异概率

    # 优化目标权重
    FITNESS_WEIGHTS = (0.25, 0.35, 0.25, 0.15)  # 收益,夏普,回撤,胜率


# ========== 输出配置 ==========
class OutputConfig:
    """输出配置"""

    # 目录
    REPORTS_DIR = './reports'
    OPTIMIZATION_DIR = './optimization_results'

    # 文件名
    MONITORING_DASHBOARD = 'monitoring_dashboard.png'
    TOP_STOCKS_ANALYSIS = 'top_stocks_analysis.png'
    DAILY_HOLDINGS_DETAIL = 'daily_holdings_detail.csv'
    PERFORMANCE_REPORT = 'performance_report.txt'


# ========== 通知配置 ==========
class NotificationConfig:
    """通知配置"""

    # 邮件配置
    EMAIL_ENABLED = False
    EMAIL_SENDER = "your_email@example.com"
    EMAIL_PASSWORD = "your_password"
    EMAIL_RECEIVER = "receiver@example.com"
    EMAIL_SMTP_SERVER = "smtp.example.com"
    EMAIL_SMTP_PORT = 465

    # 企业微信
    WECHAT_ENABLED = False
    WECHAT_WEBHOOK = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxx"

    # 钉钉
    DINGTALK_ENABLED = False
    DINGTALK_WEBHOOK = "https://oapi.dingtalk.com/robot/send?access_token=xxx"


# ========== 便捷函数 ==========
def get_config(config_class):
    """获取配置字典"""
    return {
        k: v for k, v in config_class.__dict__.items()
        if not k.startswith('_')
    }


def print_all_configs():
    """打印所有配置"""
    print("\n" + "=" * 80)
    print("📋 当前配置 v2.0 - 整合最佳现金管理")
    print("=" * 80)

    print("\n【策略版本】⭐")
    for k, v in get_config(StrategyConfig).items():
        if k == 'CASH_RESERVE_RATIO':
            print(f"  {k}: {v:.1%} (目标资金利用率: {1-v:.1%})")
        else:
            print(f"  {k}: {v}")

    print("\n【回测参数】")
    for k, v in get_config(BacktestConfig).items():
        print(f"  {k}: {v}")

    print("\n【因子风控】🎯")
    for k, v in get_config(RiskControlConfig).items():
        print(f"  {k}: {v}")

    print("\n【交易成本】")
    for k, v in get_config(TradingCostConfig).items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f} ({v*10000:.1f}bp)")
        else:
            print(f"  {k}: {v}")

    print("\n【数据配置】")
    for k, v in get_config(DataConfig).items():
        print(f"  {k}: {v}")

    print("\n【因子配置】")
    for k, v in get_config(FactorConfig).items():
        print(f"  {k}: {v}")

    print("\n【高级ML】🤖")
    for k, v in get_config(MLConfig).items():
        print(f"  {k}: {v}")

    print("\n【遗传算法】")
    for k, v in get_config(GeneticConfig).items():
        print(f"  {k}: {v}")

    print("\n【输出配置】")
    for k, v in get_config(OutputConfig).items():
        print(f"  {k}: {v}")

    print()


def validate_configs():
    """验证配置参数的合理性"""
    warnings = []

    # 验证现金保留比例
    if not (0 <= StrategyConfig.CASH_RESERVE_RATIO <= 0.20):
        warnings.append(f"⚠️  现金保留比例 {StrategyConfig.CASH_RESERVE_RATIO:.1%} 超出推荐范围 [0%, 20%]")

    # 验证持仓数量
    if BacktestConfig.POSITION_SIZE < 5 or BacktestConfig.POSITION_SIZE > 30:
        warnings.append(f"⚠️  持仓数量 {BacktestConfig.POSITION_SIZE} 超出推荐范围 [5, 30]")

    # 验证调仓周期
    if BacktestConfig.REBALANCE_DAYS < 1 or BacktestConfig.REBALANCE_DAYS > 20:
        warnings.append(f"⚠️  调仓周期 {BacktestConfig.REBALANCE_DAYS} 超出推荐范围 [1, 20]天")

    # 验证风控参数
    if RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN > -0.05:
        warnings.append(f"⚠️  组合回撤保护 {RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN:.1%} 可能过于宽松")

    if RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN < -0.30:
        warnings.append(f"⚠️  组合回撤保护 {RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN:.1%} 可能过于严格")

    # 打印警告
    if warnings:
        print("\n" + "=" * 80)
        print("⚠️  配置验证警告")
        print("=" * 80)
        for warning in warnings:
            print(f"  {warning}")
        print()
    else:
        print("\n✅ 配置验证通过")


def get_strategy_params():
    """获取策略运行所需的完整参数（便捷函数）"""
    return {
        # 基础参数
        'start_date': BacktestConfig.START_DATE,
        'end_date': BacktestConfig.END_DATE,
        'capital_base': BacktestConfig.CAPITAL_BASE,
        'position_size': BacktestConfig.POSITION_SIZE,
        'rebalance_days': BacktestConfig.REBALANCE_DAYS,

        # v2.0 新增
        'cash_reserve_ratio': StrategyConfig.CASH_RESERVE_RATIO,

        # 风控参数
        'enable_score_decay_stop': RiskControlConfig.ENABLE_SCORE_DECAY_STOP,
        'score_decay_threshold': RiskControlConfig.SCORE_DECAY_THRESHOLD,
        'min_holding_days': RiskControlConfig.MIN_HOLDING_DAYS,
        'enable_rank_stop': RiskControlConfig.ENABLE_RANK_STOP,
        'rank_percentile_threshold': RiskControlConfig.RANK_PERCENTILE_THRESHOLD,
        'max_portfolio_drawdown': RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN,
        'reduce_position_ratio': RiskControlConfig.REDUCE_POSITION_RATIO,
        'enable_industry_rotation': RiskControlConfig.ENABLE_INDUSTRY_ROTATION,
        'max_industry_weight': RiskControlConfig.MAX_INDUSTRY_WEIGHT,
        'extreme_loss_threshold': RiskControlConfig.EXTREME_LOSS_THRESHOLD,
        'portfolio_loss_threshold': RiskControlConfig.PORTFOLIO_LOSS_THRESHOLD,

        # 交易成本
        'buy_cost': TradingCostConfig.BUY_COST,
        'sell_cost': TradingCostConfig.SELL_COST,
        'tax_ratio': TradingCostConfig.TAX_RATIO,

        # 调试
        'debug': StrategyConfig.DEBUG_MODE,
    }


def print_config_comparison():
    """打印v1.0和v2.0配置对比"""
    print("\n" + "=" * 80)
    print("📊 v1.0 vs v2.0 配置对比")
    print("=" * 80)

    print("\n主要差异：")
    print(f"  策略版本: {StrategyConfig.STRATEGY_VERSION}")
    print(f"\n  【v2.0 新增特性】")
    print(f"    ✨ 现金保留比例: {StrategyConfig.CASH_RESERVE_RATIO:.1%}")
    print(f"    ✨ 目标资金利用率: {1-StrategyConfig.CASH_RESERVE_RATIO:.1%}")
    print(f"    ✨ 仓位分配: 动态等权（从剩余现金中等分）")
    print(f"    ✨ 预期改进: 资金利用率提升50%+")

    print(f"\n  【v1.0 特有参数】")
    print(f"    • 仓位方法: {BacktestConfig.POSITION_METHOD}")
    print(f"    • 评分衰减率: {BacktestConfig.SCORE_DECAY_RATE}")
    print(f"    • 强制换仓天数: {BacktestConfig.FORCE_REPLACE_DAYS}")

    print(f"\n  【通用参数】（两版本共享）")
    print(f"    • 持仓数量: {BacktestConfig.POSITION_SIZE}")
    print(f"    • 调仓周期: {BacktestConfig.REBALANCE_DAYS}天")
    print(f"    • 因子衰减止损: {RiskControlConfig.ENABLE_SCORE_DECAY_STOP}")
    print(f"    • 相对排名止损: {RiskControlConfig.ENABLE_RANK_STOP}")
    print()


# ========== 配置预设 ==========
class ConfigPresets:
    """配置预设（快速切换场景）"""

    @staticmethod
    def aggressive():
        """激进型配置"""
        StrategyConfig.CASH_RESERVE_RATIO = 0.02
        BacktestConfig.POSITION_SIZE = 15
        RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN = -0.20
        print("✓ 已切换到【激进型】配置")

    @staticmethod
    def balanced():
        """平衡型配置（默认）"""
        StrategyConfig.CASH_RESERVE_RATIO = 0.05
        BacktestConfig.POSITION_SIZE = 10
        RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN = -0.15
        print("✓ 已切换到【平衡型】配置")

    @staticmethod
    def conservative():
        """保守型配置"""
        StrategyConfig.CASH_RESERVE_RATIO = 0.10
        BacktestConfig.POSITION_SIZE = 8
        RiskControlConfig.MAX_PORTFOLIO_DRAWDOWN = -0.10
        print("✓ 已切换到【保守型】配置")


if __name__ == "__main__":
    print_all_configs()
    validate_configs()
    print_config_comparison()

    # 测试配置预设
    print("\n" + "=" * 80)
    print("🎛️  配置预设测试")
    print("=" * 80)
    print("\n原始配置:")
    print(f"  现金保留: {StrategyConfig.CASH_RESERVE_RATIO:.1%}")
    print(f"  持仓数量: {BacktestConfig.POSITION_SIZE}")

    print("\n切换到激进型:")
    ConfigPresets.aggressive()
    print(f"  现金保留: {StrategyConfig.CASH_RESERVE_RATIO:.1%}")
    print(f"  持仓数量: {BacktestConfig.POSITION_SIZE}")

    print("\n切换回平衡型:")
    ConfigPresets.balanced()
    print(f"  现金保留: {StrategyConfig.CASH_RESERVE_RATIO:.1%}")
    print(f"  持仓数量: {BacktestConfig.POSITION_SIZE}")