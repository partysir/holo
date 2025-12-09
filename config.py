"""
config.py - 统一配置文件
集中管理所有参数，方便调整
"""

import os
from datetime import datetime, timedelta

# ========== Tushare配置 ==========
TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"

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

    # 风险控制
    STOP_LOSS = -0.15  # 止损阈值
    TAKE_PROFIT = None  # 止盈阈值（None=不止盈）
    SCORE_THRESHOLD = 0.10  # 换仓阈值

    # 打印控制
    PRINT_INTERVAL = 5  # 每N天打印一次


# ========== 数据配置 ==========
class DataConfig:
    """数据配置"""

    # 缓存目录
    CACHE_DIR = './data_cache'

    # 股票选择
    USE_SAMPLING = False  # 是否使用智能抽样
    SAMPLE_SIZE = 3950  # 抽样数量
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
    print("📋 当前配置")
    print("=" * 80)

    print("\n【回测参数】")
    for k, v in get_config(BacktestConfig).items():
        print(f"  {k}: {v}")

    print("\n【数据配置】")
    for k, v in get_config(DataConfig).items():
        print(f"  {k}: {v}")

    print("\n【因子配置】")
    for k, v in get_config(FactorConfig).items():
        print(f"  {k}: {v}")

    print("\n【遗传算法】")
    for k, v in get_config(GeneticConfig).items():
        print(f"  {k}: {v}")

    print()


if __name__ == "__main__":
    print_all_configs()