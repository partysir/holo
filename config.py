"""
config.py - 统一配置文件 v2.3 (性能优化版)
集中管理所有参数，方便调整

版本：v2.3 - 优化风控参数解决收益过低问题
日期：2025-12-29

核心优化：
  ✅ 放宽止损条件，减少误杀
  ✅ 延长持仓时间，降低交易频率
  ✅ 提高资金使用率
  ✅ 关闭过于激进的排名止损
"""

import os
from datetime import datetime, timedelta

# ========== Tushare配置 ==========
TUSHARE_TOKEN = "2876ea85cb005fb5fa17c809a98174f2d5aae8b1f830110a5ead6211"

# ========== 策略版本配置 ==========
class StrategyConfig:
    """策略版本配置"""
    # 策略版本
    STRATEGY_VERSION = "v2.0"

    # v2.0 新增：最佳现金管理参数
    CASH_RESERVE_RATIO = 0.02  # ✅ 优化：现金保留比例改为2%（原5%），提高资金使用率
    """
    现金保留比例建议：
    - 激进型（高胜率策略）: 0.02 (98%仓位) ← 当前
    - 平衡型（推荐）: 0.05 (95%仓位)
    - 保守型（波动市场）: 0.10 (90%仓位)
    """

    # 调试模式
    DEBUG_MODE = False

    # 大盘择时
    ENABLE_MARKET_TIMING = False  # 暂时关闭择时，避免频繁空仓


# ========== 回测参数 ==========
class BacktestConfig:
    """回测配置"""
    # 日期范围
    START_DATE = "2023-01-01"
    END_DATE = datetime.now().strftime('%Y-%m-%d')

    # 资金配置
    CAPITAL_BASE = 1000000

    # 持仓配置
    POSITION_SIZE = 10
    REBALANCE_DAYS = 15  # ✅ 优化：调仓周期改为15天（原10天）
    POSITION_METHOD = 'equal'

    # 风险控制（通用）
    STOP_LOSS = -0.15
    TAKE_PROFIT = None
    SCORE_THRESHOLD = 0.10
    SCORE_DECAY_RATE = 1.0
    FORCE_REPLACE_DAYS = 20

    # 打印控制
    PRINT_INTERVAL = 5


# ========== 因子风控参数 ==========
class RiskControlConfig:
    """因子风控配置（适用于v1.0和v2.0）"""
    # 1. 因子衰减止损
    ENABLE_SCORE_DECAY_STOP = True
    SCORE_DECAY_THRESHOLD = 0.50  # ✅ 优化：评分下降50%才止损（原40%）
    MIN_HOLDING_DAYS = 8  # ✅ 优化：最少持有8天（原5天）

    # 2. 相对排名止损
    ENABLE_RANK_STOP = False  # ✅ 优化：暂时关闭排名止损（原True）
    RANK_PERCENTILE_THRESHOLD = 0.90  # 优化：跌出前90%才止损（原80%）

    # 3. 组合回撤保护
    MAX_PORTFOLIO_DRAWDOWN = -0.20  # ✅ 优化：组合回撤-20%降仓（原-15%）
    REDUCE_POSITION_RATIO = 0.7  # ✅ 优化：降仓到70%（原50%）

    # 4. 行业轮动控制
    ENABLE_INDUSTRY_ROTATION = True
    MAX_INDUSTRY_WEIGHT = 0.50  # ✅ 优化：单行业最大50%（原40%）

    # 5. 极端亏损保护
    EXTREME_LOSS_THRESHOLD = -0.20  # ✅ 优化：单股极端亏损-20%（原-15%）
    PORTFOLIO_LOSS_THRESHOLD = -0.30  # ✅ 优化：组合极端亏损-30%（原-25%）


# ========== 交易成本配置 ==========
class TradingCostConfig:
    """交易成本配置"""
    BUY_COST = 0.0003  # 买入佣金（万3）
    SELL_COST = 0.0003  # 卖出佣金（万3）
    TAX_RATIO = 0.0005  # 印花税（千分之0.5，仅卖出）
    SLIPPAGE = 0.0  # 滑点比例


# ========== 数据配置 ==========
class DataConfig:
    """数据配置"""
    # 缓存目录
    CACHE_DIR = './data_cache'

    # 股票选择
    USE_SAMPLING = False
    SAMPLE_SIZE = 4000
    MAX_STOCKS = 5000

    # 性能优化
    MAX_WORKERS = 10
    FORCE_FULL_UPDATE = False


# ========== 因子配置 ==========
class FactorConfig:
    """因子配置"""
    # 模型选择
    USE_STOCKRANKER = True
    USE_FUNDAMENTAL = True
    USE_MONEY_FLOW = True

    # 自定义权重（None=使用默认）
    CUSTOM_WEIGHTS = None

    # IC调整
    ENABLE_IC_ADJUSTMENT = True
    IC_ADJUSTMENT_DECAY = 0.7

    # 资金流因子配置
    MONEY_FLOW_CONFIG = {
        'use_full_tick': False,
        'weight_main_netflow': 0.10,
        'weight_main_strength': 0.08,
        'weight_large_netflow': 0.07,
        'main_continuous_inflow': 0.05,
        'main_vs_retail_ratio': 0.05,
        'main_activity': 0.05,
    }


# ========== 高级ML配置 ==========
class MLConfig:
    """高级ML配置"""
    # ML开关
    USE_ADVANCED_ML = True

    # 模型参数
    ML_MODEL_TYPE = 'xgboost'
    ML_TARGET_PERIOD = 5
    ML_TOP_PERCENTILE = 0.20

    # 训练参数
    ML_USE_CLASSIFICATION = True
    ML_USE_IC_FEATURES = True
    ML_TRAIN_MONTHS = 12

    # 选股参数
    ML_MIN_SCORE = 0.6

    # 文件与模式
    ML_MODEL_DIR = './models'
    ML_AUTO_SAVE = True
    ML_PREDICT_ONLY = False
    ML_FORCE_RETRAIN = False


# ========== 遗传算法配置 ==========
class GeneticConfig:
    """遗传算法配置"""
    # 种群参数
    GENERATIONS = 30
    POPULATION_SIZE = 50

    # 遗传操作
    CROSSOVER_PROB = 0.7
    MUTATION_PROB = 0.3

    # 优化目标权重
    FITNESS_WEIGHTS = (0.25, 0.35, 0.25, 0.15)


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


# ========== 实盘交易配置 ==========
class LiveTradingConfig:
    """实盘交易配置"""

    # 是否启用自动交易
    # ⚠️ 警告：首次使用请设为 False，先检查订单文件
    ENABLE_AUTO_TRADE = False

    # 券商配置
    BROKER = 'guosen'  # 'guosen' | 'gf' | 'ht' | 'yh' | 'yjb'

    # 初始资金（用于计算买入数量）
    INITIAL_CAPITAL = 1000000

    # 最小买入金额
    MIN_BUY_AMOUNT = 5000

    # 账户信息（从环境变量读取，更安全）
    # 使用方法：
    # export BROKER_ACCOUNT="your_account"
    # export BROKER_PASSWORD="your_password"
    ACCOUNT = None  # 留空，从环境变量读取
    PASSWORD = None
    COMM_PASSWORD = None

    # 输出配置
    OUTPUT_DIR = './live_trading'
    SAVE_CSV = True
    SAVE_TXT = True
    SAVE_JSON = True

    # 日志级别
    LOG_LEVEL = 'INFO'  # 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'

    # 是否发送通知
    ENABLE_NOTIFICATION = False

    # 通知方式
    # - 'email': 邮件
    # - 'wechat': 微信（需要企业微信）
    # - 'dingtalk': 钉钉
    NOTIFICATION_METHOD = 'email'

    # ML相关配置
    ML_CACHE_MODELS = True  # 缓存训练好的模型
    ML_TRAIN_MONTHS = 12  # 训练窗口（月）
    USE_ML_SCORING = True  # 启用ML评分

    # 策略参数
    REBALANCE_DAYS = 5  # 调仓周期
    POSITION_SIZE = 10  # 持仓数量

    # 数据配置
    SAMPLE_SIZE = 4000
    USE_SAMPLING = False


# ========== 券商配置模板 ==========
class BrokerConfig:
    """券商API配置"""

    # 国信证券
    GUOSEN = {
        'broker': 'guosen',
        'account': '',
        'password': '',
        'comm_password': '',
        'ip': '',
        'port': 0,
    }

    # 广发证券
    GUANGFA = {
        'broker': 'gf',
        'account': '',
        'password': '',
    }

    # 华泰证券
    HUATAI = {
        'broker': 'ht',
        'account': '',
        'password': '',
    }

    # 银河证券
    YINHE = {
        'broker': 'yh',
        'account': '',
        'password': '',
    }

    # 一键报盘
    YJB = {
        'broker': 'yjb',
        'account': '',
        'password': '',
    }


# ========== 通用函数 ==========
def get_strategy_params():
    """获取策略运行所需的完整参数"""
    return {
        # 基础参数
        'start_date': BacktestConfig.START_DATE,
        'end_date': BacktestConfig.END_DATE,
        'capital_base': BacktestConfig.CAPITAL_BASE,
        'position_size': BacktestConfig.POSITION_SIZE,
        'rebalance_days': BacktestConfig.REBALANCE_DAYS,

        # v2.0 新增
        'cash_reserve_ratio': StrategyConfig.CASH_RESERVE_RATIO,
        
        # 大盘择时
        'enable_market_timing': StrategyConfig.ENABLE_MARKET_TIMING,

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

def get_live_trading_params():
    """获取实盘交易运行所需的完整参数"""
    return {
        # 实盘交易控制
        'enable_auto_trade': LiveTradingConfig.ENABLE_AUTO_TRADE,
        'broker': LiveTradingConfig.BROKER,
        'initial_capital': LiveTradingConfig.INITIAL_CAPITAL,
        'min_buy_amount': LiveTradingConfig.MIN_BUY_AMOUNT,
        'account': LiveTradingConfig.ACCOUNT,
        'password': LiveTradingConfig.PASSWORD,
        'comm_password': LiveTradingConfig.COMM_PASSWORD,

        # 输出配置
        'output_dir': LiveTradingConfig.OUTPUT_DIR,
        'save_csv': LiveTradingConfig.SAVE_CSV,
        'save_txt': LiveTradingConfig.SAVE_TXT,
        'save_json': LiveTradingConfig.SAVE_JSON,
        'log_level': LiveTradingConfig.LOG_LEVEL,
        'enable_notification': LiveTradingConfig.ENABLE_NOTIFICATION,
        'notification_method': LiveTradingConfig.NOTIFICATION_METHOD,
    }