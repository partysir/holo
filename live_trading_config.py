"""
live_trading_config.py - 实盘交易统一配置文件

使用方法：
1. 修改此配置文件
2. 运行 main_live_trading_enhanced.py

版本: v1.0
"""


# ============================================================================
# 基础策略配置
# ============================================================================

class StrategyConfig:
    """策略配置"""

    # 调仓周期（天）
    REBALANCE_DAYS = 5

    # 仓位分配方法
    # - 'equal': 等权
    # - 'score_weighted': 按评分加权
    # - 'volatility_weighted': 按波动率加权
    POSITION_METHOD = 'equal'

    # 持仓数量
    POSITION_SIZE = 10

    # 风控参数
    STOP_LOSS = -0.15  # 单只止损线 -15%
    TAKE_PROFIT = 0.30  # 单只止盈线 +30% (可选)
    MAX_POSITION_RATIO = 0.15  # 单只最大仓位 15%
    SCORE_THRESHOLD = 0.15  # 换仓评分阈值

    # 强制评估周期（天）
    FORCE_REPLACE_DAYS = 45


# ============================================================================
# ML模型配置
# ============================================================================

class MLConfig:
    """ML模型配置"""

    # 是否启用ML评分
    USE_ML_SCORING = True

    # 训练模式
    # - 'rolling': 滚动训练（推荐，适应性强）
    # - 'single': 一次性训练（快速，适合测试）
    TRAINING_MODE = 'rolling'

    # 滚动训练窗口（月）
    TRAIN_MONTHS = 12

    # 预测周期（天）
    TARGET_PERIOD = 5

    # Top股票百分比
    TOP_PERCENTILE = 0.2

    # 模型缓存
    CACHE_MODELS = True

    # 特征工程
    NEUTRALIZE_MARKET = True
    NEUTRALIZE_INDUSTRY = True

    # 集成策略
    VOTING_STRATEGY = 'average'  # 'average' | 'weighted'


# ============================================================================
# 数据配置
# ============================================================================

class DataConfig:
    """数据配置"""

    # 股票池大小
    SAMPLE_SIZE = 3950
    USE_SAMPLING = False

    # 数据源
    USE_STOCKRANKER = True  # 使用StockRanker因子
    USE_FUNDAMENTAL = True  # 使用基本面因子
    USE_MONEY_FLOW = True  # 使用资金流因子

    # 上市时间过滤（天）
    MIN_DAYS_LISTED = 180

    # 缓存配置
    CACHE_DIR = './data_cache'
    FORCE_FULL_UPDATE = False

    # 并行处理
    MAX_WORKERS = 10


# ============================================================================
# 交易成本配置
# ============================================================================

class CostConfig:
    """交易成本配置"""

    # 买入手续费（万3）
    BUY_COST = 0.0003

    # 卖出手续费（万3）
    SELL_COST = 0.0003

    # 印花税（千1）
    TAX_RATIO = 0.0005

    # 滑点（估计值）
    SLIPPAGE = 0.001


# ============================================================================
# 实盘交易配置
# ============================================================================

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


# ============================================================================
# 输出配置
# ============================================================================

class OutputConfig:
    """输出配置"""

    # 输出目录
    OUTPUT_DIR = './live_trading'

    # 报告格式
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


# ============================================================================
# 券商配置模板
# ============================================================================

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


# ============================================================================
# 预设配置方案
# ============================================================================

class PresetConfigs:
    """预设配置方案"""

    @staticmethod
    def conservative():
        """保守型配置"""
        StrategyConfig.REBALANCE_DAYS = 10
        StrategyConfig.POSITION_SIZE = 15
        StrategyConfig.STOP_LOSS = -0.10
        MLConfig.USE_ML_SCORING = True
        MLConfig.TRAIN_MONTHS = 18
        print("✓ 已切换到【保守型】配置")

    @staticmethod
    def balanced():
        """平衡型配置（推荐）"""
        StrategyConfig.REBALANCE_DAYS = 5
        StrategyConfig.POSITION_SIZE = 10
        StrategyConfig.STOP_LOSS = -0.15
        MLConfig.USE_ML_SCORING = True
        MLConfig.TRAIN_MONTHS = 12
        print("✓ 已切换到【平衡型】配置（推荐）")

    @staticmethod
    def aggressive():
        """激进型配置"""
        StrategyConfig.REBALANCE_DAYS = 3
        StrategyConfig.POSITION_SIZE = 8
        StrategyConfig.STOP_LOSS = -0.20
        MLConfig.USE_ML_SCORING = True
        MLConfig.TRAIN_MONTHS = 6
        print("✓ 已切换到【激进型】配置")

    @staticmethod
    def fast_test():
        """快速测试配置"""
        StrategyConfig.REBALANCE_DAYS = 5
        StrategyConfig.POSITION_SIZE = 5
        MLConfig.USE_ML_SCORING = True
        MLConfig.TRAINING_MODE = 'single'  # 使用单次训练加速
        DataConfig.SAMPLE_SIZE = 500  # 减少股票池
        DataConfig.USE_SAMPLING = True
        print("✓ 已切换到【快速测试】配置")


# ============================================================================
# 配置验证
# ============================================================================

def validate_config():
    """验证配置合法性"""
    errors = []
    warnings = []

    # 检查必填项
    if StrategyConfig.REBALANCE_DAYS < 1:
        errors.append("调仓周期必须 >= 1天")

    if StrategyConfig.POSITION_SIZE < 1:
        errors.append("持仓数量必须 >= 1只")

    if MLConfig.TRAIN_MONTHS < 3:
        errors.append("训练窗口必须 >= 3个月")

    # 检查风险参数
    if abs(StrategyConfig.STOP_LOSS) > 0.30:
        warnings.append(f"止损线 {StrategyConfig.STOP_LOSS:.1%} 过大，建议 < 30%")

    if StrategyConfig.POSITION_SIZE > 20:
        warnings.append(f"持仓数量 {StrategyConfig.POSITION_SIZE} 过多，分散效果递减")

    # 检查自动交易
    if LiveTradingConfig.ENABLE_AUTO_TRADE:
        if not LiveTradingConfig.ACCOUNT:
            warnings.append("自动交易已启用但未配置账户信息")

    # 打印结果
    if errors:
        print("\n❌ 配置错误:")
        for err in errors:
            print(f"  - {err}")
        return False

    if warnings:
        print("\n⚠️  配置警告:")
        for warn in warnings:
            print(f"  - {warn}")

    print("\n✓ 配置验证通过")
    return True


def print_current_config():
    """打印当前配置"""
    print("\n" + "=" * 80)
    print("当前配置")
    print("=" * 80)

    print("\n【策略配置】")
    print(f"  调仓周期: {StrategyConfig.REBALANCE_DAYS} 天")
    print(f"  持仓数量: {StrategyConfig.POSITION_SIZE} 只")
    print(f"  仓位方法: {StrategyConfig.POSITION_METHOD}")
    print(f"  止损线: {StrategyConfig.STOP_LOSS:.1%}")

    print("\n【ML配置】")
    print(f"  ML评分: {'启用' if MLConfig.USE_ML_SCORING else '禁用'}")
    if MLConfig.USE_ML_SCORING:
        print(f"  训练模式: {MLConfig.TRAINING_MODE}")
        print(f"  训练窗口: {MLConfig.TRAIN_MONTHS} 个月")
        print(f"  模型缓存: {'启用' if MLConfig.CACHE_MODELS else '禁用'}")

    print("\n【数据配置】")
    print(f"  股票池: {DataConfig.SAMPLE_SIZE} 只")
    print(f"  基本面: {'启用' if DataConfig.USE_FUNDAMENTAL else '禁用'}")
    print(f"  资金流: {'启用' if DataConfig.USE_MONEY_FLOW else '禁用'}")

    print("\n【交易配置】")
    print(f"  自动交易: {'启用' if LiveTradingConfig.ENABLE_AUTO_TRADE else '禁用'}")
    print(f"  初始资金: ¥{LiveTradingConfig.INITIAL_CAPITAL:,.0f}")
    print(f"  券商: {LiveTradingConfig.BROKER}")

    print("\n" + "=" * 80)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("实盘交易配置文件")
    print("\n使用预设配置:")
    print("  1. 保守型: PresetConfigs.conservative()")
    print("  2. 平衡型: PresetConfigs.balanced()  # 推荐")
    print("  3. 激进型: PresetConfigs.aggressive()")
    print("  4. 测试型: PresetConfigs.fast_test()")

    # 示例：使用平衡型配置
    PresetConfigs.balanced()

    # 验证配置
    validate_config()

    # 打印配置
    print_current_config()