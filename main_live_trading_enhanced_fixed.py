"""
main_live_trading_enhanced_fixed.py - Part 1: 配置与初始化

修复内容：
✅ 方案2: 扩展数据历史至540天（18个月）
✅ 方案3: 智能自适应训练模式
✅ 优化数据加载逻辑
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import time
import traceback

import tushare as ts

# ========== 配置导入 ==========
from config import (
    TUSHARE_TOKEN,
    StrategyConfig,
    BacktestConfig,
    RiskControlConfig,
    TradingCostConfig,
    DataConfig,
    FactorConfig,
    MLConfig,
    OutputConfig
)

ts.set_token(TUSHARE_TOKEN)

# ========== 数据模块 ==========
from data_module import DataCache, TushareDataSource
from data_module_incremental import load_data_with_incremental_update

# ========== ML模块 ==========
ML_AVAILABLE = False
try:
    from ml_factor_scoring_fixed import (
        AdvancedMLScorer,
        ICCalculator,
        IndustryBasedScorer,
        EnhancedStockSelector
    )
    ML_AVAILABLE = True
    print("✓ ML评分模块加载成功")
except ImportError as e:
    print(f"⚠️  ML模块未找到: {e}")

# ========== ML修复补丁 ==========
ML_FIX_AVAILABLE = False
try:
    from ml_factor_scoring_fixed import (
        quick_fix_ml_scorer,
        diagnose_prediction_gap,
        FixedAdvancedMLScorer
    )
    ML_FIX_AVAILABLE = True
    print("✓ ML修复补丁加载成功")
except ImportError as e:
    print(f"⚠️  ML修复补丁未加载: {e}")

# ========== 舆情风控 ==========
SENTIMENT_AVAILABLE = False
try:
    from sentiment_risk_control import (
        apply_sentiment_control,
        SentimentRiskController
    )
    SENTIMENT_AVAILABLE = True
    print("✓ 舆情风控模块加载成功")
except ImportError as e:
    print(f"⚠️  舆情风控未加载: {e}")


# ========== 实盘配置（优化版） ==========
class LiveTradingConfig:
    """实盘交易配置（修复版）"""
    
    # 策略参数（从回测最优配置继承）
    REBALANCE_DAYS = 5  # 5日调仓
    POSITION_METHOD = 'equal'  # 等权
    POSITION_SIZE = 10  # 持仓10只
    
    # 智能调仓参数
    BUFFER_RANK = 18  # 缓冲区排名（前18名不主动卖出）
    SCORE_IMPROVEMENT_THRESHOLD = 0.05  # 换仓评分提升门槛
    
    # 风控参数
    STOP_LOSS = -0.15  # 硬止损-15%
    MIN_DAYS_LISTED = 180  # 最短上市时间（天）
    
    # 交易成本
    BUY_COST = 0.0003
    SELL_COST = 0.0003
    TAX_RATIO = 0.0005
    
    # 🔧 数据配置（方案2：扩展历史）
    DATA_HISTORY_DAYS = 540  # 从365扩展至540天（约18个月）✅
    USE_SAMPLING = False
    SAMPLE_SIZE = 5000
    
    # 🔧 ML配置（方案3：自适应训练）
    USE_ML_SCORING = True
    ML_TRAIN_MONTHS = 10  # 默认10个月（可自适应调整）✅
    ML_MIN_TRAIN_MONTHS = 6  # 最小训练月份
    ML_AUTO_ADJUST = True  # 启用自动调整 ✅
    USE_SENTIMENT_CONTROL = True
    
    # 实盘控制
    ENABLE_AUTO_TRADE = False  # 默认仅生成建议
    
    # 国信证券配置
    GUOSEN_CONFIG = {
        'broker': 'guosen',
        'account': '',
        'password': '',
        'comm_password': '',
        'ip': '',
        'port': 0,
    }


def print_banner():
    """打印启动横幅"""
    print("\n" + "="*80)
    print("    🚀 实盘交易系统 v3.1 - 自适应训练版")
    print("="*80)
    print("\n🎯 核心特性:")
    print("  ✅ 完整因子处理流程（行业中性化、因子增强）")
    print("  ✅ ML高级评分（智能自适应训练）")  # 修改
    print("  ✅ 最新数据预测修复（确保信号不中断）")
    print("  ✅ 舆情风控（一票否决 + 加分提权）")
    print("  ✅ 智能缓冲调仓（减少交易摩擦）")
    print("  ✅ 前视偏差防护（剔除次新股）")
    print("  🆕 扩展数据历史（18个月训练集）")  # 新增
    print("  🆕 自动降级策略（数据不足时智能处理）")  # 新增
    print()


def check_trading_day():
    """检查是否是交易日"""
    try:
        pro = ts.pro_api()
        today = datetime.now().strftime('%Y%m%d')
        
        cal = pro.trade_cal(
            exchange='SSE',
            start_date=today,
            end_date=today
        )
        
        if len(cal) == 0:
            return False
        
        return cal.iloc[0]['is_open'] == 1
    except Exception as e:
        print(f"⚠️  交易日检查失败: {e}")
        return True


def load_historical_state():
    """加载历史状态"""
    state_file = './live_trading_state.json'
    
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    
    return {
        'last_rebalance_date': None,
        'positions': {},
        'rebalance_history': []
    }


def save_current_state(state):
    """保存当前状态"""
    with open('./live_trading_state.json', 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def should_rebalance(state):
    """判断是否应该调仓"""
    last_date = state.get('last_rebalance_date')
    
    if last_date is None:
        return True, "首次运行"
    
    try:
        last_dt = datetime.strptime(last_date, '%Y-%m-%d')
    except ValueError:
        return True, "日期格式重置"
    
    today = datetime.now()
    days_diff = (today - last_dt).days
    
    if days_diff >= LiveTradingConfig.REBALANCE_DAYS:
        return True, f"距上次调仓{days_diff}天"
    
    return False, f"距上次调仓仅{days_diff}天"

def process_factors_with_ml(factor_data, price_data, cache_manager):
    """
    🔥 核心函数：完整的因子处理 + ML评分流程（自适应版）
    
    方案3：智能自适应训练
    - 数据充足：Walk-Forward训练（多窗口）
    - 数据有限：简单训练模式（80/20切分）
    - 数据不足：因子等权备用方案
    
    Returns:
        factor_data: 带有 ml_score 列的因子数据
    """
    
    # ============ 步骤1: 补全行业数据 ============
    print("\n" + "="*80)
    print("🏭 步骤1: 补全行业数据")
    print("="*80)
    
    try:
        ds = TushareDataSource(token=TUSHARE_TOKEN, cache_manager=cache_manager)
        unique_stocks = factor_data['instrument'].unique().tolist()
        industry_df = ds.get_industry_data(unique_stocks, use_cache=True)
        
        if industry_df is not None and not industry_df.empty:
            if 'industry' in factor_data.columns:
                del factor_data['industry']
            factor_data = factor_data.merge(industry_df, on='instrument', how='left')
            factor_data['industry'] = factor_data['industry'].fillna('其他')
            print(f"  ✓ 成功合并行业数据: {factor_data['industry'].nunique()} 个行业")
        else:
            factor_data['industry'] = 'Unknown'
    except Exception as e:
        print(f"  ⚠️  行业数据获取失败: {e}")
        if 'industry' not in factor_data.columns:
            factor_data['industry'] = 'Unknown'
    
    # ============ 步骤2: 数据质量优化 ============
    try:
        print("\n" + "="*80)
        print("🔍 步骤2: 数据质量优化")
        print("="*80)
        from data_quality_optimizer import optimize_data_quality
        price_data, factor_data = optimize_data_quality(
            price_data, factor_data, cache_manager=cache_manager
        )
    except Exception as e:
        print(f"  ⚠️  数据质量优化警告: {e}")
    
    # ============ 步骤3: 因子增强处理 ============
    try:
        print("\n" + "="*80)
        print("🎯 步骤3: 因子增强处理（行业中性化）")
        print("="*80)
        
        from enhanced_factor_processor import EnhancedFactorProcessor
        
        factor_processor = EnhancedFactorProcessor(
            neutralize_industry=True,
            neutralize_market=False
        )
        
        exclude_columns = [
            'date', 'instrument', 'open', 'high', 'low', 'close', 
            'volume', 'amount', 'industry'
        ]
        factor_columns = [
            col for col in factor_data.columns 
            if col not in exclude_columns and pd.api.types.is_numeric_dtype(factor_data[col])
        ]
        
        print(f"  检测到 {len(factor_columns)} 个有效因子")
        
        if len(factor_columns) > 0:
            factor_data = factor_processor.process_factors(factor_data, factor_columns)
            print(f"  ✓ 因子增强完成")
    except Exception as e:
        print(f"  ⚠️  因子增强警告: {e}")
        traceback.print_exc()
    
    # ============ 步骤4: ML评分（自适应版） ============
    if LiveTradingConfig.USE_ML_SCORING and ML_AVAILABLE:
        try:
            print("\n" + "="*80)
            print("🚀 步骤4: ML高级评分（智能自适应训练）")
            print("="*80)
            
            # 清理污染列
            污染列 = ['ml_score', 'position', 'score_rank', 'composite_score']
            factor_data_clean = factor_data.copy()
            for col in 污染列:
                if col in factor_data_clean.columns:
                    factor_data_clean = factor_data_clean.drop(columns=[col])
            
            # 🔧 初始化ML评分器（使用优化参数）
            ml_scorer = AdvancedMLScorer(
                model_type=MLConfig.ML_MODEL_TYPE,
                target_period=MLConfig.ML_TARGET_PERIOD,
                top_percentile=MLConfig.ML_TOP_PERCENTILE,
                use_classification=MLConfig.ML_USE_CLASSIFICATION,
                use_ic_features=MLConfig.ML_USE_IC_FEATURES,
                use_active_return=True,
                train_months=LiveTradingConfig.ML_TRAIN_MONTHS  # 使用配置的月份
            )
            
            # 准备训练数据
            print("  [1/5] 准备训练数据...")
            X, y, merged_df = ml_scorer.prepare_training_data(
                factor_data_clean,
                price_data,
                factor_columns
            )
            
            # 🔧 智能诊断：检测数据月份
            print("  [2/5] 数据量诊断...")
            merged_df['year_month'] = pd.to_datetime(merged_df['date']).dt.to_period('M')
            unique_months = merged_df['year_month'].nunique()
            month_list = sorted(merged_df['year_month'].unique())
            
            required_months = LiveTradingConfig.ML_TRAIN_MONTHS + 2  # train + valid + test
            min_required_months = LiveTradingConfig.ML_MIN_TRAIN_MONTHS + 2
            
            print(f"\n  📊 数据诊断报告:")
            print(f"     可用月份: {unique_months} ({month_list[0]} ~ {month_list[-1]})")
            print(f"     理想需求: {required_months}月 (训练{LiveTradingConfig.ML_TRAIN_MONTHS} + 验证1 + 测试1)")
            print(f"     最小需求: {min_required_months}月 (训练{LiveTradingConfig.ML_MIN_TRAIN_MONTHS} + 验证1 + 测试1)")
            
            # 🔧 自适应训练决策
            training_mode = None
            
            if unique_months >= required_months:
                # 情况1: 数据充足 - Walk-Forward训练
                training_mode = 'walk_forward'
                n_splits = min(3, unique_months - required_months + 1)  # 最多3个窗口
                print(f"\n  ✅ 数据充足，使用 Walk-Forward 训练 ({n_splits} 个窗口)")
                
            elif unique_months >= min_required_months and LiveTradingConfig.ML_AUTO_ADJUST:
                # 情况2: 数据有限但可调整 - 压缩训练窗口
                training_mode = 'walk_forward_adjusted'
                adjusted_train = unique_months - 2  # 减去验证+测试
                adjusted_train = max(adjusted_train, LiveTradingConfig.ML_MIN_TRAIN_MONTHS)
                
                print(f"\n  🔧 自动调整训练参数:")
                print(f"     训练月份: {LiveTradingConfig.ML_TRAIN_MONTHS} → {adjusted_train}")
                
                # 临时修改评分器配置
                ml_scorer.train_months = adjusted_train
                n_splits = 2  # 有限窗口数
                print(f"  ⚠️  使用压缩的 Walk-Forward 训练 ({n_splits} 个窗口)")
                
            elif unique_months >= 4:
                # 情况3: 数据不足 - 简单训练
                training_mode = 'simple'
                print(f"\n  ⚠️  数据月份不足，降级到简单训练模式 (80/20切分)")
                
            else:
                # 情况4: 数据严重不足 - 使用备用方案
                training_mode = 'fallback'
                print(f"\n  ❌ 数据严重不足 ({unique_months}月)，使用因子等权备用方案")
            
            # 🔧 执行训练
            print(f"\n  [3/5] 执行训练 (模式: {training_mode})...")
            
            if training_mode in ['walk_forward', 'walk_forward_adjusted']:
                ml_scorer.train_walk_forward(X, y, merged_df, n_splits=n_splits)
                
            elif training_mode == 'simple':
                ml_scorer._train_simple(X, y)
                
            elif training_mode == 'fallback':
                # 跳过ML训练，直接使用因子等权
                print("  ⏭️  跳过ML训练，使用因子等权")
                factor_data['ml_score'] = factor_data[factor_columns].mean(axis=1)
                factor_data['ml_score'] = factor_data.groupby('date')['ml_score'].rank(pct=True)
                return factor_data  # 提前返回
            
            # 应用最新数据修复
            print("  [4/5] 应用最新数据预测修复...")
            if ML_FIX_AVAILABLE:
                factor_data = quick_fix_ml_scorer(
                    ml_scorer=ml_scorer,
                    factor_data=factor_data,
                    price_data=price_data,
                    factor_columns=factor_columns
                )
            else:
                print("  ⚠️  修复补丁未加载，使用原始预测")
                factor_data_predicted = ml_scorer.predict_scores(merged_df)
                prediction_cols = ['date', 'instrument', 'ml_score', 'position']
                prediction_df = factor_data_predicted[prediction_cols]
                
                # 清理并合并
                for col in ['ml_score', 'position']:
                    if col in factor_data.columns:
                        factor_data = factor_data.drop(columns=[col])
                
                factor_data = factor_data.merge(
                    prediction_df, on=['date', 'instrument'], how='left'
                )
            
            # 验证修复效果
            latest_date = factor_data['date'].max()
            latest_scores = factor_data[factor_data['date'] == latest_date]
            valid_scores = latest_scores['ml_score'].notna().sum()
            
            print(f"\n  [5/5] 预测结果验证:")
            print(f"     最新日期: {latest_date}")
            print(f"     有效评分: {valid_scores}/{len(latest_scores)} 只 ({valid_scores/len(latest_scores):.1%})")
            
            if valid_scores == 0:
                print(f"     ⚠️  警告：无有效评分，回退到备用方案")
                factor_data['ml_score'] = factor_data[factor_columns].mean(axis=1)
                factor_data['ml_score'] = factor_data.groupby('date')['ml_score'].rank(pct=True)
            
            # 打印特征重要性
            try:
                importance = ml_scorer.get_feature_importance(top_n=10)
                if importance is not None:
                    print("\n  📊 TOP 10 关键因子:")
                    for idx, row in importance.iterrows():
                        print(f"     {row['feature']:<25}: {row['importance']:.4f}")
            except Exception as e:
                print(f"  ⚠️  特征重要性分析失败: {e}")
        
        except Exception as e:
            print(f"  ❌ ML评分失败: {e}")
            traceback.print_exc()
            # 备用方案
            if 'ml_score' not in factor_data.columns and len(factor_columns) > 0:
                print("  ⚠️  启用备用评分：因子等权")
                factor_data['ml_score'] = factor_data[factor_columns].mean(axis=1)
                factor_data['ml_score'] = factor_data.groupby('date')['ml_score'].rank(pct=True)
    else:
        print("\n  ℹ️  ML评分未启用，使用因子等权")
        if len(factor_columns) > 0:
            factor_data['ml_score'] = factor_data[factor_columns].mean(axis=1)
            factor_data['ml_score'] = factor_data.groupby('date')['ml_score'].rank(pct=True)
    
    return factor_data

def apply_sentiment_filter(factor_data, price_data, cache_manager):
    """
    🛡️ 应用舆情风控
    """
    if not LiveTradingConfig.USE_SENTIMENT_CONTROL or not SENTIMENT_AVAILABLE:
        print("\n  ℹ️  舆情风控未启用")
        return factor_data
    
    try:
        print("\n" + "="*80)
        print("🛡️  步骤5: 舆情风控")
        print("="*80)
        
        latest_date = factor_data['date'].max()
        latest_mask = factor_data['date'] == latest_date
        latest_stocks = factor_data[latest_mask].copy()
        
        print(f"  分析对象: {len(latest_stocks)} 只股票")
        
        # 应用舆情过滤
        filtered_latest = apply_sentiment_control(
            selected_stocks=latest_stocks,
            factor_data=factor_data,
            price_data=price_data,
            tushare_token=TUSHARE_TOKEN,
            cache_manager=cache_manager,
            enable_veto=True,
            enable_boost=True,
            lookback_days=30
        )
        
        # 更新factor_data
        removed_stocks = set(latest_stocks['instrument']) - set(filtered_latest['instrument'])
        if removed_stocks:
            print(f"  🚫 剔除风险股票: {len(removed_stocks)} 只")
            factor_data = factor_data[
                ~((factor_data['date'] == latest_date) & 
                  (factor_data['instrument'].isin(removed_stocks)))
            ]
        
        # 更新评分
        score_col = 'ml_score' if 'ml_score' in factor_data.columns else 'position'
        boost_count = 0
        
        for _, row in filtered_latest.iterrows():
            stock = row['instrument']
            new_score = row[score_col]
            
            mask = (factor_data['date'] == latest_date) & (factor_data['instrument'] == stock)
            if mask.any():
                old_score = factor_data.loc[mask, score_col].values[0]
                if abs(new_score - old_score) > 0.01:
                    factor_data.loc[mask, score_col] = new_score
                    boost_count += 1
        
        if boost_count > 0:
            print(f"  📈 加分提权: {boost_count} 只")
        
        print(f"  ✅ 舆情风控完成 ({len(latest_stocks)} → {len(filtered_latest)} 只)")
    
    except Exception as e:
        print(f"  ⚠️  舆情风控出错: {e}")
        traceback.print_exc()
    
    return factor_data


def get_today_signals_enhanced(factor_data, price_data):
    """
    增强版信号生成（使用ML评分）
    """
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 获取最新日期数据
    latest_date = factor_data['date'].max()
    today_factors = factor_data[factor_data['date'] == latest_date]
    
    print(f"  📅 使用数据日期: {latest_date}")
    
    # 优先使用ml_score
    score_col = 'ml_score' if 'ml_score' in today_factors.columns else 'position'
    
    # 检查有效评分
    valid_scores = today_factors[score_col].notna().sum()
    if valid_scores == 0:
        print(f"  ❌ 无有效评分，无法生成信号")
        return pd.DataFrame()
    
    print(f"  ✅ 有效评分: {valid_scores}/{len(today_factors)} 只 ({valid_scores/len(today_factors):.1%})")
    
    # 排序取Top N
    top_stocks = today_factors.nlargest(LiveTradingConfig.POSITION_SIZE, score_col)
    
    # 等权分配
    weight = 1.0 / LiveTradingConfig.POSITION_SIZE
    
    # 获取价格
    latest_price_date = price_data['date'].max()
    today_prices = price_data[price_data['date'] == latest_price_date]
    
    signals = []
    for _, row in top_stocks.iterrows():
        stock = row['instrument']
        score = row[score_col]
        
        price_row = today_prices[today_prices['instrument'] == stock]
        price = price_row['close'].iloc[0] if len(price_row) > 0 else 0
        
        signals.append({
            'stock': stock,
            'score': score,
            'target_weight': weight,
            'current_price': price,
            'date': latest_date
        })
    
    return pd.DataFrame(signals)


def compare_with_current_positions_enhanced(signals, current_positions, factor_data,
                                           buffer_rank=18, score_improvement_threshold=0.05):
    """
    智能持仓对比（参考原版逻辑）
    """
    print("\n  🔍 智能持仓分析:")
    
    if signals.empty:
        print("  ⚠️  无有效信号，建议清仓")
        return pd.DataFrame(), list(current_positions.keys())
    
    latest_date = signals['date'].iloc[0]
    
    # 识别评分列
    score_col = 'ml_score' if 'ml_score' in factor_data.columns else 'position'
    
    today_data = factor_data[factor_data['date'] == latest_date]
    today_all_ranks = today_data.sort_values(score_col, ascending=False)
    today_all_ranks['rank'] = range(1, len(today_all_ranks) + 1)
    
    stock_to_rank = today_all_ranks.set_index('instrument')['rank'].to_dict()
    stock_to_score = today_all_ranks.set_index('instrument')[score_col].to_dict()
    
    current_stocks = set(current_positions.keys())
    
    to_sell_list = []
    to_buy_list = []
    kept_stocks = []
    
    # === 卖出逻辑 ===
    for stock in current_stocks:
        current_rank = stock_to_rank.get(stock, 9999)
        current_score = stock_to_score.get(stock, 0)
        
        if current_rank > buffer_rank:
            print(f"    🔻 淘汰: {stock:10s} 排名 {current_rank:3d} (> {buffer_rank})")
            to_sell_list.append(stock)
        else:
            print(f"    ⚓ 保留: {stock:10s} 排名 {current_rank:3d}")
            kept_stocks.append(stock)
    
    # === 买入逻辑 ===
    open_slots = LiveTradingConfig.POSITION_SIZE - len(kept_stocks)
    candidates = signals[~signals['stock'].isin(current_stocks)].sort_values('score', ascending=False)
    
    for _, row in candidates.iterrows():
        stock_name = row['stock']
        new_score = row['score']
        
        if open_slots > 0:
            to_buy_list.append(row)
            open_slots -= 1
            print(f"    🟢 买入(填补): {stock_name:10s} 评分 {new_score:.4f}")
        else:
            if not kept_stocks:
                break
            
            weakest_stock = min(kept_stocks, key=lambda x: stock_to_score.get(x, 0))
            weakest_score = stock_to_score.get(weakest_stock, 0)
            
            if new_score > weakest_score + score_improvement_threshold:
                print(f"    🔄 换仓: {stock_name}({new_score:.3f}) 替换 "
                      f"{weakest_stock}({weakest_score:.3f})")
                to_buy_list.append(row)
                to_sell_list.append(weakest_stock)
                kept_stocks.remove(weakest_stock)
                kept_stocks.append(stock_name)
            else:
                break
    
    to_buy_df = pd.DataFrame(to_buy_list) if to_buy_list else pd.DataFrame(columns=signals.columns)
    
    return to_buy_df, to_sell_list


def generate_trading_orders(to_buy_df, to_sell_list, current_positions, 
                           available_cash, total_value):
    """生成交易订单"""
    orders = []
    
    # 卖出
    for stock in to_sell_list:
        shares = current_positions.get(stock, 0)
        if shares > 0:
            orders.append({
                'stock': stock,
                'action': 'sell',
                'shares': shares,
                'price': 0,
                'amount': 0,
                'reason': '排名下滑/优化换仓'
            })
    
    # 买入
    for _, row in to_buy_df.iterrows():
        target_amount = total_value * row['target_weight']
        price = row['current_price']
        
        if price and price > 0:
            shares = int(target_amount / price / 100) * 100
            
            if shares >= 100:
                orders.append({
                    'stock': row['stock'],
                    'action': 'buy',
                    'shares': shares,
                    'price': price,
                    'amount': shares * price,
                    'reason': f"ML评分: {row['score']:.4f}"
                })
    
    return pd.DataFrame(orders)


def save_trading_orders(orders_df, signals_df, output_dir='./live_trading'):
    """保存交易订单和信号详情"""
    os.makedirs(output_dir, exist_ok=True)
    
    today = datetime.now().strftime('%Y%m%d')
    
    # 保存订单
    orders_path = os.path.join(output_dir, f'trading_orders_{today}.csv')
    orders_df.to_csv(orders_path, index=False, encoding='utf-8-sig')
    
    # 保存信号详情
    signals_path = os.path.join(output_dir, f'signals_{today}.csv')
    signals_df.to_csv(signals_path, index=False, encoding='utf-8-sig')
    
    print(f"\n💾 文件已保存:")
    print(f"   订单: {orders_path}")
    print(f"   信号: {signals_path}")
    
    # 生成简化指令
    simple_orders = []
    for _, order in orders_df.iterrows():
        if order['action'] == 'buy':
            simple_orders.append(f"买入 {order['stock']} {order['shares']}股 @ ¥{order['price']:.2f}")
        elif order['action'] == 'sell':
            simple_orders.append(f"卖出 {order['stock']} {order['shares']}股")
    
    simple_path = os.path.join(output_dir, f'trading_instructions_{today}.txt')
    with open(simple_path, 'w', encoding='utf-8') as f:
        f.write(f"交易日期: {datetime.now().strftime('%Y-%m-%d')}\n")
        f.write(f"策略版本: v3.1 自适应训练版\n")
        f.write(f"调仓模式: 智能缓冲 (Buffer={LiveTradingConfig.BUFFER_RANK}, "
                f"Threshold={LiveTradingConfig.SCORE_IMPROVEMENT_THRESHOLD})\n")
        f.write(f"数据历史: {LiveTradingConfig.DATA_HISTORY_DAYS}天 (约18个月)\n")
        f.write("=" * 60 + "\n\n")
        
        # 写入Top信号
        f.write("📊 今日Top信号:\n\n")
        for i, row in signals_df.iterrows():
            f.write(f"{i+1:2d}. {row['stock']:10s} | 评分: {row['score']:.4f} | "
                   f"权重: {row['target_weight']:.1%}\n")
        
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("📋 交易指令:\n\n")
        if not simple_orders:
            f.write("✅ 无需交易（持仓结构稳定）\n")
        else:
            for i, instruction in enumerate(simple_orders, 1):
                f.write(f"{i}. {instruction}\n")
        
        # 风控提示
        f.write("\n" + "=" * 60 + "\n\n")
        f.write("⚠️  风控提示:\n")
        f.write("1. 此信号已通过舆情风控筛选\n")
        f.write("2. 已剔除ST、立案调查等风险股票\n")
        f.write("3. 建议开盘后观察流动性再执行\n")
        f.write("4. 遇停牌/涨停可顺延至下一候选股\n")
    
    print(f"   指令: {simple_path}")
    
    return orders_path


def main():
    """主函数（修复版）"""
    print_banner()
    
    print(f"📅 运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 策略配置: {LiveTradingConfig.REBALANCE_DAYS}日调仓 | "
          f"{LiveTradingConfig.POSITION_METHOD} | {LiveTradingConfig.POSITION_SIZE}只")
    print(f"🤖 智能模式: ML评分={LiveTradingConfig.USE_ML_SCORING} | "
          f"舆情风控={LiveTradingConfig.USE_SENTIMENT_CONTROL}")
    print(f"📊 数据配置: 历史{LiveTradingConfig.DATA_HISTORY_DAYS}天 | "
          f"自适应训练={LiveTradingConfig.ML_AUTO_ADJUST}")
    
    # ============ 步骤1: 检查交易日 ============
    print("\n" + "="*80)
    print("【步骤1/7】检查交易日")
    print("="*80)
    
    if not check_trading_day():
        print("  ℹ️  今天不是交易日，程序退出")
        return
    print("  ✅ 确认为交易日")
    
    # ============ 步骤2: 加载历史状态 ============
    print("\n" + "="*80)
    print("【步骤2/7】加载历史状态")
    print("="*80)
    
    state = load_historical_state()
    current_positions = state.get('positions', {})
    
    if state['last_rebalance_date']:
        print(f"  上次调仓: {state['last_rebalance_date']}")
        print(f"  当前持仓: {len(current_positions)} 只")
        if current_positions:
            for stock, shares in list(current_positions.items())[:5]:
                print(f"     • {stock}: {shares} 股")
            if len(current_positions) > 5:
                print(f"     ... 还有 {len(current_positions)-5} 只")
    else:
        print("  首次运行")
    
    # ============ 步骤3: 判断调仓时机 ============
    print("\n" + "="*80)
    print("【步骤3/7】判断调仓时机")
    print("="*80)
    
    need_rebalance, reason = should_rebalance(state)
    print(f"  是否调仓: {'✅ 是' if need_rebalance else '❌ 否'} ({reason})")
    
    if not need_rebalance:
        print("\n  今日无需调仓，程序退出")
        return
    
    # ============ 步骤4: 加载最新数据（修复版） ============
    print("\n" + "="*80)
    print("【步骤4/7】加载最新数据（扩展历史）")
    print("="*80)
    
    # 🔧 方案2：扩展数据历史至540天（约18个月）
    START_DATE = (datetime.now() - timedelta(days=LiveTradingConfig.DATA_HISTORY_DAYS)).strftime('%Y-%m-%d')
    END_DATE = datetime.now().strftime('%Y-%m-%d')
    
    print(f"  数据区间: {START_DATE} ~ {END_DATE}")
    print(f"  历史长度: {LiveTradingConfig.DATA_HISTORY_DAYS} 天 (约 {LiveTradingConfig.DATA_HISTORY_DAYS/30:.1f} 个月)")
    
    cache_manager = DataCache(cache_dir='./data_cache')
    
    try:
        factor_data, price_data = load_data_with_incremental_update(
            START_DATE,
            END_DATE,
            max_stocks=LiveTradingConfig.SAMPLE_SIZE,
            cache_manager=cache_manager,
            use_stockranker=FactorConfig.USE_STOCKRANKER,
            custom_weights=FactorConfig.CUSTOM_WEIGHTS,
            tushare_token=TUSHARE_TOKEN,
            use_fundamental=FactorConfig.USE_FUNDAMENTAL,
            use_sampling=LiveTradingConfig.USE_SAMPLING,
            sample_size=LiveTradingConfig.SAMPLE_SIZE,
            max_workers=DataConfig.MAX_WORKERS,
            force_full_update=False,
            min_days_listed=LiveTradingConfig.MIN_DAYS_LISTED  # 🔥 前视偏差防护
        )
    except Exception as e:
        print(f"  ❌ 数据加载失败: {e}")
        traceback.print_exc()
        return
    
    if factor_data is None or price_data is None or factor_data.empty or price_data.empty:
        print("  ❌ 数据为空，无法继续")
        return
    
    # 🔧 数据诊断
    print(f"\n  ✅ 数据加载完成")
    print(f"     股票数: {factor_data['instrument'].nunique()}")
    print(f"     日期范围: {factor_data['date'].min()} ~ {factor_data['date'].max()}")
    print(f"     数据行数: {len(factor_data):,}")
    
    # 计算实际月份数
    factor_data['temp_month'] = pd.to_datetime(factor_data['date']).dt.to_period('M')
    actual_months = factor_data['temp_month'].nunique()
    month_list = sorted(factor_data['temp_month'].unique())
    factor_data = factor_data.drop(columns=['temp_month'])
    
    print(f"     实际月份: {actual_months} ({month_list[0]} ~ {month_list[-1]})")
    
    if actual_months < LiveTradingConfig.ML_MIN_TRAIN_MONTHS + 2:
        print(f"\n  ⚠️  警告：数据月份({actual_months})少于最小需求({LiveTradingConfig.ML_MIN_TRAIN_MONTHS + 2})")
        print(f"  建议：增加 DATA_HISTORY_DAYS 或使用因子等权备用方案")
    
    # ============ 步骤5: 因子处理 + ML评分（自适应版） ============
    print("\n" + "="*80)
    print("【步骤5/7】因子处理 + ML评分（自适应训练）")
    print("="*80)
    
    try:
        factor_data = process_factors_with_ml(factor_data, price_data, cache_manager)
    except Exception as e:
        print(f"  ❌ 因子处理失败: {e}")
        traceback.print_exc()
        return
    
    # ============ 步骤6: 舆情风控 ============
    try:
        factor_data = apply_sentiment_filter(factor_data, price_data, cache_manager)
    except Exception as e:
        print(f"  ⚠️  舆情风控警告: {e}")
    
    # ============ 步骤7: 生成交易信号 ============
    print("\n" + "="*80)
    print("【步骤7/7】生成交易信号")
    print("="*80)
    
    signals = get_today_signals_enhanced(factor_data, price_data)
    
    if signals.empty:
        print("\n  ❌ 无有效信号，建议检查数据完整性")
        return
    
    print(f"\n  📊 今日Top {len(signals)} 候选:")
    for i, row in signals.iterrows():
        print(f"     {i+1:2d}. {row['stock']:10s} | 评分: {row['score']:.4f} | "
              f"价格: ¥{row['current_price']:.2f}")
    
    # ============ 智能持仓对比 ============
    to_buy_df, to_sell_list = compare_with_current_positions_enhanced(
        signals,
        current_positions,
        factor_data,
        buffer_rank=LiveTradingConfig.BUFFER_RANK,
        score_improvement_threshold=LiveTradingConfig.SCORE_IMPROVEMENT_THRESHOLD
    )
    
    print(f"\n  📋 交易计划:")
    print(f"     卖出: {len(to_sell_list)} 只")
    print(f"     买入: {len(to_buy_df)} 只")
    
    # ============ 生成订单 ============
    if len(to_buy_df) > 0 or len(to_sell_list) > 0:
        available_cash = 1000000  # 实盘应从券商接口获取
        total_value = 1000000
        
        orders = generate_trading_orders(
            to_buy_df, to_sell_list, current_positions, 
            available_cash, total_value
        )
        
        if len(orders) > 0:
            print(f"\n  💼 交易订单明细 ({len(orders)} 条):")
            print("  " + "-"*70)
            print(f"  {'操作':<6} | {'股票':<10} | {'股数':<8} | {'价格':<8} | {'原因'}")
            print("  " + "-"*70)
            
            for _, order in orders.iterrows():
                action_icon = "🔵买入" if order['action'] == 'buy' else "🔴卖出"
                print(f"  {action_icon:<6} | {order['stock']:<10} | "
                      f"{order['shares']:>8.0f} | {order['price']:>8.2f} | "
                      f"{order.get('reason','')}")
            
            print("  " + "-"*70)
            
            # 保存订单
            save_trading_orders(orders, signals)
            
            # 更新状态
            new_positions = current_positions.copy()
            
            # 移除卖出的
            for stock in to_sell_list:
                if stock in new_positions:
                    del new_positions[stock]
            
            # 添加买入的
            for _, row in orders[orders['action']=='buy'].iterrows():
                new_positions[row['stock']] = row['shares']
            
            state['last_rebalance_date'] = datetime.now().strftime('%Y-%m-%d')
            state['positions'] = new_positions
            state['rebalance_history'].append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'orders_count': len(orders),
                'buy_count': len(orders[orders['action']=='buy']),
                'sell_count': len(orders[orders['action']=='sell']),
                'ml_enabled': LiveTradingConfig.USE_ML_SCORING,
                'sentiment_enabled': LiveTradingConfig.USE_SENTIMENT_CONTROL,
                'data_months': actual_months,
                'training_mode': 'adaptive'  # 标记使用了自适应训练
            })
            
            save_current_state(state)
            
            print(f"\n  ✅ 状态已更新")
            print(f"     新持仓: {len(new_positions)} 只")
            
            # 打印新持仓
            if new_positions:
                print("\n  📊 调仓后持仓:")
                for stock in list(new_positions.keys())[:10]:
                    shares = new_positions[stock]
                    # 尝试获取评分
                    latest_date = factor_data['date'].max()
                    score_col = 'ml_score' if 'ml_score' in factor_data.columns else 'position'
                    stock_data = factor_data[
                        (factor_data['date']==latest_date) & 
                        (factor_data['instrument']==stock)
                    ]
                    score = stock_data[score_col].values[0] if len(stock_data)>0 else 0
                    print(f"     • {stock}: {shares} 股 | 评分: {score:.4f}")
        else:
            print("\n  ℹ️  生成订单为空（可能因价格异常等原因）")
    else:
        print("\n  🍵 持仓结构稳定，无需交易")
        # 更新检查点
        state['last_rebalance_date'] = datetime.now().strftime('%Y-%m-%d')
        save_current_state(state)
    
    # ============ 完成 ============
    print("\n" + "="*80)
    print("✅ 实盘交易流程完成！")
    print("="*80)
    
    print("\n💡 下一步:")
    print("  1. 查看 live_trading/trading_instructions_*.txt 获取交易指令")
    print("  2. 开盘后手动或自动执行订单（需启用 ENABLE_AUTO_TRADE）")
    print("  3. 如需自动交易，请配置 GUOSEN_CONFIG 并安装 easytrader")
    
    print("\n📊 本次运行统计:")
    print(f"  数据历史: {LiveTradingConfig.DATA_HISTORY_DAYS}天 ({actual_months}个月)")
    print(f"  训练模式: 自适应 (最优={LiveTradingConfig.ML_TRAIN_MONTHS}月, 最小={LiveTradingConfig.ML_MIN_TRAIN_MONTHS}月)")
    print(f"  ML评分: {'✅ 启用' if LiveTradingConfig.USE_ML_SCORING else '❌ 未启用'}")
    print(f"  舆情风控: {'✅ 启用' if LiveTradingConfig.USE_SENTIMENT_CONTROL else '❌ 未启用'}")
    
    if not LiveTradingConfig.ENABLE_AUTO_TRADE:
        print("\n  ⚠️  当前为模拟模式，仅生成建议文件")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        traceback.print_exc()
