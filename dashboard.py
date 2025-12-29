"""
dashboard.py - AI量化策略监控台 (ML增强版)

新增功能:
✅ 展示ML模型评分分布
✅ 实盘信号可视化图表
✅ 持仓收益跟踪
✅ 模型性能监控
✅ 风险指标仪表盘

版本: v3.0
日期: 2025-12-27
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import json
from PIL import Image
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ==============================================================================
# 页面配置
# ==============================================================================
st.set_page_config(
    page_title="AI量化策略监控台 v3.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        padding: 20px;
        border-radius: 5px;
        text-align: center;
    }
    .stDataFrame { border: 1px solid #262730; }
    .css-1544g2n { padding-top: 2rem; }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
    .highlight-green {
        color: #27ae60;
        font-weight: bold;
    }
    .highlight-red {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 辅助函数
# ==============================================================================

def load_live_state():
    """加载实盘状态"""
    state_path = "./live_trading_state.json"
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    return None


def calculate_portfolio_metrics(signals_df, state_data):
    """计算组合指标"""
    if signals_df.empty:
        return {}

    metrics = {
        'total_stocks': len(signals_df),
        'avg_score': signals_df['score'].mean() if 'score' in signals_df.columns else 0,
        'score_std': signals_df['score'].std() if 'score' in signals_df.columns else 0,
        'top_industry': signals_df['industry'].mode()[
            0] if 'industry' in signals_df.columns and not signals_df.empty else 'Unknown',
        'concentration': signals_df['target_weight'].max() if 'target_weight' in signals_df.columns else 0,
    }

    # 从状态数据获取历史信息
    if state_data:
        metrics['last_rebalance'] = state_data.get('last_rebalance_date', 'N/A')
        metrics['last_ml_train'] = state_data.get('last_ml_train_date', 'N/A')
        metrics['position_count'] = len(state_data.get('positions', {}))

    return metrics


def plot_score_distribution(signals_df):
    """绘制评分分布图"""
    if 'score' not in signals_df.columns or signals_df.empty:
        return None

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=signals_df['score'],
        nbinsx=20,
        name='Score Distribution',
        marker=dict(
            color='#00F0FF',
            line=dict(color='#ffffff', width=1)
        )
    ))

    fig.update_layout(
        title='<b>📊 ML评分分布</b>',
        xaxis_title='ML Score',
        yaxis_title='Count',
        template='plotly_dark',
        height=300,
        showlegend=False
    )

    return fig


def plot_industry_distribution(signals_df):
    """绘制行业分布图"""
    if 'industry' not in signals_df.columns or signals_df.empty:
        return None

    industry_counts = signals_df['industry'].value_counts().head(10)

    fig = go.Figure(data=[
        go.Bar(
            x=industry_counts.values,
            y=industry_counts.index,
            orientation='h',
            marker=dict(
                color='#9b59b6',
                line=dict(color='#ffffff', width=1)
            )
        )
    ])

    fig.update_layout(
        title='<b>🏭 行业分布</b>',
        xaxis_title='Count',
        yaxis_title='',
        template='plotly_dark',
        height=300,
        showlegend=False
    )

    return fig


def plot_position_radar(signals_df):
    """绘制持仓雷达图"""
    if signals_df.empty or len(signals_df) < 3:
        return None

    # 选择前6只股票
    top_6 = signals_df.nlargest(6, 'score')

    categories = top_6['stock'].tolist()
    scores = top_6['score'].tolist()

    # 闭合
    categories += [categories[0]]
    scores += [scores[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        line=dict(color='#00F0FF', width=2),
        fillcolor='rgba(0, 240, 255, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor='#444'
            ),
            angularaxis=dict(gridcolor='#444'),
            bgcolor='#0e1117'
        ),
        title='<b>🎯 Top 6 股票评分雷达</b>',
        template='plotly_dark',
        height=400,
        showlegend=False
    )

    return fig


def plot_rebalance_history(state_data):
    """绘制调仓历史"""
    if not state_data or 'rebalance_history' not in state_data:
        return None

    history = state_data['rebalance_history']
    if not history:
        return None

    df = pd.DataFrame(history)
    df['date'] = pd.to_datetime(df['date'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['orders_count'],
        mode='lines+markers',
        name='Orders Count',
        line=dict(color='#e74c3c', width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='<b>📈 调仓历史</b>',
        xaxis_title='Date',
        yaxis_title='Orders Count',
        template='plotly_dark',
        height=300,
        showlegend=False
    )

    return fig


# ==============================================================================
# 侧边栏：控制台
# ==============================================================================
st.sidebar.title("🎛️ 控制台")

# --- 模块 1: 回测报告 (Backtest) ---
st.sidebar.markdown("### 📊 回测分析 (Backtest)")
REPORTS_DIR = "./reports"

backtest_dates = []
if os.path.exists(REPORTS_DIR):
    subfolders = [f.path for f in os.scandir(REPORTS_DIR) if f.is_dir()]
    backtest_dates = sorted([os.path.basename(f) for f in subfolders], reverse=True)

if backtest_dates:
    selected_bt_date = st.sidebar.selectbox("📅 选择回测报告日期", backtest_dates)
    bt_report_path = os.path.join(REPORTS_DIR, selected_bt_date)
else:
    st.sidebar.warning("未找到回测报告")
    selected_bt_date = None
    bt_report_path = None

# --- 模块 2: 实盘/模拟盘 (Live) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 实盘监控 (Live)")
LIVE_DIR = "./live_trading"

live_dates = []
if os.path.exists(LIVE_DIR):
    order_files = glob.glob(os.path.join(LIVE_DIR, "trading_orders_*.csv"))
    live_dates = sorted([os.path.basename(f).split('_')[-1].replace('.csv', '') for f in order_files], reverse=True)

if live_dates:
    selected_live_date = st.sidebar.selectbox("📡 选择实盘信号日期", live_dates)
else:
    st.sidebar.info("暂无实盘记录")
    selected_live_date = None

# --- 系统状态 ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 系统状态")

state_data = load_live_state()
if state_data:
    st.sidebar.success("✓ 系统在线")
    if state_data.get('last_rebalance_date'):
        st.sidebar.metric("上次调仓", state_data['last_rebalance_date'])
    if state_data.get('last_ml_train_date'):
        st.sidebar.metric("上次ML训练", state_data['last_ml_train_date'])
    st.sidebar.metric("当前持仓", f"{len(state_data.get('positions', {}))} 只")
else:
    st.sidebar.warning("系统离线")

# 全局刷新按钮
st.sidebar.markdown("---")
if st.sidebar.button("🔄 刷新所有数据"):
    st.rerun()

# ==============================================================================
# 主界面逻辑
# ==============================================================================
st.title("🚀 AI 量化策略监控台 v3.0")
st.caption("集成滚动训练ML模型 | 实时风险监控 | 智能选股建议")

# 创建6个标签页
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 资金曲线",
    "🏆 本周金股",
    "💼 持仓明细",
    "📝 交易日志",
    "🤖 实盘建议",
    "📈 模型监控"  # ✨ 新增
])

# ------------------------------------------------------------------------------
# Tab 1: 资金曲线 (回测)
# ------------------------------------------------------------------------------
with tab1:
    if selected_bt_date and bt_report_path:
        col_curve, col_shap = st.columns([2, 1])

        with col_curve:
            st.subheader(f"📈 策略净值走势 ({selected_bt_date})")
            equity_html = os.path.join(bt_report_path, "video_assets", "01_equity_curve.html")
            if os.path.exists(equity_html):
                with open(equity_html, 'r', encoding='utf-8') as f:
                    components.html(f.read(), height=600, scrolling=True)
            else:
                st.warning("未找到资金曲线图表")

        with col_shap:
            st.subheader("🤖 AI 模型解释 (SHAP)")
            shap_img = os.path.join(bt_report_path, "video_assets", "02_shap_summary.png")
            if os.path.exists(shap_img):
                st.image(Image.open(shap_img), caption="因子重要性", use_container_width=True)
            else:
                st.info("未生成SHAP图")
    else:
        st.warning("请在左侧侧边栏选择回测日期")

# ------------------------------------------------------------------------------
# Tab 2: 选股榜单 (回测)
# ------------------------------------------------------------------------------
with tab2:
    if selected_bt_date and bt_report_path:
        st.subheader(f"🎯 模型预测 Top 榜单 ({selected_bt_date})")
        top_picks_html = os.path.join(bt_report_path, "video_assets", "03_weekly_top_picks.html")
        if os.path.exists(top_picks_html):
            with open(top_picks_html, 'r', encoding='utf-8') as f:
                components.html(f.read(), height=600)
        else:
            st.info("未找到选股榜单图表")
    else:
        st.warning("请在左侧侧边栏选择回测日期")

# ------------------------------------------------------------------------------
# Tab 3: 持仓明细 (回测)
# ------------------------------------------------------------------------------
with tab3:
    if selected_bt_date and bt_report_path:
        st.subheader(f"💼 回测期末持仓 ({selected_bt_date})")
        holdings_csv = os.path.join(bt_report_path, "daily_holdings.csv")
        if os.path.exists(holdings_csv):
            df_h = pd.read_csv(holdings_csv)


            # 添加收益率颜色
            def color_negative_red(val):
                try:
                    if isinstance(val, str):
                        val = float(val.strip('%')) / 100
                    color = 'color: #27ae60' if val > 0 else 'color: #e74c3c'
                    return color
                except:
                    return ''


            st.dataframe(
                df_h.style.format({"成本价": "{:.2f}", "现价": "{:.2f}", "收益率": "{:.2%}"})
                .applymap(color_negative_red, subset=['收益率']),
                use_container_width=True
            )
        else:
            st.info("该日期无持仓记录")
    else:
        st.warning("请在左侧侧边栏选择回测日期")

# ------------------------------------------------------------------------------
# Tab 4: 交易日志 (回测)
# ------------------------------------------------------------------------------
with tab4:
    if selected_bt_date and bt_report_path:
        st.subheader("📝 回测调仓记录")
        trades_csv = os.path.join(bt_report_path, "trades.csv")
        if os.path.exists(trades_csv):
            df_trades = pd.read_csv(trades_csv)

            # 添加筛选器
            col1, col2 = st.columns(2)
            with col1:
                action_filter = st.multiselect(
                    "操作类型",
                    options=['buy', 'sell'],
                    default=['buy', 'sell']
                )
            with col2:
                date_filter = st.date_input(
                    "日期范围",
                    value=(pd.to_datetime(df_trades['date'].min()), pd.to_datetime(df_trades['date'].max()))
                )

            # 应用筛选
            filtered_trades = df_trades[df_trades['action'].isin(action_filter)]

            st.dataframe(filtered_trades, use_container_width=True)

            # 统计信息
            col1, col2, col3 = st.columns(3)
            col1.metric("总交易笔数", len(filtered_trades))
            col2.metric("买入笔数", len(filtered_trades[filtered_trades['action'] == 'buy']))
            col3.metric("卖出笔数", len(filtered_trades[filtered_trades['action'] == 'sell']))
        else:
            st.info("未找到交易记录文件 (trades.csv)")
    else:
        st.warning("请在左侧侧边栏选择回测日期")

# ------------------------------------------------------------------------------
# Tab 5: 实盘建议 (Live) ✨ 增强版
# ------------------------------------------------------------------------------
with tab5:
    if selected_live_date:
        st.header(f"🤖 实盘交易建议 - {selected_live_date}")

        # 定义文件路径
        orders_path = os.path.join(LIVE_DIR, f"trading_orders_{selected_live_date}.csv")
        signals_path = os.path.join(LIVE_DIR, f"signals_{selected_live_date}.csv")
        instructions_path = os.path.join(LIVE_DIR, f"trading_instructions_{selected_live_date}.txt")

        # ========== 顶部指标卡片 ==========
        if os.path.exists(signals_path):
            df_signals = pd.read_csv(signals_path)
            metrics = calculate_portfolio_metrics(df_signals, state_data)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("🎯 持仓数量", f"{metrics['total_stocks']} 只")
            col2.metric("📊 平均评分", f"{metrics['avg_score']:.4f}")
            col3.metric("📈 评分标准差", f"{metrics['score_std']:.4f}")
            col4.metric("🏭 主导行业", metrics['top_industry'])
            col5.metric("⚖️ 集中度", f"{metrics['concentration']:.1%}")

            st.markdown("---")

        # ========== 主内容区 ==========
        col_main, col_side = st.columns([2, 1])

        with col_main:
            # 1. 交易指令
            st.subheader("📝 交易指令单")
            if os.path.exists(instructions_path):
                with open(instructions_path, 'r', encoding='utf-8') as f:
                    txt_content = f.read()
                st.text_area("复制以下指令:", txt_content, height=250, key="instructions")

                # 下载按钮
                st.download_button(
                    label="📥 下载指令",
                    data=txt_content,
                    file_name=f"trading_instructions_{selected_live_date}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("未找到指令文件")

            st.markdown("---")

            # 2. 交易订单
            st.subheader("📋 交易订单明细")
            if os.path.exists(orders_path):
                df_orders = pd.read_csv(orders_path)


                def highlight_action(row):
                    if row['action'] == 'buy':
                        return ['background-color: #d4edda'] * len(row)
                    elif row['action'] == 'sell':
                        return ['background-color: #f8d7da'] * len(row)
                    return [''] * len(row)


                st.dataframe(
                    df_orders.style.apply(highlight_action, axis=1)
                    .format({"price": "{:.2f}", "amount": "{:,.0f}"}),
                    use_container_width=True
                )

                # 订单统计
                if not df_orders.empty:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📝 总订单数", len(df_orders))
                    col2.metric("🔵 买入笔数", len(df_orders[df_orders['action'] == 'buy']))
                    col3.metric("🔴 卖出笔数", len(df_orders[df_orders['action'] == 'sell']))
            else:
                st.info("本日无交易订单 (可能无需调仓)")

            st.markdown("---")

            # 3. 信号详情
            st.subheader("📡 原始信号池")
            if os.path.exists(signals_path):
                df_signals = pd.read_csv(signals_path)

                # 添加排名列
                df_signals['rank'] = range(1, len(df_signals) + 1)

                # 格式化显示
                st.dataframe(
                    df_signals.style.format({
                        "score": "{:.4f}",
                        "target_weight": "{:.1%}",
                        "current_price": "{:.2f}"
                    }),
                    use_container_width=True
                )

                # 可视化
                st.markdown("### 📊 信号可视化")

                col_viz1, col_viz2 = st.columns(2)

                with col_viz1:
                    fig_score = plot_score_distribution(df_signals)
                    if fig_score:
                        st.plotly_chart(fig_score, use_container_width=True)

                with col_viz2:
                    fig_industry = plot_industry_distribution(df_signals)
                    if fig_industry:
                        st.plotly_chart(fig_industry, use_container_width=True)

                # 雷达图
                fig_radar = plot_position_radar(df_signals)
                if fig_radar:
                    st.plotly_chart(fig_radar, use_container_width=True)

            else:
                st.info("未找到信号文件")

        with col_side:
            # 4. 系统状态
            st.subheader("💼 系统状态")

            if state_data:
                # 关键指标
                st.markdown("#### 📊 关键指标")
                st.metric("上次调仓", state_data.get('last_rebalance_date', 'N/A'))
                st.metric("上次ML训练", state_data.get('last_ml_train_date', 'N/A'))
                st.metric("当前持仓", f"{len(state_data.get('positions', {}))} 只")

                st.markdown("---")

                # 调仓历史
                st.markdown("#### 📈 调仓历史")
                fig_history = plot_rebalance_history(state_data)
                if fig_history:
                    st.plotly_chart(fig_history, use_container_width=True)

                st.markdown("---")

                # 当前持仓
                st.markdown("#### 💼 当前持仓")
                positions = state_data.get('positions', {})
                if positions:
                    df_pos = pd.DataFrame(
                        list(positions.items()),
                        columns=['股票代码', '股数']
                    )
                    st.dataframe(df_pos, hide_index=True, use_container_width=True)
                else:
                    st.info("当前空仓")

                # 完整状态JSON
                with st.expander("🔍 查看完整状态"):
                    st.json(state_data)
            else:
                st.warning("未找到状态文件")
                st.info("请运行实盘交易系统生成状态文件")

    else:
        st.info("👈 请在左侧选择实盘信号日期")
        st.markdown("""
        ### 📖 使用指南

        #### 生成实盘数据
        1. 运行增强版实盘系统：
        ```bash
        python main_live_trading_enhanced.py
        ```

        2. 刷新本页面，选择日期查看

        #### 功能说明
        - **交易指令**: 可直接复制到交易软件
        - **订单明细**: 包含买卖方向、数量、价格
        - **信号池**: 展示ML评分和行业分布
        - **可视化**: 评分分布、行业分布、雷达图
        - **系统状态**: 调仓历史、持仓监控
        """)

# ------------------------------------------------------------------------------
# Tab 6: 模型监控 ✨ 新增
# ------------------------------------------------------------------------------
with tab6:
    st.header("📈 ML模型性能监控")

    # 检查是否有实盘数据
    if selected_live_date and os.path.exists(signals_path):
        df_signals = pd.read_csv(signals_path)

        st.markdown("### 📊 模型评分分析")

        # 评分统计
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最高评分", f"{df_signals['score'].max():.4f}")
        col2.metric("最低评分", f"{df_signals['score'].min():.4f}")
        col3.metric("平均评分", f"{df_signals['score'].mean():.4f}")
        col4.metric("标准差", f"{df_signals['score'].std():.4f}")

        # 评分分布直方图
        st.markdown("#### 📊 评分分布")
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df_signals['score'],
            nbinsx=30,
            marker=dict(color='#00F0FF', line=dict(color='#ffffff', width=1))
        ))
        fig_hist.update_layout(
            xaxis_title='ML Score',
            yaxis_title='Count',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # 评分 vs 行业
        st.markdown("#### 🏭 行业评分对比")
        if 'industry' in df_signals.columns:
            industry_scores = df_signals.groupby('industry')['score'].agg(['mean', 'std', 'count']).reset_index()
            industry_scores = industry_scores.sort_values('mean', ascending=False).head(10)

            fig_industry = go.Figure()
            fig_industry.add_trace(go.Bar(
                x=industry_scores['industry'],
                y=industry_scores['mean'],
                error_y=dict(type='data', array=industry_scores['std']),
                marker=dict(color='#9b59b6')
            ))
            fig_industry.update_layout(
                xaxis_title='Industry',
                yaxis_title='Average Score',
                template='plotly_dark',
                height=400
            )
            st.plotly_chart(fig_industry, use_container_width=True)

        # 模型训练信息
        st.markdown("---")
        st.markdown("### 🤖 模型训练信息")

        if state_data:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**上次训练日期**: {state_data.get('last_ml_train_date', 'N/A')}")
                st.info(f"**训练窗口**: 12个月 (滚动)")
                st.info(f"**模型类型**: XGBoost + LightGBM 集成")

            with col2:
                st.info(f"**特征数量**: 约30-50个")
                st.info(f"**中性化**: 市场 + 行业")
                st.info(f"**投票策略**: 平均")

        # 数据质量监控
        st.markdown("---")
        st.markdown("### 📋 数据质量监控")

        col1, col2, col3 = st.columns(3)

        missing_price = df_signals['current_price'].isna().sum()
        col1.metric("缺失价格", f"{missing_price} / {len(df_signals)}")

        missing_industry = df_signals['industry'].isna().sum() if 'industry' in df_signals.columns else 0
        col2.metric("缺失行业", f"{missing_industry} / {len(df_signals)}")

        col3.metric("有效信号", f"{len(df_signals[df_signals['score'] > 0.5])} / {len(df_signals)}")

    else:
        st.info("请选择实盘日期以查看模型监控")

        st.markdown("""
        ### 📖 模型监控说明

        该页面展示ML模型的性能指标和数据质量：

        #### 监控内容
        - 📊 **评分分布**: 检查评分是否合理分布
        - 🏭 **行业分析**: 对比不同行业的评分差异
        - 🤖 **训练信息**: 模型配置和训练时间
        - 📋 **数据质量**: 缺失值和异常值监控

        #### 健康指标
        - ✅ 评分范围: 0.0 - 1.0
        - ✅ 标准差: 0.1 - 0.3 (适度分散)
        - ✅ 缺失价格: < 5%
        - ✅ 有效信号: > 70%
        """)

# 页脚
st.sidebar.markdown("---")
st.sidebar.caption("© 2025 AI Quant Dashboard v3.0")
st.sidebar.caption("Powered by Streamlit + ML")