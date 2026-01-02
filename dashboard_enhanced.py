"""
dashboard_enhanced.py - 完整整合video_visualization的增强版仪表盘

功能整合:
1. ✅ 基础推荐展示（原dashboard）
2. ✅ Top 10横向条形图（video_visualization）
3. ✅ 评分对比散点图（video_visualization）
4. ✅ 评分时序图（video_visualization）
5. ✅ 行业分布图（video_visualization）
6. ✅ 风险-收益气泡图
7. ✅ 双评分对比（StockRanker vs ML）
8. ✅ 交互式数据表格

版本: v3.1
运行: streamlit run dashboard_enhanced.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import numpy as np

# ========== 页面配置 ==========
st.set_page_config(
    page_title="A股量化交易系统 - 增强版",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 自定义CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(120deg, #e0f7fa 0%, #b2ebf2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stock-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .positive { color: #d32f2f; font-weight: bold; }
    .negative { color: #388e3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ========== 数据加载函数 ==========

@st.cache_data(ttl=300)
def load_recommendations(date=None):
    """加载推荐数据"""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    csv_path = f'./live_trading/stock_recommendations_{date}.csv'

    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        return df
    except Exception as e:
        st.error(f"加载数据失败: {e}")
        return None


@st.cache_data(ttl=300)
def load_factor_data():
    """加载因子数据（用于高级可视化）"""
    try:
        cache_dir = './data_cache'
        if not os.path.exists(cache_dir):
            return None

        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('factor_data_incr') and f.endswith('.csv')]
        if cache_files:
            latest = sorted(cache_files)[-1]
            df = pd.read_csv(f'{cache_dir}/{latest}')
            return df
    except Exception as e:
        st.warning(f"无法加载因子数据: {e}")
    return None


def load_state():
    """加载交易状态"""
    state_file = './live_trading_state.json'

    if not os.path.exists(state_file):
        return {'last_rebalance_date': None, 'positions': {}, 'cash': 1000000}

    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {'last_rebalance_date': None, 'positions': {}, 'cash': 1000000}


# ========== 高级可视化函数（整合video_visualization）==========

def plot_top_picks_bar(df):
    """Top 10横向条形图（来自video_visualization）"""
    df_sorted = df.sort_values('position', ascending=True).copy()

    fig = go.Figure()

    # 主条形图 - 最终评分
    fig.add_trace(go.Bar(
        x=df_sorted['position'],
        y=df_sorted['instrument'],
        orientation='h',
        name='Final Score',
        marker=dict(color='#00F0FF', line=dict(color='#00A0CC', width=1)),
        text=df_sorted['position'].apply(lambda x: f"{x:.3f}"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Final Score: %{x:.4f}<extra></extra>'
    ))

    # 如果有双评分，添加对比标记
    if 'stockranker_score' in df_sorted.columns and 'ml_score' in df_sorted.columns:
        fig.add_trace(go.Scatter(
            x=df_sorted['stockranker_score'],
            y=df_sorted['instrument'],
            mode='markers',
            name='StockRanker',
            marker=dict(color='#f39c12', size=12, symbol='diamond', line=dict(width=1, color='white')),
            hovertemplate='StockRanker: %{x:.4f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=df_sorted['ml_score'],
            y=df_sorted['instrument'],
            mode='markers',
            name='ML Score',
            marker=dict(color='#9b59b6', size=12, symbol='circle', line=dict(width=1, color='white')),
            hovertemplate='ML: %{x:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title={'text': '<b>Top 10 Stock Recommendations</b>', 'x': 0.5, 'xanchor': 'center'},
        template='plotly_white',
        xaxis_title='Score',
        yaxis_title='',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.8)'),
        xaxis=dict(range=[0, 1.05]),
        hovermode='y unified'
    )

    return fig


def plot_score_comparison(df):
    """StockRanker vs ML评分对比散点图（来自video_visualization）"""
    if 'stockranker_score' not in df.columns or 'ml_score' not in df.columns:
        return None

    fig = go.Figure()

    # 散点图
    fig.add_trace(go.Scatter(
        x=df['stockranker_score'],
        y=df['ml_score'],
        mode='markers',
        marker=dict(
            size=df['position'] * 20 + 5,
            color=df['position'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Final<br>Score"),
            line=dict(width=1, color='white')
        ),
        text=df['instrument'],
        customdata=df[['recommend_level', 'position']],
        hovertemplate='<b>%{text}</b><br>StockRanker: %{x:.4f}<br>ML Score: %{y:.4f}<br>Final: %{customdata[1]:.4f}<br>Level: %{customdata[0]}<extra></extra>'
    ))

    # 对角线
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash', width=2),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 象限分割线
    fig.add_hline(y=0.5, line_dash="dot", line_color="lightgray", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dot", line_color="lightgray", opacity=0.5)

    fig.update_layout(
        title={'text': '<b>StockRanker vs ML Score Comparison</b>', 'x': 0.5, 'xanchor': 'center'},
        template='plotly_white',
        xaxis_title='StockRanker Score',
        yaxis_title='ML Score',
        height=600,
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
        annotations=[
            dict(text="High ML<br>Low SR", x=0.1, y=0.9, showarrow=False, font=dict(size=10, color="gray")),
            dict(text="High SR<br>Low ML", x=0.9, y=0.1, showarrow=False, font=dict(size=10, color="gray")),
            dict(text="Both High", x=0.9, y=0.9, showarrow=False, font=dict(size=10, color="gray")),
            dict(text="Both Low", x=0.1, y=0.1, showarrow=False, font=dict(size=10, color="gray"))
        ]
    )

    return fig


def plot_score_timeline(factor_data, top_n=5):
    """Top N股票评分时序图（来自video_visualization）"""
    if factor_data is None:
        return None

    score_col = 'position' if 'position' in factor_data.columns else 'ml_score'

    if score_col not in factor_data.columns:
        return None

    try:
        latest_date = factor_data['date'].max()
        top_stocks = factor_data[factor_data['date'] == latest_date].nlargest(top_n, score_col)['instrument'].tolist()

        df_subset = factor_data[factor_data['instrument'].isin(top_stocks)].copy()
        df_subset['date'] = pd.to_datetime(df_subset['date'])

        fig = go.Figure()

        colors = px.colors.qualitative.Set2
        for idx, stock in enumerate(top_stocks):
            stock_data = df_subset[df_subset['instrument'] == stock].sort_values('date')
            fig.add_trace(go.Scatter(
                x=stock_data['date'],
                y=stock_data[score_col],
                mode='lines+markers',
                name=stock,
                line=dict(width=2, color=colors[idx % len(colors)]),
                marker=dict(size=6)
            ))

        fig.update_layout(
            title={'text': f'<b>Top {top_n} Stock Score Timeline</b>', 'x': 0.5, 'xanchor': 'center'},
            template='plotly_white',
            xaxis_title='Date',
            yaxis_title='Score',
            hovermode='x unified',
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.8)')
        )

        return fig
    except Exception as e:
        st.warning(f"评分时序图生成失败: {e}")
        return None


def plot_sector_distribution(df):
    """行业分布图（来自video_visualization）"""
    if 'industry' not in df.columns:
        return None

    industry_counts = df['industry'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=industry_counts.index,
        values=industry_counts.values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set3, line=dict(color='white', width=2)),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>'
    )])

    fig.update_layout(
        title={'text': '<b>Industry Distribution (Top 10)</b>', 'x': 0.5, 'xanchor': 'center'},
        template='plotly_white',
        height=400,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1)
    )

    return fig


def plot_risk_return_bubble(df):
    """风险-收益气泡图"""
    if 'return_5d' not in df.columns or 'volatility_20d' not in df.columns:
        return None

    fig = px.scatter(
        df,
        x='volatility_20d',
        y='return_5d',
        size='position',
        color='recommend_level',
        hover_data={'instrument': True, 'position': ':.4f', 'volatility_20d': ':.2%', 'return_5d': ':.2%'},
        labels={'volatility_20d': '波动率 (20日)', 'return_5d': '5日收益率', 'position': 'Final Score',
                'recommend_level': '推荐等级'},
        title='<b>Risk-Return Profile</b>',
        color_discrete_map={'Strong Buy': '#27ae60', 'Buy': '#3498db', 'Accumulate': '#f39c12', 'Hold': '#95a5a6'}
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        template='plotly_white',
        height=500,
        xaxis_title='20日波动率',
        yaxis_title='5日收益率',
        title={'x': 0.5, 'xanchor': 'center'}
    )

    return fig


# ========== 主界面 ==========

def main():
    st.markdown('<div class="main-header">📊 A股量化交易系统 - 增强版仪表盘</div>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 控制面板")

        selected_date = st.date_input("选择日期", value=datetime.now(), max_value=datetime.now())
        date_str = selected_date.strftime('%Y%m%d')

        st.subheader("📊 视图选择")
        view_mode = st.radio("选择视图", ["概览", "详细分析", "高级图表"], index=0)

        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        st.subheader("📌 系统状态")
        state = load_state()

        if state['last_rebalance_date']:
            st.info(f"上次调仓: {state['last_rebalance_date']}")
        else:
            st.warning("尚未调仓")

        st.metric("当前持仓", f"{len(state.get('positions', {}))} 只")
        st.metric("可用资金", f"¥{state.get('cash', 0):,.0f}")

    # 加载数据
    df = load_recommendations(date_str)

    if df is None:
        st.error(f"❌ 未找到 {date_str} 的推荐数据")
        st.info("请先运行主程序生成推荐")
        st.code("streamlit run dashboard_enhanced.py", language="bash")
        return

    # ========== 概览视图 ==========
    if view_mode == "概览":
        st.header("📈 今日概览")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("推荐股票数", len(df))
        with col2:
            strong_buy = len(df[df['recommend_level'] == 'Strong Buy'])
            st.metric("强力推荐", strong_buy, delta=f"{strong_buy / len(df) * 100:.0f}%")
        with col3:
            if 'return_5d' in df.columns:
                avg_return = df['return_5d'].mean()
                st.metric("平均5日收益", f"{avg_return:.2%}", delta=f"{'📈' if avg_return > 0 else '📉'}")
        with col4:
            avg_score = df['position'].mean()
            st.metric("平均评分", f"{avg_score:.3f}")

        st.divider()

        st.header("⭐ 重点推荐 (Top 3)")

        cols = st.columns(3)

        for idx, (i, row) in enumerate(df.head(3).iterrows()):
            with cols[idx]:
                st.markdown(f"""
                <div class="stock-card">
                    <h3>#{idx + 1} {row['instrument']}</h3>
                    <p><strong>推荐等级:</strong> {row.get('recommend_level', 'N/A')}</p>
                    <p><strong>综合评分:</strong> {row['position']:.4f}</p>
                    <p><strong>当前价格:</strong> ¥{row.get('close', 0):.2f}</p>
                </div>
                """, unsafe_allow_html=True)

                if 'return_5d' in row and pd.notna(row['return_5d']):
                    ret = row['return_5d']
                    color_class = 'positive' if ret > 0 else 'negative'
                    st.markdown(f'<p class="{color_class}">5日涨跌: {ret:+.2%}</p>', unsafe_allow_html=True)

                if 'industry' in row and pd.notna(row['industry']):
                    st.caption(f"行业: {row['industry']}")

        st.divider()

        st.header("🎯 Top 10 推荐")

        display_cols = ['instrument', 'recommend_level', 'position']
        if 'stockranker_score' in df.columns:
            display_cols.append('stockranker_score')
        if 'ml_score' in df.columns:
            display_cols.append('ml_score')
        if 'close' in df.columns:
            display_cols.append('close')
        if 'return_5d' in df.columns:
            display_cols.append('return_5d')
        if 'risk_level' in df.columns:
            display_cols.append('risk_level')

        df_display = df[display_cols].copy()

        if 'close' in df_display.columns:
            df_display['close'] = df_display['close'].apply(lambda x: f"¥{x:.2f}")
        if 'return_5d' in df_display.columns:
            df_display['return_5d'] = df_display['return_5d'].apply(lambda x: f"{x:.2%}")

        rename_dict = {
            'instrument': '代码', 'recommend_level': '推荐等级', 'position': '综合评分',
            'stockranker_score': '多因子', 'ml_score': 'ML评分', 'close': '价格',
            'return_5d': '5日涨跌', 'risk_level': '风险'
        }

        df_display = df_display.rename(columns=rename_dict)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ========== 详细分析视图 ==========
    elif view_mode == "详细分析":
        st.header("📊 详细分析")

        st.subheader("🎯 Top 10评分对比")
        fig1 = plot_top_picks_bar(df)
        st.plotly_chart(fig1, use_container_width=True)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏭 行业分布")
            fig2 = plot_sector_distribution(df)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("行业数据不可用")

        with col2:
            st.subheader("📈 风险-收益分布")
            fig3 = plot_risk_return_bubble(df)
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("技术指标数据不可用")

    # ========== 高级图表视图 ==========
    elif view_mode == "高级图表":
        st.header("🔬 高级分析图表")

        if 'stockranker_score' in df.columns and 'ml_score' in df.columns:
            st.subheader("🎯 StockRanker vs ML 评分对比")
            fig4 = plot_score_comparison(df)
            if fig4:
                st.plotly_chart(fig4, use_container_width=True)

                with st.expander("💡 图表解读"):
                    st.markdown("""
                    **象限说明：**
                    - **右上角**：两个模型都看好（最佳机会）
                    - **左下角**：两个模型都不看好（规避）
                    - **右下角**：StockRanker看好，ML谨慎（可能估值偏高或短期风险）
                    - **左上角**：ML看好，StockRanker谨慎（可能存在短期催化剂）

                    **气泡大小**：代表最终融合评分
                    **气泡颜色**：最终评分从低到高（深色到浅色）
                    """)
        else:
            st.info("评分对比需要同时包含StockRanker和ML评分")

        st.divider()

        factor_data = load_factor_data()
        if factor_data is not None:
            st.subheader("📈 Top 5评分时序")
            fig5 = plot_score_timeline(factor_data, top_n=5)
            if fig5:
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.warning("评分时序数据不可用")
        else:
            st.info("需要因子数据来显示时序图。请确保data_cache目录包含因子数据文件。")

    st.divider()
    st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("⚠️ 本系统仅供参考，不构成投资建议。投资有风险，入市需谨慎。")


if __name__ == "__main__":
    main()