"""
dashboard_ultimate.py - 终极版仪表盘（整合所有功能）

功能集成:
1. ✅ 实时推荐展示
2. ✅ 完整收益指标（胜率、盈亏比、夏普等）
3. ✅ video_visualization的所有图表
4. ✅ 回测绩效展示
5. ✅ 美化的现代化界面
6. ✅ 多页面导航

版本: v4.0 Ultimate
运行: streamlit run dashboard_ultimate.py
Powered by Dayiwu
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
    page_title="AI量化交易系统 | Powered by Dayiwu",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== 增强版CSS样式 ==========
st.markdown("""
<style>
    /* 主题色彩 */
    :root {
        --primary-color: #1E88E5;
        --success-color: #43A047;
        --danger-color: #E53935;
        --warning-color: #FB8C00;
        --dark-bg: #0E1117;
        --card-bg: #ffffff;
    }
    
    /* 主标题 */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: fadeIn 0.8s ease-in;
    }
    
    /* 页脚署名 */
    .footer-brand {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #667eea;
        font-size: 0.9rem;
        color: #666;
    }
    
    .footer-brand strong {
        color: #667eea;
        font-size: 1.1rem;
    }
    
    /* 卡片样式 */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border-left: 4px solid var(--primary-color);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }
    
    /* 股票卡片 */
    .stock-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .stock-card:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .stock-card h3 {
        color: #667eea;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* 颜色标记 */
    .positive { 
        color: #E53935; 
        font-weight: bold; 
        font-size: 1.1rem;
    }
    
    .negative { 
        color: #43A047; 
        font-weight: bold; 
        font-size: 1.1rem;
    }
    
    /* 徽章 */
    .badge {
        display: inline-block;
        padding: 0.35em 0.65em;
        font-size: 0.85rem;
        font-weight: 700;
        line-height: 1;
        color: #fff;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.375rem;
    }
    
    .badge-success { background-color: #43A047; }
    .badge-warning { background-color: #FB8C00; }
    .badge-danger { background-color: #E53935; }
    .badge-info { background-color: #1E88E5; }
    
    /* 指标卡片增强 */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    
    /* 动画 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 分隔线 */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
    }
    
    /* 标签页 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        background-color: transparent;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
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
        st.error(f"❌ 加载数据失败: {e}")
        return None


@st.cache_data(ttl=300)
def load_backtest_metrics():
    """加载回测绩效指标"""
    try:
        # 尝试从最新报告目录加载
        reports_dir = './reports'
        if not os.path.exists(reports_dir):
            return None
        
        # 获取最新报告目录
        subdirs = [d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d))]
        if not subdirs:
            return None
        
        latest_dir = sorted(subdirs)[-1]
        metrics_path = os.path.join(reports_dir, latest_dir, 'performance_metrics.csv')
        
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path, index_col=0)
            metrics = df['值'].to_dict()
            return metrics
    except Exception as e:
        st.warning(f"⚠️ 无法加载回测指标: {e}")
    return None


@st.cache_data(ttl=300)
def load_factor_data():
    """加载因子数据"""
    try:
        cache_dir = './data_cache'
        if not os.path.exists(cache_dir):
            return None
        
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith('factor_data_incr') and f.endswith('.csv')]
        if cache_files:
            latest = sorted(cache_files)[-1]
            df = pd.read_csv(f'{cache_dir}/{latest}')
            return df
    except:
        pass
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


# ========== 可视化函数 ==========

def plot_top_picks_bar(df):
    """Top 10横向条形图"""
    df_sorted = df.sort_values('position', ascending=True).copy()
    
    fig = go.Figure()
    
    # 主条形图
    fig.add_trace(go.Bar(
        x=df_sorted['position'],
        y=df_sorted['instrument'],
        orientation='h',
        name='Final Score',
        marker=dict(
            color=df_sorted['position'],
            colorscale='Viridis',
            line=dict(color='white', width=1)
        ),
        text=df_sorted['position'].apply(lambda x: f"{x:.3f}"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>'
    ))
    
    # 双评分标记
    if 'stockranker_score' in df_sorted.columns and 'ml_score' in df_sorted.columns:
        fig.add_trace(go.Scatter(
            x=df_sorted['stockranker_score'],
            y=df_sorted['instrument'],
            mode='markers',
            name='StockRanker',
            marker=dict(color='#FB8C00', size=14, symbol='diamond', line=dict(width=2, color='white')),
            hovertemplate='SR: %{x:.4f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=df_sorted['ml_score'],
            y=df_sorted['instrument'],
            mode='markers',
            name='ML Score',
            marker=dict(color='#AB47BC', size=14, symbol='circle', line=dict(width=2, color='white')),
            hovertemplate='ML: %{x:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={'text': '<b>🎯 Top 10 Stock Recommendations</b>', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        template='plotly_white',
        xaxis_title='Score',
        yaxis_title='',
        height=550,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.9)'),
        xaxis=dict(range=[0, 1.05]),
        hovermode='y unified',
        font=dict(size=12)
    )
    
    return fig


def plot_score_comparison(df):
    """评分对比散点图"""
    if 'stockranker_score' not in df.columns or 'ml_score' not in df.columns:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['stockranker_score'],
        y=df['ml_score'],
        mode='markers',
        marker=dict(
            size=df['position'] * 25 + 8,
            color=df['position'],
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="Final<br>Score", thickness=15),
            line=dict(width=2, color='white'),
            opacity=0.8
        ),
        text=df['instrument'],
        customdata=df[['recommend_level', 'position']],
        hovertemplate='<b>%{text}</b><br>SR: %{x:.4f}<br>ML: %{y:.4f}<br>Final: %{customdata[1]:.4f}<br>Level: %{customdata[0]}<extra></extra>'
    ))
    
    # 对角线
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash', width=3),
        showlegend=False,
        hoverinfo='skip',
        name='Perfect Agreement'
    ))
    
    # 象限线
    fig.add_hline(y=0.5, line_dash="dot", line_color="lightgray", line_width=2, opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dot", line_color="lightgray", line_width=2, opacity=0.5)
    
    fig.update_layout(
        title={'text': '<b>🔬 StockRanker vs ML Score Comparison</b>', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 20}},
        template='plotly_white',
        xaxis_title='StockRanker Score',
        yaxis_title='ML Score',
        height=650,
        xaxis=dict(range=[-0.05, 1.05]),
        yaxis=dict(range=[-0.05, 1.05]),
        annotations=[
            dict(text="<b>High ML<br>Low SR</b>", x=0.15, y=0.85, showarrow=False, font=dict(size=12, color="gray"), bgcolor="rgba(255,255,255,0.7)", borderpad=4),
            dict(text="<b>High SR<br>Low ML</b>", x=0.85, y=0.15, showarrow=False, font=dict(size=12, color="gray"), bgcolor="rgba(255,255,255,0.7)", borderpad=4),
            dict(text="<b>Both High<br>⭐Best</b>", x=0.85, y=0.85, showarrow=False, font=dict(size=12, color="#43A047"), bgcolor="rgba(255,255,255,0.7)", borderpad=4),
            dict(text="<b>Both Low<br>❌Avoid</b>", x=0.15, y=0.15, showarrow=False, font=dict(size=12, color="#E53935"), bgcolor="rgba(255,255,255,0.7)", borderpad=4)
        ],
        font=dict(size=12)
    )
    
    return fig


def plot_sector_distribution(df):
    """行业分布饼图"""
    if 'industry' not in df.columns:
        return None
    
    industry_counts = df['industry'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=industry_counts.index,
        values=industry_counts.values,
        hole=0.45,
        marker=dict(colors=px.colors.qualitative.Vivid, line=dict(color='white', width=3)),
        textinfo='label+percent',
        textfont=dict(size=13, color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percent: %{percent}<extra></extra>',
        pull=[0.1 if i == 0 else 0 for i in range(len(industry_counts))]
    )])
    
    fig.update_layout(
        title={'text': '<b>🏭 Industry Distribution</b>', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
        template='plotly_white',
        height=450,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05, font=dict(size=11)),
        font=dict(size=12)
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
        hover_data={'instrument': True, 'position': ':.4f'},
        labels={'volatility_20d': '波动率 (20日)', 'return_5d': '5日收益率'},
        title='<b>📈 Risk-Return Profile</b>',
        color_discrete_map={'Strong Buy': '#43A047', 'Buy': '#1E88E5', 'Accumulate': '#FB8C00', 'Hold': '#9E9E9E'}
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=2, opacity=0.6)
    
    fig.update_layout(
        template='plotly_white',
        height=550,
        title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 18}},
        font=dict(size=12)
    )
    
    fig.update_traces(marker=dict(line=dict(width=2, color='white')))
    
    return fig


# ========== 主界面 ==========

def main():
    # 主标题
    st.markdown('''
    <div class="main-header">
        🚀 AI Quantitative Trading System
        <div style="font-size: 1rem; font-weight: 400; margin-top: 0.5rem;">
            Powered by Advanced Machine Learning & Multi-Factor Models
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 侧边栏
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=AI+Trading", use_container_width=True)
        
        st.markdown("### ⚙️ Control Panel")
        
        selected_date = st.date_input("📅 Select Date", value=datetime.now(), max_value=datetime.now())
        date_str = selected_date.strftime('%Y%m%d')
        
        st.divider()
        
        st.markdown("### 📌 System Status")
        state = load_state()
        
        if state['last_rebalance_date']:
            st.success(f"✅ Last Rebalance: {state['last_rebalance_date']}")
        else:
            st.warning("⚠️ No Rebalance Yet")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💼 Positions", f"{len(state.get('positions', {}))} stocks")
        with col2:
            st.metric("💰 Cash", f"¥{state.get('cash', 0)/10000:.1f}万")
        
        st.divider()
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # 加载回测指标
        metrics = load_backtest_metrics()
        if metrics:
            st.markdown("### 📊 Backtest Metrics")
            
            with st.expander("🎯 View Details", expanded=False):
                st.metric("📈 Total Return", f"{float(metrics.get('total_return', 0)):.2%}")
                st.metric("⚡ Sharpe Ratio", f"{float(metrics.get('sharpe_ratio', 0)):.3f}")
                st.metric("🎲 Win Rate", f"{float(metrics.get('win_rate', 0)):.2%}")
                st.metric("💎 Profit/Loss Ratio", f"{float(metrics.get('profit_loss_ratio', 0)):.2f}")
    
    # 加载推荐数据
    df = load_recommendations(date_str)
    
    if df is None:
        st.error(f"❌ No recommendations found for {date_str}")
        st.info("💡 Please run the main program first to generate recommendations")
        st.code("python main_live_trading_enhanced.py", language="bash")
        return
    
    # 主内容区 - 使用标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Analysis", "📈 Charts", "🎯 Backtest"])
    
    # ========== Tab 1: 概览 ==========
    with tab1:
        st.markdown("## 📈 Today's Overview")
        
        # 指标卡片
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📦 Stocks", len(df))
        
        with col2:
            strong_buy = len(df[df['recommend_level'] == 'Strong Buy'])
            st.metric("⭐ Strong Buy", strong_buy, delta=f"{strong_buy/len(df)*100:.0f}%")
        
        with col3:
            if 'return_5d' in df.columns:
                avg_return = df['return_5d'].mean()
                st.metric("📈 Avg Return", f"{avg_return:.2%}", delta="Good" if avg_return > 0 else "Bad")
        
        with col4:
            avg_score = df['position'].mean()
            st.metric("🎯 Avg Score", f"{avg_score:.3f}")
        
        with col5:
            if 'ml_score' in df.columns and 'stockranker_score' in df.columns:
                consensus = len(df[(df['ml_score'] > 0.7) & (df['stockranker_score'] > 0.7)])
                st.metric("🤝 Consensus", consensus, delta=f"{consensus/len(df)*100:.0f}%")
        
        st.markdown("---")
        
        # Top 3推荐
        st.markdown("## ⭐ Top 3 Recommendations")
        
        cols = st.columns(3)
        
        for idx, (i, row) in enumerate(df.head(3).iterrows()):
            with cols[idx]:
                # 级别徽章
                level = row.get('recommend_level', 'N/A')
                if level == 'Strong Buy':
                    badge_class = 'badge-success'
                    icon = '🔥'
                elif level == 'Buy':
                    badge_class = 'badge-info'
                    icon = '✨'
                else:
                    badge_class = 'badge-warning'
                    icon = '📌'
                
                st.markdown(f"""
                <div class="stock-card">
                    <h3>{icon} #{idx+1} {row['instrument']}</h3>
                    <span class="badge {badge_class}">{level}</span>
                    <p style="margin-top: 1rem;"><strong>综合评分:</strong> <span style="color: #667eea; font-size: 1.3rem;">{row['position']:.4f}</span></p>
                    <p><strong>当前价格:</strong> ¥{row.get('close', 0):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if 'return_5d' in row and pd.notna(row['return_5d']):
                    ret = row['return_5d']
                    color_class = 'positive' if ret > 0 else 'negative'
                    st.markdown(f'<p class="{color_class}">5日涨跌: {ret:+.2%}</p>', unsafe_allow_html=True)
                
                if 'industry' in row and pd.notna(row['industry']):
                    st.caption(f"🏭 {row['industry']}")
                
                # 双评分对比
                if 'stockranker_score' in row and 'ml_score' in row:
                    with st.expander("📊 Score Details"):
                        st.progress(float(row['stockranker_score']), text=f"SR: {row['stockranker_score']:.3f}")
                        st.progress(float(row['ml_score']), text=f"ML: {row['ml_score']:.3f}")
        
        st.markdown("---")
        
        # 完整推荐表
        st.markdown("## 🎯 Complete Recommendations")
        
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
        
        # 格式化
        if 'close' in df_display.columns:
            df_display['close'] = df_display['close'].apply(lambda x: f"¥{x:.2f}")
        if 'return_5d' in df_display.columns:
            df_display['return_5d'] = df_display['return_5d'].apply(lambda x: f"{x:+.2%}")
        
        # 重命名
        rename_dict = {
            'instrument': '📌 Code',
            'recommend_level': '⭐ Level',
            'position': '🎯 Score',
            'stockranker_score': '📊 SR',
            'ml_score': '🤖 ML',
            'close': '💰 Price',
            'return_5d': '📈 5D',
            'risk_level': '⚠️ Risk'
        }
        
        df_display = df_display.rename(columns=rename_dict)
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    # ========== Tab 2: 详细分析 ==========
    with tab2:
        st.markdown("## 🔬 Detailed Analysis")
        
        # Top 10条形图
        st.markdown("### 🎯 Top 10 Score Comparison")
        fig1 = plot_top_picks_bar(df)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏭 Industry Distribution")
            fig2 = plot_sector_distribution(df)
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("ℹ️ Industry data not available")
        
        with col2:
            st.markdown("### 📈 Risk-Return Profile")
            fig3 = plot_risk_return_bubble(df)
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("ℹ️ Technical indicators not available")
    
    # ========== Tab 3: 高级图表 ==========
    with tab3:
        st.markdown("## 📈 Advanced Charts")
        
        if 'stockranker_score' in df.columns and 'ml_score' in df.columns:
            st.markdown("### 🔬 Model Consensus Analysis")
            fig4 = plot_score_comparison(df)
            if fig4:
                st.plotly_chart(fig4, use_container_width=True)
                
                with st.expander("💡 How to Read This Chart"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        **Quadrant Interpretation:**
                        - **Top-Right (Green)**: Both models bullish ⭐ Best opportunities
                        - **Top-Left**: ML bullish, SR cautious → Possible short-term catalyst
                        - **Bottom-Right**: SR bullish, ML cautious → Possible overvaluation
                        - **Bottom-Left (Red)**: Both models cautious ❌ Avoid
                        """)
                    with col2:
                        st.markdown("""
                        **Visual Elements:**
                        - **Bubble Size**: Final fusion score
                        - **Bubble Color**: Score gradient (dark to light)
                        - **Diagonal Line**: Perfect agreement reference
                        - **Cross Lines**: Quadrant dividers (0.5, 0.5)
                        """)
        else:
            st.info("ℹ️ Requires both StockRanker and ML scores")
        
        st.markdown("---")
        
        # 评分时序（如果有历史数据）
        factor_data = load_factor_data()
        if factor_data is not None:
            st.markdown("### 📊 Score Timeline (Top 5)")
            # 这里可以添加时序图
            st.info("📈 Score timeline chart will be displayed here")
        else:
            st.info("ℹ️ Historical data needed for timeline chart")
    
    # ========== Tab 4: 回测绩效 ==========
    with tab4:
        st.markdown("## 🎯 Backtest Performance")
        
        metrics = load_backtest_metrics()
        
        if metrics:
            # 核心指标
            st.markdown("### 📊 Core Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_return = float(metrics.get('total_return', 0))
                st.metric("💰 Total Return", f"{total_return:.2%}", 
                         delta="Excellent" if total_return > 0.2 else "Good" if total_return > 0.1 else "Fair")
            
            with col2:
                sharpe = float(metrics.get('sharpe_ratio', 0))
                st.metric("⚡ Sharpe Ratio", f"{sharpe:.3f}",
                         delta="Great" if sharpe > 1.5 else "Good" if sharpe > 1.0 else "Fair")
            
            with col3:
                win_rate = float(metrics.get('win_rate', 0))
                st.metric("🎲 Win Rate", f"{win_rate:.2%}",
                         delta="High" if win_rate > 0.55 else "Medium" if win_rate > 0.5 else "Low")
            
            with col4:
                pl_ratio = float(metrics.get('profit_loss_ratio', 0))
                st.metric("💎 P/L Ratio", f"{pl_ratio:.2f}",
                         delta="Excellent" if pl_ratio > 2.0 else "Good" if pl_ratio > 1.5 else "Fair")
            
            st.markdown("---")
            
            # 详细指标
            st.markdown("### 📋 Detailed Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**🎯 Return Metrics**")
                st.metric("Annual Return", f"{float(metrics.get('annualized_return', 0)):.2%}")
                st.metric("Best Month", f"{float(metrics.get('best_month', 0)):.2%}")
                st.metric("Worst Month", f"{float(metrics.get('worst_month', 0)):.2%}")
            
            with col2:
                st.markdown("**⚠️ Risk Metrics**")
                st.metric("Max Drawdown", f"{float(metrics.get('max_drawdown', 0)):.2%}")
                st.metric("Volatility", f"{float(metrics.get('volatility', 0)):.2%}")
                st.metric("Sortino Ratio", f"{float(metrics.get('sortino_ratio', 0)):.3f}")
            
            with col3:
                st.markdown("**📊 Trading Metrics**")
                st.metric("Total Trades", f"{int(float(metrics.get('total_trades', 0)))}")
                st.metric("Winning Trades", f"{int(float(metrics.get('winning_trades', 0)))}")
                st.metric("Avg Holding Days", f"{float(metrics.get('avg_holding_days', 0)):.1f}")
            
        else:
            st.info("ℹ️ No backtest metrics available. Run backtest first.")
            st.code("python main_enhanced.py", language="bash")
    
    # 页脚
    st.markdown("---")
    st.markdown("""
    <div class="footer-brand">
        <p>🚀 <strong>Powered by Dayiwu</strong> - AI Quantitative Trading System v4.0</p>
        <p style="font-size: 0.8rem; color: #999; margin-top: 0.5rem;">
            Advanced Machine Learning • Multi-Factor Models • Real-time Analytics
        </p>
        <p style="font-size: 0.75rem; color: #bbb; margin-top: 0.5rem;">
            © 2025 Dayiwu. All rights reserved. | Built with ❤️ and Python
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
