"""
dashboard_fixed.py - 修复版Streamlit仪表盘

修复内容:
1. ✅ 修复UTF-8编码问题
2. ✅ 添加正确的运行说明
3. ✅ 改进错误处理

运行方式:
streamlit run dashboard_fixed.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json

# ========== 页面配置 ==========
st.set_page_config(
    page_title="A股量化交易系统",
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
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem;
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

@st.cache_data(ttl=300)  # 缓存5分钟
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
def load_report(date=None):
    """加载文本报告"""
    if date is None:
        date = datetime.now().strftime('%Y%m%d')

    report_path = f'./live_trading/stock_recommendations_{date}.txt'

    if not os.path.exists(report_path):
        return None

    try:
        # ✅ 修复: 显式指定UTF-8编码
        with open(report_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试GBK
        try:
            with open(report_path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception as e:
            st.error(f"报告编码错误: {e}")
            return None
    except Exception as e:
        st.error(f"加载报告失败: {e}")
        return None


def load_state():
    """加载交易状态"""
    state_file = './live_trading_state.json'

    if not os.path.exists(state_file):
        return {
            'last_rebalance_date': None,
            'positions': {},
            'cash': 1000000
        }

    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {
            'last_rebalance_date': None,
            'positions': {},
            'cash': 1000000
        }


# ========== 可视化函数 ==========

def plot_score_distribution(df):
    """评分分布图"""
    fig = go.Figure()

    if 'stockranker_score' in df.columns:
        fig.add_trace(go.Histogram(
            x=df['stockranker_score'],
            name='StockRanker',
            opacity=0.7,
            marker_color='#1f77b4'
        ))

    if 'ml_score' in df.columns:
        fig.add_trace(go.Histogram(
            x=df['ml_score'],
            name='ML Score',
            opacity=0.7,
            marker_color='#ff7f0e'
        ))

    fig.add_trace(go.Histogram(
        x=df['position'],
        name='Final Score',
        opacity=0.7,
        marker_color='#2ca02c'
    ))

    fig.update_layout(
        title='评分分布',
        xaxis_title='Score',
        yaxis_title='Count',
        barmode='overlay',
        height=400
    )

    return fig


def plot_risk_return(df):
    """风险-收益散点图"""
    if 'return_5d' not in df.columns or 'volatility_20d' not in df.columns:
        return None

    fig = px.scatter(
        df,
        x='volatility_20d',
        y='return_5d',
        size='position',
        color='recommend_level',
        hover_data=['instrument'],
        labels={
            'volatility_20d': '波动率 (20日)',
            'return_5d': '5日收益率',
            'position': 'Final Score'
        },
        title='风险-收益分布'
    )

    fig.update_layout(height=400)

    return fig


def plot_sector_distribution(df):
    """行业分布图"""
    if 'industry' not in df.columns:
        return None

    sector_counts = df['industry'].value_counts()

    fig = go.Figure(data=[
        go.Pie(
            labels=sector_counts.index,
            values=sector_counts.values,
            hole=0.4
        )
    ])

    fig.update_layout(
        title='行业分布',
        height=400
    )

    return fig


# ========== 主界面 ==========

def main():
    # 标题
    st.markdown('<div class="main-header">📊 A股量化交易系统仪表盘</div>', unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 控制面板")

        # 日期选择
        selected_date = st.date_input(
            "选择日期",
            value=datetime.now(),
            max_value=datetime.now()
        )
        date_str = selected_date.strftime('%Y%m%d')

        # 刷新按钮
        if st.button("🔄 刷新数据", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # 系统状态
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
        st.info("请先运行 `python main_live_trading_enhanced.py` 生成推荐")
        return

    # ========== 概览指标 ==========
    st.header("📈 今日概览")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "推荐股票数",
            len(df),
            delta=None
        )

    with col2:
        strong_buy = len(df[df['recommend_level'] == 'Strong Buy'])
        st.metric(
            "强力推荐",
            strong_buy,
            delta=f"{strong_buy/len(df)*100:.0f}%"
        )

    with col3:
        if 'return_5d' in df.columns:
            avg_return = df['return_5d'].mean()
            st.metric(
                "平均5日收益",
                f"{avg_return:.2%}",
                delta=f"{'📈' if avg_return > 0 else '📉'}"
            )

    with col4:
        avg_score = df['position'].mean()
        st.metric(
            "平均评分",
            f"{avg_score:.3f}",
            delta=None
        )

    st.divider()

    # ========== 详细推荐 ==========
    st.header("🎯 Top 10 推荐")

    # 创建数据表
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

    display_cols = [c for c in display_cols if c in df.columns]

    # 格式化显示
    df_display = df[display_cols].copy()

    if 'close' in df_display.columns:
        df_display['close'] = df_display['close'].apply(lambda x: f"¥{x:.2f}")
    if 'return_5d' in df_display.columns:
        df_display['return_5d'] = df_display['return_5d'].apply(lambda x: f"{x:.2%}")

    # 重命名列
    rename_dict = {
        'instrument': '代码',
        'recommend_level': '推荐等级',
        'position': '综合评分',
        'stockranker_score': '多因子',
        'ml_score': 'ML评分',
        'close': '价格',
        'return_5d': '5日涨跌',
        'risk_level': '风险'
    }

    df_display = df_display.rename(columns=rename_dict)

    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True
    )

    # ========== Top 3 重点推荐 ==========
    st.header("⭐ 重点推荐 (Top 3)")

    cols = st.columns(3)

    for idx, (i, row) in enumerate(df.head(3).iterrows()):
        with cols[idx]:
            st.markdown(f"""
            <div class="stock-card">
                <h3>#{idx+1} {row['instrument']}</h3>
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

    # ========== 可视化分析 ==========
    st.header("📊 可视化分析")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = plot_score_distribution(df)
        st.plotly_chart(fig1, use_container_width=True)

        fig3 = plot_sector_distribution(df)
        if fig3:
            st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig2 = plot_risk_return(df)
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    # ========== 完整报告 ==========
    st.header("📄 完整报告")

    report = load_report(date_str)

    if report:
        with st.expander("查看详细报告", expanded=False):
            st.text(report)
    else:
        st.warning("未找到完整报告文件")

    # ========== 页脚 ==========
    st.divider()
    st.caption(f"最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption("⚠️ 本系统仅供参考，不构成投资建议。投资有风险，入市需谨慎。")


# ========== 程序入口 ==========
if __name__ == "__main__":
    main()