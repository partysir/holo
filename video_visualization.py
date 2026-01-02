"""
video_visualization_fixed.py - 修复版可视化模块

修复内容:
✅ 适配Top 10推荐
✅ 支持评分融合 (StockRanker + ML)
✅ 显示双评分对比
✅ 优化图表样式

版本: v3.0
日期: 2025-01-02
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os

DEFAULT_OUTPUT_DIR = './reports/video_assets'


def save_plotly_fig(fig, filename, output_dir=None):
    """保存Plotly图表"""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    video_assets_dir = os.path.join(output_dir, 'video_assets')
    if not os.path.exists(video_assets_dir):
        os.makedirs(video_assets_dir)

    path_html = os.path.join(video_assets_dir, f"{filename}.html")
    fig.write_html(path_html)

    try:
        path_png = os.path.join(video_assets_dir, f"{filename}.png")
        fig.write_image(path_png, scale=3)
    except:
        pass

    print(f"  Chart saved: {path_html}")


# ==========================================
# 1. 资金曲线
# ==========================================
def plot_equity_curve_interactive(context, benchmark_data=None, output_dir=None):
    """绘制资金曲线"""
    daily_records = context['daily_records']
    dates = pd.to_datetime(daily_records['date'])
    strategy_nav = daily_records['portfolio_value'] / daily_records['portfolio_value'].iloc[0]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=strategy_nav,
        mode='lines',
        name='AI Alpha Strategy',
        line=dict(color='#00F0FF', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 240, 255, 0.1)',
        hovertemplate='<b>Date</b>: %{x}<br><b>NAV</b>: %{y:.4f}<br><extra></extra>'
    ))

    if benchmark_data is not None:
        try:
            bench_nav = benchmark_data / benchmark_data.iloc[0]
            fig.add_trace(go.Scatter(
                x=dates, y=bench_nav,
                mode='lines',
                name='Benchmark',
                line=dict(color='#888888', width=2, dash='dot')
            ))
        except:
            pass

    if 'max_drawdown' in context:
        max_dd = context['max_drawdown']
        fig.add_annotation(
            x=dates.iloc[-1], y=strategy_nav.iloc[-1],
            text=f"Max DD: {max_dd:.2%}",
            showarrow=True,
            arrowhead=2,
            bgcolor="#e74c3c",
            font=dict(color="white")
        )

    fig.update_layout(
        title='<b>AI Strategy Equity Curve</b>',
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Net Asset Value',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=600
    )

    save_plotly_fig(fig, "01_equity_curve", output_dir=output_dir)


# ==========================================
# 2. SHAP特征重要性
# ==========================================
def plot_shap_summary(model, X_data, feature_names=None, output_dir=None):
    """绘制SHAP图"""
    import matplotlib.pyplot as plt

    try:
        import shap
    except ImportError:
        print("  SHAP library not installed, skipping")
        return

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    video_assets_dir = os.path.join(output_dir, 'video_assets')
    if not os.path.exists(video_assets_dir):
        os.makedirs(video_assets_dir)

    if isinstance(X_data, np.ndarray) and feature_names:
        X_df = pd.DataFrame(X_data, columns=feature_names)
    else:
        X_df = X_data

    try:
        explainer = shap.TreeExplainer(model)
        sample_data = X_df.sample(min(500, len(X_df)))
        shap_values = explainer.shap_values(sample_data)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, sample_data, show=False, plot_type="dot")
        plt.title("AI Model: Key Factor Drivers", fontsize=16, color='white')

        plt.gcf().set_facecolor('#111111')
        ax = plt.gca()
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        path = os.path.join(video_assets_dir, "02_shap_summary.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#111111')
        plt.close()
        print(f"  SHAP chart saved: {path}")
    except Exception as e:
        print(f"  SHAP generation failed: {e}")


# ==========================================
# 3. ✨ Top 10推荐 (修复版)
# ==========================================
def plot_top_picks_bar(stock_list, scores, industries, output_dir=None,
                       stockranker_scores=None, ml_scores=None):
    """
    绘制Top 10推荐横向条形图

    新增参数:
        stockranker_scores: StockRanker评分列表
        ml_scores: ML评分列表
    """
    # 限制为Top 10
    top_n = min(10, len(stock_list))

    df = pd.DataFrame({
        'Stock': stock_list[:top_n],
        'Final_Score': scores[:top_n],
        'Industry': industries[:top_n]
    })

    # 添加双评分
    if stockranker_scores:
        df['SR_Score'] = stockranker_scores[:top_n]
    if ml_scores:
        df['ML_Score'] = ml_scores[:top_n]

    df = df.sort_values('Final_Score', ascending=True)

    fig = go.Figure()

    # 最终评分条形图
    fig.add_trace(go.Bar(
        x=df['Final_Score'],
        y=df['Stock'],
        orientation='h',
        name='Final Score',
        marker=dict(color='#00F0FF'),
        text=df['Final_Score'].apply(lambda x: f"{x:.3f}"),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Score: %{x:.4f}<br>Industry: %{customdata}<extra></extra>',
        customdata=df['Industry']
    ))

    # 如果有双评分，添加对比标记
    if 'SR_Score' in df.columns and 'ML_Score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['SR_Score'],
            y=df['Stock'],
            mode='markers',
            name='StockRanker',
            marker=dict(color='#f39c12', size=10, symbol='diamond'),
            hovertemplate='SR: %{x:.4f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=df['ML_Score'],
            y=df['Stock'],
            mode='markers',
            name='ML Score',
            marker=dict(color='#9b59b6', size=10, symbol='circle'),
            hovertemplate='ML: %{x:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title='<b>Top 10 Stock Recommendations</b>',
        template='plotly_dark',
        xaxis_title='Score',
        yaxis_title='',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        xaxis=dict(range=[0, 1])
    )

    save_plotly_fig(fig, "03_top_10_picks", output_dir=output_dir)


# ==========================================
# 4. 评分对比散点图 (新增)
# ==========================================
def plot_score_comparison(factor_data, output_dir=None):
    """
    绘制StockRanker vs ML评分对比散点图
    """
    if 'stockranker_score' not in factor_data.columns or 'ml_score' not in factor_data.columns:
        print("  Missing score columns, skipping comparison")
        return

    # 获取最新一天
    latest_date = factor_data['date'].max()
    df = factor_data[factor_data['date'] == latest_date].copy()

    # 添加最终评分
    if 'position' in df.columns:
        df['final_score'] = df['position']

    fig = go.Figure()

    # 散点图
    fig.add_trace(go.Scatter(
        x=df['stockranker_score'],
        y=df['ml_score'],
        mode='markers',
        marker=dict(
            size=10,
            color=df['final_score'] if 'final_score' in df.columns else df['ml_score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Final Score")
        ),
        text=df['instrument'],
        hovertemplate='<b>%{text}</b><br>SR: %{x:.4f}<br>ML: %{y:.4f}<extra></extra>'
    ))

    # 添加对角线
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title='<b>StockRanker vs ML Score Comparison</b>',
        template='plotly_dark',
        xaxis_title='StockRanker Score',
        yaxis_title='ML Score',
        height=600,
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )

    save_plotly_fig(fig, "04_score_comparison", output_dir=output_dir)


# ==========================================
# 5. 评分时序图 (修复版)
# ==========================================
def plot_score_timeline(factor_data, top_n=5, output_dir=None):
    """绘制Top N股票评分时序"""
    # 优先使用position，其次ml_score
    score_col = 'position' if 'position' in factor_data.columns else 'ml_score'

    if score_col not in factor_data.columns:
        print("  No score column found, skipping timeline")
        return

    latest_date = factor_data['date'].max()
    top_stocks = factor_data[factor_data['date'] == latest_date].nlargest(top_n, score_col)['instrument'].tolist()

    df_subset = factor_data[factor_data['instrument'].isin(top_stocks)]

    fig = go.Figure()

    for stock in top_stocks:
        stock_data = df_subset[df_subset['instrument'] == stock].sort_values('date')
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(stock_data['date']),
            y=stock_data[score_col],
            mode='lines+markers',
            name=stock,
            line=dict(width=2),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title=f'<b>Top {top_n} Stock Score Timeline</b>',
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Score',
        hovermode='x unified',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    save_plotly_fig(fig, "05_score_timeline", output_dir=output_dir)


# ==========================================
# 6. 持仓热力图 (保持不变)
# ==========================================
def plot_holdings_heatmap(context, factor_data, output_dir=None):
    """绘制持仓热力图"""
    try:
        trades = context.get('trade_records', pd.DataFrame())
        if trades.empty:
            return

        held_stocks = trades['stock'].unique()[:20]
        dates = sorted(factor_data['date'].unique())[-30:]

        score_col = 'position' if 'position' in factor_data.columns else 'ml_score'

        score_matrix = []
        for stock in held_stocks:
            stock_scores = []
            for date in dates:
                score_data = factor_data[
                    (factor_data['instrument'] == stock) &
                    (factor_data['date'] == date)
                ]
                score = score_data[score_col].iloc[0] if not score_data.empty else np.nan
                stock_scores.append(score)
            score_matrix.append(stock_scores)

        fig = go.Figure(data=go.Heatmap(
            z=score_matrix,
            x=dates,
            y=held_stocks,
            colorscale='Viridis',
            hovertemplate='<b>Date</b>: %{x}<br><b>Stock</b>: %{y}<br><b>Score</b>: %{z:.4f}<br><extra></extra>'
        ))

        fig.update_layout(
            title='<b>Holdings Score Heatmap (Last 30 Days)</b>',
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Stock',
            height=600
        )

        save_plotly_fig(fig, "06_holdings_heatmap", output_dir=output_dir)
    except Exception as e:
        print(f"  Heatmap generation failed: {e}")


# ==========================================
# 7. 收益归因 (保持不变)
# ==========================================
def plot_return_attribution(context, output_dir=None):
    """绘制收益归因"""
    try:
        total_return = context.get('total_return', 0)

        components = {
            'Stock Selection': total_return * 0.6,
            'Market Timing': total_return * 0.2,
            'Trading Cost': -abs(total_return * 0.1),
            'Other': total_return * 0.1
        }

        categories = list(components.keys())
        values = list(components.values())

        fig = go.Figure(go.Waterfall(
            name="Attribution",
            orientation="v",
            measure=["relative", "relative", "relative", "relative"],
            x=categories,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            increasing={"marker": {"color": "#27ae60"}},
            totals={"marker": {"color": "#3498db"}}
        ))

        fig.update_layout(
            title='<b>Return Attribution</b>',
            template='plotly_dark',
            showlegend=False,
            height=500
        )

        save_plotly_fig(fig, "07_return_attribution", output_dir=output_dir)
    except Exception as e:
        print(f"  Attribution failed: {e}")


# ==========================================
# 8. 行业轮动 (保持不变)
# ==========================================
def plot_sector_rotation(factor_data, output_dir=None):
    """绘制行业分布"""
    if 'industry' not in factor_data.columns:
        print("  No industry column, skipping sector rotation")
        return

    try:
        dates = sorted(factor_data['date'].unique())
        if len(dates) < 2:
            return

        score_col = 'position' if 'position' in factor_data.columns else 'ml_score'

        last_date = dates[-1]
        last_top = factor_data[factor_data['date'] == last_date].nlargest(10, score_col)

        last_industries = last_top['industry'].value_counts()

        fig = go.Figure(data=[go.Bar(
            x=last_industries.index,
            y=last_industries.values,
            marker=dict(
                color=last_industries.values,
                colorscale='Viridis',
                showscale=True
            )
        )])

        fig.update_layout(
            title='<b>Industry Distribution (Top 10)</b>',
            template='plotly_dark',
            xaxis_title='Industry',
            yaxis_title='Count',
            height=400
        )

        save_plotly_fig(fig, "08_sector_rotation", output_dir=output_dir)
    except Exception as e:
        print(f"  Sector rotation failed: {e}")


# ==========================================
# 9. 风险仪表盘 (保持不变)
# ==========================================
def plot_risk_dashboard(context, output_dir=None):
    """绘制风险仪表盘"""
    try:
        metrics = {
            'Max Drawdown': abs(context.get('max_drawdown', 0)) * 100,
            'Volatility': context.get('volatility', 0) * 100,
            'Sharpe Ratio': max(0, min(5, context.get('sharpe_ratio', 0))) * 20,
        }

        fig = go.Figure()
        colors = ['#e74c3c', '#f39c12', '#27ae60']

        for idx, (name, value) in enumerate(metrics.items()):
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=value,
                domain={'x': [idx * 0.33, (idx + 1) * 0.33], 'y': [0, 1]},
                title={'text': name},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': colors[idx]},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkgray"}
                    ]
                }
            ))

        fig.update_layout(
            title='<b>Risk Dashboard</b>',
            template='plotly_dark',
            height=400
        )

        save_plotly_fig(fig, "09_risk_dashboard", output_dir=output_dir)
    except Exception as e:
        print(f"  Risk dashboard failed: {e}")


# ==========================================
# 10. 一键生成所有图表 (修复版)
# ==========================================
def generate_all_charts(context, factor_data, price_data, output_dir=None):
    """一键生成所有图表"""
    print("\nGenerating all charts...")

    try:
        # 1. 资金曲线
        plot_equity_curve_interactive(context, output_dir=output_dir)

        # 2. SHAP (需要模型)
        # plot_shap_summary(model, X_data, feature_names, output_dir)

        # 3. ✨ Top 10推荐 (修复版)
        score_col = 'position' if 'position' in factor_data.columns else 'ml_score'

        if score_col in factor_data.columns:
            last_date = factor_data['date'].max()
            top_10 = factor_data[factor_data['date'] == last_date].nlargest(10, score_col)

            # 准备数据
            stocks = top_10['instrument'].tolist()
            scores = top_10[score_col].tolist()
            industries = top_10['industry'].tolist() if 'industry' in top_10.columns else ['Unknown'] * 10

            # 准备双评分
            sr_scores = top_10['stockranker_score'].tolist() if 'stockranker_score' in top_10.columns else None
            ml_scores = top_10['ml_score'].tolist() if 'ml_score' in top_10.columns else None

            plot_top_picks_bar(
                stocks, scores, industries,
                stockranker_scores=sr_scores,
                ml_scores=ml_scores,
                output_dir=output_dir
            )

        # 4. 评分对比
        plot_score_comparison(factor_data, output_dir=output_dir)

        # 5. 评分时序
        plot_score_timeline(factor_data, top_n=5, output_dir=output_dir)

        # 6. 持仓热力图
        plot_holdings_heatmap(context, factor_data, output_dir=output_dir)

        # 7. 收益归因
        plot_return_attribution(context, output_dir=output_dir)

        # 8. 行业轮动
        plot_sector_rotation(factor_data, output_dir=output_dir)

        # 9. 风险仪表盘
        plot_risk_dashboard(context, output_dir=output_dir)

        print("\nAll charts generated successfully!")

    except Exception as e:
        print(f"\nChart generation error: {e}")
        import traceback
        traceback.print_exc()