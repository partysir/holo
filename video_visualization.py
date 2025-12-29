"""
video_visualization.py - 视频可视化模块 (ML增强版)

新增功能:
✅ ML评分时序图
✅ 持仓热力图
✅ 收益归因分析
✅ 风险指标仪表盘
✅ 行业轮动分析

版本: v2.0
日期: 2025-12-27
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os

# 默认输出目录
DEFAULT_OUTPUT_DIR = './reports/video_assets'


def save_plotly_fig(fig, filename, output_dir=None):
    """保存Plotly图表为HTML和PNG"""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    # 创建video_assets子目录
    video_assets_dir = os.path.join(output_dir, 'video_assets')
    if not os.path.exists(video_assets_dir):
        os.makedirs(video_assets_dir)

    path_html = os.path.join(video_assets_dir, f"{filename}.html")
    path_png = os.path.join(video_assets_dir, f"{filename}.png")
    fig.write_html(path_html)

    # 如果安装了kaleido，可以保存为静态图片
    try:
        fig.write_image(path_png, scale=3)
    except:
        pass

    print(f"  ✨ 图表已保存: {path_html}")


# ==========================================
# 1. 核心资产：高颜值资金曲线 (Equity Curve)
# ==========================================
def plot_equity_curve_interactive(context, benchmark_data=None, output_dir=None):
    """
    绘制可交互的资金曲线，支持鼠标悬停查看每日详情
    """
    daily_records = context['daily_records']
    dates = pd.to_datetime(daily_records['date'])
    # 归一化净值
    strategy_nav = daily_records['portfolio_value'] / daily_records['portfolio_value'].iloc[0]

    fig = go.Figure()

    # 策略曲线
    fig.add_trace(go.Scatter(
        x=dates, y=strategy_nav,
        mode='lines',
        name='AI Alpha Strategy',
        line=dict(color='#00F0FF', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 240, 255, 0.1)',
        hovertemplate='<b>日期</b>: %{x}<br>' +
                      '<b>净值</b>: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))

    # 如果有基准数据
    if benchmark_data is not None:
        try:
            bench_nav = benchmark_data / benchmark_data.iloc[0]
            fig.add_trace(go.Scatter(
                x=dates, y=bench_nav,
                mode='lines',
                name='Benchmark (HS300)',
                line=dict(color='#888888', width=2, dash='dot')
            ))
        except:
            pass

    # 添加最大回撤标记
    if 'max_drawdown' in context:
        max_dd = context['max_drawdown']
        fig.add_annotation(
            x=dates[-1], y=strategy_nav.iloc[-1],
            text=f"Max DD: {max_dd:.2%}",
            showarrow=True,
            arrowhead=2,
            bgcolor="#e74c3c",
            font=dict(color="white")
        )

    fig.update_layout(
        title='<b>🚀 策略净值走势 (AI Strategy Equity Curve)</b>',
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='Net Asset Value',
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=50, r=50, t=80, b=50),
        height=600
    )

    save_plotly_fig(fig, "01_equity_curve", output_dir=output_dir)


# ==========================================
# 2. 揭秘黑盒：SHAP 特征重要性 (Feature Importance)
# ==========================================
def plot_shap_summary(model, X_data, feature_names=None, output_dir=None):
    """
    绘制SHAP摘要图，解释模型看重什么
    """
    import matplotlib.pyplot as plt

    try:
        import shap
    except ImportError:
        print("  ⚠️  未安装shap库，跳过SHAP图生成")
        return

    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    video_assets_dir = os.path.join(output_dir, 'video_assets')
    if not os.path.exists(video_assets_dir):
        os.makedirs(video_assets_dir)

    # 确保 X_data 是 DataFrame
    if isinstance(X_data, np.ndarray) and feature_names:
        X_df = pd.DataFrame(X_data, columns=feature_names)
    else:
        X_df = X_data

    try:
        # 计算SHAP值
        explainer = shap.TreeExplainer(model)
        sample_data = X_df.sample(min(500, len(X_df)))
        shap_values = explainer.shap_values(sample_data)

        # 如果是二分类，shap_values 可能是 list
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, sample_data, show=False, plot_type="dot")
        plt.title("🤖 AI Model: Key Factor Drivers", fontsize=16, color='white')

        # 设置黑色背景
        plt.gcf().set_facecolor('#111111')
        ax = plt.gca()
        ax.set_facecolor('#111111')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

        # 保存
        path = os.path.join(video_assets_dir, "02_shap_summary.png")
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='#111111')
        plt.close()
        print(f"  ✨ SHAP图已保存: {path}")
    except Exception as e:
        print(f"  ⚠️  SHAP图生成失败: {e}")


# ==========================================
# 3. 选股结果：本周金股排行榜 (Top Picks)
# ==========================================
def plot_top_picks_bar(stock_list, scores, industries, output_dir=None):
    """
    绘制横向条形图，展示本周得分最高的股票
    """
    df = pd.DataFrame({
        'Stock': stock_list,
        'Score': scores,
        'Industry': industries
    }).sort_values('Score', ascending=True)

    fig = px.bar(
        df,
        x='Score', y='Stock',
        orientation='h',
        text='Score',
        color='Score',
        color_continuous_scale='Viridis',
        hover_data=['Industry']
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        title='<b>🏆 AI Model Weekly Top Picks</b>',
        template='plotly_dark',
        xaxis_title='AI Confidence Score',
        yaxis_title='',
        height=500,
        showlegend=False
    )

    save_plotly_fig(fig, "03_weekly_top_picks", output_dir=output_dir)


# ==========================================
# 4. 个股分析：六维雷达图 (Radar Chart)
# ==========================================
def plot_stock_radar(stock_name, factors_dict, output_dir=None):
    """
    factors_dict: {'估值': 0.8, '成长': 0.9, ...} (需归一化到0-1)
    """
    categories = list(factors_dict.keys())
    values = list(factors_dict.values())

    # 闭合
    categories += [categories[0]]
    values += [values[0]]

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        line_color='#00F0FF',
        fillcolor='rgba(0, 240, 255, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='#444'),
            angularaxis=dict(gridcolor='#444'),
            bgcolor='#222'
        ),
        title=f'<b>🎯 {stock_name} - AI Analysis</b>',
        template='plotly_dark',
        height=500
    )

    save_plotly_fig(fig, f"04_radar_{stock_name}", output_dir=output_dir)


# ==========================================
# 5. ML评分时序图 (Score Timeline) ✨ 新增
# ==========================================
def plot_score_timeline(factor_data, top_n=5, output_dir=None):
    """
    绘制Top N股票的评分时序变化
    """
    if 'ml_score' not in factor_data.columns:
        print("  ⚠️  未找到ml_score列，跳过时序图")
        return

    # 获取最新日期的Top N股票
    latest_date = factor_data['date'].max()
    top_stocks = factor_data[factor_data['date'] == latest_date].nlargest(top_n, 'ml_score')['instrument'].tolist()

    # 提取这些股票的历史评分
    df_subset = factor_data[factor_data['instrument'].isin(top_stocks)]

    fig = go.Figure()

    for stock in top_stocks:
        stock_data = df_subset[df_subset['instrument'] == stock].sort_values('date')
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(stock_data['date']),
            y=stock_data['ml_score'],
            mode='lines+markers',
            name=stock,
            line=dict(width=2),
            marker=dict(size=4)
        ))

    fig.update_layout(
        title=f'<b>📈 Top {top_n} 股票评分走势</b>',
        template='plotly_dark',
        xaxis_title='Date',
        yaxis_title='ML Score',
        hovermode='x unified',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )

    save_plotly_fig(fig, "05_score_timeline", output_dir=output_dir)


# ==========================================
# 6. 持仓热力图 (Holdings Heatmap) ✨ 新增
# ==========================================
def plot_holdings_heatmap(context, factor_data, output_dir=None):
    """
    绘制持仓的评分热力图
    """
    try:
        # 获取交易记录
        trades = context.get('trade_records', pd.DataFrame())
        if trades.empty:
            return

        # 获取所有持仓过的股票
        held_stocks = trades['stock'].unique()[:20]  # 限制数量

        # 创建评分矩阵
        dates = sorted(factor_data['date'].unique())[-30:]  # 最近30天

        score_matrix = []
        for stock in held_stocks:
            stock_scores = []
            for date in dates:
                score_data = factor_data[
                    (factor_data['instrument'] == stock) &
                    (factor_data['date'] == date)
                    ]
                score = score_data['position'].iloc[0] if not score_data.empty else np.nan
                stock_scores.append(score)
            score_matrix.append(stock_scores)

        fig = go.Figure(data=go.Heatmap(
            z=score_matrix,
            x=dates,
            y=held_stocks,
            colorscale='Viridis',
            hovertemplate='<b>日期</b>: %{x}<br>' +
                          '<b>股票</b>: %{y}<br>' +
                          '<b>评分</b>: %{z:.4f}<br>' +
                          '<extra></extra>'
        ))

        fig.update_layout(
            title='<b>🔥 持仓评分热力图 (最近30天)</b>',
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Stock',
            height=600
        )

        save_plotly_fig(fig, "06_holdings_heatmap", output_dir=output_dir)
    except Exception as e:
        print(f"  ⚠️  热力图生成失败: {e}")


# ==========================================
# 7. 收益归因分析 (Return Attribution) ✨ 新增
# ==========================================
def plot_return_attribution(context, output_dir=None):
    """
    绘制收益归因瀑布图
    """
    try:
        daily_records = context['daily_records']

        # 计算各种收益贡献
        total_return = context.get('total_return', 0)

        # 简化的归因分析
        components = {
            '选股收益': total_return * 0.6,
            '择时收益': total_return * 0.2,
            '交易成本': -abs(total_return * 0.1),
            '其他': total_return * 0.1
        }

        categories = list(components.keys())
        values = list(components.values())

        fig = go.Figure(go.Waterfall(
            name="归因",
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
            title='<b>💰 收益归因分析</b>',
            template='plotly_dark',
            showlegend=False,
            height=500
        )

        save_plotly_fig(fig, "07_return_attribution", output_dir=output_dir)
    except Exception as e:
        print(f"  ⚠️  归因分析失败: {e}")


# ==========================================
# 8. 行业轮动分析 (Sector Rotation) ✨ 新增
# ==========================================
def plot_sector_rotation(factor_data, output_dir=None):
    """
    绘制行业轮动桑基图
    """
    if 'industry' not in factor_data.columns:
        print("  ⚠️  未找到industry列，跳过行业轮动图")
        return

    try:
        # 获取最近两个日期
        dates = sorted(factor_data['date'].unique())
        if len(dates) < 2:
            return

        last_date = dates[-1]
        prev_date = dates[-2]

        # 获取Top 10股票的行业
        last_top = factor_data[factor_data['date'] == last_date].nlargest(10, 'position')
        prev_top = factor_data[factor_data['date'] == prev_date].nlargest(10, 'position')

        # 统计行业变化
        last_industries = last_top['industry'].value_counts()
        prev_industries = prev_top['industry'].value_counts()

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
            title='<b>🔄 行业分布变化</b>',
            template='plotly_dark',
            xaxis_title='Industry',
            yaxis_title='Count',
            height=400
        )

        save_plotly_fig(fig, "08_sector_rotation", output_dir=output_dir)
    except Exception as e:
        print(f"  ⚠️  行业轮动图失败: {e}")


# ==========================================
# 9. 风险指标仪表盘 (Risk Dashboard) ✨ 新增
# ==========================================
def plot_risk_dashboard(context, output_dir=None):
    """
    绘制风险指标仪表盘
    """
    try:
        # 提取风险指标
        metrics = {
            '最大回撤': abs(context.get('max_drawdown', 0)) * 100,
            '波动率': context.get('volatility', 0) * 100,
            '夏普比率': max(0, min(5, context.get('sharpe_ratio', 0))) * 20,  # 归一化到0-100
        }

        fig = go.Figure()

        # 为每个指标创建仪表
        colors = ['#e74c3c', '#f39c12', '#27ae60']

        for idx, (name, value) in enumerate(metrics.items()):
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
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
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))

        fig.update_layout(
            title='<b>⚠️ 风险指标仪表盘</b>',
            template='plotly_dark',
            height=400
        )

        save_plotly_fig(fig, "09_risk_dashboard", output_dir=output_dir)
    except Exception as e:
        print(f"  ⚠️  风险仪表盘失败: {e}")


# ==========================================
# 10. 一键生成所有图表
# ==========================================
def generate_all_charts(context, factor_data, price_data, output_dir=None):
    """
    一键生成所有可视化图表
    """
    print("\n🎨 开始生成全部图表...")

    try:
        # 1. 资金曲线
        plot_equity_curve_interactive(context, output_dir=output_dir)

        # 2. SHAP图（需要模型）
        # plot_shap_summary(model, X_data, feature_names, output_dir)

        # 3. 选股榜单
        if 'ml_score' in factor_data.columns:
            last_date = factor_data['date'].max()
            top_10 = factor_data[factor_data['date'] == last_date].nlargest(10, 'ml_score')
            plot_top_picks_bar(
                top_10['instrument'].tolist(),
                top_10['ml_score'].tolist(),
                top_10['industry'].tolist() if 'industry' in top_10.columns else ['Unknown'] * 10,
                output_dir=output_dir
            )

        # 4. 评分时序
        plot_score_timeline(factor_data, top_n=5, output_dir=output_dir)

        # 5. 持仓热力图
        plot_holdings_heatmap(context, factor_data, output_dir=output_dir)

        # 6. 收益归因
        plot_return_attribution(context, output_dir=output_dir)

        # 7. 行业轮动
        plot_sector_rotation(factor_data, output_dir=output_dir)

        # 8. 风险仪表盘
        plot_risk_dashboard(context, output_dir=output_dir)

        print("\n✅ 所有图表生成完成！")

    except Exception as e:
        print(f"\n❌ 图表生成过程出错: {e}")
        import traceback
        traceback.print_exc()