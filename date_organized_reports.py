"""
date_organized_reports.py - 按日期组织报告的模块

功能:
- 创建按日期命名的报告目录
- 将所有报告文件移动到对应的日期目录中
"""

import os
import shutil
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def get_current_date_folder():
    """
    获取当前日期的文件夹名称
    
    Returns:
        str: 日期格式为 YYYY-MM-DD 的字符串
    """
    return datetime.now().strftime('%Y-%m-%d')


def create_date_folder(base_dir='./reports'):
    """
    创建按当前日期命名的文件夹
    
    Args:
        base_dir (str): 基础目录路径
        
    Returns:
        str: 创建的日期文件夹路径
    """
    # 获取当前日期
    date_folder_name = get_current_date_folder()
    
    # 创建完整路径
    date_folder_path = os.path.join(base_dir, date_folder_name)
    
    # 创建目录（如果不存在）
    os.makedirs(date_folder_path, exist_ok=True)
    
    return date_folder_path


def move_reports_to_date_folder(source_dir='./reports', target_dir=None):
    """
    将报告文件移动到按日期命名的文件夹中
    
    Args:
        source_dir (str): 源目录路径
        target_dir (str): 目标日期目录路径，如果为None则自动创建
        
    Returns:
        str: 目标日期目录路径
    """
    # 如果没有指定目标目录，则创建一个新的日期目录
    if target_dir is None:
        target_dir = create_date_folder(source_dir)
    
    # 定义需要移动的文件列表
    report_files = [
        'monitoring_dashboard.png',
        'top_stocks_analysis.png',
        'performance_report.txt',
        'daily_holdings_detail.csv',
        'daily_holdings_summary.csv',
        'stock_holding_stats.csv',
        'today_holdings_dashboard.png',
        'today_holdings.csv'
    ]
    
    # 移动文件
    moved_files = []
    for filename in report_files:
        source_file = os.path.join(source_dir, filename)
        if os.path.exists(source_file):
            target_file = os.path.join(target_dir, filename)
            try:
                shutil.move(source_file, target_file)
                moved_files.append(filename)
            except Exception as e:
                print(f"⚠️  移动文件 {filename} 失败: {e}")
    
    print(f"✓ 已将 {len(moved_files)} 个报告文件移动到: {target_dir}")
    
    return target_dir


def generate_date_organized_reports(context, factor_data, price_data, base_dir='./reports'):
    """
    生成按日期组织的报告
    
    Args:
        context: 回测上下文
        factor_data: 因子数据
        price_data: 价格数据
        base_dir (str): 基础目录路径
        
    Returns:
        str: 日期目录路径
    """
    from visualization_module import (
        plot_monitoring_results,
        plot_top_stocks_evolution,
        generate_performance_report
    )
    
    from show_today_holdings import show_today_holdings_dashboard
    from holdings_monitor import generate_daily_holdings_report
    
    # 创建日期目录
    date_folder = create_date_folder(base_dir)
    
    print(f"\n📁 报告将保存到: {date_folder}")
    
    # 1. 监控面板
    print("  生成监控面板...")
    plot_monitoring_results(context, output_dir=date_folder)
    
    # 2. TOP股票分析
    print("  生成股票分析图...")
    plot_top_stocks_evolution(context, output_dir=date_folder)
    
    # 3. 绩效报告
    print("  生成绩效报告...")
    generate_performance_report(context, output_dir=date_folder)
    
    # 4. 每日持仓历史报告
    print("  生成持仓历史报告...")
    generate_daily_holdings_report(
        context=context,
        factor_data=factor_data,
        price_data=price_data,
        output_dir=date_folder,
        print_to_console=False,
        save_to_csv=True
    )
    
    # 5. 今日持仓仪表板
    print("\n  生成今日持仓仪表板...")
    show_today_holdings_dashboard(
        context=context,
        factor_data=factor_data,
        price_data=price_data,
        output_dir=date_folder
    )
    
    return date_folder


# 导出函数
__all__ = [
    'get_current_date_folder',
    'create_date_folder',
    'move_reports_to_date_folder',
    'generate_date_organized_reports'
]