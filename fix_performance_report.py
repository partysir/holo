#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复绩效报告中的异常收益率问题
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def analyze_performance_report(report_path):
    """分析绩效报告中的问题"""
    print("🔍 分析绩效报告")
    print("=" * 50)
    
    if not os.path.exists(report_path):
        print(f"❌ 报告文件不存在: {report_path}")
        return None
    
    # 读取报告内容
    with open(report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 提取关键信息
    initial_capital = 0
    final_value = 0
    total_return = 0
    annualized_return = 0
    
    for line in lines:
        if '初始资金' in line:
            initial_capital = float(line.split('¥')[1].replace(',', '').strip())
        elif '最终资产' in line:
            final_value = float(line.split('¥')[1].replace(',', '').strip())
        elif '总收益率' in line and '总收益率:' in line:
            total_return = float(line.split(':')[1].replace('%', '').replace('+', '').strip()) / 100
        elif '年化收益率' in line:
            annualized_return = float(line.split(':')[1].replace('%', '').replace('+', '').strip()) / 100
    
    print(f"初始资金: ¥{initial_capital:,.2f}")
    print(f"最终资产: ¥{final_value:,.2f}")
    print(f"总收益率: {total_return:+.2%}")
    print(f"年化收益率: {annualized_return:+.2%}")
    
    # 检查是否存在异常
    if total_return > 1000:  # 如果收益率超过1000%，则认为异常
        print("\n⚠️  检测到异常高的收益率!")
        print("可能的原因:")
        print("1. 初始资金设置过低")
        print("2. 最终资产计算错误")
        print("3. 存在数据错误或计算错误")
        
        # 计算合理的收益率
        correct_return = (final_value - initial_capital) / initial_capital
        print(f"\n✅ 正确的总收益率应该是: {correct_return:+.2%}")
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'reported_return': total_return,
            'correct_return': correct_return,
            'annualized_return': annualized_return
        }
    
    print("\n✅ 收益率在合理范围内")
    return None

def fix_performance_report(report_path):
    """修复绩效报告"""
    print("\n🔧 修复绩效报告")
    print("=" * 50)
    
    # 分析报告
    result = analyze_performance_report(report_path)
    if not result:
        print("无需修复")
        return
    
    # 读取报告内容
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    backup_path = report_path.replace('.txt', '_backup.txt')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ 原报告已备份到: {backup_path}")
    
    # 修复收益率显示
    # 替换总收益率
    old_total_return_line = f"总收益率: +{result['reported_return']*100:.2f}%"
    new_total_return_line = f"总收益率: +{result['correct_return']*100:.2f}%"
    content = content.replace(old_total_return_line, new_total_return_line)
    
    # 替换净收益率
    old_net_return_line = f"净收益率: +{result['reported_return']*100:.2f}%"
    new_net_return_line = f"净收益率: +{result['correct_return']*100:.2f}%"
    content = content.replace(old_net_return_line, new_net_return_line)
    
    # 重新计算年化收益率
    # 假设报告中有交易天数信息
    lines = content.split('\n')
    trading_days = 714  # 默认值，从报告中提取
    for line in lines:
        if '回测交易天数:' in line and '天' in line:
            try:
                trading_days = int(line.split('天')[0].split(':')[-1].strip().split()[0])
                break
            except:
                pass
    
    years = trading_days / 365
    correct_annualized_return = result['annualized_return']  # 默认值
    if years > 0 and result['correct_return'] > -1:
        correct_annualized_return = (1 + result['correct_return']) ** (1 / years) - 1
        old_annualized_line = f"年化收益率: +{result['annualized_return']*100:.2f}%"
        new_annualized_line = f"年化收益率: +{correct_annualized_return*100:.2f}%"
        content = content.replace(old_annualized_line, new_annualized_line)
        print(f"✅ 年化收益率已修正: {correct_annualized_return:+.2%}")
    
    # 保存修复后的报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 报告已修复并保存到: {report_path}")
    
    return {
        'original_return': result['reported_return'],
        'corrected_return': result['correct_return'],
        'original_annualized': result['annualized_return'],
        'corrected_annualized': correct_annualized_return
    }

def main():
    """主函数"""
    print("📈 绩效报告修复工具")
    print("=" * 50)
    
    # 查找最新的报告文件
    reports_dir = './reports'
    if not os.path.exists(reports_dir):
        print(f"❌ 报告目录不存在: {reports_dir}")
        return
    
    # 获取最新的报告文件
    latest_report = None
    latest_date = None
    
    for root, dirs, files in os.walk(reports_dir):
        for file in files:
            if file == 'performance_report.txt':
                report_path = os.path.join(root, file)
                # 提取日期
                try:
                    date_str = os.path.basename(root)  # 假设目录名是日期
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    if latest_date is None or date_obj > latest_date:
                        latest_date = date_obj
                        latest_report = report_path
                except:
                    if latest_report is None:
                        latest_report = report_path
    
    if latest_report:
        print(f"📄 找到最新报告: {latest_report}")
        fix_result = fix_performance_report(latest_report)
        if fix_result:
            print(f"\n📊 修复结果:")
            print(f"   原始总收益率: {fix_result['original_return']:+.2%}")
            print(f"   修正总收益率: {fix_result['corrected_return']:+.2%}")
            print(f"   原始年化收益率: {fix_result['original_annualized']:+.2%}")
            print(f"   修正年化收益率: {fix_result['corrected_annualized']:+.2%}")
    else:
        print("❌ 未找到绩效报告文件")

if __name__ == "__main__":
    main()