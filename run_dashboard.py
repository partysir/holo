"""
run_dashboard.py - Dashboard快速启动脚本

功能：
1. 检查环境和依赖
2. 验证数据完整性
3. 启动Streamlit Dashboard
4. 自动打开浏览器

使用方法：
python run_dashboard.py
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def print_banner():
    """打印欢迎横幅"""
    print("\n" + "=" * 80)
    print("    AI 量化策略监控台 - Dashboard 启动器 v3.0")
    print("=" * 80)
    print()


def check_streamlit():
    """检查Streamlit是否安装"""
    try:
        import streamlit
        print("✓ Streamlit 已安装 (v{})".format(streamlit.__version__))
        return True
    except ImportError:
        print("✗ Streamlit 未安装")
        print("\n请安装 Streamlit:")
        print("  pip install streamlit")
        return False


def check_dependencies():
    """检查必要的依赖"""
    print("\n【检查依赖】")

    dependencies = {
        'streamlit': 'Streamlit (Web框架)',
        'pandas': 'Pandas (数据处理)',
        'plotly': 'Plotly (可视化)',
        'PIL': 'Pillow (图像处理)'
    }

    missing = []

    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name}")
            missing.append(module)

    if missing:
        print(f"\n缺少依赖: {', '.join(missing)}")
        print("\n安装命令:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def check_data():
    """检查数据完整性"""
    print("\n【检查数据】")

    reports_dir = Path("./reports")
    live_dir = Path("./live_trading")

    # 检查回测报告
    if reports_dir.exists():
        report_count = len(list(reports_dir.iterdir()))
        print(f"  ✓ 回测报告: {report_count} 个")
    else:
        print("  ⚠️  回测报告目录不存在")
        print("     请先运行: python main-2.py")

    # 检查实盘数据
    if live_dir.exists():
        order_files = list(live_dir.glob("trading_orders_*.csv"))
        print(f"  ✓ 实盘记录: {len(order_files)} 个")
    else:
        print("  ⚠️  实盘数据目录不存在")
        print("     请先运行: python main_live_trading_enhanced.py")

    # 检查状态文件
    state_file = Path("./live_trading_state.json")
    if state_file.exists():
        print("  ✓ 系统状态文件")
    else:
        print("  ℹ️  状态文件不存在（首次运行正常）")


def check_dashboard_file():
    """检查dashboard.py是否存在"""
    dashboard_file = Path("./dashboard.py")

    if not dashboard_file.exists():
        print("\n❌ 未找到 dashboard.py 文件")
        print("   请确保 dashboard.py 在当前目录下")
        return False

    print("\n✓ Dashboard 文件就绪")
    return True


def start_dashboard(port=8501, open_browser=True):
    """启动Streamlit Dashboard"""
    print(f"\n【启动 Dashboard】")
    print(f"  端口: {port}")
    print(f"  URL: http://localhost:{port}")
    print("\n提示: 按 Ctrl+C 停止服务")
    print("-" * 80)

    # 构建命令
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "dashboard.py",
        "--server.port", str(port),
        "--server.headless", "true"
    ]

    # 启动进程
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )

        # 等待启动
        print("\n⏳ 正在启动...")
        time.sleep(3)

        # 自动打开浏览器
        if open_browser:
            print("🌐 打开浏览器...")
            webbrowser.open(f"http://localhost:{port}")

        print("\n✅ Dashboard 已启动！\n")

        # 持续输出日志
        for line in process.stdout:
            print(line, end='')

        process.wait()

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断，正在关闭...")
        process.terminate()
        process.wait()
        print("✓ Dashboard 已关闭")

    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        return False

    return True


def show_usage_tips():
    """显示使用提示"""
    print("\n" + "=" * 80)
    print("📖 使用提示")
    print("=" * 80)
    print("""
1. **首次使用**
   - 确保已运行回测系统生成报告
   - 运行命令: python main-2.py

2. **实盘监控**
   - 运行实盘系统: python main_live_trading_enhanced.py
   - 在Dashboard中选择实盘日期查看

3. **功能导航**
   - Tab 1-4: 回测分析
   - Tab 5: 实盘建议
   - Tab 6: 模型监控

4. **常见问题**
   - 数据不显示: 检查是否选择了日期
   - 图表缺失: 查看相关报告文件是否存在
   - 刷新数据: 点击侧边栏的刷新按钮

5. **停止Dashboard**
   - 在终端按 Ctrl+C
   - 或关闭浏览器标签页
""")


def main():
    """主函数"""
    print_banner()

    # 1. 检查环境
    if not check_streamlit():
        return

    if not check_dependencies():
        return

    if not check_dashboard_file():
        return

    # 2. 检查数据
    check_data()

    # 3. 显示使用提示
    show_usage_tips()

    # 4. 询问是否启动
    print("\n" + "=" * 80)
    response = input("\n是否启动 Dashboard？ (y/n): ").lower()

    if response != 'y':
        print("\n已取消启动")
        return

    # 5. 启动Dashboard
    success = start_dashboard(port=8501, open_browser=True)

    if not success:
        print("\n启动失败，请检查错误信息")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()