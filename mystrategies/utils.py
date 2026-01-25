import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from datetime import datetime
import platform

# 设置中文字体 - 根据操作系统选择合适字体
def setup_chinese_font():
    """根据操作系统设置中文字体"""
    system = platform.system()
    if system == 'Darwin':  # macOS
        # macOS 常见中文字体
        chinese_fonts = ['PingFang SC', 'STHeiti', 'Heiti SC', 'Arial Unicode MS', 'STSong']
    elif system == 'Windows':
        # Windows 常见中文字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']
    else:  # Linux
        # Linux 常见中文字体
        chinese_fonts = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC']
    
    # 尝试设置字体
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    for font in chinese_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            print(f'已设置中文字体: {font}')
            break
    else:
        # 如果找不到，尝试使用系统默认字体
        print('警告: 未找到合适的中文字体，可能无法正确显示中文')
        # 使用 Arial Unicode MS 作为备选（如果可用）
        if 'Arial Unicode MS' in available_fonts:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] + plt.rcParams['font.sans-serif']
    
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    font_manager._log.setLevel(40)
