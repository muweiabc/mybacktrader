#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# 简化版yfinance测试程序
#
###############################################################################

import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

def simple_test():
    """简单的yfinance测试"""
    print("开始测试yfinance...")
    
    try:
        # 下载AAPL数据
        print("正在下载AAPL 2024年数据...")
        aapl = yf.download("AAPL", start="2024-01-01", end="2024-12-31")
        
        if aapl.empty:
            print("错误：没有获取到数据")
            return False
            
        print(f"成功下载 {len(aapl)} 条数据")
        print(f"数据范围：{aapl.index[0].strftime('%Y-%m-%d')} 到 {aapl.index[-1].strftime('%Y-%m-%d')}")
        aapl.to_csv("aapl_2024.csv")

        # 显示前几行数据
        print("\n前5行数据：")
        print(aapl.head())
        
        # 绘制简单图表
        plt.figure(figsize=(12, 6))
        plt.plot(aapl.index, aapl['Close'], linewidth=2, color='blue')
        plt.title('AAPL 2024年收盘价走势', fontsize=14, fontweight='bold')
        plt.xlabel('日期')
        plt.ylabel('收盘价 ($)')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig('aapl_simple_test.png', dpi=300, bbox_inches='tight')
        print("图表已保存为 aapl_simple_test.png")
        
        # 显示图表
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        return False

if __name__ == "__main__":
    success = simple_test()
    if success:
        print("yfinance测试成功！")
    else:
        print("yfinance测试失败！") 