#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
"""
测试BTFD策略的简单脚本
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_btfd_local():
    """测试使用本地数据文件运行BTFD策略"""
    print("测试BTFD策略（使用本地数据文件）...")
    
    # 导入btfd模块
    from btfd import runstrat
    
    # 设置命令行参数
    sys.argv = [
        'btfd.py',
        '--offline',
        '--data', 'yhoo-1996-2014.txt',
        '--fromdate', '2000-01-01',
        '--todate', '2005-12-31',
        '--plot', 'volume=False'
    ]
    
    try:
        runstrat()
        print("✅ BTFD策略测试成功！")
    except Exception as e:
        print(f"❌ BTFD策略测试失败: {e}")
        return False
    
    return True

def test_btfd_yahoo():
    """测试使用Yahoo Finance数据运行BTFD策略"""
    print("测试BTFD策略（使用Yahoo Finance数据）...")
    
    # 导入btfd模块
    from btfd import runstrat
    
    # 设置命令行参数
    sys.argv = [
        'btfd.py',
        '--data', 'AAPL',
        '--fromdate', '2020-01-01',
        '--todate', '2021-12-31',
        '--plot', 'volume=False'
    ]
    
    try:
        runstrat()
        print("✅ BTFD策略（Yahoo Finance）测试成功！")
    except Exception as e:
        print(f"❌ BTFD策略（Yahoo Finance）测试失败: {e}")
        print("这是正常的，因为Yahoo Finance API可能不稳定")
        return False
    
    return True

if __name__ == '__main__':
    print("开始BTFD策略测试...")
    print("=" * 50)
    
    # 测试本地数据文件
    local_success = test_btfd_local()
    
    print("\n" + "=" * 50)
    
    # 测试Yahoo Finance数据（可选）
    yahoo_success = test_btfd_yahoo()
    
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"本地数据文件测试: {'✅ 成功' if local_success else '❌ 失败'}")
    print(f"Yahoo Finance测试: {'✅ 成功' if yahoo_success else '❌ 失败'}")
    
    if local_success:
        print("\n🎉 BTFD策略可以正常运行！")
        print("建议使用以下命令运行策略：")
        print("  python btfd.py --offline --data yhoo-1996-2014.txt --plot")
    else:
        print("\n⚠️  需要进一步检查问题") 