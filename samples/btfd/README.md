# BTFD 策略使用指南

## 问题解决

原来的 `FileNotFoundError: [Errno 2] No such file or directory: '^GSPC'` 错误已经修复。这个错误是因为程序试图从 Yahoo Finance 下载数据时失败。

## 解决方案

### 1. 使用本地数据文件（推荐）

项目中的 `datas/` 目录包含多个可用的数据文件：

```bash
# 使用本地数据文件运行策略
python btfd.py --offline --data yhoo-1996-2014.txt --plot

# 其他可用的数据文件
python btfd.py --offline --data nvda-1999-2014.txt --plot
python btfd.py --offline --data orcl-1995-2014.txt --plot
```

### 2. 使用 Yahoo Finance 数据（需要网络连接）

```bash
# 使用 Yahoo Finance 数据（需要稳定的网络连接）
python btfd.py --data AAPL --fromdate 2020-01-01 --todate 2021-12-31 --plot
```

## 可用的数据文件

在 `datas/` 目录中有以下数据文件：

- `yhoo-1996-2014.txt` - Yahoo 股票数据 (1996-2014)
- `yhoo-1996-2015.txt` - Yahoo 股票数据 (1996-2015)
- `nvda-1999-2014.txt` - NVIDIA 股票数据 (1999-2014)
- `orcl-1995-2014.txt` - Oracle 股票数据 (1995-2014)
- `orcl-2003-2005.txt` - Oracle 股票数据 (2003-2005)
- 以及其他多个数据文件

## 参数说明

### 基本参数
- `--offline`: 使用本地数据文件
- `--data`: 指定数据文件或股票代码
- `--fromdate`: 开始日期
- `--todate`: 结束日期
- `--plot`: 显示图表

### 策略参数
- `--strat approach="highlow"`: 计算下跌幅度的方法
  - `closeclose`: 收盘价相对于前一日收盘价的跌幅
  - `openclose`: 收盘价相对于当日开盘价的跌幅
  - `highclose`: 收盘价相对于当日最高价的跌幅
  - `highlow`: 当日最低价相对于最高价的跌幅（默认）

### 其他参数
- `--broker cash=100000.0`: 初始资金
- `--comminfo leverage=2.0`: 杠杆倍数

## 使用示例

### 示例 1：基本使用
```bash
python btfd.py --offline --data yhoo-1996-2014.txt --plot
```

### 示例 2：自定义参数
```bash
python btfd.py --offline --data yhoo-1996-2014.txt \
    --strat 'fall=-0.02,hold=5,approach="closeclose"' \
    --fromdate 2000-01-01 --todate 2005-12-31 \
    --plot 'volume=False'
```

### 示例 3：测试脚本
```bash
python test_btfd.py
```

## 策略说明

BTFD (Buy The Fucking Dip) 策略是一个简单的反转策略：

1. **入场条件**: 当价格下跌超过设定阈值时买入
2. **出场条件**: 持有固定时间后卖出
3. **仓位管理**: 使用目标百分比仓位

### 策略特点
- 适合波动性较大的市场
- 适合有明确趋势的市场
- 适合做短线的市场
- 不适合单边下跌市场
- 不适合低波动性市场

## 错误处理

如果遇到数据文件不存在的问题，程序会：
1. 显示错误信息
2. 列出可用的数据文件
3. 提供使用建议

## 注意事项

1. 确保在正确的目录下运行脚本
2. 本地数据文件路径是相对于 `datas/` 目录的
3. Yahoo Finance 数据需要稳定的网络连接
4. 建议先使用本地数据文件测试策略 