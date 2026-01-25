"""
市值分组多空策略 (Market Cap Long-Short Strategy)

策略逻辑:
- 使用成交额(AMT)作为市值的代理变量
- 每月初将所有股票按成交额分成10组
- 做多第10组(最小市值组)，做空第1组(最大市值组)
- 回测过去5年的表现
- 适用于震荡市或者熊市，不适用牛市会把大盘上涨丢掉

数据来源:
- 市值代理: ~/etf_daily.parquet 或 ~/wind_daily_2010_adj.parquet 中的 AMT 列
- 日线数据: ~/wind_daily_2010_adj.parquet
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import pandas as pd
import numpy as np
from utils import setup_chinese_font
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

# 初始化中文字体
setup_chinese_font()


class MarketCapLongShortStrategy(bt.Strategy):
    """
    市值分组多空策略
    - 每月初根据前一个月的平均成交额将股票分成10组
    - 做多第10组(最小市值/成交额)，做空第1组(最大市值/成交额)
    - 等权重配置
    """
    params = (
        ('num_groups', 10),           # 分组数量
        ('long_group', 10),           # 做多的组 (最小市值)
        ('short_group', 1),           # 做空的组 (最大市值)
        ('lookback_days', 20),        # 计算平均成交额的回看天数
        ('rebalance_period', 'monthly'),  # 换仓周期
        ('long_weight', 0.5),         # 多头仓位占比
        ('short_weight', 0.5),        # 空头仓位占比
    )

    def __init__(self):
        self.order_refs = {}
        self.last_rebalance_key = None
        
        # 记录每日资产价值用于绘图
        self.portfolio_values = []
        self.portfolio_dates = []
        self.peak_values = []
        self.drawdowns = []
        
        # 存储每只股票的成交额数据
        self.amt_data = {}
        for data in self.datas:
            self.amt_data[data._name] = []

    def prenext(self):
        """在所有数据对齐前也执行策略"""
        self.next()

    def next(self):
        # 获取当前日期 - 使用第一个有数据的数据源
        current_date = None
        for data in self.datas:
            if len(data) > 0:
                current_date = data.datetime.date(0)
                break
        
        if current_date is None:
            return
            
        current_value = self.broker.getvalue()
        
        self.portfolio_values.append(current_value)
        self.portfolio_dates.append(current_date)
        
        # 计算回撤
        if len(self.peak_values) == 0:
            self.peak_values.append(current_value)
            self.drawdowns.append(0.0)
        else:
            peak = max(self.peak_values[-1], current_value)
            self.peak_values.append(peak)
            drawdown = (peak - current_value) / peak * 100 if peak > 0 else 0.0
            self.drawdowns.append(drawdown)
        
        # 更新成交额数据
        for data in self.datas:
            if len(data) > 0:
                # 获取成交额 (使用volume*close作为成交额代理)
                try:
                    amt = data.volume[0] * data.close[0]
                except:
                    amt = 0
                self.amt_data[data._name].append(amt)
        
        # 检查是否是换仓日
        if self.p.rebalance_period == 'weekly':
            current_key = (current_date.year, current_date.isocalendar()[1])
            period_name = '周'
        else:
            current_key = (current_date.year, current_date.month)
            period_name = '月'
        
        if self.last_rebalance_key == current_key:
            return
        
        self.last_rebalance_key = current_key
        self.log(f'--- 换仓日 {current_date} (按{period_name}换仓)：进行市值分组 ---')
        
        # 计算每只股票的平均成交额
        candidates = []
        for data in self.datas:
            name = data._name
            
            # 确保有足够的数据
            if len(self.amt_data[name]) < self.p.lookback_days:
                continue
            
            # 检查价格是否有效
            if data.close[0] <= 0 or pd.isna(data.close[0]):
                continue
            
            # 计算过去lookback_days的平均成交额
            recent_amt = self.amt_data[name][-self.p.lookback_days:]
            avg_amt = np.mean([a for a in recent_amt if a > 0])
            
            if avg_amt > 0 and not np.isnan(avg_amt):
                candidates.append({
                    'name': name,
                    'data': data,
                    'avg_amt': avg_amt
                })
        
        if len(candidates) < self.p.num_groups:
            self.log(f'股票数量不足 ({len(candidates)})，需要至少 {self.p.num_groups} 只')
            return
        
        # 按平均成交额从高到低排序 (第1组是最大市值，第10组是最小市值)
        candidates.sort(key=lambda x: x['avg_amt'], reverse=True)
        
        # 分组
        group_size = len(candidates) // self.p.num_groups
        groups = {}
        for i in range(self.p.num_groups):
            start_idx = i * group_size
            if i == self.p.num_groups - 1:
                # 最后一组包含剩余所有股票
                groups[i + 1] = candidates[start_idx:]
            else:
                groups[i + 1] = candidates[start_idx:start_idx + group_size]
        
        # 获取做多和做空的股票列表
        long_stocks = groups.get(self.p.long_group, [])
        short_stocks = groups.get(self.p.short_group, [])
        
        self.log(f'第{self.p.long_group}组(做多，小市值): {len(long_stocks)}只股票')
        self.log(f'第{self.p.short_group}组(做空，大市值): {len(short_stocks)}只股票')
        
        # 获取所有需要持仓的股票名称
        long_names = [s['name'] for s in long_stocks]
        short_names = [s['name'] for s in short_stocks]
        all_target_names = set(long_names + short_names)
        
        # 平仓不在目标列表中的股票
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size != 0 and data._name not in all_target_names:
                self.log(f'平仓: {data._name}, 当前持仓: {pos.size}')
                self.close(data=data)
        
        # 计算每只股票的目标仓位
        total_value = self.broker.getvalue()
        
        # 做多仓位
        if long_stocks:
            long_value_per_stock = (total_value * self.p.long_weight) / len(long_stocks)
            for stock in long_stocks:
                data = stock['data']
                price = data.close[0]
                if price > 0:
                    target_size = int(long_value_per_stock * 0.95 / price)
                    current_size = self.getposition(data).size
                    
                    if target_size > 0 and current_size != target_size:
                        # 调整仓位
                        diff = target_size - current_size
                        if diff > 0:
                            self.log(f'做多建仓: {stock["name"]}, 价格: {price:.2f}, 目标数量: {target_size}')
                            self.buy(data=data, size=diff)
                        elif diff < 0:
                            self.log(f'减仓: {stock["name"]}, 数量: {-diff}')
                            self.sell(data=data, size=-diff)
        
        # 做空仓位 (卖空)
        if short_stocks:
            short_value_per_stock = (total_value * self.p.short_weight) / len(short_stocks)
            for stock in short_stocks:
                data = stock['data']
                price = data.close[0]
                if price > 0:
                    target_size = -int(short_value_per_stock * 0.95 / price)  # 负数表示空头
                    current_size = self.getposition(data).size
                    
                    if target_size < 0 and current_size != target_size:
                        diff = target_size - current_size
                        if diff < 0:
                            self.log(f'做空建仓: {stock["name"]}, 价格: {price:.2f}, 目标数量: {target_size}')
                            self.sell(data=data, size=-diff)
                        elif diff > 0:
                            self.log(f'空头平仓: {stock["name"]}, 数量: {diff}')
                            self.buy(data=data, size=diff)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Size: %d' % 
                         (order.executed.price, order.executed.size))
            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Size: %d' % 
                         (order.executed.price, order.executed.size))
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Failed/Rejected')


def plot_equity_curve(strategy, initial_value, final_value, output_file='graphs/marketcap_longshort_strategy_curve.png'):
    """绘制收益曲线图"""
    if not hasattr(strategy, 'portfolio_values') or len(strategy.portfolio_values) == 0:
        print('警告: 没有资产价值数据，无法绘制图表')
        return
    
    dates = pd.to_datetime(strategy.portfolio_dates)
    values = strategy.portfolio_values
    drawdowns = strategy.drawdowns
    
    # 找到最大回撤点
    max_dd_idx = drawdowns.index(max(drawdowns))
    max_dd_value = drawdowns[max_dd_idx]
    max_dd_date = dates[max_dd_idx]
    max_dd_portfolio_value = values[max_dd_idx]
    
    peak_before_dd = max(values[:max_dd_idx+1]) if max_dd_idx > 0 else values[0]
    peak_idx = values.index(peak_before_dd)
    peak_date = dates[peak_idx]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 资产价值曲线
    ax1.plot(dates, values, label='资产价值', linewidth=1.5, color='#2E86AB')
    ax1.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5, label='初始资金')
    ax1.axhline(y=final_value, color='green', linestyle='--', alpha=0.5, label='最终资产')
    
    # 标注最大回撤
    ax1.plot([peak_date, max_dd_date], [peak_before_dd, max_dd_portfolio_value], 
             'r--', linewidth=2, alpha=0.7, label=f'最大回撤: {max_dd_value:.2f}%')
    ax1.scatter([peak_date, max_dd_date], [peak_before_dd, max_dd_portfolio_value], 
                color='red', s=100, zorder=5)
    
    ax1.set_ylabel('资产价值 (元)', fontsize=12)
    ax1.set_title('市值分组多空策略 - 收益曲线 (做多小市值 vs 做空大市值)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    
    # 回撤曲线
    ax2.fill_between(dates, 0, drawdowns, alpha=0.3, color='red', label='回撤')
    ax2.plot(dates, drawdowns, color='red', linewidth=1.5)
    ax2.axhline(y=max_dd_value, color='darkred', linestyle='--', alpha=0.7, 
                label=f'最大回撤: {max_dd_value:.2f}%')
    
    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_title('回撤曲线', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # 格式化x轴
    date_span = (dates[-1] - dates[0]).days
    if date_span > 365 * 5:
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    elif date_span > 365:
        ax1.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'收益曲线图已保存到: {output_file}')
    plt.close()  # 关闭图表，不显示


def main(top_amt_stocks=300, num_groups=10, long_group=10, short_group=1):
    """
    运行市值分组多空策略回测
    
    参数:
    - top_amt_stocks: top_amt_stocks
    - num_groups: 分组数量（默认10组）
    - long_group: 做多的组号（默认10，最小市值组）
    - short_group: 做空的组号（默认1，最大市值组）
    """
    import os
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1000000.0)  # 初始资金100万
    cerebro.broker.setcommission(commission=0.001)
    
    # 允许做空
    cerebro.broker.set_shortcash(True)
    
    # 读取数据
    parquet_file = os.path.expanduser('~/wind_daily_2010_adj.parquet')
    print(f'正在读取 {parquet_file}...')
    
    try:
        raw_data = pd.read_parquet(parquet_file)
        print(f'成功读取数据，共 {len(raw_data)} 行')
        
        # 字段映射
        date_col = None
        if isinstance(raw_data.index, pd.DatetimeIndex):
            date_col = None
        else:
            for col in ['日期', 'date', 'trade_date']:
                if col in raw_data.columns:
                    date_col = col
                    break
        
        if date_col:
            raw_data[date_col] = pd.to_datetime(raw_data[date_col])
            raw_data.set_index(date_col, inplace=True)
        
        code_col = 'CODE'
        
        # 过滤最近5年的数据 (2020-2025)
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2025, 12, 31)
        
        # 筛选在回测开始日期前就有数据的股票
        # 计算每只股票的第一条数据日期
        first_date_by_stock = raw_data.groupby(code_col).apply(lambda x: x.index.min())
        
        # 只选择在2020年1月1日前就有数据的股票
        valid_stocks = first_date_by_stock[first_date_by_stock < start_date].index.tolist()
        print(f'在2020年前有数据的股票数量: {len(valid_stocks)}')
        
        # 计算每只股票在回测期间的平均成交额，用于筛选
        recent_data = raw_data[(raw_data.index >= start_date) & (raw_data.index <= end_date)]
        recent_data = recent_data[recent_data[code_col].isin(valid_stocks)]
        
        # 计算每只股票的平均成交额
        avg_amt_by_stock = recent_data.groupby(code_col)['AMT'].mean().sort_values(ascending=False)
        
        # 选择平均成交额最高的股票（确保有足够流动性）
        selected_stocks = avg_amt_by_stock.head(top_amt_stocks).index.tolist()
        print(f'选择了 {len(selected_stocks)} 只股票进行回测')
        
        # 为每个股票创建数据源 - 使用groupby提高效率
        added_count = 0
        # 先筛选出selected_stocks的数据，再groupby，避免多次全表扫描
        selected_raw_data = raw_data[raw_data[code_col].isin(selected_stocks)]
        ohlc_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE']
        
        for stock_code, stock_data in selected_raw_data.groupby(code_col):
            stock_data = stock_data.copy()
            stock_data.sort_index(inplace=True)
            stock_data = stock_data.drop(columns=[code_col])
            
            if len(stock_data) == 0:
                continue
            
            # 数据清理 - 一次性构建mask
            valid_mask = pd.Series(True, index=stock_data.index)
            for col in ohlc_cols:
                if col in stock_data.columns:
                    valid_mask &= (stock_data[col] > 0) & (stock_data[col].notna()) & (stock_data[col] != float('inf'))
            stock_data = stock_data[valid_mask]
            
            # 检查在回测期间是否有足够数据
            backtest_data = stock_data[(stock_data.index >= start_date) & (stock_data.index <= end_date)]
            if len(backtest_data) < 500:  # 至少需要500个交易日的数据（约2年）
                continue
            
            data_feed = bt.feeds.PandasData(
                dataname=stock_data,
                fromdate=start_date,
                todate=end_date,
                open='OPEN',
                high='HIGH',
                low='LOW',
                close='CLOSE',
                volume='VOLUME',
                name=str(stock_code)
            )
            
            cerebro.adddata(data_feed)
            added_count += 1
        
        print(f'总共添加了 {added_count} 个数据源到 cerebro')
        
    except Exception as e:
        print(f'读取数据时出错: {e}')
        import traceback
        traceback.print_exc()
        raise

    # 添加策略
    cerebro.addstrategy(
        MarketCapLongShortStrategy, 
        num_groups=num_groups,
        long_group=long_group,
        short_group=short_group
    )
    
    # 允许策略在所有数据对齐前就开始运行
    cerebro.run_once = False

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns',
                       timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annualreturn')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.Calmar, _name='calmar',
                        timeframe=bt.TimeFrame.Days)

    print('=' * 60)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('=' * 60)
    
    results = cerebro.run()
    strat = results[0]
    
    print('=' * 60)
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('=' * 60)
    
    # 打印分析结果
    print('\n' + '=' * 60)
    print('市值分组多空策略 - 分析报告')
    print(f'策略说明: 做多第{long_group}组(小市值) vs 做空第{short_group}组(大市值)')
    print('=' * 60)
    
    initial_value = cerebro.broker.startingcash
    final_value = cerebro.broker.getvalue()
    total_return_pct = (final_value - initial_value) / initial_value * 100
    
    # 收益率分析
    returns = strat.analyzers.returns.get_analysis()
    print('\n【收益率分析】')
    print(f'  初始资产: {initial_value:,.2f}')
    print(f'  最终资产: {final_value:,.2f}')
    print(f'  总收益率: {total_return_pct:.2f}%')
    print(f'  总收益倍数: {final_value / initial_value:.2f}x')
    import math
    rtot_log = returns.get("rtot", 0)
    if rtot_log != float('-inf') and rtot_log != 0:
        rtot_pct = (math.exp(rtot_log) - 1) * 100
        print(f'  对数收益率 (rtot): {rtot_log:.4f} = {rtot_pct:.2f}%')
    print(f'  年化收益率: {returns.get("rnorm", 0) * 100:.2f}%')
    
    # 夏普率
    sharpe = strat.analyzers.sharpe.get_analysis()
    print('\n【夏普率分析】')
    if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
        print(f'  夏普率: {sharpe["sharperatio"]:.4f}')
    else:
        print('  夏普率: 数据不足，无法计算')
    
    # 回撤分析
    drawdown = strat.analyzers.drawdown.get_analysis()
    print('\n【回撤分析】')
    max_dd = drawdown.get("max", {}).get("drawdown", 0)
    print(f'  最大回撤: {max_dd:.2f}%')
    print(f'  最大回撤金额: {drawdown.get("max", {}).get("moneydown", 0):,.2f}')
    print(f'  最大回撤长度: {drawdown.get("max", {}).get("len", 0)} 天')
    
    # 交易分析
    trades = strat.analyzers.trades.get_analysis()
    print('\n【交易分析】')
    print(f'  总交易次数: {trades.get("total", {}).get("total", 0)}')
    print(f'  盈利交易: {trades.get("won", {}).get("total", 0)}')
    print(f'  亏损交易: {trades.get("lost", {}).get("total", 0)}')
    total_trades = trades.get("total", {}).get("total", 1)
    won_trades = trades.get("won", {}).get("total", 0)
    if total_trades > 0:
        print(f'  胜率: {won_trades / total_trades * 100:.2f}%')
    
    # 年化收益率
    annual_return = strat.analyzers.annualreturn.get_analysis()
    print('\n【年化收益率】')
    for year, ret in sorted(annual_return.items()):
        print(f'  {year}: {ret * 100:.2f}%')
    
    # SQN
    sqn = strat.analyzers.sqn.get_analysis()
    print('\n【系统质量指标 (SQN)】')
    if 'sqn' in sqn:
        sqn_value = sqn['sqn']
        print(f'  SQN值: {sqn_value:.4f}')
        if sqn_value > 2.5:
            print('  评价: 优秀 (>2.5)')
        elif sqn_value > 1.6:
            print('  评价: 良好 (1.6-2.5)')
        elif sqn_value > 1.0:
            print('  评价: 一般 (1.0-1.6)')
        else:
            print('  评价: 较差 (<1.0)')
    
    # 卡玛比率
    calmar = strat.analyzers.calmar.get_analysis()
    print('\n【卡玛比率】')
    if 'calmar' in calmar and calmar['calmar'] is not None:
        print(f'  卡玛比率: {calmar["calmar"]:.4f}')
    
    print('\n' + '=' * 60)
    print('分析报告结束')
    print('=' * 60)

    # 绘制收益曲线
    print('\n正在生成收益曲线图...')
    plot_equity_curve(strat, initial_value, final_value, 'market_cap_strategy_curve.png')
    
    return strat, cerebro


if __name__ == '__main__':
    # 运行回测
    # 参数说明:
    # - num_stocks: 选择200只股票进行回测
    # - num_groups: 分成10组
    # - long_group: 做多第10组(最小市值)
    # - short_group: 做空第1组(最大市值)
    main(top_amt_stocks=1000, num_groups=10, long_group=10, short_group=1)

