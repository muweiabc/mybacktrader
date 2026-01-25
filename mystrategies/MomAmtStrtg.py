from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import random
import argparse
import backtrader as bt
import pandas as pd
import numpy as np
from utils import setup_chinese_font
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from pathlib import Path
from runlogger import RunLogger, build_dashboard
try:
    from mystrategies.config_loader import load_config, get_repo_root, resolve_path
except ImportError:
    from config_loader import load_config, get_repo_root, resolve_path
# 初始化中文字体
setup_chinese_font()


class PandasDataWithAmt(bt.feeds.PandasData):
    """扩展PandasData以包含成交额(AMT)字段"""
    lines = ('amt',)
    params = (('amt', -1),)  # -1 表示使用列名映射


class MomAmtStrtg(bt.Strategy):
    """
    一个简化的动量与市值双因子选股策略
    - 动量：过去20个交易日的回报率
    - 市值：以成交量作为代理因子
    composite_score = mom_score + (amt_score * val_weight) #
    """
    params = (
        ('momentum_period', 20),  # 动量计算周期
        ('top_n_stocks', 10),      # 每期买入排名前N的股票数量
        ('rebalance_period', 'monthly'),  # 换仓周期: 'monthly' (每月) 或 'weekly' (每周)
        ('amt_weight', 100),  # 估值权重
        ('amt_norm_method', 'zscore'),  # 市值归一化: 'zscore' 或 'minmax'
    )

    def __init__(self):
        self.ranking = []
        self.order_refs = {} # 用于追踪订单
        self.last_rebalance_key = None  # 记录上次换仓的时间键（月份或周数）
        
        # 记录每日资产价值用于绘图
        self.portfolio_values = []
        self.portfolio_dates = []
        self.peak_values = []  # 记录峰值
        self.drawdowns = []    # 记录回撤
        
        # 1. 初始化指标：为每个数据源计算动量因子 (RateOfChange)
        self.momentum_indicators = {}
        for data in self.datas:
            # 计算过去 self.p.momentum_period 的回报率 (MOM)
            roc = bt.indicators.RateOfChange(data.close, 
                                           period=self.p.momentum_period
                                          )
            self.momentum_indicators[data._name] = roc

    # def prenext(self):
    #     self.next()

    def next(self):
        # for d in self.datas:
        #     if not len(d):  # 等价于：数据还没开始
        #         continue     # 新股跳过
        #     # 对已有数据的股票执行策略逻辑
        self.next1()

    def next1(self):
        # 记录每日资产价值
        
        current_date = self.datas[0].datetime.date(0)
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
            if peak > 0:
                drawdown = (peak - current_value) / peak * 100
            else:
                drawdown = 0.0
            self.drawdowns.append(drawdown)
        
        # 2. 检查是否是换仓日
        # 根据 rebalance_period 参数决定换仓周期
        if self.p.rebalance_period == 'weekly':
            # 按周换仓：使用 (年份, 周数) 作为键
            current_key = (current_date.year, current_date.isocalendar()[1])
            period_name = '周'
        else:  # 默认按月换仓
            # 按月换仓：使用 (年份, 月份) 作为键
            current_key = (current_date.year, current_date.month)
            period_name = '月'
        
        # 如果当前周期与上次换仓周期相同，跳过
        if self.last_rebalance_key == current_key:
            return
        
        # 这是新周期的第一个交易日，执行换仓
        self.last_rebalance_key = current_key
        self.log(f'--- 换仓日 {current_date} (按{period_name}换仓)：开始计算因子并轮动 ---')

        # 3. 计算因子并进行排序

        candidates = []
        amt_scores = []
        for data in self.datas:
            name = data._name
            mom_ind = self.momentum_indicators[name]

            # 确保数据已就绪
            if len(data) < self.p.momentum_period or mom_ind[0] is None or pd.isna(mom_ind[0]):
                continue
            
            # 检查价格是否有效（避免除零错误）
            if data.close[0] <= 0 or pd.isna(data.close[0]):
                continue

            # 因子值获取
            mom_score = mom_ind[0]             # 动量得分 (越高越好)
            # 使用成交额(AMT)的倒数作为估值因子（成交额越低越好，用倒数让它越高越好）
            amt_value = data.amt[0] if hasattr(data, 'amt') and data.amt[0] > 0 else 0
            raw_amt_score = 1.0 / amt_value if amt_value > 0 else 0.0    # 成交额倒数 (越高越好)

            candidates.append({
                'name': name,
                'data': data,
                'score': None,
                'mom': mom_score,
                'amt_score': raw_amt_score
            })
            amt_scores.append(raw_amt_score)

        if not candidates:
            return

        # 市值得分归一化（避免过小影响综合得分）
        norm_method = (self.p.amt_norm_method or "zscore").lower()
        if norm_method == "minmax":
            norm_scores = self._normalize_amt_minmax(amt_scores)
        else:
            norm_scores = self._normalize_amt_zscore(amt_scores)

        for stock, norm_amt_score in zip(candidates, norm_scores):
            stock['amt_score'] = norm_amt_score
            stock['score'] = stock['mom'] + (norm_amt_score * self.p.amt_weight)
            print(f"{stock['name']} 动量得分: {stock['mom']}, 市值得分: {norm_amt_score}, 综合得分: {stock['score']}")

        # 按综合得分从高到低排序
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # 选出表现最好的前 N 只股票
        top_stocks = candidates[:self.p.top_n_stocks]
        top_names = [s['name'] for s in top_stocks]
        
        self.log(f'本期入选股票 ({len(top_names)}只): {top_names}')

        # 4. 交易操作：卖出非入选股票，买入入选股票

        # 步骤 A: 平仓非入选股票 (卖出表现差的)
        for data in self.datas:
            if self.getposition(data).size > 0 and data._name not in top_names:
                self.log(f'平仓: {data._name}')
                self.close(data=data)

        # 步骤 B: 计算等权重买入量，并执行买入
        
        # 假设我们将所有资金平均分配给 top_n_stocks
        cash_for_each = self.broker.getcash() / len(top_stocks) if top_stocks else 0

        for stock in top_stocks:
            data = stock['data']
            
            # 如果该股票当前没有持仓，则买入
            if self.getposition(data).size == 0:
                price = data.close[0]
                # 计算买入股数 (必须是整数，且 backtrader默认资金管理)
                size = int((cash_for_each * 0.99) / price) # 留一点余量
                
                if size > 0:
                    self.log(f"建仓: {stock['name']}, 价格: {price:.2f}, 数量: {size}")
                    self.buy(data=data, size=size)
            # 简化处理：如果已经持仓，我们不调整仓位（不重新分配资金）
            # 实际策略需要更复杂的 rebalance_percent 逻辑
    
    # --- 日志和订单处理函数（可选，但推荐） ---
    def log(self, txt, dt=None):
        """
        输出日志

        Args:
            txt: 日志内容
            dt: 指定日期（默认使用当前数据日期）
        """
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def _normalize_amt_minmax(self, scores):
        if not scores:
            return []
        min_v = float(np.min(scores))
        max_v = float(np.max(scores))
        denom = max_v - min_v
        if denom <= 0:
            return [0.0 for _ in scores]
        return [(s - min_v) / denom for s in scores]

    def _normalize_amt_zscore(self, scores):
        if not scores:
            return []
        mean_v = float(np.mean(scores))
        std_v = float(np.std(scores, ddof=0))
        denom = std_v if std_v > 0 else 1.0
        return [(s - mean_v) / denom for s in scores]

    def notify_order(self, order):
        """
        订单状态通知回调

        Args:
            order: Backtrader 订单对象
        """
        # 订单状态通知
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

# --- 绘图函数 ---
def plot_equity_curve(strategy, initial_value, final_value, output_file=None, show=False):
    """
    绘制收益曲线图，标注最大回撤
    输出到equity_curve.png

    Args:
        strategy: 回测得到的策略实例（需要包含资产曲线相关字段）
        initial_value: 初始资金
        final_value: 最终资产
    """
    if not hasattr(strategy, 'portfolio_values') or len(strategy.portfolio_values) == 0:
        print('警告: 没有资产价值数据，无法绘制图表')
        return
    
    # 准备数据
    dates = pd.to_datetime(strategy.portfolio_dates)
    values = strategy.portfolio_values
    drawdowns = strategy.drawdowns
    
    # 计算收益率（相对于初始值）
    returns_pct = [(v - initial_value) / initial_value * 100 for v in values]
    
    # 找到最大回撤点
    max_dd_idx = drawdowns.index(max(drawdowns))
    max_dd_value = drawdowns[max_dd_idx]
    max_dd_date = dates[max_dd_idx]
    max_dd_portfolio_value = values[max_dd_idx]
    
    # 找到最大回撤前的峰值
    peak_before_dd = max(values[:max_dd_idx+1]) if max_dd_idx > 0 else values[0]
    peak_idx = values.index(peak_before_dd)
    peak_date = dates[peak_idx]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    
    # 第一个子图：资产价值曲线
    ax1.plot(dates, values, label='资产价值', linewidth=1.5, color='#2E86AB')
    ax1.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5, label='初始资金')
    ax1.axhline(y=final_value, color='green', linestyle='--', alpha=0.5, label='最终资产')
    
    # 标注最大回撤
    ax1.plot([peak_date, max_dd_date], [peak_before_dd, max_dd_portfolio_value], 
             'r--', linewidth=2, alpha=0.7, label=f'最大回撤: {max_dd_value:.2f}%')
    ax1.scatter([peak_date, max_dd_date], [peak_before_dd, max_dd_portfolio_value], 
                color='red', s=100, zorder=5)
    ax1.annotate(f'峰值\n{peak_before_dd:,.0f}', 
                 xy=(peak_date, peak_before_dd), 
                 xytext=(10, 20), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    ax1.annotate(f'最大回撤点\n{max_dd_portfolio_value:,.0f}\n回撤: {max_dd_value:.2f}%', 
                 xy=(max_dd_date, max_dd_portfolio_value), 
                 xytext=(10, -30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax1.set_ylabel('资产价值 (元)', fontsize=12)
    ax1.set_title('策略收益曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    
    # 第二个子图：回撤曲线
    ax2.fill_between(dates, 0, drawdowns, alpha=0.3, color='red', label='回撤')
    ax2.plot(dates, drawdowns, color='red', linewidth=1.5)
    ax2.axhline(y=max_dd_value, color='darkred', linestyle='--', alpha=0.7, 
                label=f'最大回撤: {max_dd_value:.2f}%')
    ax2.scatter([max_dd_date], [max_dd_value], color='darkred', s=100, zorder=5)
    
    ax2.set_ylabel('回撤 (%)', fontsize=12)
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_title('回撤曲线', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # 回撤图通常倒置显示
    
    # 格式化x轴日期 - 更详细的时间标注
    # 根据数据跨度选择合适的时间间隔
    date_span = (dates[-1] - dates[0]).days
    if date_span > 365 * 5:  # 超过5年，按年显示
        ax1.xaxis.set_major_locator(mdates.YearLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))  # 半年次刻度
    elif date_span > 365:  # 超过1年，按半年显示
        ax1.xaxis.set_major_locator(mdates.MonthLocator((1, 7)))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_minor_locator(mdates.MonthLocator())
    else:  # 少于1年，按月显示
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_minor_locator(mdates.WeekdayLocator())
    
    # 应用相同的格式到两个子图
    ax2.xaxis.set_major_locator(ax1.xaxis.get_major_locator())
    ax2.xaxis.set_major_formatter(ax1.xaxis.get_major_formatter())
    ax2.xaxis.set_minor_locator(ax1.xaxis.get_minor_locator())
    
    # 旋转标签以避免重叠
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    
    # 保存图片（可选）
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f'收益曲线图已保存到: {output_file}')
    
    # 显示图表或关闭
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

# --- 回测运行部分 ---

import os
import math


def load_and_validate_data(parquet_file, code_col, col_open, col_high, col_low, col_close, col_volume, col_amt='AMT'):
    """
    加载并验证 parquet 数据文件

    Args:
        parquet_file: parquet 文件路径（支持 ~ 展开）
        code_col: 股票代码列名
        col_open: 开盘价列名
        col_high: 最高价列名
        col_low: 最低价列名
        col_close: 收盘价列名
        col_volume: 成交量列名
        col_amt: 成交额列名（默认 AMT）
    """
    parquet_file = os.path.expanduser(parquet_file)
    print(f'正在读取 {parquet_file}...')
    
    raw_data = pd.read_parquet(parquet_file)
    print(f'成功读取数据，共 {len(raw_data)} 行')
    print(f'数据列名: {list(raw_data.columns)}')

    # 验证股票代码列
    if code_col not in raw_data.columns:
        raise ValueError(f'无法找到股票代码列 {code_col}')
    print(f'使用 {code_col} 作为股票代码列')
    
    # 验证OHLCV列
    ohlcv_cols = {
        'open': col_open,
        'high': col_high,
        'low': col_low,
        'close': col_close,
        'volume': col_volume
    }
    for field, col_name in ohlcv_cols.items():
        if col_name not in raw_data.columns:
            raise ValueError(f'无法找到 {field} 字段: {col_name}')
    
    # 验证AMT列
    if col_amt not in raw_data.columns:
        print(f'警告: 无法找到成交额列 {col_amt}，将使用 volume*close 作为替代')
        raw_data[col_amt] = raw_data[col_volume] * raw_data[col_close]
    
    print(f'OHLCV列: open={col_open}, high={col_high}, low={col_low}, close={col_close}, volume={col_volume}, amt={col_amt}')
    
    return raw_data


def add_data_feeds(cerebro, raw_data, code_col, col_open, col_high, col_low, col_close, col_volume,
                   num_stocks, min_data_length, fromdate, todate, col_amt='AMT'):
    """
    添加股票数据源到 cerebro

    Args:
        cerebro: Backtrader Cerebro 实例
        raw_data: 原始行情数据 DataFrame
        code_col: 股票代码列名
        col_open: 开盘价列名
        col_high: 最高价列名
        col_low: 最低价列名
        col_close: 收盘价列名
        col_volume: 成交量列名
        num_stocks: 抽样股票数量
        min_data_length: 最小数据长度（过滤不足样本）
        fromdate: 回测起始日期
        todate: 回测结束日期
        col_amt: 成交额列名（默认 AMT）
    """
    ohlc_cols = [col_open, col_high, col_low, col_close]
    grouped = raw_data.groupby(code_col)
    print(f'找到 {len(grouped)} 只股票')
    
    # 筛选在 fromdate 之前已上市的股票（有数据的）
    all_codes = list(grouped.groups.keys())
    eligible_codes = []
    for code in all_codes:
        df = grouped.get_group(code)
        if not np.isnan(df.iloc[0][col_close]):
            eligible_codes.append(code)
    
    print(f'在 {fromdate.date()} 之前已上市的股票: {len(eligible_codes)} 只')
    
    stock_codes_sample = random.sample(eligible_codes, min(num_stocks, len(eligible_codes)))
    
    added_count = 0
    for stock_code in stock_codes_sample:
        stock_data = grouped.get_group(stock_code).copy()
        
        # 确保数据按日期排序，移除股票代码列
        stock_data = stock_data.sort_index().drop(columns=[code_col])
        
        # 数据清理：过滤掉价格为 0、NaN 或无效的数据
        valid_mask = pd.Series(True, index=stock_data.index)
        for col in ohlc_cols:
            valid_mask &= (stock_data[col] > 0) & stock_data[col].notna() & (stock_data[col] != float('inf'))
        stock_data = stock_data[valid_mask]
        
        # 确保数据量足够
        if len(stock_data) < min_data_length:
            continue
        
        # 创建 PandasDataWithAmt feed (包含AMT字段)
        data_feed = PandasDataWithAmt(
            dataname=stock_data,
            fromdate=fromdate,
            todate=todate,
            open=col_open,
            high=col_high,
            low=col_low,
            close=col_close,
            volume=col_volume,
            amt=col_amt,
            name=str(stock_code)
        )
        
        cerebro.adddata(data_feed)
        added_count += 1
        print(f'已添加股票 {stock_code}，共 {len(stock_data)} 条记录')
    
    print(f'总共添加了 {added_count} 个数据源到 cerebro')
    return added_count


def add_analyzers(cerebro):
    """
    添加分析器到 cerebro

    Args:
        cerebro: Backtrader Cerebro 实例
    """
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
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn',
                        timeframe=bt.TimeFrame.Months)


def print_analysis_report(strat, initial_value, final_value):
    """
    打印策略分析报告

    Args:
        strat: 回测得到的策略实例
        initial_value: 初始资金
        final_value: 最终资产
    """
    print('\n' + '=' * 60)
    print('策略分析报告')
    print('=' * 60)
    
    total_return_pct = (final_value - initial_value) / initial_value * 100
    
    # 1. 收益率分析
    returns = strat.analyzers.returns.get_analysis()
    print('\n【收益率分析】')
    print(f'  初始资产: {initial_value:,.2f}')
    print(f'  最终资产: {final_value:,.2f}')
    print(f'  总收益率: {total_return_pct:.2f}%')
    print(f'  总收益倍数: {final_value / initial_value:.2f}x')
    rtot_log = returns.get("rtot", 0)
    if rtot_log != float('-inf'):
        rtot_pct = (math.exp(rtot_log) - 1) * 100
        print(f'  对数收益率 (rtot): {rtot_log:.4f} = {rtot_pct:.2f}%')
    print(f'  年化收益率: {returns.get("rnorm", 0) * 100:.2f}%')
    print(f'  年化收益率 (100): {returns.get("rnorm100", 0):.2f}%')
    
    # 2. 夏普率
    sharpe = strat.analyzers.sharpe.get_analysis()
    print('\n【夏普率分析】')
    if 'sharperatio' in sharpe:
        print(f'  夏普率: {sharpe["sharperatio"]:.4f}')
    else:
        print('  夏普率: 数据不足，无法计算')
    
    # 3. 回撤分析
    drawdown = strat.analyzers.drawdown.get_analysis()
    print('\n【回撤分析】')
    print(f'  最大回撤: {drawdown.get("max", {}).get("drawdown", 0):.2f}%')
    print(f'  最大回撤金额: {drawdown.get("max", {}).get("moneydown", 0):.2f}')
    print(f'  最大回撤长度: {drawdown.get("max", {}).get("len", 0)} 天')
    
    # 4. 交易分析
    trades = strat.analyzers.trades.get_analysis()
    print('\n【交易分析】')
    print(f'  总交易次数: {trades.get("total", {}).get("total", 0)}')
    print(f'  盈利交易: {trades.get("won", {}).get("total", 0)}')
    print(f'  亏损交易: {trades.get("lost", {}).get("total", 0)}')
    if trades.get("won", {}).get("total", 0) > 0:
        print(f'  胜率: {trades.get("won", {}).get("total", 0) / trades.get("total", {}).get("total", 1) * 100:.2f}%')
    print(f'  总盈利: {trades.get("won", {}).get("pnl", {}).get("net", 0):.2f}')
    print(f'  总亏损: {trades.get("lost", {}).get("pnl", {}).get("net", 0):.2f}')
    print(f'  平均盈利: {trades.get("won", {}).get("pnl", {}).get("average", 0):.2f}')
    print(f'  平均亏损: {trades.get("lost", {}).get("pnl", {}).get("average", 0):.2f}')
    
    # 5. 年化收益率
    annual_return = strat.analyzers.annualreturn.get_analysis()
    print('\n【年化收益率】')
    for year, ret in sorted(annual_return.items()):
        print(f'  {year}: {ret * 100:.2f}%')
    
    # 6. 系统质量指标 (SQN)
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
    
    # 7. 卡玛比率
    calmar = strat.analyzers.calmar.get_analysis()
    print('\n【卡玛比率】')
    if 'calmar' in calmar:
        print(f'  卡玛比率: {calmar["calmar"]:.4f}')
    
    print('\n' + '=' * 60)
    print('分析报告结束')
    print('=' * 60)


def main(config: dict):
    """
    回测主函数

    Args:
        config: 配置字典（来自 JSON/YAML）
    """
    config = dict(config)
    parquet_file = resolve_path(config["parquet_file"], base_dir=get_repo_root())
    config["parquet_file"] = parquet_file
    num_stocks = config.get("num_stocks", 100)
    min_data_length = config.get("min_data_length", 210)
    fromdate = config.get("fromdate", datetime(2020, 1, 1))
    todate = config.get("todate", datetime(2025, 12, 31))
    code_col = config.get("code_col", "CODE")
    col_open = config.get("col_open", "OPEN")
    col_high = config.get("col_high", "HIGH")
    col_low = config.get("col_low", "LOW")
    col_close = config.get("col_close", "CLOSE")
    col_volume = config.get("col_volume", "VOLUME")
    col_amt = config.get("col_amt", "AMT")
    top_n_stocks = config.get("top_n_stocks", 3)
    momentum_period = config.get("momentum_period", 20)
    amt_weight = config.get("amt_weight", 100)
    amt_norm_method = config.get("amt_norm_method", "zscore")
    rebalance_period = config.get("rebalance_period", "monthly")
    initial_cash = config.get("initial_cash", 100000.0)
    commission = config.get("commission", 0.001)
    # ===== 运行配置与日志 =====
    config = {
        "strategy": "MomAmtStrtg",
        "parquet_file": parquet_file,
        "num_stocks": num_stocks,
        "min_data_length": min_data_length,
        "fromdate": fromdate,
        "todate": todate,
        "code_col": code_col,
        "col_open": col_open,
        "col_high": col_high,
        "col_low": col_low,
        "col_close": col_close,
        "col_volume": col_volume,
        "col_amt": col_amt,
        "top_n_stocks": top_n_stocks,
        "momentum_period": momentum_period,
        "amt_weight": amt_weight,
        "amt_norm_method": amt_norm_method,
        "rebalance_period": rebalance_period,
        "initial_cash": initial_cash,
        "commission": commission,
    }

    logger = RunLogger(root_dir="experiments", index_file="experiments/index.csv")
    run_id, run_dir = logger.start(config)

    # 初始化 cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission)
    
    # 加载数据
    try:
        raw_data = load_and_validate_data(
            parquet_file, code_col, col_open, col_high, col_low, col_close, col_volume, col_amt
        )
        
        # 添加数据源
        add_data_feeds(
            cerebro, raw_data, code_col, col_open, col_high, col_low, col_close, col_volume,
            num_stocks, min_data_length, fromdate, todate, col_amt
        )
    except FileNotFoundError:
        print(f'错误: 找不到文件 {parquet_file}')
        raise
    except Exception as e:
        print(f'读取数据时出错: {e}')
        import traceback
        traceback.print_exc()
        raise

    # 添加策略
    cerebro.addstrategy(
        MomAmtStrtg,
        top_n_stocks=top_n_stocks,
        momentum_period=momentum_period,
        amt_weight=amt_weight,
        amt_norm_method=amt_norm_method,
        rebalance_period=rebalance_period
    )

    # 添加分析器
    add_analyzers(cerebro)

    # 运行回测
    print('=' * 60)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('=' * 60)
    
    results = cerebro.run()
    strat = results[0]
    
    print('=' * 60)
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('=' * 60)
    
    # 打印分析报告
    initial_value = cerebro.broker.startingcash
    final_value = cerebro.broker.getvalue()
    print_analysis_report(strat, initial_value, final_value)

    # ===== 汇总指标 =====
    returns = strat.analyzers.returns.get_analysis()
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    calmar = strat.analyzers.calmar.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    total_return_pct = (final_value - initial_value) / initial_value * 100
    metrics = {
        "run_id": run_id,
        "strategy": "MomAmtStrtg",
        "initial_value": initial_value,
        "final_value": final_value,
        "total_return_pct": total_return_pct,
        "sharpe": sharpe.get("sharperatio"),
        "max_drawdown": drawdown.get("max", {}).get("drawdown", 0),
        "calmar": calmar.get("calmar"),
        "total_trades": trades.get("total", {}).get("total", 0),
        "rtot": returns.get("rtot", 0),
        "rnorm": returns.get("rnorm", 0),
    }

    # ===== 保存 artifacts =====
    # 曲线数据
    curve_df = pd.DataFrame({
        "date": pd.to_datetime(strat.portfolio_dates),
        "value": strat.portfolio_values,
        "drawdown": strat.drawdowns,
    })
    curve_path = logger.save_df("equity_curve.csv", curve_df)

    # 曲线图
    print('\n正在生成收益曲线图...')
    fig = plot_equity_curve(strat, initial_value, final_value, output_file=None, show=False)
    curve_img_path = logger.save_fig("equity_curve.png", fig)

    # 保存指标
    metrics_path = logger.save_json("metrics.json", metrics)

    # 生成单次运行HTML
    run_index = Path(run_dir) / "index.html"
    run_template = Path("dashboard/experiment_run.html")
    if run_template.exists():
        run_html = run_template.read_text(encoding="utf-8").format(
            run_id=run_id,
            strategy="MomAmtStrtg",
            run_time=logger.run_time or ""
        )
    else:
        run_html = f"<h1>Run {run_id}</h1>"
    run_index.write_text(run_html, encoding="utf-8")

    # ===== 写索引 =====
    logger.append_index_row({
        "run_id": run_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "run_time": logger.run_time,
        "strategy": "MomAmtStrtg",
        "total_return_pct": total_return_pct,
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "run_dir": str(run_dir),
        "equity_curve": curve_img_path,
        "metrics_path": metrics_path,
        "curve_csv": curve_path,
    })

    # ===== 生成总览dashboard =====
    build_dashboard(index_file="experiments/index.csv", output_file="experiments/index.html")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MomAmtStrtg backtest runner")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON/YAML config (default: configs/mom_amt_strategy.json)",
    )
    args = parser.parse_args()
    default_config = get_repo_root() / "configs" / "mom_amt_strategy.json"
    config_path = args.config or str(default_config)
    config = load_config(config_path)
    main(config)

