"""
参数扫描脚本：测试不同 top_n_stocks 值对策略表现的影响
绘制收益率、夏普率等指标关于 top_n_stocks 的图像
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import platform
import os
import math

# 导入策略类和绘图函数
from SimpleFactorStrategy import SimpleFactorStrategy, plot_equity_curve
from utils import setup_chinese_font
# 设置中文字体
setup_chinese_font()


def load_data(file_path='~/etf_daily.parquet'):
    """加载并预处理数据"""
    parquet_file = os.path.expanduser(file_path)
    print(f'正在读取 {parquet_file}...')
    
    raw_data = pd.read_parquet(parquet_file)
    print(f'成功读取数据，共 {len(raw_data)} 行')
    
    # 字段名映射
    date_cols = ['日期', 'date', 'trade_date', '交易日期', 'datetime']
    code_cols = ['股票代码', 'code', 'wind_code', 'ts_code', 'symbol']
    field_mapping = {
        'open': ['开盘', 'open', 'OPEN'],
        'high': ['最高', 'high', 'HIGH'],
        'low': ['最低', 'low', 'LOW'],
        'close': ['收盘', 'close', 'CLOSE'],
        'volume': ['成交量', 'volume', 'vol', 'VOL', 'VOLUME']
    }
    
    # 自动检测日期列
    date_col = None
    for col in date_cols:
        if col in raw_data.columns:
            date_col = col
            break
    if date_col is None and isinstance(raw_data.index, pd.DatetimeIndex):
        date_col = None
    elif date_col is None:
        for col in raw_data.columns:
            if 'date' in col.lower() or '日期' in col.lower():
                date_col = col
                break
    
    if date_col:
        raw_data[date_col] = pd.to_datetime(raw_data[date_col])
        raw_data.set_index(date_col, inplace=True)
    
    # 自动检测股票代码列
    code_col = None
    for col in code_cols:
        if col in raw_data.columns:
            code_col = col
            break
    if code_col is None:
        for col in raw_data.columns:
            if 'code' in col.lower() or '代码' in col.lower():
                code_col = col
                break
    
    # 自动检测OHLCV列
    detected_fields = {}
    for field, possible_names in field_mapping.items():
        for name in possible_names:
            if name in raw_data.columns:
                detected_fields[field] = name
                break
    
    return raw_data, code_col, detected_fields


def prepare_data_feeds(raw_data, code_col, detected_fields, max_stocks=None,backtest_start=datetime(2020, 1, 1),backtest_end=datetime(2025, 12, 31)):
    """准备数据源列表（优化版：使用 groupby 预分组）"""
    import time
    t0 = time.time()
    
    print(f'找到 {len(raw_data[code_col].unique())} 只股票')
    
    ohlc_cols = [detected_fields['open'], detected_fields['high'], 
                detected_fields['low'], detected_fields['close']]
    
    # Step 1: 在原始数据上计算每只股票的起始日期（用于筛选从回测开始就有数据的股票）
    t1 = time.time()
    first_dates = raw_data.groupby(code_col)[ohlc_cols[0]].apply(lambda x: x.index.min())
    # 允许 backtest_start 后一周内就有数据的股票（容忍节假日等情况）
    grace_period = pd.Timedelta(days=7)
    valid_early_codes = set(first_dates[first_dates <= backtest_start + grace_period].index)
    print(f'从 {backtest_start.date()} 前后就有数据的股票: {len(valid_early_codes)} 只 ({time.time()-t1:.2f}s)')
    
    # Step 2: 过滤数据（日期范围 + OHLC 有效性）
    t2 = time.time()
    valid_mask = (
        (raw_data.index >= backtest_start) & 
        (raw_data.index <= backtest_end) &
        (raw_data[code_col].isin(valid_early_codes))  # 只保留早期有数据的股票
    )
    for col in ohlc_cols:
        if col in raw_data.columns:
            valid_mask &= (raw_data[col] > 0) & raw_data[col].notna()
    
    filtered_data = raw_data[valid_mask]
    print(f'过滤后数据量: {len(filtered_data)} 行 ({time.time()-t2:.2f}s)')
    
    # Step 3: 按股票分组并筛选数据量足够的股票
    t3 = time.time()
    grouped = filtered_data.groupby(code_col, sort=False)
    stock_counts = grouped.size()
    valid_codes = stock_counts[stock_counts >= 21].index.tolist()
    
    if max_stocks is not None:
        selected_codes = sorted(valid_codes)[:max_stocks]
    else:
        selected_codes = valid_codes
    
    print(f'有效股票: {len(selected_codes)} 只 ({time.time()-t3:.2f}s)')
    
    # Step 4: 构建数据源，确保每个股票清理后的起始日期足够早
    t4 = time.time()
    selected_set = set(selected_codes)
    
    # 设置允许的最晚起始日期（例如回测开始后30天内）
    max_start_date = backtest_start + pd.Timedelta(days=30)
    
    data_feeds = []
    skipped_late_start = 0
    
    for code, group in grouped:
        if code in selected_set:
            stock_data = group.drop(columns=[code_col]).sort_index()
            if len(stock_data) >= 21:
                # 检查清理后的起始日期是否足够早
                actual_start = stock_data.index.min()
                if actual_start <= max_start_date:
                    data_feeds.append({'data': stock_data, 'name': str(code)})
                else:
                    skipped_late_start += 1
    
    if skipped_late_start > 0:
        print(f'跳过 {skipped_late_start} 只起始日期过晚的股票 ({time.time()-t4:.2f}s)')
    
    print(f'准备了 {len(data_feeds)} 个有效数据源 (总耗时: {time.time()-t0:.2f}s)')
    return data_feeds


def run_backtest(data_feeds, detected_fields, top_n_stocks, initial_cash=100000.0, verbose=False):
    """运行单次回测，返回指标"""
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)
    
    # 添加数据源
    for feed_info in data_feeds:
        data_feed = bt.feeds.PandasData(
            dataname=feed_info['data'],
            fromdate=datetime(2020, 1, 1),
            todate=datetime(2025, 12, 31),
            open=detected_fields['open'],
            high=detected_fields['high'],
            low=detected_fields['low'],
            close=detected_fields['close'],
            volume=detected_fields['volume'],
            name=feed_info['name']
        )
        cerebro.adddata(data_feed)
        print(f'已添加股票 {feed_info["name"]}, 共 {len(feed_info["data"])} 条记录')
    # 添加策略
    cerebro.addstrategy(SimpleFactorStrategy, top_n_stocks=top_n_stocks)
    
    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns',
                       timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    cerebro.addanalyzer(bt.analyzers.Calmar, _name='calmar',
                        timeframe=bt.TimeFrame.Days)
    
    if verbose:
        print(f'\n运行回测: top_n_stocks = {top_n_stocks}')
        print(f'初始资产: {cerebro.broker.getvalue():.2f}')
    
    # 运行回测
    results = cerebro.run()
    strat = results[0]
    
    # 收集指标
    final_value = cerebro.broker.getvalue()
    total_return_pct = (final_value - initial_cash) / initial_cash * 100
    
    # 夏普率
    sharpe = strat.analyzers.sharpe.get_analysis()
    sharpe_ratio = sharpe.get('sharperatio', None)
    if sharpe_ratio is None or np.isnan(sharpe_ratio):
        sharpe_ratio = 0.0
    
    # 最大回撤 (backtrader 返回的已经是百分比形式)
    drawdown = strat.analyzers.drawdown.get_analysis()
    max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
    
    # 交易分析
    trades = strat.analyzers.trades.get_analysis()
    total_trades = trades.get('total', {}).get('total', 0)
    won_trades = trades.get('won', {}).get('total', 0)
    lost_trades = trades.get('lost', {}).get('total', 0)
    win_rate = won_trades / total_trades * 100 if total_trades > 0 else 0.0
    
    # 年化收益率
    returns = strat.analyzers.returns.get_analysis()
    annual_return = returns.get('rnorm100', 0)
    
    # SQN
    sqn = strat.analyzers.sqn.get_analysis()
    sqn_value = sqn.get('sqn', 0)
    if sqn_value is None or np.isnan(sqn_value):
        sqn_value = 0.0
    
    # Calmar比率
    calmar = strat.analyzers.calmar.get_analysis()
    calmar_value = list(calmar.values())[0] if calmar else 0.0
    if calmar_value is None or np.isnan(calmar_value):
        calmar_value = 0.0
    
    metrics = {
        'top_n_stocks': top_n_stocks,
        'final_value': final_value,
        'total_return': total_return_pct,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'sqn': sqn_value,
        'calmar': calmar_value
    }
    
    if verbose:
        print(f'最终资产: {final_value:.2f}')
        print(f'总收益率: {total_return_pct:.2f}%')
        print(f'夏普率: {sharpe_ratio:.4f}')
        print(f'最大回撤: {max_drawdown:.2f}%')
    
    # 返回指标和策略对象（用于绘图）
    return metrics, strat, initial_cash, final_value


def plot_param_sweep_results(results_df):
    """绘制参数扫描结果图像"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    x = results_df['top_n_stocks']
    
    # 1. 总收益率
    ax1 = axes[0, 0]
    ax1.bar(x, results_df['total_return'], color='#2E86AB', alpha=0.8)
    ax1.set_xlabel('持仓股票数量 (top_n_stocks)', fontsize=11)
    ax1.set_ylabel('总收益率 (%)', fontsize=11)
    ax1.set_title('总收益率 vs 持仓数量', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # 标注最大值
    max_idx = results_df['total_return'].idxmax()
    max_val = results_df.loc[max_idx, 'total_return']
    max_x = results_df.loc[max_idx, 'top_n_stocks']
    ax1.annotate(f'最优: {max_x}\n收益: {max_val:.1f}%', 
                 xy=(max_x, max_val), xytext=(max_x+1, max_val*1.1),
                 fontsize=9, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    # 2. 夏普率
    ax2 = axes[0, 1]
    ax2.bar(x, results_df['sharpe_ratio'], color='#28A745', alpha=0.8)
    ax2.set_xlabel('持仓股票数量 (top_n_stocks)', fontsize=11)
    ax2.set_ylabel('夏普率', fontsize=11)
    ax2.set_title('夏普率 vs 持仓数量', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='夏普率=1')
    # 标注最大值
    max_idx = results_df['sharpe_ratio'].idxmax()
    max_val = results_df.loc[max_idx, 'sharpe_ratio']
    max_x = results_df.loc[max_idx, 'top_n_stocks']
    ax2.annotate(f'最优: {max_x}\n夏普: {max_val:.2f}', 
                 xy=(max_x, max_val), xytext=(max_x+1, max_val*1.1),
                 fontsize=9, color='red',
                 arrowprops=dict(arrowstyle='->', color='red', lw=1))
    
    # 3. 最大回撤
    ax3 = axes[1, 0]
    ax3.bar(x, results_df['max_drawdown'], color='#DC3545', alpha=0.8)
    ax3.set_xlabel('持仓股票数量 (top_n_stocks)', fontsize=11)
    ax3.set_ylabel('最大回撤 (%)', fontsize=11)
    ax3.set_title('最大回撤 vs 持仓数量', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    # 标注最小值（最好）
    min_idx = results_df['max_drawdown'].idxmin()
    min_val = results_df.loc[min_idx, 'max_drawdown']
    min_x = results_df.loc[min_idx, 'top_n_stocks']
    ax3.annotate(f'最优: {min_x}\n回撤: {min_val:.1f}%', 
                 xy=(min_x, min_val), xytext=(min_x+1, min_val*0.8),
                 fontsize=9, color='green',
                 arrowprops=dict(arrowstyle='->', color='green', lw=1))
    
    # 4. 年化收益率
    ax4 = axes[1, 1]
    ax4.bar(x, results_df['annual_return'], color='#6F42C1', alpha=0.8)
    ax4.set_xlabel('持仓股票数量 (top_n_stocks)', fontsize=11)
    ax4.set_ylabel('年化收益率 (%)', fontsize=11)
    ax4.set_title('年化收益率 vs 持仓数量', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # 5. 胜率
    ax5 = axes[2, 0]
    ax5.bar(x, results_df['win_rate'], color='#FD7E14', alpha=0.8)
    ax5.set_xlabel('持仓股票数量 (top_n_stocks)', fontsize=11)
    ax5.set_ylabel('胜率 (%)', fontsize=11)
    ax5.set_title('胜率 vs 持仓数量', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
    
    # 6. 交易次数
    ax6 = axes[2, 1]
    ax6.bar(x, results_df['total_trades'], color='#17A2B8', alpha=0.8)
    ax6.set_xlabel('持仓股票数量 (top_n_stocks)', fontsize=11)
    ax6.set_ylabel('交易次数', fontsize=11)
    ax6.set_title('交易次数 vs 持仓数量', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('策略参数扫描: top_n_stocks 敏感性分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # 保存图片
    output_file = 'param_sweep_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'\n参数扫描结果图已保存到: {output_file}')
    
    plt.show()
    
    return fig


def plot_combined_metrics(results_df):
    """绘制综合指标对比图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = results_df['top_n_stocks']
    
    # 归一化数据以便在同一图上显示
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) * 100 if series.max() != series.min() else series
    
    ax.plot(x, normalize(results_df['total_return']), 'o-', label='总收益率 (归一化)', linewidth=2, markersize=8)
    ax.plot(x, normalize(results_df['sharpe_ratio']), 's-', label='夏普率 (归一化)', linewidth=2, markersize=8)
    ax.plot(x, 100 - normalize(results_df['max_drawdown']), '^-', label='风险控制 (归一化)', linewidth=2, markersize=8)
    ax.plot(x, normalize(results_df['win_rate']), 'd-', label='胜率 (归一化)', linewidth=2, markersize=8)
    
    ax.set_xlabel('持仓股票数量 (top_n_stocks)', fontsize=12)
    ax.set_ylabel('归一化得分 (0-100)', fontsize=12)
    ax.set_title('策略指标综合对比', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    
    plt.tight_layout()
    
    output_file = 'param_sweep_combined.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'综合指标对比图已保存到: {output_file}')
    
    plt.show()
    
    return fig


def plot_2d_heatmap(results_df, metric='total_return', title='参数扫描热力图'):
    """绘制二维参数扫描热力图"""
    import seaborn as sns
    
    # 创建透视表
    pivot = results_df.pivot(index='max_stocks', columns='top_n', values=metric)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制热力图
    cmap = 'RdYlGn' if metric in ['total_return', 'sharpe_ratio', 'annual_return'] else 'RdYlGn_r'
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap=cmap, ax=ax, 
                cbar_kws={'label': metric})
    
    ax.set_xlabel('持仓数量 (top_n_stocks)', fontsize=12)
    ax.set_ylabel('股票池大小 (max_stocks)', fontsize=12)
    ax.set_title(f'{title}: {metric}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_file = f'heatmap_{metric}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'热力图已保存到: {output_file}')
    plt.close()
    
    return fig


def run_2d_param_sweep(raw_data, code_col, detected_fields, 
                       max_stocks_values, top_n_values,
                       backtest_start, backtest_end,
                       plot_curves=True, save_dir='param_sweep_curves'):
    """
    运行二维参数扫描
    
    参数:
        max_stocks_values: 股票池大小列表
        top_n_values: 持仓数量列表
        plot_curves: 是否绘制每次回测的收益曲线
        save_dir: 收益曲线保存目录
    """
    import os
    
    if plot_curves and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    all_results = []
    total_runs = len(max_stocks_values) * len(top_n_values)
    current_run = 0
    
    print(f'\n总共需要运行 {total_runs} 次回测')
    print('=' * 60)
    
    for max_stocks in max_stocks_values:
        # 为每个 max_stocks 准备数据源
        print(f'\n准备数据源: max_stocks = {max_stocks}')
        data_feeds = prepare_data_feeds(raw_data, code_col, detected_fields, 
                                        max_stocks=max_stocks,
                                        backtest_start=backtest_start, 
                                        backtest_end=backtest_end)
        
        if len(data_feeds) == 0:
            print(f'警告: max_stocks={max_stocks} 没有有效数据源，跳过')
            continue
        
        for top_n in top_n_values:
            current_run += 1
            print(f'\n[{current_run}/{total_runs}] max_stocks={max_stocks}, top_n={top_n}')
            
            try:
                metrics, strat, initial_value, final_value = run_backtest(
                    data_feeds, detected_fields, top_n, verbose=True
                )
                
                # 添加 max_stocks 到结果
                metrics['max_stocks'] = max_stocks
                metrics['top_n'] = top_n
                all_results.append(metrics)
                
                # 绘制收益曲线
                if plot_curves and hasattr(strat, 'portfolio_values') and len(strat.portfolio_values) > 0:
                    # 保存当前图像设置
                    plt.figure(figsize=(15, 10))
                    
                    # 调用 plot_equity_curve（修改为保存到指定文件）
                    curve_file = os.path.join(save_dir, f'curve_max{max_stocks}_top{top_n}.png')
                    plot_equity_curve_to_file(strat, initial_value, final_value, curve_file,
                                              title=f'max_stocks={max_stocks}, top_n={top_n}')
                    plt.close('all')
                    
            except Exception as e:
                print(f'错误: {e}')
                import traceback
                traceback.print_exc()
                continue
    
    return pd.DataFrame(all_results)


def plot_equity_curve_to_file(strategy, initial_value, final_value, output_file, title=''):
    """绘制收益曲线并保存到文件"""
    import matplotlib.dates as mdates
    
    if not hasattr(strategy, 'portfolio_values') or len(strategy.portfolio_values) == 0:
        print('警告: 没有资产价值数据，无法绘制图表')
        return
    
    # 准备数据
    dates = pd.to_datetime(strategy.portfolio_dates)
    values = strategy.portfolio_values
    drawdowns = strategy.drawdowns
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    
    # 第一个子图：资产价值曲线
    ax1.plot(dates, values, label='资产价值', linewidth=1.5, color='#2E86AB')
    ax1.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.5, label='初始资金')
    ax1.axhline(y=final_value, color='green', linestyle='--', alpha=0.5, label='最终资产')
    
    # 标注最大回撤
    ax1.plot([peak_date, max_dd_date], [peak_before_dd, max_dd_portfolio_value], 
             'r--', linewidth=2, alpha=0.7)
    ax1.scatter([peak_date, max_dd_date], [peak_before_dd, max_dd_portfolio_value], 
                color='red', s=80, zorder=5)
    
    total_return = (final_value - initial_value) / initial_value * 100
    ax1.set_ylabel('资产价值 (元)', fontsize=11)
    ax1.set_title(f'{title}\n总收益: {total_return:.1f}%, 最大回撤: {max_dd_value:.1f}%', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    
    # 第二个子图：回撤曲线
    ax2.fill_between(dates, 0, drawdowns, alpha=0.3, color='red')
    ax2.plot(dates, drawdowns, color='red', linewidth=1)
    ax2.axhline(y=max_dd_value, color='darkred', linestyle='--', alpha=0.7)
    
    ax2.set_ylabel('回撤 (%)', fontsize=11)
    ax2.set_xlabel('日期', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # 格式化日期
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'  收益曲线已保存到: {output_file}')


def run_1d_param_sweep(raw_data, code_col, detected_fields, 
                       top_n_values, max_stocks=None,
                       backtest_start=datetime(2020, 1, 1), 
                       backtest_end=datetime(2025, 12, 31),
                       plot_curves=True, save_dir='param_sweep_curves'):
    """
    运行一维参数扫描（只扫描 top_n）
    
    参数:
        top_n_values: 持仓数量列表
        max_stocks: 股票池大小（固定值，None 表示使用所有股票）
        plot_curves: 是否绘制每次回测的收益曲线
        save_dir: 收益曲线保存目录
    """
    import os
    
    if plot_curves and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 准备数据源（只准备一次）
    print(f'\n准备数据源: max_stocks = {max_stocks if max_stocks else "全部"}')
    data_feeds = prepare_data_feeds(raw_data, code_col, detected_fields, 
                                    max_stocks=max_stocks,
                                    backtest_start=backtest_start, 
                                    backtest_end=backtest_end)
    
    if len(data_feeds) == 0:
        print('错误: 没有有效数据源')
        return pd.DataFrame()
    
    all_results = []
    total_runs = len(top_n_values)
    
    print(f'\n总共需要运行 {total_runs} 次回测')
    print('=' * 60)
    
    for i, top_n in enumerate(top_n_values, 1):
        print(f'\n[{i}/{total_runs}] top_n = {top_n}')
        
        try:
            metrics, strat, initial_value, final_value = run_backtest(
                data_feeds, detected_fields, top_n, verbose=True
            )
            
            metrics['top_n'] = top_n
            all_results.append(metrics)
            
            # 绘制收益曲线
            if plot_curves and hasattr(strat, 'portfolio_values') and len(strat.portfolio_values) > 0:
                curve_file = os.path.join(save_dir, f'curve_top{top_n}.png')
                plot_equity_curve_to_file(strat, initial_value, final_value, curve_file,
                                          title=f'top_n = {top_n}')
                plt.close('all')
                
        except Exception as e:
            print(f'错误: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    return pd.DataFrame(all_results)


def plot_sweep_summary(results_df):
    """绘制参数扫描结果汇总图（三个指标柱状图）"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = results_df['top_n']
    x_labels = [str(v) for v in x]
    x_pos = range(len(x))
    
    # 1. 总收益率
    ax1 = axes[0]
    bars1 = ax1.bar(x_pos, results_df['total_return'], color='#2E86AB', alpha=0.8, edgecolor='white')
    ax1.set_xlabel('持仓数量 (top_n)', fontsize=11)
    ax1.set_ylabel('总收益率 (%)', fontsize=11)
    ax1.set_title('总收益率', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    # 在柱子上显示数值
    for bar, val in zip(bars1, results_df['total_return']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    # 标注最大值
    max_idx = results_df['total_return'].idxmax()
    bars1[max_idx].set_color('#FF6B6B')
    bars1[max_idx].set_edgecolor('red')
    bars1[max_idx].set_linewidth(2)
    
    # 2. 夏普率
    ax2 = axes[1]
    bars2 = ax2.bar(x_pos, results_df['sharpe_ratio'], color='#28A745', alpha=0.8, edgecolor='white')
    ax2.set_xlabel('持仓数量 (top_n)', fontsize=11)
    ax2.set_ylabel('夏普率', fontsize=11)
    ax2.set_title('夏普率', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='夏普率=1')
    for bar, val in zip(bars2, results_df['sharpe_ratio']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                 f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    max_idx = results_df['sharpe_ratio'].idxmax()
    bars2[max_idx].set_color('#90EE90')
    bars2[max_idx].set_edgecolor('darkgreen')
    bars2[max_idx].set_linewidth(2)
    
    # 3. 最大回撤
    ax3 = axes[2]
    bars3 = ax3.bar(x_pos, results_df['max_drawdown'], color='#DC3545', alpha=0.8, edgecolor='white')
    ax3.set_xlabel('持仓数量 (top_n)', fontsize=11)
    ax3.set_ylabel('最大回撤 (%)', fontsize=11)
    ax3.set_title('最大回撤', fontsize=13, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels)
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars3, results_df['max_drawdown']):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    min_idx = results_df['max_drawdown'].idxmin()
    bars3[min_idx].set_color('#90EE90')
    bars3[min_idx].set_edgecolor('darkgreen')
    bars3[min_idx].set_linewidth(2)
    
    plt.suptitle('参数扫描结果汇总', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = 'param_sweep_summary.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f'汇总图已保存到: {output_file}')
    plt.show()


if __name__ == '__main__':
    print('=' * 60)
    print('参数扫描: top_n_stocks 敏感性分析')
    print('=' * 60)
    
    # 加载数据
    raw_data, code_col, detected_fields = load_data('~/etf_daily.parquet')
    # raw_data, code_col, detected_fields = load_data('~/wind_daily_2010_adj.parquet')
    # 定义参数空间
    top_n_values = [1, 5]  # 持仓数量
    # top_n_values = [3]
    max_stocks = None  # 限制股票数量加快测试
    
    # 回测时间范围
    backtest_start = datetime(2020, 2, 1)
    backtest_end = datetime(2025, 12, 31)
    
    # 运行一维参数扫描
    results_df = run_1d_param_sweep(
        raw_data, code_col, detected_fields,
        top_n_values, max_stocks=max_stocks,
        backtest_start=backtest_start, backtest_end=backtest_end,
        plot_curves=True,
        save_dir='param_sweep_curves'
    )
    
    if len(results_df) == 0:
        print('错误: 没有有效的回测结果')
        exit(1)
    
    # 打印汇总表格
    print('\n' + '=' * 60)
    print('参数扫描结果汇总')
    print('=' * 60)
    print(results_df[['top_n', 'total_return', 'sharpe_ratio', 'max_drawdown', 
                      'annual_return', 'total_trades', 'win_rate']].to_string(index=False))
    
    # 保存结果到 CSV
    results_df.to_csv('param_sweep_results.csv', index=False)
    print('\n结果已保存到: param_sweep_results.csv')
    
    # 找出最优参数
    print('\n' + '=' * 60)
    print('最优参数分析')
    print('=' * 60)
    
    best_return_idx = results_df['total_return'].idxmax()
    best_sharpe_idx = results_df['sharpe_ratio'].idxmax()
    best_drawdown_idx = results_df['max_drawdown'].idxmin()
    
    print(f'最高收益率: top_n = {results_df.loc[best_return_idx, "top_n"]}, '
          f'收益率 = {results_df.loc[best_return_idx, "total_return"]:.2f}%')
    
    print(f'最高夏普率: top_n = {results_df.loc[best_sharpe_idx, "top_n"]}, '
          f'夏普率 = {results_df.loc[best_sharpe_idx, "sharpe_ratio"]:.4f}')
    
    print(f'最低回撤:   top_n = {results_df.loc[best_drawdown_idx, "top_n"]}, '
          f'最大回撤 = {results_df.loc[best_drawdown_idx, "max_drawdown"]:.2f}%')
    
    # 绘制汇总图并显示
    print('\n正在生成汇总图...')
    plot_sweep_summary(results_df)
    
    print('\n实验完成！')

