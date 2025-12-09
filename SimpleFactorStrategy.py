from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

# 初始化中文字体
setup_chinese_font()

class SimpleFactorStrategy(bt.Strategy):
    """
    一个简化的动量与估值双因子选股策略
    - 动量：过去20个交易日的回报率
    - 估值：以当前收盘价的倒数作为代理因子 (越低越好，但我们用倒数让它越高越好)
    """
    params = (
        ('momentum_period', 20),  # 动量计算周期
        ('top_n_stocks', 5),      # 每期买入排名前N的股票数量
        ('rebalance_monthday', 1),# 每月1号换仓
    )

    def __init__(self):
        self.ranking = []
        self.order_refs = {} # 用于追踪订单
        self.last_rebalance_date = None
        
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

    def next(self):
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
        
        # 2. 检查是否是换仓日 (例如，每月的第一天)
        
        # 仅在指定日期执行，避免在同一天重复计算
        if current_date.day != self.p.rebalance_monthday:
            return

        # 检查是否是本月第一次执行
        if self.last_rebalance_date and self.last_rebalance_date.month == current_date.month:
            return
        
        self.last_rebalance_date = current_date
        self.log(f'--- 换仓日 {current_date}：开始计算因子并轮动 ---')

        # 3. 计算因子并进行排序

        candidates = []
        for data in self.datas:
            name = data._name
            mom_ind = self.momentum_indicators[name]

            # 确保数据已就绪
            if len(data) < self.p.momentum_period or mom_ind[0] is None:
                continue
            
            # 检查价格是否有效（避免除零错误）
            if data.close[0] <= 0 or pd.isna(data.close[0]):
                continue

            # 因子值获取
            mom_score = mom_ind[0]             # 动量得分 (越高越好)
            # 使用安全的除法，避免除零错误
            val_score = 1.0 / data.close[0] if data.close[0] > 0 else 0.0    # 估值得分 (股价倒数，越高越好)

            # 组合因子 (简化为相加)
            # 实际中会进行标准化、加权和回归等复杂处理
            composite_score = mom_score + (val_score * 1000) # 估值权重放大

            candidates.append({
                'name': name,
                'data': data,
                'score': composite_score,
                'mom': mom_score,
                'val': val_score
            })

        if not candidates:
            return

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
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
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
def plot_equity_curve(strategy, initial_value, final_value):
    """
    绘制收益曲线图，标注最大回撤
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
    
    # 保存图片
    output_file = 'equity_curve.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'收益曲线图已保存到: {output_file}')
    
    # 显示图表
    plt.show()

# --- 回测运行部分 ---

if __name__ == '__main__':
    import os
    
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    # 读取 wind_daily_2010_adj.parquet 文件
    parquet_file = os.path.expanduser('~/etf_daily.parquet')
    print(f'正在读取 {parquet_file}...')
    
    try:
        raw_data = pd.read_parquet(parquet_file)
        print(f'成功读取数据，共 {len(raw_data)} 行')
        print(f'数据列名: {list(raw_data.columns)}')
        
        # 字段名映射：支持中英文字段名
        # 日期字段可能的名称
        date_cols = ['日期', 'date', 'trade_date', '交易日期', 'datetime']
        # 股票代码字段可能的名称
        code_cols = ['股票代码', 'code', 'wind_code', 'ts_code', 'symbol', '股票代码']
        # OHLCV字段可能的名称（中文和英文）
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
            # 如果索引已经是日期类型，使用索引
            date_col = None
            print('使用索引作为日期')
        elif date_col is None:
            # 尝试查找包含date的列
            for col in raw_data.columns:
                if 'date' in col.lower() or '日期' in col.lower():
                    date_col = col
                    break
        
        if date_col:
            raw_data[date_col] = pd.to_datetime(raw_data[date_col])
            raw_data.set_index(date_col, inplace=True)
            print(f'使用 {date_col} 作为日期列')
        elif not isinstance(raw_data.index, pd.DatetimeIndex):
            raise ValueError('无法找到日期列，请确保数据包含日期信息')
        
        # 自动检测股票代码列
        code_col = None
        for col in code_cols:
            if col in raw_data.columns:
                code_col = col
                break
        if code_col is None:
            # 尝试查找包含code的列
            for col in raw_data.columns:
                if 'code' in col.lower() or '代码' in col.lower():
                    code_col = col
                    break
        
        if code_col is None:
            raise ValueError('无法找到股票代码列')
        print(f'使用 {code_col} 作为股票代码列')
        
        # 自动检测OHLCV列
        detected_fields = {}
        for field, possible_names in field_mapping.items():
            for name in possible_names:
                if name in raw_data.columns:
                    detected_fields[field] = name
                    break
            if field not in detected_fields:
                raise ValueError(f'无法找到 {field} 字段，尝试过的名称: {possible_names}')
        
        print(f'字段映射: {detected_fields}')
        
        # 按股票代码分组
        stock_codes = raw_data[code_col].unique()
        print(f'找到 {len(stock_codes)} 只股票')
        
        # 为每个股票代码创建数据源并添加到 cerebro
        # 限制处理前20只股票，避免数据量过大
        added_count = 0
        for stock_code in sorted(stock_codes)[:3]:
            stock_data = raw_data[raw_data[code_col] == stock_code].copy()
            
            # 确保数据按日期排序
            stock_data.sort_index(inplace=True)
            
            # 移除股票代码列（不再需要）
            stock_data = stock_data.drop(columns=[code_col])
            
            # 检查数据是否为空
            if len(stock_data) == 0:
                print(f'警告: 股票 {stock_code} 没有数据，跳过')
                continue
            
            # 数据清理：过滤掉价格为 0、NaN 或无效的数据
            # 检查 OHLC 列是否有效
            ohlc_cols = [detected_fields['open'], detected_fields['high'], 
                        detected_fields['low'], detected_fields['close']]
            for col in ohlc_cols:
                if col in stock_data.columns:
                    # 过滤掉价格为 0、NaN、负数或无穷大的行
                    stock_data = stock_data[
                        (stock_data[col] > 0) & 
                        (stock_data[col].notna()) &
                        (stock_data[col] != float('inf')) &
                        (stock_data[col] != float('-inf'))
                    ]
            
            # 再次检查数据是否为空（清理后）
            if len(stock_data) == 0:
                # print(f'警告: 股票 {stock_code} 清理后没有有效数据，跳过')
                continue
            
            # 确保数据量足够（至少需要 momentum_period + 1 条记录）
            if len(stock_data) < 21:  # 至少需要 20 + 1 条记录用于计算动量
                # print(f'警告: 股票 {stock_code} 数据量不足 ({len(stock_data)} 条)，跳过')
                continue
            
            # 创建 PandasData feed，使用检测到的字段名
            data_feed = bt.feeds.PandasData(
                dataname=stock_data,
                # 告诉 backtrader 你的 DataFrame 中对应 OHLCV 的列名
                fromdate=datetime(2020, 1, 1),
                todate=datetime(2025, 12, 31),
                open=detected_fields['open'],
                high=detected_fields['high'],
                low=detected_fields['low'],
                close=detected_fields['close'],
                volume=detected_fields['volume'],
                name=str(stock_code)  # 设置数据源名称
            )
            
            # 添加到 cerebro
            cerebro.adddata(data_feed)
            added_count += 1
            print(f'已添加股票 {stock_code}，共 {len(stock_data)} 条记录')
        
        print(f'总共添加了 {added_count} 个数据源到 cerebro')
        
    except FileNotFoundError:
        print(f'错误: 找不到文件 {parquet_file}')
        print('请确保 history.parquet 文件在当前目录或提供正确的路径')
        raise
    except Exception as e:
        print(f'读取数据时出错: {e}')
        import traceback
        traceback.print_exc()
        raise

    # 添加策略
    cerebro.addstrategy(SimpleFactorStrategy, top_n_stocks=3) # 只持有表现最好的 3 只

    # 添加常用的分析器
    # 1. 夏普率 (Sharpe Ratio) - 衡量风险调整后的收益
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Days, annualize=True)
    
    # 2. 回撤分析 (DrawDown) - 最大回撤和回撤持续时间
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    
    # 3. 交易分析 (TradeAnalyzer) - 交易统计信息
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    
    # 4. 收益率分析 (Returns) - 总收益率和年化收益率
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns',
                       timeframe=bt.TimeFrame.Days)
    
    # 5. 年化收益率 (AnnualReturn) - 每年的收益率
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annualreturn')
    
    # 6. 系统质量指标 (SQN) - 评估交易系统质量
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    # 7. 卡玛比率 (Calmar) - 年化收益率与最大回撤的比率
    cerebro.addanalyzer(bt.analyzers.Calmar, _name='calmar',
                        timeframe=bt.TimeFrame.Days)
    
    # 8. 时间收益率 (TimeReturn) - 按时间框架的收益率
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timereturn',
                       timeframe=bt.TimeFrame.Months)

    print('=' * 60)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('=' * 60)
    
    # 运行回测
    results = cerebro.run()
    strat = results[0]
    
    print('=' * 60)
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    print('=' * 60)
    
    # 打印分析结果
    print('\n' + '=' * 60)
    print('策略分析报告')
    print('=' * 60)
    
    # 计算实际的初始和最终资产价值
    initial_value = cerebro.broker.startingcash
    final_value = cerebro.broker.getvalue()
    total_return_pct = (final_value - initial_value) / initial_value * 100
    
    # 1. 收益率分析
    returns = strat.analyzers.returns.get_analysis()
    print('\n【收益率分析】')
    print(f'  初始资产: {initial_value:,.2f}')
    print(f'  最终资产: {final_value:,.2f}')
    print(f'  总收益率: {total_return_pct:.2f}%')
    print(f'  总收益倍数: {final_value / initial_value:.2f}x')
    # rtot是对数收益率，转换为百分比收益率: (e^rtot - 1) * 100%
    import math
    rtot_log = returns.get("rtot", 0)
    if rtot_log != float('-inf'):
        rtot_pct = (math.exp(rtot_log) - 1) * 100
        print(f'  对数收益率 (rtot): {rtot_log:.4f} = {rtot_pct:.2f}%')
    print(f'  年化收益率: {returns.get("rnorm", 0) * 100:.2f}%')
    print(f'  年化收益率 (100): {returns.get("rnorm100", 0):.2f}%')
    
    # 2. 夏普率
    sharpe = strat.analyzers.sharpe.get_analysis()
    print('\n【夏普率分析】')
    # if 'sharperatio' in sharpe:
    #     print(f'  夏普率: {sharpe["sharperatio"]:.4f}')
    # else:
    #     print('  夏普率: 数据不足，无法计算')
    
    # 3. 回撤分析
    drawdown = strat.analyzers.drawdown.get_analysis()
    print('\n【回撤分析】')
    print(f'  最大回撤: {drawdown.get("max", {}).get("drawdown", 0) * 100:.2f}%')
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

    # 绘制收益曲线和回撤图
    print('\n正在生成收益曲线图...')
    plot_equity_curve(strat, initial_value, final_value)
    
    # cerebro.plot() # 取消注释可以绘制结果