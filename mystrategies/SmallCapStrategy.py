import backtrader as bt
import argparse
import pandas as pd
import os
from datetime import datetime
from datetime import date
import logging
import matplotlib.pyplot as plt
from backtrader import strategy
from utils import setup_chinese_font
try:
    from mystrategies.config_loader import load_config, get_repo_root, resolve_path
except ImportError:
    from config_loader import load_config, get_repo_root, resolve_path

setup_chinese_font()

# 配置日志：级别为 DEBUG（输出所有级别）、格式包含时间/级别/模块/内容
logging.basicConfig(
    level=logging.DEBUG,  # 输出级别：DEBUG 及以上
    format="%(module)s:%(lineno)d - %(message)s",  # 格式
    datefmt="%Y-%m-%d %H:%M:%S"  # 时间格式
)

class ASharePandasData(bt.feeds.PandasData):
    """支持流通市值的 PandasData"""
    lines = ('free_mktcap',)
    params = (
        ('free_mktcap', 'FREE_MKTCAP'),  # 列名，如果不存在会被忽略
    )


class SmallCapMomentumStrategy(bt.Strategy):
    params = dict(
        lookback=60,
        ma_period=120,
        rebalance_months=1,
        smallcap_pct=0.3,   # 后30%小盘股
        hold_num=10,
    )

    def __init__(self):
        self.mas = {}
        self.returns = {}
        self.last_rebalance_month = None
        self.portfolio_values = []
        self.portfolio_dates = []

        for d in self.datas:
            self.mas[d] = bt.indicators.SMA(d.close, period=self.p.ma_period)
            self.returns[d] = d.close / d.close(-self.p.lookback) - 1

    def prenext(self):
        self.next()

    def next(self):
        current_date = self.datas[0].datetime.date(0)
        current_value = self.broker.getvalue()
        
        self.portfolio_values.append(current_value)
        self.portfolio_dates.append(current_date)
        dt = self.datas[0].datetime.date(0)

        # 控制月频调仓
        if self.last_rebalance_month == (dt.year, dt.month):
            return
        self.last_rebalance_month = (dt.year, dt.month)

        candidates = []

        for d in self.datas:
            if len(d) < max(self.p.lookback, self.p.ma_period):
                continue

            if d.close[0] <= self.mas[d][0]:
                continue

            if self.returns[d][0] <= 0:
                continue

            candidates.append(d)

        if not candidates:
            return

        # ===== 按流通市值或收盘价排序（小盘在前） =====
        # 如果有 amt 字段则使用，否则用收盘价作为替代
        # ===== 综合因子：按成交额(市值) & 动量 =====
        stock_factors = []
        for d in candidates:
            # 首选用 amt 作为市值
            
            amt_val = d.amt[0]
            
            # 动量
            momentum_val = self.returns[d][0] if pd.notna(self.returns[d][0]) else 0
            # 归一化市值[市值越小因子越高]，先简单取倒数（实际可z-score/分组等）
            value_factor = 1.0 / amt_val if amt_val > 0 else 0
            # 综合分（可以自定义权重，这里都权重为1）
            composite_score = value_factor + momentum_val
            stock_factors.append({'data': d, 'score': composite_score, 'amt': amt_val, 'momentum': momentum_val})

        # 按综合得分降序排列，选出目标池
        stock_factors.sort(key=lambda x: x['score'], reverse=True)

        targets = [x['data'] for x in stock_factors[:self.p.hold_num]]
        target_set = set(targets)

        # ===== 卖出不在目标池的 =====
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0 and d not in target_set:
                self.close(d)
                logging.debug(f'{dt},卖出 {d._name} {pos.size} 股,价格 {d.close[0]:.2f}')

        # ===== 等权买入 =====
        cash = self.broker.getcash()
        print(cash)
        if targets:
            alloc = cash / len(targets)

        for d in targets:
            price = d.close[0]
            size = int(alloc / price / 100) * 100
            if size > 0:
                self.order_target_size(d, size)
                logging.debug(f'{dt},买入 {d._name} {size} 股,价格 {price:.2f}')

def load_parquet_data(parquet_path, max_stocks=None, backtest_start=None, backtest_end=None,
                      mktcap_path="~/wind_mktcap.csv"):
    """
    从 parquet 文件加载股票数据
    
    Args:
        parquet_path: parquet 文件路径
        max_stocks: 最大加载股票数量（可选）
        backtest_start: 回测开始日期（可选）
        backtest_end: 回测结束日期（可选）
    
    Returns:
        data_feeds: 包含每只股票数据的列表
    """
    df_mktcap = pd.read_csv(os.path.expanduser(mktcap_path), index_col=0)
    df_mktcap.sort_values(by='MKT_FREESHARES', ascending=True, inplace=True)
    smallcap_codelist = df_mktcap.head(max_stocks).index.values

    print(f'正在读取 {parquet_path}...')
    df = pd.read_parquet(os.path.expanduser(parquet_path))
    print(f'数据量: {len(df)} 行')
    print(f'列名: {df.columns.tolist()}')
    
    
    # 检测列名（支持不同数据源）
    col_mapping = {}
    for col in df.columns:
        col_upper = col.upper()
        if 'CODE' in col_upper or '代码' in col:
            col_mapping['code'] = col
        elif col_upper == 'OPEN' or '开盘' in col:
            col_mapping['open'] = col
        elif col_upper == 'HIGH' or '最高' in col:
            col_mapping['high'] = col
        elif col_upper == 'LOW' or '最低' in col:
            col_mapping['low'] = col
        elif col_upper == 'CLOSE' or '收盘' in col:
            col_mapping['close'] = col
        elif col_upper == 'VOLUME' or '成交量' in col:
            col_mapping['volume'] = col
        elif 'FREE_MKTCAP' in col_upper or '流通市值' in col:
            col_mapping['free_mktcap'] = col
    
    code_col = col_mapping.get('code', 'CODE')
    print(f'股票代码列: {code_col}')
    
    # 确保索引是日期类型
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns.str.lower().tolist():
            date_col = [c for c in df.columns if c.lower() == 'date'][0]
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
        else:
            df.index = pd.to_datetime(df.index)
    
    # 过滤日期范围
    if backtest_start:
        df = df[df.index >= backtest_start]
    if backtest_end:
        df = df[df.index <= backtest_end]
    
    # 过滤无效数据
    close_col = col_mapping.get('close', 'CLOSE')
    df = df[(df[close_col] > 0) & (df[close_col].notna())]
    
    # 按股票代码分组
    stock_codes = df[code_col].unique()
    print(f'共 {len(stock_codes)} 只股票')
    
    # if max_stocks:
    #     stock_codes = sorted(stock_codes)[:max_stocks]
    #     print(f'限制为 {len(stock_codes)} 只股票')
    
    data_feeds = []
    grouped = df.groupby(code_col)
    tmpdf = pd.DataFrame()
    for stock_code in stock_codes:
        if stock_code not in smallcap_codelist:
            continue
        
        stock_data = grouped.get_group(stock_code).copy()
        stock_data.sort_index(inplace=True)
        
        # 确保有足够的数据
        if len(stock_data) < 120:  # 至少需要 ma_period 天数据
            continue
        tmpdf = pd.concat([tmpdf, pd.DataFrame({'datemin': stock_data.index.min(), 'index': stock_code},index=[0])])
        # 重命名列以匹配 backtrader 标准
        rename_dict = {}
        if 'open' in col_mapping:
            rename_dict[col_mapping['open']] = 'open'
        if 'high' in col_mapping:
            rename_dict[col_mapping['high']] = 'high'
        if 'low' in col_mapping:
            rename_dict[col_mapping['low']] = 'low'
        if 'close' in col_mapping:
            rename_dict[col_mapping['close']] = 'close'
        if 'volume' in col_mapping:
            rename_dict[col_mapping['volume']] = 'volume'
        if 'free_mktcap' in col_mapping:
            rename_dict[col_mapping['free_mktcap']] = 'free_mktcap'
        
        stock_data.rename(columns=rename_dict, inplace=True)
        
        # 移除股票代码列
        stock_data = stock_data.drop(columns=[code_col], errors='ignore')
        
        data_feeds.append({
            'data': stock_data,
            'name': str(stock_code)
        })
    
    print(f'准备了 {len(data_feeds)} 个有效数据源')
    tmpdf.to_csv('data_feeds.csv',index=False)
    return data_feeds, 'free_mktcap' in col_mapping


def main(config: dict):
    config = dict(config)
    parquet_path = resolve_path(config["parquet_path"], base_dir=get_repo_root())
    config["parquet_path"] = parquet_path
    backtest_start = config.get("backtest_start", datetime(2015, 1, 1))
    backtest_end = config.get("backtest_end", datetime(2019, 12, 31))
    max_stocks = config.get("max_stocks", 1500)
    mktcap_path = resolve_path(config.get("mktcap_path", "~/wind_mktcap.csv"), base_dir=get_repo_root())
    initial_cash = config.get("initial_cash", 1_000_000)
    commission = config.get("commission", 0.0003)
    strategy_params = config.get("strategy_params", {})

    cerebro = bt.Cerebro()
    cerebro.addstrategy(SmallCapMomentumStrategy, **strategy_params)

    data_feeds, has_mktcap = load_parquet_data(
        parquet_path,
        max_stocks=max_stocks,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        mktcap_path=mktcap_path,
    )

    if not has_mktcap:
        print('\n警告: 数据中没有流通市值字段，策略将使用收盘价作为替代排序依据')

    for feed in data_feeds:
        if has_mktcap and 'free_mktcap' in feed['data'].columns:
            data = ASharePandasData(
                dataname=feed['data'],
                fromdate=backtest_start,
                todate=backtest_end,
                free_mktcap='free_mktcap',
            )
        else:
            data = bt.feeds.PandasData(
                dataname=feed['data'],
                fromdate=backtest_start,
                todate=backtest_end,
            )
        cerebro.adddata(data, name=feed['name'])

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=commission, stocklike=True)

    print(f'\n开始回测: {backtest_start.date()} 到 {backtest_end.date()}')
    print('Start Value:', cerebro.broker.getvalue())
    results = cerebro.run()
    print('Final Value:', cerebro.broker.getvalue())

    initial = initial_cash
    final = cerebro.broker.getvalue()
    total_return = (final - initial) / initial * 100
    print(f'总收益率: {total_return:.2f}%')

    strategy = results[0]
    dates = pd.to_datetime(strategy.portfolio_dates)
    values = strategy.portfolio_values
    plt.plot(dates, values)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SmallCapMomentumStrategy backtest runner")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON/YAML config (default: configs/small_cap_strategy.json)",
    )
    args = parser.parse_args()
    default_config = get_repo_root() / "configs" / "small_cap_strategy.json"
    config_path = args.config or str(default_config)
    config = load_config(config_path)
    main(config)
