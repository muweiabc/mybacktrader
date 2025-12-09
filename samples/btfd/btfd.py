#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2023 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# References:
#  - https://www.reddit.com/r/algotrading/comments/5jez2b/can_anyone_replicate_this_strategy/
#  - http://dark-bid.com/BTFD-only-strategy-that-matters.html

import argparse
import datetime
import os
import sys
import backtrader as bt


class ValueUnlever(bt.observers.Value):
    '''Extension of regular Value observer to add leveraged view'''
    lines = ('value_lever', 'asset')
    params = (('assetstart', 100000.0), ('lever', True),)

    def next(self):
        super(ValueUnlever, self).next()
        if self.p.lever:
            self.lines.value_lever[0] = self._owner.broker._valuelever

        if len(self) == 1:
            self.lines.asset[0] = self.p.assetstart
        else:
            change = self.data[0] / self.data[-1]
            self.lines.asset[0] = change * self.lines.asset[-1]


class St(bt.Strategy):
    params = (
        ('fall', -0.01),      # 下跌阈值：-1%
        ('hold', 2),          # 持仓时间：2个bar
        ('approach', 'highlow'), # 计算下跌幅度的方法
        ('target', 1.0),      # 目标仓位：100%
        ('prorder', False),   # 是否打印订单信息
        ('prtrade', False),   # 是否打印交易信息
        ('prdata', False),    # 是否打印数据信息
    )

    def __init__(self):
        # 四种方法对比：
        # closeclose：收盘价相对于前一日收盘价的跌幅
        # openclose：收盘价相对于当日开盘价的跌幅
        # highclose：收盘价相对于当日最高价的跌幅
        # highlow：当日最低价相对于最高价的跌幅（默认方法）
        if self.p.approach == 'closeclose':
            self.pctdown = self.data.close / self.data.close(-1) - 1.0
        elif self.p.approach == 'openclose':
            self.pctdown = self.data.close / self.data.open - 1.0
        elif self.p.approach == 'highclose':
            self.pctdown = self.data.close / self.data.high - 1.0
        elif self.p.approach == 'highlow':
            self.pctdown = self.data.low / self.data.high - 1.0
        
    def next(self):
        if self.position:
            if len(self) == self.barexit:   # 达到预设的退出时间
                self.close()
                if self.p.prdata:
                    print(','.join(str(x) for x in
                                   ['DATA', 'CLOSE',
                                    self.data.datetime.date().isoformat(),
                                    self.data.close[0],
                                    float('NaN')]))
        else:
            if self.pctdown <= self.p.fall: # 下跌幅度超过阈值（-1%）
                self.order_target_percent(target=self.p.target) # 买入100%仓位
                self.barexit = len(self) + self.p.hold  # 设置退出时间

                if self.p.prdata:
                    print(','.join(str(x) for x in
                                   ['DATA', 'OPEN',
                                    self.data.datetime.date().isoformat(),
                                    self.data.close[0],
                                    self.pctdown[0]]))

    def start(self):
        if self.p.prtrade:
            print(','.join(
                ['TRADE', 'Status', 'Date', 'Value', 'PnL', 'Commission']))
        if self.p.prorder:
            print(','.join(
                ['ORDER', 'Type', 'Date', 'Price', 'Size', 'Commission']))
        if self.p.prdata:
            print(','.join(['DATA', 'Action', 'Date', 'Price', 'PctDown']))

    def notify_order(self, order):
        if order.status in [order.Margin, order.Rejected, order.Canceled]:
            print('ORDER FAILED with status:', order.getstatusname())
        elif order.status == order.Completed:
            if self.p.prorder:
                print(','.join(map(str, [
                    'ORDER', 'BUY' * order.isbuy() or 'SELL',
                    self.data.num2date(order.executed.dt).date().isoformat(),
                    order.executed.price,
                    order.executed.size,
                    order.executed.comm,
                ]
                )))

    def notify_trade(self, trade):
        if not self.p.prtrade:
            return

        if trade.isclosed:
            print(','.join(map(str, [
                'TRADE', 'CLOSE',
                self.data.num2date(trade.dtclose).date().isoformat(),
                trade.value,
                trade.pnl,
                trade.commission,
            ]
            )))
        elif trade.justopened:
            print(','.join(map(str, [
                'TRADE', 'OPEN',
                self.data.num2date(trade.dtopen).date().isoformat(),
                trade.value,
                trade.pnl,
                trade.commission,
            ]
            )))


def runstrat(args=None):
    args = parse_args(args)

    cerebro = bt.Cerebro()

    # Data feed kwargs
    kwargs = dict()

    # Parse from/to-date
    dtfmt, tmfmt = '%Y-%m-%d', 'T%H:%M:%S'
    for a, d in ((getattr(args, x), x) for x in ['fromdate', 'todate']):
        kwargs[d] = datetime.datetime.strptime(a, dtfmt + tmfmt * ('T' in a))

    if args.offline:
        # 使用本地CSV文件
        data_path = os.path.join('datas', args.data)
        if not os.path.exists(data_path):
            print(f"错误：找不到数据文件 {data_path}")
            print("可用的数据文件：")
            datas_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'datas')
            if os.path.exists(datas_dir):
                for file in os.listdir(datas_dir):
                    if file.endswith(('.txt', '.csv')):
                        print(f"  - {file}")
            print(f"\n请使用 --data 参数指定一个可用的数据文件，例如：")
            print(f"  python btfd.py --offline --data yhoo-1996-2014.txt")
            sys.exit(1)
        
        YahooData = bt.feeds.YahooFinanceCSVData

    # Data feed - no plot - observer will do the job
        data = YahooData(dataname=data_path, plot=False, **kwargs)
    else:
        # 尝试从Yahoo Finance下载数据
        try:
            YahooData = bt.feeds.YahooFinanceData
            data = YahooData(dataname=args.data, plot=False, **kwargs)
        except Exception as e:
            print(f"从Yahoo Finance下载数据失败: {e}")
            print("建议使用本地数据文件，请尝试以下命令：")
            print(f"  python btfd.py --offline --data yhoo-1996-2014.txt")
            sys.exit(1)
    cerebro.adddata(data)

    # Broker
    cerebro.broker = bt.brokers.BackBroker(**eval('dict(' + args.broker + ')'))

    # Add a commission
    cerebro.broker.setcommission(**eval('dict(' + args.comminfo + ')'))

    # Strategy
    cerebro.addstrategy(St, **eval('dict(' + args.strat + ')'))

    # Add specific observer
    cerebro.addobserver(ValueUnlever, **eval('dict(' + args.valobserver + ')'))

    # Execute
    cerebro.run(**eval('dict(' + args.cerebro + ')'))

    if args.plot:  # Plot if requested to
        cerebro.plot(**eval('dict(' + args.plot + ')'))


def parse_args(pargs=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(' - '.join([
            'BTFD',
            'http://dark-bid.com/BTFD-only-strategy-that-matters.html',
            ('https://www.reddit.com/r/algotrading/comments/5jez2b/'
             'can_anyone_replicate_this_strategy/')]))
        )

    parser.add_argument('--offline', required=False, action='store_true',
                        help='Use offline file with ticker name')

    parser.add_argument('--data', required=False, default='yhoo-1996-2014.txt',
                        metavar='TICKER', help='Yahoo ticker to download or local file name')


    parser.add_argument('--fromdate', required=False, default='1990-01-01',
                        metavar='YYYY-MM-DD[THH:MM:SS]',
                        help='Starting date[time]')

    parser.add_argument('--todate', required=False, default='2016-10-01',
                        metavar='YYYY-MM-DD[THH:MM:SS]',
                        help='Ending date[time]')

    parser.add_argument('--cerebro', required=False, default='stdstats=False',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--broker', required=False,
                        default='cash=100000.0, coc=True',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--valobserver', required=False,
                        default='assetstart=100000.0',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--strat', required=False,
                        default='approach="highlow"',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--comminfo', required=False, default='leverage=2.0',
                        metavar='kwargs', help='kwargs in key=value format')

    parser.add_argument('--plot', required=False, default='',
                        nargs='?', const='volume=False',
                        metavar='kwargs', help='kwargs in key=value format')

    return parser.parse_args(pargs)


if __name__ == '__main__':
    runstrat()

# 策略使用场景
# 适合的市场：
# 波动性较大的市场
# 有明确趋势的市场
# 适合做短线的市场
# 不适合的市场：
# 单边下跌市场
# 低波动性市场
# 需要长期持有的投资