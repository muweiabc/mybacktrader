"""
使用 Wind API 获取股票数据并保存到 parquet 文件
"""
import pandas as pd
import os
from datetime import datetime, timedelta

try:
    from WindPy import w
except ImportError:
    print("错误: 未安装 WindPy，请先安装 Wind Python SDK")
    exit(1)


def get_all_a_stocks():
    """获取全部 A 股股票代码"""
    # 获取沪深A股成分股
    result = w.wset("sectorconstituent", "date=" + datetime.now().strftime('%Y-%m-%d') + ";sectorid=a001010100000000")
    if result.ErrorCode != 0:
        print(f"获取股票列表失败: {result.Data}")
        return []
    
    codes = result.Data[1]  # 股票代码在第二列
    print(f"获取到 {len(codes)} 只 A 股股票")
    return codes


def fetch_stock_data(codes, start_date, end_date, fields):
    """
    使用 wsd 获取股票日数据
    
    Args:
        codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        fields: 字段列表
    
    Returns:
        DataFrame
    """
    all_data = []
    total = len(codes)
    
    # 分批获取数据（每批100只）
    batch_size = 100
    for i in range(0, total, batch_size):
        batch_codes = codes[i:i+batch_size]
        print(f"正在获取第 {i+1}-{min(i+batch_size, total)}/{total} 只股票...")
        
        # wsd 获取日数据
        result = w.wsd(
            batch_codes,
            fields,
            start_date,
            end_date,
            "PriceAdj=F"  # 前复权
        )
        
        if result.ErrorCode != 0:
            print(f"获取数据失败: {result.Data}")
            continue
        
        # 解析数据
        dates = result.Times
        field_list = fields.split(',')
        
        for code_idx, code in enumerate(batch_codes):
            for date_idx, date in enumerate(dates):
                row = {'DATE': date, 'CODE': code}
                for field_idx, field in enumerate(field_list):
                    # wsd 返回的数据结构: Data[field_idx][code_idx * len(dates) + date_idx]
                    # 或者: Data[field_idx][date_idx] 如果只有一个股票
                    if len(batch_codes) == 1:
                        value = result.Data[field_idx][date_idx]
                    else:
                        # 多股票时，数据按 (field, code, date) 排列
                        value = result.Data[field_idx][code_idx * len(dates) + date_idx] if len(dates) > 1 else result.Data[field_idx][code_idx]
                    row[field.upper()] = value
                all_data.append(row)
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    return df


def fetch_stock_data_wss(codes, trade_date, fields):
    """
    使用 wss 获取某一天的截面数据
    
    Args:
        codes: 股票代码列表
        trade_date: 交易日期
        fields: 字段列表
    
    Returns:
        DataFrame
    """
    print(f"正在获取 {trade_date} 的数据，共 {len(codes)} 只股票...")
    
    result = w.wss(
        codes,
        fields,
        f"tradeDate={trade_date};priceAdj=F"
    )
    
    if result.ErrorCode != 0:
        print(f"获取数据失败: ErrorCode={result.ErrorCode}")
        return pd.DataFrame()
    
    # 构建 DataFrame
    field_list = [f.strip().upper() for f in fields.split(',')]
    data = {'CODE': result.Codes}
    
    for idx, field in enumerate(field_list):
        data[field] = result.Data[idx]
    
    df = pd.DataFrame(data)
    df['DATE'] = pd.to_datetime(trade_date)
    return df


def get_trading_days(start_date, end_date):
    """获取交易日列表"""
    result = w.tdays(start_date, end_date)
    if result.ErrorCode != 0:
        print(f"获取交易日失败: {result.Data}")
        return []
    return [d.strftime('%Y-%m-%d') for d in result.Data]


def main():
    # 初始化 Wind
    print("正在连接 Wind...")
    w.start()
    
    if not w.isconnected():
        print("Wind 连接失败，请检查 Wind 客户端是否已启动")
        return
    
    print("Wind 连接成功!")
    
    # 设置日期范围（过去1周）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    print(f"数据日期范围: {start_str} 到 {end_str}")
    
    # 获取交易日
    trading_days = get_trading_days(start_str, end_str)
    print(f"交易日: {trading_days}")
    
    if not trading_days:
        print("该时间段内没有交易日")
        return
    
    # 获取股票列表
    codes = get_all_a_stocks()
    if not codes:
        print("未获取到股票列表")
        return
    
    # 定义要获取的字段
    # wss 字段名映射:
    # sec_name: 股票名称
    # open: 开盘价
    # close: 收盘价
    # high: 最高价
    # low: 最低价
    # volume: 成交量
    # amt: 成交额
    # mkt_cap_ard: 总市值
    # mkt_freeshares: 流通股本
    fields = "sec_name,open,close,high,low,volume,amt,mkt_cap_ard,mkt_freeshares"
    
    # 逐个交易日获取数据
    all_data = []
    for trade_date in trading_days:
        df = fetch_stock_data_wss(codes, trade_date, fields)
        if not df.empty:
            all_data.append(df)
            print(f"  {trade_date}: {len(df)} 条数据")
    
    if not all_data:
        print("未获取到任何数据")
        return
    
    # 合并所有数据
    final_df = pd.concat(all_data, ignore_index=True)
    
    # 重命名列
    final_df.rename(columns={
        'SEC_NAME': 'NAME',
        'MKT_CAP_ARD': 'MKT_CAP',
        'MKT_FREESHARES': 'FREE_SHARES'
    }, inplace=True)
    
    # 设置日期为索引
    final_df.set_index('DATE', inplace=True)
    final_df.sort_index(inplace=True)
    
    print(f"\n总数据量: {len(final_df)} 行")
    print(f"列名: {final_df.columns.tolist()}")
    print(f"\n数据预览:")
    print(final_df.head(10))
    
    # 保存到 parquet
    output_path = os.path.expanduser('~/wind_weekly_data.parquet')
    final_df.to_parquet(output_path)
    print(f"\n数据已保存到: {output_path}")
    
    # 关闭 Wind 连接
    w.close()
    print("Wind 连接已关闭")


if __name__ == '__main__':
    main()





