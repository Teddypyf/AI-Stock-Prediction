import yfinance as yf
import pandas as pd

def get_stock_info(symbol, date):
    # 获取给定日期前100天的数据
    start_date = pd.to_datetime(date) - pd.DateOffset(days=100)
    end_date = date
    stock_info = yf.download(symbol, start=start_date, end=end_date)
    return stock_info

def calculate_moving_average(data, window):
    # 计算移动平均值
    moving_avg = data['Close'].rolling(window=window, min_periods=1).mean()
    return moving_avg

if __name__ == "__main__":
    # 输入你感兴趣的股票代码
    stock_symbol = input("请输入股票代码：")
    
    # 输入感兴趣的日期
    date = input("请输入日期（格式：YYYY-MM-DD）：")
    
    # 获取给定日期前100天的股票信息
    stock_info = get_stock_info(stock_symbol, date)
    
    # 计算当日的7、20和100天移动平均价
    stock_info['MA_7'] = calculate_moving_average(stock_info, 7)
    stock_info['MA_20'] = calculate_moving_average(stock_info, 20)
    stock_info['MA_100'] = calculate_moving_average(stock_info, 100)
    
    # 显示股票信息
    print("股票代码:", stock_symbol)
    print("日期:", date)
    print("开盘价:", stock_info['Open'][0])
    print("最高价:", stock_info['High'][0])
    print("最低价:", stock_info['Low'][0])
    print("收盘价:", stock_info['Close'][0])
    print("调整后收盘价:", stock_info['Adj Close'][0])
    print("成交量:", stock_info['Volume'][0])
    print("当日7日平均价:", stock_info['MA_7'][0])
    print("当日20日平均价:", stock_info['MA_20'][0])
    print("当日100日平均价:", stock_info['MA_100'][0])
