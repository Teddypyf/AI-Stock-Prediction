import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start_date, end_date):
    # 使用yfinance获取股票数据
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    return stock_data

def calculate_moving_average(data, window):
    # 计算移动平均值，当数据不足时取已有数据的平均值填充缺失值
    moving_avg = data['Close'].rolling(window=window, min_periods=1).mean()
    return moving_avg

def export_to_csv(data, filename):
    # 将数据保存到CSV文件中
    data.to_csv(filename)

if __name__ == "__main__":
    # 输入你感兴趣的股票代码
    stock_symbol = input("请输入股票代码：")
    
    # 输入起始日期和结束日期
    start_date = input("请输入起始日期（格式：YYYY-MM-DD）：")
    end_date = input("请输入结束日期（格式：YYYY-MM-DD）：")
    
    # 获取股票数据
    stock_data = get_stock_data(stock_symbol, start_date, end_date)
    
    # 计算7、20和100天移动平均价
    stock_data['MA_7'] = calculate_moving_average(stock_data, 7)
    stock_data['MA_20'] = calculate_moving_average(stock_data, 20)
    stock_data['MA_100'] = calculate_moving_average(stock_data, 100)
    
    # 将数据保存为CSV文件
    filename = "data_with_Date_og_Title.csv"
    export_to_csv(stock_data, filename)
    
    print(f"股票数据已保存到 {filename}")
