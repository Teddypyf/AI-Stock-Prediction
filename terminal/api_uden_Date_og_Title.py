import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start_date, end_date):
    # Get stock data using yfinance
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    
    return stock_data

def calculate_moving_average(data, window):
    # Calculate the moving average, and when there is insufficient data, take the average of the existing data to fill in the missing values.
    moving_avg = data['Close'].rolling(window=window, min_periods=1).mean()
    return moving_avg

def move_column_last(dataframe, column_name):
    # the function to move the adjusted closing price column to the last column
    cols = list(dataframe.columns)
    cols.remove(column_name)
    cols.append(column_name)
    return dataframe.reindex(columns=cols)

def export_to_csv(data, filename):
    # Save data to CSV file without dates and column headers
    data.to_csv(filename, index=False, header=False)

if __name__ == "__main__":
    # enter the stock code
    stock_symbol = input("Please enter the stock code:")
    
    # enter the starting and end date
    start_date = input("Please enter the starting date (YYYY-MM-DD):")
    end_date = input("Please enter the end date (YYYY-MM-DD):")
    
    # Get stock data
    stock_data = get_stock_data(stock_symbol, start_date, end_date)
    
    # Calculate 7, 20 and 100 day moving average prices
    stock_data['MA_7'] = calculate_moving_average(stock_data, 7)
    stock_data['MA_20'] = calculate_moving_average(stock_data, 20)
    stock_data['MA_100'] = calculate_moving_average(stock_data, 100)
    
    # Move Adjusted Close to last column
    stock_data = move_column_last(stock_data, 'Adj Close')
    
    # Save data as CSV file
    filename = "data.csv"
    export_to_csv(stock_data, filename)
    
    print(f"Stock data has been saved to {filename}")
