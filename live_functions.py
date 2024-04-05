
import re
import os
import quantstats as qs
import time

import matplotlib.pyplot as plt

import nest_asyncio # install this package to avoid running in loop

import urllib

import pandas_market_calendars as mcal

import pandas_ta as pta
import asyncio
import datetime
qs.extend_pandas()

# import dask.dataframe as dd
# from dask.distributed import Client
from concurrent.futures import ThreadPoolExecutor

#import talib

import pandas as pd
import requests
import urllib.parse

import sys
#from datetime import datetime , timedelta


from strategy_functions import *   # Import the module

from more_functions import*


def initialise_strategy_directory(strategy_name='test' , root_folder=r"/home/aakash_tandon/Python_strategies"):
    
    """
    Function to initialise strategy folder to store trades , dataframes and live daily positions and other stuff
    
    """
    
    
    
    #==== First we initialise the strategy_folder
        
    strategy_folder = os.path.join(root_folder, strategy_name)
    if not os.path.exists(strategy_folder):
        print("\n Making strategy folder as " , strategy_folder )
        os.makedirs(strategy_folder)
    
    
    #=== Now we create symbol wise dataframes ( mainly for debug)
        
    df_folder = os.path.join(strategy_folder, "symbol_wise_dataframe")
    if not os.path.exists(df_folder):
        print("\n Making symbol wise dataframe folder as : " , df_folder )
        os.makedirs(df_folder)
       

    #----- symbol_wise_trades 

    trades_strategy_folder = os.path.join(strategy_folder, "symbol_wise_trades")
    
    if not os.path.exists(trades_strategy_folder):
        
        print("\n Making folder for symbol wise trades of strategy" , trades_strategy_folder)
        os.makedirs(trades_strategy_folder)
    
    position_folder = os.path.join(strategy_folder , "position_folder" )
    
    if not os.path.exists(position_folder):
        print("\n Making position folder as " , position_folder )
        os.makedirs(position_folder)
       
    
    return  df_folder , trades_strategy_folder , position_folder , strategy_folder
    

    
    

def fetch_historical_data(symbol, time_interval=30, bar_count=1000):
    """
    Fetch historical data for a given symbol from the specified URL.
    
    Parameters:
    - symbol (str): The symbol for which to fetch data.
    - time_interval (int): The time interval for the data.
    - bar_count (int): The number of data points to fetch.
    
    Returns:
    - pd.DataFrame: DataFrame containing the historical data or None if data is empty.
    """
    # URL encode the symbol to handle spaces and special characters
    encoded_symbol = urllib.parse.quote(symbol)
    
    # Construct the URL
    pth = f"http://candles.multyfi.com/api/candles/latest?instrument={encoded_symbol}&interval={time_interval}&count={bar_count}"
    
    # Send GET request
    try:
        response = requests.get(pth)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    # Convert response to JSON
    data_json = response.json()
    
    # Check if data is empty
    if not data_json.get('data'):
        print("Data is empty.")
        return None
    
    # Create DataFrame from data
    df = pd.DataFrame(data_json['data'])
    
    # Convert 'Timestamp' to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    return df


#http://65.0.126.220:8002/api/stocks/candles?interval=15&instrument=NSE:ACC&from=2024-01-02&to=2024-01-10



import pandas as pd
import requests
import urllib.parse

def fetch_historical_data_from_date(symbol, time_interval=30, start_date=datetime.date(2023 , 9 , 1 ,) , end_date=datetime.date(2024 , 3 , 1) ):
    """
    Fetch historical data for a given symbol from the specified URL.
    
    Parameters:
    - symbol (str): The symbol for which to fetch data.
    - time_interval (int): The time interval for the data.
    - bar_count (int): The number of data points to fetch.
    
    Returns:
    - pd.DataFrame: DataFrame containing the historical data or None if data is empty.
    """
    # URL encode the symbol to handle spaces and special characters
    encoded_symbol = urllib.parse.quote(symbol)
    
    # Convert dates to strings in the format 'YYYY-MM-DD'
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    
    # Construct the URL with the given parameters
    url = f"http://65.0.126.220:8002/api/stocks/candles?interval={time_interval}&instrument={encoded_symbol}&from={start_date_str}&to={end_date_str}"

    # Send GET request
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    # Convert response to JSON
    data_json = response.json()
    
    # Check if data is empty
    if not data_json.get('data'):
        print("Data is empty.")
        return None
    
    # Create DataFrame from data
    df = pd.DataFrame(data_json['data'])
    
    # Convert 'Timestamp' to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    return df



def fetch_historical_index_from_date(symbol, time_interval=30, start_date=datetime.date(2023 , 9 , 1 ,) , end_date=datetime.date(2024 , 3 , 1) ):
    """
    Fetch historical data for a given symbol from the specified URL.
    
    
    Has symbols: NSE:NIFTY 50 , NSE:NIFTY BANK , "NSE:NIFTY FIN SERVICE", "NSE:NIFTY MID SELECT" 
    
    Parameters:
    - symbol (str): The symbol for which to fetch data.
    - time_interval (int): The time interval for the data.
    - bar_count (int): The number of data points to fetch.
    
    Returns:
    - pd.DataFrame: DataFrame containing the historical data or None if data is empty.
    """
    # URL encode the symbol to handle spaces and special characters
    encoded_symbol = urllib.parse.quote(symbol)
    
    # Convert dates to strings in the format 'YYYY-MM-DD'
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    
    # Construct the URL with the given parameters
    url = f"http://65.0.126.220:8002/api/indexes/candles?interval={time_interval}&instrument={encoded_symbol}&from={start_date_str}&to={end_date_str}"

    # Send GET request
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    # Convert response to JSON
    data_json = response.json()
    
    # Check if data is empty
    if not data_json.get('data'):
        print("Data is empty.")
        return None
    
    # Create DataFrame from data
    df = pd.DataFrame(data_json['data'])
    
    # Convert 'Timestamp' to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    return df




def fetch_live_ltp(symbol):
    
    """
    Fetch live LTP for a given symbol from the specified URL.
    
    Parameters:
    - symbol (str): The symbol for which to fetch data.
    
    """
    
    
    encoded_symbol = urllib.parse.quote(symbol)
    
    url = f"http://65.0.126.220:8002/api/ltp?instrument={encoded_symbol}"

    # Send GET request
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    # Convert response to JSON
    data_json = response.json()
    
    # Check if data is empty
    if not data_json.get('data'):
        print("Data is empty.")
        return None
    
    return data_json.get('data')
    

def fetch_live_candles(symbol , lookback_days=10 , product='stocks' , interval = 30  ):
    
    """
    Fetch live candles for any symbol upto for today
    
    Parameters:
    - symbol (str): The symbol for which to fetch data.
    - lookback_days : the number of days of historical too u want
    - product: 'stocks' for equity , 'indexes' for indices
    - interval : the interval to resample and create candles to
    
    """
    
    
    # URL encode the symbol to handle spaces and special characters
    encoded_symbol = urllib.parse.quote(symbol)
    
    
    #end_date = datetime.date.today()
    end_date = datetime.datetime.today().date()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Convert dates to strings in the format 'YYYY-MM-DD'
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
        
    
    # Construct the URL with the given parameters
    # Construct the URL with the given parameters
    url = f"http://65.0.126.220:8002/api/{product}/candles?interval={interval}&instrument={encoded_symbol}&from={start_date_str}&to={end_date_str}"


    
    # Send GET request
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError if the response status code was unsuccessful
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None
    
    # Convert response to JSON
    data_json = response.json()
    
    # Check if data is empty
    if not data_json.get('data'):
        print("Data is empty.")
        return None
    
    # Create DataFrame from data
    df = pd.DataFrame(data_json['data'])
    
    # Convert 'Timestamp' to datetime and set as index
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    if product == 'indexes':
        
        df = df.rename(columns={'Open': 'indOpen','High': 'indHigh', 'Low': 'indLow', 'Close': 'indClose', 'Volume':'indVolume', 'Symbol': 'indSymbol'})
    
    
    return df
   



#=== Code to group the symbol dataframe by positions.. ie each group is a trade
#--- helps in finding all the trades( entry  , exit)


def create_tradelog(df , Symbol=None ):
    
    
    """
    This function groups the positions into a dataframe for each symbol and returns the trade_log.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with Close and position , order and bar-by-bar returns
                             

    Returns:
    pandas.dataframe: A dataframe with each trade in format 
                      : Symbol	Side	EntryPrice	Stop Loss	Time of Entry	Time of Exit	Exit Reason	inposition	Exit Price

    """
    
    
     # Check if DataFrame is empty
    if df.empty:
        print("DataFrame is empty. No data to process.")
        return pd.DataFrame()  # Return an empty DataFrame

    
    print("\nSymbol passed to trade log: " , Symbol , "\n\n")
    
    
    #df.set_index('entry_time', inplace=True)

    
    
    # Assuming 'df' is your DataFrame with the columns 'A_Close', 'A_position', and 'A_ret'.

    # Detect changes in 'A_position'
    if Symbol+'_position' in df.columns:
        df['position_change'] = df[Symbol+'_position'].diff()
    
    else:
        print("\n Position column ( 1 , 0 , -1) missing in dataframe please check")
        
        
    # Assign a group number to each sequence of 1's
    df['group'] = (df['position_change'] == 1).cumsum()

    # Filter out the rows where 'A_position' is not 1
    df_positions_1 = df[df[Symbol+'_position'] == 1]
    
    if df_positions_1.empty:
        print(f"No trade entries found for symbol {Symbol}.")
        return pd.DataFrame()
    
    # Now you can group by the 'group' column
    grouped = df_positions_1.groupby('group')

    # If you want to see the groups

    # Initialize a list to store your results
    trade_returns = []

    for name, group in grouped:
        #print(f"Group {name}:")
        #print(group)

        start_price = group[Symbol+'_Close'].iloc[0]
        end_price = group[Symbol+'_Close'].iloc[-1]
        trade_return = end_price / start_price - 1
        start_time = group.index[0]
        end_time = group.index[-1]
        exit_order = group[Symbol+'_order'].iloc[-1]
        # Assume the position is consistent within the group and take the first one
        position = group[Symbol+'_position'].iloc[0]
        side = 'long' if position == 1 else 'Short' if position == -1 else 'flat'

        
        
        trade_returns.append({
            'Symbol': Symbol,
            'Side': side,
            'entry_time': start_time,
            'entry_price': start_price,
            'exit_time': end_time, 
            'exit_price': end_price,
            'order' : exit_order
        })

        
    # Convert the list of dictionaries to a DataFrame
    trade_returns_df = pd.DataFrame(trade_returns)
    #print("\nThe trade log is" , trade_returns_df)
    if not trade_returns_df.empty:
        trade_returns_df.set_index('entry_time', inplace=True)
    return trade_returns_df
    


def get_fno_stocks():
    import pandas as pd
    kite_csv = pd.read_csv('https://api.kite.trade/instruments')
    #Filter by  instrument_type,segment,exchange
    filtered_kite_csv = kite_csv[(kite_csv['instrument_type'] == 'FUT') & (kite_csv['exchange'] == 'NFO')]
    
    filtered_kite_csv = filtered_kite_csv['name'].unique()
    
    return filtered_kite_csv



def process_expiry_dates_from_file(df):
    
    
    """
    Reads the expiry dataframe and converts date to correct format as required by future functions
    """
    
    if 'Date' in df.columns:
        
        df['Date'] = pd.to_datetime(df['Date'] , format="%d%b%y").dt.date
    else:
        
        print("\n No expiry dates found in" , df)
        return None 
    
    df = df.sort_values(by='Date')
    
    # Selecting only the 'dates' column if needed
    
    df = df[['Date']]
    
    return df    
        


    
def fetch_index_historial(index_symbol='NSE:NIFTY 50' , lookback_days=40 , bar_interval=30):
    
    
    """
    Fetches the index data from previous day upto a lookback period defined by lookback_days for a bar interval(candle timeframe)
    """
    
    #=== Note this function doesnt fetch today data but till yesterday
        
    end_date = datetime.datetime.today().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days)
    
        
    ind_df = fetch_historical_index_from_date(symbol=index_symbol , time_interval=bar_interval , start_date=start_date , end_date=end_date)

    if ind_df is not None:
        print(ind_df.head())
    else:
        print("No data returned.")
        
    #ind_df = process_index(ind_df )
    
    ind_df = ind_df.rename(columns={'Open': 'indOpen','High': 'indHigh', 'Low': 'indLow', 'Close': 'indClose', 'Volume':'indVolume', 'Symbol': 'indSymbol'})
    
    return ind_df    
        
    
    



def get_next_run_time(start_hour, start_minute, interval):
    
    
    """
    Calculate the next scheduled run time for a task based on a given start hour, start minute, and interval.

    Parameters:
    - start_hour (int): The hour at which the task should first run for the day, using a 24-hour clock format.
    - start_minute (int): The minute of the hour at which the task should first run.
    - interval (int): The frequency, in minutes, at which the task should repeat after the initial start time.

    Returns:
    - next_run (datetime): The next scheduled run time for the task.

    This function uses the pytz library to work with the timezone for India ('Asia/Kolkata').
    It calculates the next run time ensuring it's always in the future relative to when the function is called.
    """
    
    
    import pytz
    india_tz = pytz.timezone('Asia/Kolkata')
    
    # Get the current time in the specified timezone
    now = datetime.datetime.now(india_tz)

    #print(now)
    
    # Create the start time for today with the specified hour and minute
    start_time = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
 
 
    # If now is before the initial start time, schedule the first run at start_time + interval
    if now < start_time:
        next_run = start_time + datetime.timedelta(minutes=interval)
        print("\n First run at " , next_run)
        
    else:
        # Calculate how much time has passed since the start_time today
        time_passed_since_start = now - start_time

        # Calculate the total minutes since the start_time
        total_minutes_since_start = time_passed_since_start.total_seconds() / 60

        # Find how many intervals have passed since start_time
        intervals_since_start = total_minutes_since_start // interval

        # Calculate the next run time
        next_run = start_time + datetime.timedelta(minutes=(intervals_since_start + 1) * interval)
        #print("\n next run outside loop: " , next_run)
        # Ensure next run time is in the future
        if next_run <= now:
            next_run += datetime.timedelta(minutes=interval)
            #print("\n next_run is " , next_run)
            
    return next_run

