

import pandas_market_calendars as mcal

def get_data_timeframe(df):
    
    
    if isinstance(df.index, pd.DatetimeIndex):
        
        df.sort_index(inplace=True)
        
        time_diffs = np.diff(df.index)
        min_time_diff = min(diff for diff in time_diffs if diff != np.timedelta64(0, 'ns'))
        time_frame = int(min_time_diff.total_seconds() / 60)
        
        #print("\n Current timeframe is: " , time_frame)
            
        return time_frame
    
    else:
        print("\n Index is not DatetimeIndex ... convert index to datetime for further processing ")

        
def resample_data(df , highcol = 'High' , lowcol = 'Low' , closecol = 'Close' ,opencol = 'Open'  , volcol = 'Volume' , timeframe = '30min'):
    
    """
    Gets the resampled upsample time-series for a OHLCV dataframe
    
    
    """
    orig_tf = get_data_timeframe(df)
    
    # Parse timeframe string to check if it's in minutes
    if 'T' in timeframe or 'min' in timeframe:
        # Extract numeric value from timeframe string
        timeframe_numeric = int(''.join(filter(str.isdigit, timeframe)))
        
        # Perform comparison if orig_tf is not None
        if orig_tf is not None and timeframe_numeric <= orig_tf:
            print("\nInvalid timeframe resample request.")
            return
    elif orig_tf is None:
        print("\n Original timeframe could not be determined.")
        return
        
    
    rdf = pd.DataFrame()

    rdf['High'] = df[highcol].resample(timeframe , origin='start').max().dropna()
    rdf['Low'] = df[lowcol].resample(timeframe , origin='start' ).min().dropna()
    rdf['Close'] = df[closecol].resample(timeframe, origin='start').last().dropna()
    rdf['Open'] = df[opencol].resample(timeframe , origin='start').first().dropna()
    if volcol in df.columns:
        rdf['Volume'] = df[volcol].resample(timeframe , origin='start').sum().dropna()
    
    return rdf    

import numpy as np
import pandas as pd

def find_slope_and_prediction(df, column='Close', window=21):
    
    def calc_slope_intercept(y):
        if len(y) < window:
            return np.nan, np.nan
        x = np.arange(len(y))
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_x2 = np.sum(x**2)
        sum_xy = np.sum(x*y)
        N = len(x)

        # Calculate slope and intercept
        slope = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x**2)
        intercept = (sum_y - slope * sum_x) / N
        return slope, intercept

    # Apply the function over a rolling window and extract results
    results = df[column].rolling(window=window).apply(
        lambda y: calc_slope_intercept(y)[0], raw=True
    )

    slopes = results
    intercepts = df[column].rolling(window=window).apply(
        lambda y: calc_slope_intercept(y)[1], raw=True
    )

    # Calculate predicted value for the last point in each window
    predictions = slopes * (window - 1) + intercepts

    return slopes, predictions

# Example usage
# slopes, predictions = find_slope_and_prediction(df, 'Close', 21)



def get_x_day_range(df  , n , high_col = 'High' , low_col='Low' ):
    
    
    xdh = get_x_day_high(df , n=n , column=high_col)
    xdl = get_x_day_low(df , n=n , column=low_col)
    
    x_day_range = xdh/xdl-1
    
    return x_day_range

def rsi_timeframe(df , column='Close' , period=14,  highcol='High' , lowcol='Low' , closecol='Close' , opencol='Open' , volcol='Volume' , timeframe='1D'):
    import pandas_ta as pta
    
    resamp = resample_data(df=df , highcol=highcol , lowcol=lowcol , closecol=closecol , opencol=opencol , volcol=volcol , timeframe=timeframe)
    
    rsi_htf = pta.rsi(resamp[column].shift(1) , length=period) 
    
    rsi_htf = rsi_htf.reindex(df.index, method='ffill')
    
    
    return rsi_htf


def my_ATR(df, length=14 , highcol='High' , lowcol='Low' , closecol='Close'):
    
    
     
    """
    Finds the ATR using the basic Simple moving average 
    
    Prerequisites
    
    Parameters:
    df =  (pandas.Dataframe): A pandas dataframe with OHLC and datetime as index
    length = (number): period for which to calculate ATR 
    
    Returns:
    ATR values
    
    """
    
    
    
    if df is None or length <= 0 or len(df) < length:
        print("Dataframe is very small or too large value of input period\n")
        return []
    
    # Calculate True Range
    high = df[highcol].values
    low = df[lowcol].values
    close = df[closecol].shift(1).values
    tr = np.maximum(high - low, high - close, close - low)
    
    # Calculate ATR
    atr = np.empty_like(tr)
    atr[:length] = np.nan
    atr[length-1] = tr[1:length].mean()
    for i in range(length, len(df)):
        atr[i] = (atr[i-1]*(length-1) + tr[i]) / float(length)

    return atr


def ATR(highcol='High' , lowcol = 'Low' , closecol='Close', length=21 , mamode='RMA'  , talib=True  , percent=False):
    
    import pandas_ta as pta
    
    atrv = pta.atr(highcol , lowcol , closecol , length=length , mamode=mamode  , talib=talib  , percent=percent )
    
    return atrv
    

    #=== to-do
def US_stock_monthly_expiry(df , expiry=0):
    
    from datetime import date, timedelta
    
    """
    Finds the nearest monthly expiry for all dates between start date and end_date 
    
    
    """
    start_date = df.index.date.min()
    end_date = df.index.date.max()+ timedelta(days=360)
    
    us_cal = mcal.get_calendar('CBOE_Equity_Options')
    
    valid_US_days = us_cal.valid_days(start_date=start_date , end_date=end_date).date
    
    
     # Find all Thursdays
    all_days = pd.date_range(start_date, end_date, freq='B')
    fridays = all_days[all_days.to_series().dt.weekday == 4] 
    #print("all fridays are: \n " ,fridays )
    
    
    # Get the last Thursday of each month
    third_friday = fridays.to_series().groupby([fridays.year, fridays.month]).nth(2)  # nth is 0-based, so 2 means the third item
    last_fridays = fridays.to_series().groupby([fridays.year, fridays.month]).last()
    
    
    
    # If the last Thursday is not a valid day, find the previous valid day
    #last_thursdays = last_thursdays.apply(lambda d: valid_nse_days[valid_nse_days <= pd.to_datetime(d).date()].max() if pd.to_datetime(d).date() not in valid_nse_days else d)
    
    third_fridays = third_friday.apply(lambda d: valid_US_days[valid_US_days <= pd.to_datetime(d).date()].max() if pd.to_datetime(d).date() not in valid_US_days else d)
    

        # Remove the time part from 'last_thursdays'
    # Convert back to Timestamp and remove the time part from 'last_thursdays'
    #last_thursdays_date = pd.to_datetime(last_thursdays).dt.date
    
    third_friday_date = pd.to_datetime(third_fridays).dt.date
    

    # Remove the time part from the DataFrame's index
    df_date = df.index.date

    # Find the nearest expiry date for each date in the DataFrame
    #df['Expiry_Date'] = [last_thursdays_date[last_thursdays_date >= date].min() for date in df_date]
    
    
    
    # Find the nearest expiry date for each date in the DataFrame
    if expiry>=0: 
        df['Expiry_Date'] = [sorted(third_friday_date[third_friday_date >= date])[expiry] if len(third_friday_date[third_friday_date >= date]) > expiry else np.nan for date in df_date]
    elif expiry<0:
        
        df['Expiry_Date'] = [sorted(third_friday_date[third_friday_date < date])[expiry]  if len(third_friday_date[third_friday_date < date]) > abs(expiry) - 1 else np.nan for date in df_date]
    
        
    return df





def supertrend( highcol='High' , lowcol= 'Low' , closecol = 'Close' , length=2 , multiplier=2.5):
    import pandas_ta as pta

    super_df  = pta.supertrend(highcol , lowcol , closecol , length=length , multiplier=multiplier)
    # Check if the DataFrame is empty or all values are NaN
    if super_df.empty or super_df.isnull().all().all():
        print("The DataFrame is either empty or contains only NaN values.")
        return None  # handle this as appropriate for your use case
    
    # If DataFrame is non-empty and not all NaN, return the first column
    return super_df.iloc[:, 0]


import pandas as pd
from datetime import timedelta


def get_x_day_low(df, n , column='Low'):
    """
    Get the previous low of the last n known dates for each row in the DataFrame.

    n>=1
    """
    df = df.copy()
    df['Date'] = df.index.normalize()
    unique_dates = df['Date'].unique()

    # Create a dictionary to store the x_day_low for each date
    x_day_low_dict = {}

    for i, current_date in enumerate(unique_dates):
        # Find the start date index such that the difference in known dates is at least n days
        start_date_index = i - n
        if start_date_index < 0:
            start_date_index = 0

        start_date = unique_dates[start_date_index]
        mask = (df['Date'] < current_date) & (df['Date'] >= start_date)
        x_day_low_dict[current_date] = df.loc[mask, column].min()

    # Apply the x_day_low values from the dictionary to the DataFrame
    x_day_low = df['Date'].map(x_day_low_dict)

    # Remove the temporary Date column
    #df.drop(columns='Date', inplace=True)

    return x_day_low


def get_x_day_high(df, n , column='high'):
    """
    Get the previous low of the last n known dates for each row in the DataFrame.

    n>=1
    """
    df = df.copy()
    df['Date'] = df.index.normalize()
    unique_dates = df['Date'].unique()

    # Create a dictionary to store the x_day_low for each date
    x_day_low_dict = {}

    for i, current_date in enumerate(unique_dates):
        # Find the start date index such that the difference in known dates is at least n days
        start_date_index = i - n
        if start_date_index < 0:
            start_date_index = 0

        start_date = unique_dates[start_date_index]
        mask = (df['Date'] < current_date) & (df['Date'] >= start_date)
        x_day_low_dict[current_date] = df.loc[mask, column].max()

    # Apply the x_day_low values from the dictionary to the DataFrame
    x_day_high = df['Date'].map(x_day_low_dict)

    # Remove the temporary Date column
    #df.drop(columns='Date', inplace=True)

    return x_day_high



def get_x_day_range(df  , n , high_col = 'High' , low_col='Low' ):
    
    
    xdh = get_x_day_high(df , n=n , column=high_col)
    xdl = get_x_day_low(df , n=n , column=low_col)
    
    x_day_range = xdh/xdl-1
    
    return x_day_range



def z_score_price(df, column, window_size=60):
    
    """
    This function calculates the rolling Z-Score for the specified column in the given DataFrame.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    column (str): Column for which to calculate the rolling Z-Score.
    window_size (int): Size of the rolling window.

    Returns:
    pandas.Series: A Series representing the rolling Z-Score of the specified column.
    """
    
     # Calculate the rolling mean and standard deviation
    rolling_mean = df[column].rolling(window=window_size , min_periods=3).mean()
    rolling_std = df[column].rolling(window=window_size, min_periods=3).std()

    # Calculate the rolling Z-Score and handle division by zero
    rolling_z_score = (df[column] - rolling_mean) / rolling_std

    # Replace infinities with NaNs if any
    rolling_z_score.replace([np.inf, -np.inf], np.nan, inplace=True)

    return rolling_z_score
    
    


def x_min_low(df , column , start='09:15' , end='10:15'):
    """
    Get the previous x min lowest low for value of your choice for each row in the DataFrame.
    example you want first hour low:
        x_min_low(df , 'Low' , start='09:15' , end='10:15'):
    
    """
   
    if column in df.columns:
    
        xm_low = df.groupby(df.index.date)[column].transform(lambda x: x.between_time(start, end).expanding().min()).ffill()
        return xm_low

def x_min_high(df , column , start='09:15' , end='10:15' ):
    """
    Get the previous x min high high for value of your choice for each row in the DataFrame.
    example you want first hour highest high:
        x_min_low(df , 'High' , start='09:15' , end='10:15'):
    
    """
    
    if column in df.columns:
    
        xm_high = df.groupby(df.index.date)[column].transform(lambda x: x.between_time(start, end).expanding().max()).ffill()
        return xm_high
        
        
 
        
def x_min_cum_vol(df ,column , start='09:15' , end='10:15' ):
    
    
    
    if column in df.columns:
    
        xm_cvol = df.groupby(df.index.date)[column].transform(lambda x: x.between_time(start, end).expanding().sum()).ffill()
        return xm_cvol
 

def bars_since_condition(df, column):
    # Calculate the cumulative sum which resets at each '1' to create group identifiers
    group_ids = (df[column] == 1).cumsum()
    
    # Use 'cumcount' to count the number of occurrences since the last reset, adding 1 to start from 1
    bars_since = df.groupby(group_ids).cumcount() 
    
    # Wherever the condition is true, reset 'bars_since' to 1
    return bars_since.where(df[column] == 0, 1)


#== x-min VWAP

#--- 30 min ka chaiye




def find_peak(df , column='High' , threshold=4 , distance=3):
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(df[column] , threshold=threshold , distance=distance )
    
        # Extracting peak values
    peak_values = df[column].iloc[peaks]
    #print(peak_values)

    return peak_values


def find_trough(df , column='Low' , threshold=4 , distance=3):
    from scipy.signal import find_peaks
    

    # Invert the data to turn troughs into peaks
    inverted_data = -df[column]

    # Find peaks in the inverted data, which correspond to troughs in the original data
    troughs, _ = find_peaks(inverted_data  , threshold=threshold , distance=distance )

    # Extracting trough values from the original data
    trough_values = df[column].iloc[troughs]
    
    
    return trough_values


def find_peak_multi_tf(df , column='High', highcol='High' , lowcol='Low' , closecol='Close' , opencol='Open' , volcol='Volume' , threshold=4 , distance=3 , timeframe='1D'):
    
    #=== resample the data to the timeframe 
    resamp = resample_data(df=df , highcol=highcol , lowcol=lowcol , closecol=closecol , opencol=opencol , volcol=volcol , timeframe=timeframe)
    
    #=== find the peak series and reindex to the resampled dataframe
    resamp = resamp.shift(1)
    
    peak_values = find_peak(resamp , column=column , threshold=threshold , distance=distance)
    
    peakss = peak_values.reindex(df.index, method='ffill')
    
    return peakss
    
def find_trough_multi_tf(df , column='High', highcol='High' , lowcol='Low' , closecol='Close' , opencol='Open' , volcol='Volume' , threshold=4 , distance=3 , timeframe='1D'):     
    
    
     #=== resample the data to the timeframe 
    resamp = resample_data(df=df , highcol=highcol , lowcol=lowcol , closecol=closecol , opencol=opencol , volcol=volcol , timeframe=timeframe)
    
    print(resamp)
    resamp = resamp.shift(1)
    
    trough_values = find_trough(resamp , column=column , threshold=threshold , distance=distance)
    
    troughs = trough_values.reindex(df.index , method='ffill')
    
    return troughs




def daily_ROC(df , column='Close' , ROC_period=3 , z_score_period=20 , highcol='High' , lowcol='Low' , closecol='Close' , opencol='Open'  , timeframe='1D' , volcol='Volume'):
    
    resamp = resample_data(df=df , highcol=highcol , lowcol=lowcol , closecol=closecol , opencol=opencol , volcol=volcol , timeframe=timeframe)
    resamp = resamp.shift(1)
    
    #print(resamp)
        
    
    resamp["ROC"] = (resamp[column] - resamp['Open'].shift(ROC_period)) / (resamp['Open'].shift(ROC_period)) * 100
    resamp["ROC_z_score"] = z_score_price(df=resamp , column='ROC' , window_size=z_score_period)
    
    resamp = resamp.reindex(df.index , method='ffill')
        
    return resamp['ROC_z_score']
    



def calculate_rolling_beta_from_intraday(df, window , stock_col = 'Close' , index_col = 'indClose'):
    # Calculate daily returns
    
    
    
    df_copy = df.copy(deep=True)
    
    
    dfd = resample_data(df , highcol = 'High' , lowcol = 'Low' , closecol = 'Close' ,opencol = 'Open'  , volcol = 'Volume' , timeframe = '1D')
    #df[closecol].resample(timeframe, origin='start').last().dropna()
    #print("\n Initial dataframe is " , df)
    dfd['index_dclose'] = df[index_col].resample('D', origin='start').last().dropna()
    
    #print(dfd)
    
    
    dfd['Stock_Returns'] = dfd[stock_col].pct_change()
    dfd['Index_Returns'] = dfd['index_dclose'].pct_change()
    
    # Calculate rolling covariance of stock returns with index returns
    rolling_cov = dfd['Stock_Returns'].rolling(window=window).cov(dfd['Index_Returns'])
    
    # Calculate rolling variance of index returns
    rolling_var = dfd['Index_Returns'].rolling(window=window).var()
    
    # Rolling beta is the rolling covariance divided by the rolling variance
    rolling_beta = rolling_cov / rolling_var
    
    rolling_beta = rolling_beta.reindex(df.index, method='ffill')
    
    return rolling_beta



#===========================================================

def calculate_rolling_beta(df, window , stock_col = 'Close' , index_col = 'Close_index'):
    
    # Calculate daily returns
    
    #--- but u need the daily 
    
    
    
    df_copy = df.copy(deep=True)
    
    df['Stock_Returns'] = df[stock_col].pct_change()
    df['Index_Returns'] = df[index_col].pct_change()
    
    # Calculate rolling covariance of stock returns with index returns
    rolling_cov = df['Stock_Returns'].rolling(window=window).cov(df['Index_Returns'])
    
    # Calculate rolling variance of index returns
    rolling_var = df['Index_Returns'].rolling(window=window).var()
    
    # Rolling beta is the rolling covariance divided by the rolling variance
    rolling_beta = rolling_cov / rolling_var
    
    
    
    
    return rolling_beta
    









