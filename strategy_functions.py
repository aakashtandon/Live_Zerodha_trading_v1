#=================================
import pandas as pd
import numpy as np
from datetime import timedelta
from config_read import config_object
import pandas_ta as ta



def fetch_VIX(vix_file_path):
    
    df = pd.read_csv(vix_file_path)
    
    if 'Date' in df.columns:
        df.Date = pd.to_datetime(df['Date'] , format="%Y-%m-%d")
        df.set_index('Date' , inplace=True)
    return df


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
    


def supertrend( highcol='High' , lowcol= 'Low' , closecol = 'Close' , length=2 , multiplier=2.5):
    import pandas_ta as pta

    super_df  = pta.supertrend(highcol , lowcol , closecol , length=length , multiplier=multiplier)
    # Check if the DataFrame is empty or all values are NaN
    if super_df.empty or super_df.isnull().all().all():
        print("The DataFrame is either empty or contains only NaN values.")
        return None  # handle this as appropriate for your use case
    
    # If DataFrame is non-empty and not all NaN, return the first column
    return super_df.iloc[:, 0]
    
    

# pta.supertrend(rdf['High'] , rdf['Low'] , rdf['Close'] , length=14 , multiplier=2)

def ATR(highcol='High' , lowcol = 'Low' , closecol='Close', length=21 , mamode='RMA'  , talib=True  , percent=False):
    
    import pandas_ta as pta
    
    atrv = pta.atr(highcol , lowcol , closecol , length=length , mamode=mamode  , talib=talib  , percent=percent )
    
    return atrv



def daily_ATR(df , length=14  , highcol='high' , lowcol = 'low' , closecol='close'  ,  mamode='RMA'  , talib=True  , percent=False):


    import pandas as pd
    import pandas_ta as pta

    # Assuming 'data' is your DataFrame with daily data
    # Replace 'YourHighColumn', 'YourLowColumn', and 'YourCloseColumn' with actual column names from your data
    daily_data = df.resample('D').agg({
        highcol: 'max',
        lowcol: 'min',
        closecol: 'last'
    })

    daily_data.dropna(how='all' , inplace=True)
    #print( "\n resapleed data is \n " , daily_data)
    # Calculate daily ATR for the last 3 days
    last_3_days_atr = pta.atr(daily_data[highcol], daily_data[lowcol], daily_data[closecol], length=length ,  mamode=mamode  , talib=talib  , percent=percent)
    #print(last_3_days_atr)
    
    rolling_mean = last_3_days_atr.rolling(window=40 , min_periods=3).mean()
    rolling_std = last_3_days_atr.rolling(window=40, min_periods=3).std()

    # Calculate the rolling Z-Score and handle division by zero
    rolling_z_score = (last_3_days_atr - rolling_mean) / rolling_std

    # Replace infinities with NaNs if any
    rolling_z_score.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    
    
    last_3_days_atr = last_3_days_atr.reindex(df.index, method='ffill')
    rolling_z_score = rolling_z_score.reindex(df.index, method='ffill')
        
    
    print( "\n \n Z_score", rolling_z_score)
    
    # Print or use the ATR values for the last 3 days
    #last_3_days_atr.plot()
    return rolling_z_score
        

def get_data_timeframe(df):
    
    
    if isinstance(df.index, pd.DatetimeIndex):
        
        df.sort_index(inplace=True)
        
        time_diffs = np.diff(df.index)
        min_time_diff = min(diff for diff in time_diffs if diff != np.timedelta64(0, 'ns'))
        time_frame = min_time_diff.astype('timedelta64[m]').astype(int)
        
        
        return time_frame
    
    else:
        print("\n Index is not DatetimeIndex ... convert index to datetime for further processing ")
    


def add_time_to_column_or_index(df, column=None, specific_time="09:15"):
    
    """
    Adds a specific time to the date in a DataFrame column or index if the time component is missing.

    :param - df: DataFrame to which the operation will be applied.
    :param - column: The column name to which the operation will be applied. If None, applies to the index.
    :param - specific_time: The specific time to add if the time component is missing. Default is "09:15".
    :return: DataFrame with updated times.
    """

    def add_specific_time(timestamp):
        if timestamp.time() == pd.Timestamp("00:00").time():
            return pd.Timestamp(f"{timestamp.date()} {specific_time}")
        else:
            return timestamp

    if column:
        # Apply to a specific column
        corrected_ts = pd.to_datetime(df[column]).map(add_specific_time)
    else:
        # Apply to the index
        corrected_ts = pd.to_datetime(df.index).map(add_specific_time)

    return corrected_ts




def intraday_high(df, high_col, agg_func):
    if agg_func not in ['max']:
        raise ValueError("agg_func must be - 'max'")

    intraday_high_col = f'intraday_{high_col}_{agg_func}'
    
    if agg_func == 'max':
       int_high =  df.groupby(df.index.date)[high_col].transform(lambda x: x.expanding().max())
    else:
       int_high = None
    return int_high



def intraday_low(df, low_col, agg_func):
    if agg_func not in ['min']:
        raise ValueError("agg_func must be - 'min'")

    intraday_low_col = f'intraday_{low_col}_{agg_func}'
    
    if agg_func == 'min':
        int_low = df.groupby(df.index.date)[low_col].transform(lambda x: x.expanding().min())
    else:
        int_low = None
    return int_low



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


def get_x_day_close_vectorized(df, n, column='Close'):
    """
    Efficiently get the closing price of the last n calendar days for each row in the DataFrame
    using vectorized operations for improved performance.

    Parameters:
    - df: DataFrame with a DateTimeIndex.
    - n: Integer, the number of calendar days to look back for the last known close price.
    - column: String, the name of the column from which to get the close price.
    
    Returns:
    - A Series containing the last known close price from the last n calendar days for each row in the DataFrame.
    """
    # Normalize the DateTimeIndex to remove time and keep only dates
    
    normalized_dates = df.index.normalize()
    
    # Find the unique dates to establish the range of trading days
    unique_dates = pd.Series(normalized_dates.unique()).sort_values()
    
    # Map each date in the DataFrame to its 'n days ago' equivalent
    n_days_ago_mapping = {date: unique_dates[unique_dates.searchsorted(date, side='left') - n] 
                          for date in unique_dates[n:]}
    
    # Create a mapping from each date to the last close of its 'n days ago' date
    last_close_mapping = df[column].groupby(normalized_dates).last().reindex(unique_dates).fillna(method='ffill')
    last_close_mapping = {date: last_close_mapping[n_days_ago] for date, n_days_ago in n_days_ago_mapping.items()}
    
    # Apply the mapping to get the 'n days ago' close price for each date in the DataFrame
    x_day_close = normalized_dates.map(last_close_mapping)
    
    return x_day_close




def get_data_timeframe(df):
    
    
    if isinstance(df.index, pd.DatetimeIndex):
        
        df.sort_index(inplace=True)
        
        time_diffs = np.diff(df.index)
        min_time_diff = min(diff for diff in time_diffs if diff != np.timedelta64(0, 'ns'))
        time_frame = int(min_time_diff.total_seconds() / 60)
        
        
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


    
def rsi_timeframe(df , column='Close' , period=14,  highcol='High' , lowcol='Low' , closecol='Close' , opencol='Open' , volcol='Volume' , timeframe='1D'):
    import pandas_ta as pta
    
    resamp = resample_data(df=df , highcol=highcol , lowcol=lowcol , closecol=closecol , opencol=opencol , volcol=volcol , timeframe=timeframe)
    
    rsi_htf = pta.rsi(resamp[column].shift(1) , length=period) 
    
    rsi_htf = rsi_htf.reindex(df.index, method='ffill')
    
    
    return rsi_htf



def daily_percentile(df, column='close' , window=10 , output_col_name='daily-percentile'):
    
    """
    Get the rolling Daily percentile of a series from intraday data 

    n>=1
    """
    
    
    if column not in df.columns:
        raise ValueError("Column not found in df")
    
    df_daily = df.resample('D')[column].last().dropna()

    # Define the window size
    window_size = window
    df_daily = df_daily.to_frame()
    # Calculate the rolling percentile rank
    
    df_daily[output_col_name] = df_daily[column].shift().rolling(window_size).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

    df_daily = df_daily.reindex(df.index, method='ffill')
    df = pd.concat([df, df_daily[output_col_name]], axis=1)
    return df




    
def pivot_points_classic(df , close_col = 'Close' , low_col = 'Low' , high_col='High'):
    Prev_close = df[close_col].groupby(df.index.date).last().shift().reindex(df.index.date).values
    Prev_High = df[high_col].groupby(df.index.date).max().shift().reindex(df.index.date).values
    Prev_low = df[low_col].groupby(df.index.date).min().shift().reindex(df.index.date).values
    
    Pivot = (Prev_High + Prev_low + Prev_close) / 3
    R1 = 2 * Pivot - Prev_low
    S1 = 2 * Pivot - Prev_High
    R2 = Pivot + (Prev_High - Prev_low)
    S2 = Pivot - (Prev_High - Prev_low)
    R3 = Pivot + 2 * (Prev_High - Prev_low)
    S3 = Pivot - 2 * (Prev_High - Prev_low)

    # Add columns to the DataFrame
    df['Pivot'] = Pivot
    df['R1'] = R1
    df['S1'] = S1
    df['R2'] = R2
    df['S2'] = S2
    df['R3'] = R3
    df['S3'] = S3

    return df


def pivot_points_each(df , close_col = 'Close' , low_col = 'Low' , high_col='High'):
    
    Prev_close = df[close_col].groupby(df.index.date).last().shift().reindex(df.index.date).values
    Prev_High = df[high_col].groupby(df.index.date).max().shift().reindex(df.index.date).values
    Prev_low = df[low_col].groupby(df.index.date).min().shift().reindex(df.index.date).values
    
    Pivot = (Prev_High + Prev_low + Prev_close) / 3
    R1 = 2 * Pivot - Prev_low
    S1 = 2 * Pivot - Prev_High
    R2 = Pivot + (Prev_High - Prev_low)
    S2 = Pivot - (Prev_High - Prev_low)
    R3 = Pivot + 2 * (Prev_High - Prev_low)
    S3 = Pivot - 2 * (Prev_High - Prev_low)

    return pd.Series(R1, index=df.index), pd.Series(S1, index=df.index), pd.Series(R2, index=df.index), pd.Series(S2, index=df.index)

    
    




def bars_since_condition(df, column):
    # Calculate the cumulative sum which resets at each '1' to create group identifiers
    group_ids = (df[column] == 1).cumsum()
    
    # Use 'cumcount' to count the number of occurrences since the last reset, adding 1 to start from 1
    bars_since = df.groupby(group_ids).cumcount() 
    
    # Wherever the condition is true, reset 'bars_since' to 1
    return bars_since.where(df[column] == 0, 1)


# def get_nearest_future_exp(date_series , symbol='NIFTY'):
#     """
#     Finds the nearest future expiry (>= current date) for all rows in a dataframe 
    
#     Parameters:
#     date_series (pandas.Series): A pandas series of datetime from which you want to find the nearest expiry
    
#     Returns:
#     nearest_future_expiry (pandas.Series): A pandas series with nearest expiry (>= current date) for each datetime in date_series
    
#     """   
#     #== find all valid expiries between start and end of df
#     expirylist = fetchExpiryDays(symbol=symbol)
#     exp_days = expirylist['dates'].tolist()
        
#     df_dates = np.array(date_series ,dtype='datetime64[ns]')
#     exp_dates = np.array(exp_days, dtype='datetime64[ns]')
    
#     future_diffs = exp_dates - df_dates[:, np.newaxis]
#     future_diffs[future_diffs < np.timedelta64(0)] = np.timedelta64(99999999, 'D') # Set past values to large number
#     nearest_future_indices = np.argmin(future_diffs, axis=1)
   
#     # Use the nearest indices to lookup the corresponding dates in exp_dates
#     nearest_future_dates = exp_dates[nearest_future_indices]
    
#     return nearest_future_dates


def get_prev_exp(date_series ,  symbol , expirylist):
    
    """
    Finds the previous expiry for all rows in a dataframe 
    
    Parameters:
    date_series (pandas.Series): A pandas series of datetime from which you want to find the nearest expiry
    expiry_dt (pandas.Series): A pandas series of expiry datetime 
    
    Returns:
    days_to_expiry (pandas.Series): A pandas series with business days difference to nearest expiry for each datetime in date_series
    
    """
    
    # Generate a list of all business days between start and end dates
    start_date = date_series.min().date()- timedelta(days=10)
    end_date = date_series.max().date()+ timedelta(days=10)
    
    #expirylist = fetchExpiryDays(symbol=symbol)
    exp_days = expirylist['Date'].tolist()
    #exp_days = [date.date() for date in exp_days]
    df_dates = np.array(date_series ,dtype='datetime64[ns]')
    exp_dates = np.array(exp_days, dtype='datetime64[ns]')
    
    prev_diffs = df_dates[:, np.newaxis] - exp_dates
    prev_diffs[prev_diffs < np.timedelta64(0)] = np.timedelta64(99999999, 'D') # Set negative values to large number
    nearest_previous_indices = np.argmin(prev_diffs, axis=1)
   
    # Use the nearest indices to lookup the corresponding dates in exp_dates
    nearest_previous_dates = exp_dates[nearest_previous_indices]
    #prev_exp = [date+datetime.time(15 , 15) for date in nearest_previous_dates]
    
    #nearest_previous_dates = pd.Series(nearest_previous_dates)+pd.Timedelta(hours=15, minutes=15)
    
    return nearest_previous_dates


def get_expiry(dt,expiry_offset , symbol , expiry_list):
    """
        get_expiry- calulates the expiry to trade for a datetime(row of df) on based on the expiry_offset
            
        Parameters
        ----------
        dt : datetime.date
            current date of the trade. 
            
        expiry_offset: int
            nth expiry away from current(latest) expiry 
            
        expiry_list : pd.DataFrame
            a dataframe with dates of all expiries    
                        
    """
    
    #expiriesList = fetchExpiryDays(symbol=symbol)
    possible_dates = expiry_list[expiry_list['dates'] >= dt]
    if not possible_dates.empty:
        if 0 <= expiry_offset < len(possible_dates):
            return possible_dates.iloc[expiry_offset]['dates']
        else:
            return "Not found"
    return "Not found"






# def days_to_expiry_weekly(date_series , symbol='NIFTY'):
    
#     """
#     Finds the business days to nearest expiry for weekly Indian options
    
#     Parameters:
#     date_series (pandas.Series): A pandas series of datetime from which you want to find the nearest expiry
#     expiry_dt (pandas.Series): A pandas series of expiry datetime 
    
#     Returns:
#     days_to_expiry (pandas.Series): A pandas series with business days difference to nearest expiry for each datetime in date_series
    
#     """
        
#     # Generate a list of all business days between start and end dates
#     start_date = date_series.min().date()
#     end_date = date_series.max().date()+ timedelta(days=10)
    
#     #== find all valid expiries between start and end of df
#     expirylist = fetchExpiryDays(symbol=symbol)
#     exp_days = expirylist['dates'].tolist()
    
    
#     df_dates = np.array(date_series.date )
#     exp_dates = np.array(exp_days)

#     trading_days = fetchTradingDays()
#     nse_cal = trading_days['dates']

#     nearest_indices = np.argmax(exp_dates[:, np.newaxis] >= df_dates, axis=0)
    
#     # Use the nearest indices to lookup the corresponding dates in exp_dates
#     nearest_dates = exp_dates[nearest_indices]


#     # Calculate the number of business days between each date in df and the nearest date in expiry
    
#     # , holidays=nse_cal.holidays().holidays
#     business_days = np.busday_count(df_dates.astype('datetime64[D]'), nearest_dates.astype('datetime64[D]') , weekmask='1111100' )
    
#     # Check if the nearest date is the same as the input date, and set business days to 0 if so
#     same_date_indices = np.where(nearest_dates == df_dates)[0]
#     business_days[same_date_indices] = 0
    
#     return business_days

# #==========================================================

# def days_since_expiry_weekly(date_series , symbol='NIFTY'):
    
#     """
#     Finds the business days from nearest  previous expiry for weekly Indian options
    
#     Parameters:
#     date_series (pandas.Series): A pandas series of datetime from which you want to find the nearest expiry
    
    
#     Returns:
#     days_to_expiry (pandas.Series): A pandas series with business days difference to nearest expiry for each datetime in date_series
    
#     """
    
    
    
#     start_date = date_series.min().date()
#     end_date = date_series.max().date()+ timedelta(days=10)
        
#     #== find all valid expiries between start and end of df
#     expirylist = fetchExpiryDays(symbol=symbol)
#     exp_days = expirylist['dates'].tolist()
        
        
#     df_dates = np.array(date_series.date )
#     exp_dates = np.array(exp_days)

#     # Convert to numpy arrays of type 'datetime64[D]'
#     df_dates_np = np.array([np.datetime64(d) for d in df_dates])
#     exp_dates_np = np.array([np.datetime64(d) for d in exp_dates])

#     # Find the nearest date in exp_dates that is less than or equal to each date in df_dates
#     nearest_dates = np.array([exp_dates_np[exp_dates_np <= d][-1] if np.any(exp_dates_np <= d) else np.datetime64('NaT') 
#                             for d in df_dates_np])


#     days_since_expiry = np.busday_count(nearest_dates ,df_dates_np  , weekmask='1111100' )

#     return days_since_expiry










def calculate_anchored_vwap(df, open_col, high_col, low_col, close_col, volume_col, start_col):
    """
    Calculate Anchored VWAP for each row in the DataFrame, restarting the calculation each time 'a condition' is 1.
    
    Can be used to find vwap from say an expiry or an event
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the price and volume data.
    open_col (str): Name of the column containing open prices.
    high_col (str): Name of the column containing high prices.
    low_col (str): Name of the column containing low prices.
    close_col (str): Name of the column containing close prices.
    volume_col (str): Name of the column containing volume data.
    start_col (str): Name of the binary column indicating the start of the VWAP period.
    
    Returns:
    pd.Series: A Series containing the Anchored VWAP values.
    """
    # Create a rolling identifier for each VWAP period
    vwap_period_id = df[start_col].cumsum()

    # Calculate typical price
    typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3

    # Calculate price-volume product
    pv_product = typical_price * df[volume_col]

    # Calculate cumulative price-volume and cumulative volume for each period
    cumulative_pv = pv_product.groupby(vwap_period_id).cumsum()
    cumulative_vol = df[volume_col].groupby(vwap_period_id).cumsum()

    # Calculate Anchored VWAP
    anchored_vwap = cumulative_pv / cumulative_vol

    return anchored_vwap








def calc_intraday_low(series):
    # intraday_low_col = f'intraday_low_{agg_func}'
    spread_intraday_low = series.groupby(series.index.date).transform(lambda x: x.expanding().min())

    
def calc_spread(action,ratio,close_series):
    """
        calc_spread- calculates the spread of the options legs
            
        Parameters
        ----------
        action : string
            "buy"/"sell" based on this the spread is calculated. 
            
        ratio: int
            multiplier of the leg position
            
        close_series    : pd.series    
            close price of the option contract
            
        low_series    : pd.series    
            low price of the option contract
    """
    if(action =='buy'):
        spread_price = spread_price.add(ratio*close_series, fill_value=0)
    elif(action == 'sell'):
        spread_price = spread_price.sub(ratio*close_series, fill_value=0)
    

# def get_nth_day(dt,offset):
#         """
#             get_nth_day- calulates the day  based on the offset
                
#             Parameters
#             ----------
#             dt : datetime.date
#                 current date of the trade. 
                
#             offset: int
#                 nth day from current(latest) day        
#         """
#         possible_dates = [fetchTradingDays['dates'] >= dt]
#         if not possible_dates.empty:
#             if 0 <= offset < len(possible_dates):
#                 return possible_dates.iloc[offset]['dates']
#             else:
#                 return "Not found"
#         return "Not found"


def intraday_vwap(df , high_col = 'High' , low_col='Low' , close_col = 'Close' , vol_col = 'Volume'):
    """
    Get the intraday vwap for a df.
    
    requirements: df with datetime as index and HLCV
    
    """
    df2 = df.copy(deep=True)


    df2['typical_p'] = ((df2[high_col] + df2[low_col] + df2[close_col])/3).astype('float64') 
    #print(df2['typical_p'])

    cum_vol = df2.groupby(df2.index.date)[vol_col].apply(lambda x: x.expanding().sum())

    #print(df2['cumulative_volume'])

    df2['cumm'] = df2['typical_p']*df2[vol_col]

    expanding_sum = df2.groupby(df.index.date)['cumm'].apply(lambda x: x.expanding().sum())

    xx = expanding_sum/(cum_vol)
    
    return xx.values
    
#=== This is faster 
   
def vwap(df, label='vwap', window=3, fillna=True , highcol='High' , lowcol='Low' , closecol='Close' , opencol = 'Open' , volcol='Volume'):
        from ta.volume import VolumeWeightedAveragePrice
        
        df[label] = VolumeWeightedAveragePrice(high=df[highcol], low=df[lowcol], close=df[closecol], volume=df[volcol], window=window, fillna=fillna).volume_weighted_average_price()
        
        return df    

def daily_moving_average(df, timeframe, column, periods=3, agg_func='last'):
    # Calculate the moving average excluding the latest value
    moving_average = df.resample(timeframe)[column].agg(agg_func).dropna().rolling(window=periods).mean().shift(1)

    # Reindex the moving average back to the original DataFrame
    moving_average_reindexed = moving_average.reindex(df.index, method='ffill')

    # Add the moving average to the original DataFrame
     

    return moving_average_reindexed 


def analyze_market_condition(price_change, oi_change):
    
    """
    Get the OI intrepetation for a list with price_change and oi_change
            
    """
   
    if price_change > 0 and oi_change > 0:
        return 'Long Buildup'
    elif price_change < 0 and oi_change < 0:
        return 'Long unwinding'
    elif price_change < 0 and oi_change > 0:
        return 'Short Buildup'
    elif price_change > 0 and oi_change < 0:
        return 'Short Covering'
    else:
        return 'Indeterminate/Other'    
    
    


# def fetch_prev_day_OI_analysis(und_df , symbol , callput ):

    
#     """
#     Finds the previous day OI analysis for a symbol using its curretn options.
#     Finds the top two strikes with highest OI and its initrepretation using price and OI change
    
    
#     requirements: underlying data(OHLC and future expiry dates) with datetime as index and symbol name
    
#     """

#     # Initialize variable to keep track of the previous date
#     previous_date = None

#     unique_dates = np.unique(und_df.index.date)
    
#     comb_df = pd.DataFrame()
    
#     #=== Process option chain for each day 
    
#     for current_date in unique_dates:
        
#         print("\n Processing date" , current_date)
               
#         #-- skip the first date as else we can see the future
#         if previous_date is not None:
            
#             #print( "\n current_date", current_date )
#             #print( "\n \n ============== previous_date", previous_date )
        
#             data_day = und_df[und_df.index.date==current_date]
            
#             #print("\n Days data is " , data_day)
            
            
#             #=== OI is only relevant for days with atleast 1 day since expiry
                      
                
#             if data_day['days_since_expiry'].iloc[0]>=1:

#                 # Determine the day's low and high, rounded to the nearest 100
                
#                 day_low = np.ceil(data_day['Low'].min() / 100) * 100 
#                 day_high = np.ceil(data_day['High'].max() / 100) * 100
                
#                 call_limit = np.ceil(data_day['High'].max()*1.02 / 100) * 100
#                 put_low_limit = np.floor(data_day['Low'].min()*0.98 / 100) * 100
                
#                 strike_range_call = np.arange(day_low, call_limit , 100)
#                 strike_range_put = np.arange(put_low_limit, day_high , 100)
                                
                
#                 # Get start , end and expiry dates
#                 exp_Date = data_day['next_exp'].dt.date.iloc[0]
#                 #print("\n"  , "put_range: " ,strike_range_put  , exp_Date)
                
#                 #======================
                
#                 start_date = min(data_day['near_Exp'].dt.date.iloc[0] ,previous_date)-timedelta(days=5) 
#                 end_date = previous_date
                
#                 chain = fetchAllOptions(symbol=symbol , startDate=start_date , endDate=end_date , upper_strike=strike_range_call[-1] , lower_strike=strike_range_call[0] , expiryDate=exp_Date , callPut=callput , timeframe='1D')
                
#                 #print("\n Unique days found in chain" , chain.index.unique())               
                
#                 chain = chain.assign(
#                                 OI_pct_change=lambda x: x.groupby('Strike')['OI'].pct_change(),
#                                 Close_pct_change=lambda x: x.groupby('Strike')['Close'].pct_change()
#                             )
                
#                 #print( "\n the chain is : ", chain)
                
                
#                 #== Fetch the Option chain and create a pivot table for Close and OI and changes in them
#                 pivot = chain.pivot_table(index=chain.index.date, columns='Strike', values=['OI' , 'Close' , 'OI_pct_change' ,'Close_pct_change' ], aggfunc='max')
                                
#                 #==== Filter to last row which is previous date
#                 if previous_date in pivot.index:
#                     pivot = pivot.loc[previous_date]
                    
#                     pivot = pivot.dropna(how='all')
                    
#                     if pivot is None or pivot.empty:
#                         print("\n ========Empty pivot: " )
                        
#                     top_2_strikes = pivot['OI'].nlargest(2).index.get_level_values('Strike')

#                     # Create a list of tuples for the column headers
#                     top_2_strikes_columns = [(level, strike) for strike in top_2_strikes for level in [ 'OI', 'OI_pct_change', 'Close_pct_change']]
#                     #print("\n top_2_strikes_columns are  " , top_2_strikes_columns)
#                     existing_columns = [col for col in top_2_strikes_columns if col in pivot.index]
#                     #print( "\n Exising columns are: ", existing_columns)
                    
#                     if top_2_strikes_columns==existing_columns:
#                         # Retrieve all columns for these top 2 strikes
#                         #print("\n" , "Columns are equal")
#                         top_2_strikes_data = pivot[top_2_strikes_columns]
#                         #print(top_2_strikes_data)
                    
#                     # To view the result
#                     #print( "\n======= Data of top 2 strikes", top_2_strikes_data)
                    
#                     if (top_2_strikes_data is None or  top_2_strikes_data.empty):
#                         print("\n ========Empty dataframe: " )
                                                        
#                                     # Extract unique strikes
#                     unique_strikes = top_2_strikes_data.index.get_level_values('Strike').unique()

                    
                    
#                     rows = []

#                     for strike in unique_strikes:
#                         price_change = top_2_strikes_data.loc[('Close_pct_change', strike)]
#                         oi_change = top_2_strikes_data.loc[('OI_pct_change', strike)]
#                         condition = analyze_market_condition(price_change, oi_change)

#                         # Create a dictionary for each row
#                         row_data = {
#                             'Date': previous_date,
#                             'Strike': strike,
#                             'Buildup': condition
#                         }
#                         rows.append(row_data)

#                     # Create DataFrame from the list of dictionaries
#                     df1 = pd.DataFrame(rows)
        
                                        
#                     comb_df = pd.concat([comb_df , df1] , axis=0)
                    
                    
#                     # ... rest of your code to process pivot_day_data
#                 else:
#                     print(f"No data available for {previous_date}")
#                                 #pivot = pivot.loc[previous_date]
               
                
#         previous_date = current_date
   
    
#     comb_df = comb_df.groupby('Date').agg({'Strike': list, 'Buildup': list}).reset_index()    
#     # Add time of 9:15 to the 'Date' column
#     #comb_df['Date'] = pd.to_datetime(comb_df['Date']) + pd.Timedelta(hours=9, minutes=15)
#     #print(comb_df['Date'])
#     comb_df.set_index('Date' , inplace=True)
       
#     return comb_df        
        
    


# def fetch_single_option(symbol , expiry_list , und_df_row,und_df , strike_ref_col='Close' , moneyness=0 , min_strike_chang=100 , expiry_offset=0 , option_type='CE'):
    
    
#     option_data = []  #
    
        
#     curr_date = und_df_row.name.date()
#     #print("\n\n", curr_date)
    
#     expiry_to_trade = get_expiry( curr_date,expiry_offset , symbol ,  expiry_list )
    
#     strike_of_focus = round(und_df_row[strike_ref_col],-2)
    
#     strike_to_fetch = strike_of_focus if moneyness == 0 else ((moneyness*min_strike_chang) + strike_of_focus  if option_type == 'CE' else strike_of_focus - (moneyness*min_strike_chang))

#     print("\n Strike to fetch - " , strike_to_fetch)

#     end_data = expiry_to_trade
    
#     curr_date = curr_date - datetime.timedelta(days=3)
    
#     #print("\n Starting and ending days: " , curr_date  , end_data)
#     option_df_temp = fetchDataOptions(symbol=symbol,startDate = curr_date, endDate= end_data ,strike = strike_to_fetch ,expiryDate = expiry_to_trade,callPut = option_type,timeframe = '15min')
    
#     new_columns = option_df_temp.columns.map(lambda x: 'opt_' + str(x))
#     option_df_temp.columns = new_columns
#     # option_data.append(option_df_temp)
#     # del(option_df_temp)

#     #== Now we combine the signal(underlying) data with option data 
#     filtered_df = und_df[(und_df.index >= option_df_temp.index[0]) & (und_df.index <= option_df_temp.index[-1])]
#     #print(filtered_df)
#     result = pd.concat([option_df_temp, filtered_df], axis=1,join='outer')
    
#     return result
    
    
def get_leg_info(leg_id):
    for leg_info in config_object.legs_info:
        if leg_info['leg_id'] == leg_id:
            return leg_info
    return None  # return None if no matching leg_id is found    



# def fetch_all_options_combine_und(config , symbol, expiry_list,und_df_row, und_df , strike_ref_col , min_strike_chang , time_frame , trading_days ):
    
#     option_data = []  #
    
#     for leg in config['legs_info']:
        
#         #print("\n Processing leg info" , leg)
        
#         curr_date = und_df_row.name.date()
                    
#         expiry_to_trade = get_expiry(curr_date,leg['expiry_offset'] , symbol , expiry_list )
        
#         strike_of_focus = round(und_df_row[strike_ref_col],-2)
        
#         if leg["call_put"] == 'CE':
#             strike_to_fetch = (leg["moneyness"] * min_strike_chang) + strike_of_focus
#         else:
#             strike_to_fetch = strike_of_focus - (leg["moneyness"] * min_strike_chang)
        
        
#         end_data = expiry_to_trade
        
#         #print( "\n Expiry : " , expiry_to_trade , "\n strike: " , strike_to_fetch)
        
#         curr_date = curr_date - datetime.timedelta(days=3)
                
#         option_df_temp = fetchDataOptions(symbol=symbol,startDate = curr_date, endDate= end_data ,strike = strike_to_fetch ,expiryDate = expiry_to_trade,callPut = leg['call_put'],timeframe = time_frame , trading_days=trading_days)
#         #leg_id = add_leg(name=f"{leg['action']}_{leg['ratio']}_{leg['call_put']}_{strike_of_focus}_{str(expiry_to_trade)}")
        
#         if option_df_temp is not None and not option_df_temp.empty and len(option_df_temp) > 30:
            
#             new_columns = option_df_temp.columns.map(lambda x: leg["leg_id"]+'_' + str(x))
            
#             option_df_temp.columns = new_columns
            
#             option_data.append(option_df_temp)
            
#             del(option_df_temp)

    
#     if option_data:  # This ensures that option_data is not empty
#         combined_df = pd.concat(option_data, axis=1)
#         filtered_df = und_df[(und_df.index >= combined_df.index[0]) & (und_df.index <= combined_df.index[-1])]
#         result = pd.concat([combined_df, filtered_df], axis=1, join='outer')
#     else:
#         print("Warning: No options data fetched. Returning an empty DataFrame.")
#         result = None  # Return an empty DataFrame or handle as needed
   
   
#     #print("\n\n\n" , "Result or net dataframe is : \n" ,result )
#     return result



def expanding_high_since_expiry(df, expiry_series , column='indHigh'):
    
    #==== Function to find the expanding high since a date column..eg expiry
    
    
    # Ensure the expiry_series is aligned with df's index
    #df['Temp_Expiry'] = expiry_series.reindex(df.index)

    # Use groupby on Temp_Expiry and then compute expanding max on the 'High' column
    expanding_high = df.groupby(expiry_series)[column].expanding().max().reset_index(level=0, drop=True)
    
    # Drop the temporary expiry column
    #df.drop(columns=['Temp_Expiry'], inplace=True)
    
    return expanding_high


def expanding_low_since_expiry(df, expiry_series , column='indLow'):
    
    #==== Function to find the expanding high since a date column..eg expiry
    
    
    # Ensure the expiry_series is aligned with df's index
    #df['Temp_Expiry'] = expiry_series.reindex(df.index)

    # Use groupby on Temp_Expiry and then compute expanding max on the 'High' column
    expanding_low = df.groupby(expiry_series)[column].expanding().min().reset_index(level=0, drop=True)
    
    # Drop the temporary expiry column
    #df.drop(columns=['Temp_Expiry'], inplace=True)
    
    return expanding_low

def expanding_high_since_condition(df, high_col='High', condition_col='is_exp'):
    # Create a segment identifier that increments each time condition_col is 1
    df['segment'] = df[condition_col].cumsum()

    # Calculate expanding high within each segment
    expanding_highs = df.groupby('segment')[high_col].expanding().max().reset_index(level=0, drop=True)

    return expanding_highs



# def plot_signal2(config ,  tradelog ,unddf, signal_date):
#         """
#             plot_signal- Iterates over the tradelist and plot the signal u want
            
#             Parameters
#             ----------
            
#             signal_date: string or datetime
#                 the date for which you want to see the signals
                      
        
#          """
         
#         import mplfinance as mpf 
#         import matplotlib.pyplot as plt
#         import matplotlib.gridspec as gridspec

        
        
#         opt_timeframe = str(config['timeframe']) + "min" 
#         print("\n Plotting on timeframe : " , opt_timeframe)
         
#         if isinstance(signal_date, str):
#             signal_date = datetime.datetime.strptime(signal_date, '%Y-%m-%d %H:%M:%S')
        
#         df_date =tradelog[tradelog['Time of Entry'].dt.date == signal_date]
#         df_date.reset_index(inplace=True)
                
#         fig = plt.figure(figsize=(12, 8))

#         # create GridSpec with 2 rows and 2 columns
#         gs = gridspec.GridSpec(2, 2, figure=fig)
#         # Create my own `marketcolors` style:
#         mc = mpf.make_marketcolors(up='b', down='r')
#         s = mpf.make_mpf_style(base_mpl_style='bmh', marketcolors=mc)
        
#         for i, trade in df_date.iterrows():
#             # Fetch historical data
#             entry_time = trade['Time of Entry']
#             exit_time = trade['Time of Exit']

#             string_leg = trade['name']
                        
#             # Split the string if '_' is present, else assign 'NA'
#             parts = string_leg.split('_') if '_' in string_leg else ['NA', 'NA', 'NA']

#             # Unpack the parts to callput, strike, and expiry_to_trade
#             callput, strike, expiry_to_trade = parts[:3]
            
                                
#             option_df_temp = fetchDataOptions(symbol=config['symbol'],startDate = entry_time.date(), endDate= exit_time.date() ,strike = strike ,expiryDate = expiry_to_trade,callPut = callput,timeframe = opt_timeframe , trading_days=fetchTradingDays())
#             #print(option_df_temp)
#             t = fig.add_subplot(gs[0, i])
#             t.grid(True, color='darkgrey')
#             mpf.plot(option_df_temp,style=s,type='candle',vlines=dict(vlines=[f'{str(entry_time)}',f'{str(exit_time)}'],linewidths=(1,1)),ax=t,axtitle=f"Strikes = {strike}")

#         #For underlyiung
#         # Modify the date filtering for und_df
#         unddf = unddf[(unddf.index.date >= signal_date - timedelta(days=1)) & (unddf.index.date <= signal_date + timedelta(days=1))]

#         t = fig.add_subplot(gs[1, :])
#         t.grid(True, color='darkgrey')
#         mpf.plot(unddf,style=s,type='candle',ax=t,axtitle=f"Underlying")
#         plt.tight_layout()
#         plt.show()



# def process_VIX():
    
#     vixfile = r"C:\Users\Administrator\Desktop\Data\Nifty_index_and_options\India VIX.csv"

#     vixdf = pd.read_csv(vixfile)
#     vixdf['TIMESTAMP'] = pd.to_datetime(vixdf['TIMESTAMP'] , format="%Y-%m-%d")
#     vixdf.set_index('TIMESTAMP' , inplace=True)
#     vixdf['TIMESTAMP'] = vixdf['TIMESTAMP'] + timedelta(hours=15, minutes=15)
#     vixdf = vixdf['VIX_Close']
    
#     return vixdf    
        

def resistance_all_levels(und_df , price_increase_percentage):
    
    import pandas as pd
    import numpy as np

    # Assuming und_df is your DataFrame and it's already defined with necessary columns
    
    # Your existing code
    
    target_price = und_df['today_open'] + (und_df['prevday_rng'] * price_increase_percentage)
     
    resistance_cols = ['prevday_close', 'prevdayh', 'expiry_high', 'R1', 'S1', 'Pivot', '20DMA', 'R2', 'today_open', 'all_time_high', 'nearest_100_fhr', 'round_level' , 'ind_3highc' , '100DEMA']
     
    # Convert DataFrame to NumPy array for efficient computation
    
    resistance_values = und_df[resistance_cols].values
    target_prices = target_price.values[:, np.newaxis]  # Convert to column vector for broadcasting
    today_opens = und_df['today_open'].values[:, np.newaxis]

    # Filter out values below target price and today's open
    valid_resistances = np.where((resistance_values > target_prices) & (resistance_values > today_opens), resistance_values, np.nan)

    # Sort along axis=1 and take the first three values
    sorted_resistances = np.sort(valid_resistances, axis=1)
    nearest_three_resistances = sorted_resistances[:, :3]  # Take first three columns which are the nearest resistances

    # Convert back to DataFrame
    resistance_three = list(nearest_three_resistances)

    return resistance_three



#--- Make global the tradelog dictionary

def make_tradelog( action , position_book, id ,name  ,dtime ,Side ,price , qty , trade_book):
        """
        make_tradelog - adds trade values to 'temp_dict_entry'
        
        Parameters
        ----------
            action : string
                entry/exit
                
            dtime: datetime
                timestamp of the trade
                
            price  : float    
                price point of the trade
        """        

        if(action == 'entry'):
            position_book[id]['name'] = name
            position_book[id]['inposition'] = True
            position_book[id]['Time of Entry'] = dtime
            position_book[id]['Side'] = Side
            position_book[id]['Entry Price'] = price
            position_book[id]['qty'] = qty
            #tradelog = pd.concat([tradelog,pd.DataFrame([position_book[id]])]) 
                        
        elif(action == 'exit'):
            position_book[id]['inposition'] = False
            position_book[id]['Time of Exit'] = dtime
            #position_book[id]['Side'] = Side           
            position_book[id]['Exit Price'] = price
            #position_book[id]['Exit Reason'] =  'reason'
            position_book[id]['qty'] = qty
            trade = pd.DataFrame([position_book[id]])
            print("\n \n Entire_trade in position_book is " ,trade )
            
            return trade
            
            
    
   
  


def support_all_levels(und_df , price_increase_percentage):
    
    """
    Finds the support levels below today open and % of previous range from open.
    Levels considered are 5 ,20 , 100 DMA and Pivot points and prevday low and closes etc
    
    
    requirements: underlying data(OHLC) with datetime as index and price_increase_percentage() e.g) 0.2
    
    """
    
    
    import pandas as pd
    import numpy as np

    # Assuming und_df is your DataFrame and it's already defined with necessary columns
    
    # Your existing code
    
    target_price = und_df['today_open'] - (und_df['prevday_rng'] * price_increase_percentage)
     
    resistance_cols = ['prevday_close', 'prevdayl', 'expiry_low', 'R1', 'S1', 'S2', 'Pivot', '20DMA', 'R2', 'fh_low_round', 'round_support' , 'ind_3dlowc' , '100DEMA']
     
    # Convert DataFrame to NumPy array for efficient computation
    
    resistance_values = und_df[resistance_cols].values
    target_prices = target_price.values[:, np.newaxis]  # Convert to column vector for broadcasting
    today_opens = und_df['today_open'].values[:, np.newaxis]

    # Filter out values above target price and above today's open
    valid_supports = np.where((resistance_values < target_prices) & (resistance_values < today_opens), resistance_values, today_opens*0.99)


    # Sort along axis=1 and take the first three values
    sorted_resistances = -np.sort(-valid_supports, axis=1)

    nearest_three_resistances = sorted_resistances[:, :3]  # Take first three columns which are the nearest resistances

    # Convert back to DataFrame
    resistance_three = list(nearest_three_resistances)

    return resistance_three

    
    

def yang_zhang_vol(df , periods=20 , trading_days=252 , clean=True , highcol = 'High' , lowcol='Low' , closecol='Close' , opencol='Open'):
    
    import math
    """
    Calculate the Yang-Zhang estimator of volatility for an array of open, high, low, and closing prices( daily is better).
    K is a factor to correct for bias. Default is 0.34 for daily returns.

    :param open_prices: array-like of opening prices
    :param high_prices: array-like of high prices
    :param low_prices: array-like of low prices
    :param close_prices: array-like of closing prices
    :return: the Yang-Zhang estimate of volatility
    """
    
    
    logho = (df[highcol]/df[opencol]).apply(np.log)
    
    loglo = (df[lowcol]/df[opencol]).apply(np.log)
    
    logco = (df[closecol]/df[opencol]).apply(np.log)
    
    logoc = (df[opencol]/df[closecol].shift(1)).apply(np.log)
    
    logoc_sq = logoc ** 2
    
    logcc = (df[closecol]/df[closecol].shift(1)).apply(np.log)
    
    logcc_sq = logcc **2
    
    
    rs = logho *( logho - logco) + loglo * (loglo - logco)
    
    close_vol = logcc_sq.rolling(window=periods , center=False ).sum() * ( 1.0 /(periods - 1.0))
    
    open_vol = logoc_sq.rolling(window=periods , center=False).sum() * ( 1.0 /(periods - 1.0))
    
    window_rs = rs.rolling(window=periods , center=False).sum() * (1/(periods-1))
    
    
    k = 0.34/ (1 + (periods +1) / (periods -1)) 
    
    result = (open_vol + k*close_vol + (1-k)*window_rs).apply(np.sqrt)* math.sqrt(trading_days)
    
    if clean:
        return result.dropna()
    else:
        return result
        


