import pandas as pd
import datetime
import psycopg2
import psycopg2.extras as psycopg2_e


#
ip = "13.126.50.178"

#establish a connection to a PostgreSQL database

conn = psycopg2.connect(f"dbname='qdb' user='admin' host='{ip}' port='8812' password='quest'")

cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)


def fetchTradingDays():
    
    """
    Fetch the valid trading days of India from database
    
    
    Returns:
    df(pd.Datafram): A Series containing the trading (datetime) dates.
    """
    
    global ip
    global conn
    global cursor
    query_ = f"""
                SELECT DISTINCT DATE_TRUNC('day', Datetime) AS dates
                FROM NIFTY_INDEX
            """
    try:
        cursor.execute(query_)
    except Exception as e:
        try:
            cursor.close()
            cursor = conn.cursor()
        except:
            conn.close()
            ip = "13.126.50.178"
            conn = psycopg2.connect(f"dbname='qdb' user='admin' host='{ip}' port='8812' password='quest'")
                    
        cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)
        cursor.execute(query_)
    
    
    df = pd.DataFrame(cursor.fetchall())
    df['dates'] = pd.to_datetime(df['dates']).dt.date
    return df


# def fetchExpiryDays(symbol):
#     # global ip
#     # global conn
#     # global cursor
    
        
#     #==== stock fututes expiry_list
#     stock_Exp = r"/home/aakash_tandon/Python_strategies/India_stock_futures_expiries.csv"
#     bk_exp = r"/home/aakash_tandon/Python_strategies/Banknifty_weekly_expiries.csv"
#     nif_exp = r"/home/aakash_tandon/Python_strategies/nifty_weekly_expiries.csv"
#     mid_exp = ''
#     fin_exp = r"/home/aakash_tandon/Python_strategies/Finnifty_weekly_expiries.csv"
#     sensex_exp = ''
        
    
#     # Map symbols to their expiry date files
#     file_map = {
#         'NIFTY': nif_exp,
#         'BANKNIFTY': bk_exp,
#         'FINNIFTY': fin_exp,
#         'MIDCPNIFTY': mid_exp,
#         'SENSEX': sensex_exp
#     }
#     if(symbol not in ['NIFTY','BANKNIFTY','FINNIFTY','MIDCPNIFTY' , 'SENSEX']):
#         print('invalid symbol')
#         return None
#     # query_ = f"""
#     #             SELECT DISTINCT Expiry
#     #             FROM {symbol}_OPTIONS_MODIFIED
#     #             ORDER BY Expiry ASC;
#     #         """
#     # try:
#     #     cursor.execute(query_)
#     # except Exception as e:
#     #     try:
#     #         cursor.close()
#     #         cursor = conn.cursor()
#     #     except:
#     #         conn.close()
#     #         ip = "13.126.50.178"
#     #         conn = psycopg2.connect(f"dbname='qdb' user='admin' host='{ip}' port='8812' password='quest'")
                    
#     #     cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)
#     #     cursor.execute(query_)

#     else:
        
        
        
        
    

#     df = pd.DataFrame(cursor.fetchall())
    
#     # Convert the string column to datetime
#     df['dates'] = pd.to_datetime(df['Expiry'], format='%d-%m-%Y').dt.date
    
#     # Sort the DataFrame by the 'dates' column in ascending order
#     df = df.sort_values(by='dates')
    
#     # Selecting only the 'dates' column if needed
#     df = df[['dates']]
#     return df


def fetchDataUnder(symbol,startDate,endDate,timeframe):
    global ip
    global conn
    global cursor
    if(symbol not in ['NIFTY','BANKNIFTY','FINNIFTY','MIDCPNIFTY']):
        print('invalid symbol')
        return None
        
    if isinstance(startDate, datetime.date):
        startDate = startDate.strftime('%Y-%m-%d')
    
    if isinstance(endDate, datetime.date):
        endDate = endDate.strftime('%Y-%m-%d')
        
    endDate = datetime.datetime.strptime(endDate, '%Y-%m-%d')
    endDate = endDate + datetime.timedelta(days=1)
    endDate = endDate.strftime('%Y-%m-%d')
    
    query_ = f"""
            with 
              meta_query  as ((select Datetime as timestamp, Open, High, low, close from {symbol}_INDEX WHERE Datetime >= '{startDate}' AND Datetime <= '{endDate}' ORDER BY Datetime) timestamp(timestamp))
              SELECT timestamp, first(Open) as Open, max(High) as High, min(Low) as Low, last(Close) as Close from meta_query 
              WHERE  
              timestamp in '2000-01-01T09:15;375m;1d;10000' 
              SAMPLE BY {timeframe}m;
            """
            
    try:
        cursor.execute(query_)
    except Exception as e:
        print(e)
        try:
            cursor.close()
            cursor = conn.cursor()
        except:
            conn.close()
            ip = "13.126.50.178"
            conn = psycopg2.connect(f"dbname='qdb' user='admin' host='{ip}' port='8812' password='quest'")
                    
        cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)
        cursor.execute(query_)

    df = pd.DataFrame(cursor.fetchall())
    df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
    return df


#=====================================================================================


# def fetchDataOptions(symbol,startDate,endDate,strike,expiryDate,callPut,timeframe):
#     global ip
#     global conn
#     global cursor
#     if(symbol not in ['NIFTY','BANKNIFTY','FINNIFTY','MIDCPNIFTY']):
#         print('invalid symbol')
#         return None
    
    
    
#     # Format dates in 'YYYY-MM-DD' for SQL compatibility
#     if isinstance(startDate, datetime.date):
#         startDate = startDate.strftime('%Y-%m-%d')
#     if isinstance(endDate, datetime.date):
#         endDate = endDate.strftime('%Y-%m-%d')
#     if isinstance(expiryDate, datetime.date):
#         expiryDate = expiryDate.strftime('%d-%m-%Y')

#     endDate = datetime.datetime.strptime(endDate, '%Y-%m-%d')
#     endDate = endDate + datetime.timedelta(days=1)
#     endDate = endDate.strftime('%Y-%m-%d')

#     query_ = f"""
#             with 
#             meta_query  as (select  Datetime as timestamp,OptionType,Strike,Expiry,Open,High,Low,Close,Volume,OI,Datetime from '{symbol}_OPTIONS_MODIFIED' where Datetime > '{startDate}' and Datetime < '{endDate}')
#             SELECT min(timestamp) as timestamp, OptionType,Strike,Expiry,first(Open) as Open, max(High) as High, min(Low) as Low, last(Close) as Close , sum(Volume) as Volume , last(OI) as OI from meta_query where timestamp IN '2000-01-01T09:15;404m;1d;10000' and  (Strike = {strike}) and (Expiry = '{expiryDate}') and (OptionType='{callPut}') SAMPLE BY 1m; 
#             """
#         #
#     try:
#         #print(query_)
#         cursor.execute(query_)
#     except Exception as e:
#         print(e)
#         try:
#             cursor.close()
#             cursor = conn.cursor()
#         except:
#             conn.close()
#             ip = "13.126.50.178"
#             conn = psycopg2.connect(f"dbname='qdb' user='admin' host='{ip}' port='8812' password='quest'")
                    
#         cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)
#         cursor.execute(query_)

#     df = pd.DataFrame( cursor.fetchall())
#     #df.empty or
#     if( len(df) <= 0):
#         print(f"\nOption data Not found,- {startDate}__{endDate}__{strike}__{expiryDate}__{callPut}")
#         print(df)
#         return  None
    
#     #=== if 1-min data is less than 330 minutes of data than data is inadequate
    
#     elif(len(df)<330):
#         print("Data Not GOOD")
#         return None
#     else:
#         df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
#         df = df.set_index("timestamp")
        
        
        
        
        
#         # Check if the first timestamp is not 09:15 and handle it
#         market_start_time = pd.to_datetime(f"{df.index[0].date()} 09:15:00")
#         if df.index[0] > market_start_time:
            
#             print("\n \n Data does not start from 09:15... Please check ---" , "--- First index is : " , df.index[0] , "\n\n-----------" )
#             missing_row = pd.DataFrame(index=[market_start_time], columns=df.columns)


#         # Generate a complete timestamp index (including 09:15)
#         start_date = df.index[0].date()
#         end_date = df.index[-1].date()
#         complete_index = pd.date_range(start=start_date, end=end_date, freq='T', closed='right')

#         # Reindex the DataFrame to include all timestamps
#         df = df.reindex(complete_index)
                
        
        
#         try:
            
#             if 'T' in timeframe or 'min' in timeframe or 'm' in timeframe or 'Min' in timeframe:
#             # Extract numeric value from timeframe string
#                 timeframe2 = int(''.join(filter(str.isdigit, timeframe)))
#                 timeframe2 = str(timeframe2) + 'T'
#             else:
#                 timeframe2 = timeframe
                
#             df_resampled = df.resample(f"{timeframe2}", origin='start').agg({
#                 'OptionType': 'first',
#                 'Strike': 'first',
#                 'Expiry': 'first',
#                 'Open': 'first',
#                 'High': 'max',
#                 'Low': 'min',
#                 'Close': 'last' ,
#                 'Volume': 'sum' , 
#                 'OI':'last'
#             })
#             # Filter only the valid market hours
#             filtered_df = df_resampled.between_time('09:15', '15:30')
#             # To drop rows which are entirely NaN (which might happen after resampling)
#             df = filtered_df.dropna()
#             df.index = df.index.round('s')
#             print("After resampling the data has first index as : " , df.index[0])
#             print(df.head(5))
#         except Exception as e:
#             print("Error while fetching one-minute data and resampling:", e)        
#             return None
#     return df




def fetchDataOptions( symbol,startDate,endDate,strike,expiryDate,callPut,timeframe , trading_days ):
    global ip
    global conn
    global cursor
    if(symbol not in ['NIFTY','BANKNIFTY','FINNIFTY','MIDCPNIFTY']):
        print('invalid symbol')
        return None
    
    #print( "\n Start date is: " , startDate)
    
    # Filter trading_days DataFrame for the range between startDate and endDate
    filtered_trading_days = trading_days[((trading_days['dates'] >= startDate) & (trading_days['dates'] <= endDate))]
    #print(filtered_trading_days)
    
    complete_trading_timestamps = pd.DatetimeIndex([ time for date in filtered_trading_days['dates']
    for time in pd.date_range(start=date.strftime('%Y-%m-%d') + ' 09:15', end=date.strftime('%Y-%m-%d') + ' 15:29', freq='T')])
    #print(complete_trading_timestamps)
    
    
    # Format dates in 'YYYY-MM-DD' for SQL compatibility
    if isinstance(startDate, datetime.date):
        startDate = startDate.strftime('%Y-%m-%d')
    if isinstance(endDate, datetime.date):
        endDate = endDate.strftime('%Y-%m-%d')
    if isinstance(expiryDate, datetime.date):
        expiryDate = expiryDate.strftime('%d-%m-%Y')

    endDate = datetime.datetime.strptime(endDate, '%Y-%m-%d')
    endDate = endDate + datetime.timedelta(days=1)
    endDate = endDate.strftime('%Y-%m-%d')

    query_ = f"""
            with 
            meta_query  as (select  Datetime as timestamp,OptionType,Strike,Expiry,Open,High,Low,Close,Volume,OI,Datetime from '{symbol}_OPTIONS_MODIFIED' where Datetime > '{startDate}' and Datetime < '{endDate}')
            SELECT min(timestamp) as timestamp, OptionType,Strike,Expiry,first(Open) as Open, max(High) as High, min(Low) as Low, last(Close) as Close , sum(Volume) as Volume , last(OI) as OI from meta_query where timestamp IN '2000-01-01T09:15;404m;1d;10000' and  (Strike = {strike}) and (Expiry = '{expiryDate}') and (OptionType='{callPut}') SAMPLE BY 1m; 
            """
        #
    try:
        #print(query_)
        cursor.execute(query_)
    except Exception as e:
        print(e)
        try:
            cursor.close()
            cursor = conn.cursor()
        except:
            conn.close()
            ip = "13.126.50.178"
            conn = psycopg2.connect(f"dbname='qdb' user='admin' host='{ip}' port='8812' password='quest'")
                    
        cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)
        cursor.execute(query_)

    df = pd.DataFrame( cursor.fetchall())
    #df.empty or
    if( len(df) <= 0):
        print(f"\nOption data Not found,- {startDate}__{endDate}__{strike}__{expiryDate}__{callPut}")
        print(df)
        return  None
    
    #=== if 1-min data is less than 330 minutes of data than data is inadequate
    
    elif(len(df)<330):
        print("Data Not GOOD")
        return None
    else:
        df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
        df = df.set_index("timestamp")
               
        
        # Check if the first timestamp is not 09:15 and handle it
        market_start_time = pd.to_datetime(f"{df.index[0].date()} 09:15:00")
        # if df.index[0] > market_start_time:
            
        #     print("\n \n Data does not start from 09:15... Please check ---" , "--- First index is : " , df.index[0]  )
           

      
        
        df = df.reindex(complete_trading_timestamps)
        
        #print( "\nAfter reindex with missing timestamps dataframe is  ", df )
                    
        
        try:
            
            if 'T' in timeframe or 'min' in timeframe or 'm' in timeframe or 'Min' in timeframe:
            # Extract numeric value from timeframe string
                timeframe2 = int(''.join(filter(str.isdigit, timeframe)))
                timeframe2 = str(timeframe2) + 'T'
            else:
                timeframe2 = timeframe
                
            df_resampled = df.resample(f"{timeframe2}", origin='start').agg({
                'OptionType': 'first',
                'Strike': 'first',
                'Expiry': 'first',
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last' ,
                'Volume': 'sum' , 
                'OI':'last'
            })
            # Filter only the valid market hours
            filtered_df = df_resampled.between_time('09:15', '15:30')
            # To drop rows which are entirely NaN (which might happen after resampling)
            df = filtered_df.dropna()
            df.index = df.index.round('s')
            # print("After resampling the data has first index as : " , df.index[0])
            # print(df.head(5))
        except Exception as e:
            print("Error while fetching one-minute data and resampling:", e)        
            return None
    return df








































#===================================================================================
    

#==== Function to fetch entire Option chain data for a range of dates based on expiry..

def fetchAllOptions(symbol,startDate,endDate, upper_strike , lower_strike,  expiryDate,callPut,timeframe):
    
    
    """
    Fetch the entire option chain for one symbol and one expiry for a range of strikes and dates
    i.e example from 2020/1/1 to 2020/1/2 find all options within a range fof strikes for some expiry
    like the option chain
    
    Returns:
    df(pd.Datafram): A dataframe containing the OHLCV , OI for the specified timeframe of options data
    """
    
    
    
    global ip
    global conn
    global cursor
    if(symbol not in ['NIFTY','BANKNIFTY','FINNIFTY','MIDCPNIFTY']):
        print('invalid symbol')
        return None

    # Format dates in 'YYYY-MM-DD' for SQL compatibility
    if isinstance(startDate, datetime.date):
        startDate = startDate.strftime('%Y-%m-%d')
    if isinstance(endDate, datetime.date):
        endDate = endDate.strftime('%Y-%m-%d')
    if isinstance(expiryDate, datetime.date):
        expiryDate = expiryDate.strftime('%d-%m-%Y')

    endDate = datetime.datetime.strptime(endDate, '%Y-%m-%d')
    endDate = endDate + datetime.timedelta(days=1)
    endDate = endDate.strftime('%Y-%m-%d')

    query_ = f"""
            with 
            meta_query  as (select  Datetime as timestamp,OptionType,Expiry,Open,High,Low,Close,Volume ,Strike,OI,Datetime from '{symbol}_OPTIONS_MODIFIED' where Datetime > '{startDate}' and Datetime < '{endDate}')
            SELECT timestamp, OptionType,Expiry,Open,  High, Low,  Close , Volume, Strike , OI from meta_query where (Expiry = '{expiryDate}') and (Strike>={lower_strike}) and (Strike<={upper_strike}) and (OptionType='{callPut}'); 
            """
        #
    try:
        #print(query_)
        cursor.execute(query_)
    except Exception as e:
        print(e)
        try:
            cursor.close()
            cursor = conn.cursor()
        except:
            conn.close()
            ip = "13.126.50.178"
            conn = psycopg2.connect(f"dbname='qdb' user='admin' host='{ip}' port='8812' password='quest'")
                    
        cursor = conn.cursor(cursor_factory=psycopg2_e.RealDictCursor)
        cursor.execute(query_)

    df = pd.DataFrame( cursor.fetchall())
       
    
    if(df.empty or len(df) <= 0):
        print(f"\nOption data Not found,- {startDate}__{endDate}__{expiryDate}__{callPut}")
        return  None
    
    #=== if 1-min data is less than 330 minutes of data than data is inadequate
    
    elif(len(df)<100):
        print("Data Not GOOD")
        return None
    else:
        df.timestamp = pd.to_datetime(df.timestamp, format="%Y-%m-%d %H:%M:%S")
        df = df.set_index("timestamp")
        #df = df[df.index.dayofweek < 5]
        try:
            
            if 'T' in timeframe or 'min' in timeframe or 'm' in timeframe or 'Min' in timeframe:
            # Extract numeric value from timeframe string
                timeframe2 = int(''.join(filter(str.isdigit, timeframe)))
                timeframe2 = str(timeframe2) + 'T'
            else:
                timeframe2 = timeframe
            
            #print("\n" , df)
            #print("\n" , df.Strike.unique())
            
            #=== group data by strikes to resample it to higher frequency
            grp = df.groupby(by=df.Strike)
            
            com_df = pd.DataFrame()
            
            for strik , groups in grp:
                #print( "\n Processing strike" , strik , groups)
                            
                df_resampled = groups.resample(f"{timeframe2}", origin='start').agg({
                    'OptionType': 'first',
                    'Strike': 'first',
                    'Expiry': 'first',
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last' ,
                    'Volume': 'sum' , 
                    'OI':'last'
                })
                # Filter only the valid market hours
                filtered_df = df_resampled.between_time('09:15', '15:30')
                # To drop rows which are entirely NaN (which might happen after resampling)
                df2 = filtered_df.dropna()
                df2.index = df2.index.round('s')
                
                com_df = pd.concat([com_df , df2])
                com_df = com_df[com_df.index.dayofweek < 5]
                #break
            #print("\n Combined dataframe " , com_df)    
                
        except Exception as e:
            print("Error while fetching one-minute data and resampling:", e)        
            return None
    return com_df


    



