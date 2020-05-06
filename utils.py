import os
import pandas as pd

def get_data(file='Sweden Load Data 2005-2017.csv'):
    df = pd.read_csv(file)
    df.rename({'cet_cest_timestamp':'time', 'SE_load_actual_tso':'load'},
            axis='columns',
            inplace=True)
    df['time'] = pd.to_datetime(df['time'],errors='ignore', utc=True)
    df['weekday'] = df['time'].dt.weekday
    return df

def make_dir(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)
