#data/data.py

import os
import pandas as pd
import numpy as np
from datetime import datetime
from eodhd import APIClient
from dotenv import load_dotenv

def fetch_asset(
    api_key: str,
    symbol: str = 'BTC-USD.CC',  # Default to Bitcoin
    start_date: str = '2014-01-01',
    end_date:   str = None,
    out_path:   str = 'data/btc_2014_now.csv'
):

    # download daily adjusted close data; compute return; uses EODHD API
    # saves as CSV with columns: ['adjusted_close', 'return']
    # defaults to 2014-01-01 → today for BTC.
   
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    client = APIClient(api_key)
    raw = client.get_eod_historical_stock_market_data(
        symbol=symbol,
        period='d',                  # daily
        from_date=start_date,
        to_date=end_date,
        order='a'
    )

    df = pd.DataFrame(raw)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # compute daily returns
    df['return'] = np.log(df['adjusted_close']).diff().fillna(0)

    out_df = df[['adjusted_close', 'return']]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path)
    print(f"Saved {symbol} daily to {out_path}")

if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv("EODHD_API_KEY")
    assert api_key, "Please set EODHD_API_KEY in your environment"
    # fetches Bitcoin by default
    fetch_asset(api_key)
