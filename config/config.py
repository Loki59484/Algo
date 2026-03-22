# configuration file for anatomy.py and gui.py

import asyncio
import threading
from core import upstox_func as ustox
from concurrent.futures import ThreadPoolExecutor
import datetime as dt

sandbox_orders = True
stopevent = threading.Event()
market_close = asyncio.Event()
tick_ready = threading.Event()
looprun = asyncio.Event()
log_queue = None
latency: float = 0.5
current_index = 0
previous_target = 1157611
target_profit = 77
prev_days = 50
prev_starting = 452
units = lambda balance, close: max(
    65, int((balance / close) - ((balance / close) % 65))
)
pbar = 0

if __name__ == "__main__":

    import os
    import pandas as pd
    import traceback as tb



    """
    expiry_date = ustox.get_expiry(ustox.get_options())
    print("Trading options with expiry on: ", expiry_date)
    suitable_puts, suitable_calls = ustox.get_suitable(
        expiry_date,
        funds=300000,
    )
    keys = [
    suitable_calls[0]["instrument_key"],
    suitable_puts[0]["instrument_key"],
    "NSE_INDEX|Nifty 50",
    ]
    for key in keys:
        targdir = ustox.directory + '/sim_database/'+expiry_date
        os.makedirs(targdir,exist_ok=True)
        history = ustox.get_historical(
                dtype="intraday",
                instrument_key=key,
                expiry_date=expiry_date,
            )
        history.to_json(targdir+'/'+key+'.json')
    exit()


    """
    count = 0
    simdir = ustox.directory + "/sim_database/"
    dates = os.listdir(simdir)
    # dates.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
    for date in dates:
        datedir = ustox.directory + "/sim_database/" + date
        files = os.listdir(datedir)
        if not "NSE_INDEX.json" in files:
            print(date)
            date_dt = dt.datetime.strptime(date,"%Y-%m-%d").date()
            data = ustox.get_historical(to_date=date_dt,from_date=date_dt)
            save_dir = simdir + "/" + date
            data.to_json(save_dir+'/'+ "NSE_INDEX.json")
            continue
        for file in files:
            count += 1
            print(count," Processing: ",file,end='\r')
            filepath = datedir + "/" + file
            df = pd.read_json(filepath)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["date"] = df["timestamp"].dt.date
            grouped = df.groupby("date")
            for date_obj, dailydf in grouped:
                date_str = date_obj.strftime(format="%Y-%m-%d")
                save_dir = simdir + "/" + date_str
                os.makedirs(save_dir, exist_ok=True)
                dailydf.to_json(save_dir+'/'+file)
            print(count," Processed: ",file,end='\r')

    
 
