from multiprocessing import Pool,Manager,pool,Process
from contextlib import redirect_stdout
import scripts.prodigy as ui
from core import upstox_func as ustox
from numpy import arange
import pandas as pd
import subprocess
import asyncio
import config
import tqdm
import sys
import os

logfile = LOG_FILE_PATH = os.path.join(ustox.directory, "parallel_results.csv")

def istarmap(self, func, iterable, chunksize=1):
    """Modified istarmap for Python 3.11+ compatibility"""
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1 or greater.")
    
    # In 3.11+, we iterate directly or use the internal task generator
    def generator():
        for item in iterable:
            yield self.apply_async(func, item).get()
            
    return generator()
pool.Pool.istarmap = istarmap

async def run_single_backtest(date_idx=None,verbose=None,log_queue=None):
    if not date_idx is None:
        local_buffer = asyncio.Queue(maxsize=5)
        try:
            # This will run the GUI, finish the sim, and then the function returns
            await ui.main(dateidx=date_idx,verbose=verbose,log_queue=log_queue,buffer=local_buffer)
        except Exception as e:
            print(f"Error during execution: {e}")
            return
    else:
        print("No date index provided.")
        return


def logger_listener(queue, file_path):
    with open(file_path, 'a', buffering=1) as f:
        while True:
            message = queue.get()
            if message == "STOP":
                break   
            f.write(f"{message}\n")

def run_process(date_idx,queue):
    config.stopevent.clear()
    config.market_close.clear()
    config.tick_ready.clear()
    from anatomy import Trader
    for trader in Trader._instances:
        trader._tickqueue = None
        trader._history_lock = None
    return asyncio.run(run_single_backtest(date_idx=date_idx,verbose=True,log_queue=queue))

def worker_unpack(args):
    return run_process(*args)

def run_simulations():
    with open(logfile, "w") as f:
        f.write("Date,Type,Sell Price,Buy Price,Points,Units,P&L\n")
        f.close()

    log_manager = Manager()
    log_queue = log_manager.Queue()
    listener = Process(target=logger_listener, args=(log_queue,logfile))
    listener.daemon = True
    listener.start()

    sim_dir = os.path.join(ustox.directory, "sim_database/")
    testdata_all = os.listdir(sim_dir)
    testdata = testdata_all
    dateidxs = [testdata_all.index(x) for x in testdata]
    tasks = [(idx, log_queue) for idx in dateidxs]
    print("Starting simulation for the dates: ", testdata_all)
    with Pool(processes=os.cpu_count()) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(worker_unpack, tasks), 
                           total=len(tasks), 
                           desc="Backtesting Progress"):
            pass
    log_queue.put("STOP")
    listener.join()

if __name__ == "__main__":
    try:
        run_simulations()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        f = pd.read_csv(logfile)
        print(f)
        print("Total points captured: ", f["Points"].sum())
        print("Gross P&L: ",f["P&L"].sum())