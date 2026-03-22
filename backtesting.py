from functools import partial
import multiprocessing as mp
import asyncio
import pandas as pd
import tqdm
import os
import sys
import traceback
import time
import prodigy as ui
from core import upstox_func as ustox
import config
import logging

LOG_FILE_PATH = os.path.join(ustox.directory, "parallel_results.csv")
original_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
logger = logging.getLogger(__name__)

def init_worker(shared_lock):
    """Fires once per worker when the pool starts, securely setting the tqdm lock."""
    tqdm.tqdm.set_lock(shared_lock)
    
def logger_process(log_queue, file_path):

    with open(file_path, "w") as f:
        f.write(
            "Index,Date,Type,Buy Price,Sell Price,Stoploss,Tick Close,Points,Units,P&L,Stoploss Hit,ADX,DI+,DI-\n"
        )
    with open(file_path, "a", buffering=1) as f:
        while True:
            try:
                message = log_queue.get()
                if message == "STOP":
                    break
                f.write(f"{message}\n")
                f.flush()  
            except Exception as e:
                logger.exception(f"\nLogger Process Error: {e}")
            except (EOFError, BrokenPipeError, FileNotFoundError):
                break


def run_worker(date_idx, log_queue, date = None):
    from core import anatomy as ana
    start_time = time.strftime("%H:%M:%S")
    worker_id = mp.current_process()._identity[0]
    config.pbar = worker_id
    logger.debug(f"Process started for {date[date_idx]}")

    async def _run_sim():
        ana.ticks_ready = asyncio.Event()
        ana.Trader.lock = asyncio.Lock()
        config.looprun = asyncio.Event()
        config.tick_ready = asyncio.Event()
        local_buffer = asyncio.Queue(maxsize=15)
        await ui.main(
            dateidx=date_idx, verbose=True, log_queue=log_queue, buffer=local_buffer
        )
        logger.debug(f"Process completed successfully for {date[date_idx]}")
    try:
        asyncio.run(_run_sim())
        return date_idx, "Success"
    except BaseException as e:
        logger.error(f"Error while running simulation processes for {date[date_idx]} | \n{e}")
        error_trace = traceback.format_exc()
        return date_idx, f"Failed: {type(e).__name__} - {str(e)}\n{error_trace}"
    finally:
        ana.pool.shutdown(wait=False, cancel_futures=True)


def run_simulations():
    manager = mp.Manager()
    log_queue = manager.Queue()

    listener = mp.Process(target=logger_process, args=(log_queue, LOG_FILE_PATH))
    listener.daemon = True
    listener.start()

    sim_dir = os.path.join(ustox.directory, "sim_database/")
    testdata = os.listdir(sim_dir)
    testdata_slice = testdata
    dateidxs = [testdata.index(x) for x in testdata_slice]
    lock = mp.RLock()
    tqdm.tqdm.set_lock(lock)
    print(f"Starting parallel simulation for {len(testdata)} dates...",file=sys.stderr)
    worker_with_queue = partial(run_worker, log_queue=log_queue, date=testdata)
    with mp.Pool(processes=os.cpu_count(), maxtasksperchild=1, initializer=init_worker, initargs=(lock,)) as pool:
        with tqdm.tqdm(total=len(dateidxs) - 2, desc="Backtesting",position=0) as pbar:
            for result in pool.imap_unordered(worker_with_queue, dateidxs):
                completed_idx, status = result      
                if status == "Success":
                    tqdm.tqdm.write(f"✓ Index {completed_idx} completed.")
                else:
                    tqdm.tqdm.write(f"✗ Index {completed_idx} crashed: {status}")
                pbar.update(1)
    log_queue.put("STOP")
    logger.info("Stopping")
    listener.join()

if __name__ == "__main__":
    try:
        run_simulations()
        pass
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        sys.stdout = original_stdout

        pd.set_option('display.max_rows', None)
        if os.path.exists(LOG_FILE_PATH):
            df = pd.read_csv(LOG_FILE_PATH, on_bad_lines="skip")
            print("\n--- Final Results ---")
            print(df.sort_values(by=["Date", "Index"]))
            print("#----Stoploss Trades----#")
            print(df[df["Sell Price"] == df["Stoploss"]])
            print("#----Winning Trades----#")
            print(df[df["P&L"]>0])
            print("Total: ",df[df["P&L"]>0]["P&L"].sum())
            print("#----Losing Trades----#")
            print(df[df["P&L"]<0])
            print("Total: ",df[df["P&L"]<0]["P&L"].sum())
            points = pd.to_numeric(df["Points"], errors="coerce").sum()
            pnl = pd.to_numeric(df["P&L"], errors="coerce").sum()
            print(f"\nTotal points captured: {points}")
            print(f"Gross P&L: {pnl}")
        else:
            print(f"Log file not found at {LOG_FILE_PATH}")
