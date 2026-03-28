from core import upstox_func as ustox
import pandas as pd
from queue import Queue
from core import anatomy as ana
import threading
import traceback
import logging
import asyncio
import config
from os.path import exists
logger = logging.getLogger(__name__) 

plotting = False
ana.Funds.opening = ana.Funds.balance = 30000
pos = ana.Position.history
config.market_close.set()
ana.Funds.totalpnl = pos[-1]["pnl"] if len(pos) > 0 else 0
ana.Funds.opening = ana.Funds.balance = 30000

class App_cli:
    def __init__(self, dateidx=None, verbose=True,log_queue=None,buffer=None):
        self.logconsole.verbose = verbose
        self.logconsole.log_queue = log_queue
        self.buffer = buffer
        self.dateidx = dateidx
        self.data_queue = Queue()
        self.updatelock = threading.Lock()
        self.pause = True
        self.simulation = True
        self.tradethread = None
        self.traders_ok = False

    def init_variables(self):
        try:
            (
            self.keys,
            self.simdate,
            self.call_plotdata,
            self.put_plotdata,
            self.scrip_plotdata,
            self.live_index,
            self.livetotal,
            config.latency,
            ) = ana.setup_data(self.simulation, self.dateidx)
            config.latency = 0.0005
        except TypeError as e:
            logger.critical(e,stack_info=True,exc_info=True)
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            print("Fatal error. Exiting simultaion")
            exit()

    class logconsole:
        verbose = True
        _logfile = None
        log_queue = None

        @staticmethod
        def log_info(msg):
            if App_cli.logconsole.verbose:
                print("INFO : ", str(msg))

        @staticmethod
        def log_error(msg):
            if App_cli.logconsole.verbose:
                print("ERROR : ", str(msg))

        @staticmethod
        def log_debug(msg, val=None):
            if App_cli.logconsole.verbose:
                print("DEBUG : ", str(msg))
            if not val is None:
                if App_cli.logconsole.log_queue is not None:
                    App_cli.logconsole.log_queue.put(val)
                else:
                    App_cli.logconsole._logfile.writelines(f"{val}\n")
                    App_cli.logconsole._logfile.flush()

    def updatepos(self):
        pass

    def updatefunds(self, val=None):
        pass

    async def async_main(self):
        try:
            config.looprun = asyncio.Event()
            config.tick_ready = asyncio.Event()
            ana.ticks_ready = asyncio.Event()
            ana.Tester.lock = asyncio.Lock()
            
            if self.buffer is None:
                self.buffer = asyncio.Queue(maxsize=15)

            self.Trader_under = ana.Tester(
                self.keys[2],
                self.scrip_plotdata,
                gui=self,
                tag="underlying",
                simdate=self.simdate,
            )
            self.logconsole.log_info(f"Index initialized")

            self.Trader1 = ana.Tester(
                self.keys[0],
                self.call_plotdata,
                gui=self,
                tag="call",
                simdate=self.simdate,
                underlying=self.Trader_under,
            )
            self.logconsole.log_info(f"Trader1 initialized for {self.keys[0]}")
            self.Trader2 = ana.Tester(
                self.keys[1],
                self.put_plotdata,
                gui=self,
                tag="put",
                simdate=self.simdate,
                underlying=self.Trader_under,
            )
            self.logconsole.log_info(f"Trader2 initialized for {self.keys[1]}")
            if config.market_close.is_set() or self.simulation:
                tasks_to_run = [
                    ana.sim_live(
                        keys=self.keys,
                        buffer=self.buffer if self.buffer is not None else asyncio.Queue(maxsize=15),
                        current_index=self.live_index,
                        total_indices=self.livetotal,
                    ),
                    ana.publisher(buffer=self.buffer if self.buffer is not None else asyncio.Queue(maxsize=15)),
                    self.Trader1.analyse(),
                    self.Trader2.analyse(),
                    self.Trader_under.analyse(),
                ]
            else:
                tasks_to_run = [
                    ustox.get_live(
                        access_token=ustox.access_token,
                        instrument_key=self.keys,
                        output=self.buffer if self.buffer is not None else asyncio.Queue(maxsize=15),
                        write=False,
                    ),
                    ana.publisher(buffer=self.buffer if self.buffer is not None else asyncio.Queue(maxsize=15)),
                    self.Trader1.analyse(),
                    self.Trader2.analyse(),
                    self.Trader_under.analyse(),
                ]

            self.async_tasks = [asyncio.create_task(coro) for coro in tasks_to_run]
            config.looprun.set()
            await asyncio.gather(*self.async_tasks, return_exceptions=False)

        except asyncio.CancelledError:
            logger.info("Async tasks cancelled.")
            self.logconsole.log_info("Async tasks cancelled.")
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt by user.")
            config.stopevent.set()
        except Exception as e:
            logger.exception(f"Exception in App_cli | \n{e}")
        finally:
            config.stopevent.set()
            return

    def startasyncloop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.async_main())


async def main(dateidx=None, verbose=True,log_queue=None,buffer=None):
    try:
        ui = App_cli(dateidx=dateidx, verbose=verbose,log_queue=log_queue,buffer=buffer)
        ui.init_variables()
        if __name__ == "__main__":
            App_cli.logconsole._logfile = open(ustox.directory + "/logfile.csv", "w")   
            App_cli.logconsole._logfile.write("Index,Date,Type,Buy Price,Sell Price,Stoploss,Tick Close,Points,Units,P&L,Stoploss Hit,ADX,DI+,DI-\n")
            App_cli.logconsole._logfile.close()
            App_cli.logconsole._logfile = open(ustox.directory + "/logfile.csv", "a")

        await ui.async_main()
    except KeyboardInterrupt:
        for task in ui.async_tasks:
            if not task.done():
                task.cancel()
    except Exception as e:
        logger.exception(f"Exception occured in main ui loop. | \n{e}")
        tb = traceback.format_exc()
        print(tb)
        exit()
    finally:
        if not config.stopevent.is_set():
            config.stopevent.set()
        print("Main function finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main(dateidx=43))
        result_file_path = ustox.directory+'/logfile.csv'
        pd.set_option('display.max_rows', None)
        if exists(result_file_path):
            df = pd.read_csv(result_file_path, on_bad_lines="skip")
            print("\n--- Final Results ---")
            print(df)
            print("#----Stoploss Trades----#")
            print(df[df["Sell Price"] == df["Stoploss"]])
            print("#----Winning Trades----#")
            print(df[df["P&L"]>0])
            print("Total: ",df[df["P&L"]>0]["P&L"].sum())
            print("#----Losing Trades----#")
            print(df[df["P&L"]<0])
            print("Total: ",df[df["P&L"]<0]["P&L"].sum())
            print("#----DI Analysis----#")
            print(df[df["DI+"] < df["DI-"]])
            points = pd.to_numeric(df["Points"], errors="coerce").sum()
            pnl = pd.to_numeric(df["P&L"], errors="coerce").sum()
            print(f"\nTotal points captured: {points}")
            print(f"Gross P&L: {pnl}")
        else:
            print(f"Log file not found at {result_file_path}")
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        config.stopevent.set()
        exit(1)
