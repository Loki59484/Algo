"""Main file to run the live trading algorithm. This will be used to run the algo in production."""
from functools import partial
from textual.app import App, ComposeResult
from textual import work
from textual.widgets import (
    Header,
    Footer,
    Static,
    RichLog,
    TabbedContent,
    DataTable,
    TabPane,
    Input,
    Button
)
from textual.events import Key
from textual.constants import TEXTUAL_ANIMATIONS
from textual.containers import Vertical, Container
from types import SimpleNamespace
from textual_plotext import PlotextPlot
import textual_pandas as tpd
from zmq import SUB, SUBSCRIBE
import zmq.asyncio
import asyncio
import json
import sys
from pathlib import Path
import logging
import os
import io
import pandas as pd
import numpy as np
from collections import deque,defaultdict


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from core.datatypes import *
from core.anatomy import Trader
from core.upstox_methods import *

os.environ["TEXTUAL_LOG"] = str(ROOT_DIR / "logs" / "logs.log")
logger = logging.getLogger(__name__)


class TradingTUI(App):
    # CSS = TUI_CSS
    CSS_PATH = "layout.tcss"
    BINDINGS = [("q", "quit", "Quit Engine")]

    def __init__(self, trader: Trader, buckets: Bucket | list[Bucket], simulate=False,client: UpstoxClient = None, **kwargs):
        super().__init__()
        self.trader = trader
        self.buckets = [buckets] if isinstance(buckets,Bucket) else buckets
        self.tick_buffer = kwargs.get("port", asyncio.Queue(maxsize=100))
        self.portfolio_buffer = asyncio.Queue(maxsize=100)
        self.simulate = simulate
        self.simulating = False
        self.portfolio_routers = {
            "order": self.handle_order_update,
            "position": self.handle_position_update,
        }
        self.context: zmq.asyncio.Context | None = None
        self.socket: zmq.asyncio.Context.socket | None = None
        self.bucketattr: defaultdict = defaultdict(dict )
        self.cmd_history_idx= 0
        self.cmd_history = []
        self.local_env = {'self': self.app}
        self.ustox = UpstoxClient() if client is None else client

    def set_breakpoint(self):
        with self.suspend():
            breakpoint(header='TUI Breakpoint')


    def compose(self) -> ComposeResult:
        """This builds the UI components."""
        yield Header(show_clock=True)

        with TabbedContent():
            for idx,bucket in enumerate(self.buckets):
                idx +=1
                self.bucketattr[f'bucket_{idx}']['bucket'] = asdict(bucket)
                call_plt = self.bucketattr[f'bucket_{idx}']['call_plt'] = PlotextPlot(id=f"call_plot_{idx}", classes='tickplots')
                put_plt = self.bucketattr[f'bucket_{idx}']['put_plt'] = PlotextPlot(id=f"put_plot_{idx}", classes='tickplots')
                with TabPane(f"Bucket_{idx}"):
                    with Container(id=f"dashboard_{idx}",classes='dashboards'):
                        with Vertical(
                            id=f"market_data_container",classes='data_containers'
                        ) as self.bucketattr[f'bucket_{idx}']['data_container']:
                            yield call_plt
                            yield put_plt
                        yield Static(id="order_pane")
                        yield Static(id="funds_pane")

                        with Vertical(id="console_container"):
                            yield RichLog(
                                id="bash_log",
                                highlight=True,
                                markup=True,
                            )
                        yield Static(id="position_pane")

            with TabPane("Analysis", id="trade_analysis"):
                yield Button('Refresh',compact= True,id='refresh_table')
                yield tpd.DataFrameTable(id='recent-trades')
                with Vertical(id="analysis_console_container"):
                    yield RichLog(id="analysis_log",
                                    highlight=True,
                                    markup=True,
                                )
                    yield Input(
                        placeholder=">>>",
                        id="analysis_input"
                    )
            with TabPane("Settings", id="tab_settings"):
                yield Static(
                    "This is the second tab. Add your settings or alternative views here."
                )

        yield Footer()

    def on_button_pressed(self,event: Button.Pressed):
        if event.button.id == 'refresh_table':
            self.analysis_table.clear()
            df = self.trader.portfolio.get_report()
            if df.empty:
                self.analysis_table.border_subtitle = "No trades executed yet"
                return
            self.analysis_table.border_subtitle = ''    
            self.analysis_table.add_df(df)

    @work
    async def table_refresher(self):
            self.on_button_pressed(Button.Pressed(self.query_one('#refresh_table',Button)))
        
    def on_key(self, event:Key):
        if event.key == 'up':
            self.cmd_history_up()
        elif event.key == 'down':
            self.cmd_history_down()

    def cmd_history_up(self):
        input_widget = self.query_one("#analysis_input", Input)
        if self.cmd_history and self.cmd_history_idx > 0:
            self.cmd_history_idx-=1
            input_widget.value = f"{self.cmd_history[self.cmd_history_idx]}"
            
    def cmd_history_down(self):
        input_widget = self.query_one("#analysis_input", Input)
        if self.cmd_history and self.cmd_history_idx >= len(self.cmd_history)-1:
            input_widget.value = ''
            return
        if self.cmd_history and self.cmd_history_idx < len(self.cmd_history)-1:
            self.cmd_history_idx+=1
            input_widget.value = f"{self.cmd_history[self.cmd_history_idx]}"
        
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        
        command = event.value
        input_widget = self.query_one("#analysis_input", Input)
        log = self.query_one("#analysis_log", RichLog)

        # Clear the input box
        input_widget.value = ""
        log.write(f"[cyan]>>> {command}")
        if not command.strip():
            return

        if not self.cmd_history or self.cmd_history[-1]!=command:
            self.cmd_history.append(command)
        self.cmd_history_idx = len(self.cmd_history) 
        # Run the bash command asynchronously so we don't freeze the trading UI
        if command == 'clear':
            log.clear()
            return

        old_stderr,old_stdout = sys.stderr,sys.stdout
        redirected_out = io.StringIO()
        sys.stderr = sys.stdout = redirected_out
        try:
            try:
                result = eval(command,globals(),self.local_env)
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                exec(command, globals(),self.local_env)
        except Exception as e:
            log.write(f"[red]Error executing command: {e}")
        finally:
            sys.stderr,sys.stdout = old_stderr,old_stdout 
        output = redirected_out.getvalue()
        log.write(output)


    @work(exit_on_error=True)
    async def tick_processor(self):
        
        def parse_data(tick):
            #with self.suspend():
            #    breakpoint()
            if isinstance(tick, Tick):
                ltp = f"[bold green]{tick.ltpc.ltp}[/]"
                oi = tick.oi
                lts = pd.to_datetime(tick.timestamp).strftime("%H:%M")
                vol = tick.ohlc_1m.volume
                open_price = tick.ohlc_1m.open
                high_price = tick.ohlc_1m.high
                low_price = tick.ohlc_1m.low
                close_price = tick.ohlc_1m.close
                market_open = tick.market_open
                row = SimpleNamespace(
                    oi=oi,
                    vol=vol,
                    ltp=int(ltp),
                    lts=lts,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                    market_open=market_open,
                )

            elif isinstance(tick, (Candle, SimpleNamespace)):
                ltp = tick.ltp if tick.ltp else tick.close
                lts = pd.to_datetime(tick.timestamp).strftime("%H:%M")
                oi = 0
                vol = tick.volume 
                open_price = tick.open
                high_price = tick.high
                low_price = tick.low
                close_price = tick.close
                row = SimpleNamespace(
                    oi=oi,
                    vol=vol,
                    ltp=ltp,
                    lts=lts,
                    open_price=open_price,
                    high_price=high_price,
                    low_price=low_price,
                    close_price=close_price,
                )
                return row

        def gen_plot(
            row: SimpleNamespace,
            plt: PlotextPlot.plt,
            plotdata: dict,
            ts: list,
            row_keys: list,
        ):
            if row.ltp == row.close_price:
                plotdata["Open"].append(row.open_price)
                plotdata["High"].append(row.high_price)
                plotdata["Low"].append(row.low_price)
                plotdata["Close"].append(row.close_price)
                plotdata["Close"].append(row.close_price)
            else:
                logger.info("Replacing ltp.")
                plotdata["Open"][-1]=row.open_price
                plotdata["High"][-1]=row.high_price
                plotdata["Low"][-1]=row.low_price
                plotdata["Close"][-1]=row.ltp

            ts.append(row.lts)
            plt.clear_data()
            plt.ylim(min(plotdata["Low"])-3 , max(plotdata["High"])+3)

            x_indices = list(range(0, len(plotdata["Open"]), 1))

            if len(x_indices) > 0:
                plt.xlim(-1, len(x_indices))

            row_key = plt.candlestick(
                dates=x_indices,
                data={
                    "Open": list(plotdata["Open"]),
                    "High": list(plotdata["High"]),
                    "Low": list(plotdata["Low"]),
                    "Close": list(plotdata["Close"]),
                },
                colors=["white", "gray"],
            )
            plt.xticks(x_indices[::5], list(ts)[::5])
            row_keys.append(row_key)

        try:
            if isinstance(self.tick_buffer, str):
                self.context = zmq.asyncio.Context()
                self.socket = self.context.socket(SUB)
                self.socket.connect(self.tick_buffer)
                self.socket.setsockopt_string(SUBSCRIBE, "")

            for key,item in self.bucketattr.items():
                call_ins = item['bucket']['legs']["CE"]
                put_ins = item['bucket']['legs']["PE"]
                call_key = item['bucket']['legs']["CE"].key
                put_key = item['bucket']['legs']["PE"].key
                last_rows = {
                    call_key: None,
                    put_key: None,
                }
                put_plt = item['put_plt'].plt
                call_plt = item['call_plt'].plt

                for plt in [call_plt, put_plt]:
                    plt.clear_color()
                    plt.frame(False)

                call_plotdata = {
                    "Open": deque(maxlen=40),
                    "High": deque(maxlen=40),
                    "Low": deque(maxlen=40),
                    "Close": deque(maxlen=40),
                }
                call_ts = deque(maxlen=40)

                for candle in call_ins.historical_candles:
                    call_plotdata["Open"].append(candle.open)
                    call_plotdata["High"].append(candle.high)
                    call_plotdata["Low"].append(candle.low)
                    call_plotdata["Close"].append(candle.close)
                    call_ts.append(pd.to_datetime(candle.timestamp).strftime("%H:%M"))

                put_plotdata = {
                    "Open": deque(maxlen=40),
                    "High": deque(maxlen=40),
                    "Low": deque(maxlen=40),
                    "Close": deque(maxlen=40),
                }
                put_ts = deque(maxlen=40)
                for candle in put_ins.historical_candles:
                    put_plotdata["Open"].append(candle.open)
                    put_plotdata["High"].append(candle.high)
                    put_plotdata["Low"].append(candle.low)
                    put_plotdata["Close"].append(candle.close)
                    put_ts.append(pd.to_datetime(candle.timestamp).strftime("%H:%M"))

                item['call_key']=call_key
                item['put_key']=put_key
                item['last_rows']=last_rows
                item['call_plotdata']=call_plotdata
                item['call_ts']=call_ts
                item['put_plotdata']=put_plotdata
                item['put_ts']=put_ts
                item['call_row_keys']=[]
                item['put_row_keys']=[]
                
            await asyncio.sleep(1)
            while True:
                await asyncio.sleep(0.01)
                ticks: dict = await self.socket.recv_json()                
                try:
                    for key, tick in ticks.items():
                        tick = SimpleNamespace(**json.loads(tick))
                        row = parse_data(tick)
                        if any(val is None for val in vars(row).values()):
                            continue
                        for attrs in self.bucketattr.values():
                            
                            call_key = attrs["call_key"]
                            put_key = attrs["put_key"]
                            data_container = attrs["data_container"]
                            last_rows = attrs["last_rows"]
                            call_row_keys = attrs["call_row_keys"]
                            put_row_keys = attrs["put_row_keys"]
                            call_plt = attrs['call_plt']
                            call_plotdata = attrs['call_plotdata'] 
                            call_ts = attrs['call_ts']
                            put_plt = attrs['put_plt']
                            put_plotdata = attrs['put_plotdata']
                            put_ts = attrs['put_ts']
                            market_status = getattr(row,'market_open',None)
                            if market_status is not None and not market_status:
                                self.log_pane.write("NSE_FO market is closed!.")
                                data_container.border_subtitle = "Status:❗MARKET CLOSED"
                                break


                            if (
                                last_rows.get(call_key) is not None
                                or last_rows.get(put_key) is not None
                                and not self.simulating
                            ):
                                data_container.border_subtitle = "Status: ✅ Connected"

                            if row == last_rows.get(key):
                                logger.debug("Skipped updating identical ticks.")
                                continue
                            last_rows[key] = row

                            if key == call_key:
                                gen_plot(
                                    row, call_plt.plt, call_plotdata, call_ts, call_row_keys
                                )
                            elif key == put_key:
                                gen_plot(row, put_plt.plt, put_plotdata, put_ts, put_row_keys)
                            
                            call_plt.refresh()
                            put_plt.refresh()


                except Exception as e:
                    logger.exception(f"Error processing tick data: {e}")
                    self.log_pane.write(f"[red]Tick processor error: {e}")

        except Exception as e:
            logger.exception(f"Error in tick processor {e}")

    @work(exit_on_error=True)
    async def portfolio_streamer(self):
        self.log_pane.write("[cyan]Initiating Backend Portfolio Stream...[/]")
        self.portfolio_processor()
        while True:
            try:
                await self.ustox.subscribe_portfolio(self.portfolio_buffer)
            except Exception as e:
                self.log_pane.write(f"[red]Portfolio stream error: {e}[/]")
                self.log_pane.write("[dim]Retrying portfolio stream in 5 seconds...")

    @work(exit_on_error=True)
    async def portfolio_processor(self):
        self.log.info("Started portfolio update processor.")
        while True:
            update = await self.portfolio_buffer.get()
            update = SimpleNamespace(**update)
            try:
                if update.update_type in self.portfolio_routers:
                    handler = self.portfolio_routers[update.update_type]
                    if handler:
                        await handler(update)
                else:
                    self.log_pane.write(
                        f"[dim]Ignored unmapped portfolio update: {update.update_type}[/dim]"
                    )
            except Exception as e:
                self.log_pane.write(
                    f"[red]Error processing portfolio update: {e}[/red]"
                )
            finally:
                self.portfolio_buffer.task_done()

    async def handle_order_update(self, update: SimpleNamespace):
        self.log_pane.write("[blue]Order update recieved![/]")
        udpate_str = f"""
[yellow]Timestamp[/]: {update.order_timestamp}\n
[yellow]Order ID[/]: {update.order_id}\n
[yellow]Order status[/]: {f"[blue]{update.status}" if update.status == 'complete' else f"[yellow]{update.status}" if update.status == 'open' else f"[red]{update.status}" }[/]\n
[yellow]Instrument[/]: {update.trading_symbol}\n
[yellow]Filled Quantity[/]: {update.filled_quantity}\n
[yellow]Pending Quantity[/]": {update.pending_quantity}
"""
        self.order_pane.update(udpate_str)

    async def handle_position_update(self, update):
        update_str = f"""
[yellow]Instrument Key:[/] {update.instrument_key}\n
[yellow]Average Price:[/] {update.average_price}\n
[yellow]Buy Value:[/] {update.buy_value}\n
[yellow]Multiplier:[/] {update.multiplier}\n
[yellow]Quantity:[/] {update.quantity}\n
[yellow]Product:[/] {update.product}\n
[yellow]Sell Value:[/] {update.sell_value}\n
[yellow]Buy Price:[/] {update.buy_price}\n
[yellow]Sell Price:[/] {update.sell_price}
"""
        self.positon_pane.update(update_str)

    def on_mount(self) -> None:
        """Called when the app starts."""
        self.title = "Trading Engine"
        funds_pane = self.query_one("#funds_pane", Static)
        for item in self.bucketattr.values():

            item['data_container'].border_title = "MARKET DATA"
            item['data_container'].border_subtitle = "Status: ⭕ Not connected"
            item['call_plt'].border_title = f"CALL: {item['bucket']['legs']['CE'].key}"
            item['put_plt'].border_title = f"PUT: {item['bucket']['legs']['PE'].key}"

        console_pane = self.query_one("#console_container", Vertical)
        self.positon_pane = self.query_one("#position_pane", Static)
        self.order_pane = self.query_one("#order_pane", Static)

        # UPDATING PANE TITLES AND SUBTITLES
        funds_pane.border_title = "LIVE PORTFOLIO"
        console_pane.border_title = "LOG CONSOLE"
        self.positon_pane.border_title = "OPEN POSITIONS"
        self.order_pane.border_title = "OPEN ORDERS"


        # GREETING THE USER
        funds_pane.update("[italic]Loading portfolio data...")
        self.positon_pane.update("[italic]No open positions yet...")
        self.order_pane.update("[italic]No open orders to display...")
        self.log_pane = self.query_one("#bash_log", RichLog)
        self.log_pane.write("[green]System initialized.")
        self.analysis_table = self.query_one('#recent-trades',tpd.DataFrameTable)
        self.analysis_console = self.query_one('#analysis_console_container',Vertical)
        self.analysis_console.border_title = 'ANALYISIS CONSOLE'
        

    def on_ready(self):
        # Start the background trading loop
        self.portfolio_streamer()
        self.set_interval(2.0,self.table_refresher)
        self.tick_processor()

    def on_unmount(self) -> None:
        """Gracefully shutdown the app
        Returns
        -------
        None
        """
        self.socket.close()
        self.context.term()
        os.system("cls" if os.name == "nt" else "clear")


if __name__ == "__main__":
    app = TradingTUI()
    app.run()
