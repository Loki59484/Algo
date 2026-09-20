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
    TabPane,
    Input,
    Button
)
from textual.events import Key
from textual.constants import TEXTUAL_ANIMATIONS
from textual.containers import Vertical, Container, Horizontal
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

    def compose(self) -> ComposeResult:
        """This builds the UI components."""
        yield Header(show_clock=True)

        with TabbedContent():
            for idx,bucket in enumerate(self.buckets):
                idx +=1
                self.bucketattr[f'bucket_{idx}']['bucket'] = asdict(bucket)
                spot_plt = self.bucketattr[f'bucket_{idx}']['spot_plt'] = PlotextPlot(
                    id=f"spot_plot_{idx}", classes='spottickplot'
                )
                
                call_plt = self.bucketattr[f'bucket_{idx}']['call_plt'] = PlotextPlot(
                    id=f"call_plot_{idx}", classes='calltickplots'
                )
                
                put_plt = self.bucketattr[f'bucket_{idx}']['put_plt'] = PlotextPlot(
                    id=f"put_plot_{idx}", classes='puttickplots'
                )                
                ind_plt = self.bucketattr[f'bucket_{idx}']['ind_plt'] = PlotextPlot(id=f"ind_plot_{idx}", classes='indicator_plots')

                with TabPane(f"Bucket_{idx}"):
                    with Container(id=f"dashboard_{idx}",classes='dashboards'):
                        with Container(
                            id="market_data_container",classes='data_containers'
                        ) as self.bucketattr[f'bucket_{idx}']['data_container']:
                            yield spot_plt
                            yield call_plt
                            yield put_plt
                        yield Static(id="order_pane")
                        yield Static(id="funds_pane")

                        with Horizontal(id="console_row"):
                            with Vertical(id="console_container"):
                                yield RichLog(
                                    id="bash_log",
                                    highlight=True,
                                    markup=True,
                                )
                            yield ind_plt # Drops in the new vertical bars on the left
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
            try:
                
                lts_raw = getattr(tick, 'timestamp', getattr(tick, 'lts', None))
                lts = pd.to_datetime(lts_raw).strftime("%H:%M") if lts_raw else "00:00"
                close_price = getattr(tick, 'close', getattr(tick, 'close_price', None))            
                ltp = getattr(tick, 'ltp', None)
                
                if ltp is None:
                    ltp = close_price

                row = SimpleNamespace(
                    oi=getattr(tick, 'oi', 0),
                    vol=getattr(tick, 'volume', getattr(tick, 'vol', 0)),
                    ltp=ltp,
                    lts=lts,
                    open_price=getattr(tick, 'open', getattr(tick, 'open_price', close_price)),
                    high_price=getattr(tick, 'high', getattr(tick, 'high_price', close_price)),
                    low_price=getattr(tick, 'low', getattr(tick, 'low_price', close_price)),
                    close_price=close_price,
                    market_open=getattr(tick, 'market_open', True)
                )
                return row
            except Exception as e:
                raise ValueError(f"Parser failed: {e}")

        def gen_plot(row, plt, plotdata, ts, row_keys):
            is_new_candle = (len(ts) == 0) or (row.lts != ts[-1])

            if is_new_candle:
                plotdata["Open"].append(row.open_price)
                plotdata["High"].append(row.high_price)
                plotdata["Low"].append(row.low_price)
                plotdata["Close"].append(row.ltp)
                ts.append(row.lts) 
            else:
                plotdata["High"][-1] = max(plotdata["High"][-1], row.high_price)
                plotdata["Low"][-1] = min(plotdata["Low"][-1], row.low_price)
                plotdata["Close"][-1] = row.ltp

            plt.clear_data()
            if not plotdata["High"] or not plotdata["Low"]: return
            
            plt.ylim(min(plotdata["Low"]) - 3, max(plotdata["High"]) + 3)
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
                step = max(1, len(x_indices) // 5)
                plt.xticks(x_indices[::step], list(ts)[::step])
                row_keys.append(row_key)

        try:
            if isinstance(self.tick_buffer, str):
                self.context = zmq.asyncio.Context()
                self.socket = self.context.socket(SUB)
                self.socket.connect(self.tick_buffer)
                self.socket.setsockopt_string(SUBSCRIBE, "")

            # ... [Keep your existing Bucket attribute initialization here] ...
            for key,item in self.bucketattr.items():
                spot_ins = item['bucket']['spot']
                call_ins = item['bucket']['legs']["CE"]
                put_ins = item['bucket']['legs']["PE"]
                spot_key = spot_ins.key
                call_key = call_ins.key
                put_key = put_ins.key
                last_rows = {call_key: None, put_key: None}
                put_plt = item['put_plt'].plt
                call_plt = item['call_plt'].plt
                spot_plt = item['spot_plt'].plt
                for plt in [call_plt, put_plt, spot_plt]:
                    plt.clear_color()
                    plt.frame(False)

                # Initialize plot data
                item['spot_plotdata'] = {"Open": deque(maxlen=40), "High": deque(maxlen=40), "Low": deque(maxlen=40), "Close": deque(maxlen=40)}
                item['call_plotdata'] = {"Open": deque(maxlen=40), "High": deque(maxlen=40), "Low": deque(maxlen=40), "Close": deque(maxlen=40)}
                item['put_plotdata'] = {"Open": deque(maxlen=40), "High": deque(maxlen=40), "Low": deque(maxlen=40), "Close": deque(maxlen=40)}
                item['spot_ts'] = deque(maxlen=40)
                item['call_ts'] = deque(maxlen=40)
                item['put_ts'] = deque(maxlen=40)
                
                # Pre-fill historicals
                for candle in call_ins.historical_candles:
                    item['call_plotdata']["Open"].append(candle.open)
                    item['call_plotdata']["High"].append(candle.high)
                    item['call_plotdata']["Low"].append(candle.low)
                    item['call_plotdata']["Close"].append(candle.close)
                    item['call_ts'].append(pd.to_datetime(candle.timestamp).strftime("%H:%M"))
                    
                for candle in put_ins.historical_candles:
                    item['put_plotdata']["Open"].append(candle.open)
                    item['put_plotdata']["High"].append(candle.high)
                    item['put_plotdata']["Low"].append(candle.low)
                    item['put_plotdata']["Close"].append(candle.close)
                    item['put_ts'].append(pd.to_datetime(candle.timestamp).strftime("%H:%M"))

                for candle in spot_ins.historical_candles:
                    item['spot_plotdata']["Open"].append(candle.open)
                    item['spot_plotdata']["High"].append(candle.high)
                    item['spot_plotdata']["Low"].append(candle.low)
                    item['spot_plotdata']["Close"].append(candle.close)
                    item['spot_ts'].append(pd.to_datetime(candle.timestamp).strftime("%H:%M"))
                    
                item['spot_key'] = spot_key
                item['call_key'] = call_key
                item['put_key'] = put_key
                item['last_rows'] = last_rows
                item['call_row_keys'] = []
                item['put_row_keys'] = []
                item['spot_row_keys'] = []
                ind_plt = item['ind_plt'].plt
                for plt in [call_plt, put_plt, ind_plt]:
                    plt.clear_color()
                    plt.frame(False)
            await asyncio.sleep(1)
            
            while True:
                await asyncio.sleep(0.01)
                ticks = await self.socket.recv_json()     
                
                # WRAP THE LOOP IN A LOUD TRY BLOCK
                try:
                    if not isinstance(ticks, dict):
                        self.log_pane.write(f"[red]Expected Dict, got {type(ticks)}[/]")
                        continue

                    for key, tick_raw in ticks.items():
                        try:
                            # Safely load JSON or dict
                            if isinstance(tick_raw, str):
                                tick_obj = SimpleNamespace(**json.loads(tick_raw))
                            else:
                                tick_obj = SimpleNamespace(**tick_raw)
                                
                            row = parse_data(tick_obj)
                            # Catch missing OHLC data explicitly
                            if row.open_price is None or row.close_price is None:
                                self.log_pane.write(f"[yellow]Skipping {key}: Missing OHLC data -> {vars(row)}[/]")
                                continue
                                
                            matched_any = False
                            for attrs in self.bucketattr.values():
                                call_key = attrs["call_key"]
                                put_key = attrs["put_key"]
                                spot_key = attrs['spot_key']
                                if getattr(row, 'market_open', True) is False:
                                    attrs["data_container"].border_subtitle = "Status: ❗MARKET CLOSED"
                                    continue

                                attrs["data_container"].border_subtitle = "Status: ✅ Connected"

                                if row == attrs["last_rows"].get(key):
                                    continue
                                    
                                attrs["last_rows"][key] = row

                                if key == call_key:
                                    gen_plot(row, attrs['call_plt'].plt, attrs['call_plotdata'], attrs['call_ts'], attrs['call_row_keys'])
                                    attrs['call_plt'].refresh()
                                    matched_any = True
                                elif key == put_key:
                                    gen_plot(row, attrs['put_plt'].plt, attrs['put_plotdata'], attrs['put_ts'], attrs['put_row_keys'])
                                    attrs['put_plt'].refresh()
                                    matched_any = True
                                elif key == spot_key:
                                    gen_plot(row, attrs['spot_plt'].plt, attrs['spot_plotdata'], attrs['spot_ts'], attrs['spot_row_keys'])
                                    attrs['spot_plt'].refresh()
                                    matched_any = True

                            # Extract indicators (defaulting to 0 if not yet calculated)
                            rsi_val = getattr(row, 'RSI_14', getattr(tick_obj, 'RSI_14', 0))
                            adx_val = getattr(row, 'ADX_14', getattr(tick_obj, 'ADX_14', 0))
                            
                            bbu = getattr(row, 'BBU_20_2.0', getattr(tick_obj, 'BBU_20_2.0_2.0', 0))
                            bbl = getattr(row, 'BBL_20_2.0', getattr(tick_obj, 'BBL_20_2.0_2.0', 0))
                            spot_close = row.close_price
                            bb_width_raw = ((bbu - bbl) / spot_close) if spot_close > 0 else 0
                            
                            # 3. Scale BB Width by 1000 for the UI (0.068 becomes 68)
                            bb_val_scaled = bb_width_raw * 1000
                            
                            # 4. Determine Colors 
                            # (Note: BB Width turns green when it is BELOW the 0.068 threshold)
                            rsi_color = "white" if rsi_val >= 45 else "gray"
                            adx_color = "white" if adx_val >= 20 else "gray"
                            bb_color = "white" if 0 < bb_width_raw < 0.068 else "gray"
                            
                            # 5. Draw the chart
                            ind_plt_widget = attrs['ind_plt']
                            p = ind_plt_widget.plt
                            p.clear_data()
                            p.bar(["RSI", "ADX", "BBw"], [rsi_val, adx_val, bb_val_scaled], color=[rsi_color, adx_color, bb_color])
                            p.ylim(0, 100)
                            ind_plt_widget.refresh()

                            if not matched_any:
                                logger.info(f"Tick ignored. Key '{key}' doesn't match INDEX/CE/PE targets.")
                                
                        except Exception as inner_err:
                            logger.exception(f"{inner_err}")
                            with self.suspend():
                                breakpoint(header=f"{inner_err}")
                            self.log_pane.write(f"[red]Error parsing tick {key}: {inner_err}[/]")
                            
                except Exception as loop_err:
                    self.log_pane.write(f"[bold red]Loop Error: {loop_err}[/]")

        except Exception as e:
            logger.exception(f"Fatal Tick Processor Error: {e}")
            if hasattr(self, 'log_pane'):
                self.log_pane.write(f"[bold red]FATAL TICK PROCESSOR ERROR: {e}[/]")

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
            item['spot_plt'].border_title = f"INDEX: {item['bucket']['spot'].key}"
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
        #self.set_interval(2.0,self.table_refresher)
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
