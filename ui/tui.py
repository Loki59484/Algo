"""Main file to run the live trading algorithm. This will be used to run the algo in production."""

from core.upstox_methods import *
from core.datatypes import *
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static, Input, RichLog, DataTable
from textual.containers import Vertical
from types import SimpleNamespace
import pytz
from datetime import datetime, time
import asyncio
from functools import wraps

ustox = UpstoxClient()
class TradingTUI(App):
    # CSS = TUI_CSS
    CSS_PATH = "layout.tcss"
    BINDINGS = [("q", "quit", "Quit Engine")]

    def __init__(self, trader, simulate=False, **kwargs):

        super().__init__(**kwargs)
        self.trader = trader
        self.bucket = trader.buckets[0]
        self.tick_buffer = asyncio.Queue(maxsize=100)
        self.portfolio_buffer = asyncio.Queue(maxsize=100)
        self.simulate = simulate
        self.simulating = False
        self.call_row_keys = []  # Keys for the call option rows in the DataTable
        self.put_row_keys = []  # Keys for the put option rows in the DataTable
        self.portfolio_routers = {
            "order": self.handle_order_update,
            "position": self.handle_position_update,
        }

    def enforce_market_hours(func):
        """
        Gatekeeper decorator: Prevents the wrapped function from running
        if the NSE/BSE market is closed, and updates the UI automatically.
        """

        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # 1. strictly define Indian Standard Time
            ist = pytz.timezone("Asia/Kolkata")
            now = datetime.now(ist)

            # 2. Define NSE trading hours (09:15 to 15:30)
            market_open = time(9, 15)
            market_close = time(15, 30)

            is_market_open = ustox.exchanges_status()

            # 3. If the market is closed, hijack the UI and block execution
            if is_market_open != "NORMAL_OPEN":
                try:
                    # Since we have 'self', we can safely reach into the Textual App!
                    log = self.query_one("#bash_log", RichLog)
                    market_pane = self.query_one("#market_data_container", Vertical)

                    log.write(f"[red]🚫 Connection Blocked: Market is currently closed")
                    market_pane.border_subtitle = "Status: 🛑 Market Closed"
                except Exception:
                    pass
                if self.simulate:
                    self.log_pane.write(f"[yellow]Simulation mode active!")
                    market_pane.border_subtitle = "Status: ⏪ SIMULATION MODE"
                    self.run_worker(self.tick_simulator())
                    return
                return

            return await func(self, *args, **kwargs)

        return wrapper

    def compose(self) -> ComposeResult:
        """This builds the UI components."""
        yield Header()

        with Vertical(
            id="market_data_container",
        ):
            yield DataTable(id="call_ticks")
            yield DataTable(id="put_ticks")

        yield Static(id="position_pane")

        yield Static(id="funds_pane")

        with Vertical(id="console_container"):
            yield RichLog(
                id="bash_log",
                highlight=True,
                markup=True,
            )
            yield Input(
                placeholder="Enter bash command here (e.g., ls, top, ps)...",
                id="bash_input",
                compact=True,
            )

        yield Static(id="order_pane")
        yield Footer()

    async def tick_simulator(self):
        """Simulates incoming tick data for testing purposes."""
        if (
            len(self.bucket.legs["CE"].historical_candles) == 0
            or len(self.bucket.legs["PE"].historical_candles) == 0
        ):
            files = list((DATA_DIR / "historical").rglob("*.parquet"))
            files = [file for file in files if "INDEX" not in str(file)]
            files.sort()
            insts = Instrument.load_multiple(source=files[-2:], lookback=2)
            bucket = Bucket(
                date=datetime.today().date(), legs={"CE": insts[0], "PE": insts[1]}
            )
            self.bucket = bucket

        call: Instrument = self.bucket.legs["CE"]
        put: Instrument = self.bucket.legs["PE"]
        if len(call.historical_candles) == 0 or len(put.historical_candles) == 0:
            self.log_pane.write("[red]No historical data available for simulation!")
            return
        self.run_worker(self.tick_processor())
        i = 0
        self.simulating = True
        try:
            while i < len(call.historical_candles) and i < len(put.historical_candles):
                simulated_ticks = {
                    call.key: call.historical_candles[i],
                    put.key: put.historical_candles[i],
                }
                i += 1
                await self.tick_buffer.put(simulated_ticks)
                await asyncio.sleep(0.5)  # Simulate real-time delay

        except Exception as e:
            self.log_pane.write(f"[red]Simulation error: {e}")

    @enforce_market_hours
    async def tick_streamer(self):
        self.run_worker(self.tick_processor())
        while True:
            self.log_pane.write("[yellow]Connecting to Upstox...[/]")
            try:
                await ustox.subscribe_ticks(
                    self.tick_buffer,
                    [instrument.key for instrument in self.bucket.legs.values()],
                )

                self.log_pane.write("[red]Broker closed connection. Reconnecting...[/]")

            except asyncio.CancelledError:
                self.log_pane.write(
                    "[bold red]✗ Stream worker received shutdown signal.[/]"
                )
                break

            except Exception as e:
                # If the network drops or throws an error, we catch it here
                self.log_pane.write(f"[red]Stream crashed: {str(e)}[/]")
            self.log_pane.write("[dim]Waiting 5 seconds before reconnect...")
            await asyncio.sleep(3)

    async def tick_processor(self):
        call_key = self.bucket.legs["CE"].key
        put_key = self.bucket.legs["PE"].key

        last_rows = {
            call_key: None,
            put_key: None,
        }  # To track the last row for each instrument

        while True:
            ticks: dict = await self.tick_buffer.get()
            if (
                last_rows.get(call_key) is not None
                or last_rows.get(put_key) is not None
                and not self.simulating
            ):
                market_pane = self.query_one("#market_data_container", Vertical)
                market_pane.border_subtitle = "Status: ✅ Connected"
            try:

                for key, tick in ticks.items():
                    if isinstance(tick, Tick):
                        ltp = f"[bold green]{tick.ltpc.ltp}[/]"
                        oi = tick.oi
                        vol = tick.ohlc_1m.volume
                        open_price = tick.ohlc_1m.open
                        high_price = tick.ohlc_1m.high
                        low_price = tick.ohlc_1m.low
                        close_price = tick.ohlc_1m.close
                        row = (
                            oi,
                            vol,
                            ltp,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                        )
                    elif isinstance(tick, Candle):
                        ltp = f"[bold green]{tick.close}[/]"
                        oi = 0
                        vol = tick.volume
                        open_price = tick.open
                        high_price = tick.high
                        low_price = tick.low
                        close_price = tick.close
                        row = (
                            oi,
                            vol,
                            ltp,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                        )
                    if row == last_rows.get(key):
                        continue  # Skip UI update if data hasn't changed
                    last_rows[key] = row  # Update the last row for this instrument
                    if key == call_key:
                        if (
                            len(self.call_row_keys) > 20
                        ):  # Limit the number of rows to prevent UI overload
                            self.call_pane.remove_row(self.call_row_keys.pop(0))
                        row_key = self.call_pane.add_row(*row)
                        self.call_row_keys.append(
                            row_key
                        )  # Store the key for potential future updates
                        self.call_pane.scroll_end(animate=False)

                    elif key == put_key:
                        if (
                            len(self.put_row_keys) > 20
                        ):  # Limit the number of rows to prevent UI overload
                            self.put_pane.remove_row(self.put_row_keys.pop(0))
                        row_key = self.put_pane.add_row(*row)
                        self.put_row_keys.append(
                            row_key
                        )  # Store the key for potential future updates
                        self.put_pane.scroll_end(animate=False)

            except Exception as e:
                logger.exception(f"Error processing tick data: {e}")
                self.log_pane.write(f"[red]Tick processor error: {e}")
            finally:
                self.tick_buffer.task_done()

            # Here you would add logic to update the UI with the new tick data

    async def portfolio_streamer(self):
        self.log_pane.write("[cyan]Initiating Backend Portfolio Stream...[/]")
        self.run_worker(self.portfolio_processor())
        while True:
            try:
                await ustox.subscribe_portfolio(self.portfolio_buffer)
            except Exception as e:
                self.log_pane.write(f"[red]Portfolio stream error: {e}[/]")
                self.log_pane.write("[dim]Retrying portfolio stream in 5 seconds...")

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
        # Start the background trading loop

        # FETCHING DIFFERENT PANES
        funds_pane = self.query_one("#funds_pane", Static)
        market_pane = self.query_one("#market_data_container", Vertical)
        console_pane = self.query_one("#console_container", Vertical)
        self.positon_pane = self.query_one("#position_pane", Static)
        self.order_pane = self.query_one("#order_pane", Static)
        self.call_pane = self.query_one("#call_ticks", DataTable)
        self.put_pane = self.query_one("#put_ticks", DataTable)

        # UPDATING PANE TITLES AND SUBTITLES
        funds_pane.border_title = "LIVE PORTFOLIO"
        market_pane.border_title = "MARKET DATA"
        console_pane.border_title = "TERMINAL"
        self.positon_pane.border_title = "OPEN POSITIONS"
        self.order_pane.border_title = "OPEN ORDERS"
        self.call_pane.border_title = f"[Bold]CALL: {self.bucket.legs['CE'].key}"
        self.call_pane.add_column("OI", width=10)
        self.call_pane.add_column("VOL", width=10)
        self.call_pane.add_column(
            "LTP",
            width=10,
        )
        self.call_pane.add_column("OPEN", width=10)
        self.call_pane.add_column("HIGH", width=10)
        self.call_pane.add_column("LOW", width=10)
        self.call_pane.add_column("CLOSE", width=10)
        self.call_pane.zebra_stripes = True
        self.call_pane.cursor_type = "row"
        self.put_pane.border_title = f"PUT: {self.bucket.legs['PE'].key}"
        self.put_pane.add_column("OI", width=10)
        self.put_pane.add_column("VOL", width=10)
        self.put_pane.add_column("LTP", width=10)
        self.put_pane.add_column("OPEN", width=10)
        self.put_pane.add_column("HIGH", width=10)
        self.put_pane.add_column("LOW", width=10)
        self.put_pane.add_column("CLOSE", width=10)
        self.put_pane.zebra_stripes = True
        self.put_pane.cursor_type = "row"
        market_pane.border_subtitle = "Status: ⭕ Not connected"

        # GREETING THE USER
        funds_pane.update("[italic]Loading portfolio data...")
        self.positon_pane.update("[italic]No open positions yet...")
        self.order_pane.update("[italic]No open orders to display...")
        self.log_pane = self.query_one("#bash_log", RichLog)
        self.log_pane.write("[green]System initialized.")

        # Start the background trading loop
        self.run_worker(self.tick_streamer())
        self.run_worker(self.portfolio_streamer())

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Captures the Enter key on the input bar and runs a bash command."""
        command = event.value
        input_widget = self.query_one("#bash_input", Input)
        log = self.query_one("#bash_log", RichLog)

        # Clear the input box
        input_widget.value = ""
        log.write(f"\n[blue]$ {command}")

        # Run the bash command asynchronously so we don't freeze the trading UI
        try:
            process = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if stdout:
                log.write(stdout.decode().strip())
            if stderr:
                log.write(f"[red]{stderr.decode().strip()}")
        except Exception as e:
            log.write(f"[red]Error executing command: {e}")


if __name__ == "__main__":
    app = TradingTUI()
    app.run()
