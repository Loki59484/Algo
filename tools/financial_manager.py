import sys
import json
import logging
import time
import math
from pathlib import Path

from textual_pandas.widgets import DataFrameTable
from textual import work
from textual.screen import ModalScreen
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Center, Grid, Container
from textual.widgets import (
    Header,
    Footer,
    Static,
    Input,
    Button,
    RichLog,
    Label,
    Tree,
    DataTable,
    TabbedContent,
    TabPane,

)

from textual.widget import Widget
from rich.text import Text

# Adding root directory to sys.path for module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
TOOL_DIR = ROOT_DIR / "tools"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from core.datatypes import Funds
from core.upstox_methods import UpstoxClient, AWS_TAILSCALE_IP
from tools.charges_calculator import charges_calculator
from planner import Planner
logger = logging.getLogger(__name__)
STATE_FILE = TOOL_DIR / "finance_state.json"
ustox = UpstoxClient()

# -----------------------------------------------------------------------------
# CUSTOM WIDGETS & MODALS
# -----------------------------------------------------------------------------

class Pie(Widget):
    """A custom pie chart widget that mathematically corrects terminal aspect ratios."""

    def __init__(self, data: dict, radius: int = 10, marker="⣿", **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.radius = radius
        self.marker = marker[0]
        self.pie_colors = ["#ff0055", "#aaff00", "#00ffff", "#ffff00"]

    def render(self) -> Text:
        total = sum(self.data.values())
        if total == 0:
            return Text("No funds available to display.", style="dim", justify="center")

        slices = []
        current_angle = 0
        for key, value in self.data.items():
            fraction = value / total
            angle = fraction * 2 * math.pi
            slices.append((key, current_angle, current_angle + angle, value))
            current_angle += angle

        result = Text()
        result.justify = "center"
        aspect_ratio = 2.0
        width = int(self.radius * aspect_ratio)

        for y in range(-self.radius, self.radius + 1):
            for x in range(-width, width + 1):
                nx = x / aspect_ratio
                distance = math.sqrt(nx**2 + y**2)

                if distance <= self.radius + 0.1:
                    angle = math.atan2(y, nx)
                    if angle < 0:
                        angle += 2 * math.pi

                    char_color = "white"
                    for i, (name, start, end, val) in enumerate(slices):
                        if start <= angle <= end:
                            char_color = self.pie_colors[i % len(self.pie_colors)]
                            break

                    result.append(self.marker, style=char_color)
                else:
                    result.append(" ")
            result.append("\n")

        result.append("\n")
        for i, (name, start, end, val) in enumerate(slices):
            color = self.pie_colors[i % len(self.pie_colors)]
            result.append(f"■ {name} (₹{val:,.2f})   ", style=color)

        return result


class SetupScreen(ModalScreen[dict]):
    """A popup screen to ask for initial balances if no save file exists."""
    CSS_PATH = "SetupScreen.tcss"

    def compose(self) -> ComposeResult:
        with Vertical(id="setup_dialog"):
            with Center():
                yield Label("[bold yellow]Corporate Setup initialization[/bold yellow]\nPlease establish your company parameters:\n")
                with Grid(id="data_grid"):
                    yield Input(placeholder="Name", id="init_user", classes="setup-input")
                    yield Input(placeholder="Current Debt (e.g., 1900000)", id="init_debt", classes="setup-input")
                    yield Input(placeholder="AUM (e.g., 1000000)", id="init_capital", classes="setup-input")
                    yield Input(placeholder="Financial Target (e.g., 2000000)", id="init_target", classes="setup-input")
                    yield Input(placeholder="Savings (e.g., 400000)", id="init_savings", classes="setup-input")
                    yield Input(placeholder="Expected Base Pay (e.g., 50000/month)", id="init_base_pay", classes="setup-input")

            with Center():
                with Horizontal(id="control_container"):
                    yield Button("Establish Corporation", id="btn_save_setup", variant="success", classes="control_buttons")
                    yield Button("Fetch from Upstox", id="btn_fetch_upstox", variant="success", classes="control_buttons")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "btn_save_setup":
            try:
                debt_val = self.query_one("#init_debt", Input).value
                capital_val = self.query_one("#init_capital", Input).value
                savings_val = self.query_one("#init_savings", Input).value
                target_val = self.query_one("#init_target", Input).value
                base_pay_val = self.query_one("#init_base_pay", Input).value

                new_state = {
                    "user": self.query_one("#init_user", Input).value or "User",
                    "debt": float(debt_val) if debt_val else 0.0,
                    "trading_capital": float(capital_val) if capital_val else 0.0,
                    "savings": float(savings_val) if savings_val else 0.0,
                    "unrealized_profit": 0.0,
                    "target": float(target_val) if target_val else 0.0,
                    "base_pay": float(base_pay_val) if base_pay_val else 50000.0,
                }
                self.dismiss(new_state)
            except ValueError:
                self.query_one(Label).update("[bold red]Error: Please enter valid numbers![/bold red]")

        if event.button.id == "btn_fetch_upstox":
            try:
                capital_input = self.query_one("#init_capital", Input)
                capital_input.value = "30000"
            except Exception:
                self.query_one(Label).update("[bold red]Error in fetching upstox data. Please enter manually.[/bold red]")


# -----------------------------------------------------------------------------
# MAIN DASHBOARD
# -----------------------------------------------------------------------------

class CFOTracker(App):
    TITLE = "Algo's Personal CFO"
    CSS_PATH = "cfo_tracker.tcss"
    BINDINGS = [("ctrl+q", "quit", "Quit the Application")]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_shutting_down = False
        self.incoming_data = {"Live Trader": 0, "Risk Monitor": 0, "AWS Server": 0}

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Dashboard"):
                with Container(id="dashboard"):
                    with Horizontal():
                        yield Static(id="debt_display", classes="balance-box alert")
                        yield Static(id="capital_display", classes="balance-box")
                        yield Static(id="savings_display", classes="balance-box")
                        yield Static(id="profit_display", classes="balance-box")
                        yield Static(id="target_display", classes="balance-box")
                        yield Static(id="base_pay_display", classes="balance-box")

                with Horizontal():
                    with Grid(id="data_grid"):
                        yield Pie(data={"Trading Capital": 0, "Savings": 0}, radius=7, id="division_plot", classes="plotext_plot")

                        # REFACTOR 1: A single Consolidated Tree for Financial Ledger (Zero UI lag)
                        with Static(id="history", classes="data_panes"):
                            with Vertical():
                                yield Tree("Financial Ledger", id="recents_tree")
                                with Horizontal(id="tree_control_buttons"):
                                    yield Button("Expand All", id="btn_toggle_trees", classes="tree_buttons")

                        # REFACTOR 2: Professional Investment DataTable
                        with Static(id="investments", classes="data_panes"):
                            yield DataTable(id="investments_table")

                        with Static(id="monitor", classes="data_panes"):
                            yield DataTable(id="system_status_table")

                    with Vertical():
                        with Container(id="log_container"):
                            yield RichLog(id="activity_log", highlight=True, markup=True, max_lines=100, auto_scroll=True)
                        yield Static(id="package")
            with TabPane("Planner"):
                with Container():
                    with Horizontal():
                        yield DataFrameTable(id='plan_table',zebra_stripes = True)
        yield Footer()

    def on_mount(self):
        status_online = "🟢 Online"
        status_offline = "🔴 Offline"
        log = self.query_one(RichLog)
        planner = Planner()
        args = planner.setup_cli()
        plan_df = planner.create(args)

        if not STATE_FILE.exists() or STATE_FILE.stat().st_size <= 0:
            self.push_screen(SetupScreen(), self.init_state_callback)
        else:
            self.state = self.load_state()
            self.update_ui()
            log.write(f"[bold cyan]Hello {self.state.get('user', 'User')}, This is your personal CFO![/bold cyan]")

        # Apply borders
        self.query_one("#dashboard", Container).border_title = "Financial Summary"
        self.query_one("#division_plot", Pie).border_title = "Division of Funds"
        self.query_one("#history", Static).border_title = "Recents Ledger"
        self.query_one("#investments", Static).border_title = "Current Investments"
        self.query_one("#monitor", Static).border_title = "System Monitor"
        self.query_one("#log_container", Container).border_title = "Logs"
        self.query_one("#package", Static).border_title = "Financial Package"
        self.query_one("#plan_table", DataFrameTable).add_df(plan_df)

        # Initialize the Recents Tree
        recents_tree = self.query_one("#recents_tree", Tree)
        recents_tree.root.expand()
        self.t_expenses = recents_tree.root.add("Expenses", expand=False)
        self.t_payouts = recents_tree.root.add("Payouts", expand=False)
        self.t_payins = recents_tree.root.add("Payins", expand=False)
        self.t_wants = recents_tree.root.add("Wants", expand=False)

        inv_table = self.query_one("#investments_table", DataTable)
        inv_table.cursor_type = "row"
        
        # FIX 1: Explicitly set the keys for the Investment columns
        inv_table.add_column("Asset Class", key="Asset Class")
        inv_table.add_column("Invested (₹)", key="Invested (₹)")
        inv_table.add_column("Current (₹)", key="Current (₹)")
        inv_table.add_column("Status/P&L", key="Status/P&L")
        inv_table.add_column("Duration/Summary", key="Duration/Summary")
        
        # We assign keys to these rows so we can update them seamlessly later
        inv_table.add_row("Mutual Funds", "0.0", "0.0", "-", "Long Term", key="mf")
        inv_table.add_row("Stocks", "0.0", "0.0", "-", "Swing/Hold", key="stk")
        inv_table.add_row("Options", "0.0", "0.0", "-", "Awaiting Sync", key="opt")
        inv_table.add_row("Fixed Deposits", "0.0", "0.0", "-", "Locked", key="fd")

        # Initialize System Monitor DataTable
        monitor_table = self.query_one("#system_status_table", DataTable)
        monitor_table.cursor_type = "none" 
        
        # FIX 2: Explicitly set the keys for the Monitor columns
        monitor_table.add_column("Component", key="Component")
        monitor_table.add_column("Status", key="Status")
        monitor_table.add_column("Last Seen", key="Last Seen")     
        
        monitor_table.add_row("Live Trader", status_offline, "Never", key="live_trader")
        monitor_table.add_row("Risk Monitor", status_offline, "Never", key="risk_monitor")
        monitor_table.add_row("AWS Server", status_offline, "Never", key="aws_server")
        # Map your dictionary to those exact string keys
        self.system_components = {
            "Live Trader": {"row_key": "live_trader"},
            "Risk Monitor": {"row_key": "risk_monitor"},
            "AWS Server": {"row_key": "aws_server"}
        }

        self.incoming_data = {"Live Trader": 0, "Risk Monitor": 0, "AWS Server": 0}

        self.zmq_listener()
        self.set_interval(1.0, self.check_heartbeats)
        
        # ADD THIS: Run the API fetch in the background once on startup
        self.sync_investment_data()
        
        # ADD THIS (Optional): Automatically refresh the charges from the API every 1 hour (3600 seconds)
        self.set_interval(3600.0, self.sync_investment_data)    
    
    def _log_remote_error(self, component: str, msg: str):
        log_widget = self.query_one(RichLog)
        log_widget.write(f"[bold red][{component} ALERT][/bold red] {msg}")

    @work(thread=True)
    def zmq_listener(self):
        import zmq
        import time

        context = zmq.Context.instance()
        sub_socket = context.socket(zmq.SUB)

        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5557")
        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5558")
        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5567")
        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5568")

        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        while not self._is_shutting_down:
            try:
                message = sub_socket.recv_string(flags=zmq.NOBLOCK)
                
                if message.startswith("PING:"):
                    component_name = message.split(":")[1]
                    self.incoming_data[component_name] = time.time()
                elif message.startswith("ERROR:"):
                    parts = message.split(":", 2)
                    if len(parts) == 3:
                        component = parts[1]
                        error_text = parts[2]
                        self.call_from_thread(self._log_remote_error, component, error_text)

            except zmq.Again:
                time.sleep(0.1)
            except Exception as e:
                self.call_from_thread(self._log_remote_error, "System", str(e))
                time.sleep(1)

        sub_socket.close()
        context.term()

    def init_state_callback(self, new_state: dict):
        self.state = new_state
        self.save_state()
        self.update_ui()
        log = self.query_one(RichLog)
        log.write("[bold green]Corporate setup complete. Data initialized![/bold green]")
        log.write(f"[bold cyan]Hello {self.state.get('user', 'User')}, This is your personal CFO![/bold cyan]")

    def check_heartbeats(self):
        monitor_table = self.query_one("#system_status_table", DataTable)
        current_time = time.time()
        
        for name, data in self.system_components.items():
            last_ping = self.incoming_data.get(name, 0)
            time_diff = current_time - last_ping
            row_key = data["row_key"]
            
            if time_diff <= 3 and last_ping > 0:
                status = "🟢 Online"
                seen_text = "Just now"
            else:
                status = "🔴 Offline"
                seen_text = f"{round(int(time_diff),-1)}s ago" if last_ping > 0 else "Never"
                
            monitor_table.update_cell(row_key, "Status", status)
            monitor_table.update_cell(row_key, "Last Seen", seen_text)

    def load_state(self):
        with open(STATE_FILE, "r") as f:
            return json.load(f)

    def save_state(self):
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f, indent=4)
        self.update_ui()

    def update_ui(self):
        if not hasattr(self, "state"):
            return

        self.query_one("#debt_display", Static).update(f"Corporate Debt\n₹{self.state['debt']:,.2f}")
        self.query_one("#capital_display", Static).update(f"AUM (Trading Capital)\n₹{self.state['trading_capital']:,.2f}")
        self.query_one("#savings_display", Static).update(f"Treasury (Buffer)\n₹{self.state['savings']:,.2f}")
        self.query_one("#profit_display", Static).update(f"Unrealized Alpha\n₹{self.state['unrealized_profit']:,.2f}")
        self.query_one("#target_display", Static).update(f"Financial Target\n₹{self.state['target']:,.2f}")
        self.query_one("#base_pay_display", Static).update(f"Payroll Liability\n₹{self.state.get('base_pay', 0):,.2f}/mo")

        package_text = (
            f"[bold cyan]Entity Name:[/bold cyan] {self.state.get('user', 'User')}\n"
            f"[bold cyan]Designation:[/bold cyan] Chief Investment Officer & Sole Proprietor\n\n"
            f"[bold yellow]Compensation Agreement:[/bold yellow]\n"
            f"[bold]Base Salary[/bold]: [green]₹{self.state.get('base_pay', 0):,.2f} per month[/green]\n"
            f"[bold]Daywise estimate[/bold] (20 days a month basis): [green]₹{(self.state.get('base_pay', 0) / 20):,.2f} per day[/green]\n"
            f"[bold]Profit Sharing[/bold]: 100% of reinvested compounding equity\n"
            f"[bold]Performance Target[/bold]: ₹{self.state.get('target', 0):,.2f} AUM\n\n"
            f"[dim italic]**Base salary is processed automatically based on treasury surplus. "
            f"Manual intervention is restricted to prevent emotional capital allocation**.[/dim italic]"
        )
        self.query_one("#package", Static).update(package_text)

# ... [Keep the package_text update] ...

        plot_widget = self.query_one("#division_plot", Pie)
        plot_widget.data = {
            "Trading Capital": self.state["trading_capital"],
            "Savings (Treasury)": self.state["savings"],
        }
        plot_widget.refresh()
        
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_save_setup":
            return

        if event.button.id == "btn_toggle_trees":
            tree = self.query_one("#recents_tree", Tree)
            if tree.root.is_expanded:
                tree.root.collapse_all()
                event.button.label = "Expand All"
            else:
                tree.root.expand_all()
                event.button.label = "Collapse All"
                
        self.save_state()

    @work(exclusive=True, thread=True)
    def sync_investment_data(self):
        """Fetches heavy API data like charges in the background without freezing the UI."""
        
        # 1. Tell the UI we are fetching (Safely pushed to Main Thread)
        def set_fetching():
            fetching = "[bold yellow]Fetching...[/bold yellow]"
            try:
                inv_table = self.query_one("#investments_table", DataTable)
                inv_table.update_cell("opt", "Duration/Summary", fetching)
                inv_table.update_cell("opt", "Status/P&L", fetching)
                inv_table.update_cell("opt", "Current (₹)", fetching)
            except Exception:
                pass
                
        self.call_from_thread(set_fetching)
        
        # 2. Do the heavy network lifting in the background thread
        try:
            charges = charges_calculator()
            capital: Funds = Funds.update_from_json(ustox.get_funds())
            # 3. Tell the UI to apply the fetched data (Safely pushed to Main Thread)
            def _calculate_current_pnl() -> float:
                """Fetches live positions to calculate today's realized PnL."""
                positions = ustox.get_positions()
                return sum(item.get("realised", 0.0) for item in positions)
            
            capital.pnl = _calculate_current_pnl()
            
            def apply_data():
                try:
                    inv_table = self.query_one("#investments_table", DataTable)
                    inv_table.update_cell("opt", "Duration/Summary", f"Charges: ₹{charges}")
                    inv_table.update_cell("opt", "Status/P&L", f"₹{capital.pnl:,.2f}")
                    inv_table.update_cell("opt", "Current (₹)", f"₹{capital.available_margin:,.2f}")
                except Exception:
                    pass
                    
            self.call_from_thread(apply_data)
            
        except Exception as e:
            logger.exception(f"Network error syncing charges: {e}")
            
            # 4. Handle errors cleanly on the UI
            def set_error():
                try:
                    inv_table = self.query_one("#investments_table", DataTable)
                    inv_table.update_cell("opt", "Duration/Summary", "Sync Failed (Network)")
                    inv_table.update_cell("opt", "Status/P&L", "Error")
                    inv_table.update_cell("opt", "Current (₹)", "Error")
                except Exception:
                    pass
                    
            self.call_from_thread(set_error)

    def action_quit(self):
        """Called automatically when Ctrl+Q is pressed."""
        self._is_shutting_down = True
        logger.info("Initiating hard shutdown via Ctrl+Q...")
        self.exit()

if __name__ == "__main__":
    app = CFOTracker()
    app.run()