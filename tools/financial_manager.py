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

# Ensure FinancialState is imported
from core.datatypes import Funds, FinancialState
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


class SetupScreen(ModalScreen[FinancialState]):
    """A popup screen to ask for initial balances if no save file exists."""
    CSS_PATH = "SetupScreen.tcss"

    def compose(self) -> ComposeResult:
        with Vertical(id="setup_dialog"):
            with Center():
                yield Label("[bold yellow]Corporate Setup initialization[/bold yellow]\nPlease establish your company parameters:\n")
                with Grid(id="data_grid"):
                    yield Input(placeholder="Name (e.g., Lokesh)", id="init_user", classes="setup-input")
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

                new_state = FinancialState(
                    user=self.query_one("#init_user", Input).value or "Lokesh",
                    debt=float(debt_val) if debt_val else 0.0,
                    trading_capital=float(capital_val) if capital_val else 0.0,
                    savings=float(savings_val) if savings_val else 0.0,
                    unrealized_profit=0.0,
                    target=float(target_val) if target_val else 0.0,
                    base_pay=float(base_pay_val) if base_pay_val else 50000.0,
                    growth_factor=0.1,
                    external_pnl=0.0
                )
                self.dismiss(new_state)
            except ValueError:
                self.query_one(Label).update("[bold red]Error: Please enter valid numbers![/bold red]")

        if event.button.id == "btn_fetch_upstox":
            try:
                capital_input = self.query_one("#init_capital", Input)
                capital_input.value = "30000"
            except Exception:
                self.query_one(Label).update("[bold red]Error in fetching upstox data. Please enter manually.[/bold red]")


class EditPlanParamsScreen(ModalScreen[dict]):
    """A popup screen to edit all planner parameters at once."""
    CSS_PATH = "edit_params.tcss"

    def __init__(self, planner, **kwargs):
        super().__init__(**kwargs)
        self.planner = planner

    def compose(self) -> ComposeResult:
        with Vertical(id="edit_plan_dialog"):
            yield Label("[b]Edit Financial Plan Parameters[/b]\n")
            
            yield Label("Multiplier:")
            yield Input(value=str(getattr(self.planner.financial_state, 'growth_factor', "")), id="inp_multiplier", classes="param-input")
            
            yield Label("Starting Amount (₹):")
            yield Input(value=str(getattr(self.planner.financial_state, 'trading_capital', "")), id="inp_starting_amount", classes="param-input")
            
            yield Label("Target (₹):")
            yield Input(value=str(getattr(self.planner.financial_state, 'target', "")), id="inp_target", classes="param-input")
            
            with Horizontal(id="edit_buttons"):
                yield Button("Save & Recalculate", id="btn_save_plan", variant="success")
                yield Button("Cancel", id="btn_cancel_plan", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_save_plan":
            try:
                new_args = {
                    "multiplier": float(self.query_one("#inp_multiplier", Input).value),
                    "starting": float(self.query_one("#inp_starting_amount", Input).value),
                    "target": float(self.query_one("#inp_target", Input).value),
                }
                self.dismiss(new_args)
            except ValueError:
                pass 
        elif event.button.id == "btn_cancel_plan":
            self.dismiss(None)


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

                        with Static(id="history", classes="data_panes"):
                            with Vertical():
                                yield Tree("Financial Ledger", id="recents_tree")
                                with Horizontal(id="tree_control_buttons"):
                                    yield Button("Expand All", id="btn_toggle_trees", classes="tree_buttons")

                        with Static(id="investments", classes="data_panes"):
                            yield DataTable(id="investments_table")

                        with Static(id="monitor", classes="data_panes"):
                            yield DataTable(id="system_status_table")

                    with Vertical():
                        with Container(id="log_container"):
                            yield RichLog(id="activity_log", highlight=True, markup=True, max_lines=100, auto_scroll=True)
                        yield Static(id="package")

            with TabPane("Planner"):
                with Horizontal():
                    yield DataFrameTable(id='plan_table', zebra_stripes=True)
                    with Vertical():
                        yield Static(id='plan_summary')
                        yield DataTable(id='plan_params', cursor_type="none")
                        with Horizontal(id="planner_action_buttons"):
                            yield Button("Edit Parameters", id="btn_edit_plan", variant="primary")
                            yield Button("Save Plan State", id="btn_save_plan_state", variant="success")
                            
        yield Footer()

    def on_mount(self):
        status_offline = "🔴 Offline"
        
        self.query_one("#dashboard", Container).border_title = "Financial Summary"
        self.query_one("#division_plot", Pie).border_title = "Division of Funds"
        self.query_one("#history", Static).border_title = "Recents Ledger"
        self.query_one("#investments", Static).border_title = "Current Investments"
        self.query_one("#monitor", Static).border_title = "System Monitor"
        self.query_one("#log_container", Container).border_title = "Logs"
        self.query_one("#package", Static).border_title = "Financial Package"

        # Initialize Base Trees and Tables
        recents_tree = self.query_one("#recents_tree", Tree)
        recents_tree.root.expand()
        self.t_expenses = recents_tree.root.add("Expenses", expand=False)
        self.t_payouts = recents_tree.root.add("Payouts", expand=False)
        self.t_payins = recents_tree.root.add("Payins", expand=False)
        self.t_wants = recents_tree.root.add("Wants", expand=False)

        inv_table = self.query_one("#investments_table", DataTable)
        inv_table.cursor_type = "row"
        inv_table.add_column("Asset Class", key="Asset Class")
        inv_table.add_column("Invested (₹)", key="Invested (₹)")
        inv_table.add_column("Current (₹)", key="Current (₹)")
        inv_table.add_column("Status/P&L", key="Status/P&L")
        inv_table.add_column("Duration/Summary", key="Duration/Summary")
        inv_table.add_row("Mutual Funds", "0.0", "0.0", "-", "Long Term", key="mf")
        inv_table.add_row("Stocks", "0.0", "0.0", "-", "Swing/Hold", key="stk")
        inv_table.add_row("Options", "0.0", "0.0", "-", "Awaiting Sync", key="opt")
        inv_table.add_row("Fixed Deposits", "0.0", "0.0", "-", "Locked", key="fd")

        monitor_table = self.query_one("#system_status_table", DataTable)
        monitor_table.cursor_type = "none" 
        monitor_table.add_column("Component", key="Component")
        monitor_table.add_column("Status", key="Status")
        monitor_table.add_column("Last Seen", key="Last Seen")     
        monitor_table.add_row("Live Trader", status_offline, "Never", key="live_trader")
        monitor_table.add_row("Risk Monitor", status_offline, "Never", key="risk_monitor")
        monitor_table.add_row("AWS Server", status_offline, "Never", key="aws_server")
        
        self.system_components = {
            "Live Trader": {"row_key": "live_trader"},
            "Risk Monitor": {"row_key": "risk_monitor"},
            "AWS Server": {"row_key": "aws_server"}
        }

        # Check for State File before initializing Planner
        if not STATE_FILE.exists() or STATE_FILE.stat().st_size <= 0:
            self.push_screen(SetupScreen(), self.init_state_callback)
        else:
            self.state = FinancialState.load_from_file(STATE_FILE)
            self.startup_sequence()

        self.zmq_listener()
        self.set_interval(1.0, self.check_heartbeats)
        self.sync_investment_data()
        self.set_interval(3600.0, self.sync_investment_data)    

    def init_state_callback(self, new_state: FinancialState):
        """Called automatically when SetupScreen completes."""
        self.state = new_state
        self.state.save_to_file(STATE_FILE)
        self.startup_sequence()
        
        log = self.query_one(RichLog)
        log.write("[bold green]Corporate setup complete. Data initialized and saved![/bold green]")

    def startup_sequence(self):
        """Initializes the planner and updates UI components tied to state."""
        log = self.query_one(RichLog)
        log.write(f"[bold cyan]Hello {self.state.user}, This is your personal CFO![/bold cyan]")
        
        self.planner = Planner()
        plan_df = self.planner.create(save_state=False)
        
        # Setup Plan Params DataTable
        plan_table = self.query_one("#plan_params", DataTable)
        plan_table.add_column("Parameter", key="Parameter")
        plan_table.add_column("Value", key="Value")
        plan_table.add_row("Multiplier", f"{self.planner.financial_state.growth_factor}", key="multiplier")
        plan_table.add_row("Starting Amount", f"₹{self.planner.financial_state.trading_capital:,.2f}", key="starting_amount")
        plan_table.add_row("Target", f"₹{self.planner.financial_state.target:,.2f}", key="target")
        
        self.query_one("#plan_table", DataFrameTable).add_df(plan_df)
        
        self.update_ui()
        self.update_plan_summary()

    def _log_remote_error(self, component: str, msg: str):
        log_widget = self.query_one(RichLog)
        log_widget.write(f"[bold red][{component} ALERT][/bold red] {msg}")

    @work(thread=True)
    def zmq_listener(self):
        import zmq

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

    def update_plan_summary(self):
        """Extracts stats from the planner and updates the summary widget."""
        import datetime as dt
        
        if not hasattr(self, "planner"):
            return

        days_left_text = "N/A"
        target_date_text = "N/A"
        if self.planner.df_plan is not None and not self.planner.df_plan.empty:
            final_date_str = self.planner.df_plan.iloc[-1]["Date"]
            final_date_obj = dt.datetime.strptime(final_date_str, "%Y-%m-%d").date()
            days_left = (final_date_obj - dt.datetime.today().date()).days
            
            days_left_text = f"{days_left} days"
            target_date_text = final_date_obj.strftime('%d %B %Y')

        if self.planner.surplus > 0:
            surplus_text = f"[bold green]+₹{self.planner.surplus:,.2f} (Ahead of plan 🚀)[/bold green]"
        elif self.planner.surplus < 0:
            surplus_text = f"[bold red]-₹{abs(self.planner.surplus):,.2f} (Behind plan ⚠️)[/bold red]"
        else:
            surplus_text = "[bold]₹0.00 (Exactly on track)[/bold]"

        summary_text = (
            f"[bold cyan]📊 TRADING PLAN REPORT[/bold cyan]\n"
            f"{'═'*50}\n"
            f"[b]Target Remaining[/b]:              ₹{self.planner.target:,.2f}\n"
            f"[b]Day's Starting Amount[/b]:         ₹{self.planner.financial_state.trading_capital:,.2f}\n"
            f"[b]Today's Planned Target[/b]:        ₹{self.planner.chronological_target:,.2f}\n"
            f"[b]Today's Current P&L[/b]:           ₹{self.planner.pnl:,.2f}\n"
            f"{'─'*50}\n"
            f"[b]Performance Surplus[/b]:           {surplus_text}\n"
            f"[b]Projected Trading Days Left[/b]:   {self.planner.days_remaining} days\n"
            f"[b]Projected Normal Days Left[/b]:    {days_left_text}\n"
            f"[b]Projected Target Date[/b]:         [bold yellow]{target_date_text}[/bold yellow]\n"
        )
        
        self.query_one("#plan_summary", Static).update(summary_text)
    
    def update_ui(self):
        if not hasattr(self, "state"):
            return

        self.query_one("#debt_display", Static).update(f"Corporate Debt\n₹{self.state.debt:,.2f}")
        self.query_one("#capital_display", Static).update(f"AUM (Trading Capital)\n₹{self.state.trading_capital:,.2f}")
        self.query_one("#savings_display", Static).update(f"Treasury (Buffer)\n₹{self.state.savings:,.2f}")
        self.query_one("#profit_display", Static).update(f"Unrealized Alpha\n₹{self.state.unrealized_profit:,.2f}")
        self.query_one("#target_display", Static).update(f"Financial Target\n₹{self.state.target:,.2f}")
        self.query_one("#base_pay_display", Static).update(f"Payroll Liability\n₹{self.state.base_pay:,.2f}/mo")

        package_text = (
            f"[bold cyan]Entity Name:[/bold cyan] {self.state.user}\n"
            f"[bold cyan]Designation:[/bold cyan] Chief Investment Officer & Sole Proprietor\n\n"
            f"[bold yellow]Compensation Agreement:[/bold yellow]\n"
            f"[bold]Base Salary[/bold]: [green]₹{self.state.base_pay:,.2f} per month[/green]\n"
            f"[bold]Daywise estimate[/bold] (20 days a month basis): [green]₹{(self.state.base_pay / 20):,.2f} per day[/green]\n"
            f"[bold]Profit Sharing[/bold]: 100% of reinvested compounding equity\n"
            f"[bold]Performance Target[/bold]: ₹{self.state.target:,.2f} AUM\n\n"
            f"[dim italic]**Base salary is processed automatically based on treasury surplus. "
            f"Manual intervention is restricted to prevent emotional capital allocation**.[/dim italic]"
        )
        self.query_one("#package", Static).update(package_text)

        plot_widget = self.query_one("#division_plot", Pie)
        plot_widget.data = {
            "Trading Capital": self.state.trading_capital,
            "Savings (Treasury)": self.state.savings,
        }
        plot_widget.refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_save_setup":
            return
            
        if event.button.id == "btn_edit_plan":
            def check_plan_edit(new_args: dict | None):
                if new_args is not None:
                    # Update parameter display table visually
                    plan_table_params = self.query_one("#plan_params", DataTable)
                    plan_table_params.update_cell("multiplier", "Value", f"{new_args['multiplier']}")
                    plan_table_params.update_cell("starting_amount", "Value", f"₹{new_args['starting']:,.2f}")
                    plan_table_params.update_cell("target", "Value", f"₹{new_args['target']:,.2f}")

                    # Recalculate plan in memory
                    new_plan_df = self.planner.create(
                        target=new_args['target'],
                        starting=new_args['starting'],
                        multiplier=new_args['multiplier'],
                        force=True,
                        save_state=False
                    )                    
                    
                    # Update main plan table and internal state
                    self.query_one("#plan_table", DataFrameTable).update_df(new_plan_df)
                    self.state = self.planner.financial_state
                    
                    # Refresh memory UI states without saving to disk
                    self.update_ui()
                    self.update_plan_summary()
                    
                    log = self.query_one(RichLog)
                    log.write("[bold yellow]Plan recalculated in sandbox mode.[/bold yellow] Click '[bold green]Save Plan State[/bold green]' to commit changes to disk.")

            self.push_screen(EditPlanParamsScreen(self.planner), check_plan_edit)
            return

        if event.button.id == "btn_save_plan_state":
            if hasattr(self, "state") and self.state is not None:
                self.state.save_to_file(STATE_FILE)
                log = self.query_one(RichLog)
                log.write("[bold green]Financial state and plan successfully committed to disk![/bold green]")
            return

        if event.button.id == "btn_toggle_trees":
            tree = self.query_one("#recents_tree", Tree)
            if tree.root.is_expanded:
                tree.root.collapse_all()
                event.button.label = "Expand All"
            else:
                tree.root.expand_all()
                event.button.label = "Collapse All"

    @work(exclusive=True, thread=True)
    def sync_investment_data(self):
        """Fetches heavy API data like charges in the background without freezing the UI."""
        
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
        
        try:
            charges = charges_calculator()
            capital: Funds = Funds.update_from_json(ustox.get_funds())
            
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