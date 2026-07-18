import sys
import json
import logging
import time
import zmq
from pathlib import Path
from textual import work
from textual.screen import ModalScreen
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Center, Grid, Container
from textual.widgets import Header, Footer, Static, Input, Button, RichLog, Label, Tree, Collapsible, DataTable

# Adding root directory to sys.path for module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
TOOL_DIR = ROOT_DIR / "tools"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.upstox_methods import UpstoxClient, AWS_TAILSCALE_IP
from core.datatypes import Funds

# Intitiating logger
logger = logging.getLogger(__name__)
STATE_FILE = TOOL_DIR / "finance_state.json"
ustox = UpstoxClient()

import math
from textual.widget import Widget
from rich.text import Text


class Pie(Widget):
    """A custom pie chart widget that mathematically corrects terminal aspect ratios."""

    def __init__(self, data: dict, radius: int = 10, marker="⣿", **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.radius = radius
        self.marker = marker[0]
        # Rename this variable so it doesn't fight Textual's internals
        self.pie_colors = ["#ff0055", "#aaff00", "#00ffff", "#ffff00"]

    def render(self) -> Text:
        total = sum(self.data.values())
        if total == 0:
            return Text("No funds available to display.", style="dim", justify="center")

        # 1. Calculate angles for each slice
        slices = []
        current_angle = 0
        for key, value in self.data.items():
            fraction = value / total
            angle = fraction * 2 * math.pi
            slices.append((key, current_angle, current_angle + angle, value))
            current_angle += angle

        # 2. Initialize Text object and CENTER IT
        result = Text()
        result.justify = "center"  # <-- THIS CENTERS EVERYTHING (Pie + Legend)

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

                    result.append(
                        self.marker, style=char_color
                    )  # Or use "⡿", "⠿", etc.
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
                yield Label(
                    "[bold yellow]Corporate Setup initialization[/bold yellow]\nPlease establish your company parameters:\n",
                )
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
                base_pay_val = self.query_one("#init_base_pay", Input).value # NEW
                
                new_state = {
                    "user": self.query_one("#init_user", Input).value or "User",
                    "debt": float(debt_val) if debt_val else 0.0,
                    "trading_capital": float(capital_val) if capital_val else 0.0,
                    "savings": float(savings_val) if savings_val else 0.0,
                    "unrealized_profit": 0.0,
                    "target": float(target_val) if target_val else 0.0,
                    "base_pay": float(base_pay_val) if base_pay_val else 50000.0, # NEW
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


class CFOTracker(App):
    TITLE = "Algo's Personal CFO"
    CSS_PATH = "cfo_tracker.tcss"

    BINDINGS = [
        ("ctrl+q", "quit", "Quit the Application")
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_shutting_down = False

    def compose(self) -> ComposeResult:

        yield Header()

        with Container(id="dashboard"):
            with Horizontal():
                yield Static(id="debt_display", classes="balance-box alert")
                yield Static(id="capital_display", classes="balance-box")
                yield Static(id="savings_display", classes="balance-box")
                yield Static(id="profit_display", classes="balance-box")
                yield Static(id="target_display", classes="balance-box")
                yield Static(id="base_pay_display", classes="balance-box")

        # Grid to show other related data
        with Horizontal():
            with Grid(id="data_grid"):
                # Plot of available funds in different divisions
                yield Pie(
                    data={"Trading Capital": 0, "Savings": 0},
                    radius=7,  # Adjust radius to fit your layout panel
                    id="division_plot",
                    classes="plotext_plot",
                )

                # History of recent events like transactions, debt settlements, payments etc.
                with Static(id="history", classes="data_panes"):
                    with Vertical():
                        payments = Tree(id="payments", label="Expenses", classes="trees")
                        payout = Tree(id="payout", label="Payouts", classes="trees")
                        payin = Tree(id="payin", label="Payins", classes="trees")
                        wants = Tree(id="wants", label="Wants", classes="trees")
                        
                        with Vertical(id="trees_container"):
                            yield payments
                            yield payout
                            yield payin
                            yield wants

                        with Horizontal(id="tree_control_buttons"):
                            yield Button(label="Expand All", id="btn_toggle_trees", classes="tree_buttons")

                with Static(id="investments", classes="data_panes"):
                    with Vertical():
                        mutual_funds = Collapsible(id="payments", classes="trees")
                        options = Collapsible(id="payout", classes="trees")
                        stocks = Collapsible(id="payin", classes="trees")
                        fixed_deposits = Collapsible(id="wants", classes="trees")
                        
                        with Vertical(id="trees_container"):
                            yield mutual_funds
                            yield options
                            yield stocks
                            yield fixed_deposits

                        with Horizontal(id="tree_control_buttons"):
                            yield Button(label="Expand All", id="btn_toggle_collapsibles", classes="tree_buttons")
                with Static(id="monitor", classes="data_panes"):
                    yield DataTable(id="system_status_table")
                    
    

            with Vertical():
                with Container(id="log_container"):
                    yield RichLog(
                        id="activity_log",
                        highlight=True,
                        markup=True,
                        max_lines=100,
                        auto_scroll=True,
                    )
                yield Static(id="package")
                    
                
        yield Footer()

    def _log_remote_error(self, component: str, msg: str):
        """Safely writes remote AWS errors to the RichLog in red."""
        log_widget = self.query_one(RichLog)
        log_widget.write(f"[bold red][{component} ALERT][/bold red] {msg}")

    @work(thread=True)
    def zmq_listener(self):
        import zmq
        import time

        context = zmq.Context.instance()
        sub_socket = context.socket(zmq.SUB)
        
        # Connect to Heartbeat Ports
        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5557") 
        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5558") 
        
        # Connect to the NEW Error Log Ports
        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5567") 
        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5568") 
        
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        while not self._is_shutting_down:
            try:
                message = sub_socket.recv_string(flags=zmq.NOBLOCK)
                logger.info(f"Recieved msg : {message}")
                # ... [Keep your existing message handling logic here] ...

            except zmq.Again:
                time.sleep(0.1)
            except Exception as e:
                self.call_from_thread(self._log_remote_error, "System", str(e))
                time.sleep(1)
                
        sub_socket.close()
        context.term()

    def on_mount(self):
        log = self.query_one(RichLog)
        
        # Check if file exists immediately on startup
        if not STATE_FILE.exists() or STATE_FILE.stat().st_size <= 0:
            # If no file, push the setup screen and wait for the callback
            self.push_screen(SetupScreen(), self.init_state_callback)
        else:
            # File exists, load normally
            self.state = self.load_state()
            self.update_ui()
            # ADD IT HERE INSTEAD
            log.write(
                f"[bold cyan]Hello {self.state.get('user', 'User')}, This is your personal CFO![/bold cyan]"
            )
            
        self.query_one("#dashboard", Container).border_title = "Financial Summary"
        self.query_one("#division_plot", Pie).border_title = "Division of Funds"
        self.query_one("#history", Static).border_title = "Recents"
        self.query_one("#investments", Static).border_title = "Current Investments"
        self.query_one("#monitor", Static).border_title = "System Monitor"
        self.query_one("#log_container", Container).border_title = "Logs"
        self.query_one("#package", Static).border_title = "Financial Package"

        monitor_table = self.query_one("#system_status_table", DataTable)
        monitor_table.cursor_type = "none" 
        monitor_table.add_column("Component", key="Component")
        monitor_table.add_column("Status", key="Status")
        monitor_table.add_column("Last Seen", key="Last Seen")        
        
        self.system_components = {
            "Live Trader": {"row_key": monitor_table.add_row("Live Trader", "🔴 Offline", "Never")},
            "Risk Monitor": {"row_key": monitor_table.add_row("Risk Monitor", "🔴 Offline", "Never")},
            "AWS Server": {"row_key": monitor_table.add_row("AWS Server", "🔴 Offline", "Never")}
        }

        self.incoming_data = {
            "Live Trader": 0,
            "Risk Monitor": 0,
            "AWS Server": 0 
        }
        
        # 2. Start the background ZMQ listener thread
        self.zmq_listener()
        
        # 3. Start the UI updater loop (every 1 second)
        self.set_interval(1.0, self.check_heartbeats)
        
        # REMOVED log.write(...) FROM HERE

    def init_state_callback(self, new_state: dict):
        """Called automatically when the SetupScreen is dismissed."""
        self.state = new_state
        self.save_state()
        self.update_ui()
        
        log = self.query_one(RichLog)
        log.write("[bold green]Corporate setup complete. Data initialized![/bold green]")
        log.write(f"[bold cyan]Hello {self.state.get('user', 'User')}, This is your personal CFO![/bold cyan]")
    
    def check_heartbeats(self):
        """Timer task that updates the DataTable with live statuses."""
        monitor_table = self.query_one("#system_status_table", DataTable)
        current_time = time.time()
        
        for name, data in self.system_components.items():
            # Read from the shared dictionary updated by the background thread
            last_ping = self.incoming_data.get(name, 0)
            time_diff = current_time - last_ping
            row_key = data["row_key"]
            
            # If pinged within the last 3 seconds, it's online
            if time_diff <= 3 and last_ping > 0:
                status = "🟢 Online"
                seen_text = "Just now"
            else:
                status = "🔴 Offline"
                seen_text = f"{round(int(time_diff),-1)}s ago" if last_ping > 0 else "Never"
                
            # Update the specific cells in the table
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
        if hasattr(self, "state"):

            self.query_one("#debt_display", Static).update(
                f"Corporate Debt\n₹{self.state['debt']:,.2f}"
            )
            self.query_one("#capital_display", Static).update(
                f"AUM (Trading Capital)\n₹{self.state['trading_capital']:,.2f}"
            )
            self.query_one("#savings_display", Static).update(
                f"Treasury (Buffer)\n₹{self.state['savings']:,.2f}"
            )
            self.query_one("#profit_display", Static).update(
                f"Unrealized Alpha\n₹{self.state['unrealized_profit']:,.2f}"
            )
            self.query_one("#target_display", Static).update(
                f"Financial Target\n₹{self.state['target']:,.2f}"
            )
            
            self.query_one("#base_pay_display", Static).update(
                f"Payroll Liability\n₹{self.state.get('base_pay', 0):,.2f}/mo"
            )

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

            plot_widget = self.query_one("#division_plot", Pie)
            plot_widget.data = {
                "Trading Capital": self.state["trading_capital"],
                "Savings (Treasury)": self.state["savings"],
            }
            plot_widget.refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        # Ignore button presses from the setup screen
        if event.button.id == "btn_save_setup":
            return

        if event.button.id == "btn_toggle_trees":
            for tree in self.query(Tree):
                tree.root.toggle_all()
                event.button.label = "Collapse All" if tree.root.is_expanded else "Expand All"
        elif event.button.id == "btn_toggle_collapsibles":
            for item in self.query(Collapsible):
                item.collapsed = not item.collapsed
                event.button.label = "Expand All" if item.collapsed else "Collapse All"
        self.save_state()

    def action_quit(self):
        """Called automatically when Ctrl+Q is pressed."""
        # Tell the ZMQ thread to break its loop
        self._is_shutting_down = True
        
        # Log the graceful exit 
        logger.info("Initiating graceful shutdown via Ctrl+Q...")
        
        # Tell Textual to close the application
        self.exit()
        
if __name__ == "__main__":
    app = CFOTracker()
    app.run()
    