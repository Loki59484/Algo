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
                        "[bold yellow]First time? Let's get you setup![/bold yellow]\nPlease enter your starting balances:\n",
                    )
                    with Grid(id="data_grid"):
                        yield Input(
                            placeholder="Name (e.g., John Doe)",
                            id="init_user",
                            classes="setup-input",
                        )
                        yield Input(
                            placeholder="Current Debt (e.g., 1900000)",
                            id="init_debt",
                            classes="setup-input",
                        )
                        yield Input(
                            placeholder="Trading Capital (e.g., 1000000)",
                            id="init_capital",
                            classes="setup-input",
                        )
                        yield Input(
                            placeholder="Financial Target (e.g., 2000000)",
                            id="init_target",
                            classes="setup-input",
                        )
                        yield Input(
                            placeholder="Savings/Buffer (e.g., 400000)",
                            id="init_savings",
                            classes="setup-input",
                        )
                        yield Input(
                            placeholder="Paycut Ratio (e.g., 0.1)",
                            id="init_paycut",
                            classes="setup-input",
                        )

                with Center():
                    with Horizontal(id="control_container"):
                        yield Button(
                            "Save & Start",
                            id="btn_save_setup",
                            variant="success",
                            classes="control_buttons",
                        )
                        yield Button(
                            "Fetch from Upstox",
                            id="btn_fetch_upstox",
                            variant="success",
                            classes="control_buttons",
                        )  # fetches available data from upstox

        def on_button_pressed(self, event: Button.Pressed) -> None:
            event.stop()
            if event.button.id == "btn_save_setup":
                try:
                    # Retrieve inputs, defaulting to 0 if left blank
                    debt_val = self.query_one("#init_debt", Input).value
                    capital_val = self.query_one("#init_capital", Input).value
                    savings_val = self.query_one("#init_savings", Input).value
                    target_val = self.query_one("#init_target", Input).value
                    paycut_ratio = self.query_one("#init_paycut", Input).value
                    new_state = {
                        "user": self.query_one("#init_user", Input).value or "User",
                        "debt": float(debt_val) if debt_val else 0.0,
                        "trading_capital": float(capital_val) if capital_val else 0.0,
                        "savings": float(savings_val) if savings_val else 0.0,
                        "unrealized_profit": 0.0,
                        "target": float(target_val) if target_val else 0.0,
                        "paycut_ratio": float(paycut_ratio) if paycut_ratio else 0.5,
                    }
                    # Dismiss the modal and return the new state to the main app
                    self.dismiss(new_state)
                except ValueError:
                    # If they type non-numbers, we can just replace the label text to warn them
                    self.query_one(Label).update(
                        "[bold red]Error: Please enter valid numbers![/bold red]"
                    )

            if event.button.id == "btn_fetch_upstox":
                try:
                    # capital: Funds = Funds.update_from_json(ustox.get_funds())
                    # available = capital.available_margin
                    capital_val = self.query_one("#init_capital", Input)
                    capital_val.replace(capital_val, 30000)
                except Exception:
                    self.query_one(Label).update(
                        "[bold red]Error in fetching upstox data. Please enter manually.[/bold red]"
                    )


    class CFOTracker(App):
        TITLE = "Algo's Personal CFO"
        CSS_PATH = "cfo_tracker.tcss"

        def compose(self) -> ComposeResult:

            yield Header()

            # Container for available funds in different catagories
            with Container(id="dashboard"):
                with Horizontal():
                    yield Static(id="debt_display", classes="balance-box alert")
                    yield Static(id="capital_display", classes="balance-box")
                    yield Static(id="savings_display", classes="balance-box")
                    yield Static(id="profit_display", classes="balance-box")
                    yield Static(id="target_display", classes="balance-box")
                    yield Static(id="paycut_ratio", classes="balance-box")

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

                with Container(id="log_container"):
                    yield RichLog(
                        id="activity_log",
                        highlight=True,
                        markup=True,
                        max_lines=100,
                        auto_scroll=True,
                    )
            yield Footer()
        
@work(thread=True)
    def zmq_listener(self):
        """Runs in the background, listening for ZMQ messages and updating timestamps."""
        import zmq
        import time

        context = zmq.Context()
        sub_socket = context.socket(zmq.SUB)

        AWS_TAILSCALE_IP = "100.115.92.50" 
        
        # Connect to all 3 ports (Data, Risk Monitor, Live Trader Heartbeat)
        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5557") # Risk Monitor Heartbeat
        sub_socket.connect(f"tcp://{AWS_TAILSCALE_IP}:5558") # Live Trader Heartbeat
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        while True:
            try:
                message = sub_socket.recv_string(flags=zmq.NOBLOCK)
                if message.startswith("PING:"):
                    component_name = message.split(":")[1]
                    self.incoming_data[component_name] = time.time()
            except zmq.Again:
                time.sleep(0.1)
            except Exception as e:
                # Use call_from_thread to safely write to the UI from a background thread
                self.call_from_thread(self._log_zmq_error, str(e))
                time.sleep(1) # Prevent spamming logs if it loops
                
    def _log_zmq_error(self, error_msg: str):
        """Helper to safely write errors to the RichLog."""
        self.query_one(RichLog).write(f"[red]ZMQ Error: {error_msg}[/red]")
        
        def on_mount(self):
            # Check if file exists immediately on startup
            log = self.query_one(RichLog)
            if not STATE_FILE.exists() or STATE_FILE.stat().st_size <= 0:
                # If no file, push the setup screen and wait for the callback
                self.push_screen(SetupScreen(), self.init_state_callback)
            else:
                # File exists, load normally
                self.state = self.load_state()
                self.update_ui()
            self.query_one("#dashboard", Container).border_title = "Financial Summary"
            self.query_one("#division_plot", Pie).border_title = "Division of Funds"
            self.query_one("#history", Static).border_title = "Recents"
            self.query_one("#investments", Static).border_title = "Current Investments"
            self.query_one("#monitor", Static).border_title = "System Monitor"
            self.query_one("#log_container", Container).border_title = "Logs"

            monitor_table = self.query_one("#system_status_table", DataTable)
            monitor_table.cursor_type = "none" # Hide the selection cursor# ADD THESE LINES:
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
            log.write(
                f"[bold cyan]Hello {self.state.get('user', 'User')}, This is your personal CFO![/bold cyan]"
            )

        def init_state_callback(self, new_state: dict):
            """Called automatically when the SetupScreen is dismissed."""
            self.state = new_state
            self.save_state()
            self.update_ui()
            self.query_one(RichLog).write(
                "[bold green]Initial setup complete. Data saved![/bold green]"
            )
        
        def check_heartbeats(self):
            """Timer task that updates the DataTable with live statuses."""
            monitor_table = self.query_one("#system_status_table", DataTable)
            current_time = time.time()
            
            for name, data in self.system_components.items():
                # Read from the shared dictionary updated by the background thread
                last_ping = self.incoming_data.get(name, 0)
                logger.info(f"{last_ping}")
                time_diff = current_time - last_ping
                row_key = data["row_key"]
                
                # If pinged within the last 3 seconds, it's online
                if time_diff <= 3 and last_ping > 0:
                    status = "🟢 Online"
                    seen_text = "Just now"
                else:
                    status = "🔴 Offline"
                    seen_text = f"{int(time_diff)}s ago" if last_ping > 0 else "Never"
                    
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
                    f"Debt\n₹{self.state['debt']:,.2f}"
                )
                self.query_one("#capital_display", Static).update(
                    f"Trading Capital\n₹{self.state['trading_capital']:,.2f}"
                )
                self.query_one("#savings_display", Static).update(
                    f"Savings (Buffer)\n₹{self.state['savings']:,.2f}"
                )
                self.query_one("#profit_display", Static).update(
                    f"Unsplit Monthly Profit\n₹{self.state['unrealized_profit']:,.2f}"
                )
                self.query_one("#target_display", Static).update(
                    f"Financial Target\n₹{self.state['target']:,.2f}"
                )
                self.query_one("#paycut_ratio", Static).update(
                    f"Paycut Ratio\n{self.state['paycut_ratio']:,.2f}"
                )
                plot_widget = self.query_one("#division_plot", Pie)

                # Update the widget's internal data dictionary
                plot_widget.data = {
                    "Trading Capital": self.state["trading_capital"],
                    "Savings": self.state["savings"],
                }

                # Force the widget to recalculate the math and redraw
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


    if __name__ == "__main__":
        app = CFOTracker()
        app.run()
