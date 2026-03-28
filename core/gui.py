# ------Import Libraries------
from dearpygui.dearpygui import *
from dearpygui_ext import logger
from queue import Queue
from pprint import pprint
import pandas as pd
import config
from core import upstox_func as ustox
import traceback
from core import anatomy as ana
import numpy as np
import threading
import asyncio
import report
import math

# ----------------------------
"""
Module for gui of the to test live trading using Upstox API. 
"""
# ----------------------------

tick_buffer = asyncio.Queue(maxsize=5)


class App:

    def __init__(self, dateidx=None):
        from screeninfo import get_monitors

        monitors = get_monitors()
        primary_monitor = monitors[0]
        self.date_idx = dateidx
        self.width = primary_monitor.width
        self.height = primary_monitor.height
        self.data_queue = Queue()
        self.updatelock = threading.Lock()
        self.pause = True
        self.simulation = None
        self.loop = asyncio.new_event_loop()
        self.tradethread = None
        self.traders_ok = False
        self.strikes = False

    def init_variables(self):
        (
            self.keys,
            self.simdate,
            self.call_plotdata,
            self.put_plotdata,
            self.scrip_plotdata,
            self.live_index,
            self.livetotal,
            config.latency,
        ) = ana.setup_data(self.simulation, self.date_idx)

    def setup_gui(self):
        create_context()
        create_viewport(
            title="Trader",
            resizable=True,
            min_height=600,
            min_width=800,
            height=self.height,
            width=self.width,
        )
        """Font Regisrtry"""
        with font_registry():
            with font(
                f"{ustox.directory}fonts/Ubuntu-Regular.ttf", 16
            ) as self.basefont:
                add_font_range_hint(mvFontRangeHint_Default)
            with font(
                f"{ustox.directory}fonts/Ubuntu-Regular.ttf", 14
            ) as self.smallfont:
                add_font_range_hint(mvFontRangeHint_Default)
            with font(
                f"{ustox.directory}fonts/Ubuntu-Regular.ttf", 18
            ) as self.largefont:
                add_font_range_hint(mvFontRangeHint_Default)

        """Theme Regisrtry"""
        with theme() as self.globaltheme:
            with theme_component(mvAll):
                add_theme_color(
                    mvThemeCol_WindowBg, (0, 0, 0), category=mvThemeCat_Core
                )
                add_theme_color(
                    mvThemeCol_ChildBg, (0, 0, 25), category=mvThemeCat_Core
                )
                add_theme_style(
                    mvStyleVar_WindowBorderSize, 0, 0, category=mvThemeCat_Core
                )
                add_theme_style(mvStyleVar_FramePadding, 4, 4, category=mvThemeCat_Core)
                add_theme_style(
                    mvStyleVar_WindowPadding, 8, 8, category=mvThemeCat_Core
                )
                add_theme_style(
                    mvStyleVar_ChildBorderSize, 1, 1, category=mvThemeCat_Core
                )
                add_theme_color(
                    mvThemeCol_Border, (100, 100, 100), category=mvThemeCat_Core
                )
                add_theme_color(
                    mvThemeCol_Button, (48, 48, 48), category=mvThemeCat_Core
                )

        with theme() as self.windowtheme:
            with theme_component(mvAll):
                add_theme_color(
                    mvThemeCol_TitleBg, (32, 32, 32), category=mvThemeCat_Core
                )
                add_theme_color(
                    mvThemeCol_Text, (200, 200, 200), category=mvThemeCat_Core
                )
                add_theme_color(
                    mvThemeCol_Border, (100, 100, 100), category=mvThemeCat_Core
                )
                add_theme_color(
                    mvThemeCol_FrameBg, (32, 32, 32), category=mvThemeCat_Core
                )

        with theme() as self.plottheme:
            with theme_component(mvPlot):
                add_theme_color(
                    mvPlotCol_PlotBorder, (64, 64, 64), category=mvThemeCat_Plots
                )
                add_theme_color(mvPlotCol_PlotBg, (0, 0, 20), category=mvThemeCat_Plots)
                add_theme_color(
                    mvPlotCol_AxisGrid, (0, 0, 0), category=mvThemeCat_Plots
                )
                add_theme_style(
                    mvPlotStyleVar_PlotPadding, 0, 10, category=mvThemeCat_Plots
                )
                add_theme_color(mvThemeCol_Border, (0, 0, 0), category=mvThemeCat_Core)
            with theme_component(mvShadeSeries):
                add_theme_color(
                    mvPlotCol_Fill, (64, 64, 64, 50), category=mvThemeCat_Plots
                )
            with theme_component(mvLineSeries):
                add_theme_style(
                    mvPlotStyleVar_LineWeight, 1.5, category=mvThemeCat_Plots
                )
            with theme_component(mvInfLineSeries):
                add_theme_color(
                    mvPlotCol_Line, (150, 150, 150, 150), category=mvThemeCat_Plots
                )

        with theme() as self.buytheme:
            with theme_component(mvScatterSeries):
                add_theme_color(
                    mvPlotCol_Line, (74, 199, 60), category=mvThemeCat_Plots
                )
                add_theme_style(
                    mvPlotStyleVar_Marker, mvPlotMarker_Up, category=mvThemeCat_Plots
                )
                add_theme_style(mvPlotStyleVar_MarkerSize, 8, category=mvThemeCat_Plots)

        with theme() as self.selltheme:
            with theme_component(mvScatterSeries):
                add_theme_color(
                    mvPlotCol_Line, (235, 85, 85), category=mvThemeCat_Plots
                )
                add_theme_style(
                    mvPlotStyleVar_Marker, mvPlotMarker_Down, category=mvThemeCat_Plots
                )
                add_theme_style(mvPlotStyleVar_MarkerSize, 8, category=mvThemeCat_Plots)

        with handler_registry():
            add_key_press_handler(key=mvKey_Spacebar, callback=self.pause_play)
            add_key_press_handler(key=mvKey_Tab, callback=self.fastforward)
            add_key_press_handler(key=mvKey_F11, callback=toggle_viewport_fullscreen)
            add_key_press_handler(
                key=mvKey_S,
                callback=(
                    self.sell_button_handler
                    if is_key_down(mvKey_LControl) or is_key_down(mvKey_RControl)
                    else None
                ),
            )
            add_key_press_handler(
                key=mvKey_U,
                callback=lambda: configure_item(
                    "underlying",
                    show=(not is_item_shown("underlying")),
                ),
            )
            add_key_press_handler(
                key=mvKey_R,
                callback=lambda: configure_item(
                    "Report",
                    show=(not is_item_shown("Report")),
                ),
            )

    def draw_gui(self):
        width = self.width
        height = self.height
        """ Setup for the windows"""
        with window(
            tag="primary", no_move=False, no_collapse=False, no_background=True
        ):  # Main window
            with child_window(
                label="Charts",
                tag="charts",
                height=3 * height // 4 + 40,
                width=width,
                pos=(0, 0),
            ):  # Charts window
                bind_item_font("charts", self.smallfont)
                with group(horizontal=True, tag="chartgroup", height=3 * height // 4):
                    bind_item_theme("chartgroup", self.plottheme)
                    with subplots(
                        label=self.call_plotdata.key,
                        rows=2,
                        columns=1,  # Call options plots
                        tag="callgroup",
                        height=-1,
                        width=(width // 2) - 15,
                        parent="chartgroup",
                        no_resize=False,
                        no_menus=True,
                        share_series=True,
                        link_all_x=True,
                    ):
                        bind_item_theme("callgroup", self.plottheme)

                        with plot(
                            tag="call",
                            no_frame=True,
                            tracked=True,
                            crosshairs=True,
                            pan_mod=True,
                        ):
                            add_plot_axis(
                                mvXAxis,
                                no_tick_labels=True,
                                tag="callxaxis",
                                no_gridlines=True,
                            )
                            add_plot_axis(mvYAxis, tag="callclose", no_gridlines=True)
                            set_axis_limits_auto("callclose")
                            add_candle_series(
                                dates=[np.nan] * 50,
                                opens=[np.nan] * 50,
                                highs=[np.nan] * 50,
                                lows=[np.nan] * 50,
                                closes=[np.nan] * 50,
                                parent="callclose",
                                tag="callcloseline",
                            )
                            add_plot_annotation(tag="callclose_annot", parent="call")
                            add_line_series(
                                [np.nan] * 50,
                                [np.nan] * 50,
                                parent="callclose",
                                tag="callsupertrend",
                            )
                            add_scatter_series(
                                [np.nan] * 50,
                                [np.nan] * 50,
                                parent="callclose",
                                tag="callbuys",
                            )
                            add_scatter_series(
                                [np.nan] * 50,
                                [np.nan] * 50,
                                parent="callclose",
                                tag="callsells",
                            )
                            bind_item_theme("callbuys", theme=self.buytheme)
                            bind_item_theme("callsells", theme=self.selltheme)
                            add_shade_series(
                                parent="callclose",
                                x=(0, 0),
                                y1=(1000, 1000),
                                tag="calltraderegion",
                            )
                            add_plot_legend(parent="call")
                        with plot(
                            tag="call_oscillation",
                            no_frame=True,
                            tracked=True,
                            crosshairs=True,
                            pan_mod=True,
                        ):
                            add_plot_legend()
                            add_plot_axis(
                                mvXAxis,
                                no_tick_labels=True,
                                tag="call_osc_x",
                                no_gridlines=False,
                            )
                            add_plot_axis(
                                mvYAxis,
                                tag="call_osc_y",
                                no_gridlines=True,
                            )
                            set_axis_limits_auto("call_osc_y")
                            add_line_series(
                                [np.nan] * 50,
                                [np.nan] * 50,
                                parent="call_osc_y",
                                tag="call_osc_line"
                            )

                    with subplots(
                        label=self.put_plotdata.key,
                        rows=2,
                        columns=1,  # Put options plots
                        tag="putgroup",
                        height=-1,
                        parent="chartgroup",
                        width=(width // 2) - 15,
                        no_menus=True,
                        no_resize=False,
                        link_all_x=True,
                        share_series=True,
                    ):

                        bind_item_theme("putgroup", self.plottheme)

                        with plot(
                            tag="put",
                            no_frame=True,
                            tracked=True,
                            crosshairs=True,
                            pan_mod=True,
                        ):
                            add_plot_legend()
                            add_plot_axis(
                                mvXAxis,
                                no_tick_labels=True,
                                tag="putxaxis",
                                no_gridlines=False,
                            )
                            add_plot_axis(
                                mvYAxis,
                                tag="putclose",
                                opposite=True,
                                no_gridlines=True,
                            )
                            set_axis_limits_auto("putclose")
                            add_candle_series(
                                dates=[np.nan] * 50,
                                opens=[np.nan] * 50,
                                highs=[np.nan] * 50,
                                lows=[np.nan] * 50,
                                closes=[np.nan] * 50,
                                parent="putclose",
                                tag="putcloseline",
                            )
                            add_plot_annotation(tag="putclose_annot", parent="put")

                            add_line_series(
                                [np.nan] * 50,
                                [np.nan] * 50,
                                parent="putclose",
                                tag="putsupertrend",
                            )
                            add_scatter_series(
                                [np.nan] * 50,
                                [np.nan] * 50,
                                parent="putclose",
                                tag="putbuys",
                            )
                            add_scatter_series(
                                [np.nan] * 50,
                                [np.nan] * 50,
                                parent="putclose",
                                tag="putsells",
                            )
                            bind_item_theme("putbuys", theme=self.buytheme)
                            bind_item_theme("putsells", theme=self.selltheme)
                            add_shade_series(
                                parent="putclose",
                                x=(0, 0),
                                y1=(1000, 1000),
                                tag="puttraderegion",
                            )
                        with plot(
                            tag="put_oscillation",
                            no_frame=True,
                            tracked=True,
                            crosshairs=True,
                            pan_mod=True,
                        ):
                            add_plot_legend()
                            add_plot_axis(
                                mvXAxis,
                                no_tick_labels=True,
                                tag="put_osc_x",
                                no_gridlines=False,
                            )
                            add_plot_axis(
                                mvYAxis,
                                tag="put_osc_y",
                                opposite=True,
                                no_gridlines=True,
                            )
                            set_axis_limits_auto("put_osc_y")
                            add_line_series(
                                [np.nan] * 50,
                                [np.nan] * 50,
                                parent="put_osc_y",
                                tag="put_osc_line",
                            )

                with group(
                    tag="controlpanel", horizontal=True, parent="Charts"
                ):  # Control panel
                    add_button(
                        label="Start", callback=self.pause_play, tag="playpausebutton"
                    )

            with group(
                parent="primary",
                tag="datapanel",
                horizontal=True,
                pos=(0, 3 * height // 4 + 40),
            ):  # Control panel

                with child_window(
                    label="Position",
                    tag="position",
                    height=-1,
                    width=width // 5,
                    border=True,
                ):  # Position child_window
                    add_text("POSITION", color=(255, 255, 255))
                    add_separator()
                    bind_item_theme("position", self.windowtheme)
                    add_text(f"Key: {ana.Position.key}", tag="poskey")
                    add_text(f"Symbol: {ana.Position.symbol}", tag="possym")
                    add_text(f"Quantity: {ana.Position.quantity}", tag="posqty")
                    add_text(
                        f"Current Value: {ana.Position.unrealised}", tag="poscurrentval"
                    )

                with child_window(
                    label="Funds", tag="funds", height=-1, width=width // 5, border=True
                ):  # Funds child_window
                    add_text("FUNDS", color=(255, 255, 255))
                    add_separator()
                    bind_item_theme("funds", self.windowtheme)
                    add_text(
                        f"Opening Balance : {ana.Funds.opening}", tag="openingbalance"
                    )
                    add_text(
                        f"Current Balance : {ana.Funds.balance}", tag="currentbalance"
                    )
                    add_text(f"Gross P&L : {ana.Funds.totalpnl :.2f}", tag="Gpnl")
                    add_text(f"Total charges: {ana.Position.charges}", tag="charges")
                    add_text(
                        f"Net P&L : {ana.Funds.totalpnl-ana.Position.charges}",
                        tag="Npnl",
                    )

                with child_window(
                    label="OPERATION LOGS",
                    tag="logconsole",
                    height=-1,
                    width=-1,
                    border=True,
                ):
                    add_text("OPERATION LOGS", color=(255, 255, 255))
                    add_separator()
                    self.logconsole = logger.mvLogger(parent="logconsole")
                    bind_item_theme("logconsole", self.windowtheme)

        with window(
            tag="underlying",
            label="UNDERLYING",
            height=0.75 * height,
            width=0.75 * width,
            show=False,
            no_title_bar=True,
        ):

            with plot(
                tag="underlying_plot",
                no_frame=True,
                tracked=True,
                crosshairs=True,
                pan_mod=True,
                height=-1,
                width=-1,
            ):
                add_plot_legend()
                add_plot_axis(
                    mvXAxis,
                    no_tick_labels=True,
                    tag="underlyingxaxis",
                )
                add_plot_axis(
                    mvYAxis,
                    tag="underlyingclose",
                )
                add_candle_series(
                    dates=[np.nan] * 50,
                    opens=[np.nan] * 50,
                    highs=[np.nan] * 50,
                    lows=[np.nan] * 50,
                    closes=[np.nan] * 50,
                    parent="underlyingclose",
                    tag="underlyingcloseline",
                )
                add_line_series(
                    [np.nan] * 50,
                    [np.nan] * 50,
                    parent="underlyingclose",
                    tag="underlyingsupertrend",
                )
            bind_item_theme("underlying_plot", self.plottheme)

        add_button(
            label=f"Buy {self.keys[0]}",
            callback=self.buy_button_handlerT1,
            tag="buybuttonT1",
            parent="controlpanel",
        )
        add_button(
            label="Underlying",
            callback=lambda: configure_item(
                item="underlying", show=not is_item_shown("underlying")
            ),
            tag="toggle_underlying",
            parent="controlpanel",
        )
        add_button(
            label=f"Buy {self.keys[1]}",
            callback=self.buy_button_handlerT2,
            tag="buybuttonT2",
            parent="controlpanel",
        )
        add_button(
            label="Sell",
            callback=self.sell_button_handler,
            tag="sellbutton",
            parent="controlpanel",
        )
        set_primary_window("primary", True)
        bind_theme(self.globaltheme)
        bind_font(self.basefont)

    def fastforward(self):
        if not config.latency == 0.000005:
            configure_item(item="fastforwardbutton", label="Slow Down")
            config.latency = 0.000005
            self.logconsole.log_info(f"Speed : {config.latency}")
        else:
            config.latency = 0.5
            configure_item(item="fastforwardbutton", label="Fast Forward")
            self.logconsole.log_info(f"Speed : {config.latency}")

    def sell_button_handler(self):
        self.Trader1.sell_order()
        self.Trader2.sell_order()

    def buy_button_handlerT1(self):
        self.Trader1.buy_order()

    def buy_button_handlerT2(self):
        self.Trader2.buy_order()

    def toggle_sandbox(self):
        if config.sandbox_orders:
            config.sandbox_orders = False
            self.logconsole.log_info("Sandbox deactivated.")
            configure_item("sandboxbutton", label="Sandbox: OFF")

        elif not config.sandbox_orders:
            config.sandbox_orders = True
            self.logconsole.log_info("Sandbox activated.")
            configure_item("sandboxbutton", label="Sanbox: ON")

    def replay(self):
        try:
            self.logconsole.clear_log()
            if not (config.market_close.is_set() or self.simulation):
                self.logconsole.log_info("Cannot replay during a live session.")
                return
            if not self.pause:
                self.pause_play()

            if hasattr(self, "tradethread") and self.tradethread.is_alive():
                for task in self.async_tasks:
                    self.loop.call_soon_threadsafe(task.cancel)

                self.tradethread.join()
            config.stopevent.clear()
            self.live_index = 0
            self.call_plotdata = ana.History(self.keys[0])
            self.put_plotdata = ana.History(self.keys[1])
            self.logconsole.log_info("Trade data has been reset.")
            self.updater()
            self.tradethread = threading.Thread(target=self.startasyncloop, daemon=True)
            self.tradethread.start()
            configure_item("playpausebutton", label="Start")
            self.logconsole.log_info("Ready to replay. Press Start to begin.")

        except Exception as e:

            tb = traceback.format_exc()
            self.logconsole.log_error(tb)
            print(tb)

    def toggle_simulation(self, switch):
        try:
            self.simulation = switch
            self.init_variables()
            self.draw_gui()
            if does_item_exist("initwin"):
                delete_item("initwin")
            self.tradethread = threading.Thread(target=self.startasyncloop, daemon=True)
            self.tradethread.start()

            if self.simulation:  # Simulation mode
                self.logconsole.log_info("Simulation Mode Active")
                ana.Funds.totalpnl = pos[-1]["pnl"] if len(pos) > 0 else 0
                ana.Funds.opening = ana.Funds.balance = 30000 
                config.market_close.set()
                self.updatefunds(val=ana.Funds.balance)
                add_button(
                    label="Fast Forward",
                    callback=self.fastforward,
                    tag="fastforwardbutton",
                    parent="controlpanel",
                )
                add_button(
                    label="Replay",
                    callback=self.replay,
                    tag="replaybutton",
                    parent="controlpanel",
                )
                add_progress_bar(
                    tag="progressbar", parent="controlpanel", label="Progress Bar"
                )

            else:

                add_button(
                    label="Sandbox : ON",
                    callback=self.toggle_sandbox,
                    tag="sandboxbutton",
                    parent="controlpanel",
                )

            return

        except Exception as e:
            tb = traceback.format_exc()
            print(tb)

    def pause_play(self):
        if self.pause:
            self.logconsole.log_info("Trading Engine Started")
            self.pause = False
            self.loop.call_soon_threadsafe(config.looprun.set)
            configure_item("playpausebutton", label="Pause")
            return
        elif not self.pause:
            self.logconsole.log_info("Trading Engine Paused")
            self.pause = True
            self.loop.call_soon_threadsafe(config.looprun.clear)
            configure_item("playpausebutton", label="Play")
            return
    def updatefunds(self, val=None):
        if val is None:
            ana.Funds.update_funds()
        set_value(value=f"Opening Balance : {ana.Funds.opening}", item="openingbalance")
        set_value(value=f"Current Balance : {ana.Funds.balance}", item="currentbalance")
        set_value(value=f"Gross P&L : {ana.Funds.totalpnl}", item="Gpnl")
        set_value(value=f"Total Charges : {ana.Position.charges}", item="charges")
        set_value(value=f"Net P&L : {ana.Position.charges}", item="Npnl")

    def updatepos(self):
        set_value(value=f"Key: {ana.Position.key}", item="poskey")
        set_value(value=f"Symbol: {ana.Position.symbol}", item="possym")
        set_value(value=f"Quantity: {ana.Position.quantity}", item="posqty")
        set_value(
            value=f"Current Value: {ana.Position.unrealised}", item="poscurrentval"
        )

    def shutdown(self):
        if not config.stopevent.is_set():
            print("Shutdown initiated.")
            self.logconsole.log_info("Shutdown initiated.")
            config.stopevent.set()

            if self.loop and self.loop.is_running():
                self.loop.call_soon_threadsafe(config.looprun.set)

            if hasattr(self, "async_tasks") and self.loop and self.loop.is_running():
                print("Cancelling async tasks.")
                for task in self.async_tasks:
                    self.loop.call_soon_threadsafe(task.cancel)

            print("Shutting down thread pool.")
            ana.pool.shutdown(wait=True, cancel_futures=True)
            print("Thread pool shut down.")

            if self.tradethread and self.tradethread.is_alive():
                print("Waiting for trading thread to join.")
                self.tradethread.join(timeout=5)
                if self.tradethread.is_alive():
                    print("Trading thread did not join gracefully.")
                else:
                    print("Trading thread joined.")
            print("Stopping Dear PyGui.")
            stop_dearpygui()

    def run(self):
        try:
            self.setup_gui()
            setup_dearpygui()
            set_exit_callback(callback=self.shutdown)
            maximize_viewport()
            show_viewport()
            report.report_window(show=False)
            render_dearpygui_frame()
            with theme() as inittheme:
                with theme_component(mvAll):
                    add_theme_color(
                        mvThemeCol_WindowBg, (65, 60, 62, 150), category=mvThemeCat_Core
                    )
                    add_theme_color(
                        mvThemeCol_Button, (62, 62, 62), category=mvThemeCat_Core
                    )

            if not config.market_close.is_set():  # Choices for open market
                with window(
                    height=self.height,
                    width=self.width,
                    pos=(0, 0),
                    tag="initwin",
                    no_collapse=True,
                    no_move=True,
                    no_resize=True,
                    no_close=True,
                    no_title_bar=True,
                ):
                    add_text(
                        "How would you like to proceed?",
                        pos=(self.width // 2 - 120, self.height // 2 - 100),
                    )
                    add_button(
                        label="Simulate",
                        tag="simbutton",
                        pos=(self.width // 2 - 100, self.height // 2 - 70),
                        callback=lambda: self.toggle_simulation(switch=True),
                    )
                    add_button(
                        label="Go Live",
                        tag="golivebutton",
                        pos=(self.width // 2, self.height // 2 - 70),
                        callback=lambda: self.toggle_simulation(switch=False),
                    )
                    bind_item_theme("initwin", inittheme)
                    bind_item_font("initwin", self.largefont)
            else:
                self.toggle_simulation(switch=True)
            render_dearpygui_frame()
            count = 0
            while is_dearpygui_running():
                needsupdate = False
                while not self.data_queue.empty():
                    try:
                        self.data_queue.get_nowait()
                        needsupdate = True
                    except Exception:
                        break
                if needsupdate:
                    self.updater()
                    if self.simulation and not config.current_index == self.livetotal:
                        set_value("progressbar", config.current_index / self.livetotal)
                    else:
                        self.logconsole.log_info("#------Simulation Complete------#")
                render_dearpygui_frame()
            destroy_context()
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)

    def updater(self, data=None):
        """Update the plots with the latest trade data."""
        if not self.pause:
            self.update_plots(
                "underlyingcloseline",
                xaxis="underlyingxaxis",
                yaxis="underlyingclose",
                fit=True,
                opens=self.scrip_plotdata.open,
                highs=self.scrip_plotdata.high,
                lows=self.scrip_plotdata.low,
                closes=self.scrip_plotdata.close,
            )

            self.update_plots(
                "underlyingsupertrend",
                xaxis="underlyingxaxis",
                yaxis="underlyingclose",
                ydata=self.scrip_plotdata.supertrend,
            )

            self.update_plots(
                "putsupertrend",
                xaxis="putxaxis",
                yaxis="putclose",
                ydata=self.put_plotdata.supertrend,
            )

            self.update_plots(
                "callsupertrend",
                xaxis="callxaxis",
                yaxis="callclose",
                ydata=self.call_plotdata.supertrend,
            )
            self.update_plots(
                "putcloseline",
                xaxis="putxaxis",
                yaxis="putclose",
                fit=True,
                ##kwargs
                opens=self.put_plotdata.open,
                highs=self.put_plotdata.high,
                lows=self.put_plotdata.low,
                closes=self.put_plotdata.close,
            )

            configure_item(
                "putclose_annot",
                default_value=(
                    get_axis_limits("putxaxis")[-1:] + self.put_plotdata.close[-1:]
                ),
                label=f"{self.put_plotdata.close[-1]:.2f}",
            )
            configure_item(
                "callclose_annot",
                default_value=(
                    get_axis_limits("callxaxis")[-1:] + self.call_plotdata.close[-1:]
                ),
                label=f"{self.call_plotdata.close[-1]:.2f}",
            )

            self.update_plots(
                "putbuys",
                xaxis="putxaxis",
                yaxis="putclose",
                ydata=self.put_plotdata.buys,
            )
            self.update_plots(
                "putsells",
                xaxis="putxaxis",
                yaxis="putclose",
                ydata=self.put_plotdata.sells,
            )

            self.update_plots(
                "putsells",
                xaxis="putxaxis",
                yaxis="putclose",
                ydata=self.put_plotdata.sells,
            )

            self.update_plots(
                "callcloseline",
                xaxis="callxaxis",
                yaxis="callclose",
                fit=True,
                ##kwargs
                opens=self.call_plotdata.open,
                highs=self.call_plotdata.high,
                lows=self.call_plotdata.low,
                closes=self.call_plotdata.close,
            )

            self.update_plots(
                "callbuys",
                xaxis="callxaxis",
                yaxis="callclose",
                ydata=self.call_plotdata.buys,
            )

            self.update_plots(
                "callsells",
                xaxis="callxaxis",
                yaxis="callclose",
                ydata=self.call_plotdata.sells,
            )

    def update_plots(self, target, xaxis, yaxis, ydata=None, fit=False, pad=2, **ohlc):
        try:
            if ohlc:
                min_len = min([len(ohlc[cols]) for cols in ohlc.keys()])
                for cols in ohlc.keys():
                    ohlc[cols] = list(ohlc[cols])[-min_len:]
                df = pd.DataFrame(ohlc)
                df = df.dropna().reset_index(drop=True)
                if df.empty:
                    print("DF Empty for", target)
                    return
                indices = np.arange(len(df))
                data = [
                    indices.tolist(),
                    df["opens"].tolist(),
                    df["closes"].tolist(),
                    df["lows"].tolist(),
                    df["highs"].tolist(),
                ]
                set_value(target, data)
                ymin = list(filter(lambda x: not math.isnan(x) and x > 0, ohlc["lows"]))
                ymax = list(
                    filter(lambda x: not math.isnan(x) and x > 0, ohlc["highs"])
                )

                if ymin and ymax:
                    if "underlyingcloseline" in target:
                        set_axis_limits(
                            yaxis,
                            ymin=min(ymin) - 10,
                            ymax=max(ymax) + 10,
                        )

                    else:
                        set_axis_limits(
                            yaxis,
                            ymin=min(ymin) - 10,
                            ymax=max(ymax) + 10,
                        )
                set_axis_limits(xaxis, ymin=indices[-1] - 50, ymax=indices[-1] + 10 )

                if "underlying" in target:
                    lims = [min(ymin) - 20000, max(ymax)+20000]
                    lower = lims[0] - (lims[0] % 50)
                    ticks = np.arange(lower, lims[1], 50)
                    if len(ticks) > 1 and not self.strikes:
                        add_inf_line_series(
                            tag="strikes", x=ticks, parent=yaxis, horizontal=True
                        )
                        self.strikes = True
            else:
                ydata = ydata + ([np.nan] * (200 - len(ydata)))

                xdata = np.arange(len(ydata))
                if ydata and target:
                    configure_item(target, y=ydata, x=xdata)
                points = len(ydata)
                xmax = max(60, points)
                xmin = max(0, xmax - 60)

                if fit:
                    visible_y_slice = ydata[xmin:xmax]
                    finite_y_in_slice = [
                        val for val in visible_y_slice if np.isfinite(val)
                    ]
                    if finite_y_in_slice:
                        ymin = min(finite_y_in_slice)
                        ymax = max(finite_y_in_slice)
                        set_axis_limits(yaxis, ymin=ymin - pad, ymax=ymax + pad)
                    else:
                        all_finite_y = [val for val in ydata if np.isfinite(val)]
                        if all_finite_y:
                            last_value = all_finite_y[-1]
                            set_axis_limits(
                                yaxis, ymin=last_value - pad, ymax=last_value + pad
                            )
        except Exception as e:
            print(target)
            print(ydata)
            tb = traceback.format_exc()
            self.logconsole.log_error(tb)
            print(tb)

    def startasyncloop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.async_main())

    async def async_main(self):
        try:
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
                #            if self.simulation:
                tasks_to_run = [
                    ana.sim_live(
                        keys=self.keys,
                        buffer=tick_buffer,
                        current_index=self.live_index,
                        total_indices=self.livetotal,
                    ),
                    ana.publisher(buffer=tick_buffer),
                    self.Trader1.analyse(),
                    self.Trader2.analyse(),
                    self.Trader_under.analyse(),
                ]
            else:
                tasks_to_run = [
                    ustox.get_live(
                        access_token=ustox.access_token,
                        instrument_key=self.keys,
                        output=tick_buffer,
                        write=False,
                    ),
                    ana.publisher(buffer=tick_buffer),
                    self.Trader1.analyse(),
                    self.Trader2.analyse(),
                    self.Trader_under.analyse(),
                ]

            self.async_tasks = [asyncio.create_task(coro) for coro in tasks_to_run]
            await asyncio.gather(*self.async_tasks)

        except asyncio.CancelledError:
            self.logconsole.log_info("Async tasks cancelled.")
        except KeyboardInterrupt:
            config.stopevent.set()
        except Exception as e:
            tb = traceback.format_exc()
            print(tb)


# ========================================================================MAIN==========================================================

pos = ustox.get_positions()
if len(pos) > 0:
    ana.Position.open_position()
    ana.Position.close_position()
else:

    ana.Position.history.append(ana.Position.snapshot())

ana.Funds.update_funds()

# Initial setup
plotting = False
ana.Funds.opening = ana.Funds.balance = 30000
margin = ustox.get_funds()
ana.Funds.opening = ana.Funds.balance = (
    margin["equity"]["available_margin"] if margin else 30000
)
pos = ana.Position.history
market_status = ustox.exchanges_status(ustox.access_token, exchange="NSE")
logfile = open("./testlog.txt", "w")
logfile.write("")
logfile.close()
logfile = open("./testlog.txt", "a")
if market_status != "NORMAL_OPEN":
    config.market_close.set()
    ana.Funds.totalpnl = pos[-1]["pnl"] if len(pos) > 0 else 0
    ana.Funds.opening = ana.Funds.balance = 30000


async def main(dateidx=None):
    try:
        gui = App(dateidx=dateidx)
        gui.run()
    except KeyboardInterrupt:
        gui.shutdown()
    except Exception as e:
        print("An error occurred in the main GUI loop:")
        tb = traceback.format_exc()
        print(tb)
        gui.shutdown_app()
    finally:
        if not config.stopevent.is_set():
            config.stopevent.set()
        print("Main function finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main(43))
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        config.stopevent.set()
        exit(1)
