from datetime import datetime, timezone
from unittest.mock import patch
import pandas as pd
import pytest

from algo.core.datatypes import Position
from algo.core.instrument import Bucket, Instrument
from algo.core.master import InstrumentMaster
from algo.core.portfolio import Portfolio, Trade


@pytest.fixture
def mock_master():
    sample_registry = {
        "NSE_INDEX|Nifty 50": {
            "exchange": "NSE",
            "segment": "NSE_INDEX",
            "instrument_type": "INDEX",
            "lot_size": 1,
            "tick_size": 0.05,
            "strike_price": 0.0,
            "freeze_quantity": 0.0,
            "expiry": None,
        },
        "NSE_FO|58529": {
            "exchange": "NSE",
            "segment": "NSE_FO",
            "instrument_type": "CE",
            "lot_size": 25,
            "tick_size": 0.05,
            "strike_price": 24500.0,
            "freeze_quantity": 1800.0,
            "expiry": "2026-10-29",
        },
        "NSE_FO|58530": {
            "exchange": "NSE",
            "segment": "NSE_FO",
            "instrument_type": "PE",
            "lot_size": 25,
            "tick_size": 0.05,
            "strike_price": 24500.0,
            "freeze_quantity": 1800.0,
            "expiry": "2026-10-29",
        },
    }
    with patch.object(InstrumentMaster, "_load_master"):
        master = InstrumentMaster()
        master._registry = sample_registry
        yield master


class TestBucket:
    def test_bucket_initialization(self, mock_master):
        now = datetime(2026, 6, 15, 9, 15)
        spot_ins = Instrument("NSE_INDEX|Nifty 50")
        bucket = Bucket(date=now, tag="NIFTY_STRADDLE", spot=spot_ins)

        assert bucket.date == now
        assert bucket.tag == "NIFTY_STRADDLE"
        assert bucket.spot.key == "NSE_INDEX|Nifty 50"
        assert len(bucket.legs) == 0
        assert bucket.open_position is None

    def test_bucket_add_leg_with_inferred_type(self, mock_master):
        bucket = Bucket(date=datetime(2026, 6, 15))
        ce_leg = Instrument("NSE_FO|58529")

        bucket.add_leg(ce_leg)
        assert "CE" in bucket.legs
        assert bucket.legs["CE"].key == "NSE_FO|58529"

    def test_bucket_add_leg_with_explicit_alias(self, mock_master):
        bucket = Bucket(date=datetime(2026, 6, 15))
        ce_leg = Instrument("NSE_FO|58529")

        bucket.add_leg(ce_leg, leg_type="PRIMARY_CE")
        assert "PRIMARY_CE" in bucket.legs
        assert bucket.legs["PRIMARY_CE"].key == "NSE_FO|58529"

    def test_bucket_add_leg_dict(self, mock_master):
        bucket = Bucket(date=datetime(2026, 6, 15))
        ce_leg = Instrument("NSE_FO|58529")
        pe_leg = Instrument("NSE_FO|58530")

        bucket.add_leg({"ATM_CE": ce_leg, "ATM_PE": pe_leg})
        assert len(bucket.legs) == 2
        assert bucket.legs["ATM_CE"].strike_price == 24500.0
        assert bucket.legs["ATM_PE"].strike_price == 24500.0

    def test_bucket_tracks_open_position(self, mock_master):
        bucket = Bucket(date=datetime(2026, 6, 15))
        pos = Position(trading_symbol="NIFTY26OCT24500CE", quantity=25, buy_price=120.0)
        bucket.open_position = pos

        assert bucket.open_position.quantity == 25
        assert bucket.open_position.buy_price == 120.0


class TestPortfolioAndTrade:
    def test_trade_creation_defaults(self):
        trade = Trade(
            trade_id="TRD_001",
            instrument_key="NSE_FO|58529",
            side="BUY",
            buy_price=100.50,
            buy_qty=50,
            sell_price=120.50,
            sell_qty=50,
            movement=20.0,
            pnl=1000.0,
        )
        assert trade.trade_id == "TRD_001"
        assert trade.pnl == 1000.0
        assert trade.movement == 20.0
        assert trade.remark == ""

    def test_portfolio_empty_report_handling(self):
        portfolio = Portfolio()
        df = portfolio.get_report(verbose=False)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_portfolio_report_generation(self):
        portfolio = Portfolio()
        t1 = Trade(
            trade_id="T1",
            instrument_key="NSE_FO|58529",
            buy_timestamp=datetime(2026, 6, 15, 9, 20),
            buy_price=100.0,
            buy_qty=25,
            sell_timestamp=datetime(2026, 6, 15, 9, 45),
            sell_price=115.0,
            sell_qty=25,
            movement=15.0,
            pnl=375.0,
            remark="TARGET_HIT",
        )
        t2 = Trade(
            trade_id="T2",
            instrument_key="NSE_FO|58530",
            buy_timestamp=datetime(2026, 6, 15, 10, 0),
            buy_price=80.0,
            buy_qty=25,
            sell_timestamp=datetime(2026, 6, 15, 10, 15),
            sell_price=70.0,
            sell_qty=25,
            movement=-10.0,
            pnl=-250.0,
            remark="SL_HIT",
        )
        portfolio.report.extend([t2, t1])

        df = portfolio.get_report(verbose=False)

        assert len(df) == 2
        # Validates chronological ordering by buy_timestamp
        assert df.iloc[0]["trade_id"] == "T1"
        assert df.iloc[1]["trade_id"] == "T2"
        assert df["pnl"].sum() == 125.0
        assert df["movement"].sum() == 5.0

    def test_portfolio_position_state_mutation(self):
        portfolio = Portfolio()
        pos = Position(trading_symbol="NIFTY24500CE", quantity=50, buy_price=100.0)
        portfolio.positions["NSE_FO|58529"] = pos

        assert "NSE_FO|58529" in portfolio.positions
        portfolio.positions["NSE_FO|58529"].update_position(quantity=0, realised=500.0)
        assert portfolio.positions["NSE_FO|58529"].quantity == 0
        assert portfolio.positions["NSE_FO|58529"].realised == 500.0
