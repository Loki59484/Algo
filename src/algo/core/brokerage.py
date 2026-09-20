import numpy as np
from typing import Literal
from algo.core.datatypes import Order

class BrokerageEngine:
    """
    Stateful calculator that maintains running totals of all taxation and brokerage.
    Feed it completed Order objects directly from the websocket stream.
    """
    def __init__(self):
        self.total_brokerage = 0.0
        self.total_stt = 0.0
        self.total_txn = 0.0
        self.total_sebi = 0.0
        self.total_stamp = 0.0
        self.total_gst = 0.0
        self.total_charges = 0.0

    def add_charge(
        self, 
        order: Order, 
        instrument: Literal['equity', 'futures', 'options'] = 'options', 
        trade_type: Literal['intraday', 'delivery'] = 'intraday'
    ) -> float:
        """
        Calculates and adds charges for a single executed order leg.
        Returns the exact charge deducted for this specific leg.
        """
        if not order.average_price or not order.filled_quantity:
            return 0.0
            
        turnover = order.average_price * order.filled_quantity
        is_buy = order.transaction_type == 'BUY'
        is_sell = order.transaction_type == 'SELL'

        # 1. BROKERAGE (Charged on both Buy and Sell legs)
        if instrument == 'options':
            brokerage = 20.0 
        elif instrument == 'futures':
            brokerage = min(20.0, turnover * 0.0005)
        elif instrument == 'equity':
            if trade_type == 'delivery':
                brokerage = 20.0 
            else: 
                brokerage = min(20.0, turnover * 0.001)
        else:
            raise ValueError("Instrument must be 'equity', 'futures', or 'options'.")

        # 2. STT (Securities Transaction Tax)
        stt = 0.0
        if is_sell:
            if instrument == 'options':
                stt = np.round(turnover * 0.001)
            elif instrument == 'futures':
                stt = np.round(turnover * 0.0002)
            elif instrument == 'equity' and trade_type == 'intraday':
                stt = np.round(turnover * 0.00025)
        
        # Equity delivery STT applies to BOTH buy and sell
        if instrument == 'equity' and trade_type == 'delivery':
            stt = np.round(turnover * 0.001)

        # 3. EXCHANGE TRANSACTION CHARGES
        if instrument == 'options':
            txn_charge = turnover * 0.000495   
        elif instrument == 'futures':
            txn_charge = turnover * 0.0000188  
        elif instrument == 'equity':
            txn_charge = turnover * 0.0000345  

        # 4. SEBI CHARGES
        sebi_charge = turnover * 0.000001 

        # 5. STAMP DUTY (Charged on Buy Side Only)
        stamp_duty = 0.0
        if is_buy:
            if instrument == 'options':
                stamp_duty = np.round(turnover * 0.00003)
            elif instrument == 'futures':
                stamp_duty = np.round(turnover * 0.00002)
            elif instrument == 'equity':
                if trade_type == 'delivery':
                    stamp_duty = np.round(turnover * 0.00015)
                else: 
                    stamp_duty = np.round(turnover * 0.00003)

        # 6. GST (18% on Brokerage + Txn Charges + SEBI)
        gst = (brokerage + txn_charge + sebi_charge) * 0.18

        # Tally State
        leg_total = brokerage + stt + txn_charge + sebi_charge + stamp_duty + gst
        
        self.total_brokerage += brokerage
        self.total_stt += stt
        self.total_txn += txn_charge
        self.total_sebi += sebi_charge
        self.total_stamp += stamp_duty
        self.total_gst += gst
        self.total_charges += leg_total

        return round(leg_total, 2)

    @property
    def summary(self) -> dict:
        """Returns the current breakdown of all accrued charges."""
        return {
            "Brokerage": round(self.total_brokerage, 2),
            "STT": round(self.total_stt, 2),
            "Transaction Charge": round(self.total_txn, 2),
            "SEBI Charge": round(self.total_sebi, 2),
            "Stamp Duty": round(self.total_stamp, 2),
            "GST": round(self.total_gst, 2),
            "Total": round(self.total_charges, 2)
        }
