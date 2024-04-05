import pandas as pd
from pathlib import Path
import json

import datetime
BASE_DIR = Path(__file__).parent
config_file = str(BASE_DIR / "config.json")



class config_object_class(object):

    def __init__(self):

        with open(config_file, "r") as f:
            config = json.load(f)

        self.config = config
        self.symbol =self.config.get("symbol", None)
        self.entry_start_time = pd.Timestamp(self.config.get("entry_start_time", None)).time() 
        self.entry_end_time = pd.Timestamp(self.config.get("entry_end_time", None)).time() 
        self.exit_time = pd.Timestamp(self.config.get("exit_time", None)).time() 
        self.backtest_start_date = pd.Timestamp(self.config.get("backtest_start_date", None)).date() 
        self.backtest_end_date = pd.Timestamp(self.config.get("backtest_end_date", None)).date() 
        
        self.moneyness = self.config.get("moneyness", 0)
        self.fixed_SL = self.config.get("fixed_SL", 1)
        
        self.price_rng_low = self.config.get("price_rng_low", None)
        self.price_rng_high = self.config.get("price_rng_high", None) 
        
        self.minSL = self.config.get("minSL", None)
        self.margin_contract = self.config.get("margin_contract", None)
        
        self.commision_per_lot = self.config.get("comm_brokerage_per_lot", None)
        
        
        self.multiplier = self.config.get("multiplier", None)
        self.position_capital = self.config.get("position_capital", None)
        
        self.starting_capital = self.config.get("starting_capital", None)
        
        self.timeframe = self.config.get("timeframe", None)
        self.safety_margin_multiplier = self.config.get("safety_margin_multiplier",None)
        self.spread_distance_from_low = self.config.get("spread_distance_from_low",None)
        self.days_to_hold = self.config.get("days_to_hold",None)
        self.intraday = self.config.get("intraday",None)

        self.vix_perc = self.config.get("vix_perc", None)

        self.strategy_name = self.config.get("strategy_name", None)

        self.net_slippage = self.config.get("net_slippage", None)
        self.legs_info = self.config.get("legs_info",None)
        self.param_to_trade = self.config.get("param_to_trade",None)
        
config_object = config_object_class()

