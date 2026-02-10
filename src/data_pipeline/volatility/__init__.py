"""
Volatility Pipeline Module

Pipeline for collecting data based on daily top gainers and top losers.
Strategy: Identify high-volatility coins and trade momentum/reversal.
"""

from .daily_downloader import DailyDownloader
from .top_movers_calculator import TopMoversCalculator
from .five_minute_downloader import FiveMinuteDownloader
from .main_pipeline import VolatilityPipeline

__all__ = [
    "DailyDownloader",
    "TopMoversCalculator",
    "FiveMinuteDownloader",
    "VolatilityPipeline",
]
