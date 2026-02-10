"""
Top Movers Calculator

Analyzes daily data to identify top 10 gainers and top 10 losers for each trading day.

Timeline visualization:
    Daily data for all symbols
        │
        ▼
    ┌──────────────────────────────────────────────────────────────┐
    │  For each date (00:00 UTC):                                  │
    │                                                              │
    │  1. Get all symbols that have data on that date              │
    │  2. Calculate 24h % change: (close - open) / open * 100      │
    │  3. Rank by % change                                         │
    │  4. Take top 10 gainers (highest %)                          │
    │  5. Take top 10 losers (lowest %)                            │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘
        │
        ▼
    Save to top_movers.parquet with format:
    ┌──────────────┬─────────────────────────────┬─────────────────────────────┐
    │ date         │ gainers                     │ losers                      │
    │──────────────│─────────────────────────────│─────────────────────────────│
    │ 2020-01-01   │ [DOGE, SOL, XRP, ...]       │ [AAVE, LINK, ...]           │
    │ 2020-01-02   │ [ETH, BNB, ADA, ...]        │ [DOT, UNI, ...]             │
    └──────────────┴─────────────────────────────┴─────────────────────────────┘
"""

import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopMoversCalculator:
    """
    Calculates top gainers and top losers from daily data.

    Supports incremental mode: if top_movers.parquet already exists,
    only calculates for dates that don't exist yet.
    """

    def __init__(
        self,
        daily_directory: str = "data/volatility/daily",
        output_file: str = "data/volatility/top_movers.parquet",
        top_count: int = 10,
    ):
        """
        Initialize the calculator.

        Args:
            daily_directory: Directory containing daily parquet files per symbol
            output_file: Path to output file for results
            top_count: Number of top gainers and losers to extract (default: 10)
        """
        self.daily_directory = Path(daily_directory)
        self.output_file = Path(output_file)
        self.top_count = top_count

        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_all_daily_data(self) -> pl.DataFrame:
        """
        Load all daily data from directory into one DataFrame.

        Returns:
            DataFrame with columns: open_time, open, close, volume, symbol
        """
        all_files = list(self.daily_directory.glob("*.parquet"))

        if not all_files:
            raise FileNotFoundError(f"No parquet files found in {self.daily_directory}")

        all_data = []
        for file in all_files:
            symbol = file.stem
            if symbol.startswith("_"):  # Skip internal files
                continue

            try:
                df = pl.read_parquet(file)
                df = df.select(
                    ["open_time", "open", "close", "volume", "quote_volume"]
                ).with_columns(pl.lit(symbol).alias("symbol"))
                all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {file}: {e}")
                continue

        if not all_data:
            raise ValueError("No data successfully loaded")

        df_combined = pl.concat(all_data)
        logger.info(
            f"Loaded {len(all_data)} symbols with total {len(df_combined)} rows"
        )

        return df_combined

    def _calculate_daily_change(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate daily percentage change for each symbol.

        Change is calculated as: (close - open) / open * 100

        Args:
            df: DataFrame with columns open_time, open, close, symbol

        Returns:
            DataFrame with additional 'change_percent' column
        """
        df = df.with_columns(
            [
                # Extract date (without time) for grouping
                pl.col("open_time").dt.date().alias("date"),
                # Calculate percentage change
                ((pl.col("close") - pl.col("open")) / pl.col("open") * 100).alias(
                    "change_percent"
                ),
            ]
        )

        return df

    def _check_existing_data(self) -> Tuple[Optional[datetime], Set[str]]:
        """
        Check what data already exists in output file.

        Returns:
            Tuple (last_date, set_of_existing_dates)
        """
        if not self.output_file.exists():
            return None, set()

        try:
            df = pl.read_parquet(self.output_file)
            date_set = set(df["date"].cast(pl.Utf8).to_list())
            last_date = df["date"].max()
            return last_date, date_set
        except Exception as e:
            logger.warning(f"Failed to read existing file: {e}")
            return None, set()

    def _get_top_gainers_losers_for_date(self, df: pl.DataFrame, date) -> dict:
        """
        Get top gainers and losers for one date.

        Args:
            df: DataFrame with change_percent already calculated
            date: Date to process

        Returns:
            Dictionary with keys: date, gainers, losers, gainers_pct, losers_pct
        """
        # Filter for this date
        df_date = df.filter(pl.col("date") == date)

        if len(df_date) == 0:
            return None

        # Filter symbols with minimum volume (avoid dead coins)
        # Minimum $100k volume per day
        df_date = df_date.filter(pl.col("quote_volume") > 100_000)

        if len(df_date) < self.top_count * 2:
            logger.debug(
                f"Date {date}: Only {len(df_date)} symbols with sufficient volume"
            )

        # Sort and get top gainers
        df_sorted = df_date.sort("change_percent", descending=True)

        top_gainers = df_sorted.head(self.top_count)
        top_losers = df_sorted.tail(self.top_count).sort("change_percent")

        return {
            "date": date,
            "gainers": top_gainers["symbol"].to_list(),
            "losers": top_losers["symbol"].to_list(),
            "gainers_pct": top_gainers["change_percent"].to_list(),
            "losers_pct": top_losers["change_percent"].to_list(),
        }

    def calculate(self, incremental_mode: bool = True) -> pl.DataFrame:
        """
        Calculate top gainers and losers for all dates.

        Args:
            incremental_mode: If True, only calculate dates that don't exist yet

        Returns:
            DataFrame with results
        """
        logger.info("Loading daily data...")
        df = self._load_all_daily_data()

        logger.info("Calculating daily changes...")
        df = self._calculate_daily_change(df)

        # Get all unique dates
        all_dates = df["date"].unique().sort().to_list()
        logger.info(f"Found {len(all_dates)} unique dates")

        # Check existing data for incremental mode
        last_existing_date, existing_dates_set = self._check_existing_data()

        if incremental_mode and existing_dates_set:
            # Filter only dates that don't exist
            all_dates = [d for d in all_dates if str(d) not in existing_dates_set]
            logger.info(f"Incremental mode: {len(all_dates)} new dates to process")

        if not all_dates:
            logger.info("No new dates to process")
            if self.output_file.exists():
                return pl.read_parquet(self.output_file)
            return pl.DataFrame()

        # Process each date
        results = []
        for date in all_dates:
            result = self._get_top_gainers_losers_for_date(df, date)
            if result:
                results.append(result)

        if not results:
            logger.warning("No results generated")
            if self.output_file.exists():
                return pl.read_parquet(self.output_file)
            return pl.DataFrame()

        # Create DataFrame from new results
        df_new_results = pl.DataFrame(
            {
                "date": [r["date"] for r in results],
                "gainers": [r["gainers"] for r in results],
                "losers": [r["losers"] for r in results],
                "gainers_pct": [r["gainers_pct"] for r in results],
                "losers_pct": [r["losers_pct"] for r in results],
            }
        )

        # Combine with existing data if incremental
        if incremental_mode and self.output_file.exists():
            df_old = pl.read_parquet(self.output_file)
            df_results = pl.concat([df_old, df_new_results])
            df_results = df_results.unique(subset=["date"], maintain_order=True)
            df_results = df_results.sort("date")
        else:
            df_results = df_new_results.sort("date")

        # Save
        df_results.write_parquet(self.output_file)
        logger.info(f"Saved {len(df_results)} dates to {self.output_file}")
        logger.info(
            f"Date range: {df_results['date'].min()} - {df_results['date'].max()}"
        )

        return df_results

    def load_results(self) -> pl.DataFrame:
        """
        Load calculation results from file.

        Returns:
            DataFrame with top movers per date
        """
        if not self.output_file.exists():
            raise FileNotFoundError(
                f"File {self.output_file} not found. " "Run calculate() first."
            )

        return pl.read_parquet(self.output_file)

    def get_unique_symbols(self) -> Set[str]:
        """
        Get all unique symbols that have ever been in top movers.

        Returns:
            Set of unique symbols
        """
        df = self.load_results()

        symbol_set = set()

        # Get from gainers
        for gainers in df["gainers"].to_list():
            symbol_set.update(gainers)

        # Get from losers
        for losers in df["losers"].to_list():
            symbol_set.update(losers)

        logger.info(f"Found {len(symbol_set)} unique symbols in top movers")
        return symbol_set

    def get_download_schedule(self) -> pl.DataFrame:
        """
        Generate schedule of (date, symbol) pairs that need 5m data downloaded.

        Returns:
            DataFrame with columns: date, symbol
        """
        df = self.load_results()

        schedule = []
        for row in df.iter_rows(named=True):
            date = row["date"]
            # Combine gainers and losers
            for symbol in row["gainers"] + row["losers"]:
                schedule.append({"date": date, "symbol": symbol})

        df_schedule = pl.DataFrame(schedule)
        df_schedule = df_schedule.unique()  # Remove duplicates if any

        logger.info(
            f"Download schedule: {len(df_schedule)} (date, symbol) combinations"
        )

        return df_schedule

    def statistics(self) -> dict:
        """
        Generate statistics from top movers data.

        Returns:
            Dictionary with various statistics
        """
        df = self.load_results()
        unique_symbols = self.get_unique_symbols()

        # Count frequency of each symbol
        frequency = {}
        for row in df.iter_rows(named=True):
            for symbol in row["gainers"] + row["losers"]:
                frequency[symbol] = frequency.get(symbol, 0) + 1

        # Sort by frequency
        frequency_sorted = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

        return {
            "total_dates": len(df),
            "date_range": {
                "start": str(df["date"].min()),
                "end": str(df["date"].max()),
            },
            "unique_symbols_count": len(unique_symbols),
            "top_10_most_frequent": frequency_sorted[:10],
            "total_download_combinations": len(df) * self.top_count * 2,
        }


if __name__ == "__main__":
    # Test calculation
    calculator = TopMoversCalculator()

    # Calculate top movers
    df = calculator.calculate()
    print(f"\nResults: {len(df)} dates")
    print(df.head(5))

    # Show statistics
    stats = calculator.statistics()
    print("\nStatistics")
    print(f"    Total dates: {stats['total_dates']}")
    print(f"    Date range: {stats['date_range']}")
    print(f"    Unique symbols: {stats['unique_symbols_count']}")
    print(f"    Most frequent: {stats['top_10_most_frequent'][:5]}")
