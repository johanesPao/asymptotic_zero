"""
Main Volatility Pipeline

Orchestrator that runs the entire data collection pipeline.
Combines:
1. DailyDownloader - Download daily candles for all symbols
2. TopMoversCalculator - Identify top gainers/losers per day
3. FiveMinuteDownloader - Download 5m data only for top movers

Timeline visualization:
════════════════════════════════════════════════════════════════════════════════

FIRST RUN (Fresh Start)
────────────────────────────────────────────────────────────────────────────────

    [1] Download daily data for ALL symbols (5 years)
        │
        │   ~300 symbols × ~1825 days = ~550K candles
        │   Size: ~50-100MB
        │   Time: ~30-60 minutes
        │
        ▼
    [2] Calculate top 10 gainers + 10 losers per day
        │
        │   1825 days × 20 coins = ~36,500 combinations
        │   Size: ~1MB
        │   Time: ~1 minute
        │
        ▼
    [3] Download 5m data for each combination
        │
        │   ~36,500 files × 288 candles = ~10.5M candles
        │   Size: ~1-2GB
        │   Time: ~6-12 hours (with rate limiting)
        │
        ▼
    ✅ DONE - Data ready for training!


RE-RUN (Incremental Update)
────────────────────────────────────────────────────────────────────────────────

    [1] Download daily data for NEW days only (gap fill)
        │
        │   Only download the difference
        │   Time: ~1-5 minutes
        │
        ▼
    [2] Calculate top movers for NEW dates only
        │
        │   Append to existing file
        │   Time: ~few seconds
        │
        ▼
    [3] Download 5m data that DOESN'T EXIST yet
        │
        │   Skip files that already exist
        │   Time: proportional to new data
        │
        ▼
    ✅ DONE - Data updated!

════════════════════════════════════════════════════════════════════════════════
"""

import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging
import yaml
import time

from .daily_downloader import DailyDownloader
from .top_movers_calculator import TopMoversCalculator
from .five_minute_downloader import FiveMinuteDownloader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _sanitize_for_yaml(obj):
    """
    Recursively convert Python objects to YAML-safe types.
    Converts tuples to lists, handles nested structures.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_yaml(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_yaml(item) for item in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


class VolatilityPipeline:
    """
    Main orchestrator for the volatility data collection pipeline.

    Usage:
        pipeline = VolatilityPipeline()
        pipeline.run() # Run full pipeline

        # Or run step by step:
        pipeline.step_1_download_daily()
        pipeline.step_2_calculate_top_movers()
        pipeline.step_3_download_5m()
    """

    def __init__(
        self,
        base_directory: str = "data/volatility",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        top_count: int = 10,
    ):
        """
        Initialize the pipeline.

        Args:
            base_directory: Base directory for all volatility data
            start_date: Start date for data collection
            end_date: End date (default: yesterday)
            top_count: Number of top gainers/losers per day
        """
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

        self.start_date = start_date
        self.end_date = end_date or (datetime.now() - timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )
        self.top_count = top_count

        # Initialize components
        self.daily_downloader = DailyDownloader(
            output_directory=str(self.base_directory / "daily")
        )

        self.top_movers_calculator = TopMoversCalculator(
            daily_directory=str(self.base_directory / "daily"),
            output_file=str(self.base_directory / "top_movers.parquet"),
            top_count=top_count,
        )

        self.five_minute_downloader = FiveMinuteDownloader(
            output_directory=str(self.base_directory / "5m"),
            top_movers_file=str(self.base_directory / "top_movers.parquet"),
        )

        # State file for tracking progress
        self.state_file = self.base_directory / "_pipeline_state.yaml"

    def _save_state(self, state: dict):
        """Save pipeline state to file."""
        sanitized_state = _sanitize_for_yaml(state)
        with open(self.state_file, "w") as f:
            yaml.dump(sanitized_state, f, default_flow_style=False)

    def _load_state(self) -> dict:
        """Load pipeline state from file."""
        if not self.state_file.exists():
            return {}
        with open(self.state_file, "r") as f:
            return yaml.safe_load(f) or {}

    def step_1_download_daily(self) -> dict:
        """
        Step 1: Download daily data for all symbols.

        Returns:
            Dictionary with download statistics
        """
        logger.info("=" * 70)
        logger.info("STEP 1: Download Daily Data (1D)")
        logger.info("=" * 70)

        start_time = time.time()

        statistics = self.daily_downloader.download_all_symbols(
            start_date=self.start_date, end_date=self.end_date, incremental_mode=True
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Step 1 completed in {duration:.1f} seconds")

        # Update state
        state = self._load_state()
        state["step_1"] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": duration,
            "statistics": {
                k: v for k, v in statistics.items() if k != "failed_symbols"
            },
        }
        self._save_state(state)

        return statistics

    def step_2_calculate_top_movers(self) -> pl.DataFrame:
        """
        Step 2: Calculate top gainers and losers per day.

        Returns:
            DataFrame with top movers per date.
        """
        logger.info("=" * 70)
        logger.info("STEP 2: Calculate Top Movers")
        logger.info("=" * 70)

        start_time = time.time()

        df = self.top_movers_calculator.calculate(incremental_mode=True)
        statistics = self.top_movers_calculator.statistics()

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Step 2 completed in {duration:.1f} seconds")
        logger.info(f"Total dates: {statistics['total_dates']}")
        logger.info(f"Unique symbols: {statistics['unique_symbols_count']}")

        # Update state
        state = self._load_state()
        state["step_2"] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": duration,
            "statistics": statistics,
        }
        self._save_state(state)

        return df

    def step_3_download_5m(
        self, limit: Optional[int] = None, delay_between_requests: float = 0.1
    ) -> dict:
        """
        Step 3: Download 5m data for top movers.

        Args:
            limit: Limit number of downloads (for testing)
            delay_between_requests: Delay between requests in seconds

        Returns:
            Dictionary with download statistics
        """
        logger.info("=" * 70)
        logger.info("STEP 3: Download 5-Minute Data")
        logger.info("=" * 70)

        start_time = time.time()

        statistics = self.five_minute_downloader.download_all_from_top_movers(
            limit=limit, delay_between_requests=delay_between_requests
        )

        end_time = time.time()
        duration = end_time - start_time

        logger.info(f"Step 3 completed in {duration:.1f} seconds")

        # Update state
        state = self._load_state()
        state["step_3"] = {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "duration_seconds": duration,
            "statistics": {k: v for k, v in statistics.items() if k != "failed_list"},
        }
        self._save_state(state)

        return statistics

    def run(
        self,
        skip_step_1: bool = False,
        skip_step_2: bool = False,
        skip_step_3: bool = False,
        limit_5m: Optional[int] = None,
    ) -> dict:
        """
        Run the entire pipeline.

        Args:
            skip_step_1: Skip daily download
            skip_step_2: Skip top movers calculation
            skip_step_3: Skip 5m download
            limit_5m: Limit number of 5m downloads (for testing)

        Returns:
            Dictionary with results from all steps
        """
        logger.info("=" * 70)
        logger.info("STARTING VOLATILITY PIPELINE")
        logger.info(f"Date range: {self.start_date} - {self.end_date}")
        logger.info(f"Top movers: {self.top_count} gainers + {self.top_count} losers")
        logger.info("=" * 70)

        total_start_time = time.time()
        results = {}

        # Step 1
        if not skip_step_1:
            results["step_1"] = self.step_1_download_daily()
        else:
            logger.info("Step 1 skipped")

        # Step 2
        if not skip_step_2:
            results["step_2"] = self.step_2_calculate_top_movers()
        else:
            logger.info("Step 2 skipped")

        # Step 3
        if not skip_step_3:
            results["step_3"] = self.step_3_download_5m(limit=limit_5m)
        else:
            logger.info("Step 3 skipped")

        total_end_time = time.time()
        total_duration = total_end_time - total_start_time

        logger.info("=" * 70)
        logger.info(f"PIPELINE COMPLETED IN {total_duration:.1f} seconds")
        logger.info("=" * 70)

        return results

    def status(self) -> dict:
        """
        Show current pipeline status.

        Returns:
            Dictionary with status of each component
        """
        state = self._load_state()

        # Check daily data
        try:
            daily_symbols = self.daily_downloader.fetch_perpetual_symbols()
            daily_files = list((self.base_directory / "daily").glob("*.parquet"))
            daily_files = [f for f in daily_files if not f.name.startswith("_")]
        except Exception:
            daily_symbols = []
            daily_files = []

        # Check top movers
        top_movers_file = self.base_directory / "top_movers.parquet"
        if top_movers_file.exists():
            df_top = pl.read_parquet(top_movers_file)
            top_movers_info = {
                "total_dates": len(df_top),
                "date_range": f"{df_top['date'].min()} - {df_top['date'].max()}",
            }
        else:
            top_movers_info = None

        # Check 5m data
        stats_5m = self.five_minute_downloader.statistics()

        return {
            "saved_state": state,
            "daily_data": {
                "total_symbols_available": len(daily_symbols),
                "files_downloaded": len(daily_files),
            },
            "top_movers": top_movers_info,
            "five_minute_data": stats_5m,
        }

    def summary(self):
        """Print pipeline status summary."""
        status = self.status()

        print("\n" + "=" * 70)
        print("VOLATILITY PIPELINE STATUS SUMMARY")
        print("=" * 70)

        print("\n📊 Daily Data (1D):")
        print(
            f"   Symbols available: {status['daily_data']['total_symbols_available']}"
        )
        print(f"   Files downloaded: {status['daily_data']['files_downloaded']}")

        print("\n🏆 Top Movers:")
        if status["top_movers"]:
            print(f"   Total dates: {status['top_movers']['total_dates']}")
            print(f"   Date range: {status['top_movers']['date_range']}")
        else:
            print("   Not calculated yet")

        print("\n📈 5-Minute Data:")
        print(f"   Total symbols: {status['five_minute_data']['total_symbols']}")
        print(f"   Total files: {status['five_minute_data']['total_files']}")

        print("\n" + "=" * 70)


def main():
    """Entry point for running the pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Volatility Pipeline - Data Collection for DQN Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m src.data_pipeline.volatility.main_pipeline              # Run all steps
    python -m src.data_pipeline.volatility.main_pipeline --status     # Check status only
    python -m src.data_pipeline.volatility.main_pipeline --step 1     # Only step 1
    python -m src.data_pipeline.volatility.main_pipeline --step 2     # Only step 2
    python -m src.data_pipeline.volatility.main_pipeline --step 3     # Only step 3
    python -m src.data_pipeline.volatility.main_pipeline --limit 10   # Test with 10 5m files
        """,
    )

    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD), default 2020-01-01",
    )
    parser.add_argument(
        "--end", default=None, help="End date (YYYY-MM-DD), default: yesterday"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top gainers/losers per day, default: 10",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show pipeline status only, without running",
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        help="Run specific step only (1, 2, or 3)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of 5m downloads (for testing)",
    )
    parser.add_argument(
        "--directory",
        default="data/volatility",
        help="Output directory, default: data/volatility",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = VolatilityPipeline(
        base_directory=args.directory,
        start_date=args.start,
        end_date=args.end,
        top_count=args.top,
    )

    # Status mode only
    if args.status:
        pipeline.summary()
        return

    # Run specific step or all
    if args.step:
        print(f"\n🚀 Running STEP {args.step} only...\n")

        match args.step:
            case 1:
                pipeline.step_1_download_daily()
            case 2:
                pipeline.step_2_calculate_top_movers()
            case 3:
                pipeline.step_3_download_5m(limit=args.limit)
    else:
        print("\n🚀 Running FULL PIPELINE...\n")
        pipeline.run(limit_5m=args.limit)

    # Show summary
    print("\n")
    pipeline.summary()


if __name__ == "__main__":
    main()
