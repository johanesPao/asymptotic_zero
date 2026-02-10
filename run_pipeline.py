"""
Runner Script for Volatility Pipeline

Usage:
    # Run full pipeline
    python run_pipeline.py

    # Run specific step only
    python run_pipeline.py --step 1 # Only download daily
    python run_pipeline.py --step 2 # Only calculate top movers
    python run_pipeline.py --step 3 # Only download 5m

    # Check status
    python run_pipeline.py --status

    # Test with limit
    python run_pipeline.py --limit 10 # Only 10 5m files for testing
"""

import sys
from pathlib import Path
from src.data_pipeline.volatility import VolatilityPipeline

# Add src to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Volatility Pipeline - Data Collection for DQN Trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py              # Run all steps
    python run_pipeline.py --status     # Check status only
    python run_pipeline.py --step 1     # Only step 1 (download daily)
    python run_pipeline.py --step 2     # Only step 2 (calculate top movers)
    python run_pipeline.py --step 3     # Only step 3 (download 5m)
    python run_pipeline.py --limit 10   # Test with 10 5m files only
        """,
    )

    parser.add_argument(
        "--start",
        default="2020-01-01",
        help="Start date (YYYY-MM-DD), default: 2020-01-01",
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

        if args.step == 1:
            pipeline.step_1_download_daily()
        elif args.step == 2:
            pipeline.step_2_calculate_top_movers()
        elif args.step == 3:
            pipeline.step_3_download_5m(limit=args.limit)
    else:
        print("\n🚀 Running FULL PIPELINE...\n")
        pipeline.run(limit_5m=args.limit)

    # Show summary
    print("\n")
    pipeline.summary()


if __name__ == "__main__":
    main()
