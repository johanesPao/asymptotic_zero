import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from src.database.models import Base
from src.config.secrets import get_secret

try:
    print("Getting DATABASE_URL from secret host...")
    db_url = get_secret("DATABASE_URL")

    if not db_url:
        print("❌ DATABASE_URL not found in secret host!")
        exit(1)

    print("✅ Got DATABASE_URL")

    print("\nCreating database tables...")
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    print("✅ Tables created successfully!")
    print("\nCreated tables:")
    print("  - trading_sessions")
    print("  - trades")
    print("  - daily_performance")
    print("  - system_logs")
    print("  - agent_states")

    print("\n🎉 Database is ready!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
