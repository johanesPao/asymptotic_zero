"""
Database Models for Asymptotic Zero Trading Bot

Single source of truth for all trading state. The JSON metrics file is eliminated;
the dashboard reads from these tables + a live Binance poller.

Tables:
- sessions:       One row per screening cycle (typically daily)
- trades:         Individual trade open/close records
- activity_log:   Rolling human-readable activity log (survives restart)
- system_logs:    WARNING/ERROR/CRITICAL log entries
- pnl_snapshots:  Periodic equity snapshots for PnL chart
- bot_heartbeat:  Single-row UPSERT — "what is the bot doing now?"
"""

from datetime import datetime, timezone
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    SmallInteger,
    Float,
    String,
    DateTime,
    Date,
    Boolean,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class Session(Base):
    """One row per screening cycle (typically daily)."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    date = Column(Date)
    mode = Column(String(20), nullable=False)
    initial_balance = Column(Float, nullable=False)
    final_balance = Column(Float)
    total_pnl = Column(Float)
    screened_coins = Column(JSONB)
    screening_time = Column(DateTime(timezone=True))
    trading_start = Column(DateTime(timezone=True))
    trading_end = Column(DateTime(timezone=True))
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    daily_limit_hit = Column(Boolean, default=False)
    stop_reason = Column(String(200))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    trades = relationship("Trade", back_populates="session", cascade="all, delete-orphan")
    activity_entries = relationship("ActivityLog", back_populates="session", cascade="all, delete-orphan")
    pnl_snapshots = relationship("PnLSnapshot", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Session(date={self.date}, pnl={self.total_pnl}, trades={self.total_trades})>"


class Trade(Base):
    """Individual trade open/close record."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    coin_rank = Column(SmallInteger)
    is_gainer = Column(Boolean)
    action_type = Column(String(10), nullable=False)  # 'open' or 'close'
    action_id = Column(SmallInteger)
    side = Column(String(5))  # 'LONG' or 'SHORT'
    quantity = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float)
    agent_q_value = Column(Float)
    portfolio_value = Column(Float)
    open_positions = Column(SmallInteger)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    # Relationships
    session = relationship("Session", back_populates="trades")

    __table_args__ = (
        Index("idx_trade_timestamp", "timestamp"),
        Index("idx_trade_symbol", "symbol"),
    )

    def __repr__(self):
        return f"<Trade(symbol={self.symbol}, action={self.action_type}, pnl={self.pnl})>"


class ActivityLog(Base):
    """Rolling human-readable activity log. Survives restart."""

    __tablename__ = "activity_log"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    message = Column(Text, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="activity_entries")

    def __repr__(self):
        return f"<ActivityLog({self.message[:50]})>"


class SystemLog(Base):
    """WARNING/ERROR/CRITICAL log entries from all Python loggers."""

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="SET NULL"), index=True)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    level = Column(String(10), nullable=False, index=True)
    source = Column(String(50), nullable=False)
    message = Column(Text, nullable=False)
    exception = Column(Text)
    extra_data = Column(JSONB)

    __table_args__ = (
        Index("idx_log_level", "level"),
        Index("idx_log_timestamp", "timestamp"),
    )

    def __repr__(self):
        return f"<SystemLog({self.level}: {self.message[:50]})>"


class PnLSnapshot(Base):
    """Periodic equity snapshots for PnL chart seeding."""

    __tablename__ = "pnl_snapshots"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    equity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=False)

    # Relationships
    session = relationship("Session", back_populates="pnl_snapshots")

    def __repr__(self):
        return f"<PnLSnapshot(equity={self.equity}, pnl={self.pnl})>"


class BotHeartbeat(Base):
    """
    Single-row UPSERT — the dashboard's sole "what is the bot doing now?" query.

    Dashboard query: SELECT * FROM bot_heartbeat WHERE bot_id = 'main'
    """

    __tablename__ = "bot_heartbeat"

    bot_id = Column(String(20), primary_key=True, default="main")
    session_id = Column(Integer, ForeignKey("sessions.id", ondelete="SET NULL"))
    status = Column(String(20))  # 'running', 'sleeping', 'offline'
    session_started = Column(DateTime(timezone=True))
    last_updated = Column(DateTime(timezone=True))
    agent_status = Column(String(20))
    epsilon = Column(Float)
    current_step = Column(Integer)
    last_action = Column(String(100))
    avg_q = Column(Float)
    current_balance = Column(Float)
    initial_balance = Column(Float)
    trade_count = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    guardrail_status = Column(JSONB)
    coin_table = Column(JSONB)
    screened_coins = Column(JSONB)
    trading_mode = Column(String(20))

    def __repr__(self):
        return f"<BotHeartbeat(status={self.status}, step={self.current_step})>"


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def get_database_url():
    """Get database URL from Infisical (falls back to env var)."""
    from src.config.secrets import get_secret
    return get_secret("DATABASE_URL")


def create_db_engine(database_url=None):
    """Create SQLAlchemy engine."""
    url = database_url or get_database_url()
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )


def get_session_maker(engine=None):
    """Create session maker."""
    from sqlalchemy.orm import sessionmaker
    if engine is None:
        engine = create_db_engine()
    return sessionmaker(bind=engine)


def init_database(database_url=None):
    """Initialize database - create all tables."""
    engine = create_db_engine(database_url)
    Base.metadata.create_all(engine)
    print("Database tables created successfully")
    return engine


def drop_all_tables(database_url=None):
    """Drop all tables - USE WITH CAUTION!"""
    engine = create_db_engine(database_url)
    Base.metadata.drop_all(engine)
    print("All tables dropped")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        print("Initializing database...")
        init_database()
    elif len(sys.argv) > 1 and sys.argv[1] == "drop":
        response = input("Are you SURE you want to drop all tables? Type 'YES': ")
        if response == "YES":
            drop_all_tables()
        else:
            print("Aborted.")
    else:
        print("Usage:")
        print("  python models.py init  - Create all tables")
        print("  python models.py drop  - Drop all tables (DANGEROUS)")
