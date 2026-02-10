"""
Database Models for Asymptotic Zero Trading Bot

Stores all trading activity, performance metrics, and system logs in PostgreSQL.

Tables:
- trading_sessions: Daily trading sessions
- trades: Individual trades executed
- daily_performance: Aggregated daily statistics
- system_logs: Bot errors, warnings, events
"""

from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
    Index,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
import os

Base = declarative_base()


class TradingSession(Base):
    """
    Represents a daily trading session.
    One row per trading day.
    """

    __tablename__ = "trading_sessions"

    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False, unique=True, index=True)

    # Configuration
    mode = Column(String(20), nullable=False)  # 'testnet' or 'live'
    initial_balance = Column(Float, nullable=False)

    # Screening results
    screened_coins = Column(
        JSONB
    )  # {'gainers': [...], 'losers': [...], 'gainers_pct': [...], 'losers_pct': [...]}

    # Session timing
    screening_time = Column(DateTime)
    trading_start_time = Column(DateTime)
    trading_end_time = Column(DateTime)

    # Performance
    final_balance = Column(Float)
    total_pnl = Column(Float)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)

    # Safety flags
    emergency_stop = Column(Boolean, default=False)
    daily_limit_hit = Column(Boolean, default=False)
    stop_reason = Column(String(200))

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    trades = relationship(
        "Trade", back_populates="session", cascade="all, delete-orphan"
    )
    daily_performance = relationship(
        "DailyPerformance",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self):
        return f"<TradingSession(date={self.date}, pnl={self.total_pnl}, trades={self.total_trades})>"


class Trade(Base):
    """
    Individual trade execution record.
    One row per open/close action.
    """

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    session_id = Column(
        Integer, ForeignKey("trading_sessions.id"), nullable=False, index=True
    )

    # Trade details
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    coin_rank = Column(Integer)  # 0-19 (gainer/loser rank)
    is_gainer = Column(Boolean)

    # Action
    action_type = Column(
        String(20), nullable=False
    )  # 'open', 'close', 'close_all', etc.
    action_id = Column(Integer, nullable=False)  # Raw action number from agent
    side = Column(String(10))  # 'LONG' or 'SHORT'

    # Execution
    quantity = Column(Float)
    entry_price = Column(Float)
    exit_price = Column(Float)

    # Performance
    pnl = Column(Float, default=0.0)
    pnl_percent = Column(Float)
    fees = Column(Float, default=0.0)
    slippage = Column(Float, default=0.0)

    # Agent decision
    raw_action = Column(Integer)  # Original agent action before guardrails
    filtered_action = Column(Integer)  # Action after guardrails
    action_blocked = Column(Boolean, default=False)
    block_reason = Column(String(100))

    # State at time of trade
    agent_q_value = Column(Float)  # Q-value for selected action
    portfolio_value = Column(Float)
    open_positions = Column(Integer)

    # Binance order details
    order_id = Column(String(100))
    order_status = Column(String(20))
    order_response = Column(JSONB)  # Full order response from Binance

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("TradingSession", back_populates="trades")

    # Indexes for common queries
    __table_args__ = (
        Index("idx_trade_timestamp", "timestamp"),
        Index("idx_trade_symbol", "symbol"),
        Index("idx_trade_pnl", "pnl"),
    )

    def __repr__(self):
        return (
            f"<Trade(symbol={self.symbol}, action={self.action_type}, pnl={self.pnl})>"
        )


class DailyPerformance(Base):
    """
    Aggregated daily performance metrics.
    Calculated at end of each trading day.
    """

    __tablename__ = "daily_performance"

    id = Column(Integer, primary_key=True)
    session_id = Column(
        Integer,
        ForeignKey("trading_sessions.id"),
        nullable=False,
        unique=True,
        index=True,
    )
    date = Column(DateTime, nullable=False, unique=True, index=True)

    # Performance metrics
    total_pnl = Column(Float, nullable=False)
    pnl_percent = Column(Float)
    win_rate = Column(Float)  # Percentage
    profit_factor = Column(Float)

    # Trading statistics
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    largest_win = Column(Float)
    largest_loss = Column(Float)

    # Risk metrics
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)  # Calculated over rolling window

    # Activity
    actions_taken = Column(Integer)
    actions_blocked = Column(Integer)
    guardrail_blocks = Column(JSONB)  # {'cooldown': 10, 'concentration': 5, ...}

    # Comparison to backtest
    expected_pnl = Column(Float)  # From backtest
    performance_ratio = Column(Float)  # Actual / Expected

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("TradingSession", back_populates="daily_performance")

    def __repr__(self):
        return f"<DailyPerformance(date={self.date}, pnl={self.total_pnl}, wr={self.win_rate}%)>"


class SystemLog(Base):
    """
    System events, errors, warnings.
    Stores important bot lifecycle events.
    """

    __tablename__ = "system_logs"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Log details
    level = Column(
        String(10), nullable=False, index=True
    )  # 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    category = Column(
        String(50), nullable=False, index=True
    )  # 'api', 'trading', 'agent', 'database', 'system'
    message = Column(Text, nullable=False)

    # Context
    session_id = Column(Integer, ForeignKey("trading_sessions.id"), index=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), index=True)

    # Additional data
    exception = Column(Text)  # Full exception traceback if error
    extra_data = Column(JSONB)  # Any additional context

    # Indexes
    __table_args__ = (
        Index("idx_log_level", "level"),
        Index("idx_log_timestamp", "timestamp"),
        Index("idx_log_category", "category"),
    )

    def __repr__(self):
        return f"<SystemLog({self.level}: {self.message[:50]})>"


class AgentState(Base):
    """
    Periodic snapshots of agent internal state.
    Useful for debugging and analysis.
    """

    __tablename__ = "agent_states"

    id = Column(Integer, primary_key=True)
    session_id = Column(
        Integer, ForeignKey("trading_sessions.id"), nullable=False, index=True
    )
    timestamp = Column(DateTime, nullable=False, index=True)

    # Agent internal state
    epsilon = Column(Float)  # Current exploration rate
    replay_buffer_size = Column(Integer)
    avg_q_value = Column(Float)
    max_q_value = Column(Float)

    # Model metrics
    recent_loss = Column(Float)  # Most recent training loss

    # Portfolio state
    portfolio_value = Column(Float)
    cash = Column(Float)
    open_positions = Column(Integer)
    position_details = Column(JSONB)  # Full position info

    # Market state snapshot
    current_coins = Column(JSONB)  # Current trading universe

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<AgentState(timestamp={self.timestamp}, portfolio={self.portfolio_value})>"


# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION & UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def get_database_url():
    """Get database URL from environment/Infisical."""
    # This will be populated by Infisical
    return os.getenv(
        "DATABASE_URL", "postgresql://user:password@localhost:5432/asymptotic_zero"
    )


def create_db_engine(database_url=None):
    """Create SQLAlchemy engine."""
    url = database_url or get_database_url()
    return create_engine(
        url,
        pool_pre_ping=True,  # Verify connections before using
        pool_size=10,
        max_overflow=20,
        echo=False,  # Set to True for SQL debugging
    )


def get_session_maker(engine=None):
    """Create session maker."""
    if engine is None:
        engine = create_db_engine()
    return sessionmaker(bind=engine)


def init_database(database_url=None):
    """
    Initialize database - create all tables.
    Run this once during deployment setup.
    """
    engine = create_db_engine(database_url)
    Base.metadata.create_all(engine)
    print("✅ Database tables created successfully")
    return engine


def drop_all_tables(database_url=None):
    """
    Drop all tables - USE WITH CAUTION!
    Only for development/testing.
    """
    engine = create_db_engine(database_url)
    Base.metadata.drop_all(engine)
    print("⚠️  All tables dropped")


if __name__ == "__main__":
    # Test database connection and create tables
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        print("Initializing database...")
        init_database()
    elif len(sys.argv) > 1 and sys.argv[1] == "drop":
        response = input("⚠️  Are you SURE you want to drop all tables? Type 'YES': ")
        if response == "YES":
            drop_all_tables()
        else:
            print("Aborted.")
    else:
        print("Usage:")
        print("  python models.py init  - Create all tables")
        print("  python models.py drop  - Drop all tables (DANGEROUS)")
