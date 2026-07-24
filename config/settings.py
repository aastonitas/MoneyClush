from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class TradingMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"


@dataclass(frozen=True)
class PolymarketConfig:
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    clob_url: str = "https://clob.polymarket.com"


@dataclass(frozen=True)
class WalletConfig:
    private_key: str = ""
    address: str = ""


@dataclass(frozen=True)
class RiskConfig:
    max_bankroll_usd: float = 1000.0
    kelly_fraction: float = 0.25
    max_loss_daily_usd: float = 50.0
    max_position_per_market: float = 200.0
    max_uncovered_shares: int = 500
    max_correlated_positions: int = 3
    kill_switch_latency_ms: float = 2000.0
    kill_switch_stale_data_sec: float = 10.0


@dataclass(frozen=True)
class ExecutionConfig:
    default_order_type: str = "GTC"
    max_slippage_pct: float = 0.02
    split_order_count: int = 3
    min_edge_to_execute: float = 0.02


@dataclass(frozen=True)
class Settings:
    mode: TradingMode = TradingMode.PAPER
    log_level: str = "INFO"
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    wallet: WalletConfig = field(default_factory=WalletConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    binance_ws_url: str = "wss://stream.binance.com:9443/ws"


def load_settings() -> Settings:
    return Settings(
        mode=TradingMode(os.getenv("TRADING_MODE", "paper")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        polymarket=PolymarketConfig(
            api_key=os.getenv("POLYMARKET_API_KEY", ""),
            api_secret=os.getenv("POLYMARKET_API_SECRET", ""),
            api_passphrase=os.getenv("POLYMARKET_API_PASSPHRASE", ""),
            clob_url=os.getenv(
                "POLYMARKET_CLOB_URL", "https://clob.polymarket.com"
            ),
        ),
        wallet=WalletConfig(
            private_key=os.getenv("PRIVATE_KEY", ""),
            address=os.getenv("WALLET_ADDRESS", ""),
        ),
        risk=RiskConfig(
            max_bankroll_usd=float(os.getenv("MAX_BANKROLL_USD", "1000")),
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
            max_loss_daily_usd=float(os.getenv("MAX_LOSS_DAILY_USD", "50")),
        ),
        binance_ws_url=os.getenv(
            "BINANCE_WS_URL", "wss://stream.binance.com:9443/ws"
        ),
    )
