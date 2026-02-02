from __future__ import annotations

import os
from typing import Optional

import pandas as pd


class CTraderClient:
    """
    Placeholder adapter for the official cTrader Open API.

    The real cTrader Open API uses protobuf over TCP. This adapter is designed to
    be extended with proper credentials and networking logic while keeping the
    rest of the pipeline stable. If the required credentials are not provided,
    it returns None so the pipeline can fall back to Dukascopy data.
    """

    def __init__(self) -> None:
        self.client_id = os.getenv("CTRADER_CLIENT_ID")
        self.client_secret = os.getenv("CTRADER_CLIENT_SECRET")
        self.access_token = os.getenv("CTRADER_ACCESS_TOKEN")

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[pd.Timestamp],
    ) -> Optional[pd.DataFrame]:
        if not self.client_id or not self.client_secret or not self.access_token:
            return None
        raise NotImplementedError(
            "Provide a full cTrader Open API implementation with protobuf transport "
            "to enable primary-source downloads."
        )
