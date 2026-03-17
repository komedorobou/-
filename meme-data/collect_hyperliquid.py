#!/usr/bin/env python3
"""
Hyperliquid 全パーペチュアル銘柄 日足データ収集スクリプト

Hyperliquidに上場している全銘柄の日足データを
2023年1月から現在まで取得し、銘柄ごとにCSV保存する。
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- 設定 ---
BASE_URL = os.getenv("HYPERLIQUID_BASE_URL", "https://api.hyperliquid.xyz")
OUTPUT_DIR = Path("data/hyperliquid")
METADATA_FILE = OUTPUT_DIR / "_metadata.csv"
LOG_FILE = OUTPUT_DIR / "_errors.log"
SLEEP_BETWEEN_REQUESTS = 0.5  # 秒
MAX_RETRIES = 3
# 2023年1月1日からデータ取得開始
START_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def get_meta() -> list[dict]:
    """Hyperliquidの全銘柄メタデータを取得"""
    url = f"{BASE_URL}/info"
    resp = requests.post(url, json={"type": "meta"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    symbols = []
    for coin in data.get("universe", []):
        symbols.append({
            "symbol": coin["name"],
            "szDecimals": coin.get("szDecimals", 0),
            "maxLeverage": coin.get("maxLeverage", 0),
        })

    logger.info(f"Hyperliquid 銘柄数: {len(symbols)}")
    return symbols


def fetch_candles(coin: str, start_time_ms: int, end_time_ms: int) -> list:
    """指定銘柄の日足キャンドルデータを取得（1回分）"""
    url = f"{BASE_URL}/info"
    body = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": "1d",
            "startTime": start_time_ms,
            "endTime": end_time_ms,
        },
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=body, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"{coin} リトライ {attempt+1}/{MAX_RETRIES} ({wait}s待機): {e}")
                time.sleep(wait)
            else:
                raise

    return []


def collect_all_candles(coin: str) -> pd.DataFrame:
    """指定銘柄の全期間日足データを取得"""
    all_rows = []
    start_ms = int(START_DATE.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    # 約500日ずつ分割取得
    chunk_days = 500
    chunk_ms = chunk_days * 86_400_000

    current_start = start_ms
    while current_start < now_ms:
        current_end = min(current_start + chunk_ms, now_ms)

        try:
            candles = fetch_candles(coin, current_start, current_end)
        except Exception as e:
            logger.warning(f"{coin}: チャンク取得失敗 ({current_start}～) - {e}")
            current_start = current_end
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            continue

        if candles:
            for c in candles:
                # Hyperliquid returns: {"t": timestamp, "o": open, "h": high, "l": low, "c": close, "v": volume}
                # or list format depending on version
                if isinstance(c, dict):
                    ts = c.get("t", c.get("T", 0))
                    all_rows.append({
                        "open_time": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                        "open": float(c.get("o", c.get("O", 0))),
                        "high": float(c.get("h", c.get("H", 0))),
                        "low": float(c.get("l", c.get("L", 0))),
                        "close": float(c.get("c", c.get("C", 0))),
                        "volume": float(c.get("v", c.get("V", 0))),
                        "quote_volume": 0.0,  # Hyperliquidでは提供されない場合あり
                    })
                elif isinstance(c, list) and len(c) >= 6:
                    all_rows.append({
                        "open_time": datetime.fromtimestamp(c[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                        "quote_volume": 0.0,
                    })

        current_start = current_end
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def already_collected(symbol: str) -> bool:
    """既に取得済みかチェック"""
    csv_path = OUTPUT_DIR / f"{symbol}_daily.csv"
    return csv_path.exists() and csv_path.stat().st_size > 100


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Hyperliquid 全銘柄日足データ収集")
    print("=" * 60)

    # Step 1: 銘柄リスト取得
    print("\n[Step 1] 銘柄リスト取得中...")
    symbols = get_meta()
    print(f"  銘柄数: {len(symbols)}")

    # 既存スキップ数カウント
    skip_count = sum(1 for s in symbols if already_collected(s["symbol"]))
    to_collect = len(symbols) - skip_count
    print(f"  取得済みスキップ: {skip_count} 銘柄")
    print(f"  今回取得対象: {to_collect} 銘柄")

    # 推定所要時間 (約500日チャンク × 0.5秒 × チャンク数)
    # 2023/1～現在で約3年 = ~1100日 → 約3チャンク/銘柄 × 0.5秒 ≈ 2秒/銘柄
    est_seconds = to_collect * 3
    est_minutes = est_seconds / 60
    print(f"\n  推定所要時間: 約 {est_minutes:.0f} 分 ({est_seconds} 秒)")
    print()

    # Step 2: 各銘柄のデータ取得
    metadata_rows = []
    errors = []

    for sym_info in tqdm(symbols, desc="銘柄データ取得"):
        symbol = sym_info["symbol"]

        if already_collected(symbol):
            existing_df = pd.read_csv(OUTPUT_DIR / f"{symbol}_daily.csv")
            metadata_rows.append({
                "symbol": symbol,
                "szDecimals": sym_info["szDecimals"],
                "maxLeverage": sym_info["maxLeverage"],
                "rows": len(existing_df),
            })
            continue

        try:
            df = collect_all_candles(symbol)

            if df.empty:
                logger.warning(f"{symbol}: データなし")
                metadata_rows.append({
                    "symbol": symbol,
                    "szDecimals": sym_info["szDecimals"],
                    "maxLeverage": sym_info["maxLeverage"],
                    "rows": 0,
                })
                continue

            # CSV保存
            csv_path = OUTPUT_DIR / f"{symbol}_daily.csv"
            df.to_csv(csv_path, index=False)

            metadata_rows.append({
                "symbol": symbol,
                "szDecimals": sym_info["szDecimals"],
                "maxLeverage": sym_info["maxLeverage"],
                "rows": len(df),
            })

        except Exception as e:
            logger.error(f"{symbol}: 取得失敗 - {e}")
            errors.append({"symbol": symbol, "error": str(e)})

    # Step 3: メタデータ保存
    if metadata_rows:
        meta_df = pd.DataFrame(metadata_rows)
        meta_df.to_csv(METADATA_FILE, index=False)
        print(f"\nメタデータ保存: {METADATA_FILE}")

    # サマリー
    print("\n" + "=" * 60)
    print("収集完了サマリー")
    print("=" * 60)
    print(f"  成功: {len(metadata_rows)} 銘柄")
    print(f"  失敗: {len(errors)} 銘柄")
    if errors:
        print("  失敗銘柄:")
        for e in errors:
            print(f"    - {e['symbol']}: {e['error']}")
    total_rows = sum(m["rows"] for m in metadata_rows)
    print(f"  総データ行数: {total_rows:,}")
    print(f"  保存先: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
