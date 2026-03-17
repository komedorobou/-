#!/usr/bin/env python3
"""
Binance USDT-M Futures - 全銘柄日足データ収集スクリプト

全てのUSDT-Mペアの日足klineデータを上場日から現在まで取得し、
銘柄ごとにCSV保存する。
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- 設定 ---
BASE_URL = "https://fapi.binance.com"
OUTPUT_DIR = Path("data/binance")
METADATA_FILE = OUTPUT_DIR / "_metadata.csv"
LOG_FILE = OUTPUT_DIR / "_errors.log"
SLEEP_BETWEEN_REQUESTS = 0.3  # 秒
MAX_RETRIES = 3
KLINE_LIMIT = 1500  # 最大1500、ダメなら1000にフォールバック

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


def get_exchange_info() -> list[dict]:
    """Binance USDT-M先物の全銘柄情報を取得"""
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    symbols = []
    for s in data["symbols"]:
        if s["quoteAsset"] == "USDT" and s["contractType"] == "PERPETUAL":
            symbols.append({
                "symbol": s["symbol"],
                "baseAsset": s["baseAsset"],
                "status": s["status"],
                "onboardDate": s.get("onboardDate", 0),
            })

    logger.info(f"取得銘柄数: {len(symbols)} (USDT-M Perpetual)")
    return symbols


def fetch_klines(symbol: str, start_time: int, limit: int = KLINE_LIMIT) -> list:
    """指定銘柄の日足klineデータを取得（1回分）"""
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": start_time,
        "limit": limit,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 400 and limit > 1000:
                # limit=1500が使えない場合、1000にフォールバック
                return fetch_klines(symbol, start_time, limit=1000)
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"{symbol} リトライ {attempt+1}/{MAX_RETRIES} ({wait}s待機): {e}")
                time.sleep(wait)
            else:
                raise
        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"{symbol} リトライ {attempt+1}/{MAX_RETRIES} ({wait}s待機): {e}")
                time.sleep(wait)
            else:
                raise

    return []


def collect_all_klines(symbol: str, onboard_date: int) -> pd.DataFrame:
    """指定銘柄の全期間日足klineデータを取得"""
    all_rows = []
    start_time = onboard_date if onboard_date > 0 else 1_500_000_000_000  # 2017年頃

    while True:
        data = fetch_klines(symbol, start_time)
        if not data:
            break

        for k in data:
            all_rows.append({
                "open_time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc).strftime("%Y-%m-%d"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "quote_volume": float(k[7]),
            })

        # 次のページへ
        last_open_time = data[-1][0]
        start_time = last_open_time + 86_400_000  # +1日

        # 未来のデータはないのでbreak
        if start_time > int(datetime.now(timezone.utc).timestamp() * 1000):
            break

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
    print("Binance USDT-M Futures 全銘柄日足データ収集")
    print("=" * 60)

    # Step 1: 銘柄リスト取得
    print("\n[Step 1] 銘柄リスト取得中...")
    symbols = get_exchange_info()
    trading = [s for s in symbols if s["status"] == "TRADING"]
    print(f"  TRADING: {len(trading)} 銘柄")
    print(f"  全体: {len(symbols)} 銘柄")

    # 既存スキップ数カウント
    skip_count = sum(1 for s in symbols if already_collected(s["symbol"]))
    to_collect = len(symbols) - skip_count
    print(f"  取得済みスキップ: {skip_count} 銘柄")
    print(f"  今回取得対象: {to_collect} 銘柄")

    # 推定所要時間
    # 平均して1銘柄あたり3リクエスト × 0.3秒 + 処理時間 ≈ 2秒/銘柄
    est_seconds = to_collect * 2
    est_minutes = est_seconds / 60
    print(f"\n  推定所要時間: 約 {est_minutes:.0f} 分 ({est_seconds} 秒)")
    print()

    # Step 2: 各銘柄のデータ取得
    metadata_rows = []
    errors = []

    for sym_info in tqdm(symbols, desc="銘柄データ取得"):
        symbol = sym_info["symbol"]
        onboard_date = sym_info["onboardDate"]

        if already_collected(symbol):
            # メタデータは既存CSVから読み込む
            existing_df = pd.read_csv(OUTPUT_DIR / f"{symbol}_daily.csv")
            metadata_rows.append({
                "symbol": symbol,
                "baseAsset": sym_info["baseAsset"],
                "status": sym_info["status"],
                "onboardDate": datetime.fromtimestamp(onboard_date / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if onboard_date > 0 else "unknown",
                "rows": len(existing_df),
            })
            continue

        try:
            df = collect_all_klines(symbol, onboard_date)

            if df.empty:
                logger.warning(f"{symbol}: データなし")
                metadata_rows.append({
                    "symbol": symbol,
                    "baseAsset": sym_info["baseAsset"],
                    "status": sym_info["status"],
                    "onboardDate": datetime.fromtimestamp(onboard_date / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if onboard_date > 0 else "unknown",
                    "rows": 0,
                })
                continue

            # CSV保存
            csv_path = OUTPUT_DIR / f"{symbol}_daily.csv"
            df.to_csv(csv_path, index=False)

            metadata_rows.append({
                "symbol": symbol,
                "baseAsset": sym_info["baseAsset"],
                "status": sym_info["status"],
                "onboardDate": datetime.fromtimestamp(onboard_date / 1000, tz=timezone.utc).strftime("%Y-%m-%d") if onboard_date > 0 else "unknown",
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
