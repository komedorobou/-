#!/usr/bin/env python3
"""
データ結合＋基礎分析スクリプト

Binance / Hyperliquid から収集した全銘柄の日足データを統合し、
各種指標を計算して基礎テーブルを作成する。
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# --- 設定 ---
BINANCE_DIR = Path("data/binance")
HYPERLIQUID_DIR = Path("data/hyperliquid")
RESULTS_DIR = Path("results")
OUTPUT_CSV = RESULTS_DIR / "all_coins_base_data.csv"
SUMMARY_FILE = RESULTS_DIR / "summary.txt"

MULTIPLIER_LEVELS = [3, 5, 10, 20, 50, 100]


def load_all_data() -> dict:
    """全銘柄のデータを読み込む。{symbol: {"df": DataFrame, "source": str}}"""
    all_data = {}

    # Binance
    if BINANCE_DIR.exists():
        for csv_path in sorted(BINANCE_DIR.glob("*_daily.csv")):
            if csv_path.name.startswith("_"):
                continue
            symbol = csv_path.stem.replace("_daily", "")
            # BinanceのシンボルからUSDTを除去して統一名にする
            base = symbol.replace("USDT", "")
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df["open_time"] = pd.to_datetime(df["open_time"])
            df = df.sort_values("open_time").reset_index(drop=True)
            all_data[base] = {"df": df, "source": "binance", "raw_symbol": symbol}

    # Hyperliquid
    if HYPERLIQUID_DIR.exists():
        for csv_path in sorted(HYPERLIQUID_DIR.glob("*_daily.csv")):
            if csv_path.name.startswith("_"):
                continue
            symbol = csv_path.stem.replace("_daily", "")
            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            df["open_time"] = pd.to_datetime(df["open_time"])
            df = df.sort_values("open_time").reset_index(drop=True)

            if symbol in all_data:
                # 両方にある場合: データを統合（日付でマージ、Binance優先）
                existing = all_data[symbol]["df"]
                merged = pd.concat([existing, df]).drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
                all_data[symbol]["df"] = merged
                all_data[symbol]["source"] = "both"
            else:
                all_data[symbol] = {"df": df, "source": "hyperliquid", "raw_symbol": symbol}

    return all_data


def find_date_crossing_threshold(df: pd.DataFrame, col: str, threshold: float, direction: str = "below") -> str | None:
    """価格がthresholdを下回る（direction="below"）/上回る（direction="above"）最初の日付"""
    if direction == "below":
        mask = df[col] <= threshold
    else:
        mask = df[col] >= threshold
    hits = df[mask]
    if hits.empty:
        return None
    return hits.iloc[0]["open_time"].strftime("%Y-%m-%d")


def analyze_coin(symbol: str, df: pd.DataFrame, source: str) -> dict:
    """1銘柄の全指標を計算"""
    if df.empty or len(df) < 2:
        return None

    result = {"symbol": symbol, "source": source}

    # 基本情報
    result["listing_date"] = df.iloc[0]["open_time"].strftime("%Y-%m-%d")
    result["listing_close"] = df.iloc[0]["close"]
    result["current_price"] = df.iloc[-1]["close"]
    result["data_days"] = len(df)

    listing_close = df.iloc[0]["close"]
    if listing_close <= 0:
        return None

    # ATH
    ath_idx = df["high"].idxmax()
    result["ath"] = df.loc[ath_idx, "high"]
    result["ath_date"] = df.loc[ath_idx, "open_time"].strftime("%Y-%m-%d")
    result["ath_multiple"] = result["ath"] / listing_close

    # ATHからの下落率
    result["ath_drawdown_pct"] = (1 - result["current_price"] / result["ath"]) * 100

    # ATH後の最安値
    after_ath = df.loc[ath_idx:]
    if len(after_ath) > 1:
        post_ath_low_idx = after_ath["low"].idxmin()
        result["post_ath_low"] = after_ath.loc[post_ath_low_idx, "low"]
        result["post_ath_low_date"] = after_ath.loc[post_ath_low_idx, "open_time"].strftime("%Y-%m-%d")
        result["post_ath_low_drawdown_pct"] = (1 - result["post_ath_low"] / result["ath"]) * 100
    else:
        result["post_ath_low"] = result["current_price"]
        result["post_ath_low_date"] = df.iloc[-1]["open_time"].strftime("%Y-%m-%d")
        result["post_ath_low_drawdown_pct"] = result["ath_drawdown_pct"]

    # ATH -50%
    ath_50 = result["ath"] * 0.5
    after_ath_df = df.loc[ath_idx:].copy()
    ath_50_date = find_date_crossing_threshold(after_ath_df, "low", ath_50, "below")
    result["ath_minus_50_date"] = ath_50_date

    # ATH -50%到達後の分析
    result["ath_50_recovery"] = False
    result["ath_50_recovery_max"] = None
    result["ath_minus_80_date"] = None
    result["ath_50_to_80_days"] = None

    if ath_50_date:
        ath_50_dt = pd.to_datetime(ath_50_date)
        after_50 = df[df["open_time"] >= ath_50_dt].copy()

        if not after_50.empty:
            # 復帰フラグ: -50%ライン(= ATH*0.5)を上回ったことがあるか
            recovered = after_50[after_50["high"] > ath_50]
            result["ath_50_recovery"] = len(recovered) > 1  # 最初の日を除く

            # 復帰後の最高値
            result["ath_50_recovery_max"] = after_50["high"].max()

            # ATH -80%到達
            ath_80 = result["ath"] * 0.2
            ath_80_date = find_date_crossing_threshold(after_50, "low", ath_80, "below")
            result["ath_minus_80_date"] = ath_80_date

            if ath_80_date:
                days_50_to_80 = (pd.to_datetime(ath_80_date) - ath_50_dt).days
                result["ath_50_to_80_days"] = days_50_to_80

    # 各倍率到達
    for mult in MULTIPLIER_LEVELS:
        target = listing_close * mult
        reached_date = find_date_crossing_threshold(df, "high", target, "above")
        result[f"x{mult}_reached"] = reached_date is not None
        result[f"x{mult}_date"] = reached_date

        # 各倍率到達後に50%下落した日付
        result[f"x{mult}_then_50drop_date"] = None
        result[f"x{mult}_then_50drop_recovery"] = False

        if reached_date:
            reached_dt = pd.to_datetime(reached_date)
            after_reach = df[df["open_time"] >= reached_dt].copy()
            if not after_reach.empty:
                peak_after = after_reach["high"].max()
                drop_target = peak_after * 0.5
                drop_date = find_date_crossing_threshold(after_reach, "low", drop_target, "below")
                result[f"x{mult}_then_50drop_date"] = drop_date

                if drop_date:
                    drop_dt = pd.to_datetime(drop_date)
                    after_drop = df[df["open_time"] >= drop_dt].copy()
                    if not after_drop.empty:
                        result[f"x{mult}_then_50drop_recovery"] = after_drop["high"].max() > drop_target

    # 出来高
    result["avg_volume"] = df["volume"].mean()
    ath_range_low = result["ath"] * 0.9
    ath_range_high = result["ath"] * 1.1
    ath_zone = df[(df["high"] >= ath_range_low) & (df["low"] <= ath_range_high)]
    result["ath_zone_avg_volume"] = ath_zone["volume"].mean() if not ath_zone.empty else 0

    return result


def compute_short_strategy_stats(base_df: pd.DataFrame, all_data: dict) -> dict:
    """
    X倍以上達成 × ATH -50%でショート → -80%で利確 の勝率を計算
    ストップロス: +30%（ショートエントリーから30%上昇で損切り）
    """
    results = {}

    for mult in [3, 5, 10, 20, 50]:
        col_reached = f"x{mult}_reached"
        if col_reached not in base_df.columns:
            continue

        candidates = base_df[base_df[col_reached] == True].copy()
        wins = 0
        losses = 0
        profit_rates = []

        for _, row in candidates.iterrows():
            symbol = row["symbol"]
            if symbol not in all_data:
                continue
            df = all_data[symbol]["df"]
            ath = row["ath"]

            if pd.isna(ath) or ath <= 0:
                continue

            # ATH -50%ラインでショートエントリー
            entry_price = ath * 0.5
            target_price = ath * 0.2  # -80% = ATHの20%
            stop_loss_price = entry_price * 1.3  # +30%で損切り

            # ATH -50%に到達した日を探す
            ath_date = pd.to_datetime(row["ath_date"])
            after_ath = df[df["open_time"] > ath_date].copy()

            entry_found = False
            for idx, day in after_ath.iterrows():
                if day["low"] <= entry_price:
                    entry_found = True
                    # エントリー後のデータ
                    after_entry = after_ath.loc[idx:]
                    trade_done = False

                    for _, trade_day in after_entry.iterrows():
                        # ストップロス判定（高値がストップロスを超えた）
                        if trade_day["high"] >= stop_loss_price:
                            losses += 1
                            # 損失率: -(stop_loss_price / entry_price - 1) * 100
                            profit_rates.append(-(stop_loss_price / entry_price - 1) * 100)
                            trade_done = True
                            break
                        # 利確判定（安値がターゲットに到達）
                        if trade_day["low"] <= target_price:
                            wins += 1
                            profit_rates.append((entry_price - target_price) / entry_price * 100)
                            trade_done = True
                            break

                    if not trade_done:
                        # まだ決着がついていない（現在進行中）
                        current = after_entry.iloc[-1]["close"]
                        unrealized = (entry_price - current) / entry_price * 100
                        profit_rates.append(unrealized)
                        if current < entry_price:
                            wins += 1
                        else:
                            losses += 1

                    break  # 1銘柄1トレード

            if not entry_found:
                continue

        total = wins + losses
        win_rate = (wins / total * 100) if total > 0 else 0
        avg_profit = np.mean(profit_rates) if profit_rates else 0
        stop_loss_count = losses

        results[mult] = {
            "multiplier": f"x{mult}",
            "candidates": len(candidates),
            "trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "avg_profit_pct": avg_profit,
            "stop_loss_hits": stop_loss_count,
        }

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("データ結合＋基礎分析")
    print("=" * 70)

    # データ読み込み
    print("\n[Step 1] データ読み込み中...")
    all_data = load_all_data()
    print(f"  読み込み銘柄数: {len(all_data)}")

    if not all_data:
        print("データが見つかりません。先にcollect_binance.pyとcollect_hyperliquid.pyを実行してください。")
        sys.exit(1)

    # ソース別カウント
    binance_count = sum(1 for v in all_data.values() if v["source"] in ("binance", "both"))
    hl_count = sum(1 for v in all_data.values() if v["source"] in ("hyperliquid", "both"))
    both_count = sum(1 for v in all_data.values() if v["source"] == "both")
    print(f"  Binance: {binance_count}, Hyperliquid: {hl_count}, 重複: {both_count}")

    # 各銘柄の分析
    print("\n[Step 2] 各銘柄の基礎指標計算中...")
    results = []
    for symbol, info in all_data.items():
        r = analyze_coin(symbol, info["df"], info["source"])
        if r:
            results.append(r)

    if not results:
        print("分析可能な銘柄がありません。")
        sys.exit(1)

    base_df = pd.DataFrame(results)

    # CSV保存
    base_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n基礎テーブル保存: {OUTPUT_CSV} ({len(base_df)} 銘柄)")

    # ショート戦略分析
    print("\n[Step 3] ショート戦略分析中...")
    strategy_stats = compute_short_strategy_stats(base_df, all_data)

    # サマリー統計
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append(f"分析サマリー  (生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    summary_lines.append("=" * 70)
    summary_lines.append("")
    summary_lines.append(f"総銘柄数: {len(base_df)}")
    summary_lines.append(f"  Binance: {binance_count}")
    summary_lines.append(f"  Hyperliquid: {hl_count}")
    summary_lines.append(f"  重複(両方): {both_count}")
    summary_lines.append("")

    # 倍率達成数
    for mult in MULTIPLIER_LEVELS:
        col = f"x{mult}_reached"
        if col in base_df.columns:
            count = base_df[col].sum()
            summary_lines.append(f"ATH {mult}倍以上達成: {count} 銘柄")

    summary_lines.append("")

    # ATH -50%復帰
    recovered = base_df[base_df["ath_50_recovery"] == True]
    not_recovered = base_df[(base_df["ath_minus_50_date"].notna()) & (base_df["ath_50_recovery"] == False)]
    summary_lines.append(f"ATH -50%から復帰した銘柄: {len(recovered)} 銘柄")
    if not recovered.empty:
        summary_lines.append(f"  銘柄名: {', '.join(recovered['symbol'].tolist())}")
    summary_lines.append(f"ATH -50%から復帰しなかった銘柄: {len(not_recovered)} 銘柄")
    summary_lines.append("")

    # ショート戦略結果
    summary_lines.append("-" * 70)
    summary_lines.append("ショート戦略: X倍以上達成 × ATH -50%でショート → -80%で利確")
    summary_lines.append("ストップロス: +30%")
    summary_lines.append("-" * 70)

    for mult, stats in strategy_stats.items():
        summary_lines.append(f"\n  [{stats['multiplier']}以上フィルタ]")
        summary_lines.append(f"    対象銘柄数: {stats['candidates']}")
        summary_lines.append(f"    トレード数: {stats['trades']}")
        summary_lines.append(f"    勝ち: {stats['wins']}  負け: {stats['losses']}")
        summary_lines.append(f"    勝率: {stats['win_rate']:.1f}%")
        summary_lines.append(f"    ストップロス発動回数: {stats['stop_loss_hits']}")
        summary_lines.append(f"    平均利益率: {stats['avg_profit_pct']:.1f}%")

    # 最適なX
    if strategy_stats:
        # 期待値 = 勝率 × 平均利益 で比較
        best_mult = max(strategy_stats.keys(), key=lambda m: strategy_stats[m]["win_rate"] * strategy_stats[m]["avg_profit_pct"] if strategy_stats[m]["trades"] > 0 else 0)
        best = strategy_stats[best_mult]
        expected_value = best["win_rate"] / 100 * best["avg_profit_pct"]
        summary_lines.append(f"\n  >>> 最適フィルタ: x{best_mult} (期待値: {expected_value:.1f}%)")

    summary_text = "\n".join(summary_lines)

    # ファイル保存
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"サマリー保存: {SUMMARY_FILE}")

    # ターミナル出力
    print("\n" + summary_text)

    # 詳細テーブル出力
    print("\n\n" + "=" * 70)
    print("ショート戦略 詳細比較テーブル")
    print("=" * 70)
    if strategy_stats:
        header = f"{'フィルタ':>10} | {'対象':>6} | {'取引':>6} | {'勝ち':>6} | {'負け':>6} | {'勝率':>8} | {'SL発動':>6} | {'平均利益':>10} | {'期待値':>10}"
        print(header)
        print("-" * len(header))
        for mult in [3, 5, 10, 20, 50]:
            if mult in strategy_stats:
                s = strategy_stats[mult]
                ev = s["win_rate"] / 100 * s["avg_profit_pct"]
                print(f"{'x' + str(mult):>10} | {s['candidates']:>6} | {s['trades']:>6} | {s['wins']:>6} | {s['losses']:>6} | {s['win_rate']:>7.1f}% | {s['stop_loss_hits']:>6} | {s['avg_profit_pct']:>9.1f}% | {ev:>9.1f}%")

    print("\n全データは results/all_coins_base_data.csv に保存済み。")
    print("条件を変えて再分析する場合はCSVを直接読み込んでください。")


if __name__ == "__main__":
    main()
