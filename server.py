#!/usr/bin/env python3
"""
Flask server for stock data retrieval using yfinance.
Step 2: OHLCV data API
Step 3: k-medoids clustering API
Step 4: Walk-forward backtest for optimal k*

Usage:
    python server.py

Endpoints:
    GET /api/health
    GET /api/stock?ticker=AAPL&period=5y
    GET /api/tickers
    GET /api/cluster?ticker=AAPL&period=5y&k=5&window=30&stride=5&future=10
    GET /api/backtest?ticker=AAPL&period=5y&k_max=20&window=30&stride=5&future=10&split=0.7
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import yfinance as yf
import numpy as np

app = Flask(__name__)
CORS(app)

VALID_PERIODS   = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"}
VALID_INTERVALS = {"1d", "1wk", "1mo"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _z_norm(arr):
    arr = np.asarray(arr, dtype=np.float64)
    std = arr.std()
    if std < 1e-10:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _dtw(s, t, w):
    """DTW distance with Sakoe-Chiba band w, path-length normalised."""
    n = len(s)
    prev = np.full(n + 1, np.inf);  prev[0] = 0.0
    curr = np.full(n + 1, np.inf)
    for i in range(1, n + 1):
        curr[:] = np.inf
        for j in range(max(1, i - w), min(n, i + w) + 1):
            cost = (s[i - 1] - t[j - 1]) ** 2
            curr[j] = cost + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev
    return float(np.sqrt(prev[n] / n))


def _k_medoids(D, k, seed=42):
    """PAM k-medoids on precomputed distance matrix D."""
    n = D.shape[0]
    rng = np.random.RandomState(seed)

    # Initialise: most-central point, then spread (k-means++ style)
    med = [int(D.sum(axis=1).argmin())]
    for _ in range(k - 1):
        min_d = D[:, med].min(axis=1)
        probs = min_d / (min_d.sum() + 1e-12)
        med.append(int(rng.choice(n, p=probs)))

    for _ in range(50):
        asgn = D[:, med].argmin(axis=1)
        new_med = list(med)
        for c in range(k):
            pts = np.where(asgn == c)[0]
            if len(pts):
                sub = D[np.ix_(pts, pts)]
                new_med[c] = int(pts[sub.sum(axis=1).argmin()])
        if new_med == med:
            break
        med = new_med

    asgn = D[:, med].argmin(axis=1)
    return med, asgn.tolist()


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "version": "2.0.0"})


@app.route("/api/stock")
def get_stock():
    ticker   = request.args.get("ticker",   "AAPL").upper().strip()
    period   = request.args.get("period",   "5y")
    interval = request.args.get("interval", "1d")

    if period   not in VALID_PERIODS:
        return jsonify({"error": f"Invalid period '{period}'."}), 400
    if interval not in VALID_INTERVALS:
        return jsonify({"error": f"Invalid interval '{interval}'."}), 400

    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if hist.empty:
            return jsonify({"error": f"No data for '{ticker}'"}), 404
        if hist.index.tzinfo:
            hist.index = hist.index.tz_localize(None)

        return jsonify({
            "ticker":   ticker,
            "period":   period,
            "interval": interval,
            "count":    len(hist),
            "dates":    hist.index.strftime("%Y-%m-%d").tolist(),
            "open":     [round(float(v), 4) for v in hist["Open"]],
            "high":     [round(float(v), 4) for v in hist["High"]],
            "low":      [round(float(v), 4) for v in hist["Low"]],
            "close":    [round(float(v), 4) for v in hist["Close"]],
            "volume":   [int(v) for v in hist["Volume"]],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/cluster")
def cluster():
    """
    k-medoids clustering of all sliding windows.

    Query params:
        ticker  : e.g. AAPL
        period  : e.g. 5y
        k       : number of clusters (2–10, default 5)
        window  : window length in days (10–90, default 30)
        stride  : days between windows (1–20, default 5)
        bw      : DTW band ratio (default 0.10)
        future  : future horizon for return stats (default 10)
    """
    ticker  = request.args.get("ticker",  "AAPL").upper().strip()
    period  = request.args.get("period",  "5y")
    k       = max(2, min(10, int(request.args.get("k",       5))))
    win     = max(10, min(90, int(request.args.get("window", 30))))
    stride  = max(1, min(20, int(request.args.get("stride",  5))))
    bw      = max(1, int(win * float(request.args.get("bw",  0.10))))
    fut_len = max(1, min(60, int(request.args.get("future",  10))))

    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return jsonify({"error": f"No data for '{ticker}'"}), 404
        if hist.index.tzinfo:
            hist.index = hist.index.tz_localize(None)

        prices = np.array(hist["Close"].tolist(), dtype=np.float64)
        dates  = hist.index.strftime("%Y-%m-%d").tolist()
        n      = len(prices)

        # Extract sliding windows (exclude windows that lack future data)
        starts = list(range(0, n - win - fut_len, stride))
        norms  = [_z_norm(prices[i: i + win]) for i in starts]
        m      = len(norms)

        if m < k:
            return jsonify({
                "error": f"ウィンドウ数({m})がクラスタ数({k})より少ないです。"
                         f"ストライドを小さくするか、期間を長くしてください。"
            }), 400

        # Pairwise DTW distance matrix
        D = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                d = _dtw(norms[i], norms[j], bw)
                D[i, j] = D[j, i] = d

        # k-medoids
        medoid_idxs, assignments = _k_medoids(D, k)

        # Per-cluster statistics
        clusters_out = []
        for c in range(k):
            pts     = [i for i, a in enumerate(assignments) if a == c]
            mi      = medoid_idxs[c]
            ms      = starts[mi]

            fut_rets = []
            for pi in pts:
                end = starts[pi] + win
                if end + fut_len <= n:
                    ret = (float(prices[end + fut_len - 1]) / float(prices[end]) - 1) * 100
                    fut_rets.append(ret)

            clusters_out.append({
                "id":           c,
                "count":        len(pts),
                "medoid_start": ms,
                "medoid_date":  dates[ms],
                "medoid_norm":  [round(float(v), 5) for v in norms[mi]],
                "avg_return":   round(float(np.mean(fut_rets))   if fut_rets else 0.0, 3),
                "win_rate":     round(float(sum(r > 0 for r in fut_rets) / len(fut_rets) * 100)
                                      if fut_rets else 0.0, 1),
            })

        return jsonify({
            "ticker":        ticker,
            "n_clusters":    k,
            "window_len":    win,
            "n_windows":     m,
            "clusters":      clusters_out,
            "assignments":   assignments,
            "window_starts": starts,
            "dates":         dates,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest")
def backtest():
    """
    Walk-forward backtest to find optimal k*.

    Splits historical data into train/test, computes DTW distances from
    every test window to every training window, then evaluates directional
    accuracy for k = 1..k_max.  Returns per-k stats + recommended k*.

    Query params:
        ticker  : e.g. AAPL
        period  : e.g. 5y
        k_max   : max k to evaluate (2–30, default 20)
        window  : window length in days (10–90, default 30)
        stride  : days between windows (1–20, default 5)
        bw      : DTW band ratio (default 0.10)
        future  : future horizon days (default 10)
        split   : train fraction (0.5–0.9, default 0.7)
    """
    ticker  = request.args.get("ticker",  "AAPL").upper().strip()
    period  = request.args.get("period",  "5y")
    k_max   = max(2, min(30, int(request.args.get("k_max",   20))))
    win     = max(10, min(90, int(request.args.get("window", 30))))
    stride  = max(1, min(20, int(request.args.get("stride",  5))))
    bw      = max(1, int(win * float(request.args.get("bw",  0.10))))
    fut_len = max(1, min(60, int(request.args.get("future",  10))))
    split   = max(0.5, min(0.9, float(request.args.get("split", 0.7))))

    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return jsonify({"error": f"No data for '{ticker}'"}), 404
        if hist.index.tzinfo:
            hist.index = hist.index.tz_localize(None)

        prices = np.array(hist["Close"].tolist(), dtype=np.float64)
        dates  = hist.index.strftime("%Y-%m-%d").tolist()
        n      = len(prices)
        split_idx = int(n * split)

        all_starts   = list(range(0, n - win - fut_len, stride))
        train_starts = [s for s in all_starts if s + win + fut_len <= split_idx]
        test_starts  = [s for s in all_starts if s >= split_idx]

        n_tr = len(train_starts)
        n_te = len(test_starts)

        if n_tr < k_max:
            return jsonify({
                "error": f"訓練窓数({n_tr})が k_max({k_max})より少ないです。"
                         f"期間を長くするか k_max を小さくしてください。"
            }), 400
        if n_te < 5:
            return jsonify({
                "error": f"テスト窓数({n_te})が少なすぎます。"
                         f"期間を長くするか分割比率を変えてください。"
            }), 400

        train_norms = [_z_norm(prices[s: s + win]) for s in train_starts]
        test_norms  = [_z_norm(prices[s: s + win]) for s in test_starts]

        # Pairwise DTW: shape (n_te, n_tr)
        D = np.zeros((n_te, n_tr))
        for i in range(n_te):
            for j in range(n_tr):
                D[i, j] = _dtw(test_norms[i], train_norms[j], bw)

        train_rets = np.array([
            (float(prices[s + win + fut_len - 1]) / float(prices[s + win]) - 1) * 100
            for s in train_starts
        ])
        test_actual = np.array([
            (float(prices[s + win + fut_len - 1]) / float(prices[s + win]) - 1) * 100
            for s in test_starts
        ])

        sorted_idxs = np.argsort(D, axis=1)  # (n_te, n_tr)
        results = []
        for k in range(1, k_max + 1):
            pred    = train_rets[sorted_idxs[:, :k]].mean(axis=1)
            correct = np.sign(pred) == np.sign(test_actual)
            dir_acc = float(correct.mean() * 100)
            corr    = float(np.corrcoef(pred, test_actual)[0, 1]) if n_te > 1 else 0.0
            results.append({
                "k":            k,
                "dir_acc":      round(dir_acc, 1),
                "corr":         round(corr, 4),
                "avg_pred_ret": round(float(pred.mean()), 3),
            })

        best = max(results, key=lambda r: r["dir_acc"])

        return jsonify({
            "ticker":       ticker,
            "n_train":      n_tr,
            "n_test":       n_te,
            "split_date":   dates[split_idx],
            "results":      results,
            "best_k":       best["k"],
            "best_dir_acc": best["dir_acc"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tickers")
def suggest_tickers():
    return jsonify([
        {"symbol": "AAPL",    "name": "Apple Inc."},
        {"symbol": "MSFT",    "name": "Microsoft Corp."},
        {"symbol": "GOOGL",   "name": "Alphabet Inc."},
        {"symbol": "AMZN",    "name": "Amazon.com Inc."},
        {"symbol": "TSLA",    "name": "Tesla Inc."},
        {"symbol": "NVDA",    "name": "NVIDIA Corp."},
        {"symbol": "SPY",     "name": "S&P 500 ETF"},
        {"symbol": "QQQ",     "name": "Nasdaq-100 ETF"},
        {"symbol": "7203.T",  "name": "Toyota Motor (JP)"},
        {"symbol": "9984.T",  "name": "SoftBank Group (JP)"},
        {"symbol": "6758.T",  "name": "Sony Group (JP)"},
        {"symbol": "BTC-USD", "name": "Bitcoin / USD"},
        {"symbol": "GLD",     "name": "Gold ETF"},
    ])


if __name__ == "__main__":
    print("=" * 50)
    print(" Stock Pattern Finder — Flask Server v3")
    print("=" * 50)
    print("  GET /api/health")
    print("  GET /api/stock?ticker=AAPL&period=5y")
    print("  GET /api/cluster?ticker=AAPL&k=5&window=30")
    print("  GET /api/backtest?ticker=AAPL&k_max=20&window=30")
    print("  GET /api/tickers")
    print()
    print("  Server: http://localhost:5000")
    print("=" * 50)
    app.run(debug=True, port=5000, host="0.0.0.0")
