# utils/analyzer.py  -- KITE-ONLY version (no yfinance fallback)
import os
import time
import tempfile
import shutil
import logging
import json
from pathlib import Path
from threading import Lock
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from kiteconnect import KiteConnect
try:
    from kiteconnect.exceptions import KiteException
except Exception:
    try:
        from kiteconnect import KiteException
    except Exception:
        class KiteException(Exception):
            pass

from prophet import Prophet
from arch import arch_model
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

# ---------------- Logging ----------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))

# ---------------- Config & Cache ----------------
CACHE = Path(os.getenv("KITE_CACHE_DIR", "data/kite_cache"))
CACHE.mkdir(parents=True, exist_ok=True)

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 24 * 3600))
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", 3))
DOWNLOAD_BACKOFF = float(os.getenv("DOWNLOAD_BACKOFF", 2.0))
POLITE_SLEEP = float(os.getenv("POLITE_SLEEP", 0.25))
KITE_MAX_CALLS_PER_MIN = int(os.getenv("KITE_MAX_CALLS_PER_MIN", 120))
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_SESSION_FILE = Path(os.getenv("KITE_SESSION_FILE", "data/kite_session.json"))

# instrument persistence
INSTR_MAP_PATH = Path("data/kite_instruments_map.json")       # dict: symbol->token
INSTR_LIST_PATH = Path("data/kite_instruments_list.json")    # raw list returned by kite.instruments("NSE")

# ensure data dir
INSTR_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------- Helpers: atomic write ----------------
def _atomic_write_json(obj, fp: Path):
    tmp_dir = fp.parent
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_json_", dir=str(tmp_dir))
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        shutil.move(tmp_path, str(fp))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def _atomic_write_csv(df: pd.DataFrame, fp: Path):
    tmp_dir = fp.parent
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_cache_", dir=str(tmp_dir))
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=True)
        shutil.move(tmp_path, str(fp))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def _cache(sym: str, per: str) -> Path:
    safe = str(sym).replace("/", "_").replace("^", "_").replace(":", "_")
    return CACHE / f"{safe}_{per}.csv"

def _is_fresh(fp: Path, ttl_seconds: int = CACHE_TTL_SECONDS) -> bool:
    try:
        if not fp.exists():
            return False
        mtime = fp.stat().st_mtime
        return (time.time() - mtime) <= ttl_seconds
    except Exception:
        return False

def _choose_price_series(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    cols = list(df.columns)
    for c in ["close", "Close", "last_price", "close_price", "adj_close", "Adj Close"]:
        if c in cols:
            return df[c].copy()
    for c in cols:
        try:
            if np.issubdtype(df[c].dtype, np.number):
                return df[c].copy()
        except Exception:
            continue
    return pd.Series(dtype=float)

# ---------------- Kite client (lazy init) ----------------
_kite_client = None
_kite_lock = Lock()

def _load_session_token() -> str:
    try:
        if KITE_SESSION_FILE.exists():
            data = json.loads(KITE_SESSION_FILE.read_text(encoding="utf8"))
            token = data.get("access_token") or data.get("ACCESS_TOKEN") or ""
            return token
    except Exception:
        pass
    return os.getenv("KITE_ACCESS_TOKEN", "")

def _get_kite() -> Optional[KiteConnect]:
    global _kite_client
    with _kite_lock:
        if _kite_client:
            return _kite_client
        if not KITE_API_KEY:
            logger.warning("KITE_API_KEY not set.")
            return None
        try:
            kite = KiteConnect(api_key=KITE_API_KEY)
            access = _load_session_token()
            if access:
                kite.set_access_token(access)
            _kite_client = kite
            return _kite_client
        except Exception as e:
            logger.error("Failed to init Kite client: %s", e)
            return None

# ---------------- Instrument map/list management ----------------
_instr_lock = Lock()
try:
    if INSTR_MAP_PATH.exists():
        INSTRUMENT_MAP: Dict[str, Any] = json.loads(INSTR_MAP_PATH.read_text(encoding="utf8"))
    else:
        INSTRUMENT_MAP = {}
except Exception:
    INSTRUMENT_MAP = {}

try:
    if INSTR_LIST_PATH.exists():
        INSTRUMENT_LIST = json.loads(INSTR_LIST_PATH.read_text(encoding="utf8"))
    else:
        INSTRUMENT_LIST = []
except Exception:
    INSTRUMENT_LIST = []

def _persist_instruments(map_obj: Dict[str, Any], list_obj: List[Any]):
    try:
        _atomic_write_json(map_obj, INSTR_MAP_PATH)
        _atomic_write_json(list_obj, INSTR_LIST_PATH)
    except Exception:
        pass

def _build_instrument_map(force_refresh: bool = False) -> Tuple[Dict[str,int], List[dict]]:
    """
    Download instruments (NSE) from Kite and build both:
      - INSTRUMENT_MAP: dict of many symbol variants -> instrument_token
      - INSTRUMENT_LIST: raw list for fuzzy matching
    Returns (map, list)
    """
    global INSTRUMENT_MAP, INSTRUMENT_LIST
    if INSTRUMENT_MAP and INSTRUMENT_LIST and not force_refresh:
        return INSTRUMENT_MAP, INSTRUMENT_LIST
    kite = _get_kite()
    if not kite:
        logger.warning("Kite client not available; cannot build instrument map.")
        return INSTRUMENT_MAP, INSTRUMENT_LIST
    try:
        logger.info("Downloading Kite instruments (this can take ~10s)...")
        instruments = kite.instruments("NSE")  # returns list of dicts
        map_local: Dict[str,int] = {}
        # We'll keep also the raw list for fuzzy scanning
        for it in instruments:
            trad = (it.get("tradingsymbol") or "").strip()
            name = (it.get("name") or "").strip()
            exch = (it.get("exchange") or "").strip()
            token = it.get("instrument_token")
            if not trad or token is None:
                continue
            trad_up = trad.upper()
            # Add many variants for robust resolution
            map_local[trad_up] = token
            map_local[f"{trad_up}.NS"] = token
            map_local[f"{trad_up}.NSE"] = token
            map_local[f"NSE:{trad_up}"] = token
            map_local[trad_up.replace(".", "")] = token
            # also include name-based keys if name exists
            if name:
                map_local[name.upper()] = token
                map_local[f"{name.upper()}.NS"] = token
        with _instr_lock:
            INSTRUMENT_MAP = map_local
            INSTRUMENT_LIST = instruments
            _persist_instruments(INSTRUMENT_MAP, INSTRUMENT_LIST)
        logger.info("Instrument map built with %d entries.", len(INSTRUMENT_MAP))
        return INSTRUMENT_MAP, INSTRUMENT_LIST
    except Exception as e:
        logger.error("Failed to download instruments list: %s", e)
        return INSTRUMENT_MAP, INSTRUMENT_LIST

# ---------------- Robust resolver (no yfinance fallback) ----------------
def _find_candidates_in_list(query: str, max_results: int = 10) -> List[Tuple[str,int]]:
    """Return (tradingsymbol, token) matches sorted by simple score."""
    q = query.strip().upper()
    candidates = []
    for it in INSTRUMENT_LIST:
        ts = (it.get("tradingsymbol") or "").upper()
        name = (it.get("name") or "").upper()
        exch = (it.get("exchange") or "").upper()
        token = it.get("instrument_token")
        if not token:
            continue
        score = 0
        if q == ts:
            score += 100
        if q in ts:
            score += 50
        if q == name:
            score += 40
        if q in name:
            score += 20
        # prefer NSE exchange rows
        if "NSE" in exch:
            score += 5
        if score > 0:
            candidates.append((score, ts, token))
    candidates.sort(reverse=True)
    out = [(ts, token) for (_, ts, token) in candidates[:max_results]]
    return out

def resolve_kite_token(raw_sym: str) -> Optional[int]:
    """
    Resolve a user-supplied symbol (e.g., 'TATAMOTORS.NS', 'TATAMOTORS', 'NSEI') to a Kite instrument_token.
    This function:
      - checks INSTRUMENT_MAP (fast)
      - tries normalized variants
      - uses fuzzy substring search over INSTRUMENT_LIST
      - caches top match into INSTRUMENT_MAP for future
    IMPORTANT: Does NOT fall back to yfinance.
    """
    global INSTRUMENT_MAP, INSTRUMENT_LIST
    if not raw_sym:
        return None
    key = raw_sym.strip().upper()

    # ensure instrument data present
    if (not INSTRUMENT_MAP) or (not INSTRUMENT_LIST):
        _build_instrument_map()

    # 1) direct hit
    if key in INSTRUMENT_MAP:
        return INSTRUMENT_MAP[key]

    # 2) normalized variants
    base = key.split(".", 1)[0]
    variants = [key, base, f"{base}.NS", f"{base}.NSE", f"NSE:{base}", base.replace(".", "")]
    for v in variants:
        if v in INSTRUMENT_MAP:
            return INSTRUMENT_MAP[v]

    # 3) fuzzy substring search in instrument list
    matches = _find_candidates_in_list(key, max_results=5)
    if matches:
        top_ts, top_token = matches[0]
        # cache mapping for quicker future lookup
        with _instr_lock:
            INSTRUMENT_MAP[key] = top_token
            INSTRUMENT_MAP[top_ts] = top_token
            _persist_instruments(INSTRUMENT_MAP, INSTRUMENT_LIST)
        logger.info("Resolved %s -> %s (instrument %s) by fuzzy lookup", raw_sym, top_token, top_ts)
        return top_token

    # 4) No result
    logger.error("Kite: No instrument token found for %s. Add manual mapping to data/kite_instruments_map.json if necessary.", raw_sym)
    return None

# ---------------- Rate limiter ----------------
class RateLimiter:
    def __init__(self, per_minute: int = 120):
        self.capacity = max(1, int(per_minute))
        self.tokens = self.capacity
        self.updated_at = time.time()
        self.lock = Lock()
    def _refill(self):
        now = time.time()
        elapsed = now - self.updated_at
        refill = (elapsed / 60.0) * self.capacity
        if refill >= 1:
            self.tokens = min(self.capacity, self.tokens + int(refill))
            self.updated_at = now
    def consume(self, n: int = 1) -> bool:
        with self.lock:
            self._refill()
            if self.tokens >= n:
                self.tokens -= n
                return True
            return False
    def wait_for_token(self):
        while not self.consume(1):
            time.sleep(0.5)

RATE_LIMITER = RateLimiter(KITE_MAX_CALLS_PER_MIN)

# ---------------- Kite download (Kite-only) ----------------
def _kite_historical_series(sym: str, period: str = "5y") -> pd.Series:
    """
    Download daily Close series for sym via Kite. No external fallbacks.
    Returns empty Series on failure.
    """
    fp = _cache(sym, period)
    # try fresh cache first
    if _is_fresh(fp):
        try:
            df = pd.read_csv(fp, index_col=0, parse_dates=True)
            s = _choose_price_series(df)
            if not s.empty:
                s.name = sym
                return s
        except Exception:
            try:
                fp.unlink(missing_ok=True)
            except Exception:
                pass

    token = resolve_kite_token(sym)
    if token is None:
        # Do NOT fallback to yfinance. Log error and return empty Series.
        logger.error("No Kite instrument for %s — returning empty series (no fallback).", sym)
        return pd.Series(dtype=float)

    # compute date range
    to_date = pd.Timestamp.today().normalize()
    if period.endswith("y"):
        years = int(period[:-1])
        from_date = (to_date - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    elif period.endswith("d"):
        days = int(period[:-1])
        from_date = (to_date - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    else:
        from_date = (to_date - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    to_date_str = to_date.strftime("%Y-%m-%d")

    delay = 1.0
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            RATE_LIMITER.wait_for_token()
            kite = _get_kite()
            if not kite:
                raise RuntimeError("Kite client not available or access token missing.")
            logger.info("Kite historical_data for %s (token=%s) %s -> %s (attempt %d)", sym, token, from_date, to_date_str, attempt)
            candles = kite.historical_data(instrument_token=token, from_date=from_date, to_date=to_date_str, interval="day")
            if not candles:
                raise RuntimeError("Empty candles from Kite")
            df = pd.DataFrame(candles)
            # index handling
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                df = df.set_index("date").sort_index()
            elif "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
                df = df.set_index("time").sort_index()
            df.columns = [c.lower() for c in df.columns]
            if "close" in df.columns:
                series = df["close"].copy()
            elif "last_price" in df.columns:
                series = df["last_price"].copy()
            else:
                series = _choose_price_series(df)
            if series is None or series.empty:
                raise RuntimeError("No numeric price in Kite response")
            series = series.dropna()
            series.name = sym
            # cache
            try:
                fp.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write_csv(series.to_frame(name="Close"), fp)
            except Exception as e:
                logger.debug("Cache write failed for %s: %s", fp, e)
            time.sleep(POLITE_SLEEP)
            return series
        except KiteException as kex:
            logger.warning("Kite exception for %s: %s", sym, kex)
            if "Invalid access token" in str(kex) or "token" in str(kex).lower():
                break
            last_exc = kex
        except Exception as e:
            logger.info("Kite download attempt %d for %s failed: %s", attempt, sym, e)
            last_exc = e
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(delay)
                delay *= DOWNLOAD_BACKOFF
            else:
                logger.error("All Kite download attempts failed for %s: %s", sym, e)

    # if we reach here, failed
    logger.error("Failed to fetch Kite series for %s after retries.", sym)
    return pd.Series(dtype=float)

# ---------------- Sector mapping (Kite-first) ----------------
_SECTOR: Dict[str, str] = {}
def sector(sym: str) -> str:
    if sym in _SECTOR:
        return _SECTOR[sym]
    s = "Other"
    kite = _get_kite()
    try:
        token = resolve_kite_token(sym)
        # Kite doesn't give sector in instruments.json reliably; set Other or use external mapping if desired
        s = "Other"
    except Exception:
        s = "Other"
    _SECTOR[sym] = s
    time.sleep(0.05)
    return s

# ---------------- ML Utils (same as before) ----------------
def pca_diversification(ret_df: pd.DataFrame) -> float:
    try:
        if ret_df.shape[1] < 2:
            return 0.0
        X = StandardScaler().fit_transform(ret_df.dropna())
        if X.size == 0:
            return 0.0
        evr = PCA().fit(X).explained_variance_ratio_[0]
        return round(1 - float(evr), 2)
    except Exception as e:
        logger.debug("pca_diversification error: %s", e)
        return 0.0

def rf_risk_label(exp_ret: float, vol: float, sharpe: float) -> str:
    try:
        rs, vs = np.meshgrid(np.linspace(-20, 20, 9), np.linspace(5, 60, 12))
        X = np.column_stack([rs.ravel(), vs.ravel()])
        y = [0 if (r / v) > 1 else 1 if (r / v) > 0.2 else 2 for r, v in X]
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X, y)
        label = rf.predict([[float(exp_ret), float(vol)]])[0]
        return {0: "Low", 1: "Moderate", 2: "High"}.get(int(label), "High")
    except Exception as e:
        logger.debug("rf_risk_label error: %s", e)
        return "High"

# ---------------- Main Analyzer (same interface) ----------------
def analyze_portfolio(symbols: List[str], shares: List[float], avg_prices: List[float], lookback: str = "5y") -> Dict[str, Any]:
    symbols = [str(s).strip() for s in symbols]
    shares = [float(s) for s in shares]
    avg_prices = [float(a) for a in avg_prices]

    # ensure instrument map loaded
    if not INSTRUMENT_MAP or not INSTRUMENT_LIST:
        _build_instrument_map()

    # download each series synchronously using Kite only
    series_map: Dict[str, pd.Series] = {}
    for s in symbols:
        ser = _kite_historical_series(s, lookback)
        if ser is None or ser.empty:
            logger.warning("no data for %s", s)
            continue
        ser = ser.dropna()
        if ser.empty:
            continue
        series_map[s] = ser

    if not series_map:
        raise ValueError("No valid data for provided symbols.")

    df = pd.concat(series_map.values(), axis=1, keys=series_map.keys())
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    valid_symbols = [s for s in symbols if s in df.columns]
    if not valid_symbols:
        raise ValueError("No valid stocks after filtering.")

    idxs = [symbols.index(s) for s in valid_symbols]
    sh_arr = np.array([shares[i] for i in idxs], dtype=float)
    ap_arr = np.array([avg_prices[i] for i in idxs], dtype=float)
    invest = sh_arr * ap_arr
    total_invest = float(invest.sum())
    w = invest / total_invest if total_invest else np.zeros_like(invest)

    port = (df[valid_symbols] * sh_arr).sum(axis=1).dropna()
    if port.empty:
        raise ValueError("Computed portfolio time-series is empty.")

    rets = port.pct_change().dropna()
    if not rets.empty:
        exp = float(rets.mean() * 252 * 100)
        vol = float(rets.std() * np.sqrt(252) * 100)
        sharpe = float(exp / vol) if vol else 0.0
        var5 = float(np.percentile(rets, 5) * 100)
        mdd = float(((port - port.cummax()) / port.cummax()).min() * 100)
    else:
        exp = vol = sharpe = var5 = mdd = 0.0

    # Beta vs NIFTY: try to resolve a sensible NIFTY instrument token from instrument list
    try:
        nifty_candidates = ["NIFTY 50", "NIFTY50", "NIFTY", "NSEI", "NSE:NIFTY 50"]
        nifty_series = pd.Series(dtype=float)
        for ns in nifty_candidates:
            tok = resolve_kite_token(ns)
            if tok:
                # create pseudo-symbol name to fetch via token -> use token-specific logic below
                # Kite historical_data expects instrument_token; we'll wrap a tiny helper to fetch by token
                try:
                    RATE_LIMITER.wait_for_token()
                    kite = _get_kite()
                    candles = kite.historical_data(instrument_token=tok, from_date=(pd.Timestamp.today()-pd.DateOffset(years=5)).strftime("%Y-%m-%d"), to_date=pd.Timestamp.today().strftime("%Y-%m-%d"), interval="day")
                    if candles:
                        df_n = pd.DataFrame(candles)
                        if "date" in df_n.columns:
                            df_n["date"] = pd.to_datetime(df_n["date"]).dt.tz_localize(None)
                            df_n = df_n.set_index("date").sort_index()
                        if "close" in df_n.columns:
                            nifty_series = df_n["close"].copy()
                            break
                except Exception:
                    continue
        nifty_r = nifty_series.pct_change().dropna() if not nifty_series.empty else pd.Series(dtype=float)
        comb = pd.concat([rets, nifty_r], axis=1).dropna()
        if comb.shape[1] >= 2 and len(comb) > 5:
            beta = float(np.cov(comb.iloc[:,0], comb.iloc[:,1])[0,1] / np.var(comb.iloc[:,1]))
        else:
            beta = 0.0
    except Exception as e:
        logger.debug("beta calc failed: %s", e)
        beta = 0.0

    sec_w: Dict[str,float] = {}
    for s, wt in zip(valid_symbols, w):
        sec = sector(s)
        sec_w[sec] = sec_w.get(sec, 0.0) + float(wt)
    div_sector = 1 - sum(v ** 2 for v in sec_w.values()) if sec_w else 0.0

    try:
        pct_df = df[valid_symbols].pct_change().dropna()
        ml_div = pca_diversification(pct_df) if not pct_df.empty else 0.0
    except Exception as e:
        logger.debug("ml_div error: %s", e)
        ml_div = 0.0
    ml_risk = rf_risk_label(exp, vol, sharpe)

    pred_ret = 0.0
    try:
        model = Prophet(daily_seasonality=True)
        df_prop = pd.DataFrame({"ds": port.index, "y": port.values})
        model.fit(df_prop)
        future = model.make_future_dataframe(periods=30)
        pred_df = model.predict(future)
        if "yhat" in pred_df.columns:
            pred = pred_df["yhat"].iloc[-30:].values
            if len(pred) and float(port.values[-1]) != 0:
                pred_ret = float((pred[-1] - float(port.values[-1])) / float(port.values[-1]) * 100)
    except Exception as e:
        logger.debug("Prophet failed: %s", e)
        pred_ret = 0.0

    sigma = 0.0
    try:
        log_r = np.log(port / port.shift(1)).dropna() * 100
        if not log_r.empty and len(log_r) > 10:
            g = arch_model(log_r, vol="Garch", p=1, q=1, dist="normal").fit(disp="off")
            fcast = g.forecast(horizon=30, reindex=False)
            var_forecast = None
            try:
                var_forecast = fcast.variance.values
            except Exception:
                try:
                    var_forecast = fcast.variance
                except Exception:
                    var_forecast = None
            if var_forecast is not None:
                try:
                    sigma = float(np.sqrt(var_forecast.mean()))
                except Exception:
                    try:
                        sigma = float(np.sqrt(var_forecast[-1].mean()))
                    except Exception:
                        sigma = 0.0
            sigma = float(np.clip(sigma, 0, 100))
    except Exception as e:
        logger.debug("GARCH failed: %s", e)
        sigma = 0.0
    pred_sharpe = float(pred_ret / sigma) if sigma else 0.0

    table = []
    for s, sh, ap, iv, wt in zip(valid_symbols, sh_arr, ap_arr, invest, w):
        table.append({
            "symbol": s,
            "shares": int(round(sh)),
            "avg_price": float(ap),
            "investment": round(float(iv), 2),
            "weight": round(float(wt * 100), 2)
        })

    return {
        "total_investment": round(float(total_invest), 2),
        "expected_return": round(exp, 2),
        "volatility": round(vol, 2),
        "sharpe_ratio": round(sharpe, 2),
        "var_5": round(var5, 2),
        "max_drawdown": round(mdd, 2),
        "beta": round(beta, 2),
        "diversification": round(div_sector, 2),
        "ml_diversification_score": ml_div,
        "ml_risk_level": ml_risk,
        "ml_predicted_return": round(pred_ret, 2),
        "ml_predicted_volatility": round(sigma, 2),
        "ml_predicted_sharpe": round(pred_sharpe, 2),
        "portfolio_data": table
    }
