# utils/symbol_mapper.py
import json
import logging
from pathlib import Path
from thefuzz import fuzz, process
from typing import Tuple, Optional, List, Dict

logger = logging.getLogger(__name__)
if not logger.handlers:
    import sys, logging as _logging
    h = _logging.StreamHandler(sys.stdout)
    h.setFormatter(_logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel("INFO")

BASE = Path(__file__).resolve().parent.parent
INSTR_LIST_PATH = BASE / "data" / "kite_instruments_list.json"

# load instrument list once
def _load_instruments() -> List[Dict]:
    if not INSTR_LIST_PATH.exists():
        logger.error("Instrument list missing: %s. Run build_instruments.py or start app to build it.", INSTR_LIST_PATH)
        return []
    try:
        raw = json.loads(INSTR_LIST_PATH.read_text(encoding="utf8"))
        return raw if isinstance(raw, list) else []
    except Exception as e:
        logger.exception("Failed to read instrument list: %s", e)
        return []

_INSTRUMENTS = _load_instruments()

def _norm(s: str) -> str:
    return (s or "").strip().upper()

def _make_search_buckets():
    """
    Build two lists for fuzzy search:
      - tradingsymbols list
      - names list
    Also keep mapping index -> instrument dict
    """
    symbols = []
    names = []
    idx_map = []
    for it in _INSTRUMENTS:
        ts = _norm(it.get("tradingsymbol") or "")
        name = _norm(it.get("name") or "")
        token = it.get("instrument_token")
        exch = _norm(it.get("exchange") or "")
        if not ts and not name:
            continue
        symbols.append(ts)
        names.append(name)
        idx_map.append({"tradingsymbol": ts, "name": name, "exchange": exch, "instrument_token": token})
    return symbols, names, idx_map

_SYMBOLS_BUCKET, _NAMES_BUCKET, _IDX_MAP = _make_search_buckets()

def auto_resolve(user_input: str, prefer_exchange: str = "NSE") -> Tuple[Optional[str], float, Optional[int]]:
    """
    Return (best_tradingsymbol_or_None, confidence_percent (0-100), instrument_token_or_None)
    - Uses fuzzy scoring (combine symbol/name scores) and exchange preference.
    - If confidence < 60, we still return the top candidate (as you requested 'automatic'), but it logs the low confidence.
    """
    q = (user_input or "").strip()
    if not q:
        return None, 0.0, None

    Q = q.upper()

    # 1) quick exact matches: check tradingsymbol or name exact
    for it in _IDX_MAP:
        if Q == it["tradingsymbol"] or Q == it["name"]:
            return it["tradingsymbol"], 100.0, it.get("instrument_token")

    # 2) exact substring preference: if Q is substring of many tradingsymbols/names, pick exact symbol match first
    substring_matches = []
    for it in _IDX_MAP:
        ts = it["tradingsymbol"]
        name = it["name"]
        if Q in ts or Q in name:
            # score a base for substring
            score = 60
            # prefer same-exchange
            if prefer_exchange and prefer_exchange.upper() in (it.get("exchange") or ""):
                score += 10
            substring_matches.append((score, it))
    if substring_matches:
        substring_matches.sort(key=lambda x: x[0], reverse=True)
        best = substring_matches[0][1]
        conf = float(substring_matches[0][0])
        # clamp to 100
        conf = min(max(conf, 0.0), 100.0)
        return best["tradingsymbol"], conf, best.get("instrument_token")

    # 3) fuzzy match against tradingsymbols and names using thefuzz
    # Get top N matches from both buckets
    N = 10
    sym_matches = process.extract(Q, _SYMBOLS_BUCKET, limit=N, scorer=fuzz.token_sort_ratio)
    name_matches = process.extract(Q, _NAMES_BUCKET, limit=N, scorer=fuzz.token_sort_ratio)

    # Convert to unified candidate list with combined score
    cand_scores = {}  # idx -> best combined score
    for val, score in sym_matches:
        # find indices with this tradingsymbol (there may be duplicates)
        for idx, it in enumerate(_IDX_MAP):
            if it["tradingsymbol"] == val:
                cand_scores[idx] = max(cand_scores.get(idx, 0), score + 10)  # favor symbol matches slightly
    for val, score in name_matches:
        for idx, it in enumerate(_IDX_MAP):
            if it["name"] == val:
                cand_scores[idx] = max(cand_scores.get(idx, 0), score)

    if not cand_scores:
        # fallback: return top 1 name match if exists
        if name_matches:
            best_name, s = name_matches[0]
            # find index
            for idx, it in enumerate(_IDX_MAP):
                if it["name"] == best_name:
                    cand = it
                    conf = float(s)
                    if prefer_exchange and prefer_exchange.upper() in (cand.get("exchange") or ""):
                        conf += 5
                    conf = min(conf, 100.0)
                    if conf < 60:
                        logger.warning("Low-confidence auto-resolve for '%s' -> %s (conf=%.1f).", user_input, cand["tradingsymbol"], conf)
                    return cand["tradingsymbol"], conf, cand.get("instrument_token")
        return None, 0.0, None

    # pick highest scoring candidate, prefer NSE exchange if tie/close
    best_idx = max(cand_scores.items(), key=lambda x: x[1])[0]
    best_score = cand_scores[best_idx]
    # check for ties within 5 points and prefer NSE
    tied = [idx for idx, sc in cand_scores.items() if abs(sc - best_score) <= 5]
    if len(tied) > 1 and prefer_exchange:
        for idx in tied:
            exch = _IDX_MAP[idx].get("exchange") or ""
            if prefer_exchange.upper() in exch:
                best_idx = idx
                break

    cand = _IDX_MAP[best_idx]
    confidence = float(min(max(best_score, 0.0), 100.0))
    # Slightly boost confidence if tradingsymbol exactly contains the user tokens in order
    if Q.replace(" ", "") in cand["tradingsymbol"].replace(" ", ""):
        confidence = min(100.0, confidence + 5.0)

    if confidence < 60:
        logger.warning("Low-confidence auto-resolve for '%s' -> %s (conf=%.1f).", user_input, cand["tradingsymbol"], confidence)

    return cand["tradingsymbol"], confidence, cand.get("instrument_token")
