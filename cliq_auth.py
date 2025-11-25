from flask import Blueprint, request, jsonify
import os
from bson.objectid import ObjectId
from db import mongo
from utils.analyzer import analyze_portfolio  # your real analyzer

cliq_bp = Blueprint("cliq_bp", __name__)

# ---------------- Auth ----------------
def require_auth():
    api_key = os.getenv("CLIQ_API_KEY", "change-me-in-render")
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {api_key}"

# ---------------- User resolution ----------------
def _resolve_user_oid(user_id):
    """
    Try to interpret user_id as an ObjectId string, else try email, else full_name.
    Returns ObjectId or None.
    """
    try:
        return ObjectId(user_id)
    except Exception:
        pass

    try:
        u = mongo.db.users.find_one({"email": user_id})
        if u:
            return u["_id"]
    except Exception:
        pass

    try:
        u = mongo.db.users.find_one({"full_name": user_id})
        if u:
            return u["_id"]
    except Exception:
        pass

    return None

# ---------------- DB fallback ----------------
def _db_portfolio_fallback(oid):
    saved = list(mongo.db.portfolios.find({"user_id": oid}))
    holdings_list = []
    total = 0.0
    for item in saved:
        sym = item.get("symbol", "")
        try:
            qty = float(item.get("shares", 0))
        except Exception:
            qty = 0.0

        ltp_candidate = item.get("ltp") or item.get("last_price") or item.get("market_price") or item.get("avg_price") or 0
        try:
            ltp = float(ltp_candidate)
        except Exception:
            ltp = 0.0

        value = qty * ltp
        total += value
        h = {"symbol": sym, "qty": qty, "ltp": ltp, "value": value}
        if item.get("change_pct") is not None:
            h["change_pct"] = item.get("change_pct")
        holdings_list.append(h)
    return {"user_id": str(oid), "total_value": float(total), "holdings": holdings_list}

# ---------------- Core: normalized portfolio ----------------
def get_portfolio_for_user(user_id_input):
    """
    Returns normalized portfolio dict:
      { user_id, total_value, holdings: [ {symbol, qty, ltp, value, [change_pct]} ] }
    """
    oid = _resolve_user_oid(user_id_input)
    if oid is None:
        # Not resolvable -> return empty but not error
        return {"user_id": user_id_input, "total_value": 0.0, "holdings": []}

    saved = list(mongo.db.portfolios.find({"user_id": oid}))
    if not saved:
        return {"user_id": str(oid), "total_value": 0.0, "holdings": []}

    symbols = [item.get("symbol") for item in saved]
    shares_list = []
    avg_prices_list = []
    for item in saved:
        try:
            shares_list.append(float(item.get("shares", 0)))
        except Exception:
            shares_list.append(0.0)
        try:
            avg_prices_list.append(float(item.get("avg_price", 0)))
        except Exception:
            avg_prices_list.append(0.0)

    # Preferred: use your analyzer (may require Kite session)
    try:
        result = analyze_portfolio(symbols, shares_list, avg_prices_list)
        if isinstance(result, dict) and "total_investment" in result and "portfolio_data" in result:
            total_value = float(result.get("total_investment", 0.0))
            holdings_out = []
            for row in result.get("portfolio_data", []):
                sym = row.get("symbol") or row.get("ticker") or ""
                qty = float(row.get("shares", row.get("qty", 0)))
                investment = float(row.get("investment", row.get("value", 0)))
                avg_price = float(row.get("avg_price", 0))
                # derive ltp if possible: investment/qty else avg_price
                if qty:
                    ltp = investment / qty
                else:
                    ltp = avg_price
                entry = {"symbol": sym, "qty": qty, "ltp": ltp, "value": investment}
                if "change_pct" in row:
                    entry["change_pct"] = row["change_pct"]
                holdings_out.append(entry)
            return {"user_id": str(oid), "total_value": total_value, "holdings": holdings_out}
    except Exception:
        # Analyzer may fail if Kite session missing; fallback to DB below
        pass

    # Fallback: compute from DB fields
    return _db_portfolio_fallback(oid)

# ---------------- Endpoints ----------------
@cliq_bp.route("/cliq/portfolio/<user_id>", methods=["GET"])
def cliq_portfolio(user_id):
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401
    portfolio = get_portfolio_for_user(user_id)
    return jsonify(portfolio)

@cliq_bp.route("/cliq/portfolio/<user_id>/holdings", methods=["GET"])
def cliq_portfolio_holdings(user_id):
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401
    portfolio = get_portfolio_for_user(user_id)
    return jsonify({"user_id": portfolio["user_id"], "holdings": portfolio.get("holdings", [])})
