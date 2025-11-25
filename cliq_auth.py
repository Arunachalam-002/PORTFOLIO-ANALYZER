# cliq_auth.py
from flask import Blueprint, request, jsonify
import os

cliq_bp = Blueprint("cliq_bp", __name__)

# Authentication helper
def require_auth():
    api_key = os.getenv("CLIQ_API_KEY", "change-me-in-render")
    auth = request.headers.get("Authorization", "")
    return auth == f"Bearer {api_key}"

# ---- Example Endpoint for Cliq ----
@cliq_bp.route("/cliq/portfolio/<user_id>", methods=["GET"])
def cliq_portfolio(user_id):
    if not require_auth():
        return jsonify({"error": "unauthorized"}), 401

    # TODO: replace dummy with your real portfolio logic
    portfolio = {
        "user_id": user_id,
        "total_value": 12345.67,
        "holdings": [
            {
                "symbol": "RELIANCE",
                "qty": 10,
                "ltp": 2500.0,
                "value": 25000.0,
                "change_pct": 1.5
            }
        ]
    }

    return jsonify(portfolio)
