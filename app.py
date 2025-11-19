import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId

# app-specific modules
from db import mongo
from models import User

# analyzer and mapper
from utils.analyzer import analyze_portfolio, _build_instrument_map
from utils.symbol_mapper import auto_resolve

# Kite OAuth (used by kite_login/kite_callback)
from kiteconnect import KiteConnect

load_dotenv()

# ---------------- App / DB setup ----------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb://localhost:27017/portfolio_db")
mongo.init_app(app)

# Flask-Login
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    user_doc = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    return User(user_doc) if user_doc else None

# ---------------- Kite settings ----------------
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_SESSION_FILE = Path(os.getenv("KITE_SESSION_FILE", "data/kite_session.json"))
KITE_REDIRECT_URL = os.getenv("KITE_REDIRECT_URL", "http://127.0.0.1:5000/kite_callback")

def _save_kite_session(session_data: dict):
    KITE_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    KITE_SESSION_FILE.write_text(json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf8")

def _load_kite_session() -> dict:
    try:
        if KITE_SESSION_FILE.exists():
            return json.loads(KITE_SESSION_FILE.read_text(encoding="utf8"))
    except Exception:
        pass
    return {}

# ---------------- Auth routes ----------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        full_name = request.form["full_name"]
        password = request.form["password"]
        phone = request.form.get("phone", "")
        goal = request.form.get("goal", "")

        if mongo.db.users.find_one({"email": email}):
            return render_template("signup.html", error="Email already exists")

        user_data = {
            "email": email,
            "full_name": full_name,
            "password": generate_password_hash(password),
            "phone": phone,
            "goal": goal
        }
        mongo.db.users.insert_one(user_data)
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        user_doc = mongo.db.users.find_one({"email": email})
        if user_doc and check_password_hash(user_doc["password"], password):
            user = User(user_doc)
            login_user(user)
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------- Kite OAuth helpers ----------------
@app.route("/kite_login")
@login_required
def kite_login():
    """Redirect to Kite Connect login to get request_token."""
    if not KITE_API_KEY:
        flash("KITE_API_KEY not configured. Add it to .env", "danger")
        return redirect(url_for("home"))
    kite = KiteConnect(api_key=KITE_API_KEY)
    login_url = kite.login_url(redirect_uri=KITE_REDIRECT_URL)
    return redirect(login_url)

@app.route("/kite_callback")
def kite_callback():
    """
    Kite redirect callback. Exchanges request_token for access_token and saves session info to data/kite_session.json.
    """
    req_token = request.args.get("request_token")
    if not req_token:
        return "Missing request_token in callback URL.", 400
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        session_data = kite.generate_session(req_token, api_secret=KITE_API_SECRET)
        _save_kite_session(session_data)
        # Also refresh instrument map asynchronously next request; immediate build is fine:
        try:
            _build_instrument_map(force_refresh=True)
        except Exception:
            pass
        return """
            <html><body>
            <h3>Kite authorization successful.</h3>
            <p>Access token saved. You can close this tab and return to the app.</p>
            </body></html>
        """
    except Exception as e:
        return f"Failed to generate session: {e}", 500

@app.route("/kite_status")
@login_required
def kite_status():
    """Return basic kite session status to the UI (JSON)."""
    sess = _load_kite_session()
    has = bool(sess.get("access_token"))
    return jsonify({"has_session": has, "session": {"user_id": sess.get("user_id")}})

# ---------------- Admin / util routes ----------------
@app.route("/refresh_cache")
@login_required
def refresh_cache():
    cache_dir = os.path.join("data", "kite_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    # also refresh instrument map
    try:
        _build_instrument_map(force_refresh=True)
    except Exception:
        pass
    return redirect(url_for("home"))

@app.route("/add_manual_mapping", methods=["POST"])
@login_required
def add_manual_mapping():
    """Admin endpoint to add a manual mapping to data/kite_instruments_map.json"""
    data = request.get_json(force=True)
    key = data.get("key")
    token = data.get("token")
    if not key or not token:
        return jsonify({"error":"key and token required"}), 400
    p = Path("data/kite_instruments_map.json")
    if p.exists():
        content = json.loads(p.read_text(encoding="utf8"))
    else:
        content = {}
    content[key.strip().upper()] = int(token)
    p.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf8")
    return jsonify({"ok": True, "key": key, "token": token})

# ---------------- Main UI / analyzer ----------------
@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    error = None
    if request.method == "POST":
        companies = request.form.getlist("company")
        shares = request.form.getlist("shares")
        avg_prices = request.form.getlist("avg_price")
        portfolio_date = request.form.get("portfolio_date", "")

        if not (len(companies) == len(shares) == len(avg_prices)):
            return render_template("index.html", error="Input lists length mismatch.")

        portfolio_rows = []
        try:
            # ensure instrument map exists (lazy)
            try:
                _build_instrument_map()
            except Exception:
                app.logger.info("Instrument build skipped or failed (will try resolving anyway).")

            for c, s, p in zip(companies, shares, avg_prices):
                user_input = c.strip()
                if not user_input:
                    raise ValueError("Empty company field")
                # auto-resolve using Kite instrument list (no CSV, no user prompt)
                resolved_sym, conf, token = auto_resolve(user_input)
                if not resolved_sym:
                    raise ValueError(f"Could not resolve '{user_input}' to any Kite instrument.")
                # log low confidence for later auditing
                if conf < 60:
                    app.logger.warning("Auto-resolve low confidence for input '%s' -> %s (%.1f%%)", user_input, resolved_sym, conf)
                # use resolved tradingsymbol as canonical symbol
                symbol = resolved_sym
                portfolio_rows.append({
                    "symbol": symbol,
                    "shares": float(s),
                    "avg_price": float(p),
                    "date": portfolio_date,
                    "user_id": ObjectId(current_user.id)
                })
        except Exception as e:
            return render_template("index.html", error=str(e))

        # save portfolio to DB
        mongo.db.portfolios.delete_many({"user_id": ObjectId(current_user.id)})
        if portfolio_rows:
            mongo.db.portfolios.insert_many(portfolio_rows)

        # prepare inputs for analyzer
        symbols = [row["symbol"] for row in portfolio_rows]
        shares = [row["shares"] for row in portfolio_rows]
        avg_prices = [row["avg_price"] for row in portfolio_rows]

        try:
            result = analyze_portfolio(symbols, shares, avg_prices)
        except Exception as e:
            return render_template("index.html", error=f"Analysis failed: {e}")

        return render_template("result.html",
                               full_name=current_user.full_name,
                               email=current_user.email,
                               phone=current_user.phone,
                               goal=current_user.goal,
                               portfolio_date=portfolio_date,
                               result=result, **result)

    # GET: show saved portfolio if present
    saved = list(mongo.db.portfolios.find({"user_id": ObjectId(current_user.id)}))
    if saved:
        try:
            symbols = [item["symbol"] for item in saved]
            shares = [float(item["shares"]) for item in saved]
            avg_prices = [float(item["avg_price"]) for item in saved]
            portfolio_date = saved[0].get("date", "")
            result = analyze_portfolio(symbols, shares, avg_prices)
            return render_template("result.html",
                                   full_name=current_user.full_name,
                                   email=current_user.email,
                                   phone=current_user.phone,
                                   goal=current_user.goal,
                                   portfolio_date=portfolio_date,
                                   result=result, **result)
        except Exception as e:
            error = f"Failed to analyze saved data: {e}"
    return render_template("index.html", error=error)

# ---------------- Edit portfolio (uses same auto-resolve) ----------------
@app.route("/edit_portfolio", methods=["GET", "POST"])
@login_required
def edit_portfolio():
    if request.method == "POST":
        portfolio_date = request.form["portfolio_date"]
        companies = request.form.getlist("company")
        shares = request.form.getlist("shares")
        avg_prices = request.form.getlist("avg_price")

        if not (len(companies) == len(shares) == len(avg_prices)):
            return render_template("edit_portfolio.html", error="Input list length mismatch.")

        try:
            mongo.db.portfolios.delete_many({"user_id": ObjectId(current_user.id)})
            new_portfolio = []
            # build instrument map lazily
            try:
                _build_instrument_map()
            except Exception:
                pass
            for c, s, p in zip(companies, shares, avg_prices):
                user_input = c.strip()
                if not user_input:
                    raise ValueError("Empty company field")
                sym, conf, token = auto_resolve(user_input)
                if not sym:
                    raise ValueError(f"Symbol not found for '{c}'")
                if conf < 60:
                    app.logger.warning("Auto-resolve low confidence for input '%s' -> %s (%.1f%%)", user_input, sym, conf)
                new_portfolio.append({
                    "symbol": sym,
                    "shares": float(s),
                    "avg_price": float(p),
                    "date": portfolio_date,
                    "user_id": ObjectId(current_user.id)
                })
            if new_portfolio:
                mongo.db.portfolios.insert_many(new_portfolio)
            return redirect(url_for("home"))
        except Exception as e:
            return render_template("edit_portfolio.html", error=f"Update failed: {e}")

    portfolio = list(mongo.db.portfolios.find({"user_id": ObjectId(current_user.id)}))
    portfolio_date = portfolio[0].get("date", "") if portfolio else ""
    return render_template("edit_portfolio.html", portfolio=portfolio, portfolio_date=portfolio_date)

# ---------------- Run app ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # ensure instrument map exists early (non-blocking attempt)
    try:
        _build_instrument_map()
    except Exception:
        app.logger.info("Instrument map build skipped at startup.")
    app.run(host="0.0.0.0", port=port)
