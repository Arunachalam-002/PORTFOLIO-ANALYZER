import os
import json
import shutil
from pathlib import Path
from dotenv import load_dotenv
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort
from flask_login import LoginManager, login_required, current_user, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from bson.objectid import ObjectId
from bson.errors import InvalidId

# optional CSRF - will be used only if flask-wtf is installed
try:
    from flask_wtf import CSRFProtect
    _HAS_CSRF = True
except Exception:
    _HAS_CSRF = False

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
# SECRET_KEY (from Render or local .env)
app.secret_key = os.getenv("SECRET_KEY", "default-secret-key")

# read MONGO_URI and strip any accidental whitespace/newlines (safe if env var missing)
raw_mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/portfolio_db")
mongo_uri = (raw_mongo_uri or "").strip()
app.config["MONGO_URI"] = mongo_uri

# <<< CHANGE — support Atlas database name
app.config["MONGO_DBNAME"] = os.getenv("MONGO_DBNAME", "portfolio_db")

mongo.init_app(app)

if _HAS_CSRF:
    csrf = CSRFProtect(app)
    app.logger.info("CSRF protection enabled.")
else:
    app.logger.info("CSRF protection not enabled (flask-wtf missing).")

# Flask-Login
login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    try:
        user_doc = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    except Exception:
        user_doc = None
    return User(user_doc) if user_doc else None

# ---------------- Kite settings ----------------
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_SESSION_FILE = Path(os.getenv("KITE_SESSION_FILE", "data/kite_session.json"))
KITE_REDIRECT_URL = os.getenv("KITE_REDIRECT_URL", "http://127.0.0.1:5000/kite_callback")

def _save_kite_session(session_data: dict):
    KITE_SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = KITE_SESSION_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(session_data, ensure_ascii=False, indent=2), encoding="utf8")
    os.replace(str(tmp), str(KITE_SESSION_FILE))
    try:
        os.chmod(str(KITE_SESSION_FILE), 0o600)
    except Exception:
        pass

def _load_kite_session() -> dict:
    try:
        if KITE_SESSION_FILE.exists():
            return json.loads(KITE_SESSION_FILE.read_text(encoding="utf8"))
    except Exception:
        app.logger.exception("Failed to load kite session file")
    return {}

# ---------------- Simple admin decorator ----------------
def admin_required(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not getattr(current_user, "is_admin", False):
            app.logger.warning("Unauthorized admin access attempt by user id=%s", getattr(current_user, "id", "anon"))
            abort(403)
        return f(*args, **kwargs)
    return wrapped

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
            "goal": goal,
            "is_admin": False
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
    if not KITE_API_KEY:
        flash("KITE_API_KEY not configured. Add it to .env", "danger")
        return redirect(url_for("home"))
    kite = KiteConnect(api_key=KITE_API_KEY)
    login_url = kite.login_url(redirect_uri=KITE_REDIRECT_URL)
    return redirect(login_url)

@app.route("/kite_callback")
def kite_callback():
    req_token = request.args.get("request_token")
    if not req_token:
        return "Missing request_token in callback URL.", 400
    try:
        kite = KiteConnect(api_key=KITE_API_KEY)
        session_data = kite.generate_session(req_token, api_secret=KITE_API_SECRET)
        _save_kite_session(session_data)
        try:
            _build_instrument_map(force_refresh=True)
        except Exception:
            app.logger.info("Instrument map refresh deferred after kite callback.")
        flash("Kite authorization successful — session saved.", "success")
        return redirect(url_for("home"))
    except Exception as e:
        app.logger.exception("Failed generating kite session")
        return f"Failed to generate session: {e}", 500

@app.route("/kite_status")
@login_required
def kite_status():
    sess = _load_kite_session()
    has = bool(sess.get("access_token"))
    return jsonify({"has_session": has, "session": {"user_id": sess.get("user_id")}})

# ---------------- Admin / util routes ----------------
@app.route("/refresh_cache")
@login_required
@admin_required
def refresh_cache():
    cache_dir = os.path.join("data", "kite_cache")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        _build_instrument_map(force_refresh=True)
    except Exception:
        app.logger.exception("Instrument map refresh failed in refresh_cache")
    flash("Cache refreshed.", "success")
    return redirect(url_for("home"))

@app.route("/add_manual_mapping", methods=["POST"])
@login_required
@admin_required
def add_manual_mapping():
    data = request.get_json(force=True)
    key = data.get("key")
    token = data.get("token")
    if not key or not token:
        return jsonify({"error":"key and token required"}), 400
    p = Path("data/kite_instruments_map.json")
    if p.exists():
        try:
            content = json.loads(p.read_text(encoding="utf8"))
        except Exception:
            content = {}
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
        # This POST is used when user enters a portfolio on the homepage
        companies = request.form.getlist("company")
        shares = request.form.getlist("shares")
        avg_prices = request.form.getlist("avg_price")
        portfolio_date = request.form.get("portfolio_date", "")

        if not (len(companies) == len(shares) == len(avg_prices)):
            return render_template("index.html", error="Input lists length mismatch.")

        portfolio_rows = []
        try:
            try:
                _build_instrument_map()
            except Exception:
                app.logger.info("Instrument build skipped or failed (will try resolving anyway).")

            for c, s, p in zip(companies, shares, avg_prices):
                user_input = c.strip()
                if not user_input:
                    raise ValueError("Empty company field")
                resolved_sym, conf, token = auto_resolve(user_input)
                if not resolved_sym:
                    raise ValueError(f"Could not resolve '{user_input}' to any Kite instrument.")
                if conf < 60:
                    app.logger.warning("Auto-resolve low confidence for input '%s' -> %s (%.1f%%)", user_input, resolved_sym, conf)
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

        # replace DB
        mongo.db.portfolios.delete_many({"user_id": ObjectId(current_user.id)})
        if portfolio_rows:
            mongo.db.portfolios.insert_many(portfolio_rows)

        symbols = [row["symbol"] for row in portfolio_rows]
        shares_list = [row["shares"] for row in portfolio_rows]
        avg_prices_list = [row["avg_price"] for row in portfolio_rows]

        try:
            result = analyze_portfolio(symbols, shares_list, avg_prices_list)
        except Exception as e:
            return render_template("index.html", error=f"Analysis failed: {e}")

        return render_template("result.html",
                               full_name=current_user.full_name,
                               email=current_user.email,
                               phone=current_user.phone,
                               goal=current_user.goal,
                               portfolio_date=portfolio_date,
                               result=result, **result)

    # GET: show saved portfolio if present (and show analysis)
    saved = list(mongo.db.portfolios.find({"user_id": ObjectId(current_user.id)}))
    if saved:
        try:
            symbols = [item["symbol"] for item in saved]
            shares_list = [float(item["shares"]) for item in saved]
            avg_prices_list = [float(item["avg_price"]) for item in saved]
            portfolio_date = saved[0].get("date", "")
            result = analyze_portfolio(symbols, shares_list, avg_prices_list)
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

# ---------------- Edit portfolio (validate -> save -> analyze -> render result) ----------------
@app.route("/edit_portfolio", methods=["GET", "POST"])
@login_required
def edit_portfolio():
    if request.method == "POST":
        portfolio_date = request.form.get("portfolio_date", "")
        # prefer company[] style names; fallback to company
        companies = request.form.getlist("company[]") or request.form.getlist("company")
        shares = request.form.getlist("shares[]") or request.form.getlist("shares")
        avg_prices = request.form.getlist("avg_price[]") or request.form.getlist("avg_price")

        app.logger.info("edit_portfolio POST: companies=%s shares=%s avg_prices=%s", companies, shares, avg_prices)

        if not (len(companies) == len(shares) == len(avg_prices)):
            return render_template("edit_portfolio.html", error="Input list length mismatch.",
                                   portfolio=[{"symbol":c,"shares":s,"avg_price":p} for c,s,p in zip(companies,shares,avg_prices)],
                                   portfolio_date=portfolio_date)

        new_portfolio = []
        try:
            try:
                _build_instrument_map()
            except Exception:
                app.logger.info("Instrument map build skipped/failed during edit; continuing.")

            for idx, (c, s, p) in enumerate(zip(companies, shares, avg_prices), start=1):
                raw_input = (c or "").strip()
                if not raw_input:
                    raise ValueError(f"Empty company field at row {idx}")

                # attempt resolution but do not fail the entire update if resolution fails
                try:
                    sym, conf, token = auto_resolve(raw_input)
                except Exception as e:
                    app.logger.exception("auto_resolve raised for '%s' at row %d", raw_input, idx)
                    sym, conf, token = None, 0, None

                app.logger.info("auto_resolve '%s' -> %s (conf=%s)", raw_input, sym, conf)

                if not sym:
                    # fallback: use raw input as canonical symbol (uppercase) and log warning
                    app.logger.warning("auto_resolve could not resolve '%s' at row %d. Using raw input as symbol.", raw_input, idx)
                    sym = raw_input.upper()

                # numeric conversions
                try:
                    s_val = float(s)
                    p_val = float(p)
                except Exception:
                    raise ValueError(f"Invalid numeric value at row {idx} for '{raw_input}' (shares='{s}', avg_price='{p}')")

                if s_val < 0 or p_val < 0:
                    raise ValueError(f"Shares and avg_price must be non-negative at row {idx}")

                new_portfolio.append({
                    "symbol": sym,
                    "shares": s_val,
                    "avg_price": p_val,
                    "date": portfolio_date,
                    "user_id": ObjectId(current_user.id)
                })

            app.logger.info("Built new_portfolio for user %s: %s", current_user.id, new_portfolio)

            # Persist: replace existing
            mongo.db.portfolios.delete_many({"user_id": ObjectId(current_user.id)})
            if new_portfolio:
                res = mongo.db.portfolios.insert_many(new_portfolio)
                app.logger.info("Inserted portfolio ids: %s", getattr(res, "inserted_ids", None))
            else:
                app.logger.info("Inserted empty portfolio (user cleared portfolio)")

            # Immediately analyze the just-saved portfolio and render result
            symbols = [row["symbol"] for row in new_portfolio]
            shares_list = [row["shares"] for row in new_portfolio]
            avg_prices_list = [row["avg_price"] for row in new_portfolio]

            try:
                result = analyze_portfolio(symbols, shares_list, avg_prices_list)
            except Exception as e:
                app.logger.exception("Analysis failed immediately after update.")
                # even if analysis fails, go back to home with flash and the saved DB will be used next time
                flash(f"Portfolio saved but analysis failed: {e}", "warning")
                return redirect(url_for("home"))

            flash("Portfolio updated and analyzed.", "success")
            return render_template("result.html",
                                   full_name=current_user.full_name,
                                   email=current_user.email,
                                   phone=current_user.phone,
                                   goal=current_user.goal,
                                   portfolio_date=portfolio_date,
                                   result=result, **result)

        except Exception as e:
            app.logger.exception("Failed to update portfolio for user %s", current_user.id)
            portfolio_for_render = [{"symbol":c,"shares":s,"avg_price":p} for c,s,p in zip(companies,shares,avg_prices)]
            return render_template("edit_portfolio.html", error=f"Update failed: {e}", portfolio=portfolio_for_render, portfolio_date=portfolio_date)

    # GET: render edit form with DB values
    try:
        portfolio = list(mongo.db.portfolios.find({"user_id": ObjectId(current_user.id)}))
    except Exception:
        app.logger.exception("Failed to read portfolio for user %s", getattr(current_user, "id", None))
        portfolio = []
    portfolio_date = portfolio[0].get("date", "") if portfolio else ""
    return render_template("edit_portfolio.html", portfolio=portfolio, portfolio_date=portfolio_date)

# ---------------- Run app ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        _build_instrument_map()
    except Exception:
        app.logger.info("Instrument map build skipped at startup.")
    app.run(host="0.0.0.0", port=port)
