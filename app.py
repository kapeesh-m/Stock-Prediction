from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import yfinance as yf
import plotly.graph_objs as go
import plotly.io as pio
import feedparser
import os
import re
from datetime import timedelta

app = Flask(__name__)

# -------------------- SECURITY CONFIG --------------------
# FIX 1: Use environment variable for secret key, never hardcode
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)

# FIX 2: Session expiry — auto-logout after 1 hour of inactivity
app.permanent_session_lifetime = timedelta(hours=1)

# -------------------- DATABASE CONFIG --------------------
# FIX 3: Support production DB via environment variable (fallback to SQLite for dev)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# -------------------- USER MODEL --------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# -------------------- HOME --------------------
@app.route("/")
def home():
    return render_template("home.html")

# -------------------- CONTACT --------------------
@app.route("/contact")
def contact():
    return render_template("contact.html")

# -------------------- SIGNUP --------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = generate_password_hash(request.form["password"])

        # FIX 4: Basic input validation for signup
        if not username or not email or not request.form["password"]:
            flash("All fields are required.", "danger")
            return redirect(url_for("signup"))

        if len(request.form["password"]) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return redirect(url_for("signup"))

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user:
            flash("Username or Email already exists!", "danger")
            return redirect(url_for("signup"))

        new_user = User(username=username, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# -------------------- LOGIN --------------------
# FIX 5: Track failed login attempts to prevent brute force
login_attempts = {}

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        # Simple brute-force protection (in-memory, resets on server restart)
        # For production, use flask-limiter with Redis instead
        ip = request.remote_addr
        attempts = login_attempts.get(ip, 0)
        if attempts >= 5:
            flash("Too many failed attempts. Please try again later.", "danger")
            return render_template("login.html")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            # Reset failed attempts on success
            login_attempts.pop(ip, None)

            session.permanent = True   # Respect the 1-hour lifetime set above
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("predict"))
        else:
            # Increment failed attempts
            login_attempts[ip] = attempts + 1
            flash("Invalid username or password!", "danger")

    return render_template("login.html")

# -------------------- LOGOUT --------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# -------------------- PREDICT PAGE --------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if "user_id" not in session:
        flash("Please login first!", "warning")
        return redirect(url_for("login"))

    prediction = None
    today_price = None
    stock_input = None
    logo_url = None
    graph_html = None
    news_list = None
    error_message = None

    if request.method == "POST":
        stock_input = request.form["stock"].upper().strip()

        # FIX 6: Validate ticker symbol before passing to any external service
        if not re.match(r'^[A-Z0-9.\-]{1,10}$', stock_input):
            flash("Invalid ticker symbol. Use only letters, numbers, dots, or hyphens (max 10 chars).", "danger")
            return redirect(url_for("predict"))

        try:
            stock = yf.Ticker(stock_input)
            data = stock.history(period="6mo")

            if not data.empty:

                # ---------------- ACTUAL PRICE ----------------
                today_price = round(data["Close"].iloc[-1], 2)

                # ---------------- 10-DAY MOVING AVERAGE ----------------
                # FIX 7: Renamed from "Predicted" to accurately reflect it's a moving average
                data["MA10"] = data["Close"].rolling(window=10).mean()
                prediction = round(data["MA10"].iloc[-1], 2)

                # ---------------- COMPANY LOGO ----------------
                clean_symbol = stock_input.replace(".NS", "").replace(".BSE", "").lower()
                logo_url = f"https://logo.clearbit.com/{clean_symbol}.com"

                # ---------------- PLOTLY GRAPH ----------------
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name="Actual Price",
                    line=dict(color="cyan", width=2)
                ))

                # FIX 7 (continued): Label accurately in graph too
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["MA10"],
                    mode="lines",
                    name="10-Day Moving Average",
                    line=dict(color="orange", dash="dash")
                ))

                fig.update_layout(
                    template="plotly_dark",
                    title=f"{stock_input} - Actual Price vs 10-Day Moving Average",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    legend_title="Legend"
                )

                graph_html = pio.to_html(fig, full_html=False)

                # ---------------- NEWS ----------------
                news_feed = feedparser.parse(
                    f"https://news.google.com/rss/search?q={stock_input}+stock&hl=en-IN&gl=IN&ceid=IN:en"
                )

                news_list = []
                for entry in news_feed.entries[:5]:
                    news_list.append({
                        "title": entry.title,
                        "link": entry.link,
                        "published": entry.published
                    })

            else:
                # FIX 8: Handle empty data gracefully
                flash(f"No data found for '{stock_input}'. Please check the ticker symbol.", "warning")

        except Exception as e:
            # FIX 8: Show user-facing error instead of silently swallowing it
            print(f"[ERROR] Stock fetch failed for {stock_input}: {e}")
            flash(f"Could not fetch data for '{stock_input}'. Please try again later.", "danger")

    return render_template(
        "predict.html",
        prediction=prediction,
        today_price=today_price,
        stock_input=stock_input,
        logo_url=logo_url,
        graph_html=graph_html,
        news_list=news_list
    )

# -------------------- RUN APP --------------------
if __name__ == "__main__":
    # FIX 9: Debug mode controlled by environment variable, defaults to OFF
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode)