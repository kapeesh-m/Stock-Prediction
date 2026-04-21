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
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
app.permanent_session_lifetime = timedelta(hours=1)

# -------------------- DATABASE CONFIG --------------------
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL', 'sqlite:///users.db'
)

# Fix for PostgreSQL URL issue on Render
if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace(
        "postgres://", "postgresql://", 1
    )

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
        password_raw = request.form["password"]

        if not username or not email or not password_raw:
            flash("All fields are required.", "danger")
            return redirect(url_for("signup"))

        if len(password_raw) < 8:
            flash("Password must be at least 8 characters.", "danger")
            return redirect(url_for("signup"))

        existing_user = User.query.filter(
            (User.username == username) | (User.email == email)
        ).first()

        if existing_user:
            flash("Username or Email already exists!", "danger")
            return redirect(url_for("signup"))

        hashed_password = generate_password_hash(password_raw)

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash("Account created successfully! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("signup.html")

# -------------------- LOGIN --------------------
login_attempts = {}

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        ip = request.remote_addr
        attempts = login_attempts.get(ip, 0)

        if attempts >= 5:
            flash("Too many failed attempts. Try later.", "danger")
            return render_template("login.html")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_attempts.pop(ip, None)
            session.permanent = True
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("predict"))
        else:
            login_attempts[ip] = attempts + 1
            flash("Invalid username or password!", "danger")

    return render_template("login.html")

# -------------------- LOGOUT --------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# -------------------- PREDICT --------------------
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

    if request.method == "POST":
        stock_input = request.form["stock"].upper().strip()

        if not re.match(r'^[A-Z0-9.\-]{1,10}$', stock_input):
            flash("Invalid ticker symbol.", "danger")
            return redirect(url_for("predict"))

        try:
            stock = yf.Ticker(stock_input)
            data = stock.history(period="6mo")

            if not data.empty:

                today_price = round(data["Close"].iloc[-1], 2)

                data["MA10"] = data["Close"].rolling(window=10).mean()
                prediction = round(data["MA10"].iloc[-1], 2)

                clean_symbol = stock_input.replace(".NS", "").lower()
                logo_url = f"https://logo.clearbit.com/{clean_symbol}.com"

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name="Actual Price"
                ))

                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["MA10"],
                    mode="lines",
                    name="10-Day Moving Average"
                ))

                fig.update_layout(
                    template="plotly_dark",
                    title=f"{stock_input} Price Analysis"
                )

                graph_html = pio.to_html(fig, full_html=False)

                news_feed = feedparser.parse(
                    f"https://news.google.com/rss/search?q={stock_input}+stock"
                )

                news_list = []
                for entry in news_feed.entries[:5]:
                    news_list.append({
                        "title": entry.title,
                        "link": entry.link
                    })

            else:
                flash("No data found.", "warning")

        except Exception as e:
            print(e)
            flash("Error fetching data.", "danger")

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
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=debug_mode)