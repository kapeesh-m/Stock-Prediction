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

# ✅ ML IMPORTS
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# -------------------- SECURITY --------------------
app.secret_key = os.environ.get("SECRET_KEY") or os.urandom(24)
app.permanent_session_lifetime = timedelta(hours=1)

# -------------------- DATABASE --------------------
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
    'DATABASE_URL', 'sqlite:///users.db'
)

if app.config['SQLALCHEMY_DATABASE_URI'].startswith("postgres://"):
    app.config['SQLALCHEMY_DATABASE_URI'] = app.config['SQLALCHEMY_DATABASE_URI'].replace(
        "postgres://", "postgresql://", 1
    )

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# -------------------- LOAD ML MODEL --------------------
model = load_model("best_bilstm_model.h5")
scaler = MinMaxScaler(feature_range=(0, 1))

# -------------------- USER MODEL --------------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("home.html")

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

        flash("Account created successfully!", "success")
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
            flash("Too many attempts. Try later.", "danger")
            return render_template("login.html")

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_attempts.pop(ip, None)
            session["user_id"] = user.id
            session["username"] = user.username
            return redirect(url_for("predict"))
        else:
            login_attempts[ip] = attempts + 1
            flash("Invalid credentials!", "danger")

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
        flash("Login first!", "warning")
        return redirect(url_for("login"))

    prediction = None
    today_price = None
    stock_input = None
    graph_html = None
    news_list = None

    if request.method == "POST":
        stock_input = request.form["stock"].upper().strip()

        if not re.match(r'^[A-Z0-9.\-]{1,10}$', stock_input):
            flash("Invalid ticker.", "danger")
            return redirect(url_for("predict"))

        try:
            stock = yf.Ticker(stock_input)
            data = stock.history(period="6mo")

            if not data.empty:

                today_price = round(data["Close"].iloc[-1], 2)

                # ✅ ML PREDICTION
                close_prices = data["Close"].values.reshape(-1, 1)
                scaled_data = scaler.fit_transform(close_prices)

                last_60_days = scaled_data[-60:]
                X_test = np.array([last_60_days])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

                predicted_price = model.predict(X_test)
                predicted_price = scaler.inverse_transform(predicted_price)

                prediction = round(predicted_price[0][0], 2)

                # ✅ GRAPH
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data["Close"],
                    mode="lines",
                    name="Actual Price"
                ))

                fig.add_trace(go.Scatter(
                    x=[data.index[-1]],
                    y=[prediction],
                    mode="markers",
                    name="Predicted Price"
                ))

                fig.update_layout(
                    template="plotly_dark",
                    title=f"{stock_input} Prediction"
                )

                graph_html = pio.to_html(fig, full_html=False)

                # ✅ NEWS
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
        graph_html=graph_html,
        news_list=news_list
    )

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))