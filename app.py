from flask import Flask, request, jsonify
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from xgboost import XGBClassifier

app = Flask(__name__)

# 🔥 حط بياناتك هنا
TOKEN = "8504301988:AAGmj6aY_oI4c-25vOi8w3BFPpGT5x7UAAA"
CHAT_ID = "1715919167"

def send(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={"chat_id": CHAT_ID, "text": msg}
        )
    except:
        pass

def get_data():
    data = yf.download("^GSPC", interval="5m", period="5d", progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns={
        "Close": "close",
        "High": "high",
        "Low": "low",
        "Open": "open",
        "Volume": "volume"
    })

    return data.dropna()

def train():
    data = get_data()

    data["ema"] = ta.trend.ema_indicator(data["close"], 50)
    data["rsi"] = ta.momentum.rsi(data["close"], 14)
    data["atr"] = ta.volatility.average_true_range(
        data["high"], data["low"], data["close"], 14
    )

    data["trend"] = (data["close"] > data["ema"]).astype(int)
    data["momentum"] = data["close"] - data["close"].shift(3)
    data["volatility"] = data["high"] - data["low"]

    data["future"] = data["close"].shift(-5)
    data["target"] = (data["future"] > data["close"]).astype(int)

    data = data.dropna()

    X = data[["trend","momentum","volatility","rsi","atr"]]
    y = data["target"]

    model = XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss")
    model.fit(X, y)

    return model, data

model, data_ai = train()

@app.route("/webhook", methods=["POST"])
def webhook():
    global model, data_ai

    try:
        model, data_ai = train()

        data = request.json
        decision = data.get("action")
        price = float(data.get("price", 0))

        last = data_ai.iloc[-1]
        X = last[["trend","momentum","volatility","rsi","atr"]].values.reshape(1,-1)

        conf = model.predict_proba(X)[0][1]

        if conf < 0.7:
            return jsonify({"status":"weak"})

        atr = float(data_ai["atr"].iloc[-1])
        risk = atr * 1.2

        if decision == "BUY":
            sl = price - risk
            tp = price + risk*2
        else:
            sl = price + risk
            tp = price - risk*2

        msg = f"""
🚀 SIGNAL

{decision}
Price: {price}

SL: {round(sl,2)}
TP: {round(tp,2)}

AI: {round(conf,2)}
"""
        send(msg)

        return jsonify({"status":"ok"})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def home():
    return "BOT WORKING 🚀"

app.run(host="0.0.0.0", port=10000)
