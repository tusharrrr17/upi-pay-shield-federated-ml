%%writefile app.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime
import os

app = FastAPI(title="UPI Pay Shield | Federated XGBoost Engine", version="4.0.0")

os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Load Leakage-Free Preprocessing Artifacts & Global Federated XGBoost Model
scaler = joblib.load("models/standard_scaler.pkl")
ohe = joblib.load("models/onehot_encoder.pkl")
selector = joblib.load("models/feature_selector.pkl")

bst_model = xgb.Booster()
bst_model.load_model("models/global_xgb_model.json")

USER_WALLET_BALANCE = 50000.0
KNOWN_RECEIVERS = {
    "zomato@icici": {"old_bal": 1200000.0, "reports": 0},
    "mom@oksbi": {"old_bal": 45000.0, "reports": 0},
    "rahul.sharma@okaxis": {"old_bal": 18000.0, "reports": 0},
}
REPORTED_VPAS = {
    "kyc.update99@ybl": 14,
    "refund.helpdesk@axl": 9,
}

class UPIRequest(BaseModel):
    receiver_upi: str
    amount: float
    payment_mode: str
    on_active_call: bool
    impossible_travel: bool
    guardian_lock: bool
    penny_drop_seen: bool
    midnight_mode: bool
    typing_hesitation: float

def score_upi_transaction(req: UPIRequest):
    upi_clean = req.receiver_upi.lower().strip()
    current_hour = 2 if req.midnight_mode else (datetime.now().hour + 1)
    old_bal_org = USER_WALLET_BALANCE
    new_bal_org = max(0.0, old_bal_org - req.amount)

    receiver_info = KNOWN_RECEIVERS.get(upi_clean)
    if receiver_info:
        old_bal_dest = receiver_info["old_bal"]
        new_bal_dest = old_bal_dest + req.amount
    else:
        old_bal_dest = 0.0
        new_bal_dest = 0.0 if req.amount >= 15000 else req.amount

    tx_type = "PAYMENT" if req.payment_mode == "MERCHANT_QR" else "TRANSFER"

    # 1. Exact Same Preprocessing Pipeline as Training Notebook
    raw_df = pd.DataFrame([{
        "step": current_hour,
        "type": tx_type,
        "amount": req.amount,
        "oldbalanceOrg": old_bal_org,
        "newbalanceOrig": new_bal_org,
        "oldbalanceDest": old_bal_dest,
        "newbalanceDest": new_bal_dest
    }])

    raw_df["errorBalanceOrig"] = raw_df["oldbalanceOrg"] - raw_df["amount"] - raw_df["newbalanceOrig"]
    raw_df["errorBalanceDest"] = raw_df["oldbalanceDest"] + raw_df["amount"] - raw_df["newbalanceDest"]

    encoded_type = pd.DataFrame(
        ohe.transform(raw_df[["type"]]),
        columns=ohe.get_feature_names_out(["type"]),
        index=raw_df.index
    )
    proc_df = pd.concat([raw_df.drop(columns=["type"]), encoded_type], axis=1)

    log_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    for col in log_cols:
        proc_df[col] = np.log1p(proc_df[col])

    scale_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest"]
    proc_df[scale_cols] = scaler.transform(proc_df[scale_cols])

    selected_input = selector.transform(proc_df)
    base_xgb_prob = float(bst_model.predict(xgb.DMatrix(selected_input))[0])

    # 2. Risk Attribution & Real-Time Defense Layers
    shap_factors = []
    xgb_contrib = round(base_xgb_prob * 45.0, 1)
    if xgb_contrib > 2.0:
        shap_factors.append({"feature": "FedXGBoost Core Pattern", "impact": xgb_contrib})

    bonus_risk = 0.0
    reasons = []

    if req.on_active_call and req.amount >= 2000:
        bonus_risk += 25.0
        shap_factors.append({"feature": "Active Call / Screen-Share", "impact": 25.0})
        reasons.append("Active phone call or screen-sharing detected during transfer")

    if req.payment_mode == "COLLECT_REQUEST":
        bonus_risk += 22.0
        shap_factors.append({"feature": "Collect Request Trap", "impact": 22.0})
        reasons.append("COLLECT REQUEST: Entering UPI PIN will DEDUCT money from your account")

    fed_reports = REPORTED_VPAS.get(upi_clean, 0 if receiver_info else 2)
    if fed_reports >= 5:
        bonus_risk += 30.0
        shap_factors.append({"feature": "Federated Bank Blacklist", "impact": 30.0})
        reasons.append(f"Federated Alert: {fed_reports} cross-bank fraud reports on this VPA in 2h")
    elif not receiver_info:
        bonus_risk += 8.0
        shap_factors.append({"feature": "Unverified New VPA", "impact": 8.0})
        reasons.append("First-time unverified receiver across HDFC/SBI/ICICI nodes")

    if req.typing_hesitation >= 0.5:
        bio_impact = round(req.typing_hesitation * 15.0, 1)
        bonus_risk += bio_impact
        shap_factors.append({"feature": "Typing Coercion / Panic", "impact": bio_impact})
        reasons.append("Abnormal keystroke hesitation / panic typing rhythm detected")

    if req.impossible_travel:
        bonus_risk += 28.0
        shap_factors.append({"feature": "Impossible Travel (Geo)", "impact": 28.0})
        reasons.append("Location jump anomaly: Delhi to Mumbai within 10 minutes")

    if req.penny_drop_seen:
        bonus_risk += 24.0
        shap_factors.append({"feature": "Penny-Drop Mule Pattern", "impact": 24.0})
        reasons.append("Rs. 1 test deposit detected prior to high-value sweep")

    dynamic_limit = 5000.0 if (req.midnight_mode or req.on_active_call) else 50000.0
    if req.amount > dynamic_limit:
        bonus_risk += 15.0
        shap_factors.append({"feature": "Dynamic AI Limit Breach", "impact": 15.0})
        reasons.append(f"Exceeds current AI-adjusted safe limit of Rs. {int(dynamic_limit):,}")

    guardian_required = req.guardian_lock and (req.amount > 10000) and (not receiver_info)
    final_score = min(99.5, round(xgb_contrib + bonus_risk, 1))
    shap_factors.sort(key=lambda x: x["impact"], reverse=True)

    if final_score >= 70.0:
        status = "BLOCKED"
        banner = "CRITICAL THREAT: Payment Blocked by Federated AI Shield."
    elif guardian_required:
        status = "GUARDIAN_HOLD"
        banner = "GUARDIAN LOCK: Waiting for Family Trusted Contact Approval."
    elif final_score >= 35.0:
        status = "WARNING"
        banner = "SUSPICIOUS: 10-Second Cooling-Off Timer Activated."
    else:
        status = "SAFE"
        banner = "VERIFIED SAFE: Proceed to Enter 4-Digit UPI PIN."
        if not reasons:
            reasons.append("Trusted VPA & normal behavioral telemetry")

    return {
        "risk_score": final_score,
        "status": status,
        "banner": banner,
        "reasons": reasons,
        "shap_factors": shap_factors[:5],
        "dynamic_limit": int(dynamic_limit),
        "federated_status": f"Synced (3 Bank Nodes) | VPA Reports: {fed_reports}",
        "guardian_required": guardian_required
    }

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/check_upi")
async def check_upi(req: UPIRequest):
    try:
        return JSONResponse(content=score_upi_transaction(req))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
