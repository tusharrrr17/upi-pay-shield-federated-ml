from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import joblib
from datetime import datetime
import os

app = FastAPI(title="UPI Pay Shield | 10-Feature Federated AI", version="3.0.0")

os.makedirs("templates", exist_ok=True)
templates = Jinja2Templates(directory="templates")

# Load Existing Models from GitHub models/ folder
scaler = joblib.load("models/robust_scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

model = xgb.Booster()
model.load_model("models/global_xgb_model.json")

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

class FuturisticUPIRequest(BaseModel):
    receiver_upi: str
    amount: float
    payment_mode: str
    on_active_call: bool
    impossible_travel: bool
    guardian_lock: bool
    penny_drop_seen: bool
    midnight_mode: bool
    typing_hesitation: float

def run_10_feature_shield(req: FuturisticUPIRequest):
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

    raw_dict = {
        "step": current_hour,
        "amount": req.amount,
        "oldbalanceOrg": old_bal_org,
        "newbalanceOrig": new_bal_org,
        "oldbalanceDest": old_bal_dest,
        "newbalanceDest": new_bal_dest,
        f"type_{tx_type}": 1
    }
    input_df = pd.DataFrame([raw_dict])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]

    scaled_features = scaler.transform(input_df)
    base_xgb_prob = float(model.predict(xgb.DMatrix(scaled_features))[0])

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
async def check_upi(req: FuturisticUPIRequest):
    try:
        return JSONResponse(content=run_10_feature_shield(req))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
