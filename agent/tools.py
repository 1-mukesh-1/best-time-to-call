import io
import json
import os
import joblib
import numpy as np
import pandas as pd
import boto3
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "model"

S3_BUCKET = os.getenv("S3_BUCKET_NAME")
USE_S3 = bool(S3_BUCKET)

CAT_FEATURES = ["Job", "Marital", "Education", "Communication", "LastContactMonth", "Outcome"]

LEAD_FIELD_MAP = {
    "Job": "job",
    "Marital": "marital",
    "Education": "education",
    "Communication": "communication",
    "LastContactMonth": "last_contact_month",
    "Outcome": "outcome",
    "Age": "age",
    "Balance": "balance",
    "HHInsurance": "hh_insurance",
    "CarLoan": "car_loan",
    "Default": "default",
    "LastContactDay": "last_contact_day",
    "NoOfContacts": "no_of_contacts",
    "DaysPassed": "days_passed",
    "PrevAttempts": "prev_attempts",
}


def _s3_client():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"))


def _read_json_from_s3(key: str) -> object:
    s3 = _s3_client()
    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def _read_joblib_from_s3(key: str) -> object:
    s3 = _s3_client()
    response = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return joblib.load(io.BytesIO(response["Body"].read()))


def _load_leads() -> dict:
    if USE_S3:
        leads = _read_json_from_s3("data/leads.json")
    else:
        with open(DATA_DIR / "leads.json") as f:
            leads = json.load(f)
    return {lead["lead_id"]: lead for lead in leads}


def _load_call_logs() -> dict:
    if USE_S3:
        return _read_json_from_s3("data/call_logs.json")
    else:
        with open(DATA_DIR / "call_logs.json") as f:
            return json.load(f)


def _load_model_artifacts():
    if USE_S3:
        model = _read_joblib_from_s3("model/xgb_model.joblib")
        label_encoders = _read_joblib_from_s3("model/label_encoders.joblib")
        feature_cols = _read_joblib_from_s3("model/feature_cols.joblib")
    else:
        model = joblib.load(MODEL_DIR / "xgb_model.joblib")
        label_encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
        feature_cols = joblib.load(MODEL_DIR / "feature_cols.joblib")
    return model, label_encoders, feature_cols


def _summarize_transcript(transcript: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize this sales call transcript in 1-2 sentences. "
                    "Focus on: lead's intent, any objections raised, specific requests, "
                    "or reason for the call outcome. Be concise and factual."
                ),
            },
            {"role": "user", "content": transcript},
        ],
        max_tokens=120,
    )
    return response.choices[0].message.content


@tool
def get_lead_info(lead_id: str) -> dict:
    """Fetch lead profile from CRM by lead ID."""
    leads = _load_leads()
    if lead_id not in leads:
        return {"error": f"Lead {lead_id} not found"}
    return leads[lead_id]


@tool
def get_call_logs(lead_id: str) -> dict:
    """
    Fetch call history for a lead. Returns structured outcome stats
    and per-call summaries generated from transcripts.
    """
    logs = _load_call_logs()
    calls = logs.get(lead_id, [])

    if not calls:
        return {
            "total_attempts": 0,
            "outcomes": {},
            "recent_pattern": [],
            "days_since_last_contact": None,
            "per_call_summaries": [],
        }

    outcomes: dict[str, int] = {}
    for call in calls:
        o = call["outcome"]
        outcomes[o] = outcomes.get(o, 0) + 1

    recent_pattern = [c["outcome"] for c in calls[-3:]]
    last_ts = datetime.fromisoformat(calls[-1]["timestamp"])
    days_since = (datetime.now() - last_ts).days

    per_call_summaries = []
    for call in calls:
        summary = _summarize_transcript(call["transcript"])
        per_call_summaries.append(
            {
                "date": call["timestamp"][:10],
                "outcome": call["outcome"],
                "summary": summary,
            }
        )

    return {
        "total_attempts": len(calls),
        "outcomes": outcomes,
        "recent_pattern": recent_pattern,
        "days_since_last_contact": days_since,
        "per_call_summaries": per_call_summaries,
    }


@tool
def predict_best_time(lead_id: str) -> dict:
    """
    Use the XGBoost model to predict the best hour to call a lead.
    Sweeps hours 9-17 and returns the hour with highest conversion probability.
    """
    leads = _load_leads()
    if lead_id not in leads:
        return {"error": f"Lead {lead_id} not found"}

    lead = leads[lead_id]
    model, label_encoders, feature_cols = _load_model_artifacts()

    hours = list(range(9, 18))
    rows = []
    for hour in hours:
        row = {}
        for feat in feature_cols:
            if feat == "CallHour":
                row[feat] = hour
            elif feat.endswith("_encoded"):
                base = feat.replace("_encoded", "")
                raw_val = str(lead.get(LEAD_FIELD_MAP[base], "Unknown"))
                le = label_encoders[base]
                row[feat] = int(le.transform([raw_val])[0]) if raw_val in le.classes_ else -1
            elif feat in LEAD_FIELD_MAP:
                row[feat] = lead[LEAD_FIELD_MAP[feat]]
        rows.append(row)

    df = pd.DataFrame(rows, columns=feature_cols)
    probs = model.predict_proba(df)[:, 1]
    best_idx = int(np.argmax(probs))

    return {
        "best_hour": hours[best_idx],
        "best_hour_label": f"{hours[best_idx]}:00",
        "conversion_probability": round(float(probs[best_idx]), 3),
        "all_hours": {f"{h}:00": round(float(p), 3) for h, p in zip(hours, probs)},
    }


@tool
def schedule_call(lead_id: str, scheduled_time: str) -> dict:
    """
    Schedule the next call for a lead.
    scheduled_time format: 'YYYY-MM-DD HH:MM'
    """
    return {
        "status": "scheduled",
        "lead_id": lead_id,
        "scheduled_time": scheduled_time,
        "confirmation": f"Call for lead {lead_id} booked at {scheduled_time}",
    }


@tool
def escalate_to_human(lead_id: str, reason: str) -> dict:
    """Escalate a lead to a human sales representative."""
    return {
        "status": "escalated",
        "lead_id": lead_id,
        "reason": reason,
        "message": f"Lead {lead_id} escalated to human rep — {reason}",
    }


@tool
def disqualify_lead(lead_id: str, reason: str) -> dict:
    """Remove a lead from the active pipeline."""
    return {
        "status": "disqualified",
        "lead_id": lead_id,
        "reason": reason,
        "message": f"Lead {lead_id} disqualified — {reason}",
    }
