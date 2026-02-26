"""
EduShield AI — FastAPI Backend
==============================
Cloud-ready REST API to serve the trained Random Forest Classifier (model.pkl).
Supports:
  - Single-student subject risk prediction
  - Batch prediction for an entire class
  - Loophole Patch Engine (5 smart corrections)
  - Holiday Slide Tolerance
  - Health check & model metadata endpoints

Deploy with:
  uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import pickle
import numpy as np
from typing import List, Optional
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# 1.  Load the trained model
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    MODEL_LOADED = True
except FileNotFoundError:
    model = None
    MODEL_LOADED = False


# ──────────────────────────────────────────────
# 2.  Feature constants (must match training)
# ──────────────────────────────────────────────
FEATURES = [
    'Exam_1',
    'Exam_2',
    'Exam_3',
    'Exam_4',
    'Exam_5',
    'Exam_6',
    'Score_Momentum', 
    'Latest_Score',
    'Overall_Attendance_Pct', 
    'Total_Days_Absent', 
    'Max_Absent_Streak_Length'
]

# ──────────────────────────────────────────────
# 3.  Pydantic schemas
# ──────────────────────────────────────────────

class SubjectRecord(BaseModel):
    """One subject record for a single student."""
    student_id: str = Field(..., example="STU00042")
    subject_name: str = Field(..., example="Mathematics")
    exam_1: float = Field(0.0, ge=0, le=100)
    exam_2: float = Field(0.0, ge=0, le=100)
    exam_3: float = Field(0.0, ge=0, le=100)
    exam_4: float = Field(0.0, ge=0, le=100)
    exam_5: float = Field(0.0, ge=0, le=100)
    exam_6: float = Field(0.0, ge=0, le=100)
    latest_score: float = Field(..., ge=0, le=100, description="Score of the most recent exam taken")
    score_momentum: float = Field(..., description="Latest score minus average of previous scores")
    overall_attendance_pct: float = Field(..., ge=0, le=100)
    total_days_absent: int = Field(..., ge=0)
    max_absent_streak_length: int = Field(..., ge=0)
    days_enrolled: Optional[int] = Field(None, description="Days enrolled — triggers cold-start patch if < 45")
    historical_average: Optional[float] = Field(None, description="Historical average — for miracle-score patch")
    discipline_flags: Optional[int] = Field(0, description="Count of discipline issues")

    class Config:
        json_schema_extra = {
            "example": {
                "student_id": "STU00042",
                "subject_name": "Mathematics",
                "exam_1": 65.0,
                "exam_2": 68.0,
                "exam_3": 0.0,
                "exam_4": 0.0,
                "exam_5": 0.0,
                "exam_6": 0.0,
                "latest_score": 68.0,
                "score_momentum": 3.0,
                "overall_attendance_pct": 72.0,
                "total_days_absent": 14,
                "max_absent_streak_length": 3,
                "days_enrolled": 150,
                "historical_average": 66.0,
                "discipline_flags": 0,
            }
        }


class BatchRequest(BaseModel):
    records: List[SubjectRecord]
    apply_loophole_patches: bool = Field(True, description="Run the 5-patch smart correction engine")
    holiday_factor: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Holiday slide multiplier (0.5 = 50% more forgiving for post-holiday drops)",
    )


class PredictionResult(BaseModel):
    student_id: str
    subject_name: str
    at_risk: bool
    risk_probability: float  # 0.0 – 1.0
    risk_label: str          # SAFE / WARNING / CRITICAL RISK
    patches_applied: List[str]
    recommendation: str


class BatchResponse(BaseModel):
    model_version: str = "RandomForest-v1.0"
    model_accuracy: str = "98.40%"
    total_records: int
    at_risk_count: int
    safe_count: int
    results: List[PredictionResult]


# ──────────────────────────────────────────────
# 4.  Loophole Patch Engine (server-side)
# ──────────────────────────────────────────────

def apply_patches(record: SubjectRecord, class_avg_momentum: float, holiday_factor: float):
    """
    Runs EduShield's 5 smart loophole patches before inference.
    Returns (modified feature dict, list of patch notes).
    """
    patches = []
    features = {
        "Exam_1": record.exam_1,
        "Exam_2": record.exam_2,
        "Exam_3": record.exam_3,
        "Exam_4": record.exam_4,
        "Exam_5": record.exam_5,
        "Exam_6": record.exam_6,
        "Score_Momentum": record.score_momentum,
        "Latest_Score": record.latest_score,
        "Overall_Attendance_Pct": record.overall_attendance_pct,
        "Total_Days_Absent": record.total_days_absent,
        "Max_Absent_Streak_Length": record.max_absent_streak_length,
    }
    override_risk: Optional[float] = None   # Force a specific risk level (0–1)

    # PATCH 3: TRANSFER STUDENT COLD START
    if record.days_enrolled is not None and record.days_enrolled < 45:
        patches.append("PATCH 3: Cold Start — enrolled < 45 days, momentum ignored, baseline applied.")
        override_risk = 0.10
        return features, patches, override_risk

    # PATCH 1: STRICT TEACHER ANOMALY (class-wide drop)
    if class_avg_momentum < -20 and record.score_momentum < -20:
        patches.append(
            f"PATCH 1: Strict Teacher — class avg momentum {class_avg_momentum:.1f}%. "
            "Normalised individual momentum against class pain."
        )
        features["Score_Momentum"] = record.score_momentum - class_avg_momentum

    # Apply holiday slide tolerance to momentum
    if holiday_factor < 1.0 and features["Score_Momentum"] < -10:
        features["Score_Momentum"] = features["Score_Momentum"] * holiday_factor
        patches.append(
            f"Holiday Slide — momentum softened by {int((1 - holiday_factor) * 100)}% "
            "for post-holiday tolerance."
        )

    # PATCH 4: MIRACLE IMPROVEMENT (possible cheating)
    if (
        record.historical_average is not None
        and record.historical_average < 40
        and (record.latest_score - record.historical_average) > 50
    ):
        patches.append(
            "PATCH 4: Miracle Fix — score jumped > 50 points from a low baseline. "
            "Flagged for academic integrity review. Risk NOT reduced."
        )
        override_risk = 0.65

    # PATCH 5: QUIET FAIL (perfect behaviour, silently failing)
    if record.overall_attendance_pct >= 95 and (record.discipline_flags or 0) == 0:
        if features["Score_Momentum"] < -15 or record.latest_score < 40:
            patches.append(
                "PATCH 5: Quiet Fail — perfect behaviour but academically failing. "
                "System elevated risk alert."
            )
            override_risk = 0.80

    # PATCH 2: ABSOLUTE FAILURE FLOOR
    if record.latest_score < 33:
        patches.append("PATCH 2: Absolute Failure — Latest Score below pass mark (33%).")
        if override_risk is None or override_risk < 0.55:
            override_risk = 0.55

    return features, patches, override_risk


# ──────────────────────────────────────────────
# 5.  Inference helper
# ──────────────────────────────────────────────

def risk_label_and_reco(probability: float, subject: str) -> tuple[str, str]:
    if probability >= 0.75:
        return "CRITICAL RISK", f"Immediate teacher alert required. Student is failing {subject}."
    elif probability >= 0.40:
        return "WARNING", f"Monitor {subject} closely. Slight performance drop detected."
    else:
        return "SAFE", "No action required. Student is performing within expectations."


def predict_record(
    record: SubjectRecord,
    class_avg_momentum: float,
    apply_patches_flag: bool,
    holiday_factor: float,
) -> PredictionResult:
    patches: List[str] = []
    override_risk: Optional[float] = None

    if apply_patches_flag:
        feat_dict, patches, override_risk = apply_patches(record, class_avg_momentum, holiday_factor)
    else:
        feat_dict = {
            "Exam_1": record.exam_1,
            "Exam_2": record.exam_2,
            "Exam_3": record.exam_3,
            "Exam_4": record.exam_4,
            "Exam_5": record.exam_5,
            "Exam_6": record.exam_6,
            "Score_Momentum": record.score_momentum,
            "Latest_Score": record.latest_score,
            "Overall_Attendance_Pct": record.overall_attendance_pct,
            "Total_Days_Absent": record.total_days_absent,
            "Max_Absent_Streak_Length": record.max_absent_streak_length,
        }

    if override_risk is not None:
        probability = override_risk
        at_risk = probability >= 0.5
    else:
        X = np.array([[feat_dict[f] for f in FEATURES]])
        probability = float(model.predict_proba(X)[0][1])
        at_risk = bool(model.predict(X)[0] == 1)

    label, reco = risk_label_and_reco(probability, record.subject_name)

    return PredictionResult(
        student_id=record.student_id,
        subject_name=record.subject_name,
        at_risk=at_risk,
        risk_probability=round(probability, 4),
        risk_label=label,
        patches_applied=patches,
        recommendation=reco,
    )


# ──────────────────────────────────────────────
# 6.  FastAPI app
# ──────────────────────────────────────────────

app = FastAPI(
    title="EduShield AI — Risk Prediction API",
    description=(
        "Cloud-hosted REST API for the EduShield student risk prediction system. "
        "Powered by a Random Forest Classifier trained on Indian school data (98.40% accuracy). "
        "Includes the 5-patch Loophole Correction Engine and Holiday Slide Tolerance."
    ),
    version="1.0.0",
    contact={
        "name": "EduShield AI Team",
        "email": "team@edushield.ai",
    },
    license_info={
        "name": "Private — EduShield AI",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# 7.  Endpoints
# ──────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    """Welcome message & API status."""
    return {
        "project": "EduShield AI",
        "status": "online",
        "model_loaded": MODEL_LOADED,
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe — used by cloud platforms (Render, Railway, GCP, AWS) to check uptime."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="model.pkl not found — service unavailable.")
    return {"status": "healthy", "model": "loaded", "timestamp": str(date.today())}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns metadata about the deployed ML model."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": "RandomForestClassifier",
        "n_estimators": 100,
        "max_depth": 5,
        "training_records": 4484,
        "test_records": 1122,
        "overall_accuracy": "98.40%",
        "at_risk_precision": "99%",
        "at_risk_recall": "82%",
        "features": FEATURES,
        "feature_importance": FEATURE_IMPORTANCE,
        "target": "Subject_At_Risk",
        "classes": {"0": "Safe", "1": "At Risk"},
    }


@app.post("/predict", response_model=PredictionResult, tags=["Inference"])
def predict_single(
    record: SubjectRecord,
    apply_loophole_patches: bool = True,
    holiday_factor: float = 1.0,
):
    """
    Predict risk for a **single student-subject record**.

    - Optionally applies the 5-patch Loophole Correction Engine
    - Supports Holiday Slide Tolerance (set `holiday_factor=0.5` after long holidays)
    - Returns risk probability, label, and recommendation
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded. Check that model.pkl exists.")

    try:
        result = predict_record(
            record=record,
            class_avg_momentum=record.score_momentum,  # Single record — no class avg available
            apply_patches_flag=apply_loophole_patches,
            holiday_factor=holiday_factor,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchResponse, tags=["Inference"])
def predict_batch(request: BatchRequest):
    """
    Predict risk for an **entire class of student-subject records**.

    - Automatically computes class-average momentum for the Strict Teacher patch
    - Applies all 5 loophole patches per record (if enabled)
    - Returns summary counts + per-record results
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded. Check that model.pkl exists.")

    if not request.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    # Compute class average momentum for PATCH 1 (Strict Teacher)
    momentums = [r.score_momentum for r in request.records]
    class_avg_momentum = float(np.mean(momentums))

    results: List[PredictionResult] = []
    try:
        for record in request.records:
            res = predict_record(
                record=record,
                class_avg_momentum=class_avg_momentum,
                apply_patches_flag=request.apply_loophole_patches,
                holiday_factor=request.holiday_factor,
            )
            results.append(res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")

    at_risk_count = sum(1 for r in results if r.at_risk)

    return BatchResponse(
        total_records=len(results),
        at_risk_count=at_risk_count,
        safe_count=len(results) - at_risk_count,
        results=results,
    )


@app.get("/predict/student/{student_id}/summary", tags=["Inference"])
def student_summary(student_id: str):
    """
    Placeholder: Returns a sample summary for a given student ID.
    In production, wire this to a database to fetch live records and run inference.
    """
    return {
        "student_id": student_id,
        "message": (
            f"Connect this endpoint to your database to fetch live records for {student_id} "
            "and run /predict/batch automatically."
        ),
        "suggested_flow": [
            "1. Query your DB for all subject records for this student",
            "2. POST to /predict/batch with those records",
            "3. Store results back into the dashboard DB",
        ],
    }


# ──────────────────────────────────────────────
# 8.  Entrypoint (local dev only)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
