import cv2
import numpy as np
import json
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from sqlalchemy.orm import Session

# --- The Ghost-Proof Import ---
from core_db.db import get_db, PredictionRecord
from inference.replyer import DiagnosticReplyer
from inference.predictor import ClinicalPredictor

router = APIRouter()
ai_engine = ClinicalPredictor()
replyer = DiagnosticReplyer()

@router.post("/analyze")
async def analyze_skin(
    user_id: int = Form(...), 
    scan_type: str = Form(...), 
    file: UploadFile = File(...), 
    db: Session = Depends(get_db)
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    raw_report = ai_engine.predict(img_bgr)
    if raw_report.get("status") in ["rejected", "error"]:
        raise HTTPException(status_code=422, detail=raw_report)

    all_confs = raw_report["all_confidences"]
    max_score = max(all_confs.values())
    
    bulk_symptoms = {symp: score for symp, score in all_confs.items() if score >= (max_score - 15.0)}
    raw_report["margin_results"] = bulk_symptoms 

    if scan_type == "normal":
        final_report = replyer.finalize_and_route_analysis(raw_report, session_id=str(user_id))
    
    elif scan_type == "10_day":
        last_record = db.query(PredictionRecord).filter(
            PredictionRecord.user_id == user_id
        ).order_by(PredictionRecord.timestamp.desc()).first()

        if not last_record:
            raise HTTPException(status_code=400, detail="No previous scan found. Take a normal scan first to set a baseline.")

        last_report = json.loads(last_record.report_data)
        last_confs = last_report.get("all_confidences", {})

        rates = {}
        primary_symp = max(bulk_symptoms, key=bulk_symptoms.get)
        primary_diff = bulk_symptoms[primary_symp] - last_confs.get(primary_symp, 0)

        for symp in bulk_symptoms.keys():
            diff = bulk_symptoms[symp] - last_confs.get(symp, 0)
            rates[symp] = round(diff, 2)

        if primary_diff <= -5.0:
            status = "improved"
        elif primary_diff >= 5.0:
            status = "worsened"
        else:
            status = "plateau"

        final_report = replyer.finalize_and_route_progress(list(bulk_symptoms.keys()), status, session_id=str(user_id))
        
        final_report["growth_decay_rates"] = rates
        final_report["current_bulk_symptoms"] = bulk_symptoms

    else:
        raise HTTPException(status_code=400, detail="Invalid scan_type. Use 'normal' or '10_day'.")

    new_record = PredictionRecord(
        user_id=user_id,
        scan_type=scan_type,
        report_data=json.dumps(final_report)
    )
    db.add(new_record)
    db.commit()

    return final_report