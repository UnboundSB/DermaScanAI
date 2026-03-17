from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from core_db.db import get_db, PredictionRecord

router = APIRouter()

@router.get("/image/{record_id}")
def get_scan_image(record_id: int, db: Session = Depends(get_db)):
    """
    Retrieves the binary image BLOB from the database and serves it as a JPEG.
    """
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
        
    if not record.image_data:
        raise HTTPException(status_code=404, detail="No image data stored for this record")
        
    # Return the raw bytes natively as an image
    return Response(content=record.image_data, media_type="image/jpeg")


@router.delete("/{record_id}")
def delete_scan(record_id: int, db: Session = Depends(get_db)):
    """
    Deletes a specific scan record and its associated image BLOB from the database.
    """
    record = db.query(PredictionRecord).filter(PredictionRecord.id == record_id).first()
    
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
        
    db.delete(record)
    db.commit()
    
    return {"message": "Scan record successfully deleted", "id": record_id}