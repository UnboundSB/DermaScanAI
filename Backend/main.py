import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Backend.api.endpoints import users, predict

app = FastAPI(title="DermaScanAI Enterprise API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the separate endpoints
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(predict.router, prefix="/api/ml", tags=["Clinical Engine"])

@app.get("/")
def health_check():
    return {"status": "Database & AI Engines Online."}