from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core_db.db import engine, Base
from api.endpoints import users, scans
import os

# Create upload directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# Generate database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="DermaScan AI Engine")

# --- THE CORS FIX ---
# This stops the browser from blocking your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Standard Vite port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, DELETE, PUT, etc.
    allow_headers=["*"],
)

# Register our expanded routers
app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(scans.router, prefix="/api/scans", tags=["Scans"])

@app.get("/")
def health_check():
    return {"status": "DermaScan AI Backend is operational."}