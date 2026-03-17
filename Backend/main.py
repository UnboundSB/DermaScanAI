from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles # ADD THIS IMPORT
from core_db.db import engine, Base
from api.endpoints import users, scans
import os

os.makedirs("uploads", exist_ok=True)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DermaScan AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- THE IMAGE SERVER ---
# This allows React to display `<img src="http://localhost:8000/uploads/my_face.jpg" />`
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.include_router(users.router, prefix="/api/users", tags=["Users"])
app.include_router(scans.router, prefix="/api/scans", tags=["Scans"])