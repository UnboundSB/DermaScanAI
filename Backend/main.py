from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# --- The Ghost-Proof Imports ---
from core_db.db import engine, Base
from api.endpoints import users, predict, scans

# 1. Generate the Database Tables (Creates derma_history.db if it doesn't exist)
Base.metadata.create_all(bind=engine)

# 2. Initialize the AI Backend Server
app = FastAPI(title="DermaScan AI Engine", version="2.0")

# 3. --- THE CORS FIX ---
# This strictly allows your Vite/React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Standard Vite port
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, DELETE, PUT
    allow_headers=["*"],
)

# 4. Register the Routers (The API Architecture)
app.include_router(users.router, prefix="/api/users", tags=["Authentication & Identity"])
app.include_router(predict.router, prefix="/api/predict", tags=["AI Engine & Analysis"])
app.include_router(scans.router, prefix="/api/scans", tags=["History & Image Retrieval"])

# 5. System Health Check
@app.get("/health", tags=["System"])
def health_check():
    """
    A silent ping for the frontend to verify the PyTorch engine is awake.
    """
    return {"status": "DermaScan AI Backend is operational and ready for input."}