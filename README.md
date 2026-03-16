# 🔬 DermaScan AI
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi&logoColor=white)](#)
[![React](https://img.shields.io/badge/React-19.0-61DAFB.svg?logo=react&logoColor=black)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](#)

**An Advanced, AI-Powered Clinical Skin Analysis & Symptom Tracking Platform**

DermaScan AI bridges the gap between advanced machine learning and consumer health tracking. By leveraging custom PyTorch neural networks via a lightning-fast FastAPI backend, and presenting it through a highly gamified, cyberpunk-inspired React interface, it transforms standard medical tracking into a cutting-edge, engaging experience.

---

## 📑 Table of Contents
1. [Project Philosophy](#-project-philosophy)
2. [System Architecture](#-system-architecture)
3. [Key Features & User Journey](#-key-features--user-journey)
4. [Tech Stack](#-tech-stack)
5. [Prerequisites & Installation](#-prerequisites--installation)
6. [API Reference](#-api-reference)
7. [Future Roadmap](#-future-roadmap)

---

## 💡 Project Philosophy
Traditional medical portals are sterile, intimidating, and difficult to navigate. DermaScan AI flips this paradigm by utilizing a "Glassmorphism" design system and gamified tracking mechanics. Users don't just upload photos; they engage with a responsive, futuristic AI that tracks their physical progress mathematically over time.

---

## 🏗️ System Architecture

The application operates on a decoupled client-server model:

* **The Visual Layer (Frontend):** A Vite-powered React Single Page Application (SPA). State is managed locally, and styling is handled via pure CSS, strictly optimizing for mobile responsiveness and fluid animations without heavy library bloat.
* **The Clinical Engine (Backend):** A Python FastAPI server that handles high-throughput asynchronous requests. It acts as the gateway between the user's data and the ML models.
* **The Brain (AI/ML):** PyTorch models evaluate raw image matrices, while a dedicated logic controller (`replyer.py`) calculates symptom margins, delta growth/decay rates, and generates human-readable clinical prescriptions.
* **The Vault (Database):** A relational SQLite database managed by SQLAlchemy, ensuring patient scan histories and encrypted profiles are securely stored and rapidly accessible.

---

## 🎯 Key Features & User Journey

1. **Secure Onboarding:** Users create an encrypted clinical profile to persist their data.
2. **Baseline Diagnostics:** Users upload a standard facial scan. The AI detects symptom margins (e.g., dark spots, acne, fine lines) and provides a baseline prescription.
3. **The 10-Day Challenge (Delta Tracking):** Upon follow-up scans, the system compares the new image against the baseline. It mathematically calculates the percentage reduction or increase in symptoms, classifying the result as an *Improvement*, *Plateau*, or *SOS* state.
4. **Historical Dashboard:** A dedicated feed where users can visually track their diagnostic history and health score over time.

---

## 💻 Tech Stack

### Frontend
* **React 19** (Component Architecture & Hooks)
* **Vite** (Build Tool & Dev Server)
* **Axios** (HTTP Client)
* **React Router DOM** (Navigation)
* **Lucide React** (SVG Iconography)
* **Vanilla CSS3** (Glassmorphism, Flexbox, Keyframe Animations)

### Backend & AI
* **Python 3.10+**
* **FastAPI** (RESTful API routing)
* **Uvicorn** (ASGI Server)
* **PyTorch** (Computer Vision & Neural Networks)
* **SQLAlchemy** (ORM)
* **SQLite3** (Database)

---

## ⚙️ Prerequisites & Installation

### Prerequisites
* **Node.js** (v18.0 or higher)
* **Python** (v3.10 or higher)
* *(Optional but recommended)* CUDA-compatible GPU for faster PyTorch inference.

### 1. Clone & Setup Backend
```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Boot the FastAPI server
uvicorn main:app --reload
The API documentation will be available at http://localhost:8000/docs.

2. Setup Frontend
Bash
# Open a new terminal and navigate to the frontend directory
cd frontend

# Install Node modules
npm install

# Start the Vite development server
npm run dev
The application will launch at http://localhost:5173.

🔌 API Reference (Core Endpoints)
POST /api/users/register - Creates a new user profile.

POST /api/users/login - Authenticates and retrieves user_id.

GET /api/users/{user_id}/history - Fetches the user's complete diagnostic timeline.

POST /api/ml/analyze - Accepts multipart/form-data (image file + metadata) and returns the AI prescription and margin calculations.

🗺️ Future Roadmap
[ ] AWS S3 Integration: Offload image storage from the local file system to the cloud.

[ ] Data Visualization: Implement Chart.js on the dashboard for visual symptom graphs.

[ ] OAuth2.0 Security: Upgrade standard authentication to JWT token-based security.

[ ] Export to PDF: Allow users to download their 10-day clinical reports to share with real-world dermatologists.

Developed with a focus on privacy, speed, and architectural scalability.


***

How is that for a broader scope? It adds a massive layer of professionalism. 

Now, sticking to our new "blueprint first" rule: let's talk about the **Camera Capture component** (`CameraCapture.jsx`). 

When the user clicks "New Clinical Scan" on the dashboard, what should they see? 
1. **The Sci-Fi Scanner:** A live webcam feed with a pulsing targeting reticle/crosshair overlay, looking like a terminator HUD?
2. **The Clean Dropzone:** A sleek, glassmorphism box where they simply click to upload an existing file from their phone/computer gallery?
3. **A hybrid:** Options for both live webcam *and* file upload?

Tell me your vision for the user experience here!