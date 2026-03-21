import os
import sys
import json
from datetime import datetime

# --- DYNAMIC PATH INJECTION ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import your enhanced conversational intelligence
from Backend.Model.recommender.model import SkincareRecommender, ImprovementObserver

class DiagnosticReplyer:
    """
    The Fan-Out Communication Hub.
    Routes the raw AI metrics through the conversational engine,
    and sends the final payload to BOTH the frontend and a local backend file.
    """
    def __init__(self):
        print("--- BOOTING CLINICAL REPLYER ---")
        self.recommender = SkincareRecommender()
        self.observer = ImprovementObserver()
        
        # Setup the local "Database" directory to save the second copy
        self.history_dir = os.path.join(os.path.dirname(__file__), '..', 'Database', 'Patient_History')
        os.makedirs(self.history_dir, exist_ok=True)

    def finalize_and_route_analysis(self, report_dict: dict, session_id: str = "guest_user") -> dict:
        """
        1. Generates the human-readable prescription.
        2. Injects it into the JSON.
        3. Saves a copy to the backend file system.
        4. Returns the finalized JSON for the frontend.
        """
        # Step 1: Generate Prescription using your enhanced logic
        diagnoses_list = list(report_dict["margin_results"].items())
        prescription = self.recommender.generate_prescription(diagnoses_list)
        
        # Step 2: Enrich the payload
        final_payload = report_dict.copy()
        final_payload["prescription"] = prescription
        final_payload["timestamp"] = datetime.now().isoformat()
        final_payload["session_id"] = session_id

        # Step 3: Fan-Out to Backend File (The "Database")
        self._save_to_backend(final_payload, session_id, "analysis")

        # Step 4: Return for the Frontend
        return final_payload

    def finalize_and_route_progress(self, symptoms: list, status: str, session_id: str = "guest_user") -> dict:
        """
        Handles the 10-day progress evaluation with the exact same fan-out logic.
        """
        # Triggers your updated plateau/success/SOS logic
        feedback = self.observer.evaluate_10_day_trial(symptoms, status)
        
        final_payload = {
            "status": "success",
            "type": "progress_evaluation",
            "tracked_symptoms": symptoms,
            "reported_status": status,
            "prescription": feedback,  # <--- CHANGED FROM 'feedback'
            "report": f"10-Day Delta Analysis Complete. Status: {status.upper()}", # Add this so the UI title works too!
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id
        }

        self._save_to_backend(final_payload, session_id, "progress")
        
        return final_payload

    def _save_to_backend(self, payload: dict, session_id: str, record_type: str):
        """
        Silently writes the exact frontend JSON to a backend file.
        This represents the 'simultaneous' secondary routing.
        """
        filename = f"{session_id}_{record_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.history_dir, filename)
        
        try:
            with open(filepath, 'w') as f:
                json.dump(payload, f, indent=4)
            print(f"[*] Fan-Out Success: Record securely logged to {filepath}")
        except Exception as e:
            print(f"[!] Fan-Out Error: Failed to save record - {e}")

if __name__ == "__main__":
    # --- ISOLATED TEST RUN FOR NEW CONVERSATIONAL FLOW ---
    replyer = DiagnosticReplyer()
    
    mock_ai_report = {
        "status": "success",
        "quality_score": 8.75,
        "primary_diagnosis": "acne",
        "confidence": 72.50,
        "margin_results": {
            "acne": 72.50,
            "darkspots": 68.00
        }
    }
    
    print("\n--- TESTING FAN-OUT ARCHITECTURE (ANALYSIS) ---")
    final_analysis_json = replyer.finalize_and_route_analysis(mock_ai_report, session_id="test_user_001")
    print(json.dumps(final_analysis_json, indent=2))
    
    print("\n--- TESTING FAN-OUT ARCHITECTURE (WORSENED SOS PROTOCOL) ---")
    final_progress_json = replyer.finalize_and_route_progress(["acne", "darkspots"], "worsened", session_id="test_user_001")
    print(json.dumps(final_progress_json, indent=2))