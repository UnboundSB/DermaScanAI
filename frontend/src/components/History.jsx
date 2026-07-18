import React, { useState, useEffect } from 'react';
import './History.css';
import { api } from '../services/api';

const History = ({ userId, setCurrentView, setCompareScanId }) => {
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // --- 1. FETCH THE LEDGER ---
  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const response = await api.get(`/api/users/${userId}/history`);
      if (!response.ok) throw new Error('Failed to load clinical history.');
      const data = await response.json();
      setHistory(data.history || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (userId) fetchHistory();
  }, [userId]);

  // --- 2. THE DELETE ACTION ---
  const handleDelete = async (scanId) => {
    // A quick browser confirmation so they don't accidentally wipe their medical data
    if (!window.confirm("Are you sure you want to permanently delete this scan record?")) return;

    try {
      // Hits the scans.py router we built to vaporize the BLOB and the record
      const response = await api.delete(`/api/scans/${scanId}`);
      
      if (!response.ok) throw new Error('Failed to delete the record.');
      
      // Remove it from the React state immediately so the UI updates without a refresh
      setHistory(prev => prev.filter(scan => scan.id !== scanId));
    } catch (err) {
      alert(`Error: ${err.message}`);
    }
  };

  // --- 3. THE COMPARE ACTION ---
  const handleCompare = (scanId) => {
    // Save the specific scan ID to the master App.jsx memory bank
    setCompareScanId(scanId);
    // Redirect to the Compare view
    setCurrentView('compare');
  };

  // --- HELPER: FORMAT DATE ---
  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return {
      day: date.toLocaleDateString('en-US', { day: 'numeric', month: 'short', year: 'numeric' }),
      time: date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
    };
  };

  if (isLoading) {
    return (
      <div className="history-container loading">
        <div className="spinner"></div>
        <p>Accessing medical archives...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="history-container error">
        <h2>Archive Error</h2>
        <p>{error}</p>
        <button className="action-btn primary" onClick={fetchHistory}>Retry Connection</button>
      </div>
    );
  }

  return (
    <div className="history-container">
      <header className="history-header">
        <h1>Clinical History</h1>
        <p>Your complete chronological timeline of dermatological scans.</p>
      </header>

      {history.length === 0 ? (
        <div className="empty-history">
          <span className="empty-icon">📂</span>
          <h2>No Archives Found</h2>
          <p>You haven't completed any scans yet.</p>
          <button className="action-btn primary" onClick={() => setCurrentView('scan')}>
            Take First Scan
          </button>
        </div>
      ) : (
        <div className="timeline-layout">
          {history.map((scan) => {
            const dateObj = formatDate(scan.timestamp);
            
            // Extract the top symptom dynamically from the bulk_symptoms margin dictionary
            // Since we saved the whole math payload in SQLite, we just read it here!
            let primaryIssue = "Healthy Baseline";
            if (scan.report.current_bulk_symptoms && Object.keys(scan.report.current_bulk_symptoms).length > 0) {
               // Find the symptom with the highest confidence score
               const symptoms = scan.report.current_bulk_symptoms;
               primaryIssue = Object.keys(symptoms).reduce((a, b) => symptoms[a] > symptoms[b] ? a : b).replace('_', ' ');
            }

            return (
              <div key={scan.id} className="timeline-item">
                
                {/* The Timeline Node (Left side) */}
                <div className="timeline-node">
                  <div className="node-dot"></div>
                  <div className="node-line"></div>
                </div>

                {/* The History Card (Right side) */}
                <div className="history-card">
                  <div className="history-card-header">
                    <div className="date-group">
                      <span className="history-date">{dateObj.day}</span>
                      <span className="history-time">{dateObj.time}</span>
                    </div>
                    <span className={`scan-badge ${scan.scan_type === '10_day' ? 'progress-check' : 'baseline'}`}>
                      {scan.scan_type === '10_day' ? 'Progress Check' : 'Baseline'}
                    </span>
                  </div>

                  <div className="history-card-body">
                    <div className="diagnosis-highlight">
                      <strong>Primary Focus:</strong> <span className="capitalize">{primaryIssue}</span>
                    </div>
                    <p className="history-preview-text">
                      {/* Show just a snippet of the full report so the card doesn't get massive */}
                      {(scan.report.prescription || scan.report.report || "").substring(0, 150)}...
                    </p>
                  </div>

                  <div className="history-card-actions">
                    <button 
                      className="action-btn outline danger-hover" 
                      onClick={() => handleDelete(scan.id)}
                    >
                      Delete Record
                    </button>
                    <button 
                      className="action-btn primary" 
                      onClick={() => handleCompare(scan.id)}
                    >
                      Compare to Current
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default History;