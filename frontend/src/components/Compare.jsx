import React, { useState, useEffect } from 'react';
import './Compare.css';

const Compare = ({ userId, compareScanId, setCurrentView }) => {
  const [currentScan, setCurrentScan] = useState(null);
  const [historicalScan, setHistoricalScan] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchComparisonData = async () => {
      if (!userId || !compareScanId) {
        setError("Invalid comparison parameters.");
        setIsLoading(false);
        return;
      }

      try {
        // Fetch the full history to extract both the latest and the specific historical scan
        const response = await fetch(`http://localhost:8000/api/users/${userId}/history`);
        if (!response.ok) throw new Error('Failed to load clinical data for comparison.');
        
        const data = await response.json();
        const history = data.history || [];

        if (history.length === 0) {
          throw new Error("No scan history available.");
        }

        // The latest scan is always the first one in the sorted array
        const latest = history[0];
        // The historical scan is the one they clicked on
        const historical = history.find(scan => scan.id === compareScanId);

        if (!historical) {
          throw new Error("Historical scan record not found or was deleted.");
        }

        setCurrentScan(latest);
        setHistoricalScan(historical);

      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    fetchComparisonData();
  }, [userId, compareScanId]);

  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' 
    });
  };

  if (isLoading) {
    return (
      <div className="compare-container loading">
        <div className="spinner"></div>
        <p>Initializing Delta Comparison Engine...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="compare-container error">
        <h2>Comparison Failed</h2>
        <p>{error}</p>
        <button className="action-btn outline" onClick={() => setCurrentView('history')}>
          Return to Ledger
        </button>
      </div>
    );
  }

  // Edge Case: They clicked "Compare" on their most recent scan
  if (currentScan.id === historicalScan.id) {
    return (
      <div className="compare-container edge-case">
        <h2>Comparison Not Possible</h2>
        <p>You selected your most recent scan. Please select an older record to compare against your current baseline.</p>
        <button className="action-btn primary" onClick={() => setCurrentView('history')}>
          Return to Ledger
        </button>
      </div>
    );
  }

  return (
    <div className="compare-container">
      <header className="compare-header">
        <div className="header-top">
          <button className="back-btn" onClick={() => setCurrentView('history')}>
            <span>←</span> Back to Ledger
          </button>
          <span className="status-badge progress-check">Delta Analysis Active</span>
        </div>
        <h1>Comparative Analysis</h1>
        <p>Evaluating clinical progression between historical record and current baseline.</p>
      </header>

      <div className="split-screen-layout">
        
        {/* LEFT SIDE: HISTORICAL SCAN */}
        <div className="comparison-panel historical">
          <div className="panel-badge">Historical Record</div>
          <div className="panel-date">📅 {formatDate(historicalScan.timestamp)}</div>
          
          <div className="panel-content">
            <div className="data-block">
              <h3>Diagnosis at the time</h3>
              <p className="prescription-text">{historicalScan.report.prescription || historicalScan.report.report}</p>
            </div>
            
            {/* Displaying raw confidence scores if available for visual depth */}
            {historicalScan.report.current_bulk_symptoms && (
              <div className="data-block metrics">
                <h3>Symptom Confidence Scores</h3>
                <ul>
                  {Object.entries(historicalScan.report.current_bulk_symptoms).map(([symp, val]) => (
                    <li key={symp}>
                      <span className="symp-name">{symp.replace('_', ' ')}</span>
                      <span className="symp-val">{val.toFixed(1)}%</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>

        {/* MIDDLE DIVIDER */}
        <div className="vs-divider">
          <div className="vs-circle">VS</div>
          <div className="vs-line"></div>
        </div>

        {/* RIGHT SIDE: CURRENT BASELINE */}
        <div className="comparison-panel current">
          <div className="panel-badge highlight">Current Baseline</div>
          <div className="panel-date">📅 {formatDate(currentScan.timestamp)}</div>
          
          <div className="panel-content">
            <div className="data-block">
              <h3>Current Diagnosis</h3>
              <p className="prescription-text">{currentScan.report.prescription || currentScan.report.report}</p>
            </div>

            {currentScan.report.current_bulk_symptoms && (
              <div className="data-block metrics">
                <h3>Symptom Confidence Scores</h3>
                <ul>
                  {Object.entries(currentScan.report.current_bulk_symptoms).map(([symp, val]) => {
                    // Quick delta calculation just for the UI
                    const pastVal = historicalScan.report.current_bulk_symptoms?.[symp] || 0;
                    const diff = (val - pastVal).toFixed(1);
                    const isImproved = diff < 0;
                    
                    return (
                      <li key={symp} className={isImproved ? 'improved' : diff > 0 ? 'worsened' : ''}>
                        <span className="symp-name">{symp.replace('_', ' ')}</span>
                        <div className="val-group">
                          <span className="symp-val">{val.toFixed(1)}%</span>
                          {pastVal > 0 && diff != 0 && (
                            <span className={`delta-tag ${isImproved ? 'good' : 'bad'}`}>
                              {diff > 0 ? '+' : ''}{diff}%
                            </span>
                          )}
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>
        </div>

      </div>
    </div>
  );
};

export default Compare;