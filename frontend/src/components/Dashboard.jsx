import React, { useState, useEffect } from 'react';
import './Dashboard.css';

const Dashboard = ({ userId, setCurrentView }) => {
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // --- THE DATA FETCH (Backend Route Fixed!) ---
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        // FIXED: The history endpoint lives inside the users router, not scans!
        const response = await fetch(`http://localhost:8000/api/users/${userId}/history`);
        if (!response.ok) {
          throw new Error('Failed to fetch clinical data.');
        }
        const data = await response.json();
        setHistory(data.history || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };

    if (userId) {
      fetchHistory();
    }
  }, [userId]);

  // --- DATA PROCESSING FOR QUICK STATS ---
  const totalScans = history.length;
  const latestScan = totalScans > 0 ? history[0] : null; 
  
  // Format the date to look professional
  const formatDate = (isoString) => {
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric', 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  // Get badge class name
  const getBadgeClass = (scanType) => {
    if (scanType === '10_day') {
      return 'scan-badge progress-check';
    }
    return `scan-badge ${scanType || 'baseline'}`;
  };

  // Get badge text
  const getBadgeText = (scanType) => {
    if (scanType === '10_day') {
      return 'Progress Check';
    }
    return 'Baseline Scan';
  };

  if (isLoading) {
    return (
      <div className="dashboard-container loading">
        <div className="spinner"></div>
        <p>Loading Clinical Hub...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-container error">
        <h2>Connection Error</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Retry Connection</button>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Welcome, Patient #{userId}</h1>
        <p className="dashboard-subtitle">Here is your current clinical overview.</p>
      </header>

      {/* --- QUICK STATS WIDGETS --- */}
      <div className="stats-grid">
        <div className="stat-card">
          <h3>Total Scans</h3>
          <p className="stat-value">{totalScans}</p>
        </div>
        <div className="stat-card">
          <h3>Latest Analysis</h3>
          <p className="stat-value">
            {latestScan ? formatDate(latestScan.timestamp) : 'N/A'}
          </p>
        </div>
        <div className="stat-card">
          <h3>Current Status</h3>
          <p className="stat-value">
            {totalScans > 0 ? 'Active Monitoring' : 'Awaiting Baseline'}
          </p>
        </div>
      </div>

      {/* --- THE MAIN CONTENT AREA --- */}
      <div className="dashboard-main-content">
        
        {totalScans === 0 ? (
          // EMPTY STATE
          <div className="empty-state-card">
            <div className="empty-icon">📷</div>
            <h2>No Clinical Data Found</h2>
            <p>You haven't initialized your baseline scan yet. Start the AI analysis to generate your customized treatment plan.</p>
            <button 
              className="action-btn primary"
              onClick={() => setCurrentView('scan')}
            >
              Take First Scan
            </button>
          </div>
        ) : (
          // ACTIVE STATE (Showing latest scan summary)
          <div className="latest-scan-summary">
            <h2>Latest Report Summary</h2>
            <div className="summary-card">
              <div className="summary-header">
                <span className={getBadgeClass(latestScan.scan_type)}>
                  {latestScan.scan_type === '10_day' ? '📈' : '🎯'} {getBadgeText(latestScan.scan_type)}
                </span>
                <span className="summary-date">
                  🕐 {formatDate(latestScan.timestamp)}
                </span>
              </div>
              
              <div className="summary-body">
                {/* Display the AI-generated prescription/report */}
                <p className="prescription-text">
                  {latestScan.report?.prescription || latestScan.report?.report || 'No detailed report available.'}
                </p>
              </div>

              <div className="summary-actions">
                <button 
                  className="action-btn outline"
                  onClick={() => setCurrentView('history')}
                >
                  View Full History
                </button>
                <button 
                  className="action-btn primary"
                  onClick={() => setCurrentView('scan')}
                >
                  Take Follow-up Scan
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default Dashboard;