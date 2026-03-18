import React, { useState } from 'react';
import './NewScan.css';

const NewScan = ({ userId, setCurrentView }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [scanType, setScanType] = useState('normal'); 
  
  const [isScanning, setIsScanning] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  // --- FILE HANDLING ---
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setReport(null); 
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setReport(null);
      setError(null);
    } else {
      setError("Please drop a valid image file (.jpg, .png).");
    }
  };

  // --- THE AI TRIGGER ---
  const handleScan = async () => {
    if (!selectedFile) return;

    setIsScanning(true);
    setError(null);

    const formData = new FormData();
    formData.append('user_id', userId);
    formData.append('scan_type', scanType);
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/api/predict/analyze', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Analysis failed. The AI rejected the image.');
      }

      setReport(data);

    } catch (err) {
      setError(err.message);
    } finally {
      setIsScanning(false);
    }
  };

  const resetScanner = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setReport(null);
    setError(null);
  };

  return (
    <div className="new-scan-container">
      <header className="scan-header">
        <h1>Initialize Scan</h1>
        <p>Upload a clear, well-lit photo of the target area for AI analysis.</p>
      </header>

      <div className="scan-content-wrapper">
        
        {/* --- LEFT SIDE: THE UPLOAD / PREVIEW ZONE --- */}
        <div className="scan-intake-panel">
          
          <div className="scan-type-selector">
            <button 
              className={`type-btn ${scanType === 'normal' ? 'active' : ''}`}
              onClick={() => setScanType('normal')}
              disabled={isScanning || report}
            >
              🎯 Baseline Scan
            </button>
            <button 
              className={`type-btn ${scanType === '10_day' ? 'active' : ''}`}
              onClick={() => setScanType('10_day')}
              disabled={isScanning || report}
            >
              📈 10-Day Progress
            </button>
          </div>

          {!previewUrl ? (
            <div 
              className="upload-dropzone"
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <div className="dropzone-content">
                <span className="upload-icon">📸</span>
                <h3>Drag & Drop Image</h3>
                <p>or</p>
                <label className="file-browse-btn">
                  Browse Files
                  <input 
                    type="file" 
                    accept="image/jpeg, image/png" 
                    onChange={handleFileChange} 
                    hidden 
                  />
                </label>
              </div>
            </div>
          ) : (
            <div className={`image-preview-container ${isScanning ? 'scanning-active' : ''}`}>
              <img src={previewUrl} alt="Target area" className="preview-image" />
              
              {isScanning && (
                <div className="scanner-laser-overlay">
                  <div className="laser-line"></div>
                  <div className="scanning-text">Analyzing dermal layers...</div>
                </div>
              )}
            </div>
          )}

          {error && <div className="scan-error-alert">⚠️ {error}</div>}

          <div className="intake-actions">
            {previewUrl && !report && !isScanning && (
              <>
                <button className="action-btn outline" onClick={resetScanner}>Retake / Clear</button>
                <button className="action-btn primary pulse" onClick={handleScan}>Initialize AI Engine</button>
              </>
            )}
            
            {report && (
              <button className="action-btn outline" onClick={resetScanner}>Scan Another Area</button>
            )}
          </div>
        </div>

        {/* --- RIGHT SIDE: THE RESULTS REPORT CARD --- */}
        {report && (
          <div className="scan-results-panel">
            <div className="report-card">
              <div className="report-header">
                <h2>Clinical Analysis Complete</h2>
                <span className="status-badge success">Verified</span>
              </div>

              <div className="report-body">
                <h3>AI Diagnosis & Prescription</h3>
                <p className="prescription-text">{report.prescription || report.report}</p>

                {report.growth_decay_rates && (
                  <div className="delta-metrics">
                    <h3>10-Day Delta Metrics</h3>
                    <ul className="metrics-list">
                      {Object.entries(report.growth_decay_rates).map(([symptom, rate]) => (
                        // A negative rate means the symptom DECREASED (Improved).
                        <li key={symptom} className={rate < 0 ? 'improved' : rate > 0 ? 'worsened' : 'plateau'}>
                          <span className="symptom-name">{symptom.replace('_', ' ')}</span>
                          <span className="rate-value">{rate > 0 ? '+' : ''}{rate}%</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              <div className="report-footer">
                <button className="action-btn primary full-width" onClick={() => setCurrentView('dashboard')}>
                  Save & Return to Dashboard
                </button>
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default NewScan;