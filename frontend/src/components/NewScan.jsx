import React, { useState, useRef, useEffect } from 'react';
import './NewScan.css';

const NewScan = ({ userId, setCurrentView }) => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [scanType, setScanType] = useState('normal'); 
  
  const [isScanning, setIsScanning] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  // --- WEBCAM STATE & REFS ---
  const [inputMode, setInputMode] = useState('upload'); // 'upload' or 'camera'
  const [isCameraActive, setIsCameraActive] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  // --- CAMERA HANDLING ---
  // This useEffect ensures the video element exists in the DOM BEFORE we turn on the webcam
  useEffect(() => {
    if (inputMode === 'camera') {
      startCamera();
    } else {
      stopCamera();
    }
    // Cleanup on unmount
    return () => stopCamera();
  }, [inputMode]);

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } 
      });
      streamRef.current = stream;
      
      // Crucial: Attach stream ONLY if videoRef is ready
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsCameraActive(true);
    } catch (err) {
      setError("Camera access denied or no camera found. Please use upload mode.");
      setInputMode('upload');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setIsCameraActive(false);
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    canvas.toBlob((blob) => {
      const file = new File([blob], "webcam-scan.jpg", { type: "image/jpeg" });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      // Camera will automatically stop because setting previewUrl changes the UI
    }, 'image/jpeg', 0.95);
  };

  // --- FILE UPLOAD HANDLING ---
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) processFile(file);
  };

  const handleDragOver = (e) => e.preventDefault();

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    } else {
      setError("Please drop a valid image file (.jpg, .png).");
    }
  };

  const processFile = (file) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setReport(null);
    setError(null);
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

  // Toggle between Upload and Camera modes safely
  const handleModeSwitch = (mode) => {
    setInputMode(mode);
    resetScanner();
  };

  return (
    <div className="new-scan-container">
      <header className="scan-header">
        <h1>Initialize Scan</h1>
        <p>Capture or upload a clear, well-lit photo of the target area for AI analysis.</p>
      </header>

      <div className="scan-content-wrapper">
        
        {/* --- LEFT SIDE: THE INTAKE ZONE --- */}
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

          {/* Mode Switcher */}
          {!previewUrl && !report && (
            <div className="input-mode-switcher">
              <button 
                className={`mode-btn ${inputMode === 'upload' ? 'active' : ''}`}
                onClick={() => handleModeSwitch('upload')}
              >
                📁 Upload File
              </button>
              <button 
                className={`mode-btn ${inputMode === 'camera' ? 'active' : ''}`}
                onClick={() => handleModeSwitch('camera')}
              >
                📷 Live Camera
              </button>
            </div>
          )}

          {/* Dynamic Input Area */}
          {!previewUrl ? (
            inputMode === 'upload' ? (
              // UPLOAD MODE
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
                    <input type="file" accept="image/jpeg, image/png" onChange={handleFileChange} hidden />
                  </label>
                </div>
              </div>
            ) : (
              // CAMERA MODE
              <div className="camera-container">
                <video 
                  ref={videoRef} 
                  autoPlay 
                  playsInline 
                  className="live-video-feed" 
                  style={{ display: isCameraActive ? 'block' : 'none' }} 
                />
                
                {isCameraActive ? (
                  <button className="capture-shutter-btn" onClick={capturePhoto}>
                    <div className="shutter-inner"></div>
                  </button>
                ) : (
                  <div className="camera-loading">
                    <div className="spinner"></div>
                    <p>Initializing Secure Camera Feed...</p>
                  </div>
                )}
                {/* Hidden canvas to grab the frame */}
                <canvas ref={canvasRef} style={{ display: 'none' }} />
              </div>
            )
          ) : (
            // PREVIEW & SCANNING MODE
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

          {/* Action Buttons */}
          <div className="intake-actions" style={{ marginTop: '20px' }}>
            {previewUrl && !report && !isScanning && (
              <>
                <button className="action-btn outline" onClick={resetScanner}>Retake / Clear</button>
                <button className="action-btn primary pulse" onClick={handleScan}>Initialize AI Engine</button>
              </>
            )}
            
            {report && (
              <button className="action-btn outline" onClick={() => handleModeSwitch('upload')}>Scan Another Area</button>
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