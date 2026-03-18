import React, { useState, useEffect } from 'react';
import './Auth.css';

const Auth = ({ onAuthSuccess }) => {
  const [showIntro, setShowIntro] = useState(true);
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // --- DYNAMIC CALIBRATION SLIDER STATE ---
  const [calibrationValue, setCalibrationValue] = useState(0);
  const [isCalibrated, setIsCalibrated] = useState(false);
  
  // Initialize with a random number between 15 and 85
  const [targetCalibration, setTargetCalibration] = useState(() => Math.floor(Math.random() * 71) + 15);

  // Helper to generate a new random target
  const resetCalibrationTarget = () => {
    setTargetCalibration(Math.floor(Math.random() * 71) + 15);
    setCalibrationValue(0);
    setIsCalibrated(false);
  };

  // Check if slider hits the exact target
  useEffect(() => {
    if (parseInt(calibrationValue) === targetCalibration) {
      setIsCalibrated(true);
    } else {
      setIsCalibrated(false);
    }
  }, [calibrationValue, targetCalibration]);

  const handleBegin = () => {
    setShowIntro(false);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!isCalibrated) {
      setError('Please calibrate the lens to proceed.');
      return;
    }

    setIsLoading(true);
    setError('');

    const endpoint = isLogin ? '/api/users/login' : '/api/users/register';
    
    try {
      const response = await fetch(`http://localhost:8000${endpoint}?username=${encodeURIComponent(username)}&password=${encodeURIComponent(password)}`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json'
        }
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }

      onAuthSuccess(data.user_id);

    } catch (err) {
      setError(err.message);
      // Optional: Force them to re-calibrate if login fails to stop brute forcing
      resetCalibrationTarget(); 
    } finally {
      setIsLoading(false);
    }
  };

  // Intro Screen
  if (showIntro) {
    return (
      <div className="intro-screen">
        <div className="intro-particles" />
        
        <div className="scanner-orb-container">
          <div className="orb-core">
            <div className="orb-icon">🔬</div>
            <div className="orb-label">SCANNING</div>
          </div>
          <div className="scan-ring scan-ring-1" />
          <div className="scan-ring scan-ring-2" />
          <div className="scan-ring scan-ring-3" />
        </div>

        <h1 className="intro-title">DermaScanAI</h1>
        <p className="intro-subtitle">Advanced AI-Powered Skin Analysis Platform</p>
        <p className="intro-description">
          Detect dark spots, wrinkles, puffy eyes, and aging signs with precision
        </p>
        
        <button className="begin-button" onClick={handleBegin}>
          <span className="button-text">Begin Analysis</span>
          <span className="button-arrow">→</span>
        </button>

        <div className="intro-features">
          <div className="feature-badge">
            <span className="feature-icon">🎯</span>
            <span>98% Accuracy</span>
          </div>
          <div className="feature-badge">
            <span className="feature-icon">⚡</span>
            <span>Instant Results</span>
          </div>
          <div className="feature-badge">
            <span className="feature-icon">🔒</span>
            <span>HIPAA Compliant</span>
          </div>
        </div>
      </div>
    );
  }

  // Auth Form Screen
  return (
    <div className="auth-container">
      <div className="auth-background">
        <div className="bg-particle" />
        <div className="bg-particle" />
        <div className="bg-particle" />
      </div>

      <div className="auth-card">
        <div className="auth-header">
          <div className="logo-badge">
            <span className="logo-icon">🔬</span>
            <div className="scan-line" />
          </div>
          <h1 className="auth-title">DermaScanAI</h1>
          <p className="auth-subtitle">
            {isLogin ? 'Access your clinical profile' : 'Initialize your patient record'}
          </p>
        </div>

        {/* Tab Switcher */}
        <div className="auth-tabs">
          <button 
            className={`tab ${isLogin ? 'active' : ''}`}
            onClick={() => {
              setIsLogin(true);
              setError('');
              resetCalibrationTarget(); // Generates new random number
            }}
            type="button"
          >
            Sign In
          </button>
          <button 
            className={`tab ${!isLogin ? 'active' : ''}`}
            onClick={() => {
              setIsLogin(false);
              setError('');
              resetCalibrationTarget(); // Generates new random number
            }}
            type="button"
          >
            Sign Up
          </button>
        </div>

        {error && (
          <div className="auth-error">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label className="form-label">
              <span className="label-icon">👤</span>
              Patient ID (Username)
            </label>
            <input 
              type="text" 
              value={username} 
              onChange={(e) => setUsername(e.target.value)} 
              required 
              placeholder="Enter your username"
              className="form-input"
            />
          </div>

          <div className="form-group">
            <label className="form-label">
              <span className="label-icon">🔐</span>
              Security Key (Password)
            </label>
            <input 
              type="password" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              required 
              placeholder="Enter your password"
              className="form-input"
            />
          </div>

          {/* Calibration Slider - "I'm not a robot" replacement */}
          <div className={`calibration-container ${isCalibrated ? 'calibrated-success' : ''}`}>
            <div className="calibration-header">
              <label className="calibration-label">
                {isCalibrated 
                  ? '✓ Lens Calibrated. System Ready.' 
                  : `🎯 Move Slider to Mark ${targetCalibration}%`}
              </label>
              {isCalibrated && <div className="calibration-checkmark">✓</div>}
            </div>
            
            <div className="slider-wrapper">
              <div className="slider-track">
                <div 
                  className="slider-progress" 
                  style={{ width: `${calibrationValue}%` }}
                />
                <div 
                  className="target-marker" 
                  style={{ left: `${targetCalibration}%` }}
                >
                  <div className="marker-line" />
                  <div className="marker-label">{targetCalibration}%</div>
                </div>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={calibrationValue}
                onChange={(e) => setCalibrationValue(e.target.value)}
                disabled={isCalibrated || isLoading}
                className="lens-slider"
              />
              <div className="slider-readout">
                <span className="readout-value">{calibrationValue}</span>
                <span className="readout-unit">%</span>
              </div>
            </div>

            {!isCalibrated && Math.abs(calibrationValue - targetCalibration) <= 5 && parseInt(calibrationValue) !== targetCalibration && (
              <div className="calibration-hint">
                {calibrationValue < targetCalibration ? 'Almost there! Go a bit higher →' : '← Close! Go a bit lower'}
              </div>
            )}
          </div>

          <button 
            type="submit" 
            className="auth-submit-btn" 
            disabled={!isCalibrated || isLoading}
          >
            {isLoading ? (
              <>
                <span className="loading-spinner" />
                Processing...
              </>
            ) : (
              <>
                {isLogin ? 'Initialize Session' : 'Create Record'}
                <span className="submit-arrow">→</span>
              </>
            )}
          </button>
        </form>

        <div className="auth-footer">
          <div className="security-badge">
            <span className="security-icon">🔒</span>
            <span>256-bit Encryption</span>
          </div>
        </div>
      </div>

      {/* Pulse Indicators */}
      <div className="pulse-dot pulse-1" />
      <div className="pulse-dot pulse-2" />
      <div className="pulse-dot pulse-3" />
    </div>
  );
};

export default Auth;