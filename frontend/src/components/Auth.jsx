import { useState, useEffect } from 'react';
import axios from 'axios';
import './Auth.css';

export default function Auth({ setUserId }) {
  // UI State
  const [showIntro, setShowIntro] = useState(true);
  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  
  // Data State
  const [username, setUsername] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Generate random particles for the intro
  const introParticles = Array.from({ length: 30 }).map((_, i) => ({
    left: `${Math.random() * 100}%`,
    animationDuration: `${Math.random() * 8 + 8}s`,
    animationDelay: `${Math.random() * 5}s`
  }));

  // Generate random particles for the background
  const bgParticles = Array.from({ length: 50 }).map((_, i) => ({
    left: `${Math.random() * 100}%`,
    animationDuration: `${Math.random() * 10 + 10}s`,
    animationDelay: `${Math.random() * 5}s`
  }));

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Extra validation for registration
    if (!isLogin && password !== confirmPassword) {
      setError('❌ Passwords do not match!');
      return;
    }

    setLoading(true);

    const endpoint = isLogin ? '/login' : '/register';
    // Mapping their "Email Address" input directly to our backend "username" 
    const url = `http://localhost:8000/api/users${endpoint}?username=${username}&password=${password}`;

    try {
      const response = await axios.post(url);
      const id = response.data.user_id;
      
      // Save to local storage to keep user logged in
      localStorage.setItem('userId', id);
      setUserId(id);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred connecting to the server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-wrapper">
      {/* INTRO SCREEN */}
      <div className={`intro-screen ${!showIntro ? 'hidden' : ''}`} id="introScreen">
        <div className="intro-particles">
          {introParticles.map((style, i) => (
            <div key={i} className="intro-particle" style={style}></div>
          ))}
        </div>
        
        <div className="scanner-orb">
          <div className="orb-core">
            <div className="orb-text">
              <div className="orb-icon">🔬</div>
              <div className="orb-label">SCANNING</div>
            </div>
          </div>
          <div className="scan-ring"></div>
          <div className="scan-ring"></div>
          <div className="scan-ring"></div>
        </div>

        <h1 className="intro-title">DermaScanAI</h1>
        <p className="intro-subtitle">Advanced AI-Powered Skin Analysis Platform</p>
        
        <button className="start-button" onClick={() => {
          setShowIntro(false);
          setError(''); // Clear any errors on start
        }}>
          <span>Begin Analysis →</span>
        </button>
      </div>

      {/* MAIN LOGIN CONTENT */}
      <div className={`main-content ${!showIntro ? 'visible' : ''}`}>
        <div className="bg-particles">
          {bgParticles.map((style, i) => (
            <div key={i} className="particle" style={style}></div>
          ))}
        </div>

        <div className="auth-container">
          <div className="brand-section">
            <div className="logo-container">
              <div className="logo">
                🔬
                <div className="scan-overlay"></div>
              </div>
            </div>
            <h1 className="brand-title">DermaScanAI</h1>
            <p className="brand-subtitle">Advanced AI-Powered Skin Analysis</p>
            
            <div className="features">
              <div className="feature">
                <div className="feature-icon">📸</div>
                <div className="feature-text">
                  <strong>Instant Analysis</strong><br />
                  Upload your photo and get results in seconds
                </div>
              </div>
              <div className="feature">
                <div className="feature-icon">🎯</div>
                <div className="feature-text">
                  <strong>Comprehensive Detection</strong><br />
                  Dark spots, wrinkles, puffy eyes, and aging signs
                </div>
              </div>
              <div className="feature">
                <div className="feature-icon">💡</div>
                <div className="feature-text">
                  <strong>Personalized Recommendations</strong><br />
                  Tailored skincare advice to improve your skin health
                </div>
              </div>
              <div className="feature">
                <div className="feature-icon">🔒</div>
                <div className="feature-text">
                  <strong>Private & Secure</strong><br />
                  Your data is encrypted and never shared
                </div>
              </div>
            </div>
          </div>

          <div className="form-section">
            <div className="pulse-dot"></div>
            <div className="pulse-dot"></div>
            <div className="pulse-dot"></div>
            
            <div className="form-header">
              <h2 className="form-title">Get Started</h2>
              <p className="form-description">Create an account or sign in to begin your skin analysis journey</p>
            </div>

            <div className="tabs">
              <button 
                className={`tab ${isLogin ? 'active' : ''}`} 
                onClick={() => { setIsLogin(true); setError(''); }}
              >
                Sign In
              </button>
              <button 
                className={`tab ${!isLogin ? 'active' : ''}`} 
                onClick={() => { setIsLogin(false); setError(''); }}
              >
                Sign Up
              </button>
            </div>

            {error && <div className="error-message">{error}</div>}

            <form onSubmit={handleSubmit}>
              {!isLogin && (
                <div className="form-group">
                  <label className="form-label" htmlFor="registerName">Full Name</label>
                  <input 
                    type="text" 
                    id="registerName" 
                    className="form-input" 
                    placeholder="John Doe" 
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    required={!isLogin} 
                  />
                </div>
              )}

              <div className="form-group">
                {/* Using their label, but wiring it to our 'username' state for the backend */}
                <label className="form-label" htmlFor="email">Email Address</label>
                <input 
                  type="email" 
                  id="email" 
                  className="form-input" 
                  placeholder="you@example.com" 
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required 
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="password">Password</label>
                <div className="password-container">
                  <input 
                    type={showPassword ? "text" : "password"} 
                    id="password" 
                    className="form-input" 
                    placeholder={isLogin ? "Enter your password" : "Create a strong password"} 
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required 
                  />
                  <button 
                    type="button" 
                    className="toggle-password" 
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? "👁️‍🗨️" : "👁️"}
                  </button>
                </div>
              </div>

              {!isLogin && (
                <>
                  <div className="form-group">
                    <label className="form-label" htmlFor="confirmPassword">Confirm Password</label>
                    <div className="password-container">
                      <input 
                        type={showConfirmPassword ? "text" : "password"} 
                        id="confirmPassword" 
                        className="form-input" 
                        placeholder="Re-enter your password" 
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        required={!isLogin} 
                      />
                      <button 
                        type="button" 
                        className="toggle-password" 
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      >
                        {showConfirmPassword ? "👁️‍🗨️" : "👁️"}
                      </button>
                    </div>
                  </div>

                  <div className="checkbox-group">
                    <input type="checkbox" id="terms" className="checkbox" required={!isLogin} />
                    <label htmlFor="terms" className="checkbox-label">
                      I agree to the <a href="#terms">Terms of Service</a> and <a href="#privacy">Privacy Policy</a>
                    </label>
                  </div>
                </>
              )}

              {isLogin && (
                <div className="forgot-password">
                  <a href="#forgot">Forgot password?</a>
                </div>
              )}

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? "Processing..." : (isLogin ? "Sign In" : "Create Account")}
              </button>
            </form>

          </div>
        </div>
      </div>
    </div>
  );
}