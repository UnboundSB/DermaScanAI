import React, { useState, useEffect } from 'react';
import './Navbar.css';

const Navbar = ({ currentView, setCurrentView, onLogout }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Close mobile menu when view changes
  useEffect(() => {
    setIsMobileMenuOpen(false);
  }, [currentView]);

  // Close mobile menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (isMobileMenuOpen && !e.target.closest('.navbar-container') && !e.target.closest('.mobile-menu-toggle')) {
        setIsMobileMenuOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isMobileMenuOpen]);

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (isMobileMenuOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isMobileMenuOpen]);

  const handleNavClick = (view) => {
    setCurrentView(view);
    setIsMobileMenuOpen(false);
  };

  return (
    <>
      {/* Mobile Menu Toggle Button */}
      <button 
        className="mobile-menu-toggle" 
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        aria-label="Toggle menu"
      >
        {isMobileMenuOpen ? '✕' : '☰'}
      </button>

      {/* Overlay for mobile */}
      <div 
        className={`navbar-overlay ${isMobileMenuOpen ? 'active' : ''}`}
        onClick={() => setIsMobileMenuOpen(false)}
      />

      {/* Main Navbar */}
      <nav className={`navbar-container ${isMobileMenuOpen ? 'mobile-open' : ''}`}>
        {/* 1. The Brand / Logo */}
        <div className="navbar-brand" onClick={() => handleNavClick('dashboard')}>
          <span className="navbar-logo-icon">🔬</span>
          <span className="navbar-logo-text">
            DermaScan<span className="highlight">AI</span>
          </span>
        </div>

        {/* 2. The Core Navigation Links */}
        <div className="navbar-links">
          <button 
            className={`nav-btn ${currentView === 'dashboard' ? 'active' : ''}`}
            onClick={() => handleNavClick('dashboard')}
          >
            <span className="nav-icon">📊</span>
            Dashboard
          </button>
          
          <button 
            className={`nav-btn ${currentView === 'scan' ? 'active' : ''}`}
            onClick={() => handleNavClick('scan')}
          >
            <span className="nav-icon">🎯</span>
            New Scan
          </button>
          
          <button 
            className={`nav-btn ${(currentView === 'history' || currentView === 'compare') ? 'active' : ''}`}
            onClick={() => handleNavClick('history')}
          >
            <span className="nav-icon">📁</span>
            Clinical History
          </button>
        </div>

        {/* 3. User Actions (Logout) */}
        <div className="navbar-actions">
          <div className="status-indicator">
            <span className="status-dot"></span>
            System Online
          </div>
          
          <button className="logout-btn" onClick={() => {
            onLogout();
            setIsMobileMenuOpen(false);
          }}>
            <span className="logout-icon">⏏️</span>
            End Session
          </button>
        </div>
      </nav>
    </>
  );
};

export default Navbar;