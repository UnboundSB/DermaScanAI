import React, { useState } from 'react';
import './App.css'; 

// --- THE COMPONENT ROSTER ---
import Auth from './components/Auth';
import Navbar from './components/Navbar';
import Dashboard from './components/Dashboard';
import NewScan from './components/NewScan';
import History from './components/History';
import Compare from './components/Compare';

function App() {
  // --- GLOBAL STATE ---
  // 1. Tracks who is logged in
  const [userId, setUserId] = useState(null);
  
  // 2. The Custom Router (dashboard, scan, history, compare)
  const [currentView, setCurrentView] = useState('dashboard');
  
  // 3. The Memory Bank (Stores a specific scan ID when a user clicks "Compare" on the History page)
  const [compareScanId, setCompareScanId] = useState(null);

  // --- AUTHENTICATION HANDLERS ---
  const handleAuthSuccess = (id) => {
    setUserId(id);
    setCurrentView('dashboard');
  };

  const handleLogout = () => {
    setUserId(null);
    setCurrentView('dashboard');
    setCompareScanId(null);
  };

  // --- THE GATEKEEPER ---
  // If no user is logged in, they cannot pass this line.
  if (!userId) {
    return <Auth onAuthSuccess={handleAuthSuccess} />;
  }

  // --- THE MASTER APP SHELL ---
  return (
    <div className="app-layout">
      {/* The Navigation Bar sits fixed at the top/side */}
      <Navbar 
        currentView={currentView} 
        setCurrentView={setCurrentView} 
        onLogout={handleLogout} 
      />

      {/* The Dynamic Viewport */}
      <main className="main-content">
        
        {currentView === 'dashboard' && (
          <Dashboard 
            userId={userId} 
            setCurrentView={setCurrentView} 
          />
        )}
        
        {currentView === 'scan' && (
          <NewScan 
            userId={userId} 
            setCurrentView={setCurrentView} 
          />
        )}
        
        {currentView === 'history' && (
          <History 
            userId={userId} 
            setCurrentView={setCurrentView} 
            setCompareScanId={setCompareScanId} 
          />
        )}
        
        {currentView === 'compare' && (
          <Compare 
            userId={userId} 
            compareScanId={compareScanId} 
            setCurrentView={setCurrentView} 
          />
        )}

      </main>
    </div>
  );
}

export default App;