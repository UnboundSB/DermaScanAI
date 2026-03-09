import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useState, useEffect } from 'react';
import Auth from './components/Auth';
import Dashboard from './components/Dashboard';
import CameraCapture from './components/CameraCapture';
import ReportCard from './components/ReportCard';

export default function App() {
  const [userId, setUserId] = useState(localStorage.getItem('userId'));

  useEffect(() => {
    const handleStorageChange = () => setUserId(localStorage.getItem('userId'));
    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('userId');
    setUserId(null);
  };

  return (
    <Router>
      <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
        {/* Simple Global Header */}
        {userId && (
          <header className="bg-white border-b border-slate-200 px-6 py-4 flex justify-between items-center">
            <h1 className="text-xl font-bold text-blue-600">DermaScan AI</h1>
            <button onClick={handleLogout} className="text-sm text-slate-500 hover:text-slate-800">
              Sign Out
            </button>
          </header>
        )}

        <main className="max-w-4xl mx-auto p-4 md:p-8">
          <Routes>
            <Route path="/" element={!userId ? <Auth setUserId={setUserId} /> : <Navigate to="/dashboard" />} />
            <Route path="/dashboard" element={userId ? <Dashboard userId={userId} /> : <Navigate to="/" />} />
            <Route path="/scan/:type" element={userId ? <CameraCapture userId={userId} /> : <Navigate to="/" />} />
            <Route path="/report" element={userId ? <ReportCard /> : <Navigate to="/" />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}