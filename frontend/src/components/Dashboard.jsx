import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { Activity, Clock, PlusCircle } from 'lucide-react';

export default function Dashboard({ userId }) {
  const navigate = useNavigate();
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(`http://localhost:8000/api/users/${userId}/history`);
        setHistory(res.data.history);
      } catch (err) {
        console.error("Failed to load history", err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [userId]);

  return (
    <div className="space-y-8">
      {/* Action Cards */}
      <div className="grid md:grid-cols-2 gap-4">
        <button 
          onClick={() => navigate('/scan/normal')}
          className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm hover:shadow-md transition text-left flex items-start space-x-4"
        >
          <div className="p-3 bg-blue-50 text-blue-600 rounded-lg"><PlusCircle /></div>
          <div>
            <h3 className="text-lg font-bold">New Clinical Scan</h3>
            <p className="text-sm text-slate-500">Take a fresh photo to analyze your skin's current baseline.</p>
          </div>
        </button>

        <button 
          onClick={() => navigate('/scan/10_day')}
          className="p-6 bg-white border border-slate-200 rounded-2xl shadow-sm hover:shadow-md transition text-left flex items-start space-x-4"
        >
          <div className="p-3 bg-emerald-50 text-emerald-600 rounded-lg"><Activity /></div>
          <div>
            <h3 className="text-lg font-bold">10-Day Follow Up</h3>
            <p className="text-sm text-slate-500">Track your progress and adjust your prescription.</p>
          </div>
        </button>
      </div>

      {/* History Feed */}
      <div>
        <h2 className="text-xl font-bold mb-4 flex items-center"><Clock className="mr-2 h-5 w-5" /> Previous Scans</h2>
        {loading ? (
          <p className="text-slate-500">Loading history...</p>
        ) : history.length === 0 ? (
          <p className="text-slate-500 bg-white p-6 rounded-xl border border-dashed border-slate-300 text-center">No previous scans found. Start by taking a new scan.</p>
        ) : (
          <div className="space-y-4">
            {history.map((record) => (
              <div key={record.id} className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex justify-between items-center cursor-pointer hover:bg-slate-50"
                   onClick={() => navigate('/report', { state: { report: record.report } })}>
                <div>
                  <p className="font-semibold capitalize text-slate-800">
                    {record.scan_type === '10_day' ? 'Follow-Up Scan' : 'Baseline Scan'}
                  </p>
                  <p className="text-sm text-slate-500">{new Date(record.timestamp).toLocaleString()}</p>
                </div>
                <div className="text-right">
                  {record.report.primary_diagnosis && (
                    <span className="inline-block px-3 py-1 bg-blue-50 text-blue-700 rounded-full text-xs font-medium capitalize">
                      {record.report.primary_diagnosis.replace('_', ' ')}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}