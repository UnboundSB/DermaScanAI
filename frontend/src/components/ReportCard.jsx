import { useLocation, useNavigate } from 'react-router-dom';
import { ArrowLeft, TrendingDown, TrendingUp, Minus } from 'lucide-react';

export default function ReportCard() {
  const location = useLocation();
  const navigate = useNavigate();
  const report = location.state?.report;

  if (!report) {
    return (
      <div className="text-center mt-20">
        <p className="text-slate-500">No report data found.</p>
        <button onClick={() => navigate('/dashboard')} className="mt-4 text-blue-600 font-medium hover:underline">Return to Dashboard</button>
      </div>
    );
  }

  const isFollowUp = report.type === "progress_evaluation";

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <button onClick={() => navigate('/dashboard')} className="flex items-center text-slate-500 hover:text-slate-800 transition">
        <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
      </button>

      <div className="bg-white rounded-2xl shadow-lg border border-slate-100 overflow-hidden">
        {/* Header Section */}
        <div className={`p-6 ${isFollowUp ? 'bg-emerald-50' : 'bg-blue-50'} border-b border-slate-100`}>
          <h2 className="text-2xl font-bold text-slate-900">
            {isFollowUp ? "10-Day Progress Report" : "Clinical Diagnosis"}
          </h2>
          {report.quality_score && (
            <p className="text-sm mt-1 text-slate-600">Scan Quality: {report.quality_score}/10</p>
          )}
        </div>

        {/* Prescription / Feedback Body */}
        <div className="p-6 md:p-8">
          <div className="mb-8">
            <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-3">Clinical Assessment</h3>
            <p className="text-lg text-slate-800 leading-relaxed font-medium">
              {report.prescription || report.feedback}
            </p>
          </div>

          {/* Render Math for Follow-Ups */}
          {isFollowUp && report.growth_decay_rates && (
            <div>
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Symptom Tracking (10-Day Delta)</h3>
              <div className="space-y-3">
                {Object.entries(report.growth_decay_rates).map(([symptom, rate]) => {
                  const isImprovement = rate < 0;
                  const isPlateau = rate === 0;
                  return (
                    <div key={symptom} className="flex justify-between items-center p-3 bg-slate-50 rounded-lg border border-slate-100">
                      <span className="capitalize font-medium text-slate-700">{symptom.replace('_', ' ')}</span>
                      <span className={`flex items-center font-bold ${isImprovement ? 'text-emerald-600' : isPlateau ? 'text-slate-500' : 'text-red-500'}`}>
                        {isImprovement ? <TrendingDown className="mr-1 h-4 w-4" /> : isPlateau ? <Minus className="mr-1 h-4 w-4" /> : <TrendingUp className="mr-1 h-4 w-4" />}
                        {Math.abs(rate)}% {isImprovement ? 'Reduction' : isPlateau ? 'Unchanged' : 'Increase'}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Render Stats for Initial Scans */}
          {!isFollowUp && report.margin_results && (
            <div>
              <h3 className="text-sm font-bold text-slate-400 uppercase tracking-wider mb-4">Detected Margins</h3>
              <div className="flex flex-wrap gap-2">
                {Object.entries(report.margin_results).map(([symptom, conf]) => (
                  <span key={symptom} className="px-3 py-1 bg-slate-100 border border-slate-200 text-slate-700 rounded-full text-sm font-medium capitalize">
                    {symptom.replace('_', ' ')}: {conf.toFixed(1)}%
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}