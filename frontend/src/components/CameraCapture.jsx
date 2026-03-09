import { useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import axios from 'axios';
import { Camera, Upload, AlertCircle } from 'lucide-react';

export default function CameraCapture({ userId }) {
  const { type } = useParams(); // 'normal' or '10_day'
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setError('');
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);
    formData.append('scan_type', type);

    try {
      const response = await axios.post('http://localhost:8000/api/ml/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      // Navigate to the report card and pass the JSON data via React Router state
      navigate('/report', { state: { report: response.data } });
    } catch (err) {
      setError(err.response?.data?.detail?.reason || err.response?.data?.detail || "Failed to process image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-xl mx-auto bg-white p-8 rounded-2xl shadow-xl border border-slate-100 text-center">
      <h2 className="text-2xl font-bold mb-2 capitalize">
        {type === '10_day' ? '10-Day Progress Scan' : 'New Clinical Scan'}
      </h2>
      <p className="text-slate-500 mb-8">Ensure your face is well-lit and clearly visible.</p>

      {error && (
        <div className="mb-6 bg-red-50 text-red-700 p-4 rounded-lg text-sm flex items-center justify-center">
          <AlertCircle className="mr-2 h-5 w-5" /> {error}
        </div>
      )}

      {preview ? (
        <div className="space-y-6">
          <img src={preview} alt="Preview" className="w-full h-64 object-cover rounded-xl border-2 border-slate-200" />
          <div className="flex space-x-4">
            <button onClick={() => { setFile(null); setPreview(null); }} className="w-1/2 py-3 rounded-lg border border-slate-300 font-medium text-slate-700 hover:bg-slate-50">
              Retake
            </button>
            <button onClick={handleUpload} disabled={loading} className="w-1/2 py-3 rounded-lg bg-blue-600 font-medium text-white hover:bg-blue-700 disabled:bg-blue-300 flex justify-center items-center">
              {loading ? 'Analyzing...' : <><Upload className="mr-2 h-4 w-4" /> Analyze Skin</>}
            </button>
          </div>
        </div>
      ) : (
        <div className="border-2 border-dashed border-slate-300 rounded-xl p-12 hover:bg-slate-50 transition cursor-pointer relative">
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange} 
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          <Camera className="mx-auto h-12 w-12 text-slate-400 mb-4" />
          <p className="text-slate-600 font-medium">Tap to take a photo or upload</p>
        </div>
      )}
    </div>
  );
}