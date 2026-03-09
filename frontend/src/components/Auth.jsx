import { useState } from 'react';
import axios from 'axios';
import { ShieldCheck, Activity } from 'lucide-react';

export default function Auth({ setUserId }) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const endpoint = isLogin ? '/login' : '/register';
    // Matches your FastAPI backend routing exactly
    const url = `http://localhost:8000/api/users${endpoint}?username=${username}&password=${password}`;

    try {
      const response = await axios.post(url);
      const id = response.data.user_id;
      
      // Save to local storage to keep user logged in across refreshes
      localStorage.setItem('userId', id);
      setUserId(id);
    } catch (err) {
      setError(err.response?.data?.detail || 'An error occurred connecting to the server.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4">
      <div className="w-full max-w-md space-y-8 rounded-2xl bg-white p-10 shadow-xl border border-slate-100">
        <div className="text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-blue-50">
            <Activity className="h-8 w-8 text-blue-600" />
          </div>
          <h2 className="mt-6 text-3xl font-extrabold text-slate-900">
            DermaScan AI
          </h2>
          <p className="mt-2 text-sm text-slate-500">
            Clinical-grade skin analysis & tracking
          </p>
        </div>

        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          {error && (
            <div className="rounded-md bg-red-50 p-4 text-sm text-red-700 border border-red-200">
              {error}
            </div>
          )}
          
          <div className="space-y-4 rounded-md shadow-sm">
            <div>
              <input
                type="text"
                required
                className="relative block w-full rounded-lg border border-slate-300 px-3 py-3 text-slate-900 placeholder-slate-400 focus:z-10 focus:border-blue-500 focus:outline-none focus:ring-blue-500 sm:text-sm"
                placeholder="Username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            <div>
              <input
                type="password"
                required
                className="relative block w-full rounded-lg border border-slate-300 px-3 py-3 text-slate-900 placeholder-slate-400 focus:z-10 focus:border-blue-500 focus:outline-none focus:ring-blue-500 sm:text-sm"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="group relative flex w-full justify-center rounded-lg border border-transparent bg-blue-600 py-3 px-4 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-blue-400 transition-all"
          >
            {loading ? (
              "Authenticating..."
            ) : (
              <>
                <span className="absolute inset-y-0 left-0 flex items-center pl-3">
                  <ShieldCheck className="h-5 w-5 text-blue-500 group-hover:text-blue-400" />
                </span>
                {isLogin ? 'Secure Sign In' : 'Create Clinical Profile'}
              </>
            )}
          </button>
        </form>

        <div className="text-center">
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="text-sm font-medium text-blue-600 hover:text-blue-500"
          >
            {isLogin ? "Don't have a profile? Register here" : "Already have a profile? Sign in"}
          </button>
        </div>
      </div>
    </div>
  );
}