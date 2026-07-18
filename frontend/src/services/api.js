const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'https://dermascanai-l6d0.onrender.com';

/**
 * Fetch wrapper that adds retry logic with exponential backoff.
 * Useful for handling Render cold starts (502/503 errors).
 */
export const fetchWithRetry = async (endpoint, options = {}, retries = 3, backoff = 1000) => {
  const url = endpoint.startsWith('http') ? endpoint : `${API_BASE_URL}${endpoint}`;
  
  try {
    const response = await fetch(url, options);
    
    if (!response.ok) {
      // 502/503 commonly happen during cold start
      if ((response.status === 502 || response.status === 503) && retries > 0) {
        console.warn(`Server starting up, retrying in ${backoff}ms...`);
        await new Promise(resolve => setTimeout(resolve, backoff));
        return fetchWithRetry(url, options, retries - 1, backoff * 2);
      }
    }
    
    return response;
  } catch (error) {
    if (retries > 0) {
      console.warn(`Network error, retrying in ${backoff}ms...`);
      await new Promise(resolve => setTimeout(resolve, backoff));
      return fetchWithRetry(url, options, retries - 1, backoff * 2);
    }
    throw error;
  }
};

export const api = {
  get: (endpoint, options = {}) => fetchWithRetry(endpoint, { ...options, method: 'GET' }),
  post: (endpoint, options = {}) => fetchWithRetry(endpoint, { ...options, method: 'POST' }),
  put: (endpoint, options = {}) => fetchWithRetry(endpoint, { ...options, method: 'PUT' }),
  delete: (endpoint, options = {}) => fetchWithRetry(endpoint, { ...options, method: 'DELETE' }),
};
