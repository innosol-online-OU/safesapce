import React, { useState, useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import axios from 'axios';
import Login from './components/Login';
import Dashboard from './components/Dashboard';
import GPUNap from './components/GPUNap';

const PrivateRoute = ({ children }) => {
  const token = localStorage.getItem('token');
  return token ? children : <Navigate to="/login" />;
};

function App() {
  const [isSystemOffline, setIsSystemOffline] = useState(false);

  useEffect(() => {
    // Global Error Interceptor
    const interceptor = axios.interceptors.response.use(
      (response) => response,
      (error) => {
        if (!error.response) {
          // Network Error (Server down)
          setIsSystemOffline(true);
        } else if (error.response.status === 502 || error.response.status === 503) {
          // Bad Gateway / Service Unavailable (GPU suspended)
          setIsSystemOffline(true);
        }
        return Promise.reject(error);
      }
    );

    // Initial Health Check
    const checkHealth = async () => {
      try {
        await axios.get(`${import.meta.env.VITE_API_URL || '/api'}/status`);
      } catch (e) {
        // Only trigger if it's a hard failure
        if (!e.response || e.response.status >= 500) {
          setIsSystemOffline(true);
        }
      }
    };
    checkHealth();

    return () => {
      axios.interceptors.response.eject(interceptor);
    };
  }, []);

  const handleRetry = async () => {
    setIsSystemOffline(false);
    try {
      await axios.get(`${import.meta.env.VITE_API_URL || '/api'}/status`);
    } catch (e) {
      // If still down, it will re-trigger via interceptor or we catch it here
      setIsSystemOffline(true);
    }
  };

  return (
    <>
      {isSystemOffline && <GPUNap onRetry={handleRetry} />}
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/dashboard"
          element={
            <PrivateRoute>
              <Dashboard />
            </PrivateRoute>
          }
        />
        <Route path="*" element={<Navigate to="/login" />} />
      </Routes>
    </>
  );
}

export default App;
