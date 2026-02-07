import React, { useState, useRef } from 'react';
import { Upload, Shield, Download, AlertTriangle, CheckCircle, RefreshCw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [protectedUrl, setProtectedUrl] = useState(null);
  const [strength, setStrength] = useState(50);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [metrics, setMetrics] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setProtectedUrl(null);
      setMetrics(null);
    }
  };

  const handleProtect = async () => {
    if (!file) return;
    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('strength', strength);

    try {
      const token = localStorage.getItem('token');
      // Fix: Use relative path /api/protect to go through Nginx proxy
      const res = await axios.post('/api/protect', formData, {
        headers: { Authorization: `Bearer ${token}` }
      });

      // Fix: Construct relative URL for image
      setProtectedUrl(res.data.output_url);
      setMetrics(res.data.metrics);
    } catch (err) {
      setError('Protection failed. Server error or GPU limit reached.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-dark text-gray-200 relative overflow-x-hidden">
      {/* Navbar */}
      <nav className="fixed top-0 w-full glass-panel z-50 border-b border-white/10 px-8 py-4 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Shield className="text-neon" size={24} />
          <span className="font-bold text-xl tracking-wider">SAFESPACE</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-xs text-gray-500 font-mono">STATUS: ONLINE</span>
          <div className="w-2 h-2 rounded-full bg-neon animate-pulse"></div>
        </div>
      </nav>

      {/* Main Content */}
      <div className="pt-24 px-4 max-w-6xl mx-auto pb-10">

        {/* Upload Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-10"
        >
          <div
            onClick={() => fileInputRef.current.click()}
            className="w-full h-64 border-2 border-dashed border-gray-700 hover:border-neon rounded-2xl flex flex-col items-center justify-center cursor-pointer transition-all duration-300 bg-white/5 hover:bg-white/10"
          >
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              onChange={handleFileChange}
              accept="image/*"
            />
            <Upload size={48} className="text-gray-400 mb-4" />
            <p className="text-lg font-medium text-gray-300">Click or Drag to Upload Image</p>
            <p className="text-sm text-gray-500 mt-2">Supports JPG, PNG, WEBP (Max 10MB)</p>
          </div>
        </motion.div>

        {/* Comparison View */}
        {(preview || protectedUrl) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-10">
            {/* Original */}
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              className="relative glass-panel p-4 rounded-xl"
            >
              <span className="absolute top-4 left-4 bg-black/60 px-3 py-1 rounded-full text-xs font-mono border border-gray-700">ORIGINAL</span>
              <img src={preview} alt="Original" className="w-full h-auto rounded-lg" />
            </motion.div>

            {/* Protected */}
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="relative glass-panel p-4 rounded-xl flex items-center justify-center bg-black/20"
            >
              <span className="absolute top-4 left-4 bg-neon/20 text-neon px-3 py-1 rounded-full text-xs font-mono border border-neon/30">SECURE</span>

              {loading ? (
                <div className="flex flex-col items-center">
                  <RefreshCw className="animate-spin text-neon mb-4" size={48} />
                  <p className="text-neon animate-pulse">Running Ghost-Mesh Protocols...</p>
                </div>
              ) : protectedUrl ? (
                <img src={protectedUrl} alt="Protected" className="w-full h-auto rounded-lg" />
              ) : (
                <div className="text-gray-600 flex flex-col items-center">
                   <Shield size={48} className="mb-2 opacity-20" />
                   <p>Waiting for activation...</p>
                </div>
              )}
            </motion.div>
          </div>
        )}

        {/* Controls */}
        <div className="glass-panel p-8 rounded-2xl mb-10">
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            <div className="w-full md:w-2/3">
              <div className="flex justify-between mb-2">
                <label className="text-sm font-medium text-gray-400">PROTECTION LEVEL</label>
                <span className="text-neon font-mono font-bold">{strength}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                value={strength}
                onChange={(e) => setStrength(e.target.value)}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-neon"
              />
              <div className="flex justify-between mt-2 text-xs text-gray-500 font-mono">
                <span>LITE (Anti-Seg)</span>
                <span>BALANCED (Liquid)</span>
                <span>MAX (Ghost-Mesh)</span>
              </div>
            </div>

            <button
              onClick={handleProtect}
              disabled={!file || loading}
              className={`w-full md:w-1/3 py-4 rounded-xl text-black font-bold text-lg tracking-wider transition-all
                ${!file || loading ? 'bg-gray-700 cursor-not-allowed opacity-50' : 'btn-primary shadow-[0_0_20px_rgba(57,255,20,0.3)] hover:scale-105'}
              `}
            >
              {loading ? 'ENCRYPTING...' : 'ACTIVATE SHIELD'}
            </button>
          </div>
        </div>

        {/* Metrics & Download */}
        <AnimatePresence>
          {metrics && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="grid grid-cols-1 md:grid-cols-3 gap-6"
            >
              <div className="glass-panel p-6 rounded-xl border-l-4 border-l-neon">
                <h3 className="text-gray-400 text-sm uppercase mb-1">Visual Fidelity</h3>
                <p className="text-2xl font-mono font-bold text-white">{metrics.psnr ? metrics.psnr.toFixed(1) : '0.0'} dB</p>
                <p className="text-xs text-gray-500 mt-2">Human perception score (Higher is better)</p>
              </div>

              <div className="glass-panel p-6 rounded-xl border-l-4 border-l-purple-500">
                <h3 className="text-gray-400 text-sm uppercase mb-1">AI Confusion</h3>
                <p className="text-2xl font-mono font-bold text-white">
                    {metrics.qwen_passed ? '100%' : 'PARTIAL'}
                </p>
                 <p className="text-xs text-gray-500 mt-2">
                    {metrics.qwen_reason || "Identity obfuscated successfully."}
                 </p>
              </div>

              <div className="glass-panel p-6 rounded-xl flex items-center justify-center">
                <a
                  href={protectedUrl}
                  download={`secure_safespace_${Date.now()}.png`}
                  className="flex items-center gap-3 bg-white text-black px-8 py-3 rounded-full font-bold hover:bg-gray-200 transition-colors"
                >
                  <Download size={20} />
                  DOWNLOAD ASSET
                </a>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

      </div>
    </div>
  );
};

export default Dashboard;
