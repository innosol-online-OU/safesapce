import React, { useState, useRef, useEffect } from 'react';
import { Upload, Shield, Download, AlertTriangle, CheckCircle, RefreshCw, Trash2, Zap, Cpu, Cloud, Monitor, ChevronDown, ChevronUp } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const Dashboard = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [protectedUrl, setProtectedUrl] = useState(null);
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [strength, setStrength] = useState(50);
  const [optimizationSteps, setOptimizationSteps] = useState(50);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [computeMode, setComputeMode] = useState('local');
  const [systemStatus, setSystemStatus] = useState({ gpu_type: 'Unknown', studio_status: 'n/a' });
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [ghostMeshConfig, setGhostMeshConfig] = useState({
    grid_resolution: 24,
    warp_noise_balance: 0.5,
    tzone_anchoring: 0.8,
    grain_control: 0.3,
    ghost_masking: true
  });

  // Job State
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);
  const [cost, setCost] = useState(0.0);
  const [statusMessage, setStatusMessage] = useState('');

  const fetchStatus = async () => {
    try {
      const res = await axios.get('/api/status');
      if (res.data.compute_mode) setComputeMode(res.data.compute_mode);
      setSystemStatus({
        gpu_type: res.data.gpu_type || 'Unknown',
        studio_status: res.data.studio_status || 'n/a'
      });
    } catch (e) { console.error("Status check failed", e); }
  };

  const fetchGhostMeshConfig = async () => {
    try {
      const res = await axios.get('/api/config/ghost-mesh');
      setGhostMeshConfig(res.data);
    } catch (e) { console.error("Failed to load Ghost Mesh config", e); }
  };

  const updateGhostMeshConfig = async (newConfig) => {
    try {
      const token = localStorage.getItem('token');
      await axios.post('/api/config/ghost-mesh', newConfig, {
        headers: { Authorization: `Bearer ${token}` }
      });
      setGhostMeshConfig(newConfig);
    } catch (e) {
      console.error('Failed to update Ghost Mesh config', e);
    }
  };

  const fetchHistory = async () => {
    try {
      const token = localStorage.getItem('token');
      const res = await axios.get('/api/history', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setHistory(res.data);
    } catch (e) {
      console.error("Failed to load history", e);
    }
  };

  const deleteHistoryItem = async (recordId) => {
    if (!window.confirm('Delete this history item?')) return;
    try {
      const token = localStorage.getItem('token');
      await axios.delete(`/api/history/${recordId}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchHistory(); // Refresh
    } catch (e) {
      console.error('Failed to delete history item', e);
    }
  };

  useEffect(() => {
    fetchHistory();
    fetchStatus();
    fetchGhostMeshConfig();

    // Poll status every 10 seconds
    const statusInterval = setInterval(fetchStatus, 10000);
    return () => clearInterval(statusInterval);
  }, []);

  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setProtectedUrl(null);
      setHeatmapUrl(null);
      setMetrics(null);
    }
  };

  // Auto-adjust optimization steps based on strength
  useEffect(() => {
    if (strength < 30) {
      setOptimizationSteps(30);
    } else if (strength > 80) {
      setOptimizationSteps(80);
    } else {
      setOptimizationSteps(50);
    }
  }, [strength]);

  const handleProtect = async () => {
    if (!file) return;
    setLoading(true);
    setHeatmapUrl(null);
    setError('');
    setProgress(0);
    setCost(0.0);
    setStatusMessage("Initializing...");
    setProtectedUrl(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('strength', strength);

    try {
      const token = localStorage.getItem('token');
      // Start Job
      const res = await axios.post('/api/protect', formData, {
        headers: { Authorization: `Bearer ${token}` }
      });

      const { job_id } = res.data;
      setJobId(job_id);

      // Start Polling
      const pollInterval = setInterval(async () => {
        try {
          const jobRes = await axios.get(`/api/jobs/${job_id}`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          const job = jobRes.data;

          setProgress(job.progress);
          setStatusMessage(job.message);
          setCost(job.cost_estimate || 0.0);

          // Update intermediate preview if available and running
          if (job.status === 'running' && job.intermediate_url) {
            // Add timestamp to prevent caching
            setProtectedUrl(`${job.intermediate_url}?t=${Date.now()}`);
          }

          if (job.status === 'completed') {
            clearInterval(pollInterval);
            setLoading(false);
            setProtectedUrl(job.output_url);
            setHeatmapUrl(job.heatmap_url);
            setMetrics(job.metrics);
            fetchHistory();
          } else if (job.status === 'failed') {
            clearInterval(pollInterval);
            setLoading(false);
            setError(job.message || "Protection Failed");
          }
        } catch (e) {
          console.error("Polling error", e);
          // Don't stop polling immediately on one error, network might be flaky
        }
      }, 2000);

    } catch (err) {
      console.error('[Dashboard] Protection Error:', err.response?.data || err.message);
      setError(err.response?.data?.detail || 'Protection failed to start.');
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
          {/* Enhanced System Status Badge */}
          <div className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-bold border transition-all ${computeMode === 'cloud'
            ? 'bg-purple-900/40 border-purple-500 text-purple-300 shadow-lg shadow-purple-500/20'
            : systemStatus.gpu_type !== 'CPU' && systemStatus.gpu_type !== 'Unknown'
              ? 'bg-blue-900/40 border-blue-500 text-blue-300 shadow-lg shadow-blue-500/20'
              : 'bg-gray-800 border-gray-600 text-gray-400'
            }`}>
            {computeMode === 'cloud' ? (
              <>
                <Cloud size={16} className="fill-current" />
                <span>CLOUD GPU ({systemStatus.gpu_type})</span>
              </>
            ) : systemStatus.gpu_type !== 'CPU' && systemStatus.gpu_type !== 'Unknown' ? (
              <>
                <Monitor size={16} />
                <span>LOCAL GPU ({systemStatus.gpu_type})</span>
              </>
            ) : (
              <>
                <Cpu size={16} />
                <span>CPU MODE</span>
              </>
            )}
          </div>
          <button className="bg-white/10 hover:bg-white/20 px-4 py-2 rounded-lg text-sm transition-colors">
            Connect Wallet
          </button>
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
                <div className="flex flex-col items-center w-full px-8">
                  <RefreshCw className="animate-spin text-neon mb-4" size={48} />
                  <p className="text-neon animate-pulse font-bold mb-2">{statusMessage}</p>

                  {/* Progress Bar */}
                  <div className="w-full bg-gray-700 h-2 rounded-full mb-2 overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${progress}%` }}
                      className="bg-neon h-full"
                    />
                  </div>
                  <div className="flex justify-between w-full text-xs font-mono text-gray-400">
                    <span>{progress}%</span>
                    {cost > 0 && <span className="text-yellow-400">Est. Cost: ${cost.toFixed(4)}</span>}
                  </div>

                  {protectedUrl && (
                    <div className="mt-4 relative w-full h-32 rounded-lg overflow-hidden border border-white/10">
                      <img src={protectedUrl} className="w-full h-full object-cover opacity-50" alt="Processing..." />
                      <span className="absolute bottom-1 right-1 text-[10px] bg-black/60 px-2 rounded text-white">INTERMEDIATE</span>
                    </div>
                  )}
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

              {/* Optimization Steps Info */}
              <div className="mt-4 flex items-center justify-between bg-black/30 px-4 py-2 rounded-lg border border-neon/20">
                <span className="text-xs font-mono text-gray-400">OPTIMIZATION STEPS</span>
                <span className="text-xs text-neon font-bold font-mono">{optimizationSteps} iterations</span>
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

          {/* Advanced Settings - Ghost Mesh Controls */}
          <div className="mt-6 border-t border-white/10 pt-6">
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-sm font-mono text-gray-400 hover:text-neon transition-colors mb-4"
            >
              {showAdvanced ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
              <span>ADVANCED SETTINGS (Phase 18: Ghost Mesh)</span>
            </button>

            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="grid grid-cols-1 md:grid-cols-2 gap-6 overflow-hidden"
                >
                  {/* Grid Resolution */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-xs font-mono text-gray-400">GRID RESOLUTION</span>
                      <span className="text-xs text-neon font-mono">{ghostMeshConfig.grid_resolution}</span>
                    </div>
                    <input
                      type="range"
                      min="12"
                      max="48"
                      step="4"
                      value={ghostMeshConfig.grid_resolution}
                      onChange={(e) => {
                        const newConfig = { ...ghostMeshConfig, grid_resolution: Number(e.target.value) };
                        setGhostMeshConfig(newConfig);
                        updateGhostMeshConfig(newConfig);
                      }}
                      className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-gray-600 mt-1">
                      <span>12</span>
                      <span>48</span>
                    </div>
                  </div>

                  {/* Warp/Noise Balance */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-xs font-mono text-gray-400">WARP/NOISE BALANCE</span>
                      <span className="text-xs text-neon font-mono">{ghostMeshConfig.warp_noise_balance.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={ghostMeshConfig.warp_noise_balance}
                      onChange={(e) => {
                        const newConfig = { ...ghostMeshConfig, warp_noise_balance: Number(e.target.value) };
                        setGhostMeshConfig(newConfig);
                        updateGhostMeshConfig(newConfig);
                      }}
                      className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-gray-600 mt-1">
                      <span>Warp</span>
                      <span>Noise</span>
                    </div>
                  </div>

                  {/* T-Zone Anchoring */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-xs font-mono text-gray-400">T-ZONE ANCHORING</span>
                      <span className="text-xs text-neon font-mono">{ghostMeshConfig.tzone_anchoring.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={ghostMeshConfig.tzone_anchoring}
                      onChange={(e) => {
                        const newConfig = { ...ghostMeshConfig, tzone_anchoring: Number(e.target.value) };
                        setGhostMeshConfig(newConfig);
                        updateGhostMeshConfig(newConfig);
                      }}
                      className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  {/* Grain Control */}
                  <div>
                    <div className="flex justify-between mb-2">
                      <span className="text-xs font-mono text-gray-400">GRAIN CONTROL</span>
                      <span className="text-xs text-neon font-mono">{ghostMeshConfig.grain_control.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={ghostMeshConfig.grain_control}
                      onChange={(e) => {
                        const newConfig = { ...ghostMeshConfig, grain_control: Number(e.target.value) };
                        setGhostMeshConfig(newConfig);
                        updateGhostMeshConfig(newConfig);
                      }}
                      className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>

                  {/* Ghost Masking Toggle */}
                  <div className="flex items-center justify-between col-span-2 bg-black/30 p-4 rounded-lg border border-white/5">
                    <span className="text-xs font-mono text-gray-400">GHOST MASKING</span>
                    <button
                      onClick={() => {
                        const newConfig = { ...ghostMeshConfig, ghost_masking: !ghostMeshConfig.ghost_masking };
                        setGhostMeshConfig(newConfig);
                        updateGhostMeshConfig(newConfig);
                      }}
                      className={`relative w-14 h-7 rounded-full transition-all ${ghostMeshConfig.ghost_masking ? 'bg-neon' : 'bg-gray-700'
                        }`}
                    >
                      <div
                        className={`absolute top-1 left-1 w-5 h-5 bg-black rounded-full transition-transform ${ghostMeshConfig.ghost_masking ? 'translate-x-7' : ''
                          }`}
                      />
                    </button>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
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

        {/* Heatmap Visualization */}
        <AnimatePresence>
          {heatmapUrl && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="glass-panel p-8 rounded-2xl mb-10"
            >
              <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 rounded-full bg-purple-900/40 border border-purple-500 flex items-center justify-center">
                  <AlertTriangle className="text-purple-300" size={20} />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-neon">Debug Visualizations</h3>
                  <p className="text-sm text-gray-400 font-mono">Ghost-Mesh Perturbation Analysis</p>
                </div>
              </div>

              <div className="relative">
                <img
                  src={heatmapUrl}
                  alt="Protection Heatmap"
                  className="w-full h-auto rounded-lg border border-neon/30"
                />
                <div className="absolute top-4 right-4 bg-black/80 px-3 py-2 rounded-lg border border-neon/20">
                  <span className="text-xs text-neon font-mono">DIFFERENCE MAP</span>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                  <p className="text-xs text-gray-500 font-mono mb-1">WARP FIELD</p>
                  <p className="text-sm text-blue-400 font-bold">Geometric</p>
                </div>
                <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                  <p className="text-xs text-gray-500 font-mono mb-1">NOISE MAP</p>
                  <p className="text-sm text-purple-400 font-bold">Pixel Delta</p>
                </div>
                <div className="bg-black/30 p-3 rounded-lg border border-white/5">
                  <p className="text-xs text-gray-500 font-mono mb-1">COMBINED</p>
                  <p className="text-sm text-neon font-bold">Ghost Mesh</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* --- HISTORY SECTION --- */}
        {history.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="border-t border-gray-800 pt-10 mb-10"
          >
            <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
              <RefreshCw className="text-neon" size={24} />
              PROTECTION HISTORY
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {history.map((record, idx) => (
                <div key={idx} className="glass-panel p-4 rounded-xl hover:bg-white/5 transition-colors">
                  <div className="text-xs text-gray-500 mb-2 flex justify-between">
                    <span>{new Date(record.timestamp).toLocaleString()}</span>
                    <span className="text-neon font-bold">{record.config?.strength}%</span>
                  </div>
                  <div className="flex gap-2 h-32 mb-3">
                    <div className="w-1/2 relative bg-black/40 rounded overflow-hidden">
                      <span className="absolute top-1 left-1 bg-black/60 px-1 py-0.5 text-[10px] rounded text-white z-10">ORIG</span>
                      <img src={record.original} className="w-full h-full object-cover opacity-80 hover:opacity-100 transition-opacity" alt="Original" />
                    </div>
                    <div className="w-1/2 relative bg-black/40 rounded overflow-hidden">
                      <span className="absolute top-1 left-1 bg-neon/20 text-neon px-1 py-0.5 text-[10px] rounded z-10 border border-neon/30">SECURE</span>
                      <img src={record.protected} className="w-full h-full object-cover opacity-80 hover:opacity-100 transition-opacity" alt="Protected" />
                    </div>
                  </div>
                  <div className="flex justify-between items-center text-xs border-t border-white/5 pt-2">
                    <div className="text-gray-400">
                      PSNR: <span className="text-white">{record.metrics?.psnr?.toFixed(1) || '0.0'}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <a
                        href={record.protected}
                        download={`historical_${idx}.png`}
                        className="text-neon hover:text-white transition-colors flex items-center gap-1 font-bold"
                      >
                        GET ASSET <Download size={10} />
                      </a>
                      <button
                        onClick={() => deleteHistoryItem(record.id)}
                        className="text-red-500 hover:text-red-300 transition-colors"
                        title="Delete"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

      </div>
    </div>
  );
};

export default Dashboard;
