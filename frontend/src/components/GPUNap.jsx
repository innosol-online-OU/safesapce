import React from 'react';

const GPUNap = ({ onRetry }) => {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-90 backdrop-blur-sm p-4">
            <div className="bg-gray-900 border border-neon/50 rounded-2xl p-8 max-w-md text-center shadow-[0_0_30px_rgba(57,255,20,0.2)]">
                <div className="mb-6 flex justify-center">
                    {/* Sleeping Robot / Zzz Icon */}
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-24 w-24 text-neon animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                    </svg>
                </div>

                <h2 className="text-2xl font-bold text-white mb-4 tracking-wider">System Offline</h2>

                <p className="text-gray-300 text-lg mb-6 leading-relaxed">
                    The GPU varies between being suspended and unshelved, so it is
                    <span className="text-neon font-bold ml-1">taking a nap</span>.
                </p>

                <p className="text-sm text-gray-500 mb-8 italic">
                    Check back in a bit!
                </p>

                <button
                    onClick={onRetry}
                    className="px-6 py-3 bg-neon text-black font-bold rounded-lg hover:bg-green-400 transition-all transform hover:scale-105 shadow-lg shadow-neon/20"
                >
                    Wake Up Check
                </button>
            </div>
        </div>
    );
};

export default GPUNap;
