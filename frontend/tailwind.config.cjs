/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        neon: '#39ff14',
        dark: '#0a0a0f',
        glass: 'rgba(255,255,255,0.05)'
      },
    },
  },
  plugins: [],
}
