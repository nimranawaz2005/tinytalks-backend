// Add this "server.proxy" block to your existing vite.config.js
// (the file that sits next to your App.jsx / package.json for the frontend).
//
// This lets your frontend code keep calling the relative path
// "/api/cry-detect" (as already written in App.jsx) while Vite silently
// forwards those requests to your backend running on port 3001.
// No CORS headaches, no hardcoded backend URL in the frontend bundle.

import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:3001',
        changeOrigin: true,
      },
    },
  },
});
