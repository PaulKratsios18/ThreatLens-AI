import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    exclude: ['@cesium/engine', '@cesium/widgets']
  },
  build: {
    rollupOptions: {
      external: [
        'Workers/createVerticesFromHeightmap.js',
        'Workers/transferTypedArrayTest.js'
      ]
    }
  }
})
