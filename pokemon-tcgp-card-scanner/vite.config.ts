import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '',  // Changed from "/Pokemon-TCGP-Card-Scanner/" to empty string
  publicDir: 'public',
  build: {
    assetsDir: 'assets',
  }
})
