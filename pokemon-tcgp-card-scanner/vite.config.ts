import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command }) => {
  const config = {
    plugins: [react()],
    base: command === 'serve' ? '' : '/Pokemon-TCGP-Card-Scanner/',
    publicDir: 'public',
    build: {
      assetsDir: 'assets',
    }
  }
  
  console.log('Vite config:', config);
  return config;
})
