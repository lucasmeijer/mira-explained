import { defineConfig } from 'vite';
import { resolve } from 'path';
const pages = ['index', 'ch1-codec', 'ch2-flow', 'ch3-forcing', 'ch4-transformer', 'ch5-multiplayer', 'ch6-realtime', 'ch7-evidence', 'ch8-training'];
export default defineConfig({
  server: { host: '0.0.0.0', port: 5173, allowedHosts: true },
  preview: { host: '0.0.0.0', port: 5173, allowedHosts: true },
  build: { rollupOptions: { input: Object.fromEntries(pages.map((p) => [p, resolve(__dirname, `${p}.html`)])) } },
});
