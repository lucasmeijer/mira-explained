// usage: node scripts/shot.mjs <page> <out.png> [jsFile-with-async-steps] [width]
import { chromium } from 'playwright';
import fs from 'fs';
import { execSync } from 'child_process';
const [page_, out, steps, width] = process.argv.slice(2);
const exe = execSync(`find ${process.env.PLAYWRIGHT_BROWSERS_PATH || '/ms-playwright'} -path '*/chrome-linux*/chrome' -type f | sort | tail -n 1`).toString().trim();
const browser = await chromium.launch({ executablePath: exe, args: ['--enable-unsafe-webgpu', '--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: +(width || 1280), height: 900 }, deviceScaleFactor: 1 });
const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
await page.goto(`${process.env.BASE || 'http://localhost:5173'}/${page_}`, { waitUntil: 'networkidle' });
if (steps) { const fn = new Function('page', `return (async () => { ${fs.readFileSync(steps, 'utf8')} })()`); await fn(page); }
await page.screenshot({ path: out, fullPage: !process.env.NOFULL });
console.log(logs.join('\n'));
await browser.close();
