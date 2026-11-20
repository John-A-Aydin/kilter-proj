// Rebuilds dist/ whenever a site source changes, for use with Live Server (or `npm start`).
import { watch } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const web = path.dirname(fileURLToPath(import.meta.url));
const repo = path.dirname(web);
const generated = path.join(web, 'public/vendor/boardsesh/bluetooth') + path.sep;
const sources = [
  'src', 'public', 'index.html', 'how.html', 'label.html', 'experiments.html',
  'experiments-page.mjs', 'build.mjs', '../experiments/README.md', '../results',
].map(source => path.resolve(web, source));

let running = false;
let pending = false;
let timer;

function build() {
  if (running) { pending = true; return; }
  running = true;
  const started = Date.now();
  const child = spawn(process.execPath, ['build.mjs'], { cwd: web, stdio: ['ignore', 'ignore', 'inherit'] });
  child.on('exit', code => {
    running = false;
    const time = new Date().toLocaleTimeString();
    console.log(code === 0 ? `[${time}] Rebuilt in ${Date.now() - started} ms` : `[${time}] Build failed (exit ${code})`);
    if (pending) { pending = false; build(); }
  });
}

for (const source of sources) {
  watch(source, { recursive: true }, (_event, file) => {
    if (file && /(^|\/)(\.|.*~$)/.test(file)) return; // editor temp files
    // build.mjs regenerates these compiled Bluetooth modules on every run.
    if (file && path.join(source, file).endsWith('.mjs') && path.join(source, file).startsWith(generated)) return;
    clearTimeout(timer);
    timer = setTimeout(build, 150);
  });
}

console.log(`Watching ${sources.map(source => path.relative(repo, source)).join(', ')}`);
build();
