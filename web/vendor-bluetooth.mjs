// Compile the pinned, unmodified Boardsesh protocol sources for the browser.
// TypeScript 7 ships only the native compiler, so this drives its tsc CLI.
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const root = path.dirname(fileURLToPath(import.meta.url));
const directory = path.join(root, 'public/vendor/boardsesh/bluetooth');
const names = ['transport', 'aurora', 'web-transport'];
const output = await mkdtemp(path.join(tmpdir(), 'boardsesh-ble-'));
try {
  // --noCheck: the @boardsesh/* imports are rewritten below, not resolved.
  await promisify(execFile)(process.execPath, [path.join(root, 'node_modules/typescript/bin/tsc'),
    '--target', 'es2022', '--module', 'es2022', '--moduleResolution', 'bundler',
    '--noCheck', '--skipLibCheck', '--outDir', output,
    ...names.map(name => path.join(directory, `${name}.ts`))]);
  for (const name of names) {
    const browserCode = (await readFile(path.join(output, `${name}.js`), 'utf8'))
      .replaceAll("from './transport'", "from './transport.mjs'")
      .replaceAll("from '@boardsesh/board-constants/hold-states'", "from './kilter-constants.mjs'")
      .replaceAll("from '@boardsesh/board-constants/moonboard'", "from './kilter-constants.mjs'");
    if (/from '(?!\.\/[\w-]+\.mjs')/.test(browserCode)) throw new Error(`Unresolved import in ${name}.ts`);
    await writeFile(path.join(directory, `${name}.mjs`),
      '// Generated from Boardsesh 15f6a8588585992725d9f9ba181271a0de9165f9 (Apache-2.0) by vendor-bluetooth.mjs.\n'
      + browserCode);
  }
} catch (error) {
  throw new Error(`Could not compile the Boardsesh Bluetooth sources: ${error.stderr || error.stdout || error.message}`);
} finally {
  await rm(output, { recursive: true, force: true });
}
console.log('Compiled pinned Boardsesh Bluetooth sources.');
