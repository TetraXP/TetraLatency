#!/usr/bin/env node
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const pyScript = path.join(__dirname, '..', 'tlate.py');

// Spawning python3 and inheriting the terminal standard IO streams
const args = process.argv.slice(2);
const child = spawn('python3', [pyScript, ...args], { stdio: 'inherit' });

child.on('exit', (code) => {
    process.exit(code);
});
