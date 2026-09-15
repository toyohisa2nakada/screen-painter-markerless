// Screen Painter (markerless) - static file server + PeerJS signaling server.
// No inference happens here; everything runs in the browsers.
'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');
const https = require('https');
const express = require('express');
const { ExpressPeerServer } = require('peer');
const QRCode = require('qrcode');

const PORT = Number(process.env.PORT || 8443);
const CERT = process.env.CERT || path.join(__dirname, 'certs', 'cert.pem');
const KEY = process.env.KEY || path.join(__dirname, 'certs', 'key.pem');
const PC_PEER_ID = process.env.PC_PEER_ID || 'screen-painter-pc';

const app = express();

// Cross-origin isolation lets onnxruntime-web use multi-threaded WASM (the CPU fallback).
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cache-Control', 'no-cache');
  next();
});

app.use('/vendor/ort', express.static(path.join(__dirname, 'node_modules/onnxruntime-web/dist')));
app.use('/vendor/peerjs', express.static(path.join(__dirname, 'node_modules/peerjs/dist')));
app.use('/models', express.static(path.join(__dirname, 'models')));
app.use(express.static(path.join(__dirname, 'public')));

function lanAddresses() {
  const out = [];
  const ifaces = os.networkInterfaces();
  for (const name of Object.keys(ifaces)) for (const i of ifaces[name]) {
    if (i.family === 'IPv4' && !i.internal) out.push(i.address);
  }
  return out;
}
app.get('/config.json', (req, res) => res.json({ pcPeerId: PC_PEER_ID, lanAddresses: lanAddresses(), port: PORT }));

app.get('/qr.svg', async (req, res) => {
  const text = String(req.query.text || '');
  const svg = await QRCode.toString(text, { type: 'svg', margin: 1 });
  res.type('image/svg+xml').send(svg);
});

let server;
let scheme;
if (fs.existsSync(CERT) && fs.existsSync(KEY)) {
  server = https.createServer({ cert: fs.readFileSync(CERT), key: fs.readFileSync(KEY) }, app);
  scheme = 'https';
} else {
  server = http.createServer(app);
  scheme = 'http';
  console.warn('[warn] certs/cert.pem or certs/key.pem not found -> serving plain HTTP.');
  console.warn('       The phone camera needs HTTPS (see README: mkcert).');
}

// PeerJS signaling at /peerjs (both pages connect here).
const peerServer = ExpressPeerServer(server, { path: '/', proxied: false });
app.use('/peerjs', peerServer);
peerServer.on('connection', (c) => console.log('[peer] connected', c.getId()));
peerServer.on('disconnect', (c) => console.log('[peer] disconnected', c.getId()));

server.listen(PORT, '0.0.0.0', () => {
  console.log(`\nScreen Painter (markerless) - ${scheme}://localhost:${PORT}`);
  for (const a of lanAddresses()) {
    console.log(`  PC page    : ${scheme}://${a}:${PORT}/pc.html`);
    console.log(`  Phone page : ${scheme}://${a}:${PORT}/phone.html`);
  }
  console.log('');
});
