# Brush WASM Next.js Demo

A minimal Next.js application demonstrating 3D Gaussian Splatting using the Brush WASM library.

## Quick Start

1. **Install dependencies:**
```bash
npm install
```

2. **Build the WASM module and start a development server:**
```bash
npm run dev
```

Or for release mode:
```bash
npm run release
```

## Static Export

This app can be built as a static website for deployment:

**Build static export:**
```bash
npm run build
```

The static files will be generated in the `out/` directory and can be deployed to any static hosting service.
