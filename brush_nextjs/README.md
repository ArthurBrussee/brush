# Brush WASM Next.js Demo

A minimal Next.js application demonstrating 3D Gaussian Splatting using the Brush WASM library.

## Features

- 🎨 Real-time 3D Gaussian Splatting rendering
- 🖱️ Interactive camera controls (orbit, zoom, pan)
- 🌐 Load .splat and .ply files from URLs
- ⚡ Built with Next.js 14 and TypeScript
- 📱 Responsive design

## Prerequisites

- Node.js 18+
- Rust toolchain with `wasm-pack`
- Git

## Quick Start

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Build the WASM module and start development server:**
   ```bash
   npm run dev
   ```

3. **Open your browser:**
   Navigate to [http://localhost:3000](http://localhost:3000)

## Static Export

This app can be built as a static website for deployment:

1. **Build static export:**
   ```bash
   npm run export
   ```

2. **Serve locally (optional):**
   ```bash
   npm run serve
   ```

The static files will be generated in the `out/` directory and can be deployed to any static hosting service.

## Build Commands

- `npm run dev` - Start development server (with Turbopack)
- `npm run build` - Build WASM module and Next.js app for production (with Turbopack)
- `npm run export` - Build and export as static website
- `npm run serve` - Serve the static export locally
- `npm run build:wasm` - Build only the WASM module
- `npm run copy:wasm` - Copy WASM files to local pkg directory
- `npm run start` - Start production server

## Usage

### Demo Scenes
The app includes several pre-configured demo scenes:
- **Bicycle** - Classic bicycle scene
- **Garden** - Outdoor garden environment
- **Stump** - Tree stump close-up

### Custom URLs
Load your own 3D Gaussian Splat files by entering a URL to:
- `.splat` files (native format)
- `.ply` files (point cloud format)

### Camera Controls
- **Mouse drag**: Rotate camera around the scene
- **Mouse wheel**: Zoom in and out
- **Right click + drag**: Pan the camera

## Technical Details

### Architecture
- **Frontend**: Next.js 15 with TypeScript and Turbopack
- **3D Rendering**: Brush WASM (Rust + WebAssembly)
- **Build Process**: Automated WASM compilation with `wasm-pack`
- **Static Export**: Built-in Next.js static site generation

### WASM Integration
The app uses a top-level import to load the WASM module with proper TypeScript support:

```typescript
import * as BrushWasm from '../../pkg/brush_app';
```

### Build Pipeline
The build process automatically compiles and bundles the Rust WASM module:
1. `wasm-pack build --target bundler` compiles Rust to WASM for bundler compatibility
2. WASM files are copied to local `pkg/` directory
3. Next.js 15 with Turbopack bundles everything together
4. Static export generates deployable HTML/JS/WASM files

## File Structure

```
Brush/examples/nextjs-demo/
├── app/
│   ├── components/
│   │   └── BrushViewer.tsx     # Main WASM viewer component
│   ├── layout.tsx              # Root layout
│   └── page.tsx                # Home page with demo UI
├── next.config.js              # Next.js config with WASM support
├── package.json                # Dependencies and scripts
├── tsconfig.json               # TypeScript configuration
└── README.md                   # This file
```

## Troubleshooting

### WASM Module Not Found
If you get "Module not found" errors:
```bash
npm run build:wasm
```

### Build Fails
Ensure you have the Rust toolchain and wasm-pack installed:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install wasm-pack
```

### Canvas Not Rendering
Check browser console for WebGL errors. Brush requires WebGL 2.0 support.

## Performance Tips

- Use `.splat` format for best performance (smaller file size)
- Large files may take time to download and parse
- Consider hosting files on a CDN for faster loading
- Static export enables deployment to CDNs and edge networks for global performance

## Deployment

The static export in the `out/` directory can be deployed to:
- **Vercel**: `vercel --prod out/`
- **Netlify**: Drag and drop the `out/` folder
- **GitHub Pages**: Upload contents of `out/` to your repository
- **Any static hosting**: Upload the `out/` directory contents

## Links

- [Brush GitHub Repository](https://github.com/ArthurBrussee/brush)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Next.js Documentation](https://nextjs.org/docs)

## License

This demo is part of the Brush project and uses the same Apache 2.0 license.
