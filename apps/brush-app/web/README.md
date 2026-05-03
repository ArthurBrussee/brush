# Brush WASM web demo

The `brush-app` egui viewer running in the browser via WebAssembly. Vite + React.

```bash
npm install
npm run dev      # localhost:5173 — auto-rebuilds wasm before starting
npm run build    # static build under dist/, basepath /brush-demo
```

URL params (all optional):
- `url=…` — load a `.ply` / dataset URL on start
- `fullsplat=true` (legacy alias `zen=true`) — embedded viewer mode
- `focal_point=x,y,z`, `camera_rotation=x,y,z`, `focus_distance`, `min_focus_distance`, `max_focus_distance`, `speed_scale`
