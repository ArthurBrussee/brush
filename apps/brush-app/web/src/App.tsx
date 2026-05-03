import { Suspense, lazy } from 'react';
import { Vector3 } from 'three';

const BrushViewer = lazy(() => import('./BrushViewer'));

function Loading() {
  return (
    <div
      style={{
        width: '100vw',
        height: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: 'white',
        fontSize: '18px',
      }}
    >
      Loading Brush WASM...
    </div>
  );
}

function getFloat(params: URLSearchParams, name: string): number | undefined {
  const value = parseFloat(params.get(name) ?? '');
  return Number.isNaN(value) ? undefined : value;
}

function getVector3(params: URLSearchParams, name: string): Vector3 | undefined {
  const value = params.get(name);
  if (!value) return undefined;
  const parts = value.split(',').map((s) => parseFloat(s.trim()));
  return parts.length === 3 && parts.every((p) => !Number.isNaN(p))
    ? new Vector3(parts[0], parts[1], parts[2])
    : undefined;
}

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const url = params.get('url');
  // This mode used to be called "zen" mode; keep it for backwards compatibility.
  const fullsplat =
    params.get('fullsplat')?.toLowerCase() === 'true' ||
    params.get('zen')?.toLowerCase() === 'true' ||
    false;
  const focusDistance = getFloat(params, 'focus_distance');
  const minFocusDistance = getFloat(params, 'min_focus_distance');
  const maxFocusDistance = getFloat(params, 'max_focus_distance');
  const speedScale = getFloat(params, 'speed_scale');
  const focalPoint = getVector3(params, 'focal_point');
  const cameraRotation = getVector3(params, 'camera_rotation');

  return (
    <Suspense fallback={<Loading />}>
      <BrushViewer
        url={url}
        fullsplat={fullsplat}
        focusDistance={focusDistance}
        minFocusDistance={minFocusDistance}
        maxFocusDistance={maxFocusDistance}
        speedScale={speedScale}
        focalPoint={focalPoint}
        cameraRotation={cameraRotation}
      />
    </Suspense>
  );
}
