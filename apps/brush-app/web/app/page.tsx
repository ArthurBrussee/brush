'use client';

import { ReadonlyURLSearchParams, useSearchParams } from 'next/navigation';
import { Suspense, lazy } from 'react';
import { Vector3 } from 'three';

const BrushViewer = lazy(() => import('./components/BrushViewer'));

function Loading() {
  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white',
      fontSize: '18px',
    }}>
      Loading Brush WASM...
    </div>
  );
}

function getFloat(searchParams: ReadonlyURLSearchParams, name: string): number | undefined {
  const value = parseFloat(searchParams.get(name) ?? '');
  return isNaN(value) ? undefined : value;
}

function getVector3(searchParams: ReadonlyURLSearchParams, name: string): Vector3 | undefined {
  const value = searchParams.get(name);
  if (!value) {
    return undefined;
  }
  const parts = value.split(',').map(s => parseFloat(s.trim()));
  return parts.length === 3 && parts.every(p => !isNaN(p)) ? new Vector3(parts[0], parts[1], parts[2]) : undefined;
}

function Brush() {
  const searchParams = useSearchParams();
  const url = searchParams.get('url');
  // This mode used to be called "zen" mode, keep it for backwards compatibility.
  const fullsplat = searchParams.get('fullsplat')?.toLowerCase() == 'true' || searchParams.get('zen')?.toLowerCase() == 'true' || false;
  const focusDistance = getFloat(searchParams, 'focus_distance');
  const minFocusDistance = getFloat(searchParams, 'min_focus_distance');
  const maxFocusDistance = getFloat(searchParams, 'max_focus_distance');
  const speedScale = getFloat(searchParams, 'speed_scale');

  let focalPoint = getVector3(searchParams, 'focal_point');
  let cameraRotation = getVector3(searchParams, 'camera_rotation');

  return <BrushViewer
    url={url}
    fullsplat={fullsplat}
    focusDistance={focusDistance}
    minFocusDistance={minFocusDistance}
    maxFocusDistance={maxFocusDistance}
    speedScale={speedScale}
    focalPoint={focalPoint}
    cameraRotation={cameraRotation}
  />;
}

export default function Home() {
  return (
    <Suspense fallback={<Loading />}>
      <Brush />
    </Suspense>
  );
}
