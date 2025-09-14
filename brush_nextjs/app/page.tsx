'use client';

import { ReadonlyURLSearchParams, useSearchParams } from 'next/navigation';
import { Suspense, lazy } from 'react';

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

function Brush() {
  const searchParams = useSearchParams();
  const url = searchParams.get('url');
  // This mode used to be called "zen" mode, keep it for backwards compatibility.
  const fullsplat = searchParams.get('fullsplat')?.toLowerCase() == 'true' || searchParams.get('zen')?.toLowerCase() == 'true' || false;
  const focusDistance = getFloat(searchParams, 'focus_distance');
  const minFocusDistance = getFloat(searchParams, 'min_focus_distance');
  const maxFocusDistance = getFloat(searchParams, 'max_focus_distance');
  const speedScale = getFloat(searchParams, 'speed_scale');

  return <BrushViewer
    url={url}
    fullsplat={fullsplat}
    focusDistance={focusDistance}
    minFocusDistance={minFocusDistance}
    maxFocusDistance={maxFocusDistance}
    speedScale={speedScale}
  />;
}

export default function Home() {
  return (
    <Suspense fallback={<Loading />}>
      <Brush />
    </Suspense>
  );
}
