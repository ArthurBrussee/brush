'use client';

import { useSearchParams } from 'next/navigation';
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

function Brush() {
  const searchParams = useSearchParams();
  const url = searchParams.get('url');
  return <BrushViewer url={url} />;
}

export default function Home() {

  return (
    <Suspense fallback={<Loading />}>
      <Brush />
    </Suspense>
  );
}
