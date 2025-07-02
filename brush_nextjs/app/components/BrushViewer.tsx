'use client';

import { useEffect, useRef, useState } from 'react';
import { EmbeddedApp } from '../../pkg/brush_app';

interface BrushViewerProps {
  url?: string | null;
}

export default function BrushViewer({
  url = ''
}: BrushViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [app, setApp] = useState<EmbeddedApp | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvasId = `brush-canvas-${Date.now()}`;
    canvasRef.current.id = canvasId;

    try {
      const brushApp = new EmbeddedApp(canvasId, url ? `?url=${encodeURIComponent(url)}` : '');
      setApp(brushApp);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    }
  }, []);

  useEffect(() => {
    if (app && url) {
      app.load_url(url);
    }
  }, [app, url]);

  return (
    <div style={{
      width: '100vw',
      height: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
    }}>
      {error ? (
        <div style={{ color: '#ff6b6b' }}>Error: {error}</div>
      ) : (
        <canvas
          ref={canvasRef}
          style={{
            width: '100%',
            height: '100%',
            display: 'block'
          }}
        />
      )}
    </div>
  );
}
