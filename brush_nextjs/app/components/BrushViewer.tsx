'use client';

import { useEffect, useRef, useState } from 'react';
import { CameraSettings, EmbeddedApp, UiMode } from '../../pkg/brush_app';
import { Vector3 } from 'three';

interface BrushViewerProps {
  url?: string | null;
  fullsplat?: boolean;
  focusDistance?: number;
  minFocusDistance?: number;
  maxFocusDistance?: number;
  speedScale?: number;
  focalPoint?: Vector3;
  cameraRotation?: Vector3;
}

export default function BrushViewer(props: BrushViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [app, setApp] = useState<EmbeddedApp | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvasId = `brush-canvas-${Date.now()}`;
    canvasRef.current.id = canvasId;

    const initApp = async () => {
      try {
        const brushApp = new EmbeddedApp();
        await brushApp.start(canvasId);
        setApp(brushApp);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    };

    initApp();
  }, []);

  useEffect(() => {
    if (app && props.url) {
      app.load_url(props.url);
    }
  }, [app, props.url]);

  useEffect(() => {
    if (app && props.fullsplat) {
      app.set_ui_mode(props.fullsplat ? UiMode.FullScreenSplat : UiMode.Default);
    }
  }, [app, props.url, props.fullsplat]);

  useEffect(() => {
    if (app) {
      app.set_cam_settings(new CameraSettings(
        undefined, // background,
        props.speedScale,
        props.minFocusDistance,
        props.maxFocusDistance,
        undefined, // min_pitch,
        undefined, // max_pitch
        undefined, // min_yaw,
        undefined, // max_yaw
        undefined, // splat_scale
      ));
    }
  }, [app, props.url, props.speedScale, props.minFocusDistance, props.maxFocusDistance]);

  useEffect(() => {
    if (app) {
      const focalPoint = props.focalPoint ?? new Vector3(0, 0, 0);
      const focalDistance = props.focusDistance ?? 2.5;
      const cameraRotation = props.cameraRotation ?? new Vector3(0, 0, 0);
      app.set_focal_point(focalPoint, focalDistance, cameraRotation);
    }
  }, [app, props.url, props.focalPoint, props.focusDistance, props.cameraRotation]);

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
