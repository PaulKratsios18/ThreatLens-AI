import { useEffect, useRef } from 'react';
import { Viewer } from '@cesium/widgets';
import { Globe as CesiumGlobe, Cartesian3, Color } from '@cesium/engine';
import { Box } from '@mui/material';

const Globe = () => {
  const viewerRef = useRef<Viewer | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current && !viewerRef.current) {
      // Initialize Cesium viewer
      viewerRef.current = new Viewer(containerRef.current, {
        animation: false,
        baseLayerPicker: false,
        fullscreenButton: false,
        geocoder: false,
        homeButton: false,
        infoBox: false,
        sceneModePicker: false,
        selectionIndicator: false,
        timeline: false,
        navigationHelpButton: false,
        scene3DOnly: true,
      });

      // Configure globe properties
      const globe = viewerRef.current.scene.globe as CesiumGlobe;
      globe.enableLighting = true;
      globe.baseColor = Color.BLACK;
      globe.atmosphereSaturationShift = 0.1;
      globe.atmosphereHueShift = 0.0;
      globe.atmosphereBrightnessShift = -0.1;

      // Set initial camera position
      viewerRef.current.camera.setView({
        destination: Cartesian3.fromDegrees(0, 0, 20000000),
      });
    }

    return () => {
      if (viewerRef.current) {
        viewerRef.current.destroy();
        viewerRef.current = null;
      }
    };
  }, []);

  return (
    <Box
      ref={containerRef}
      sx={{
        width: '100%',
        height: 'calc(100vh - 64px)', // Subtract header height
        '& .cesium-viewer-bottom': {
          display: 'none',
        },
      }}
    />
  );
};

export default Globe; 