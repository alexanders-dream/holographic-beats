/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { ImageSegmenter, FilesetResolver } from '@mediapipe/tasks-vision';
import { motion, AnimatePresence } from 'framer-motion';
import { Camera, Loader2, AlertCircle, Settings2, Volume2, VolumeX } from 'lucide-react';

// --- Shader Code ---
const vertexShader = `
uniform sampler2D uVideoTexture;
uniform sampler2D uMaskTexture;
uniform float uTime;
uniform float uInvertMask;
uniform float uAudioLevel;

varying vec2 vUv;
varying float vVisible;
varying vec3 vColor;
varying float vIsBackground;

void main() {
    vUv = uv;
    
    // Mirror X for webcam feel
    vec2 mirroredUv = vec2(1.0 - vUv.x, vUv.y);
    
    // Sample mask. DataTexture (mask) has flipY=false, VideoTexture has flipY=true.
    // So we flip Y for the mask to align them.
    vec2 maskUv = vec2(mirroredUv.x, 1.0 - mirroredUv.y);
    vec4 maskData = texture2D(uMaskTexture, maskUv);
    
    // Check mask value
    float maskVal = maskData.r * 255.0;
    float isPerson = maskVal > 0.5 ? 1.0 : 0.0;
    
    // Invert mask if needed (fixes the issue where background is rendered instead of person)
    if (uInvertMask > 0.5) {
        isPerson = 1.0 - isPerson;
    }
    
    // Sample video for color and brightness
    vec4 videoData = texture2D(uVideoTexture, mirroredUv);
    float brightness = dot(videoData.rgb, vec3(0.299, 0.587, 0.114));
    
    vVisible = isPerson;
    vIsBackground = 1.0 - isPerson;
    
    vec3 newPosition = position;
    
    if (isPerson > 0.5) {
        // Person: displace based on brightness to create 3D shape
        float audioDisplacement = uAudioLevel * brightness * 3.0;
        newPosition.z += brightness * 1.5 + audioDisplacement;
        
        // Add subtle floating animation
        newPosition.y += sin(uTime * 1.5 + position.x * 3.0) * 0.02;
        newPosition.x += cos(uTime * 1.5 + position.y * 3.0) * 0.02;
        
        // Matrix/Hologram color theme with enhanced audio reaction
        vec3 baseColor = mix(vec3(0.0, 0.8, 1.0), vec3(0.5, 0.0, 1.0), uAudioLevel * 2.0); // Cyan to Purple
        vec3 highlightColor = mix(vec3(0.8, 1.0, 1.0), vec3(1.0, 0.2, 0.8), uAudioLevel * 1.5); // White to Pink
        
        // Add time-based color shifting when audio is loud
        vec3 timeColor = vec3(sin(uTime * 2.0) * 0.5 + 0.5, cos(uTime * 1.5) * 0.5 + 0.5, sin(uTime * 1.2) * 0.5 + 0.5);
        baseColor = mix(baseColor, timeColor, uAudioLevel * 0.8);
        
        vColor = mix(baseColor, highlightColor, brightness);
    } else {
        // Background: scatter points to look like data rain
        float rand = fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
        
        if (rand > 0.95) {
            vVisible = 1.0;
            newPosition.z -= rand * 4.0 + 1.0 + (uAudioLevel * rand * 5.0); // Push back
            newPosition.y -= mod(uTime * (rand * 1.0 + 0.5 + uAudioLevel * 2.0), 4.0); // Fall down
            if (newPosition.y < -2.0) newPosition.y += 4.0; // Wrap around
            vColor = mix(vec3(0.0, 0.4, 0.6), vec3(0.0, 0.8, 1.0), uAudioLevel); // Dimmer cyan
        } else {
            vVisible = 0.0;
        }
    }
    
    vec4 mvPosition = modelViewMatrix * vec4(newPosition, 1.0);
    
    if (isPerson > 0.5) {
        gl_PointSize = (3.0 + brightness * 2.0 + uAudioLevel * 8.0) * (1.0 / -mvPosition.z);
    } else {
        gl_PointSize = (2.0 + uAudioLevel * 3.0) * (1.0 / -mvPosition.z);
    }
    
    gl_Position = projectionMatrix * mvPosition;
}
`;

const fragmentShader = `
varying vec2 vUv;
varying float vVisible;
varying vec3 vColor;
varying float vIsBackground;

void main() {
    if (vVisible < 0.5) discard;
    
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) discard;
    
    // Soft glowing edge
    float alpha = 1.0 - (dist * 2.0);
    alpha = pow(alpha, 1.5); // More exponential glow
    
    if (vIsBackground > 0.5) {
        gl_FragColor = vec4(vColor, alpha * 0.5);
    } else {
        gl_FragColor = vec4(vColor, alpha * 0.9);
    }
}
`;

// --- Components ---

const PointCloud = ({ 
  videoElement, 
  segmenter, 
  invertMask,
  analyserRef,
  dataArrayRef
}: { 
  videoElement: HTMLVideoElement, 
  segmenter: ImageSegmenter, 
  invertMask: boolean,
  analyserRef: React.MutableRefObject<AnalyserNode | null>,
  dataArrayRef: React.MutableRefObject<Uint8Array<ArrayBuffer> | null>
}) => {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  const maskTextureRef = useRef<THREE.DataTexture | null>(null);
  const { size } = useThree();

  // Create grid geometry
  const { positions, uvs } = useMemo(() => {
    const width = 512; // Increased density
    const height = 512;
    const count = width * height;
    const positions = new Float32Array(count * 3);
    const uvs = new Float32Array(count * 2);

    for (let i = 0; i < width; i++) {
      for (let j = 0; j < height; j++) {
        const index = i * height + j;
        // Map to -2.66 to 2.66 for X (4:3 aspect ratio), -2 to 2 for Y
        positions[index * 3] = (i / (width - 1)) * 5.33 - 2.66;
        // FLIP Y to fix upside down issue
        positions[index * 3 + 1] = -((j / (height - 1)) * 4 - 2);
        positions[index * 3 + 2] = 0;
        
        uvs[index * 2] = i / (width - 1);
        uvs[index * 2 + 1] = 1.0 - (j / (height - 1)); // Flip Y for texture mapping
      }
    }
    return { positions, uvs };
  }, []);

  const videoTexture = useMemo(() => {
    if (!videoElement) return null;
    const tex = new THREE.VideoTexture(videoElement);
    tex.minFilter = THREE.LinearFilter;
    tex.magFilter = THREE.LinearFilter;
    return tex;
  }, [videoElement]);

  const uniforms = useMemo(() => ({
    uVideoTexture: { value: videoTexture },
    uMaskTexture: { value: null },
    uTime: { value: 0 },
    uInvertMask: { value: invertMask ? 1.0 : 0.0 },
    uAudioLevel: { value: 0.0 }
  }), [videoTexture, invertMask]);

  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.elapsedTime;
      materialRef.current.uniforms.uInvertMask.value = invertMask ? 1.0 : 0.0;
      
      // Update audio level
      if (analyserRef.current && dataArrayRef.current) {
        analyserRef.current.getByteFrequencyData(dataArrayRef.current);
        let sum = 0;
        for (let i = 0; i < dataArrayRef.current.length; i++) {
          sum += dataArrayRef.current[i];
        }
        const average = sum / dataArrayRef.current.length;
        const normalized = average / 255.0;
        // Smooth the audio level
        materialRef.current.uniforms.uAudioLevel.value += (normalized - materialRef.current.uniforms.uAudioLevel.value) * 0.1;
      } else {
        materialRef.current.uniforms.uAudioLevel.value *= 0.9; // Decay if disabled
      }
    }

    if (videoElement && videoElement.readyState >= 2 && segmenter) {
      try {
        const result = segmenter.segmentForVideo(videoElement, performance.now());
        if (result.categoryMask) {
          const maskData = result.categoryMask.getAsUint8Array();
          const width = result.categoryMask.width;
          const height = result.categoryMask.height;

          if (!maskTextureRef.current || maskTextureRef.current.image.width !== width) {
            maskTextureRef.current = new THREE.DataTexture(maskData, width, height, THREE.RedFormat);
            
            if (materialRef.current) {
              materialRef.current.uniforms.uMaskTexture.value = maskTextureRef.current;
            }
          } else {
            maskTextureRef.current.image.data = maskData;
          }
          maskTextureRef.current.needsUpdate = true;
          
          // Close the mask to prevent memory leaks
          result.categoryMask.close();
        }
      } catch (e) {
        console.error("Segmentation error:", e);
      }
    }
  });

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-uv"
          count={uvs.length / 2}
          array={uvs}
          itemSize={2}
        />
      </bufferGeometry>
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent={true}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
};

export default function App() {
  const [videoElement, setVideoElement] = useState<HTMLVideoElement | null>(null);
  const [segmenter, setSegmenter] = useState<ImageSegmenter | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [invertMask, setInvertMask] = useState(true); // Default to true based on the issue
  const [audioEnabled, setAudioEnabled] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const dataArrayRef = useRef<Uint8Array<ArrayBuffer> | null>(null);

  const enableAudio = async () => {
    try {
      // Try to get system audio via screen share
      const stream = await navigator.mediaDevices.getDisplayMedia({
        audio: true,
        video: true // Browsers usually require video to be true for getDisplayMedia
      });

      const audioTrack = stream.getAudioTracks()[0];
      if (!audioTrack) {
        stream.getTracks().forEach(track => track.stop());
        alert("No audio track found. Please make sure to check 'Share audio' when selecting a screen or tab.");
        return;
      }

      // Stop the video track since we only need audio
      stream.getVideoTracks().forEach(track => track.stop());

      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(new MediaStream([audioTrack]));
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);

      analyserRef.current = analyser;
      dataArrayRef.current = new Uint8Array(analyser.frequencyBinCount) as Uint8Array<ArrayBuffer>;
      setAudioEnabled(true);

      // Handle stream stop
      audioTrack.onended = () => {
        setAudioEnabled(false);
        analyserRef.current = null;
      };
    } catch (err) {
      console.error("Failed to get system audio:", err);
      alert("Could not access system audio. Please ensure you granted permission and checked 'Share audio'.");
    }
  };

  const retryInitialization = async () => {
    setError(null);
    setIsLoading(true);
    setVideoElement(null);
    setSegmenter(null);

    // Clean up existing resources
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    if (segmenter) {
      segmenter.close();
    }

    // Re-run initialization
    let active = true;

    const init = async () => {
      try {
        // 1. Request Webcam
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
          audio: false,
        });

        if (!active) return;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setVideoElement(videoRef.current);
          };
        }

        // 2. Initialize MediaPipe
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        const seg = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          outputCategoryMask: true,
        });

        if (!active) return;
        setSegmenter(seg);
        setIsLoading(false);

      } catch (err: any) {
        console.error(err);
        let errorMessage = "Failed to initialize camera or AI model.";

        if (err.name === 'NotAllowedError') {
          errorMessage = "Camera access denied. Please grant camera permission and try again.";
        } else if (err.name === 'NotFoundError') {
          errorMessage = "No camera found. Please connect a camera and try again.";
        } else if (err.name === 'NotReadableError') {
          errorMessage = "Camera is already in use by another application. Please close other apps using the camera and try again.";
        } else if (err.name === 'OverconstrainedError') {
          errorMessage = "Camera does not support the required video format.";
        } else if (err.name === 'SecurityError') {
          errorMessage = "Camera access blocked. This app requires HTTPS in production.";
        } else if (err.message) {
          errorMessage = err.message;
        }

        setError(errorMessage);
        setIsLoading(false);
      }
    };

    init();

    return () => {
      active = false;
    };
  };

  useEffect(() => {
    let active = true;

    const init = async () => {
      try {
        // 1. Request Webcam
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
          audio: false,
        });

        if (!active) return;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
            setVideoElement(videoRef.current);
          };
        }

        // 2. Initialize MediaPipe
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        const seg = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          outputCategoryMask: true,
        });

        if (!active) return;
        setSegmenter(seg);
        setIsLoading(false);

      } catch (err: any) {
        console.error(err);
        let errorMessage = "Failed to initialize camera or AI model.";

        if (err.name === 'NotAllowedError') {
          errorMessage = "Camera access denied. Please grant camera permission and refresh the page.";
        } else if (err.name === 'NotFoundError') {
          errorMessage = "No camera found. Please connect a camera and refresh the page.";
        } else if (err.name === 'NotReadableError') {
          errorMessage = "Camera is already in use by another application. Please close other apps using the camera.";
        } else if (err.name === 'OverconstrainedError') {
          errorMessage = "Camera does not support the required video format.";
        } else if (err.name === 'SecurityError') {
          errorMessage = "Camera access blocked. This app requires HTTPS in production.";
        } else if (err.message) {
          errorMessage = err.message;
        }

        setError(errorMessage);
        setIsLoading(false);
      }
    };

    init();

    return () => {
      active = false;
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
      if (segmenter) {
        segmenter.close();
      }
    };
  }, []);

  return (
    <div className="w-full h-screen bg-black text-white overflow-hidden relative font-sans">
      {/* Hidden Video Element */}
      <video
        ref={videoRef}
        className="hidden"
        playsInline
        muted
      />

      {/* UI Overlay */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/90 backdrop-blur-sm"
          >
            <Loader2 className="w-12 h-12 text-cyan-500 animate-spin mb-4" />
            <h2 className="text-xl font-medium tracking-widest text-cyan-400 uppercase">Initializing Neural Link</h2>
            <p className="text-cyan-500/60 mt-2 text-sm">Loading MediaPipe Models & Accessing Camera...</p>
          </motion.div>
        )}

        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-black/90"
          >
            <AlertCircle className="w-16 h-16 text-red-500 mb-4" />
            <h2 className="text-2xl font-bold text-white mb-2">Connection Failed</h2>
            <p className="text-red-400 max-w-md text-center">{error}</p>
            <button
              onClick={retryInitialization}
              className="mt-6 px-6 py-2 bg-red-500/20 hover:bg-red-500/40 text-red-300 rounded-full transition-colors border border-red-500/50"
            >
              Retry Connection
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 3D Scene */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 5], fov: 60 }}>
          <color attach="background" args={['#050505']} />
          <ambientLight intensity={0.5} />
          
          {videoElement && segmenter && (
            <PointCloud 
              videoElement={videoElement} 
              segmenter={segmenter} 
              invertMask={invertMask} 
              analyserRef={analyserRef}
              dataArrayRef={dataArrayRef}
            />
          )}
          
          <OrbitControls 
            enablePan={false}
            enableZoom={true}
            minDistance={2}
            maxDistance={10}
          />
        </Canvas>
      </div>

      {/* HUD Elements */}
      {!isLoading && !error && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 1 }}
          className="absolute inset-0 pointer-events-none z-10"
        >
          <div className="absolute top-6 left-6 flex items-center gap-3">
            <div className="w-3 h-3 rounded-full bg-cyan-500 animate-pulse shadow-[0_0_10px_rgba(6,182,212,0.8)]" />
            <span className="font-mono text-cyan-500 text-sm tracking-widest uppercase">Live Feed Active</span>
          </div>
          
          <div className="absolute bottom-6 left-6 pointer-events-auto">
            <div className="font-mono text-xs text-cyan-500/50 mb-2">SYSTEM STATUS</div>
            <div className="font-mono text-sm text-cyan-400 mb-1">MediaPipe Vision: ONLINE</div>
            <div className="font-mono text-sm text-cyan-400 mb-4">Three.js Engine: ONLINE</div>
            
            <button 
              onClick={() => setInvertMask(!invertMask)}
              className="flex items-center gap-2 px-3 py-1.5 bg-cyan-950/50 border border-cyan-500/30 rounded text-cyan-400 text-xs font-mono hover:bg-cyan-900/50 transition-colors mb-2"
            >
              <Settings2 className="w-3 h-3" />
              Toggle Mask Inversion
            </button>

            <button 
              onClick={enableAudio}
              disabled={audioEnabled}
              className={`flex items-center gap-2 px-3 py-1.5 border rounded text-xs font-mono transition-colors ${
                audioEnabled 
                  ? 'bg-cyan-500/20 border-cyan-500 text-cyan-300' 
                  : 'bg-cyan-950/50 border-cyan-500/30 text-cyan-400 hover:bg-cyan-900/50'
              }`}
            >
              {audioEnabled ? <Volume2 className="w-3 h-3" /> : <VolumeX className="w-3 h-3" />}
              {audioEnabled ? 'System Audio: ON' : 'Enable System Audio'}
            </button>
          </div>

          <div className="absolute bottom-6 right-6 text-right">
            <div className="font-mono text-xs text-cyan-500/50 mb-1">CONTROLS</div>
            <div className="font-mono text-sm text-cyan-400">Drag to Rotate</div>
            <div className="font-mono text-sm text-cyan-400">Scroll to Zoom</div>
          </div>
          
          {/* Decorative corners */}
          <div className="absolute top-4 left-4 w-8 h-8 border-t-2 border-l-2 border-cyan-500/30" />
          <div className="absolute top-4 right-4 w-8 h-8 border-t-2 border-r-2 border-cyan-500/30" />
          <div className="absolute bottom-4 left-4 w-8 h-8 border-b-2 border-l-2 border-cyan-500/30" />
          <div className="absolute bottom-4 right-4 w-8 h-8 border-b-2 border-r-2 border-cyan-500/30" />
        </motion.div>
      )}
    </div>
  );
}
