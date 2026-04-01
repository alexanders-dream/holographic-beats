<div align="center">
<img width="1200" height="475" alt="Holographic Point Cloud" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Holographic Point Cloud

A real-time 3D holographic point cloud visualization that transforms webcam video into an interactive particle system using AI-powered body segmentation.

## Features

- **Real-time Body Segmentation**: Uses TensorFlow.js with BodyPix/ArcFace for person segmentation
- **Interactive 3D Point Cloud**: Webcam feed rendered as 65,536 particles (256×256 grid)
- **Holographic Visuals**: Custom shaders with additive blending and depth effects
- **Camera Controls**: Orbit, zoom, and pan with mouse/touch
- **Toggle Mask Inversion**: Switch between positive and negative space visualization

## Technical Details

### Particle Resolution

- **Grid Size**: 256×256 = 65,536 particles
- **Position Range**: X: -2.66 to 2.66, Y: -2 to 2
- **Rendering**: Full screen resolution with device pixel ratio support

### Performance

- Target: 60 FPS on modern hardware
- Adjustable particle density by modifying grid size in code

## Quick Start

**Prerequisites:** Node.js 18+

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

4. Open http://localhost:3000 in your browser
5. Allow camera access when prompted

## Project Structure

```
holographic-point-cloud/
├── src/
│   ├── App.tsx         # Main application component
│   ├── main.tsx       # Application entry point
│   └── index.css      # Global styles
├── public/            # Static assets
├── index.html         # HTML entry point
├── vite.config.ts     # Vite configuration
├── tsconfig.json      # TypeScript configuration
├── package.json       # Dependencies and scripts
└── .env.example       # Environment variables template
```

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server on port 3000 |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm run lint` | Run TypeScript type checking |

## Controls

- **Left Mouse Drag**: Rotate view
- **Scroll Wheel**: Zoom in/out
- **Right Mouse Drag**: Pan (disabled by default)
- **Toggle Button**: Invert segmentation mask

## Tech Stack

- **React** 19 + TypeScript
- **Three.js** via React Three Fiber
- **TensorFlow.js** (MediaPipe for segmentation)
- **Framer Motion** for UI animations
- **Tailwind CSS** for styling
- **Vite** for build tooling

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Optional | Google Gemini API key - currently not in use but available for future AI features |
| `APP_URL` | Optional | Application URL for AI Studio integration |

## Troubleshooting

### Camera not working
- Ensure you've granted camera permissions in your browser
- Check that no other application is using the camera
- Try using a different browser (Chrome recommended)

### Performance issues
- Reduce particle count in code (modify grid size)
- Close other resource-intensive applications
- Use hardware acceleration in browser settings

## License

All rights reserved. This project is not licensed for public use.
