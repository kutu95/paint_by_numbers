# LayerPainter - Projection Paint Layers

A projection-first web app for generating paint-by-numbers layers optimized for mural and canvas painting.

## Tech Stack

- **Frontend**: Next.js 14 + React + TypeScript + Tailwind CSS
- **Backend**: FastAPI (Python) + OpenCV + NumPy + scikit-learn

## Setup

### Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (if not already created):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the backend server:
```bash
python main.py
```

The backend will be available at `http://localhost:8000`

### Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Open `http://localhost:3000` in your browser
2. Upload an image
3. Configure settings:
   - Number of colors (2-100, default 16)
   - Overpaint amount in mm (default 5)
   - Layer ordering mode (largest/smallest/manual)
   - Max processing resolution (1920 or 2400px)
   - **Gradient-aware quantization** (optional): Enable to detect smooth gradients (sky, water) and generate multi-step ramps instead of flat bands. Set gradient steps (5–15), transition mode (dither recommended), and optionally enable the **glaze pass** for a final unifying thin layer per gradient.
4. Click "Generate Layers"
5. Review the quantized preview, palette, and layer list
6. Click "Start Projection" to enter fullscreen projection mode

## Projection Mode Keyboard Shortcuts

- **C** - Toggle corner crosshairs (default: ON)
- **G** - Toggle grid
- **I** - Invert mask polarity (white on black ↔ black on white)
- **O** - Cycle outline mode (off → thin → thick → glow)
- **[** - Decrease mask opacity (40-100%, default 85%)
- **]** - Increase mask opacity
- **-** - Shrink projected image (25–100%, 5% steps)
- **+** / **=** - Enlarge projected image (100–200%, 5% steps)
- **F** - Toggle full-colour final image (see how current layer fits in) / back to current layer
- **R** - Toggle registration mode (crosshairs + grid + outline only, no fill)
- **B** - Black screen
- **W** - White screen
- **H** - Toggle HUD
- **←** / **→** - Navigate to previous/next layer
- **Space** - Next layer
- **D** - Mark current layer as Done (skipped in navigation)

## Painting gradient regions (sky, water)

When gradient detection is enabled, the app finds smooth gradient areas and replaces flat color bands with **ramp steps** (lighter to darker, top to bottom by default). Paint in layer order:

1. **Gradient steps** – Paint each gradient step layer in order (Step 1, 2, …). Use the suggested colors; transitions between steps are dithered so edges blend when viewed from a distance.
2. **Glaze pass** (optional) – If you enabled "Glaze pass", a final layer per gradient region is added. Paint it **last**, very thin/translucent, over the whole gradient area to unify the steps.

Non-gradient areas are unchanged and use the normal palette layers.

## Features

- Smart overpaint expansion: Early layers expand more than later layers for efficient base coating
- Clean mask generation: Removes tiny components and applies morphological operations
- Layer ordering: Automatic ordering by coverage or manual reordering
- Fullscreen projection mode: Optimized for wall/canvas projection
- Persistent layer state: Done layers are saved in localStorage per session
- Mouse auto-hide: Cursor hides after 2 seconds of inactivity

## Project Structure

```
Paint by Numbers/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── image_processor.py   # Image processing pipeline
│   ├── paint_manager.py     # Paint library & recipe generation
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx         # Main upload/setup page
│   │   ├── paints/
│   │   │   ├── page.tsx     # Paint library management
│   │   │   └── calibrate/
│   │   │       └── [paintId]/
│   │   │           └── page.tsx  # Calibration wizard
│   │   └── project/
│   │       └── [sessionId]/
│   │           └── page.tsx # Projection viewer
│   └── components/
├── data/
│   ├── sessions/            # Generated session files
│   └── paint/               # Paint library & calibrations
└── README.md
```

## Notes

- Sessions are stored in `./data/sessions/{session_id}/`
- Sessions older than 24 hours are automatically cleaned up
- No authentication or database required
- All processing is done server-side
- Layers are served as PNG images from the backend

## Paint Management & Recipe Generation

LayerPainter includes a paint management system that helps you generate mixing recipes for the quantized palette colors.

### Paint Library

1. Navigate to **Paint Library** (click "Manage Paints" button or go to `/paints`)
2. Add base paints with their approximate colors
3. Each paint needs to be calibrated before it can be used in recipes

### Calibration Workflow

To calibrate a paint:

1. **Prepare Tint Ladder:**
   - Mix your paint with white at these ratios: 50%, 25%, 12.5%, 6.25%, 3.125%
   - Paint small squares for each ratio on your target surface
   - Include a reference strip: white, mid-grey, black (simple printed card)
   - Take a photo straight-on with good lighting

2. **Upload & Sample:**
   - Go to Paint Library → Click "Calibrate" on a paint
   - Upload your calibration photo
   - Click each swatch in order (darkest to lightest)
   - The app will sample colors and save the calibration

### Generating Recipes

After generating layers for an image:

1. Click **"Generate Recipes"** in the Paint Recipes panel
2. The app will calculate mixing recipes for each palette color using your calibrated paints
3. Recipes show:
   - Mixing ratios (e.g., "White 92% + Blue 8%")
   - Error score (how close the match is)
   - Quality indicator (good/ok/poor)

### Recipe Types

- **One Pigment + White:** Most common, uses a single calibrated pigment
- **Two Pigments + White:** Used when single pigment match is poor (estimated)

### Verification (MVP)

After mixing a recipe, you can:
1. Paint a small swatch
2. Upload a photo and click the swatch center
3. The app will compare measured color to target and suggest corrections

## Deployment

See [deployment/DEPLOYMENT.md](deployment/DEPLOYMENT.md) for detailed instructions on deploying to an Ubuntu server.

Quick deployment steps:
1. Copy files to server at `/opt/layerpainter/`
2. Setup backend: `cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
3. Setup frontend: `cd frontend && npm install && NEXT_PUBLIC_API_BASE_URL=http://192.168.0.146:8000 npm run build`
4. Configure systemd services (see deployment files)
5. Start services: `sudo systemctl start backend.service frontend.service`

For automated deployment, use the provided `deployment/deploy.sh` script.

## Tests

Basic gradient validation tests (coverage, no leak, deterministic output):

```bash
cd backend && source venv/bin/activate && python -m unittest tests.test_gradient_validation -v
```

## Limitations

- No export/download functionality (by design)
- No authentication or user accounts
- Paint calibration requires manual swatch clicking (no auto-detection)

