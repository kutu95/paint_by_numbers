LayerPainter — Projection-Based Paint-by-Numbers App

Purpose

LayerPainter is a projection-first web app for mural and canvas painting.
It takes an input image, reduces it to N paint layers, applies smart underpainting expansion, and lets the user project each layer fullscreen for tracing or painting.

There is NO export or zip download in the MVP.
All layers are generated and displayed directly in the app.

⸻

Core Capabilities
	1.	Upload image
	2.	Choose number of colors (N)
	3.	Quantize image into N palette colors
	4.	Generate clean paint masks per color
	5.	Automatically order layers (largest coverage first)
	6.	Apply overpaint expansion to early layers
	7.	Display layers fullscreen with projection-specific tools

⸻

Tech Stack (MANDATORY)
	•	Frontend: Next.js + React + TypeScript + Tailwind
	•	Backend: FastAPI (Python)
	•	Image processing: OpenCV + numpy
	•	Storage: Temporary filesystem storage per session
	•	No authentication
	•	No database
	•	No export pipeline

⸻

User Flow

A) Setup & Processing
	1.	User uploads an image
	2.	User selects:
	•	Number of colors N (2–32, default 16)
	•	Overpaint amount in mm (default 5)
	•	Layer ordering:
	•	Largest coverage first (default)
	•	Smallest coverage first
	•	Manual reorder
	•	Max processing resolution (default 1920 or 2400 px)
	3.	Click Generate Layers
	4.	App displays:
	•	Quantized preview image
	•	Palette chips with coverage %
	•	Ordered layer list with thumbnails
	•	Reorder controls

⸻

B) Projection Mode
	•	User clicks Start Projection
	•	Fullscreen projection UI optimized for walls and canvases
	•	One layer shown at a time
	•	Minimal UI, black background

⸻

Projection View Features (CRITICAL)

Alignment Aids

Corner Crosshairs (DEFAULT)
	•	Four L-shaped corner marks
	•	Offset inward ~3% of width/height
	•	Stroke: 1–2 px at 1080p (scale with resolution)
	•	Color: mid-grey (#888)
	•	Toggle: C (on by default)

Background Grid (OPTIONAL)
	•	Grid spacing: every 5% of width/height
	•	Low contrast dark grey on black
	•	Toggle: G

⸻

Display Modes

Mask Polarity
	•	Normal: white paint on black
	•	Inverted: black paint on white
	•	Shortcut: I

Outline Overlay
	•	OFF
	•	Thin (1px)
	•	Thick (2–3px)
	•	Glow (soft halo)
	•	Cycle shortcut: O

Mask Opacity
	•	Adjustable 40% → 100%
	•	Default 85%
	•	Keyboard: [ and ]

⸻

Registration Mode

Used to realign projector across sessions.
	•	Displays:
	•	Corner crosshairs
	•	Grid
	•	Current layer outline only
	•	No filled paint area
	•	Toggle: R

⸻

Layer Navigation & State
	•	Prev / Next layer
	•	Jump via number keys
	•	Mark layer as Done
	•	Automatically skip done layers
	•	Store completion state in localStorage per session

⸻

Screen Control
	•	B → full black screen
	•	W → full white screen
	•	Mouse auto-hide after 2s inactivity
	•	HUD toggle: H

⸻

Image Processing Pipeline (Backend)

Step 1 — Normalize
	•	Read image → RGB
	•	Resize longest side to max_side (default 1920/2400)
	•	Preserve aspect ratio

⸻

Step 2 — Quantize
	•	Convert RGB → Lab
	•	K-means clustering (fixed random seed)
	•	Outputs:
	•	labels array (0..N-1)
	•	Palette RGB centers
	•	Coverage % per cluster
	•	Quantized preview image

⸻

Step 3 — Clean Masks

For each cluster:
	•	Binary mask from labels
	•	Remove tiny connected components (<0.02% image area)
	•	Morphological close → open
	•	Store as clean base mask

⸻

Step 4 — Layer Ordering

Default: Largest coverage first

⸻

Step 5 — Smart Overpaint Expansion (MANDATORY)

Purpose: allow fast base coats that will be refined later.

Variables:
	•	overpaint_mm (default 5)
	•	Approximate px_per_mm:
	•	Assume longest side ≈ 1000 mm unless user specifies wall size
	•	r_px_base = max(1, round(overpaint_mm * px_per_mm))
	•	Gamma scaling: gamma = 1.5


Algorithm:
code

painted_union = empty mask

for idx, layer in ordered_layers:
    base = clean_mask[layer]
    scale = (1 - idx/(N-1)) ** gamma
    r_px = round(r_px_base * scale)
    expanded = dilate(base, r_px)
    paint_mask = expanded AND NOT painted_union
    painted_union = painted_union OR paint_mask
    store paint_mask

Constraints:
	•	Expanded regions MUST NOT overwrite earlier layers
	•	Later layers naturally become tighter and more precise

⸻

Step 6 — Outline Generation
	•	Generate from paint mask or base mask
	•	Techniques:
	•	Morphological gradient or Canny
	•	Support thickness + glow variants
	•	Returned as transparent PNG or binary edge mask

⸻

Backend API

POST /api/sessions

Multipart:
	•	image
	•	n_colors
	•	overpaint_mm
	•	order_mode
	•	manual_order optional
	•	max_side optional

Response:

{
  session_id,
  width,
  height,
  palette: [{ index, hex, coverage }],
  order: [indices],
  quantized_preview_url,
  layers: [
    {
      layer_index,
      palette_index,
      mask_url,
      outline_url
    }
  ]
}

Sessions stored in:

/data/sessions/{session_id}/

Cleanup:
	•	Delete sessions older than X hours (simple cron or startup cleanup)

⸻

Frontend Routes
	•	/ — setup + preview
	•	/project/[sessionId] — fullscreen projection mode

⸻

Defaults (IMPORTANT)
	•	Corner crosshairs: ON
	•	Grid: OFF
	•	Mask: white on black
	•	Outline: thin
	•	Mask opacity: 85%
	•	Ordering: largest coverage first
	•	Overpaint: 5 mm

⸻

Explicit Non-Goals (DO NOT IMPLEMENT)
	•	Zip or file export
	•	Paint-by-numbers numbering
	•	Palette locking to real paints
	•	Full geometric calibration / keystone correction
	•	Authentication or persistence

⸻

Acceptance Criteria
	•	End-to-end flow works locally
	•	Layers visibly expand early and refine later
	•	Projection mode is usable from across a room
	•	App is stable and readable, not over-engineered

⸻

Cursor Build Prompt

Use this prompt in Cursor:

Build the full app exactly per CURSOR_CONTEXT.md.
Prioritize:
	1.	A correct image pipeline with smart overpaint expansion
	2.	A clean, fullscreen projection UI with crosshairs, grid, invert, outlines, and registration mode
Do NOT implement exports, authentication, or databases.
Keep the code simple, readable, and local-dev friendly. Provide clear run instructions.
