# Favicon Setup Instructions

The SVG favicon has been created at `public/icon.svg`. To generate the required PNG and ICO files:

## Option 1: Online Converter (Recommended)

1. Visit https://favicon.io/favicon-converter/
2. Upload `public/icon.svg`
3. Download the generated package
4. Extract these files to `public/`:
   - `favicon.ico` (16x16)
   - `android-chrome-192x192.png` → rename to `icon-192.png`
   - `android-chrome-512x512.png` → rename to `icon-512.png`
   - `apple-touch-icon.png` (180x180)

## Option 2: Using ImageMagick (if installed)

```bash
cd frontend/public

# Generate favicon.ico (16x16)
convert icon.svg -resize 16x16 favicon.ico

# Generate PNG files
convert icon.svg -resize 192x192 icon-192.png
convert icon.svg -resize 512x512 icon-512.png
convert icon.svg -resize 180x180 apple-touch-icon.png
```

## Option 3: Using rsvg-convert (if installed)

```bash
cd frontend/public

rsvg-convert -w 16 -h 16 icon.svg > favicon.ico
rsvg-convert -w 192 -h 192 icon.svg > icon-192.png
rsvg-convert -w 512 -h 512 icon.svg > icon-512.png
rsvg-convert -w 180 -h 180 icon.svg > apple-touch-icon.png
```

## Files Required

After generation, ensure these files exist in `public/`:
- `icon.svg` ✓ (already created)
- `favicon.ico` (16x16)
- `icon-192.png` (192x192)
- `icon-512.png` (512x512)
- `apple-touch-icon.png` (180x180)
- `site.webmanifest` ✓ (already created)

The favicon will automatically be used once these files are in place.
