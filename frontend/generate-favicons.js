#!/usr/bin/env node
/**
 * Generate favicon PNG files from SVG
 * Requires: npm install sharp (or use online converter)
 * 
 * Alternative: Use online tool like https://favicon.io/favicon-converter/
 * or https://realfavicongenerator.net/ to convert the SVG
 */

const fs = require('fs');
const path = require('path');

// SVG content for the stacked layers favicon
const svgContent = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="100" height="100">
  <!-- Bottom Layer (Red) -->
  <g transform="translate(20, 60)">
    <!-- Top face -->
    <polygon points="0,0 40,0 60,20 20,20" fill="#E74C3C"/>
    <!-- Thickness edge -->
    <polygon points="20,20 60,20 60,25 20,25" fill="#B33D31"/>
  </g>
  
  <!-- Middle Layer (Orange) -->
  <g transform="translate(15, 45)">
    <!-- Top face -->
    <polygon points="0,0 40,0 60,20 20,20" fill="#FDC742"/>
    <!-- Thickness edge -->
    <polygon points="20,20 60,20 60,25 20,25" fill="#C19932"/>
  </g>
  
  <!-- Top Layer (Teal) -->
  <g transform="translate(10, 30)">
    <!-- Top face -->
    <polygon points="0,0 40,0 60,20 20,20" fill="#4ECDC4"/>
    <!-- Thickness edge -->
    <polygon points="20,20 60,20 60,25 20,25" fill="#3BA099"/>
  </g>
</svg>`;

// Save SVG to public directory
const publicDir = path.join(__dirname, 'public');
if (!fs.existsSync(publicDir)) {
  fs.mkdirSync(publicDir, { recursive: true });
}

fs.writeFileSync(path.join(publicDir, 'icon.svg'), svgContent);
console.log('✓ Created icon.svg');

console.log('\nTo generate PNG favicons:');
console.log('1. Visit https://favicon.io/favicon-converter/');
console.log('2. Upload public/icon.svg');
console.log('3. Download and extract the generated files to public/');
console.log('4. Or use ImageMagick: convert public/icon.svg -resize 16x16 public/favicon.ico');
console.log('\nRequired files:');
console.log('  - public/favicon.ico (16x16)');
console.log('  - public/icon-192.png (192x192)');
console.log('  - public/icon-512.png (512x512)');
console.log('  - public/apple-touch-icon.png (180x180)');
