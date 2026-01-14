#!/usr/bin/env node
/**
 * Generate favicon PNG files from SVG using sharp
 * Run: npm install sharp (if not already installed)
 * Then: node generate-favicons-sharp.js
 */

const fs = require('fs');
const path = require('path');

try {
  const sharp = require('sharp');
  const publicDir = path.join(__dirname, 'public');
  const svgPath = path.join(publicDir, 'icon.svg');

  if (!fs.existsSync(svgPath)) {
    console.error('Error: icon.svg not found. Run generate-favicons.js first.');
    process.exit(1);
  }

  console.log('Generating favicon files from SVG...');

  // Generate favicon.ico (16x16) - sharp doesn't support ICO, so we'll create PNG
  sharp(svgPath)
    .resize(16, 16)
    .png()
    .toFile(path.join(publicDir, 'favicon-16.png'))
    .then(() => console.log('✓ Created favicon-16.png (use online converter for .ico)'));

  // Generate icon-192.png
  sharp(svgPath)
    .resize(192, 192)
    .png()
    .toFile(path.join(publicDir, 'icon-192.png'))
    .then(() => console.log('✓ Created icon-192.png'));

  // Generate icon-512.png
  sharp(svgPath)
    .resize(512, 512)
    .png()
    .toFile(path.join(publicDir, 'icon-512.png'))
    .then(() => console.log('✓ Created icon-512.png'));

  // Generate apple-touch-icon.png (180x180)
  sharp(svgPath)
    .resize(180, 180)
    .png()
    .toFile(path.join(publicDir, 'apple-touch-icon.png'))
    .then(() => {
      console.log('✓ Created apple-touch-icon.png');
      console.log('\nNote: For favicon.ico, use an online converter like https://favicon.io/');
      console.log('or convert favicon-16.png to favicon.ico');
    });

} catch (error) {
  if (error.code === 'MODULE_NOT_FOUND') {
    console.log('sharp not installed. Installing...');
    console.log('Run: npm install sharp');
    console.log('Then run this script again.');
  } else {
    console.error('Error:', error.message);
  }
  process.exit(1);
}
