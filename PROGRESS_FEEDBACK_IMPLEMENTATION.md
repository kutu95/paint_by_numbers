# Progress Feedback Implementation Plan

## Overview
Add real-time progress feedback for recipe generation showing which color is currently being processed.

## Approach
Use Server-Sent Events (SSE) with FastAPI's StreamingResponse to send progress updates as JSON lines.

## Implementation Steps

### Backend Changes
1. Add StreamingResponse import (done)
2. Create a generator function `generate_recipes_with_progress()` that:
   - Yields progress updates as JSON lines: `{"type": "progress", "palette_index": N, "current": X, "total": Y, "hex": "#...", "status": "processing"}`
   - Processes colors sequentially
   - Yields final result: `{"type": "complete", "recipes": [...]}`
3. Modify endpoint to use StreamingResponse with the generator

### Frontend Changes
1. Update `handleGenerateRecipes` to:
   - Read response as a stream
   - Parse JSON lines
   - Update progress state as each line arrives
   - Display: "Processing color X of Y: #HEX"
   - Show final recipes when complete

## Notes
- The endpoint is ~300 lines, so this requires significant refactoring
- Generator pattern is necessary for real-time progress
- JSON Lines format (one JSON object per line) is simple to parse
