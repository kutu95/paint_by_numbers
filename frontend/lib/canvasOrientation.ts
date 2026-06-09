/**
 * Match canvas cm dimensions to image orientation (portrait vs landscape).
 */
export function canvasCmForImageOrientation(
  imageWidth: number,
  imageHeight: number,
  canvasWidthCm: number,
  canvasHeightCm: number
): { widthCm: number; heightCm: number } {
  if (imageWidth <= 0 || imageHeight <= 0 || canvasWidthCm <= 0 || canvasHeightCm <= 0) {
    return { widthCm: canvasWidthCm, heightCm: canvasHeightCm }
  }

  const imagePortrait = imageHeight > imageWidth
  const canvasPortrait = canvasHeightCm > canvasWidthCm

  if (imageWidth === imageHeight || canvasWidthCm === canvasHeightCm) {
    return { widthCm: canvasWidthCm, heightCm: canvasHeightCm }
  }

  if (imagePortrait !== canvasPortrait) {
    return { widthCm: canvasHeightCm, heightCm: canvasWidthCm }
  }

  return { widthCm: canvasWidthCm, heightCm: canvasHeightCm }
}
