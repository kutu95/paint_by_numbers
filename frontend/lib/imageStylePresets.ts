/** Image look presets — must match backend VALID_STYLE_PRESETS (natural → none). */
export type ImageStylePreset =
  | 'none'
  | 'easy_painting'
  | 'portrait'
  | 'poster'
  | 'bold'
  | 'expressive'
  | 'graphic'
  | 'stipple'
  | 'sketch'
  | 'harmony'

export interface ImageStylePresetInfo {
  id: ImageStylePreset
  label: string
  description: string
}

export const IMAGE_STYLE_PRESETS: ImageStylePresetInfo[] = [
  {
    id: 'none',
    label: 'None (classic)',
    description:
      'No stylization — same as before presets. Turn on Easy painting below for softer regions and detail preservation.',
  },
  {
    id: 'easy_painting',
    label: 'Easy painting',
    description: 'Softer regions with optional face/eye protection — best for hand-painted portraits.',
  },
  {
    id: 'portrait',
    label: 'Portrait',
    description: 'Moderate simplify with eyes, face, and outlines always protected.',
  },
  {
    id: 'poster',
    label: 'Poster',
    description: 'Flat tonal steps — bold print/poster look (light background blur).',
  },
  {
    id: 'bold',
    label: 'Bold colours',
    description: 'Extra saturation with gentle simplification.',
  },
  {
    id: 'expressive',
    label: 'Expressive',
    description: 'Hue shift and richer colour — structure unchanged; optional detail preservation.',
  },
  {
    id: 'graphic',
    label: 'Graphic',
    description: 'Maximum simplification — large flat areas, few details.',
  },
  {
    id: 'stipple',
    label: 'Stipple / halftone',
    description: 'Dot halftone by tone — pointillist feel (highlights stay bright).',
  },
  {
    id: 'sketch',
    label: 'Sketch',
    description: 'Pencil-sketch tones with a hint of colour.',
  },
  {
    id: 'harmony',
    label: 'Colour harmony',
    description: 'Smoother, related hues — cohesive painterly palette.',
  },
]

/** Map stored / legacy values to a valid preset id. */
export function normalizeStylePreset(value: string | undefined | null): ImageStylePreset {
  const v = (value || 'none').trim().toLowerCase().replace(/-/g, '_')
  if (v === 'natural' || v === 'classic' || v === 'original' || v === 'off') return 'none'
  if (IMAGE_STYLE_PRESETS.some((p) => p.id === v)) return v as ImageStylePreset
  return 'none'
}

export function isImageStylePreset(value: string): value is ImageStylePreset {
  return normalizeStylePreset(value) === value || value === 'natural'
}

export function presetUsesLegacyEasyPainting(preset: ImageStylePreset): boolean {
  return preset === 'none'
}

export function presetShowsSimplifyControls(
  preset: ImageStylePreset,
  easyPainting: boolean
): boolean {
  if (preset === 'none') return easyPainting
  return true
}

export function presetShowsFigureDetailControls(
  preset: ImageStylePreset,
  easyPainting: boolean
): boolean {
  if (preset === 'none') return easyPainting
  return true
}

export function presetForcesFigureDetail(preset: ImageStylePreset): boolean {
  return preset === 'portrait'
}
