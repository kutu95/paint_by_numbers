export interface Layer {
  layer_index: number
  palette_index: number
  mask_url: string
  mask_pure_url?: string
  outline_thin_url: string
  outline_thick_url: string
  outline_glow_url: string
  is_finished?: boolean
  finished_url?: string
  is_gradient?: boolean
  is_glaze?: boolean
  gradient_region_id?: string
  gradient_step_index?: number
  hex?: string
  rgb?: number[]
}

export interface SessionData {
  session_id: string
  width: number
  height: number
  palette: Array<{ index: number; hex: string; coverage: number }>
  order: number[]
  layers: Layer[]
  quantized_preview_url?: string
  original_url?: string
  canvas_width_cm?: number
  canvas_height_cm?: number
}
