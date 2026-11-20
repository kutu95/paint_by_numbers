'use client'

const SECTIONS = [
  { id: 'flats', label: 'Flats first' },
  { id: 'acrylic', label: 'Acrylic handling' },
  { id: 'fade', label: 'Fading two colours' },
  { id: 'soften', label: 'Softening an edge' },
  { id: 'hair', label: 'Hair' },
  { id: 'spots', label: 'Spots and marks' },
  { id: 'skin', label: 'Skin and eyes' },
  { id: 'texture', label: 'Fur, cloth, water, leaves' },
  { id: 'projection', label: 'While you paint' },
] as const

function scrollToSection(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

export function TechniqueGuide() {
  return (
    <article className="max-w-3xl mx-auto space-y-8 pb-16">
      <header className="space-y-3">
        <h2 className="text-2xl font-bold">Technique</h2>
        <p className="text-gray-300 leading-relaxed">
          LayerPainter gives you a photograph as a set of flat colours you can project and paint in order.
          The likeness is already in those shapes. The brushwork on this page is what you add on top of the
          flats: a fade between two colours, hair, spots, and the soft or sharp edges the photograph actually has.
        </p>
        <p className="text-gray-400 text-sm leading-relaxed">
          These are ordinary acrylic methods. Mix a little, compare it with the photograph, and let each layer dry
          before you complicate it.
        </p>
        <nav className="flex flex-wrap gap-2 pt-1" aria-label="Techniques">
          {SECTIONS.map((section) => (
            <button
              key={section.id}
              type="button"
              onClick={() => scrollToSection(section.id)}
              className="px-3 py-1.5 text-sm rounded bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white"
            >
              {section.label}
            </button>
          ))}
        </nav>
      </header>

      <section id="flats" className="scroll-mt-6 space-y-3">
        <h3 className="text-lg font-semibold">Paint the flat colours first</h3>
        <p className="text-gray-300 leading-relaxed">
          Each layer is one palette colour, a solid shape. Paint that shape in the projected mask and let it dry.
          Early layers are drawn a little larger than the final patch, so the next colour can cover the join.
          Carry the early colour up to the projected edge and slightly past it. Leave the crisp drawing, the hair,
          the spots, and the fades until the neighbouring colours are on the canvas.
        </p>
        <p className="text-gray-300 leading-relaxed">
          The finished preview (F in the projection window) is those flats assembled. The original photograph (G)
          is the reference for everything you blend by hand. Switch to G when you are judging a transition.
        </p>
      </section>

      <section id="acrylic" className="scroll-mt-6 space-y-3">
        <h3 className="text-lg font-semibold">How acrylic behaves while you work</h3>
        <p className="text-gray-300 leading-relaxed">
          Acrylic usually dries a little darker, and it loses the wet shine, so a mix in the pot looks lighter than
          the same paint on the canvas an hour later. Brush a small patch, let it dry, and compare it with the
          photograph before you mix a large amount. On the Layers tab, Verify mix photographs that dried swatch and
          checks it against the target colour.
        </p>
        <p className="text-gray-300 leading-relaxed">
          The paint starts to skin over in minutes. Mix only what the passage in front of you needs. A
          slow-drying medium or retarder, used at the amount printed on the bottle, keeps a blend workable. More
          than that can leave the film tacky. A stay-wet palette, or a light mist over the mixes, helps in the
          same way.
        </p>
        <p className="text-gray-300 leading-relaxed">
          When the paint on the canvas starts to drag, stop. Brushing a half-dry film turns the colour cloudy.
          Let it dry fully, then continue with a scumble or a glaze. Thin a light wash with a little water. For a
          transparent layer you want to stay strong, use glazing medium. A film flooded with water can dry weak
          and chalky.
        </p>
        <p className="text-gray-300 leading-relaxed">
          Rinse the brush between the two ends of a fade. A dirty brush shifts every step toward the same muddy
          middle.
        </p>
      </section>

      <section id="fade" className="scroll-mt-6 space-y-3">
        <h3 className="text-lg font-semibold">Fade from one colour to another</h3>
        <p className="text-gray-300 leading-relaxed">
          A fade in a photograph is a short run of colours between two ends: lit cheek into shadow, sky into
          horizon, a highlight falling off across a sleeve. Sample those two ends, mix a paint for each, mix a
          few steps between them, and lay the steps across the form.
        </p>
        <ol className="list-decimal pl-5 space-y-3 text-gray-300 leading-relaxed">
          <li>
            On the Image tab, zoom into the transition. Choose <strong className="text-gray-100">Pick colour</strong>.
            Move across the lighter end until the swatch matches the colour of that area. A single pixel on a noisy
            photo can be a speck of the wrong colour, so rest on a small patch that looks right, then click to lock
            it. Write down the hex. Choose Cancel unless you also want that colour forced into the palette the next
            time you generate layers. Must include is for the palette. The fade itself is mixed by hand.
          </li>
          <li>
            Do the same at the darker end of the same transition. You now have two hex values.
          </li>
          <li>
            On the Layers tab, open <strong className="text-gray-100">Virtual paint mixer</strong>. Paste the first
            hex into <strong className="text-gray-100">Or custom hex</strong>. Move the paint sliders until the
            preview sits near that swatch and Mix ΔE is low. A low number means the mix is close to the sampled
            colour. If the number stays high, the paint library cannot hit that hex, so take the nearest mix and
            correct it by eye against the photograph. Write the slider amounts down. That is the recipe for the
            first colour. Clear the sliders and repeat for the second hex.
          </li>
          <li>
            Mix both colours on the palette. Then mix three steps between them by volume: about three parts of the
            first to one of the second, equal parts, and one part of the first to three of the second. Strong
            pigments, such as phthalo blue or dioxazine purple, take over a mix, so add those in smaller amounts
            and judge the step against G. Hold the mixes next to the projection if you can.
          </li>
          <li>
            Paint the steps across the transition, following the form. While they are wet, pull a clean soft brush,
            barely damp, once across each join. Leave the steps visible. A fade brushed until it is one even tone
            loses the turning form the photograph had.
          </li>
          <li>
            If the flat colour underneath has already dried, leave it alone. Set the new steps beside it, or soften
            the change with a scumble or a glaze once everything is dry.
          </li>
        </ol>
        <p className="text-gray-300 leading-relaxed">
          Use this on the boundaries that matter: the turn of a cheek, the shadow under a chin, a sky, a wall in
          raking light. The flat palette colours can stay flat everywhere else.
        </p>
      </section>

      <section id="soften" className="scroll-mt-6 space-y-3">
        <h3 className="text-lg font-semibold">Other ways to soften a change of colour</h3>
        <p className="text-gray-300 leading-relaxed">
          <strong className="text-gray-100">Wet in wet.</strong> Put both colours on the canvas while they are wet
          and pull them together with a soft brush. Work a small area. Acrylic closes up quickly, so this suits a
          short transition, such as a cloud edge or the side of a nose.
        </p>
        <p className="text-gray-300 leading-relaxed">
          <strong className="text-gray-100">Double-load the brush.</strong> Load one corner of a flat brush with the
          first colour and the other corner with the second. One stroke lays both, blended along the middle. Reload
          for each stroke. This suits a sky band, a petal, or a soft horizon.
        </p>
        <p className="text-gray-300 leading-relaxed">
          <strong className="text-gray-100">Scumble.</strong> Wait until the lower colour is dry. Pick up a little of
          the upper colour, wipe the brush until it is almost dry, and drag it lightly. The tooth of the canvas and
          the colour underneath both show through. This is the dependable way to fade after a layer has dried. It
          suits skin, distant hills, and a haze over a background.
        </p>
        <p className="text-gray-300 leading-relaxed">
          <strong className="text-gray-100">Glaze.</strong> Mix a transparent colour with glazing medium and brush a
          thin film over a dry layer. The colour underneath remains, shifted warmer, cooler, darker, or greyer. A
          cool glaze quiets a distant background. A warm glaze can put sunlight on a wall or a face. Let each glaze
          dry before the next one. A wet glaze looks lighter than it will when dry, so check it against G after it
          has dried.
        </p>
        <p className="text-gray-300 leading-relaxed">
          A hard edge belongs where the photograph is sharp: an eyelid, a collar, a window frame, the meeting of
          two flat palette colours. Paint it with a loaded brush and leave the edge. A soft edge belongs where a
          form turns or the photo is out of focus. The same object often has both. A fold is firmer on the shadow
          side and softer as it comes back toward the light.
        </p>
      </section>

      <section id="hair" className="scroll-mt-6 space-y-3">
        <h3 className="text-lg font-semibold">Hair</h3>
        <p className="text-gray-300 leading-relaxed">
          Block the hair with the palette colours as solid masses, following the projected shapes. Those masses
          carry the likeness: the dark under the jaw and behind the ear, the mid tone of the bulk, the light along
          the top plane.
        </p>
        <p className="text-gray-300 leading-relaxed">
          When the masses are dry, suggest strands with a liner or a rigger, or with an old filbert whose bristles
          have separated. Stroke in the direction the hair grows, from the root toward the tip, and follow the part.
          A darker mix goes in the gaps between masses. A lighter mix goes in thin, broken strokes only where the
          photograph shows light on the hair. Wipe the brush nearly dry and drag it so the canvas breaks the line.
          A few of those broken lights read as a full head of hair.
        </p>
        <p className="text-gray-300 leading-relaxed">
          A fan brush is for a handful of flyaway strokes at the silhouette. Where the hair meets the background,
          paint the background colour back into the edge in short strokes, so the boundary is a set of gaps. Sample
          the highlight colour with Pick colour. Light on dark hair takes the colour of the light source, and it is
          rarely pure white.
        </p>
      </section>

      <section id="spots" className="scroll-mt-6 space-y-3">
        <h3 className="text-lg font-semibold">Spots, freckles, and broken marks</h3>
        <p className="text-gray-300 leading-relaxed">
          Stipple with the tip of a round brush held upright. Touch the canvas and lift. Vary the pressure so the
          dots change size. Cluster them the way the photograph does: denser in the middle of a group, thinner at
          the edge, with a few larger and a few softer. The projection places the group. G tells you which marks
          are sharp and which are blurred.
        </p>
        <p className="text-gray-300 leading-relaxed">
          Freckles sit in the skin. Sample the skin and a freckle with Pick colour, and mix a colour a little warmer
          and darker than the skin around it. Keep most dots small. Soften some with a dry brush so they look inside
          the skin, and leave a few sharper. They gather across the nose and upper cheeks and thin out toward the
          sides of the face.
        </p>
        <p className="text-gray-300 leading-relaxed">
          Animal spots go on after the base colour is dry. Follow the projected shape for placement, then look at
          the original for the edge. Spots in a photograph are irregular. Some are broken, some overlap, and some
          have a darker rim or a lighter halo of fur. Paint that halo with a short dry-brush stroke in the direction
          of the fur.
        </p>
        <p className="text-gray-300 leading-relaxed">
          Out-of-focus lights are soft discs. Lay a pale mix and, while it is wet, feather the edge with a clean
          damp brush until the disc melts into the background. A hard dot reads as a speck, which is a different
          mark in the photograph.
        </p>
        <p className="text-gray-300 leading-relaxed">
          Foliage, stone, and a field of flowers can be dabbed with a natural sponge. Load it lightly, blot it once
          on scrap paper, then dab, turning the sponge so the pattern does not repeat. Use the lighter palette
          colour on the lit side and the darker one in the shade, over the flat layer. For sand, stars, or distant
          tiny flowers, load a toothbrush and flick the paint. Cover any part of the canvas that should stay clean.
        </p>
      </section>

      <section id="skin" className="scroll-mt-6 space-y-3">
        <h3 className="text-lg font-semibold">Skin and eyes</h3>
        <p className="text-gray-300 leading-relaxed">
          The palette colours are the planes of the face. Place them as flats. Fade only along the boundary where
          one plane turns into the next: under the cheekbone, beside the nose, along the jaw. Use the two-colour
          mix above, with the two samples taken from either side of that boundary. Leave the lit forehead and the
          features comparatively simple. Blending the whole face into one tone removes the drawing the layers gave
          you.
        </p>
        <p className="text-gray-300 leading-relaxed">
          On the Image tab, a priority region and Favor skin tones ask the palette to spend more of its colours on
          a face before you start. That is the moment to decide the skin colours. The brush blends between them later.
        </p>
        <p className="text-gray-300 leading-relaxed">
          Paint the eyes after the surrounding skin is in. Sample the white of the eye from the photograph. It is a
          warm or cool grey sitting next to the skin, and the catchlight is a small opaque dab of the light colour,
          placed where G shows it, with a crisp edge. Soften the outer iris slightly. Keep the pupil dark and definite.
        </p>
      </section>

      <section id="texture" className="scroll-mt-6 space-y-3">
        <h3 className="text-lg font-semibold">Fur, cloth, water, and leaves</h3>
        <p className="text-gray-300 leading-relaxed">
          <strong className="text-gray-100">Fur.</strong> Short strokes in the direction the fur grows, usually
          outward from the body. The same nearly dry drag used for hair breaks the edge against the background.
          Darker strokes sit in the recesses. Lighter strokes sit on the lit surface.
        </p>
        <p className="text-gray-300 leading-relaxed">
          <strong className="text-gray-100">Cloth.</strong> Put the shadow colour in the crease. That edge is sharper
          where the fold turns away from the light and softer where it comes back. A thin lighter stroke along the
          ridge of the fold is enough to describe the cloth.
        </p>
        <p className="text-gray-300 leading-relaxed">
          <strong className="text-gray-100">Water and sky.</strong> Mix a short row of steps from the upper sky to
          the horizon, or from an object down to its reflection, and lay them in horizontal bands. A reflection is a
          little darker and softer than the thing reflected. A horizontal drag of a dry brush breaks it into water.
          Calm water wants level strokes. Ripples are those strokes interrupted.
        </p>
        <p className="text-gray-300 leading-relaxed">
          <strong className="text-gray-100">Foliage.</strong> The layers give you the large clumps. Stipple or sponge
          a lighter value on the sunlit side and a darker value underneath. Paint a few clearer leaf or needle shapes
          at the outer edge, where the photograph is sharpest. The inside of the tree can stay as the flat colour
          plus a broken texture.
        </p>
        <p className="text-gray-300 leading-relaxed">
          <strong className="text-gray-100">Bark, brick, and stone.</strong> Drag a lightly loaded flat or filbert
          so the canvas tooth catches the paint. Two close colours, one dragged over the other after it has dried,
          read as a rough surface. Sample both from the photograph so the texture stays in the right hue.
        </p>
      </section>

      <section id="projection" className="scroll-mt-6 space-y-3">
        <h3 className="text-lg font-semibold">Using the projection while you do this</h3>
        <ul className="list-disc pl-5 space-y-2 text-gray-300 leading-relaxed">
          <li>
            <strong className="text-gray-100">G</strong> shows the original photograph. <strong className="text-gray-100">F</strong> shows
            the flat finished preview. Judge fades, hair highlights, and soft spots against G.
          </li>
          <li>
            <strong className="text-gray-100">K</strong> cycles the mask. In the detail view, white is the part of
            this colour that remains, and grey is overlap a later layer will cover. Spend the careful work on the white.
          </li>
          <li>
            Draw a lasso around a face, an animal, or a patch of spots. The arrow keys then walk only the colours
            inside that shape. <strong className="text-gray-100">E</strong> clears the lasso.
          </li>
          <li>
            <strong className="text-gray-100">[</strong> lowers the mask brightness when the projection washes out the
            wet paint you are trying to see.
          </li>
          <li>
            <strong className="text-gray-100">D</strong> marks the flat layer done. You can still glaze, scumble, and
            add hair or spots over it afterwards.
          </li>
          <li>
            Line the projector up with the crosshairs before a session, and use registration (R) if you come back
            to a canvas that has already been started. A fade painted a few millimetres off the projected edge will
            not meet the next colour.
          </li>
        </ul>
      </section>
    </article>
  )
}
