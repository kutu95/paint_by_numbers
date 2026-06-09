# LayerPainter — Founding Member Brief

Last updated: 2026-05-30  
Status: Draft for outreach (not yet recruiting)  
Related: `Docs/commercialisation.md`

---

## One-line pitch

**LayerPainter is the projection and paint-planning studio for mural artists working from their own acrylics.**

Upload a reference, generate paint layers in the right order, project them fullscreen onto your wall or canvas, and match colours to real tubes with calibrated recipes — not a phone tracing app, not a print-and-number PDF generator.

---

## Who we are looking for

**Ideal founding members (aim for 10–20):**

- Working muralists, sign painters, or large-canvas portrait artists
- Already use (or want to use) a projector for breakdown / underpainting
- Paint from their own acrylic set — not buying a pre-made kit
- Willing to run at least **one real job** (or serious test piece) through the tool in the first 30 days
- Comfortable giving blunt feedback on a beta product

**Not the right fit:**

- Casual hobbyists who only want a paint-by-numbers PDF
- Artists who need mobile AR tracing (Da Vinci Eye–style)
- Anyone who cannot tolerate missing features during beta (see honest gaps below)

---

## What founding members get

| Benefit | Detail |
|---------|--------|
| **Locked price** | **$19/month** (or **$199/year**) for as long as membership stays active — vs planned public **Pro at $29/month** |
| **Pro-level access** | Full pipeline, projection workflow, paint library, calibration, recipes, spot test, palette optimisation |
| **Direct line to builder** | Async feedback channel; your pain points shape the roadmap |
| **Early features** | First access to keystone, saved alignment, checklist export as they ship |
| **Founding badge** | Credit in launch materials if you opt in (case study, quote, timelapse clip) |

**Beta access:** Hosted account when Phase 1 ships; until then, guided self-hosted setup or paired onboarding session if needed.

---

## What we ask in return

1. **Onboard within 7 days** of invite — upload one image, generate layers, try projection mode.
2. **Use it on real work** within 30 days — even a test board or wall section counts.
3. **30-minute feedback call** (or written survey) after first serious use.
4. **Optional:** Short case study — before/after photo, 2–3 sentences on time or paint saved, permission to quote.

No NDA required. We want honest “this broke my workflow” feedback, not cheerleading.

---

## What works well today

- Photo → layered masks with smart overpaint and coverage-based paint order
- Fullscreen projection: crosshairs, grid, registration mode, outline modes, mask opacity, lasso navigation
- Priority region and portrait-oriented controls (detail in faces, style presets)
- Paint library with calibration curves, ΔE recipes, weight estimates, virtual mixer, spot test
- Project workflow: Image tab → Layers → Projection (single generate path)

---

## Honest gaps in beta (we will prioritise by founding feedback)

| Gap | Status |
|-----|--------|
| Keystone / corner pinning | Not yet — competitors like ProTrace have this; **top retention priority** |
| Saved alignment across sessions | Limited — registration mode helps but persistence is weak |
| Palette / layer checklist PDF | Not yet — planned for Pro launch |
| Mobile / AR | Out of scope — desktop + projector workflow |
| Hosted multi-user | Building toward launch — early beta may be invite-only hosted or assisted self-host |

Founding members should expect rough edges. The deal is **lower price + real influence** in exchange for tolerance and signal.

---

## Outreach copy (email / DM — adapt as needed)

**Subject:** Founding spot — mural projection + paint planning tool (beta)

Hi [Name],

I'm building **LayerPainter** — a web studio for muralists who project breakdown layers and mix from their own acrylics. You upload a reference, get ordered paint layers with overpaint built in, project fullscreen, and tie colours to calibrated recipes for your tube set.

I'm recruiting **10–20 founding members** before public launch:

- **$19/mo locked** (public Pro will be $29)
- Full Pro features during beta
- Your feedback directly shapes what we build next (keystone, alignment save, export checklist are top of list)

I'm looking for people who will actually run a wall or canvas through it in the next month — not casual testers.

Would you be open to a 15-minute look, or should I send a one-pager and demo link?

[Your name]

---

## Discovery questions (first call or survey)

Use these to validate ICP and willingness to pay. Score 1–5 or free text.

### Workflow today

1. How do you currently break down a mural or large canvas? (projector / grid / freehand / print / other)
2. What tools do you use today? (Da Vinci Eye, ProTrace, Photoshop, Mural Maker, nothing formal)
3. How many commercial or serious personal murals do you paint per year?
4. Typical palette size on a job? (number of mixed colours)

### Pain and value

5. Where do you lose the most time: planning colours, mixing, projection alignment, or repainting mistakes?
6. Have you ever abandoned or reworked a job because colour planning was wrong? What did that cost in time or materials?
7. If a tool saved you **half a day per job**, what would that be worth monthly? (open-ended — listen for $ range)

### Product fit

8. After a 10-minute demo, would you use this on your **next** job? Why or why not?
9. Which matters more: **better projection alignment** (keystone, saved setup) or **better paint matching** (recipes, weights)?
10. Would you pay **$29/month** for unlimited projects + full paint lab after beta, if keystone shipped within ~3 months?

### Competitive

11. What would make you stay on your current tool instead of switching?
12. Would you recommend this to another muralist at $19 founding / $29 public? (NPS-style 0–10)

### Founding commitment

13. Can you commit to one real use within 30 days of invite?
14. OK to be quoted or featured in a launch case study? (yes / no / maybe later)

---

## Selection criteria (who to accept)

| Signal | Accept | Wait-list |
|--------|--------|-----------|
| Paints murals or large canvases regularly | Yes | — |
| Uses or wants projector workflow | Yes | Phone-only tracing |
| Gives specific, critical feedback | Yes | Vague “looks cool” |
| Commits to 30-day real use | Yes | “Maybe someday” |
| Mixes own acrylics | Strong fit | Pre-made kits only |
| Teaches / runs workshops | Studio tier lead later | — |

Target mix: ~70% working pros, ~30% serious independents with clear projector setup.

---

## Success metrics for the founding cohort

After 60 days, the beta is working if:

- **≥12 of 20** complete onboarding and one real use
- **≥8** complete feedback call or survey
- **≥5** say they would pay $29/mo post-beta (question 10)
- **≥3** opt into case study or public quote
- Clear rank order on roadmap from aggregated Q9 (keystone vs paint lab vs export)

If fewer than 8 complete real use, revisit ICP or product gaps before public launch.

---

## Prerequisite: user accounts (do not recruit before this)

Founding outreach is blocked until **hosted multi-user accounts** work end-to-end. Without auth, there is no tenant boundary, no per-user project storage, and no way to issue invites or lock founding pricing.

Minimum bar before first invite:

1. Register / login (email + password is enough for v1)
2. Session cookie or token; logout works
3. `GET /api/projects` returns **only the signed-in user's projects**
4. All project read/write/generate routes enforce ownership
5. New signups get an empty workspace (optional sample project copy)

Stripe, entitlements, and OAuth can wait. See `Docs/auth-plan.md` for implementation slices.

---

## Internal checklist before sending invites

**Accounts & data (required)**

- [ ] User register / login / logout working
- [ ] Projects scoped per user (`ownerId` on manifest + API enforcement)
- [ ] Hosted demo environment with isolated accounts (not shared server dirs)
- [ ] Terms draft: image ownership, beta disclaimer, founding price lock rules

**Product & outreach (required)**

- [ ] Demo environment stable (upload → generate → projection)
- [ ] One sample project new users can open without uploading
- [ ] Feedback form or call booking link ready

**Billing (can follow first cohort manually)**

- [ ] Stripe founding SKU or manual invoicing process decided
- [ ] List of 25–40 prospects named (need ~2× target for acceptance rate)

---

## Founding price lock rules (draft)

- Price locked at **$19/mo or $199/yr** while subscription remains **continuously active**
- Lapse > 30 days → rejoin at then-current public rate unless grandfathering exception granted
- Founding benefits transfer only with explicit written approval (not casual account sharing)
- Feature set = Pro tier at time of use; new Studio-only features may be add-on later

---

## Related docs

- `Docs/commercialisation.md` — full pricing, GTM phases, marketplace
- `Docs/context.md` — technical product spec
