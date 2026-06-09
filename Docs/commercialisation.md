# LayerPainter — Commercialisation & Platform Strategy

Last updated: 2026-05-30  
Status: Planning / not implemented  
Audience: Product, engineering, future investors or partners

This document captures strategy discussed in May 2026: subscription hosting, pricing, go-to-market, competitive positioning, and a paint library marketplace. It supplements `Docs/context.md` (technical MVP spec).

---

## 1. Product positioning

**What LayerPainter is**

A projection-first mural/canvas studio that:

1. Quantizes photos into paint layers (masks per colour)
2. Projects layers fullscreen for hand-painting
3. Ties colours to real calibrated paints (recipes, weights, verification)

**One-line pitch (draft)**

> LayerPainter is the projection and paint-planning studio for mural artists working from their own acrylics.

**Primary ICP (recommended)**

Working muralists, sign painters, and serious portrait/large-canvas artists — not mass-market phone tracing users.

**Moat**

- Mural-specific pipeline (overpaint, layer order, registration projection)
- Deep paint matching (calibration curves, ΔE recipes, spot test, feedback bias, substrate compensation)
- Self-hosted / studio workflow today; hosted SaaS + marketplace tomorrow

**Explicit non-goals (from original MVP, some now contested)**

Original `context.md` excluded export, keystone, auth, and "palette locking." The product has evolved: paint-library-aware recipes and calibration are now core. Revisit non-goals before launch:

- Export: still no full PBN PDF by default; consider palette/layer checklist PDF for pros
- Keystone: high priority for paying muralists (competitive gap)
- Auth + multi-tenancy: required for commercialisation

---

## 2. Competitive landscape (summary)

| Category | Examples | Overlap |
|----------|----------|---------|
| AR tracing / breakdown | Da Vinci Eye, Mural Maker | Layers + painting guidance; mobile AR |
| Browser projection | ProTrace, Painting Stoner | Keystone warp; no layer generation |
| PBN generators | PBNify, DigitPaints, ostempel | Quantization + PDF export; no projection/recipes |

**Where competitors are ahead**

- Perspective / keystone correction (ProTrace, Painting Stoner)
- AR re-alignment across sessions (Mural Maker)
- Print/export deliverables (PBN tools)
- Mobile-native + community (Da Vinci Eye)
- Multi-device / workshop streaming (Mural Maker)

**Where LayerPainter is ahead**

- Smart overpaint + coverage-based layer order
- Per-colour projection masks (pure / expanded / detail, outlines)
- Calibrated recipe pipeline + paint weights
- Priority region, skin slots, style presets for portraits
- Virtual mixer, spot test, palette optimisation vs library
- Lasso-filtered projection navigation

**Strategic implication**

Do not compete on PBN PDFs or AR mobile. Compete on **mural studio + paint lab**. Close keystone + alignment gaps for retention.

---

## 3. Subscription model (hosted SaaS)

### Model

- Hosted multi-tenant service
- Users buy subscription (Stripe)
- Per-user project storage, processing entitlements, paint library access

### Recommended tiers (launch)

| Tier | Monthly | Annual | Target user |
|------|---------|--------|-------------|
| **Hobby** | $12 | $99 | Home murals, limited projects |
| **Pro** | $29 | $249 | Working muralists (hero tier) |
| **Studio** | $59 | $499 | Small teams / workshops |

- **14-day Pro trial** (define card-required vs not before launch)
- **Founding member** beta: e.g. $19/mo locked for early muralists + feedback

### Hobby entitlements (example)

- 3 active projects
- Full projection workflow
- Up to 24 colours per project
- No or limited paint calibration / recipes

### Pro entitlements (example)

- Unlimited projects
- Full image pipeline (priority region, presets, up to 100 colours)
- Full paint library + calibration + recipes + spot test
- Palette optimisation, weight estimates
- Commercial use allowed

### Studio entitlements (example)

- Pro + 3–5 seats (or per-seat add-on)
- Shared team paint library
- Priority processing queue

### Pricing rationale

- Da Vinci Eye ~$8/mo / $30/yr (mass market tracing)
- LayerPainter bundles more pro value → **$29 Pro** is credible if it saves hours/paint per job
- One commercial mural ($500–5000+) justifies months of subscription

### Revenue sanity (illustrative)

- 200 Pro @ $29 ≈ $5,800 MRR
- Niche SaaS viable at hundreds of paying pros with low churn + annual plans

---

## 4. Go-to-market (phased)

### Phase 0 — ICP & messaging (≈2 weeks)

- Commit to muralist ICP
- One-sentence pitch + landing page story

### Phase 1 — Minimum commercial product (≈4–8 weeks)

**Start here:** `Docs/auth-plan.md` — user accounts + project ownership (blocks founding outreach).

| Area | Must-have |
|------|-----------|
| Auth | Email/password + session; one account = one tenant (OAuth later) |
| Billing | Stripe Checkout + Customer Portal |
| Entitlements | Project limits, recipes, max colours by plan |
| Data isolation | Per-user storage (not shared server dirs) |
| Legal | Terms, privacy, image ownership, retention |
| Ops | Backups, rate limits, processing timeouts |

### Phase 2 — Packaging & onboarding

- Wizard: upload → preview → generate → projection
- Defer full calibration on day one; Pro unlock = recipes / spot test
- Sample project for trial users

### Phase 3 — Soft launch

- 10–20 founding members at discounted locked price
- Collect: time saved, willingness to pay, competitor comparisons
- Public launch with one case study

### Channels

- Mural timelapses (projection + layers)
- Street art / mural festivals, sign painting, community murals
- Avoid generic "paint by numbers" SEO (wrong buyer)

### Retention priorities (reduce churn)

1. Keystone / corner pinning in projection
2. Saved alignment per project
3. Palette + layer checklist export (PDF)

---

## 5. Paint library marketplace (platform extension)

### Concept

Two-sided market:

- **Creators** calibrate a paint set, brand it, publish, optionally sell
- **Buyers** discover, purchase/unlock, install read-only snapshot, rate quality

Turns LayerPainter from "rent the app" into "rent the app + calibration ecosystem."

### Fits existing architecture

- Libraries: `libraries/{group}.json`
- Calibrations: `calibration/{group}__{paint_id}.json`
- Recipes keyed by `library_group`
- Publishable asset = versioned bundle (manifest + calibrations + substrate settings)

### Listing schema (conceptual)

- Immutable slug + display name (naming rights on display name)
- Creator profile, version (semver), paint manifest
- Calibration bundle, substrate compensation, coverage
- Quality metrics (% calibrated, optional median ΔE on test palette)
- License (personal / commercial mural)
- Price, ratings aggregate

### Buyer install rules

- Install = **read-only snapshot** of creator bundle at purchase version
- Buyer **feedback_bias** stays private (per-user fork, not published)
- Updates: notify when creator ships v1.2; optional upgrade

### Ratings (structured, not stars-only)

| Dimension | Measures |
|-----------|----------|
| Recipe accuracy | Mixes matched targets in practice |
| Completeness | Enough colours for real projects |
| Documentation | Tubes, surface, conditions clear |
| Support | Creator fixes / updates |
| Value | Worth the price |

Rules: verified purchase only; cooldown before rating; version pinned on review; creator reply; report misleading brand claims.

### Commerce

| Model | Notes |
|-------|-------|
| One-time unlock | $5–29 typical per library |
| Marketplace pass | Pro includes N libraries / month |
| Creator listing fee | Optional $9/mo + sales commission |

Revenue split (starting point): **70/30** creator/platform on one-time sales (after payment fees); adjust to **85/15** early to attract creators.

Stack with app tiers: Hobby = buy only; Pro = publish one free listing + discount; Studio = paid publishing + lower fee.

### Trust & legal

- Trademark: "independent calibration" vs official brand badge
- Takedown for IP; no implied manufacturer affiliation without deal
- Terms: guidance not warranty; batch/surface variance disclosed
- Publish bar: e.g. ≥8 paints, ≥80% calibrated, automated validation
- Refund: broken asset / missing cal files; not "my red was slightly off"

### Marketplace rollout order

1. **Free publish + ratings** (no money) — prove discovery
2. **Paid unlocks** — Stripe + creator payouts
3. **Official / brand partnerships** — verified libraries

**Dependency:** Marketplace after auth, multi-tenancy, and subscription billing.

### Illustrative buyer pricing

| Library type | Price range |
|--------------|-------------|
| Small hobby set | $5–9 |
| Pro mural set (24–40 paints) | $15–25 |
| Official / workshop-certified | $29–49 |

---

## 6. Engineering dependencies (commercial + marketplace)

### Subscription

- User accounts, tenant isolation for `data/projects/` and `data/paint/`
- Stripe products for Hobby / Pro / Studio
- Entitlement middleware on API routes (generate, recipes, calibration)
- Migration path from single-server self-host

### Marketplace

- `marketplace/listings/{id}/` + `users/{uid}/libraries/{installed_id}/`
- Publish pipeline: package + sign + version
- Stripe Connect (or similar) for creator payouts
- Ratings store (listing_id, user_id, version, dimensions)
- Discovery: search, tags, sort by rating/sales

### Do not launch marketplace before

- Auth
- Per-user private libraries
- Automated bundle validation

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Calibration too hard for Hobby | Estimated recipes on Hobby; full lab on Pro |
| Compute cost | Regenerate limits on lower tiers |
| Users expect mobile AR | Position as wall projector / studio product |
| Churn without keystone | Prioritise warp + saved alignment |
| Marketplace fraud | Validation gates + manual first listing review |
| Stale libraries | Versioning + "last verified" date |
| Support load | Docs, video onboarding, async support at launch |

---

## 8. Open decisions

- [ ] Single paid tier vs three tiers at launch
- [ ] Trial: card required or not
- [ ] Marketplace: one-time only vs subscription bundle
- [ ] Official "LayerPainter" starter library as first flagship listing
- [ ] Rev share percentage and creator payout minimum
- [ ] Whether Hobby tier can publish free libraries
- [ ] Update `Docs/context.md` non-goals to reflect paint-library platform direction

---

## 9. Related docs

- `Docs/auth-plan.md` — user accounts implementation slices (do first)
- `Docs/founding-member-brief.md` — outreach copy, discovery questions, founding cohort criteria
- `Docs/context.md` — original technical MVP and projection spec
- `README.md` — setup and feature overview
- `deployment/DEPLOYMENT.md` — self-hosted deployment (transitional)
