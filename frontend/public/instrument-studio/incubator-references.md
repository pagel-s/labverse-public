# CO₂ incubator · reference and validation log

## Real examples inspected before modelling

- [Thermo Scientific Heracell VIOS manufacturer brochure](https://assets.thermofisher.com/TFS-Assets/LED/brochures/Heracell-VIOS-CO2-Incubator-BRCO2VIOSNA-EN.pdf): inspected front, control-panel, open-chamber and sample-vessel photographs on PDF pages 1–5. The 160-series envelope is approximately 637 mm wide × 900 mm high × 880 mm deep, including fittings. The internal chamber and three perforated shelves informed the construction.
- [BINDER CB 170 manufacturer product page](https://www.binder-world.com/uk-en/products/incubators/standard-incubators/product/cb-170): secondary comparison for a stainless chamber, inner glass door and shelf layout. This model's dimensions were not mixed into the primary envelope.

Accessed 3 October 2026. Manufacturer imagery was used for study only and is not distributed with this library. All geometry, screen graphics and surface textures are procedurally authored. No manufacturer labels or logos appear on the asset.

## Scope and fidelity

Original, unbranded interpretation of this instrument category, with approximate real-world proportions. Independent insulated outer door, glass inner door, handles, hinges, gaskets, three physically perforated sliding shelves, chamber supports, airflow panel, sensor head, shallow water pan, cabinet feet and rear fittings. Six neutral sealed demonstration dishes have stable IDs and can move between shelf positions and a small staging surface.

Dimensions describe a nominal source envelope, not a dimensionally certified CAD replica. Small fittings, ventilation details, hinge clearance, dish dimensions and materials are approximations. Door opening angles and shelf travel are illustrative. The model has no manufacturer certification or measurable photorealism score.

## Functional boundary

State and geometry are independent. Door/shelf clearance rules prevent closing doors over an extended shelf. Loading requires both doors open, and each shelf has two demonstration positions. These are deliberately limited UI affordances, not a complete instrument capacity model. Power and glass-door exposure drive a qualitative recovery bar; no real temperatures, gas concentrations, recovery predictions or processing protocols are modelled. Outer-door opening alone does not expose the chamber in this state model. No controller or hardware connection is included.

`LabIncubatorState.snapshot()` exposes IDs, locations, door flags and shelf flags. The browser emits `labverse:equipment-state` events. The Python simulator is not yet wired to these events.

## Validation

- `tests/incubator.test.mjs`: 44 assertions for door and shelf interlocks, inventory limits and identity, power/recovery, invalid inputs, and isolated snapshots.
- Browser review: inspected open/closed outer door, inner glass door, extended shelf and vessel placement. Verified shelf clearance rejection, load/unload, door closure, power off/on, and recovery indication. No console warnings or errors were reported during this review.
- Export preparation produced a 21.2 MB scene JSON blob and visible download link. The embedded browser download-completion event could not be verified; use the explicit Save scene JSON link.
- Browser automation navigation intermittently timed out; subsequent page-state and visual inspections succeeded.
- Static 3D export uses Three.js ObjectLoader JSON. Behaviour remains in the JavaScript state module; export is not GLB.
