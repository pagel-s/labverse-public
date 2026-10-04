# PCR thermocycler · reference and validation log

## Real examples inspected before modelling

- [Eppendorf Mastercycler X50 manufacturer brochure](https://www.eppendorf.com/product-media/doc/en/236881/PCR_Brochure_Mastercycler-X50-Next-Stage.pdf): inspected PDF spreads on pages 2–3, including the open lid, handle, block close-up, sloped front panel and closed instrument. These shaped the model's construction. The device screen is original generic artwork.
- [Mastercycler X50 manufacturer manual](https://www.eppendorf.com/product-media/doc/en/237979/PCR_Operating-manual_Mastercycler-X50.pdf): technical dimensions on printed page 59 supply the nominal 275 × 430 × 330 mm closed envelope.
- [Bio-Rad T100 manufacturer instrument-guide excerpt](https://www.bio-rad.com/sites/default/files/2021-08/10000145577.pdf): secondary comparison of the opening lid and reaction-module arrangement. Its dimensions and operating programs were not used.

Accessed 4 October 2026. Manufacturer images are study references only and are not bundled. The model has no manufacturer logos, wordmarks or copied screen artwork.

## Model scope

Compact sloped chassis, contrasting base, feet and vents, touchscreen, lid housing, large U-shaped handle, underside plate, hinge, 96-position block, removable neutral plate, clear cover and a staging pad. Block openings are geometry, with simplified tapered liners beneath them. The closed envelope is approximate, not certified CAD; small fittings and the lid mechanism are interpreted rather than measured. Plate and cover are demonstration props, not dimensionally validated consumables.

The model uses millimetres under a metre-scaled root, and exposes sample-block and plate-staging anchors. Geometry and state are separate modules. UI emits `labverse:equipment-state` integration events; the Python backend is not connected.

## Demonstration boundary

The shortened run is a visual progress animation. There are no actual temperature values, dwell times, ramp rates, amplification physics, biological samples or experimental programs. Cover and lid checks are simplified demonstration rules. Pausing keeps the lid locked; stopping or completion releases it without modelling real cooling. This is not a manufacturer controller or a safe-operation simulator.

## Validation

- `tests/thermocycler.test.mjs`: 36 assertions for loading/cover checks, lid locking, power, pause/resume, stop, completion, invalid time inputs and snapshot isolation.
- Browser review: inspected the open and closed housing, corrected the touchscreen placement, loaded the plate, closed the lid, and ran the demonstration through pause/resume to completion. The lid control remained disabled during pause and was released at completion.
- Export preparation produced a 14.6 MB JSON blob and a visible download link. Download completion in the embedded browser is not independently verified.
- All five gallery navigation entries and local documentation links resolve.
- Export uses Three.js ObjectLoader JSON, with separate JavaScript behaviour.
