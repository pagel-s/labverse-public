# Biological safety cabinet · reference and validation log

## Real instruments inspected before modelling

- [Thermo Scientific 1300 Series A2 manufacturer brochure](https://assets.thermofisher.com/TFS-Assets/LED/brochures/1300-Series-A2-Biological-Safety-Cabinet.pdf): viewed PDF pages 2–3 showing front housing, sloped sash, metal chamber, grilles, side utility fittings, stand, armrests and control-panel layout. The four-foot version supplies the nominal cabinet envelope of 1300 × 1568 × 798 mm; the support stand is a generic interpretation and adds 650 mm below that envelope.
- [Esco Airstream G4 manufacturer product page](https://www.escolifesciences.com/products/class-ii-biological-safety-cabinet/airstream-e-series-g4-class-ii-biological-safety-cabinet): secondary category comparison. Its dimensions are not mixed with the primary reference.

Accessed 3 October 2026. No reference photographs, logos, manufacturer screen artwork or labels are distributed. Geometry and display textures are original.

## Included

Sloped independently moving glass sash, guide rails, paired handles, powder-coated housing, stainless chamber and work deck, geometrically modelled intake slots, top grille, stand uprights, levelling feet, armrests, utility covers, generic receptacles, work light and generic display. UI exposes three illustrative sash positions, blower and work-light switches, main power, utility status and an optional schematic airflow overlay.

## Fidelity boundaries

The overall proportions follow a real cabinet category. Small fittings, stand construction, guide mechanism and internal arrangement are approximate. There is no serviceable filter assembly or real fan hardware. The schematic arrows are an explanatory overlay, not a flow calculation. Status logic shows a raised-sash warning but does not determine safe conditions. No containment, filtration, contamination, sterility or protection model is implemented, and this asset has no certification. No biological workflow, processing protocol, or UV control is included.

Geometry and state are independent, use a metre-scale root, and expose work-surface/access anchors. The `labverse:equipment-state` browser event is an integration hook; it is not connected to the Python simulator.

## Validation

- `tests/biosafety.test.mjs`: 30 assertions cover sash status, invalid sash commands, power shutdown, independent controls, restoration and snapshot isolation.
- Browser review: inspected the cabinet, sloped glass closure, raised sash, and chamber close-up. Verified blower switch, raised-sash warning, light switch, airflow overlay checkbox, utility indicator and power shutdown.
- Export preparation produced a 7.2 MB scene JSON blob with a visible download link. Download completion in the embedded browser is not independently verified.
- Export uses Three.js ObjectLoader JSON; behaviour remains in the separate state module.
