# LabVerse instrument studio

An unbranded, reference-grounded 3D asset library. The gallery contains 33 entries covering all requested core and optional equipment types, plus supporting asset sets. These are interactive visual approximations, not validated digital twins or a measured photorealism claim.

The catalog is available at `gallery.html`. New instruments use `equipment.html?id=…` with a shared camera/UI, geometric helpers and independent instrument definitions in `instruments/`.

## Open

Serve this folder locally (for example `python3 -m http.server 8766`) and open `http://localhost:8766/gallery.html` in a WebGL-capable browser, keeping this folder intact. The pinned Three.js library is bundled locally, so no internet connection is required.

Within the LabVerse frontend, start its existing Vite server and visit `/instrument-studio/`. Files in this directory are static assets and do not modify the paper replay viewer.

## Interact

Orbit by dragging; zoom with the wheel. Use the view buttons for keyboard-accessible camera presets. Load or unload a dummy tube with the 24 position buttons, or load matched pairs. Install the rotor cover, close the lid, and start a demonstration cycle. Stop decelerates before unlocking. The complete-demo button prepares a fresh matched pair automatically.

With the lid open, remove the cover and unload tubes to remove the rotor. Removed hardware is supported on separate bench pads. The incubator export first prepares the scene, then presents an explicit Save scene JSON link. Export saves the current geometry as Three.js ObjectLoader JSON (not GLB, and without the behavior code).

The CO₂ incubator is available at `incubator.html`. Open its independent outer and glass doors, extend any of three shelves, and load or unload six sealed demonstration vessels. Retract shelves before closing the doors. Power and exposure affect a qualitative recovery indicator. The chamber does not implement biological processing or real environmental physics.

The cabinet at `biosafety.html` has a sloped sliding sash, work light, blower and utility indicators, main power, a close-up chamber view, and an optional schematic airflow overlay. Status controls are illustrative and do not establish protective performance.

The existing robot is now available at `pipetting-robot.html`, with its original interactions and generic labels. It uses the bundled renderer and joins the same gallery navigation. See `pipetting-robot-references.md`; this prototype has not yet been extracted into the modular model/state architecture.

The thermocycler at `thermocycler.html` includes a hinged lid, removable covered plate, sample-block view, and a shortened run/pause/stop demonstration. It has no actual PCR temperature program or amplification model.

## Complete gallery

Core equipment: liquid-handling robot, manual pipettes, refrigerated centrifuge, mini centrifuge, thermocycler, biosafety cabinet, CO₂ incubator, autoclave, inverted microscope, refrigerator, freezer, vortex mixer, analytical balance, pH meter, hotplate/stirrer, water bath and plate reader.

Optional equipment: cell counter, electrophoresis tank and supply, gel imager, real-time PCR instrument, orbital shaker, plate shaker, plate sealer and plate transport arm.

Supporting sets: fume hood, sink bench, chemical cabinet/cold-storage rack, labware/holders, tips/closures, waste/sharps, eyewash/shower/spill supplies and PPE.

The new instruments provide guarded loading and shortened demonstration cycles; the arm supports pick, carry and place at two stations. Supporting sets expose doors, lids, sash, drawers or visibility controls as appropriate. Safety indicators and analytical displays are illustrative. They make no certification or measurement claim.

## Reuse

- `instruments/centrifuge-model.js`: `createLabCentrifuge(THREE)` returns `{model, update, targets, slotTargets, materials}`. Add `model` to a shared scene and call `update(state, dt, options)` each frame.
- `instruments/centrifuge-state.js`: DOM-independent state and interlocks. `snapshot()` provides serializable equipment state.
- `studio.js`: example viewer and UI binding. Emits `labverse:equipment-state` events on meaningful state changes. These are integration hooks, not an existing backend adapter.
- `instruments/incubator-model.js`: `createLabIncubator(THREE)` returns `{model, update, materials}`; construction uses millimetres under a metre-scaled root, like the centrifuge.
- `instruments/incubator-state.js`: `LabIncubatorState`, with door, shelf, vessel and qualitative recovery state. `incubator.js` binds the controls, events and export.
- `scene-core.js`: shared camera, lighting and orbit/view controls; it renders on change. Rigid incubator geometry is batched by material while moving assemblies remain separate.
- `instruments/biosafety-model.js` and `biosafety-state.js`: `createLabBiosafetyCabinet(THREE)` and `LabBiosafetyState`; `biosafety.js` supplies UI bindings, state events, camera presets and export preparation.
- `instruments/thermocycler-model.js` and `thermocycler-state.js`: `createLabThermocycler(THREE)` and `LabThermocyclerState`; `thermocycler.js` binds controls and emits state snapshots.
- `asset-tools.js`: shared geometry/material primitives and opaque mesh batching under metre-scaled roots.
- `equipment-state.js` and `equipment-runtime.js`: common state, controls, render scheduling and scene export for catalog modules.
- `collection-tools.js`: reusable labware, guarded drawer instruments and supporting-set definitions.
- `model.userData` contains equipment type, scale conventions and interaction anchors.

## Test

```sh
node tests/centrifuge.test.mjs
node tests/incubator.test.mjs
node tests/biosafety.test.mjs
node tests/thermocycler.test.mjs
node tests/remaining-library.test.mjs
node tests/remaining-geometry.test.mjs
```

The final 12 entries pass 129 state assertions and finite geometry/pose checks. Browser review also covers loading drawers, tank assembly, arm transfer and supporting-set controls. Other instrument checks are in `tests/`.

See each `*-references.md` file for the corresponding source study. See `references.md`, `incubator-references.md` and `biosafety-references.md` for sources, modeled features and explicit limitations. See `library-plan.md` for the authorized library scope.

## License

Original model and studio source follow the parent repository's MIT license. Three.js r128 is included under its MIT license in `vendor/three.LICENSE`. Manufacturer reference imagery is not included.
