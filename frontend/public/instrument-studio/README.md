# LabVerse instrument studio

An unbranded, reference-grounded 3D asset library. Available models: refrigerated benchtop centrifuge and CO₂ incubator.

## Open

Open `index.html` directly in a WebGL-capable browser, keeping this folder intact. The pinned Three.js library is bundled locally, so no internet connection is required.

Within the LabVerse frontend, start its existing Vite server and visit `/instrument-studio/`. Files in this directory are static assets and do not modify the paper replay viewer.

## Interact

Orbit by dragging; zoom with the wheel. Use the view buttons for keyboard-accessible camera presets. Load or unload a dummy tube with the 24 position buttons, or load matched pairs. Install the rotor cover, close the lid, and start a demonstration cycle. Stop decelerates before unlocking. The complete-demo button prepares a fresh matched pair automatically.

With the lid open, remove the cover and unload tubes to remove the rotor. Removed hardware is supported on separate bench pads. The incubator export first prepares the scene, then presents an explicit Save scene JSON link. Export saves the current geometry as Three.js ObjectLoader JSON (not GLB, and without the behavior code).

The CO₂ incubator is available at `incubator.html`. Open its independent outer and glass doors, extend any of three shelves, and load or unload six sealed demonstration vessels. Retract shelves before closing the doors. Power and exposure affect a qualitative recovery indicator. The chamber does not implement biological processing or real environmental physics.

## Reuse

- `instruments/centrifuge-model.js`: `createLabCentrifuge(THREE)` returns `{model, update, targets, slotTargets, materials}`. Add `model` to a shared scene and call `update(state, dt, options)` each frame.
- `instruments/centrifuge-state.js`: DOM-independent state and interlocks. `snapshot()` provides serializable equipment state.
- `studio.js`: example viewer and UI binding. Emits `labverse:equipment-state` events on meaningful state changes. These are integration hooks, not an existing backend adapter.
- `instruments/incubator-model.js`: `createLabIncubator(THREE)` returns `{model, update, materials}`; construction uses millimetres under a metre-scaled root, like the centrifuge.
- `instruments/incubator-state.js`: `LabIncubatorState`, with door, shelf, vessel and qualitative recovery state. `incubator.js` binds the controls, events and export.
- `scene-core.js`: shared camera, lighting and orbit/view controls; it renders on change. Rigid incubator geometry is batched by material while moving assemblies remain separate.
- `model.userData` contains equipment type, scale conventions and interaction anchors.

## Test

```sh
node tests/centrifuge.test.mjs
node tests/incubator.test.mjs
```

See `references.md` and `incubator-references.md` for sources, modeled features and explicit limitations. See `library-plan.md` for the authorized library scope.

## License

Original model and studio source follow the parent repository's MIT license. Three.js r128 is included under its MIT license in `vendor/three.LICENSE`. Manufacturer reference imagery is not included.
