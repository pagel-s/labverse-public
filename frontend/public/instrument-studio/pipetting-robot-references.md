# Pipetting robot · existing model added to the gallery

The original reference is the user-supplied photograph of an enclosed benchtop liquid-handling robot, provided when this prototype was built on 3 October 2026. It informed the enclosure, gantry and pipette head. Later user-directed revisions added an attached door handle and darkened the frame. This gallery entry reuses that completed model; it is not a newly researched or dimensionally validated replica.

## Gallery adaptation

- Preserves the existing geometry, dark frame, door handle, orbit/deck views, enclosure and label toggles, and liquid-handling controls.
- Replaces the company wordmark and head emblem with neutral labelling or plain surfaces.
- Loads the already bundled Three.js r128 locally, so the gallery no longer needs a CDN for this instrument.
- Adds navigation to and from all other gallery instruments.
- Keeps the original standalone prototype unchanged.

## Scope and limitations

Interactive demonstration includes selecting labware and wells, moving the head, attaching and discarding tips, transferring generic liquid, pausing, resetting, and a four-well demonstration. The capacity checks and quantities belong to an illustrative state model, not a real robot controller or experimental protocol. There is no calibrated liquid physics, collision avoidance or door safety interlock. The scene uses its existing millimetre-style coordinates.

This entry is still a self-contained prototype. Extracting reusable metre-scale geometry/state modules and connecting it to the LabVerse backend remain separate library tasks. The original reference image is not redistributed.

## Gallery validation · 4 October 2026

Verified all four gallery pages link to the robot and all local links resolve. Confirmed local renderer loading and removal of the company wordmark/emblem. Browser smoke check confirmed the no-tip aspiration guard and an animated tip pickup, leaving 95 tips available; no console warnings or errors were reported. Existing simulation logic was retained, rather than rewritten.

## State integration development — 21 October 2026

The existing referenced geometry is unchanged. A reusable DOM-independent
`LabPipettingRobotState` now owns transfer quantities, tip custody, strict snapshot
restore and a conceptual door-motion interlock. The viewer commits transfers
through that state and emits `labverse:equipment-state` snapshots. State tests
pass for conservation, invalid restore, custody, tip capacity and door locking.
Browser verification of this revision remains pending because the browser tool
was unavailable. Metre-scale model extraction remains outstanding. This revision
is not yet published as a completed build.

## Metre-scale geometry extraction — 21 October 2026

`createLabPipettingRobot(THREE)` now returns a metre-scaled `model`, `update`,
materials, interaction targets, metre-valued loading anchors, and moving parts.
The original referenced geometry, dark frame and attached door handle are retained.
`update(state, dt, {position})` accepts the head position in metres. The legacy
viewer wraps the model in a 1000-scale display group to preserve its millimetre
camera and motion path; reusable consumers must add `model` directly instead.
Two independently constructed models do not share moving assemblies.

All 24 library test files passed in this continuation, including finite geometry,
sub-metre envelope, state-driven liquid/door poses, and independent instances.
These are software/geometry checks, not visual inspection or collision validation.
Browser inventory is currently empty, and GitHub reports an invalid saved token.
Visual review and fresh-branch publication remain pending. The implementation is
not a certified or dimensionally validated manufacturer replica.
