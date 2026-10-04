# Sequential 3D laboratory library

User-authorized scope: all core equipment, optional instruments, and supporting assets. Each instrument or coherent asset set must first have real references inspected, then be built, visually reviewed, behavior-tested, and uploaded to a fresh branch. Brand names and logos must not appear on models. Branches remain unmerged until requested.

## Core equipment

- [ ] Automated liquid-handling robot — existing prototype added to gallery and unbranded; conversion to reusable architecture remains
- [ ] Manual pipettes
- [x] Refrigerated benchtop centrifuge — visually reviewed; 25 state assertions passed; branch `feat/3d-centrifuge-studio`
- [ ] Mini centrifuge
- [x] PCR thermocycler — visually reviewed; 36 state assertions passed; branch `feat/3d-thermocycler-studio`
- [x] Biosafety cabinet — visually reviewed; 30 state assertions passed; branch `feat/3d-biosafety-cabinet-studio`
- [x] CO2 incubator — visually reviewed; 44 state assertions passed; branch `feat/3d-incubator-studio`
- [x] Autoclave — visually reviewed; 27 state assertions and browser door-release cycle passed; branch `feat/3d-autoclave-studio`
- [ ] Inverted microscope
- [ ] Laboratory refrigerator
- [ ] Laboratory freezer
- [ ] Vortex mixer
- [ ] Analytical balance
- [ ] pH meter
- [ ] Magnetic stirrer / hotplate
- [ ] Heating block or water bath (initial choice to be recorded after reference review)
- [ ] Microplate reader or spectrophotometer (initial choice to be recorded after reference review)

## Optional instruments, now in scope

- [ ] Cell counter
- [ ] Electrophoresis system and gel imager
- [ ] Real-time PCR instrument
- [ ] Orbital shaker / shaking incubator
- [ ] Plate shaker
- [ ] Plate sealer / seal remover
- [ ] Robotic arm or plate transporter

## Supporting asset sets

- [ ] Chemical fume hood
- [ ] Benches and sink
- [ ] Chemical-storage cabinets and cold-storage racks
- [ ] Plates, tubes, flasks, bottles, reservoirs and compatible holders
- [ ] Disposable tips, tip racks, lids, caps and seals
- [ ] Segregated waste containers and sharps bins
- [ ] Eyewash, emergency shower and spill-response supplies
- [ ] PPE

## Consistency rules

Use metre-scale roots, modular geometry and state, clear loading anchors, appropriate device interlocks, and reusable sample identities. Distinguish visual approximations from validated physics. No biological protocols or real hardware control are part of this asset-building work.

## Branch history

`feat/3d-incubator-studio` is based on `feat/3d-centrifuge-studio`, so it contains both reviewed instruments. The original centrifuge branch remains unchanged. Neither branch is merged.

`feat/3d-biosafety-cabinet-studio` is based on the incubator branch and contains all three reviewed instruments. It adds cache versioning for shared scene changes and a cabinet close-up camera.

`feat/gallery-pipetting-robot` adds the existing robot to the gallery, based on the biosafety-cabinet branch. Model/state modularization remains outstanding.

`feat/3d-thermocycler-studio` is based on the robot-gallery branch and contains five gallery entries, including the pre-existing robot prototype.

`feat/3d-autoclave-studio` is based on the thermocycler branch. It adds the front-loading autoclave and a shared gallery/viewer for subsequent assets.
