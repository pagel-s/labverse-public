# Sequential 3D laboratory library

User-authorized scope: all core equipment, optional instruments, and supporting assets. Each instrument or coherent asset set must first have real references inspected, then be built, visually reviewed, behavior-tested, and uploaded to a fresh branch. Brand names and logos must not appear on models. Branches remain unmerged until requested.

## Core equipment

- [ ] Automated liquid-handling robot — existing prototype added to gallery and unbranded; conversion to reusable architecture remains
- [x] Manual pipettes — three individually selectable variants; visually reviewed; 19 state assertions passed; branch `feat/3d-manual-pipettes-studio`
- [x] Refrigerated benchtop centrifuge — visually reviewed; 25 state assertions passed; branch `feat/3d-centrifuge-studio`
- [x] Mini centrifuge — visually reviewed; 26 state assertions passed; branch `feat/3d-mini-centrifuge-studio`
- [x] PCR thermocycler — visually reviewed; 36 state assertions passed; branch `feat/3d-thermocycler-studio`
- [x] Biosafety cabinet — visually reviewed; 30 state assertions passed; branch `feat/3d-biosafety-cabinet-studio`
- [x] CO2 incubator — visually reviewed; 44 state assertions passed; branch `feat/3d-incubator-studio`
- [x] Autoclave — visually reviewed; 27 state assertions and browser door-release cycle passed; branch `feat/3d-autoclave-studio`
- [x] Inverted microscope — visually reviewed; 14 state assertions passed; branch `feat/3d-inverted-microscope-studio`
- [x] Laboratory refrigerator — visually reviewed; 15 state assertions and browser checks passed; branch `feat/3d-cold-storage-studio`
- [x] Laboratory freezer — visually reviewed; 14 state assertions and browser checks passed; branch `feat/3d-cold-storage-studio`
- [x] Vortex mixer — visually reviewed; 24 state assertions and touch-mode browser checks passed; branch `feat/3d-vortex-studio`
- [x] Analytical balance — visually reviewed; 21 state assertions and tare browser check passed; branch `feat/3d-measurement-studio`
- [x] pH meter — visually reviewed; 14 state assertions and probe/readout browser checks passed; branch `feat/3d-measurement-studio`
- [x] Magnetic stirrer / hotplate — visually reviewed; 13 assertions and browser cooldown checks passed; branch `feat/3d-thermal-and-reader-studio`
- [x] Water bath — initial thermal-bath choice; visually reviewed; 14 assertions and browser cycle passed; branch `feat/3d-thermal-and-reader-studio`
- [x] Microplate reader — initial optical-reader choice; visually reviewed; 18 assertions and browser carrier-lock/cycle passed; branch `feat/3d-thermal-and-reader-studio`

## Optional instruments, now in scope

- [x] Cell counter — visually reviewed; 15 assertions and browser slide interlock/cycle passed; branch `feat/3d-cell-counter-studio`
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

`feat/3d-vortex-studio` is based on the autoclave branch and adds the seventh gallery entry.

`feat/3d-mini-centrifuge-studio` is based on the vortex branch and adds a compact twelve-position centrifuge.

`feat/3d-manual-pipettes-studio` is based on the mini-centrifuge branch.

`feat/3d-inverted-microscope-studio` is based on the manual-pipettes branch.

`feat/3d-cold-storage-studio` is based on the inverted-microscope branch and adds separately referenced refrigerator and freezer assets.

`feat/3d-measurement-studio` is based on the cold-storage branch and adds analytical balance and pH meter.

`feat/3d-thermal-and-reader-studio` is based on the measurement branch and adds hotplate, water bath and microplate reader. All 17 core types now have gallery entries; reusable robot conversion remains.

`feat/3d-cell-counter-studio` is based on the thermal-and-reader branch and adds the first optional instrument.
