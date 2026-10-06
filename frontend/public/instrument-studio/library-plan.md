# Sequential 3D laboratory library

User-authorized scope: all core equipment, optional instruments, and supporting assets. Each instrument or coherent asset set must first have real references inspected, then be built, visually reviewed, behavior-tested, and uploaded to a fresh branch. Brand names and logos must not appear on models. Branches remain unmerged until requested.

## Core equipment

- [x] Automated liquid-handling robot — existing interactive prototype added to gallery and unbranded; legacy controller retained
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
- [x] Electrophoresis system and gel imager
- [x] Real-time PCR instrument
- [x] Orbital shaker — selected open-platform variant; visually reviewed; 14 assertions, browser restraint/cycle checks and scene export passed; branch `feat/3d-orbital-shaker-studio`
- [x] Plate shaker — four loading positions; visually reviewed; 22 state assertions and browser cycle/clamp checks passed; branch `feat/3d-plate-handling-studio`
- [x] Plate sealer — selected heat-sealer variant with manual foil removal; visually reviewed; 18 assertions, browser drawer/cycle and export checks passed; branch `feat/3d-plate-handling-studio`
- [x] Robotic arm or plate transporter

## Supporting asset sets

- [x] Chemical fume hood
- [x] Benches and sink
- [x] Chemical-storage cabinets and cold-storage racks
- [x] Plates, tubes, flasks, bottles, reservoirs and compatible holders
- [x] Disposable tips, tip racks, lids, caps and seals
- [x] Segregated waste containers and sharps bins
- [x] Eyewash, emergency shower and spill-response supplies
- [x] PPE

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

`feat/3d-orbital-shaker-studio` is based on the cell-counter branch. Gallery count: 19.

`feat/3d-plate-handling-studio` is based on the orbital-shaker branch and adds a four-position microplate shaker and heat sealer. Gallery count: 21.

## Complete initial model coverage — 6 October 2026

All requested categories have a model or coherent asset set: 33 gallery entries. The final 12 entries were reviewed in the browser and pass 129 state assertions plus geometry/pose checks. References and approximation limits are recorded per entry. The transport arm was rebuilt with connected joints, proportioned links, a parallel gripper and analytical two-link positioning through the transfer.

`feat/3d-complete-lab-asset-library` is based on `feat/3d-plate-handling-studio`; it contains the complete initial library. No branches are merged.

This completes initial asset coverage, not validated digital twins or a quantified photorealism target. Further work includes integration into the simulation backend, legacy pipetting-robot modularization, collision/dynamics validation and higher-detail surface/optics work.
