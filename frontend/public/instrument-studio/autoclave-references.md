# Autoclave reference study

Reviewed before modelling: [Tuttnauer laboratory autoclave brochure](https://tuttnauer.com/sites/default/files/2021-07/lab_line_nb_english-29-07-21.pdf), pages 4–5, with vertical and front-loading product photographs. This model follows the 3850 EL benchtop category and nominal 720 × 540 × 765 mm external envelope. The brochure's chamber and front-panel construction informed the geometry. Accessed 4 October 2026. No manufacturer photographs or labels are redistributed.

Original neutral geometry includes a hollow cylindrical chamber, hinged door with internal gasket, front gauge and generic screen, slide-out perforated tray, feet and rear vents. The neutral demonstration load is a generic prop. Door locking, cycle progress and a short release delay are illustrative UI states; pressure, temperature, sterilization efficacy, cooling and processing protocols are not modelled. Small fittings, chamber/tray clearances and door geometry are approximate. Export is Three.js scene JSON; behavior remains in code. The backend is not connected.

Validation: 27 state assertions passed. Browser checked loading, tray retraction, door locking during a run, completion, release delay and reopening. Console showed no errors or warnings.
