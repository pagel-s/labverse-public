# Instrument 01: refrigerated benchtop centrifuge

## Real examples inspected before construction

- Eppendorf Centrifuge 5425 R official operating manual, product overview (printed p. 13) and dimensions (printed p. 48): https://www.eppendorf.com/product-media/doc/en/924635/Centrifugation_Operating-manual_Centrifuge-5425-R.pdf
- Manufacturer photograph, open lid and keypad variant: https://www.eppendorf.com/product-media/img/global/964035/Eppendorf_Centrifugation_Centrifuge5425R_foil_open_filter_product.jpg?imwidth=1440
- Manufacturer product family: https://www.eppendorf.com/gb-en/Products/Centrifugation/Microcentrifuges/Centrifuge-5425-5425R-p-PF-934144
- Secondary geometry comparison: Thermo Scientific Sorvall Legend Micro 17/21 series brochure, including rotor and tube photos: https://assets.thermofisher.com/TFS-Assets/LED/brochures/D17321.pdf

Reviewed 2026-10-03. Reference images and manuals were inspected, not redistributed. All geometry and graphic labels in this asset are original procedural work. No manufacturer names, logos, or model identifiers appear on the instrument.

## Derived visual decisions

The reference housing envelope is 290 mm wide, 480 mm deep, and 260 mm high when closed. A broad rear-hinged lid covers a circular chamber; the front has an angled display/keypad panel, and side ventilation is prominent. These features, the inset lid lining, chamber gasket, rotor hub and removable cover informed the model. The control layout and colors are deliberately generic.

## Fidelity and limits

- Metre-scale exported root; construction coordinates are millimetres under a 0.001 root scale.
- Chamber, pierced rotor deck, cover, individual tube positions and lid are separate meshes.
- Internal dimensions, corner profiles and tube-holder geometry are approximations. Tube wells are simplified upright rather than manufacturer-accurate angled bores.
- Motor acceleration, deceleration and cycle duration are deliberately shortened visual demonstrations. The indicated room temperature is illustrative; refrigeration is not thermodynamically simulated.
- Equal dummy tubes in opposite pairs are the only supported loading scheme. This is not a general physical rotor-balancing algorithm.
- Cutaway mode hides the external lid only during a cycle; it does not represent a real transparent lid.
- State events are exposed for integration, but the current LabVerse Python backend is not yet connected.
- No manufacturer certification, operating-procedure accuracy, numerical physical fidelity, or percentage of photorealism is claimed.

## Validation

- 25 state assertions cover loading, cover/lid/rotor interlocks, imbalance rejection, acceleration, stopping, full cycle completion, and snapshot isolation.
- Browser review confirms the real chamber opening, corrected open/closed housing, front controls, matched-pair loading, and completion/unlocking of a demonstration cycle. No browser warnings or errors were reported in the checked run.
- The front bridge above the display was added after user review; front wall height was corrected to avoid obscuring controls.
