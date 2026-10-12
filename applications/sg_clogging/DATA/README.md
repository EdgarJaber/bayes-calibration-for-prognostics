# Steam-generator clogging data (not distributed)

The THYC-Puffer-DEPO simulations and the field measurements are EDF industrial data and are
**not** part of this repository: everything in this folder except this file is git-ignored.

To run the application, place the following files here (one set per steam generator, with the
tag `SGA` / `SGB` used throughout the code):

| file | content |
|---|---|
| `<SG>_MC_UNIFORM_HL.npy`, `<SG>_MC_UNIFORM_CL.npy` | Monte-Carlo simulations (hot / cold leg): input samples and clogging trajectories |
| `<SG>_TIME.npy` | simulation time grid (days) |
| `<SG>_METADATA.json` | chemical conditioning and cleaning dates |
| `<SG>_FIELD_DATA.csv` | field data: ESTICOL regression points and TVE field points |
| `thresholds.json` | `{"SGA": <tau_c^{*,A}>, "SGB": <tau_c^{*,B}>}`: the failure thresholds (one file for both SGs) |

See the main README for how to request the data.
