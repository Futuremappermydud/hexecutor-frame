# Prepared mixed-range Frame validation, 2026-10-30

The previously accepted prepared engine `74d31b4b2986e3b776c0b5f967dacdc46aee82a1926864b8736ecc1a1c57ed0c` passed two mixed InputSlicePad/Im2col command intervals on Frame. The preserving bootstrap was `628e362de7563a2c3b0b5cf2a02cae49493b80931e47e52d652fedae449fa147`.

| Model | Command interval | Count | Cases | All five compared pools | DSP range time |
| --- | --- | ---: | --- | --- | --- |
| Primary | `[19,60)` | 41 | smoke, random | exact | 11.665 / 11.727 ms |
| Alternate | `[30,54)` | 24 | smoke, random | exact | 5.152 / 5.133 ms |

The verifier seeded pools from CPU execution immediately before each interval, ran the entire interval on the prepared DSP executor, then compared pools 1, 3, 4, 5, and 6 byte for byte to CPU execution. All four runs passed. No native convolution calls occurred. Both sessions exited with `[0,0,0]` errors, lease phase 2, no lease error, and equal independent host VTCM fingerprints. These are bounded range measurements with a 7.5 MiB boundary upload and download, not end-to-end inference timings.

Artifacts:

- [Verifier](bench/verify_prepared_mixed_range.py), SHA-256 `5ef4ce756a798f7b97277029e99d5a95ea71aa468b9ff7ad143d6d8b30d730ae`
- `.work/frame-direct-kernels-20261030/native-prepared-mixed-primary-19-60.json`, SHA-256 `ca7d95db28d511439366549849832a096e98dc1d844979c2aa91f3b3dfcabb61`
- `.work/frame-direct-kernels-20261030/native-prepared-mixed-alternate-30-54.json`, SHA-256 `d1f217d340b87bddd22a201c2fd82660bc163352f27cc121c6a557460142f912`

Afterward `cdsp_mem` was unloaded. Final read-only state: CTRL `[1,0,1]`, callback zero, loader/payload/arena zero, and eye client parked in `S` state.
