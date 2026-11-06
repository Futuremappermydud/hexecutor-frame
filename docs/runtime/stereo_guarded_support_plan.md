# Guarded-support stereo protocol

Guarded adaptive support limits support where local evidence is ambiguous or incompatible. The fixed screen varies guard and exposure handling while retaining baseline controls and the same development cases. Method/case identities and job construction are defined in `runtime/scripts/stereo_guarded_common.py`.

Judge the guard by its effect on false-valid pixels, edge leakage, texture/exposure robustness, and the amount of useful geometry retained. A stricter guard that rejects nearly everything can appear accurate without being useful. Select configurations on development data and retain held-out scenes for the final comparison.

The runner and loader verify the planned job identities, source/input snapshots, completion inventories, and hashes. Partial runs are not a complete screen. Keep the source-hash record and output checks alongside reported summaries.

This page keeps its stable filename for those manifests. New documentation/source bytes imply a new run identity; saved bundles continue to use their own snapshots. Live GPU/occlusion promotion needs separate performance and presentation evidence.
