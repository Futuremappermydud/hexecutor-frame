# Adaptive-support stereo protocol

The adaptive-support runner screens support radius and edge threshold on the fixed development image set. Candidate selection must not consume the evaluation seed set or change the image identities after seeing results. Preserve refinement and baseline controls so a support change is not confused with a different output-selection rule.

Support may improve weakly textured surfaces while leaking across silhouettes. Assess texture, repeated patterns, exposure/noise, vertical mismatch, false-valid pixels, and boundary error together. Compare against the calibrated/baseline methods on identical data and retain invalid pixels in the accounting rather than silently excluding them.

`runtime/scripts/run_stereo_adaptive_experiment.py` records input/source hashes and the full job list before execution. Its existing source list reads this stable page through `runtime/docs`. The archived source snapshot is part of each run's provenance; rewriting this page does not retroactively change an older result.

Promoting a setting to a live matcher additionally requires device timing and real-scene geometry evidence.
