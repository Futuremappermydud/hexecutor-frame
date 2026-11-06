# Simulator reference outputs

Produced by QAIRT 2.50.0.260828's x86_64 HTP simulator from the unmodified
extracted contexts. The synthetic inputs are regenerated deterministically by
`tools/stereo_probes.py make`, and `tools/prepare_inputs.py --smoke` produces
the smoke input.

- `simulator_output_hashes.json`: SHA-256 of every native output file, per case
  and model, to check reruns and future reimplementations for bit equality.
- `stereo_probes_{primary,alternate}.txt`: interior medians of both outputs.

Across all 25 probes, the alternate's float medians are within 0.025 px of the
primary's and its uint8 medians are identical. Probes are synthetic textures,
not camera data; see the main README for interpretation limits.

Both standalone NumPy models now match every output hash in this file (25
probes per model), plus smoke and a seed-1234 random input. Evidence is in
`recovered/{primary,alternate}/emulator_{probes,smoke,random,kernels}.json`.
Run `tools/verify_htp_emu.py` to repeat the probe comparison without the SDK.
