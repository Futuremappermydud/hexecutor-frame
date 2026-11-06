# Hexecutor

Hexagon DSP execution for Steam Frame: transport, resident execution, prepared
programs, HVX/HMX kernels, scheduling, and validation tooling.

- [frame/](frame/README.md): implementation and build tools, moved unchanged from `depth-model/frame/`.
- [frame-dsp-followup/](frame-dsp-followup/README.md): early callback and DSP bring-up experiments.
- [Depth model and reference](../depth-model/README.md).

Run the existing DSP build commands from `hexecutor/frame/`. Model-resource
links in this directory preserve tools that compute their model root using
`Path(__file__).resolve()`. `depth-model/frame` is a compatibility link back
to this implementation. Existing `qd_*` C symbols and package formats retain
their names; Hexecutor is the component name.
