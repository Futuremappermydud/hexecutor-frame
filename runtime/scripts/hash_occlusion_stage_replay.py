#!/usr/bin/env python3
"""Save reproducibility hashes for a stage replay run (run from project root)."""
import hashlib
import json
import pathlib
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: hash_occlusion_stage_replay.py output-directory")
    output = pathlib.Path(sys.argv[1])
    settings = dict(line.split("=", 1) for line in (output / "replay.txt").read_text().splitlines())
    sources = [pathlib.Path(p) for p in (
        "tools/replay_occlusion_stages.cpp", "src/vulkan_stereo_pipeline.cpp",
        "include/vulkan_stereo_pipeline.hpp", "include/calibration.hpp",
        "include/occlusion_stage_diagnostics.hpp", "include/occlusion_benchmark.hpp",
        "include/occlusion_surface.hpp", "include/occlusion_completion.hpp",
        "include/depth_speckle_filter.hpp", "bin/replay_occlusion_stages", "Makefile")]
    sources += list(pathlib.Path(settings["shader_directory"]).glob("stereo_*.comp"))
    sources += list(pathlib.Path(settings["shader_directory"]).glob("stereo_*.spv"))
    sources.append(pathlib.Path(settings["recording"]))
    if settings.get("regions_file", "none") != "none":
        sources.append(pathlib.Path(settings["regions_file"]))
    result = {"settings": settings, "sha256": {str(path): sha256(path) for path in sorted(set(sources))}}
    (output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
