#!/usr/bin/env python3
"""Run an extracted context with stock qnn-net-run on a configured QNN device.

The device must already have functioning QuRT/FastRPC and matching unsigned
DSP libraries. This script does not install firmware or change services.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", required=True, type=Path)
    parser.add_argument("--target", default="aarch64-oe-linux-gcc11.2",
                        help="SDK target directory; use aarch64-android on Android")
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--input-list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dsp-libs", required=True, type=Path,
                        help="Directory containing matching V75 skel and dependencies")
    parser.add_argument("--op-package", help="Optional QNN library:provider:HTP registration string")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sdk, output = args.sdk.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lib = sdk / "lib" / args.target
    backend_config = output / "htp.json"
    backend_config.write_text(json.dumps({
        "devices": [{"device_id": 0, "soc_id": 57, "dsp_arch": "v75", "pd_session": "unsigned"}]
    }, indent=2) + "\n")
    config = output / "extensions.json"
    config.write_text(json.dumps({"backend_extensions": {
        "shared_library_path": str(lib / "libQnnHtpNetRunExtensions.so"),
        "config_file_path": str(backend_config)
    }}, indent=2) + "\n")
    cmd = [str(sdk / "bin" / args.target / "qnn-net-run"),
           "--backend", str(lib / "libQnnHtp.so"),
           "--retrieve_context", str(args.context.resolve()),
           "--input_list", str(args.input_list.resolve()),
           "--use_native_input_files", "--use_native_output_files",
           "--config_file", str(config), "--output_dir", str(output / "results"),
           "--log_level", "verbose"]
    if args.op_package:
        cmd.extend(["--op_packages", args.op_package])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    # Qualcomm DSP paths use semicolons. Preserve system/vendor search roots.
    env["ADSP_LIBRARY_PATH"] = ";".join([str(args.dsp_libs.resolve()),
        env.get("ADSP_LIBRARY_PATH", "/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp")])
    (output / "command.txt").write_text(shlex.join(cmd) + "\n")
    print(shlex.join(cmd), flush=True)
    if args.dry_run:
        return
    for p in [Path(cmd[0]), lib / "libQnnHtp.so", args.context, args.input_list, args.dsp_libs]:
        if not p.exists():
            parser.error(f"missing required path: {p}")
    with (output / "run.log").open("w") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    print((output / "run.log").read_text())
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
