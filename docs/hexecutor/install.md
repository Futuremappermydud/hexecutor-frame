# Installing the Hexecutor broker on Frame

The release installer downloads a kernel-specific ARM64 bundle, verifies its checksums, then asks for sudo once. It installs the native `hexecutord` broker, its four DSP support images, and the matching `cdsp_mem` kernel module with its source. It enables a systemd service; it does not install the spatial application, model files, Python, or ExecuTorch itself. Apps that link `HexecutorBackend` supply their own delegated `.pte` model and register its embedded graph through the broker socket.

Once this code is published and a bundle for the headset's `uname -r` has been uploaded as `hexecutord-linux-aarch64-<kernel>.tar.gz` with the adjacent `.sha256` file, run:

```sh
curl -fsSL https://raw.githubusercontent.com/Futuremappermydud/frame-spatial-runtime/monorepo/hexecutor/hexecutor/packaging/get.sh | bash
```

The installer currently targets the Steam Frame's `steamos` user and systemd. It places the daemon at `/opt/hexecutor/bin/hexecutord` and the owner-only socket at `/run/hexecutor/hexecutord.sock`. Apps running as `steamos` find that socket automatically. An app can set `HEXECUTORD_SOCKET` to an explicit path; when the socket is present the delegate sends the graph from its `.pte` blob to the daemon and dispatches each run over IPC. CPU reference mode remains explicit via `HEXECUTOR_ET_REFERENCE=1`. The existing direct-session path still works with explicit `HEXECUTOR_*` artifact paths when there is no broker socket.

To inspect the service, run `systemctl status hexecutord.service` or `journalctl -u hexecutord.service -b`. The daemon starts with no model and opens the CDSP only when an app submits a graph. The current tensor session supports eight registered graphs per daemon lifetime. The service must be restarted to clear those slots; running apps lose their connection on restart. ExecuTorch applications must link the Hexecutor delegate library; installing the daemon alone does not add the delegate to existing apps.

## Building a release bundle

The repo does not contain a redistributable verified loader, a built module for every Frame kernel, or release assets. A release maintainer must supply those explicitly, verify the loader's distribution terms, and test the complete bundle on the target kernel. The package builder refuses missing inputs and records every file's SHA-256 and the module's kernel vermagic:

```sh
hexecutor/packaging/build_bundle.sh \
  "dist/hexecutord-linux-aarch64-$(uname -r).tar.gz" \
  /path/to/arm64/hexecutord /path/to/verified/qd_loader.elf \
  /path/to/engine.elf /path/to/bootstrap.elf /path/to/chunk_probe.elf \
  /path/to/kernel-matched/cdsp_mem.ko /path/to/loader-license.txt
```

The `engine.elf`, `bootstrap.elf`, and `chunk_probe.elf` may be built with `hexecutor/build-scripts/build_tensor.sh` and the explicit SDK/toolchain settings in [tensor backend](tensor-backend.md). The loader must be the validated artifact for this firmware. Upload the tarball and its `.sha256` sidecar to a GitHub Release before advertising the one-line command. Do not put model weights or captured arrays in the release.
