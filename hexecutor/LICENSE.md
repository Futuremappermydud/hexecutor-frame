# hexecutor license

hexecutor is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General
Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option)
any later version (`LGPL-3.0-or-later`).

hexecutor is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for
more details.

- [COPYING.LESSER](COPYING.LESSER): the GNU Lesser General Public License, version 3.
- [COPYING](COPYING): the GNU General Public License, version 3, which the LGPL incorporates.

Source files carry `SPDX-License-Identifier: LGPL-3.0-or-later`.

## Exceptions

- `device/kmod/` (the `cdsp_mem`, `cdsp_bw`, `cdsp_noc` and `cdsp_nocclk` kernel modules and their scripts) is
  `GPL-2.0`, as its SPDX headers and `MODULE_LICENSE("GPL")` say. A module is combined with the Linux kernel,
  which is GPLv2-only, so it cannot be LGPLv3.
- Model weights, captures and other data are not covered. Each model keeps its own terms; see the model
  component that documents it.

## Using hexecutor from other programs

The LGPL lets a program under any license, including MIT or proprietary, use hexecutor as a library. When that
program is distributed together with hexecutor (LGPLv3 section 4), its distributor must:

- link hexecutor as a separate, replaceable shared library (for example `libhexecutor_et.so`, built with
  `-DHEXECUTOR_ET_SHARED=ON` from `hexecutor/executorch`), or else supply the program's object files so that it
  can be relinked against a modified hexecutor;
- say that the program uses hexecutor and that hexecutor is under the LGPL, and ship `COPYING` and
  `COPYING.LESSER` with it;
- provide hexecutor's corresponding source (the version that was shipped, including any changes to it);
- not forbid the user from modifying hexecutor or reverse engineering to debug such modifications.

Changes to hexecutor itself remain under the LGPL.

## DSP programs

The DSP images built from `native/` (for example `engine.elf` and `bootstrap.elf` from
`build-scripts/build_tensor.sh`) statically link the runtime libraries (`libc`, `libm`, `libhexagon`) of the
Qualcomm Hexagon Tools used to compile them. Those libraries are not covered by this license. Anyone distributing
the built images is responsible for complying with the Hexagon Tools' redistribution terms.
