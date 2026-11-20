/* SPDX-License-Identifier: LGPL-3.0-or-later */
ENTRY(callback)
PHDRS { code PT_LOAD FLAGS(5); }
SECTIONS {
    . = QD_LINK_BASE;
    .text : { KEEP(*(.text.entry)) *(.text*) } :code
    .rodata : { *(.rodata*) } :code
    .data : { *(.data*) } :code
    .bss : { *(.bss*) *(COMMON) } :code
    ASSERT(. <= QD_LINK_BASE + 4096, "HVX callback exceeds its dedicated code page")
    /DISCARD/ : { *(.comment) *(.note*) *(.eh_frame*) }
}
