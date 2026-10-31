# Exact `sign` level reduction

The existing `k_sign` expression first reduces every input element to
`s∈{-1,0,+1}`, then evaluates
`floor((double)s / output_scale + output_zero + 0.5)`, clips to `[0,65535]`,
and calls `from_i64`. `sign_levels.c` computes those three clipped values once
per output quantization. `qd_sign_select_u16` compares the input with its zero
point and returns the precomputed level; callers must still use their normal
`from_i64` path so output dtype casts remain unchanged.

Both recovered models have 13 sign commands: 12 blocked u16 `CH*2` tensors of
shape `[1,8,80,64]` and one direct u16 scalar `FH` tensor. All outputs use
zero point 32768 and scale `3.0518509447574615e-05`; their exact levels are
`(1,32768,65535)`. The large-tensor input zero points are 9196, 10701, and
61378; the scalar case uses 65432.

The exhaustive test compares the helper against an independent copy of the old
double/floor expression for every u16 input and each actual input zero point:

```sh
cc -O2 -std=c11 -ffp-contract=off -fno-fast-math -Wall -Wextra \
  frame/blocked_glue/sign_levels.c frame/blocked_glue/sign_levels_test.c \
  -lm -o /tmp/sign_levels_test
/tmp/sign_levels_test
```
