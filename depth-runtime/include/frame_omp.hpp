#pragma once

// Row/band parallel loop; a plain loop when OpenMP is off. Bodies must be independent.
#ifdef _OPENMP
#define FRAME_OMP_FOR _Pragma("omp parallel for schedule(static)")
#else
#define FRAME_OMP_FOR
#endif
