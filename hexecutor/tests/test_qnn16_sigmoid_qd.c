/* SPDX-License-Identifier: LGPL-3.0-or-later */
/* Consumes the tiny binary fixture emitted by test_qnn16_sigmoid_qd.py. */
#include "qnn16_hvx.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv){
 if(argc!=2)return 2;FILE*f=fopen(argv[1],"rb");if(!f)return 2;
 uint32_t n=0;if(fread(&n,4,1,f)!=1||!n||n>4096)return 2;
 uint16_t *input=malloc(n*2),*expected=malloc(n*2),*output=calloc(n,2),*scratch=malloc(n*2);
 if(!input||!expected||!output||!scratch||fread(input,2,n,f)!=n||fread(expected,2,n,f)!=n)return 2;fclose(f);
 Qnn16Tensor in={.base=(uint8_t*)input,.base_bytes=n*2,.dims={1,1,1,n},.full={1,1,1,n},.direct=1};
 Qnn16Tensor out={.base=(uint8_t*)output,.base_bytes=n*2,.dims={1,1,1,n},.full={1,1,1,n},.direct=1};
 if(qnn16_tensor_sigmoid(&out,&in,scratch))return 2;
 for(uint32_t i=0;i<n;i++){_Float16 a,b;memcpy(&a,output+i,2);memcpy(&b,expected+i,2);if(fabsf((float)a-(float)b)>5e-4f){fprintf(stderr,"sigmoid[%u] got %.7g expected %.7g\n",i,(float)a,(float)b);return 1;}}
 puts("PASS qnn16 sigmoid vs QD_WRITES (absolute tolerance 5e-4)");free(input);free(expected);free(output);free(scratch);return 0;
}
