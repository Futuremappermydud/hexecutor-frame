# Runtime for QAIRT's x86_64-linux-clang qnn-net-run and HTP simulator backend,
# plus a compiler for qnn_op_harness.cpp. The SDK is mounted at run time.
FROM ubuntu:22.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libatomic1 libc++1-14 libc++abi1-14 libunwind-14 g++ \
 && rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH=/usr/lib/llvm-14/lib
