FROM ubuntu:22.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-14 lld-14 llvm-14 make && rm -rf /var/lib/apt/lists/*
WORKDIR /src
