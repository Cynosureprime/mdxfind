# BUILD LAYER
# Compiles mdxfind, mdsplit, and all pinned dependencies via the Makefile.
# Uses Debian (glibc) rather than Alpine (musl) because mdxfind
# requires iconv UTF-16LE support for NTLM and related hash types.
FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    git \
    autoconf \
    automake \
    autopoint \
    libtool \
    gettext \
    pkg-config \
    perl \
    nasm \
    zlib1g-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/mdxfind
COPY . .
RUN find . -name '*.o' -delete && find . -name '*.a' -delete && \
    make deps && make all

# RUNTIME LAYER
# Minimal image containing only the built binaries.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/mdxfind/mdxfind /usr/local/bin/mdxfind
COPY --from=build /src/mdxfind/mdsplit /usr/local/bin/mdsplit

WORKDIR /data

ENTRYPOINT ["mdxfind"]
