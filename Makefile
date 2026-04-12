# Platform detection based on hashpipe Makefile framework

CC = cc
AR = ar
RANLIB = ranlib
TOPDIR := $(shell pwd)

# ---- Platform detection ----
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Architecture defines
ifeq ($(UNAME_M),x86_64)
  ARCHOPT = -DINTEL
else ifeq ($(UNAME_M),i386)
  ARCHOPT = -DINTEL
else ifeq ($(UNAME_M),i686)
  ARCHOPT = -DINTEL
else ifeq ($(UNAME_M),ppc64le)
  ARCHOPT = -DPOWERPC
else ifeq ($(UNAME_M),ppc64)
  ARCHOPT = -DPOWERPC
else ifeq ($(UNAME_M),aarch64)
  ARCHOPT = -DARM
else ifeq ($(UNAME_M),arm64)
  ARCHOPT = -DARM
else
  ARCHOPT =
endif

# OS-specific flags
ifeq ($(UNAME_S),Darwin)
  OSOPT = -DMACOSX
  ICONV = /opt/local/lib/libiconv.a
  LDEXTRA =
  INCEXTRA = -I/opt/local/include
  # Metal GPU acceleration on macOS (Apple Silicon and x86_64 with discrete/integrated GPU)
  METAL_GPU = 1
  OSOPT += -DMETAL_GPU=1
  LDEXTRA += -framework Metal -framework Foundation
else ifeq ($(UNAME_S),FreeBSD)
  OSOPT =
  ICONV = /usr/local/lib/libiconv.a
  LDEXTRA = -Wl,--allow-multiple-definition -L/usr/local/lib
  INCEXTRA = -I/usr/local/include
  # OpenCL GPU acceleration on FreeBSD (requires: pkg install opencl ocl-icd)
  ifeq ($(UNAME_M),x86_64)
    ifneq ($(wildcard /usr/include/CL/cl.h /usr/local/include/CL/cl.h),)
      OPENCL_GPU = 1
      OSOPT += -DOPENCL_GPU=1
    endif
  endif
else
  # Linux and others
  OSOPT =
  ICONV =
  LDEXTRA = -ldl
  INCEXTRA = -I/usr/local/include
  # OpenCL GPU acceleration on Linux (requires: OpenCL headers + runtime or dynload)
  ifneq ($(wildcard /usr/include/CL/cl.h /usr/local/include/CL/cl.h),)
    OPENCL_GPU = 1
    OSOPT += -DOPENCL_GPU=1
  endif
endif

# GCC needs -fgnu89-inline to emit out-of-line copies of inline functions
ifneq ($(UNAME_S),Darwin)
  OSOPT += -fgnu89-inline
endif

CFLAGS = -fomit-frame-pointer -pthread -O3 $(ARCHOPT) $(OSOPT) $(INCEXTRA) -I.
LDFLAGS = -pthread -O3

# Static libraries (expected in current directory or subdirectories)
LIBS = libssl.a libcrypto.a libsph.a libmhash.a librhash.a md6.a \
       gosthash/gost2012/gost2012.a bcrypt-master/bcrypt.a \
       argon2/argon2.a libJudy.a libpcre.a lm/lm.a liblzma.a libbz2.a libzstd.a $(ICONV)

# yescrypt (object files, not a .a archive)
YESCRYPT_OBJS = yescrypt/yescrypt-common.o yescrypt/yescrypt-opt.o \
                yescrypt/sha256.o yescrypt/insecure_memzero.o

MDXFIND_OBJS = mdxfind.o yarn.o gosthash/gosthash.o rmd128.o mymd5.o \
               ruleproc.o crypt-des.o myprogress.o
MDSPLIT_OBJS = mdsplit.o

# sha1_block.s requires yasm and is x86_64-only
ifeq ($(UNAME_M),x86_64)
  MDXFIND_OBJS += sha1_block.o sha1_shani.o
endif

# Metal GPU objects (macOS only)
ifdef METAL_GPU
  MDXFIND_OBJS += gpu_metal.o gpujob_metal.o
endif

# OpenCL GPU objects (Linux, FreeBSD, aarch64)
ifdef OPENCL_GPU
  MDXFIND_OBJS += gpu/gpu_opencl.o gpu/gpujob_opencl.o gpu/opencl_dynload.o
  CFLAGS += -Igpu
endif

# argon2 fill-block selection: SSE on x86_64, portable ref elsewhere
ifeq ($(UNAME_M),x86_64)
  ARGON2_FILL_SRC = opt.c
  ARGON2_FILL_OBJ = opt.o
else ifeq ($(UNAME_M),amd64)
  ARGON2_FILL_SRC = opt.c
  ARGON2_FILL_OBJ = opt.o
else
  ARGON2_FILL_SRC = ref.c
  ARGON2_FILL_OBJ = ref.o
endif

all: mdxfind mdsplit getpass mdxpause

mdxfind.o: mdxfind.c mdxfind.h job_types.h gpujob.h
	$(CC) $(CFLAGS) -c mdxfind.c

ruleproc.o: ruleproc.c mdxfind.h
	$(CC) $(CFLAGS) -c ruleproc.c

yarn.o: yarn.c yarn.h
	$(CC) $(CFLAGS) -c yarn.c

myprogress.o: myprogress.c
	$(CC) $(CFLAGS) -c myprogress.c

crypt-des.o: crypt-des.c
	$(CC) $(CFLAGS) -c crypt-des.c

rmd128.o: rmd128.c rmd128.h
	$(CC) $(CFLAGS) -c rmd128.c

mymd5.o: mymd5.c
	$(CC) $(CFLAGS) -c mymd5.c

gosthash/gosthash.o: gosthash/gosthash.c gosthash/gosthash.h
	$(CC) $(CFLAGS) -c -o gosthash/gosthash.o gosthash/gosthash.c

sha1_block.o: sha1_block.s
ifeq ($(UNAME_S),Darwin)
	yasm -DINTEL_SHA1_UPDATE_DEFAULT_DISPATCH=_sha1_step \
	     -DINTEL_SHA1_SINGLEBLOCK=1 \
	     -DINTEL_SHA1_UPDATE_FUNCNAME=_sha1_update_intel \
	     -f macho64 -o sha1_block.o sha1_block.s
else
	nasm -DINTEL_SHA1_UPDATE_DEFAULT_DISPATCH=sha1_step \
	     -DINTEL_SHA1_SINGLEBLOCK=1 \
	     -f elf64 -o sha1_block.o sha1_block.s
endif

sha1_shani.o: sha1_shani.c
	$(CC) -O3 -msha -msse4.1 -c sha1_shani.c

# Metal GPU source files (Objective-C++)
ifdef METAL_GPU
gpu_metal.o: gpu_metal.m gpu_metal.h gpujob.h job_types.h \
             gpu/metal_common_str.h gpu/metal_md5salt_str.h gpu/metal_md5saltpass_str.h \
             gpu/metal_md5_md5saltmd5pass_str.h gpu/metal_sha256_str.h gpu/metal_phpbb3_str.h \
             gpu/metal_descrypt_str.h gpu/metal_md5unsalted_str.h gpu/metal_md4unsalted_str.h \
             gpu/metal_sha1unsalted_str.h gpu/metal_sha256unsalted_str.h gpu/metal_sha512unsalted_str.h \
             gpu/metal_md6256unsalted_str.h gpu/metal_wrlunsalted_str.h gpu/metal_keccakunsalted_str.h \
             gpu/metal_mysql3unsalted_str.h gpu/metal_hmac_sha256_str.h gpu/metal_hmac_sha512_str.h \
             gpu/metal_hmac_rmd160_str.h gpu/metal_hmac_rmd320_str.h gpu/metal_hmac_blake2s_str.h \
             gpu/metal_streebog_str.h gpu/metal_sha256crypt_str.h gpu/metal_sha512crypt_str.h \
             gpu/metal_rmd160unsalted_str.h gpu/metal_blake2s256unsalted_str.h \
             gpu/metal_bcrypt_str.h
	$(CC) -x objective-c++ $(CFLAGS) -std=c++11 -c gpu_metal.m

gpujob_metal.o: gpujob_metal.m gpujob.h job_types.h gpu_metal.h mdxfind.h
ifeq ($(UNAME_M),x86_64)
	$(CC) -x objective-c++ $(CFLAGS) -std=c++11 -include emmintrin.h -c gpujob_metal.m
else
	$(CC) -x objective-c++ $(CFLAGS) -std=c++11 -c gpujob_metal.m
endif
endif

# Auto-generated JOB_ type constants for GPU headers
job_types.h: mdxfind.c
	(echo '/* Auto-generated from mdxfind.c -- do not edit */'; echo '#ifndef NO_JOB_TYPES'; grep '^#define JOB_' mdxfind.c; echo '#endif') > job_types.h



ifdef OPENCL_GPU
gpu/gpu_opencl.o: gpu/gpu_opencl.c gpu/gpu_opencl.h gpu/gpu_kernels_str.h gpujob.h job_types.h \
                  gpu/gpu_common_str.h gpu/gpu_md5salt_str.h gpu/gpu_md5saltpass_str.h \
                  gpu/gpu_md5iter_str.h gpu/gpu_phpbb3_str.h gpu/gpu_md5crypt_str.h \
                  gpu/gpu_md5_md5saltmd5pass_str.h gpu/gpu_sha1_str.h gpu/gpu_sha256_str.h \
                  gpu/gpu_md5mask_str.h gpu/gpu_descrypt_str.h gpu/gpu_md5unsalted_str.h \
                  gpu/gpu_md4unsalted_str.h gpu/gpu_sha1unsalted_str.h \
                  gpu/gpu_sha256unsalted_str.h gpu/gpu_sha512unsalted_str.h \
                  gpu/gpu_md6256unsalted_str.h gpu/gpu_wrlunsalted_str.h \
                  gpu/gpu_keccakunsalted_str.h gpu/gpu_mysql3unsalted_str.h \
                  gpu/gpu_hmac_sha256_str.h gpu/gpu_hmac_sha512_str.h \
                  gpu/gpu_hmac_rmd160_str.h gpu/gpu_hmac_rmd320_str.h \
                  gpu/gpu_hmac_blake2s_str.h gpu/gpu_streebog_str.h \
                  gpu/gpu_sha256crypt_str.h gpu/gpu_sha512crypt_str.h \
                  gpu/gpu_rmd160unsalted_str.h gpu/gpu_blake2s256unsalted_str.h \
                  gpu/gpu_bcrypt_str.h
	$(CC) -DOPENCL_GPU=1 -DCL_TARGET_OPENCL_VERSION=120 -I. -Igpu $(INCEXTRA) -O3 -pthread -c gpu/gpu_opencl.c -o gpu/gpu_opencl.o

gpu/gpujob_opencl.o: gpu/gpujob_opencl.c gpu/gpu_opencl.h gpujob.h job_types.h mdxfind.h
	$(CC) -DOPENCL_GPU=1 -I. -Igpu $(INCEXTRA) -O3 -pthread -c gpu/gpujob_opencl.c -o gpu/gpujob_opencl.o

gpu/opencl_dynload.o: gpu/opencl_dynload.c gpu/opencl_dynload.h
	$(CC) -DOPENCL_GPU=1 -DCL_TARGET_OPENCL_VERSION=120 -I. -Igpu $(INCEXTRA) -O3 -pthread -c gpu/opencl_dynload.c -o gpu/opencl_dynload.o
endif

mdsplit.o: mdsplit.c
	$(CC) $(CFLAGS) -c mdsplit.c

# LM hash library (bundled)
lm/lm.a:
	cd lm && $(CC) -O3 -w -c DES.c LMhash.c && \
	$(AR) rcs lm.a DES.o LMhash.o

argon2/argon2.a:
	cd argon2 && $(CC) $(CFLAGS) -c argon2.c core.c encoding.c thread.c $(ARGON2_FILL_SRC) && \
	$(CC) $(CFLAGS) -c -o blake2b.o blake2/blake2b.c && \
	$(AR) rcs argon2.a argon2.o core.o encoding.o thread.o $(ARGON2_FILL_OBJ) blake2b.o

mdxfind: $(MDXFIND_OBJS) argon2/argon2.a lm/lm.a
	$(CC) $(LDFLAGS) -o mdxfind $(MDXFIND_OBJS) $(YESCRYPT_OBJS) $(LIBS) $(LDEXTRA) -lz

mdsplit: $(MDSPLIT_OBJS)
	$(CC) $(LDFLAGS) -o mdsplit $(MDSPLIT_OBJS) libJudy.a

getpass.o: getpass.c
	$(CC) $(CFLAGS) -c getpass.c

getpass: getpass.o
	$(CC) $(LDFLAGS) -o getpass getpass.o

mdxpause: mdxpause.c
	$(CC) -O3 -o mdxpause mdxpause.c

clean:
	rm -f mdxfind mdsplit $(MDXFIND_OBJS) $(MDSPLIT_OBJS)
	rm -f argon2/*.o argon2/argon2.a
	rm -f lm/*.o lm/lm.a
	rm -f gosthash/*.o
	rm -f gpu/*.o

distclean: clean
	rm -rf deps

# ======================================================================
# Optional: pull and build all dependencies from original sources
# Usage: make deps
#
# Each library is cloned from its authoritative repository and pinned
# to a specific tag or commit hash.  After checkout, the commit hash
# is verified -- the build aborts if it does not match.
#
# Built artifacts (.a archives and headers) are copied into the
# mdxfind source tree so that "make" finds them without additional
# configuration.
#
# Requires: git, a C compiler, make, autotools (for mhash/Judy), yasm.
# ======================================================================

DEPDIR = $(TOPDIR)/deps

# ---- Pinned versions ----
# OpenSSL 1.1.1w  -- last public release of the 1.1.1 LTS branch
OPENSSL_REPO   = https://github.com/openssl/openssl.git
OPENSSL_TAG    = OpenSSL_1_1_1w
OPENSSL_COMMIT = e04bd3433fd84e1861bf258ea37928d9845e6a86

# sphlib (Thomas Pornin) -- SHA-3 candidates and classic hashes
SPHLIB_REPO    = https://github.com/pornin/sphlib.git
SPHLIB_COMMIT  = 15b6b8d8f3e4a43c58ba102d712fa6b8a3317035

# libmhash 0.9.9.9 (Distrotech mirror of SourceForge canonical)
MHASH_REPO     = https://github.com/Distrotech/mhash.git
MHASH_BRANCH   = distrotech-mhash
MHASH_COMMIT   = d8cb1ed69b146d5001de1e083a44c12dc50d2e89

# RHash 1.4.6 -- latest stable release
RHASH_REPO     = https://github.com/rhash/RHash.git
RHASH_TAG      = v1.4.6
RHASH_COMMIT   = 6562de382954d9893442b89b0e8b5c513eea6a88

# MD6 reference implementation (Ron Rivest, MIT) via retter collection
MD6_REPO       = https://github.com/brandondahler/retter.git
MD6_COMMIT     = eaba612ef34c35ac6cce6a1778e91908ec62bd0e

# Streebog / GOST R 34.11-2012 (Markku-Juhani O. Saarinen)
# Core primitives from brutus (CAESAR test framework); streebog.c/streebog.h
# wrapper from stricat (not on GitHub) bundled in gosthash/gost2012/
STREEBOG_REPO  = https://github.com/mjosaarinen/brutus.git
STREEBOG_COMMIT = 04509d7c9009015fc13ffcc49324e4bbcaa569ec

# crypt_blowfish / bcrypt (Openwall) -- tag 1.3
BCRYPT_REPO    = https://github.com/openwall/crypt_blowfish.git
BCRYPT_TAG     = CRYPT_BLOWFISH_1_3
BCRYPT_COMMIT  = 3354bb81eea489e972b0a7c63231514ab34f73a0

# libJudy (netdata fork of HP's Judy arrays) -- v1.0.5-netdata2
JUDY_REPO      = https://github.com/netdata/libjudy.git
JUDY_TAG       = v1.0.5-netdata2
JUDY_COMMIT    = 777c9f4a8faf3f524d0afa39fb4577876b6b646d

# yescrypt 1.1.0 (Openwall -- Colin Percival / Alexander Peslyak)
YESCRYPT_REPO  = https://github.com/openwall/yescrypt.git
YESCRYPT_TAG   = YESCRYPT_1_1_0
YESCRYPT_COMMIT = 0731cce8fdd1636f0bd6b7ce742e0d2a2154c6e0

# PCRE 8.45 -- last release of PCRE1 (Philip Hazel, via luvit mirror)
PCRE_REPO      = https://github.com/luvit/pcre.git
PCRE_COMMIT    = 5c78f7d5d7f41bdd4be4867ef3a1030af3e973e3

# bzip2 1.0.8
BZIP2_REPO     = https://gitlab.com/bzip2/bzip2.git
BZIP2_TAG      = bzip2-1.0.8

# xz/liblzma 5.4.5
XZ_REPO        = https://github.com/tukaani-project/xz.git
XZ_TAG         = v5.4.5

# zstd 1.5.5
ZSTD_REPO      = https://github.com/facebook/zstd.git
ZSTD_TAG       = v1.5.5

deps: dep-openssl dep-sphlib dep-mhash dep-rhash dep-md6 dep-streebog dep-bcrypt dep-judy dep-yescrypt dep-pcre dep-bzip2 dep-xz dep-zstd
	@echo ""
	@echo "All dependencies built. Run 'make' to build mdxfind and mdsplit."

# ---- OpenSSL ----
dep-openssl:
	@echo "==> OpenSSL ($(OPENSSL_TAG))"
	@if [ -f $(TOPDIR)/libssl.a ] && [ -f $(TOPDIR)/libcrypto.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone --depth 1 --branch $(OPENSSL_TAG) $(OPENSSL_REPO) $(DEPDIR)/openssl; \
	GOT=$$(cd $(DEPDIR)/openssl && git rev-parse HEAD); \
	if [ "$$GOT" != "$(OPENSSL_COMMIT)" ]; then \
		echo "ERROR: OpenSSL HEAD $$GOT != expected $(OPENSSL_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	cd $(DEPDIR)/openssl && \
	./config no-shared no-dso no-engine no-tests && \
	$(MAKE) build_libs; \
	cp $(DEPDIR)/openssl/libssl.a $(TOPDIR)/; \
	cp $(DEPDIR)/openssl/libcrypto.a $(TOPDIR)/; \
	mkdir -p $(TOPDIR)/openssl; \
	cp -r $(DEPDIR)/openssl/include/openssl/* $(TOPDIR)/openssl/; \
	echo "  libssl.a + libcrypto.a installed"

# ---- sphlib ----
dep-sphlib:
	@echo "==> sphlib ($(SPHLIB_COMMIT))"
	@if [ -f $(TOPDIR)/libsph.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone $(SPHLIB_REPO) $(DEPDIR)/sphlib; \
	GOT=$$(cd $(DEPDIR)/sphlib && git rev-parse HEAD); \
	if [ "$$GOT" != "$(SPHLIB_COMMIT)" ]; then \
		echo "ERROR: sphlib HEAD $$GOT != expected $(SPHLIB_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	cd $(DEPDIR)/sphlib/c && \
	SPH_SRCS=$$(ls *.c | grep -v '^test_' | grep -v '^hsum' | grep -v '^speed' \
		| grep -v 'sha3nist' | grep -v '^utest' | grep -v '_helper\.c') && \
	$(CC) -O3 -w -fno-strict-aliasing -c $$SPH_SRCS && \
	$(AR) rcs libsph.a *.o; \
	cp $(DEPDIR)/sphlib/c/libsph.a $(TOPDIR)/; \
	cp $(DEPDIR)/sphlib/c/sph_*.h $(TOPDIR)/; \
	echo "  libsph.a installed"

# ---- libmhash ----
dep-mhash:
	@echo "==> libmhash ($(MHASH_COMMIT))"
	@if [ -f $(TOPDIR)/libmhash.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone --branch $(MHASH_BRANCH) $(MHASH_REPO) $(DEPDIR)/mhash; \
	GOT=$$(cd $(DEPDIR)/mhash && git rev-parse HEAD); \
	if [ "$$GOT" != "$(MHASH_COMMIT)" ]; then \
		echo "ERROR: libmhash HEAD $$GOT != expected $(MHASH_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	cd $(DEPDIR)/mhash && \
	autoreconf -i && \
	CFLAGS="-O2 -w -std=gnu89" ./configure --enable-static --disable-shared && \
	$(MAKE); \
	cp $(DEPDIR)/mhash/lib/.libs/libmhash.a $(TOPDIR)/; \
	cp $(DEPDIR)/mhash/include/mhash.h $(TOPDIR)/; \
	cp -r $(DEPDIR)/mhash/include/mutils $(TOPDIR)/; \
	echo "  libmhash.a installed"

# ---- librhash ----
dep-rhash:
	@echo "==> RHash ($(RHASH_TAG))"
	@if [ -f $(TOPDIR)/librhash.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone --depth 1 --branch $(RHASH_TAG) $(RHASH_REPO) $(DEPDIR)/rhash; \
	GOT=$$(cd $(DEPDIR)/rhash && git rev-parse HEAD); \
	if [ "$$GOT" != "$(RHASH_COMMIT)" ]; then \
		echo "ERROR: RHash HEAD $$GOT != expected $(RHASH_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	cd $(DEPDIR)/rhash && \
	./configure --enable-lib-static && \
	cd librhash && \
	$(MAKE) lib-static; \
	cp $(DEPDIR)/rhash/librhash/librhash.a $(TOPDIR)/; \
	cp $(DEPDIR)/rhash/librhash/rhash.h $(TOPDIR)/; \
	cp $(DEPDIR)/rhash/librhash/rhash_torrent.h $(TOPDIR)/; \
	echo "  librhash.a installed"

# ---- md6 ----
dep-md6:
	@echo "==> MD6 (Rivest reference impl)"
	@if [ -f $(TOPDIR)/md6.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone $(MD6_REPO) $(DEPDIR)/retter; \
	GOT=$$(cd $(DEPDIR)/retter && git rev-parse HEAD); \
	if [ "$$GOT" != "$(MD6_COMMIT)" ]; then \
		echo "ERROR: MD6/retter HEAD $$GOT != expected $(MD6_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	cd $(DEPDIR)/retter/MD6 && \
	$(CC) -O3 -w -fcommon -c md6_compress.c md6_mode.c && \
	$(AR) rcs md6.a md6_compress.o md6_mode.o; \
	cp $(DEPDIR)/retter/MD6/md6.a $(TOPDIR)/; \
	cp $(DEPDIR)/retter/MD6/md6.h $(TOPDIR)/; \
	echo "  md6.a installed"

# ---- Streebog / GOST R 34.11-2012 ----
# Core primitives (sbob_pi64.c, sbob_tab64.c, stribob.h) from mjosaarinen/brutus.
# Standalone hash wrapper (streebog.c, streebog.h) from Saarinen's stricat,
# bundled in gosthash/gost2012/ (not published on GitHub).
dep-streebog:
	@echo "==> Streebog ($(STREEBOG_COMMIT))"
	@if [ -f $(TOPDIR)/gosthash/gost2012/gost2012.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone $(STREEBOG_REPO) $(DEPDIR)/brutus; \
	GOT=$$(cd $(DEPDIR)/brutus && git rev-parse HEAD); \
	if [ "$$GOT" != "$(STREEBOG_COMMIT)" ]; then \
		echo "ERROR: brutus HEAD $$GOT != expected $(STREEBOG_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	mkdir -p $(TOPDIR)/gosthash/gost2012; \
	cp $(DEPDIR)/brutus/crypto_aead_round1/stribob192r1/ref/sbob_pi64.c $(TOPDIR)/gosthash/gost2012/; \
	cp $(DEPDIR)/brutus/crypto_aead_round1/stribob192r1/ref/sbob_tab64.c $(TOPDIR)/gosthash/gost2012/; \
	cp $(DEPDIR)/brutus/crypto_aead_round1/stribob192r1/ref/stribob.h $(TOPDIR)/gosthash/gost2012/; \
	cd $(TOPDIR)/gosthash/gost2012 && \
	$(CC) -O3 -w -c sbob_pi64.c sbob_tab64.c streebog.c && \
	$(AR) rcs gost2012.a sbob_pi64.o sbob_tab64.o streebog.o; \
	echo "  gost2012.a built"

# ---- bcrypt (Openwall crypt_blowfish) ----
dep-bcrypt:
	@echo "==> crypt_blowfish ($(BCRYPT_TAG))"
	@if [ -f $(TOPDIR)/bcrypt-master/bcrypt.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone --depth 1 --branch $(BCRYPT_TAG) $(BCRYPT_REPO) $(DEPDIR)/crypt_blowfish; \
	GOT=$$(cd $(DEPDIR)/crypt_blowfish && git rev-parse HEAD); \
	if [ "$$GOT" != "$(BCRYPT_COMMIT)" ]; then \
		echo "ERROR: crypt_blowfish HEAD $$GOT != expected $(BCRYPT_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	cd $(DEPDIR)/crypt_blowfish && \
	$(CC) -O3 -w -c crypt_blowfish.c crypt_gensalt.c wrapper.c && \
	$(AR) rcs bcrypt.a crypt_blowfish.o crypt_gensalt.o wrapper.o; \
	mkdir -p $(TOPDIR)/bcrypt-master; \
	cp $(DEPDIR)/crypt_blowfish/bcrypt.a $(TOPDIR)/bcrypt-master/; \
	echo "  bcrypt.a installed"

# ---- libJudy ----
dep-judy:
	@echo "==> libJudy ($(JUDY_TAG))"
	@if [ -f $(TOPDIR)/libJudy.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone --depth 1 --branch $(JUDY_TAG) $(JUDY_REPO) $(DEPDIR)/libjudy; \
	GOT=$$(cd $(DEPDIR)/libjudy && git rev-parse HEAD); \
	if [ "$$GOT" != "$(JUDY_COMMIT)" ]; then \
		echo "ERROR: libJudy HEAD $$GOT != expected $(JUDY_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	cd $(DEPDIR)/libjudy && \
	autoreconf -i && \
	./configure --enable-static --disable-shared && \
	mkdir -p doc/man/man3 && \
	$(MAKE); \
	cp $(DEPDIR)/libjudy/src/obj/.libs/libJudy.a $(TOPDIR)/; \
	cp $(DEPDIR)/libjudy/src/Judy.h $(TOPDIR)/; \
	echo "  libJudy.a installed"

# ---- yescrypt ----
dep-yescrypt:
	@echo "==> yescrypt ($(YESCRYPT_TAG))"
	@if [ -f $(TOPDIR)/yescrypt/yescrypt-opt.o ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone --depth 1 --branch $(YESCRYPT_TAG) $(YESCRYPT_REPO) $(DEPDIR)/yescrypt; \
	GOT=$$(cd $(DEPDIR)/yescrypt && git rev-parse HEAD); \
	if [ "$$GOT" != "$(YESCRYPT_COMMIT)" ]; then \
		echo "ERROR: yescrypt HEAD $$GOT != expected $(YESCRYPT_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	cd $(DEPDIR)/yescrypt && \
	$(CC) -O3 -w -DSKIP_MEMZERO -c yescrypt-opt.c yescrypt-common.c sha256.c insecure_memzero.c; \
	mkdir -p $(TOPDIR)/yescrypt; \
	cp $(DEPDIR)/yescrypt/yescrypt-opt.o $(TOPDIR)/yescrypt/; \
	cp $(DEPDIR)/yescrypt/yescrypt-common.o $(TOPDIR)/yescrypt/; \
	cp $(DEPDIR)/yescrypt/sha256.o $(TOPDIR)/yescrypt/; \
	cp $(DEPDIR)/yescrypt/insecure_memzero.o $(TOPDIR)/yescrypt/; \
	cp $(DEPDIR)/yescrypt/yescrypt.h $(TOPDIR)/yescrypt/; \
	echo "  yescrypt objects installed"

# ---- PCRE ----
dep-pcre:
	@echo "==> PCRE 8.45 ($(PCRE_COMMIT))"
	@if [ -f $(TOPDIR)/libpcre.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone $(PCRE_REPO) $(DEPDIR)/pcre; \
	GOT=$$(cd $(DEPDIR)/pcre && git rev-parse HEAD); \
	if [ "$$GOT" != "$(PCRE_COMMIT)" ]; then \
		echo "ERROR: PCRE HEAD $$GOT != expected $(PCRE_COMMIT)"; exit 1; \
	fi; \
	echo "  verified $$GOT"; \
	cd $(DEPDIR)/pcre && \
	autoreconf -i && \
	./configure --enable-static --disable-shared --disable-cpp && \
	$(MAKE); \
	cp $(DEPDIR)/pcre/.libs/libpcre.a $(TOPDIR)/; \
	cp $(DEPDIR)/pcre/pcre.h $(TOPDIR)/; \
	echo "  libpcre.a installed"

# ---- bzip2 ----
dep-bzip2:
	@echo "==> bzip2 ($(BZIP2_TAG))"
	@if [ -f $(TOPDIR)/libbz2.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone --depth 1 --branch $(BZIP2_TAG) $(BZIP2_REPO) $(DEPDIR)/bzip2; \
	cd $(DEPDIR)/bzip2 && \
	$(MAKE) CC="$(CC)" libbz2.a; \
	cp $(DEPDIR)/bzip2/libbz2.a $(TOPDIR)/; \
	cp $(DEPDIR)/bzip2/bzlib.h $(TOPDIR)/; \
	echo "  libbz2.a installed"

# ---- xz/liblzma ----
dep-xz:
	@echo "==> xz/liblzma ($(XZ_TAG))"
	@if [ -f $(TOPDIR)/liblzma.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone --depth 1 --branch $(XZ_TAG) $(XZ_REPO) $(DEPDIR)/xz; \
	cd $(DEPDIR)/xz && \
	./autogen.sh && \
	./configure --enable-static --disable-shared --disable-xz --disable-xzdec \
		--disable-lzmadec --disable-lzmainfo --disable-scripts --disable-doc && \
	$(MAKE); \
	cp $(DEPDIR)/xz/src/liblzma/.libs/liblzma.a $(TOPDIR)/; \
	cp $(DEPDIR)/xz/src/liblzma/api/lzma.h $(TOPDIR)/; \
	cp -r $(DEPDIR)/xz/src/liblzma/api/lzma $(TOPDIR)/; \
	echo "  liblzma.a installed"

# ---- zstd ----
dep-zstd:
	@echo "==> zstd ($(ZSTD_TAG))"
	@if [ -f $(TOPDIR)/libzstd.a ]; then echo "  already built, skipping"; exit 0; fi; \
	set -e; \
	mkdir -p $(DEPDIR); \
	git clone --depth 1 --branch $(ZSTD_TAG) $(ZSTD_REPO) $(DEPDIR)/zstd; \
	cd $(DEPDIR)/zstd/lib && \
	$(MAKE) CC="$(CC)" libzstd.a; \
	cp $(DEPDIR)/zstd/lib/libzstd.a $(TOPDIR)/; \
	cp $(DEPDIR)/zstd/lib/zstd.h $(TOPDIR)/; \
	cp $(DEPDIR)/zstd/lib/zstd_errors.h $(TOPDIR)/; \
	echo "  libzstd.a installed"

.PHONY: all clean distclean deps \
        dep-openssl dep-sphlib dep-mhash dep-rhash dep-md6 \
        dep-streebog dep-bcrypt dep-judy dep-yescrypt dep-pcre \
        dep-bzip2 dep-xz dep-zstd
