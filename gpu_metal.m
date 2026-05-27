/*
 * $Revision: 1.53 $
 * $Log: gpu_metal.m,v $
 * Revision 1.52  2026/05/22 00:00:00  dlr
 * Phase 4 sub-phase 4a.2b Metal kernel-B production dispatcher built from scratch. Add gpu_metal_kernelb_dispatch_proto Metal twin of gpu_opencl_kernelb_dispatch_proto. Two-kernel pipeline JOB_MD5MD5SALT only. Kernel A1 cand_rules_phase0 producer encodes on one MTLCommandBuffer then read state slot byte overflow counters from shared-storage MTLBuffer contents update payload OCLParams num_words to actual_slots and packed_size to actual_byte_count lazy-allocate persistent buf_hashes_shown sized to cache_hash_data_count plus cache_overflow_count per feedback_proto_dispatch_lazy_alloc_dependency.md lazy-allocate persistent mtl_buf_kernelb_ovr_set and mtl_buf_kernelb_ovr_gid for the explicit atomic_uint args 16 17 the Metal emitter wires as standalone buffers OpenCL twin aliases off payload but Metal cannot cast through non atomic pointer. Encode kernel B 18 arg signature setBuffer atIndex 0 through 17 mirroring hx_emit_e347_md5md5md5salt_metal kernelb_hx_e347_phase0 signature in codegen hx_emit_metal.c rev 1.4 buffer 0 payload 1 packed 2 chunk_index 3 5 salt data off lens 6 7 compact_fp idx 8 9 hash_data data_off 10 11 hits hit_count 12 14 overflow keys hashes offsets 15 hashes_shown 16 17 ovr_set ovr_gid. tg_size 64 groups equal actual_slots times num_salt_chunks plus 63 over 64 where num_salt_chunks equal cached_salts_count plus 63 over 64. waitUntilCompleted MTLCommandBufferStatusCompleted gate FATAL on cb.error not nil. Read hcount from buf_hit_count contents copy hcount times GPU_HIT_STRIDE u32 entries from buf_hits contents into persistent h_hits return h_hits. MDXFIND_HX_CODEGEN equal 0 opt out FATAL on Metal there is no legacy hand written kernel B path the codegen IS the Metal kernel B implementation. Wire into gpu/gpujob_metal.m JOB_MD5MD5SALT branch after the existing kernel A1 A2 A3 A4 harness arms and before standard dispatch fallback. Stderr marker Metal GPU 0 kernel B PROTO dispatch fired op 347 n_words n n_rules n codegen kernelb_hx_e347_phase0 fires once per process per device. Declare in gpu_metal.h. New persistent static state mtl_buf_kernelb_ovr_set mtl_buf_kernelb_ovr_gid mtl_buf_kernelb_hashes_shown_cap zeroed at shutdown. dev1 build clean iMac authoritative ci -l. 3-cell production fixture validation on dev3.local Apple M2 Max byte-exact PASS smoke medium large. Real-world ./mdxfind -m e347 -G 0 on dev3 finds password digest d64a9d6cec245455d21f989ebdab0dd9 matching CPU oracle. Opt-out FATAL message documented. OpenCL non-regression on fpga.local Pascal GTX 1080. Kernel A1 plus e1 non-regression on dev3 unchanged.
 *
 * Revision 1.51  2026/05/21 00:00:00  dlr
 * Phase 4 sub-phase 4a.2 Metal twin of OpenCL 4a.1 production-codegen swap. Add gpu_metal_hx_codegen_enabled accessor mirroring OpenCL gpu_opencl_hx_codegen_enabled semantic (env literal "0" forces opt-out; absent or any other value selects codegen path). Add mtl_lib_kernelb_codegen and mtl_pso_kernelb_codegen module-local statics parallel to existing kernel-A1/A2/A3/A4 lib+PSO families. Add gpu_metal_kernelb_codegen_pso lazy-init helper that on first call per process runs hx_specs_lookup(JOB_MD5MD5SALT), builds an hx_specialization for e347 (iter_count=1, has_rules=1, SALT_BATCH=64, salt_minlen=1, salt_maxlen=64, iter_shape=NONE, digest_endianness=LE, emit_width=16), invokes hx_emit_kernel with HX_BACKEND_METAL, dumps the emitted Metal source to /tmp/mdxfind-codegen-e347-pid-devN.metal for post-mortem inspection, JIT-compiles via gpu_metal_jit_compile_source_with_common_keep, caches the MTLLibrary + PSO under bridge-retained pointers, and returns the PSO as opaque void * for the eventual kernel-B dispatcher swap. All failure paths FATAL per feedback_external_failures_are_fatal.md (hx spec lookup, walker rc, JIT compile, PSO creation). Include codegen headers under METAL_GPU guard mirroring the OpenCL #include set in gpu_opencl.c rev 1.190. NOTE Metal kernel-B production dispatcher does not exist yet (prior Phase 1a sub-phases shipped kernel A1-A4 only; no Metal kernel-B wrapper was wired). The 4a.2 ship lays the codegen scaffolding so the future Metal kernel-B dispatcher (separate follow-on sub-phase) can perform the OpenCL-twin-style swap directly.
 *
 * Revision 1.50  2026/05/22 04:30:00  dlr
 * sub-phase 2a.6 add three Metal entry points for the e347 byte-exact validation harness; gpu_metal_jit_compile_source_with_common_keep mirrors the 2a.4 JIT helper but creates an MTLComputePipelineState for the requested entry point and bridge-retains the MTLLibrary and PSO as opaque void pointers so mdxfind.c can hold them across dispatch; gpu_metal_e347_validate_dispatch allocates short-lived MTLBuffers for the per-invocation packed candidates and word offsets plus OCLParams payload binds them along with device-resident salt and compact and hash_data and overflow buffers under setBuffer atIndex 0 through 17 mirroring the Metal emitter signature dispatches via MTLCommandBuffer with a tg picked from maxTotalThreadsPerThreadgroup capped at 128 waits for completion via waitUntilCompleted and reads hits back; gpu_metal_jit_release_keep tears down the bridge retained handles via CFBridgingRelease idempotent on NULL; all three FATAL on any error per external-failures-are-fatal; Metal twin of OpenCL gpu_opencl_jit_compile_source_with_common_keep and gpu_opencl_e347_validate_dispatch and gpu_opencl_jit_release_keep in gpu_opencl.c rev 1.189 closes Phase 3 cross-arch byte-exact validation
 *
 * Revision 1.49  2026/05/12 18:00:00  dlr
 * Phase 2a row 6: rules-engine kernel-side wiring. New statics mtl_lib_template_md5_rules + mtl_pso_template_md5_rules; new internal helper metal_load_library_rules() builds a SEPARATE MTLLibrary from concat(metal_common_str + metal_md5_core_str + metal_md5_rules_str + metal_template_str) with MTLCompileOptions.preprocessorMacros set to GPU_TEMPLATE_HAS_RULES=1. gpu_metal_template_pso_lazy_md5_rules() public lazy creator mirrors metal_pso_template_lazy_md5(). gpu_metal_dispatch_md5_rules() extended to consume the gpu_rule_program / gpu_rule_offsets / gpu_rule_count externs (lazy MTLBuffer upload to buf_rule_program + buf_rule_offset on first rules dispatch; reused thereafter unless gpu_rule_program changes underfoot — Phase 2a holds them constant per session). When gpu_rule_count > 0 the rules PSO is selected, num_masks is set to gpu_rule_count for the kernel's n_rules read, and buffers 10 + 11 bind rule_program / rule_offset. When gpu_rule_count == 0 the kernel ships with Phase 1 no-rules PSO unchanged. Embedded metallib still no-rules-only; rules variant always JIT (single concession to keep build_metallib.sh simple — Apple driver self-caches the JIT). gpu_metal_shutdown extended to release the new statics + buffers. Per memo §3 rows 4-7.
 *
 * Revision 1.47  2026/05/12 13:55:13  dlr
 * Phase 1 gpujob stubs: minimal no-op definitions of gpujob_init, gpujob_available, gpujob_shutdown, gpujob_batch_max, gpujob_queue_depth, gpujob_free_count, gpujob_print_share_line, and gpu_op_category at end of file so the local Metal build links. gpu/gpujob_opencl.c emits no symbols on Metal builds (its outer #if is OPENCL_GPU-only). Memo §1 non-goal stands ("no gpujob_metal.m re-add"); these stubs satisfy compile+link only and will be removed when the real gpujob_metal.m lands. Behavior: gpujob_available returns 1 when mtl_device!=nil, gpu_op_category returns GPU_CAT_UNSALTED for JOB_MD5 (mirrors gpu/gpujob_opencl.c:~3287) and GPU_CAT_NONE for everything else so unsupported ops fall to CPU. Wrapped in [Phase 1 gpujob stubs] markers for grep-out. Isolated compile passes (clang -fobjc-arc -DMACOSX=1 -DMETAL_GPU=1).
 *
 * Revision 1.46  2026/05/12 13:43:49  dlr
 * Phase 1 Metal port host implementation fresh start (replaces retired 1.45). Single-device Objective-C++ host mirroring gpu/gpu_opencl.c structure for nine entry points: init/shutdown/available/set_compact_table/set_overflow/set_op/set_max_iter/compile_families/dispatch_md5_rules. Two-tier compile: embedded gpu_mdxfind_metallib[] default via newLibraryWithData + dispatch_data_create, JIT fallback when MDXFIND_METAL_JIT=1 via newLibraryWithSource over concat metal_common_str+metal_md5_core_str+metal_template_str. Lazy PSO (mtl_pso_template_md5) on first dispatch. SSH-context MTLCreateSystemDefaultDevice -> MTLCopyAllDevices[0] fallback preserved from retired source. OCLParams typedef redeclared (host-only; gpu/gpu_opencl.c keeps it static) with six _Static_asserts on sizeof+offsetof to gate wire-format parity (sizeof=128, algo_mode=120, num_words=32, max_iter=60, overflow_first_word=104, overflow_first_set=100). MetalParams parity asserted on GPU side in gpu/metal_common.metal. Defensive forward-decls of GPU_HIT_STRIDE/FAM_MD5UNSALTED/GPU_MAX_HITS allow isolated compile pre-Makefile-edit; gpujob.h gate edit makes them unreachable post-edit. Phase 1 JOB_MD5 only; other ops return NULL+nhits=0. Isolated compile passes on iMac AMD Radeon Pro 580X (Intel) + dev1.local Apple M-series.
 *
 */
/* gpu_metal.m — Objective-C++ host for the Metal GPU backend (Phase 1
 * fresh start; mirrors gpu/gpu_opencl.c structurally for the same nine
 * entry points listed in gpu_metal.h).
 *
 * Phase 1 scope (memo §1 + §11 row 12):
 *   - Single device only.
 *   - One library + one PSO (template_phase0 for JOB_MD5 unsalted).
 *   - metallib-first compile: load embedded gpu_mdxfind_metallib[] bytes
 *     via -[MTLDevice newLibraryWithData:error:].
 *   - JIT fallback when getenv("MDXFIND_METAL_JIT") == "1": concat
 *     metal_common_str + metal_md5_core_str + metal_template_str and
 *     compile via -[MTLDevice newLibraryWithSource:options:error:].
 *   - Lazy PSO creation on first dispatch (mirrors
 *     gpu_opencl_template_kernel_lazy_md5).
 *   - SSH-context fallback: MTLCreateSystemDefaultDevice() returns nil
 *     under sshd on macOS — fall back to MTLCopyAllDevices()[0]. The
 *     retired gpu_metal.m carried this pattern; preserved here.
 *
 * NOT to be compiled unless METAL_GPU is defined (the #if guard at the
 * top of the body keeps a stray .m file in the build inert).
 *
 * --- OCLParams reuse rationale ---
 *
 * The OCLParams typedef is defined in gpu/gpu_opencl.c as a static-file
 * struct (not exposed in gpu_opencl.h). To avoid widening the OpenCL
 * public ABI, this file redeclares the same 128-byte layout under the
 * same name — guaranteed byte-identical by static_assert on sizeof +
 * offsetof of every field the kernel reads. The Metal kernel's
 * MetalParams struct (gpu/metal_common.metal) also asserts the same
 * layout from the GPU side. Three independent assertions enforce wire-
 * format parity; any drift trips a compile error in at least one TU.
 *
 * --- Buffer storage choice ---
 *
 * Phase 1 uses MTLResourceStorageModeShared on all buffers. Apple Silicon
 * (M1/M2/M3) has unified memory so this is a zero-copy hand-off; Intel
 * macs (Radeon Pro 580X) pay one DMA per write but the Phase 1 dispatch
 * is small enough that this is acceptable. Phase 2 may switch large
 * read-only buffers to Managed mode for Intel discrete GPUs.
 */

#if defined(METAL_GPU)

/* Task 309 safety net (2026-05-17): Intel Mac Metal disabled — Apple
 * MTLCompilerService XPC daemon hangs on JIT PSO creation for AMD GCN
 * GPUs (e.g., Radeon Pro 580X) on macOS Sequoia 15.x. Confirmed at rev
 * 1.475 on local iMac AND nutshack. The bug is in Apple's driver and
 * not fixable from mdxfind. Compile-time guard prevents accidental
 * METAL_GPU re-enable on Intel architecture; edit the Makefile to
 * exclude this TU on Intel Mac, or build on Apple Silicon (M-series).
 */
#if defined(__APPLE__) && !defined(__aarch64__)
#error "gpu_metal.m: Intel Mac Metal disabled — Apple MTLCompilerService XPC hang on AMD GCN + macOS Sequoia (confirmed 2026-05-17 rev 1.475 on iMac + nutshack). Edit Makefile to exclude this TU on Intel Mac, or build on Apple Silicon."
#endif

#include "gpu_metal.h"

/* Phase D5a 2026-05-16 (Task #281): host-side fatal-error macros for
 * runtime GPU failures (PSO/library create, dispatch, buffer alloc).
 * Provides GPU_FATAL (no NSError) and MTL_FATAL_NSERR (unwraps NSError).
 * See feedback_external_failures_are_fatal.md. */
#include "gpu/gpu_fatal.h"

/* User directive 2026-05-17: debug stderr emissions in GPU paths MUST NOT
 * live in the shipped binary. GPU_DEBUG_FPRINTF wraps init/dispatch
 * chatter (library JIT markers, lazy PSO markers, "first dispatch issued",
 * "salts uploaded", etc.). Default builds elide both the call and the
 * format-string literal. Debug builds re-enable via
 *   make CFLAGS_EXTRA="-DMDXFIND_GPU_DEBUG=1"
 * PRODUCTION fprintf calls (device identity, capability-gap summary,
 * end-of-job stats, GPU_FATAL/MTL_FATAL_NSERR) remain UNCONDITIONAL.
 * See gpu/gpu_debug.h for the full classification policy. */
#include "gpu/gpu_debug.h"

/* gpujob.h is gated on CUDA_GPU || OPENCL_GPU; the main-session edit
 * (memo §9 + §10) adds `|| defined(METAL_GPU)` to that gate, after
 * which this include exposes FAM_MD5UNSALTED, GPU_HIT_STRIDE, and the
 * other gpu_* macros. Until that edit lands this TU still compiles in
 * isolation via the defensive forward-declarations below. */
#include "gpujob.h"
#include "job_types.h"     /* JOB_MD5 */

/* Defensive forward-declarations for the symbols gpujob.h gates on
 * CUDA_GPU || OPENCL_GPU. The main-session gate edit makes these
 * declarations unreachable (the gpujob.h ones win); keeping them
 * lets gpu_metal.m compile cleanly in isolation. */
#ifndef GPU_HIT_STRIDE
#define GPU_HIT_STRIDE   19
#endif
#ifndef FAM_MD5UNSALTED
/* Same enum slot index as in gpujob.h. If gpujob.h's enum gets
 * extended/reordered, the gate edit must include METAL_GPU before
 * the enum so this fallback never wins. */
#define FAM_MD5UNSALTED  10
#endif

/* GPU_MAX_HITS lives in gpu/gpu_opencl.c as a private constant (line 1310).
 * It's not exposed in any public header; replicate the value here. If the
 * OpenCL definition changes, this must change too (one-line audit
 * surfaces it via `grep GPU_MAX_HITS`). */
#ifndef GPU_MAX_HITS
#define GPU_MAX_HITS  32768
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <dispatch/dispatch.h>

/* Embedded metallib bytes (default compile path). Generated by
 * build_metallib.sh via xxd -i. */
#include "gpu/mdxfind_metallib.h"

/* JIT source strings (fallback when MDXFIND_METAL_JIT=1, and always for
 * the Phase 2a rules-variant library). Each header declares a
 * `static const char <name>_str[]` so we get TU-local copies here; the
 * JIT path concatenates them in source order. */
#include "gpu/metal_common_str.h"
#include "gpu/metal_md5_core_str.h"
#include "gpu/metal_md5salt_core_str.h"  /* Phase 2c salt-variant core */
#include "gpu/metal_md4_core_str.h"      /* Phase 2d.2.1b md4 core */
#include "gpu/metal_md4utf16_core_str.h" /* Phase 2d.2.2 md4utf16 core */
#include "gpu/metal_md5raw_core_str.h"   /* Phase 2d.2.3 md5raw core */
#include "gpu/metal_md5passsalt_core_str.h" /* Phase 2d.2.4 md5passsalt core (4 PSO variants) */
#include "gpu/metal_md5saltpass_core_str.h" /* Phase 2d.2.5 md5saltpass core (4 PSO variants) */
#include "gpu/metal_sha1_core_str.h"     /* Phase 2d.3.1 sha1 core (FIRST SHA-family) */
#include "gpu/metal_sha1raw_core_str.h"  /* Phase 2d.3.2 sha1raw core (binary-digest re-feed) */
#include "gpu/metal_sha1dru_core_str.h"  /* Phase 2d.3.3 sha1dru core (Drupal SHA-1, 1M-iter) */
#include "gpu/metal_sha1passsalt_core_str.h" /* Phase 2d.3.4 sha1passsalt core (SHA-1(pass||salt)) */
#include "gpu/metal_sha1saltpass_core_str.h" /* Phase 2d.3.5 sha1saltpass core (SHA-1(salt||pass)) */
#include "gpu/metal_sha256_core_str.h"   /* Phase 2d.4.1 sha256 core (FIRST SHA-2/256 family) */
#include "gpu/metal_sha256raw_core_str.h" /* Phase 2d.4.2 sha256raw core (binary-digest re-feed) */
#include "gpu/metal_sha224_core_str.h"   /* Phase 2d.4.3 sha224 core (7-word digest, reuses sha256_block) */
#include "gpu/metal_sha256passsalt_core_str.h" /* Phase 2d.4.4 sha256passsalt core (SHA-256(pass||salt)) */
#include "gpu/metal_sha256saltpass_core_str.h" /* Phase 2d.4.5 sha256saltpass core (SHA-256(salt||pass)) */
#include "gpu/metal_sha224saltpass_core_str.h" /* Phase 2d.4.6 sha224saltpass core (SHA-224(salt||pass), 7-word) */
#include "gpu/metal_sha512_core_str.h"   /* Phase 2d.5.1 sha512 core (FIRST 64-bit-state family on Metal) */
#include "gpu/metal_sha512raw_core_str.h" /* Phase 2d.5.2 sha512raw core (binary-digest re-feed, 16 uint32) */
#include "gpu/metal_sha384_core_str.h"   /* Phase 2d.5.3 sha384 core (truncated SHA-512, 12 uint32 emit) */
#include "gpu/metal_sha384raw_core_str.h" /* Phase 2d.5.4 sha384raw core (binary 48-byte digest re-feed, 12 uint32) */
#include "gpu/metal_sha512passsalt_core_str.h" /* Phase 2d.5.5 sha512passsalt core (FIRST salted SHA-512: SHA-512(pass||salt)) */
#include "gpu/metal_sha512saltpass_core_str.h" /* Phase 2d.5.6 sha512saltpass core (PREPEND-shape SHA-512(salt||pass)) */
#include "gpu/metal_sha384saltpass_core_str.h" /* Phase 2d.5.7 sha384saltpass core (PREPEND-shape SHA-384(salt||pass), 12 uint32) */
#include "gpu/metal_ripemd160_core_str.h" /* Phase 2d.6.1 ripemd160 core (FIRST RIPEMD-family Metal port, 5 uint32 LE digest) */
#include "gpu/metal_ripemd320_core_str.h" /* Phase 2d.6.2 ripemd320 core (RIPEMD-320, 10 uint32 LE digest, dual-pipeline no-merge) */
#include "gpu/metal_blake2s256_core_str.h" /* Phase 2d.7a.1 blake2s256 core (FIRST BLAKE2-family Metal port, 8 uint32 LE digest, b2s_compress) */
#include "gpu/metal_blake2b256_core_str.h" /* Phase 2d.7a.2 blake2b256 core (BLAKE2B truncated to 32 bytes = 8 uint32 LE, b2b_compress 64-bit lanes) */
#include "gpu/metal_blake2b512_core_str.h" /* Phase 2d.7a.3 blake2b512 core (full BLAKE2B 64 bytes = 16 uint32 LE, b2b_compress shared with b256) */
#include "gpu/metal_keccak256_core_str.h"  /* Phase 2d.7b.1 keccak256 core (FIRST Keccak/SHA-3 sponge Metal port, 8 uint32 LE digest, keccakf1600 shared across 8 ops, rate=136, pad=0x01) */
#include "gpu/metal_keccak224_core_str.h"  /* Phase 2d.7b.2 keccak224 core (7 uint32 LE digest = 28 bytes, rate=144, pad=0x01) */
#include "gpu/metal_keccak384_core_str.h"  /* Phase 2d.7b.3 keccak384 core (12 uint32 LE digest = 48 bytes, rate=104, pad=0x01) */
#include "gpu/metal_keccak512_core_str.h"  /* Phase 2d.7b.4 keccak512 core (16 uint32 LE digest = 64 bytes, rate=72, pad=0x01) */
#include "gpu/metal_sha3_224_core_str.h"   /* Phase 2d.7b.5 sha3_224 core (7 uint32 LE digest, rate=144, pad=0x06) */
#include "gpu/metal_sha3_256_core_str.h"   /* Phase 2d.7b.6 sha3_256 core (8 uint32 LE digest, rate=136, pad=0x06) */
#include "gpu/metal_sha3_384_core_str.h"   /* Phase 2d.7b.7 sha3_384 core (12 uint32 LE digest, rate=104, pad=0x06) */
#include "gpu/metal_sha3_512_core_str.h"   /* Phase 2d.7b.8 sha3_512 core (16 uint32 LE digest, rate=72, pad=0x06) */
#include "gpu/metal_streebog256_core_str.h" /* Phase 2d.7c.1 streebog256 core (FIRST Streebog family Metal port (CANARY), translator rev 1.8 in-body uchar* cast handling, 8 uint32 LE digest, NO metal_common deps) */
#include "gpu/metal_streebog512_core_str.h" /* Phase 2d.7c.2 streebog512 core (16 uint32 LE digest, shares streebog_g + streebog_hash_priv with streebog256 sibling) */
#include "gpu/metal_hmac_blake2s_core_str.h" /* Phase 2d.7d.1 HMAC-BLAKE2S carrier (Family I; SALTED-ONLY; single algo_mode=5; 8 uint32 LE digest; b2s_compress shared with blake2s256 via metal_common) */
#include "gpu/metal_hmac_streebog256_core_str.h" /* Phase 2d.7d.2 HMAC-STREEBOG-256 carrier (Family J; SALTED-ONLY; algo_mode 5=KSALT/6=KPASS; 8 uint32 LE digest; Typesalt for BOTH ops) */
#include "gpu/metal_hmac_streebog512_core_str.h" /* Phase 2d.7d.3 HMAC-STREEBOG-512 carrier (Family K; SALTED-ONLY; algo_mode 5=KSALT/6=KPASS; 16 uint32 LE digest; FINAL HMAC family) */
#include "gpu/metal_phpbb3_core_str.h" /* Phase 2d.8a.1 PHPBB3 carrier (op=455; SALTED-ONLY; single algo_mode=0; 4 uint32 LE digest = MD5 width; iterated MD5 chain INSIDE template_finalize; count decoded from salt[3] via phpitoa64) */
#include "gpu/metal_md5crypt_core_str.h" /* Phase 2d.8a.2 MD5CRYPT carrier (op=511; SALTED-ONLY; single algo_mode=0; 4 uint32 LE digest = MD5 width; BSD $1$ 3-step + 1000-iter MD5 chain INSIDE template_finalize; Phase 1 of Unix-crypt ladder on Metal) */
#include "gpu/metal_shacrypt_core_str.h" /* Phase 2d.8b SHACRYPT shared core (SHA256CRYPT op=512 HASH_WORDS=8 + SHA512CRYPT op=513 HASH_WORDS=16 + SHA512CRYPTMD5 op=538 HASH_WORDS=16 algo_mode=1 MD5-preprocess; SAME .metal instantiated TWICE via preprocessorMacros HASH_WORDS={8,16} + HASH_BLOCK_BYTES={64,128}; Phases 2/3/4 of Unix-crypt ladder on Metal; SHA512CRYPTMD5 aliases SHA512CRYPT compiled program) */
#include "gpu/metal_descrypt_core_str.h" /* Phase 2d.9a DESCRYPT carrier (op=500; SALTED-ONLY; single algo_mode=7; 4 uint32 LE state = h[0..1] pre-FP (l,r) + h[2..3] zero-pad; 25-iter DES Feistel INSIDE template_finalize; HAND-PORT of gpu/gpu_descrypt_core.cl rev 1.1; cl2metal.py UNSUITABLE per architect Task #293 Option A; last Unix-crypt op to migrate to Metal) */
#include "gpu/metal_bcrypt_core_str.h" /* Phase 2d.9b BCRYPT carrier (op=450; SALTED-ONLY; single algo_mode=8; HASH_WORDS=6 (first 6-word Metal family); 2^cost Eksblowfish iter INSIDE template_finalize; HAND-PORT of gpu/gpu_bcrypt_core.cl rev 1.1; cl2metal.py UNSUITABLE per architect Task #293; uses NEW threadgroup-shared sbox_pool (32 KB per WG = exactly Apple Silicon maxThreadgroupMemoryLength); requires NEW GPU_TEMPLATE_HAS_LOCAL_BUFFER scaffold extension + per-op threadsPerThreadgroup=8 dispatch-site override; FINAL Phase 2d sub-phase; 51 -> 52 families) */
#include "gpu/metal_md5_rules_str.h"
#include "gpu/metal_template_str.h"
#include "gpu/metal_kernel_a_rules_str.h" /* Phase 1a sub-phase 1a.1b-continued (2026-05-20): translator-driven kernel A1 source for Metal dispatcher */
#include "gpu/metal_kernel_a_masks_str.h" /* Phase 1a sub-phase 1a.2 (2026-05-21): translator-driven kernel A2 (masks-only) source for Metal dispatcher */
#include "gpu/metal_kernel_a_rules_masks_str.h" /* Phase 1a sub-phase 1a.3 (2026-05-21): hand-port kernel A3 (rules+masks) source for Metal dispatcher */
#include "gpu/metal_kernel_a_bruteforce_str.h" /* Phase 1a sub-phase 1a.4 (2026-05-21): hand-port kernel A4 (brute-force) source for Metal dispatcher */

/* Phase 4 sub-phase 4a.2 (2026-05-21): hx codegen integration headers
 * for the Metal-side production codegen path. Mirrors the OpenCL twin
 * inclusion set in gpu/gpu_opencl.c rev 1.190. The headers declare the
 * hx_specs_lookup table accessor, the hx_specialization struct, and the
 * hx_emit_kernel walker entry point. None of them are Metal-specific;
 * the same headers are consumed by the OpenCL TU and the mdxfind.c
 * harness blocks (both OpenCL and Metal branches per mdxfind.c rev
 * 1.494 widen). */
#include "codegen/hx_spec.h"
#include "codegen/hx_walker.h"
#include "codegen/hx_spec_entry.h"
/* Sub-phase 5a.5 (2026-05-22): family admit-predicate helper used by
 * gpu_metal_kernelb_dispatch_proto early gate to accept the seven
 * 5a-eligible MAKE_MD5PASS family JOB enums (e122/e159/e161/e163/
 * e165/e167/e169) alongside JOB_MD5MD5SALT. */
#include "gpu/gpu_codegen_eligible.h"

/* ---- Wire-format invariant: redeclare OCLParams (host-only) ---- */

/* OCLParams typedef from gpu/gpu_opencl.c line ~1334. Kept byte-identical;
 * static_asserts below enforce the layout. */
typedef struct {
    uint64_t compact_mask;        /*   0 */
    uint64_t mask_start;          /*   8 */
    uint64_t mask_base0;          /*  16 */
    uint64_t mask_base1;          /*  24 */
    uint32_t num_words;           /*  32 */
    uint32_t num_salts;           /*  36 */
    uint32_t salt_start;          /*  40 */
    uint32_t max_probe;           /*  44 */
    uint32_t hash_data_count;     /*  48 */
    uint32_t max_hits;            /*  52 */
    uint32_t overflow_count;      /*  56 */
    uint32_t max_iter;            /*  60 */
    uint32_t num_masks;           /*  64 */
    uint32_t n_prepend;           /*  68 */
    uint32_t n_append;            /*  72 */
    uint32_t iter_count;          /*  76 */
    /* base_word_idx (80) + packed_size (84): claimed from reserved32[2] by
     * the Phase 1a two-kernel pipeline (mirror of gpu/gpu_opencl.c:1421).
     * base_word_idx = kernel A source word offset (hit-attribution).
     * packed_size   = kernel A output buffer capacity in bytes. Both are
     * unused by the standard rules dispatcher (zero-filled there) and
     * unused by kernel B; only kernel A1 reads them via OCLParams typedef
     * bridge in metal_common.metal. Per feedback_rename_reserved_slots.md:
     * rename in same commit as first real use. */
    uint32_t base_word_idx;       /*  80 */
    uint32_t packed_size;         /*  84 */
    uint32_t input_cursor_start;  /*  88 */
    uint32_t rule_cursor_start;   /*  92 */
    uint32_t inner_iter;          /*  96 */
    uint32_t overflow_first_set;  /* 100 */
    uint32_t overflow_first_word; /* 104 */
    uint32_t num_rules;           /* 108: source rule count for kernel A1/A3
                                   *      (sub-phase 1a.2 rename, was
                                   *      overflow_first_rule, unused in B3) */
    uint64_t num_salts_per_page;  /* 112 */
    uint32_t algo_mode;           /* 120 */
    uint32_t mask_offset_per_word;/* 124 */
} OCLParams;

/* OCLParams wire-format gate. Mirrors the same static_asserts that
 * gpu_common.cl / metal_common.metal use on the GPU side. If ANY of
 * these trip at compile time the GPU kernel will misread the payload —
 * fix the cause, do not paper over the assertion. */
_Static_assert(sizeof(OCLParams) == 128,
               "OCLParams MUST be 128 bytes (host/GPU wire-format parity)");
_Static_assert(offsetof(OCLParams, algo_mode) == 120,
               "OCLParams.algo_mode offset MUST be 120 (B6.6 wire format)");
_Static_assert(offsetof(OCLParams, num_words) == 32,
               "OCLParams.num_words offset MUST be 32");
_Static_assert(offsetof(OCLParams, max_iter) == 60,
               "OCLParams.max_iter offset MUST be 60");
_Static_assert(offsetof(OCLParams, overflow_first_word) == 104,
               "OCLParams.overflow_first_word offset MUST be 104");
_Static_assert(offsetof(OCLParams, overflow_first_set) == 100,
               "OCLParams.overflow_first_set offset MUST be 100");

/* ---- Single-device static state (Phase 1) ----
 *
 * Memo §3: ONE device, ONE library, ONE PSO. No metal_kernel_map[], no
 * mtl_pipeline_packed[FAM_COUNT] array (the architecture deliberately
 * drops them — Phase 2+ each algo gets its own mtl_lib_template_<algo>
 * + mtl_pso_template_<algo> pair as separate statics, not array slots).
 */
static id<MTLDevice>               mtl_device              = nil;
static id<MTLCommandQueue>         mtl_queue               = nil;
static id<MTLLibrary>              mtl_lib_template_md5    = nil;
static id<MTLComputePipelineState> mtl_pso_template_md5    = nil;


/* Phase 2c row 1: salt-variant library + PSO statics. Four new variants
 * mirror Phase 2b R/M/RM exactly with an extra GPU_TEMPLATE_HAS_SALT
 * macro defined at JIT compile time. Each library concatenates
 * metal_md5salt_core_str INSTEAD OF metal_md5_core_str (the two define
 * the same fn names; mutually exclusive within a single MTLLibrary).
 * Always JIT-compiled at first use; embedded metallib is unsalted-only
 * (mirrors the rules-variant choice). */
static id<MTLLibrary>              mtl_lib_template_md5_salt            = nil;
static id<MTLComputePipelineState> mtl_pso_template_md5_salt            = nil;
static id<MTLLibrary>              mtl_lib_template_md5_salt_rules      = nil;
static id<MTLComputePipelineState> mtl_pso_template_md5_salt_rules      = nil;
static id<MTLLibrary>              mtl_lib_template_md5_salt_mask       = nil;
static id<MTLComputePipelineState> mtl_pso_template_md5_salt_mask       = nil;
static id<MTLLibrary>              mtl_lib_template_md5_salt_rules_mask = nil;
static id<MTLComputePipelineState> mtl_pso_template_md5_salt_rules_mask = nil;

/* Phase 2e: pre-salt hoist + SIMD lane-batching variant. ONE new PSO
 * (salt+rules+presalt; the only salt combo that fires on the e31
 * sm-saltfull bench). Other salt combos (salt-only / salt+mask /
 * salt+rules+mask) fall through to the non-presalt PSOs; Phase 2e.2
 * expands coverage. SALT_BATCH is per-tier (8/16/32) -- baked into
 * the JIT macros at compile time, so a different tier forces a fresh
 * library + PSO. */
static id<MTLLibrary>              mtl_lib_template_md5_salt_rules_presalt = nil;
static id<MTLComputePipelineState> mtl_pso_template_md5_salt_rules_presalt = nil;

/* Phase 1a sub-phase 1a.1b-continued (2026-05-20): kernel A1 (rules-only)
 * standalone library + PSO statics. Entry-point name `cand_rules_phase0`
 * (matches OpenCL twin gpu/gpu_kernel_a_rules.cl rev 1.4). Always JIT-
 * compiled at first use; embedded metallib excludes the A1 kernel (the
 * rules-variant precedent at line 5 documents: rules variant always JIT).
 *
 * Buffer trio is the Metal mirror of OpenCL gpu_device fields b_proto_-
 * packed_buf / b_proto_chunk_index / b_proto_kernelA_state. _proto_*_cap
 * tracks current MTLBuffer capacity (bytes) so grow-on-demand mirrors
 * gpu_opencl_kernelb_ensure_proto_bufs. host scratch h_proto_*_readback
 * mirrors the OpenCL post-dispatch host-side buffers used by Phase 4
 * hit-replay; the Metal A1-only ship does NOT need host-side hit-replay
 * (kernel B not wired) but the trace-dump path reads them for the harness. */
static id<MTLLibrary>              mtl_lib_kernel_a_rules               = nil;
static id<MTLComputePipelineState> mtl_pso_kernel_a_rules               = nil;
static id<MTLBuffer>               mtl_buf_proto_packed                 = nil;
static id<MTLBuffer>               mtl_buf_proto_chunk_index            = nil;
static id<MTLBuffer>               mtl_buf_proto_state                  = nil;
static size_t                      mtl_buf_proto_packed_cap             = 0;
static size_t                      mtl_buf_proto_index_cap              = 0;
static void                       *h_proto_packed_readback              = NULL;
static size_t                      h_proto_packed_readback_cap          = 0;
static void                       *h_proto_index_readback               = NULL;
static size_t                      h_proto_index_readback_cap           = 0;

/* Phase 1a sub-phase 1a.2 (2026-05-21): kernel A2 (masks-only) standalone
 * library + PSO statics. Mirrors the A1 pattern above (rules-variant).
 * Entry-point name `cand_masks_phase0` (matches OpenCL twin gpu/
 * gpu_kernel_a_masks.cl rev 1.3). Always JIT-compiled at first use.
 *
 * Mask wire-format buffers (4): 32 + 32 + 4096 + 64 = 4224 bytes total.
 * Allocated lazily on the first A2 dispatch with non-zero mask config.
 * Storage mode Shared mirrors the A1 buffer trio convention; sub-5 KB
 * total — UMA/discrete perf difference is irrelevant at this size. */
static id<MTLLibrary>              mtl_lib_kernel_a_masks               = nil;
static id<MTLComputePipelineState> mtl_pso_kernel_a_masks               = nil;
static id<MTLBuffer>               mtl_buf_kern_a_mask_prepend          = nil;
static id<MTLBuffer>               mtl_buf_kern_a_mask_append           = nil;
static id<MTLBuffer>               mtl_buf_kern_a_mask_charsets         = nil;
static id<MTLBuffer>               mtl_buf_kern_a_mask_counts           = nil;
static int                         mtl_buf_kern_a_mask_allocated        = 0;

/* Phase 1a sub-phase 1a.3 (2026-05-21): kernel A3 (rules + masks)
 * standalone library + PSO statics. Compositional product of A1 + A2;
 * reuses buf_rule_program + buf_rule_offset (A1) and the four
 * mtl_buf_kern_a_mask_* buffers (A2). Always JIT-compiled at first use;
 * 10-arg signature cand_rules_masks_phase0. */
static id<MTLLibrary>              mtl_lib_kernel_a_rules_masks         = nil;
static id<MTLComputePipelineState> mtl_pso_kernel_a_rules_masks         = nil;

/* Phase 1a sub-phase 1a.4 (2026-05-21): kernel A4 (brute-force) standalone
 * library + PSO statics. Pure BF; no rule axis. Reuses the four
 * mtl_buf_kern_a_mask_* buffers (A2) -- prepend buffer is zeroed by the
 * upload helper since BF invariant is MaskPrependLen == 0. Always JIT-
 * compiled at first use; 8-arg signature cand_bruteforce_phase0
 * (byte-identical to A2). */
static id<MTLLibrary>              mtl_lib_kernel_a_bruteforce          = nil;
static id<MTLComputePipelineState> mtl_pso_kernel_a_bruteforce          = nil;

/* Phase 4 sub-phase 4a.2 (2026-05-21): hx codegen kernel B production
 * library + PSO statics. Lazy-populated on first call to
 * gpu_metal_kernelb_codegen_pso. Storage type matches the validate
 * dispatcher's bridge-retain pattern from sub-phase 2a.6 (CFBridging
 * Retain returns void * that survives out-of-ARC scope; cast back to
 * id<MTLComputePipelineState> at use site). Process-lifetime; no
 * explicit release entry point (process exit reclaims).
 *
 * NOTE the kernel-B dispatcher wrapper that consumes this PSO is not
 * yet wired on Metal -- prior sub-phases shipped kernel A1-A4 only.
 * This static + the lazy-init helper below stand ready for the future
 * Metal kernel-B dispatcher (follow-on sub-phase) to consume in the
 * same kernel-selection shape as the OpenCL 4a.1 swap inside
 * gpu_opencl_kernelb_dispatch_proto. */
static void                       *mtl_lib_kernelb_codegen              = NULL;
static void                       *mtl_pso_kernelb_codegen              = NULL;

/* Sub-phase 5a.3 (2026-05-22): per-JOB slot map for the codegen kernel B
 * PSO cache, mirroring the OpenCL twin's gpu_device.hx_codegen_slots[16]
 * (gpu/gpu_opencl.c rev 1.192). Slot 0 is reserved for back-compat with
 * the 4a.2 statics above (single-PSO API gpu_metal_kernelb_codegen_pso
 * still works); family members occupy slots 1+. Bumped to 64 in 5b when
 * the remaining 22 deferred family members ship.
 *
 * Storage is CFBridgingRetain'd void* (parallel to the 4a.2 single-PSO
 * statics) so the slots can live in a C-friendly struct without ARC
 * lifetime concerns; CFBridgingRelease at process exit (or never -- the
 * cache is process-lifetime as documented in
 * gpu_metal_kernelb_codegen_pso). */
typedef struct mtl_codegen_slot_s {
    int    job_enum;     /* 0 = empty; otherwise JOB_xxx enum */
    void  *lib_void;     /* CFBridgingRetain'd MTLLibrary    */
    void  *pso_void;     /* CFBridgingRetain'd MTLComputePipelineState */
} mtl_codegen_slot;

#ifndef MTL_CODEGEN_SLOT_COUNT
#define MTL_CODEGEN_SLOT_COUNT 16
#endif
static mtl_codegen_slot mtl_codegen_slots[MTL_CODEGEN_SLOT_COUNT];

/* Phase 4 sub-phase 4a.2b (2026-05-22): Persistent device buffers for
 * the codegen kernel-B's explicit atomic_uint args ovr_set [[buffer(16)]]
 * and ovr_gid [[buffer(17)]]. The OpenCL twin aliases these atomic
 * counters off the payload buffer at offsets +100/+104; Metal cannot
 * cast through a non-atomic pointer so the emitter wires them as
 * standalone buffer args. Lazy-allocated on first dispatch in
 * gpu_metal_kernelb_dispatch_proto; reused across all subsequent
 * dispatches in the session. Both reset to their sentinel values
 * (0 for ovr_set, 0xFFFFFFFFu for ovr_gid) before each kernel B encode.
 *
 * mtl_buf_kernelb_hashes_shown_cap tracks the byte-capacity of
 * buf_hashes_shown when allocated for the kernel-B path so a subsequent
 * dispatch with a larger hash_data_count triggers reallocation rather
 * than overflowing the dedup buffer. */
static id<MTLBuffer>               mtl_buf_kernelb_ovr_set              = nil;
static id<MTLBuffer>               mtl_buf_kernelb_ovr_gid              = nil;
static size_t                      mtl_buf_kernelb_hashes_shown_cap     = 0;


static int                         metal_ready             = 0;

/* Rules data externs. Owned by mdxfind.c (set up at session start, before
 * any GPU dispatch). gpu_rule_program is the NUL-separated bytecode the
 * kernel reads; gpu_rule_offsets is a uint32 array giving the byte offset
 * of each rule within the program. gpu_rule_count includes the synthetic
 * no-rule pass at index 0 (first rule offset points at a NUL byte; the
 * kernel reads NUL -> is_no_rule=1 -> applies no transformation).
 * Mirrors gpu/gpujob_opencl.c lines 188-191. */
extern unsigned char *gpu_rule_program;
extern uint32_t      *gpu_rule_offsets;
extern uint32_t       gpu_rule_program_len;
extern int            gpu_rule_count;

/* Phase 1a sub-phase 1a.2 (2026-05-21): CPU mask machinery externs for
 * the A2 Metal dispatch host-upload path. Globals defined in mdxfind.c
 * lines 7474-7491. The A2 dispatcher reads MaskPrependPattern[] /
 * MaskAppendPattern[] directly and translates the signed classid (-1
 * for MASK_LITERAL) into the 0xff byte sentinel per D9.2.a.
 *
 * Struct tag names MUST match the mdxfind.c definitions exactly (C
 * struct compatibility across TUs requires identical tag + members).
 * Do NOT widen without coordinated change to mdxfind.c. */
#ifndef MASK_LITERAL_CPU
#define MASK_LITERAL_CPU (-1)
#endif
/* 2026-05-26: raised from 16 to 256 to match mdxfind.c MAX_MASK_POS.
 * The extern decls below MUST mirror the storage definition in mdxfind.c
 * (currently MaskAppendPattern[256] etc.). Existing call sites here in
 * gpu_metal.m iterate MIN(MaskPrependLen, KERN_A_PATTERN_LEN_CAP=16)
 * for kernel A2 wire-format upload; the kernel-side per-side cap of 16
 * is enforced upstream in mdxfind.c by the GPU upload guards plus the
 * kernel-internal `if (npre > 16) npre = 16` clamps in
 * metal_template.metal / metal_kernel_a_*. */
#ifndef MAX_MASK_POS_CPU
#define MAX_MASK_POS_CPU 256
#endif
#ifndef MASK_MAX_CLASSES_CPU
#define MASK_MAX_CLASSES_CPU 16
#endif
struct MaskClass { unsigned char chars[256]; int count; };
struct MaskPos   { int classid; unsigned char literal; };
extern struct MaskClass MaskClasses[MASK_MAX_CLASSES_CPU];
extern struct MaskPos   MaskAppendPattern[MAX_MASK_POS_CPU];
extern int                MaskAppendLen;
extern unsigned long long MaskAppendTotal;
extern struct MaskPos   MaskPrependPattern[MAX_MASK_POS_CPU];
extern int                MaskPrependLen;
extern unsigned long long MaskPrependTotal;
extern unsigned long long MaskTotal;

/* Per-device state mirrored from gpu_opencl.c. Phase 1 keeps a flat set
 * of statics (no struct) because there's only one device; Phase 4+
 * multi-device adds a `struct metal_device dev[N]` array indexed by
 * dev_idx. */
static uint64_t  cache_compact_mask     = 0;
static uint32_t  cache_hash_data_count  = 0;
static int       cache_overflow_count   = 0;
static uint32_t  cache_max_probe        = 256u;   /* matches OpenCL default */
static int       cache_gpu_op           = 0;
static int       cache_max_iter         = 1;

/* Cached MTLBuffer handles for the compact table + overflow + hash_data.
 * Owned via ARC strong refs in static __strong vars. */
static id<MTLBuffer> buf_compact_fp       = nil;
static id<MTLBuffer> buf_compact_idx      = nil;
static id<MTLBuffer> buf_hash_data        = nil;
static id<MTLBuffer> buf_hash_data_off    = nil;
static id<MTLBuffer> buf_hash_data_len    = nil;
static id<MTLBuffer> buf_overflow_keys    = nil;
static id<MTLBuffer> buf_overflow_hashes  = nil;
static id<MTLBuffer> buf_overflow_offsets = nil;
static id<MTLBuffer> buf_hashes_shown     = nil;   /* on-GPU dedup (lazy) */

/* Phase 2a row 6: rule_program + rule_offset MTLBuffers. Uploaded lazily
 * on first rules dispatch from the gpu_rule_program / gpu_rule_offsets
 * externs (mdxfind.c populates those at session start). Reused across
 * subsequent dispatches as long as the host-side pointers haven't moved.
 * We track the captured host pointers so a re-upload happens if mdxfind
 * ever rebinds them (Phase 2a does not, but the guard is cheap). */
static id<MTLBuffer> buf_rule_program     = nil;
static id<MTLBuffer> buf_rule_offset      = nil;
static unsigned char *cached_rule_program  = NULL;
static uint32_t      *cached_rule_offsets  = NULL;
static int            cached_rule_count    = 0;

/* Phase 2b row 4: mask charset table + per-position sizes MTLBuffers.
 * Populated by gpu_metal_set_mask; bound at buffers 12 and 13 in the M-
 * and RM-variant kernel signatures. Buffer capacity is the full 32-row
 * footprint (MASK_TOTAL_CAP = 32 = MASK_POS_CAP * 2; 8 KB + 128 B); active
 * rows are filled and the rest is zero-padded sentinel + size=1 so any
 * stray divmod in the kernel terminates safely. */
#ifndef METAL_MASK_POS_CAP
#define METAL_MASK_POS_CAP 16
#endif
#ifndef METAL_MASK_TOTAL_CAP
#define METAL_MASK_TOTAL_CAP (METAL_MASK_POS_CAP * 2)
#endif
static id<MTLBuffer> buf_mask_charsets    = nil;
static id<MTLBuffer> buf_mask_sizes       = nil;

/* Phase 2c row 2: salt MTLBuffers. Populated by gpu_metal_set_salt;
 * bound at buffers 15, 16, 17 in the S/RS/MS/RMS variant kernel
 * signatures. Lifetime is per-session; the worker thread refreshes
 * on every salt-list change. Cap-grow: when nsalts_packed *
 * (sizeof(uint32) + sizeof(uint16)) or total salt bytes exceed the
 * current MTLBuffer length, re-allocate; else memcpy in place. */
static id<MTLBuffer> buf_salt_data        = nil;
static id<MTLBuffer> buf_salt_off         = nil;
static id<MTLBuffer> buf_salt_lens        = nil;
static int           cached_salts_count   = 0;
static size_t        cached_salts_data_cap = 0;
static size_t        cached_salts_off_cap  = 0;
static size_t        cached_salts_lens_cap = 0;

/* Task #250: per-lane scratch pool. Replaces the kernel's previous
 * thread-local 40 KB `uchar buf[RULE_BUF_MAX]` register array, which
 * blew the M2 Max (T6020) PSO-create register-allocator gate. Sized
 * to peak num_words * RULE_BUF_MAX so each lane (one per word in the
 * post-#250 kernel restructure) owns a contiguous RULE_BUF_MAX-byte
 * slice. The pool is grown lazily and re-allocated when num_words
 * exceeds the current capacity. Always bound at buffer 14 for every
 * PSO variant.
 *
 * 16K-word peak * 40 KB = 640 MB — fits comfortably in M1 (16 GB) and
 * M2 Max (96 GB+) unified memory. Storage mode is Private (device-
 * local; no CPU access path needed — the kernel reads + writes it but
 * the host never touches the contents). Apple's MTLResource model
 * places shared buffers in a CPU-visible region with non-trivial cache-
 * line ping-ponging; Private bypasses that. */
static id<MTLBuffer> buf_scratch_pool     = nil;
static uint32_t      buf_scratch_pool_words_cap = 0;

#ifndef METAL_RULE_BUF_MAX
/* MUST match RULE_BUF_MAX in gpu/metal_common.metal (40960). Defined here
 * because gpu_metal.m doesn't include the kernel headers — same idiom as
 * METAL_MIN_BUFFER_BYTES + GPU_MAX_HITS forward-decls. If RULE_BUF_MAX
 * ever changes in metal_common.metal this constant MUST move in lockstep
 * (a build-time static_assert is not feasible since metal_common.metal
 * is an .metal file, not a header). */
#define METAL_RULE_BUF_MAX 40960
#endif

/* Phase 2b row 4: cached host-side mask state for the hit-replay path in
 * gpu/gpujob_metal.m. The OpenCL twin owns gpu_mask_n_prepend /
 * gpu_mask_n_append / gpu_mask_total / gpu_mask_sizes[] as file-scope
 * symbols in gpu/gpu_opencl.c (line ~3993). gpujob_opencl.c externs them
 * (line ~242). We mirror that ownership here: NON-static (linker-visible)
 * so gpu/gpujob_metal.m externs and reads them in its hit-replay block. */
int      gpu_mask_n_prepend  = 0;
int      gpu_mask_n_append   = 0;
uint64_t gpu_mask_total      = 0;
uint8_t  gpu_mask_sizes[METAL_MASK_TOTAL_CAP];
/* Host-side charset rows mirrored byte-for-byte for hit-replay (gpujob_metal.m
 * reads these to reconstruct the prepend+append bytes for each hit). Row
 * layout matches the device-side buf_mask_charsets: rows [0..npre) prepend,
 * [npre..npre+napp) append; only the first gpu_mask_sizes[i] bytes of each
 * row are meaningful. Linker-visible so gpujob_metal.m externs it.
 * Mirrors the OpenCL twin's gpu_mask_desc[ntotal + i*256 + pidx] pattern
 * (gpu_opencl.c rev 1.<set_mask>; gpujob_opencl.c:1623). */
uint8_t  gpu_mask_charsets_host[METAL_MASK_TOTAL_CAP][256];

/* Persistent host-side hits buffer (caller does not free; returned by
 * gpu_metal_dispatch_md5_rules). Sized once to GPU_MAX_HITS *
 * GPU_HIT_STRIDE * 4 bytes. */
static uint32_t *h_hits = NULL;

/* MIN_BUFFER_BYTES from gpu_opencl.c — driver-side validators on some
 * platforms reject sub-floor buffers at NDRange time. Metal on macOS
 * doesn't have the same NVIDIA Windows pathology, but we keep the floor
 * for layout parity (and one Phase 1 hashes_shown placeholder allocation
 * will land below it without it). */
#ifndef METAL_MIN_BUFFER_BYTES
#define METAL_MIN_BUFFER_BYTES 4096
#endif

/* ---- Internal helpers ---- */

/* Resolve an MTLDevice. MTLCreateSystemDefaultDevice() can return nil
 * under sshd context on macOS; the retired gpu_metal.m (rev 1.45) had
 * a fallback to MTLCopyAllDevices()[0] for that case. Preserved here
 * so SSH-driven test runs from dev1.local still work. */
static id<MTLDevice> metal_resolve_device(void)
{
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (d != nil) return d;
    NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
    if (all != nil && [all count] > 0) {
        return [all objectAtIndex:0];
    }
    return nil;
}

/* Load the kernel library. Two-tier:
 *   default: load embedded gpu_mdxfind_metallib[] bytes via
 *            -[MTLDevice newLibraryWithData:error:].
 *   getenv("MDXFIND_METAL_JIT") == "1": concat the three _str.h sources
 *            and compile via -[MTLDevice newLibraryWithSource:...].
 * Returns a strong ref (nil on failure; caller's __strong static binds). */
static id<MTLLibrary> metal_load_library(id<MTLDevice> device)
{
    const char *jit_env = getenv("MDXFIND_METAL_JIT");
    int want_jit = (jit_env != NULL && jit_env[0] == '1' && jit_env[1] == '\0');

    NSError *err = nil;

    if (want_jit) {
        /* JIT path — concat metal_common + metal_md5_core + metal_template
         * in source order (matches build_metallib.sh build order). */
        size_t total = strlen(metal_common_str)
                     + strlen(metal_md5_core_str)
                     + strlen(metal_template_str)
                     + 16;
        char *src = (char *)malloc(total);
        if (src == NULL) {
            fprintf(stderr, "Metal: JIT concat malloc(%zu) failed\n", total);
            return nil;
        }
        strcpy(src, metal_common_str);
        strcat(src, "\n");
        strcat(src, metal_md5_core_str);
        strcat(src, "\n");
        strcat(src, metal_template_str);

        NSString *nsrc = [NSString stringWithUTF8String:src];
        free(src);
        if (nsrc == nil) {
            fprintf(stderr, "Metal: JIT source NSString conversion failed\n");
            return nil;
        }
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> lib = [device newLibraryWithSource:nsrc
                                                  options:opts
                                                    error:&err];
        if (lib == nil) {
            fprintf(stderr, "Metal: JIT compile failed: %s\n",
                    err ? [[err localizedDescription] UTF8String] : "(no error)");
            return nil;
        }
        GPU_DEBUG_FPRINTF(stderr, "Metal: library loaded via JIT (MDXFIND_METAL_JIT=1)\n");
        return lib;
    }

    /* Default path: embedded metallib bytes. dispatch_data_t with a
     * NULL destructor (DISPATCH_DATA_DESTRUCTOR_DEFAULT actually copies;
     * we pass DISPATCH_DATA_DESTRUCTOR_DEFAULT which copies the bytes
     * into a private buffer the dispatch_data_t owns). */
    dispatch_data_t dd = dispatch_data_create((const void *)gpu_mdxfind_metallib,
                                              (size_t)gpu_mdxfind_metallib_len,
                                              dispatch_get_main_queue(),
                                              DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    if (dd == NULL) {
        fprintf(stderr, "Metal: dispatch_data_create(%u bytes) returned NULL\n",
                gpu_mdxfind_metallib_len);
        return nil;
    }
    id<MTLLibrary> lib = [device newLibraryWithData:dd error:&err];
    if (lib == nil) {
        fprintf(stderr, "Metal: embedded metallib load failed (%u bytes): %s\n",
                gpu_mdxfind_metallib_len,
                err ? [[err localizedDescription] UTF8String] : "(no error)");
        return nil;
    }
    GPU_DEBUG_FPRINTF(stderr, "Metal: library loaded from embedded metallib (%u bytes)\n",
            gpu_mdxfind_metallib_len);
    return lib;
}

/* Lazy PSO creation. Mirrors gpu_opencl_template_kernel_lazy_md5.
 * Returns 0 on success, -1 on failure. */
static int metal_pso_template_lazy_md5(void)
{
    if (mtl_pso_template_md5 != nil) return 0;
    if (mtl_lib_template_md5 == nil) {
        fprintf(stderr, "Metal: PSO requested but library is nil\n");
        return -1;
    }
    id<MTLFunction> fn = [mtl_lib_template_md5
                          newFunctionWithName:@"template_phase0"];
    if (fn == nil) {
        fprintf(stderr, "Metal: function 'template_phase0' not found in library\n");
        return -1;
    }
    NSError *err = nil;
    mtl_pso_template_md5 = [mtl_device newComputePipelineStateWithFunction:fn
                                                                     error:&err];
    if (mtl_pso_template_md5 == nil) {
        fprintf(stderr, "Metal: PSO create failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(no error)");
        return -1;
    }
    GPU_DEBUG_FPRINTF(stderr, "Metal: PSO template_phase0 (MD5) created lazily\n");
    return 0;
}


/* Phase 2c row 1: salt-variant library loaders. Helper concatenates the
 * 4 source strings in build-order (common + md5_rules + md5salt_core +
 * template), then JIT-compiles with the requested preprocessorMacros.
 * Note metal_md5salt_core_str REPLACES metal_md5_core_str — the two
 * define the same fn names (template_state, template_init,
 * template_finalize, template_digest_compare, template_emit_hit_or_overflow,
 * template_iterate, template_transform) so they're mutually exclusive
 * within a single MTLLibrary. The 6-arg template_finalize lives in
 * metal_md5salt_core; the 3-arg form lives in metal_md5_core. */
static id<MTLLibrary> metal_load_library_salt_variant(id<MTLDevice> device,
                                                      NSDictionary *macros,
                                                      const char *label)
{
    size_t total = strlen(metal_common_str)
                 + strlen(metal_md5salt_core_str)
                 + strlen(metal_md5_rules_str)
                 + strlen(metal_template_str)
                 + 16;
    char *src = (char *)malloc(total);
    if (src == NULL) {
        fprintf(stderr, "Metal: %s JIT concat malloc(%zu) failed\n",
                label, total);
        return nil;
    }
    strcpy(src, metal_common_str);
    strcat(src, "\n");
    strcat(src, metal_md5salt_core_str);
    strcat(src, "\n");
    strcat(src, metal_md5_rules_str);
    strcat(src, "\n");
    strcat(src, metal_template_str);

    NSString *nsrc = [NSString stringWithUTF8String:src];
    free(src);
    if (nsrc == nil) {
        fprintf(stderr, "Metal: %s JIT source NSString conversion failed\n",
                label);
        return nil;
    }

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.preprocessorMacros = macros;

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:nsrc
                                              options:opts
                                                error:&err];
    if (lib == nil) {
        fprintf(stderr, "Metal: %s JIT compile failed: %s\n", label,
                err ? [[err localizedDescription] UTF8String] : "(no error)");
        return nil;
    }
    return lib;
}

static id<MTLLibrary> metal_load_library_salt(id<MTLDevice> device)
{
    id<MTLLibrary> lib = metal_load_library_salt_variant(
        device,
        @{ @"GPU_TEMPLATE_HAS_SALT": @1 },
        "salt");
    if (lib != nil) {
        GPU_DEBUG_FPRINTF(stderr,
            "Metal: salt-variant library JIT-compiled "
            "(GPU_TEMPLATE_HAS_SALT=1)\n");
    }
    return lib;
}

static id<MTLLibrary> metal_load_library_salt_rules(id<MTLDevice> device)
{
    id<MTLLibrary> lib = metal_load_library_salt_variant(
        device,
        @{ @"GPU_TEMPLATE_HAS_RULES": @1,
           @"GPU_TEMPLATE_HAS_SALT":  @1 },
        "rules+salt");
    if (lib != nil) {
        GPU_DEBUG_FPRINTF(stderr,
            "Metal: rules+salt-variant library JIT-compiled "
            "(GPU_TEMPLATE_HAS_RULES=1 GPU_TEMPLATE_HAS_SALT=1)\n");
    }
    return lib;
}

static id<MTLLibrary> metal_load_library_salt_mask(id<MTLDevice> device)
{
    id<MTLLibrary> lib = metal_load_library_salt_variant(
        device,
        @{ @"GPU_TEMPLATE_HAS_MASK": @1,
           @"GPU_TEMPLATE_HAS_SALT": @1 },
        "mask+salt");
    if (lib != nil) {
        GPU_DEBUG_FPRINTF(stderr,
            "Metal: mask+salt-variant library JIT-compiled "
            "(GPU_TEMPLATE_HAS_MASK=1 GPU_TEMPLATE_HAS_SALT=1)\n");
    }
    return lib;
}

static id<MTLLibrary> metal_load_library_salt_rules_mask(id<MTLDevice> device)
{
    id<MTLLibrary> lib = metal_load_library_salt_variant(
        device,
        @{ @"GPU_TEMPLATE_HAS_RULES": @1,
           @"GPU_TEMPLATE_HAS_MASK":  @1,
           @"GPU_TEMPLATE_HAS_SALT":  @1 },
        "rules+mask+salt");
    if (lib != nil) {
        GPU_DEBUG_FPRINTF(stderr,
            "Metal: rules+mask+salt-variant library JIT-compiled "
            "(GPU_TEMPLATE_HAS_RULES=1 GPU_TEMPLATE_HAS_MASK=1 GPU_TEMPLATE_HAS_SALT=1)\n");
    }
    return lib;
}

/* Phase 2c row 1: 4 lazy PSO creators for salt-variant kernels. Mirror
 * the Phase 2b R/M/RM lazy creators exactly with the new lib loaders.
 * Idempotent; cache-pinned at first invocation. */
int gpu_metal_template_pso_lazy_md5_salt(void)
{
    if (mtl_pso_template_md5_salt != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr, "Metal: salt PSO requested but device is nil\n");
        return -1;
    }

    @autoreleasepool {
        if (mtl_lib_template_md5_salt == nil) {
            mtl_lib_template_md5_salt = metal_load_library_salt(mtl_device);
            if (mtl_lib_template_md5_salt == nil) return -1;
        }

        id<MTLFunction> fn = [mtl_lib_template_md5_salt
                              newFunctionWithName:@"template_phase0"];
        if (fn == nil) {
            fprintf(stderr, "Metal: salt-variant 'template_phase0' "
                            "not found in JIT library\n");
            return -1;
        }
        NSError *err = nil;
        mtl_pso_template_md5_salt =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (mtl_pso_template_md5_salt == nil) {
            fprintf(stderr, "Metal: salt PSO create failed: %s\n",
                    err ? [[err localizedDescription] UTF8String]
                        : "(no error)");
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr, "Metal: PSO template_phase0 (MD5+SALT) created lazily\n");
    return 0;
}

int gpu_metal_template_pso_lazy_md5_salt_rules(void)
{
    if (mtl_pso_template_md5_salt_rules != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr, "Metal: salt+rules PSO requested but device is nil\n");
        return -1;
    }

    @autoreleasepool {
        if (mtl_lib_template_md5_salt_rules == nil) {
            mtl_lib_template_md5_salt_rules =
                metal_load_library_salt_rules(mtl_device);
            if (mtl_lib_template_md5_salt_rules == nil) return -1;
        }

        id<MTLFunction> fn = [mtl_lib_template_md5_salt_rules
                              newFunctionWithName:@"template_phase0"];
        if (fn == nil) {
            fprintf(stderr, "Metal: salt+rules-variant 'template_phase0' "
                            "not found in JIT library\n");
            return -1;
        }
        NSError *err = nil;
        mtl_pso_template_md5_salt_rules =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (mtl_pso_template_md5_salt_rules == nil) {
            fprintf(stderr, "Metal: salt+rules PSO create failed: %s\n",
                    err ? [[err localizedDescription] UTF8String]
                        : "(no error)");
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr, "Metal: PSO template_phase0 (MD5+SALT+RULES) created lazily\n");
    return 0;
}

int gpu_metal_template_pso_lazy_md5_salt_mask(void)
{
    if (mtl_pso_template_md5_salt_mask != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr, "Metal: salt+mask PSO requested but device is nil\n");
        return -1;
    }

    @autoreleasepool {
        if (mtl_lib_template_md5_salt_mask == nil) {
            mtl_lib_template_md5_salt_mask =
                metal_load_library_salt_mask(mtl_device);
            if (mtl_lib_template_md5_salt_mask == nil) return -1;
        }

        id<MTLFunction> fn = [mtl_lib_template_md5_salt_mask
                              newFunctionWithName:@"template_phase0"];
        if (fn == nil) {
            fprintf(stderr, "Metal: salt+mask-variant 'template_phase0' "
                            "not found in JIT library\n");
            return -1;
        }
        NSError *err = nil;
        mtl_pso_template_md5_salt_mask =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (mtl_pso_template_md5_salt_mask == nil) {
            fprintf(stderr, "Metal: salt+mask PSO create failed: %s\n",
                    err ? [[err localizedDescription] UTF8String]
                        : "(no error)");
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr, "Metal: PSO template_phase0 (MD5+SALT+MASK) created lazily\n");
    return 0;
}

int gpu_metal_template_pso_lazy_md5_salt_rules_mask(void)
{
    if (mtl_pso_template_md5_salt_rules_mask != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr, "Metal: salt+rules+mask PSO requested but device is nil\n");
        return -1;
    }

    @autoreleasepool {
        if (mtl_lib_template_md5_salt_rules_mask == nil) {
            mtl_lib_template_md5_salt_rules_mask =
                metal_load_library_salt_rules_mask(mtl_device);
            if (mtl_lib_template_md5_salt_rules_mask == nil) return -1;
        }

        id<MTLFunction> fn = [mtl_lib_template_md5_salt_rules_mask
                              newFunctionWithName:@"template_phase0"];
        if (fn == nil) {
            fprintf(stderr, "Metal: salt+rules+mask-variant 'template_phase0' "
                            "not found in JIT library\n");
            return -1;
        }
        NSError *err = nil;
        mtl_pso_template_md5_salt_rules_mask =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (mtl_pso_template_md5_salt_rules_mask == nil) {
            fprintf(stderr, "Metal: salt+rules+mask PSO create failed: %s\n",
                    err ? [[err localizedDescription] UTF8String]
                        : "(no error)");
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr, "Metal: PSO template_phase0 (MD5+SALT+RULES+MASK) created lazily\n");
    return 0;
}

/* Phase 1a sub-phase 1a.1b-continued (2026-05-20): Metal A1 (rules-only)
 * standalone kernel library loader. Concatenates metal_common_str +
 * metal_kernel_a_rules_str (mirror of OpenCL twin's two-source build at
 * gpu/gpu_opencl.c:9621). NO preprocessor macros -- KERNEL_A_VARIANT
 * baking happens in the source itself (variant=1 == rules-only is the
 * only shape A1 ships). Always JIT (no embedded-metallib entry per the
 * rules-variant precedent). */
static id<MTLLibrary> metal_load_library_kernel_a_rules(id<MTLDevice> device)
{
    size_t total = strlen(metal_common_str)
                 + strlen(metal_kernel_a_rules_str)
                 + 16;
    char *src = (char *)malloc(total);
    if (src == NULL) {
        fprintf(stderr, "Metal: kernel A1 JIT concat malloc(%zu) failed\n", total);
        return nil;
    }
    strcpy(src, metal_common_str);
    strcat(src, "\n");
    strcat(src, metal_kernel_a_rules_str);

    NSString *nsrc = [NSString stringWithUTF8String:src];
    free(src);
    if (nsrc == nil) {
        fprintf(stderr, "Metal: kernel A1 JIT source NSString conversion failed\n");
        return nil;
    }
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    /* No preprocessorMacros -- variant=1 is baked into the source. */
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:nsrc
                                              options:opts
                                                error:&err];
    if (lib == nil) {
        fprintf(stderr, "Metal: kernel A1 JIT compile failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(no error)");
        return nil;
    }
    GPU_DEBUG_FPRINTF(stderr,
        "Metal: kernel A1 (rules-only) library JIT-compiled\n");
    return lib;
}

/* Phase 1a sub-phase 1a.1b-continued (2026-05-20): lazy PSO creator for
 * kernel A1 standalone entry point `cand_rules_phase0`. Mirrors
 * gpu_metal_template_pso_lazy_md5_salt_rules contract: idempotent on
 * subsequent invocations (cache-pinned at first call); returns 0 on
 * success, -1 on failure. */
static int metal_kernel_a_rules_pso_lazy(void)
{
    if (mtl_pso_kernel_a_rules != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr, "Metal: kernel A1 PSO requested but device is nil\n");
        return -1;
    }

    @autoreleasepool {
        if (mtl_lib_kernel_a_rules == nil) {
            mtl_lib_kernel_a_rules = metal_load_library_kernel_a_rules(mtl_device);
            if (mtl_lib_kernel_a_rules == nil) return -1;
        }

        id<MTLFunction> fn = [mtl_lib_kernel_a_rules
                              newFunctionWithName:@"cand_rules_phase0"];
        if (fn == nil) {
            fprintf(stderr, "Metal: kernel A1 entry 'cand_rules_phase0' "
                            "not found in JIT library\n");
            return -1;
        }
        NSError *err = nil;
        mtl_pso_kernel_a_rules =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (mtl_pso_kernel_a_rules == nil) {
            fprintf(stderr, "Metal: kernel A1 PSO create failed: %s\n",
                    err ? [[err localizedDescription] UTF8String]
                        : "(no error)");
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr, "Metal: PSO cand_rules_phase0 (kernel A1) created lazily\n");
    return 0;
}

/* Phase 1a sub-phase 1a.2 (2026-05-21): Metal A2 (masks-only) standalone
 * kernel library loader. Mirrors metal_load_library_kernel_a_rules above;
 * concatenates metal_common_str + metal_kernel_a_masks_str. No
 * preprocessor macros; variant=2 is baked into the source. */
static id<MTLLibrary> metal_load_library_kernel_a_masks(id<MTLDevice> device)
{
    size_t total = strlen(metal_common_str)
                 + strlen(metal_kernel_a_masks_str)
                 + 16;
    char *src = (char *)malloc(total);
    if (src == NULL) {
        fprintf(stderr, "Metal: kernel A2 JIT concat malloc(%zu) failed\n", total);
        return nil;
    }
    strcpy(src, metal_common_str);
    strcat(src, "\n");
    strcat(src, metal_kernel_a_masks_str);

    NSString *nsrc = [NSString stringWithUTF8String:src];
    free(src);
    if (nsrc == nil) {
        fprintf(stderr, "Metal: kernel A2 JIT source NSString conversion failed\n");
        return nil;
    }
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:nsrc
                                              options:opts
                                                error:&err];
    if (lib == nil) {
        fprintf(stderr, "Metal: kernel A2 JIT compile failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(no error)");
        return nil;
    }
    GPU_DEBUG_FPRINTF(stderr,
        "Metal: kernel A2 (masks-only) library JIT-compiled\n");
    return lib;
}

/* Phase 1a sub-phase 1a.2 (2026-05-21): lazy PSO creator for the A2
 * standalone entry point `cand_masks_phase0`. Mirrors metal_kernel_a_-
 * rules_pso_lazy contract: idempotent, returns 0 on success, -1 on
 * failure. */
static int metal_kernel_a_masks_pso_lazy(void)
{
    if (mtl_pso_kernel_a_masks != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr, "Metal: kernel A2 PSO requested but device is nil\n");
        return -1;
    }

    @autoreleasepool {
        if (mtl_lib_kernel_a_masks == nil) {
            mtl_lib_kernel_a_masks = metal_load_library_kernel_a_masks(mtl_device);
            if (mtl_lib_kernel_a_masks == nil) return -1;
        }

        id<MTLFunction> fn = [mtl_lib_kernel_a_masks
                              newFunctionWithName:@"cand_masks_phase0"];
        if (fn == nil) {
            fprintf(stderr, "Metal: kernel A2 entry 'cand_masks_phase0' "
                            "not found in JIT library\n");
            return -1;
        }
        NSError *err = nil;
        mtl_pso_kernel_a_masks =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (mtl_pso_kernel_a_masks == nil) {
            fprintf(stderr, "Metal: kernel A2 PSO create failed: %s\n",
                    err ? [[err localizedDescription] UTF8String]
                        : "(no error)");
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr, "Metal: PSO cand_masks_phase0 (kernel A2) created lazily\n");
    return 0;
}

/* Phase 1a sub-phase 1a.3 (2026-05-21): Metal A3 (rules+masks) standalone
 * kernel library loader. Mirrors metal_load_library_kernel_a_masks above;
 * concatenates metal_common_str + metal_kernel_a_rules_masks_str. No
 * preprocessor macros; variant=3 is baked into the source. */
static id<MTLLibrary> metal_load_library_kernel_a_rules_masks(id<MTLDevice> device)
{
    size_t total = strlen(metal_common_str)
                 + strlen(metal_kernel_a_rules_masks_str)
                 + 16;
    char *src = (char *)malloc(total);
    if (src == NULL) {
        fprintf(stderr, "Metal: kernel A3 JIT concat malloc(%zu) failed\n", total);
        return nil;
    }
    strcpy(src, metal_common_str);
    strcat(src, "\n");
    strcat(src, metal_kernel_a_rules_masks_str);

    NSString *nsrc = [NSString stringWithUTF8String:src];
    free(src);
    if (nsrc == nil) {
        fprintf(stderr, "Metal: kernel A3 JIT source NSString conversion failed\n");
        return nil;
    }
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:nsrc
                                              options:opts
                                                error:&err];
    if (lib == nil) {
        fprintf(stderr, "Metal: kernel A3 JIT compile failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(no error)");
        return nil;
    }
    GPU_DEBUG_FPRINTF(stderr,
        "Metal: kernel A3 (rules+masks) library JIT-compiled\n");
    return lib;
}

/* Phase 1a sub-phase 1a.3 (2026-05-21): lazy PSO creator for the A3
 * standalone entry point `cand_rules_masks_phase0`. Mirrors metal_kernel_-
 * a_masks_pso_lazy contract: idempotent, returns 0 on success, -1 on
 * failure. */
static int metal_kernel_a_rules_masks_pso_lazy(void)
{
    if (mtl_pso_kernel_a_rules_masks != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr, "Metal: kernel A3 PSO requested but device is nil\n");
        return -1;
    }

    @autoreleasepool {
        if (mtl_lib_kernel_a_rules_masks == nil) {
            mtl_lib_kernel_a_rules_masks =
                metal_load_library_kernel_a_rules_masks(mtl_device);
            if (mtl_lib_kernel_a_rules_masks == nil) return -1;
        }

        id<MTLFunction> fn = [mtl_lib_kernel_a_rules_masks
                              newFunctionWithName:@"cand_rules_masks_phase0"];
        if (fn == nil) {
            fprintf(stderr, "Metal: kernel A3 entry 'cand_rules_masks_phase0' "
                            "not found in JIT library\n");
            return -1;
        }
        NSError *err = nil;
        mtl_pso_kernel_a_rules_masks =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (mtl_pso_kernel_a_rules_masks == nil) {
            fprintf(stderr, "Metal: kernel A3 PSO create failed: %s\n",
                    err ? [[err localizedDescription] UTF8String]
                        : "(no error)");
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr,
        "Metal: PSO cand_rules_masks_phase0 (kernel A3) created lazily\n");
    return 0;
}

/* Phase 1a sub-phase 1a.4 (2026-05-21): A4 standalone Metal kernel
 * library loader. Mirrors metal_load_library_kernel_a_rules_masks above;
 * concatenates metal_common_str + metal_kernel_a_bruteforce_str. No
 * preprocessor macros; variant=4 is baked into the source. */
static id<MTLLibrary> metal_load_library_kernel_a_bruteforce(id<MTLDevice> device)
{
    size_t total = strlen(metal_common_str)
                 + strlen(metal_kernel_a_bruteforce_str)
                 + 16;
    char *src = (char *)malloc(total);
    if (src == NULL) {
        fprintf(stderr, "Metal: kernel A4 JIT concat malloc(%zu) failed\n", total);
        return nil;
    }
    strcpy(src, metal_common_str);
    strcat(src, "\n");
    strcat(src, metal_kernel_a_bruteforce_str);

    NSString *nsrc = [NSString stringWithUTF8String:src];
    free(src);
    if (nsrc == nil) {
        fprintf(stderr, "Metal: kernel A4 JIT source NSString conversion failed\n");
        return nil;
    }
    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:nsrc
                                              options:opts
                                                error:&err];
    if (lib == nil) {
        fprintf(stderr, "Metal: kernel A4 JIT compile failed: %s\n",
                err ? [[err localizedDescription] UTF8String] : "(no error)");
        return nil;
    }
    GPU_DEBUG_FPRINTF(stderr,
        "Metal: kernel A4 (bruteforce) library JIT-compiled\n");
    return lib;
}

/* Phase 1a sub-phase 1a.4 (2026-05-21): lazy PSO creator for the A4
 * standalone entry point `cand_bruteforce_phase0`. Mirrors metal_kernel_-
 * a_rules_masks_pso_lazy contract: idempotent, returns 0 on success,
 * -1 on failure. */
static int metal_kernel_a_bruteforce_pso_lazy(void)
{
    if (mtl_pso_kernel_a_bruteforce != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr, "Metal: kernel A4 PSO requested but device is nil\n");
        return -1;
    }

    @autoreleasepool {
        if (mtl_lib_kernel_a_bruteforce == nil) {
            mtl_lib_kernel_a_bruteforce =
                metal_load_library_kernel_a_bruteforce(mtl_device);
            if (mtl_lib_kernel_a_bruteforce == nil) return -1;
        }

        id<MTLFunction> fn = [mtl_lib_kernel_a_bruteforce
                              newFunctionWithName:@"cand_bruteforce_phase0"];
        if (fn == nil) {
            fprintf(stderr, "Metal: kernel A4 entry 'cand_bruteforce_phase0' "
                            "not found in JIT library\n");
            return -1;
        }
        NSError *err = nil;
        mtl_pso_kernel_a_bruteforce =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (mtl_pso_kernel_a_bruteforce == nil) {
            fprintf(stderr, "Metal: kernel A4 PSO create failed: %s\n",
                    err ? [[err localizedDescription] UTF8String]
                        : "(no error)");
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr,
        "Metal: PSO cand_bruteforce_phase0 (kernel A4) created lazily\n");
    return 0;
}

/* Phase 2e.1: monotonic microsecond clock for per-dispatch wall
 * measurement. Local copy (gpu/gpujob_metal.m has a `static` peer
 * we cannot link from this TU). */
static uint64_t metal_now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ull + (uint64_t)(ts.tv_nsec / 1000);
}

/* Phase 2e: per-tier SALT_BATCH selector.
 *
 * SALT_BATCH controls the inner-salt-loop tile width under
 * GPU_TEMPLATE_HAS_PRE_SALT. On Metal one-thread-per-word the tile is
 * an unroll hint to the compiler -- the kernel still iterates the full
 * salt list, just in stride-SALT_BATCH chunks. The pre-salt hoist
 * itself is the main perf win (one inner-MD5+hex32 per (word, rule,
 * mask) instead of one per salt); SALT_BATCH gives the Metal compiler
 * a hint to unroll the inner cycle and amortise register loads.
 *
 * Per-tier table (Phase 2e §3):
 *   M1 (8-core GPU):     SALT_BATCH = 8   (conservative; smaller
 *                                          register budget)
 *   M2 / M2 Max:         SALT_BATCH = 16  (sweet spot per OpenCL twin
 *                                          empirical data on similar
 *                                          register-budget cores)
 *   M3+ / unknown:       SALT_BATCH = 32  (more registers; defer 2e.2
 *                                          adaptive measurement)
 *
 * MTLDevice.name regex selection. Env override MDXFIND_METAL_SALT_BATCH
 * mirrors the OpenCL OCL_SALT_BATCH pattern -- if the operator wants
 * to force a particular value it takes precedence. */
static uint32_t metal_select_salt_batch(id<MTLDevice> device)
{
    /* Env override wins regardless of device tier. */
    const char *env = getenv("MDXFIND_METAL_SALT_BATCH");
    if (env != NULL) {
        long v = strtol(env, NULL, 10);
        if (v >= 1 && v <= 256) return (uint32_t)v;
    }

    if (device == nil) return 16u;  /* defensive default */

    NSString *name = [device name];
    if (name == nil) return 16u;
    const char *cname = [name UTF8String];
    if (cname == NULL) return 16u;

    /* Cheapest possible classifier -- substring on the device name. The
     * MTLDevice.name strings on Apple Silicon are stable: "Apple M1",
     * "Apple M1 Pro", "Apple M1 Max", "Apple M2", "Apple M2 Max", etc.
     * We branch on M1 vs M2 vs M3+ family, ignoring the suffix because
     * the inner-loop register budget tracks the GPU core architecture,
     * not the variant. */
    if (strstr(cname, "M3") != NULL ||
        strstr(cname, "M4") != NULL ||
        strstr(cname, "M5") != NULL) {
        return 32u;
    }
    if (strstr(cname, "M2") != NULL) {
        return 16u;
    }
    if (strstr(cname, "M1") != NULL) {
        return 8u;
    }

    /* Intel discrete / unknown -- the presalt variant is e31-specific
     * and unlikely to fire on Intel iMac in production, but pick a
     * conservative default in case it's exercised by harness smoke. */
    return 16u;
}

/* Phase 2e.1: per-tier SALT_CHUNK selector.
 *
 * Caps salt iteration per dispatch so the per-dispatch wall stays well
 * under Apple's ~2s ImpactingInteractivity watchdog ceiling. The outer
 * salt-chunk loop in gpu_metal_dispatch_md5_rules iterates salt-pages
 * of this size; each dispatch sees num_salts_per_page = this_chunk and
 * kernel emits combined_ridx page-locally. Host post-process (Path 1b)
 * rewrites hits to global encoding after each dispatch.
 *
 * Per-tier table (Phase 2e.1 §3):
 *   M1 (8-core GPU):     SALT_CHUNK = 64    (watchdog headroom on M1)
 *   M2 / M2 Max:         SALT_CHUNK = 256   (sweet spot per memo §6)
 *   M3+ / unknown:       SALT_CHUNK = 1024  (more headroom; 2e.2 auto-tune)
 *
 * MTLDevice.name substring selection. Env override MDXFIND_METAL_SALT_CHUNK
 * (mirrors MDXFIND_METAL_RULE_CHUNK / MDXFIND_METAL_SALT_BATCH) wins. */
static uint32_t metal_select_salt_chunk(id<MTLDevice> device)
{
    const char *env = getenv("MDXFIND_METAL_SALT_CHUNK");
    if (env != NULL) {
        long v = strtol(env, NULL, 10);
        if (v >= 1 && v <= 1000000) return (uint32_t)v;
    }

    if (device == nil) return 256u;

    NSString *name = [device name];
    if (name == nil) return 256u;
    const char *cname = [name UTF8String];
    if (cname == NULL) return 256u;

    if (strstr(cname, "M3") != NULL ||
        strstr(cname, "M4") != NULL ||
        strstr(cname, "M5") != NULL) {
        return 1024u;
    }
    if (strstr(cname, "M2") != NULL) {
        return 256u;
    }
    if (strstr(cname, "M1") != NULL) {
        return 64u;
    }

    return 256u;
}

/* Phase 2e: salt+rules+presalt MTLLibrary loader. Mirrors
 * metal_load_library_salt_rules but adds GPU_TEMPLATE_HAS_PRE_SALT=1
 * + SALT_BATCH=N to the preprocessor macros. The SALT_BATCH value is
 * baked into the JIT compile -- a different per-tier value produces
 * a separate MTLLibrary + PSO. */
static id<MTLLibrary> metal_load_library_salt_rules_presalt(id<MTLDevice> device,
                                                            uint32_t salt_batch)
{
    NSDictionary *macros = @{
        @"GPU_TEMPLATE_HAS_RULES":    @1,
        @"GPU_TEMPLATE_HAS_SALT":     @1,
        @"GPU_TEMPLATE_HAS_PRE_SALT": @1,
        @"SALT_BATCH":                @(salt_batch),
    };
    id<MTLLibrary> lib = metal_load_library_salt_variant(
        device, macros, "salt+rules+presalt");
    if (lib != nil) {
        GPU_DEBUG_FPRINTF(stderr,
            "Metal: salt+rules+presalt-variant library JIT-compiled "
            "(GPU_TEMPLATE_HAS_RULES=1 GPU_TEMPLATE_HAS_SALT=1 "
            "GPU_TEMPLATE_HAS_PRE_SALT=1 SALT_BATCH=%u)\n",
            (unsigned)salt_batch);
    }
    return lib;
}

/* Phase 2e: salt+rules+presalt lazy PSO creator. Mirrors
 * gpu_metal_template_pso_lazy_md5_salt_rules with the new library
 * loader; preserves the Tier 2 eager-compile pattern (so macOS 26.3
 * G13 AGX UserShaderFactory crash on first-dispatch PSO finalize
 * stays sidestepped). */
int gpu_metal_template_pso_lazy_md5_salt_rules_presalt(void)
{
    if (mtl_pso_template_md5_salt_rules_presalt != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr, "Metal: salt+rules+presalt PSO requested but device is nil\n");
        return -1;
    }

    @autoreleasepool {
        if (mtl_lib_template_md5_salt_rules_presalt == nil) {
            uint32_t salt_batch = metal_select_salt_batch(mtl_device);
            mtl_lib_template_md5_salt_rules_presalt =
                metal_load_library_salt_rules_presalt(mtl_device, salt_batch);
            if (mtl_lib_template_md5_salt_rules_presalt == nil) return -1;
        }

        id<MTLFunction> fn = [mtl_lib_template_md5_salt_rules_presalt
                              newFunctionWithName:@"template_phase0"];
        if (fn == nil) {
            fprintf(stderr, "Metal: salt+rules+presalt-variant 'template_phase0' "
                            "not found in JIT library\n");
            return -1;
        }
        NSError *err = nil;
        mtl_pso_template_md5_salt_rules_presalt =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (mtl_pso_template_md5_salt_rules_presalt == nil) {
            fprintf(stderr, "Metal: salt+rules+presalt PSO create failed: %s\n",
                    err ? [[err localizedDescription] UTF8String]
                        : "(no error)");
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr, "Metal: PSO template_phase0 (MD5+SALT+RULES+PRESALT) created lazily\n");
    return 0;
}

/* Phase 2c row 2: gpu_metal_set_salt — bind a salt list to the GPU.
 * Mirrors gpu/gpu_opencl.c gpu_opencl_set_salts (line ~3805) exactly.
 *
 * The kernel reads three flat buffers:
 *   salt_buf[]         concatenated salt bytes; salt N at byte offset
 *                      salt_offsets[N], length salt_lens[N].
 *   salt_off[]         per-salt uint32 byte offset.
 *   salt_lens[]        per-salt uint16 length.
 *
 * Cap-grow pattern: when current MTLBuffer capacity is exceeded, re-
 * allocate via newBufferWithLength + memcpy; else memcpy in place into
 * the existing buffer's contents (Apple unified memory makes the in-place
 * path zero-copy on M-series; Intel pays a DMA per refresh — both
 * acceptable for the per-batch refresh rate).
 *
 * Returns 0 on success, -1 on error. Per-session lifetime; safe to call
 * many times per session (the worker calls it whenever the salt snapshot
 * changes). Logs `Metal: salts uploaded (...)` marker on success. */
int gpu_metal_set_salt(const char *salts, const uint32_t *salt_offsets,
                       const uint16_t *salt_lens, int num_salts)
{
    if (!metal_ready) {
        fprintf(stderr, "Metal: set_salt called before init\n");
        return -1;
    }
    if (num_salts <= 0) {
        fprintf(stderr, "Metal: set_salt: num_salts=%d, nothing to bind\n",
                num_salts);
        return -1;
    }
    if (salts == NULL || salt_offsets == NULL || salt_lens == NULL) {
        fprintf(stderr,
            "Metal: set_salt: NULL pointer (salts=%p offsets=%p lens=%p)\n",
            (const void *)salts, (const void *)salt_offsets,
            (const void *)salt_lens);
        return -1;
    }

    /* Compute the total salt-data bytes from offsets+lens (matches
     * gpu_opencl_set_salts behavior — caller's `salts` buffer is sized
     * to fit all entries contiguously). The largest reachable byte is
     * salt_offsets[num_salts-1] + salt_lens[num_salts-1]. */
    size_t data_bytes = (size_t)salt_offsets[num_salts - 1]
                      + (size_t)salt_lens[num_salts - 1];
    size_t off_bytes  = (size_t)num_salts * sizeof(uint32_t);
    size_t lens_bytes = (size_t)num_salts * sizeof(uint16_t);

    if (data_bytes < METAL_MIN_BUFFER_BYTES) data_bytes = METAL_MIN_BUFFER_BYTES;
    if (off_bytes  < METAL_MIN_BUFFER_BYTES) off_bytes  = METAL_MIN_BUFFER_BYTES;
    if (lens_bytes < METAL_MIN_BUFFER_BYTES) lens_bytes = METAL_MIN_BUFFER_BYTES;

    @autoreleasepool {
        /* Cap-grow salt_data. */
        if (buf_salt_data == nil || data_bytes > cached_salts_data_cap) {
            buf_salt_data =
                [mtl_device newBufferWithLength:data_bytes
                                        options:MTLResourceStorageModeShared];
            if (buf_salt_data == nil) {
                /* Phase D5a (Task #281): newBufferWithLength returns nil
                 * silently on alloc failure (no NSError out-param). Fatal. */
                GPU_FATAL("Metal: salt_data newBuffer(%zu bytes) failed (num_salts=%d)",
                          data_bytes, num_salts);
            }
            cached_salts_data_cap = data_bytes;
        }
        memset([buf_salt_data contents], 0, cached_salts_data_cap);
        /* The actual salt-byte count is the smaller of data_bytes and
         * the live span (offsets[N-1] + lens[N-1]); memcpy that. */
        size_t live_data = (size_t)salt_offsets[num_salts - 1]
                         + (size_t)salt_lens[num_salts - 1];
        memcpy([buf_salt_data contents], salts, live_data);

        /* Cap-grow salt_off. */
        if (buf_salt_off == nil || off_bytes > cached_salts_off_cap) {
            buf_salt_off =
                [mtl_device newBufferWithLength:off_bytes
                                        options:MTLResourceStorageModeShared];
            if (buf_salt_off == nil) {
                /* Phase D5a (Task #281): alloc failure -> fatal. */
                GPU_FATAL("Metal: salt_off newBuffer(%zu bytes) failed (num_salts=%d)",
                          off_bytes, num_salts);
            }
            cached_salts_off_cap = off_bytes;
        }
        memset([buf_salt_off contents], 0, cached_salts_off_cap);
        memcpy([buf_salt_off contents], salt_offsets,
               (size_t)num_salts * sizeof(uint32_t));

        /* Cap-grow salt_lens. */
        if (buf_salt_lens == nil || lens_bytes > cached_salts_lens_cap) {
            buf_salt_lens =
                [mtl_device newBufferWithLength:lens_bytes
                                        options:MTLResourceStorageModeShared];
            if (buf_salt_lens == nil) {
                /* Phase D5a (Task #281): alloc failure -> fatal. */
                GPU_FATAL("Metal: salt_lens newBuffer(%zu bytes) failed (num_salts=%d)",
                          lens_bytes, num_salts);
            }
            cached_salts_lens_cap = lens_bytes;
        }
        memset([buf_salt_lens contents], 0, cached_salts_lens_cap);
        memcpy([buf_salt_lens contents], salt_lens,
               (size_t)num_salts * sizeof(uint16_t));
    }

    cached_salts_count = num_salts;
    GPU_DEBUG_FPRINTF(stderr, "Metal: salts uploaded (%d entries, %zu bytes)\n",
            num_salts, (size_t)salt_offsets[num_salts - 1]
                     + (size_t)salt_lens[num_salts - 1]);
    return 0;
}

/* Phase 2a row 6: lazy upload of the rule program + offset table to
 * device-side MTLBuffers. Reads gpu_rule_program / gpu_rule_offsets /
 * gpu_rule_program_len / gpu_rule_count from mdxfind.c.
 *
 * Returns 0 on success, -1 if the rule globals are not populated or a
 * buffer allocation fails. Idempotent: re-uses existing buffers if the
 * host pointers + count haven't changed since the prior call. */
static int metal_upload_rules_lazy(void)
{
    if (gpu_rule_program == NULL || gpu_rule_offsets == NULL
        || gpu_rule_program_len == 0 || gpu_rule_count <= 0) {
        fprintf(stderr, "Metal: rules upload requested but rule globals "
                        "unset (program=%p offsets=%p len=%u count=%d)\n",
                (void *)gpu_rule_program, (void *)gpu_rule_offsets,
                gpu_rule_program_len, gpu_rule_count);
        return -1;
    }

    /* Cache hit: same host pointers + count -> nothing to do. */
    if (buf_rule_program != nil && buf_rule_offset != nil
        && cached_rule_program == gpu_rule_program
        && cached_rule_offsets == gpu_rule_offsets
        && cached_rule_count   == gpu_rule_count)
        return 0;

    /* (Re)upload. ARC drops old refs when we overwrite the statics. */
    size_t prog_bytes = (size_t)gpu_rule_program_len;
    if (prog_bytes < METAL_MIN_BUFFER_BYTES) prog_bytes = METAL_MIN_BUFFER_BYTES;
    size_t off_bytes  = (size_t)gpu_rule_count * sizeof(uint32_t);
    if (off_bytes < METAL_MIN_BUFFER_BYTES) off_bytes = METAL_MIN_BUFFER_BYTES;

    buf_rule_program = [mtl_device newBufferWithLength:prog_bytes
                                               options:MTLResourceStorageModeShared];
    if (buf_rule_program == nil) {
        /* Phase D5a (Task #281): alloc failure -> fatal. */
        GPU_FATAL("Metal: rule_program newBuffer(%zu bytes) failed (rule_count=%d)",
                  prog_bytes, gpu_rule_count);
    }
    memset([buf_rule_program contents], 0, prog_bytes);
    memcpy([buf_rule_program contents], gpu_rule_program,
           (size_t)gpu_rule_program_len);

    buf_rule_offset = [mtl_device newBufferWithLength:off_bytes
                                              options:MTLResourceStorageModeShared];
    if (buf_rule_offset == nil) {
        /* Phase D5a (Task #281): alloc failure -> fatal. */
        GPU_FATAL("Metal: rule_offset newBuffer(%zu bytes) failed (rule_count=%d)",
                  off_bytes, gpu_rule_count);
    }
    memset([buf_rule_offset contents], 0, off_bytes);
    memcpy([buf_rule_offset contents], gpu_rule_offsets,
           (size_t)gpu_rule_count * sizeof(uint32_t));

    cached_rule_program = gpu_rule_program;
    cached_rule_offsets = gpu_rule_offsets;
    cached_rule_count   = gpu_rule_count;

    GPU_DEBUG_FPRINTF(stderr, "Metal: rule_program (%u bytes) + rule_offset "
                    "(%d entries) uploaded\n",
            gpu_rule_program_len, gpu_rule_count);
    return 0;
}

/* Phase 2b row 4: gpu_metal_set_mask — bind a multi-position prepend+
 * append charset table to the GPU. Mirrors gpu/gpu_opencl.c
 * gpu_opencl_set_mask (line ~3998) exactly. mdxfind.c calls this from
 * two sites (rules-engine mask activation and BF activation) — both
 * extracted from the existing OPENCL_GPU-gated arms in a parallel
 * commit in the main session.
 *
 * The kernel reads two flat buffers:
 *   mask_charsets[MASK_TOTAL_CAP * 256] = 8 KB, row-major. Rows
 *     [0..npre) are prepend charsets; rows [npre..npre+napp) are append
 *     charsets. Unused rows are zero-filled sentinel.
 *   mask_sizes[MASK_TOTAL_CAP]          = 128 B uint32. mask_sizes[i] is
 *     the modulus for position i; unused entries are 1 sentinel.
 *
 * Idempotent: if invoked twice with the same args the buffers are
 * recreated (cheap — Apple unified memory). Side effects:
 *   - Allocates buf_mask_charsets + buf_mask_sizes via
 *     MTLResourceStorageModeShared.
 *   - Populates gpu_mask_n_prepend / gpu_mask_n_append / gpu_mask_total /
 *     gpu_mask_sizes[] (linker-visible) so gpujob_metal.m's hit-replay
 *     block can decode mask_idx without re-uploading the host descriptor.
 *
 * Returns 0 on success; -1 on bounds violation or alloc failure. */
int gpu_metal_set_mask(const uint8_t *sizes, const uint8_t tables[][256],
                       int npre, int napp)
{
    if (!metal_ready) {
        fprintf(stderr, "Metal: set_mask called before init\n");
        return -1;
    }
    if (sizes == NULL || tables == NULL) {
        fprintf(stderr, "Metal: set_mask: NULL sizes/tables (npre=%d napp=%d)\n",
                npre, napp);
        return -1;
    }
    if (npre < 0 || napp < 0
        || npre > METAL_MASK_POS_CAP || napp > METAL_MASK_POS_CAP) {
        fprintf(stderr, "Metal: set_mask: position count out of range "
                        "(npre=%d napp=%d, cap=%d)\n",
                npre, napp, METAL_MASK_POS_CAP);
        return -1;
    }
    int ntotal = npre + napp;
    if (ntotal < 1) {
        fprintf(stderr, "Metal: set_mask: npre+napp == 0 (nothing to bind)\n");
        return -1;
    }

    /* Update host-visible cache + descriptor for hit-replay decode. */
    gpu_mask_n_prepend = npre;
    gpu_mask_n_append  = napp;
    memset(gpu_mask_sizes, 1, sizeof(gpu_mask_sizes));
    for (int i = 0; i < ntotal; i++) gpu_mask_sizes[i] = sizes[i];
    memset(gpu_mask_charsets_host, 0, sizeof(gpu_mask_charsets_host));
    for (int i = 0; i < ntotal; i++) {
        memcpy(gpu_mask_charsets_host[i], tables[i], 256);
    }
    uint64_t total = 1;
    for (int i = 0; i < ntotal; i++) {
        uint32_t sz = sizes[i] ? sizes[i] : 1u;
        total *= sz;
    }
    gpu_mask_total = total;

    /* Build the packed kernel-side buffers. Pattern matches gpu_opencl_set_-
     * mask's b7_charsets / b7_sizes packing (lines 4047-4063). */
    uint8_t  b7_charsets[METAL_MASK_TOTAL_CAP * 256];
    uint32_t b7_sizes[METAL_MASK_TOTAL_CAP];
    memset(b7_charsets, 0, sizeof(b7_charsets));
    for (int i = 0; i < METAL_MASK_TOTAL_CAP; i++) b7_sizes[i] = 1u;
    for (int i = 0; i < ntotal; i++) {
        memcpy(b7_charsets + i * 256, tables[i], 256);
        b7_sizes[i] = (uint32_t)(sizes[i] ? sizes[i] : 1);
    }

    @autoreleasepool {
        /* ARC drops the prior strong refs when we overwrite. Recreate on
         * every call — idempotent, low-cost on Apple unified memory. */
        buf_mask_charsets =
            [mtl_device newBufferWithBytes:b7_charsets
                                    length:sizeof(b7_charsets)
                                   options:MTLResourceStorageModeShared];
        if (buf_mask_charsets == nil) {
            fprintf(stderr, "Metal: set_mask: mask_charsets newBuffer(%zu) failed\n",
                    sizeof(b7_charsets));
            return -1;
        }
        buf_mask_sizes =
            [mtl_device newBufferWithBytes:b7_sizes
                                    length:sizeof(b7_sizes)
                                   options:MTLResourceStorageModeShared];
        if (buf_mask_sizes == nil) {
            fprintf(stderr, "Metal: set_mask: mask_sizes newBuffer(%zu) failed\n",
                    sizeof(b7_sizes));
            buf_mask_charsets = nil;
            return -1;
        }
    }

    GPU_DEBUG_FPRINTF(stderr, "Metal: mask binding (%d positions, %llu combos)\n",
            ntotal, (unsigned long long)gpu_mask_total);
    return 0;
}

/* ---- Public API ---- */

/* Forward declaration for the built-in family registration helper. The
 * family struct + resolvers + register call live further down (after the
 * lazy creators they reference); gpu_metal_init invokes this at the end
 * so the dispatcher's family-lookup path sees the registry populated. */
static void metal_register_builtin_families(void);

int gpu_metal_init(void)
{
    if (metal_ready) return 0;

    @autoreleasepool {
        mtl_device = metal_resolve_device();
        if (mtl_device == nil) {
            fprintf(stderr, "Metal: no GPU device available "
                            "(MTLCreateSystemDefaultDevice + MTLCopyAllDevices both empty)\n");
            return -1;
        }
        fprintf(stderr, "Metal: device = %s\n",
                [[mtl_device name] UTF8String]);

        mtl_queue = [mtl_device newCommandQueue];
        if (mtl_queue == nil) {
            fprintf(stderr, "Metal: newCommandQueue returned nil\n");
            mtl_device = nil;
            return -1;
        }

        mtl_lib_template_md5 = metal_load_library(mtl_device);
        if (mtl_lib_template_md5 == nil) {
            /* metal_load_library already logged the cause. */
            mtl_queue  = nil;
            mtl_device = nil;
            return -1;
        }
    }

    /* Persistent host hits buffer. Sized to the max one dispatch can
     * emit; reused across all dispatches. */
    if (h_hits == NULL) {
        size_t bytes = (size_t)GPU_MAX_HITS * GPU_HIT_STRIDE * sizeof(uint32_t);
        h_hits = (uint32_t *)calloc(1, bytes);
        if (h_hits == NULL) {
            fprintf(stderr, "Metal: h_hits calloc(%zu) failed\n", bytes);
            return -1;
        }
    }

    /* Phase 2d.2.1a: populate the family registry. Must happen after
     * metal_ready = 1 because the lazy creators (invoked later via the
     * resolvers in gpu_metal_compile_families) test metal_ready and
     * early-return if it is clear. */
    metal_ready = 1;
    metal_register_builtin_families();
    GPU_DEBUG_FPRINTF(stderr, "Metal GPU: 1 device initialized\n");
    return 0;
}

void gpu_metal_shutdown(void)
{
    if (!metal_ready) return;
    metal_ready = 0;

    @autoreleasepool {
        /* Drain any outstanding work. A no-op if no dispatch ran. */
        if (mtl_queue != nil) {
            id<MTLCommandBuffer> cb = [mtl_queue commandBuffer];
            [cb commit];
            [cb waitUntilCompleted];
        }
    }

    /* ARC: assign nil to drop strong refs. The buffer statics
     * are autoreleased when their last dispatch's autoreleasepool
     * drains; the static vars themselves go nil here. */
    buf_compact_fp       = nil;
    buf_compact_idx      = nil;
    buf_hash_data        = nil;
    buf_hash_data_off    = nil;
    buf_hash_data_len    = nil;
    buf_overflow_keys    = nil;
    buf_overflow_hashes  = nil;
    buf_overflow_offsets = nil;
    buf_hashes_shown     = nil;
    buf_rule_program     = nil;
    buf_rule_offset      = nil;
    cached_rule_program  = NULL;
    cached_rule_offsets  = NULL;
    cached_rule_count    = 0;
    buf_mask_charsets    = nil;
    buf_mask_sizes       = nil;
    buf_salt_data        = nil;
    buf_salt_off         = nil;
    buf_salt_lens        = nil;
    cached_salts_count   = 0;
    cached_salts_data_cap = 0;
    cached_salts_off_cap  = 0;
    cached_salts_lens_cap = 0;
    buf_scratch_pool     = nil;
    buf_scratch_pool_words_cap = 0;
    gpu_mask_n_prepend   = 0;
    gpu_mask_n_append    = 0;
    gpu_mask_total       = 0;
    memset(gpu_mask_sizes, 0, sizeof(gpu_mask_sizes));
    memset(gpu_mask_charsets_host, 0, sizeof(gpu_mask_charsets_host));
    /* Phase 1a sub-phase 1a.1b-continued: kernel A1 statics. */
    mtl_pso_kernel_a_rules      = nil;
    mtl_lib_kernel_a_rules      = nil;
    mtl_buf_proto_packed        = nil;
    mtl_buf_proto_chunk_index   = nil;
    mtl_buf_proto_state         = nil;
    mtl_buf_proto_packed_cap    = 0;
    mtl_buf_proto_index_cap     = 0;
    /* Phase 4 sub-phase 4a.2b (2026-05-22): kernel B production
     * dispatcher persistent atomic_uint buffers + hashes_shown cap. */
    mtl_buf_kernelb_ovr_set     = nil;
    mtl_buf_kernelb_ovr_gid     = nil;
    mtl_buf_kernelb_hashes_shown_cap = 0;
    /* Phase 1a sub-phase 1a.2 (2026-05-21): kernel A2 statics. */
    mtl_pso_kernel_a_masks       = nil;
    mtl_lib_kernel_a_masks       = nil;
    /* Phase 1a sub-phase 1a.3 (2026-05-21): A3 (rules+masks) statics. */
    mtl_pso_kernel_a_rules_masks = nil;
    mtl_lib_kernel_a_rules_masks = nil;
    /* Phase 1a sub-phase 1a.4 (2026-05-21): A4 (brute-force) statics. */
    mtl_pso_kernel_a_bruteforce  = nil;
    mtl_lib_kernel_a_bruteforce  = nil;
    mtl_buf_kern_a_mask_prepend  = nil;
    mtl_buf_kern_a_mask_append   = nil;
    mtl_buf_kern_a_mask_charsets = nil;
    mtl_buf_kern_a_mask_counts   = nil;
    mtl_buf_kern_a_mask_allocated = 0;
    if (h_proto_packed_readback != NULL) {
        free(h_proto_packed_readback);
        h_proto_packed_readback     = NULL;
        h_proto_packed_readback_cap = 0;
    }
    if (h_proto_index_readback != NULL) {
        free(h_proto_index_readback);
        h_proto_index_readback     = NULL;
        h_proto_index_readback_cap = 0;
    }
    /* Phase 2c salt-variant statics. */
    mtl_pso_template_md5_salt_rules_presalt = nil;
    mtl_lib_template_md5_salt_rules_presalt = nil;
    mtl_pso_template_md5_salt_rules_mask = nil;
    mtl_lib_template_md5_salt_rules_mask = nil;
    mtl_pso_template_md5_salt_mask       = nil;
    mtl_lib_template_md5_salt_mask       = nil;
    mtl_pso_template_md5_salt_rules      = nil;
    mtl_lib_template_md5_salt_rules      = nil;
    mtl_pso_template_md5_salt            = nil;
    mtl_lib_template_md5_salt            = nil;
    mtl_pso_template_md5 = nil;
    mtl_lib_template_md5 = nil;

    mtl_queue            = nil;
    mtl_device           = nil;

    if (h_hits != NULL) {
        free(h_hits);
        h_hits = NULL;
    }
}

int gpu_metal_available(void) { return metal_ready; }

int gpu_metal_set_compact_table(int dev_idx,
    uint32_t *compact_fp, uint32_t *compact_idx,
    uint64_t compact_size, uint64_t compact_mask,
    unsigned char *hash_data_buf, size_t hash_data_buf_size,
    size_t *hash_data_off, size_t hash_data_count,
    unsigned short *hash_data_len)
{
    (void)hash_data_len;  /* Phase 1: not yet uploaded as a buffer. */
    if (!metal_ready || dev_idx != 0) return -1;
    if (compact_fp == NULL || compact_idx == NULL) return -1;

    @autoreleasepool {
        size_t fp_bytes  = compact_size * sizeof(uint32_t);
        size_t idx_bytes = compact_size * sizeof(uint32_t);

        buf_compact_fp = [mtl_device newBufferWithBytes:compact_fp
                                                 length:fp_bytes
                                                options:MTLResourceStorageModeShared];
        if (buf_compact_fp == nil) {
            fprintf(stderr, "Metal: compact_fp newBuffer(%zu) failed\n", fp_bytes);
            return -1;
        }
        buf_compact_idx = [mtl_device newBufferWithBytes:compact_idx
                                                  length:idx_bytes
                                                 options:MTLResourceStorageModeShared];
        if (buf_compact_idx == nil) {
            fprintf(stderr, "Metal: compact_idx newBuffer(%zu) failed\n", idx_bytes);
            buf_compact_fp = nil;
            return -1;
        }

        if (hash_data_buf != NULL && hash_data_buf_size > 0) {
            buf_hash_data = [mtl_device newBufferWithBytes:hash_data_buf
                                                    length:hash_data_buf_size
                                                   options:MTLResourceStorageModeShared];
            if (buf_hash_data == nil) {
                fprintf(stderr, "Metal: hash_data newBuffer(%zu) failed\n",
                        hash_data_buf_size);
                buf_compact_fp = buf_compact_idx = nil;
                return -1;
            }
        }

        /* hash_data_off is size_t[] on the host but the kernel reads
         * uint64. On a 64-bit host they're the same width; copy through
         * a uint64 staging array regardless so a future 32-bit-host port
         * stays sound. */
        if (hash_data_off != NULL && hash_data_count > 0) {
            size_t bytes = hash_data_count * sizeof(uint64_t);
            uint64_t *staging = (uint64_t *)malloc(bytes);
            if (staging == NULL) {
                fprintf(stderr, "Metal: hash_data_off staging malloc(%zu) failed\n",
                        bytes);
                buf_compact_fp = buf_compact_idx = buf_hash_data = nil;
                return -1;
            }
            for (size_t i = 0; i < hash_data_count; i++)
                staging[i] = (uint64_t)hash_data_off[i];
            buf_hash_data_off = [mtl_device newBufferWithBytes:staging
                                                        length:bytes
                                                       options:MTLResourceStorageModeShared];
            free(staging);
            if (buf_hash_data_off == nil) {
                fprintf(stderr, "Metal: hash_data_off newBuffer(%zu) failed\n", bytes);
                buf_compact_fp = buf_compact_idx = buf_hash_data = nil;
                return -1;
            }
        }

        if (hash_data_len != NULL && hash_data_count > 0) {
            size_t bytes = hash_data_count * sizeof(unsigned short);
            buf_hash_data_len = [mtl_device newBufferWithBytes:hash_data_len
                                                        length:bytes
                                                       options:MTLResourceStorageModeShared];
            /* Non-fatal if this fails — the Phase 1 kernel doesn't read it. */
        }
    }

    cache_compact_mask    = compact_mask;
    cache_hash_data_count = (uint32_t)hash_data_count;

    GPU_DEBUG_FPRINTF(stderr, "Metal GPU[0]: compact table registered "
            "(%llu slots, %u hashes)\n",
            (unsigned long long)compact_size, (unsigned)hash_data_count);
    return 0;
}

int gpu_metal_set_overflow(int dev_idx,
    const uint64_t *keys, const unsigned char *hashes,
    const uint32_t *offsets, const uint16_t *lengths, int count)
{
    (void)lengths;
    if (!metal_ready || dev_idx != 0) return -1;

    @autoreleasepool {
        if (count <= 0) {
            /* Bind METAL_MIN_BUFFER_BYTES zero placeholders so the kernel's
             * device-buffer args are always valid pointers. The kernel
             * guards `if (overflow_count > 0u)` before reading these. */
            void *zero = calloc(1, METAL_MIN_BUFFER_BYTES);
            if (zero == NULL) {
                fprintf(stderr, "Metal: overflow placeholder calloc failed\n");
                return -1;
            }
            buf_overflow_keys = [mtl_device newBufferWithBytes:zero
                                                        length:METAL_MIN_BUFFER_BYTES
                                                       options:MTLResourceStorageModeShared];
            buf_overflow_hashes = [mtl_device newBufferWithBytes:zero
                                                          length:METAL_MIN_BUFFER_BYTES
                                                         options:MTLResourceStorageModeShared];
            buf_overflow_offsets = [mtl_device newBufferWithBytes:zero
                                                           length:METAL_MIN_BUFFER_BYTES
                                                          options:MTLResourceStorageModeShared];
            free(zero);
            cache_overflow_count = 0;
            return 0;
        }

        /* B5 sub-batch 7 pad parity: last entry padded to 16 bytes (or
         * its real length, whichever is larger) so the GPU's 4xuint32
         * probe never spills into the next entry. Mirrors gpu_opencl.c. */
        size_t last_pad = (lengths != NULL && lengths[count - 1] >= 16)
                          ? lengths[count - 1] : 16;
        size_t hash_bytes = offsets[count - 1] + last_pad;
        size_t keys_bytes = (size_t)count * sizeof(uint64_t);
        size_t offs_bytes = (size_t)count * sizeof(uint32_t);

        buf_overflow_keys = [mtl_device newBufferWithBytes:keys
                                                    length:keys_bytes
                                                   options:MTLResourceStorageModeShared];
        buf_overflow_hashes = [mtl_device newBufferWithBytes:hashes
                                                      length:hash_bytes
                                                     options:MTLResourceStorageModeShared];
        buf_overflow_offsets = [mtl_device newBufferWithBytes:offsets
                                                       length:offs_bytes
                                                      options:MTLResourceStorageModeShared];
        if (buf_overflow_keys == nil || buf_overflow_hashes == nil ||
            buf_overflow_offsets == nil) {
            fprintf(stderr, "Metal: overflow buffer alloc failed "
                            "(keys=%zu hashes=%zu offsets=%zu)\n",
                    keys_bytes, hash_bytes, offs_bytes);
            return -1;
        }
        cache_overflow_count = count;
    }
    return 0;
}

void gpu_metal_set_op(int dev_idx, int op)
{
    if (dev_idx != 0) return;
    cache_gpu_op = op;
}

void gpu_metal_set_max_iter(int dev_idx, int max_iter)
{
    if (dev_idx != 0) return;
    cache_max_iter = (max_iter < 1) ? 1 : max_iter;
}

/* ---- Phase 2d.2.1a: per-family struct + op-keyed dispatcher ----
 *
 * Family registry. Linear scan over a small fixed array; the expected
 * algorithm count is on the order of 10s, not 1000s. Two families are
 * registered today: md5 (op=JOB_MD5) and md5salt (op=JOB_MD5SALT).
 * Subsequent phases (2d.2.1b md4 onwards) extend this without touching
 * the dispatcher cascade — only an additional gpu_metal_register_family
 * call in metal_register_builtin_families plus a new pso_for_variant
 * resolver. */
#ifndef METAL_FAMILY_CAP
/* Phase 2d.7b (2026-05-16): bumped from 32 to 64. Phase 2d.7b added 8
 * Keccak/SHA-3 sponge families (KECCAK-{224,256,384,512} + SHA3-{224,256,
 * 384,512}); registry count grew from 30 to 38, overrunning the prior
 * 32-entry cap. 64 gives headroom for Phase 2d.7c (Streebog) +
 * Phase 2d.7d (HMAC siblings) without another bump. */
#define METAL_FAMILY_CAP 64
#endif

static struct gpu_metal_family *metal_families[METAL_FAMILY_CAP];
static int                      metal_family_count = 0;

/* === D5b Wave 1 shared-loader refactor 2026-05-16 ===
 *
 * Hidden parallel arrays for the generic loader path. Indexed by
 * fam->fam_idx (populated by metal_assign_fam_indices below) and variant_bits
 * (0..7). When a family migrates to .pso_for_variant_v2, its libraries +
 * PSOs land in these arrays instead of per-family statics. Mirrors the
 * existing metal_admit_mask[] parallel-array pattern at line 19737. */
static __strong id<MTLLibrary>              metal_family_libs[METAL_FAMILY_CAP][8];
static __strong id<MTLComputePipelineState> metal_family_psos[METAL_FAMILY_CAP][8];

void gpu_metal_register_family(struct gpu_metal_family *f)
{
    if (f == NULL) return;
    /* Default fam_idx to -1 (sentinel) so a family that forgets the post-
     * registration walk is loudly indexed-out-of-bounds rather than
     * silently aliasing slot 0. The walk in metal_assign_fam_indices
     * overwrites this with the correct registry index. */
    f->fam_idx = -1;
    /* Idempotent on op: replace any existing entry. */
    for (int i = 0; i < metal_family_count; i++) {
        if (metal_families[i]->op == f->op) {
            metal_families[i] = f;
            return;
        }
    }
    if (metal_family_count >= METAL_FAMILY_CAP) {
        fprintf(stderr,
                "Metal: family registry full (%d entries) -- dropping op=%d (%s)\n",
                METAL_FAMILY_CAP, f->op, f->name ? f->name : "?");
        return;
    }
    metal_families[metal_family_count++] = f;
}

/* Post-registration walk: assign fam_idx for every registered family.
 * Called from metal_register_builtin_families() AFTER the last register
 * call. Idempotent (safe to re-run if registry mutates). */
static void metal_assign_fam_indices(void)
{
    for (int i = 0; i < metal_family_count; i++) {
        if (metal_families[i] != NULL) metal_families[i]->fam_idx = i;
    }
}

/* === D5b Wave 1 generic loader / lazy / resolver ===
 *
 * Generic library loader. JIT-compiles a Metal library from concat of:
 *   metal_common_str
 *   fam->core_str            (the per-family algo body)
 *   metal_md5_rules_str      (only if variant_bits & V_R)
 *   metal_template_str
 *
 * Macros: copy fam->base_macros (NSDictionary) and add variant-derived
 *   GPU_TEMPLATE_HAS_RULES / _HAS_MASK / _HAS_SALT per variant_bits.
 *
 * On runtime failure during JIT, returns nil and the caller's eager-probe
 * tier (gpu_metal_compile_families) treats nil as "capability missing"
 * per D5a §6 Option B carve-out. At dispatch time (post-init) the resolver
 * indirection in gpu_metal_dispatch_md5_rules promotes nil to GPU_FATAL.
 *
 * For families whose V_NONE variant ships in the embedded metallib
 * (currently only md5), the v2 resolver SHOULD bypass this loader for
 * V_NONE and continue to use the embedded-metallib path (mtl_lib_template_*
 * statics populated at gpu_metal_init time). Wave 1 md5 keeps that
 * behavior; non-md5 families never had embedded metallib anyway. */
static id<MTLLibrary> metal_load_library_generic(id<MTLDevice> device,
                                                 struct gpu_metal_family *fam,
                                                 uint8_t variant_bits)
{
    if (fam == NULL || fam->core_str == NULL) {
        fprintf(stderr,
                "Metal: generic loader called with missing family or core_str "
                "(fam=%p)\n", (void *)fam);
        return nil;
    }

    int has_rules = (variant_bits & V_R) ? 1 : 0;
    int has_mask  = (variant_bits & V_M) ? 1 : 0;
    int has_salt  = (variant_bits & V_S) ? 1 : 0;

    size_t total = strlen(metal_common_str)
                 + strlen(fam->core_str)
                 + (has_rules ? strlen(metal_md5_rules_str) : 0)
                 + strlen(metal_template_str)
                 + 64;
    char *src = (char *)malloc(total);
    if (src == NULL) {
        fprintf(stderr,
                "Metal: generic loader malloc(%zu) failed for %s vbits=0x%x\n",
                total, fam->name ? fam->name : "?", (unsigned)variant_bits);
        return nil;
    }
    strcpy(src, metal_common_str);
    strcat(src, "\n");
    strcat(src, fam->core_str);
    if (has_rules) {
        strcat(src, "\n");
        strcat(src, metal_md5_rules_str);
    }
    strcat(src, "\n");
    strcat(src, metal_template_str);

    NSString *nsrc = [NSString stringWithUTF8String:src];
    free(src);
    if (nsrc == nil) {
        fprintf(stderr,
                "Metal: generic loader NSString conversion failed for %s vbits=0x%x\n",
                fam->name ? fam->name : "?", (unsigned)variant_bits);
        return nil;
    }

    NSMutableDictionary *macros = nil;
    if (fam->base_macros != NULL) {
        NSDictionary *base = (__bridge NSDictionary *)fam->base_macros;
        macros = [base mutableCopy];
    } else {
        macros = [NSMutableDictionary dictionary];
    }
    if (has_rules) macros[@"GPU_TEMPLATE_HAS_RULES"] = @1;
    if (has_mask)  macros[@"GPU_TEMPLATE_HAS_MASK"]  = @1;
    if (has_salt)  macros[@"GPU_TEMPLATE_HAS_SALT"]  = @1;

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    opts.preprocessorMacros = macros;

    NSError *err = nil;
    id<MTLLibrary> lib = [device newLibraryWithSource:nsrc
                                              options:opts
                                                error:&err];
    if (lib == nil) {
        fprintf(stderr,
                "Metal: %s vbits=0x%x generic library JIT-compile failed: %s\n",
                fam->name ? fam->name : "?", (unsigned)variant_bits,
                err ? [[err localizedDescription] UTF8String] : "(no error)");
        return nil;
    }
    GPU_DEBUG_FPRINTF(stderr,
            "Metal: %s-variant library JIT-compiled (generic vbits=0x%x rules=%d mask=%d salt=%d)\n",
            fam->name ? fam->name : "?", (unsigned)variant_bits,
            has_rules, has_mask, has_salt);
    return lib;
}

/* Generic lazy PSO creator. Reads fam->fam_idx to index into the parallel
 * arrays. Returns 0 on success (PSO present at metal_family_psos[fi][vb]),
 * -1 on failure (library or PSO compile failed). Idempotent. */
static int metal_pso_lazy_generic(struct gpu_metal_family *fam,
                                  uint8_t variant_bits)
{
    if (fam == NULL) {
        fprintf(stderr, "Metal: lazy_generic called with NULL family\n");
        return -1;
    }
    if (fam->fam_idx < 0 || fam->fam_idx >= METAL_FAMILY_CAP) {
        fprintf(stderr,
                "Metal: lazy_generic: family %s has invalid fam_idx=%d "
                "(post-registration walk did not run?)\n",
                fam->name ? fam->name : "?", fam->fam_idx);
        return -1;
    }
    if (variant_bits >= 8u) {
        fprintf(stderr,
                "Metal: lazy_generic: family %s invalid variant_bits=0x%x\n",
                fam->name ? fam->name : "?", (unsigned)variant_bits);
        return -1;
    }
    int fi = fam->fam_idx;
    if (metal_family_psos[fi][variant_bits] != nil) return 0;
    if (mtl_device == nil) {
        fprintf(stderr,
                "Metal: lazy_generic: device nil (family %s vbits=0x%x)\n",
                fam->name ? fam->name : "?", (unsigned)variant_bits);
        return -1;
    }
    @autoreleasepool {
        if (metal_family_libs[fi][variant_bits] == nil) {
            metal_family_libs[fi][variant_bits] =
                metal_load_library_generic(mtl_device, fam, variant_bits);
            if (metal_family_libs[fi][variant_bits] == nil) return -1;
        }
        id<MTLFunction> fn = [metal_family_libs[fi][variant_bits]
                              newFunctionWithName:@"template_phase0"];
        if (fn == nil) {
            fprintf(stderr,
                    "Metal: lazy_generic: 'template_phase0' not found "
                    "(family %s vbits=0x%x)\n",
                    fam->name ? fam->name : "?", (unsigned)variant_bits);
            return -1;
        }
        NSError *err = nil;
        metal_family_psos[fi][variant_bits] =
            [mtl_device newComputePipelineStateWithFunction:fn error:&err];
        if (metal_family_psos[fi][variant_bits] == nil) {
            fprintf(stderr,
                    "Metal: lazy_generic: PSO create failed (family %s vbits=0x%x): %s\n",
                    fam->name ? fam->name : "?", (unsigned)variant_bits,
                    err ? [[err localizedDescription] UTF8String] : "(no error)");
            return -1;
        }
    }
    GPU_DEBUG_FPRINTF(stderr,
            "Metal: PSO template_phase0 (%s vbits=0x%x) created lazily via generic\n",
            fam->name ? fam->name : "?", (unsigned)variant_bits);
    return 0;
}

/* Default v2 resolver for families that use the generic loader exclusively. */
static void *metal_pso_for_variant_default(struct gpu_metal_family *fam,
                                           uint8_t variant_bits)
{
    if (metal_pso_lazy_generic(fam, variant_bits) < 0) return NULL;
    return (__bridge void *)metal_family_psos[fam->fam_idx][variant_bits];
}

struct gpu_metal_family *gpu_metal_lookup_family(int op)
{
    for (int i = 0; i < metal_family_count; i++) {
        if (metal_families[i]->op == op)
            return metal_families[i];
    }
    return NULL;
}

/* D5b Wave 1 v2 resolver for md5. Uses generic loader for V_R/V_M/V_RM;
 * V_NONE continues to use the embedded-metallib path (mtl_pso_template_md5)
 * which is loaded at gpu_metal_init time -- migrating that to JIT would
 * be a behavior change without a parity benefit. */
static void *md5_pso_for_variant_v2(struct gpu_metal_family *fam,
                                    uint8_t variant_bits)
{
    if (variant_bits & V_S) return NULL;
    if (variant_bits == V_NONE) {
        /* Embedded metallib path -- preserved unchanged for Wave 1. */
        if (metal_pso_template_lazy_md5() < 0) return NULL;
        return (__bridge void *)mtl_pso_template_md5;
    }
    /* V_R / V_M / V_RM -- generic JIT path. */
    return metal_pso_for_variant_default(fam, variant_bits);
}

/* Resolver for the md5salt family (op=JOB_MD5SALT).
 *
 * Supported variants: V_S, V_S|V_R, V_S|V_M, V_S|V_R|V_M.
 * Internally folds the Phase 2e PRESALT optimization into the V_S|V_R
 * arm — when neither V_M is set (i.e., variant_bits == V_S|V_R exactly),
 * the resolver returns the PRESALT PSO if it lazy-creates successfully,
 * else falls back to the non-PRESALT salt_rules PSO. This hides the
 * PRESALT detail from variant_bits per design decision 1 in the brief.
 *
 * Returns NULL on lazy-creation failure or on unsupported variant_bits
 * (e.g., V_NONE — the unsalted no-rule path lives in the md5 family).
 *
 * D5b Wave 3 2026-05-16: PARTIALLY ALIVE -- the V_S|V_R PRESALT branch
 * stays here; non-PRESALT variants migrate to the generic loader via
 * md5salt_pso_for_variant_v2 below. This resolver remains invoked
 * exclusively by the v2 wrapper for the V_S|V_R case. Final cleanup
 * deferred to Wave N+1 after Wave 4 lands (PRESALT may stay custom even
 * post-cleanup if no generic equivalent is built). */
static void *md5salt_pso_for_variant(uint8_t variant_bits)
{
    /* md5salt requires V_S — the family is the salted branch. */
    if (!(variant_bits & V_S)) return NULL;

    switch (variant_bits) {
    case V_S:
        if (gpu_metal_template_pso_lazy_md5_salt() < 0) return NULL;
        return (__bridge void *)mtl_pso_template_md5_salt;

    case (V_S | V_R):
        /* PRESALT fold: the V_S|V_R variant has a Phase 2e pre-salt
         * hoist optimization. Try the PRESALT lazy creator first; if it
         * succeeds use the PRESALT PSO. If it fails fall back to the
         * non-PRESALT salt_rules PSO so the dispatch still works. */
        if (gpu_metal_template_pso_lazy_md5_salt_rules_presalt() == 0
            && mtl_pso_template_md5_salt_rules_presalt != nil) {
            return (__bridge void *)mtl_pso_template_md5_salt_rules_presalt;
        }
        if (gpu_metal_template_pso_lazy_md5_salt_rules() < 0) return NULL;
        return (__bridge void *)mtl_pso_template_md5_salt_rules;

    case (V_S | V_M):
        if (gpu_metal_template_pso_lazy_md5_salt_mask() < 0) return NULL;
        return (__bridge void *)mtl_pso_template_md5_salt_mask;

    case (V_S | V_R | V_M):
        if (gpu_metal_template_pso_lazy_md5_salt_rules_mask() < 0) return NULL;
        return (__bridge void *)mtl_pso_template_md5_salt_rules_mask;

    default:
        return NULL;
    }
}

/* D5b Wave 3 2026-05-16: v2 resolver for md5salt -- PARTIAL migration.
 *
 * md5salt is the one Wave 3 family that retains a per-family resolver:
 * the V_S|V_R PRESALT fold (Phase 2e hoist of the salt-mix into a
 * pre-compiled PSO) is custom and stays on the legacy path. The other
 * three variants (V_S, V_S|V_M, V_S|V_R|V_M) have no PRESALT equivalent
 * and route through the generic loader.
 *
 * Wiring: metal_family_md5salt.pso_for_variant_v2 = md5salt_pso_for_variant_v2;
 * the legacy .pso_for_variant slot retained so any future runtime clearing
 * of v2 still falls back to the original behavior (PRESALT included).
 *
 * Per architect memo §B "Quirky resolvers stay custom (Wave 4 -- keep custom
 * signatures matching the new (fam, vbits) pattern)" -- md5salt was listed
 * in Wave 3 because most of its variants ARE genericizable. The PRESALT
 * V_S|V_R arm stays custom. */
static void *md5salt_pso_for_variant_v2(struct gpu_metal_family *fam,
                                        uint8_t variant_bits)
{
    if (!(variant_bits & V_S)) return NULL;
    if (variant_bits == (V_S | V_R)) {
        /* PRESALT fold preserved: delegate to the legacy resolver which
         * tries the PRESALT lazy creator first and falls back to the
         * non-PRESALT salt_rules PSO on failure. */
        return md5salt_pso_for_variant(variant_bits);
    }
    /* V_S, V_S|V_M, V_S|V_R|V_M -- generic JIT path. */
    return metal_pso_for_variant_default(fam, variant_bits);
}

/* Static family descriptors.
 *
 * supported_variants encodes admissible variant_bits values as a bitmask
 * where bit-position N is set iff variant_bits == N is admissible. That
 * is, supported_variants & (1u << variant_bits) is non-zero for every
 * admissible variant_bits. variant_bits ranges 0..7 (3 bits: V_R V_M V_S)
 * so supported_variants fits in 8 bits. The resolver still does its own
 * switch on variant_bits and enforces admissibility; supported_variants
 * is for the eager-compile loop in gpu_metal_compile_families and any
 * future capability-query callers. */
#define VBIT(combo) (1u << (combo))

static struct gpu_metal_family metal_family_md5 = {
    .op                 = JOB_MD5,
    .name               = "md5",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    /* D5b Wave 1: v2 resolver + core_str populated below. base_macros set
     * in metal_register_builtin_families because @{...} dict literals can
     * only appear in fn-scope Obj-C context. */
    .pso_for_variant_v2 = md5_pso_for_variant_v2,
    .core_str           = NULL,   /* populated in metal_register_builtin_families */
    .base_macros        = NULL,   /* populated in metal_register_builtin_families */
    .dispatch_tg_size   = 0,
    .fam_idx            = -1,     /* populated by metal_assign_fam_indices */
};

static struct gpu_metal_family metal_family_md5salt = {
    .op                 = JOB_MD5SALT,
    .name               = "md5salt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = md5salt_pso_for_variant,
    /* D5b Wave 3 2026-05-16: PARTIAL generic-loader migration. v2 wrapper
     * preserves PRESALT custom path for V_S|V_R; other variants generic.
     * core_str + base_macros populated in metal_register_builtin_families. */
    .pso_for_variant_v2 = md5salt_pso_for_variant_v2,
};

static struct gpu_metal_family metal_family_md4 = {
    .op                 = JOB_MD4,
    .name               = "md4",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    /* D5b Wave 2 2026-05-16: generic-loader v2 path. core_str + base_macros
     * populated in metal_register_builtin_families. */
    .pso_for_variant_v2 = metal_pso_for_variant_default,
};

static struct gpu_metal_family metal_family_md4utf16 = {
    .op                 = JOB_MD4UTF16,
    .name               = "md4utf16",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_md5raw = {
    .op                 = JOB_MD5RAW,
    .name               = "md5raw",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_md5passsalt = {
    .op                 = JOB_MD5PASSSALT,
    .name               = "md5passsalt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_md5saltpass = {
    .op                 = JOB_MD5SALTPASS,
    .name               = "md5saltpass",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_sha1 = {
    .op                 = JOB_SHA1,
    .name               = "sha1",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha1raw = {
    .op                 = JOB_SHA1RAW,
    .name               = "sha1raw",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha1dru = {
    .op                 = JOB_SHA1DRU,
    .name               = "sha1dru",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha1passsalt = {
    .op                 = JOB_SHA1PASSSALT,
    .name               = "sha1passsalt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_sha1saltpass = {
    .op                 = JOB_SHA1SALTPASS,
    .name               = "sha1saltpass",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_sha256 = {
    .op                 = JOB_SHA256,
    .name               = "sha256",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha256raw = {
    .op                 = JOB_SHA256RAW,
    .name               = "sha256raw",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha224 = {
    .op                 = JOB_SHA224,
    .name               = "sha224",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha256passsalt = {
    .op                 = JOB_SHA256PASSSALT,
    .name               = "sha256passsalt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_sha256saltpass = {
    .op                 = JOB_SHA256SALTPASS,
    .name               = "sha256saltpass",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_sha224saltpass = {
    .op                 = JOB_SHA224SALTPASS,
    .name               = "sha224saltpass",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_sha512 = {
    .op                 = JOB_SHA512,
    .name               = "sha512",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha512raw = {
    .op                 = JOB_SHA512RAW,
    .name               = "sha512raw",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha384 = {
    .op                 = JOB_SHA384,
    .name               = "sha384",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha384raw = {
    .op                 = JOB_SHA384RAW,
    .name               = "sha384raw",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha512passsalt = {
    .op                 = JOB_SHA512PASSSALT,
    .name               = "sha512passsalt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_sha512saltpass = {
    .op                 = JOB_SHA512SALTPASS,
    .name               = "sha512saltpass",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_sha384saltpass = {
    .op                 = JOB_SHA384SALTPASS,
    .name               = "sha384saltpass",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_ripemd160 = {
    .op                 = JOB_RMD160,
    .name               = "ripemd160",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_ripemd320 = {
    .op                 = JOB_RMD320,
    .name               = "ripemd320",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_blake2s256 = {
    .op                 = JOB_BLAKE2S256,
    .name               = "blake2s256",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_blake2b256 = {
    .op                 = JOB_BLAKE2B256,
    .name               = "blake2b256",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_blake2b512 = {
    .op                 = JOB_BLAKE2B512,
    .name               = "blake2b512",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_keccak256 = {
    .op                 = JOB_KECCAK256,
    .name               = "keccak256",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};
static struct gpu_metal_family metal_family_keccak224 = {
    .op                 = JOB_KECCAK224,
    .name               = "keccak224",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_keccak384 = {
    .op                 = JOB_KECCAK384,
    .name               = "keccak384",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_keccak512 = {
    .op                 = JOB_KECCAK512,
    .name               = "keccak512",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha3_224 = {
    .op                 = JOB_SHA3_224,
    .name               = "sha3_224",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha3_256 = {
    .op                 = JOB_SHA3_256,
    .name               = "sha3_256",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha3_384 = {
    .op                 = JOB_SHA3_384,
    .name               = "sha3_384",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

static struct gpu_metal_family metal_family_sha3_512 = {
    .op                 = JOB_SHA3_512,
    .name               = "sha3_512",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};


static struct gpu_metal_family metal_family_streebog256 = {
    .op                 = JOB_STREEBOG_32,
    .name               = "streebog256",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};


static struct gpu_metal_family metal_family_streebog512 = {
    .op                 = JOB_STREEBOG_64,
    .name               = "streebog512",
    .op_category        = GPU_CAT_UNSALTED,
    .supported_variants = (uint8_t)(
                            VBIT(V_NONE)
                          | VBIT(V_R)
                          | VBIT(V_M)
                          | VBIT(V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 2 */
};

/* D5b Wave 4 2026-05-16: HMAC-BLAKE2S v2 resolver. Single-struct family (no
 * dual-struct sibling); route all 4 variants through the generic loader.
 * core_str + base_macros set in metal_register_builtin_families below. */
static struct gpu_metal_family metal_family_hmac_blake2s = {
    .op                 = JOB_HMAC_BLAKE2S,
    .name               = "hmac_blake2s",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 4 */
};

/* Two family struct entries: one per op. Same resolver function pointer.
 * gpu_metal_register_family() inserts both into metal_families[] so
 * gpu_metal_lookup_family(op) resolves either op to the shared resolver.
 *
 * D5b Wave 4 2026-05-16: HMAC-STREEBOG-256 SHARED v2 resolver. Both KSALT
 * and KPASS struct entries point at this function. The CANONICAL fam_idx
 * (where libs/psos cache in metal_family_libs[][]/metal_family_psos[][])
 * is metal_family_hmac_streebog256_ksalt.fam_idx -- the KPASS entry's slot
 * in the parallel arrays stays NULL throughout the process lifetime.
 *
 * Why: ONE compiled MTLLibrary serves BOTH ops. algo_mode (5=KSALT,
 * 6=KPASS) is host-set at dispatch time per op (see algo_mode setter in
 * gpu_metal_dispatch_md5_rules), so the same PSO produces different
 * runtime behavior for the two ops. Caching the PSO under one fam_idx
 * avoids duplicate compiles + duplicate JIT log lines.
 *
 * Per architect memo §F risk 3 (HMAC dual-struct ordering breakage):
 * the canonical fam (KSALT) MUST be the one looked up here; using fam
 * (caller's fam) would point the parallel-array index at KPASS for KPASS
 * dispatches, causing KSALT and KPASS to compile TWO PSOs instead of
 * sharing ONE.
 *
 * Function body lives AFTER the struct definitions below -- newer clang
 * (dev1) rejects forward-decl-then-definition of a static struct object,
 * so the wrappers are full forward declarations with the body deferred
 * until after the canonical fam_struct is defined. */
static void *hmac_streebog256_shared_pso_for_variant_v2(
    struct gpu_metal_family *fam, uint8_t variant_bits);

static struct gpu_metal_family metal_family_hmac_streebog256_ksalt = {
    .op                 = JOB_HMAC_STREEBOG256_KSALT,
    .name               = "hmac_streebog256_ksalt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = hmac_streebog256_shared_pso_for_variant_v2,  /* Wave 4 */
};

static struct gpu_metal_family metal_family_hmac_streebog256_kpass = {
    .op                 = JOB_HMAC_STREEBOG256_KPASS,
    .name               = "hmac_streebog256_kpass",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = hmac_streebog256_shared_pso_for_variant_v2,  /* Wave 4 */
};

/* Definition deferred from forward-decl above -- now safe to take address
 * of metal_family_hmac_streebog256_ksalt. (void)fam suppresses unused-
 * param warning -- the caller's fam is intentionally discarded in favor
 * of the canonical KSALT fam. */
static void *hmac_streebog256_shared_pso_for_variant_v2(
    struct gpu_metal_family *fam, uint8_t variant_bits)
{
    (void)fam;
    return metal_pso_for_variant_default(
        &metal_family_hmac_streebog256_ksalt, variant_bits);
}

/* D5b Wave 4 2026-05-16: HMAC-STREEBOG-512 SHARED v2 resolver. Same dual-
 * struct pattern as HMAC-STREEBOG-256 above at HASH_WORDS=16. Canonical
 * fam_idx = metal_family_hmac_streebog512_ksalt.fam_idx. Function body
 * deferred until after the canonical fam_struct is defined (same clang-
 * strict-static-redefinition reason as 256-bit sibling above). */
static void *hmac_streebog512_shared_pso_for_variant_v2(
    struct gpu_metal_family *fam, uint8_t variant_bits);

static struct gpu_metal_family metal_family_hmac_streebog512_ksalt = {
    .op                 = JOB_HMAC_STREEBOG512_KSALT,
    .name               = "hmac_streebog512_ksalt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = hmac_streebog512_shared_pso_for_variant_v2,  /* Wave 4 */
};

static struct gpu_metal_family metal_family_hmac_streebog512_kpass = {
    .op                 = JOB_HMAC_STREEBOG512_KPASS,
    .name               = "hmac_streebog512_kpass",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = hmac_streebog512_shared_pso_for_variant_v2,  /* Wave 4 */
};

/* Definition deferred from forward-decl above. */
static void *hmac_streebog512_shared_pso_for_variant_v2(
    struct gpu_metal_family *fam, uint8_t variant_bits)
{
    (void)fam;
    return metal_pso_for_variant_default(
        &metal_family_hmac_streebog512_ksalt, variant_bits);
}

static struct gpu_metal_family metal_family_phpbb3 = {
    .op                 = JOB_PHPBB3,
    .name               = "phpbb3",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

static struct gpu_metal_family metal_family_md5crypt = {
    .op                 = JOB_MD5CRYPT,
    .name               = "md5crypt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

/* D5b Wave 4 2026-05-16: SHA256CRYPT migrates to generic loader. Shares
 * core_str = metal_shacrypt_core_str with sha512crypt; differentiated by
 * base_macros HASH_WORDS=8 vs HASH_WORDS=16. Per
 * feedback_defines_via_build_opts.md the macros are passed via
 * preprocessorMacros (Metal honors them as -D flags), so the two
 * instantiations of the SAME source compile as DIFFERENT PSOs cached
 * separately in metal_family_libs[][] / metal_family_psos[][]. */
static struct gpu_metal_family metal_family_sha256crypt = {
    .op                 = JOB_SHA256CRYPT,
    .name               = "sha256crypt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 4 */
};

/* D5b Wave 4 2026-05-16: SHA512CRYPT migrates to generic loader. Shares
 * core_str = metal_shacrypt_core_str with sha256crypt; differentiated by
 * base_macros HASH_WORDS=16 + HASH_BLOCK_BYTES=128. SHA512CRYPTMD5 below
 * aliases this family's compiled PSO via a custom v2 resolver. */
static struct gpu_metal_family metal_family_sha512crypt = {
    .op                 = JOB_SHA512CRYPT,
    .name               = "sha512crypt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 4 */
};

/* D5b Wave 4 2026-05-16: SHA512CRYPTMD5 ALIAS v2 resolver. No own compiled
 * library/PSO -- routes through metal_pso_for_variant_default with the
 * sha512crypt family struct so the alias dispatches the SAME PSO that
 * sha512crypt compiled. The aliased PSOs carry no per-op state -- algo_mode
 * is host-set at dispatch time (params->algo_mode field), so the same PSO
 * serves both ops with different runtime behavior (algo_mode=0 vs algo_mode=1
 * triggering the kernel-side MD5-preprocess at the top of template_finalize).
 *
 * The sha512cryptmd5 family's own slot in metal_family_libs[][] /
 * metal_family_psos[][] stays NULL throughout the process lifetime --
 * sha512crypt.fam_idx is canonical. Matches the HMAC dual-struct pattern
 * above but with a separate sibling family (sha512crypt is not declared
 * "alias"; the alias-of-alias relationship is intentional asymmetry). */
static void *sha512cryptmd5_pso_for_variant_v2(
    struct gpu_metal_family *fam, uint8_t variant_bits)
{
    (void)fam;
    return metal_pso_for_variant_default(
        &metal_family_sha512crypt, variant_bits);
}

static struct gpu_metal_family metal_family_sha512cryptmd5 = {
    .op                 = JOB_SHA512CRYPTMD5,
    .name               = "sha512cryptmd5",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = sha512cryptmd5_pso_for_variant_v2,  /* Wave 4 */
};

static struct gpu_metal_family metal_family_descrypt = {
    .op                 = JOB_DESCRYPT,
    .name               = "descrypt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 3 */
};

/* D5b Wave 4 2026-05-16: BCRYPT migrates to generic loader. Per architect
 * memo §C Wave 4 + §E row 9, .dispatch_tg_size=8 declares the per-family
 * threadsPerThreadgroup override; the dispatch site reads fam->dispatch_tg_size
 * instead of the hardcoded `cache_gpu_op == JOB_BCRYPT` check, generalizing
 * the TG override for any future family that needs one. core_str +
 * base_macros (carrying GPU_TEMPLATE_HAS_LOCAL_BUFFER=1 +
 * GPU_TEMPLATE_LOCAL_BUFFER_PER_LANE=1024 + BCRYPT_WG_SIZE=8 +
 * HASH_WORDS=6 + HASH_BLOCK_BYTES=64 + BASE_ALGO=bcrypt) set in
 * metal_register_builtin_families below. */
static struct gpu_metal_family metal_family_bcrypt = {
    .op                 = JOB_BCRYPT,
    .name               = "bcrypt",
    .op_category        = GPU_CAT_MASK,
    .supported_variants = (uint8_t)(
                            VBIT(V_S)
                          | VBIT(V_S | V_R)
                          | VBIT(V_S | V_M)
                          | VBIT(V_S | V_R | V_M)),
    .pso_for_variant    = NULL,  /* legacy fn deleted in cleanup wave; v2 wrapper canonical */
    .pso_for_variant_v2 = metal_pso_for_variant_default,  /* Wave 4 */
    .dispatch_tg_size   = 8,                              /* Wave 4 */
};


/* Register all built-in families. Called from gpu_metal_init at end. */
static void metal_register_builtin_families(void)
{
    /* D5b Wave 1 2026-05-16: populate per-family generic-loader descriptors
     * BEFORE registration. base_macros uses an NSDictionary literal which
     * cannot live at file scope. The generic loader reads these via
     * fam->core_str / fam->base_macros when pso_for_variant_v2 is invoked.
     * Unmigrated families leave both NULL and dispatch falls back through
     * the legacy pso_for_variant slot.
     *
     * Non-ARC: the dictionary literal is autoreleased; CFBridgingRetain
     * gives us +1 retain count that we deliberately never release (the
     * dict lives for process lifetime, just like the metal_families[]
     * registry itself). This is symmetric to how mtl_lib_template_* are
     * never released. ARC-on builds (iMac Makefile) treat CFBridgingRetain
     * the same way. */
    metal_family_md5.core_str    = metal_md5_core_str;
    metal_family_md5.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"md5"
    });

    /* D5b Wave 2 2026-05-16: 28 unsalted no-quirk families migrated to the
     * generic loader path. Per-family core_str + base_macros initialization
     * mirrors the md5 canary above. HASH_WORDS / HASH_BLOCK_BYTES values
     * match the `#ifndef`-guarded defaults inside each <algo>_core_str
     * verbatim; base_macros makes them explicit at the registry site and
     * positions Wave 3 (salted) to reuse the same dict-init pattern.
     *
     * Each family's struct literal sets .pso_for_variant_v2 =
     * metal_pso_for_variant_default; the legacy .pso_for_variant slot is
     * retained as DEAD CODE migration-safety fallback per Wave 1 precedent.
     * Wave N+1 cleanup wave will delete the per-family loaders + lazy
     * creators + resolvers once Waves 3/4 also land. */
    metal_family_md4.core_str    = metal_md4_core_str;
    metal_family_md4.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"md4"
    });
    metal_family_md4utf16.core_str    = metal_md4utf16_core_str;
    metal_family_md4utf16.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"md4utf16"
    });
    metal_family_md5raw.core_str    = metal_md5raw_core_str;
    metal_family_md5raw.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"md5raw"
    });
    metal_family_sha1.core_str    = metal_sha1_core_str;
    metal_family_sha1.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @5,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha1"
    });
    metal_family_sha1raw.core_str    = metal_sha1raw_core_str;
    metal_family_sha1raw.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @5,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha1raw"
    });
    metal_family_sha1dru.core_str    = metal_sha1dru_core_str;
    metal_family_sha1dru.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @5,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha1dru"
    });
    metal_family_sha256.core_str    = metal_sha256_core_str;
    metal_family_sha256.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha256"
    });
    metal_family_sha256raw.core_str    = metal_sha256raw_core_str;
    metal_family_sha256raw.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha256raw"
    });
    metal_family_sha224.core_str    = metal_sha224_core_str;
    metal_family_sha224.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @7,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha224"
    });
    metal_family_sha512.core_str    = metal_sha512_core_str;
    metal_family_sha512.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"sha512"
    });
    metal_family_sha512raw.core_str    = metal_sha512raw_core_str;
    metal_family_sha512raw.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"sha512raw"
    });
    metal_family_sha384.core_str    = metal_sha384_core_str;
    metal_family_sha384.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @12,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"sha384"
    });
    metal_family_sha384raw.core_str    = metal_sha384raw_core_str;
    metal_family_sha384raw.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @12,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"sha384raw"
    });
    metal_family_ripemd160.core_str    = metal_ripemd160_core_str;
    metal_family_ripemd160.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @5,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"ripemd160"
    });
    metal_family_ripemd320.core_str    = metal_ripemd320_core_str;
    metal_family_ripemd320.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @10,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"ripemd320"
    });
    metal_family_blake2s256.core_str    = metal_blake2s256_core_str;
    metal_family_blake2s256.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"blake2s256"
    });
    metal_family_blake2b256.core_str    = metal_blake2b256_core_str;
    metal_family_blake2b256.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"blake2b256"
    });
    metal_family_blake2b512.core_str    = metal_blake2b512_core_str;
    metal_family_blake2b512.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"blake2b512"
    });
    /* Keccak / SHA-3 sponge families: HASH_BLOCK_BYTES is the SPONGE RATE
     * (output block size) per rate-dependent Keccak parameterization, NOT
     * the compression-input block size. Values verified verbatim against
     * each metal_<algo>_core_str.h #define defaults 2026-05-16. */
    metal_family_keccak224.core_str    = metal_keccak224_core_str;
    metal_family_keccak224.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @7,
        @"HASH_BLOCK_BYTES": @144,
        @"BASE_ALGO":        @"keccak224"
    });
    metal_family_keccak256.core_str    = metal_keccak256_core_str;
    metal_family_keccak256.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @136,
        @"BASE_ALGO":        @"keccak256"
    });
    metal_family_keccak384.core_str    = metal_keccak384_core_str;
    metal_family_keccak384.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @12,
        @"HASH_BLOCK_BYTES": @104,
        @"BASE_ALGO":        @"keccak384"
    });
    metal_family_keccak512.core_str    = metal_keccak512_core_str;
    metal_family_keccak512.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @72,
        @"BASE_ALGO":        @"keccak512"
    });
    metal_family_sha3_224.core_str    = metal_sha3_224_core_str;
    metal_family_sha3_224.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @7,
        @"HASH_BLOCK_BYTES": @144,
        @"BASE_ALGO":        @"sha3_224"
    });
    metal_family_sha3_256.core_str    = metal_sha3_256_core_str;
    metal_family_sha3_256.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @136,
        @"BASE_ALGO":        @"sha3_256"
    });
    metal_family_sha3_384.core_str    = metal_sha3_384_core_str;
    metal_family_sha3_384.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @12,
        @"HASH_BLOCK_BYTES": @104,
        @"BASE_ALGO":        @"sha3_384"
    });
    metal_family_sha3_512.core_str    = metal_sha3_512_core_str;
    metal_family_sha3_512.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @72,
        @"BASE_ALGO":        @"sha3_512"
    });
    metal_family_streebog256.core_str    = metal_streebog256_core_str;
    metal_family_streebog256.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"streebog256"
    });
    metal_family_streebog512.core_str    = metal_streebog512_core_str;
    metal_family_streebog512.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"streebog512"
    });

    /* D5b Wave 3 2026-05-16: 14 salted no-quirk families migrated to the
     * generic loader path. md5salt is PARTIAL -- its V_S|V_R PRESALT fold
     * stays on the custom legacy resolver via md5salt_pso_for_variant_v2;
     * the other 3 variants and all 13 sibling families route through
     * metal_pso_for_variant_default. HASH_WORDS / HASH_BLOCK_BYTES values
     * verified verbatim against each metal_<algo>_core_str.h `#ifndef`
     * defaults pre-flight (4/64, 5/64, 7/64, 8/64, 12/128, 16/128 per
     * digest width). CFBridgingRetain per feedback_metal_arc_mismatch_imac_dev1.md
     * works under both ARC (iMac Makefile) and non-ARC (dev1 Makefile). */
    metal_family_md5salt.core_str    = metal_md5salt_core_str;
    metal_family_md5salt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"md5"
    });
    metal_family_md5passsalt.core_str    = metal_md5passsalt_core_str;
    metal_family_md5passsalt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"md5"
    });
    metal_family_md5saltpass.core_str    = metal_md5saltpass_core_str;
    metal_family_md5saltpass.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"md5"
    });
    metal_family_sha1passsalt.core_str    = metal_sha1passsalt_core_str;
    metal_family_sha1passsalt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @5,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha1"
    });
    metal_family_sha1saltpass.core_str    = metal_sha1saltpass_core_str;
    metal_family_sha1saltpass.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @5,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha1"
    });
    metal_family_sha224saltpass.core_str    = metal_sha224saltpass_core_str;
    metal_family_sha224saltpass.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @7,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha224"
    });
    metal_family_sha256passsalt.core_str    = metal_sha256passsalt_core_str;
    metal_family_sha256passsalt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha256"
    });
    metal_family_sha256saltpass.core_str    = metal_sha256saltpass_core_str;
    metal_family_sha256saltpass.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha256"
    });
    metal_family_sha384saltpass.core_str    = metal_sha384saltpass_core_str;
    metal_family_sha384saltpass.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @12,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"sha512"
    });
    metal_family_sha512passsalt.core_str    = metal_sha512passsalt_core_str;
    metal_family_sha512passsalt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"sha512"
    });
    metal_family_sha512saltpass.core_str    = metal_sha512saltpass_core_str;
    metal_family_sha512saltpass.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"sha512"
    });
    metal_family_phpbb3.core_str    = metal_phpbb3_core_str;
    metal_family_phpbb3.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"phpbb3"
    });
    metal_family_md5crypt.core_str    = metal_md5crypt_core_str;
    metal_family_md5crypt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"md5crypt"
    });
    metal_family_descrypt.core_str    = metal_descrypt_core_str;
    metal_family_descrypt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @4,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"descrypt"
    });

    /* D5b Wave 4 2026-05-16: 9 quirky families migrated to the generic
     * loader path. Final new-migration wave -- 52 of 52 Metal families now
     * route through metal_pso_for_variant_default (the canonical resolver)
     * or a Wave 4 v2 wrapper (hmac_streebog256/512 dual-struct,
     * sha512cryptmd5 alias). Per architect memo §C Wave 4 + §F risks 3+4:
     *   - hmac_blake2s: single-struct family, default v2 resolver
     *   - hmac_streebog256_{ksalt,kpass}: DUAL-STRUCT both pointing at
     *     shared resolver -> canonical fam_idx (KSALT) -> one cached PSO
     *   - hmac_streebog512_{ksalt,kpass}: same dual-struct pattern
     *   - sha256crypt: shacrypt core at HASH_WORDS=8
     *   - sha512crypt: shacrypt core at HASH_WORDS=16 (DIFFERENT PSO from
     *     sha256crypt -- cache key disambiguated by HASH_WORDS+HASH_BLOCK_BYTES
     *     macros per feedback_defines_via_build_opts.md)
     *   - sha512cryptmd5: ALIAS -- v2 wrapper routes through sha512crypt
     *     fam_idx; no own compiled PSO
     *   - bcrypt: default v2 resolver + .dispatch_tg_size=8 override (replaces
     *     hardcoded cache_gpu_op==JOB_BCRYPT check at dispatch site)
     *
     * HASH_WORDS / HASH_BLOCK_BYTES values verified verbatim against each
     * metal_<algo>_core_str.h #ifndef defaults pre-flight. BCRYPT carries
     * additional macros (HAS_LOCAL_BUFFER + LOCAL_BUFFER_PER_LANE +
     * BCRYPT_WG_SIZE) into base_macros so the generic loader gates the
     * threadgroup-shared sbox_pool + 8-arg template_finalize variant in
     * metal_template.metal (Phase 2d.9b scaffold extension). */

    metal_family_hmac_blake2s.core_str    = metal_hmac_blake2s_core_str;
    metal_family_hmac_blake2s.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"hmac_blake2s"
    });
    /* hmac_streebog256: BOTH ksalt and kpass entries need core_str +
     * base_macros populated for the eager-probe loop's lookup of the
     * caller's fam.  Even though the shared v2 resolver discards the
     * caller's fam (uses canonical KSALT instead), defensive completeness
     * keeps the registry consistent in case a future code path reads
     * fam->core_str directly. Both entries get identical core_str +
     * base_macros (same kernel program). */
    metal_family_hmac_streebog256_ksalt.core_str    = metal_hmac_streebog256_core_str;
    metal_family_hmac_streebog256_ksalt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"hmac_streebog256"
    });
    metal_family_hmac_streebog256_kpass.core_str    = metal_hmac_streebog256_core_str;
    metal_family_hmac_streebog256_kpass.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"hmac_streebog256"
    });
    metal_family_hmac_streebog512_ksalt.core_str    = metal_hmac_streebog512_core_str;
    metal_family_hmac_streebog512_ksalt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"hmac_streebog512"
    });
    metal_family_hmac_streebog512_kpass.core_str    = metal_hmac_streebog512_core_str;
    metal_family_hmac_streebog512_kpass.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"hmac_streebog512"
    });
    /* sha256crypt + sha512crypt: SHARE core_str = metal_shacrypt_core_str.
     * DIFFERENT base_macros (HASH_WORDS=8 vs HASH_WORDS=16); the generic
     * loader's preprocessorMacros pass these as -D flags so the shared
     * source compiles as TWO distinct PSOs cached separately in
     * metal_family_psos[][]. */
    metal_family_sha256crypt.core_str    = metal_shacrypt_core_str;
    metal_family_sha256crypt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @8,
        @"HASH_BLOCK_BYTES": @64,
        @"BASE_ALGO":        @"sha256crypt"
    });
    metal_family_sha512crypt.core_str    = metal_shacrypt_core_str;
    metal_family_sha512crypt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"sha512crypt"
    });
    /* sha512cryptmd5: ALIAS -- the v2 resolver (sha512cryptmd5_pso_for_variant_v2)
     * discards the caller's fam and routes through sha512crypt. Still set
     * core_str + base_macros here defensively (in case a future code path
     * reads fam->core_str without going through the resolver), but the
     * generic loader will NEVER be called with the sha512cryptmd5 fam --
     * its libs/psos slots in the parallel arrays stay NULL forever. */
    metal_family_sha512cryptmd5.core_str    = metal_shacrypt_core_str;
    metal_family_sha512cryptmd5.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":       @16,
        @"HASH_BLOCK_BYTES": @128,
        @"BASE_ALGO":        @"sha512crypt"  /* same as sha512crypt -- aliased */
    });
    /* bcrypt: HAS_LOCAL_BUFFER scaffold + per-lane buffer size + WG=8 +
     * 6-word digest emit. dispatch_tg_size=8 is set in the struct literal
     * above (not in base_macros -- it drives the host's MTLSize, not the
     * kernel's preprocessor). */
    metal_family_bcrypt.core_str    = metal_bcrypt_core_str;
    metal_family_bcrypt.base_macros = (void *)CFBridgingRetain(@{
        @"HASH_WORDS":                        @6,
        @"HASH_BLOCK_BYTES":                  @64,
        @"BASE_ALGO":                         @"bcrypt",
        @"GPU_TEMPLATE_HAS_LOCAL_BUFFER":     @1,
        @"GPU_TEMPLATE_LOCAL_BUFFER_PER_LANE": @1024,
        @"BCRYPT_WG_SIZE":                    @8
    });

    gpu_metal_register_family(&metal_family_md5);
    gpu_metal_register_family(&metal_family_md5salt);
    gpu_metal_register_family(&metal_family_md4);
    gpu_metal_register_family(&metal_family_md4utf16);
    gpu_metal_register_family(&metal_family_md5raw);
    gpu_metal_register_family(&metal_family_md5passsalt);
    gpu_metal_register_family(&metal_family_md5saltpass);
    gpu_metal_register_family(&metal_family_sha1);
    gpu_metal_register_family(&metal_family_sha1raw);
    gpu_metal_register_family(&metal_family_sha1dru);
    gpu_metal_register_family(&metal_family_sha1passsalt);
    gpu_metal_register_family(&metal_family_sha1saltpass);
    gpu_metal_register_family(&metal_family_sha256);
    gpu_metal_register_family(&metal_family_sha256raw);
    gpu_metal_register_family(&metal_family_sha224);
    gpu_metal_register_family(&metal_family_sha256passsalt);
    gpu_metal_register_family(&metal_family_sha256saltpass);
    gpu_metal_register_family(&metal_family_sha224saltpass);
    gpu_metal_register_family(&metal_family_sha512);
    gpu_metal_register_family(&metal_family_sha512raw);
    gpu_metal_register_family(&metal_family_sha384);
    gpu_metal_register_family(&metal_family_sha384raw);
    gpu_metal_register_family(&metal_family_sha512passsalt);
    gpu_metal_register_family(&metal_family_sha512saltpass);
    gpu_metal_register_family(&metal_family_sha384saltpass);
    gpu_metal_register_family(&metal_family_ripemd160);
    gpu_metal_register_family(&metal_family_ripemd320);
    gpu_metal_register_family(&metal_family_blake2s256);
    gpu_metal_register_family(&metal_family_blake2b256);
    gpu_metal_register_family(&metal_family_blake2b512);
    gpu_metal_register_family(&metal_family_keccak256);
    gpu_metal_register_family(&metal_family_keccak224);
    gpu_metal_register_family(&metal_family_keccak384);
    gpu_metal_register_family(&metal_family_keccak512);
    gpu_metal_register_family(&metal_family_sha3_224);
    gpu_metal_register_family(&metal_family_sha3_256);
    gpu_metal_register_family(&metal_family_sha3_384);
    gpu_metal_register_family(&metal_family_sha3_512);
    gpu_metal_register_family(&metal_family_streebog256);
    gpu_metal_register_family(&metal_family_streebog512);
    /* Phase 2d.7d HMAC siblings (3 carrier kernels, 5 ops via dual-op
     * struct entries for the streebog HMAC carriers). */
    gpu_metal_register_family(&metal_family_hmac_blake2s);
    gpu_metal_register_family(&metal_family_hmac_streebog256_ksalt);
    gpu_metal_register_family(&metal_family_hmac_streebog256_kpass);
    gpu_metal_register_family(&metal_family_hmac_streebog512_ksalt);
    gpu_metal_register_family(&metal_family_hmac_streebog512_kpass);
    /* Phase 2d.8a PHPBB3 + MD5CRYPT (2 ops, 2 carrier kernels). Both
     * salted-only iterated-MD5 algorithms with internal iter loops INSIDE
     * template_finalize. PHPBB3 is canary; MD5CRYPT is Phase 1 of the
     * Unix-crypt ladder. Registry count: 45 -> 47 (cap=64, headroom 17). */
    gpu_metal_register_family(&metal_family_phpbb3);
    gpu_metal_register_family(&metal_family_md5crypt);
    /* Phase 2d.8b SHACRYPT triple (3 ops, 1 SHARED carrier kernel, 2
     * compiled PSO sets). SHA256CRYPT (op=512, HASH_WORDS=8) is canary;
     * SHA512CRYPT (op=513, HASH_WORDS=16) instantiates the SAME source
     * with HASH_WORDS=16 + HASH_BLOCK_BYTES=128; SHA512CRYPTMD5 (op=538,
     * HASH_WORDS=16, algo_mode=1) ALIASES the SHA512CRYPT compiled PSOs
     * (sha512cryptmd5_pso_for_variant returns the same statics as
     * sha512crypt). Phases 2/3/4 of the Unix-crypt ladder on Metal --
     * SHA512CRYPTMD5 is the FINAL phase. Registry count: 47 -> 50
     * (cap=64, headroom 14). */
    gpu_metal_register_family(&metal_family_sha256crypt);
    gpu_metal_register_family(&metal_family_sha512crypt);
    gpu_metal_register_family(&metal_family_sha512cryptmd5);
    /* Phase 2d.9a DESCRYPT (1 op, 1 hand-ported carrier kernel). Single
     * algo_mode=7 salted-only DES (25-iter Feistel inside template_finalize).
     * Hand-port (cl2metal.py UNSUITABLE per architect Task #293). LAST
     * Unix-crypt op to migrate from CPU-only to Metal. Registry count:
     * 50 -> 51 (cap=64, headroom 13). */
    gpu_metal_register_family(&metal_family_descrypt);
    /* Phase 2d.9b BCRYPT (1 op, 1 hand-ported carrier kernel). Single
     * algo_mode=8 salted-only Eksblowfish (2^cost iter loop inside
     * template_finalize). Hand-port (cl2metal.py UNSUITABLE per architect
     * Task #293). FIRST Metal family with GPU_TEMPLATE_HAS_LOCAL_BUFFER
     * scaffold extension (32 KB threadgroup-shared S-boxes) + per-op
     * threadsPerThreadgroup=8 dispatch-site override + HASH_WORDS=6 emit.
     * Registry count: 51 -> 52 (cap=64, headroom 12). FINAL Phase 2d
     * sub-phase. */
    gpu_metal_register_family(&metal_family_bcrypt);

    /* D5b Wave 1 2026-05-16: post-registration walk to populate fam_idx
     * for every registered family. The generic loader / lazy / resolver
     * (metal_load_library_generic / metal_pso_lazy_generic /
     * metal_pso_for_variant_default) index the hidden parallel arrays
     * metal_family_libs[][] / metal_family_psos[][] via fam_idx. Unmigrated
     * families also get a valid fam_idx; they just don't use it because
     * their pso_for_variant_v2 is NULL and dispatch falls back to legacy. */
    metal_assign_fam_indices();
}

/* Phase D5a (Task #281+#282) 2026-05-16: per-(family,variant) admission
 * bitmap. metal_admit_mask[i] holds an 8-bit mask of admitted variant_bits
 * for family at index i (parallel to metal_families[]). A 1-bit at position
 * v means pso_for_variant(v) returned non-NULL at gpu_metal_compile_families
 * time; a 0-bit means EITHER (a) v is not in the family's supported_variants
 * (so we never probed) OR (b) v was probed and pso_for_variant returned
 * NULL (the PSO compile failed -- capability missing for this device, e.g.
 * the Phase 2d.5 Ventura M2 Max SHA-2/512 compiler bug).
 *
 * Populated by gpu_metal_compile_families; queried by
 * gpu_metal_op_variant_admitted (called from dispatch / future gpu_ops[]
 * pruning in mdxfind.c). */
static uint8_t metal_admit_mask[METAL_FAMILY_CAP];
static int     metal_admitted_family_count = 0;
static int     metal_compile_families_ran  = 0;

void gpu_metal_compile_families(unsigned int fam_mask)
{
    if (!metal_ready) return;
    /* Phase 2c tier-2 dev1 mitigation: eagerly create all PSO variants
     * on the MAIN thread at init time. macOS 26.3 + Apple Silicon G13
     * crashes inside AGX::UserShaderFactory::loadDynamicLibrariesForFunctions
     * when the salt-variant PSO is finalized on the gpujob worker thread
     * (see crash log mdxfind-2026-05-13-120230.ips). Pre-creating on the
     * main thread sidesteps the worker-thread context that triggers the
     * bug. Cost: ~5-10s one-time JIT at startup; Apple driver self-caches.
     * Lazy creators are idempotent (early return when PSO already created),
     * so the worker thread's first-dispatch path becomes a no-op.
     *
     * Phase 2d.2.1a: iterate registered families. Each family's resolver
     * lazy-creates the PSOs for its admissible variant_bits set. PRESALT
     * is folded inside the md5salt resolver (V_S|V_R arm); we do NOT
     * expose it as a separate variant_bits combo.
     *
     * Phase D5a (Task #281+#282) §6 Option B 2026-05-16: capability check
     * at INIT is the right boundary for PSO compile failures (per the
     * external-failures-are-fatal discipline §6 carve-out). Track per-
     * (family, variant) admission in metal_admit_mask[]; loud STDERR per
     * pruned combo and summary after the loop. Did NOT make the lazy
     * creators themselves fatal because they're called from BOTH this
     * init-time eager probe and the runtime dispatch -- the runtime fatal
     * lives at the dispatch site (gpu_metal_dispatch_md5_rules, work
     * item 4). */
    (void)fam_mask;

    /* Reset admission state. Safe across re-invocations (compile_families
     * is currently called once from gpu_metal_init's set_compact_table
     * path, but future multi-init use needs the reset). */
    for (int i = 0; i < METAL_FAMILY_CAP; i++) metal_admit_mask[i] = 0;
    metal_admitted_family_count = 0;
    metal_compile_families_ran  = 0;  /* set to 1 at end-of-function */

    int total_probed = 0;
    int total_admitted = 0;
    int total_pruned = 0;

    for (int i = 0; i < metal_family_count; i++) {
        struct gpu_metal_family *fam = metal_families[i];
        uint8_t sv = fam->supported_variants;
        uint8_t admit_for_fam = 0;
        /* Probe every (R, M, S) tuple admitted by supported_variants.
         * Bit-position v in sv encodes whether variant_bits == v is an
         * admissible combo for this family. */
        for (uint8_t v = 0; v < 8u; v++) {
            if (((sv >> v) & 1u) == 0) continue;
            total_probed++;
            /* D5b Wave 1: prefer v2 resolver when present (migrated family).
             * Unmigrated families have pso_for_variant_v2 == NULL and we
             * fall back to the legacy 1-arg pointer. */
            void *pso = fam->pso_for_variant_v2
                          ? fam->pso_for_variant_v2(fam, v)
                          : fam->pso_for_variant(v);
            if (pso == NULL) {
                /* Phase D5a §6 Option B: capability check -- the lazy
                 * creator already emitted its own "<family>: PSO create
                 * failed: <error>" diagnostic from inside the failing
                 * newComputePipelineStateWithFunction:error: callsite.
                 * Add a per-pruned-combo line so the operator sees the
                 * admission decision in one place. */
                fprintf(stderr,
                        "STDERR: GPU admission: family %s variant_bits=0x%x "
                        "NOT admitted (PSO compile failed at init)\n",
                        fam->name ? fam->name : "?", (unsigned)v);
                total_pruned++;
            } else {
                admit_for_fam |= (uint8_t)(1u << v);
                total_admitted++;
            }
        }
        metal_admit_mask[i] = admit_for_fam;
        if (admit_for_fam != 0) metal_admitted_family_count++;
    }

    /* Summary line: cohort-level admission state at end of init.
     * Debug-gated per 2026-05-17 user directive — chatter under success
     * conditions. The per-pruned-combo line above (3274) stays
     * unconditional because it only fires on real prune events that the
     * operator must see. */
    GPU_DEBUG_FPRINTF(stderr,
            "STDERR: GPU admission: %d families admitted, %d families CPU-only "
            "(probed=%d admitted-variants=%d pruned-variants=%d)\n",
            metal_admitted_family_count,
            metal_family_count - metal_admitted_family_count,
            total_probed, total_admitted, total_pruned);

    metal_compile_families_ran = 1;
}

int gpu_metal_op_variant_admitted(int op, uint8_t variant_bits_mask)
{
    /* Pre-init / pre-compile-families: treat as not-admitted. Callers
     * that want a definitive answer must run after gpu_metal_compile_-
     * families. This is a safe default: the dispatch path can route to
     * CPU when admission is unknown. */
    if (!metal_compile_families_ran) return 0;

    for (int i = 0; i < metal_family_count; i++) {
        if (metal_families[i]->op != op) continue;
        /* At least one bit in variant_bits_mask AND admit_mask -> admitted. */
        return (metal_admit_mask[i] & variant_bits_mask) ? 1 : 0;
    }
    return 0;  /* op not registered */
}

int gpu_metal_admitted_family_count(void)
{
    return metal_admitted_family_count;
}

/* Phase 1a sub-phase 1a.1b (2026-05-20): Metal twin of
 * gpu_opencl_kernel_a_proto_enabled (declared in gpu/gpu_opencl.h, defined
 * in gpu/gpu_opencl.c:9847-9880).
 *
 * This sub-phase Phase 0 ships the symbol so mdxfind.c links cleanly on
 * Metal builds (the unconditional call at mdxfind.c:11448-11450 and
 * mdxfind.c:38075-38081 landed in rev 1.488 / 2026-05-19 under the false
 * assumption that the symbol was backend-shared; it is OpenCL-only). The
 * call site is now #if defined backend-dispatched.
 *
 * Phase 1 of 1a.1b ships the translator-driven kernel source
 * (metal_kernel_a_rules.metal + _str.h) but DOES NOT YET ship the host-
 * side Metal dispatcher. Until the dispatcher (a 700-line port of
 * gpu_opencl.c:9974-10680) lands in a follow-up sub-phase, this function
 * checks the env flags and prints a one-time stderr advisory if the
 * operator requested A1 on Metal — then RETURNS 0 to let the existing
 * MDXFIND_KERNEL_B_PROTO / production wiring take over (per the spec's
 * "correctness ship" directive — do not crash, do not silently no-op).
 *
 * The implementation otherwise mirrors gpu_opencl_kernel_a_proto_enabled
 * for future drop-in once the Metal dispatcher exists: cached decision,
 * variant validator, stderr advisory shape. The only divergence is the
 * unconditional return 0 at the bottom; once the dispatcher is wired the
 * `return 0` is flipped to `cached_decision = 1; return 1;`. */
/* Shared cache used by gpu_metal_kernel_a_proto_enabled() and the
 * sibling gpu_metal_kernel_a_active_variant(). Single env-var read.
 *
 *   _cached_metal_a_variant == -2 -> not yet computed
 *   _cached_metal_a_variant ==  0 -> MDXFIND_KERNEL_A_PROTO unset or != 1
 *   _cached_metal_a_variant ==  N -> proto enabled, variant N (1..4)
 *
 * Sub-phase 1a.2 (2026-05-21) extends recognized variants from {1} to
 * {1, 2}. Variant 2 routes JOB_MD5MD5SALT (canary op) to the new
 * gpu_metal_kernelA_masks_dispatch (cand_masks_phase0 PSO). */
static int _cached_metal_a_variant = -2;

static void gpu_metal_kernel_a_variant_compute(void)
{
    if (_cached_metal_a_variant != -2) return;

    const char *e_a   = getenv("MDXFIND_KERNEL_A_PROTO");
    const char *e_var = getenv("MDXFIND_KERNEL_A_VARIANT");

    if (!e_a || strcmp(e_a, "1") != 0) {
        _cached_metal_a_variant = 0;
        return;
    }
    int variant = 1;
    if (e_var && *e_var) {
        variant = atoi(e_var);
    }
    if (variant != 1 && variant != 2 && variant != 3 && variant != 4) {
        fprintf(stderr,
            "Metal: MDXFIND_KERNEL_A_PROTO=1 with MDXFIND_KERNEL_A_VARIANT=%d "
            "is not implemented yet (sub-phase 1a.4 ships VARIANT=1/2/3/4 "
            "(rules/masks/rules+masks/bruteforce)); falling through to "
            "existing wiring.\n",
            variant);
        _cached_metal_a_variant = 0;
        return;
    }
    fprintf(stderr,
        "Metal: Phase 1a kernel A%d (%s) production path enabled "
        "via MDXFIND_KERNEL_A_PROTO=1 (variant=%d).\n",
        variant,
        variant == 1 ? "rules-only" :
        variant == 2 ? "masks-only" :
        variant == 3 ? "rules+masks" : "bruteforce",
        variant);
    _cached_metal_a_variant = variant;
}

int gpu_metal_kernel_a_proto_enabled(void)
{
    gpu_metal_kernel_a_variant_compute();
    return _cached_metal_a_variant > 0 ? 1 : 0;
}

/* gpu_metal_kernel_a_active_variant -- Phase 1a sub-phase 1a.2 sibling
 * per D9.4.a. Returns the cached variant int:
 *   0 -> proto disabled
 *   1 -> VARIANT=1 (rules-only, cand_rules_phase0)
 *   2 -> VARIANT=2 (masks-only, cand_masks_phase0)
 *   3 -> VARIANT=3 (rules+masks, cand_rules_masks_phase0; sub-phase 1a.3)
 *   4 -> reserved (1a.4 BF) */
int gpu_metal_kernel_a_active_variant(void)
{
    gpu_metal_kernel_a_variant_compute();
    return _cached_metal_a_variant > 0 ? _cached_metal_a_variant : 0;
}

/* Phase 4 sub-phase 4a.2 (2026-05-21): D13.3.a opt-out accessor for the
 * hx codegen production path on Metal. Mirrors gpu_opencl_hx_codegen_-
 * enabled (gpu/gpu_opencl.c rev 1.190 line 10598). Returns 1 when the
 * env var is unset or set to anything other than the literal "0";
 * returns 0 when the env var equals the literal "0" exactly.
 *
 * Centralized predicate so the future Metal kernel-B dispatcher and any
 * downstream gpujob_metal.m gate share one definition. Sunsets in
 * Phase 5 alongside the OpenCL twin. */
int gpu_metal_hx_codegen_enabled(void)
{
    const char *e = getenv("MDXFIND_HX_CODEGEN");
    if (e == NULL) return 1;
    if (e[0] == '0' && e[1] == '\0') return 0;
    return 1;
}

/* Phase 4 sub-phase 4a.2 (2026-05-21): hx codegen kernel B lazy-init
 * helper for JOB_MD5MD5SALT Metal production dispatch. Metal twin of
 * gpu_opencl_kernelb_codegen_kernel (gpu/gpu_opencl.c rev 1.190 line
 * 12022).
 *
 * On first call per process: runs hx_specs_lookup(JOB_MD5MD5SALT),
 * builds an hx_specialization for the e347 chain, invokes hx_emit_-
 * kernel with HX_BACKEND_METAL, dumps the emitted Metal source to
 * /tmp/mdxfind-codegen-e347-<pid>-dev<N>.metal for post-mortem
 * inspection (non-fatal if the dump fails), JIT-compiles via
 * gpu_metal_jit_compile_source_with_common_keep (which concatenates
 * metal_common_str + emitted source, compiles, looks up entry-point
 * `kernelb_hx_e347_phase0`, creates the MTLComputePipelineState), and
 * caches the bridge-retained MTLLibrary + PSO in module-local statics
 * (mtl_lib_kernelb_codegen / mtl_pso_kernelb_codegen).
 *
 * On subsequent calls returns the cached PSO immediately (fast path).
 * The codegen-side memoization saves walker + dump + Metal newLibrary
 * cost; Apple's MTLCompiler service does its own opaque cross-process
 * PSO caching transparently in the driver, so no explicit cache key
 * extension is needed on the Metal side (per spec §2 note).
 *
 * Returns the PSO as a non-NULL opaque void *; the caller (future
 * Metal kernel-B dispatcher) casts back to id<MTLComputePipelineState>
 * via __bridge before setting on the encoder.
 *
 * FATAL on any failure (hx spec lookup outlier/compile_failed/null
 * program, walker rc != 0, JIT compile, PSO creation) per feedback_-
 * external_failures_are_fatal.md. Diagnostic includes file:line,
 * hostname, dev_idx, error string where available, and the dumped
 * source path so post-mortem inspection is possible on the unhappy
 * path.
 *
 * NEVER silent fallback to a broken kernel -- on Metal there is no
 * legacy kernel-B path at all today, so the only valid behaviors are
 * (a) cached fast-path or (b) FATAL on any failure. The
 * MDXFIND_HX_CODEGEN=0 opt-out cannot route to a broken Metal kernel
 * (none exists); on Metal the opt-out manifests as the future Metal
 * kernel-B dispatcher refusing to dispatch and FATAL-erroring with a
 * specific message ("opt-out (MDXFIND_HX_CODEGEN=0) is not supported
 * on Metal -- Metal has no legacy kernel B path"). That FATAL belongs
 * inside the dispatcher's else branch when it ships in a follow-on
 * sub-phase; the lazy-init helper here is only called from the
 * codegen-active arm. */
/* Sub-phase 5a.3 (2026-05-22): per-JOB parameterized version of
 * gpu_metal_kernelb_codegen_pso. Linear scan over mtl_codegen_slots[];
 * lazy-init on miss; FATAL on cap exhaustion. Mirror of OpenCL twin
 * gpu_opencl_kernelb_codegen_kernel_for (gpu/gpu_opencl.c rev 1.192
 * line ~12302).
 *
 * job_enum selects the kernel to build:
 *   JOB_MD5MD5SALT  (347): e347 hand-tuned chain; entry point
 *                          kernelb_hx_e347_phase0; cached as slot 0
 *                          (and ALSO into the legacy single-PSO statics
 *                          for back-compat with code that still calls
 *                          the 0-arity variant).
 *   JOB_SHA1MD5PASS (161): family MAKE_MD5PASS; entry point
 *                          kernelb_hx_codegen_phase0; family-emit shape;
 *                          cached as slot 1+.
 *   Other family eN: same family-emit shape; entry point
 *                          kernelb_hx_codegen_phase0; allocated to next
 *                          free slot.
 *
 * Returns the cached PSO as opaque void*; the caller casts back to
 * id<MTLComputePipelineState> via __bridge before setting on the encoder.
 * FATAL on any failure per feedback_external_failures_are_fatal.md. */
void *gpu_metal_kernelb_codegen_pso_for(int dev_idx, int job_enum)
{
    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname) - 1);

    if (dev_idx != 0) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen (Metal): dev_idx=%d out of "
            "range (Phase 1 Metal is single-device) on %s\n",
            __FILE__, __LINE__, dev_idx, hostname);
        exit(1);
    }
    if (job_enum <= 0) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen (Metal): job_enum=%d invalid on %s\n",
            __FILE__, __LINE__, job_enum, hostname);
        exit(1);
    }

    /* Cache hit? Linear scan; SLOT_COUNT==16 is tiny. */
    for (int s = 0; s < MTL_CODEGEN_SLOT_COUNT; s++) {
        if (mtl_codegen_slots[s].job_enum == job_enum &&
            mtl_codegen_slots[s].pso_void != NULL) {
            return mtl_codegen_slots[s].pso_void;
        }
    }

    /* Lookup spec entry for this JOB. */
    const struct hx_spec_entry *entry = hx_specs_lookup(job_enum);
    if (!entry || entry->is_outlier || entry->compile_failed
        || !entry->program) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen e%d (Metal) lookup failed on %s "
            "dev[%d] (entry=%p outlier=%d compile_failed=%d program=%p)\n",
            __FILE__, __LINE__, job_enum, hostname, dev_idx,
            (const void*)entry,
            entry ? (int)entry->is_outlier : -1,
            entry ? (int)entry->compile_failed : -1,
            entry ? (const void*)entry->program : NULL);
        exit(1);
    }

    /* Build specialization. e347 uses the original tp0 spec; family
     * MAKE_MD5PASS members are unsalted single-emit; both cases set
     * iter_count=1, has_rules=1, no masks/BF. The salt_minlen/maxlen +
     * salt_count_regime fields are descriptive only for the emitter (no
     * functional effect on the family kernel which ignores salts). */
    struct hx_specialization spec;
    memset(&spec, 0, sizeof(spec));
    spec.iter_count_if_fixed = 1;
    spec.has_rules           = 1;
    spec.has_masks           = 0;
    spec.has_bf              = 0;
    spec.salt_minlen         = 1;
    spec.salt_maxlen         = 64;
    spec.salt_count_regime   = HX_SALT_BATCH_64;
    spec.iter_shape          = HX_ITER_NONE;
    spec.digest_endianness   = HX_DIGEST_LE;
    spec.emit_width          = 16;

    char *src = NULL;
    size_t src_cap = 0;
    int rc = hx_emit_kernel(entry->program, &spec, HX_BACKEND_METAL,
                            entry, &src, &src_cap);
    if (rc != 0 || !src) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen e%d (Metal) walker rc=%d on %s "
            "dev[%d] (src=%p cap=%zu)\n",
            __FILE__, __LINE__, job_enum, rc, hostname, dev_idx,
            (void*)src, src_cap);
        if (src) free(src);
        exit(1);
    }

    /* Source dump (post-mortem). */
    char dump_path[256];
    snprintf(dump_path, sizeof(dump_path),
             "/tmp/mdxfind-codegen-e%d-%d-dev%d.metal",
             job_enum, (int)getpid(), dev_idx);
    {
        FILE *df = fopen(dump_path, "w");
        if (df) {
            fwrite(src, 1, strlen(src), df);
            fclose(df);
        } else {
            fprintf(stderr,
                "hx codegen (Metal): source dump to %s failed on %s "
                "dev[%d] (errno=%d %s) -- proceeding without dump\n",
                dump_path, hostname, dev_idx, errno, strerror(errno));
        }
    }

    /* Entry-point name selection:
     *   JOB_MD5MD5SALT (e347) uses the original hand-named entry
     *     kernelb_hx_e347_phase0 (back-compat with 4a.2/4a.2b code).
     *   FAMILY_MD5PASS members use the family emit entry name
     *     kernelb_hx_codegen_phase0 (parallel to OpenCL 5a.2 naming). */
    const char *entry_point_name = (job_enum == JOB_MD5MD5SALT)
        ? "kernelb_hx_e347_phase0"
        : "kernelb_hx_codegen_phase0";

    void *lib_out = NULL;
    void *pso_out = NULL;
    int jit_rc = gpu_metal_jit_compile_source_with_common_keep(
        dev_idx, src, entry_point_name, &lib_out, &pso_out);
    if (jit_rc != 0 || lib_out == NULL || pso_out == NULL) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen e%d (Metal) JIT failed on %s "
            "dev[%d] (rc=%d lib=%p pso=%p) -- source dumped to %s\n",
            __FILE__, __LINE__, job_enum, hostname, dev_idx,
            jit_rc, lib_out, pso_out, dump_path);
        free(src);
        exit(1);
    }

    /* Insert into first free slot. */
    int placed = -1;
    for (int s = 0; s < MTL_CODEGEN_SLOT_COUNT; s++) {
        if (mtl_codegen_slots[s].job_enum == 0 &&
            mtl_codegen_slots[s].pso_void == NULL) {
            mtl_codegen_slots[s].job_enum = job_enum;
            mtl_codegen_slots[s].lib_void = lib_out;
            mtl_codegen_slots[s].pso_void = pso_out;
            placed = s;
            break;
        }
    }
    if (placed < 0) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen e%d (Metal): mtl_codegen_slots[] "
            "full on %s (cap=%d). Bump MTL_CODEGEN_SLOT_COUNT in "
            "gpu_metal.m.\n",
            __FILE__, __LINE__, job_enum, hostname,
            MTL_CODEGEN_SLOT_COUNT);
        free(src);
        exit(1);
    }

    /* Mirror slot 0 (e347) into the legacy single-PSO statics so the
     * 4a.2 0-arity API (gpu_metal_kernelb_codegen_pso) still hands back
     * the right PSO for code paths that have not yet migrated to the
     * _for() variant. Other JOBs do not back-populate (they would
     * clobber the e347 cache; the 0-arity API is e347-only by design). */
    if (job_enum == JOB_MD5MD5SALT) {
        mtl_lib_kernelb_codegen = lib_out;
        mtl_pso_kernelb_codegen = pso_out;
    }

    fprintf(stderr,
        "Metal GPU[%d]: hx codegen e%d kernel B PSO created (entry=%s, "
        "source %zu bytes, slot=%d, dump=%s) on %s\n",
        dev_idx, job_enum, entry_point_name, strlen(src), placed,
        dump_path, hostname);

    free(src);
    return pso_out;
}

void *gpu_metal_kernelb_codegen_pso(int dev_idx)
{
    /* Sub-phase 5a.3 (2026-05-22): back-compat shim. Existing 4a.2/4a.2b
     * callers (gpu_metal_kernelb_dispatch_proto + the harness branch
     * still pass dev_idx only) get the e347 PSO by default; new family
     * callers use gpu_metal_kernelb_codegen_pso_for(dev_idx, job_enum).
     *
     * Preserve the original fast-path single-static lookup so a hot loop
     * of the OpenCL-twin kernel-B production dispatcher does not pay the
     * linear-scan cost on every dispatch (the per-JOB _for variant pays
     * it but only once per JOB per process; this 0-arity shim is the
     * hot e347 path). */
    if (mtl_pso_kernelb_codegen != NULL) return mtl_pso_kernelb_codegen;

    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname) - 1);

    if (dev_idx != 0) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen e347 (Metal): dev_idx=%d out of "
            "range (Phase 1 Metal is single-device) on %s\n",
            __FILE__, __LINE__, dev_idx, hostname);
        exit(1);
    }

    const struct hx_spec_entry *entry = hx_specs_lookup(JOB_MD5MD5SALT);
    if (!entry || entry->is_outlier || entry->compile_failed
        || !entry->program) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen e347 (Metal) lookup failed on %s "
            "dev[%d] (entry=%p outlier=%d compile_failed=%d program=%p)\n",
            __FILE__, __LINE__, hostname, dev_idx,
            (const void*)entry,
            entry ? (int)entry->is_outlier : -1,
            entry ? (int)entry->compile_failed : -1,
            entry ? (const void*)entry->program : NULL);
        exit(1);
    }

    /* e347 specialization: iter_count=1, has_rules=1, no masks/BF,
     * SALT_BATCH=64 outer loop. Mirrors the harness-mode setup in
     * mdxfind.c rev 1.494 Metal branch ~line 39173. Identical to the
     * OpenCL twin's specialization in gpu/gpu_opencl.c line 12047. */
    struct hx_specialization spec;
    memset(&spec, 0, sizeof(spec));
    spec.iter_count_if_fixed = 1;
    spec.has_rules           = 1;
    spec.has_masks           = 0;
    spec.has_bf              = 0;
    spec.salt_minlen         = 1;
    spec.salt_maxlen         = 64;
    spec.salt_count_regime   = HX_SALT_BATCH_64;
    spec.iter_shape          = HX_ITER_NONE;
    spec.digest_endianness   = HX_DIGEST_LE;
    spec.emit_width          = 16;

    char *src = NULL;
    size_t src_cap = 0;
    /* Sub-phase 5a.2 (2026-05-22): walker signature takes entry. */
    int rc = hx_emit_kernel(entry->program, &spec, HX_BACKEND_METAL,
                            entry, &src, &src_cap);
    if (rc != 0 || !src) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen e347 (Metal) walker rc=%d on %s "
            "dev[%d] (src=%p cap=%zu)\n",
            __FILE__, __LINE__, rc, hostname, dev_idx,
            (void*)src, src_cap);
        if (src) free(src);
        exit(1);
    }

    /* Always-on source dump for post-mortem (mirrors OpenCL twin spec
     * §3). Extension .metal so dumps from both backends coexist when
     * the user inspects /tmp side-by-side. Path is <pid>-<dev> so
     * multi-process / future multi-device runs do not clobber.
     * Non-fatal if the dump fails. */
    char dump_path[256];
    snprintf(dump_path, sizeof(dump_path),
             "/tmp/mdxfind-codegen-e347-%d-dev%d.metal",
             (int)getpid(), dev_idx);
    {
        FILE *df = fopen(dump_path, "w");
        if (df) {
            fwrite(src, 1, strlen(src), df);
            fclose(df);
        } else {
            fprintf(stderr,
                "hx codegen (Metal): source dump to %s failed on %s "
                "dev[%d] (errno=%d %s) -- proceeding without dump\n",
                dump_path, hostname, dev_idx, errno, strerror(errno));
        }
    }

    /* JIT compile via the 2a.6 keep-wrapper. Concatenates metal_-
     * common_str + emitted source, compiles via -[MTLDevice newLibrary
     * WithSource:options:error:], looks up `kernelb_hx_e347_phase0`,
     * creates MTLComputePipelineState. FATAL on any failure -- helper
     * already exits(1) with full diagnostic. Returns 0 on success. */
    void *lib_out = NULL;
    void *pso_out = NULL;
    int jit_rc = gpu_metal_jit_compile_source_with_common_keep(
        dev_idx, src, "kernelb_hx_e347_phase0", &lib_out, &pso_out);
    if (jit_rc != 0 || lib_out == NULL || pso_out == NULL) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen e347 (Metal) JIT failed on %s "
            "dev[%d] (rc=%d lib=%p pso=%p) -- source dumped to %s\n",
            __FILE__, __LINE__, hostname, dev_idx,
            jit_rc, lib_out, pso_out, dump_path);
        free(src);
        exit(1);
    }

    mtl_lib_kernelb_codegen = lib_out;
    mtl_pso_kernelb_codegen = pso_out;

    fprintf(stderr,
        "Metal GPU[%d]: hx codegen e347 kernel B PSO created "
        "(source %zu bytes, dump=%s) on %s\n",
        dev_idx, strlen(src), dump_path, hostname);

    free(src);
    return mtl_pso_kernelb_codegen;
}

/* ====================================================================
 * Phase 4 sub-phase 4a.2b (2026-05-22): Metal kernel-B production
 * dispatcher.
 *
 * Metal twin of gpu_opencl_kernelb_dispatch_proto (gpu/gpu_opencl.c rev
 * 1.190 line 12158). Two-kernel pipeline for JOB_MD5MD5SALT:
 *   (A) kernel A1 -- cand_rules_phase0 -- writes packed candidates
 *   (B) kernel B  -- codegen-emitted kernelb_hx_e347_phase0 -- consumes
 *                    the kernel-A output and emits hits via the
 *                    EMIT_HIT_4_DEDUP_OR_OVERFLOW macro
 *
 * Prior sub-phases (1a.1b-continued through 1a.4) shipped only kernel
 * A1/A2/A3/A4 on Metal in harness mode (no hit pipeline). Sub-phase
 * 4a.2 added codegen scaffolding (mtl_pso_kernelb_codegen +
 * gpu_metal_kernelb_codegen_pso lazy-init helper) but NO production
 * caller. This sub-phase BUILDS that caller -- after 4a.2b ships,
 * real-world `./mdxfind -m e347 -G 0` on Metal works.
 *
 * The dispatcher reuses three buffer families:
 *   - mtl_buf_proto_packed / mtl_buf_proto_chunk_index / mtl_buf_proto_state
 *     (kernel A scratch; already lazy-allocated by
 *     gpu_metal_kernelA_ensure_proto_bufs)
 *   - buf_salt_data / buf_salt_off / buf_salt_lens / buf_compact_fp /
 *     buf_compact_idx / buf_hash_data / buf_hash_data_off /
 *     buf_overflow_keys / buf_overflow_hashes / buf_overflow_offsets
 *     (device-resident; uploaded by gpu_metal_set_salt + gpu_metal_set_-
 *     compact_table + gpu_metal_set_overflow)
 *   - buf_hashes_shown (on-GPU dedup; lazy-init below per
 *     feedback_proto_dispatch_lazy_alloc_dependency.md)
 *
 * Two NEW persistent device-buffer statics (mtl_buf_kernelb_ovr_set /
 * mtl_buf_kernelb_ovr_gid) hold the explicit atomic_uint args 16/17
 * that the Metal emitter wires as standalone buffers (the OpenCL twin
 * aliases them off payload+100/+104; Metal can't cast through a non-
 * atomic pointer so they MUST be separate buffers).
 *
 * MDXFIND_HX_CODEGEN=0 opt-out: on Metal there is NO legacy kernel-B
 * path. The opt-out is therefore a FATAL "not supported" message,
 * documented in feedback_external_failures_are_fatal.md compliance.
 * Users on Metal cannot revert to a broken path; the codegen IS the
 * Metal kernel-B implementation.
 *
 * Returns h_hits on dispatch fire (caller does not free); NULL when
 * preconditions fail (op != JOB_MD5MD5SALT, metal not ready, no rules,
 * etc). *nhits_out reports hit count (0..GPU_MAX_HITS).
 * ==================================================================== */

/* Forward decl for the A1 scratch-buffer helper (defined below in the
 * Phase 1a A1 infrastructure block). The kernel-B production dispatcher
 * is placed above A1 for narrative cohesion with the codegen PSO helper;
 * the actual function definition lives further down the file. */
static int gpu_metal_kernelA_ensure_proto_bufs(size_t need_packed_bytes,
                                               size_t need_index_slots);

uint32_t *gpu_metal_kernelb_dispatch_proto(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out)
{
    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname) - 1);

    if (nhits_out != NULL) *nhits_out = 0;

    /* Early structural gates. Mirror OpenCL twin.
     *
     * Sub-phase 5a.5 (2026-05-22): the seven 5a-eligible MAKE_MD5PASS
     * family JOB enums (e122/e159/e161/e163/e165/e167/e169) join
     * JOB_MD5MD5SALT in the accepted set. Family kernels are looked up
     * per-JOB via gpu_metal_kernelb_codegen_pso_for(dev_idx, op) below
     * (per-JOB slot map established in 5a.3). The family is unsalted;
     * the upstream gpujob_metal.m route gate uploads a 1-byte zero
     * placeholder salt for family ops so the kernel's salt buffer
     * bindings are non-NULL (the family body does (void)salts;
     * (void)salt_offsets; (void)salt_lens; so contents are ignored). */
    if (op != JOB_MD5MD5SALT
        && !gpu_codegen_kernelb_family_md5pass_eligible(op)) {
        return NULL;
    }
    if (!metal_ready || dev_idx != 0) return NULL;
    if (packed_words == NULL || packed_size == 0 ||
        word_offset == NULL || num_words == 0) return NULL;
    if (gpu_rule_count <= 0 || gpu_rule_program == NULL ||
        gpu_rule_offsets == NULL || gpu_rule_program_len == 0) {
        return NULL;
    }

    /* MDXFIND_HX_CODEGEN=0 opt-out FATAL per 4a.2 design. On Metal
     * there is no legacy hand-written kernel B path to fall back to;
     * the codegen IS the Metal kernel-B implementation. The opt-out
     * is therefore a documented-but-unsupported configuration. */
    if (!gpu_metal_hx_codegen_enabled()) {
        fprintf(stderr,
            "FATAL: %s:%d Metal GPU[%d]: MDXFIND_HX_CODEGEN=0 opt-out "
            "is not supported on Metal -- Metal has no legacy kernel B "
            "path (the codegen IS the Metal kernel-B implementation). "
            "Unset MDXFIND_HX_CODEGEN or set it to any value other than "
            "the literal \"0\" to use the codegen path on %s\n",
            __FILE__, __LINE__, dev_idx, hostname);
        exit(1);
    }

    /* Build kernel A1 PSO lazily (once per process). FATAL on failure
     * via the helper's internal GPU_FATAL path. */
    if (metal_kernel_a_rules_pso_lazy() < 0) {
        fprintf(stderr,
            "FATAL: %s:%d Metal GPU[%d]: kernel A1 PSO lazy-create "
            "failed on %s\n",
            __FILE__, __LINE__, dev_idx, hostname);
        exit(1);
    }

    /* Acquire codegen kernel B PSO (FATAL on any failure inside).
     * Sub-phase 5a.3 (2026-05-22): per-JOB variant -- the production
     * dispatcher still routes JOB_MD5MD5SALT only (gated above), but
     * the _for(dev_idx, op) call exercises the same cache lookup path
     * the family routing will use in 5a.5. Slot 0 holds the e347 PSO
     * after first call; subsequent dispatches hit cache. */
    void *kb_pso_void = gpu_metal_kernelb_codegen_pso_for(dev_idx, op);
    if (kb_pso_void == NULL) {
        fprintf(stderr,
            "FATAL: %s:%d Metal GPU[%d]: kernel B codegen PSO unavailable "
            "on %s\n",
            __FILE__, __LINE__, dev_idx, hostname);
        exit(1);
    }
    id<MTLComputePipelineState> kb_pso =
        (__bridge id<MTLComputePipelineState>)kb_pso_void;

    /* One-shot per-process-per-op stderr marker mirroring OpenCL twin
     * shape. Log scraping treats both backends symmetrically.
     *
     * Sub-phase 5a.5 (2026-05-22): entry-point name picked per op
     * (e347 uses kernelb_hx_e347_phase0; the seven family JOBs share
     * kernelb_hx_codegen_phase0). The per-op once-only logic uses a
     * small static table keyed on op so each family member announces
     * its first dispatch independently rather than the e347-only
     * single static used by 4a.2b. */
    {
        static int _kernelb_seen_ops[16];   /* sentinel 0 = empty slot */
        static int _kernelb_seen_count = 0;
        int _seen = 0;
        for (int _i = 0; _i < _kernelb_seen_count; _i++) {
            if (_kernelb_seen_ops[_i] == op) { _seen = 1; break; }
        }
        if (!_seen) {
            if (_kernelb_seen_count <
                (int)(sizeof(_kernelb_seen_ops) /
                      sizeof(_kernelb_seen_ops[0]))) {
                _kernelb_seen_ops[_kernelb_seen_count++] = op;
            }
            const char *_entry_name = (op == JOB_MD5MD5SALT)
                ? "kernelb_hx_e347_phase0"
                : "kernelb_hx_codegen_phase0";
            fprintf(stderr,
                "Metal GPU[%d]: kernel B PROTO dispatch fired "
                "(op=%d n_words=%u n_rules=%d) -- codegen %s\n",
                dev_idx, op, num_words, gpu_rule_count, _entry_name);
        }
    }

    uint32_t n_rules = (uint32_t)gpu_rule_count;

    /* --- 1. Ensure persistent kernel-A scratch buffers exist + are
     * sized for the worst-case (num_words * n_rules) slots. */
    size_t max_slots  = (size_t)num_words * (size_t)n_rules;
    size_t max_packed = max_slots * 256u;
    if (max_packed < METAL_MIN_BUFFER_BYTES) max_packed = METAL_MIN_BUFFER_BYTES;
    if (gpu_metal_kernelA_ensure_proto_bufs(max_packed, max_slots) < 0) {
        fprintf(stderr,
            "FATAL: %s:%d Metal GPU[%d]: kernelA_ensure_proto_bufs "
            "failed on %s (max_packed=%zu max_slots=%zu)\n",
            __FILE__, __LINE__, dev_idx, hostname, max_packed, max_slots);
        exit(1);
    }

    /* --- 2. Ensure rule program / rule offset MTLBuffers are uploaded
     * (idempotent). */
    if (metal_upload_rules_lazy() < 0) {
        fprintf(stderr,
            "FATAL: %s:%d Metal GPU[%d]: metal_upload_rules_lazy failed "
            "on %s (rule_count=%d program_len=%u)\n",
            __FILE__, __LINE__, dev_idx, hostname,
            gpu_rule_count, gpu_rule_program_len);
        exit(1);
    }

    /* --- 3. Precondition check on the device-resident buffer set used
     * by kernel B. Mirrors the harness-mode validator. */
    if (buf_compact_fp == nil || buf_compact_idx == nil ||
        buf_hash_data == nil || buf_hash_data_off == nil ||
        buf_salt_data == nil || buf_salt_off == nil || buf_salt_lens == nil) {
        fprintf(stderr,
            "FATAL: %s:%d Metal GPU[%d]: kernel B requires compact-table + "
            "salt buffers (compact_fp=%p compact_idx=%p hash_data=%p "
            "hash_data_off=%p salt_data=%p salt_off=%p salt_lens=%p) -- "
            "did caller invoke gpu_metal_set_compact_table + gpu_metal_set_salt "
            "before dispatch on %s?\n",
            __FILE__, __LINE__, dev_idx,
            (void *)buf_compact_fp, (void *)buf_compact_idx,
            (void *)buf_hash_data, (void *)buf_hash_data_off,
            (void *)buf_salt_data, (void *)buf_salt_off,
            (void *)buf_salt_lens, hostname);
        exit(1);
    }

    /* Overflow placeholders. set_compact_table does NOT provision them;
     * call set_overflow(..., 0) once to materialize floor-sized buffers
     * (the kernel guards reads with `if (overflow_count > 0u)`). */
    if (buf_overflow_keys == nil || buf_overflow_hashes == nil ||
        buf_overflow_offsets == nil) {
        if (gpu_metal_set_overflow(0, NULL, NULL, NULL, NULL, 0) != 0) {
            fprintf(stderr,
                "FATAL: %s:%d Metal GPU[%d]: overflow placeholder "
                "provisioning failed on %s\n",
                __FILE__, __LINE__, dev_idx, hostname);
            exit(1);
        }
    }

    /* All ARC-managed Obj-C allocations live inside the autoreleasepool
     * so transient MTLBuffer wrappers (payload, hits) drop their strong
     * refs before this function returns. */
    int   ret_nhits      = 0;
    uint32_t actual_slots      = 0;
    uint32_t actual_byte_count = 0;
    uint32_t overflow_flag     = 0;

    @autoreleasepool {
        /* --- 4. Allocate kernel-A payload (OCLParams + word_offset[] +
         * packed_words tail). Same wire layout as harness mode and
         * OpenCL twin: [OCLParams@0][hit_count@128][word_offset[]@132]
         * [packed_words@132+4N]. Allocated fresh per dispatch (Shared
         * storage; ARC reclaims at autoreleasepool drain). */
        size_t wo_size      = (size_t)num_words * sizeof(uint32_t);
        size_t payload_pkt  = 132 + wo_size;
        size_t payload_size = payload_pkt + packed_size;
        if (payload_size < METAL_MIN_BUFFER_BYTES) payload_size = METAL_MIN_BUFFER_BYTES;

        id<MTLBuffer> buf_payload =
            [mtl_device newBufferWithLength:payload_size
                                    options:MTLResourceStorageModeShared];
        if (buf_payload == nil) {
            fprintf(stderr,
                "FATAL: %s:%d Metal GPU[%d]: kernel B payload newBuffer "
                "(%zu bytes) failed on %s\n",
                __FILE__, __LINE__, dev_idx, payload_size, hostname);
            exit(1);
        }

        uint8_t *p = (uint8_t *)[buf_payload contents];
        memset(p, 0, payload_size);

        OCLParams params;
        memset(&params, 0, sizeof(params));
        params.compact_mask     = cache_compact_mask;
        params.num_words        = num_words;
        params.num_rules        = n_rules;
        params.max_probe        = 256u;
        params.hash_data_count  = cache_hash_data_count;
        params.max_hits         = (uint32_t)GPU_MAX_HITS;
        params.overflow_count   = (uint32_t)cache_overflow_count;
        params.max_iter         = 1u;
        params.iter_count       = 1u;
        params.salt_start       = 0u;
        params.num_salts        = (uint32_t)cached_salts_count;
        params.base_word_idx    = 0u;
        params.packed_size      = (uint32_t)max_packed;  /* capacity for kernel A */
        memcpy(p, &params, sizeof(params));

        /* hit_count slot at offset 128 (already zeroed). */
        memcpy(p + 132, word_offset, wo_size);
        memcpy(p + payload_pkt, packed_words, packed_size);

        /* Zero the kernel-A state buffer (slot/byte/overflow counters). */
        memset([mtl_buf_proto_state contents], 0, 3u * sizeof(uint32_t));

        /* --- 5. Encode + dispatch kernel A1 (6-arg signature). */
        const NSUInteger tg_size_a = 64;
        NSUInteger global_a = (NSUInteger)num_words * (NSUInteger)n_rules;
        NSUInteger padded_a = ((global_a + tg_size_a - 1) / tg_size_a) * tg_size_a;
        NSUInteger groups_a = padded_a / tg_size_a;

        MTLSize tg_a   = MTLSizeMake(tg_size_a, 1, 1);
        MTLSize grid_a = MTLSizeMake(groups_a, 1, 1);

        id<MTLCommandBuffer> cb_a = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc_a = [cb_a computeCommandEncoder];
        [enc_a setComputePipelineState:mtl_pso_kernel_a_rules];
        [enc_a setBuffer:buf_payload                offset:0 atIndex:0];
        [enc_a setBuffer:buf_rule_program           offset:0 atIndex:1];
        [enc_a setBuffer:buf_rule_offset            offset:0 atIndex:2];
        [enc_a setBuffer:mtl_buf_proto_packed       offset:0 atIndex:3];
        [enc_a setBuffer:mtl_buf_proto_chunk_index  offset:0 atIndex:4];
        [enc_a setBuffer:mtl_buf_proto_state        offset:0 atIndex:5];
        [enc_a dispatchThreadgroups:grid_a threadsPerThreadgroup:tg_a];
        [enc_a endEncoding];
        [cb_a commit];
        [cb_a waitUntilCompleted];

        if (cb_a.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr,
                "FATAL: %s:%d Metal GPU[%d]: kernel A1 cmdbuffer status=%ld "
                "on %s (error=%s)\n",
                __FILE__, __LINE__, dev_idx, (long)cb_a.status, hostname,
                cb_a.error ? [[cb_a.error localizedDescription] UTF8String]
                           : "(none)");
            exit(1);
        }

        /* --- 6. Read kernel-A state (slot/byte/overflow counters).
         * Shared storage: [contents] is host-visible after waitUntilCompleted. */
        {
            uint32_t state[3] = {0, 0, 0};
            memcpy(state, [mtl_buf_proto_state contents], 3u * sizeof(uint32_t));
            actual_slots      = state[0];
            actual_byte_count = state[1];
            overflow_flag     = state[2];
        }

        if (overflow_flag) {
            /* Kernel A overflowed the candidate buffer; skip kernel B
             * for this chunk. Mirrors OpenCL twin proto behavior. */
            fprintf(stderr,
                "[kernelb_proto] Metal GPU[%d]: kernel A overflow "
                "(slots=%u bytes=%u packed_cap=%zu) -- skipping kernel B "
                "for this chunk on %s\n",
                dev_idx, actual_slots, actual_byte_count, max_packed,
                hostname);
            ret_nhits = 0;
        } else if (actual_slots == 0) {
            /* Kernel A produced zero candidates; no kernel B work. */
            ret_nhits = 0;
        } else {
            /* --- 7. Update OCLParams for kernel B in the same payload
             * buffer (Shared storage; direct host write is visible to
             * the GPU once the next command buffer is committed). */
            params.num_words   = actual_slots;        /* output slot count */
            params.packed_size = actual_byte_count;   /* bytes written */
            params.salt_start  = 0u;
            /* hit_count slot at offset 128 stays zero (already cleared). */
            memcpy(p, &params, sizeof(params));
            /* Reassert hit_count = 0 (in case prior dispatch left
             * non-zero; defensive). */
            *((uint32_t *)(p + 128)) = 0u;

            /* --- 8. Ensure persistent buf_hashes_shown is allocated +
             * sized for hash_data_count + overflow_count. Mirrors
             * feedback_proto_dispatch_lazy_alloc_dependency.md
             * compliance: the codegen kernel's EMIT_HIT_4_DEDUP_OR_OVERFLOW
             * macro dereferences hashes_shown[matched_idx]; passing a
             * smaller-than-needed buffer crashes the GPU on first hit. */
            size_t hs_need = (size_t)cache_hash_data_count
                           + (size_t)cache_overflow_count;
            if (hs_need == 0) hs_need = 1;
            size_t hs_bytes = hs_need * sizeof(uint32_t);
            if (hs_bytes < METAL_MIN_BUFFER_BYTES) hs_bytes = METAL_MIN_BUFFER_BYTES;
            if (buf_hashes_shown == nil
                || hs_bytes > mtl_buf_kernelb_hashes_shown_cap) {
                buf_hashes_shown = nil;  /* ARC drop old before realloc */
                buf_hashes_shown =
                    [mtl_device newBufferWithLength:hs_bytes
                                            options:MTLResourceStorageModeShared];
                if (buf_hashes_shown == nil) {
                    fprintf(stderr,
                        "FATAL: %s:%d Metal GPU[%d]: kernel B "
                        "buf_hashes_shown newBuffer(%zu bytes) failed on "
                        "%s (hash_data_count=%u overflow_count=%d)\n",
                        __FILE__, __LINE__, dev_idx, hs_bytes, hostname,
                        cache_hash_data_count, cache_overflow_count);
                    exit(1);
                }
                mtl_buf_kernelb_hashes_shown_cap = hs_bytes;
            }
            /* Zero the active region before each dispatch. */
            memset([buf_hashes_shown contents], 0, hs_bytes);

            /* --- 9. Ensure persistent ovr_set + ovr_gid atomic_uint
             * buffers exist. Allocated once per session; both zeroed
             * before each dispatch (ovr_gid starts at 0xFFFFFFFFu per
             * the harness reference; sentinel for "no overflow yet"). */
            if (mtl_buf_kernelb_ovr_set == nil) {
                mtl_buf_kernelb_ovr_set =
                    [mtl_device newBufferWithLength:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
                if (mtl_buf_kernelb_ovr_set == nil) {
                    fprintf(stderr,
                        "FATAL: %s:%d Metal GPU[%d]: kernel B ovr_set "
                        "newBuffer failed on %s\n",
                        __FILE__, __LINE__, dev_idx, hostname);
                    exit(1);
                }
            }
            if (mtl_buf_kernelb_ovr_gid == nil) {
                mtl_buf_kernelb_ovr_gid =
                    [mtl_device newBufferWithLength:sizeof(uint32_t)
                                            options:MTLResourceStorageModeShared];
                if (mtl_buf_kernelb_ovr_gid == nil) {
                    fprintf(stderr,
                        "FATAL: %s:%d Metal GPU[%d]: kernel B ovr_gid "
                        "newBuffer failed on %s\n",
                        __FILE__, __LINE__, dev_idx, hostname);
                    exit(1);
                }
            }
            *((uint32_t *)[mtl_buf_kernelb_ovr_set contents]) = 0u;
            *((uint32_t *)[mtl_buf_kernelb_ovr_gid contents]) = 0xFFFFFFFFu;

            /* --- 10. Allocate kernel-B hit_count + hits buffers (per-
             * dispatch; ARC reclaims at pool drain). Sized to
             * GPU_MAX_HITS like the existing dispatch_md5_rules path. */
            id<MTLBuffer> buf_hit_count =
                [mtl_device newBufferWithLength:sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
            if (buf_hit_count == nil) {
                fprintf(stderr,
                    "FATAL: %s:%d Metal GPU[%d]: kernel B hit_count "
                    "newBuffer failed on %s\n",
                    __FILE__, __LINE__, dev_idx, hostname);
                exit(1);
            }
            *((uint32_t *)[buf_hit_count contents]) = 0u;

            size_t hits_bytes = (size_t)GPU_MAX_HITS * GPU_HIT_STRIDE * sizeof(uint32_t);
            id<MTLBuffer> buf_hits =
                [mtl_device newBufferWithLength:hits_bytes
                                        options:MTLResourceStorageModeShared];
            if (buf_hits == nil) {
                fprintf(stderr,
                    "FATAL: %s:%d Metal GPU[%d]: kernel B hits "
                    "newBuffer(%zu bytes) failed on %s\n",
                    __FILE__, __LINE__, dev_idx, hits_bytes, hostname);
                exit(1);
            }
            memset([buf_hits contents], 0, hits_bytes);

            /* --- 11. Encode + dispatch kernel B (18-arg signature
             * mirroring the Metal emitter's [[buffer(N)]] layout). */
            uint32_t num_salt_chunks =
                ((uint32_t)cached_salts_count + 64u - 1u) / 64u;
            if (num_salt_chunks == 0u) num_salt_chunks = 1u;
            NSUInteger total_threads =
                (NSUInteger)actual_slots * (NSUInteger)num_salt_chunks;

            const NSUInteger tg_size_b = 64;
            NSUInteger groups_b = (total_threads + tg_size_b - 1) / tg_size_b;
            if (groups_b == 0) groups_b = 1;
            MTLSize tg_b   = MTLSizeMake(tg_size_b, 1, 1);
            MTLSize grid_b = MTLSizeMake(groups_b, 1, 1);

            id<MTLCommandBuffer> cb_b = [mtl_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc_b = [cb_b computeCommandEncoder];
            [enc_b setComputePipelineState:kb_pso];
            [enc_b setBuffer:buf_payload               offset:0 atIndex:0];
            [enc_b setBuffer:mtl_buf_proto_packed      offset:0 atIndex:1];
            [enc_b setBuffer:mtl_buf_proto_chunk_index offset:0 atIndex:2];
            [enc_b setBuffer:buf_salt_data             offset:0 atIndex:3];
            [enc_b setBuffer:buf_salt_off              offset:0 atIndex:4];
            [enc_b setBuffer:buf_salt_lens             offset:0 atIndex:5];
            [enc_b setBuffer:buf_compact_fp            offset:0 atIndex:6];
            [enc_b setBuffer:buf_compact_idx           offset:0 atIndex:7];
            [enc_b setBuffer:buf_hash_data             offset:0 atIndex:8];
            [enc_b setBuffer:buf_hash_data_off         offset:0 atIndex:9];
            [enc_b setBuffer:buf_hits                  offset:0 atIndex:10];
            [enc_b setBuffer:buf_hit_count             offset:0 atIndex:11];
            [enc_b setBuffer:buf_overflow_keys         offset:0 atIndex:12];
            [enc_b setBuffer:buf_overflow_hashes       offset:0 atIndex:13];
            [enc_b setBuffer:buf_overflow_offsets      offset:0 atIndex:14];
            [enc_b setBuffer:buf_hashes_shown          offset:0 atIndex:15];
            [enc_b setBuffer:mtl_buf_kernelb_ovr_set   offset:0 atIndex:16];
            [enc_b setBuffer:mtl_buf_kernelb_ovr_gid   offset:0 atIndex:17];
            [enc_b dispatchThreadgroups:grid_b threadsPerThreadgroup:tg_b];
            [enc_b endEncoding];
            [cb_b commit];
            [cb_b waitUntilCompleted];

            if (cb_b.status != MTLCommandBufferStatusCompleted) {
                fprintf(stderr,
                    "FATAL: %s:%d Metal GPU[%d]: kernel B cmdbuffer status=%ld "
                    "on %s (error=%s)\n",
                    __FILE__, __LINE__, dev_idx, (long)cb_b.status, hostname,
                    cb_b.error ? [[cb_b.error localizedDescription] UTF8String]
                               : "(none)");
                exit(1);
            }

            /* --- 12. Read hits back. h_hits is the persistent host
             * buffer (allocated in gpu_metal_init); copy from device-
             * visible Shared MTLBuffer contents. */
            uint32_t hcount = *((uint32_t *)[buf_hit_count contents]);
            if (hcount > (uint32_t)GPU_MAX_HITS) hcount = (uint32_t)GPU_MAX_HITS;
            if (hcount > 0) {
                memcpy(h_hits, [buf_hits contents],
                       (size_t)hcount * GPU_HIT_STRIDE * sizeof(uint32_t));
            }
            ret_nhits = (int)hcount;
        }
    }  /* autoreleasepool */

    if (nhits_out != NULL) *nhits_out = ret_nhits;
    return h_hits;
}

/* ====================================================================
 * Phase 1a sub-phase 1a.1b-continued (2026-05-20): Metal A1 dispatcher
 * infrastructure. Port of gpu/gpu_opencl.c:9747-10303 (the A1 portion of
 * the OpenCL two-kernel pipeline). Kernel B is NOT ported here per
 * sub-phase scope -- A1 correctness ships first, kernel B follow-on is
 * a separate sub-phase.
 *
 * Three buffer types (mirror OpenCL gpu_device fields):
 *   mtl_buf_proto_packed       -- candidate plaintext concatenation (length-
 *                                 prefixed). Capacity grows on demand.
 *   mtl_buf_proto_chunk_index  -- uint32 byte-offset per slot in packed.
 *   mtl_buf_proto_state        -- 12 bytes: [slot_counter][byte_counter]
 *                                 [overflow_flag]. atomic_uint on device.
 *
 * Storage mode: MTLResourceStorageModeShared on all three (per Phase 1
 * memo §3 -- existing infrastructure uses Shared exclusively; the Apple
 * Silicon UMA vs AMD discrete discrepancy from architect §4 risk T9 is
 * a perf concern not a correctness concern; matching the existing pattern
 * is safer for the A1 correctness ship). All MTLBuffer reads happen via
 * [buf contents] after [cb waitUntilCompleted] -- same convention as
 * gpu_metal_dispatch_md5_rules.
 * ==================================================================== */

/* gpu_metal_kernelA_ensure_proto_bufs -- allocate (or grow) the three
 * per-device buffers used by the A1 dispatch. Mirror of OpenCL twin
 * gpu_opencl_kernelb_ensure_proto_bufs (gpu/gpu_opencl.c:9759). Returns 0
 * on success. Any allocation failure is FATAL per feedback_external_-
 * failures_are_fatal.md. */
static int gpu_metal_kernelA_ensure_proto_bufs(size_t need_packed_bytes,
                                                size_t need_index_slots)
{
    if (mtl_device == nil) {
        GPU_FATAL("Metal: kernelA_ensure_proto_bufs called with device=nil");
    }
    size_t need_index_bytes = need_index_slots * sizeof(uint32_t);

    /* Grow packed buf if needed. */
    if (mtl_buf_proto_packed == nil || need_packed_bytes > mtl_buf_proto_packed_cap) {
        size_t alloc_bytes = need_packed_bytes;
        if (alloc_bytes < METAL_MIN_BUFFER_BYTES) alloc_bytes = METAL_MIN_BUFFER_BYTES;
        mtl_buf_proto_packed = nil;  /* ARC drop old before realloc */
        mtl_buf_proto_packed = [mtl_device newBufferWithLength:alloc_bytes
                                                       options:MTLResourceStorageModeShared];
        if (mtl_buf_proto_packed == nil) {
            GPU_FATAL("Metal: mtl_buf_proto_packed newBuffer(%zu bytes) failed",
                      alloc_bytes);
        }
        mtl_buf_proto_packed_cap = alloc_bytes;
    }

    /* Grow chunk index buf if needed. */
    if (mtl_buf_proto_chunk_index == nil || need_index_bytes > mtl_buf_proto_index_cap) {
        size_t alloc_bytes = need_index_bytes;
        if (alloc_bytes < METAL_MIN_BUFFER_BYTES) alloc_bytes = METAL_MIN_BUFFER_BYTES;
        mtl_buf_proto_chunk_index = nil;
        mtl_buf_proto_chunk_index = [mtl_device newBufferWithLength:alloc_bytes
                                                            options:MTLResourceStorageModeShared];
        if (mtl_buf_proto_chunk_index == nil) {
            GPU_FATAL("Metal: mtl_buf_proto_chunk_index newBuffer(%zu bytes) failed",
                      alloc_bytes);
        }
        mtl_buf_proto_index_cap = alloc_bytes;
    }

    /* State buf is fixed-size 12 bytes (3 atomic uints). Allocate once;
     * caller zeroes before each dispatch via [contents] memset. */
    if (mtl_buf_proto_state == nil) {
        size_t alloc_bytes = METAL_MIN_BUFFER_BYTES;  /* >= 12 bytes */
        mtl_buf_proto_state = [mtl_device newBufferWithLength:alloc_bytes
                                                      options:MTLResourceStorageModeShared];
        if (mtl_buf_proto_state == nil) {
            GPU_FATAL("Metal: mtl_buf_proto_state newBuffer(%zu bytes) failed",
                      alloc_bytes);
        }
    }

    return 0;
}

/* gpu_metal_kernel_a_trace_dump -- Phase 1a A1 buffer-quadruple trace.
 * Mirrors OpenCL twin gpu_opencl_kernel_a_trace_dump byte-for-byte
 * (gpu/gpu_opencl.c:9899). The harness reads these three files and
 * compares them byte-exact across backends, so the format MUST match.
 *
 * Buffer formats (raw binary, little-endian):
 *   <prefix>.state    : 12 bytes [slot_counter][byte_counter][overflow_flag]
 *   <prefix>.chunkidx : 4 * actual_slots bytes (uint32 byte offsets)
 *   <prefix>.packed   : actual_byte_count bytes ([len][bytes]...)
 *
 * All failures are FATAL per feedback_external_failures_are_fatal.md. */
static void gpu_metal_kernel_a_trace_dump(int dev_idx,
    const uint32_t state[3],
    const void *index_data, size_t index_bytes,
    const void *packed_data, size_t packed_bytes)
{
    const char *prefix = getenv("MDXFIND_KERNEL_A_TRACE");
    if (prefix == NULL || prefix[0] == '\0') return;

    char path[1024];
    FILE *f;

    snprintf(path, sizeof(path), "%s.state", prefix);
    f = fopen(path, "wb");
    if (f == NULL) {
        GPU_FATAL("Metal GPU[%d]: kernel A trace fopen(%s) failed: %s",
                  dev_idx, path, strerror(errno));
    }
    if (fwrite(state, sizeof(uint32_t), 3, f) != 3) {
        GPU_FATAL("Metal GPU[%d]: kernel A trace state write failed", dev_idx);
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s.chunkidx", prefix);
    f = fopen(path, "wb");
    if (f == NULL) {
        GPU_FATAL("Metal GPU[%d]: kernel A trace fopen(%s) failed: %s",
                  dev_idx, path, strerror(errno));
    }
    if (index_bytes > 0 && fwrite(index_data, 1, index_bytes, f) != index_bytes) {
        GPU_FATAL("Metal GPU[%d]: kernel A trace chunkidx write failed", dev_idx);
    }
    fclose(f);

    snprintf(path, sizeof(path), "%s.packed", prefix);
    f = fopen(path, "wb");
    if (f == NULL) {
        GPU_FATAL("Metal GPU[%d]: kernel A trace fopen(%s) failed: %s",
                  dev_idx, path, strerror(errno));
    }
    if (packed_bytes > 0 && fwrite(packed_data, 1, packed_bytes, f) != packed_bytes) {
        GPU_FATAL("Metal GPU[%d]: kernel A trace packed write failed", dev_idx);
    }
    fclose(f);

    fprintf(stderr,
        "Metal GPU[%d]: kernel A1 trace written (prefix=%s, slots=%u, bytes=%u, overflow=%u)\n",
        dev_idx, prefix, state[0], state[1], state[2]);
}

/* gpu_metal_kernelA_dispatch_proto -- Metal A1 dispatch entry point.
 * Port of OpenCL twin gpu_opencl_kernelb_dispatch_proto (gpu/gpu_opencl.c:
 * 9974) but A1-only (kernel B not wired in this sub-phase).
 *
 * CALLER CONTRACT (matches OpenCL twin):
 *   - gpu_metal_set_op() / set_rules() called for the dispatch.
 *   - packed_words / packed_size / word_offset / num_words follow the
 *     gpu_metal_dispatch_md5_rules convention.
 *
 * RETURN: always returns a non-NULL pointer (mtl_buf_proto_state's
 * contents reinterpreted) when the dispatch fired; NULL when env-gate
 * was unset or device not ready. *nhits_out is always set to 0 in
 * this sub-phase (no kernel B == no cracks; the harness reads cracks
 * via the trace files, not via the hit buffer).
 *
 * Sibling to gpu_metal_dispatch_md5_rules -- does NOT call/modify/
 * replace that function. */
uint32_t *gpu_metal_kernelA_dispatch_proto(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out)
{
    if (nhits_out != NULL) *nhits_out = 0;

    /* Gate: either legacy MDXFIND_KERNEL_B_PROTO or new
     * MDXFIND_KERNEL_A_PROTO=1 (variant 1) per sub-phase 1a.1 union. */
    if (getenv("MDXFIND_KERNEL_B_PROTO") == NULL &&
        !gpu_metal_kernel_a_proto_enabled()) {
        return NULL;
    }
    if (op != JOB_MD5MD5SALT) return NULL;
    if (!metal_ready || dev_idx != 0) return NULL;
    if (packed_words == NULL || packed_size == 0 ||
        word_offset == NULL || num_words == 0) return NULL;
    if (gpu_rule_count <= 0 || gpu_rule_program == NULL ||
        gpu_rule_offsets == NULL || gpu_rule_program_len == 0) {
        return NULL;
    }

    /* Build kernel A1 program lazily (once per process). */
    if (metal_kernel_a_rules_pso_lazy() < 0) {
        /* Soft-fall through to caller's standard dispatch on lazy
         * creator failure; the operator already saw the stderr from
         * the lazy-create site. */
        return NULL;
    }

    /* One-shot stderr marker per feedback_verify_gpu_fired_post_build.md.
     * Mirror OpenCL twin's "kernel A1 dispatch fired" marker shape so
     * the harness Gate 2 grep is backend-agnostic. */
    {
        static int _kernel_a1_first = 0;
        if (!_kernel_a1_first) {
            _kernel_a1_first = 1;
            fprintf(stderr,
                "Metal GPU[%d]: kernel A1 dispatch fired "
                "(variant=1 op=%d n_words=%u n_rules=%d)\n",
                dev_idx, op, num_words, gpu_rule_count);
        }
    }

    uint32_t n_rules = (uint32_t)gpu_rule_count;

    /* --- Size + (re)allocate prototype buffers ---
     * Worst-case slot count = num_words * n_rules (rejection-free upper).
     * Worst-case packed bytes = slots * 256 (1 len byte + up to 255). */
    size_t max_slots  = (size_t)num_words * (size_t)n_rules;
    size_t max_packed = max_slots * 256u;
    if (max_packed < METAL_MIN_BUFFER_BYTES) max_packed = METAL_MIN_BUFFER_BYTES;

    if (gpu_metal_kernelA_ensure_proto_bufs(max_packed, max_slots) < 0) {
        return NULL;
    }

    @autoreleasepool {
        /* --- Build OCLParams + word_offset + packed payload buffer ---
         * Same layout as the standard dispatch payload:
         *   offset 0..127   OCLParams
         *   offset 128..131 uint hit_count (unused for A1; zero)
         *   offset 132..    uint word_offset[num_words]
         *   offset 132+4N.. packed bytes
         *
         * Allocated fresh per dispatch (Shared storage; small enough
         * the allocator pool handles this efficiently). */
        size_t wo_size      = (size_t)num_words * sizeof(uint32_t);
        size_t payload_pkt  = 132 + wo_size;
        size_t payload_size = payload_pkt + packed_size;
        if (payload_size < METAL_MIN_BUFFER_BYTES) payload_size = METAL_MIN_BUFFER_BYTES;

        id<MTLBuffer> buf_payload =
            [mtl_device newBufferWithLength:payload_size
                                    options:MTLResourceStorageModeShared];
        if (buf_payload == nil) {
            GPU_FATAL("Metal: kernel A1 payload newBuffer(%zu bytes) failed "
                      "(op=%d num_words=%u)",
                      payload_size, op, num_words);
        }

        uint8_t *p = (uint8_t *)[buf_payload contents];
        memset(p, 0, payload_size);

        OCLParams *params = (OCLParams *)p;
        params->num_words   = num_words;
        params->num_rules   = n_rules;        /* sub-phase 1a.2 rename
                                               * (was params->num_masks
                                               * before D9.1.b; kernel A1
                                               * reads n_rules from offset
                                               * 108, not 64) */
        params->max_probe   = 256u;
        params->max_iter    = 1u;
        params->packed_size = (uint32_t)max_packed;  /* capacity for kernel A */

        /* hit_count zero at offset 128 (already zeroed by memset). */

        /* word_offset at offset 132. */
        memcpy(p + 132, word_offset, wo_size);

        /* packed words at offset 132 + wo_size. */
        memcpy(p + 132 + wo_size, packed_words, packed_size);

        /* --- Zero the b_proto_state buffer (slot/byte/overflow counters)
         * before each dispatch. Per OpenCL twin contract S7.2. */
        memset([mtl_buf_proto_state contents], 0, 3u * sizeof(uint32_t));

        /* Ensure rule_program and rule_offset MTLBuffers are uploaded.
         * Reuse the existing helper that the standard dispatch uses --
         * it's idempotent and binds the externs into buf_rule_program /
         * buf_rule_offset statics. */
        if (metal_upload_rules_lazy() < 0) {
            GPU_FATAL("Metal: metal_upload_rules_lazy failed in kernel A1 "
                      "dispatch (op=%d rule_count=%d program_len=%u)",
                      op, gpu_rule_count, gpu_rule_program_len);
        }

        /* --- Encode + dispatch kernel A ---
         * 6-arg signature (matches metal_kernel_a_rules.metal:
         *   arg 0 payload
         *   arg 1 rule_program
         *   arg 2 rule_offset
         *   arg 3 b_packed_buf
         *   arg 4 b_chunk_index
         *   arg 5 b_kernelA_state (device atomic_uint *)
         *
         * Threadgroup size 64 matches OpenCL twin (lsize_a=64). Global
         * threads = num_words * n_rules, padded to WG multiple. */
        const NSUInteger tg_size_x = 64;
        NSUInteger global_threads = (NSUInteger)num_words * (NSUInteger)n_rules;
        NSUInteger padded = ((global_threads + tg_size_x - 1) / tg_size_x) * tg_size_x;
        NSUInteger num_groups = padded / tg_size_x;

        MTLSize threadgroup = MTLSizeMake(tg_size_x, 1, 1);
        MTLSize grid        = MTLSizeMake(num_groups, 1, 1);

        id<MTLCommandBuffer> cb = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mtl_pso_kernel_a_rules];

        [enc setBuffer:buf_payload                offset:0 atIndex:0];
        [enc setBuffer:buf_rule_program           offset:0 atIndex:1];
        [enc setBuffer:buf_rule_offset            offset:0 atIndex:2];
        [enc setBuffer:mtl_buf_proto_packed       offset:0 atIndex:3];
        [enc setBuffer:mtl_buf_proto_chunk_index  offset:0 atIndex:4];
        [enc setBuffer:mtl_buf_proto_state        offset:0 atIndex:5];

        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.error != nil) {
            MTL_FATAL_NSERR(cb.error,
                "Metal kernel A1 dispatch error op=%d num_words=%u n_rules=%u",
                op, num_words, n_rules);
        }

        /* --- Read back state buffer (slot/byte/overflow counters) --- */
        uint32_t state[3] = {0, 0, 0};
        memcpy(state, [mtl_buf_proto_state contents], 3u * sizeof(uint32_t));
        uint32_t actual_slots      = state[0];
        uint32_t actual_byte_count = state[1];
        uint32_t overflow_flag     = state[2];

        /* --- Snapshot packed + chunk_index host-side for the trace channel.
         * Skip if overflow or zero-slot (no useful payload). */
        if (!overflow_flag && actual_slots > 0) {
            if (actual_byte_count > h_proto_packed_readback_cap) {
                free(h_proto_packed_readback);
                h_proto_packed_readback = calloc(1, actual_byte_count + METAL_MIN_BUFFER_BYTES);
                if (h_proto_packed_readback == NULL) {
                    GPU_FATAL("Metal: h_proto_packed_readback calloc(%u) failed", actual_byte_count);
                }
                h_proto_packed_readback_cap = actual_byte_count + METAL_MIN_BUFFER_BYTES;
            }
            size_t index_bytes = (size_t)actual_slots * sizeof(uint32_t);
            if (index_bytes > h_proto_index_readback_cap) {
                free(h_proto_index_readback);
                h_proto_index_readback = calloc(1, index_bytes + METAL_MIN_BUFFER_BYTES);
                if (h_proto_index_readback == NULL) {
                    GPU_FATAL("Metal: h_proto_index_readback calloc(%zu) failed", index_bytes);
                }
                h_proto_index_readback_cap = index_bytes + METAL_MIN_BUFFER_BYTES;
            }
            /* Shared storage: [contents] is already host-visible after
             * waitUntilCompleted. Snapshot into host scratch for stable
             * pointer (in case the MTLBuffer is reused before trace_dump). */
            memcpy(h_proto_packed_readback,
                   [mtl_buf_proto_packed contents],
                   (size_t)actual_byte_count);
            memcpy(h_proto_index_readback,
                   [mtl_buf_proto_chunk_index contents],
                   index_bytes);
        }

        /* --- Trace dump (zero overhead when MDXFIND_KERNEL_A_TRACE unset). */
        gpu_metal_kernel_a_trace_dump(dev_idx, state,
            h_proto_index_readback,
            (size_t)actual_slots * sizeof(uint32_t),
            h_proto_packed_readback,
            (size_t)actual_byte_count);

        if (overflow_flag) {
            fprintf(stderr,
                "[kernelA_proto] Metal GPU[%d]: kernel A overflow "
                "(slots=%u bytes=%u packed_cap=%zu) -- A1-only ship; "
                "no kernel B retry path\n",
                dev_idx, actual_slots, actual_byte_count, max_packed);
        }
    }  /* autoreleasepool */

    /* Sub-phase 1a.1b-continued: A1 only. No hit buffer / no kernel B
     * means *nhits_out stays 0; caller's hit-replay loop sees no work.
     * Returning a non-NULL pointer signals "dispatch fired" to the
     * worker (mirrors OpenCL twin's d->h_hits return). The pointer
     * itself is never dereferenced for hit data in this sub-phase --
     * the trace channel is the validation surface. */
    return (uint32_t *)[mtl_buf_proto_state contents];
}

/* ====================================================================
 * Phase 1a sub-phase 1a.2 (2026-05-21): kernel A2 (masks-only) Metal
 * dispatcher infrastructure. Metal twin of gpu/gpu_opencl.c:gpu_opencl_-
 * kernel_a_masks_dispatch + supporting helpers. Mirrors A1's three-
 * buffer prototype layout, replacing rule_program/rule_offset args with
 * the 4-buffer mask wire format (prepend, append, charsets, counts).
 *
 * Reuses A1's mtl_buf_proto_packed / mtl_buf_proto_chunk_index /
 * mtl_buf_proto_state buffers + the gpu_metal_kernel_a_trace_dump
 * helper; the buffer-quadruple format is variant-agnostic.
 * ==================================================================== */

/* gpu_metal_kernelA_upload_mask_buffers -- Metal mirror of OpenCL twin
 * gpu_opencl_kernel_a_upload_mask_buffers. Allocates four MTLBuffers
 * (Shared storage) on first invocation, composes host staging arrays
 * from the CPU mask globals, encodes MASK_LITERAL (-1) as 0xff per
 * D9.2.a, copies into the Shared-mode buffers (no explicit
 * write-buffer call; [buf contents] memcpy suffices).
 *
 * Per feedback_external_failures_are_fatal.md: any MTLBuffer alloc
 * failure is FATAL via GPU_FATAL. Idempotent: subsequent calls early-
 * return after a successful first upload. */
static int gpu_metal_kernelA_upload_mask_buffers(void)
{
    if (mtl_buf_kern_a_mask_allocated) return 0;
    if (mtl_device == nil) {
        GPU_FATAL("Metal: kernel A2 mask upload called with device=nil");
    }

    /* 2026-05-26: pattern_bytes stays at 32 bytes (16 positions * 2)
     * regardless of MAX_MASK_POS_CPU (256 on CPU side). The kernel A2
     * private scratch is uchar[16] and the metal_template internally
     * clamps n_prepend/n_append to 16. The mdxfind.c upload guards
     * refuse longer masks upstream; KERN_A_PATTERN_LEN_CAP defends in
     * depth. */
    const size_t KERN_A_PATTERN_LEN_CAP = 16u;
    const size_t pattern_bytes  = KERN_A_PATTERN_LEN_CAP * 2u;            /* 32 */
    const size_t charsets_bytes = (size_t)MASK_MAX_CLASSES_CPU * 256u;    /* 4096 */
    const size_t counts_bytes   = (size_t)MASK_MAX_CLASSES_CPU * sizeof(uint32_t); /* 64 */

    size_t alloc_pattern  = pattern_bytes  < METAL_MIN_BUFFER_BYTES ? METAL_MIN_BUFFER_BYTES : pattern_bytes;
    size_t alloc_charsets = charsets_bytes < METAL_MIN_BUFFER_BYTES ? METAL_MIN_BUFFER_BYTES : charsets_bytes;
    size_t alloc_counts   = counts_bytes   < METAL_MIN_BUFFER_BYTES ? METAL_MIN_BUFFER_BYTES : counts_bytes;

    mtl_buf_kern_a_mask_prepend = [mtl_device newBufferWithLength:alloc_pattern
                                                          options:MTLResourceStorageModeShared];
    if (mtl_buf_kern_a_mask_prepend == nil) {
        GPU_FATAL("Metal: mtl_buf_kern_a_mask_prepend newBuffer(%zu) failed", alloc_pattern);
    }
    mtl_buf_kern_a_mask_append = [mtl_device newBufferWithLength:alloc_pattern
                                                         options:MTLResourceStorageModeShared];
    if (mtl_buf_kern_a_mask_append == nil) {
        GPU_FATAL("Metal: mtl_buf_kern_a_mask_append newBuffer(%zu) failed", alloc_pattern);
    }
    mtl_buf_kern_a_mask_charsets = [mtl_device newBufferWithLength:alloc_charsets
                                                           options:MTLResourceStorageModeShared];
    if (mtl_buf_kern_a_mask_charsets == nil) {
        GPU_FATAL("Metal: mtl_buf_kern_a_mask_charsets newBuffer(%zu) failed", alloc_charsets);
    }
    mtl_buf_kern_a_mask_counts = [mtl_device newBufferWithLength:alloc_counts
                                                         options:MTLResourceStorageModeShared];
    if (mtl_buf_kern_a_mask_counts == nil) {
        GPU_FATAL("Metal: mtl_buf_kern_a_mask_counts newBuffer(%zu) failed", alloc_counts);
    }

    /* Compose + write staging. D9.2.a: classid -1 -> 0xff. */
    unsigned char *prepend_wire = (unsigned char *)[mtl_buf_kern_a_mask_prepend contents];
    unsigned char *append_wire  = (unsigned char *)[mtl_buf_kern_a_mask_append  contents];
    unsigned char *charsets_p   = (unsigned char *)[mtl_buf_kern_a_mask_charsets contents];
    uint32_t      *counts_p     = (uint32_t *)     [mtl_buf_kern_a_mask_counts   contents];

    memset(prepend_wire, 0, alloc_pattern);
    memset(append_wire,  0, alloc_pattern);
    memset(charsets_p,   0, alloc_charsets);
    memset(counts_p,     0, alloc_counts);

    for (int i = 0; i < MaskPrependLen && i < (int)KERN_A_PATTERN_LEN_CAP; i++) {
        int cid = MaskPrependPattern[i].classid;
        prepend_wire[i * 2]     = (cid == MASK_LITERAL_CPU)
                                  ? 0xffu
                                  : (unsigned char)(cid & 0xff);
        prepend_wire[i * 2 + 1] = MaskPrependPattern[i].literal;
    }
    for (int i = 0; i < MaskAppendLen && i < (int)KERN_A_PATTERN_LEN_CAP; i++) {
        int cid = MaskAppendPattern[i].classid;
        append_wire[i * 2]     = (cid == MASK_LITERAL_CPU)
                                 ? 0xffu
                                 : (unsigned char)(cid & 0xff);
        append_wire[i * 2 + 1] = MaskAppendPattern[i].literal;
    }
    for (int c = 0; c < MASK_MAX_CLASSES_CPU; c++) {
        counts_p[c] = (uint32_t)MaskClasses[c].count;
        if (MaskClasses[c].count > 0) {
            memcpy(charsets_p + (size_t)c * 256,
                   MaskClasses[c].chars,
                   (size_t)MaskClasses[c].count);
        }
    }

    mtl_buf_kern_a_mask_allocated = 1;
    fprintf(stderr,
        "Metal: kernel A2 mask wire-format uploaded "
        "(prepend_len=%d append_len=%d total=%llu)\n",
        MaskPrependLen, MaskAppendLen,
        (unsigned long long)MaskTotal);
    return 0;
}

/* gpu_metal_kernelA_masks_dispatch -- A2 (masks-only) Metal dispatch
 * entry. Mirrors gpu_metal_kernelA_dispatch_proto (A1) shape but with
 * the 8-arg cand_masks_phase0 signature and the mask wire-format
 * buffers. v1 ships harness-mode: returns a non-NULL pointer on
 * dispatch fire (mtl_buf_proto_state contents), NULL otherwise;
 * *nhits_out always 0.
 *
 * Per feedback_metal_is_salted_op_widening.md note: kernel A2 v1
 * targets JOB_MD5MD5SALT (the canary op same as A1). The OpenCL
 * twin's is_salted_pack widening is a kernel B concern; A2 v1 does
 * not invoke kernel B and so does not require an analogous widening
 * here. When A2-with-kernelB lands the Metal salted-op admit list
 * (gpu/gpujob_metal.m:is_salted_op switch) will need extension. */
uint32_t *gpu_metal_kernelA_masks_dispatch(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out)
{
    if (nhits_out != NULL) *nhits_out = 0;

    if (gpu_metal_kernel_a_active_variant() != 2) return NULL;
    if (op != JOB_MD5MD5SALT) return NULL;
    if (!metal_ready || dev_idx != 0) return NULL;
    if (packed_words == NULL || packed_size == 0 ||
        word_offset == NULL || num_words == 0) return NULL;

    if (MaskPrependLen + MaskAppendLen == 0 || MaskTotal == 0) {
        static int _empty_warned = 0;
        if (!_empty_warned) {
            _empty_warned = 1;
            fprintf(stderr,
                "Metal GPU[%d]: kernel A2 dispatch invoked with no active "
                "mask (MaskPrependLen=%d MaskAppendLen=%d MaskTotal=%llu); "
                "falling through.\n",
                dev_idx, MaskPrependLen, MaskAppendLen,
                (unsigned long long)MaskTotal);
        }
        return NULL;
    }

    if (MaskTotal >= (1ULL << 31)) {
        GPU_FATAL("Metal GPU[%d]: kernel A2 v1 requires MaskTotal < 2^31 "
                  "(current=%llu); chunking not yet implemented (sub-phase 1a.2.x)",
                  dev_idx, (unsigned long long)MaskTotal);
    }

    if (metal_kernel_a_masks_pso_lazy() < 0) {
        return NULL;
    }
    if (gpu_metal_kernelA_upload_mask_buffers() < 0) {
        return NULL;
    }

    {
        static int _kernel_a2_first = 0;
        if (!_kernel_a2_first) {
            _kernel_a2_first = 1;
            fprintf(stderr,
                "Metal GPU[%d]: kernel A2 dispatch fired "
                "(variant=2 op=%d n_words=%u mask_total=%llu)\n",
                dev_idx, op, num_words,
                (unsigned long long)MaskTotal);
        }
    }

    uint32_t n_masks = (uint32_t)MaskTotal;

    /* Size + (re)allocate the prototype buffer trio (shared with A1). */
    size_t max_slots  = (size_t)num_words * (size_t)n_masks;
    size_t max_packed = max_slots * 256u;
    if (max_packed < METAL_MIN_BUFFER_BYTES) max_packed = METAL_MIN_BUFFER_BYTES;

    if (gpu_metal_kernelA_ensure_proto_bufs(max_packed, max_slots) < 0) {
        return NULL;
    }

    @autoreleasepool {
        size_t wo_size      = (size_t)num_words * sizeof(uint32_t);
        size_t payload_pkt  = 132 + wo_size;
        size_t payload_size = payload_pkt + packed_size;
        if (payload_size < METAL_MIN_BUFFER_BYTES) payload_size = METAL_MIN_BUFFER_BYTES;

        id<MTLBuffer> buf_payload =
            [mtl_device newBufferWithLength:payload_size
                                    options:MTLResourceStorageModeShared];
        if (buf_payload == nil) {
            GPU_FATAL("Metal: kernel A2 payload newBuffer(%zu) failed "
                      "(op=%d num_words=%u)",
                      payload_size, op, num_words);
        }

        uint8_t *p = (uint8_t *)[buf_payload contents];
        memset(p, 0, payload_size);

        OCLParams *params = (OCLParams *)p;
        params->num_words   = num_words;
        params->num_masks   = n_masks;
        params->n_prepend   = (uint32_t)MaskPrependLen;
        params->n_append    = (uint32_t)MaskAppendLen;
        params->num_rules   = 0;
        params->max_probe   = 256u;
        params->max_iter    = 1u;
        params->packed_size = (uint32_t)max_packed;

        memcpy(p + 132, word_offset, wo_size);
        memcpy(p + 132 + wo_size, packed_words, packed_size);

        /* Zero state buffer pre-dispatch. */
        memset([mtl_buf_proto_state contents], 0, 3u * sizeof(uint32_t));

        /* Encode + dispatch kernel A2 (8-arg signature).
         *   arg 0 payload
         *   arg 1 mask_pattern_prepend
         *   arg 2 mask_pattern_append
         *   arg 3 mask_charsets
         *   arg 4 mask_class_counts
         *   arg 5 b_packed_buf
         *   arg 6 b_chunk_index
         *   arg 7 b_kernelA_state (device atomic_uint *) */
        const NSUInteger tg_size_x = 64;
        NSUInteger global_threads = (NSUInteger)num_words * (NSUInteger)n_masks;
        NSUInteger padded = ((global_threads + tg_size_x - 1) / tg_size_x) * tg_size_x;
        NSUInteger num_groups = padded / tg_size_x;

        MTLSize threadgroup = MTLSizeMake(tg_size_x, 1, 1);
        MTLSize grid        = MTLSizeMake(num_groups, 1, 1);

        id<MTLCommandBuffer> cb = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mtl_pso_kernel_a_masks];

        [enc setBuffer:buf_payload                   offset:0 atIndex:0];
        [enc setBuffer:mtl_buf_kern_a_mask_prepend   offset:0 atIndex:1];
        [enc setBuffer:mtl_buf_kern_a_mask_append    offset:0 atIndex:2];
        [enc setBuffer:mtl_buf_kern_a_mask_charsets  offset:0 atIndex:3];
        [enc setBuffer:mtl_buf_kern_a_mask_counts    offset:0 atIndex:4];
        [enc setBuffer:mtl_buf_proto_packed          offset:0 atIndex:5];
        [enc setBuffer:mtl_buf_proto_chunk_index     offset:0 atIndex:6];
        [enc setBuffer:mtl_buf_proto_state           offset:0 atIndex:7];

        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.error != nil) {
            MTL_FATAL_NSERR(cb.error,
                "Metal kernel A2 dispatch error op=%d num_words=%u n_masks=%u",
                op, num_words, n_masks);
        }

        uint32_t state[3] = {0, 0, 0};
        memcpy(state, [mtl_buf_proto_state contents], 3u * sizeof(uint32_t));
        uint32_t actual_slots      = state[0];
        uint32_t actual_byte_count = state[1];
        uint32_t overflow_flag     = state[2];

        if (!overflow_flag && actual_slots > 0) {
            if (actual_byte_count > h_proto_packed_readback_cap) {
                free(h_proto_packed_readback);
                h_proto_packed_readback = calloc(1, actual_byte_count + METAL_MIN_BUFFER_BYTES);
                if (h_proto_packed_readback == NULL) {
                    GPU_FATAL("Metal: A2 h_proto_packed_readback calloc(%u) failed", actual_byte_count);
                }
                h_proto_packed_readback_cap = actual_byte_count + METAL_MIN_BUFFER_BYTES;
            }
            size_t index_bytes = (size_t)actual_slots * sizeof(uint32_t);
            if (index_bytes > h_proto_index_readback_cap) {
                free(h_proto_index_readback);
                h_proto_index_readback = calloc(1, index_bytes + METAL_MIN_BUFFER_BYTES);
                if (h_proto_index_readback == NULL) {
                    GPU_FATAL("Metal: A2 h_proto_index_readback calloc(%zu) failed", index_bytes);
                }
                h_proto_index_readback_cap = index_bytes + METAL_MIN_BUFFER_BYTES;
            }
            memcpy(h_proto_packed_readback,
                   [mtl_buf_proto_packed contents],
                   (size_t)actual_byte_count);
            memcpy(h_proto_index_readback,
                   [mtl_buf_proto_chunk_index contents],
                   index_bytes);
        }

        gpu_metal_kernel_a_trace_dump(dev_idx, state,
            h_proto_index_readback,
            (size_t)actual_slots * sizeof(uint32_t),
            h_proto_packed_readback,
            (size_t)actual_byte_count);

        if (overflow_flag) {
            fprintf(stderr,
                "[kernelA2_proto] Metal GPU[%d]: kernel A2 overflow "
                "(slots=%u bytes=%u packed_cap=%zu) -- A2-only ship; no retry\n",
                dev_idx, actual_slots, actual_byte_count, max_packed);
        }
    }  /* autoreleasepool */

    /* Sub-phase 1a.2 v1: harness-mode. *nhits_out stays 0; the non-NULL
     * return signals "dispatch fired" to the caller. The mtl_buf_proto_-
     * state contents pointer is never dereferenced for hit data here --
     * the trace channel is the validation surface. */
    return (uint32_t *)[mtl_buf_proto_state contents];
}

/* ====================================================================
 * Phase 1a sub-phase 1a.3 (2026-05-21): kernel A3 (rules + masks) Metal
 * dispatcher infrastructure. Metal twin of gpu/gpu_opencl.c:gpu_opencl_-
 * kernel_a_rules_masks_dispatch. Compositional product of A1+A2: reuses
 * buf_rule_program/buf_rule_offset (A1) and the four mtl_buf_kern_a_-
 * mask_* buffers (A2). No new persistent MTLBuffers.
 *
 * Per D10.1.a: host sets params.num_rules = gpu_n_rules - 1 (source
 * count); kernel adds +1 internally for the no-rule fast path. Per
 * D10.2.a: slot order producer-defined; harness multiset-compares.
 *
 * v1 ship: single-shot dispatch; cursor-restart chunking is sub-phase
 * 1a.3.x territory.
 * ==================================================================== */

/* gpu_metal_kernelA_rules_masks_dispatch -- A3 (rules+masks) Metal
 * dispatch entry. Mirrors OpenCL twin gpu_opencl_kernel_a_rules_masks_-
 * dispatch shape but with the 10-arg cand_rules_masks_phase0 signature.
 * v1 ships harness-mode: returns non-NULL on dispatch fire, NULL when
 * gates miss; *nhits_out always 0.
 *
 * Per feedback_metal_is_salted_op_widening.md note: A3 v1 targets
 * JOB_MD5MD5SALT only and does not invoke kernel B; the OpenCL twin's
 * is_salted_pack widening is a kernel-B concern, not relevant here. */
uint32_t *gpu_metal_kernelA_rules_masks_dispatch(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out)
{
    if (nhits_out != NULL) *nhits_out = 0;

    if (gpu_metal_kernel_a_active_variant() != 3) return NULL;
    if (op != JOB_MD5MD5SALT) return NULL;
    if (!metal_ready || dev_idx != 0) return NULL;
    if (packed_words == NULL || packed_size == 0 ||
        word_offset == NULL || num_words == 0) return NULL;

    /* Rules + masks both required for A3. */
    if (gpu_rule_count <= 0 || gpu_rule_program == NULL ||
        gpu_rule_offsets == NULL || gpu_rule_program_len == 0) {
        return NULL;
    }
    if (MaskPrependLen + MaskAppendLen == 0 || MaskTotal == 0) {
        static int _empty_warned = 0;
        if (!_empty_warned) {
            _empty_warned = 1;
            fprintf(stderr,
                "Metal GPU[%d]: kernel A3 dispatch invoked with no active "
                "mask (MaskPrependLen=%d MaskAppendLen=%d MaskTotal=%llu); "
                "falling through.\n",
                dev_idx, MaskPrependLen, MaskAppendLen,
                (unsigned long long)MaskTotal);
        }
        return NULL;
    }

    if (MaskTotal >= (1ULL << 31)) {
        GPU_FATAL("Metal GPU[%d]: kernel A3 v1 requires MaskTotal < 2^31 "
                  "(current=%llu); cursor-restart chunking not yet "
                  "implemented (sub-phase 1a.3.x)",
                  dev_idx, (unsigned long long)MaskTotal);
    }

    if (metal_kernel_a_rules_masks_pso_lazy() < 0) {
        return NULL;
    }
    if (gpu_metal_kernelA_upload_mask_buffers() < 0) {
        return NULL;
    }

    /* Ensure rule_program / rule_offset MTLBuffers uploaded (reuses
     * the A1 helper -- idempotent). */
    if (metal_upload_rules_lazy() < 0) {
        GPU_FATAL("Metal: metal_upload_rules_lazy failed in kernel A3 "
                  "dispatch (op=%d rule_count=%d program_len=%u)",
                  op, gpu_rule_count, gpu_rule_program_len);
    }

    {
        static int _kernel_a3_first = 0;
        if (!_kernel_a3_first) {
            _kernel_a3_first = 1;
            fprintf(stderr,
                "Metal GPU[%d]: kernel A3 dispatch fired "
                "(variant=3 op=%d n_words=%u n_rules=%d mask_total=%llu)\n",
                dev_idx, op, num_words, gpu_rule_count,
                (unsigned long long)MaskTotal);
        }
    }

    /* gpu_rule_count includes the synthetic ':' prepend rule; per D10.1.a
     * the kernel handles the no-rule pass internally via rule_idx==0
     * fast path. We pass num_rules = gpu_rule_count - 1 (= Numrules);
     * the synth ':' at rule_offset[0] is still visited at rule_idx=1 but
     * the kernel's no-op detector skips the duplicate. */
    uint32_t n_rules = (uint32_t)(gpu_rule_count > 0 ? gpu_rule_count - 1 : 0);
    uint32_t n_masks = (uint32_t)MaskTotal;

    /* Size + (re)allocate the prototype buffer trio (shared with A1/A2). */
    size_t max_slots  = (size_t)num_words * (size_t)(n_rules + 1u)
                         * (size_t)n_masks;
    size_t max_packed = max_slots * 256u;
    if (max_packed < METAL_MIN_BUFFER_BYTES) max_packed = METAL_MIN_BUFFER_BYTES;

    if (gpu_metal_kernelA_ensure_proto_bufs(max_packed, max_slots) < 0) {
        return NULL;
    }

    @autoreleasepool {
        size_t wo_size      = (size_t)num_words * sizeof(uint32_t);
        size_t payload_pkt  = 132 + wo_size;
        size_t payload_size = payload_pkt + packed_size;
        if (payload_size < METAL_MIN_BUFFER_BYTES) payload_size = METAL_MIN_BUFFER_BYTES;

        id<MTLBuffer> buf_payload =
            [mtl_device newBufferWithLength:payload_size
                                    options:MTLResourceStorageModeShared];
        if (buf_payload == nil) {
            GPU_FATAL("Metal: kernel A3 payload newBuffer(%zu) failed "
                      "(op=%d num_words=%u)",
                      payload_size, op, num_words);
        }

        uint8_t *p = (uint8_t *)[buf_payload contents];
        memset(p, 0, payload_size);

        OCLParams *params = (OCLParams *)p;
        params->num_words   = num_words;
        params->num_rules   = n_rules;            /* D10.1.a source count */
        params->num_masks   = n_masks;            /* MaskTotal */
        params->n_prepend   = (uint32_t)MaskPrependLen;
        params->n_append    = (uint32_t)MaskAppendLen;
        params->max_probe   = 256u;
        params->max_iter    = 1u;
        params->packed_size = (uint32_t)max_packed;
        /* Cursor axes -- v1 single-shot; all start at 0. */
        params->input_cursor_start = 0;
        params->rule_cursor_start  = 0;
        params->mask_start         = 0;

        memcpy(p + 132, word_offset, wo_size);
        memcpy(p + 132 + wo_size, packed_words, packed_size);

        /* Zero state buffer pre-dispatch. */
        memset([mtl_buf_proto_state contents], 0, 3u * sizeof(uint32_t));

        /* Encode + dispatch kernel A3 (10-arg signature).
         *   arg 0 payload
         *   arg 1 rule_program
         *   arg 2 rule_offset
         *   arg 3 mask_pattern_prepend
         *   arg 4 mask_pattern_append
         *   arg 5 mask_charsets
         *   arg 6 mask_class_counts
         *   arg 7 b_packed_buf
         *   arg 8 b_chunk_index
         *   arg 9 b_kernelA_state (device atomic_uint *) */
        const NSUInteger tg_size_x = 64;
        NSUInteger global_threads = (NSUInteger)num_words *
                                    (NSUInteger)(n_rules + 1u);
        NSUInteger padded = ((global_threads + tg_size_x - 1) / tg_size_x) * tg_size_x;
        NSUInteger num_groups = padded / tg_size_x;

        MTLSize threadgroup = MTLSizeMake(tg_size_x, 1, 1);
        MTLSize grid        = MTLSizeMake(num_groups, 1, 1);

        id<MTLCommandBuffer> cb = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mtl_pso_kernel_a_rules_masks];

        [enc setBuffer:buf_payload                   offset:0 atIndex:0];
        [enc setBuffer:buf_rule_program              offset:0 atIndex:1];
        [enc setBuffer:buf_rule_offset               offset:0 atIndex:2];
        [enc setBuffer:mtl_buf_kern_a_mask_prepend   offset:0 atIndex:3];
        [enc setBuffer:mtl_buf_kern_a_mask_append    offset:0 atIndex:4];
        [enc setBuffer:mtl_buf_kern_a_mask_charsets  offset:0 atIndex:5];
        [enc setBuffer:mtl_buf_kern_a_mask_counts    offset:0 atIndex:6];
        [enc setBuffer:mtl_buf_proto_packed          offset:0 atIndex:7];
        [enc setBuffer:mtl_buf_proto_chunk_index     offset:0 atIndex:8];
        [enc setBuffer:mtl_buf_proto_state           offset:0 atIndex:9];

        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.error != nil) {
            MTL_FATAL_NSERR(cb.error,
                "Metal kernel A3 dispatch error op=%d num_words=%u "
                "n_rules=%u n_masks=%u",
                op, num_words, n_rules, n_masks);
        }

        uint32_t state[3] = {0, 0, 0};
        memcpy(state, [mtl_buf_proto_state contents], 3u * sizeof(uint32_t));
        uint32_t actual_slots      = state[0];
        uint32_t actual_byte_count = state[1];
        uint32_t overflow_flag     = state[2];

        if (!overflow_flag && actual_slots > 0) {
            if (actual_byte_count > h_proto_packed_readback_cap) {
                free(h_proto_packed_readback);
                h_proto_packed_readback = calloc(1, actual_byte_count + METAL_MIN_BUFFER_BYTES);
                if (h_proto_packed_readback == NULL) {
                    GPU_FATAL("Metal: A3 h_proto_packed_readback calloc(%u) failed", actual_byte_count);
                }
                h_proto_packed_readback_cap = actual_byte_count + METAL_MIN_BUFFER_BYTES;
            }
            size_t index_bytes = (size_t)actual_slots * sizeof(uint32_t);
            if (index_bytes > h_proto_index_readback_cap) {
                free(h_proto_index_readback);
                h_proto_index_readback = calloc(1, index_bytes + METAL_MIN_BUFFER_BYTES);
                if (h_proto_index_readback == NULL) {
                    GPU_FATAL("Metal: A3 h_proto_index_readback calloc(%zu) failed", index_bytes);
                }
                h_proto_index_readback_cap = index_bytes + METAL_MIN_BUFFER_BYTES;
            }
            memcpy(h_proto_packed_readback,
                   [mtl_buf_proto_packed contents],
                   (size_t)actual_byte_count);
            memcpy(h_proto_index_readback,
                   [mtl_buf_proto_chunk_index contents],
                   index_bytes);
        }

        gpu_metal_kernel_a_trace_dump(dev_idx, state,
            h_proto_index_readback,
            (size_t)actual_slots * sizeof(uint32_t),
            h_proto_packed_readback,
            (size_t)actual_byte_count);

        if (overflow_flag) {
            fprintf(stderr,
                "[kernelA3_proto] Metal GPU[%d]: kernel A3 overflow "
                "(slots=%u bytes=%u packed_cap=%zu) -- A3 v1 single-shot; "
                "cursor-restart chunking is sub-phase 1a.3.x\n",
                dev_idx, actual_slots, actual_byte_count, max_packed);
        }
    }  /* autoreleasepool */

    /* Sub-phase 1a.3 v1: harness-mode. *nhits_out stays 0; non-NULL
     * return signals "dispatch fired" to the caller. */
    return (uint32_t *)[mtl_buf_proto_state contents];
}

/* Phase 1a sub-phase 1a.4 (2026-05-21): A4 (brute-force) Metal dispatch
 * entry. Mirrors OpenCL twin gpu_opencl_kernel_a_bruteforce_dispatch shape
 * with the 8-arg cand_bruteforce_phase0 signature (byte-identical to A2;
 * mask_pattern_prepend slot bound to the zeroed mtl_buf_kern_a_mask_-
 * prepend buffer since BF invariant is MaskPrependLen == 0).
 *
 * Signature differs from A1/A2/A3: takes struct jobg *g directly (no
 * packed_words / word_offset / num_words args). Reads g->bf_chunk,
 * g->bf_mask_start, g->bf_offset_per_word, g->bf_num_masks, g->packed_count
 * + global MaskAppendLen + MaskAppendPattern + MaskClasses[].
 *
 * v1 ships harness-mode: returns non-NULL on dispatch fire, NULL when
 * gates miss; *nhits_out always 0. */
uint32_t *gpu_metal_kernelA_bruteforce_dispatch(int dev_idx,
    struct jobg *g, int *nhits_out)
{
    if (nhits_out != NULL) *nhits_out = 0;

    if (gpu_metal_kernel_a_active_variant() != 4) return NULL;
    if (!metal_ready || dev_idx != 0) return NULL;
    if (g == NULL) return NULL;

    /* BF chunk gate. */
    if (!g->bf_chunk) return NULL;

    /* Append-mask must be present (BF candidate IS the mask expansion). */
    if (MaskAppendLen == 0) {
        static int _empty_warned = 0;
        if (!_empty_warned) {
            _empty_warned = 1;
            fprintf(stderr,
                "Metal GPU[%d]: kernel A4 dispatch invoked with no active "
                "append mask (MaskAppendLen=0); falling through.\n",
                dev_idx);
        }
        return NULL;
    }

    /* BF invariant: prepend mask MUST be zero (per spec R2 + chunk-producer
     * fast-path gate at mdxfind.c:49032). Violation = FATAL per
     * feedback_external_failures_are_fatal.md. */
    if (MaskPrependLen != 0) {
        GPU_FATAL("Metal GPU[%d]: kernel A4 BF invariant violation: "
                  "MaskPrependLen=%d (must be 0 for BF chunks)",
                  dev_idx, MaskPrependLen);
    }

    if (metal_kernel_a_bruteforce_pso_lazy() < 0) {
        return NULL;
    }
    if (gpu_metal_kernelA_upload_mask_buffers() < 0) {
        return NULL;
    }

    /* Synthetic lane count: pre-sized by BF chunk producer
     * (mdxfind.c:10498-10506). g->packed_count is the synthetic lane
     * count for BF chunks. */
    uint32_t num_words    = g->packed_count;
    uint32_t bf_num_masks = g->bf_num_masks;
    if (num_words == 0 || bf_num_masks == 0) return NULL;

    {
        static int _kernel_a4_first = 0;
        if (!_kernel_a4_first) {
            _kernel_a4_first = 1;
            fprintf(stderr,
                "Metal GPU[%d]: kernel A4 dispatch fired "
                "(variant=4 op=%d num_words=%u bf_mask_start=%llu "
                "bf_num_masks=%u bf_offset_per_word=%u app_len=%d)\n",
                dev_idx, g->op, num_words,
                (unsigned long long)g->bf_mask_start,
                bf_num_masks, g->bf_offset_per_word,
                MaskAppendLen);
        }
    }

    /* Worst-case slot count = num_words * bf_num_masks. Per spec §6
     * max_packed = slots * (1 + MaskAppendLen) <= 256 MB. Host caps and
     * FATAL-exits per R5. */
    static const size_t MAX_KERNEL_A_PACKED_METAL = (size_t)256 * 1024u * 1024u;
    size_t max_slots  = (size_t)num_words * (size_t)bf_num_masks;
    size_t max_packed = max_slots * (size_t)(1u + (uint32_t)MaskAppendLen);
    if (max_packed > MAX_KERNEL_A_PACKED_METAL) {
        GPU_FATAL("Metal GPU[%d]: kernel A4 max_packed=%zu B exceeds cap=%zu B "
                  "(num_words=%u bf_num_masks=%u app_len=%d); host-side "
                  "adaptive_bf_chunk_size produced an oversize chunk",
                  dev_idx, max_packed, MAX_KERNEL_A_PACKED_METAL,
                  num_words, bf_num_masks, MaskAppendLen);
    }
    if (max_packed < METAL_MIN_BUFFER_BYTES) max_packed = METAL_MIN_BUFFER_BYTES;

    if (gpu_metal_kernelA_ensure_proto_bufs(max_packed, max_slots) < 0) {
        return NULL;
    }

    @autoreleasepool {
        /* Payload is 132 bytes (no word_offset/packed_words tail per spec §6 / R1). */
        size_t payload_size = 132;
        if (payload_size < METAL_MIN_BUFFER_BYTES) payload_size = METAL_MIN_BUFFER_BYTES;

        id<MTLBuffer> buf_payload =
            [mtl_device newBufferWithLength:payload_size
                                    options:MTLResourceStorageModeShared];
        if (buf_payload == nil) {
            GPU_FATAL("Metal: kernel A4 payload newBuffer(%zu) failed "
                      "(op=%d num_words=%u)",
                      payload_size, g->op, num_words);
        }

        uint8_t *p = (uint8_t *)[buf_payload contents];
        memset(p, 0, payload_size);

        OCLParams *params = (OCLParams *)p;
        params->mask_start          = g->bf_mask_start;
        params->num_words           = num_words;
        params->num_masks           = bf_num_masks;
        params->n_prepend           = 0u;                       /* BF invariant */
        params->n_append            = (uint32_t)MaskAppendLen;
        params->max_probe           = 256u;
        params->max_iter            = 1u;
        params->packed_size         = (uint32_t)max_packed;
        params->input_cursor_start  = 0u;
        params->rule_cursor_start   = 0u;
        params->inner_iter          = 1u;                       /* v1: ignored by kernel */
        params->mask_offset_per_word = g->bf_offset_per_word;

        /* Zero state buffer pre-dispatch. */
        memset([mtl_buf_proto_state contents], 0, 3u * sizeof(uint32_t));

        /* Encode + dispatch kernel A4 (8-arg signature byte-identical to A2):
         *   arg 0 payload
         *   arg 1 mask_pattern_prepend (zeroed; BF invariant)
         *   arg 2 mask_pattern_append
         *   arg 3 mask_charsets
         *   arg 4 mask_class_counts
         *   arg 5 b_packed_buf
         *   arg 6 b_chunk_index
         *   arg 7 b_kernelA_state (device atomic_uint *) */
        const NSUInteger tg_size_x = 64;
        NSUInteger global_threads = (NSUInteger)num_words *
                                    (NSUInteger)bf_num_masks;
        NSUInteger padded = ((global_threads + tg_size_x - 1) / tg_size_x) * tg_size_x;
        NSUInteger num_groups = padded / tg_size_x;

        MTLSize threadgroup = MTLSizeMake(tg_size_x, 1, 1);
        MTLSize grid        = MTLSizeMake(num_groups, 1, 1);

        id<MTLCommandBuffer> cb = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:mtl_pso_kernel_a_bruteforce];

        [enc setBuffer:buf_payload                   offset:0 atIndex:0];
        [enc setBuffer:mtl_buf_kern_a_mask_prepend   offset:0 atIndex:1];
        [enc setBuffer:mtl_buf_kern_a_mask_append    offset:0 atIndex:2];
        [enc setBuffer:mtl_buf_kern_a_mask_charsets  offset:0 atIndex:3];
        [enc setBuffer:mtl_buf_kern_a_mask_counts    offset:0 atIndex:4];
        [enc setBuffer:mtl_buf_proto_packed          offset:0 atIndex:5];
        [enc setBuffer:mtl_buf_proto_chunk_index     offset:0 atIndex:6];
        [enc setBuffer:mtl_buf_proto_state           offset:0 atIndex:7];

        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.error != nil) {
            MTL_FATAL_NSERR(cb.error,
                "Metal kernel A4 dispatch error op=%d num_words=%u "
                "bf_num_masks=%u",
                g->op, num_words, bf_num_masks);
        }

        uint32_t state[3] = {0, 0, 0};
        memcpy(state, [mtl_buf_proto_state contents], 3u * sizeof(uint32_t));
        uint32_t actual_slots      = state[0];
        uint32_t actual_byte_count = state[1];
        uint32_t overflow_flag     = state[2];

        if (!overflow_flag && actual_slots > 0) {
            if (actual_byte_count > h_proto_packed_readback_cap) {
                free(h_proto_packed_readback);
                h_proto_packed_readback = calloc(1, actual_byte_count + METAL_MIN_BUFFER_BYTES);
                if (h_proto_packed_readback == NULL) {
                    GPU_FATAL("Metal: A4 h_proto_packed_readback calloc(%u) failed", actual_byte_count);
                }
                h_proto_packed_readback_cap = actual_byte_count + METAL_MIN_BUFFER_BYTES;
            }
            size_t index_bytes = (size_t)actual_slots * sizeof(uint32_t);
            if (index_bytes > h_proto_index_readback_cap) {
                free(h_proto_index_readback);
                h_proto_index_readback = calloc(1, index_bytes + METAL_MIN_BUFFER_BYTES);
                if (h_proto_index_readback == NULL) {
                    GPU_FATAL("Metal: A4 h_proto_index_readback calloc(%zu) failed", index_bytes);
                }
                h_proto_index_readback_cap = index_bytes + METAL_MIN_BUFFER_BYTES;
            }
            memcpy(h_proto_packed_readback,
                   [mtl_buf_proto_packed contents],
                   (size_t)actual_byte_count);
            memcpy(h_proto_index_readback,
                   [mtl_buf_proto_chunk_index contents],
                   index_bytes);
        }

        gpu_metal_kernel_a_trace_dump(dev_idx, state,
            h_proto_index_readback,
            (size_t)actual_slots * sizeof(uint32_t),
            h_proto_packed_readback,
            (size_t)actual_byte_count);

        if (overflow_flag) {
            GPU_FATAL("Metal GPU[%d]: kernel A4 overflow "
                      "(slots=%u bytes=%u packed_cap=%zu); A4 v1 host-side "
                      "cap should have caught this -- check adaptive_bf_-"
                      "chunk_size sizing",
                      dev_idx, actual_slots, actual_byte_count, max_packed);
        }
    }  /* autoreleasepool */

    /* Sub-phase 1a.4 v1: harness-mode. *nhits_out stays 0; non-NULL
     * return signals "dispatch fired" to the caller. */
    return (uint32_t *)[mtl_buf_proto_state contents];
}

/* Task #250: grow buf_scratch_pool to hold at least `need_words` slots
 * of RULE_BUF_MAX bytes each. Re-allocates when current capacity is
 * insufficient (the typical case is num_words==WORD_BATCH==16384 per
 * job; on the very first dispatch capacity is 0 so this allocates the
 * full 640 MB peak). Subsequent dispatches reuse the buffer.
 *
 * Storage mode is Private — the kernel reads + writes it but the host
 * never accesses contents. Private avoids the CPU-shared cache-line
 * ping-pong that Shared mode incurs (Apple Silicon unified memory has
 * one physical pool but two cache hierarchies; Private keeps the buffer
 * in the GPU-side cache exclusively).
 *
 * Returns 0 on success, -1 on allocation failure. */
static int metal_ensure_buf_scratch_pool(uint32_t need_words)
{
    if (need_words == 0) need_words = 1;
    if (buf_scratch_pool != nil && buf_scratch_pool_words_cap >= need_words) {
        return 0;  /* current capacity sufficient */
    }

    /* Drop the existing buffer if it exists at a smaller capacity. */
    buf_scratch_pool = nil;

    size_t bytes = (size_t)need_words * (size_t)METAL_RULE_BUF_MAX;
    if (bytes < METAL_MIN_BUFFER_BYTES) bytes = METAL_MIN_BUFFER_BYTES;

    buf_scratch_pool =
        [mtl_device newBufferWithLength:bytes
                                options:MTLResourceStorageModePrivate];
    if (buf_scratch_pool == nil) {
        /* Phase D5a (Task #281): alloc failure -> fatal. */
        GPU_FATAL("Metal: buf_scratch_pool newBuffer(%zu bytes / %u words) failed",
                  bytes, need_words);
    }
    buf_scratch_pool_words_cap = need_words;

    /* One-shot marker so the operator sees the pool came up. Mirrors the
     * "PSO ... created lazily" idiom. */
    GPU_DEBUG_FPRINTF(stderr,
            "Metal: buf_scratch_pool allocated (%zu bytes / %u words / RULE_BUF_MAX=%u)\n",
            bytes, need_words, (unsigned)METAL_RULE_BUF_MAX);
    return 0;
}

uint32_t *gpu_metal_dispatch_md5_rules(int dev_idx,
    const char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    int op, int *nhits_out,
    uint64_t mask_start, uint32_t mask_offset_per_word, uint32_t bf_num_masks,
    uint32_t inner_iter,
    int bf_fast_eligible)
{
    (void)packed_size;
    (void)bf_num_masks;
    (void)bf_fast_eligible;

    *nhits_out = 0;
    if (!metal_ready || dev_idx != 0) return NULL;
    if (packed_words == NULL || word_offset == NULL || num_words == 0)
        return NULL;
    /* Phase 2d.2.1a: op-gate via registered-family lookup. Replaces the
     * Phase 2c hardcoded `op != JOB_MD5 && op != JOB_MD5SALT` whitelist.
     * Future families register themselves at init time and become
     * admissible here automatically. */
    struct gpu_metal_family *fam = gpu_metal_lookup_family(op);
    if (fam == NULL) return NULL;
    if (buf_compact_fp == nil || buf_compact_idx == nil) return NULL;

    /* Phase 2a-cont marker per feedback_verify_gpu_fired_post_build.md.
     * Byte-exact CPU/Metal alone is insufficient — silent CPU fallback can
     * produce identical cracks. This one-shot stderr emit confirms the
     * dispatch path actually fires. Phase 2b extended the marker with
     * mask info; Phase 2c extends further with salt info so the operator
     * sees which of the eight PSO variants fired. */
    static int first_dispatch_logged = 0;
    int use_rules_first = (gpu_rule_count > 0
                           && gpu_rule_program != NULL
                           && gpu_rule_offsets != NULL
                           && gpu_rule_program_len > 0);
    if (!first_dispatch_logged) {
        first_dispatch_logged = 1;
        GPU_DEBUG_FPRINTF(stderr,
                "Metal GPU[0]: first dispatch issued (rules=%d, mask=%llu, salts=%d, op=%d)\n",
                use_rules_first ? gpu_rule_count : 0,
                (unsigned long long)gpu_mask_total,
                cached_salts_count,
                op);
    }

    @autoreleasepool {
        /* Phase 2a row 6: PSO + rule-data selection.
         *
         * Rules mode is selected when gpu_rule_count > 0 (the synthetic
         * no-rule pass alone counts as 1 — mdxfind.c always uploads at
         * least the synthetic `\0` program when GPU is active). The
         * no-rules PSO is kept for the truly-Phase-1 case where
         * gpu_rule_count is 0 (e.g., harness smoke tests that bypass
         * mdxfind's rule-load path). */
        int use_rules = (gpu_rule_count > 0
                         && gpu_rule_program != NULL
                         && gpu_rule_offsets != NULL
                         && gpu_rule_program_len > 0);

        /* Phase 2b row 4: mask mode is selected when gpu_metal_set_mask
         * has been called AND the resulting combinatorial space is > 1
         * (i.e., at least one position has size > 1). When gpu_mask_total
         * == 1 the kernel collapses to the (word, rule) shape and binding
         * the mask buffers would still work, but selecting the cheaper
         * non-mask PSO avoids the JIT-compile cost when no real mask
         * is active. */
        int use_mask = (gpu_mask_total > 1
                        && buf_mask_charsets != nil
                        && buf_mask_sizes    != nil
                        && (gpu_mask_n_prepend + gpu_mask_n_append) >= 1);

        /* Phase 2c row 1: salt mode is selected when gpu_metal_set_salt
         * has been called for this batch (cached_salts_count > 0) AND
         * the op is a salt-template op. Phase 2c admitted JOB_MD5SALT
         * only; Phase 2d.2.4 widened to JOB_MD5PASSSALT; Phase 2d.2.5
         * widened to JOB_MD5SALTPASS (LAST md5-family fan-out entry --
         * MD5(salt || pass) PREPEND shape, mirror image of md5passsalt).
         * Phase 2d.3.4 widens to JOB_SHA1PASSSALT (FIRST SHA-family
         * salted port -- SHA-1(pass || salt) APPEND); Phase 2d.3.5
         * widens to JOB_SHA1SALTPASS (SHA-1(salt || pass) PREPEND).
         * Future 2d siblings extend this list one op at a time. */
        int use_salt = (cached_salts_count > 0
                        && buf_salt_data != nil
                        && buf_salt_off  != nil
                        && buf_salt_lens != nil
                        && (op == JOB_MD5SALT
                            || op == JOB_MD5PASSSALT
                            || op == JOB_MD5SALTPASS
                            || op == JOB_SHA1PASSSALT
                            || op == JOB_SHA1SALTPASS
                            || op == JOB_SHA256PASSSALT
                            || op == JOB_SHA256SALTPASS
                            || op == JOB_SHA224SALTPASS
                            || op == JOB_SHA512PASSSALT
                            || op == JOB_SHA512SALTPASS
                            || op == JOB_SHA384SALTPASS
                            /* Phase 2d.7d HMAC siblings: 5 ops via 3 carrier
                             * kernels. All salted (GPU_CAT_MASK with salt
                             * upload via standard salt_buf/salt_off/salt_lens
                             * trio). Per feedback_hmac_salt_judy_typeopt.md,
                             * the STREEBOG HMAC family routes BOTH KSALT and
                             * KPASS via Typesalt (TYPEOPT_NEEDSALT) -- distinct
                             * from Families A-H. Salt-routing decision lives
                             * in gpu_salt_judy() (gpujob_metal.m). */
                            || op == JOB_HMAC_BLAKE2S
                            || op == JOB_HMAC_STREEBOG256_KSALT
                            || op == JOB_HMAC_STREEBOG256_KPASS
                            || op == JOB_HMAC_STREEBOG512_KSALT
                            || op == JOB_HMAC_STREEBOG512_KPASS
                            /* Phase 2d.8a PHPBB3 + MD5CRYPT: both
                             * salted-only iterated-MD5 ops. Salt buffer
                             * carries algorithm-specific prefix
                             * ("$H$<cost><8>" for PHPBB3; "$1$<salt>$"
                             * for MD5CRYPT). Hit-replay arms in
                             * gpu/gpujob_metal.m route via checkhashbb
                             * (PHPBB3) and hybrid_check+md5crypt_b64encode
                             * (MD5CRYPT) -- NOT through the standard
                             * checkhashsalt/checkhashkey paths. */
                            || op == JOB_PHPBB3
                            || op == JOB_MD5CRYPT
                            /* Phase 2d.8b SHACRYPT triple: SHA256CRYPT
                             * (op=512), SHA512CRYPT (op=513), and
                             * SHA512CRYPTMD5 (op=538). All salted-only
                             * (Typesalt holds the FULL "$5$..." / "$6$..."
                             * hash line including b64 tail; b64-tail strip
                             * happens in hit-replay via last-`$` scan).
                             * Hit-replay arms in gpu/gpujob_metal.m route
                             * via hybrid_check + sha{256,512}crypt_-
                             * b64encode + prfound (mirrors gpujob_opencl.c
                             * 2031-2073 / 2127-2159 byte-for-byte) --
                             * SHA512CRYPTMD5 shares the SHA512CRYPT arm
                             * (same $6$ output format, same Typesalt). */
                            || op == JOB_SHA256CRYPT
                            || op == JOB_SHA512CRYPT
                            || op == JOB_SHA512CRYPTMD5
                            /* Phase 2d.9a DESCRYPT (op=500): salted-only
                             * single-algo_mode (7) DES. Salt buffer carries
                             * the 2-char phpitoa64 salt prefix (saltlen=2;
                             * extended-DES `_CCCCSSSS` 9-char salts skipped
                             * at the salt-pack filter so they CPU-fallback
                             * through bsd_crypt_des). Hit-replay arm in
                             * gpu/gpujob_metal.m routes via metal_des_-
                             * reconstruct + JudyJ[JOB_DESCRYPT] lookup +
                             * prfound (mirrors gpujob_opencl.c DESCRYPT arm
                             * at line 2204-2232 byte-for-byte). NOT through
                             * checkhashbb/checkhashsalt/checkhashkey --
                             * DESCRYPT has its own bespoke 13-char crypt(3)
                             * output format. Phase 5 of Unix-crypt ladder
                             * on Metal -- LAST Unix-crypt op. */
                            || op == JOB_DESCRYPT
                            /* Phase 2d.9b BCRYPT (op=450): salted-only
                             * single-algo_mode (8) Eksblowfish. Salt buffer
                             * carries the full 60-char "$2[abxy]$NN$<salt><hash>"
                             * line per mdxfind.c BCRYPT loader; the kernel
                             * decodes cost from bytes [4..5] and the 22-char
                             * base64 salt at offset 7 (+16 raw bytes).
                             * Hit-replay arm in gpu/gpujob_metal.m
                             * reconstructs the 60-char crypt(3) hash via
                             * bf_encode_23 (duplicated with citation from
                             * gpujob_opencl.c) + JudyJ[JOB_BCRYPT] lookup +
                             * prfound. NOT through checkhashbb/salt/key --
                             * BCRYPT has its own bespoke 60-char crypt(3)
                             * output format. FINAL Phase 2d sub-phase --
                             * 52nd Metal family. */
                            || op == JOB_BCRYPT));

        /* Phase 2d.2.1a: per-family struct + op-keyed resolver. Replaces
         * the Phase 2c 9-arm if/else cascade. variant_bits encodes the
         * (R, M, S) axes; the registered family's pso_for_variant returns
         * an already-lazy-created PSO. PRESALT is folded inside the
         * md5salt resolver's V_S|V_R arm — NOT exposed in variant_bits.
         *
         * metal_upload_rules_lazy() is called here (post-resolve) when
         * V_R is active; it is idempotent and must run on every rules
         * dispatch so the rule_program / rule_offset MTLBuffers reflect
         * the current host-side gpu_rule_program. */
        uint8_t variant_bits = (uint8_t)(
                                (use_rules ? V_R : 0u)
                              | (use_mask  ? V_M : 0u)
                              | (use_salt  ? V_S : 0u));

        /* D5b Wave 1: prefer v2 resolver when present (migrated family).
         * Unmigrated families have pso_for_variant_v2 == NULL and we fall
         * back to the legacy 1-arg pointer. */
        void *pso_void = fam->pso_for_variant_v2
                           ? fam->pso_for_variant_v2(fam, variant_bits)
                           : fam->pso_for_variant(variant_bits);
        id<MTLComputePipelineState> active_pso =
            (__bridge id<MTLComputePipelineState>)pso_void;
        if (active_pso == nil) {
            /* Phase D5a (Task #281+#282) 2026-05-16: post-init lazy-PSO
             * resolution failure -> fatal. By the time dispatch runs, the
             * eager-compile loop (gpu_metal_compile_families, work item 5)
             * has already probed every (family, variant) admissible combo
             * and pruned the inadmissible ones from the dispatch path. So
             * a nil here means EITHER (a) an admitted variant regressed
             * post-init (driver state corruption) OR (b) admission-pruning
             * missed a combo (admission-prune bug). Both are runtime
             * failures: silently returning NULL drops words at
             * gpujob_metal_worker's return_jobg without CPU re-queue.
             * Same class of bug as Phase 2d.5 dev3 cb.error != nil case
             * (see work item 2). */
            GPU_FATAL("Metal: pso_for_variant returned nil at dispatch "
                      "(op=%d family=%s variant_bits=0x%x) -- post-init "
                      "PSO regression or admission-prune gap",
                      op, fam->name ? fam->name : "?",
                      (unsigned)variant_bits);
        }
        if (use_rules) {
            if (metal_upload_rules_lazy() < 0) {
                /* Phase D5a (Task #281): metal_upload_rules_lazy returns
                 * -1 on newBuffer failure (which is now itself fatal --
                 * see work item 3 rule_program/rule_offset sites) OR on
                 * missing rule globals (host-side invariant violation).
                 * Either way, post-init failure -> fatal. */
                GPU_FATAL("Metal: metal_upload_rules_lazy failed at dispatch "
                          "(op=%d rule_count=%d program_len=%u)",
                          op, gpu_rule_count, gpu_rule_program_len);
            }
        }

        /* Phase 2e: one-shot stderr marker reporting the SALT_BATCH
         * value baked into the JIT compile of the active presalt
         * library. Per feedback_verify_gpu_fired_post_build.md the
         * PRESALT PSO firing must be visible in stderr; silent fallback
         * to non-presalt would invalidate Gate B's perf measurement.
         *
         * Detection: the resolver returns the PRESALT PSO only for the
         * md5salt family at variant_bits == (V_S|V_R). We mirror that
         * here by comparing against the PRESALT PSO static — equivalent
         * to the Phase 2c `use_presalt` predicate but driven by the
         * resolver's actual return (which honors the PRESALT fallback
         * to non-PRESALT salt_rules if the PRESALT lazy create failed). */
        if (active_pso == mtl_pso_template_md5_salt_rules_presalt
            && mtl_pso_template_md5_salt_rules_presalt != nil) {
            static int presalt_marker_logged = 0;
            if (!presalt_marker_logged) {
                presalt_marker_logged = 1;
                uint32_t sb = metal_select_salt_batch(mtl_device);
                GPU_DEBUG_FPRINTF(stderr,
                        "Metal GPU[0]: first dispatch issued via PRESALT variant "
                        "(rules=%d, salts=%d, salt_batch=%u, op=%d)\n",
                        gpu_rule_count, cached_salts_count, (unsigned)sb, op);
            }
        }

        /* Task #250: ensure the device-side scratch pool is sized for the
         * current word count. All four PSO variants read buffer 14, so the
         * pool must exist before the first dispatch — regardless of which
         * variant is selected. */
        if (metal_ensure_buf_scratch_pool(num_words) < 0) return NULL;

        /* --- Payload layout (matches gpu_opencl.c b_dispatch_payload):
         *   offset 0..127     OCLParams
         *   offset 128..131   uint hit_count (kernel atomic)
         *   offset 132..      uint word_offset[num_words]
         *   offset 132+4N..   packed words (length byte + bytes)
         */
        size_t wo_size      = (size_t)num_words * sizeof(uint32_t);
        size_t payload_pkt  = 132 + wo_size;
        size_t payload_size = payload_pkt + packed_size;
        if (payload_size < METAL_MIN_BUFFER_BYTES)
            payload_size = METAL_MIN_BUFFER_BYTES;

        id<MTLBuffer> buf_payload =
            [mtl_device newBufferWithLength:payload_size
                                    options:MTLResourceStorageModeShared];
        if (buf_payload == nil) {
            /* Phase D5a (Task #281): alloc failure -> fatal. */
            GPU_FATAL("Metal: payload newBuffer(%zu bytes) failed (op=%d num_words=%u)",
                      payload_size, op, num_words);
        }

        uint8_t *p = (uint8_t *)[buf_payload contents];
        memset(p, 0, payload_size);

        OCLParams *params = (OCLParams *)p;
        params->compact_mask         = cache_compact_mask;
        params->mask_start           = mask_start;
        params->num_words            = num_words;
        /* Phase 2c row 12 + Phase 2e.1: num_salts is overloaded as:
         *   - HAS_SALT undef: use_mask ? mask_size : 1 (Phase 2b layout).
         *   - HAS_SALT defined: this_salt_chunk * mask_size  (per-dispatch
         *     when salt-chunking is active; was nsalts_packed * mask_size
         *     in Phase 2c when one dispatch covered all salts). Kernel
         *     reads mask_size = num_salts / num_salts_per_page; with
         *     num_salts_per_page = this_salt_chunk this gives
         *     mask_size = this_salt_chunk * mask_size / this_salt_chunk =
         *     mask_size. Same overload as gpu_template.cl 251-256.
         *
         * Phase 2e.1: salt_start, num_salts, num_salts_per_page are now
         * REWRITTEN per outer salt-chunk iteration in the new salt-chunk
         * loop below. The initial values here are the non-salt defaults
         * (salt_total == 1 collapses cleanly). */
        uint32_t mask_size_for_pack = use_mask ? (uint32_t)gpu_mask_total : 1u;
        params->num_salts            = mask_size_for_pack;
        /* Phase 2e.1: salt_start is the per-dispatch salt-page base
         * (was always 0 in Phase 2c when one dispatch covered the full
         * salt list). Rewritten per outer salt-chunk iteration. */
        params->salt_start           = 0;
        params->max_probe            = cache_max_probe;
        params->hash_data_count      = cache_hash_data_count;
        params->max_hits             = GPU_MAX_HITS;
        params->overflow_count       = (uint32_t)cache_overflow_count;
        params->max_iter             = (uint32_t)cache_max_iter;
        /* Phase 2d.3.3: JOB_SHA1DRU's 1M-iteration loop is INSIDE
         * template_finalize; the kernel's outer iter loop in template_phase0
         * must run exactly ONCE so only the FINAL state (after 1M inner
         * iterations) is probed. CPU semantics at mdxfind.c:15187-15205
         * also probe exactly once after the for-loop. Force max_iter=1
         * regardless of user `-i` (which doesn't apply to SHA1DRU at the
         * CPU level either -- the 1M is hardcoded in the algorithm). Mirror
         * of OpenCL twin gpu_opencl.c:10793 (B6.11 host-side forcing). */
        if (cache_gpu_op == JOB_SHA1DRU) params->max_iter = 1;
        /* Phase 2d.7d: HMAC ops have NO CPU iter loop (CPU calls
         * checkhashsalt(..., iter=0, ...) once after the HMAC body, see
         * mdxfind.c:31345 / 31732 / 31786 / 31845 / 31898). The HMAC
         * body runs INSIDE template_finalize on the GPU; the kernel's
         * outer iter loop must run exactly ONCE so only the FINAL state
         * (after the HMAC body) is probed. Force max_iter=1 regardless of
         * user `-i`. Mirrors OpenCL twin's host-side forcing at the rules-
         * engine pack site. */
        if (cache_gpu_op == JOB_HMAC_BLAKE2S ||
            cache_gpu_op == JOB_HMAC_STREEBOG256_KSALT ||
            cache_gpu_op == JOB_HMAC_STREEBOG256_KPASS ||
            cache_gpu_op == JOB_HMAC_STREEBOG512_KSALT ||
            cache_gpu_op == JOB_HMAC_STREEBOG512_KPASS) {
            params->max_iter = 1;
        }
        /* Phase 2d.8a: PHPBB3 + MD5CRYPT iter-loop algorithms. Both have
         * their iteration loops INSIDE template_finalize (PHPBB3: count
         * decoded from salt[3], typically 128..2^30; MD5CRYPT: FIXED 1000
         * iters). The kernel's outer iter loop in template_phase0 must
         * run exactly ONCE so only the FINAL state is probed. Force
         * max_iter=1 regardless of user `-i`. Mirrors SHA1DRU pattern. */
        if (cache_gpu_op == JOB_PHPBB3 ||
            cache_gpu_op == JOB_MD5CRYPT) {
            params->max_iter = 1;
        }
        /* Phase 2d.8b: SHACRYPT triple (SHA256CRYPT + SHA512CRYPT +
         * SHA512CRYPTMD5). Each runs the full 5-step glibc crypt-sha2
         * chain (default 5000 inner rounds; configurable via "rounds=N$"
         * salt prefix decoded INSIDE template_finalize). The kernel's
         * outer iter loop in template_phase0 must run exactly ONCE so
         * only the FINAL state is probed. Force max_iter=1 regardless
         * of user `-i`. Mirrors OpenCL twin's host-side forcing at the
         * rules-engine pack site (gpu_opencl.c host-side max_iter
         * override for SHACRYPT family). */
        if (cache_gpu_op == JOB_SHA256CRYPT ||
            cache_gpu_op == JOB_SHA512CRYPT ||
            cache_gpu_op == JOB_SHA512CRYPTMD5) {
            params->max_iter = 1;
        }
        /* Phase 2d.9a: DESCRYPT (op=500) runs the FIXED 25-iteration
         * DES Feistel chain INSIDE template_finalize. The kernel's outer
         * iter loop in template_phase0 must run exactly ONCE so only the
         * FINAL state is probed (CPU semantics at mdxfind.c:23673 calls
         * JSLG once after the for-loop in bsd_crypt_des). Force max_iter=1
         * regardless of user `-i`. Mirrors PHPBB3/MD5CRYPT/SHACRYPT
         * pattern. */
        if (cache_gpu_op == JOB_DESCRYPT) {
            params->max_iter = 1;
        }
        /* Phase 2d.9b: BCRYPT (op=450) runs the 2^cost Eksblowfish iter
         * loop INSIDE template_finalize. The kernel's outer iter loop in
         * template_phase0 must run exactly ONCE so only the FINAL state
         * is probed (CPU semantics: bcrypt yields a single 24-byte digest
         * after 2^cost rounds). Force max_iter=1 regardless of user `-i`.
         * Mirrors PHPBB3/MD5CRYPT/SHACRYPT/DESCRYPT pattern. */
        if (cache_gpu_op == JOB_BCRYPT) {
            params->max_iter = 1;
        }
        /* num_masks is overloaded as the kernel's n_rules in the rules-
         * engine dispatch path (mirrors OCLParams convention, gpu_template.cl
         * line 215). Phase 1 no-rules: num_masks=1 (kernel skips the rules
         * walker entirely under #ifdef). Phase 2a rules: num_masks =
         * gpu_rule_count so the kernel sees one lane per (word, rule).
         * Note: this field is OVERWRITTEN per chunk in the rule-axis sub-
         * batching loop below to carry the per-chunk rule_count; the
         * initial value here is the default for the no-sub-batching case. */
        params->num_masks            = use_rules
                                       ? (uint32_t)gpu_rule_count : 1u;
        /* Phase 2b row 4: kernel reads n_prepend / n_append to gate the
         * mask block. Set from the host-cached descriptor when use_mask
         * is true; zero otherwise so the kernel's `(npre>=1||napp>=1)`
         * gate stays off in non-mask dispatches. */
        params->n_prepend            = use_mask
                                       ? (uint32_t)gpu_mask_n_prepend : 0u;
        params->n_append             = use_mask
                                       ? (uint32_t)gpu_mask_n_append  : 0u;
        params->iter_count           = 0;
        params->input_cursor_start   = 0;
        /* Phase 2c row 13 + §8 migration: rule_cursor_start (offset 92)
         * is the per-dispatch rule_base now that the kernel reads it
         * directly under HAS_RULES (was overloaded via salt_start in
         * Phase 2b). Initialized to 0; OVERWRITTEN per chunk in the
         * rule-axis sub-batching loop below. */
        params->rule_cursor_start    = 0;
        params->inner_iter           = inner_iter;
        params->overflow_first_set   = 0;
        params->overflow_first_word  = 0xFFFFFFFFu;  /* CAS-min sentinel */
        /* D9.5.a (sub-phase 1a.2): was overflow_first_rule CAS-min sentinel.
         * Renamed to num_rules; A1/A3 dispatcher payload populator overwrites
         * pre-enqueue. Init to safe default 0 here (B3 path does not read). */
        params->num_rules            = 0;
        /* Phase 2c row 12 + Phase 2e.1: num_salts_per_page carries the
         * per-dispatch salt-chunk count when HAS_SALT is active (was
         * cached_salts_count in 2c when one dispatch covered the whole
         * list). For unsalted dispatches it stays at 1 — the kernel's
         * optional HAS_SALT block reads this only when the macro is set.
         * Rewritten per outer salt-chunk iteration in the loop below. */
        params->num_salts_per_page   = 1ull;
        /* algo_mode = 0 for JOB_MD5SALT in Phase 2c (e31 lowercase hex).
         * Phase 2d.7d HMAC siblings extend the per-op switch (mirrors
         * gpu_opencl.c lines 10667-10768). HMAC body in the carrier kernel
         * gates on `if (algo_mode >= 5u)` (or `== 5u` for blake2s); host
         * must set this correctly or the HMAC body silently never fires
         * (defensive fallback runs, producing wrong digest). */
        if      (op == JOB_HMAC_BLAKE2S)           params->algo_mode = 5u;
        else if (op == JOB_HMAC_STREEBOG256_KSALT) params->algo_mode = 5u;
        else if (op == JOB_HMAC_STREEBOG256_KPASS) params->algo_mode = 6u;
        else if (op == JOB_HMAC_STREEBOG512_KSALT) params->algo_mode = 5u;
        else if (op == JOB_HMAC_STREEBOG512_KPASS) params->algo_mode = 6u;
        /* Phase 2d.8b: SHA512CRYPTMD5 (op=538) selects kernel-side MD5-
         * preprocess via algo_mode=1u. The kernel's template_finalize
         * checks `if (algo_mode == 1u)` at the top and substitutes the
         * password with the 32-byte ASCII hex of MD5(password) BEFORE
         * running the SHA-crypt-512 chain. Mirrors CPU semantics at
         * mdxfind.c:12199-12212. SHA256CRYPT (op=512) and SHA512CRYPT
         * (op=513) use algo_mode=0u (default arm; no MD5-preprocess).
         * SHA512CRYPT and SHA512CRYPTMD5 share the SAME compiled PSO
         * (HASH_WORDS=16) -- algo_mode is the runtime discriminator. */
        else if (op == JOB_SHA512CRYPTMD5)         params->algo_mode = 1u;
        /* Phase 2d.9a: DESCRYPT uses algo_mode=7 (bespoke kernel; single
         * mode). The kernel ignores the value (cast to (void) inside
         * template_finalize) but host sets it for cache-key consistency
         * with the BASE_ALGO=descrypt defines_str discipline. */
        else if (op == JOB_DESCRYPT)               params->algo_mode = 7u;
        /* Phase 2d.9b: BCRYPT (op=450) uses algo_mode=8u. Reserved range
         * 8-15 for future BCRYPT-family variants (BCRYPTMD5/BCRYPTSHA1/
         * BCRYPTSHA512 will share algo_mode=8 because host preprocesses
         * the input before pack; future kernel-side preprocess variants
         * would claim 9-15). Mirrors gpu_bcrypt_core.cl rev 1.1 algo_mode
         * gate at template_finalize entry. */
        else if (op == JOB_BCRYPT)                 params->algo_mode = 8u;
        else                                       params->algo_mode = 0u;
        params->mask_offset_per_word = mask_offset_per_word;

        /* hit_count slot at offset 128 (already zeroed by memset). */
        memcpy(p + 132, word_offset, wo_size);
        memcpy(p + payload_pkt, packed_words, packed_size);

        /* Ensure the on-GPU dedup buffer exists. Sized to
         * hash_data_count + overflow_count slots; persisted across
         * dispatches in Phase 2+ (recreated each dispatch in Phase 1
         * for simplicity — the cost is small for the first-light
         * smoke). */
        size_t need_slots = (size_t)cache_hash_data_count
                          + (size_t)cache_overflow_count;
        if (need_slots == 0) need_slots = 1;
        size_t hs_bytes = need_slots * sizeof(uint32_t);
        if (hs_bytes < METAL_MIN_BUFFER_BYTES) hs_bytes = METAL_MIN_BUFFER_BYTES;
        if (buf_hashes_shown == nil) {
            buf_hashes_shown =
                [mtl_device newBufferWithLength:hs_bytes
                                        options:MTLResourceStorageModeShared];
            if (buf_hashes_shown == nil) {
                /* Phase D5a (Task #281): alloc failure -> fatal. */
                GPU_FATAL("Metal: hashes_shown newBuffer(%zu bytes) failed (op=%d need_slots=%zu)",
                          hs_bytes, op, need_slots);
            }
            memset([buf_hashes_shown contents], 0, hs_bytes);
        }

        /* Hits output buffer (GPU side). HIT_STRIDE uint32 per hit. */
        size_t hits_bytes = (size_t)GPU_MAX_HITS * GPU_HIT_STRIDE * sizeof(uint32_t);
        id<MTLBuffer> buf_hits =
            [mtl_device newBufferWithLength:hits_bytes
                                    options:MTLResourceStorageModeShared];
        if (buf_hits == nil) {
            /* Phase D5a (Task #281): alloc failure -> fatal. */
            GPU_FATAL("Metal: hits newBuffer(%zu bytes) failed (op=%d)",
                      hits_bytes, op);
        }
        memset([buf_hits contents], 0, hits_bytes);

        /* Defensive placeholders for nil read-only buffers — the kernel
         * derefs by signature, but the args are never read at runtime
         * unless overflow_count > 0 / hash_data_count > 0. */
        id<MTLBuffer> bo_keys    = buf_overflow_keys;
        id<MTLBuffer> bo_hashes  = buf_overflow_hashes;
        id<MTLBuffer> bo_offsets = buf_overflow_offsets;
        id<MTLBuffer> bhd        = buf_hash_data;
        id<MTLBuffer> bhdoff     = buf_hash_data_off;
        if (bo_keys == nil || bo_hashes == nil || bo_offsets == nil ||
            bhd == nil || bhdoff == nil) {
            /* Synthesize floor-sized zero placeholder lazily (one-shot,
             * persists for session). */
            static id<MTLBuffer> floor_zero = nil;
            if (floor_zero == nil) {
                floor_zero =
                    [mtl_device newBufferWithLength:METAL_MIN_BUFFER_BYTES
                                            options:MTLResourceStorageModeShared];
                if (floor_zero == nil) {
                    /* Phase D5a (Task #281): alloc failure -> fatal. The
                     * prior silent-nil path would pass nil into the kernel
                     * arg bind and the kernel would dereference nil at
                     * runtime. */
                    GPU_FATAL("Metal: floor_zero placeholder newBuffer(%u bytes) failed (op=%d)",
                              (unsigned)METAL_MIN_BUFFER_BYTES, op);
                }
                memset([floor_zero contents], 0, METAL_MIN_BUFFER_BYTES);
            }
            if (bo_keys    == nil) bo_keys    = floor_zero;
            if (bo_hashes  == nil) bo_hashes  = floor_zero;
            if (bo_offsets == nil) bo_offsets = floor_zero;
            if (bhd        == nil) bhd        = floor_zero;
            if (bhdoff     == nil) bhdoff     = floor_zero;
        }

        /* Threadgroup sizing: one lane per WORD post task #250 (the kernel
         * folds rule × mask axes into an inner double-loop so each lane
         * owns one RULE_BUF_MAX scratch slot for its entire run). Rounded
         * up to a multiple of the PSO's threadExecutionWidth. */
        NSUInteger tg_w = active_pso.threadExecutionWidth;
        NSUInteger max_tg = active_pso.maxTotalThreadsPerThreadgroup;
        NSUInteger tg = tg_w;
        if (tg > max_tg) tg = max_tg;
        if (tg == 0) tg = 32;
        /* Phase 2d.9b: per-op threadsPerThreadgroup override.
         * Mirrors gpu_opencl.c:11760-11762 `local = 8` for BCRYPT. The
         * BCRYPT kernel declares a threadgroup-shared sbox_pool =
         * BCRYPT_WG_SIZE * 1024 uints = 32 KB at WG=8, which EXACTLY
         * MATCHES Apple Silicon's maxThreadgroupMemoryLength = 32 KB.
         * WG > 8 would overflow the threadgroup memory cap and refuse to
         * dispatch. Apple's PSO validator will likely return
         * maxTotalThreadsPerThreadgroup appropriately constrained for
         * this PSO, but defense-in-depth: force tg=8 explicitly so a
         * future PSO-property regression doesn't silently widen the WG
         * and corrupt the threadgroup-memory layout. Per TRAP 2 of the
         * Phase 2d.9b architect brief (§E item 13).
         *
         * D5b Wave 4 2026-05-16: replaced the hardcoded `cache_gpu_op ==
         * JOB_BCRYPT` check with fam-driven lookup. .dispatch_tg_size is
         * a uint16_t field on struct gpu_metal_family (Wave 1 extension);
         * 0 means "use the default tg computed above". metal_family_bcrypt
         * sets it to 8. Any future family that needs a TG override sets
         * .dispatch_tg_size = N in its struct literal -- no dispatch-site
         * edit required. Per architect §E row 9 + §C Wave 4. */
        if (fam != NULL && fam->dispatch_tg_size != 0) {
            tg = fam->dispatch_tg_size;
        }

        /* Grid = num_words (was num_words × n_rules × mask_total in the
         * pre-task-#250 layout). The kernel iterates rule_idx and
         * mask_idx_local internally; combined_ridx packing semantics for
         * the hit-replay path are unchanged. */
        uint64_t total64 = (uint64_t)num_words;
        if (total64 == 0) total64 = 1;

        MTLSize threadgroup = MTLSizeMake(tg, 1, 1);
        NSUInteger n_groups = (NSUInteger)((total64 + tg - 1) / tg);
        MTLSize grid = MTLSizeMake(n_groups, 1, 1);

        /* Task #250: rule-axis sub-batching. The kernel inner-loops over
         * rule_idx in [rule_base, rule_base + rule_count). For 100K-rule
         * programs an unsplit dispatch would inner-loop 100K rules per
         * word lane → per-thread kernel runtime ~10s → hits Apple's
         * `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` watchdog.
         *
         * METAL_RULE_CHUNK_SIZE caps rules-per-dispatch; the host loops
         * over chunks. Each chunk's kernel runtime stays well under the
         * watchdog (< 1 second observed on M1 at 16K words × 1000 rules).
         * Hit accumulation: hit_count is an atomic across all chunks, so
         * resetting payload + 128 happens once (already done above via
         * memset). buf_hashes_shown is the dedup bitfield, also persistent
         * across chunks.
         *
         * Non-rules dispatches (use_rules == 0) execute exactly one
         * iteration with rule_base=0 and rule_count=1 — the chunk loop
         * collapses cleanly. */
        uint32_t total_rules = use_rules ? (uint32_t)gpu_rule_count : 1u;
#ifndef METAL_RULE_CHUNK_SIZE
        /* Default chunk: 8192 rules. Empirical perf curve on dev1.local
         * (Apple M1) for 16K words × 100K HashMob.100k.rule HMD5+rockyou:
         *   chunk=1024 -> 57 MH/s  (per-dispatch overhead dominates)
         *   chunk=8192 -> 115 MH/s (sweet spot)
         *   chunk=32768 -> 117 MH/s (chunk-overhead amortized; device-buf
         *                            latency is the residual cost vs the
         *                            145 MH/s thread-buf baseline)
         * Watchdog headroom: each chunk runs ~0.6s on M1 at 16K-word grid;
         * stays well under Apple's ~2s `kIOGPUCommandBufferCallbackError
         * ImpactingInteractivity` cap. M2 Max is faster per chunk, more
         * headroom still. Tunable via MDXFIND_METAL_RULE_CHUNK env. */
#define METAL_RULE_CHUNK_SIZE 8192u
#endif
        uint32_t rule_chunk_size = METAL_RULE_CHUNK_SIZE;
        {
            const char *env_chunk = getenv("MDXFIND_METAL_RULE_CHUNK");
            if (env_chunk != NULL) {
                long v = strtol(env_chunk, NULL, 10);
                if (v >= 1 && v <= 1000000) rule_chunk_size = (uint32_t)v;
            }
        }
        if (!use_rules) rule_chunk_size = 1u;  /* one-shot for non-rules */
        if (rule_chunk_size > total_rules) rule_chunk_size = total_rules;

        /* Phase 2e.1: salt-axis chunking. Outer salt loop iterates pages
         * of size salt_chunk_size; the existing rule-chunk loop sits
         * inside. Per-dispatch wall is bounded by
         *   words * rule_chunk * mask_size * salt_chunk
         * which keeps Apple's ~2s ImpactingInteractivity watchdog at bay
         * even on M1 with sm-saltfull-class workloads (589k salts).
         *
         * When use_salt is false the outer loop collapses to one
         * iteration with salt_total=1 and the params writes below are
         * bit-identical to the prior Phase 2c behavior.
         *
         * Per-tier defaults via metal_select_salt_chunk():
         *   M1 = 64, M2/M2 Max = 256, M3+ = 1024. Env override
         *   MDXFIND_METAL_SALT_CHUNK wins. */
        uint32_t salt_total      = use_salt ? (uint32_t)cached_salts_count : 1u;
        if (salt_total == 0u) salt_total = 1u;
        uint32_t salt_chunk_size = 1u;
        const char *salt_chunk_source = "default";
        if (use_salt) {
            uint32_t tier_pick = metal_select_salt_chunk(mtl_device);
            if (getenv("MDXFIND_METAL_SALT_CHUNK") != NULL)
                salt_chunk_source = "env";
            salt_chunk_size = tier_pick;
            if (salt_chunk_size > salt_total) salt_chunk_size = salt_total;
            if (salt_chunk_size == 0u) salt_chunk_size = 1u;
        }

        /* Rule-chunk scale-down: when both axes are large the
         * multiplicative budget breaches the watchdog. Drop rule_chunk
         * to max(256, METAL_RULE_CHUNK_SIZE / 32) when salt-chunking is
         * active and the salt axis exceeds one chunk. Mirrors memo §3
         * watchdog math: 16384 * 256 * 64 ≈ 2.7e8 ops/dispatch on M1. */
        if (use_salt && salt_total > salt_chunk_size && rule_chunk_size > 256u) {
            uint32_t scaled = METAL_RULE_CHUNK_SIZE / 32u;
            if (scaled < 256u) scaled = 256u;
            if (rule_chunk_size > scaled) rule_chunk_size = scaled;
            if (rule_chunk_size > total_rules) rule_chunk_size = total_rules;
        }

        /* Phase 2e.1 first-dispatch marker extension. One-shot stderr
         * line documenting the salt-chunk size + source for the operator
         * (mirrors the presalt marker). Per
         * feedback_verify_gpu_fired_post_build.md: visibility into
         * chunking is required so silent fall-through to single-shot
         * dispatch is detectable. */
        static int salt_chunk_marker_logged = 0;
        if (use_salt && !salt_chunk_marker_logged) {
            salt_chunk_marker_logged = 1;
            GPU_DEBUG_FPRINTF(stderr,
                    "Metal: salt-chunked dispatch (salt_chunk_size=%u "
                    "nsalts=%u rule_chunk=%u source=%s)\n",
                    (unsigned)salt_chunk_size, (unsigned)salt_total,
                    (unsigned)rule_chunk_size, salt_chunk_source);
        }

        /* Phase 2e.1 Path 1b post-process state: track the hit count at
         * the end of the prior dispatch so the post-process loop only
         * rewrites THIS dispatch's new entries. Reset to 0 before the
         * salt-chunk loop; advances per dispatch. */
        uint32_t prev_hit_count = 0u;

        /* End-of-batch summary counters (Phase 2e.1 §6). */
        uint32_t total_salt_chunks = 0u;
        uint32_t total_rule_chunks_per_salt = 0u;
        uint32_t total_dispatches = 0u;
        uint64_t total_dispatch_wall_us = 0ull;

        for (uint32_t salt_base = 0u;
             salt_base < salt_total;
             salt_base += salt_chunk_size) {

            uint32_t this_salt_chunk = salt_chunk_size;
            if (salt_base + this_salt_chunk > salt_total)
                this_salt_chunk = salt_total - salt_base;

            /* Phase 2e.1 row 3: rewrite salt-related params for this
             * salt-chunk. For !use_salt this collapses to the
             * single-iteration defaults (salt_start=0,
             * num_salts_per_page=1, num_salts=mask_size_for_pack). */
            params->salt_start = use_salt ? (uint32_t)salt_base : 0u;
            params->num_salts_per_page = use_salt
                                         ? (uint64_t)this_salt_chunk : 1ull;
            params->num_salts = use_salt
                                ? (uint32_t)this_salt_chunk * mask_size_for_pack
                                : mask_size_for_pack;

            total_salt_chunks++;
            uint32_t rule_chunks_this_salt = 0u;

        for (uint32_t rule_base = 0u;
             rule_base < total_rules;
             rule_base += rule_chunk_size) {

            uint32_t this_chunk = rule_chunk_size;
            if (rule_base + this_chunk > total_rules)
                this_chunk = total_rules - rule_base;

            /* Repack params for this chunk. The wire-format payload was
             * memset to 0 above; we re-write the OCLParams prefix each
             * chunk so rule_cursor_start (rule_base) and num_masks
             * (rule_count) carry chunk-specific values. The kernel reads
             * them once into local registers at entry.
             *
             * Phase 2c §8 migration: previously this site wrote
             * params->salt_start = rule_base (overloading salt_start
             * because HAS_SALT was undef and the field had no other
             * meaning). With the salt axis arriving, params->salt_start
             * returns to its OpenCL semantics (salt-page start, rewritten
             * per outer salt-chunk iteration above), and
             * params->rule_cursor_start (the pre-existing field at offset
             * 92, claimed for real use) carries rule_base.
             * metal_template.metal rev 1.5 reads params.rule_cursor_start
             * under HAS_RULES. */
            params->rule_cursor_start = (uint32_t)rule_base;
            params->num_masks  = use_rules ? this_chunk : 1u;

            /* Encode + dispatch one chunk. */
            id<MTLCommandBuffer> cb = [mtl_queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:active_pso];

            /* Buffer indices match metal_template.metal binding order:
             *   0 payload
             *   1 compact_fp
             *   2 compact_idx
             *   3 hash_data_buf
             *   4 hash_data_off
             *   5 hits
             *   6 overflow_keys
             *   7 overflow_hashes
             *   8 overflow_offsets
             *   9 hashes_shown
             *  Phase 2a rules variant additionally binds:
             *  10 rule_program
             *  11 rule_offset
             *  Phase 2b mask variant additionally binds:
             *  12 mask_charsets
             *  13 mask_sizes
             *  Task #250 (all variants):
             *  14 buf_scratch_pool
             * Indices are gap-tolerant - M-alone (use_rules==0, use_mask==1)
             * binds 0..9, 12, 13 with the kernel signature stripping 10/11
             * via the preprocessor. */
            [enc setBuffer:buf_payload          offset:0 atIndex:0];
            [enc setBuffer:buf_compact_fp       offset:0 atIndex:1];
            [enc setBuffer:buf_compact_idx      offset:0 atIndex:2];
            [enc setBuffer:bhd                  offset:0 atIndex:3];
            [enc setBuffer:bhdoff               offset:0 atIndex:4];
            [enc setBuffer:buf_hits             offset:0 atIndex:5];
            [enc setBuffer:bo_keys              offset:0 atIndex:6];
            [enc setBuffer:bo_hashes            offset:0 atIndex:7];
            [enc setBuffer:bo_offsets           offset:0 atIndex:8];
            [enc setBuffer:buf_hashes_shown     offset:0 atIndex:9];
            if (use_rules) {
                [enc setBuffer:buf_rule_program offset:0 atIndex:10];
                [enc setBuffer:buf_rule_offset  offset:0 atIndex:11];
            }
            if (use_mask) {
                [enc setBuffer:buf_mask_charsets offset:0 atIndex:12];
                [enc setBuffer:buf_mask_sizes    offset:0 atIndex:13];
            }
            [enc setBuffer:buf_scratch_pool offset:0 atIndex:14];
            /* Phase 2c row 14: salt buffers bound at gap-tolerant indices
             * 15/16/17 when the salt PSO variant is in use. M-alone
             * salted (use_salt && use_mask && !use_rules) binds
             * 0..9,12,13,14,15,16,17 (gaps at 10,11 are preprocessor-
             * stripped in the kernel signature; Apple Metal accepts). */
            if (use_salt) {
                [enc setBuffer:buf_salt_data offset:0 atIndex:15];
                [enc setBuffer:buf_salt_off  offset:0 atIndex:16];
                [enc setBuffer:buf_salt_lens offset:0 atIndex:17];
            }

            uint64_t _disp_t0 = metal_now_us();
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            uint64_t _disp_t1 = metal_now_us();
            total_dispatch_wall_us += (_disp_t1 - _disp_t0);
            total_dispatches++;
            rule_chunks_this_salt++;

            if (cb.error != nil) {
                /* Phase D5a (Task #281+#282) 2026-05-16: external runtime
                 * failure -> fatal exit. The prior "return NULL" path
                 * silently dropped words at gpujob_metal_worker's
                 * return_jobg label (no CPU re-queue) -- this was the
                 * Phase 2d.5 dev3 PSO silent-data-loss bug. Never silently
                 * fall back; operator sees the error at first occurrence
                 * with full context. */
                MTL_FATAL_NSERR(cb.error,
                    "Metal dispatch error op=%d salt_base=%u salt_count=%u "
                    "rule_base=%u rule_count=%u total_rules=%u",
                    op, salt_base, this_salt_chunk,
                    rule_base, this_chunk, total_rules);
            }

            /* Phase 2e.1 Path 1b post-process: rewrite this dispatch's
             * hits from per-chunk encoding to global encoding. Mirrors
             * gpu/gpu_opencl.c:11947-11982 verbatim.
             *
             * Kernel emits (metal_template.metal:611-617):
             *   combined_ridx_chunk = X * nspp + salt_local
             * where X = rule_idx * mask_size + mask_local,
             *       nspp = this_salt_chunk (per-dispatch),
             *       salt_local in [0, this_salt_chunk).
             *
             * Hit-replay (gpu/gpujob_metal.m:751-808) decodes via
             * `% nsalts_packed`. To make that decode recover
             * salt_idx_global directly, rewrite each new hit to:
             *   combined_ridx_global = X * nsalts_packed + salt_idx_global
             * where salt_idx_global = salt_base + salt_local.
             *
             * Implicit cap: combined_ridx is uint32. For sm-saltfull e31
             * (rule=1, mask=1, nsalts=589940) max global ridx = 589,939;
             * for best64.rule (rule=64) max = 63 * 589940 + 589939 =
             * 37.7M. Fits with very large headroom. */
            if (use_salt) {
                uint32_t new_hit_count = 0u;
                memcpy(&new_hit_count, p + 128, sizeof(new_hit_count));
                if (new_hit_count > GPU_MAX_HITS) new_hit_count = GPU_MAX_HITS;
                uint32_t nspp = (uint32_t)this_salt_chunk;
                if (nspp == 0u) nspp = 1u;
                uint32_t total_salts_dyn = (uint32_t)cached_salts_count;
                if (total_salts_dyn == 0u) total_salts_dyn = 1u;
                uint32_t *hit_words = (uint32_t *)[buf_hits contents];
                for (uint32_t h = prev_hit_count;
                     h < new_hit_count && h < GPU_MAX_HITS;
                     h++) {
                    uint32_t cri = hit_words[h * GPU_HIT_STRIDE + 1];
                    uint32_t salt_local      = cri % nspp;
                    uint32_t x               = cri / nspp;
                    uint32_t salt_idx_global = (uint32_t)salt_base + salt_local;
                    hit_words[h * GPU_HIT_STRIDE + 1] =
                        x * total_salts_dyn + salt_idx_global;
                }
                prev_hit_count = new_hit_count;
            }
        }

            if (total_salt_chunks == 1u)
                total_rule_chunks_per_salt = rule_chunks_this_salt;
        }

        /* Phase 2e.1 end-of-batch summary: one-shot stderr line after
         * the first batch (gated by the same one-shot flag the
         * salt-chunk marker uses, so it co-fires with the warm-up). */
        static int salt_chunk_summary_logged = 0;
        if (use_salt && !salt_chunk_summary_logged && total_dispatches > 0) {
            salt_chunk_summary_logged = 1;
            uint64_t avg_us = total_dispatch_wall_us / total_dispatches;
            GPU_DEBUG_FPRINTF(stderr,
                    "Metal: chunked dispatch summary: %u salt chunks x %u "
                    "rule chunks = %u dispatches, avg %llu us/dispatch\n",
                    (unsigned)total_salt_chunks,
                    (unsigned)total_rule_chunks_per_salt,
                    (unsigned)total_dispatches,
                    (unsigned long long)avg_us);
        }

        /* Hit count lives at payload offset 128. Per-slot drain: by the
         * time we reach here, the Phase 2e.1 Path 1b post-process has
         * rewritten every hit in buf_hits to global encoding, so the
         * existing hit-replay decode in gpu/gpujob_metal.m:751-808
         * (which divmods by nsalts_packed) recovers the correct
         * salt_idx_global / mask_idx / ridx without further host work. */
        uint32_t raw_nhits = 0;
        memcpy(&raw_nhits, p + 128, sizeof(raw_nhits));
        uint32_t emitted = raw_nhits > GPU_MAX_HITS ? GPU_MAX_HITS : raw_nhits;

        if (emitted > 0) {
            size_t bytes = (size_t)emitted * GPU_HIT_STRIDE * sizeof(uint32_t);
            memcpy(h_hits, [buf_hits contents], bytes);
        }
        *nhits_out = (int)emitted;
        return (emitted > 0) ? h_hits : NULL;
    }
}

/* Phase 1 gpujob stubs removed in Phase 2a — real worker thread + symbol
 * exports now live in gpu/gpujob_metal.m. */

/* ====================================================================
 * hx codegen sub-phase 2a.4 (2026-05-21): Metal twin of
 * gpu_opencl_jit_compile_source_with_common (gpu/gpu_opencl.c line 3431).
 *
 * Concatenate metal_common_str + caller source and JIT-compile via
 * -[MTLDevice newLibraryWithSource:options:error:]. The codegen-emitted
 * Metal source from hx_emit_e347_md5md5md5salt_metal references symbols
 * defined in metal_common.metal (md5_block, OCLParams/MetalParams
 * typedef bridge, HIT_STRIDE, EMIT_HIT_4_DEDUP_OR_OVERFLOW,
 * probe_compact_idx); metal_common_str must therefore be prepended at
 * compile time (Metal's newLibraryWithSource does NOT run a C-style
 * preprocessor on #include directives at JIT scope).
 *
 * FATAL on any error per feedback_external_failures_are_fatal.md. The
 * compiled library is released before return; harness mode only -- not
 * yet wired into the runtime dispatch path (Phase 4 territory). The
 * kernel function is looked up by name and immediately released; this
 * sub-phase verifies JIT-clean only, not PSO creation (PSO creation is
 * a Phase 4 dispatcher-binding concern).
 *
 * Symbol is METAL_GPU-guarded so the call site in mdxfind.c can be
 * mirrored under #if defined(METAL_GPU) symmetric with the OpenCL twin
 * call under #if defined(OPENCL_GPU). */
int gpu_metal_jit_compile_source_with_common(int dev_idx, const char *src)
{
    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname) - 1);

    if (!metal_ready) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen JIT requested but Metal not "
            "initialized on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }
    if (dev_idx != 0) {
        /* Phase 1 Metal is single-device; dev_idx must be 0. */
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT dev_idx=%d out of range "
            "(Phase 1 single-device) on %s\n",
            __FILE__, __LINE__, dev_idx, hostname);
        exit(1);
    }
    if (!src) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT source pointer is NULL on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }
    if (mtl_device == nil) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT mtl_device is nil on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }

    /* Concat metal_common_str + src into a single buffer; same build-
     * order pattern as metal_load_library() and metal_load_library_salt_
     * variant() above. */
    size_t total = strlen(metal_common_str) + strlen(src) + 16;
    char *concat = (char *)malloc(total);
    if (!concat) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT concat malloc(%zu) failed "
            "on %s\n",
            __FILE__, __LINE__, total, hostname);
        exit(1);
    }
    strcpy(concat, metal_common_str);
    strcat(concat, "\n");
    strcat(concat, src);

    NSString *nsstr = [NSString stringWithUTF8String:concat];
    free(concat);
    if (nsstr == nil) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT NSString conversion failed "
            "on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    NSError *err = nil;
    id<MTLLibrary> lib = [mtl_device newLibraryWithSource:nsstr
                                                  options:opts
                                                    error:&err];
    if (lib == nil) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal newLibraryWithSource(with_common) "
            "failed on %s dev[%d]: %s\n",
            __FILE__, __LINE__, hostname, dev_idx,
            err ? [[err localizedDescription] UTF8String] : "(no error)");
        exit(1);
    }

    /* Look up the known entry points so a library that compiles but
     * lacks ANY recognised kernel function is still surfaced as FATAL.
     * The walker emits one of two kernel names: `kernelb_hx_e347_phase0`
     * for the pattern-recognised e347 fast path, or `hx_trivial_kernel`
     * for the generic per-opcode dispatch (2a.2 placeholder). Both are
     * accepted; the lookup proceeds in pattern-first order so the
     * fast-path success log marker is preferred when both names exist
     * (the trivial walker never co-emits both). */
    id<MTLFunction> fn = [lib newFunctionWithName:@"kernelb_hx_e347_phase0"];
    if (fn == nil) {
        fn = [lib newFunctionWithName:@"hx_trivial_kernel"];
    }
    if (fn == nil) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal library compiled but no known "
            "entry point (kernelb_hx_e347_phase0 / hx_trivial_kernel) "
            "found on %s dev[%d]\n",
            __FILE__, __LINE__, hostname, dev_idx);
        exit(1);
    }

    fprintf(stderr,
            "hx codegen: Metal JIT (with common) succeeded on "
            "%s dev[%d]\n",
            hostname, dev_idx);

    /* Release lib + fn (harness mode -- no PSO retained for Phase 2-3). */
    fn = nil;
    lib = nil;
    return 0;
}

/* hx codegen sub-phase 2a.6 (2026-05-22): retain-the-PSO variant of the
 * 2a.4 JIT helper. Returns the MTLLibrary + MTLComputePipelineState as
 * bridge-retained opaque void* pointers so the caller (mdxfind.c harness)
 * can dispatch against real data after JIT, then release via
 * gpu_metal_jit_release_keep when done. FATAL on any error. */
int gpu_metal_jit_compile_source_with_common_keep(
    int dev_idx, const char *src, const char *entry_point_name,
    void **out_library, void **out_pso)
{
    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname) - 1);

    if (!metal_ready) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen JIT (keep) requested but Metal not "
            "initialized on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }
    if (dev_idx != 0) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT (keep) dev_idx=%d out of "
            "range (Phase 1 single-device) on %s\n",
            __FILE__, __LINE__, dev_idx, hostname);
        exit(1);
    }
    if (!src || !entry_point_name || !out_library || !out_pso) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT (keep) NULL arg: src=%p "
            "entry=%p out_lib=%p out_pso=%p on %s\n",
            __FILE__, __LINE__, (void *)src, (void *)entry_point_name,
            (void *)out_library, (void *)out_pso, hostname);
        exit(1);
    }
    if (mtl_device == nil) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT (keep) mtl_device is nil "
            "on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }

    size_t total = strlen(metal_common_str) + strlen(src) + 16;
    char *concat = (char *)malloc(total);
    if (!concat) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT (keep) concat malloc(%zu) "
            "failed on %s\n",
            __FILE__, __LINE__, total, hostname);
        exit(1);
    }
    strcpy(concat, metal_common_str);
    strcat(concat, "\n");
    strcat(concat, src);

    NSString *nsstr = [NSString stringWithUTF8String:concat];
    free(concat);
    if (nsstr == nil) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT (keep) NSString conversion "
            "failed on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }

    MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
    NSError *err = nil;
    id<MTLLibrary> lib = [mtl_device newLibraryWithSource:nsstr
                                                  options:opts
                                                    error:&err];
    if (lib == nil) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal newLibraryWithSource "
            "(keep) failed on %s dev[%d]: %s\n",
            __FILE__, __LINE__, hostname, dev_idx,
            err ? [[err localizedDescription] UTF8String] : "(no error)");
        exit(1);
    }

    NSString *entry_ns = [NSString stringWithUTF8String:entry_point_name];
    id<MTLFunction> fn = [lib newFunctionWithName:entry_ns];
    if (fn == nil) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal JIT (keep): entry point "
            "\"%s\" not found in library on %s dev[%d]\n",
            __FILE__, __LINE__, entry_point_name, hostname, dev_idx);
        exit(1);
    }

    NSError *pso_err = nil;
    id<MTLComputePipelineState> pso =
        [mtl_device newComputePipelineStateWithFunction:fn error:&pso_err];
    if (pso == nil) {
        fprintf(stderr,
            "FATAL: %s:%d hx codegen Metal newComputePipelineStateWith"
            "Function(\"%s\") failed on %s dev[%d]: %s\n",
            __FILE__, __LINE__, entry_point_name, hostname, dev_idx,
            pso_err ? [[pso_err localizedDescription] UTF8String]
                    : "(no error)");
        exit(1);
    }

    fprintf(stderr,
            "hx codegen: Metal JIT (with common, retained) succeeded on "
            "%s dev[%d] entry=%s pso=%p\n",
            hostname, dev_idx, entry_point_name, (void *)pso);

    /* Bridge-retain the ObjC refs so they survive out of ARC scope. The
     * caller will release via gpu_metal_jit_release_keep -> CFBridgingRelease. */
    *out_library = (void *)CFBridgingRetain(lib);
    *out_pso     = (void *)CFBridgingRetain(pso);
    return 0;
}

/* hx codegen sub-phase 2a.6 (2026-05-22): cleanup pair for the keep
 * variant. Releases the bridge-retained MTLLibrary + MTLComputePipeline
 * State that gpu_metal_jit_compile_source_with_common_keep returned.
 * Idempotent on NULL. */
void gpu_metal_jit_release_keep(void *library, void *pso)
{
    if (pso)     CFBridgingRelease(pso);
    if (library) CFBridgingRelease(library);
}

/* hx codegen sub-phase 2a.6 (2026-05-22): byte-exact validation dispatch
 * for the codegen-emitted e347 (MD5MD5SALT) Metal kernel. Metal twin of
 * gpu_opencl_e347_validate_dispatch.
 *
 * Caller has already JIT-compiled the kernel via _with_common_keep,
 * uploaded the salts via gpu_metal_set_salt, and the compact table +
 * overflow + hash_data via gpu_metal_set_compact_table. We allocate
 * short-lived MTLBuffers for the per-invocation inputs (packed candidates,
 * word offsets, payload), bind the 18 kernel args mirroring the Metal
 * emitter signature, dispatch, and read hits back.
 *
 * Returns 0 on success; FATAL on any setup or dispatch error. */
int gpu_metal_e347_validate_dispatch(
    int dev_idx, void *pso,
    const unsigned char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    uint32_t num_salts,
    uint32_t *out_hits, int *out_n_hits, int max_hits)
{
    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname) - 1);

    if (!metal_ready) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): not initialized on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }
    if (dev_idx != 0) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): dev_idx=%d out of range "
            "on %s\n",
            __FILE__, __LINE__, dev_idx, hostname);
        exit(1);
    }
    if (!pso || !packed_words || !word_offset || !out_hits || !out_n_hits ||
        num_words == 0 || packed_size == 0 || num_salts == 0)
    {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): bad args on %s "
            "(pso=%p pw=%p wo=%p oh=%p onh=%p nw=%u ps=%u ns=%u)\n",
            __FILE__, __LINE__, hostname,
            pso, (void *)packed_words, (void *)word_offset,
            (void *)out_hits, (void *)out_n_hits,
            num_words, packed_size, num_salts);
        exit(1);
    }
    if (mtl_device == nil || mtl_queue == nil) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): mtl_device or mtl_queue "
            "nil on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }
    if (buf_compact_fp == nil || buf_compact_idx == nil ||
        buf_hash_data == nil || buf_hash_data_off == nil ||
        buf_salt_data == nil || buf_salt_off == nil || buf_salt_lens == nil)
    {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): required device-resident "
            "buffer missing on %s (compact_fp=%p compact_idx=%p hash_data=%p "
            "hash_data_off=%p salt_data=%p salt_off=%p salt_lens=%p) -- "
            "did caller invoke gpu_metal_set_compact_table + "
            "gpu_metal_set_salt?\n",
            __FILE__, __LINE__, hostname,
            (void *)buf_compact_fp, (void *)buf_compact_idx,
            (void *)buf_hash_data, (void *)buf_hash_data_off,
            (void *)buf_salt_data, (void *)buf_salt_off,
            (void *)buf_salt_lens);
        exit(1);
    }

    /* Overflow buffer placeholders. gpu_metal_set_compact_table does NOT
     * provision the overflow MTLBuffers (that's gpu_metal_set_overflow's
     * job); when overflow_count == 0 the harness never has reason to call
     * the overflow setter, but the kernel still binds the three overflow
     * buffer args (guarded inside the kernel by `if (overflow_count > 0u)`).
     * Provision tiny placeholders via gpu_metal_set_overflow(..., 0). */
    if (buf_overflow_keys == nil || buf_overflow_hashes == nil ||
        buf_overflow_offsets == nil) {
        if (gpu_metal_set_overflow(0, NULL, NULL, NULL, NULL, 0) != 0) {
            fprintf(stderr,
                "FATAL: %s:%d e347 validate (Metal): overflow placeholder "
                "provisioning failed on %s\n",
                __FILE__, __LINE__, hostname);
            exit(1);
        }
    }

    id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)pso;

    /* --- 1. Per-invocation MTLBuffers (packed candidates + word offsets). */
    id<MTLBuffer> buf_packed =
        [mtl_device newBufferWithBytes:packed_words
                                length:packed_size
                               options:MTLResourceStorageModeShared];
    if (buf_packed == nil) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): packed newBuffer(%u B) "
            "failed on %s\n",
            __FILE__, __LINE__, packed_size, hostname);
        exit(1);
    }
    id<MTLBuffer> buf_chunk =
        [mtl_device newBufferWithBytes:word_offset
                                length:(size_t)num_words * sizeof(uint32_t)
                               options:MTLResourceStorageModeShared];
    if (buf_chunk == nil) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): chunk newBuffer(%u entries) "
            "failed on %s\n",
            __FILE__, __LINE__, num_words, hostname);
        exit(1);
    }

    /* --- 2. Payload (OCLParams + trailing slop for ovr_set/ovr_gid). */
    size_t payload_bytes = sizeof(OCLParams) + 32;
    id<MTLBuffer> buf_payload_local =
        [mtl_device newBufferWithLength:payload_bytes
                                options:MTLResourceStorageModeShared];
    if (buf_payload_local == nil) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): payload newBuffer(%zu) "
            "failed on %s\n",
            __FILE__, __LINE__, payload_bytes, hostname);
        exit(1);
    }
    {
        uint8_t *p = (uint8_t *)[buf_payload_local contents];
        memset(p, 0, payload_bytes);
        OCLParams params;
        memset(&params, 0, sizeof(params));
        params.compact_mask     = cache_compact_mask;
        params.num_words        = num_words;
        params.num_salts        = num_salts;
        params.salt_start       = 0;
        params.max_probe        = 128;
        params.hash_data_count  = cache_hash_data_count;
        params.max_hits         = (uint32_t)max_hits;
        params.overflow_count   = (uint32_t)cache_overflow_count;
        params.max_iter         = 1;
        params.iter_count       = 1;
        params.num_rules        = 1;
        params.base_word_idx    = 0;
        params.packed_size      = packed_size;
        memcpy(p, &params, sizeof(params));
    }

    /* --- 3. hit_count + hashes_shown buffers. Use dedicated short-lived
     * buffers so this dispatch can't trample on a concurrent gpu_metal
     * production dispatcher state. */
    id<MTLBuffer> buf_hit_count =
        [mtl_device newBufferWithLength:sizeof(uint32_t)
                                options:MTLResourceStorageModeShared];
    if (buf_hit_count == nil) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): hit_count newBuffer "
            "failed on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }
    *((uint32_t *)[buf_hit_count contents]) = 0u;

    size_t hs_slots = (size_t)cache_hash_data_count + (size_t)cache_overflow_count;
    if (hs_slots == 0) hs_slots = 1;
    size_t hs_bytes = hs_slots * sizeof(uint32_t);
    if (hs_bytes < 64) hs_bytes = 64;
    id<MTLBuffer> buf_hs_local =
        [mtl_device newBufferWithLength:hs_bytes
                                options:MTLResourceStorageModeShared];
    if (buf_hs_local == nil) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): hashes_shown newBuffer "
            "(%zu B) failed on %s\n",
            __FILE__, __LINE__, hs_bytes, hostname);
        exit(1);
    }
    memset([buf_hs_local contents], 0, hs_bytes);

    /* --- 4. Hits buffer (max_hits * HIT_STRIDE u32). */
    size_t hits_bytes = (size_t)max_hits * GPU_HIT_STRIDE * sizeof(uint32_t);
    id<MTLBuffer> buf_hits_local =
        [mtl_device newBufferWithLength:hits_bytes
                                options:MTLResourceStorageModeShared];
    if (buf_hits_local == nil) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): hits newBuffer(%zu B) "
            "failed on %s\n",
            __FILE__, __LINE__, hits_bytes, hostname);
        exit(1);
    }
    memset([buf_hits_local contents], 0, hits_bytes);

    /* --- 5. ovr_set + ovr_gid (separate atomic counters). The OpenCL
     * twin aliases them off payload+100/+104; the Metal emitter wires
     * them as explicit atomic_uint kernel args (atomic types can't be
     * cast through a non-atomic pointer in Metal). Tiny dedicated
     * buffers; zeroed init. */
    id<MTLBuffer> buf_ovr_set =
        [mtl_device newBufferWithLength:sizeof(uint32_t)
                                options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_ovr_gid =
        [mtl_device newBufferWithLength:sizeof(uint32_t)
                                options:MTLResourceStorageModeShared];
    if (buf_ovr_set == nil || buf_ovr_gid == nil) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): ovr_set/ovr_gid newBuffer "
            "failed on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }
    *((uint32_t *)[buf_ovr_set contents]) = 0u;
    *((uint32_t *)[buf_ovr_gid contents]) = 0xFFFFFFFFu;

    /* (Overflow buffer placeholder check moved to the precondition block
     * above; this slot retained as a comment anchor for code reading.) */

    /* --- 7. Encode + dispatch. */
    id<MTLCommandBuffer> cb = [mtl_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ps];
    [enc setBuffer:buf_payload_local   offset:0 atIndex:0];
    [enc setBuffer:buf_packed          offset:0 atIndex:1];
    [enc setBuffer:buf_chunk           offset:0 atIndex:2];
    [enc setBuffer:buf_salt_data       offset:0 atIndex:3];
    [enc setBuffer:buf_salt_off        offset:0 atIndex:4];
    [enc setBuffer:buf_salt_lens       offset:0 atIndex:5];
    [enc setBuffer:buf_compact_fp      offset:0 atIndex:6];
    [enc setBuffer:buf_compact_idx     offset:0 atIndex:7];
    [enc setBuffer:buf_hash_data       offset:0 atIndex:8];
    [enc setBuffer:buf_hash_data_off   offset:0 atIndex:9];
    [enc setBuffer:buf_hits_local      offset:0 atIndex:10];
    [enc setBuffer:buf_hit_count       offset:0 atIndex:11];
    [enc setBuffer:buf_overflow_keys   offset:0 atIndex:12];
    [enc setBuffer:buf_overflow_hashes offset:0 atIndex:13];
    [enc setBuffer:buf_overflow_offsets offset:0 atIndex:14];
    [enc setBuffer:buf_hs_local        offset:0 atIndex:15];
    [enc setBuffer:buf_ovr_set         offset:0 atIndex:16];
    [enc setBuffer:buf_ovr_gid         offset:0 atIndex:17];

    /* Grid: gid is decomposed inside the kernel as
     *   word_idx       = gid / num_salt_chunks
     *   salt_chunk_idx = gid % num_salt_chunks
     * with SALT_BATCH=64 inner serial loop. Total threads =
     * num_words * num_salt_chunks. */
    uint32_t num_salt_chunks = (num_salts + 64u - 1u) / 64u;
    if (num_salt_chunks == 0u) num_salt_chunks = 1u;
    NSUInteger total_threads = (NSUInteger)num_words * (NSUInteger)num_salt_chunks;

    NSUInteger tg = ps.maxTotalThreadsPerThreadgroup;
    if (tg == 0) tg = 64;
    if (tg > 128) tg = 128;       /* keep WG small for harness predictability */
    if (tg > total_threads) tg = total_threads;
    if (tg == 0) tg = 1;
    MTLSize threadgroup = MTLSizeMake(tg, 1, 1);
    NSUInteger n_groups = (total_threads + tg - 1) / tg;
    MTLSize grid = MTLSizeMake(n_groups, 1, 1);

    [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];

    if (cb.status != MTLCommandBufferStatusCompleted) {
        fprintf(stderr,
            "FATAL: %s:%d e347 validate (Metal): command buffer status=%ld "
            "on %s (error=%s)\n",
            __FILE__, __LINE__, (long)cb.status, hostname,
            cb.error ? [[cb.error localizedDescription] UTF8String]
                     : "(none)");
        exit(1);
    }

    /* --- 8. Read hits back. */
    uint32_t hcount = *((uint32_t *)[buf_hit_count contents]);
    if ((int)hcount > max_hits) hcount = (uint32_t)max_hits;
    if (hcount > 0) {
        memcpy(out_hits, [buf_hits_local contents],
               (size_t)hcount * GPU_HIT_STRIDE * sizeof(uint32_t));
    }
    *out_n_hits = (int)hcount;

    /* ARC releases the local buffer refs at function exit. */
    return 0;
}

/* hx codegen sub-phase 5a.3 (2026-05-22): byte-exact validation dispatch
 * for the codegen-emitted MAKE_MD5PASS family Metal kernel. Metal sibling
 * of gpu_opencl_kernelb_family_validate_dispatch (gpu/gpu_opencl.c rev 1.192).
 *
 * Differences from gpu_metal_e347_validate_dispatch (the sibling above):
 *   - num_salts is fixed to 1 (kernel ignores; the harness still needs
 *     gpu_metal_set_salt to provision the per-device salt MTLBuffers
 *     for the kernel arg bindings; family kernel body never reads them).
 *   - Per-thread topology (no SALT_BATCH outer loop); grid = padded
 *     (num_words rounded up to tg_size==64 multiple). Kernel's
 *     `if (word_idx >= params.num_words) return;` early-exits padding
 *     threads.
 *   - Kernel signature is identical to e347 (18 args at the same
 *     [[buffer(N)]] indices). Bindings unchanged. */
int gpu_metal_kernelb_family_validate_dispatch(
    int dev_idx, void *pso,
    const unsigned char *packed_words, uint32_t packed_size,
    const uint32_t *word_offset, uint32_t num_words,
    uint32_t *out_hits, int *out_n_hits, int max_hits)
{
    char hostname[256] = "unknown";
    gethostname(hostname, sizeof(hostname) - 1);

    if (!metal_ready) {
        fprintf(stderr,
            "FATAL: %s:%d family validate (Metal): not initialized on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }
    if (dev_idx != 0) {
        fprintf(stderr,
            "FATAL: %s:%d family validate (Metal): dev_idx=%d out of range "
            "on %s\n",
            __FILE__, __LINE__, dev_idx, hostname);
        exit(1);
    }
    if (!pso || !packed_words || !word_offset || !out_hits || !out_n_hits ||
        num_words == 0 || packed_size == 0)
    {
        fprintf(stderr,
            "FATAL: %s:%d family validate (Metal): bad args on %s "
            "(pso=%p pw=%p wo=%p oh=%p onh=%p nw=%u ps=%u)\n",
            __FILE__, __LINE__, hostname,
            pso, (void *)packed_words, (void *)word_offset,
            (void *)out_hits, (void *)out_n_hits,
            num_words, packed_size);
        exit(1);
    }
    if (mtl_device == nil || mtl_queue == nil) {
        fprintf(stderr,
            "FATAL: %s:%d family validate (Metal): mtl_device or mtl_queue "
            "nil on %s\n",
            __FILE__, __LINE__, hostname);
        exit(1);
    }
    if (buf_compact_fp == nil || buf_compact_idx == nil ||
        buf_hash_data == nil || buf_hash_data_off == nil ||
        buf_salt_data == nil || buf_salt_off == nil || buf_salt_lens == nil)
    {
        fprintf(stderr,
            "FATAL: %s:%d family validate (Metal): required device-resident "
            "buffer missing on %s (compact_fp=%p compact_idx=%p hash_data=%p "
            "hash_data_off=%p salt_data=%p salt_off=%p salt_lens=%p) -- "
            "did caller invoke gpu_metal_set_compact_table + "
            "gpu_metal_set_salt?\n",
            __FILE__, __LINE__, hostname,
            (void *)buf_compact_fp, (void *)buf_compact_idx,
            (void *)buf_hash_data, (void *)buf_hash_data_off,
            (void *)buf_salt_data, (void *)buf_salt_off,
            (void *)buf_salt_lens);
        exit(1);
    }

    /* Overflow buffer placeholders (same precondition as e347 sibling). */
    if (buf_overflow_keys == nil || buf_overflow_hashes == nil ||
        buf_overflow_offsets == nil) {
        if (gpu_metal_set_overflow(0, NULL, NULL, NULL, NULL, 0) != 0) {
            fprintf(stderr,
                "FATAL: %s:%d family validate (Metal): overflow placeholder "
                "provisioning failed on %s\n",
                __FILE__, __LINE__, hostname);
            exit(1);
        }
    }

    id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)pso;

    @autoreleasepool {
        /* --- 1. Per-invocation MTLBuffers. */
        id<MTLBuffer> buf_packed =
            [mtl_device newBufferWithBytes:packed_words
                                    length:packed_size
                                   options:MTLResourceStorageModeShared];
        if (buf_packed == nil) {
            fprintf(stderr,
                "FATAL: %s:%d family validate (Metal): packed newBuffer(%u B) "
                "failed on %s\n",
                __FILE__, __LINE__, packed_size, hostname);
            exit(1);
        }
        id<MTLBuffer> buf_chunk =
            [mtl_device newBufferWithBytes:word_offset
                                    length:(size_t)num_words * sizeof(uint32_t)
                                   options:MTLResourceStorageModeShared];
        if (buf_chunk == nil) {
            fprintf(stderr,
                "FATAL: %s:%d family validate (Metal): chunk newBuffer "
                "(%u entries) failed on %s\n",
                __FILE__, __LINE__, num_words, hostname);
            exit(1);
        }

        /* --- 2. Payload. num_salts=1 (kernel ignores; the buffer slot
         * still needs to be bound). */
        size_t payload_bytes = sizeof(OCLParams) + 32;
        id<MTLBuffer> buf_payload_local =
            [mtl_device newBufferWithLength:payload_bytes
                                    options:MTLResourceStorageModeShared];
        if (buf_payload_local == nil) {
            fprintf(stderr,
                "FATAL: %s:%d family validate (Metal): payload newBuffer(%zu) "
                "failed on %s\n",
                __FILE__, __LINE__, payload_bytes, hostname);
            exit(1);
        }
        {
            uint8_t *p = (uint8_t *)[buf_payload_local contents];
            memset(p, 0, payload_bytes);
            OCLParams params;
            memset(&params, 0, sizeof(params));
            params.compact_mask     = cache_compact_mask;
            params.num_words        = num_words;
            params.num_salts        = 1;
            params.salt_start       = 0;
            params.max_probe        = 128;
            params.hash_data_count  = cache_hash_data_count;
            params.max_hits         = (uint32_t)max_hits;
            params.overflow_count   = (uint32_t)cache_overflow_count;
            params.max_iter         = 1;
            params.iter_count       = 1;
            params.num_rules        = 1;
            params.base_word_idx    = 0;
            params.packed_size      = packed_size;
            memcpy(p, &params, sizeof(params));
        }

        /* --- 3. hit_count + hashes_shown. */
        id<MTLBuffer> buf_hit_count =
            [mtl_device newBufferWithLength:sizeof(uint32_t)
                                    options:MTLResourceStorageModeShared];
        if (buf_hit_count == nil) {
            fprintf(stderr,
                "FATAL: %s:%d family validate (Metal): hit_count newBuffer "
                "failed on %s\n",
                __FILE__, __LINE__, hostname);
            exit(1);
        }
        *((uint32_t *)[buf_hit_count contents]) = 0u;

        size_t hs_slots = (size_t)cache_hash_data_count +
                          (size_t)cache_overflow_count;
        if (hs_slots == 0) hs_slots = 1;
        size_t hs_bytes = hs_slots * sizeof(uint32_t);
        if (hs_bytes < 64) hs_bytes = 64;
        id<MTLBuffer> buf_hs_local =
            [mtl_device newBufferWithLength:hs_bytes
                                    options:MTLResourceStorageModeShared];
        if (buf_hs_local == nil) {
            fprintf(stderr,
                "FATAL: %s:%d family validate (Metal): hashes_shown "
                "newBuffer(%zu B) failed on %s\n",
                __FILE__, __LINE__, hs_bytes, hostname);
            exit(1);
        }
        memset([buf_hs_local contents], 0, hs_bytes);

        /* --- 4. Hits buffer. */
        size_t hits_bytes = (size_t)max_hits * GPU_HIT_STRIDE * sizeof(uint32_t);
        id<MTLBuffer> buf_hits_local =
            [mtl_device newBufferWithLength:hits_bytes
                                    options:MTLResourceStorageModeShared];
        if (buf_hits_local == nil) {
            fprintf(stderr,
                "FATAL: %s:%d family validate (Metal): hits newBuffer(%zu B) "
                "failed on %s\n",
                __FILE__, __LINE__, hits_bytes, hostname);
            exit(1);
        }
        memset([buf_hits_local contents], 0, hits_bytes);

        /* --- 5. ovr_set + ovr_gid (separate atomic counters). */
        id<MTLBuffer> buf_ovr_set =
            [mtl_device newBufferWithLength:sizeof(uint32_t)
                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_ovr_gid =
            [mtl_device newBufferWithLength:sizeof(uint32_t)
                                    options:MTLResourceStorageModeShared];
        if (buf_ovr_set == nil || buf_ovr_gid == nil) {
            fprintf(stderr,
                "FATAL: %s:%d family validate (Metal): ovr_set/ovr_gid "
                "newBuffer failed on %s\n",
                __FILE__, __LINE__, hostname);
            exit(1);
        }
        *((uint32_t *)[buf_ovr_set contents]) = 0u;
        *((uint32_t *)[buf_ovr_gid contents]) = 0xFFFFFFFFu;

        /* --- 6. Encode + dispatch. */
        id<MTLCommandBuffer> cb = [mtl_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ps];
        [enc setBuffer:buf_payload_local    offset:0 atIndex:0];
        [enc setBuffer:buf_packed           offset:0 atIndex:1];
        [enc setBuffer:buf_chunk            offset:0 atIndex:2];
        [enc setBuffer:buf_salt_data        offset:0 atIndex:3];
        [enc setBuffer:buf_salt_off         offset:0 atIndex:4];
        [enc setBuffer:buf_salt_lens        offset:0 atIndex:5];
        [enc setBuffer:buf_compact_fp       offset:0 atIndex:6];
        [enc setBuffer:buf_compact_idx      offset:0 atIndex:7];
        [enc setBuffer:buf_hash_data        offset:0 atIndex:8];
        [enc setBuffer:buf_hash_data_off    offset:0 atIndex:9];
        [enc setBuffer:buf_hits_local       offset:0 atIndex:10];
        [enc setBuffer:buf_hit_count        offset:0 atIndex:11];
        [enc setBuffer:buf_overflow_keys    offset:0 atIndex:12];
        [enc setBuffer:buf_overflow_hashes  offset:0 atIndex:13];
        [enc setBuffer:buf_overflow_offsets offset:0 atIndex:14];
        [enc setBuffer:buf_hs_local         offset:0 atIndex:15];
        [enc setBuffer:buf_ovr_set          offset:0 atIndex:16];
        [enc setBuffer:buf_ovr_gid          offset:0 atIndex:17];

        /* Per-thread; one thread per word. Pad gsize up to tg_size=64
         * multiple. Kernel's `if (word_idx >= params.num_words) return;`
         * early-exits the padding threads. Mirrors OpenCL family
         * dispatch lsize=64 + padded gsize. */
        const NSUInteger tg_size = 64;
        NSUInteger total_threads = (NSUInteger)num_words;
        NSUInteger n_groups = (total_threads + tg_size - 1) / tg_size;
        if (n_groups == 0) n_groups = 1;
        MTLSize threadgroup = MTLSizeMake(tg_size, 1, 1);
        MTLSize grid = MTLSizeMake(n_groups, 1, 1);

        [enc dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if (cb.status != MTLCommandBufferStatusCompleted) {
            fprintf(stderr,
                "FATAL: %s:%d family validate (Metal): command buffer "
                "status=%ld on %s (error=%s)\n",
                __FILE__, __LINE__, (long)cb.status, hostname,
                cb.error ? [[cb.error localizedDescription] UTF8String]
                         : "(none)");
            exit(1);
        }

        /* --- 7. Read hits back. */
        uint32_t hcount = *((uint32_t *)[buf_hit_count contents]);
        if ((int)hcount > max_hits) hcount = (uint32_t)max_hits;
        if (hcount > 0) {
            memcpy(out_hits, [buf_hits_local contents],
                   (size_t)hcount * GPU_HIT_STRIDE * sizeof(uint32_t));
        }
        *out_n_hits = (int)hcount;
    }  /* autoreleasepool */

    return 0;
}

#endif /* METAL_GPU */
