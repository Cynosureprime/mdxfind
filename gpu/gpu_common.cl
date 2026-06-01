/*
 * $Revision: 1.34 $
 * $Log: gpu_common.cl,v $
 * Revision 1.34  2026/05/28 04:44:24  dlr
 * 5b.4b.2: lift gost_block + 4KB GOST_SBOX_1..4 (TEST set, precomputed host-side from gosthash_init) into gpu_common.cl; donor gosthash/gosthash.c gosthash_compress; chi compression 8-iter U-V key schedule + GOST 28147-89 32-round Feistel + 3 LFSR product-matrix stages; noinline R6; R8 line comments; body+macros byte-identical to validated C-mirror 27 of 27 PASS; sum8 checksum carry and dual finalization carried by emit helper not block
 *
 * Revision 1.33  2026/05/28 04:16:44  dlr
 * sub-phase 5b4a1 lift snefru_block primitive into gpu_common.cl from RHash-master librhash snefru.c rhash_snefru_process_block lines 765-841 plus 16 KB rhash_snefru_sbox 4096 uint table snefru.c line 50 librhash live CPU oracle for e175 e177 ONE PARAMETERISED block per D18.1.a snefru_block state block is256 handles BOTH Snefru-128 is256 0 and Snefru-256 is256 1 same 8-round S-box transform for both widths only 3 sites differ on width collapse to compile-time if is256 branches JIT folds is256 literal per emit kernel stays fully unrolled per variant site 1 W fill SNE256 loads hash 4 to 7 into W 4 to 7 then 8 message words 32-byte data block SNE128 loads only hash 0 to 3 then 12 message words 48-byte data block site 2 final state XOR-back SNE256 also writes hash 4 to 7 site 3 block size 48 vs 32 bytes handled by emit helper round count FIXED at 8 SNEFRU_NUMBER_OF_ROUNDS no security pass parameter schedule and state output BIG-ENDIAN donor be2me_32 message load be32_copy state output message words assembled via SNEFRU_BE32 BE byte order state output byte-swap handled in emit helper per feedback_be_state_primitives_need_byteswap_in_codegen rot 0x18100810 4-cycle ROTR via OpenCL rotate with 32 minus low-byte count SNEFERU_UPDATE_W macro SNEFRU_UPD on W 0 to 15 plus W 0 W 15 ROTR each cycle const budget SNEFRU_SBOX 16 KB cumulative post-Tier-3 plus 16 KB equals 42-43 KB of 64 KB Pascal Apple Silicon budget noinline per R6 R8 no nested block comments donor stripped line comments only inserted after haval5_block C-mirror test_snefru_port 56 of 56 cells PASS vs librhash both widths 28 lengths incl block-boundary 31 32 33 47 48 49 63 64 65 95 96 97 127 128 129 byte-exact R15 pre-flight test_snefru_vectors 230 of 230 PASS used by emit_outer_snefru_concat_then_hash 5b4a3 family helper
 *
 * Revision 1.32  2026/05/28 03:52:21  dlr
 * sub-phase 5b3c2 lift haval5_block primitive into gpu_common.cl from mhash-0.9.9.9 lib haval.c havalTransform5 lines 412-615 public-domain donor Paulo Barreto 1998 5-pass HAVAL compression 8-uint state 32-uint LE-packed M 128-byte block pass 1 F1 no constants pass 2 F2 32 RC pass 3 F3 32 RC pass 4 F4 32 RC pass 5 F5 32 RC 0xBA3BF050 to 0x409F60C4 128 round constants total CRITICAL pass 1-4 word schedules and F-arg orderings DIFFER from BOTH havalTransform3 AND havalTransform4 verbatim transcription NOT a reuse of haval3 or haval4 passes feedforward state plus T final 8 steps shares HAVAL_F1-F5 HAVAL_ROTR32 HAVAL_IV with haval3_block noinline per R6 R8 no nested block comments inserted after haval4_block C-mirror test_haval5_port 60 of 60 cells PASS vs sph_haval all 5 widths x 12 inputs incl multi-block boundary R13 block 118 0x29 5-pass byte differs from 4-pass 0x21 and 3-pass 0x19 verified in dumped e131 e155 kernels completes 15-variant HAVAL family Tier 3
 *
 * Revision 1.31  2026/05/28 03:18:38  dlr
 * sub-phase 5b3b2 lift haval4_block primitive into gpu_common cl from mhash-0.9.9.9 lib haval c havalTransform4 lines 244-409 public-domain donor Paulo Barreto 1998 4-pass HAVAL compression 8-uint state 32-uint LE-packed M 128-byte block pass 1 F1 no constants pass 2 F2 32 RC pass 3 F3 32 RC pass 4 F4 32 RC 0x7A325381 to 0x137A3BE4 128 round constants total CRITICAL pass 1-3 word schedules and F-arg orderings DIFFER from havalTransform3 verbatim transcription NOT a reuse of haval3 passes feedforward state plus T final 8 steps shares HAVAL_F1-F5 HAVAL_ROTR32 HAVAL_IV with haval3_block noinline per R6 R8 no nested block comments inserted after haval3_block C-mirror test_haval4_port 60 of 60 cells PASS vs sph_haval all 5 widths x 12 inputs incl multi-block boundary R13 block 118 0x21 4-pass byte differs from 3-pass 0x19 verified in dumped kernel
 *
 * Revision 1.30  2026/05/28 02:16:40  dlr
 * sub-phase 5b3a1 lift haval3_block primitive into gpu_common cl from mhash-0.9.9.9 lib haval c havalTransform3 lines 113-241 public-domain donor Paulo Barreto 1998 3-pass HAVAL compression canonical default per Zheng-Pieprzyk-Seberry 1993 paper 8-uint state 32-uint LE-packed M 128-byte block 5 Boolean F1-F5 round functions inline macros HAVAL_F1 through F5 XOR AND OR NOT only no bitselect HAVAL_ROTR32 via OpenCL rotate with complemented count 96 round constants inlined as compile-time hex literals pass 1 F1 no constants pass 2 F2 0x452821E6 to 0xC25A59B5 pass 3 F3 0x9C30D539 to 0x6C24CF5C feedforward state plus T final 8 steps HAVAL_IV exposed as __constant 32 bytes Pi-fractional constants noinline per R6 feedback_md5_block_noinline_pascal R7 no nested block comments donor stripped line comments only inserted after tiger_block keeping LE-family clustered used by emit_outer_haval_concat_then_hash 5b3a3 family helper post-port C-mirror test 60 of 60 cells PASS vs sph_haval all 5 widths x 12 inputs incl multi-block boundary cases 84 85 86 118 119 120 bytes validates compression body 0x01 pad toggle block 118 119 param encoding all 5 digest folds R1 R3 R13 byte-exact
 *
 * Revision 1.29  2026/05/27 23:03:08  dlr
 * sub-phase 5b2b1 lift tiger_block primitive into gpu_common.cl from RHash-master librhash tiger.c rhash_tiger_process_block lines 109-151 and tiger_sbox.c rhash_tiger_sboxes 4x256 ulong tables 8KB constant memory noinline per R5 caller convention mirrors sha512_block and wrl_block but with LE-packed M0..M7 and LE state output direct extract no byte-swap epilogue Tiger IV 0123456789abcdef fedcba9876543210 f096a5b4c3b2e187 initialized by caller donor le2me_64 swap elided emit helper packs M in LE-ulong form 3-pass round structure pass1 mul 5 KeySchedule pass2 mul 7 KeySchedule pass3 mul 9 with rotated arg order c a b then b c a per tiger.c line 140 142 144 feedforward state0 XOR a state1 b SUB state1 state2 PLUS c per tiger.c 148-150 CPU translation tested standalone against 7 NESSIE vectors plus 1M-a stress PASS byte-exact vs rhash_tiger and sph_tiger R12 pre-flight 16 of 16 cells PASS 2026-05-27 R7 no nested block comments donor stripped uses line comments only inserted between wrl_block and rmd160_block keeping LE-family clustered used by emit_outer_tiger_concat_then_hash 5b2b3 family helper
 *
 * Revision 1.28  2026/05/27 22:20:24  dlr
 * sub-phase 5b2a1 lift wrl_block primitive into gpu_common.cl from librhash whirlpool.c lines 60-128 and whirlpool_sbox.c rhash_whirlpool_sbox 8x256 ulong tables noinline per R5 16KB constant memory budget within Pascal and Apple Silicon 64KB CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE caller convention mirrors sha512_block M0 through M7 must be BE-packed ulongs no internal byte swap needed donor be2me_64 swap elided emit helper packs M in BE-ulong form caller-side WRL_RC 10 round constants WRL_OP macro mirrors WHIRLPOOL_OP librhash whirlpool.c lines 43 to 52 8 S-box rows gather by i-th byte of state j Miyaguchi-Preneel compression 10 round AES-style mini-cipher state IV all zero CPU translation test against 8 NESSIE vectors PASS byte-exact vs librhash and OpenSSL R12 pre-flight 16 of 16 cells PASS 2026-05-27 R7 no nested block comments donor stripped uses line comments only inserted between sha512_to_hex_lc and rmd160_block keeping BE-family clustered used by emit_outer_wrl_concat_then_hash 5b2a3 family helper
 *
 * Revision 1.27  2026/05/27 17:45:05  dlr
 * sub-phase 5b1b1 lift rmd128_block primitive into gpu_common.cl from rmd128.c compress lines 39-196 Bosselaers 1996 reference 4-uint state vs rmd160 5-uint dual pipeline left line F1 F2 F3 F4 right line F4 F3 F2 F1 Bosselaers Table 4 R2 ordering with inline comments citing rmd128.c lines 117-186 right pipeline ordering reuses RMD_F1 through F4 macros from rmd160 section above defines local RMD128_STEP 4-arg variant without E and without C-rotation matching rmd128.h FF GG HH II macro shape directly LL1 through LL4 left line round-K macros RR1 through RR4 right line round-K macros K constants 0x50a28be6 0x5c4dd124 0x6d703ef3 0 LE schedule packing matches BYTES_TO_DWORD rmd128.h convention output hash 0 to 3 LE uint32 chaining values CPU oracle RIPEMD128 byte-exact match noinline per feedback_md5_block_noinline_pascal R5 cross-mix at end mirrors rmd128.c lines 188-193 with hash 1 to 3 written before hash 0 overwritten R7 no nested block comments donor stripped uses line comments only inserted between rmd160_block and rmd320_block keeping RMD family clustered used by emit_outer_rmd128_concat_then_hash 5b1b3 family helper
 *
 * Revision 1.26  2026/05/27 16:57:21  dlr
 * sub-phase 5b1a1 lift md2_block primitive into gpu_common.cl from md2 md2.c md2_transform B-Con and sph_md2 reference 16-byte data block 48-byte state 16-byte checksum 256-byte MD2_PI S-box in __constant address space 18-round state transform plus per-block checksum update R3 copy-paste S-box from md2 md2.c lines 17-34 no retype R5 noinline discipline R4 __constant S-box no premature __local cache R7 no nested block comments inner update_checksum flag selects per-data-block checksum update versus final-call skip per RFC errata signature uchar state plus uchar checksum plus const uchar data plus int update_checksum used by emit_outer_md2_concat_then_hash family helper
 *
 * Revision 1.25  2026/05/23 05:22:54  dlr
 * sub-phase 5a.4 lift md4_block primitive into gpu_common.cl from gpu_md4_core.cl md4_compress for the family MD5PASS hx codegen e122 emit body byte-for-byte mirror of gpu_md4_core.cl md4_compress 3 rounds 16 steps F G H round-2 constant 0x5A827999u round-3 constant 0x6ED9EBA1u round-1 constant 0 same IV as MD5 noinline matches md5_block signature uint pointer to 4 chaining values plus uint M[16] LE-schedule byte pack output is LE so no byte-swap before compact_fp probe needed wired in 5a.4 e122 MD4MD5PASS Metal twin lifts md4_block into metal_common.metal in same commit family MD5PASS gains MD4 outer primitive support validated PASS 8 of 8 on Pascal GTX 1080 OpenCL byte-exact
 *
 * Revision 1.24  2026/05/21 12:40:52  dlr
 * Phase 1 sub-phase 1a.2 D9.1.b rename overflow_first_rule to num_rules at offset 108 OCLParams field repurposed for kernel A1 A3 source rule count B3 path stops using slot host writes 1 when not applicable update prose comments accordingly
 *
 * Revision 1.23  2026/05/19 21:25:09  dlr
 * Phase 1 two-kernel pipeline: rename OCLParams reserved32[2] to base_word_idx + packed_size. ABI preserved bit-exactly - same offsets 80-87, same 128-byte struct. No live kernel reads either field (packed kernels retired B7.9). Named fields communicate actual purpose to Phase 2 implementing agents. Comment block updated to reference retirement. OpenCL only; Metal twin deferred.
 *
 * Revision 1.22  2026/05/19 05:45:13  dlr
 * Phase 1 Step A: replace sha512_block with flat 80-step scalar body. Lifts hashcat flat-unrolled SHA-512 transform pattern: 16 scalar w0_t..wf_t instead of W[80] array, eliminating local-mem spill on Pascal NVIDIA. MDX_SHA512_STEP_S/EXPAND_S/F0o/F1o macros added with USE_BITSELECT gate. Function signature void sha512_block(ulong *state, ulong *M) unchanged. OpenCL only; Metal twin deferred.
 *
 * Revision 1.21  2026/05/18 14:30:46  dlr
 * Phase 2h-A 2026-05-18 - add md5_block_from8 helper (noinline) that runs MD5 rounds 9-64 from a pre-rolled state. Symmetric with gpu_md5salt_core.cl template_pre_salt 8-round pre-roll. Saves 12.5 percent of outer MD5 work per salt for salted-MD5 chains. Same 64-round shape as md5_block; noinline matches Pascal register safety.
 *
 * Revision 1.20  2026/05/11 05:22:01  dlr
 * Backfill $Revision/$Log RCS keyword stanzas per feedback_rcs_keyword_stanzas.md. Passive 4-line comment block at top of file; no behavioral change. Hand-authored .cl file was missing required stanzas (per memory: all hand-authored .c/.h/.cl/.frag/.tmpl/.py/.sh files MUST contain $Revision/$Log keyword stanzas). Build green on .205 against the post-add files; OpenCL compile strips comments so no kernel behavior change.
 *
 */
/* md5salt.cl — OpenCL kernels for mdxfind MD5SALT GPU acceleration */

/* GPU Params: 128-byte uniform API. uint64 first, then uint32, then reserved.
 * Must match host-side OCLParams/MetalParams exactly.
 *
 * Memo B B1 (2026-05-03): cursor skeleton fields added at offsets 88-111
 * for two-cursor overflow restart (project_memo_b_dispatch_template.md §2).
 * Rules kernel does NOT use these in B1 (cursor=0 == today's behavior is the
 * locked contract; B3 wires read+CAS-min on overflow). base_word_idx + packed_size
 * (formerly reserved32[0..1]) at offsets 80-87. Packed kernels retired B7.9;
 * these slots are now named base_word_idx + packed_size per Phase 1 rename.
 *
 * B6 salt-axis (2026-05-06): num_salts_per_page at offset 112 (was reserved64[0])
 * communicates salt-page size to the kernel for combined_ridx packing.
 * Populated only when the salt-axis is active; ignored by unsalted kernels.
 *
 * B6.6 algo_mode (2026-05-06): generic per-algorithm runtime mode flag at
 * offset 120 (was reserved64[1] / reserved64). For kernels that share
 * infrastructure across minor variants (e.g., MD5SALT family: e31/e350/
 * e541/e542 differ only in inner-digest hex encoding step). Host sets this
 * field at dispatch time based on op; kernel branches on it. Reusable for
 * other hash families that have similar minor-variant clusters. */
typedef struct {
    ulong compact_mask;       /*  0: hash table mask */
    ulong mask_start;         /*  8: mask keyspace offset */
    ulong mask_base0;         /* 16: pre-decomposed positions 0-7 */
    ulong mask_base1;         /* 24: pre-decomposed positions 8-15 */
    uint  num_words;          /* 32: words in batch */
    uint  num_salts;          /* 36: salts for dispatch */
    uint  salt_start;         /* 40: starting salt index */
    uint  max_probe;          /* 44: compact table probe depth */
    uint  hash_data_count;    /* 48: hash_data entries */
    uint  max_hits;           /* 52: hit buffer capacity */
    uint  overflow_count;     /* 56: overflow table entries */
    uint  max_iter;           /* 60: iteration count (-i) */
    uint  num_masks;          /* 64: mask combinations per chunk */
    uint  n_prepend;          /* 68: prepend mask positions (-N) */
    uint  n_append;           /* 72: append mask positions (-n) */
    uint  iter_count;         /* 76: per-dispatch iteration (PHPBB3) */
    uint  base_word_idx;      /* 80-83: source word index this dispatch operates on.
                             *        Packed kernels (retired B7.9): was reserved32[0] / word_start.
                             *        Two-kernel pipeline (Phase 2+): kernel A source word;
                             *        kernel B prefix-cache key when candidates share base. */
    uint  packed_size;        /* 84-87: bytes in the packed candidate data.
                             *        Packed kernels (retired B7.9): was reserved32[1].
                             *        Two-kernel pipeline (Phase 2+): total bytes in
                             *        b_packed_buf for this dispatch. */
    /* B1 cursor skeleton — rules kernel reads as 0 in B1, B3 will wire. */
    uint  input_cursor_start; /* 88: B3 input cursor (lanes < cursor early-return) */
    uint  rule_cursor_start;  /* 92: B3 rule cursor */
    uint  inner_iter;         /* 96: BF Phase 1.8 — kernel inner iteration count
                               *      for BF chunks. Repurposed from B3
                               *      output_cursor_start (zero-read audit
                               *      2026-05-10: no kernel reads ever existed).
                               *      0 or 1 = today's behavior (loop runs once,
                               *      bit-identical). >1 = each work-item processes
                               *      inner_iter consecutive mask values for the
                               *      same (word, rule). Cap = 16. Unsalted BF
                               *      only; salted path forces inner_iter=1. */
    uint  overflow_first_set; /* 100: B3 kernel sets to 1 on first overflow lane */
    uint  overflow_first_word;/* 104: B3 word_idx CAS-min target */
    uint  num_rules;          /* 108: source rule count for kernel A1/A3;
                               *      reads as 1 when not applicable */
    ulong num_salts_per_page; /* 112: B6 salt-axis paging (was reserved64[0]) */
    uint  algo_mode;          /* 120: B6.6 per-algorithm runtime variant flag (was reserved64[1] high half) */
    uint  mask_offset_per_word; /* 124: BF chunk: word stride per BF chunk; 0 == not a BF chunk. Default 0 = today's behavior. */
} OCLParams;

/* Universal hit entry: fixed stride 19 uint32 words.
 * [0] word_idx  [1] salt_idx  [2] iter_num  [3..18] hash[0..15] */
#define HIT_STRIDE 19

/* Phase 6 BCRYPT (2026-05-08): workgroup-size constant for the BCRYPT
 * carrier kernel. Mirrors slab gpu_bcrypt.cl's BCRYPT_WG_SIZE=8 default
 * (line 26): each lane owns a 1024-uint __local partition (4 KB);
 * BCRYPT_WG_SIZE=8 lanes × 4 KB = 32 KB per WG, fits Pascal (48 KB),
 * RDNA (64 KB), Mali-T860 32 KB cap exactly (Phase 4 runtime probe via
 * CL_DEVICE_LOCAL_MEM_SIZE handles the Mali edge case — host disables
 * BCRYPT path on devices reporting <32 KB). The host's per-kernel local-
 * size override (gpu_opencl.c at the dispatch site) MUST match this
 * value so the kernel's reqd_work_group_size attribute is honored.
 * #ifndef guard keeps build_opts -DBCRYPT_WG_SIZE override path open
 * (e.g., Mali fallback to BCRYPT_WG_SIZE=4 future option). */
#ifndef BCRYPT_WG_SIZE
#define BCRYPT_WG_SIZE 8
#endif

#define EMIT_HIT_4(hits, hit_count, max_hits, widx, sidx, iter, a, b, c, d) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        hits[_base+3] = (a); hits[_base+4] = (b); hits[_base+5] = (c); hits[_base+6] = (d); \
        for (uint _z = 7; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); } }

#define EMIT_HIT_5(hits, hit_count, max_hits, widx, sidx, iter, h) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 5; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 8; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); } }

#define EMIT_HIT_6(hits, hit_count, max_hits, widx, sidx, iter, h) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 6; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 9; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); } }

#define EMIT_HIT_7(hits, hit_count, max_hits, widx, sidx, iter, h) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 7; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 10; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); } }

#define EMIT_HIT_8(hits, hit_count, max_hits, widx, sidx, iter, h) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 8; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 11; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); } }

/* B5 sub-batch 2: RIPEMD-320 emits 10 uint32 LE state words = 320-bit
 * digest. 3 + 10 = 13 hit-stride words; slots 13..18 zeroed. */
#define EMIT_HIT_10(hits, hit_count, max_hits, widx, sidx, iter, h) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 10; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 13; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); } }

#define EMIT_HIT_12(hits, hit_count, max_hits, widx, sidx, iter, h) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 12; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 15; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); } }

#define EMIT_HIT_16(hits, hit_count, max_hits, widx, sidx, iter, h) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 16; _i++) hits[_base+3+_i] = (h)[_i]; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); } }

/* ---- B3 overflow-aware EMIT_HIT_N wrappers (Memo B §2) ----
 *
 * These wrap the EMIT_HIT_N macros above with the two-cursor overflow
 * protocol. When the kernel's atomic_add on hit_count returns a slot
 * >= max_hits, the lane records its geometric position into
 * overflow_first_word (CAS-min on the lane's gid) and signals
 * overflow_first_set, then exits without writing the hit. The host
 * re-issues with cursor_start advanced past overflow_first.
 *
 * Why CAS-min on gid (uint32) rather than on the (word, rule) pair:
 *   - OpenCL 1.2 portable atomics are 32-bit (atomic_min on uint).
 *     64-bit atomics require cl_khr_int64_base_atomics, not universal.
 *   - gid = rule_idx * n_words + word_idx is monotonic in lex order on
 *     (rule_idx, word_idx) given fixed n_words, so atomic_min on gid
 *     selects the lex-smallest (rule, word) lane.
 *   - For total <= 2^32 (16K * 100K = 1.638G fits), gid is a uint.
 *
 * Host re-derives word_cursor = overflow_gid % n_words,
 *               rule_cursor = overflow_gid / n_words.
 * The OCLParams.num_rules field at offset 108 (formerly
 * overflow_first_rule, unused in B3) is repurposed for sub-phase 1a.2:
 * kernel A1/A3 source rule count; B3 host writes 1 when not applicable.
 * Kernel CAS-min is on overflow_first_word interpreted as the lane gid.
 *
 * Sentinel: host inits overflow_first_word to 0xFFFFFFFFu (never a
 * valid lane gid). First overflowing lane wins the CAS-min unconditionally.
 *
 * Args added to N_or_overflow variants:
 *   ovr_set    : __global volatile uint* to overflow_first_set (offset 100)
 *   ovr_gid    : __global volatile uint* to overflow_first_word (offset 104)
 *   lane_gid   : the lane's gid (rule_idx * n_words + word_idx)
 *
 * Lane behavior:
 *   - atomic_add(hit_count, 1) -> slot
 *   - if slot < max_hits: normal emit
 *   - else: CAS-min loop on ovr_gid; on success, atomic_or(ovr_set, 1).
 *     No emit. Caller MUST return (or otherwise stop) immediately.
 *
 * Note: B3 minimal scope leaves output_cursor de-dupe optimization OFF
 * (host trusts cursor + re-dispatch uniquely covers all lanes). */

/* Helper: CAS-min on a __global volatile uint, OpenCL 1.2 portable.
 * Loops until our value either lost (bigger than current) or wrote. */
#define _OVR_CASMIN_GID(ovr_gid, lane_gid) \
    do { \
        uint _cur, _new = (uint)(lane_gid); \
        do { \
            _cur = *(ovr_gid); \
            if (_new >= _cur) break; \
        } while (atomic_cmpxchg((ovr_gid), _cur, _new) != _cur); \
    } while (0)

#define EMIT_HIT_4_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, a, b, c, d, \
                               ovr_set, ovr_gid, lane_gid) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        hits[_base+3] = (a); hits[_base+4] = (b); hits[_base+5] = (c); hits[_base+6] = (d); \
        for (uint _z = 7; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); \
      } else { \
        _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
        atomic_or((ovr_set), 1u); \
      } }

#define EMIT_HIT_5_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                               ovr_set, ovr_gid, lane_gid) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 5; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 8; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); \
      } else { \
        _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
        atomic_or((ovr_set), 1u); \
      } }

#define EMIT_HIT_7_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                               ovr_set, ovr_gid, lane_gid) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 7; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 10; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); \
      } else { \
        _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
        atomic_or((ovr_set), 1u); \
      } }

#define EMIT_HIT_8_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                               ovr_set, ovr_gid, lane_gid) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 8; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 11; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); \
      } else { \
        _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
        atomic_or((ovr_set), 1u); \
      } }

/* B5 sub-batch 2: RIPEMD-320 (10 uint32 LE state words). */
#define EMIT_HIT_10_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                ovr_set, ovr_gid, lane_gid) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 10; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 13; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); \
      } else { \
        _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
        atomic_or((ovr_set), 1u); \
      } }

/* ---- B3 dedup-aware EMIT_HIT_N_DEDUP_OR_OVERFLOW wrappers ----
 *
 * The simpler EMIT_HIT_N_OR_OVERFLOW wrappers above interact incorrectly
 * with hashes_shown[]-based on-GPU dedup: if a lane sets the dedup bit
 * (winning the atomic_or "first cracker" race) and THEN fails to emit
 * because the hit-buffer is full, the dedup bit stays set -- blocking
 * any RE-ISSUE lane from emitting the same target. Result: silent
 * crack loss on overflow.
 *
 * The DEDUP_OR_OVERFLOW variants take three extra args
 *   (hashes_shown, matched_idx, dedup_mask)
 * and atomically roll back the dedup bit on overflow. Caller must NOT
 * pre-set the dedup bit; the macro does it.
 *
 * Sequence:
 *   1. atomic_or hashes_shown[matched_idx] with dedup_mask. If old bit
 *      was set, another lane already emitted -- skip silently.
 *   2. atomic_inc hit_count. If slot < max_hits, emit normally.
 *   3. Else: atomic_and hashes_shown[matched_idx] with ~dedup_mask
 *      (clears OUR bit; we know we set it because step 1 saw it unset).
 *      Then CAS-min lane_gid + signal ovr_set.
 *
 * Race-safety on the rollback (step 3):
 *   - We are the sole lane to have set this bit (atomic_or step 1 was
 *     uncontested for our mask).
 *   - Between our atomic_or and our atomic_and, no other lane can set
 *     OUR bit (already set; OR is idempotent, returns "already set").
 *     Other lanes see the bit set and skip. They don't emit.
 *   - After our atomic_and, the bit is unset. A re-issue lane may now
 *     set it and emit. Correctness preserved. */
#define EMIT_HIT_4_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, \
                                      a, b, c, d, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _dm = (uint)(dedup_mask); \
        uint _mi = (uint)(matched_idx); \
        if ((atomic_or(&(hashes_shown)[_mi], _dm) & _dm) == 0u) { \
            uint _slot = atomic_add((hit_count), 1u); \
            if (_slot < (max_hits)) { \
                uint _base = _slot * HIT_STRIDE; \
                (hits)[_base]   = (widx); (hits)[_base+1] = (sidx); (hits)[_base+2] = (iter); \
                (hits)[_base+3] = (a);    (hits)[_base+4] = (b); \
                (hits)[_base+5] = (c);    (hits)[_base+6] = (d); \
                for (uint _z = 7; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0; \
                mem_fence(CLK_GLOBAL_MEM_FENCE); \
            } else { \
                atomic_and(&(hashes_shown)[_mi], ~_dm); \
                _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
                atomic_or((ovr_set), 1u); \
            } \
        } \
    } while (0)

#define EMIT_HIT_5_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _dm = (uint)(dedup_mask); \
        uint _mi = (uint)(matched_idx); \
        if ((atomic_or(&(hashes_shown)[_mi], _dm) & _dm) == 0u) { \
            uint _slot = atomic_add((hit_count), 1u); \
            if (_slot < (max_hits)) { \
                uint _base = _slot * HIT_STRIDE; \
                (hits)[_base]   = (widx); (hits)[_base+1] = (sidx); (hits)[_base+2] = (iter); \
                for (uint _i = 0; _i < 5; _i++) (hits)[_base+3+_i] = (h)[_i]; \
                for (uint _z = 8; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0; \
                mem_fence(CLK_GLOBAL_MEM_FENCE); \
            } else { \
                atomic_and(&(hashes_shown)[_mi], ~_dm); \
                _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
                atomic_or((ovr_set), 1u); \
            } \
        } \
    } while (0)

/* Phase 6 BCRYPT (2026-05-08): 6-uint32 hash word emit (24-byte digest
 * for bcrypt). Mirrors EMIT_HIT_5/7 pattern. Used by gpu_bcrypt_core.cl
 * template_emit_hit_or_overflow. Probe-side fingerprint is first 16
 * bytes (4 words); emit packs all 6 LE words at hits[_base+3..+8]. */
#define EMIT_HIT_6_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _dm = (uint)(dedup_mask); \
        uint _mi = (uint)(matched_idx); \
        if ((atomic_or(&(hashes_shown)[_mi], _dm) & _dm) == 0u) { \
            uint _slot = atomic_add((hit_count), 1u); \
            if (_slot < (max_hits)) { \
                uint _base = _slot * HIT_STRIDE; \
                (hits)[_base]   = (widx); (hits)[_base+1] = (sidx); (hits)[_base+2] = (iter); \
                for (uint _i = 0; _i < 6; _i++) (hits)[_base+3+_i] = (h)[_i]; \
                for (uint _z = 9; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0; \
                mem_fence(CLK_GLOBAL_MEM_FENCE); \
            } else { \
                atomic_and(&(hashes_shown)[_mi], ~_dm); \
                _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
                atomic_or((ovr_set), 1u); \
            } \
        } \
    } while (0)

#define EMIT_HIT_7_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _dm = (uint)(dedup_mask); \
        uint _mi = (uint)(matched_idx); \
        if ((atomic_or(&(hashes_shown)[_mi], _dm) & _dm) == 0u) { \
            uint _slot = atomic_add((hit_count), 1u); \
            if (_slot < (max_hits)) { \
                uint _base = _slot * HIT_STRIDE; \
                (hits)[_base]   = (widx); (hits)[_base+1] = (sidx); (hits)[_base+2] = (iter); \
                for (uint _i = 0; _i < 7; _i++) (hits)[_base+3+_i] = (h)[_i]; \
                for (uint _z = 10; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0; \
                mem_fence(CLK_GLOBAL_MEM_FENCE); \
            } else { \
                atomic_and(&(hashes_shown)[_mi], ~_dm); \
                _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
                atomic_or((ovr_set), 1u); \
            } \
        } \
    } while (0)

#define EMIT_HIT_8_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _dm = (uint)(dedup_mask); \
        uint _mi = (uint)(matched_idx); \
        if ((atomic_or(&(hashes_shown)[_mi], _dm) & _dm) == 0u) { \
            uint _slot = atomic_add((hit_count), 1u); \
            if (_slot < (max_hits)) { \
                uint _base = _slot * HIT_STRIDE; \
                (hits)[_base]   = (widx); (hits)[_base+1] = (sidx); (hits)[_base+2] = (iter); \
                for (uint _i = 0; _i < 8; _i++) (hits)[_base+3+_i] = (h)[_i]; \
                for (uint _z = 11; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0; \
                mem_fence(CLK_GLOBAL_MEM_FENCE); \
            } else { \
                atomic_and(&(hashes_shown)[_mi], ~_dm); \
                _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
                atomic_or((ovr_set), 1u); \
            } \
        } \
    } while (0)

/* B5 sub-batch 2: RIPEMD-320 dedup+overflow-aware emit (10 uint32). */
#define EMIT_HIT_10_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                       hashes_shown, matched_idx, dedup_mask, \
                                       ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _dm = (uint)(dedup_mask); \
        uint _mi = (uint)(matched_idx); \
        if ((atomic_or(&(hashes_shown)[_mi], _dm) & _dm) == 0u) { \
            uint _slot = atomic_add((hit_count), 1u); \
            if (_slot < (max_hits)) { \
                uint _base = _slot * HIT_STRIDE; \
                (hits)[_base]   = (widx); (hits)[_base+1] = (sidx); (hits)[_base+2] = (iter); \
                for (uint _i = 0; _i < 10; _i++) (hits)[_base+3+_i] = (h)[_i]; \
                for (uint _z = 13; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0; \
                mem_fence(CLK_GLOBAL_MEM_FENCE); \
            } else { \
                atomic_and(&(hashes_shown)[_mi], ~_dm); \
                _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
                atomic_or((ovr_set), 1u); \
            } \
        } \
    } while (0)

/* ---- B5 SHA384/SHA512 EMIT_HIT_{12,16} overflow-aware wrappers ----
 *
 * 64-bit-state algorithms (SHA-384 = 6 ulong = 12 uint32; SHA-512 = 8
 * ulong = 16 uint32) extend the EMIT_HIT family. Same protocol as the
 * 4/5/7/8-width variants above: simple _OR_OVERFLOW (no on-GPU dedup
 * — caller does its own dedup at the host) and _DEDUP_OR_OVERFLOW (the
 * macro performs the hashes_shown atomic_or + atomic_and rollback).
 *
 * EMIT_HIT_16 fills the entire HIT_STRIDE (3 + 16 = 19 words). EMIT_HIT_12
 * leaves a 4-word zeroed tail (3 + 12 = 15, slots 15..18 zeroed). */
#define EMIT_HIT_12_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                ovr_set, ovr_gid, lane_gid) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 12; _i++) hits[_base+3+_i] = (h)[_i]; \
        for (uint _z = 15; _z < HIT_STRIDE; _z++) hits[_base+_z] = 0; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); \
      } else { \
        _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
        atomic_or((ovr_set), 1u); \
      } }

#define EMIT_HIT_16_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                ovr_set, ovr_gid, lane_gid) \
    { uint _slot = atomic_add(hit_count, 1u); \
      if (_slot < max_hits) { \
        uint _base = _slot * HIT_STRIDE; \
        hits[_base] = (widx); hits[_base+1] = (sidx); hits[_base+2] = (iter); \
        for (uint _i = 0; _i < 16; _i++) hits[_base+3+_i] = (h)[_i]; \
        mem_fence(CLK_GLOBAL_MEM_FENCE); \
      } else { \
        _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
        atomic_or((ovr_set), 1u); \
      } }

#define EMIT_HIT_12_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                       hashes_shown, matched_idx, dedup_mask, \
                                       ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _dm = (uint)(dedup_mask); \
        uint _mi = (uint)(matched_idx); \
        if ((atomic_or(&(hashes_shown)[_mi], _dm) & _dm) == 0u) { \
            uint _slot = atomic_add((hit_count), 1u); \
            if (_slot < (max_hits)) { \
                uint _base = _slot * HIT_STRIDE; \
                (hits)[_base]   = (widx); (hits)[_base+1] = (sidx); (hits)[_base+2] = (iter); \
                for (uint _i = 0; _i < 12; _i++) (hits)[_base+3+_i] = (h)[_i]; \
                for (uint _z = 15; _z < HIT_STRIDE; _z++) (hits)[_base+_z] = 0; \
                mem_fence(CLK_GLOBAL_MEM_FENCE); \
            } else { \
                atomic_and(&(hashes_shown)[_mi], ~_dm); \
                _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
                atomic_or((ovr_set), 1u); \
            } \
        } \
    } while (0)

#define EMIT_HIT_16_DEDUP_OR_OVERFLOW(hits, hit_count, max_hits, widx, sidx, iter, h, \
                                       hashes_shown, matched_idx, dedup_mask, \
                                       ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _dm = (uint)(dedup_mask); \
        uint _mi = (uint)(matched_idx); \
        if ((atomic_or(&(hashes_shown)[_mi], _dm) & _dm) == 0u) { \
            uint _slot = atomic_add((hit_count), 1u); \
            if (_slot < (max_hits)) { \
                uint _base = _slot * HIT_STRIDE; \
                (hits)[_base]   = (widx); (hits)[_base+1] = (sidx); (hits)[_base+2] = (iter); \
                for (uint _i = 0; _i < 16; _i++) (hits)[_base+3+_i] = (h)[_i]; \
                mem_fence(CLK_GLOBAL_MEM_FENCE); \
            } else { \
                atomic_and(&(hashes_shown)[_mi], ~_dm); \
                _OVR_CASMIN_GID((ovr_gid), (lane_gid)); \
                atomic_or((ovr_set), 1u); \
            } \
        } \
    } while (0)

__constant uint K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};

#define FF(a,b,c,d,m,s,k) { a += ((b&c)|(~b&d)) + m + k; a = b + rotate(a,s); }
#define GG(a,b,c,d,m,s,k) { a += ((d&b)|(~d&c)) + m + k; a = b + rotate(a,s); }
#define HH(a,b,c,d,m,s,k) { a += (b^c^d) + m + k; a = b + rotate(a,s); }
#define II(a,b,c,d,m,s,k) { a += (c^(~d|b)) + m + k; a = b + rotate(a,s); }

/* ---- MD2 S-box pi_subst[256] (RFC 1319 Table T; identical to sph_md2's
 *      sph_md2_PI table and B-Con md2.c s[]). Copy-paste-no-retype from
 *      md2/md2.c lines 17-34 (Tier 1 risk R3 mitigation). Used by
 *      md2_block compression and md2_checksum_update below. Placed in
 *      __constant address space per Tier 1 risk R4. */
__constant uchar MD2_PI[256] = {
    41, 46, 67, 201, 162, 216, 124, 1, 61, 54, 84, 161, 236, 240, 6,
    19, 98, 167, 5, 243, 192, 199, 115, 140, 152, 147, 43, 217, 188,
    76, 130, 202, 30, 155, 87, 60, 253, 212, 224, 22, 103, 66, 111, 24,
    138, 23, 229, 18, 190, 78, 196, 214, 218, 158, 222, 73, 160, 251,
    245, 142, 187, 47, 238, 122, 169, 104, 121, 145, 21, 178, 7, 63,
    148, 194, 16, 137, 11, 34, 95, 33, 128, 127, 93, 154, 90, 144, 50,
    39, 53, 62, 204, 231, 191, 247, 151, 3, 255, 25, 48, 179, 72, 165,
    181, 209, 215, 94, 146, 42, 172, 86, 170, 198, 79, 184, 56, 210,
    150, 164, 125, 182, 118, 252, 107, 226, 156, 116, 4, 241, 69, 157,
    112, 89, 100, 113, 135, 32, 134, 91, 207, 101, 230, 45, 168, 2, 27,
    96, 37, 173, 174, 176, 185, 246, 28, 70, 97, 105, 52, 64, 126, 15,
    85, 71, 163, 35, 221, 81, 175, 58, 195, 92, 249, 206, 186, 197,
    234, 38, 44, 83, 13, 110, 133, 40, 132, 9, 211, 223, 205, 244, 65,
    129, 77, 82, 106, 220, 55, 200, 108, 193, 171, 250, 36, 225, 123,
    8, 12, 189, 177, 74, 120, 136, 149, 139, 227, 99, 232, 109, 233,
    203, 213, 254, 59, 0, 29, 57, 242, 239, 183, 14, 102, 88, 208, 228,
    166, 119, 114, 248, 235, 117, 75, 10, 49, 68, 80, 180, 143, 237,
    31, 26, 219, 153, 141, 51, 159, 17, 131, 20
};

/* ---- MD2 block compression (lifted from md2/md2.c md2_transform 2026-05-27
 *      for hx codegen sub-phase 5b.1a e120 MD2MD5PASS family emit).
 *
 * RFC 1319 (with errata applied; matches sph_md2 + B-Con md2.c). MD2
 * is byte-oriented: 16-byte data block, 48-byte state, 16-byte
 * checksum. Compression mutates BOTH state and checksum in place
 * (the checksum update is per-data-block AND uses the running
 * checksum's last byte as the initial t -- a key spec subtlety).
 *
 * Signature: state and checksum are pointer-to-uchar (48 + 16 bytes
 * respectively). data is pointer-to-uchar (16 bytes). Caller manages
 * the 16-byte fill buffer + checksum carry across update calls. The
 * "final" step calls md2_block once with the PKCS-padded last block,
 * then once more with checksum-as-block (and passes a NULL or unused
 * checksum pointer the second time -- but per RFC errata the checksum
 * IS NOT updated on the second call; caller distinguishes via the
 * update_checksum flag).
 *
 * 18-round state transform + per-data-block checksum update. The
 * checksum carry-byte t persists across blocks; caller stores it in
 * checksum[15] between calls (matches B-Con md2.c md2_transform body).
 *
 * R5 noinline discipline per feedback_md5_block_noinline_pascal.md
 * (Pascal register budget; mirrors md4_block / md5_block / rmd160_block).
 * R7 NO nested block comments: all donor block comments stripped during
 * port; this header uses only the surrounding outer block. */
__attribute__((noinline)) void md2_block(uchar *state, uchar *checksum,
                                         const uchar *data,
                                         int update_checksum) {
    int j, k;
    uint t;

    // Spec step 1: copy block into state[16..31]; xor into state[32..47].
    for (j = 0; j < 16; j++) {
        state[j + 16] = data[j];
        state[j + 32] = (uchar)(state[j + 16] ^ state[j]);
    }

    // Spec step 2: 18 rounds of state transform.
    t = 0;
    for (j = 0; j < 18; j++) {
        for (k = 0; k < 48; k++) {
            state[k] = (uchar)(state[k] ^ MD2_PI[t]);
            t = state[k];
        }
        t = (t + (uint)j) & 0xFFu;
    }

    // Spec step 3: per-block checksum update (uses MD2_PI[data[j] ^ prev]).
    // Skipped on the final checksum-block call (per RFC errata: the
    // checksum block itself is NOT folded back into the checksum).
    if (update_checksum) {
        t = checksum[15];
        for (j = 0; j < 16; j++) {
            checksum[j] = (uchar)(checksum[j] ^ MD2_PI[data[j] ^ t]);
            t = checksum[j];
        }
    }
}

/* ---- MD4 block function (lifted from gpu_md4_core.cl 2026-05-23 for
 *      hx codegen sub-phase 5a.4 e122 MD4MD5PASS family emit).
 *
 * RFC 1320 MD4. Same IV as MD5. State is 4 uint32 LE chaining values.
 * Message schedule is 16 uint32 LE words; caller packs the 64-byte
 * block into M[0..15] little-endian. Three rounds of 16 steps each;
 * F/G/H round functions; round-2 constant 0x5A827999u; round-3
 * constant 0x6ED9EBA1u; round-1 constant 0.
 *
 * Signature mirrors md5_block (uint *h0..h3 + uint *M). Caller may
 * reuse the same M[16] buffer used by md5_block since the schedule
 * convention (LE-packed bytes) is identical.
 *
 * Output is LE; NO byte-swap needed before compact_fp probe. */
__attribute__((noinline)) void md4_block(uint *h0, uint *h1, uint *h2, uint *h3, uint *M) {
    uint a = *h0, b = *h1, c = *h2, d = *h3;
#define MD4_F(x,y,z) (((x)&(y)) | ((~(x))&(z)))
#define MD4_G(x,y,z) (((x)&(y)) | ((x)&(z)) | ((y)&(z)))
#define MD4_H(x,y,z) ((x)^(y)^(z))
#define MD4_R1(a,b,c,d,k,s) a = rotate(a + MD4_F(b,c,d) + M[k], (uint)(s))
#define MD4_R2(a,b,c,d,k,s) a = rotate(a + MD4_G(b,c,d) + M[k] + 0x5A827999u, (uint)(s))
#define MD4_R3(a,b,c,d,k,s) a = rotate(a + MD4_H(b,c,d) + M[k] + 0x6ED9EBA1u, (uint)(s))
    MD4_R1(a,b,c,d, 0, 3); MD4_R1(d,a,b,c, 1, 7); MD4_R1(c,d,a,b, 2,11); MD4_R1(b,c,d,a, 3,19);
    MD4_R1(a,b,c,d, 4, 3); MD4_R1(d,a,b,c, 5, 7); MD4_R1(c,d,a,b, 6,11); MD4_R1(b,c,d,a, 7,19);
    MD4_R1(a,b,c,d, 8, 3); MD4_R1(d,a,b,c, 9, 7); MD4_R1(c,d,a,b,10,11); MD4_R1(b,c,d,a,11,19);
    MD4_R1(a,b,c,d,12, 3); MD4_R1(d,a,b,c,13, 7); MD4_R1(c,d,a,b,14,11); MD4_R1(b,c,d,a,15,19);
    MD4_R2(a,b,c,d, 0, 3); MD4_R2(d,a,b,c, 4, 5); MD4_R2(c,d,a,b, 8, 9); MD4_R2(b,c,d,a,12,13);
    MD4_R2(a,b,c,d, 1, 3); MD4_R2(d,a,b,c, 5, 5); MD4_R2(c,d,a,b, 9, 9); MD4_R2(b,c,d,a,13,13);
    MD4_R2(a,b,c,d, 2, 3); MD4_R2(d,a,b,c, 6, 5); MD4_R2(c,d,a,b,10, 9); MD4_R2(b,c,d,a,14,13);
    MD4_R2(a,b,c,d, 3, 3); MD4_R2(d,a,b,c, 7, 5); MD4_R2(c,d,a,b,11, 9); MD4_R2(b,c,d,a,15,13);
    MD4_R3(a,b,c,d, 0, 3); MD4_R3(d,a,b,c, 8, 9); MD4_R3(c,d,a,b, 4,11); MD4_R3(b,c,d,a,12,15);
    MD4_R3(a,b,c,d, 2, 3); MD4_R3(d,a,b,c,10, 9); MD4_R3(c,d,a,b, 6,11); MD4_R3(b,c,d,a,14,15);
    MD4_R3(a,b,c,d, 1, 3); MD4_R3(d,a,b,c, 9, 9); MD4_R3(c,d,a,b, 5,11); MD4_R3(b,c,d,a,13,15);
    MD4_R3(a,b,c,d, 3, 3); MD4_R3(d,a,b,c,11, 9); MD4_R3(c,d,a,b, 7,11); MD4_R3(b,c,d,a,15,15);
#undef MD4_F
#undef MD4_G
#undef MD4_H
#undef MD4_R1
#undef MD4_R2
#undef MD4_R3
    *h0 += a; *h1 += b; *h2 += c; *h3 += d;
}

__attribute__((noinline)) void md5_block(uint *h0, uint *h1, uint *h2, uint *h3, uint *M) {
    uint a = *h0, b = *h1, c = *h2, d = *h3;
    FF(a,b,c,d,M[0],(uint)7,0xd76aa478u);  FF(d,a,b,c,M[1],(uint)12,0xe8c7b756u);
    FF(c,d,a,b,M[2],(uint)17,0x242070dbu);  FF(b,c,d,a,M[3],(uint)22,0xc1bdceeeu);
    FF(a,b,c,d,M[4],(uint)7,0xf57c0fafu);   FF(d,a,b,c,M[5],(uint)12,0x4787c62au);
    FF(c,d,a,b,M[6],(uint)17,0xa8304613u);  FF(b,c,d,a,M[7],(uint)22,0xfd469501u);
    FF(a,b,c,d,M[8],(uint)7,0x698098d8u);   FF(d,a,b,c,M[9],(uint)12,0x8b44f7afu);
    FF(c,d,a,b,M[10],(uint)17,0xffff5bb1u); FF(b,c,d,a,M[11],(uint)22,0x895cd7beu);
    FF(a,b,c,d,M[12],(uint)7,0x6b901122u);  FF(d,a,b,c,M[13],(uint)12,0xfd987193u);
    FF(c,d,a,b,M[14],(uint)17,0xa679438eu); FF(b,c,d,a,M[15],(uint)22,0x49b40821u);
    GG(a,b,c,d,M[1],(uint)5,0xf61e2562u);   GG(d,a,b,c,M[6],(uint)9,0xc040b340u);
    GG(c,d,a,b,M[11],(uint)14,0x265e5a51u); GG(b,c,d,a,M[0],(uint)20,0xe9b6c7aau);
    GG(a,b,c,d,M[5],(uint)5,0xd62f105du);   GG(d,a,b,c,M[10],(uint)9,0x02441453u);
    GG(c,d,a,b,M[15],(uint)14,0xd8a1e681u); GG(b,c,d,a,M[4],(uint)20,0xe7d3fbc8u);
    GG(a,b,c,d,M[9],(uint)5,0x21e1cde6u);   GG(d,a,b,c,M[14],(uint)9,0xc33707d6u);
    GG(c,d,a,b,M[3],(uint)14,0xf4d50d87u);  GG(b,c,d,a,M[8],(uint)20,0x455a14edu);
    GG(a,b,c,d,M[13],(uint)5,0xa9e3e905u);  GG(d,a,b,c,M[2],(uint)9,0xfcefa3f8u);
    GG(c,d,a,b,M[7],(uint)14,0x676f02d9u);  GG(b,c,d,a,M[12],(uint)20,0x8d2a4c8au);
    HH(a,b,c,d,M[5],(uint)4,0xfffa3942u);   HH(d,a,b,c,M[8],(uint)11,0x8771f681u);
    HH(c,d,a,b,M[11],(uint)16,0x6d9d6122u); HH(b,c,d,a,M[14],(uint)23,0xfde5380cu);
    HH(a,b,c,d,M[1],(uint)4,0xa4beea44u);   HH(d,a,b,c,M[4],(uint)11,0x4bdecfa9u);
    HH(c,d,a,b,M[7],(uint)16,0xf6bb4b60u);  HH(b,c,d,a,M[10],(uint)23,0xbebfbc70u);
    HH(a,b,c,d,M[13],(uint)4,0x289b7ec6u);  HH(d,a,b,c,M[0],(uint)11,0xeaa127fau);
    HH(c,d,a,b,M[3],(uint)16,0xd4ef3085u);  HH(b,c,d,a,M[6],(uint)23,0x04881d05u);
    HH(a,b,c,d,M[9],(uint)4,0xd9d4d039u);   HH(d,a,b,c,M[12],(uint)11,0xe6db99e5u);
    HH(c,d,a,b,M[15],(uint)16,0x1fa27cf8u); HH(b,c,d,a,M[2],(uint)23,0xc4ac5665u);
    II(a,b,c,d,M[0],(uint)6,0xf4292244u);   II(d,a,b,c,M[7],(uint)10,0x432aff97u);
    II(c,d,a,b,M[14],(uint)15,0xab9423a7u); II(b,c,d,a,M[5],(uint)21,0xfc93a039u);
    II(a,b,c,d,M[12],(uint)6,0x655b59c3u);  II(d,a,b,c,M[3],(uint)10,0x8f0ccc92u);
    II(c,d,a,b,M[10],(uint)15,0xffeff47du); II(b,c,d,a,M[1],(uint)21,0x85845dd1u);
    II(a,b,c,d,M[8],(uint)6,0x6fa87e4fu);   II(d,a,b,c,M[15],(uint)10,0xfe2ce6e0u);
    II(c,d,a,b,M[6],(uint)15,0xa3014314u);  II(b,c,d,a,M[13],(uint)21,0x4e0811a1u);
    II(a,b,c,d,M[4],(uint)6,0xf7537e82u);   II(d,a,b,c,M[11],(uint)10,0xbd3af235u);
    II(c,d,a,b,M[2],(uint)15,0x2ad7d2bbu);  II(b,c,d,a,M[9],(uint)21,0xeb86d391u);
    *h0 += a; *h1 += b; *h2 += c; *h3 += d;
}

/* md5_block_from8: Phase 2h-A 2026-05-18 — runs MD5 compression rounds 9-64
 * starting from a pre-rolled state (a8,b8,c8,d8) that already reflects
 * rounds 1-8 applied to message words M[0..7] starting from the IV.
 *
 * Use case: salted-MD5 chains where the same hex32 prefix (M[0..7]) is
 * combined with many different salts (M[8..15]). The caller pre-rolls
 * rounds 1-8 ONCE per (word, rule, mask) in template_pre_salt, then this
 * helper finishes the remaining 56 rounds ONCE per salt. Saves 8/64 =
 * 12.5% of outer MD5 work per (word, salt).
 *
 * h0..h3 inputs should be the original MD5 IV (0x67452301u etc) — the
 * epilogue adds them to the final round-64 state, matching md5_block's
 * `*h0 += a` semantics. (a8,b8,c8,d8) is the IV-relative round-8 output,
 * NOT IV-added; the IV addition is the WHOLE-block-end accumulator.
 *
 * `noinline` matches md5_block to preserve Pascal register budget.
 * Verified empirically 2026-05-18 fpga GTX 1080: removing noinline
 * regressed e31 wall 454s → 510s, throughput 780 → 745 Mh/s. Even with
 * the minimal kernel (gpu_md5salt_minimal.cl) reducing baseline register
 * pressure 2.4x, post-inline reg pressure dropped occupancy enough to
 * outweigh function-call savings. DO NOT REMOVE without re-testing. */
__attribute__((noinline)) void md5_block_from8(uint *h0, uint *h1, uint *h2, uint *h3,
                                                uint a8, uint b8, uint c8, uint d8,
                                                uint *M) {
    uint a = a8, b = b8, c = c8, d = d8;
    /* Rounds 9-16 (FF, uses M[8..15]) */
    FF(a,b,c,d,M[8],(uint)7,0x698098d8u);   FF(d,a,b,c,M[9],(uint)12,0x8b44f7afu);
    FF(c,d,a,b,M[10],(uint)17,0xffff5bb1u); FF(b,c,d,a,M[11],(uint)22,0x895cd7beu);
    FF(a,b,c,d,M[12],(uint)7,0x6b901122u);  FF(d,a,b,c,M[13],(uint)12,0xfd987193u);
    FF(c,d,a,b,M[14],(uint)17,0xa679438eu); FF(b,c,d,a,M[15],(uint)22,0x49b40821u);
    /* Rounds 17-32 (GG) */
    GG(a,b,c,d,M[1],(uint)5,0xf61e2562u);   GG(d,a,b,c,M[6],(uint)9,0xc040b340u);
    GG(c,d,a,b,M[11],(uint)14,0x265e5a51u); GG(b,c,d,a,M[0],(uint)20,0xe9b6c7aau);
    GG(a,b,c,d,M[5],(uint)5,0xd62f105du);   GG(d,a,b,c,M[10],(uint)9,0x02441453u);
    GG(c,d,a,b,M[15],(uint)14,0xd8a1e681u); GG(b,c,d,a,M[4],(uint)20,0xe7d3fbc8u);
    GG(a,b,c,d,M[9],(uint)5,0x21e1cde6u);   GG(d,a,b,c,M[14],(uint)9,0xc33707d6u);
    GG(c,d,a,b,M[3],(uint)14,0xf4d50d87u);  GG(b,c,d,a,M[8],(uint)20,0x455a14edu);
    GG(a,b,c,d,M[13],(uint)5,0xa9e3e905u);  GG(d,a,b,c,M[2],(uint)9,0xfcefa3f8u);
    GG(c,d,a,b,M[7],(uint)14,0x676f02d9u);  GG(b,c,d,a,M[12],(uint)20,0x8d2a4c8au);
    /* Rounds 33-48 (HH) */
    HH(a,b,c,d,M[5],(uint)4,0xfffa3942u);   HH(d,a,b,c,M[8],(uint)11,0x8771f681u);
    HH(c,d,a,b,M[11],(uint)16,0x6d9d6122u); HH(b,c,d,a,M[14],(uint)23,0xfde5380cu);
    HH(a,b,c,d,M[1],(uint)4,0xa4beea44u);   HH(d,a,b,c,M[4],(uint)11,0x4bdecfa9u);
    HH(c,d,a,b,M[7],(uint)16,0xf6bb4b60u);  HH(b,c,d,a,M[10],(uint)23,0xbebfbc70u);
    HH(a,b,c,d,M[13],(uint)4,0x289b7ec6u);  HH(d,a,b,c,M[0],(uint)11,0xeaa127fau);
    HH(c,d,a,b,M[3],(uint)16,0xd4ef3085u);  HH(b,c,d,a,M[6],(uint)23,0x04881d05u);
    HH(a,b,c,d,M[9],(uint)4,0xd9d4d039u);   HH(d,a,b,c,M[12],(uint)11,0xe6db99e5u);
    HH(c,d,a,b,M[15],(uint)16,0x1fa27cf8u); HH(b,c,d,a,M[2],(uint)23,0xc4ac5665u);
    /* Rounds 49-64 (II) */
    II(a,b,c,d,M[0],(uint)6,0xf4292244u);   II(d,a,b,c,M[7],(uint)10,0x432aff97u);
    II(c,d,a,b,M[14],(uint)15,0xab9423a7u); II(b,c,d,a,M[5],(uint)21,0xfc93a039u);
    II(a,b,c,d,M[12],(uint)6,0x655b59c3u);  II(d,a,b,c,M[3],(uint)10,0x8f0ccc92u);
    II(c,d,a,b,M[10],(uint)15,0xffeff47du); II(b,c,d,a,M[1],(uint)21,0x85845dd1u);
    II(a,b,c,d,M[8],(uint)6,0x6fa87e4fu);   II(d,a,b,c,M[15],(uint)10,0xfe2ce6e0u);
    II(c,d,a,b,M[6],(uint)15,0xa3014314u);  II(b,c,d,a,M[13],(uint)21,0x4e0811a1u);
    II(a,b,c,d,M[4],(uint)6,0xf7537e82u);   II(d,a,b,c,M[11],(uint)10,0xbd3af235u);
    II(c,d,a,b,M[2],(uint)15,0x2ad7d2bbu);  II(b,c,d,a,M[9],(uint)21,0xeb86d391u);
    *h0 += a; *h1 += b; *h2 += c; *h3 += d;
}

/* MD5 compress for padding block of 64-byte message: M={0x80,0,..,0,512,0}.
 * All constants — zero memory access. Compiler folds M[g] into round constants. */
void md5_block_pad64(uint *h0, uint *h1, uint *h2, uint *h3) {
    uint a = *h0, b = *h1, c = *h2, d = *h3;
    FF(a,b,c,d, 0x80u, (uint)7,  0xd76aa478u);  FF(d,a,b,c, 0u,(uint)12, 0xe8c7b756u);
    FF(c,d,a,b, 0u,   (uint)17, 0x242070dbu);  FF(b,c,d,a, 0u,(uint)22, 0xc1bdceeeu);
    FF(a,b,c,d, 0u,    (uint)7, 0xf57c0fafu);  FF(d,a,b,c, 0u,(uint)12, 0x4787c62au);
    FF(c,d,a,b, 0u,   (uint)17, 0xa8304613u);  FF(b,c,d,a, 0u,(uint)22, 0xfd469501u);
    FF(a,b,c,d, 0u,    (uint)7, 0x698098d8u);  FF(d,a,b,c, 0u,(uint)12, 0x8b44f7afu);
    FF(c,d,a,b, 0u,   (uint)17, 0xffff5bb1u);  FF(b,c,d,a, 0u,(uint)22, 0x895cd7beu);
    FF(a,b,c,d, 0u,    (uint)7, 0x6b901122u);  FF(d,a,b,c, 0u,(uint)12, 0xfd987193u);
    FF(c,d,a,b, 512u, (uint)17, 0xa679438eu);  FF(b,c,d,a, 0u,(uint)22, 0x49b40821u);
    GG(a,b,c,d, 0u,    (uint)5, 0xf61e2562u);  GG(d,a,b,c, 0u, (uint)9, 0xc040b340u);
    GG(c,d,a,b, 0u,   (uint)14, 0x265e5a51u);  GG(b,c,d,a, 0x80u,(uint)20,0xe9b6c7aau);
    GG(a,b,c,d, 0u,    (uint)5, 0xd62f105du);  GG(d,a,b,c, 0u, (uint)9, 0x02441453u);
    GG(c,d,a,b, 0u,   (uint)14, 0xd8a1e681u);  GG(b,c,d,a, 0u,(uint)20, 0xe7d3fbc8u);
    GG(a,b,c,d, 0u,    (uint)5, 0x21e1cde6u);  GG(d,a,b,c, 512u,(uint)9, 0xc33707d6u);
    GG(c,d,a,b, 0u,   (uint)14, 0xf4d50d87u);  GG(b,c,d,a, 0u,(uint)20, 0x455a14edu);
    GG(a,b,c,d, 0u,    (uint)5, 0xa9e3e905u);  GG(d,a,b,c, 0u, (uint)9, 0xfcefa3f8u);
    GG(c,d,a,b, 0u,   (uint)14, 0x676f02d9u);  GG(b,c,d,a, 0u,(uint)20, 0x8d2a4c8au);
    HH(a,b,c,d, 0u,    (uint)4, 0xfffa3942u);  HH(d,a,b,c, 0u,(uint)11, 0x8771f681u);
    HH(c,d,a,b, 0u,   (uint)16, 0x6d9d6122u);  HH(b,c,d,a, 512u,(uint)23,0xfde5380cu);
    HH(a,b,c,d, 0u,    (uint)4, 0xa4beea44u);  HH(d,a,b,c, 0u,(uint)11, 0x4bdecfa9u);
    HH(c,d,a,b, 0u,   (uint)16, 0xf6bb4b60u);  HH(b,c,d,a, 0u,(uint)23, 0xbebfbc70u);
    HH(a,b,c,d, 0u,    (uint)4, 0x289b7ec6u);  HH(d,a,b,c, 0x80u,(uint)11,0xeaa127fau);
    HH(c,d,a,b, 0u,   (uint)16, 0xd4ef3085u);  HH(b,c,d,a, 0u,(uint)23, 0x04881d05u);
    HH(a,b,c,d, 0u,    (uint)4, 0xd9d4d039u);  HH(d,a,b,c, 0u,(uint)11, 0xe6db99e5u);
    HH(c,d,a,b, 0u,   (uint)16, 0x1fa27cf8u);  HH(b,c,d,a, 0u,(uint)23, 0xc4ac5665u);
    II(a,b,c,d, 0x80u, (uint)6, 0xf4292244u);  II(d,a,b,c, 0u,(uint)10, 0x432aff97u);
    II(c,d,a,b, 512u, (uint)15, 0xab9423a7u);  II(b,c,d,a, 0u,(uint)21, 0xfc93a039u);
    II(a,b,c,d, 0u,    (uint)6, 0x655b59c3u);  II(d,a,b,c, 0u,(uint)10, 0x8f0ccc92u);
    II(c,d,a,b, 0u,   (uint)15, 0xffeff47du);  II(b,c,d,a, 0u,(uint)21, 0x85845dd1u);
    II(a,b,c,d, 0u,    (uint)6, 0x6fa87e4fu);  II(d,a,b,c, 0u,(uint)10, 0xfe2ce6e0u);
    II(c,d,a,b, 0u,   (uint)15, 0xa3014314u);  II(b,c,d,a, 0u,(uint)21, 0x4e0811a1u);
    II(a,b,c,d, 0u,    (uint)6, 0xf7537e82u);  II(d,a,b,c, 0u,(uint)10, 0xbd3af235u);
    II(c,d,a,b, 0u,   (uint)15, 0x2ad7d2bbu);  II(b,c,d,a, 0u,(uint)21, 0xeb86d391u);
    *h0 += a; *h1 += b; *h2 += c; *h3 += d;
}

ulong compact_mix(ulong k) { return k ^ (k >> 32); }

int probe_compact(uint hx, uint hy, uint hz, uint hw,
    __global const uint *compact_fp, __global const uint *compact_idx,
    ulong compact_mask, uint max_probe, uint hash_data_count,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, uint overflow_count)
{
    ulong key = ((ulong)hy << 32) | hx;
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    ulong pos = compact_mix(key) & compact_mask;
    for (int p = 0; p < (int)max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < hash_data_count) {
                ulong off = hash_data_off[idx];
                __global const uint *ref = (__global const uint *)(hash_data_buf + off);
                if (hx == ref[0] && hy == ref[1] && hz == ref[2] && hw == ref[3])
                    return 1;
            }
        }
        pos = (pos + 1) & compact_mask;
    }
    if (overflow_count > 0) {
        int lo = 0, hi = (int)overflow_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            ulong mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                __global const uint *oref = (__global const uint *)(overflow_hashes + ooff);
                if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;
                for (int d = mid-1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;
                }
                for (int d = mid+1; d < (int)overflow_count && overflow_keys[d] == key; d++) {
                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) return 1;
                }
                break;
            }
        }
    }
    return 0;
}

/* probe_compact_idx — variant of probe_compact that, on a hit, also
 * returns the matched target's index (its position in hash_data[]).
 * Used by on-GPU dedup (hashes_shown[]): the kernel atomic-increments
 * hashes_shown[idx]; only emit a hit when previous count was zero.
 * Mirrors hashcat's final_hash_pos semantics.
 *
 * Return: 1 on hit (and *out_idx is set), 0 on miss (*out_idx untouched).
 *
 * For overflow-table hits we do NOT have a stable hash_data[] index —
 * overflow entries are a separate table. We allocate dedup slots beyond
 * hash_data_count for the overflow table: idx = hash_data_count + overflow_pos.
 * The host-side dedup gate at mdxfind.c:8044 (*match_flags != job->op) is
 * the safety net for any race or sentinel collision.
 */
int probe_compact_idx(uint hx, uint hy, uint hz, uint hw,
    __global const uint *compact_fp, __global const uint *compact_idx,
    ulong compact_mask, uint max_probe, uint hash_data_count,
    __global const uchar *hash_data_buf, __global const ulong *hash_data_off,
    __global const ulong *overflow_keys, __global const uchar *overflow_hashes,
    __global const uint *overflow_offsets, uint overflow_count,
    uint *out_idx)
{
    ulong key = ((ulong)hy << 32) | hx;
    uint fp = (uint)(key >> 32);
    if (fp == 0) fp = 1;
    ulong pos = compact_mix(key) & compact_mask;
    for (int p = 0; p < (int)max_probe; p++) {
        uint cfp = compact_fp[pos];
        if (cfp == 0) break;
        if (cfp == fp) {
            uint idx = compact_idx[pos];
            if (idx < hash_data_count) {
                ulong off = hash_data_off[idx];
                __global const uint *ref = (__global const uint *)(hash_data_buf + off);
                if (hx == ref[0] && hy == ref[1] && hz == ref[2] && hw == ref[3]) {
                    *out_idx = idx;
                    return 1;
                }
            }
        }
        pos = (pos + 1) & compact_mask;
    }
    if (overflow_count > 0) {
        int lo = 0, hi = (int)overflow_count - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            ulong mkey = overflow_keys[mid];
            if (key < mkey) hi = mid - 1;
            else if (key > mkey) lo = mid + 1;
            else {
                uint ooff = overflow_offsets[mid];
                __global const uint *oref = (__global const uint *)(overflow_hashes + ooff);
                if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) {
                    *out_idx = hash_data_count + (uint)mid;
                    return 1;
                }
                for (int d = mid-1; d >= 0 && overflow_keys[d] == key; d--) {
                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) {
                        *out_idx = hash_data_count + (uint)d;
                        return 1;
                    }
                }
                for (int d = mid+1; d < (int)overflow_count && overflow_keys[d] == key; d++) {
                    oref = (__global const uint *)(overflow_hashes + overflow_offsets[d]);
                    if (hx == oref[0] && hy == oref[1] && hz == oref[2] && hw == oref[3]) {
                        *out_idx = hash_data_count + (uint)d;
                        return 1;
                    }
                }
                break;
            }
        }
    }
    return 0;
}

/* ---- Byte-swap utilities ---- */

uint bswap32(uint x) {
    return ((x >> 24) & 0xff) | ((x >> 8) & 0xff00) |
           ((x << 8) & 0xff0000) | ((x << 24) & 0xff000000u);
}

ulong bswap64(ulong x) {
    return ((x >> 56) & 0xffUL) | ((x >> 40) & 0xff00UL) |
           ((x >> 24) & 0xff0000UL) | ((x >> 8) & 0xff000000UL) |
           ((x << 8) & 0xff00000000UL) | ((x << 24) & 0xff0000000000UL) |
           ((x << 40) & 0xff000000000000UL) | ((x << 56) & 0xff00000000000000UL);
}

ulong rotr64(ulong x, uint n) { return (x >> n) | (x << (64 - n)); }

/* ---- Hex encoding helpers ---- */

uint hex_byte_lc(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    uint hc = hi + ((hi < 10) ? '0' : ('a' - 10));
    uint lc = lo + ((lo < 10) ? '0' : ('a' - 10));
    return hc | (lc << 8);
}

uint hex_byte_uc(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    uint hc = hi + ((hi < 10) ? '0' : ('A' - 10));
    uint lc = lo + ((lo < 10) ? '0' : ('A' - 10));
    return hc | (lc << 8);
}

ulong hex_byte_be64(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    return ((ulong)(hi + ((hi < 10) ? '0' : ('a' - 10))) << 8)
         |  (ulong)(lo + ((lo < 10) ? '0' : ('a' - 10)));
}

void md5_to_hex_lc(uint hx, uint hy, uint hz, uint hw, uint *M) {
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xff, b1 = (v[i]>>8) & 0xff;
        uint b2 = (v[i]>>16) & 0xff, b3 = (v[i]>>24) & 0xff;
        M[i*2]   = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[i*2+1] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
    }
}

void md5_to_hex_uc(uint hx, uint hy, uint hz, uint hw, uint *M) {
    uint v[4]; v[0]=hx; v[1]=hy; v[2]=hz; v[3]=hw;
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xff, b1 = (v[i]>>8) & 0xff;
        uint b2 = (v[i]>>16) & 0xff, b3 = (v[i]>>24) & 0xff;
        M[i*2]   = hex_byte_uc(b0) | (hex_byte_uc(b1) << 16);
        M[i*2+1] = hex_byte_uc(b2) | (hex_byte_uc(b3) << 16);
    }
}

/* ---- Byte manipulation helpers ---- */

/* Copy bytes from global memory into little-endian uint32 M[] (for MD5) */
void M_copy_bytes(uint *M, int off, __global const uchar *src, int len) {
    for (int i = 0; i < len; i++) {
        int pos = off + i;
        int word = pos >> 2;
        int shift = (pos & 3) << 3;
        M[word] |= ((uint)src[i]) << shift;
    }
}

/* Set a single byte in little-endian uint32 M[] */
void M_set_byte(uint *M, int pos, uint val) {
    M[pos >> 2] |= val << ((pos & 3) << 3);
}

/* Copy bytes from global memory into big-endian uint32 M[] (for SHA1/SHA256) */
void S_copy_bytes(uint *M, int byte_off, __global const uchar *src, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        int wi = (byte_off + i) / 4;
        int bi = 3 - ((byte_off + i) % 4);
        M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)src[i] << (bi * 8));
    }
}

void S_set_byte(uint *M, int byte_off, uchar val) {
    int wi = byte_off / 4;
    int bi = 3 - (byte_off % 4);
    M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)val << (bi * 8));
}

/* Copy bytes into big-endian ulong M[] (for SHA512) */
void S512_copy_bytes(ulong *M, int byte_off, __global const uchar *src, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        int wi = (byte_off + i) / 8; int bi = 7 - ((byte_off + i) % 8);
        M[wi] = (M[wi] & ~(0xffUL << (bi * 8))) | ((ulong)src[i] << (bi * 8));
    }
}

void S512_set_byte(ulong *M, int byte_off, uchar val) {
    int wi = byte_off / 8; int bi = 7 - (byte_off % 8);
    M[wi] = (M[wi] & ~(0xffUL << (bi * 8))) | ((ulong)val << (bi * 8));
}

/* ---- SHA1 block function ---- */

void sha1_block(uint *state, uint *M) {
    uint W[80];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 80; i++)
        W[i] = rotate(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], (uint)1);

    uint a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
    uint t;
    for (int i = 0; i < 20; i++) {
        t = rotate(a, (uint)5) + ((b & c) | (~b & d)) + e + 0x5A827999u + W[i];
        e = d; d = c; c = rotate(b, (uint)30); b = a; a = t;
    }
    for (int i = 20; i < 40; i++) {
        t = rotate(a, (uint)5) + (b ^ c ^ d) + e + 0x6ED9EBA1u + W[i];
        e = d; d = c; c = rotate(b, (uint)30); b = a; a = t;
    }
    for (int i = 40; i < 60; i++) {
        t = rotate(a, (uint)5) + ((b & c) | (b & d) | (c & d)) + e + 0x8F1BBCDCu + W[i];
        e = d; d = c; c = rotate(b, (uint)30); b = a; a = t;
    }
    for (int i = 60; i < 80; i++) {
        t = rotate(a, (uint)5) + (b ^ c ^ d) + e + 0xCA62C1D6u + W[i];
        e = d; d = c; c = rotate(b, (uint)30); b = a; a = t;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e;
}

/* ---- SHA256 block function ---- */

__constant uint SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define S256_ROTR(x,n) rotate((x),(uint)(32-(n)))
#define S256_CH(x,y,z)  ((x & y) ^ (~x & z))
#define S256_MAJ(x,y,z) ((x & y) ^ (x & z) ^ (y & z))
#define S256_EP0(x)  (S256_ROTR(x,2)  ^ S256_ROTR(x,13) ^ S256_ROTR(x,22))
#define S256_EP1(x)  (S256_ROTR(x,6)  ^ S256_ROTR(x,11) ^ S256_ROTR(x,25))
#define S256_SIG0(x) (S256_ROTR(x,7)  ^ S256_ROTR(x,18) ^ (x >> 3))
#define S256_SIG1(x) (S256_ROTR(x,17) ^ S256_ROTR(x,19) ^ (x >> 10))

void sha256_block(uint *state, uint *M) {
    uint W[64];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 64; i++)
        W[i] = S256_SIG1(W[i-2]) + W[i-7] + S256_SIG0(W[i-15]) + W[i-16];

    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; i++) {
        uint t1 = h + S256_EP1(e) + S256_CH(e,f,g) + SHA256_K[i] + W[i];
        uint t2 = S256_EP0(a) + S256_MAJ(a,b,c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* ---- SHA512 block function ---- */

// SHA-512 scalar helper macros for flat-unrolled body.
// Avoids W[80] array spill to local memory on Pascal NVIDIA GPUs.
// MDX_SHA512_S0_S/S1_S: big-sigma (compression). S2_S/S3_S: small-sigma (schedule).
// MDX_SHA512_F0o/F1o: Ch/Maj with optional bitselect gate (USE_BITSELECT).
// MDX_SHA512_STEP_S: one round; caller rotates arg order, not register names.
// MDX_SHA512_EXPAND_S: w[i] for i>=16 from four prior words.
#define MDX_SHA512_S0_S(x) (rotr64((x), 28) ^ rotr64((x), 34) ^ rotr64((x), 39))
#define MDX_SHA512_S1_S(x) (rotr64((x), 14) ^ rotr64((x), 18) ^ rotr64((x), 41))
#define MDX_SHA512_S2_S(x) (rotr64((x),  1) ^ rotr64((x),  8) ^ ((x) >> 7))
#define MDX_SHA512_S3_S(x) (rotr64((x), 19) ^ rotr64((x), 61) ^ ((x) >> 6))

#ifdef USE_BITSELECT
#define MDX_SHA512_F0o(x,y,z) (bitselect((z),(y),(x)))
#define MDX_SHA512_F1o(x,y,z) (bitselect((x),(y),((x)^(z))))
#else
#define MDX_SHA512_F0o(x,y,z) ((z) ^ ((x) & ((y) ^ (z))))
#define MDX_SHA512_F1o(x,y,z) (((x) & (y)) | ((z) & ((x) ^ (y))))
#endif

// One SHA-512 compression round. Uses hashcat arg-rotation convention:
// caller passes (a,b,c,d,e,f,g,h) cycling each step; h is the accumulator.
#define MDX_SHA512_STEP_S(a,b,c,d,e,f,g,h,x,K)  \
{                                                  \
    (h) += (K);                                    \
    (h) += (x);                                    \
    (h) += MDX_SHA512_S1_S(e);                     \
    (h) += MDX_SHA512_F0o((e),(f),(g));             \
    (d) += (h);                                    \
    (h) += MDX_SHA512_S0_S(a);                     \
    (h) += MDX_SHA512_F1o((a),(b),(c));             \
}

// Message schedule expansion: w[i] = sigma1(w[i-2]) + w[i-7] + sigma0(w[i-15]) + w[i-16]
// Args: (w[i-2], w[i-7], w[i-15], w[i-16])
#define MDX_SHA512_EXPAND_S(x,y,z,w) \
    (MDX_SHA512_S3_S(x) + (y) + MDX_SHA512_S2_S(z) + (w))


__constant ulong K512[80] = {
    0x428a2f98d728ae22UL, 0x7137449123ef65cdUL, 0xb5c0fbcfec4d3b2fUL, 0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL, 0x59f111f1b605d019UL, 0x923f82a4af194f9bUL, 0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL, 0x12835b0145706fbeUL, 0x243185be4ee4b28cUL, 0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL, 0x80deb1fe3b1696b1UL, 0x9bdc06a725c71235UL, 0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL, 0xefbe4786384f25e3UL, 0x0fc19dc68b8cd5b5UL, 0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL, 0x4a7484aa6ea6e483UL, 0x5cb0a9dcbd41fbd4UL, 0x76f988da831153b5UL,
    0x983e5152ee66dfabUL, 0xa831c66d2db43210UL, 0xb00327c898fb213fUL, 0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL, 0xd5a79147930aa725UL, 0x06ca6351e003826fUL, 0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL, 0x2e1b21385c26c926UL, 0x4d2c6dfc5ac42aedUL, 0x53380d139d95b3dfUL,
    0x650a73548baf63deUL, 0x766a0abb3c77b2a8UL, 0x81c2c92e47edaee6UL, 0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL, 0xa81a664bbc423001UL, 0xc24b8b70d0f89791UL, 0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL, 0xd69906245565a910UL, 0xf40e35855771202aUL, 0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL, 0x1e376c085141ab53UL, 0x2748774cdf8eeb99UL, 0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL, 0x4ed8aa4ae3418acbUL, 0x5b9cca4f7763e373UL, 0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL, 0x78a5636f43172f60UL, 0x84c87814a1f0ab72UL, 0x8cc702081a6439ecUL,
    0x90befffa23631e28UL, 0xa4506cebde82bde9UL, 0xbef9a3f7b2c67915UL, 0xc67178f2e372532bUL,
    0xca273eceea26619cUL, 0xd186b8c721c0c207UL, 0xeada7dd6cde0eb1eUL, 0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL, 0x0a637dc5a2c898a6UL, 0x113f9804bef90daeUL, 0x1b710b35131c471bUL,
    0x28db77f523047d84UL, 0x32caab7b40c72493UL, 0x3c9ebe0a15c9bebcUL, 0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL, 0x597f299cfc657e2aUL, 0x5fcb6fab3ad6faecUL, 0x6c44198c4a475817UL
};

void sha512_block(ulong *state, ulong *M) {
    // Load message words as 16 scalar ulong registers (avoids W[80] local-mem spill).
    // Flat-unrolled 80-step body; register rotation via arg-order cycling.
    ulong w0_t = M[0];
    ulong w1_t = M[1];
    ulong w2_t = M[2];
    ulong w3_t = M[3];
    ulong w4_t = M[4];
    ulong w5_t = M[5];
    ulong w6_t = M[6];
    ulong w7_t = M[7];
    ulong w8_t = M[8];
    ulong w9_t = M[9];
    ulong wa_t = M[10];
    ulong wb_t = M[11];
    ulong wc_t = M[12];
    ulong wd_t = M[13];
    ulong we_t = M[14];
    ulong wf_t = M[15];
    ulong a = state[0], b = state[1], c = state[2], d = state[3];
    ulong e = state[4], f = state[5], g = state[6], h = state[7];
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, K512[0]);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, K512[1]);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, K512[2]);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, K512[3]);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, K512[4]);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, K512[5]);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, K512[6]);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, K512[7]);
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, K512[8]);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, K512[9]);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, K512[10]);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, K512[11]);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, K512[12]);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, K512[13]);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, K512[14]);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, K512[15]);
    w0_t = MDX_SHA512_EXPAND_S(we_t, w9_t, w1_t, w0_t);
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, K512[16]);
    w1_t = MDX_SHA512_EXPAND_S(wf_t, wa_t, w2_t, w1_t);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, K512[17]);
    w2_t = MDX_SHA512_EXPAND_S(w0_t, wb_t, w3_t, w2_t);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, K512[18]);
    w3_t = MDX_SHA512_EXPAND_S(w1_t, wc_t, w4_t, w3_t);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, K512[19]);
    w4_t = MDX_SHA512_EXPAND_S(w2_t, wd_t, w5_t, w4_t);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, K512[20]);
    w5_t = MDX_SHA512_EXPAND_S(w3_t, we_t, w6_t, w5_t);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, K512[21]);
    w6_t = MDX_SHA512_EXPAND_S(w4_t, wf_t, w7_t, w6_t);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, K512[22]);
    w7_t = MDX_SHA512_EXPAND_S(w5_t, w0_t, w8_t, w7_t);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, K512[23]);
    w8_t = MDX_SHA512_EXPAND_S(w6_t, w1_t, w9_t, w8_t);
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, K512[24]);
    w9_t = MDX_SHA512_EXPAND_S(w7_t, w2_t, wa_t, w9_t);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, K512[25]);
    wa_t = MDX_SHA512_EXPAND_S(w8_t, w3_t, wb_t, wa_t);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, K512[26]);
    wb_t = MDX_SHA512_EXPAND_S(w9_t, w4_t, wc_t, wb_t);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, K512[27]);
    wc_t = MDX_SHA512_EXPAND_S(wa_t, w5_t, wd_t, wc_t);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, K512[28]);
    wd_t = MDX_SHA512_EXPAND_S(wb_t, w6_t, we_t, wd_t);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, K512[29]);
    we_t = MDX_SHA512_EXPAND_S(wc_t, w7_t, wf_t, we_t);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, K512[30]);
    wf_t = MDX_SHA512_EXPAND_S(wd_t, w8_t, w0_t, wf_t);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, K512[31]);
    w0_t = MDX_SHA512_EXPAND_S(we_t, w9_t, w1_t, w0_t);
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, K512[32]);
    w1_t = MDX_SHA512_EXPAND_S(wf_t, wa_t, w2_t, w1_t);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, K512[33]);
    w2_t = MDX_SHA512_EXPAND_S(w0_t, wb_t, w3_t, w2_t);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, K512[34]);
    w3_t = MDX_SHA512_EXPAND_S(w1_t, wc_t, w4_t, w3_t);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, K512[35]);
    w4_t = MDX_SHA512_EXPAND_S(w2_t, wd_t, w5_t, w4_t);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, K512[36]);
    w5_t = MDX_SHA512_EXPAND_S(w3_t, we_t, w6_t, w5_t);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, K512[37]);
    w6_t = MDX_SHA512_EXPAND_S(w4_t, wf_t, w7_t, w6_t);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, K512[38]);
    w7_t = MDX_SHA512_EXPAND_S(w5_t, w0_t, w8_t, w7_t);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, K512[39]);
    w8_t = MDX_SHA512_EXPAND_S(w6_t, w1_t, w9_t, w8_t);
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, K512[40]);
    w9_t = MDX_SHA512_EXPAND_S(w7_t, w2_t, wa_t, w9_t);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, K512[41]);
    wa_t = MDX_SHA512_EXPAND_S(w8_t, w3_t, wb_t, wa_t);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, K512[42]);
    wb_t = MDX_SHA512_EXPAND_S(w9_t, w4_t, wc_t, wb_t);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, K512[43]);
    wc_t = MDX_SHA512_EXPAND_S(wa_t, w5_t, wd_t, wc_t);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, K512[44]);
    wd_t = MDX_SHA512_EXPAND_S(wb_t, w6_t, we_t, wd_t);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, K512[45]);
    we_t = MDX_SHA512_EXPAND_S(wc_t, w7_t, wf_t, we_t);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, K512[46]);
    wf_t = MDX_SHA512_EXPAND_S(wd_t, w8_t, w0_t, wf_t);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, K512[47]);
    w0_t = MDX_SHA512_EXPAND_S(we_t, w9_t, w1_t, w0_t);
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, K512[48]);
    w1_t = MDX_SHA512_EXPAND_S(wf_t, wa_t, w2_t, w1_t);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, K512[49]);
    w2_t = MDX_SHA512_EXPAND_S(w0_t, wb_t, w3_t, w2_t);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, K512[50]);
    w3_t = MDX_SHA512_EXPAND_S(w1_t, wc_t, w4_t, w3_t);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, K512[51]);
    w4_t = MDX_SHA512_EXPAND_S(w2_t, wd_t, w5_t, w4_t);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, K512[52]);
    w5_t = MDX_SHA512_EXPAND_S(w3_t, we_t, w6_t, w5_t);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, K512[53]);
    w6_t = MDX_SHA512_EXPAND_S(w4_t, wf_t, w7_t, w6_t);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, K512[54]);
    w7_t = MDX_SHA512_EXPAND_S(w5_t, w0_t, w8_t, w7_t);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, K512[55]);
    w8_t = MDX_SHA512_EXPAND_S(w6_t, w1_t, w9_t, w8_t);
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, K512[56]);
    w9_t = MDX_SHA512_EXPAND_S(w7_t, w2_t, wa_t, w9_t);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, K512[57]);
    wa_t = MDX_SHA512_EXPAND_S(w8_t, w3_t, wb_t, wa_t);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, K512[58]);
    wb_t = MDX_SHA512_EXPAND_S(w9_t, w4_t, wc_t, wb_t);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, K512[59]);
    wc_t = MDX_SHA512_EXPAND_S(wa_t, w5_t, wd_t, wc_t);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, K512[60]);
    wd_t = MDX_SHA512_EXPAND_S(wb_t, w6_t, we_t, wd_t);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, K512[61]);
    we_t = MDX_SHA512_EXPAND_S(wc_t, w7_t, wf_t, we_t);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, K512[62]);
    wf_t = MDX_SHA512_EXPAND_S(wd_t, w8_t, w0_t, wf_t);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, K512[63]);
    w0_t = MDX_SHA512_EXPAND_S(we_t, w9_t, w1_t, w0_t);
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w0_t, K512[64]);
    w1_t = MDX_SHA512_EXPAND_S(wf_t, wa_t, w2_t, w1_t);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w1_t, K512[65]);
    w2_t = MDX_SHA512_EXPAND_S(w0_t, wb_t, w3_t, w2_t);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, w2_t, K512[66]);
    w3_t = MDX_SHA512_EXPAND_S(w1_t, wc_t, w4_t, w3_t);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, w3_t, K512[67]);
    w4_t = MDX_SHA512_EXPAND_S(w2_t, wd_t, w5_t, w4_t);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, w4_t, K512[68]);
    w5_t = MDX_SHA512_EXPAND_S(w3_t, we_t, w6_t, w5_t);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, w5_t, K512[69]);
    w6_t = MDX_SHA512_EXPAND_S(w4_t, wf_t, w7_t, w6_t);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, w6_t, K512[70]);
    w7_t = MDX_SHA512_EXPAND_S(w5_t, w0_t, w8_t, w7_t);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, w7_t, K512[71]);
    w8_t = MDX_SHA512_EXPAND_S(w6_t, w1_t, w9_t, w8_t);
    MDX_SHA512_STEP_S(a, b, c, d, e, f, g, h, w8_t, K512[72]);
    w9_t = MDX_SHA512_EXPAND_S(w7_t, w2_t, wa_t, w9_t);
    MDX_SHA512_STEP_S(h, a, b, c, d, e, f, g, w9_t, K512[73]);
    wa_t = MDX_SHA512_EXPAND_S(w8_t, w3_t, wb_t, wa_t);
    MDX_SHA512_STEP_S(g, h, a, b, c, d, e, f, wa_t, K512[74]);
    wb_t = MDX_SHA512_EXPAND_S(w9_t, w4_t, wc_t, wb_t);
    MDX_SHA512_STEP_S(f, g, h, a, b, c, d, e, wb_t, K512[75]);
    wc_t = MDX_SHA512_EXPAND_S(wa_t, w5_t, wd_t, wc_t);
    MDX_SHA512_STEP_S(e, f, g, h, a, b, c, d, wc_t, K512[76]);
    wd_t = MDX_SHA512_EXPAND_S(wb_t, w6_t, we_t, wd_t);
    MDX_SHA512_STEP_S(d, e, f, g, h, a, b, c, wd_t, K512[77]);
    we_t = MDX_SHA512_EXPAND_S(wc_t, w7_t, wf_t, we_t);
    MDX_SHA512_STEP_S(c, d, e, f, g, h, a, b, we_t, K512[78]);
    wf_t = MDX_SHA512_EXPAND_S(wd_t, w8_t, w0_t, wf_t);
    MDX_SHA512_STEP_S(b, c, d, e, f, g, h, a, wf_t, K512[79]);
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

void sha512_to_hex_lc(ulong *state, ulong *M) {
    for (int i = 0; i < 8; i++) {
        ulong s = state[i];
        uint b0 = (s >> 56) & 0xff, b1 = (s >> 48) & 0xff;
        uint b2 = (s >> 40) & 0xff, b3 = (s >> 32) & 0xff;
        uint b4 = (s >> 24) & 0xff, b5 = (s >> 16) & 0xff;
        uint b6 = (s >> 8)  & 0xff, b7 = s & 0xff;
        M[i*2]   = (hex_byte_be64(b0) << 48) | (hex_byte_be64(b1) << 32)
                  | (hex_byte_be64(b2) << 16) | hex_byte_be64(b3);
        M[i*2+1] = (hex_byte_be64(b4) << 48) | (hex_byte_be64(b5) << 32)
                  | (hex_byte_be64(b6) << 16) | hex_byte_be64(b7);
    }
}


// ---- Whirlpool (ISO/IEC 10118-3) block function ----
//
// Phase 5b Tier 2 sub-phase 5b.2a.1 (2026-05-27): lift wrl_block from
// RHash-master/librhash/whirlpool.c rhash_whirlpool_process_block lines
// 60-128 and RHash-master/librhash/whirlpool_sbox.c rhash_whirlpool_sbox.
//
// Caller convention (mirrors sha512_block): M[0..7] must already be
// BE-packed (8 message bytes per ulong, MSB = first byte). State is 8
// ulongs natural endian; the IV is all zero. Output state is BE-laid
// when reinterpreted as bytes, matching Whirlpool's spec digest order.
//
// Donor source librhash applies be2me_64() inside the block function;
// we elide that swap because the GPU emit helper already packs M in
// BE-ulong form. Net effect is byte-identical to librhash and to
// OpenSSL WHIRLPOOL() per the R12 NESSIE pre-flight test (16/16 PASS
// against published vectors on iMac 2026-05-27).
//
// Constant memory budget: 8 * 256 * 8 = 16 KB (WRL_SBOX) + 80 B
// (WRL_RC) = 16.4 KB. Pascal GTX 1080 and Apple Silicon M2 Max both
// expose >= 64 KB CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE; comfortable
// headroom for Tier 3+4 additions. Mali / AMD legacy device budgets
// not validated this sub-phase (production targets = fpga + dev3).
//
// noinline per feedback_md5_block_noinline_pascal.md (Pascal register
// budget preservation). R7 no nested block comments (line comments
// only inside body; donor /* */ stripped).

__constant ulong WRL_RC[10] = {
    0x1823c6e887b8014fUL, 0x36a6d2f5796f9152UL, 0x60bc9b8ea30c7b35UL,
    0x1de0d7c22e4bfe57UL, 0x157737e59ff04adaUL, 0x58c9290ab1a06b85UL,
    0xbd5d10f4cb3e0567UL, 0xe427418ba77d95d8UL, 0xfbee7c66dd17479eUL,
    0xca2dbf07ad5a8333UL
};

__constant ulong WRL_SBOX[8][256] = {
    {
        
        0x18186018c07830d8UL, 0x23238c2305af4626UL, 0xc6c63fc67ef991b8UL, 0xe8e887e8136fcdfbUL,
        0x878726874ca113cbUL, 0xb8b8dab8a9626d11UL, 0x0101040108050209UL, 0x4f4f214f426e9e0dUL,
        0x3636d836adee6c9bUL, 0xa6a6a2a6590451ffUL, 0xd2d26fd2debdb90cUL, 0xf5f5f3f5fb06f70eUL,
        0x7979f979ef80f296UL, 0x6f6fa16f5fcede30UL, 0x91917e91fcef3f6dUL, 0x52525552aa07a4f8UL,
        0x60609d6027fdc047UL, 0xbcbccabc89766535UL, 0x9b9b569baccd2b37UL, 0x8e8e028e048c018aUL,
        0xa3a3b6a371155bd2UL, 0x0c0c300c603c186cUL, 0x7b7bf17bff8af684UL, 0x3535d435b5e16a80UL,
        0x1d1d741de8693af5UL, 0xe0e0a7e05347ddb3UL, 0xd7d77bd7f6acb321UL, 0xc2c22fc25eed999cUL,
        0x2e2eb82e6d965c43UL, 0x4b4b314b627a9629UL, 0xfefedffea321e15dUL, 0x575741578216aed5UL,
        0x15155415a8412abdUL, 0x7777c1779fb6eee8UL, 0x3737dc37a5eb6e92UL, 0xe5e5b3e57b56d79eUL,
        0x9f9f469f8cd92313UL, 0xf0f0e7f0d317fd23UL, 0x4a4a354a6a7f9420UL, 0xdada4fda9e95a944UL,
        0x58587d58fa25b0a2UL, 0xc9c903c906ca8fcfUL, 0x2929a429558d527cUL, 0x0a0a280a5022145aUL,
        0xb1b1feb1e14f7f50UL, 0xa0a0baa0691a5dc9UL, 0x6b6bb16b7fdad614UL, 0x85852e855cab17d9UL,
        0xbdbdcebd8173673cUL, 0x5d5d695dd234ba8fUL, 0x1010401080502090UL, 0xf4f4f7f4f303f507UL,
        0xcbcb0bcb16c08bddUL, 0x3e3ef83eedc67cd3UL, 0x0505140528110a2dUL, 0x676781671fe6ce78UL,
        0xe4e4b7e47353d597UL, 0x27279c2725bb4e02UL, 0x4141194132588273UL, 0x8b8b168b2c9d0ba7UL,
        0xa7a7a6a7510153f6UL, 0x7d7de97dcf94fab2UL, 0x95956e95dcfb3749UL, 0xd8d847d88e9fad56UL,
        0xfbfbcbfb8b30eb70UL, 0xeeee9fee2371c1cdUL, 0x7c7ced7cc791f8bbUL, 0x6666856617e3cc71UL,
        0xdddd53dda68ea77bUL, 0x17175c17b84b2eafUL, 0x4747014702468e45UL, 0x9e9e429e84dc211aUL,
        0xcaca0fca1ec589d4UL, 0x2d2db42d75995a58UL, 0xbfbfc6bf9179632eUL, 0x07071c07381b0e3fUL,
        0xadad8ead012347acUL, 0x5a5a755aea2fb4b0UL, 0x838336836cb51befUL, 0x3333cc3385ff66b6UL,
        0x636391633ff2c65cUL, 0x02020802100a0412UL, 0xaaaa92aa39384993UL, 0x7171d971afa8e2deUL,
        0xc8c807c80ecf8dc6UL, 0x19196419c87d32d1UL, 0x494939497270923bUL, 0xd9d943d9869aaf5fUL,
        0xf2f2eff2c31df931UL, 0xe3e3abe34b48dba8UL, 0x5b5b715be22ab6b9UL, 0x88881a8834920dbcUL,
        0x9a9a529aa4c8293eUL, 0x262698262dbe4c0bUL, 0x3232c8328dfa64bfUL, 0xb0b0fab0e94a7d59UL,
        0xe9e983e91b6acff2UL, 0x0f0f3c0f78331e77UL, 0xd5d573d5e6a6b733UL, 0x80803a8074ba1df4UL,
        0xbebec2be997c6127UL, 0xcdcd13cd26de87ebUL, 0x3434d034bde46889UL, 0x48483d487a759032UL,
        0xffffdbffab24e354UL, 0x7a7af57af78ff48dUL, 0x90907a90f4ea3d64UL, 0x5f5f615fc23ebe9dUL,
        0x202080201da0403dUL, 0x6868bd6867d5d00fUL, 0x1a1a681ad07234caUL, 0xaeae82ae192c41b7UL,
        0xb4b4eab4c95e757dUL, 0x54544d549a19a8ceUL, 0x93937693ece53b7fUL, 0x222288220daa442fUL,
        0x64648d6407e9c863UL, 0xf1f1e3f1db12ff2aUL, 0x7373d173bfa2e6ccUL, 0x12124812905a2482UL,
        0x40401d403a5d807aUL, 0x0808200840281048UL, 0xc3c32bc356e89b95UL, 0xecec97ec337bc5dfUL,
        0xdbdb4bdb9690ab4dUL, 0xa1a1bea1611f5fc0UL, 0x8d8d0e8d1c830791UL, 0x3d3df43df5c97ac8UL,
        0x97976697ccf1335bUL, 0x0000000000000000UL, 0xcfcf1bcf36d483f9UL, 0x2b2bac2b4587566eUL,
        0x7676c57697b3ece1UL, 0x8282328264b019e6UL, 0xd6d67fd6fea9b128UL, 0x1b1b6c1bd87736c3UL,
        0xb5b5eeb5c15b7774UL, 0xafaf86af112943beUL, 0x6a6ab56a77dfd41dUL, 0x50505d50ba0da0eaUL,
        0x45450945124c8a57UL, 0xf3f3ebf3cb18fb38UL, 0x3030c0309df060adUL, 0xefef9bef2b74c3c4UL,
        0x3f3ffc3fe5c37edaUL, 0x55554955921caac7UL, 0xa2a2b2a2791059dbUL, 0xeaea8fea0365c9e9UL,
        0x656589650fecca6aUL, 0xbabad2bab9686903UL, 0x2f2fbc2f65935e4aUL, 0xc0c027c04ee79d8eUL,
        0xdede5fdebe81a160UL, 0x1c1c701ce06c38fcUL, 0xfdfdd3fdbb2ee746UL, 0x4d4d294d52649a1fUL,
        0x92927292e4e03976UL, 0x7575c9758fbceafaUL, 0x06061806301e0c36UL, 0x8a8a128a249809aeUL,
        0xb2b2f2b2f940794bUL, 0xe6e6bfe66359d185UL, 0x0e0e380e70361c7eUL, 0x1f1f7c1ff8633ee7UL,
        0x6262956237f7c455UL, 0xd4d477d4eea3b53aUL, 0xa8a89aa829324d81UL, 0x96966296c4f43152UL,
        0xf9f9c3f99b3aef62UL, 0xc5c533c566f697a3UL, 0x2525942535b14a10UL, 0x59597959f220b2abUL,
        0x84842a8454ae15d0UL, 0x7272d572b7a7e4c5UL, 0x3939e439d5dd72ecUL, 0x4c4c2d4c5a619816UL,
        0x5e5e655eca3bbc94UL, 0x7878fd78e785f09fUL, 0x3838e038ddd870e5UL, 0x8c8c0a8c14860598UL,
        0xd1d163d1c6b2bf17UL, 0xa5a5aea5410b57e4UL, 0xe2e2afe2434dd9a1UL, 0x616199612ff8c24eUL,
        0xb3b3f6b3f1457b42UL, 0x2121842115a54234UL, 0x9c9c4a9c94d62508UL, 0x1e1e781ef0663ceeUL,
        0x4343114322528661UL, 0xc7c73bc776fc93b1UL, 0xfcfcd7fcb32be54fUL, 0x0404100420140824UL,
        0x51515951b208a2e3UL, 0x99995e99bcc72f25UL, 0x6d6da96d4fc4da22UL, 0x0d0d340d68391a65UL,
        0xfafacffa8335e979UL, 0xdfdf5bdfb684a369UL, 0x7e7ee57ed79bfca9UL, 0x242490243db44819UL,
        0x3b3bec3bc5d776feUL, 0xabab96ab313d4b9aUL, 0xcece1fce3ed181f0UL, 0x1111441188552299UL,
        0x8f8f068f0c890383UL, 0x4e4e254e4a6b9c04UL, 0xb7b7e6b7d1517366UL, 0xebeb8beb0b60cbe0UL,
        0x3c3cf03cfdcc78c1UL, 0x81813e817cbf1ffdUL, 0x94946a94d4fe3540UL, 0xf7f7fbf7eb0cf31cUL,
        0xb9b9deb9a1676f18UL, 0x13134c13985f268bUL, 0x2c2cb02c7d9c5851UL, 0xd3d36bd3d6b8bb05UL,
        0xe7e7bbe76b5cd38cUL, 0x6e6ea56e57cbdc39UL, 0xc4c437c46ef395aaUL, 0x03030c03180f061bUL,
        0x565645568a13acdcUL, 0x44440d441a49885eUL, 0x7f7fe17fdf9efea0UL, 0xa9a99ea921374f88UL,
        0x2a2aa82a4d825467UL, 0xbbbbd6bbb16d6b0aUL, 0xc1c123c146e29f87UL, 0x53535153a202a6f1UL,
        0xdcdc57dcae8ba572UL, 0x0b0b2c0b58271653UL, 0x9d9d4e9d9cd32701UL, 0x6c6cad6c47c1d82bUL,
        0x3131c43195f562a4UL, 0x7474cd7487b9e8f3UL, 0xf6f6fff6e309f115UL, 0x464605460a438c4cUL,
        0xacac8aac092645a5UL, 0x89891e893c970fb5UL, 0x14145014a04428b4UL, 0xe1e1a3e15b42dfbaUL,
        0x16165816b04e2ca6UL, 0x3a3ae83acdd274f7UL, 0x6969b9696fd0d206UL, 0x09092409482d1241UL,
        0x7070dd70a7ade0d7UL, 0xb6b6e2b6d954716fUL, 0xd0d067d0ceb7bd1eUL, 0xeded93ed3b7ec7d6UL,
        0xcccc17cc2edb85e2UL, 0x424215422a578468UL, 0x98985a98b4c22d2cUL, 0xa4a4aaa4490e55edUL,
        0x2828a0285d885075UL, 0x5c5c6d5cda31b886UL, 0xf8f8c7f8933fed6bUL, 0x8686228644a411c2UL,
        }, {
        
        0xd818186018c07830UL, 0x2623238c2305af46UL, 0xb8c6c63fc67ef991UL, 0xfbe8e887e8136fcdUL,
        0xcb878726874ca113UL, 0x11b8b8dab8a9626dUL, 0x0901010401080502UL, 0x0d4f4f214f426e9eUL,
        0x9b3636d836adee6cUL, 0xffa6a6a2a6590451UL, 0x0cd2d26fd2debdb9UL, 0x0ef5f5f3f5fb06f7UL,
        0x967979f979ef80f2UL, 0x306f6fa16f5fcedeUL, 0x6d91917e91fcef3fUL, 0xf852525552aa07a4UL,
        0x4760609d6027fdc0UL, 0x35bcbccabc897665UL, 0x379b9b569baccd2bUL, 0x8a8e8e028e048c01UL,
        0xd2a3a3b6a371155bUL, 0x6c0c0c300c603c18UL, 0x847b7bf17bff8af6UL, 0x803535d435b5e16aUL,
        0xf51d1d741de8693aUL, 0xb3e0e0a7e05347ddUL, 0x21d7d77bd7f6acb3UL, 0x9cc2c22fc25eed99UL,
        0x432e2eb82e6d965cUL, 0x294b4b314b627a96UL, 0x5dfefedffea321e1UL, 0xd5575741578216aeUL,
        0xbd15155415a8412aUL, 0xe87777c1779fb6eeUL, 0x923737dc37a5eb6eUL, 0x9ee5e5b3e57b56d7UL,
        0x139f9f469f8cd923UL, 0x23f0f0e7f0d317fdUL, 0x204a4a354a6a7f94UL, 0x44dada4fda9e95a9UL,
        0xa258587d58fa25b0UL, 0xcfc9c903c906ca8fUL, 0x7c2929a429558d52UL, 0x5a0a0a280a502214UL,
        0x50b1b1feb1e14f7fUL, 0xc9a0a0baa0691a5dUL, 0x146b6bb16b7fdad6UL, 0xd985852e855cab17UL,
        0x3cbdbdcebd817367UL, 0x8f5d5d695dd234baUL, 0x9010104010805020UL, 0x07f4f4f7f4f303f5UL,
        0xddcbcb0bcb16c08bUL, 0xd33e3ef83eedc67cUL, 0x2d0505140528110aUL, 0x78676781671fe6ceUL,
        0x97e4e4b7e47353d5UL, 0x0227279c2725bb4eUL, 0x7341411941325882UL, 0xa78b8b168b2c9d0bUL,
        0xf6a7a7a6a7510153UL, 0xb27d7de97dcf94faUL, 0x4995956e95dcfb37UL, 0x56d8d847d88e9fadUL,
        0x70fbfbcbfb8b30ebUL, 0xcdeeee9fee2371c1UL, 0xbb7c7ced7cc791f8UL, 0x716666856617e3ccUL,
        0x7bdddd53dda68ea7UL, 0xaf17175c17b84b2eUL, 0x454747014702468eUL, 0x1a9e9e429e84dc21UL,
        0xd4caca0fca1ec589UL, 0x582d2db42d75995aUL, 0x2ebfbfc6bf917963UL, 0x3f07071c07381b0eUL,
        0xacadad8ead012347UL, 0xb05a5a755aea2fb4UL, 0xef838336836cb51bUL, 0xb63333cc3385ff66UL,
        0x5c636391633ff2c6UL, 0x1202020802100a04UL, 0x93aaaa92aa393849UL, 0xde7171d971afa8e2UL,
        0xc6c8c807c80ecf8dUL, 0xd119196419c87d32UL, 0x3b49493949727092UL, 0x5fd9d943d9869aafUL,
        0x31f2f2eff2c31df9UL, 0xa8e3e3abe34b48dbUL, 0xb95b5b715be22ab6UL, 0xbc88881a8834920dUL,
        0x3e9a9a529aa4c829UL, 0x0b262698262dbe4cUL, 0xbf3232c8328dfa64UL, 0x59b0b0fab0e94a7dUL,
        0xf2e9e983e91b6acfUL, 0x770f0f3c0f78331eUL, 0x33d5d573d5e6a6b7UL, 0xf480803a8074ba1dUL,
        0x27bebec2be997c61UL, 0xebcdcd13cd26de87UL, 0x893434d034bde468UL, 0x3248483d487a7590UL,
        0x54ffffdbffab24e3UL, 0x8d7a7af57af78ff4UL, 0x6490907a90f4ea3dUL, 0x9d5f5f615fc23ebeUL,
        0x3d202080201da040UL, 0x0f6868bd6867d5d0UL, 0xca1a1a681ad07234UL, 0xb7aeae82ae192c41UL,
        0x7db4b4eab4c95e75UL, 0xce54544d549a19a8UL, 0x7f93937693ece53bUL, 0x2f222288220daa44UL,
        0x6364648d6407e9c8UL, 0x2af1f1e3f1db12ffUL, 0xcc7373d173bfa2e6UL, 0x8212124812905a24UL,
        0x7a40401d403a5d80UL, 0x4808082008402810UL, 0x95c3c32bc356e89bUL, 0xdfecec97ec337bc5UL,
        0x4ddbdb4bdb9690abUL, 0xc0a1a1bea1611f5fUL, 0x918d8d0e8d1c8307UL, 0xc83d3df43df5c97aUL,
        0x5b97976697ccf133UL, 0x0000000000000000UL, 0xf9cfcf1bcf36d483UL, 0x6e2b2bac2b458756UL,
        0xe17676c57697b3ecUL, 0xe68282328264b019UL, 0x28d6d67fd6fea9b1UL, 0xc31b1b6c1bd87736UL,
        0x74b5b5eeb5c15b77UL, 0xbeafaf86af112943UL, 0x1d6a6ab56a77dfd4UL, 0xea50505d50ba0da0UL,
        0x5745450945124c8aUL, 0x38f3f3ebf3cb18fbUL, 0xad3030c0309df060UL, 0xc4efef9bef2b74c3UL,
        0xda3f3ffc3fe5c37eUL, 0xc755554955921caaUL, 0xdba2a2b2a2791059UL, 0xe9eaea8fea0365c9UL,
        0x6a656589650feccaUL, 0x03babad2bab96869UL, 0x4a2f2fbc2f65935eUL, 0x8ec0c027c04ee79dUL,
        0x60dede5fdebe81a1UL, 0xfc1c1c701ce06c38UL, 0x46fdfdd3fdbb2ee7UL, 0x1f4d4d294d52649aUL,
        0x7692927292e4e039UL, 0xfa7575c9758fbceaUL, 0x3606061806301e0cUL, 0xae8a8a128a249809UL,
        0x4bb2b2f2b2f94079UL, 0x85e6e6bfe66359d1UL, 0x7e0e0e380e70361cUL, 0xe71f1f7c1ff8633eUL,
        0x556262956237f7c4UL, 0x3ad4d477d4eea3b5UL, 0x81a8a89aa829324dUL, 0x5296966296c4f431UL,
        0x62f9f9c3f99b3aefUL, 0xa3c5c533c566f697UL, 0x102525942535b14aUL, 0xab59597959f220b2UL,
        0xd084842a8454ae15UL, 0xc57272d572b7a7e4UL, 0xec3939e439d5dd72UL, 0x164c4c2d4c5a6198UL,
        0x945e5e655eca3bbcUL, 0x9f7878fd78e785f0UL, 0xe53838e038ddd870UL, 0x988c8c0a8c148605UL,
        0x17d1d163d1c6b2bfUL, 0xe4a5a5aea5410b57UL, 0xa1e2e2afe2434dd9UL, 0x4e616199612ff8c2UL,
        0x42b3b3f6b3f1457bUL, 0x342121842115a542UL, 0x089c9c4a9c94d625UL, 0xee1e1e781ef0663cUL,
        0x6143431143225286UL, 0xb1c7c73bc776fc93UL, 0x4ffcfcd7fcb32be5UL, 0x2404041004201408UL,
        0xe351515951b208a2UL, 0x2599995e99bcc72fUL, 0x226d6da96d4fc4daUL, 0x650d0d340d68391aUL,
        0x79fafacffa8335e9UL, 0x69dfdf5bdfb684a3UL, 0xa97e7ee57ed79bfcUL, 0x19242490243db448UL,
        0xfe3b3bec3bc5d776UL, 0x9aabab96ab313d4bUL, 0xf0cece1fce3ed181UL, 0x9911114411885522UL,
        0x838f8f068f0c8903UL, 0x044e4e254e4a6b9cUL, 0x66b7b7e6b7d15173UL, 0xe0ebeb8beb0b60cbUL,
        0xc13c3cf03cfdcc78UL, 0xfd81813e817cbf1fUL, 0x4094946a94d4fe35UL, 0x1cf7f7fbf7eb0cf3UL,
        0x18b9b9deb9a1676fUL, 0x8b13134c13985f26UL, 0x512c2cb02c7d9c58UL, 0x05d3d36bd3d6b8bbUL,
        0x8ce7e7bbe76b5cd3UL, 0x396e6ea56e57cbdcUL, 0xaac4c437c46ef395UL, 0x1b03030c03180f06UL,
        0xdc565645568a13acUL, 0x5e44440d441a4988UL, 0xa07f7fe17fdf9efeUL, 0x88a9a99ea921374fUL,
        0x672a2aa82a4d8254UL, 0x0abbbbd6bbb16d6bUL, 0x87c1c123c146e29fUL, 0xf153535153a202a6UL,
        0x72dcdc57dcae8ba5UL, 0x530b0b2c0b582716UL, 0x019d9d4e9d9cd327UL, 0x2b6c6cad6c47c1d8UL,
        0xa43131c43195f562UL, 0xf37474cd7487b9e8UL, 0x15f6f6fff6e309f1UL, 0x4c464605460a438cUL,
        0xa5acac8aac092645UL, 0xb589891e893c970fUL, 0xb414145014a04428UL, 0xbae1e1a3e15b42dfUL,
        0xa616165816b04e2cUL, 0xf73a3ae83acdd274UL, 0x066969b9696fd0d2UL, 0x4109092409482d12UL,
        0xd77070dd70a7ade0UL, 0x6fb6b6e2b6d95471UL, 0x1ed0d067d0ceb7bdUL, 0xd6eded93ed3b7ec7UL,
        0xe2cccc17cc2edb85UL, 0x68424215422a5784UL, 0x2c98985a98b4c22dUL, 0xeda4a4aaa4490e55UL,
        0x752828a0285d8850UL, 0x865c5c6d5cda31b8UL, 0x6bf8f8c7f8933fedUL, 0xc28686228644a411UL,
        }, {
        
        0x30d818186018c078UL, 0x462623238c2305afUL, 0x91b8c6c63fc67ef9UL, 0xcdfbe8e887e8136fUL,
        0x13cb878726874ca1UL, 0x6d11b8b8dab8a962UL, 0x0209010104010805UL, 0x9e0d4f4f214f426eUL,
        0x6c9b3636d836adeeUL, 0x51ffa6a6a2a65904UL, 0xb90cd2d26fd2debdUL, 0xf70ef5f5f3f5fb06UL,
        0xf2967979f979ef80UL, 0xde306f6fa16f5fceUL, 0x3f6d91917e91fcefUL, 0xa4f852525552aa07UL,
        0xc04760609d6027fdUL, 0x6535bcbccabc8976UL, 0x2b379b9b569baccdUL, 0x018a8e8e028e048cUL,
        0x5bd2a3a3b6a37115UL, 0x186c0c0c300c603cUL, 0xf6847b7bf17bff8aUL, 0x6a803535d435b5e1UL,
        0x3af51d1d741de869UL, 0xddb3e0e0a7e05347UL, 0xb321d7d77bd7f6acUL, 0x999cc2c22fc25eedUL,
        0x5c432e2eb82e6d96UL, 0x96294b4b314b627aUL, 0xe15dfefedffea321UL, 0xaed5575741578216UL,
        0x2abd15155415a841UL, 0xeee87777c1779fb6UL, 0x6e923737dc37a5ebUL, 0xd79ee5e5b3e57b56UL,
        0x23139f9f469f8cd9UL, 0xfd23f0f0e7f0d317UL, 0x94204a4a354a6a7fUL, 0xa944dada4fda9e95UL,
        0xb0a258587d58fa25UL, 0x8fcfc9c903c906caUL, 0x527c2929a429558dUL, 0x145a0a0a280a5022UL,
        0x7f50b1b1feb1e14fUL, 0x5dc9a0a0baa0691aUL, 0xd6146b6bb16b7fdaUL, 0x17d985852e855cabUL,
        0x673cbdbdcebd8173UL, 0xba8f5d5d695dd234UL, 0x2090101040108050UL, 0xf507f4f4f7f4f303UL,
        0x8bddcbcb0bcb16c0UL, 0x7cd33e3ef83eedc6UL, 0x0a2d050514052811UL, 0xce78676781671fe6UL,
        0xd597e4e4b7e47353UL, 0x4e0227279c2725bbUL, 0x8273414119413258UL, 0x0ba78b8b168b2c9dUL,
        0x53f6a7a7a6a75101UL, 0xfab27d7de97dcf94UL, 0x374995956e95dcfbUL, 0xad56d8d847d88e9fUL,
        0xeb70fbfbcbfb8b30UL, 0xc1cdeeee9fee2371UL, 0xf8bb7c7ced7cc791UL, 0xcc716666856617e3UL,
        0xa77bdddd53dda68eUL, 0x2eaf17175c17b84bUL, 0x8e45474701470246UL, 0x211a9e9e429e84dcUL,
        0x89d4caca0fca1ec5UL, 0x5a582d2db42d7599UL, 0x632ebfbfc6bf9179UL, 0x0e3f07071c07381bUL,
        0x47acadad8ead0123UL, 0xb4b05a5a755aea2fUL, 0x1bef838336836cb5UL, 0x66b63333cc3385ffUL,
        0xc65c636391633ff2UL, 0x041202020802100aUL, 0x4993aaaa92aa3938UL, 0xe2de7171d971afa8UL,
        0x8dc6c8c807c80ecfUL, 0x32d119196419c87dUL, 0x923b494939497270UL, 0xaf5fd9d943d9869aUL,
        0xf931f2f2eff2c31dUL, 0xdba8e3e3abe34b48UL, 0xb6b95b5b715be22aUL, 0x0dbc88881a883492UL,
        0x293e9a9a529aa4c8UL, 0x4c0b262698262dbeUL, 0x64bf3232c8328dfaUL, 0x7d59b0b0fab0e94aUL,
        0xcff2e9e983e91b6aUL, 0x1e770f0f3c0f7833UL, 0xb733d5d573d5e6a6UL, 0x1df480803a8074baUL,
        0x6127bebec2be997cUL, 0x87ebcdcd13cd26deUL, 0x68893434d034bde4UL, 0x903248483d487a75UL,
        0xe354ffffdbffab24UL, 0xf48d7a7af57af78fUL, 0x3d6490907a90f4eaUL, 0xbe9d5f5f615fc23eUL,
        0x403d202080201da0UL, 0xd00f6868bd6867d5UL, 0x34ca1a1a681ad072UL, 0x41b7aeae82ae192cUL,
        0x757db4b4eab4c95eUL, 0xa8ce54544d549a19UL, 0x3b7f93937693ece5UL, 0x442f222288220daaUL,
        0xc86364648d6407e9UL, 0xff2af1f1e3f1db12UL, 0xe6cc7373d173bfa2UL, 0x248212124812905aUL,
        0x807a40401d403a5dUL, 0x1048080820084028UL, 0x9b95c3c32bc356e8UL, 0xc5dfecec97ec337bUL,
        0xab4ddbdb4bdb9690UL, 0x5fc0a1a1bea1611fUL, 0x07918d8d0e8d1c83UL, 0x7ac83d3df43df5c9UL,
        0x335b97976697ccf1UL, 0x0000000000000000UL, 0x83f9cfcf1bcf36d4UL, 0x566e2b2bac2b4587UL,
        0xece17676c57697b3UL, 0x19e68282328264b0UL, 0xb128d6d67fd6fea9UL, 0x36c31b1b6c1bd877UL,
        0x7774b5b5eeb5c15bUL, 0x43beafaf86af1129UL, 0xd41d6a6ab56a77dfUL, 0xa0ea50505d50ba0dUL,
        0x8a5745450945124cUL, 0xfb38f3f3ebf3cb18UL, 0x60ad3030c0309df0UL, 0xc3c4efef9bef2b74UL,
        0x7eda3f3ffc3fe5c3UL, 0xaac755554955921cUL, 0x59dba2a2b2a27910UL, 0xc9e9eaea8fea0365UL,
        0xca6a656589650fecUL, 0x6903babad2bab968UL, 0x5e4a2f2fbc2f6593UL, 0x9d8ec0c027c04ee7UL,
        0xa160dede5fdebe81UL, 0x38fc1c1c701ce06cUL, 0xe746fdfdd3fdbb2eUL, 0x9a1f4d4d294d5264UL,
        0x397692927292e4e0UL, 0xeafa7575c9758fbcUL, 0x0c3606061806301eUL, 0x09ae8a8a128a2498UL,
        0x794bb2b2f2b2f940UL, 0xd185e6e6bfe66359UL, 0x1c7e0e0e380e7036UL, 0x3ee71f1f7c1ff863UL,
        0xc4556262956237f7UL, 0xb53ad4d477d4eea3UL, 0x4d81a8a89aa82932UL, 0x315296966296c4f4UL,
        0xef62f9f9c3f99b3aUL, 0x97a3c5c533c566f6UL, 0x4a102525942535b1UL, 0xb2ab59597959f220UL,
        0x15d084842a8454aeUL, 0xe4c57272d572b7a7UL, 0x72ec3939e439d5ddUL, 0x98164c4c2d4c5a61UL,
        0xbc945e5e655eca3bUL, 0xf09f7878fd78e785UL, 0x70e53838e038ddd8UL, 0x05988c8c0a8c1486UL,
        0xbf17d1d163d1c6b2UL, 0x57e4a5a5aea5410bUL, 0xd9a1e2e2afe2434dUL, 0xc24e616199612ff8UL,
        0x7b42b3b3f6b3f145UL, 0x42342121842115a5UL, 0x25089c9c4a9c94d6UL, 0x3cee1e1e781ef066UL,
        0x8661434311432252UL, 0x93b1c7c73bc776fcUL, 0xe54ffcfcd7fcb32bUL, 0x0824040410042014UL,
        0xa2e351515951b208UL, 0x2f2599995e99bcc7UL, 0xda226d6da96d4fc4UL, 0x1a650d0d340d6839UL,
        0xe979fafacffa8335UL, 0xa369dfdf5bdfb684UL, 0xfca97e7ee57ed79bUL, 0x4819242490243db4UL,
        0x76fe3b3bec3bc5d7UL, 0x4b9aabab96ab313dUL, 0x81f0cece1fce3ed1UL, 0x2299111144118855UL,
        0x03838f8f068f0c89UL, 0x9c044e4e254e4a6bUL, 0x7366b7b7e6b7d151UL, 0xcbe0ebeb8beb0b60UL,
        0x78c13c3cf03cfdccUL, 0x1ffd81813e817cbfUL, 0x354094946a94d4feUL, 0xf31cf7f7fbf7eb0cUL,
        0x6f18b9b9deb9a167UL, 0x268b13134c13985fUL, 0x58512c2cb02c7d9cUL, 0xbb05d3d36bd3d6b8UL,
        0xd38ce7e7bbe76b5cUL, 0xdc396e6ea56e57cbUL, 0x95aac4c437c46ef3UL, 0x061b03030c03180fUL,
        0xacdc565645568a13UL, 0x885e44440d441a49UL, 0xfea07f7fe17fdf9eUL, 0x4f88a9a99ea92137UL,
        0x54672a2aa82a4d82UL, 0x6b0abbbbd6bbb16dUL, 0x9f87c1c123c146e2UL, 0xa6f153535153a202UL,
        0xa572dcdc57dcae8bUL, 0x16530b0b2c0b5827UL, 0x27019d9d4e9d9cd3UL, 0xd82b6c6cad6c47c1UL,
        0x62a43131c43195f5UL, 0xe8f37474cd7487b9UL, 0xf115f6f6fff6e309UL, 0x8c4c464605460a43UL,
        0x45a5acac8aac0926UL, 0x0fb589891e893c97UL, 0x28b414145014a044UL, 0xdfbae1e1a3e15b42UL,
        0x2ca616165816b04eUL, 0x74f73a3ae83acdd2UL, 0xd2066969b9696fd0UL, 0x124109092409482dUL,
        0xe0d77070dd70a7adUL, 0x716fb6b6e2b6d954UL, 0xbd1ed0d067d0ceb7UL, 0xc7d6eded93ed3b7eUL,
        0x85e2cccc17cc2edbUL, 0x8468424215422a57UL, 0x2d2c98985a98b4c2UL, 0x55eda4a4aaa4490eUL,
        0x50752828a0285d88UL, 0xb8865c5c6d5cda31UL, 0xed6bf8f8c7f8933fUL, 0x11c28686228644a4UL,
        }, {
        
        0x7830d818186018c0UL, 0xaf462623238c2305UL, 0xf991b8c6c63fc67eUL, 0x6fcdfbe8e887e813UL,
        0xa113cb878726874cUL, 0x626d11b8b8dab8a9UL, 0x0502090101040108UL, 0x6e9e0d4f4f214f42UL,
        0xee6c9b3636d836adUL, 0x0451ffa6a6a2a659UL, 0xbdb90cd2d26fd2deUL, 0x06f70ef5f5f3f5fbUL,
        0x80f2967979f979efUL, 0xcede306f6fa16f5fUL, 0xef3f6d91917e91fcUL, 0x07a4f852525552aaUL,
        0xfdc04760609d6027UL, 0x766535bcbccabc89UL, 0xcd2b379b9b569bacUL, 0x8c018a8e8e028e04UL,
        0x155bd2a3a3b6a371UL, 0x3c186c0c0c300c60UL, 0x8af6847b7bf17bffUL, 0xe16a803535d435b5UL,
        0x693af51d1d741de8UL, 0x47ddb3e0e0a7e053UL, 0xacb321d7d77bd7f6UL, 0xed999cc2c22fc25eUL,
        0x965c432e2eb82e6dUL, 0x7a96294b4b314b62UL, 0x21e15dfefedffea3UL, 0x16aed55757415782UL,
        0x412abd15155415a8UL, 0xb6eee87777c1779fUL, 0xeb6e923737dc37a5UL, 0x56d79ee5e5b3e57bUL,
        0xd923139f9f469f8cUL, 0x17fd23f0f0e7f0d3UL, 0x7f94204a4a354a6aUL, 0x95a944dada4fda9eUL,
        0x25b0a258587d58faUL, 0xca8fcfc9c903c906UL, 0x8d527c2929a42955UL, 0x22145a0a0a280a50UL,
        0x4f7f50b1b1feb1e1UL, 0x1a5dc9a0a0baa069UL, 0xdad6146b6bb16b7fUL, 0xab17d985852e855cUL,
        0x73673cbdbdcebd81UL, 0x34ba8f5d5d695dd2UL, 0x5020901010401080UL, 0x03f507f4f4f7f4f3UL,
        0xc08bddcbcb0bcb16UL, 0xc67cd33e3ef83eedUL, 0x110a2d0505140528UL, 0xe6ce78676781671fUL,
        0x53d597e4e4b7e473UL, 0xbb4e0227279c2725UL, 0x5882734141194132UL, 0x9d0ba78b8b168b2cUL,
        0x0153f6a7a7a6a751UL, 0x94fab27d7de97dcfUL, 0xfb374995956e95dcUL, 0x9fad56d8d847d88eUL,
        0x30eb70fbfbcbfb8bUL, 0x71c1cdeeee9fee23UL, 0x91f8bb7c7ced7cc7UL, 0xe3cc716666856617UL,
        0x8ea77bdddd53dda6UL, 0x4b2eaf17175c17b8UL, 0x468e454747014702UL, 0xdc211a9e9e429e84UL,
        0xc589d4caca0fca1eUL, 0x995a582d2db42d75UL, 0x79632ebfbfc6bf91UL, 0x1b0e3f07071c0738UL,
        0x2347acadad8ead01UL, 0x2fb4b05a5a755aeaUL, 0xb51bef838336836cUL, 0xff66b63333cc3385UL,
        0xf2c65c636391633fUL, 0x0a04120202080210UL, 0x384993aaaa92aa39UL, 0xa8e2de7171d971afUL,
        0xcf8dc6c8c807c80eUL, 0x7d32d119196419c8UL, 0x70923b4949394972UL, 0x9aaf5fd9d943d986UL,
        0x1df931f2f2eff2c3UL, 0x48dba8e3e3abe34bUL, 0x2ab6b95b5b715be2UL, 0x920dbc88881a8834UL,
        0xc8293e9a9a529aa4UL, 0xbe4c0b262698262dUL, 0xfa64bf3232c8328dUL, 0x4a7d59b0b0fab0e9UL,
        0x6acff2e9e983e91bUL, 0x331e770f0f3c0f78UL, 0xa6b733d5d573d5e6UL, 0xba1df480803a8074UL,
        0x7c6127bebec2be99UL, 0xde87ebcdcd13cd26UL, 0xe468893434d034bdUL, 0x75903248483d487aUL,
        0x24e354ffffdbffabUL, 0x8ff48d7a7af57af7UL, 0xea3d6490907a90f4UL, 0x3ebe9d5f5f615fc2UL,
        0xa0403d202080201dUL, 0xd5d00f6868bd6867UL, 0x7234ca1a1a681ad0UL, 0x2c41b7aeae82ae19UL,
        0x5e757db4b4eab4c9UL, 0x19a8ce54544d549aUL, 0xe53b7f93937693ecUL, 0xaa442f222288220dUL,
        0xe9c86364648d6407UL, 0x12ff2af1f1e3f1dbUL, 0xa2e6cc7373d173bfUL, 0x5a24821212481290UL,
        0x5d807a40401d403aUL, 0x2810480808200840UL, 0xe89b95c3c32bc356UL, 0x7bc5dfecec97ec33UL,
        0x90ab4ddbdb4bdb96UL, 0x1f5fc0a1a1bea161UL, 0x8307918d8d0e8d1cUL, 0xc97ac83d3df43df5UL,
        0xf1335b97976697ccUL, 0x0000000000000000UL, 0xd483f9cfcf1bcf36UL, 0x87566e2b2bac2b45UL,
        0xb3ece17676c57697UL, 0xb019e68282328264UL, 0xa9b128d6d67fd6feUL, 0x7736c31b1b6c1bd8UL,
        0x5b7774b5b5eeb5c1UL, 0x2943beafaf86af11UL, 0xdfd41d6a6ab56a77UL, 0x0da0ea50505d50baUL,
        0x4c8a574545094512UL, 0x18fb38f3f3ebf3cbUL, 0xf060ad3030c0309dUL, 0x74c3c4efef9bef2bUL,
        0xc37eda3f3ffc3fe5UL, 0x1caac75555495592UL, 0x1059dba2a2b2a279UL, 0x65c9e9eaea8fea03UL,
        0xecca6a656589650fUL, 0x686903babad2bab9UL, 0x935e4a2f2fbc2f65UL, 0xe79d8ec0c027c04eUL,
        0x81a160dede5fdebeUL, 0x6c38fc1c1c701ce0UL, 0x2ee746fdfdd3fdbbUL, 0x649a1f4d4d294d52UL,
        0xe0397692927292e4UL, 0xbceafa7575c9758fUL, 0x1e0c360606180630UL, 0x9809ae8a8a128a24UL,
        0x40794bb2b2f2b2f9UL, 0x59d185e6e6bfe663UL, 0x361c7e0e0e380e70UL, 0x633ee71f1f7c1ff8UL,
        0xf7c4556262956237UL, 0xa3b53ad4d477d4eeUL, 0x324d81a8a89aa829UL, 0xf4315296966296c4UL,
        0x3aef62f9f9c3f99bUL, 0xf697a3c5c533c566UL, 0xb14a102525942535UL, 0x20b2ab59597959f2UL,
        0xae15d084842a8454UL, 0xa7e4c57272d572b7UL, 0xdd72ec3939e439d5UL, 0x6198164c4c2d4c5aUL,
        0x3bbc945e5e655ecaUL, 0x85f09f7878fd78e7UL, 0xd870e53838e038ddUL, 0x8605988c8c0a8c14UL,
        0xb2bf17d1d163d1c6UL, 0x0b57e4a5a5aea541UL, 0x4dd9a1e2e2afe243UL, 0xf8c24e616199612fUL,
        0x457b42b3b3f6b3f1UL, 0xa542342121842115UL, 0xd625089c9c4a9c94UL, 0x663cee1e1e781ef0UL,
        0x5286614343114322UL, 0xfc93b1c7c73bc776UL, 0x2be54ffcfcd7fcb3UL, 0x1408240404100420UL,
        0x08a2e351515951b2UL, 0xc72f2599995e99bcUL, 0xc4da226d6da96d4fUL, 0x391a650d0d340d68UL,
        0x35e979fafacffa83UL, 0x84a369dfdf5bdfb6UL, 0x9bfca97e7ee57ed7UL, 0xb44819242490243dUL,
        0xd776fe3b3bec3bc5UL, 0x3d4b9aabab96ab31UL, 0xd181f0cece1fce3eUL, 0x5522991111441188UL,
        0x8903838f8f068f0cUL, 0x6b9c044e4e254e4aUL, 0x517366b7b7e6b7d1UL, 0x60cbe0ebeb8beb0bUL,
        0xcc78c13c3cf03cfdUL, 0xbf1ffd81813e817cUL, 0xfe354094946a94d4UL, 0x0cf31cf7f7fbf7ebUL,
        0x676f18b9b9deb9a1UL, 0x5f268b13134c1398UL, 0x9c58512c2cb02c7dUL, 0xb8bb05d3d36bd3d6UL,
        0x5cd38ce7e7bbe76bUL, 0xcbdc396e6ea56e57UL, 0xf395aac4c437c46eUL, 0x0f061b03030c0318UL,
        0x13acdc565645568aUL, 0x49885e44440d441aUL, 0x9efea07f7fe17fdfUL, 0x374f88a9a99ea921UL,
        0x8254672a2aa82a4dUL, 0x6d6b0abbbbd6bbb1UL, 0xe29f87c1c123c146UL, 0x02a6f153535153a2UL,
        0x8ba572dcdc57dcaeUL, 0x2716530b0b2c0b58UL, 0xd327019d9d4e9d9cUL, 0xc1d82b6c6cad6c47UL,
        0xf562a43131c43195UL, 0xb9e8f37474cd7487UL, 0x09f115f6f6fff6e3UL, 0x438c4c464605460aUL,
        0x2645a5acac8aac09UL, 0x970fb589891e893cUL, 0x4428b414145014a0UL, 0x42dfbae1e1a3e15bUL,
        0x4e2ca616165816b0UL, 0xd274f73a3ae83acdUL, 0xd0d2066969b9696fUL, 0x2d12410909240948UL,
        0xade0d77070dd70a7UL, 0x54716fb6b6e2b6d9UL, 0xb7bd1ed0d067d0ceUL, 0x7ec7d6eded93ed3bUL,
        0xdb85e2cccc17cc2eUL, 0x578468424215422aUL, 0xc22d2c98985a98b4UL, 0x0e55eda4a4aaa449UL,
        0x8850752828a0285dUL, 0x31b8865c5c6d5cdaUL, 0x3fed6bf8f8c7f893UL, 0xa411c28686228644UL,
        }, {
        
        0xc07830d818186018UL, 0x05af462623238c23UL, 0x7ef991b8c6c63fc6UL, 0x136fcdfbe8e887e8UL,
        0x4ca113cb87872687UL, 0xa9626d11b8b8dab8UL, 0x0805020901010401UL, 0x426e9e0d4f4f214fUL,
        0xadee6c9b3636d836UL, 0x590451ffa6a6a2a6UL, 0xdebdb90cd2d26fd2UL, 0xfb06f70ef5f5f3f5UL,
        0xef80f2967979f979UL, 0x5fcede306f6fa16fUL, 0xfcef3f6d91917e91UL, 0xaa07a4f852525552UL,
        0x27fdc04760609d60UL, 0x89766535bcbccabcUL, 0xaccd2b379b9b569bUL, 0x048c018a8e8e028eUL,
        0x71155bd2a3a3b6a3UL, 0x603c186c0c0c300cUL, 0xff8af6847b7bf17bUL, 0xb5e16a803535d435UL,
        0xe8693af51d1d741dUL, 0x5347ddb3e0e0a7e0UL, 0xf6acb321d7d77bd7UL, 0x5eed999cc2c22fc2UL,
        0x6d965c432e2eb82eUL, 0x627a96294b4b314bUL, 0xa321e15dfefedffeUL, 0x8216aed557574157UL,
        0xa8412abd15155415UL, 0x9fb6eee87777c177UL, 0xa5eb6e923737dc37UL, 0x7b56d79ee5e5b3e5UL,
        0x8cd923139f9f469fUL, 0xd317fd23f0f0e7f0UL, 0x6a7f94204a4a354aUL, 0x9e95a944dada4fdaUL,
        0xfa25b0a258587d58UL, 0x06ca8fcfc9c903c9UL, 0x558d527c2929a429UL, 0x5022145a0a0a280aUL,
        0xe14f7f50b1b1feb1UL, 0x691a5dc9a0a0baa0UL, 0x7fdad6146b6bb16bUL, 0x5cab17d985852e85UL,
        0x8173673cbdbdcebdUL, 0xd234ba8f5d5d695dUL, 0x8050209010104010UL, 0xf303f507f4f4f7f4UL,
        0x16c08bddcbcb0bcbUL, 0xedc67cd33e3ef83eUL, 0x28110a2d05051405UL, 0x1fe6ce7867678167UL,
        0x7353d597e4e4b7e4UL, 0x25bb4e0227279c27UL, 0x3258827341411941UL, 0x2c9d0ba78b8b168bUL,
        0x510153f6a7a7a6a7UL, 0xcf94fab27d7de97dUL, 0xdcfb374995956e95UL, 0x8e9fad56d8d847d8UL,
        0x8b30eb70fbfbcbfbUL, 0x2371c1cdeeee9feeUL, 0xc791f8bb7c7ced7cUL, 0x17e3cc7166668566UL,
        0xa68ea77bdddd53ddUL, 0xb84b2eaf17175c17UL, 0x02468e4547470147UL, 0x84dc211a9e9e429eUL,
        0x1ec589d4caca0fcaUL, 0x75995a582d2db42dUL, 0x9179632ebfbfc6bfUL, 0x381b0e3f07071c07UL,
        0x012347acadad8eadUL, 0xea2fb4b05a5a755aUL, 0x6cb51bef83833683UL, 0x85ff66b63333cc33UL,
        0x3ff2c65c63639163UL, 0x100a041202020802UL, 0x39384993aaaa92aaUL, 0xafa8e2de7171d971UL,
        0x0ecf8dc6c8c807c8UL, 0xc87d32d119196419UL, 0x7270923b49493949UL, 0x869aaf5fd9d943d9UL,
        0xc31df931f2f2eff2UL, 0x4b48dba8e3e3abe3UL, 0xe22ab6b95b5b715bUL, 0x34920dbc88881a88UL,
        0xa4c8293e9a9a529aUL, 0x2dbe4c0b26269826UL, 0x8dfa64bf3232c832UL, 0xe94a7d59b0b0fab0UL,
        0x1b6acff2e9e983e9UL, 0x78331e770f0f3c0fUL, 0xe6a6b733d5d573d5UL, 0x74ba1df480803a80UL,
        0x997c6127bebec2beUL, 0x26de87ebcdcd13cdUL, 0xbde468893434d034UL, 0x7a75903248483d48UL,
        0xab24e354ffffdbffUL, 0xf78ff48d7a7af57aUL, 0xf4ea3d6490907a90UL, 0xc23ebe9d5f5f615fUL,
        0x1da0403d20208020UL, 0x67d5d00f6868bd68UL, 0xd07234ca1a1a681aUL, 0x192c41b7aeae82aeUL,
        0xc95e757db4b4eab4UL, 0x9a19a8ce54544d54UL, 0xece53b7f93937693UL, 0x0daa442f22228822UL,
        0x07e9c86364648d64UL, 0xdb12ff2af1f1e3f1UL, 0xbfa2e6cc7373d173UL, 0x905a248212124812UL,
        0x3a5d807a40401d40UL, 0x4028104808082008UL, 0x56e89b95c3c32bc3UL, 0x337bc5dfecec97ecUL,
        0x9690ab4ddbdb4bdbUL, 0x611f5fc0a1a1bea1UL, 0x1c8307918d8d0e8dUL, 0xf5c97ac83d3df43dUL,
        0xccf1335b97976697UL, 0x0000000000000000UL, 0x36d483f9cfcf1bcfUL, 0x4587566e2b2bac2bUL,
        0x97b3ece17676c576UL, 0x64b019e682823282UL, 0xfea9b128d6d67fd6UL, 0xd87736c31b1b6c1bUL,
        0xc15b7774b5b5eeb5UL, 0x112943beafaf86afUL, 0x77dfd41d6a6ab56aUL, 0xba0da0ea50505d50UL,
        0x124c8a5745450945UL, 0xcb18fb38f3f3ebf3UL, 0x9df060ad3030c030UL, 0x2b74c3c4efef9befUL,
        0xe5c37eda3f3ffc3fUL, 0x921caac755554955UL, 0x791059dba2a2b2a2UL, 0x0365c9e9eaea8feaUL,
        0x0fecca6a65658965UL, 0xb9686903babad2baUL, 0x65935e4a2f2fbc2fUL, 0x4ee79d8ec0c027c0UL,
        0xbe81a160dede5fdeUL, 0xe06c38fc1c1c701cUL, 0xbb2ee746fdfdd3fdUL, 0x52649a1f4d4d294dUL,
        0xe4e0397692927292UL, 0x8fbceafa7575c975UL, 0x301e0c3606061806UL, 0x249809ae8a8a128aUL,
        0xf940794bb2b2f2b2UL, 0x6359d185e6e6bfe6UL, 0x70361c7e0e0e380eUL, 0xf8633ee71f1f7c1fUL,
        0x37f7c45562629562UL, 0xeea3b53ad4d477d4UL, 0x29324d81a8a89aa8UL, 0xc4f4315296966296UL,
        0x9b3aef62f9f9c3f9UL, 0x66f697a3c5c533c5UL, 0x35b14a1025259425UL, 0xf220b2ab59597959UL,
        0x54ae15d084842a84UL, 0xb7a7e4c57272d572UL, 0xd5dd72ec3939e439UL, 0x5a6198164c4c2d4cUL,
        0xca3bbc945e5e655eUL, 0xe785f09f7878fd78UL, 0xddd870e53838e038UL, 0x148605988c8c0a8cUL,
        0xc6b2bf17d1d163d1UL, 0x410b57e4a5a5aea5UL, 0x434dd9a1e2e2afe2UL, 0x2ff8c24e61619961UL,
        0xf1457b42b3b3f6b3UL, 0x15a5423421218421UL, 0x94d625089c9c4a9cUL, 0xf0663cee1e1e781eUL,
        0x2252866143431143UL, 0x76fc93b1c7c73bc7UL, 0xb32be54ffcfcd7fcUL, 0x2014082404041004UL,
        0xb208a2e351515951UL, 0xbcc72f2599995e99UL, 0x4fc4da226d6da96dUL, 0x68391a650d0d340dUL,
        0x8335e979fafacffaUL, 0xb684a369dfdf5bdfUL, 0xd79bfca97e7ee57eUL, 0x3db4481924249024UL,
        0xc5d776fe3b3bec3bUL, 0x313d4b9aabab96abUL, 0x3ed181f0cece1fceUL, 0x8855229911114411UL,
        0x0c8903838f8f068fUL, 0x4a6b9c044e4e254eUL, 0xd1517366b7b7e6b7UL, 0x0b60cbe0ebeb8bebUL,
        0xfdcc78c13c3cf03cUL, 0x7cbf1ffd81813e81UL, 0xd4fe354094946a94UL, 0xeb0cf31cf7f7fbf7UL,
        0xa1676f18b9b9deb9UL, 0x985f268b13134c13UL, 0x7d9c58512c2cb02cUL, 0xd6b8bb05d3d36bd3UL,
        0x6b5cd38ce7e7bbe7UL, 0x57cbdc396e6ea56eUL, 0x6ef395aac4c437c4UL, 0x180f061b03030c03UL,
        0x8a13acdc56564556UL, 0x1a49885e44440d44UL, 0xdf9efea07f7fe17fUL, 0x21374f88a9a99ea9UL,
        0x4d8254672a2aa82aUL, 0xb16d6b0abbbbd6bbUL, 0x46e29f87c1c123c1UL, 0xa202a6f153535153UL,
        0xae8ba572dcdc57dcUL, 0x582716530b0b2c0bUL, 0x9cd327019d9d4e9dUL, 0x47c1d82b6c6cad6cUL,
        0x95f562a43131c431UL, 0x87b9e8f37474cd74UL, 0xe309f115f6f6fff6UL, 0x0a438c4c46460546UL,
        0x092645a5acac8aacUL, 0x3c970fb589891e89UL, 0xa04428b414145014UL, 0x5b42dfbae1e1a3e1UL,
        0xb04e2ca616165816UL, 0xcdd274f73a3ae83aUL, 0x6fd0d2066969b969UL, 0x482d124109092409UL,
        0xa7ade0d77070dd70UL, 0xd954716fb6b6e2b6UL, 0xceb7bd1ed0d067d0UL, 0x3b7ec7d6eded93edUL,
        0x2edb85e2cccc17ccUL, 0x2a57846842421542UL, 0xb4c22d2c98985a98UL, 0x490e55eda4a4aaa4UL,
        0x5d8850752828a028UL, 0xda31b8865c5c6d5cUL, 0x933fed6bf8f8c7f8UL, 0x44a411c286862286UL,
        }, {
        
        0x18c07830d8181860UL, 0x2305af462623238cUL, 0xc67ef991b8c6c63fUL, 0xe8136fcdfbe8e887UL,
        0x874ca113cb878726UL, 0xb8a9626d11b8b8daUL, 0x0108050209010104UL, 0x4f426e9e0d4f4f21UL,
        0x36adee6c9b3636d8UL, 0xa6590451ffa6a6a2UL, 0xd2debdb90cd2d26fUL, 0xf5fb06f70ef5f5f3UL,
        0x79ef80f2967979f9UL, 0x6f5fcede306f6fa1UL, 0x91fcef3f6d91917eUL, 0x52aa07a4f8525255UL,
        0x6027fdc04760609dUL, 0xbc89766535bcbccaUL, 0x9baccd2b379b9b56UL, 0x8e048c018a8e8e02UL,
        0xa371155bd2a3a3b6UL, 0x0c603c186c0c0c30UL, 0x7bff8af6847b7bf1UL, 0x35b5e16a803535d4UL,
        0x1de8693af51d1d74UL, 0xe05347ddb3e0e0a7UL, 0xd7f6acb321d7d77bUL, 0xc25eed999cc2c22fUL,
        0x2e6d965c432e2eb8UL, 0x4b627a96294b4b31UL, 0xfea321e15dfefedfUL, 0x578216aed5575741UL,
        0x15a8412abd151554UL, 0x779fb6eee87777c1UL, 0x37a5eb6e923737dcUL, 0xe57b56d79ee5e5b3UL,
        0x9f8cd923139f9f46UL, 0xf0d317fd23f0f0e7UL, 0x4a6a7f94204a4a35UL, 0xda9e95a944dada4fUL,
        0x58fa25b0a258587dUL, 0xc906ca8fcfc9c903UL, 0x29558d527c2929a4UL, 0x0a5022145a0a0a28UL,
        0xb1e14f7f50b1b1feUL, 0xa0691a5dc9a0a0baUL, 0x6b7fdad6146b6bb1UL, 0x855cab17d985852eUL,
        0xbd8173673cbdbdceUL, 0x5dd234ba8f5d5d69UL, 0x1080502090101040UL, 0xf4f303f507f4f4f7UL,
        0xcb16c08bddcbcb0bUL, 0x3eedc67cd33e3ef8UL, 0x0528110a2d050514UL, 0x671fe6ce78676781UL,
        0xe47353d597e4e4b7UL, 0x2725bb4e0227279cUL, 0x4132588273414119UL, 0x8b2c9d0ba78b8b16UL,
        0xa7510153f6a7a7a6UL, 0x7dcf94fab27d7de9UL, 0x95dcfb374995956eUL, 0xd88e9fad56d8d847UL,
        0xfb8b30eb70fbfbcbUL, 0xee2371c1cdeeee9fUL, 0x7cc791f8bb7c7cedUL, 0x6617e3cc71666685UL,
        0xdda68ea77bdddd53UL, 0x17b84b2eaf17175cUL, 0x4702468e45474701UL, 0x9e84dc211a9e9e42UL,
        0xca1ec589d4caca0fUL, 0x2d75995a582d2db4UL, 0xbf9179632ebfbfc6UL, 0x07381b0e3f07071cUL,
        0xad012347acadad8eUL, 0x5aea2fb4b05a5a75UL, 0x836cb51bef838336UL, 0x3385ff66b63333ccUL,
        0x633ff2c65c636391UL, 0x02100a0412020208UL, 0xaa39384993aaaa92UL, 0x71afa8e2de7171d9UL,
        0xc80ecf8dc6c8c807UL, 0x19c87d32d1191964UL, 0x497270923b494939UL, 0xd9869aaf5fd9d943UL,
        0xf2c31df931f2f2efUL, 0xe34b48dba8e3e3abUL, 0x5be22ab6b95b5b71UL, 0x8834920dbc88881aUL,
        0x9aa4c8293e9a9a52UL, 0x262dbe4c0b262698UL, 0x328dfa64bf3232c8UL, 0xb0e94a7d59b0b0faUL,
        0xe91b6acff2e9e983UL, 0x0f78331e770f0f3cUL, 0xd5e6a6b733d5d573UL, 0x8074ba1df480803aUL,
        0xbe997c6127bebec2UL, 0xcd26de87ebcdcd13UL, 0x34bde468893434d0UL, 0x487a75903248483dUL,
        0xffab24e354ffffdbUL, 0x7af78ff48d7a7af5UL, 0x90f4ea3d6490907aUL, 0x5fc23ebe9d5f5f61UL,
        0x201da0403d202080UL, 0x6867d5d00f6868bdUL, 0x1ad07234ca1a1a68UL, 0xae192c41b7aeae82UL,
        0xb4c95e757db4b4eaUL, 0x549a19a8ce54544dUL, 0x93ece53b7f939376UL, 0x220daa442f222288UL,
        0x6407e9c86364648dUL, 0xf1db12ff2af1f1e3UL, 0x73bfa2e6cc7373d1UL, 0x12905a2482121248UL,
        0x403a5d807a40401dUL, 0x0840281048080820UL, 0xc356e89b95c3c32bUL, 0xec337bc5dfecec97UL,
        0xdb9690ab4ddbdb4bUL, 0xa1611f5fc0a1a1beUL, 0x8d1c8307918d8d0eUL, 0x3df5c97ac83d3df4UL,
        0x97ccf1335b979766UL, 0x0000000000000000UL, 0xcf36d483f9cfcf1bUL, 0x2b4587566e2b2bacUL,
        0x7697b3ece17676c5UL, 0x8264b019e6828232UL, 0xd6fea9b128d6d67fUL, 0x1bd87736c31b1b6cUL,
        0xb5c15b7774b5b5eeUL, 0xaf112943beafaf86UL, 0x6a77dfd41d6a6ab5UL, 0x50ba0da0ea50505dUL,
        0x45124c8a57454509UL, 0xf3cb18fb38f3f3ebUL, 0x309df060ad3030c0UL, 0xef2b74c3c4efef9bUL,
        0x3fe5c37eda3f3ffcUL, 0x55921caac7555549UL, 0xa2791059dba2a2b2UL, 0xea0365c9e9eaea8fUL,
        0x650fecca6a656589UL, 0xbab9686903babad2UL, 0x2f65935e4a2f2fbcUL, 0xc04ee79d8ec0c027UL,
        0xdebe81a160dede5fUL, 0x1ce06c38fc1c1c70UL, 0xfdbb2ee746fdfdd3UL, 0x4d52649a1f4d4d29UL,
        0x92e4e03976929272UL, 0x758fbceafa7575c9UL, 0x06301e0c36060618UL, 0x8a249809ae8a8a12UL,
        0xb2f940794bb2b2f2UL, 0xe66359d185e6e6bfUL, 0x0e70361c7e0e0e38UL, 0x1ff8633ee71f1f7cUL,
        0x6237f7c455626295UL, 0xd4eea3b53ad4d477UL, 0xa829324d81a8a89aUL, 0x96c4f43152969662UL,
        0xf99b3aef62f9f9c3UL, 0xc566f697a3c5c533UL, 0x2535b14a10252594UL, 0x59f220b2ab595979UL,
        0x8454ae15d084842aUL, 0x72b7a7e4c57272d5UL, 0x39d5dd72ec3939e4UL, 0x4c5a6198164c4c2dUL,
        0x5eca3bbc945e5e65UL, 0x78e785f09f7878fdUL, 0x38ddd870e53838e0UL, 0x8c148605988c8c0aUL,
        0xd1c6b2bf17d1d163UL, 0xa5410b57e4a5a5aeUL, 0xe2434dd9a1e2e2afUL, 0x612ff8c24e616199UL,
        0xb3f1457b42b3b3f6UL, 0x2115a54234212184UL, 0x9c94d625089c9c4aUL, 0x1ef0663cee1e1e78UL,
        0x4322528661434311UL, 0xc776fc93b1c7c73bUL, 0xfcb32be54ffcfcd7UL, 0x0420140824040410UL,
        0x51b208a2e3515159UL, 0x99bcc72f2599995eUL, 0x6d4fc4da226d6da9UL, 0x0d68391a650d0d34UL,
        0xfa8335e979fafacfUL, 0xdfb684a369dfdf5bUL, 0x7ed79bfca97e7ee5UL, 0x243db44819242490UL,
        0x3bc5d776fe3b3becUL, 0xab313d4b9aabab96UL, 0xce3ed181f0cece1fUL, 0x1188552299111144UL,
        0x8f0c8903838f8f06UL, 0x4e4a6b9c044e4e25UL, 0xb7d1517366b7b7e6UL, 0xeb0b60cbe0ebeb8bUL,
        0x3cfdcc78c13c3cf0UL, 0x817cbf1ffd81813eUL, 0x94d4fe354094946aUL, 0xf7eb0cf31cf7f7fbUL,
        0xb9a1676f18b9b9deUL, 0x13985f268b13134cUL, 0x2c7d9c58512c2cb0UL, 0xd3d6b8bb05d3d36bUL,
        0xe76b5cd38ce7e7bbUL, 0x6e57cbdc396e6ea5UL, 0xc46ef395aac4c437UL, 0x03180f061b03030cUL,
        0x568a13acdc565645UL, 0x441a49885e44440dUL, 0x7fdf9efea07f7fe1UL, 0xa921374f88a9a99eUL,
        0x2a4d8254672a2aa8UL, 0xbbb16d6b0abbbbd6UL, 0xc146e29f87c1c123UL, 0x53a202a6f1535351UL,
        0xdcae8ba572dcdc57UL, 0x0b582716530b0b2cUL, 0x9d9cd327019d9d4eUL, 0x6c47c1d82b6c6cadUL,
        0x3195f562a43131c4UL, 0x7487b9e8f37474cdUL, 0xf6e309f115f6f6ffUL, 0x460a438c4c464605UL,
        0xac092645a5acac8aUL, 0x893c970fb589891eUL, 0x14a04428b4141450UL, 0xe15b42dfbae1e1a3UL,
        0x16b04e2ca6161658UL, 0x3acdd274f73a3ae8UL, 0x696fd0d2066969b9UL, 0x09482d1241090924UL,
        0x70a7ade0d77070ddUL, 0xb6d954716fb6b6e2UL, 0xd0ceb7bd1ed0d067UL, 0xed3b7ec7d6eded93UL,
        0xcc2edb85e2cccc17UL, 0x422a578468424215UL, 0x98b4c22d2c98985aUL, 0xa4490e55eda4a4aaUL,
        0x285d8850752828a0UL, 0x5cda31b8865c5c6dUL, 0xf8933fed6bf8f8c7UL, 0x8644a411c2868622UL,
        }, {
        
        0x6018c07830d81818UL, 0x8c2305af46262323UL, 0x3fc67ef991b8c6c6UL, 0x87e8136fcdfbe8e8UL,
        0x26874ca113cb8787UL, 0xdab8a9626d11b8b8UL, 0x0401080502090101UL, 0x214f426e9e0d4f4fUL,
        0xd836adee6c9b3636UL, 0xa2a6590451ffa6a6UL, 0x6fd2debdb90cd2d2UL, 0xf3f5fb06f70ef5f5UL,
        0xf979ef80f2967979UL, 0xa16f5fcede306f6fUL, 0x7e91fcef3f6d9191UL, 0x5552aa07a4f85252UL,
        0x9d6027fdc0476060UL, 0xcabc89766535bcbcUL, 0x569baccd2b379b9bUL, 0x028e048c018a8e8eUL,
        0xb6a371155bd2a3a3UL, 0x300c603c186c0c0cUL, 0xf17bff8af6847b7bUL, 0xd435b5e16a803535UL,
        0x741de8693af51d1dUL, 0xa7e05347ddb3e0e0UL, 0x7bd7f6acb321d7d7UL, 0x2fc25eed999cc2c2UL,
        0xb82e6d965c432e2eUL, 0x314b627a96294b4bUL, 0xdffea321e15dfefeUL, 0x41578216aed55757UL,
        0x5415a8412abd1515UL, 0xc1779fb6eee87777UL, 0xdc37a5eb6e923737UL, 0xb3e57b56d79ee5e5UL,
        0x469f8cd923139f9fUL, 0xe7f0d317fd23f0f0UL, 0x354a6a7f94204a4aUL, 0x4fda9e95a944dadaUL,
        0x7d58fa25b0a25858UL, 0x03c906ca8fcfc9c9UL, 0xa429558d527c2929UL, 0x280a5022145a0a0aUL,
        0xfeb1e14f7f50b1b1UL, 0xbaa0691a5dc9a0a0UL, 0xb16b7fdad6146b6bUL, 0x2e855cab17d98585UL,
        0xcebd8173673cbdbdUL, 0x695dd234ba8f5d5dUL, 0x4010805020901010UL, 0xf7f4f303f507f4f4UL,
        0x0bcb16c08bddcbcbUL, 0xf83eedc67cd33e3eUL, 0x140528110a2d0505UL, 0x81671fe6ce786767UL,
        0xb7e47353d597e4e4UL, 0x9c2725bb4e022727UL, 0x1941325882734141UL, 0x168b2c9d0ba78b8bUL,
        0xa6a7510153f6a7a7UL, 0xe97dcf94fab27d7dUL, 0x6e95dcfb37499595UL, 0x47d88e9fad56d8d8UL,
        0xcbfb8b30eb70fbfbUL, 0x9fee2371c1cdeeeeUL, 0xed7cc791f8bb7c7cUL, 0x856617e3cc716666UL,
        0x53dda68ea77bddddUL, 0x5c17b84b2eaf1717UL, 0x014702468e454747UL, 0x429e84dc211a9e9eUL,
        0x0fca1ec589d4cacaUL, 0xb42d75995a582d2dUL, 0xc6bf9179632ebfbfUL, 0x1c07381b0e3f0707UL,
        0x8ead012347acadadUL, 0x755aea2fb4b05a5aUL, 0x36836cb51bef8383UL, 0xcc3385ff66b63333UL,
        0x91633ff2c65c6363UL, 0x0802100a04120202UL, 0x92aa39384993aaaaUL, 0xd971afa8e2de7171UL,
        0x07c80ecf8dc6c8c8UL, 0x6419c87d32d11919UL, 0x39497270923b4949UL, 0x43d9869aaf5fd9d9UL,
        0xeff2c31df931f2f2UL, 0xabe34b48dba8e3e3UL, 0x715be22ab6b95b5bUL, 0x1a8834920dbc8888UL,
        0x529aa4c8293e9a9aUL, 0x98262dbe4c0b2626UL, 0xc8328dfa64bf3232UL, 0xfab0e94a7d59b0b0UL,
        0x83e91b6acff2e9e9UL, 0x3c0f78331e770f0fUL, 0x73d5e6a6b733d5d5UL, 0x3a8074ba1df48080UL,
        0xc2be997c6127bebeUL, 0x13cd26de87ebcdcdUL, 0xd034bde468893434UL, 0x3d487a7590324848UL,
        0xdbffab24e354ffffUL, 0xf57af78ff48d7a7aUL, 0x7a90f4ea3d649090UL, 0x615fc23ebe9d5f5fUL,
        0x80201da0403d2020UL, 0xbd6867d5d00f6868UL, 0x681ad07234ca1a1aUL, 0x82ae192c41b7aeaeUL,
        0xeab4c95e757db4b4UL, 0x4d549a19a8ce5454UL, 0x7693ece53b7f9393UL, 0x88220daa442f2222UL,
        0x8d6407e9c8636464UL, 0xe3f1db12ff2af1f1UL, 0xd173bfa2e6cc7373UL, 0x4812905a24821212UL,
        0x1d403a5d807a4040UL, 0x2008402810480808UL, 0x2bc356e89b95c3c3UL, 0x97ec337bc5dfececUL,
        0x4bdb9690ab4ddbdbUL, 0xbea1611f5fc0a1a1UL, 0x0e8d1c8307918d8dUL, 0xf43df5c97ac83d3dUL,
        0x6697ccf1335b9797UL, 0x0000000000000000UL, 0x1bcf36d483f9cfcfUL, 0xac2b4587566e2b2bUL,
        0xc57697b3ece17676UL, 0x328264b019e68282UL, 0x7fd6fea9b128d6d6UL, 0x6c1bd87736c31b1bUL,
        0xeeb5c15b7774b5b5UL, 0x86af112943beafafUL, 0xb56a77dfd41d6a6aUL, 0x5d50ba0da0ea5050UL,
        0x0945124c8a574545UL, 0xebf3cb18fb38f3f3UL, 0xc0309df060ad3030UL, 0x9bef2b74c3c4efefUL,
        0xfc3fe5c37eda3f3fUL, 0x4955921caac75555UL, 0xb2a2791059dba2a2UL, 0x8fea0365c9e9eaeaUL,
        0x89650fecca6a6565UL, 0xd2bab9686903babaUL, 0xbc2f65935e4a2f2fUL, 0x27c04ee79d8ec0c0UL,
        0x5fdebe81a160dedeUL, 0x701ce06c38fc1c1cUL, 0xd3fdbb2ee746fdfdUL, 0x294d52649a1f4d4dUL,
        0x7292e4e039769292UL, 0xc9758fbceafa7575UL, 0x1806301e0c360606UL, 0x128a249809ae8a8aUL,
        0xf2b2f940794bb2b2UL, 0xbfe66359d185e6e6UL, 0x380e70361c7e0e0eUL, 0x7c1ff8633ee71f1fUL,
        0x956237f7c4556262UL, 0x77d4eea3b53ad4d4UL, 0x9aa829324d81a8a8UL, 0x6296c4f431529696UL,
        0xc3f99b3aef62f9f9UL, 0x33c566f697a3c5c5UL, 0x942535b14a102525UL, 0x7959f220b2ab5959UL,
        0x2a8454ae15d08484UL, 0xd572b7a7e4c57272UL, 0xe439d5dd72ec3939UL, 0x2d4c5a6198164c4cUL,
        0x655eca3bbc945e5eUL, 0xfd78e785f09f7878UL, 0xe038ddd870e53838UL, 0x0a8c148605988c8cUL,
        0x63d1c6b2bf17d1d1UL, 0xaea5410b57e4a5a5UL, 0xafe2434dd9a1e2e2UL, 0x99612ff8c24e6161UL,
        0xf6b3f1457b42b3b3UL, 0x842115a542342121UL, 0x4a9c94d625089c9cUL, 0x781ef0663cee1e1eUL,
        0x1143225286614343UL, 0x3bc776fc93b1c7c7UL, 0xd7fcb32be54ffcfcUL, 0x1004201408240404UL,
        0x5951b208a2e35151UL, 0x5e99bcc72f259999UL, 0xa96d4fc4da226d6dUL, 0x340d68391a650d0dUL,
        0xcffa8335e979fafaUL, 0x5bdfb684a369dfdfUL, 0xe57ed79bfca97e7eUL, 0x90243db448192424UL,
        0xec3bc5d776fe3b3bUL, 0x96ab313d4b9aababUL, 0x1fce3ed181f0ceceUL, 0x4411885522991111UL,
        0x068f0c8903838f8fUL, 0x254e4a6b9c044e4eUL, 0xe6b7d1517366b7b7UL, 0x8beb0b60cbe0ebebUL,
        0xf03cfdcc78c13c3cUL, 0x3e817cbf1ffd8181UL, 0x6a94d4fe35409494UL, 0xfbf7eb0cf31cf7f7UL,
        0xdeb9a1676f18b9b9UL, 0x4c13985f268b1313UL, 0xb02c7d9c58512c2cUL, 0x6bd3d6b8bb05d3d3UL,
        0xbbe76b5cd38ce7e7UL, 0xa56e57cbdc396e6eUL, 0x37c46ef395aac4c4UL, 0x0c03180f061b0303UL,
        0x45568a13acdc5656UL, 0x0d441a49885e4444UL, 0xe17fdf9efea07f7fUL, 0x9ea921374f88a9a9UL,
        0xa82a4d8254672a2aUL, 0xd6bbb16d6b0abbbbUL, 0x23c146e29f87c1c1UL, 0x5153a202a6f15353UL,
        0x57dcae8ba572dcdcUL, 0x2c0b582716530b0bUL, 0x4e9d9cd327019d9dUL, 0xad6c47c1d82b6c6cUL,
        0xc43195f562a43131UL, 0xcd7487b9e8f37474UL, 0xfff6e309f115f6f6UL, 0x05460a438c4c4646UL,
        0x8aac092645a5acacUL, 0x1e893c970fb58989UL, 0x5014a04428b41414UL, 0xa3e15b42dfbae1e1UL,
        0x5816b04e2ca61616UL, 0xe83acdd274f73a3aUL, 0xb9696fd0d2066969UL, 0x2409482d12410909UL,
        0xdd70a7ade0d77070UL, 0xe2b6d954716fb6b6UL, 0x67d0ceb7bd1ed0d0UL, 0x93ed3b7ec7d6ededUL,
        0x17cc2edb85e2ccccUL, 0x15422a5784684242UL, 0x5a98b4c22d2c9898UL, 0xaaa4490e55eda4a4UL,
        0xa0285d8850752828UL, 0x6d5cda31b8865c5cUL, 0xc7f8933fed6bf8f8UL, 0x228644a411c28686UL,
        }, {
        
        0x186018c07830d818UL, 0x238c2305af462623UL, 0xc63fc67ef991b8c6UL, 0xe887e8136fcdfbe8UL,
        0x8726874ca113cb87UL, 0xb8dab8a9626d11b8UL, 0x0104010805020901UL, 0x4f214f426e9e0d4fUL,
        0x36d836adee6c9b36UL, 0xa6a2a6590451ffa6UL, 0xd26fd2debdb90cd2UL, 0xf5f3f5fb06f70ef5UL,
        0x79f979ef80f29679UL, 0x6fa16f5fcede306fUL, 0x917e91fcef3f6d91UL, 0x525552aa07a4f852UL,
        0x609d6027fdc04760UL, 0xbccabc89766535bcUL, 0x9b569baccd2b379bUL, 0x8e028e048c018a8eUL,
        0xa3b6a371155bd2a3UL, 0x0c300c603c186c0cUL, 0x7bf17bff8af6847bUL, 0x35d435b5e16a8035UL,
        0x1d741de8693af51dUL, 0xe0a7e05347ddb3e0UL, 0xd77bd7f6acb321d7UL, 0xc22fc25eed999cc2UL,
        0x2eb82e6d965c432eUL, 0x4b314b627a96294bUL, 0xfedffea321e15dfeUL, 0x5741578216aed557UL,
        0x155415a8412abd15UL, 0x77c1779fb6eee877UL, 0x37dc37a5eb6e9237UL, 0xe5b3e57b56d79ee5UL,
        0x9f469f8cd923139fUL, 0xf0e7f0d317fd23f0UL, 0x4a354a6a7f94204aUL, 0xda4fda9e95a944daUL,
        0x587d58fa25b0a258UL, 0xc903c906ca8fcfc9UL, 0x29a429558d527c29UL, 0x0a280a5022145a0aUL,
        0xb1feb1e14f7f50b1UL, 0xa0baa0691a5dc9a0UL, 0x6bb16b7fdad6146bUL, 0x852e855cab17d985UL,
        0xbdcebd8173673cbdUL, 0x5d695dd234ba8f5dUL, 0x1040108050209010UL, 0xf4f7f4f303f507f4UL,
        0xcb0bcb16c08bddcbUL, 0x3ef83eedc67cd33eUL, 0x05140528110a2d05UL, 0x6781671fe6ce7867UL,
        0xe4b7e47353d597e4UL, 0x279c2725bb4e0227UL, 0x4119413258827341UL, 0x8b168b2c9d0ba78bUL,
        0xa7a6a7510153f6a7UL, 0x7de97dcf94fab27dUL, 0x956e95dcfb374995UL, 0xd847d88e9fad56d8UL,
        0xfbcbfb8b30eb70fbUL, 0xee9fee2371c1cdeeUL, 0x7ced7cc791f8bb7cUL, 0x66856617e3cc7166UL,
        0xdd53dda68ea77bddUL, 0x175c17b84b2eaf17UL, 0x47014702468e4547UL, 0x9e429e84dc211a9eUL,
        0xca0fca1ec589d4caUL, 0x2db42d75995a582dUL, 0xbfc6bf9179632ebfUL, 0x071c07381b0e3f07UL,
        0xad8ead012347acadUL, 0x5a755aea2fb4b05aUL, 0x8336836cb51bef83UL, 0x33cc3385ff66b633UL,
        0x6391633ff2c65c63UL, 0x020802100a041202UL, 0xaa92aa39384993aaUL, 0x71d971afa8e2de71UL,
        0xc807c80ecf8dc6c8UL, 0x196419c87d32d119UL, 0x4939497270923b49UL, 0xd943d9869aaf5fd9UL,
        0xf2eff2c31df931f2UL, 0xe3abe34b48dba8e3UL, 0x5b715be22ab6b95bUL, 0x881a8834920dbc88UL,
        0x9a529aa4c8293e9aUL, 0x2698262dbe4c0b26UL, 0x32c8328dfa64bf32UL, 0xb0fab0e94a7d59b0UL,
        0xe983e91b6acff2e9UL, 0x0f3c0f78331e770fUL, 0xd573d5e6a6b733d5UL, 0x803a8074ba1df480UL,
        0xbec2be997c6127beUL, 0xcd13cd26de87ebcdUL, 0x34d034bde4688934UL, 0x483d487a75903248UL,
        0xffdbffab24e354ffUL, 0x7af57af78ff48d7aUL, 0x907a90f4ea3d6490UL, 0x5f615fc23ebe9d5fUL,
        0x2080201da0403d20UL, 0x68bd6867d5d00f68UL, 0x1a681ad07234ca1aUL, 0xae82ae192c41b7aeUL,
        0xb4eab4c95e757db4UL, 0x544d549a19a8ce54UL, 0x937693ece53b7f93UL, 0x2288220daa442f22UL,
        0x648d6407e9c86364UL, 0xf1e3f1db12ff2af1UL, 0x73d173bfa2e6cc73UL, 0x124812905a248212UL,
        0x401d403a5d807a40UL, 0x0820084028104808UL, 0xc32bc356e89b95c3UL, 0xec97ec337bc5dfecUL,
        0xdb4bdb9690ab4ddbUL, 0xa1bea1611f5fc0a1UL, 0x8d0e8d1c8307918dUL, 0x3df43df5c97ac83dUL,
        0x976697ccf1335b97UL, 0x0000000000000000UL, 0xcf1bcf36d483f9cfUL, 0x2bac2b4587566e2bUL,
        0x76c57697b3ece176UL, 0x82328264b019e682UL, 0xd67fd6fea9b128d6UL, 0x1b6c1bd87736c31bUL,
        0xb5eeb5c15b7774b5UL, 0xaf86af112943beafUL, 0x6ab56a77dfd41d6aUL, 0x505d50ba0da0ea50UL,
        0x450945124c8a5745UL, 0xf3ebf3cb18fb38f3UL, 0x30c0309df060ad30UL, 0xef9bef2b74c3c4efUL,
        0x3ffc3fe5c37eda3fUL, 0x554955921caac755UL, 0xa2b2a2791059dba2UL, 0xea8fea0365c9e9eaUL,
        0x6589650fecca6a65UL, 0xbad2bab9686903baUL, 0x2fbc2f65935e4a2fUL, 0xc027c04ee79d8ec0UL,
        0xde5fdebe81a160deUL, 0x1c701ce06c38fc1cUL, 0xfdd3fdbb2ee746fdUL, 0x4d294d52649a1f4dUL,
        0x927292e4e0397692UL, 0x75c9758fbceafa75UL, 0x061806301e0c3606UL, 0x8a128a249809ae8aUL,
        0xb2f2b2f940794bb2UL, 0xe6bfe66359d185e6UL, 0x0e380e70361c7e0eUL, 0x1f7c1ff8633ee71fUL,
        0x62956237f7c45562UL, 0xd477d4eea3b53ad4UL, 0xa89aa829324d81a8UL, 0x966296c4f4315296UL,
        0xf9c3f99b3aef62f9UL, 0xc533c566f697a3c5UL, 0x25942535b14a1025UL, 0x597959f220b2ab59UL,
        0x842a8454ae15d084UL, 0x72d572b7a7e4c572UL, 0x39e439d5dd72ec39UL, 0x4c2d4c5a6198164cUL,
        0x5e655eca3bbc945eUL, 0x78fd78e785f09f78UL, 0x38e038ddd870e538UL, 0x8c0a8c148605988cUL,
        0xd163d1c6b2bf17d1UL, 0xa5aea5410b57e4a5UL, 0xe2afe2434dd9a1e2UL, 0x6199612ff8c24e61UL,
        0xb3f6b3f1457b42b3UL, 0x21842115a5423421UL, 0x9c4a9c94d625089cUL, 0x1e781ef0663cee1eUL,
        0x4311432252866143UL, 0xc73bc776fc93b1c7UL, 0xfcd7fcb32be54ffcUL, 0x0410042014082404UL,
        0x515951b208a2e351UL, 0x995e99bcc72f2599UL, 0x6da96d4fc4da226dUL, 0x0d340d68391a650dUL,
        0xfacffa8335e979faUL, 0xdf5bdfb684a369dfUL, 0x7ee57ed79bfca97eUL, 0x2490243db4481924UL,
        0x3bec3bc5d776fe3bUL, 0xab96ab313d4b9aabUL, 0xce1fce3ed181f0ceUL, 0x1144118855229911UL,
        0x8f068f0c8903838fUL, 0x4e254e4a6b9c044eUL, 0xb7e6b7d1517366b7UL, 0xeb8beb0b60cbe0ebUL,
        0x3cf03cfdcc78c13cUL, 0x813e817cbf1ffd81UL, 0x946a94d4fe354094UL, 0xf7fbf7eb0cf31cf7UL,
        0xb9deb9a1676f18b9UL, 0x134c13985f268b13UL, 0x2cb02c7d9c58512cUL, 0xd36bd3d6b8bb05d3UL,
        0xe7bbe76b5cd38ce7UL, 0x6ea56e57cbdc396eUL, 0xc437c46ef395aac4UL, 0x030c03180f061b03UL,
        0x5645568a13acdc56UL, 0x440d441a49885e44UL, 0x7fe17fdf9efea07fUL, 0xa99ea921374f88a9UL,
        0x2aa82a4d8254672aUL, 0xbbd6bbb16d6b0abbUL, 0xc123c146e29f87c1UL, 0x535153a202a6f153UL,
        0xdc57dcae8ba572dcUL, 0x0b2c0b582716530bUL, 0x9d4e9d9cd327019dUL, 0x6cad6c47c1d82b6cUL,
        0x31c43195f562a431UL, 0x74cd7487b9e8f374UL, 0xf6fff6e309f115f6UL, 0x4605460a438c4c46UL,
        0xac8aac092645a5acUL, 0x891e893c970fb589UL, 0x145014a04428b414UL, 0xe1a3e15b42dfbae1UL,
        0x165816b04e2ca616UL, 0x3ae83acdd274f73aUL, 0x69b9696fd0d20669UL, 0x092409482d124109UL,
        0x70dd70a7ade0d770UL, 0xb6e2b6d954716fb6UL, 0xd067d0ceb7bd1ed0UL, 0xed93ed3b7ec7d6edUL,
        0xcc17cc2edb85e2ccUL, 0x4215422a57846842UL, 0x985a98b4c22d2c98UL, 0xa4aaa4490e55eda4UL,
        0x28a0285d88507528UL, 0x5c6d5cda31b8865cUL, 0xf8c7f8933fed6bf8UL, 0x86228644a411c286UL,
    }
};

// WRL_OP gathers 8 S-box rows by the i-th byte of state[j] where the
// (i, shift) pair selects which byte of which state word. Mirrors
// librhash WHIRLPOOL_OP macro at whirlpool.c lines 43-52.
#define WRL_OP(src, shift) ( \
    WRL_SBOX[0][(int)((src[ (shift)      & 7] >> 56)       )] ^ \
    WRL_SBOX[1][(int)((src[((shift) + 7) & 7] >> 48) & 0xff)] ^ \
    WRL_SBOX[2][(int)((src[((shift) + 6) & 7] >> 40) & 0xff)] ^ \
    WRL_SBOX[3][(int)((src[((shift) + 5) & 7] >> 32) & 0xff)] ^ \
    WRL_SBOX[4][(int)((src[((shift) + 4) & 7] >> 24) & 0xff)] ^ \
    WRL_SBOX[5][(int)((src[((shift) + 3) & 7] >> 16) & 0xff)] ^ \
    WRL_SBOX[6][(int)((src[((shift) + 2) & 7] >>  8) & 0xff)] ^ \
    WRL_SBOX[7][(int)((src[((shift) + 1) & 7]      ) & 0xff)])

__attribute__((noinline)) void wrl_block(ulong *hash, ulong *p_block) {
    ulong K[2][8];
    ulong state[2][8];
    int m = 0;

    // Map message block to first state row and XOR into hash (Miyaguchi-
    // Preneel seed). M is already BE-packed by caller; no swap needed.
    for (int i = 0; i < 8; i++) {
        K[0][i] = hash[i];
        state[0][i] = p_block[i] ^ hash[i];
        hash[i] = state[0][i];
    }

    // 10-round AES-style mini-cipher.
    for (int i = 0; i < 10; i++) {
        K[m ^ 1][0] = WRL_OP(K[m], 0) ^ WRL_RC[i];
        K[m ^ 1][1] = WRL_OP(K[m], 1);
        K[m ^ 1][2] = WRL_OP(K[m], 2);
        K[m ^ 1][3] = WRL_OP(K[m], 3);
        K[m ^ 1][4] = WRL_OP(K[m], 4);
        K[m ^ 1][5] = WRL_OP(K[m], 5);
        K[m ^ 1][6] = WRL_OP(K[m], 6);
        K[m ^ 1][7] = WRL_OP(K[m], 7);

        state[m ^ 1][0] = WRL_OP(state[m], 0) ^ K[m ^ 1][0];
        state[m ^ 1][1] = WRL_OP(state[m], 1) ^ K[m ^ 1][1];
        state[m ^ 1][2] = WRL_OP(state[m], 2) ^ K[m ^ 1][2];
        state[m ^ 1][3] = WRL_OP(state[m], 3) ^ K[m ^ 1][3];
        state[m ^ 1][4] = WRL_OP(state[m], 4) ^ K[m ^ 1][4];
        state[m ^ 1][5] = WRL_OP(state[m], 5) ^ K[m ^ 1][5];
        state[m ^ 1][6] = WRL_OP(state[m], 6) ^ K[m ^ 1][6];
        state[m ^ 1][7] = WRL_OP(state[m], 7) ^ K[m ^ 1][7];

        m = m ^ 1;
    }

    // Miyaguchi-Preneel compression: final state XORs back into hash.
    hash[0] ^= state[0][0];
    hash[1] ^= state[0][1];
    hash[2] ^= state[0][2];
    hash[3] ^= state[0][3];
    hash[4] ^= state[0][4];
    hash[5] ^= state[0][5];
    hash[6] ^= state[0][6];
    hash[7] ^= state[0][7];
}

// ---- Tiger (Anderson + Biham 1996) block function ----
//
// Phase 5b Tier 2 sub-phase 5b.2b.1 (2026-05-27): lift tiger_block from
// RHash-master/librhash/tiger.c rhash_tiger_process_block lines 109-151
// and RHash-master/librhash/tiger_sbox.c rhash_tiger_sboxes (4 x 256
// ulong = 8 KB total).
//
// Caller convention (mirrors sha512_block / wrl_block but LE-packed):
// M[0..7] must already be LE-packed (8 message bytes per ulong, LSB =
// first byte). State is 3 ulongs natural endian; IV is the Tiger
// initial chaining value (0x0123456789abcdefUL, 0xfedcba9876543210UL,
// 0xf096a5b4c3b2e187UL). Output state, when reinterpreted as 3 LE
// ulongs, is the 24-byte digest in spec order (Tiger output is LE,
// matching MD-family convention, UNLIKE Whirlpool/SHA-2 which are BE).
//
// Donor source librhash applies le2me_64() inside the block function;
// we elide that swap because the GPU emit helper already packs M in
// LE-ulong form. Net effect is byte-identical to sph_tiger and rhash
// per the R12 NESSIE pre-flight test (16/16 PASS against published
// vectors on iMac 2026-05-27).
//
// Constant memory budget: 4 * 256 * 8 = 8 KB (TIGER_SBOX). Combined
// with WRL_SBOX (16 KB) and WRL_RC (80 B) post-Tier-2 total is ~24 KB
// of `__constant`; Pascal GTX 1080 and Apple Silicon M2 Max both
// expose >= 64 KB CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE; comfortable
// headroom remaining for Tier 3+4 additions.
//
// noinline per feedback_md5_block_noinline_pascal.md (Pascal register
// budget preservation). R7 no nested block comments (line comments
// only inside body; donor block comments stripped). 3-pass round
// structure transcribed with explicit pass1/pass2/pass3 boundaries
// citing tiger.c line numbers (the highest-risk transcription point
// per architect R2).

__constant ulong TIGER_SBOX[4][256] = {
    {
        0x02aab17cf7e90c5eUL, 0xac424b03e243a8ecUL, 0x72cd5be30dd5fcd3UL, 0x6d019b93f6f97f3aUL,
        0xcd9978ffd21f9193UL, 0x7573a1c9708029e2UL, 0xb164326b922a83c3UL, 0x46883eee04915870UL,
        0xeaace3057103ece6UL, 0xc54169b808a3535cUL, 0x4ce754918ddec47cUL, 0x0aa2f4dfdc0df40cUL,
        0x10b76f18a74dbefaUL, 0xc6ccb6235ad1ab6aUL, 0x13726121572fe2ffUL, 0x1a488c6f199d921eUL,
        0x4bc9f9f4da0007caUL, 0x26f5e6f6e85241c7UL, 0x859079dbea5947b6UL, 0x4f1885c5c99e8c92UL,
        0xd78e761ea96f864bUL, 0x8e36428c52b5c17dUL, 0x69cf6827373063c1UL, 0xb607c93d9bb4c56eUL,
        0x7d820e760e76b5eaUL, 0x645c9cc6f07fdc42UL, 0xbf38a078243342e0UL, 0x5f6b343c9d2e7d04UL,
        0xf2c28aeb600b0ec6UL, 0x6c0ed85f7254bcacUL, 0x71592281a4db4fe5UL, 0x1967fa69ce0fed9fUL,
        0xfd5293f8b96545dbUL, 0xc879e9d7f2a7600bUL, 0x860248920193194eUL, 0xa4f9533b2d9cc0b3UL,
        0x9053836c15957613UL, 0xdb6dcf8afc357bf1UL, 0x18beea7a7a370f57UL, 0x037117ca50b99066UL,
        0x6ab30a9774424a35UL, 0xf4e92f02e325249bUL, 0x7739db07061ccae1UL, 0xd8f3b49ceca42a05UL,
        0xbd56be3f51382f73UL, 0x45faed5843b0bb28UL, 0x1c813d5c11bf1f83UL, 0x8af0e4b6d75fa169UL,
        0x33ee18a487ad9999UL, 0x3c26e8eab1c94410UL, 0xb510102bc0a822f9UL, 0x141eef310ce6123bUL,
        0xfc65b90059ddb154UL, 0xe0158640c5e0e607UL, 0x884e079826c3a3cfUL, 0x930d0d9523c535fdUL,
        0x35638d754e9a2b00UL, 0x4085fccf40469dd5UL, 0xc4b17ad28be23a4cUL, 0xcab2f0fc6a3e6a2eUL,
        0x2860971a6b943fcdUL, 0x3dde6ee212e30446UL, 0x6222f32ae01765aeUL, 0x5d550bb5478308feUL,
        0xa9efa98da0eda22aUL, 0xc351a71686c40da7UL, 0x1105586d9c867c84UL, 0xdcffee85fda22853UL,
        0xccfbd0262c5eef76UL, 0xbaf294cb8990d201UL, 0xe69464f52afad975UL, 0x94b013afdf133e14UL,
        0x06a7d1a32823c958UL, 0x6f95fe5130f61119UL, 0xd92ab34e462c06c0UL, 0xed7bde33887c71d2UL,
        0x79746d6e6518393eUL, 0x5ba419385d713329UL, 0x7c1ba6b948a97564UL, 0x31987c197bfdac67UL,
        0xde6c23c44b053d02UL, 0x581c49fed002d64dUL, 0xdd474d6338261571UL, 0xaa4546c3e473d062UL,
        0x928fce349455f860UL, 0x48161bbacaab94d9UL, 0x63912430770e6f68UL, 0x6ec8a5e602c6641cUL,
        0x87282515337ddd2bUL, 0x2cda6b42034b701bUL, 0xb03d37c181cb096dUL, 0xe108438266c71c6fUL,
        0x2b3180c7eb51b255UL, 0xdf92b82f96c08bbcUL, 0x5c68c8c0a632f3baUL, 0x5504cc861c3d0556UL,
        0xabbfa4e55fb26b8fUL, 0x41848b0ab3baceb4UL, 0xb334a273aa445d32UL, 0xbca696f0a85ad881UL,
        0x24f6ec65b528d56cUL, 0x0ce1512e90f4524aUL, 0x4e9dd79d5506d35aUL, 0x258905fac6ce9779UL,
        0x2019295b3e109b33UL, 0xf8a9478b73a054ccUL, 0x2924f2f934417eb0UL, 0x3993357d536d1bc4UL,
        0x38a81ac21db6ff8bUL, 0x47c4fbf17d6016bfUL, 0x1e0faadd7667e3f5UL, 0x7abcff62938beb96UL,
        0xa78dad948fc179c9UL, 0x8f1f98b72911e50dUL, 0x61e48eae27121a91UL, 0x4d62f7ad31859808UL,
        0xeceba345ef5ceaebUL, 0xf5ceb25ebc9684ceUL, 0xf633e20cb7f76221UL, 0xa32cdf06ab8293e4UL,
        0x985a202ca5ee2ca4UL, 0xcf0b8447cc8a8fb1UL, 0x9f765244979859a3UL, 0xa8d516b1a1240017UL,
        0x0bd7ba3ebb5dc726UL, 0xe54bca55b86adb39UL, 0x1d7a3afd6c478063UL, 0x519ec608e7669eddUL,
        0x0e5715a2d149aa23UL, 0x177d4571848ff194UL, 0xeeb55f3241014c22UL, 0x0f5e5ca13a6e2ec2UL,
        0x8029927b75f5c361UL, 0xad139fabc3d6e436UL, 0x0d5df1a94ccf402fUL, 0x3e8bd948bea5dfc8UL,
        0xa5a0d357bd3ff77eUL, 0xa2d12e251f74f645UL, 0x66fd9e525e81a082UL, 0x2e0c90ce7f687a49UL,
        0xc2e8bcbeba973bc5UL, 0x000001bce509745fUL, 0x423777bbe6dab3d6UL, 0xd1661c7eaef06eb5UL,
        0xa1781f354daacfd8UL, 0x2d11284a2b16affcUL, 0xf1fc4f67fa891d1fUL, 0x73ecc25dcb920adaUL,
        0xae610c22c2a12651UL, 0x96e0a810d356b78aUL, 0x5a9a381f2fe7870fUL, 0xd5ad62ede94e5530UL,
        0xd225e5e8368d1427UL, 0x65977b70c7af4631UL, 0x99f889b2de39d74fUL, 0x233f30bf54e1d143UL,
        0x9a9675d3d9a63c97UL, 0x5470554ff334f9a8UL, 0x166acb744a4f5688UL, 0x70c74caab2e4aeadUL,
        0xf0d091646f294d12UL, 0x57b82a89684031d1UL, 0xefd95a5a61be0b6bUL, 0x2fbd12e969f2f29aUL,
        0x9bd37013feff9fe8UL, 0x3f9b0404d6085a06UL, 0x4940c1f3166cfe15UL, 0x09542c4dcdf3defbUL,
        0xb4c5218385cd5ce3UL, 0xc935b7dc4462a641UL, 0x3417f8a68ed3b63fUL, 0xb80959295b215b40UL,
        0xf99cdaef3b8c8572UL, 0x018c0614f8fcb95dUL, 0x1b14accd1a3acdf3UL, 0x84d471f200bb732dUL,
        0xc1a3110e95e8da16UL, 0x430a7220bf1a82b8UL, 0xb77e090d39df210eUL, 0x5ef4bd9f3cd05e9dUL,
        0x9d4ff6da7e57a444UL, 0xda1d60e183d4a5f8UL, 0xb287c38417998e47UL, 0xfe3edc121bb31886UL,
        0xc7fe3ccc980ccbefUL, 0xe46fb590189bfd03UL, 0x3732fd469a4c57dcUL, 0x7ef700a07cf1ad65UL,
        0x59c64468a31d8859UL, 0x762fb0b4d45b61f6UL, 0x155baed099047718UL, 0x68755e4c3d50baa6UL,
        0xe9214e7f22d8b4dfUL, 0x2addbf532eac95f4UL, 0x32ae3909b4bd0109UL, 0x834df537b08e3450UL,
        0xfa209da84220728dUL, 0x9e691d9b9efe23f7UL, 0x0446d288c4ae8d7fUL, 0x7b4cc524e169785bUL,
        0x21d87f0135ca1385UL, 0xcebb400f137b8aa5UL, 0x272e2b66580796beUL, 0x3612264125c2b0deUL,
        0x057702bdad1efbb2UL, 0xd4babb8eacf84be9UL, 0x91583139641bc67bUL, 0x8bdc2de08036e024UL,
        0x603c8156f49f68edUL, 0xf7d236f7dbef5111UL, 0x9727c4598ad21e80UL, 0xa08a0896670a5fd7UL,
        0xcb4a8f4309eba9cbUL, 0x81af564b0f7036a1UL, 0xc0b99aa778199abdUL, 0x959f1ec83fc8e952UL,
        0x8c505077794a81b9UL, 0x3acaaf8f056338f0UL, 0x07b43f50627a6778UL, 0x4a44ab49f5eccc77UL,
        0x3bc3d6e4b679ee98UL, 0x9cc0d4d1cf14108cUL, 0x4406c00b206bc8a0UL, 0x82a18854c8d72d89UL,
        0x67e366b35c3c432cUL, 0xb923dd61102b37f2UL, 0x56ab2779d884271dUL, 0xbe83e1b0ff1525afUL,
        0xfb7c65d4217e49a9UL, 0x6bdbe0e76d48e7d4UL, 0x08df828745d9179eUL, 0x22ea6a9add53bd34UL,
        0xe36e141c5622200aUL, 0x7f805d1b8cb750eeUL, 0xafe5c7a59f58e837UL, 0xe27f996a4fb1c23cUL,
        0xd3867dfb0775f0d0UL, 0xd0e673de6e88891aUL, 0x123aeb9eafb86c25UL, 0x30f1d5d5c145b895UL,
        0xbb434a2dee7269e7UL, 0x78cb67ecf931fa38UL, 0xf33b0372323bbf9cUL, 0x52d66336fb279c74UL,
        0x505f33ac0afb4eaaUL, 0xe8a5cd99a2cce187UL, 0x534974801e2d30bbUL, 0x8d2d5711d5876d90UL,
        0x1f1a412891bc038eUL, 0xd6e2e71d82e56648UL, 0x74036c3a497732b7UL, 0x89b67ed96361f5abUL,
        0xffed95d8f1ea02a2UL, 0xe72b3bd61464d43dUL, 0xa6300f170bdc4820UL, 0xebc18760ed78a77aUL,
    },
    {
        0xe6a6be5a05a12138UL, 0xb5a122a5b4f87c98UL, 0x563c6089140b6990UL, 0x4c46cb2e391f5dd5UL,
        0xd932addbc9b79434UL, 0x08ea70e42015aff5UL, 0xd765a6673e478cf1UL, 0xc4fb757eab278d99UL,
        0xdf11c6862d6e0692UL, 0xddeb84f10d7f3b16UL, 0x6f2ef604a665ea04UL, 0x4a8e0f0ff0e0dfb3UL,
        0xa5edeef83dbcba51UL, 0xfc4f0a2a0ea4371eUL, 0xe83e1da85cb38429UL, 0xdc8ff882ba1b1ce2UL,
        0xcd45505e8353e80dUL, 0x18d19a00d4db0717UL, 0x34a0cfeda5f38101UL, 0x0be77e518887caf2UL,
        0x1e341438b3c45136UL, 0xe05797f49089ccf9UL, 0xffd23f9df2591d14UL, 0x543dda228595c5cdUL,
        0x661f81fd99052a33UL, 0x8736e641db0f7b76UL, 0x15227725418e5307UL, 0xe25f7f46162eb2faUL,
        0x48a8b2126c13d9feUL, 0xafdc541792e76eeaUL, 0x03d912bfc6d1898fUL, 0x31b1aafa1b83f51bUL,
        0xf1ac2796e42ab7d9UL, 0x40a3a7d7fcd2ebacUL, 0x1056136d0afbbcc5UL, 0x7889e1dd9a6d0c85UL,
        0xd33525782a7974aaUL, 0xa7e25d09078ac09bUL, 0xbd4138b3eac6edd0UL, 0x920abfbe71eb9e70UL,
        0xa2a5d0f54fc2625cUL, 0xc054e36b0b1290a3UL, 0xf6dd59ff62fe932bUL, 0x3537354511a8ac7dUL,
        0xca845e9172fadcd4UL, 0x84f82b60329d20dcUL, 0x79c62ce1cd672f18UL, 0x8b09a2add124642cUL,
        0xd0c1e96a19d9e726UL, 0x5a786a9b4ba9500cUL, 0x0e020336634c43f3UL, 0xc17b474aeb66d822UL,
        0x6a731ae3ec9baac2UL, 0x8226667ae0840258UL, 0x67d4567691caeca5UL, 0x1d94155c4875adb5UL,
        0x6d00fd985b813fdfUL, 0x51286efcb774cd06UL, 0x5e8834471fa744afUL, 0xf72ca0aee761ae2eUL,
        0xbe40e4cdaee8e09aUL, 0xe9970bbb5118f665UL, 0x726e4beb33df1964UL, 0x703b000729199762UL,
        0x4631d816f5ef30a7UL, 0xb880b5b51504a6beUL, 0x641793c37ed84b6cUL, 0x7b21ed77f6e97d96UL,
        0x776306312ef96b73UL, 0xae528948e86ff3f4UL, 0x53dbd7f286a3f8f8UL, 0x16cadce74cfc1063UL,
        0x005c19bdfa52c6ddUL, 0x68868f5d64d46ad3UL, 0x3a9d512ccf1e186aUL, 0x367e62c2385660aeUL,
        0xe359e7ea77dcb1d7UL, 0x526c0773749abe6eUL, 0x735ae5f9d09f734bUL, 0x493fc7cc8a558ba8UL,
        0xb0b9c1533041ab45UL, 0x321958ba470a59bdUL, 0x852db00b5f46c393UL, 0x91209b2bd336b0e5UL,
        0x6e604f7d659ef19fUL, 0xb99a8ae2782ccb24UL, 0xccf52ab6c814c4c7UL, 0x4727d9afbe11727bUL,
        0x7e950d0c0121b34dUL, 0x756f435670ad471fUL, 0xf5add442615a6849UL, 0x4e87e09980b9957aUL,
        0x2acfa1df50aee355UL, 0xd898263afd2fd556UL, 0xc8f4924dd80c8fd6UL, 0xcf99ca3d754a173aUL,
        0xfe477bacaf91bf3cUL, 0xed5371f6d690c12dUL, 0x831a5c285e687094UL, 0xc5d3c90a3708a0a4UL,
        0x0f7f903717d06580UL, 0x19f9bb13b8fdf27fUL, 0xb1bd6f1b4d502843UL, 0x1c761ba38fff4012UL,
        0x0d1530c4e2e21f3bUL, 0x8943ce69a7372c8aUL, 0xe5184e11feb5ce66UL, 0x618bdb80bd736621UL,
        0x7d29bad68b574d0bUL, 0x81bb613e25e6fe5bUL, 0x071c9c10bc07913fUL, 0xc7beeb7909ac2d97UL,
        0xc3e58d353bc5d757UL, 0xeb017892f38f61e8UL, 0xd4effb9c9b1cc21aUL, 0x99727d26f494f7abUL,
        0xa3e063a2956b3e03UL, 0x9d4a8b9a4aa09c30UL, 0x3f6ab7d500090fb4UL, 0x9cc0f2a057268ac0UL,
        0x3dee9d2dedbf42d1UL, 0x330f49c87960a972UL, 0xc6b2720287421b41UL, 0x0ac59ec07c00369cUL,
        0xef4eac49cb353425UL, 0xf450244eef0129d8UL, 0x8acc46e5caf4deb6UL, 0x2ffeab63989263f7UL,
        0x8f7cb9fe5d7a4578UL, 0x5bd8f7644e634635UL, 0x427a7315bf2dc900UL, 0x17d0c4aa2125261cUL,
        0x3992486c93518e50UL, 0xb4cbfee0a2d7d4c3UL, 0x7c75d6202c5ddd8dUL, 0xdbc295d8e35b6c61UL,
        0x60b369d302032b19UL, 0xce42685fdce44132UL, 0x06f3ddb9ddf65610UL, 0x8ea4d21db5e148f0UL,
        0x20b0fce62fcd496fUL, 0x2c1b912358b0ee31UL, 0xb28317b818f5a308UL, 0xa89c1e189ca6d2cfUL,
        0x0c6b18576aaadbc8UL, 0xb65deaa91299fae3UL, 0xfb2b794b7f1027e7UL, 0x04e4317f443b5bebUL,
        0x4b852d325939d0a6UL, 0xd5ae6beefb207ffcUL, 0x309682b281c7d374UL, 0xbae309a194c3b475UL,
        0x8cc3f97b13b49f05UL, 0x98a9422ff8293967UL, 0x244b16b01076ff7cUL, 0xf8bf571c663d67eeUL,
        0x1f0d6758eee30da1UL, 0xc9b611d97adeb9b7UL, 0xb7afd5887b6c57a2UL, 0x6290ae846b984fe1UL,
        0x94df4cdeacc1a5fdUL, 0x058a5bd1c5483affUL, 0x63166cc142ba3c37UL, 0x8db8526eb2f76f40UL,
        0xe10880036f0d6d4eUL, 0x9e0523c9971d311dUL, 0x45ec2824cc7cd691UL, 0x575b8359e62382c9UL,
        0xfa9e400dc4889995UL, 0xd1823ecb45721568UL, 0xdafd983b8206082fUL, 0xaa7d29082386a8cbUL,
        0x269fcd4403b87588UL, 0x1b91f5f728bdd1e0UL, 0xe4669f39040201f6UL, 0x7a1d7c218cf04adeUL,
        0x65623c29d79ce5ceUL, 0x2368449096c00bb1UL, 0xab9bf1879da503baUL, 0xbc23ecb1a458058eUL,
        0x9a58df01bb401eccUL, 0xa070e868a85f143dUL, 0x4ff188307df2239eUL, 0x14d565b41a641183UL,
        0xee13337452701602UL, 0x950e3dcf3f285e09UL, 0x59930254b9c80953UL, 0x3bf299408930da6dUL,
        0xa955943f53691387UL, 0xa15edecaa9cb8784UL, 0x29142127352be9a0UL, 0x76f0371fff4e7afbUL,
        0x0239f450274f2228UL, 0xbb073af01d5e868bUL, 0xbfc80571c10e96c1UL, 0xd267088568222e23UL,
        0x9671a3d48e80b5b0UL, 0x55b5d38ae193bb81UL, 0x693ae2d0a18b04b8UL, 0x5c48b4ecadd5335fUL,
        0xfd743b194916a1caUL, 0x2577018134be98c4UL, 0xe77987e83c54a4adUL, 0x28e11014da33e1b9UL,
        0x270cc59e226aa213UL, 0x71495f756d1a5f60UL, 0x9be853fb60afef77UL, 0xadc786a7f7443dbfUL,
        0x0904456173b29a82UL, 0x58bc7a66c232bd5eUL, 0xf306558c673ac8b2UL, 0x41f639c6b6c9772aUL,
        0x216defe99fda35daUL, 0x11640cc71c7be615UL, 0x93c43694565c5527UL, 0xea038e6246777839UL,
        0xf9abf3ce5a3e2469UL, 0x741e768d0fd312d2UL, 0x0144b883ced652c6UL, 0xc20b5a5ba33f8552UL,
        0x1ae69633c3435a9dUL, 0x97a28ca4088cfdecUL, 0x8824a43c1e96f420UL, 0x37612fa66eeea746UL,
        0x6b4cb165f9cf0e5aUL, 0x43aa1c06a0abfb4aUL, 0x7f4dc26ff162796bUL, 0x6cbacc8e54ed9b0fUL,
        0xa6b7ffefd2bb253eUL, 0x2e25bc95b0a29d4fUL, 0x86d6a58bdef1388cUL, 0xded74ac576b6f054UL,
        0x8030bdbc2b45805dUL, 0x3c81af70e94d9289UL, 0x3eff6dda9e3100dbUL, 0xb38dc39fdfcc8847UL,
        0x123885528d17b87eUL, 0xf2da0ed240b1b642UL, 0x44cefadcd54bf9a9UL, 0x1312200e433c7ee6UL,
        0x9ffcc84f3a78c748UL, 0xf0cd1f72248576bbUL, 0xec6974053638cfe4UL, 0x2ba7b67c0cec4e4cUL,
        0xac2f4df3e5ce32edUL, 0xcb33d14326ea4c11UL, 0xa4e9044cc77e58bcUL, 0x5f513293d934fcefUL,
        0x5dc9645506e55444UL, 0x50de418f317de40aUL, 0x388cb31a69dde259UL, 0x2db4a83455820a86UL,
        0x9010a91e84711ae9UL, 0x4df7f0b7b1498371UL, 0xd62a2eabc0977179UL, 0x22fac097aa8d5c0eUL,
    },
    {
        0xf49fcc2ff1daf39bUL, 0x487fd5c66ff29281UL, 0xe8a30667fcdca83fUL, 0x2c9b4be3d2fcce63UL,
        0xda3ff74b93fbbbc2UL, 0x2fa165d2fe70ba66UL, 0xa103e279970e93d4UL, 0xbecdec77b0e45e71UL,
        0xcfb41e723985e497UL, 0xb70aaa025ef75017UL, 0xd42309f03840b8e0UL, 0x8efc1ad035898579UL,
        0x96c6920be2b2abc5UL, 0x66af4163375a9172UL, 0x2174abdcca7127fbUL, 0xb33ccea64a72ff41UL,
        0xf04a4933083066a5UL, 0x8d970acdd7289af5UL, 0x8f96e8e031c8c25eUL, 0xf3fec02276875d47UL,
        0xec7bf310056190ddUL, 0xf5adb0aebb0f1491UL, 0x9b50f8850fd58892UL, 0x4975488358b74de8UL,
        0xa3354ff691531c61UL, 0x0702bbe481d2c6eeUL, 0x89fb24057deded98UL, 0xac3075138596e902UL,
        0x1d2d3580172772edUL, 0xeb738fc28e6bc30dUL, 0x5854ef8f63044326UL, 0x9e5c52325add3bbeUL,
        0x90aa53cf325c4623UL, 0xc1d24d51349dd067UL, 0x2051cfeea69ea624UL, 0x13220f0a862e7e4fUL,
        0xce39399404e04864UL, 0xd9c42ca47086fcb7UL, 0x685ad2238a03e7ccUL, 0x066484b2ab2ff1dbUL,
        0xfe9d5d70efbf79ecUL, 0x5b13b9dd9c481854UL, 0x15f0d475ed1509adUL, 0x0bebcd060ec79851UL,
        0xd58c6791183ab7f8UL, 0xd1187c5052f3eee4UL, 0xc95d1192e54e82ffUL, 0x86eea14cb9ac6ca2UL,
        0x3485beb153677d5dUL, 0xdd191d781f8c492aUL, 0xf60866baa784ebf9UL, 0x518f643ba2d08c74UL,
        0x8852e956e1087c22UL, 0xa768cb8dc410ae8dUL, 0x38047726bfec8e1aUL, 0xa67738b4cd3b45aaUL,
        0xad16691cec0dde19UL, 0xc6d4319380462e07UL, 0xc5a5876d0ba61938UL, 0x16b9fa1fa58fd840UL,
        0x188ab1173ca74f18UL, 0xabda2f98c99c021fUL, 0x3e0580ab134ae816UL, 0x5f3b05b773645abbUL,
        0x2501a2be5575f2f6UL, 0x1b2f74004e7e8ba9UL, 0x1cd7580371e8d953UL, 0x7f6ed89562764e30UL,
        0xb15926ff596f003dUL, 0x9f65293da8c5d6b9UL, 0x6ecef04dd690f84cUL, 0x4782275fff33af88UL,
        0xe41433083f820801UL, 0xfd0dfe409a1af9b5UL, 0x4325a3342cdb396bUL, 0x8ae77e62b301b252UL,
        0xc36f9e9f6655615aUL, 0x85455a2d92d32c09UL, 0xf2c7dea949477485UL, 0x63cfb4c133a39ebaUL,
        0x83b040cc6ebc5462UL, 0x3b9454c8fdb326b0UL, 0x56f56a9e87ffd78cUL, 0x2dc2940d99f42bc6UL,
        0x98f7df096b096e2dUL, 0x19a6e01e3ad852bfUL, 0x42a99ccbdbd4b40bUL, 0xa59998af45e9c559UL,
        0x366295e807d93186UL, 0x6b48181bfaa1f773UL, 0x1fec57e2157a0a1dUL, 0x4667446af6201ad5UL,
        0xe615ebcacfb0f075UL, 0xb8f31f4f68290778UL, 0x22713ed6ce22d11eUL, 0x3057c1a72ec3c93bUL,
        0xcb46acc37c3f1f2fUL, 0xdbb893fd02aaf50eUL, 0x331fd92e600b9fcfUL, 0xa498f96148ea3ad6UL,
        0xa8d8426e8b6a83eaUL, 0xa089b274b7735cdcUL, 0x87f6b3731e524a11UL, 0x118808e5cbc96749UL,
        0x9906e4c7b19bd394UL, 0xafed7f7e9b24a20cUL, 0x6509eadeeb3644a7UL, 0x6c1ef1d3e8ef0edeUL,
        0xb9c97d43e9798fb4UL, 0xa2f2d784740c28a3UL, 0x7b8496476197566fUL, 0x7a5be3e6b65f069dUL,
        0xf96330ed78be6f10UL, 0xeee60de77a076a15UL, 0x2b4bee4aa08b9bd0UL, 0x6a56a63ec7b8894eUL,
        0x02121359ba34fef4UL, 0x4cbf99f8283703fcUL, 0x398071350caf30c8UL, 0xd0a77a89f017687aUL,
        0xf1c1a9eb9e423569UL, 0x8c7976282dee8199UL, 0x5d1737a5dd1f7abdUL, 0x4f53433c09a9fa80UL,
        0xfa8b0c53df7ca1d9UL, 0x3fd9dcbc886ccb77UL, 0xc040917ca91b4720UL, 0x7dd00142f9d1dcdfUL,
        0x8476fc1d4f387b58UL, 0x23f8e7c5f3316503UL, 0x032a2244e7e37339UL, 0x5c87a5d750f5a74bUL,
        0x082b4cc43698992eUL, 0xdf917becb858f63cUL, 0x3270b8fc5bf86ddaUL, 0x10ae72bb29b5dd76UL,
        0x576ac94e7700362bUL, 0x1ad112dac61efb8fUL, 0x691bc30ec5faa427UL, 0xff246311cc327143UL,
        0x3142368e30e53206UL, 0x71380e31e02ca396UL, 0x958d5c960aad76f1UL, 0xf8d6f430c16da536UL,
        0xc8ffd13f1be7e1d2UL, 0x7578ae66004ddbe1UL, 0x05833f01067be646UL, 0xbb34b5ad3bfe586dUL,
        0x095f34c9a12b97f0UL, 0x247ab64525d60ca8UL, 0xdcdbc6f3017477d1UL, 0x4a2e14d4decad24dUL,
        0xbdb5e6d9be0a1eebUL, 0x2a7e70f7794301abUL, 0xdef42d8a270540fdUL, 0x01078ec0a34c22c1UL,
        0xe5de511af4c16387UL, 0x7ebb3a52bd9a330aUL, 0x77697857aa7d6435UL, 0x004e831603ae4c32UL,
        0xe7a21020ad78e312UL, 0x9d41a70c6ab420f2UL, 0x28e06c18ea1141e6UL, 0xd2b28cbd984f6b28UL,
        0x26b75f6c446e9d83UL, 0xba47568c4d418d7fUL, 0xd80badbfe6183d8eUL, 0x0e206d7f5f166044UL,
        0xe258a43911cbca3eUL, 0x723a1746b21dc0bcUL, 0xc7caa854f5d7cdd3UL, 0x7cac32883d261d9cUL,
        0x7690c26423ba942cUL, 0x17e55524478042b8UL, 0xe0be477656a2389fUL, 0x4d289b5e67ab2da0UL,
        0x44862b9c8fbbfd31UL, 0xb47cc8049d141365UL, 0x822c1b362b91c793UL, 0x4eb14655fb13dfd8UL,
        0x1ecbba0714e2a97bUL, 0x6143459d5cde5f14UL, 0x53a8fbf1d5f0ac89UL, 0x97ea04d81c5e5b00UL,
        0x622181a8d4fdb3f3UL, 0xe9bcd341572a1208UL, 0x1411258643cce58aUL, 0x9144c5fea4c6e0a4UL,
        0x0d33d06565cf620fUL, 0x54a48d489f219ca1UL, 0xc43e5eac6d63c821UL, 0xa9728b3a72770dafUL,
        0xd7934e7b20df87efUL, 0xe35503b61a3e86e5UL, 0xcae321fbc819d504UL, 0x129a50b3ac60bfa6UL,
        0xcd5e68ea7e9fb6c3UL, 0xb01c90199483b1c7UL, 0x3de93cd5c295376cUL, 0xaed52edf2ab9ad13UL,
        0x2e60f512c0a07884UL, 0xbc3d86a3e36210c9UL, 0x35269d9b163951ceUL, 0x0c7d6e2ad0cdb5faUL,
        0x59e86297d87f5733UL, 0x298ef221898db0e7UL, 0x55000029d1a5aa7eUL, 0x8bc08ae1b5061b45UL,
        0xc2c31c2b6c92703aUL, 0x94cc596baf25ef42UL, 0x0a1d73db22540456UL, 0x04b6a0f9d9c4179aUL,
        0xeffdafa2ae3d3c60UL, 0xf7c8075bb49496c4UL, 0x9cc5c7141d1cd4e3UL, 0x78bd1638218e5534UL,
        0xb2f11568f850246aUL, 0xedfabcfa9502bc29UL, 0x796ce5f2da23051bUL, 0xaae128b0dc93537cUL,
        0x3a493da0ee4b29aeUL, 0xb5df6b2c416895d7UL, 0xfcabbd25122d7f37UL, 0x70810b58105dc4b1UL,
        0xe10fdd37f7882a90UL, 0x524dcab5518a3f5cUL, 0x3c9e85878451255bUL, 0x4029828119bd34e2UL,
        0x74a05b6f5d3ceccbUL, 0xb610021542e13ecaUL, 0x0ff979d12f59e2acUL, 0x6037da27e4f9cc50UL,
        0x5e92975a0df1847dUL, 0xd66de190d3e623feUL, 0x5032d6b87b568048UL, 0x9a36b7ce8235216eUL,
        0x80272a7a24f64b4aUL, 0x93efed8b8c6916f7UL, 0x37ddbff44cce1555UL, 0x4b95db5d4b99bd25UL,
        0x92d3fda169812fc0UL, 0xfb1a4a9a90660bb6UL, 0x730c196946a4b9b2UL, 0x81e289aa7f49da68UL,
        0x64669a0f83b1a05fUL, 0x27b3ff7d9644f48bUL, 0xcc6b615c8db675b3UL, 0x674f20b9bcebbe95UL,
        0x6f31238275655982UL, 0x5ae488713e45cf05UL, 0xbf619f9954c21157UL, 0xeabac46040a8eae9UL,
        0x454c6fe9f2c0c1cdUL, 0x419cf6496412691cUL, 0xd3dc3bef265b0f70UL, 0x6d0e60f5c3578a9eUL,
    },
    {
        0x5b0e608526323c55UL, 0x1a46c1a9fa1b59f5UL, 0xa9e245a17c4c8ffaUL, 0x65ca5159db2955d7UL,
        0x05db0a76ce35afc2UL, 0x81eac77ea9113d45UL, 0x528ef88ab6ac0a0dUL, 0xa09ea253597be3ffUL,
        0x430ddfb3ac48cd56UL, 0xc4b3a67af45ce46fUL, 0x4ececfd8fbe2d05eUL, 0x3ef56f10b39935f0UL,
        0x0b22d6829cd619c6UL, 0x17fd460a74df2069UL, 0x6cf8cc8e8510ed40UL, 0xd6c824bf3a6ecaa7UL,
        0x61243d581a817049UL, 0x048bacb6bbc163a2UL, 0xd9a38ac27d44cc32UL, 0x7fddff5baaf410abUL,
        0xad6d495aa804824bUL, 0xe1a6a74f2d8c9f94UL, 0xd4f7851235dee8e3UL, 0xfd4b7f886540d893UL,
        0x247c20042aa4bfdaUL, 0x096ea1c517d1327cUL, 0xd56966b4361a6685UL, 0x277da5c31221057dUL,
        0x94d59893a43acff7UL, 0x64f0c51ccdc02281UL, 0x3d33bcc4ff6189dbUL, 0xe005cb184ce66af1UL,
        0xff5ccd1d1db99beaUL, 0xb0b854a7fe42980fUL, 0x7bd46a6a718d4b9fUL, 0xd10fa8cc22a5fd8cUL,
        0xd31484952be4bd31UL, 0xc7fa975fcb243847UL, 0x4886ed1e5846c407UL, 0x28cddb791eb70b04UL,
        0xc2b00be2f573417fUL, 0x5c9590452180f877UL, 0x7a6bddfff370eb00UL, 0xce509e38d6d9d6a4UL,
        0xebeb0f00647fa702UL, 0x1dcc06cf76606f06UL, 0xe4d9f28ba286ff0aUL, 0xd85a305dc918c262UL,
        0x475b1d8732225f54UL, 0x2d4fb51668ccb5feUL, 0xa679b9d9d72bba20UL, 0x53841c0d912d43a5UL,
        0x3b7eaa48bf12a4e8UL, 0x781e0e47f22f1ddfUL, 0xeff20ce60ab50973UL, 0x20d261d19dffb742UL,
        0x16a12b03062a2e39UL, 0x1960eb2239650495UL, 0x251c16fed50eb8b8UL, 0x9ac0c330f826016eUL,
        0xed152665953e7671UL, 0x02d63194a6369570UL, 0x5074f08394b1c987UL, 0x70ba598c90b25ce1UL,
        0x794a15810b9742f6UL, 0x0d5925e9fcaf8c6cUL, 0x3067716cd868744eUL, 0x910ab077e8d7731bUL,
        0x6a61bbdb5ac42f61UL, 0x93513efbf0851567UL, 0xf494724b9e83e9d5UL, 0xe887e1985c09648dUL,
        0x34b1d3c675370cfdUL, 0xdc35e433bc0d255dUL, 0xd0aab84234131be0UL, 0x08042a50b48b7eafUL,
        0x9997c4ee44a3ab35UL, 0x829a7b49201799d0UL, 0x263b8307b7c54441UL, 0x752f95f4fd6a6ca6UL,
        0x927217402c08c6e5UL, 0x2a8ab754a795d9eeUL, 0xa442f7552f72943dUL, 0x2c31334e19781208UL,
        0x4fa98d7ceaee6291UL, 0x55c3862f665db309UL, 0xbd0610175d53b1f3UL, 0x46fe6cb840413f27UL,
        0x3fe03792df0cfa59UL, 0xcfe700372eb85e8fUL, 0xa7be29e7adbce118UL, 0xe544ee5cde8431ddUL,
        0x8a781b1b41f1873eUL, 0xa5c94c78a0d2f0e7UL, 0x39412e2877b60728UL, 0xa1265ef3afc9a62cUL,
        0xbcc2770c6a2506c5UL, 0x3ab66dd5dce1ce12UL, 0xe65499d04a675b37UL, 0x7d8f523481bfd216UL,
        0x0f6f64fcec15f389UL, 0x74efbe618b5b13c8UL, 0xacdc82b714273e1dUL, 0xdd40bfe003199d17UL,
        0x37e99257e7e061f8UL, 0xfa52626904775aaaUL, 0x8bbbf63a463d56f9UL, 0xf0013f1543a26e64UL,
        0xa8307e9f879ec898UL, 0xcc4c27a4150177ccUL, 0x1b432f2cca1d3348UL, 0xde1d1f8f9f6fa013UL,
        0x606602a047a7ddd6UL, 0xd237ab64cc1cb2c7UL, 0x9b938e7225fcd1d3UL, 0xec4e03708e0ff476UL,
        0xfeb2fbda3d03c12dUL, 0xae0bced2ee43889aUL, 0x22cb8923ebfb4f43UL, 0x69360d013cf7396dUL,
        0x855e3602d2d4e022UL, 0x073805bad01f784cUL, 0x33e17a133852f546UL, 0xdf4874058ac7b638UL,
        0xba92b29c678aa14aUL, 0x0ce89fc76cfaadcdUL, 0x5f9d4e0908339e34UL, 0xf1afe9291f5923b9UL,
        0x6e3480f60f4a265fUL, 0xeebf3a2ab29b841cUL, 0xe21938a88f91b4adUL, 0x57dfeff845c6d3c3UL,
        0x2f006b0bf62caaf2UL, 0x62f479ef6f75ee78UL, 0x11a55ad41c8916a9UL, 0xf229d29084fed453UL,
        0x42f1c27b16b000e6UL, 0x2b1f76749823c074UL, 0x4b76eca3c2745360UL, 0x8c98f463b91691bdUL,
        0x14bcc93cf1ade66aUL, 0x8885213e6d458397UL, 0x8e177df0274d4711UL, 0xb49b73b5503f2951UL,
        0x10168168c3f96b6bUL, 0x0e3d963b63cab0aeUL, 0x8dfc4b5655a1db14UL, 0xf789f1356e14de5cUL,
        0x683e68af4e51dac1UL, 0xc9a84f9d8d4b0fd9UL, 0x3691e03f52a0f9d1UL, 0x5ed86e46e1878e80UL,
        0x3c711a0e99d07150UL, 0x5a0865b20c4e9310UL, 0x56fbfc1fe4f0682eUL, 0xea8d5de3105edf9bUL,
        0x71abfdb12379187aUL, 0x2eb99de1bee77b9cUL, 0x21ecc0ea33cf4523UL, 0x59a4d7521805c7a1UL,
        0x3896f5eb56ae7c72UL, 0xaa638f3db18f75dcUL, 0x9f39358dabe9808eUL, 0xb7defa91c00b72acUL,
        0x6b5541fd62492d92UL, 0x6dc6dee8f92e4d5bUL, 0x353f57abc4beea7eUL, 0x735769d6da5690ceUL,
        0x0a234aa642391484UL, 0xf6f9508028f80d9dUL, 0xb8e319a27ab3f215UL, 0x31ad9c1151341a4dUL,
        0x773c22a57bef5805UL, 0x45c7561a07968633UL, 0xf913da9e249dbe36UL, 0xda652d9b78a64c68UL,
        0x4c27a97f3bc334efUL, 0x76621220e66b17f4UL, 0x967743899acd7d0bUL, 0xf3ee5bcae0ed6782UL,
        0x409f753600c879fcUL, 0x06d09a39b5926db6UL, 0x6f83aeb0317ac588UL, 0x01e6ca4a86381f21UL,
        0x66ff3462d19f3025UL, 0x72207c24ddfd3bfbUL, 0x4af6b6d3e2ece2ebUL, 0x9c994dbec7ea08deUL,
        0x49ace597b09a8bc4UL, 0xb38c4766cf0797baUL, 0x131b9373c57c2a75UL, 0xb1822cce61931e58UL,
        0x9d7555b909ba1c0cUL, 0x127fafdd937d11d2UL, 0x29da3badc66d92e4UL, 0xa2c1d57154c2ecbcUL,
        0x58c5134d82f6fe24UL, 0x1c3ae3515b62274fUL, 0xe907c82e01cb8126UL, 0xf8ed091913e37fcbUL,
        0x3249d8f9c80046c9UL, 0x80cf9bede388fb63UL, 0x1881539a116cf19eUL, 0x5103f3f76bd52457UL,
        0x15b7e6f5ae47f7a8UL, 0xdbd7c6ded47e9ccfUL, 0x44e55c410228bb1aUL, 0xb647d4255edb4e99UL,
        0x5d11882bb8aafc30UL, 0xf5098bbb29d3212aUL, 0x8fb5ea14e90296b3UL, 0x677b942157dd025aUL,
        0xfb58e7c0a390acb5UL, 0x89d3674c83bd4a01UL, 0x9e2da4df4bf3b93bUL, 0xfcc41e328cab4829UL,
        0x03f38c96ba582c52UL, 0xcad1bdbd7fd85db2UL, 0xbbb442c16082ae83UL, 0xb95fe86ba5da9ab0UL,
        0xb22e04673771a93fUL, 0x845358c9493152d8UL, 0xbe2a488697b4541eUL, 0x95a2dc2dd38e6966UL,
        0xc02c11ac923c852bUL, 0x2388b1990df2a87bUL, 0x7c8008fa1b4f37beUL, 0x1f70d0c84d54e503UL,
        0x5490adec7ece57d4UL, 0x002b3c27d9063a3aUL, 0x7eaea3848030a2bfUL, 0xc602326ded2003c0UL,
        0x83a7287d69a94086UL, 0xc57a5fcb30f57a8aUL, 0xb56844e479ebe779UL, 0xa373b40f05dcbce9UL,
        0xd71a786e88570ee2UL, 0x879cbacdbde8f6a0UL, 0x976ad1bcc164a32fUL, 0xab21e25e9666d78bUL,
        0x901063aae5e5c33cUL, 0x9818b34448698d90UL, 0xe36487ae3e1e8abbUL, 0xafbdf931893bdcb4UL,
        0x6345a0dc5fbbd519UL, 0x8628fe269b9465caUL, 0x1e5d01603f9c51ecUL, 0x4de44006a15049b7UL,
        0xbf6c70e5f776cbb1UL, 0x411218f2ef552bedUL, 0xcb0c0708705a36a3UL, 0xe74d14754f986044UL,
        0xcd56d9430ea8280eUL, 0xc12591d7535f5065UL, 0xc83223f1720aef96UL, 0xc3a0396f7363a51fUL
    }
};

// TIGER round/pass/key_schedule macros mirror RHash-master/librhash/tiger.c
// lines 47-101 (CPU_X64 path; GPU has native 64-bit ulong). Per donor
// macro: round(a,b,c,x,mul) does c ^= x; a -= t1[byte0(c)] ^ t2[byte2(c)]
// ^ t3[byte4(c)] ^ t4[byte6(c)]; b += t4[byte1(c)] ^ t3[byte3(c)] ^
// t2[byte5(c)] ^ t1[byte7(c)]; b *= mul. Then pass = 8 rotating rounds
// over (a,b,c). key_schedule mutates message words x0..x7 between passes.
#define TIGER_T1 TIGER_SBOX[0]
#define TIGER_T2 TIGER_SBOX[1]
#define TIGER_T3 TIGER_SBOX[2]
#define TIGER_T4 TIGER_SBOX[3]

#define TIGER_ROUND(a, b, c, x, mul) \
    (c) ^= (x); \
    (a) -= TIGER_T1[(int)((uchar)(c))] ^ \
           TIGER_T2[(int)((uchar)((c) >> 16))] ^ \
           TIGER_T3[(int)((uchar)((c) >> 32))] ^ \
           TIGER_T4[(int)((uchar)((c) >> 48))]; \
    (b) += TIGER_T4[(int)((uchar)((c) >>  8))] ^ \
           TIGER_T3[(int)((uchar)((c) >> 24))] ^ \
           TIGER_T2[(int)((uchar)((c) >> 40))] ^ \
           TIGER_T1[(int)((uchar)((c) >> 56))]; \
    (b) *= (mul);

#define TIGER_PASS(a, b, c, mul) \
    TIGER_ROUND(a, b, c, x0, mul) \
    TIGER_ROUND(b, c, a, x1, mul) \
    TIGER_ROUND(c, a, b, x2, mul) \
    TIGER_ROUND(a, b, c, x3, mul) \
    TIGER_ROUND(b, c, a, x4, mul) \
    TIGER_ROUND(c, a, b, x5, mul) \
    TIGER_ROUND(a, b, c, x6, mul) \
    TIGER_ROUND(b, c, a, x7, mul)

#define TIGER_KEY_SCHEDULE \
    x0 -= x7 ^ 0xa5a5a5a5a5a5a5a5UL; \
    x1 ^= x0; \
    x2 += x1; \
    x3 -= x2 ^ ((~x1) << 19); \
    x4 ^= x3; \
    x5 += x4; \
    x6 -= x5 ^ ((~x4) >> 23); \
    x7 ^= x6; \
    x0 += x7; \
    x1 -= x0 ^ ((~x7) << 19); \
    x2 ^= x1; \
    x3 += x2; \
    x4 -= x3 ^ ((~x2) >> 23); \
    x5 ^= x4; \
    x6 += x5; \
    x7 -= x6 ^ 0x0123456789abcdefUL;

// tiger_block: 3-pass Tiger compression. Caller convention mirrors
// sha512_block / wrl_block: state[0..2] is the 3-ulong chaining value
// (initialized by caller to the Tiger IV on first invocation), M[0..7]
// is the 64-byte block as 8 ulongs in LITTLE-ENDIAN packing (matches
// Tiger spec; donor le2me_64 elided -- emit helper packs M in LE-ulong
// form). Output: state mutated in place; reinterpret as 3 LE ulongs =
// 24 bytes for digest. Cites RHash-master/librhash/tiger.c:109-151
// (rhash_tiger_process_block) for the 3-pass / 2-key_schedule structure
// and the feedforward (a XOR, b SUB, c ADD).
__attribute__((noinline)) void tiger_block(ulong *state, ulong *M) {
    ulong a, b, c;
    ulong x0, x1, x2, x3, x4, x5, x6, x7;

    // Load message words (LE-packed by caller; donor le2me_64 elided).
    x0 = M[0]; x1 = M[1]; x2 = M[2]; x3 = M[3];
    x4 = M[4]; x5 = M[5]; x6 = M[6]; x7 = M[7];

    a = state[0];
    b = state[1];
    c = state[2];

    // Pass 1: mul = 5 (CPU_X64 path -- tiger.c line 140)
    TIGER_PASS(a, b, c, 5UL)
    TIGER_KEY_SCHEDULE
    // Pass 2: mul = 7, rotated arg order (c, a, b) per tiger.c line 142
    TIGER_PASS(c, a, b, 7UL)
    TIGER_KEY_SCHEDULE
    // Pass 3: mul = 9, rotated arg order (b, c, a) per tiger.c line 144
    TIGER_PASS(b, c, a, 9UL)

    // Feedforward: state[0] XOR a, state[1] = b - state[1], state[2] += c
    // per tiger.c line 148-150.
    state[0] = a ^ state[0];
    state[1] = b - state[1];
    state[2] = c + state[2];
}

// Phase 5b Tier 3 sub-phase 5b.3a.1 (2026-05-27): lift haval3_block from
// mhash-0.9.9.9/lib/haval.c havalTransform3 lines 113-241. Public-domain
// donor (Paulo S.L.M. Barreto 1998). 3-pass HAVAL compression -- the
// canonical default per the HAVAL paper (Zheng-Pieprzyk-Seberry 1993).
//
// Caller convention:
//   - state[0..7] is the 8-uint HAVAL chaining value. Caller initializes
//     to HAVAL_IV (see below) on first invocation, or to previous-block
//     output for the multi-block path.
//   - M[0..31] is the 128-byte block as 32 uint32 in LE-packed form
//     (matches donor convention: donor reads (mutils_word32 *) D[32]
//     directly on LE machines, the swap path is for BE only and elided
//     here). Emit helper packs M LE.
//   - Output: state mutated in place; reinterpret state[0..7] as 8 LE
//     uints = 32 bytes for the 256-bit raw state. The post-compression
//     digest fold (per havalFinal lines 816-911) is applied in the
//     emit helper, not here.
//
// HAVAL uses 5 Boolean F-functions (F1-F5) defined inline as macros
// below (matches donor lines 65-82); LE-only no swap needed; round
// constants per donor inlined as 32-bit hex literals at each round step.
// 96 round constants total for 3-pass (32 per pass for passes 2 and 3;
// pass 1 has no round constants per HAVAL spec).
//
// Constant memory budget: HAVAL_IV (32 B). No __constant arrays needed
// for the 3-pass core -- all round constants are compile-time-known
// literals inlined into each step (matches donor structure; lets the
// JIT compiler fold them into the immediate operands).
//
// R6 noinline per feedback_md5_block_noinline_pascal.md: keep
// haval<P>_block functions noinline so Pascal's per-thread register
// budget isn't blown by inlining 130+ LOC of compression code into the
// host kernel.
//
// R8 no nested block comments per feedback_no_nested_block_comments_in_-
// cl.md: donor /* */ blocks stripped; only line comments used here.

#define HAVAL_F1(X6, X5, X4, X3, X2, X1, X0) \
    (((X1) & ((X4) ^ (X0))) ^ ((X2) & (X5)) ^ ((X3) & (X6)) ^ (X0))

#define HAVAL_F2(X6, X5, X4, X3, X2, X1, X0) \
    (((X2) & (((X1) & (~(X3))) ^ ((X4) & (X5)) ^ (X6) ^ (X0))) ^ \
     (((X4) & ((X1) ^ (X5))) ^ ((X3) & (X5)) ^ (X0)))

#define HAVAL_F3(X6, X5, X4, X3, X2, X1, X0) \
    (((X3) & (((X1) & (X2)) ^ (X6) ^ (X0))) ^ ((X1) & (X4)) ^ \
     ((X2) & (X5)) ^ (X0))

#define HAVAL_F4(X6, X5, X4, X3, X2, X1, X0) \
    (((X4) & (((~(X2)) & (X5)) ^ ((X3) | (X6)) ^ (X1) ^ (X0))) ^ \
     ((X3) & (((X1) & (X2)) ^ (X5) ^ (X6))) ^ ((X2) & (X6)) ^ (X0))

#define HAVAL_F5(X6, X5, X4, X3, X2, X1, X0) \
    (((X1) & ((X4) ^ ((X0) & (X2) & (X3)))) ^ \
     (((X2) ^ (X0)) & (X5)) ^ ((X3) & (X6)) ^ (X0))

// HAVAL's ROTR_32 -- rotate-right of a uint32 by n bits. Use OpenCL's
// rotate() with the count expressed as (32 - n) since OpenCL rotate is
// rotate-LEFT. Matches donor donor ROTR semantics.
#define HAVAL_ROTR32(v, n) rotate((uint)(v), (uint)(32 - (n)))

// HAVAL initial value (digest[0..7] per donor havalInit lines 657-664).
// Public per donor (Pi-fractional constants from HAVAL spec).
// Caller initializes state[] to HAVAL_IV before first haval3_block call.
//
// HAVAL_IV exposed as a __constant array for callers that need a single
// definition. Emit helpers use it as the IV source for fresh hashing.
__constant uint HAVAL_IV[8] = {
    0x243F6A88u, 0x85A308D3u, 0x13198A2Eu, 0x03707344u,
    0xA4093822u, 0x299F31D0u, 0x082EFA98u, 0xEC4E6C89u
};

// haval3_block: HAVAL 3-pass compression on a 128-byte block. Caller
// supplies state[8] (the chaining value; init to HAVAL_IV) and M[32]
// (the 128-byte block as 32 LE-packed uint32 words). Mutates state in
// place. The post-compression digest fold for 128/160/192/224 widths
// happens in the emit helper after the FINAL haval3_block call (256-bit
// has no fold).
//
// Donor lineage: mhash-0.9.9.9/lib/haval.c havalTransform3 lines 113-241.
// Donor uses a separate T[8] temporary array; we localize via stack uints
// T0..T7 (initialized from state) so all data lives in registers.
//
// Pass 1: 32 steps using F1, no round constants (HAVAL spec).
// Pass 2: 32 steps using F2, with 32 round constants 0x452821E6..0xC25A59B5.
// Pass 3: 32 steps using F3, with 32 round constants 0x9C30D539..0x6C24CF5C.
//
// Word index permutations (W[0]..W[31]) are inline per donor; matches
// the per-step word ordering published in the HAVAL paper.
__attribute__((noinline)) void haval3_block(uint *state, const uint *M) {
    uint T0, T1, T2, T3, T4, T5, T6, T7;
    uint E0, E1, E2, E3, E4, E5, E6, E7;

    // Snapshot input state for feedforward at end of pass 3.
    E0 = state[0]; E1 = state[1]; E2 = state[2]; E3 = state[3];
    E4 = state[4]; E5 = state[5]; E6 = state[6]; E7 = state[7];

    // PASS 1 (F1, no round constants).
    T7 = HAVAL_ROTR32(HAVAL_F1(E1, E0, E3, E5, E6, E2, E4), 7) + HAVAL_ROTR32(E7, 11) + M[ 0];
    T6 = HAVAL_ROTR32(HAVAL_F1(E0, T7, E2, E4, E5, E1, E3), 7) + HAVAL_ROTR32(E6, 11) + M[ 1];
    T5 = HAVAL_ROTR32(HAVAL_F1(T7, T6, E1, E3, E4, E0, E2), 7) + HAVAL_ROTR32(E5, 11) + M[ 2];
    T4 = HAVAL_ROTR32(HAVAL_F1(T6, T5, E0, E2, E3, T7, E1), 7) + HAVAL_ROTR32(E4, 11) + M[ 3];
    T3 = HAVAL_ROTR32(HAVAL_F1(T5, T4, T7, E1, E2, T6, E0), 7) + HAVAL_ROTR32(E3, 11) + M[ 4];
    T2 = HAVAL_ROTR32(HAVAL_F1(T4, T3, T6, E0, E1, T5, T7), 7) + HAVAL_ROTR32(E2, 11) + M[ 5];
    T1 = HAVAL_ROTR32(HAVAL_F1(T3, T2, T5, T7, E0, T4, T6), 7) + HAVAL_ROTR32(E1, 11) + M[ 6];
    T0 = HAVAL_ROTR32(HAVAL_F1(T2, T1, T4, T6, T7, T3, T5), 7) + HAVAL_ROTR32(E0, 11) + M[ 7];

    T7 = HAVAL_ROTR32(HAVAL_F1(T1, T0, T3, T5, T6, T2, T4), 7) + HAVAL_ROTR32(T7, 11) + M[ 8];
    T6 = HAVAL_ROTR32(HAVAL_F1(T0, T7, T2, T4, T5, T1, T3), 7) + HAVAL_ROTR32(T6, 11) + M[ 9];
    T5 = HAVAL_ROTR32(HAVAL_F1(T7, T6, T1, T3, T4, T0, T2), 7) + HAVAL_ROTR32(T5, 11) + M[10];
    T4 = HAVAL_ROTR32(HAVAL_F1(T6, T5, T0, T2, T3, T7, T1), 7) + HAVAL_ROTR32(T4, 11) + M[11];
    T3 = HAVAL_ROTR32(HAVAL_F1(T5, T4, T7, T1, T2, T6, T0), 7) + HAVAL_ROTR32(T3, 11) + M[12];
    T2 = HAVAL_ROTR32(HAVAL_F1(T4, T3, T6, T0, T1, T5, T7), 7) + HAVAL_ROTR32(T2, 11) + M[13];
    T1 = HAVAL_ROTR32(HAVAL_F1(T3, T2, T5, T7, T0, T4, T6), 7) + HAVAL_ROTR32(T1, 11) + M[14];
    T0 = HAVAL_ROTR32(HAVAL_F1(T2, T1, T4, T6, T7, T3, T5), 7) + HAVAL_ROTR32(T0, 11) + M[15];

    T7 = HAVAL_ROTR32(HAVAL_F1(T1, T0, T3, T5, T6, T2, T4), 7) + HAVAL_ROTR32(T7, 11) + M[16];
    T6 = HAVAL_ROTR32(HAVAL_F1(T0, T7, T2, T4, T5, T1, T3), 7) + HAVAL_ROTR32(T6, 11) + M[17];
    T5 = HAVAL_ROTR32(HAVAL_F1(T7, T6, T1, T3, T4, T0, T2), 7) + HAVAL_ROTR32(T5, 11) + M[18];
    T4 = HAVAL_ROTR32(HAVAL_F1(T6, T5, T0, T2, T3, T7, T1), 7) + HAVAL_ROTR32(T4, 11) + M[19];
    T3 = HAVAL_ROTR32(HAVAL_F1(T5, T4, T7, T1, T2, T6, T0), 7) + HAVAL_ROTR32(T3, 11) + M[20];
    T2 = HAVAL_ROTR32(HAVAL_F1(T4, T3, T6, T0, T1, T5, T7), 7) + HAVAL_ROTR32(T2, 11) + M[21];
    T1 = HAVAL_ROTR32(HAVAL_F1(T3, T2, T5, T7, T0, T4, T6), 7) + HAVAL_ROTR32(T1, 11) + M[22];
    T0 = HAVAL_ROTR32(HAVAL_F1(T2, T1, T4, T6, T7, T3, T5), 7) + HAVAL_ROTR32(T0, 11) + M[23];

    T7 = HAVAL_ROTR32(HAVAL_F1(T1, T0, T3, T5, T6, T2, T4), 7) + HAVAL_ROTR32(T7, 11) + M[24];
    T6 = HAVAL_ROTR32(HAVAL_F1(T0, T7, T2, T4, T5, T1, T3), 7) + HAVAL_ROTR32(T6, 11) + M[25];
    T5 = HAVAL_ROTR32(HAVAL_F1(T7, T6, T1, T3, T4, T0, T2), 7) + HAVAL_ROTR32(T5, 11) + M[26];
    T4 = HAVAL_ROTR32(HAVAL_F1(T6, T5, T0, T2, T3, T7, T1), 7) + HAVAL_ROTR32(T4, 11) + M[27];
    T3 = HAVAL_ROTR32(HAVAL_F1(T5, T4, T7, T1, T2, T6, T0), 7) + HAVAL_ROTR32(T3, 11) + M[28];
    T2 = HAVAL_ROTR32(HAVAL_F1(T4, T3, T6, T0, T1, T5, T7), 7) + HAVAL_ROTR32(T2, 11) + M[29];
    T1 = HAVAL_ROTR32(HAVAL_F1(T3, T2, T5, T7, T0, T4, T6), 7) + HAVAL_ROTR32(T1, 11) + M[30];
    T0 = HAVAL_ROTR32(HAVAL_F1(T2, T1, T4, T6, T7, T3, T5), 7) + HAVAL_ROTR32(T0, 11) + M[31];

    // PASS 2 (F2, with round constants).
    T7 = HAVAL_ROTR32(HAVAL_F2(T4, T2, T1, T0, T5, T3, T6), 7) + HAVAL_ROTR32(T7, 11) + M[ 5] + 0x452821E6u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T3, T1, T0, T7, T4, T2, T5), 7) + HAVAL_ROTR32(T6, 11) + M[14] + 0x38D01377u;
    T5 = HAVAL_ROTR32(HAVAL_F2(T2, T0, T7, T6, T3, T1, T4), 7) + HAVAL_ROTR32(T5, 11) + M[26] + 0xBE5466CFu;
    T4 = HAVAL_ROTR32(HAVAL_F2(T1, T7, T6, T5, T2, T0, T3), 7) + HAVAL_ROTR32(T4, 11) + M[18] + 0x34E90C6Cu;
    T3 = HAVAL_ROTR32(HAVAL_F2(T0, T6, T5, T4, T1, T7, T2), 7) + HAVAL_ROTR32(T3, 11) + M[11] + 0xC0AC29B7u;
    T2 = HAVAL_ROTR32(HAVAL_F2(T7, T5, T4, T3, T0, T6, T1), 7) + HAVAL_ROTR32(T2, 11) + M[28] + 0xC97C50DDu;
    T1 = HAVAL_ROTR32(HAVAL_F2(T6, T4, T3, T2, T7, T5, T0), 7) + HAVAL_ROTR32(T1, 11) + M[ 7] + 0x3F84D5B5u;
    T0 = HAVAL_ROTR32(HAVAL_F2(T5, T3, T2, T1, T6, T4, T7), 7) + HAVAL_ROTR32(T0, 11) + M[16] + 0xB5470917u;

    T7 = HAVAL_ROTR32(HAVAL_F2(T4, T2, T1, T0, T5, T3, T6), 7) + HAVAL_ROTR32(T7, 11) + M[ 0] + 0x9216D5D9u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T3, T1, T0, T7, T4, T2, T5), 7) + HAVAL_ROTR32(T6, 11) + M[23] + 0x8979FB1Bu;
    T5 = HAVAL_ROTR32(HAVAL_F2(T2, T0, T7, T6, T3, T1, T4), 7) + HAVAL_ROTR32(T5, 11) + M[20] + 0xD1310BA6u;
    T4 = HAVAL_ROTR32(HAVAL_F2(T1, T7, T6, T5, T2, T0, T3), 7) + HAVAL_ROTR32(T4, 11) + M[22] + 0x98DFB5ACu;
    T3 = HAVAL_ROTR32(HAVAL_F2(T0, T6, T5, T4, T1, T7, T2), 7) + HAVAL_ROTR32(T3, 11) + M[ 1] + 0x2FFD72DBu;
    T2 = HAVAL_ROTR32(HAVAL_F2(T7, T5, T4, T3, T0, T6, T1), 7) + HAVAL_ROTR32(T2, 11) + M[10] + 0xD01ADFB7u;
    T1 = HAVAL_ROTR32(HAVAL_F2(T6, T4, T3, T2, T7, T5, T0), 7) + HAVAL_ROTR32(T1, 11) + M[ 4] + 0xB8E1AFEDu;
    T0 = HAVAL_ROTR32(HAVAL_F2(T5, T3, T2, T1, T6, T4, T7), 7) + HAVAL_ROTR32(T0, 11) + M[ 8] + 0x6A267E96u;

    T7 = HAVAL_ROTR32(HAVAL_F2(T4, T2, T1, T0, T5, T3, T6), 7) + HAVAL_ROTR32(T7, 11) + M[30] + 0xBA7C9045u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T3, T1, T0, T7, T4, T2, T5), 7) + HAVAL_ROTR32(T6, 11) + M[ 3] + 0xF12C7F99u;
    T5 = HAVAL_ROTR32(HAVAL_F2(T2, T0, T7, T6, T3, T1, T4), 7) + HAVAL_ROTR32(T5, 11) + M[21] + 0x24A19947u;
    T4 = HAVAL_ROTR32(HAVAL_F2(T1, T7, T6, T5, T2, T0, T3), 7) + HAVAL_ROTR32(T4, 11) + M[ 9] + 0xB3916CF7u;
    T3 = HAVAL_ROTR32(HAVAL_F2(T0, T6, T5, T4, T1, T7, T2), 7) + HAVAL_ROTR32(T3, 11) + M[17] + 0x0801F2E2u;
    T2 = HAVAL_ROTR32(HAVAL_F2(T7, T5, T4, T3, T0, T6, T1), 7) + HAVAL_ROTR32(T2, 11) + M[24] + 0x858EFC16u;
    T1 = HAVAL_ROTR32(HAVAL_F2(T6, T4, T3, T2, T7, T5, T0), 7) + HAVAL_ROTR32(T1, 11) + M[29] + 0x636920D8u;
    T0 = HAVAL_ROTR32(HAVAL_F2(T5, T3, T2, T1, T6, T4, T7), 7) + HAVAL_ROTR32(T0, 11) + M[ 6] + 0x71574E69u;

    T7 = HAVAL_ROTR32(HAVAL_F2(T4, T2, T1, T0, T5, T3, T6), 7) + HAVAL_ROTR32(T7, 11) + M[19] + 0xA458FEA3u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T3, T1, T0, T7, T4, T2, T5), 7) + HAVAL_ROTR32(T6, 11) + M[12] + 0xF4933D7Eu;
    T5 = HAVAL_ROTR32(HAVAL_F2(T2, T0, T7, T6, T3, T1, T4), 7) + HAVAL_ROTR32(T5, 11) + M[15] + 0x0D95748Fu;
    T4 = HAVAL_ROTR32(HAVAL_F2(T1, T7, T6, T5, T2, T0, T3), 7) + HAVAL_ROTR32(T4, 11) + M[13] + 0x728EB658u;
    T3 = HAVAL_ROTR32(HAVAL_F2(T0, T6, T5, T4, T1, T7, T2), 7) + HAVAL_ROTR32(T3, 11) + M[ 2] + 0x718BCD58u;
    T2 = HAVAL_ROTR32(HAVAL_F2(T7, T5, T4, T3, T0, T6, T1), 7) + HAVAL_ROTR32(T2, 11) + M[25] + 0x82154AEEu;
    T1 = HAVAL_ROTR32(HAVAL_F2(T6, T4, T3, T2, T7, T5, T0), 7) + HAVAL_ROTR32(T1, 11) + M[31] + 0x7B54A41Du;
    T0 = HAVAL_ROTR32(HAVAL_F2(T5, T3, T2, T1, T6, T4, T7), 7) + HAVAL_ROTR32(T0, 11) + M[27] + 0xC25A59B5u;

    // PASS 3 (F3, with round constants). Last 8 steps include the
    // feedforward state[i] += T[i] (donor lines 229-236).
    T7 = HAVAL_ROTR32(HAVAL_F3(T6, T1, T2, T3, T4, T5, T0), 7) + HAVAL_ROTR32(T7, 11) + M[19] + 0x9C30D539u;
    T6 = HAVAL_ROTR32(HAVAL_F3(T5, T0, T1, T2, T3, T4, T7), 7) + HAVAL_ROTR32(T6, 11) + M[ 9] + 0x2AF26013u;
    T5 = HAVAL_ROTR32(HAVAL_F3(T4, T7, T0, T1, T2, T3, T6), 7) + HAVAL_ROTR32(T5, 11) + M[ 4] + 0xC5D1B023u;
    T4 = HAVAL_ROTR32(HAVAL_F3(T3, T6, T7, T0, T1, T2, T5), 7) + HAVAL_ROTR32(T4, 11) + M[20] + 0x286085F0u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T2, T5, T6, T7, T0, T1, T4), 7) + HAVAL_ROTR32(T3, 11) + M[28] + 0xCA417918u;
    T2 = HAVAL_ROTR32(HAVAL_F3(T1, T4, T5, T6, T7, T0, T3), 7) + HAVAL_ROTR32(T2, 11) + M[17] + 0xB8DB38EFu;
    T1 = HAVAL_ROTR32(HAVAL_F3(T0, T3, T4, T5, T6, T7, T2), 7) + HAVAL_ROTR32(T1, 11) + M[ 8] + 0x8E79DCB0u;
    T0 = HAVAL_ROTR32(HAVAL_F3(T7, T2, T3, T4, T5, T6, T1), 7) + HAVAL_ROTR32(T0, 11) + M[22] + 0x603A180Eu;

    T7 = HAVAL_ROTR32(HAVAL_F3(T6, T1, T2, T3, T4, T5, T0), 7) + HAVAL_ROTR32(T7, 11) + M[29] + 0x6C9E0E8Bu;
    T6 = HAVAL_ROTR32(HAVAL_F3(T5, T0, T1, T2, T3, T4, T7), 7) + HAVAL_ROTR32(T6, 11) + M[14] + 0xB01E8A3Eu;
    T5 = HAVAL_ROTR32(HAVAL_F3(T4, T7, T0, T1, T2, T3, T6), 7) + HAVAL_ROTR32(T5, 11) + M[25] + 0xD71577C1u;
    T4 = HAVAL_ROTR32(HAVAL_F3(T3, T6, T7, T0, T1, T2, T5), 7) + HAVAL_ROTR32(T4, 11) + M[12] + 0xBD314B27u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T2, T5, T6, T7, T0, T1, T4), 7) + HAVAL_ROTR32(T3, 11) + M[24] + 0x78AF2FDAu;
    T2 = HAVAL_ROTR32(HAVAL_F3(T1, T4, T5, T6, T7, T0, T3), 7) + HAVAL_ROTR32(T2, 11) + M[30] + 0x55605C60u;
    T1 = HAVAL_ROTR32(HAVAL_F3(T0, T3, T4, T5, T6, T7, T2), 7) + HAVAL_ROTR32(T1, 11) + M[16] + 0xE65525F3u;
    T0 = HAVAL_ROTR32(HAVAL_F3(T7, T2, T3, T4, T5, T6, T1), 7) + HAVAL_ROTR32(T0, 11) + M[26] + 0xAA55AB94u;

    T7 = HAVAL_ROTR32(HAVAL_F3(T6, T1, T2, T3, T4, T5, T0), 7) + HAVAL_ROTR32(T7, 11) + M[31] + 0x57489862u;
    T6 = HAVAL_ROTR32(HAVAL_F3(T5, T0, T1, T2, T3, T4, T7), 7) + HAVAL_ROTR32(T6, 11) + M[15] + 0x63E81440u;
    T5 = HAVAL_ROTR32(HAVAL_F3(T4, T7, T0, T1, T2, T3, T6), 7) + HAVAL_ROTR32(T5, 11) + M[ 7] + 0x55CA396Au;
    T4 = HAVAL_ROTR32(HAVAL_F3(T3, T6, T7, T0, T1, T2, T5), 7) + HAVAL_ROTR32(T4, 11) + M[ 3] + 0x2AAB10B6u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T2, T5, T6, T7, T0, T1, T4), 7) + HAVAL_ROTR32(T3, 11) + M[ 1] + 0xB4CC5C34u;
    T2 = HAVAL_ROTR32(HAVAL_F3(T1, T4, T5, T6, T7, T0, T3), 7) + HAVAL_ROTR32(T2, 11) + M[ 0] + 0x1141E8CEu;
    T1 = HAVAL_ROTR32(HAVAL_F3(T0, T3, T4, T5, T6, T7, T2), 7) + HAVAL_ROTR32(T1, 11) + M[18] + 0xA15486AFu;
    T0 = HAVAL_ROTR32(HAVAL_F3(T7, T2, T3, T4, T5, T6, T1), 7) + HAVAL_ROTR32(T0, 11) + M[27] + 0x7C72E993u;

    // Final 8 steps: compute T[i] AND feedforward into state[i]
    // (donor lines 229-236: E[7] += T[7] = ...).
    T7 = HAVAL_ROTR32(HAVAL_F3(T6, T1, T2, T3, T4, T5, T0), 7) + HAVAL_ROTR32(T7, 11) + M[13] + 0xB3EE1411u;
    state[7] = E7 + T7;
    T6 = HAVAL_ROTR32(HAVAL_F3(T5, T0, T1, T2, T3, T4, T7), 7) + HAVAL_ROTR32(T6, 11) + M[ 6] + 0x636FBC2Au;
    state[6] = E6 + T6;
    T5 = HAVAL_ROTR32(HAVAL_F3(T4, T7, T0, T1, T2, T3, T6), 7) + HAVAL_ROTR32(T5, 11) + M[21] + 0x2BA9C55Du;
    state[5] = E5 + T5;
    T4 = HAVAL_ROTR32(HAVAL_F3(T3, T6, T7, T0, T1, T2, T5), 7) + HAVAL_ROTR32(T4, 11) + M[10] + 0x741831F6u;
    state[4] = E4 + T4;
    T3 = HAVAL_ROTR32(HAVAL_F3(T2, T5, T6, T7, T0, T1, T4), 7) + HAVAL_ROTR32(T3, 11) + M[23] + 0xCE5C3E16u;
    state[3] = E3 + T3;
    T2 = HAVAL_ROTR32(HAVAL_F3(T1, T4, T5, T6, T7, T0, T3), 7) + HAVAL_ROTR32(T2, 11) + M[11] + 0x9B87931Eu;
    state[2] = E2 + T2;
    T1 = HAVAL_ROTR32(HAVAL_F3(T0, T3, T4, T5, T6, T7, T2), 7) + HAVAL_ROTR32(T1, 11) + M[ 5] + 0xAFD6BA33u;
    state[1] = E1 + T1;
    T0 = HAVAL_ROTR32(HAVAL_F3(T7, T2, T3, T4, T5, T6, T1), 7) + HAVAL_ROTR32(T0, 11) + M[ 2] + 0x6C24CF5Cu;
    state[0] = E0 + T0;
}

// Phase 5b Tier 3 sub-phase 5b.3b.2 (2026-05-27): lift haval4_block from
// mhash-0.9.9.9/lib/haval.c havalTransform4 lines 244-409. Public-domain
// donor (Paulo S.L.M. Barreto 1998). 4-pass HAVAL compression.
//
// Same caller convention as haval3_block: state[0..7] is the 8-uint HAVAL
// chaining value (init to HAVAL_IV on first invocation); M[0..31] is the
// 128-byte block as 32 LE-packed uint32 words. Mutates state in place; the
// post-compression digest fold for 128/160/192/224 widths happens in the
// emit helper after the FINAL haval4_block call (256-bit has no fold).
//
// 4-pass structure: pass 1 F1 no constants; pass 2 F2 + 32 RC; pass 3 F3
// + 32 RC; pass 4 F4 + 32 RC (0x7A325381..0x137A3BE4). 128 round constants
// total for 4-pass (32 each in passes 2/3/4; pass 1 has none).
//
// CRITICAL: the per-step F-function argument orderings and W[] word-index
// permutations in passes 1-3 of havalTransform4 DIFFER from havalTransform3
// (the message-word schedule and the rotor are pass-count specific). This
// is a verbatim transcription of havalTransform4, NOT a reuse of the
// haval3_block passes.
//
// R6 noinline per feedback_md5_block_noinline_pascal.md. R8 no nested
// block comments: donor /* */ stripped, line comments only.
//
// HAVAL_F1-F5, HAVAL_ROTR32, HAVAL_IV are shared with haval3_block above.
__attribute__((noinline)) void haval4_block(uint *state, const uint *M) {
    uint T0, T1, T2, T3, T4, T5, T6, T7;
    uint E0, E1, E2, E3, E4, E5, E6, E7;

    // Snapshot input state for feedforward at end of pass 4.
    E0 = state[0]; E1 = state[1]; E2 = state[2]; E3 = state[3];
    E4 = state[4]; E5 = state[5]; E6 = state[6]; E7 = state[7];

    // PASS 1 (F1, no round constants).
    T7 = HAVAL_ROTR32(HAVAL_F1(E2, E6, E1, E4, E5, E3, E0), 7) + HAVAL_ROTR32(E7, 11) + M[ 0];
    T6 = HAVAL_ROTR32(HAVAL_F1(E1, E5, E0, E3, E4, E2, T7), 7) + HAVAL_ROTR32(E6, 11) + M[ 1];
    T5 = HAVAL_ROTR32(HAVAL_F1(E0, E4, T7, E2, E3, E1, T6), 7) + HAVAL_ROTR32(E5, 11) + M[ 2];
    T4 = HAVAL_ROTR32(HAVAL_F1(T7, E3, T6, E1, E2, E0, T5), 7) + HAVAL_ROTR32(E4, 11) + M[ 3];
    T3 = HAVAL_ROTR32(HAVAL_F1(T6, E2, T5, E0, E1, T7, T4), 7) + HAVAL_ROTR32(E3, 11) + M[ 4];
    T2 = HAVAL_ROTR32(HAVAL_F1(T5, E1, T4, T7, E0, T6, T3), 7) + HAVAL_ROTR32(E2, 11) + M[ 5];
    T1 = HAVAL_ROTR32(HAVAL_F1(T4, E0, T3, T6, T7, T5, T2), 7) + HAVAL_ROTR32(E1, 11) + M[ 6];
    T0 = HAVAL_ROTR32(HAVAL_F1(T3, T7, T2, T5, T6, T4, T1), 7) + HAVAL_ROTR32(E0, 11) + M[ 7];

    T7 = HAVAL_ROTR32(HAVAL_F1(T2, T6, T1, T4, T5, T3, T0), 7) + HAVAL_ROTR32(T7, 11) + M[ 8];
    T6 = HAVAL_ROTR32(HAVAL_F1(T1, T5, T0, T3, T4, T2, T7), 7) + HAVAL_ROTR32(T6, 11) + M[ 9];
    T5 = HAVAL_ROTR32(HAVAL_F1(T0, T4, T7, T2, T3, T1, T6), 7) + HAVAL_ROTR32(T5, 11) + M[10];
    T4 = HAVAL_ROTR32(HAVAL_F1(T7, T3, T6, T1, T2, T0, T5), 7) + HAVAL_ROTR32(T4, 11) + M[11];
    T3 = HAVAL_ROTR32(HAVAL_F1(T6, T2, T5, T0, T1, T7, T4), 7) + HAVAL_ROTR32(T3, 11) + M[12];
    T2 = HAVAL_ROTR32(HAVAL_F1(T5, T1, T4, T7, T0, T6, T3), 7) + HAVAL_ROTR32(T2, 11) + M[13];
    T1 = HAVAL_ROTR32(HAVAL_F1(T4, T0, T3, T6, T7, T5, T2), 7) + HAVAL_ROTR32(T1, 11) + M[14];
    T0 = HAVAL_ROTR32(HAVAL_F1(T3, T7, T2, T5, T6, T4, T1), 7) + HAVAL_ROTR32(T0, 11) + M[15];

    T7 = HAVAL_ROTR32(HAVAL_F1(T2, T6, T1, T4, T5, T3, T0), 7) + HAVAL_ROTR32(T7, 11) + M[16];
    T6 = HAVAL_ROTR32(HAVAL_F1(T1, T5, T0, T3, T4, T2, T7), 7) + HAVAL_ROTR32(T6, 11) + M[17];
    T5 = HAVAL_ROTR32(HAVAL_F1(T0, T4, T7, T2, T3, T1, T6), 7) + HAVAL_ROTR32(T5, 11) + M[18];
    T4 = HAVAL_ROTR32(HAVAL_F1(T7, T3, T6, T1, T2, T0, T5), 7) + HAVAL_ROTR32(T4, 11) + M[19];
    T3 = HAVAL_ROTR32(HAVAL_F1(T6, T2, T5, T0, T1, T7, T4), 7) + HAVAL_ROTR32(T3, 11) + M[20];
    T2 = HAVAL_ROTR32(HAVAL_F1(T5, T1, T4, T7, T0, T6, T3), 7) + HAVAL_ROTR32(T2, 11) + M[21];
    T1 = HAVAL_ROTR32(HAVAL_F1(T4, T0, T3, T6, T7, T5, T2), 7) + HAVAL_ROTR32(T1, 11) + M[22];
    T0 = HAVAL_ROTR32(HAVAL_F1(T3, T7, T2, T5, T6, T4, T1), 7) + HAVAL_ROTR32(T0, 11) + M[23];

    T7 = HAVAL_ROTR32(HAVAL_F1(T2, T6, T1, T4, T5, T3, T0), 7) + HAVAL_ROTR32(T7, 11) + M[24];
    T6 = HAVAL_ROTR32(HAVAL_F1(T1, T5, T0, T3, T4, T2, T7), 7) + HAVAL_ROTR32(T6, 11) + M[25];
    T5 = HAVAL_ROTR32(HAVAL_F1(T0, T4, T7, T2, T3, T1, T6), 7) + HAVAL_ROTR32(T5, 11) + M[26];
    T4 = HAVAL_ROTR32(HAVAL_F1(T7, T3, T6, T1, T2, T0, T5), 7) + HAVAL_ROTR32(T4, 11) + M[27];
    T3 = HAVAL_ROTR32(HAVAL_F1(T6, T2, T5, T0, T1, T7, T4), 7) + HAVAL_ROTR32(T3, 11) + M[28];
    T2 = HAVAL_ROTR32(HAVAL_F1(T5, T1, T4, T7, T0, T6, T3), 7) + HAVAL_ROTR32(T2, 11) + M[29];
    T1 = HAVAL_ROTR32(HAVAL_F1(T4, T0, T3, T6, T7, T5, T2), 7) + HAVAL_ROTR32(T1, 11) + M[30];
    T0 = HAVAL_ROTR32(HAVAL_F1(T3, T7, T2, T5, T6, T4, T1), 7) + HAVAL_ROTR32(T0, 11) + M[31];

    // PASS 2 (F2, with round constants).
    T7 = HAVAL_ROTR32(HAVAL_F2(T3, T5, T2, T0, T1, T6, T4), 7) + HAVAL_ROTR32(T7, 11) + M[ 5] + 0x452821E6u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T2, T4, T1, T7, T0, T5, T3), 7) + HAVAL_ROTR32(T6, 11) + M[14] + 0x38D01377u;
    T5 = HAVAL_ROTR32(HAVAL_F2(T1, T3, T0, T6, T7, T4, T2), 7) + HAVAL_ROTR32(T5, 11) + M[26] + 0xBE5466CFu;
    T4 = HAVAL_ROTR32(HAVAL_F2(T0, T2, T7, T5, T6, T3, T1), 7) + HAVAL_ROTR32(T4, 11) + M[18] + 0x34E90C6Cu;
    T3 = HAVAL_ROTR32(HAVAL_F2(T7, T1, T6, T4, T5, T2, T0), 7) + HAVAL_ROTR32(T3, 11) + M[11] + 0xC0AC29B7u;
    T2 = HAVAL_ROTR32(HAVAL_F2(T6, T0, T5, T3, T4, T1, T7), 7) + HAVAL_ROTR32(T2, 11) + M[28] + 0xC97C50DDu;
    T1 = HAVAL_ROTR32(HAVAL_F2(T5, T7, T4, T2, T3, T0, T6), 7) + HAVAL_ROTR32(T1, 11) + M[ 7] + 0x3F84D5B5u;
    T0 = HAVAL_ROTR32(HAVAL_F2(T4, T6, T3, T1, T2, T7, T5), 7) + HAVAL_ROTR32(T0, 11) + M[16] + 0xB5470917u;

    T7 = HAVAL_ROTR32(HAVAL_F2(T3, T5, T2, T0, T1, T6, T4), 7) + HAVAL_ROTR32(T7, 11) + M[ 0] + 0x9216D5D9u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T2, T4, T1, T7, T0, T5, T3), 7) + HAVAL_ROTR32(T6, 11) + M[23] + 0x8979FB1Bu;
    T5 = HAVAL_ROTR32(HAVAL_F2(T1, T3, T0, T6, T7, T4, T2), 7) + HAVAL_ROTR32(T5, 11) + M[20] + 0xD1310BA6u;
    T4 = HAVAL_ROTR32(HAVAL_F2(T0, T2, T7, T5, T6, T3, T1), 7) + HAVAL_ROTR32(T4, 11) + M[22] + 0x98DFB5ACu;
    T3 = HAVAL_ROTR32(HAVAL_F2(T7, T1, T6, T4, T5, T2, T0), 7) + HAVAL_ROTR32(T3, 11) + M[ 1] + 0x2FFD72DBu;
    T2 = HAVAL_ROTR32(HAVAL_F2(T6, T0, T5, T3, T4, T1, T7), 7) + HAVAL_ROTR32(T2, 11) + M[10] + 0xD01ADFB7u;
    T1 = HAVAL_ROTR32(HAVAL_F2(T5, T7, T4, T2, T3, T0, T6), 7) + HAVAL_ROTR32(T1, 11) + M[ 4] + 0xB8E1AFEDu;
    T0 = HAVAL_ROTR32(HAVAL_F2(T4, T6, T3, T1, T2, T7, T5), 7) + HAVAL_ROTR32(T0, 11) + M[ 8] + 0x6A267E96u;

    T7 = HAVAL_ROTR32(HAVAL_F2(T3, T5, T2, T0, T1, T6, T4), 7) + HAVAL_ROTR32(T7, 11) + M[30] + 0xBA7C9045u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T2, T4, T1, T7, T0, T5, T3), 7) + HAVAL_ROTR32(T6, 11) + M[ 3] + 0xF12C7F99u;
    T5 = HAVAL_ROTR32(HAVAL_F2(T1, T3, T0, T6, T7, T4, T2), 7) + HAVAL_ROTR32(T5, 11) + M[21] + 0x24A19947u;
    T4 = HAVAL_ROTR32(HAVAL_F2(T0, T2, T7, T5, T6, T3, T1), 7) + HAVAL_ROTR32(T4, 11) + M[ 9] + 0xB3916CF7u;
    T3 = HAVAL_ROTR32(HAVAL_F2(T7, T1, T6, T4, T5, T2, T0), 7) + HAVAL_ROTR32(T3, 11) + M[17] + 0x0801F2E2u;
    T2 = HAVAL_ROTR32(HAVAL_F2(T6, T0, T5, T3, T4, T1, T7), 7) + HAVAL_ROTR32(T2, 11) + M[24] + 0x858EFC16u;
    T1 = HAVAL_ROTR32(HAVAL_F2(T5, T7, T4, T2, T3, T0, T6), 7) + HAVAL_ROTR32(T1, 11) + M[29] + 0x636920D8u;
    T0 = HAVAL_ROTR32(HAVAL_F2(T4, T6, T3, T1, T2, T7, T5), 7) + HAVAL_ROTR32(T0, 11) + M[ 6] + 0x71574E69u;

    T7 = HAVAL_ROTR32(HAVAL_F2(T3, T5, T2, T0, T1, T6, T4), 7) + HAVAL_ROTR32(T7, 11) + M[19] + 0xA458FEA3u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T2, T4, T1, T7, T0, T5, T3), 7) + HAVAL_ROTR32(T6, 11) + M[12] + 0xF4933D7Eu;
    T5 = HAVAL_ROTR32(HAVAL_F2(T1, T3, T0, T6, T7, T4, T2), 7) + HAVAL_ROTR32(T5, 11) + M[15] + 0x0D95748Fu;
    T4 = HAVAL_ROTR32(HAVAL_F2(T0, T2, T7, T5, T6, T3, T1), 7) + HAVAL_ROTR32(T4, 11) + M[13] + 0x728EB658u;
    T3 = HAVAL_ROTR32(HAVAL_F2(T7, T1, T6, T4, T5, T2, T0), 7) + HAVAL_ROTR32(T3, 11) + M[ 2] + 0x718BCD58u;
    T2 = HAVAL_ROTR32(HAVAL_F2(T6, T0, T5, T3, T4, T1, T7), 7) + HAVAL_ROTR32(T2, 11) + M[25] + 0x82154AEEu;
    T1 = HAVAL_ROTR32(HAVAL_F2(T5, T7, T4, T2, T3, T0, T6), 7) + HAVAL_ROTR32(T1, 11) + M[31] + 0x7B54A41Du;
    T0 = HAVAL_ROTR32(HAVAL_F2(T4, T6, T3, T1, T2, T7, T5), 7) + HAVAL_ROTR32(T0, 11) + M[27] + 0xC25A59B5u;

    // PASS 3 (F3, with round constants).
    T7 = HAVAL_ROTR32(HAVAL_F3(T1, T4, T3, T6, T0, T2, T5), 7) + HAVAL_ROTR32(T7, 11) + M[19] + 0x9C30D539u;
    T6 = HAVAL_ROTR32(HAVAL_F3(T0, T3, T2, T5, T7, T1, T4), 7) + HAVAL_ROTR32(T6, 11) + M[ 9] + 0x2AF26013u;
    T5 = HAVAL_ROTR32(HAVAL_F3(T7, T2, T1, T4, T6, T0, T3), 7) + HAVAL_ROTR32(T5, 11) + M[ 4] + 0xC5D1B023u;
    T4 = HAVAL_ROTR32(HAVAL_F3(T6, T1, T0, T3, T5, T7, T2), 7) + HAVAL_ROTR32(T4, 11) + M[20] + 0x286085F0u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T5, T0, T7, T2, T4, T6, T1), 7) + HAVAL_ROTR32(T3, 11) + M[28] + 0xCA417918u;
    T2 = HAVAL_ROTR32(HAVAL_F3(T4, T7, T6, T1, T3, T5, T0), 7) + HAVAL_ROTR32(T2, 11) + M[17] + 0xB8DB38EFu;
    T1 = HAVAL_ROTR32(HAVAL_F3(T3, T6, T5, T0, T2, T4, T7), 7) + HAVAL_ROTR32(T1, 11) + M[ 8] + 0x8E79DCB0u;
    T0 = HAVAL_ROTR32(HAVAL_F3(T2, T5, T4, T7, T1, T3, T6), 7) + HAVAL_ROTR32(T0, 11) + M[22] + 0x603A180Eu;

    T7 = HAVAL_ROTR32(HAVAL_F3(T1, T4, T3, T6, T0, T2, T5), 7) + HAVAL_ROTR32(T7, 11) + M[29] + 0x6C9E0E8Bu;
    T6 = HAVAL_ROTR32(HAVAL_F3(T0, T3, T2, T5, T7, T1, T4), 7) + HAVAL_ROTR32(T6, 11) + M[14] + 0xB01E8A3Eu;
    T5 = HAVAL_ROTR32(HAVAL_F3(T7, T2, T1, T4, T6, T0, T3), 7) + HAVAL_ROTR32(T5, 11) + M[25] + 0xD71577C1u;
    T4 = HAVAL_ROTR32(HAVAL_F3(T6, T1, T0, T3, T5, T7, T2), 7) + HAVAL_ROTR32(T4, 11) + M[12] + 0xBD314B27u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T5, T0, T7, T2, T4, T6, T1), 7) + HAVAL_ROTR32(T3, 11) + M[24] + 0x78AF2FDAu;
    T2 = HAVAL_ROTR32(HAVAL_F3(T4, T7, T6, T1, T3, T5, T0), 7) + HAVAL_ROTR32(T2, 11) + M[30] + 0x55605C60u;
    T1 = HAVAL_ROTR32(HAVAL_F3(T3, T6, T5, T0, T2, T4, T7), 7) + HAVAL_ROTR32(T1, 11) + M[16] + 0xE65525F3u;
    T0 = HAVAL_ROTR32(HAVAL_F3(T2, T5, T4, T7, T1, T3, T6), 7) + HAVAL_ROTR32(T0, 11) + M[26] + 0xAA55AB94u;

    T7 = HAVAL_ROTR32(HAVAL_F3(T1, T4, T3, T6, T0, T2, T5), 7) + HAVAL_ROTR32(T7, 11) + M[31] + 0x57489862u;
    T6 = HAVAL_ROTR32(HAVAL_F3(T0, T3, T2, T5, T7, T1, T4), 7) + HAVAL_ROTR32(T6, 11) + M[15] + 0x63E81440u;
    T5 = HAVAL_ROTR32(HAVAL_F3(T7, T2, T1, T4, T6, T0, T3), 7) + HAVAL_ROTR32(T5, 11) + M[ 7] + 0x55CA396Au;
    T4 = HAVAL_ROTR32(HAVAL_F3(T6, T1, T0, T3, T5, T7, T2), 7) + HAVAL_ROTR32(T4, 11) + M[ 3] + 0x2AAB10B6u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T5, T0, T7, T2, T4, T6, T1), 7) + HAVAL_ROTR32(T3, 11) + M[ 1] + 0xB4CC5C34u;
    T2 = HAVAL_ROTR32(HAVAL_F3(T4, T7, T6, T1, T3, T5, T0), 7) + HAVAL_ROTR32(T2, 11) + M[ 0] + 0x1141E8CEu;
    T1 = HAVAL_ROTR32(HAVAL_F3(T3, T6, T5, T0, T2, T4, T7), 7) + HAVAL_ROTR32(T1, 11) + M[18] + 0xA15486AFu;
    T0 = HAVAL_ROTR32(HAVAL_F3(T2, T5, T4, T7, T1, T3, T6), 7) + HAVAL_ROTR32(T0, 11) + M[27] + 0x7C72E993u;

    T7 = HAVAL_ROTR32(HAVAL_F3(T1, T4, T3, T6, T0, T2, T5), 7) + HAVAL_ROTR32(T7, 11) + M[13] + 0xB3EE1411u;
    T6 = HAVAL_ROTR32(HAVAL_F3(T0, T3, T2, T5, T7, T1, T4), 7) + HAVAL_ROTR32(T6, 11) + M[ 6] + 0x636FBC2Au;
    T5 = HAVAL_ROTR32(HAVAL_F3(T7, T2, T1, T4, T6, T0, T3), 7) + HAVAL_ROTR32(T5, 11) + M[21] + 0x2BA9C55Du;
    T4 = HAVAL_ROTR32(HAVAL_F3(T6, T1, T0, T3, T5, T7, T2), 7) + HAVAL_ROTR32(T4, 11) + M[10] + 0x741831F6u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T5, T0, T7, T2, T4, T6, T1), 7) + HAVAL_ROTR32(T3, 11) + M[23] + 0xCE5C3E16u;
    T2 = HAVAL_ROTR32(HAVAL_F3(T4, T7, T6, T1, T3, T5, T0), 7) + HAVAL_ROTR32(T2, 11) + M[11] + 0x9B87931Eu;
    T1 = HAVAL_ROTR32(HAVAL_F3(T3, T6, T5, T0, T2, T4, T7), 7) + HAVAL_ROTR32(T1, 11) + M[ 5] + 0xAFD6BA33u;
    T0 = HAVAL_ROTR32(HAVAL_F3(T2, T5, T4, T7, T1, T3, T6), 7) + HAVAL_ROTR32(T0, 11) + M[ 2] + 0x6C24CF5Cu;

    // PASS 4 (F4, with round constants). Last 8 steps include the
    // feedforward state[i] += T[i] (donor lines 397-404).
    T7 = HAVAL_ROTR32(HAVAL_F4(T6, T4, T0, T5, T2, T1, T3), 7) + HAVAL_ROTR32(T7, 11) + M[24] + 0x7A325381u;
    T6 = HAVAL_ROTR32(HAVAL_F4(T5, T3, T7, T4, T1, T0, T2), 7) + HAVAL_ROTR32(T6, 11) + M[ 4] + 0x28958677u;
    T5 = HAVAL_ROTR32(HAVAL_F4(T4, T2, T6, T3, T0, T7, T1), 7) + HAVAL_ROTR32(T5, 11) + M[ 0] + 0x3B8F4898u;
    T4 = HAVAL_ROTR32(HAVAL_F4(T3, T1, T5, T2, T7, T6, T0), 7) + HAVAL_ROTR32(T4, 11) + M[14] + 0x6B4BB9AFu;
    T3 = HAVAL_ROTR32(HAVAL_F4(T2, T0, T4, T1, T6, T5, T7), 7) + HAVAL_ROTR32(T3, 11) + M[ 2] + 0xC4BFE81Bu;
    T2 = HAVAL_ROTR32(HAVAL_F4(T1, T7, T3, T0, T5, T4, T6), 7) + HAVAL_ROTR32(T2, 11) + M[ 7] + 0x66282193u;
    T1 = HAVAL_ROTR32(HAVAL_F4(T0, T6, T2, T7, T4, T3, T5), 7) + HAVAL_ROTR32(T1, 11) + M[28] + 0x61D809CCu;
    T0 = HAVAL_ROTR32(HAVAL_F4(T7, T5, T1, T6, T3, T2, T4), 7) + HAVAL_ROTR32(T0, 11) + M[23] + 0xFB21A991u;

    T7 = HAVAL_ROTR32(HAVAL_F4(T6, T4, T0, T5, T2, T1, T3), 7) + HAVAL_ROTR32(T7, 11) + M[26] + 0x487CAC60u;
    T6 = HAVAL_ROTR32(HAVAL_F4(T5, T3, T7, T4, T1, T0, T2), 7) + HAVAL_ROTR32(T6, 11) + M[ 6] + 0x5DEC8032u;
    T5 = HAVAL_ROTR32(HAVAL_F4(T4, T2, T6, T3, T0, T7, T1), 7) + HAVAL_ROTR32(T5, 11) + M[30] + 0xEF845D5Du;
    T4 = HAVAL_ROTR32(HAVAL_F4(T3, T1, T5, T2, T7, T6, T0), 7) + HAVAL_ROTR32(T4, 11) + M[20] + 0xE98575B1u;
    T3 = HAVAL_ROTR32(HAVAL_F4(T2, T0, T4, T1, T6, T5, T7), 7) + HAVAL_ROTR32(T3, 11) + M[18] + 0xDC262302u;
    T2 = HAVAL_ROTR32(HAVAL_F4(T1, T7, T3, T0, T5, T4, T6), 7) + HAVAL_ROTR32(T2, 11) + M[25] + 0xEB651B88u;
    T1 = HAVAL_ROTR32(HAVAL_F4(T0, T6, T2, T7, T4, T3, T5), 7) + HAVAL_ROTR32(T1, 11) + M[19] + 0x23893E81u;
    T0 = HAVAL_ROTR32(HAVAL_F4(T7, T5, T1, T6, T3, T2, T4), 7) + HAVAL_ROTR32(T0, 11) + M[ 3] + 0xD396ACC5u;

    T7 = HAVAL_ROTR32(HAVAL_F4(T6, T4, T0, T5, T2, T1, T3), 7) + HAVAL_ROTR32(T7, 11) + M[22] + 0x0F6D6FF3u;
    T6 = HAVAL_ROTR32(HAVAL_F4(T5, T3, T7, T4, T1, T0, T2), 7) + HAVAL_ROTR32(T6, 11) + M[11] + 0x83F44239u;
    T5 = HAVAL_ROTR32(HAVAL_F4(T4, T2, T6, T3, T0, T7, T1), 7) + HAVAL_ROTR32(T5, 11) + M[31] + 0x2E0B4482u;
    T4 = HAVAL_ROTR32(HAVAL_F4(T3, T1, T5, T2, T7, T6, T0), 7) + HAVAL_ROTR32(T4, 11) + M[21] + 0xA4842004u;
    T3 = HAVAL_ROTR32(HAVAL_F4(T2, T0, T4, T1, T6, T5, T7), 7) + HAVAL_ROTR32(T3, 11) + M[ 8] + 0x69C8F04Au;
    T2 = HAVAL_ROTR32(HAVAL_F4(T1, T7, T3, T0, T5, T4, T6), 7) + HAVAL_ROTR32(T2, 11) + M[27] + 0x9E1F9B5Eu;
    T1 = HAVAL_ROTR32(HAVAL_F4(T0, T6, T2, T7, T4, T3, T5), 7) + HAVAL_ROTR32(T1, 11) + M[12] + 0x21C66842u;
    T0 = HAVAL_ROTR32(HAVAL_F4(T7, T5, T1, T6, T3, T2, T4), 7) + HAVAL_ROTR32(T0, 11) + M[ 9] + 0xF6E96C9Au;

    // Final 8 steps: compute T[i] AND feedforward into state[i]
    // (donor lines 397-404: E[7] += T[7] = ...).
    T7 = HAVAL_ROTR32(HAVAL_F4(T6, T4, T0, T5, T2, T1, T3), 7) + HAVAL_ROTR32(T7, 11) + M[ 1] + 0x670C9C61u;
    state[7] = E7 + T7;
    T6 = HAVAL_ROTR32(HAVAL_F4(T5, T3, T7, T4, T1, T0, T2), 7) + HAVAL_ROTR32(T6, 11) + M[29] + 0xABD388F0u;
    state[6] = E6 + T6;
    T5 = HAVAL_ROTR32(HAVAL_F4(T4, T2, T6, T3, T0, T7, T1), 7) + HAVAL_ROTR32(T5, 11) + M[ 5] + 0x6A51A0D2u;
    state[5] = E5 + T5;
    T4 = HAVAL_ROTR32(HAVAL_F4(T3, T1, T5, T2, T7, T6, T0), 7) + HAVAL_ROTR32(T4, 11) + M[15] + 0xD8542F68u;
    state[4] = E4 + T4;
    T3 = HAVAL_ROTR32(HAVAL_F4(T2, T0, T4, T1, T6, T5, T7), 7) + HAVAL_ROTR32(T3, 11) + M[17] + 0x960FA728u;
    state[3] = E3 + T3;
    T2 = HAVAL_ROTR32(HAVAL_F4(T1, T7, T3, T0, T5, T4, T6), 7) + HAVAL_ROTR32(T2, 11) + M[10] + 0xAB5133A3u;
    state[2] = E2 + T2;
    T1 = HAVAL_ROTR32(HAVAL_F4(T0, T6, T2, T7, T4, T3, T5), 7) + HAVAL_ROTR32(T1, 11) + M[16] + 0x6EEF0B6Cu;
    state[1] = E1 + T1;
    T0 = HAVAL_ROTR32(HAVAL_F4(T7, T5, T1, T6, T3, T2, T4), 7) + HAVAL_ROTR32(T0, 11) + M[13] + 0x137A3BE4u;
    state[0] = E0 + T0;
}

// Phase 5b Tier 3 sub-phase 5b.3c.2 (2026-05-27): lift haval5_block from
// mhash-0.9.9.9/lib/haval.c havalTransform5 lines 412-615. Public-domain
// donor (Paulo S.L.M. Barreto 1998). 5-pass HAVAL compression.
//
// Same caller convention as haval3_block / haval4_block: state[0..7] is the
// 8-uint HAVAL chaining value (init to HAVAL_IV on first invocation);
// M[0..31] is the 128-byte block as 32 LE-packed uint32 words. Mutates state
// in place; the post-compression digest fold for 128/160/192/224 widths
// happens in the emit helper after the FINAL haval5_block call (256-bit has
// no fold).
//
// 5-pass structure: pass 1 F1 no constants; pass 2 F2 + 32 RC; pass 3 F3 +
// 32 RC; pass 4 F4 + 32 RC; pass 5 F5 + 32 RC (0xBA3BF050..0x409F60C4). 128
// round constants total for 5-pass (32 each in passes 2/3/4/5; pass 1 none).
//
// CRITICAL: the per-step F-function argument orderings and W[] word-index
// permutations in passes 1-4 of havalTransform5 DIFFER from BOTH
// havalTransform3 AND havalTransform4 (the message-word schedule and the
// rotor are pass-count specific across ALL passes). This is a verbatim
// transcription of havalTransform5, NOT a reuse of the haval3_block or
// haval4_block passes.
//
// R6 noinline per feedback_md5_block_noinline_pascal.md. R8 no nested
// block comments: donor /* */ stripped, line comments only.
//
// HAVAL_F1-F5, HAVAL_ROTR32, HAVAL_IV are shared with haval3_block above.
__attribute__((noinline)) void haval5_block(uint *state, const uint *M) {
    uint T0, T1, T2, T3, T4, T5, T6, T7;
    uint E0, E1, E2, E3, E4, E5, E6, E7;

    // Snapshot input state for feedforward at end of pass 5.
    E0 = state[0]; E1 = state[1]; E2 = state[2]; E3 = state[3];
    E4 = state[4]; E5 = state[5]; E6 = state[6]; E7 = state[7];

    // PASS 1 (F1, no round constants).
    T7 = HAVAL_ROTR32(HAVAL_F1(E3, E4, E1, E0, E5, E2, E6), 7) + HAVAL_ROTR32(E7, 11) + M[ 0];
    T6 = HAVAL_ROTR32(HAVAL_F1(E2, E3, E0, T7, E4, E1, E5), 7) + HAVAL_ROTR32(E6, 11) + M[ 1];
    T5 = HAVAL_ROTR32(HAVAL_F1(E1, E2, T7, T6, E3, E0, E4), 7) + HAVAL_ROTR32(E5, 11) + M[ 2];
    T4 = HAVAL_ROTR32(HAVAL_F1(E0, E1, T6, T5, E2, T7, E3), 7) + HAVAL_ROTR32(E4, 11) + M[ 3];
    T3 = HAVAL_ROTR32(HAVAL_F1(T7, E0, T5, T4, E1, T6, E2), 7) + HAVAL_ROTR32(E3, 11) + M[ 4];
    T2 = HAVAL_ROTR32(HAVAL_F1(T6, T7, T4, T3, E0, T5, E1), 7) + HAVAL_ROTR32(E2, 11) + M[ 5];
    T1 = HAVAL_ROTR32(HAVAL_F1(T5, T6, T3, T2, T7, T4, E0), 7) + HAVAL_ROTR32(E1, 11) + M[ 6];
    T0 = HAVAL_ROTR32(HAVAL_F1(T4, T5, T2, T1, T6, T3, T7), 7) + HAVAL_ROTR32(E0, 11) + M[ 7];

    T7 = HAVAL_ROTR32(HAVAL_F1(T3, T4, T1, T0, T5, T2, T6), 7) + HAVAL_ROTR32(T7, 11) + M[ 8];
    T6 = HAVAL_ROTR32(HAVAL_F1(T2, T3, T0, T7, T4, T1, T5), 7) + HAVAL_ROTR32(T6, 11) + M[ 9];
    T5 = HAVAL_ROTR32(HAVAL_F1(T1, T2, T7, T6, T3, T0, T4), 7) + HAVAL_ROTR32(T5, 11) + M[10];
    T4 = HAVAL_ROTR32(HAVAL_F1(T0, T1, T6, T5, T2, T7, T3), 7) + HAVAL_ROTR32(T4, 11) + M[11];
    T3 = HAVAL_ROTR32(HAVAL_F1(T7, T0, T5, T4, T1, T6, T2), 7) + HAVAL_ROTR32(T3, 11) + M[12];
    T2 = HAVAL_ROTR32(HAVAL_F1(T6, T7, T4, T3, T0, T5, T1), 7) + HAVAL_ROTR32(T2, 11) + M[13];
    T1 = HAVAL_ROTR32(HAVAL_F1(T5, T6, T3, T2, T7, T4, T0), 7) + HAVAL_ROTR32(T1, 11) + M[14];
    T0 = HAVAL_ROTR32(HAVAL_F1(T4, T5, T2, T1, T6, T3, T7), 7) + HAVAL_ROTR32(T0, 11) + M[15];

    T7 = HAVAL_ROTR32(HAVAL_F1(T3, T4, T1, T0, T5, T2, T6), 7) + HAVAL_ROTR32(T7, 11) + M[16];
    T6 = HAVAL_ROTR32(HAVAL_F1(T2, T3, T0, T7, T4, T1, T5), 7) + HAVAL_ROTR32(T6, 11) + M[17];
    T5 = HAVAL_ROTR32(HAVAL_F1(T1, T2, T7, T6, T3, T0, T4), 7) + HAVAL_ROTR32(T5, 11) + M[18];
    T4 = HAVAL_ROTR32(HAVAL_F1(T0, T1, T6, T5, T2, T7, T3), 7) + HAVAL_ROTR32(T4, 11) + M[19];
    T3 = HAVAL_ROTR32(HAVAL_F1(T7, T0, T5, T4, T1, T6, T2), 7) + HAVAL_ROTR32(T3, 11) + M[20];
    T2 = HAVAL_ROTR32(HAVAL_F1(T6, T7, T4, T3, T0, T5, T1), 7) + HAVAL_ROTR32(T2, 11) + M[21];
    T1 = HAVAL_ROTR32(HAVAL_F1(T5, T6, T3, T2, T7, T4, T0), 7) + HAVAL_ROTR32(T1, 11) + M[22];
    T0 = HAVAL_ROTR32(HAVAL_F1(T4, T5, T2, T1, T6, T3, T7), 7) + HAVAL_ROTR32(T0, 11) + M[23];

    T7 = HAVAL_ROTR32(HAVAL_F1(T3, T4, T1, T0, T5, T2, T6), 7) + HAVAL_ROTR32(T7, 11) + M[24];
    T6 = HAVAL_ROTR32(HAVAL_F1(T2, T3, T0, T7, T4, T1, T5), 7) + HAVAL_ROTR32(T6, 11) + M[25];
    T5 = HAVAL_ROTR32(HAVAL_F1(T1, T2, T7, T6, T3, T0, T4), 7) + HAVAL_ROTR32(T5, 11) + M[26];
    T4 = HAVAL_ROTR32(HAVAL_F1(T0, T1, T6, T5, T2, T7, T3), 7) + HAVAL_ROTR32(T4, 11) + M[27];
    T3 = HAVAL_ROTR32(HAVAL_F1(T7, T0, T5, T4, T1, T6, T2), 7) + HAVAL_ROTR32(T3, 11) + M[28];
    T2 = HAVAL_ROTR32(HAVAL_F1(T6, T7, T4, T3, T0, T5, T1), 7) + HAVAL_ROTR32(T2, 11) + M[29];
    T1 = HAVAL_ROTR32(HAVAL_F1(T5, T6, T3, T2, T7, T4, T0), 7) + HAVAL_ROTR32(T1, 11) + M[30];
    T0 = HAVAL_ROTR32(HAVAL_F1(T4, T5, T2, T1, T6, T3, T7), 7) + HAVAL_ROTR32(T0, 11) + M[31];

    // PASS 2 (F2, with round constants).
    T7 = HAVAL_ROTR32(HAVAL_F2(T6, T2, T1, T0, T3, T4, T5), 7) + HAVAL_ROTR32(T7, 11) + M[ 5] + 0x452821E6u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T5, T1, T0, T7, T2, T3, T4), 7) + HAVAL_ROTR32(T6, 11) + M[14] + 0x38D01377u;
    T5 = HAVAL_ROTR32(HAVAL_F2(T4, T0, T7, T6, T1, T2, T3), 7) + HAVAL_ROTR32(T5, 11) + M[26] + 0xBE5466CFu;
    T4 = HAVAL_ROTR32(HAVAL_F2(T3, T7, T6, T5, T0, T1, T2), 7) + HAVAL_ROTR32(T4, 11) + M[18] + 0x34E90C6Cu;
    T3 = HAVAL_ROTR32(HAVAL_F2(T2, T6, T5, T4, T7, T0, T1), 7) + HAVAL_ROTR32(T3, 11) + M[11] + 0xC0AC29B7u;
    T2 = HAVAL_ROTR32(HAVAL_F2(T1, T5, T4, T3, T6, T7, T0), 7) + HAVAL_ROTR32(T2, 11) + M[28] + 0xC97C50DDu;
    T1 = HAVAL_ROTR32(HAVAL_F2(T0, T4, T3, T2, T5, T6, T7), 7) + HAVAL_ROTR32(T1, 11) + M[ 7] + 0x3F84D5B5u;
    T0 = HAVAL_ROTR32(HAVAL_F2(T7, T3, T2, T1, T4, T5, T6), 7) + HAVAL_ROTR32(T0, 11) + M[16] + 0xB5470917u;

    T7 = HAVAL_ROTR32(HAVAL_F2(T6, T2, T1, T0, T3, T4, T5), 7) + HAVAL_ROTR32(T7, 11) + M[ 0] + 0x9216D5D9u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T5, T1, T0, T7, T2, T3, T4), 7) + HAVAL_ROTR32(T6, 11) + M[23] + 0x8979FB1Bu;
    T5 = HAVAL_ROTR32(HAVAL_F2(T4, T0, T7, T6, T1, T2, T3), 7) + HAVAL_ROTR32(T5, 11) + M[20] + 0xD1310BA6u;
    T4 = HAVAL_ROTR32(HAVAL_F2(T3, T7, T6, T5, T0, T1, T2), 7) + HAVAL_ROTR32(T4, 11) + M[22] + 0x98DFB5ACu;
    T3 = HAVAL_ROTR32(HAVAL_F2(T2, T6, T5, T4, T7, T0, T1), 7) + HAVAL_ROTR32(T3, 11) + M[ 1] + 0x2FFD72DBu;
    T2 = HAVAL_ROTR32(HAVAL_F2(T1, T5, T4, T3, T6, T7, T0), 7) + HAVAL_ROTR32(T2, 11) + M[10] + 0xD01ADFB7u;
    T1 = HAVAL_ROTR32(HAVAL_F2(T0, T4, T3, T2, T5, T6, T7), 7) + HAVAL_ROTR32(T1, 11) + M[ 4] + 0xB8E1AFEDu;
    T0 = HAVAL_ROTR32(HAVAL_F2(T7, T3, T2, T1, T4, T5, T6), 7) + HAVAL_ROTR32(T0, 11) + M[ 8] + 0x6A267E96u;

    T7 = HAVAL_ROTR32(HAVAL_F2(T6, T2, T1, T0, T3, T4, T5), 7) + HAVAL_ROTR32(T7, 11) + M[30] + 0xBA7C9045u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T5, T1, T0, T7, T2, T3, T4), 7) + HAVAL_ROTR32(T6, 11) + M[ 3] + 0xF12C7F99u;
    T5 = HAVAL_ROTR32(HAVAL_F2(T4, T0, T7, T6, T1, T2, T3), 7) + HAVAL_ROTR32(T5, 11) + M[21] + 0x24A19947u;
    T4 = HAVAL_ROTR32(HAVAL_F2(T3, T7, T6, T5, T0, T1, T2), 7) + HAVAL_ROTR32(T4, 11) + M[ 9] + 0xB3916CF7u;
    T3 = HAVAL_ROTR32(HAVAL_F2(T2, T6, T5, T4, T7, T0, T1), 7) + HAVAL_ROTR32(T3, 11) + M[17] + 0x0801F2E2u;
    T2 = HAVAL_ROTR32(HAVAL_F2(T1, T5, T4, T3, T6, T7, T0), 7) + HAVAL_ROTR32(T2, 11) + M[24] + 0x858EFC16u;
    T1 = HAVAL_ROTR32(HAVAL_F2(T0, T4, T3, T2, T5, T6, T7), 7) + HAVAL_ROTR32(T1, 11) + M[29] + 0x636920D8u;
    T0 = HAVAL_ROTR32(HAVAL_F2(T7, T3, T2, T1, T4, T5, T6), 7) + HAVAL_ROTR32(T0, 11) + M[ 6] + 0x71574E69u;

    T7 = HAVAL_ROTR32(HAVAL_F2(T6, T2, T1, T0, T3, T4, T5), 7) + HAVAL_ROTR32(T7, 11) + M[19] + 0xA458FEA3u;
    T6 = HAVAL_ROTR32(HAVAL_F2(T5, T1, T0, T7, T2, T3, T4), 7) + HAVAL_ROTR32(T6, 11) + M[12] + 0xF4933D7Eu;
    T5 = HAVAL_ROTR32(HAVAL_F2(T4, T0, T7, T6, T1, T2, T3), 7) + HAVAL_ROTR32(T5, 11) + M[15] + 0x0D95748Fu;
    T4 = HAVAL_ROTR32(HAVAL_F2(T3, T7, T6, T5, T0, T1, T2), 7) + HAVAL_ROTR32(T4, 11) + M[13] + 0x728EB658u;
    T3 = HAVAL_ROTR32(HAVAL_F2(T2, T6, T5, T4, T7, T0, T1), 7) + HAVAL_ROTR32(T3, 11) + M[ 2] + 0x718BCD58u;
    T2 = HAVAL_ROTR32(HAVAL_F2(T1, T5, T4, T3, T6, T7, T0), 7) + HAVAL_ROTR32(T2, 11) + M[25] + 0x82154AEEu;
    T1 = HAVAL_ROTR32(HAVAL_F2(T0, T4, T3, T2, T5, T6, T7), 7) + HAVAL_ROTR32(T1, 11) + M[31] + 0x7B54A41Du;
    T0 = HAVAL_ROTR32(HAVAL_F2(T7, T3, T2, T1, T4, T5, T6), 7) + HAVAL_ROTR32(T0, 11) + M[27] + 0xC25A59B5u;

    // PASS 3 (F3, with round constants).
    T7 = HAVAL_ROTR32(HAVAL_F3(T2, T6, T0, T4, T3, T1, T5), 7) + HAVAL_ROTR32(T7, 11) + M[19] + 0x9C30D539u;
    T6 = HAVAL_ROTR32(HAVAL_F3(T1, T5, T7, T3, T2, T0, T4), 7) + HAVAL_ROTR32(T6, 11) + M[ 9] + 0x2AF26013u;
    T5 = HAVAL_ROTR32(HAVAL_F3(T0, T4, T6, T2, T1, T7, T3), 7) + HAVAL_ROTR32(T5, 11) + M[ 4] + 0xC5D1B023u;
    T4 = HAVAL_ROTR32(HAVAL_F3(T7, T3, T5, T1, T0, T6, T2), 7) + HAVAL_ROTR32(T4, 11) + M[20] + 0x286085F0u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T6, T2, T4, T0, T7, T5, T1), 7) + HAVAL_ROTR32(T3, 11) + M[28] + 0xCA417918u;
    T2 = HAVAL_ROTR32(HAVAL_F3(T5, T1, T3, T7, T6, T4, T0), 7) + HAVAL_ROTR32(T2, 11) + M[17] + 0xB8DB38EFu;
    T1 = HAVAL_ROTR32(HAVAL_F3(T4, T0, T2, T6, T5, T3, T7), 7) + HAVAL_ROTR32(T1, 11) + M[ 8] + 0x8E79DCB0u;
    T0 = HAVAL_ROTR32(HAVAL_F3(T3, T7, T1, T5, T4, T2, T6), 7) + HAVAL_ROTR32(T0, 11) + M[22] + 0x603A180Eu;

    T7 = HAVAL_ROTR32(HAVAL_F3(T2, T6, T0, T4, T3, T1, T5), 7) + HAVAL_ROTR32(T7, 11) + M[29] + 0x6C9E0E8Bu;
    T6 = HAVAL_ROTR32(HAVAL_F3(T1, T5, T7, T3, T2, T0, T4), 7) + HAVAL_ROTR32(T6, 11) + M[14] + 0xB01E8A3Eu;
    T5 = HAVAL_ROTR32(HAVAL_F3(T0, T4, T6, T2, T1, T7, T3), 7) + HAVAL_ROTR32(T5, 11) + M[25] + 0xD71577C1u;
    T4 = HAVAL_ROTR32(HAVAL_F3(T7, T3, T5, T1, T0, T6, T2), 7) + HAVAL_ROTR32(T4, 11) + M[12] + 0xBD314B27u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T6, T2, T4, T0, T7, T5, T1), 7) + HAVAL_ROTR32(T3, 11) + M[24] + 0x78AF2FDAu;
    T2 = HAVAL_ROTR32(HAVAL_F3(T5, T1, T3, T7, T6, T4, T0), 7) + HAVAL_ROTR32(T2, 11) + M[30] + 0x55605C60u;
    T1 = HAVAL_ROTR32(HAVAL_F3(T4, T0, T2, T6, T5, T3, T7), 7) + HAVAL_ROTR32(T1, 11) + M[16] + 0xE65525F3u;
    T0 = HAVAL_ROTR32(HAVAL_F3(T3, T7, T1, T5, T4, T2, T6), 7) + HAVAL_ROTR32(T0, 11) + M[26] + 0xAA55AB94u;

    T7 = HAVAL_ROTR32(HAVAL_F3(T2, T6, T0, T4, T3, T1, T5), 7) + HAVAL_ROTR32(T7, 11) + M[31] + 0x57489862u;
    T6 = HAVAL_ROTR32(HAVAL_F3(T1, T5, T7, T3, T2, T0, T4), 7) + HAVAL_ROTR32(T6, 11) + M[15] + 0x63E81440u;
    T5 = HAVAL_ROTR32(HAVAL_F3(T0, T4, T6, T2, T1, T7, T3), 7) + HAVAL_ROTR32(T5, 11) + M[ 7] + 0x55CA396Au;
    T4 = HAVAL_ROTR32(HAVAL_F3(T7, T3, T5, T1, T0, T6, T2), 7) + HAVAL_ROTR32(T4, 11) + M[ 3] + 0x2AAB10B6u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T6, T2, T4, T0, T7, T5, T1), 7) + HAVAL_ROTR32(T3, 11) + M[ 1] + 0xB4CC5C34u;
    T2 = HAVAL_ROTR32(HAVAL_F3(T5, T1, T3, T7, T6, T4, T0), 7) + HAVAL_ROTR32(T2, 11) + M[ 0] + 0x1141E8CEu;
    T1 = HAVAL_ROTR32(HAVAL_F3(T4, T0, T2, T6, T5, T3, T7), 7) + HAVAL_ROTR32(T1, 11) + M[18] + 0xA15486AFu;
    T0 = HAVAL_ROTR32(HAVAL_F3(T3, T7, T1, T5, T4, T2, T6), 7) + HAVAL_ROTR32(T0, 11) + M[27] + 0x7C72E993u;

    T7 = HAVAL_ROTR32(HAVAL_F3(T2, T6, T0, T4, T3, T1, T5), 7) + HAVAL_ROTR32(T7, 11) + M[13] + 0xB3EE1411u;
    T6 = HAVAL_ROTR32(HAVAL_F3(T1, T5, T7, T3, T2, T0, T4), 7) + HAVAL_ROTR32(T6, 11) + M[ 6] + 0x636FBC2Au;
    T5 = HAVAL_ROTR32(HAVAL_F3(T0, T4, T6, T2, T1, T7, T3), 7) + HAVAL_ROTR32(T5, 11) + M[21] + 0x2BA9C55Du;
    T4 = HAVAL_ROTR32(HAVAL_F3(T7, T3, T5, T1, T0, T6, T2), 7) + HAVAL_ROTR32(T4, 11) + M[10] + 0x741831F6u;
    T3 = HAVAL_ROTR32(HAVAL_F3(T6, T2, T4, T0, T7, T5, T1), 7) + HAVAL_ROTR32(T3, 11) + M[23] + 0xCE5C3E16u;
    T2 = HAVAL_ROTR32(HAVAL_F3(T5, T1, T3, T7, T6, T4, T0), 7) + HAVAL_ROTR32(T2, 11) + M[11] + 0x9B87931Eu;
    T1 = HAVAL_ROTR32(HAVAL_F3(T4, T0, T2, T6, T5, T3, T7), 7) + HAVAL_ROTR32(T1, 11) + M[ 5] + 0xAFD6BA33u;
    T0 = HAVAL_ROTR32(HAVAL_F3(T3, T7, T1, T5, T4, T2, T6), 7) + HAVAL_ROTR32(T0, 11) + M[ 2] + 0x6C24CF5Cu;

    // PASS 4 (F4, with round constants).
    T7 = HAVAL_ROTR32(HAVAL_F4(T1, T5, T3, T2, T0, T4, T6), 7) + HAVAL_ROTR32(T7, 11) + M[24] + 0x7A325381u;
    T6 = HAVAL_ROTR32(HAVAL_F4(T0, T4, T2, T1, T7, T3, T5), 7) + HAVAL_ROTR32(T6, 11) + M[ 4] + 0x28958677u;
    T5 = HAVAL_ROTR32(HAVAL_F4(T7, T3, T1, T0, T6, T2, T4), 7) + HAVAL_ROTR32(T5, 11) + M[ 0] + 0x3B8F4898u;
    T4 = HAVAL_ROTR32(HAVAL_F4(T6, T2, T0, T7, T5, T1, T3), 7) + HAVAL_ROTR32(T4, 11) + M[14] + 0x6B4BB9AFu;
    T3 = HAVAL_ROTR32(HAVAL_F4(T5, T1, T7, T6, T4, T0, T2), 7) + HAVAL_ROTR32(T3, 11) + M[ 2] + 0xC4BFE81Bu;
    T2 = HAVAL_ROTR32(HAVAL_F4(T4, T0, T6, T5, T3, T7, T1), 7) + HAVAL_ROTR32(T2, 11) + M[ 7] + 0x66282193u;
    T1 = HAVAL_ROTR32(HAVAL_F4(T3, T7, T5, T4, T2, T6, T0), 7) + HAVAL_ROTR32(T1, 11) + M[28] + 0x61D809CCu;
    T0 = HAVAL_ROTR32(HAVAL_F4(T2, T6, T4, T3, T1, T5, T7), 7) + HAVAL_ROTR32(T0, 11) + M[23] + 0xFB21A991u;

    T7 = HAVAL_ROTR32(HAVAL_F4(T1, T5, T3, T2, T0, T4, T6), 7) + HAVAL_ROTR32(T7, 11) + M[26] + 0x487CAC60u;
    T6 = HAVAL_ROTR32(HAVAL_F4(T0, T4, T2, T1, T7, T3, T5), 7) + HAVAL_ROTR32(T6, 11) + M[ 6] + 0x5DEC8032u;
    T5 = HAVAL_ROTR32(HAVAL_F4(T7, T3, T1, T0, T6, T2, T4), 7) + HAVAL_ROTR32(T5, 11) + M[30] + 0xEF845D5Du;
    T4 = HAVAL_ROTR32(HAVAL_F4(T6, T2, T0, T7, T5, T1, T3), 7) + HAVAL_ROTR32(T4, 11) + M[20] + 0xE98575B1u;
    T3 = HAVAL_ROTR32(HAVAL_F4(T5, T1, T7, T6, T4, T0, T2), 7) + HAVAL_ROTR32(T3, 11) + M[18] + 0xDC262302u;
    T2 = HAVAL_ROTR32(HAVAL_F4(T4, T0, T6, T5, T3, T7, T1), 7) + HAVAL_ROTR32(T2, 11) + M[25] + 0xEB651B88u;
    T1 = HAVAL_ROTR32(HAVAL_F4(T3, T7, T5, T4, T2, T6, T0), 7) + HAVAL_ROTR32(T1, 11) + M[19] + 0x23893E81u;
    T0 = HAVAL_ROTR32(HAVAL_F4(T2, T6, T4, T3, T1, T5, T7), 7) + HAVAL_ROTR32(T0, 11) + M[ 3] + 0xD396ACC5u;

    T7 = HAVAL_ROTR32(HAVAL_F4(T1, T5, T3, T2, T0, T4, T6), 7) + HAVAL_ROTR32(T7, 11) + M[22] + 0x0F6D6FF3u;
    T6 = HAVAL_ROTR32(HAVAL_F4(T0, T4, T2, T1, T7, T3, T5), 7) + HAVAL_ROTR32(T6, 11) + M[11] + 0x83F44239u;
    T5 = HAVAL_ROTR32(HAVAL_F4(T7, T3, T1, T0, T6, T2, T4), 7) + HAVAL_ROTR32(T5, 11) + M[31] + 0x2E0B4482u;
    T4 = HAVAL_ROTR32(HAVAL_F4(T6, T2, T0, T7, T5, T1, T3), 7) + HAVAL_ROTR32(T4, 11) + M[21] + 0xA4842004u;
    T3 = HAVAL_ROTR32(HAVAL_F4(T5, T1, T7, T6, T4, T0, T2), 7) + HAVAL_ROTR32(T3, 11) + M[ 8] + 0x69C8F04Au;
    T2 = HAVAL_ROTR32(HAVAL_F4(T4, T0, T6, T5, T3, T7, T1), 7) + HAVAL_ROTR32(T2, 11) + M[27] + 0x9E1F9B5Eu;
    T1 = HAVAL_ROTR32(HAVAL_F4(T3, T7, T5, T4, T2, T6, T0), 7) + HAVAL_ROTR32(T1, 11) + M[12] + 0x21C66842u;
    T0 = HAVAL_ROTR32(HAVAL_F4(T2, T6, T4, T3, T1, T5, T7), 7) + HAVAL_ROTR32(T0, 11) + M[ 9] + 0xF6E96C9Au;

    T7 = HAVAL_ROTR32(HAVAL_F4(T1, T5, T3, T2, T0, T4, T6), 7) + HAVAL_ROTR32(T7, 11) + M[ 1] + 0x670C9C61u;
    T6 = HAVAL_ROTR32(HAVAL_F4(T0, T4, T2, T1, T7, T3, T5), 7) + HAVAL_ROTR32(T6, 11) + M[29] + 0xABD388F0u;
    T5 = HAVAL_ROTR32(HAVAL_F4(T7, T3, T1, T0, T6, T2, T4), 7) + HAVAL_ROTR32(T5, 11) + M[ 5] + 0x6A51A0D2u;
    T4 = HAVAL_ROTR32(HAVAL_F4(T6, T2, T0, T7, T5, T1, T3), 7) + HAVAL_ROTR32(T4, 11) + M[15] + 0xD8542F68u;
    T3 = HAVAL_ROTR32(HAVAL_F4(T5, T1, T7, T6, T4, T0, T2), 7) + HAVAL_ROTR32(T3, 11) + M[17] + 0x960FA728u;
    T2 = HAVAL_ROTR32(HAVAL_F4(T4, T0, T6, T5, T3, T7, T1), 7) + HAVAL_ROTR32(T2, 11) + M[10] + 0xAB5133A3u;
    T1 = HAVAL_ROTR32(HAVAL_F4(T3, T7, T5, T4, T2, T6, T0), 7) + HAVAL_ROTR32(T1, 11) + M[16] + 0x6EEF0B6Cu;
    T0 = HAVAL_ROTR32(HAVAL_F4(T2, T6, T4, T3, T1, T5, T7), 7) + HAVAL_ROTR32(T0, 11) + M[13] + 0x137A3BE4u;

    // PASS 5 (F5, with round constants). Final 8 steps compute T[i] AND
    // feedforward into state[i] (donor lines 603-610: E[7] += T[7] = ...).
    T7 = HAVAL_ROTR32(HAVAL_F5(T2, T5, T0, T6, T4, T3, T1), 7) + HAVAL_ROTR32(T7, 11) + M[27] + 0xBA3BF050u;
    T6 = HAVAL_ROTR32(HAVAL_F5(T1, T4, T7, T5, T3, T2, T0), 7) + HAVAL_ROTR32(T6, 11) + M[ 3] + 0x7EFB2A98u;
    T5 = HAVAL_ROTR32(HAVAL_F5(T0, T3, T6, T4, T2, T1, T7), 7) + HAVAL_ROTR32(T5, 11) + M[21] + 0xA1F1651Du;
    T4 = HAVAL_ROTR32(HAVAL_F5(T7, T2, T5, T3, T1, T0, T6), 7) + HAVAL_ROTR32(T4, 11) + M[26] + 0x39AF0176u;
    T3 = HAVAL_ROTR32(HAVAL_F5(T6, T1, T4, T2, T0, T7, T5), 7) + HAVAL_ROTR32(T3, 11) + M[17] + 0x66CA593Eu;
    T2 = HAVAL_ROTR32(HAVAL_F5(T5, T0, T3, T1, T7, T6, T4), 7) + HAVAL_ROTR32(T2, 11) + M[11] + 0x82430E88u;
    T1 = HAVAL_ROTR32(HAVAL_F5(T4, T7, T2, T0, T6, T5, T3), 7) + HAVAL_ROTR32(T1, 11) + M[20] + 0x8CEE8619u;
    T0 = HAVAL_ROTR32(HAVAL_F5(T3, T6, T1, T7, T5, T4, T2), 7) + HAVAL_ROTR32(T0, 11) + M[29] + 0x456F9FB4u;

    T7 = HAVAL_ROTR32(HAVAL_F5(T2, T5, T0, T6, T4, T3, T1), 7) + HAVAL_ROTR32(T7, 11) + M[19] + 0x7D84A5C3u;
    T6 = HAVAL_ROTR32(HAVAL_F5(T1, T4, T7, T5, T3, T2, T0), 7) + HAVAL_ROTR32(T6, 11) + M[ 0] + 0x3B8B5EBEu;
    T5 = HAVAL_ROTR32(HAVAL_F5(T0, T3, T6, T4, T2, T1, T7), 7) + HAVAL_ROTR32(T5, 11) + M[12] + 0xE06F75D8u;
    T4 = HAVAL_ROTR32(HAVAL_F5(T7, T2, T5, T3, T1, T0, T6), 7) + HAVAL_ROTR32(T4, 11) + M[ 7] + 0x85C12073u;
    T3 = HAVAL_ROTR32(HAVAL_F5(T6, T1, T4, T2, T0, T7, T5), 7) + HAVAL_ROTR32(T3, 11) + M[13] + 0x401A449Fu;
    T2 = HAVAL_ROTR32(HAVAL_F5(T5, T0, T3, T1, T7, T6, T4), 7) + HAVAL_ROTR32(T2, 11) + M[ 8] + 0x56C16AA6u;
    T1 = HAVAL_ROTR32(HAVAL_F5(T4, T7, T2, T0, T6, T5, T3), 7) + HAVAL_ROTR32(T1, 11) + M[31] + 0x4ED3AA62u;
    T0 = HAVAL_ROTR32(HAVAL_F5(T3, T6, T1, T7, T5, T4, T2), 7) + HAVAL_ROTR32(T0, 11) + M[10] + 0x363F7706u;

    T7 = HAVAL_ROTR32(HAVAL_F5(T2, T5, T0, T6, T4, T3, T1), 7) + HAVAL_ROTR32(T7, 11) + M[ 5] + 0x1BFEDF72u;
    T6 = HAVAL_ROTR32(HAVAL_F5(T1, T4, T7, T5, T3, T2, T0), 7) + HAVAL_ROTR32(T6, 11) + M[ 9] + 0x429B023Du;
    T5 = HAVAL_ROTR32(HAVAL_F5(T0, T3, T6, T4, T2, T1, T7), 7) + HAVAL_ROTR32(T5, 11) + M[14] + 0x37D0D724u;
    T4 = HAVAL_ROTR32(HAVAL_F5(T7, T2, T5, T3, T1, T0, T6), 7) + HAVAL_ROTR32(T4, 11) + M[30] + 0xD00A1248u;
    T3 = HAVAL_ROTR32(HAVAL_F5(T6, T1, T4, T2, T0, T7, T5), 7) + HAVAL_ROTR32(T3, 11) + M[18] + 0xDB0FEAD3u;
    T2 = HAVAL_ROTR32(HAVAL_F5(T5, T0, T3, T1, T7, T6, T4), 7) + HAVAL_ROTR32(T2, 11) + M[ 6] + 0x49F1C09Bu;
    T1 = HAVAL_ROTR32(HAVAL_F5(T4, T7, T2, T0, T6, T5, T3), 7) + HAVAL_ROTR32(T1, 11) + M[28] + 0x075372C9u;
    T0 = HAVAL_ROTR32(HAVAL_F5(T3, T6, T1, T7, T5, T4, T2), 7) + HAVAL_ROTR32(T0, 11) + M[24] + 0x80991B7Bu;

    T7 = HAVAL_ROTR32(HAVAL_F5(T2, T5, T0, T6, T4, T3, T1), 7) + HAVAL_ROTR32(T7, 11) + M[ 2] + 0x25D479D8u;
    state[7] = E7 + T7;
    T6 = HAVAL_ROTR32(HAVAL_F5(T1, T4, T7, T5, T3, T2, T0), 7) + HAVAL_ROTR32(T6, 11) + M[23] + 0xF6E8DEF7u;
    state[6] = E6 + T6;
    T5 = HAVAL_ROTR32(HAVAL_F5(T0, T3, T6, T4, T2, T1, T7), 7) + HAVAL_ROTR32(T5, 11) + M[16] + 0xE3FE501Au;
    state[5] = E5 + T5;
    T4 = HAVAL_ROTR32(HAVAL_F5(T7, T2, T5, T3, T1, T0, T6), 7) + HAVAL_ROTR32(T4, 11) + M[22] + 0xB6794C3Bu;
    state[4] = E4 + T4;
    T3 = HAVAL_ROTR32(HAVAL_F5(T6, T1, T4, T2, T0, T7, T5), 7) + HAVAL_ROTR32(T3, 11) + M[ 4] + 0x976CE0BDu;
    state[3] = E3 + T3;
    T2 = HAVAL_ROTR32(HAVAL_F5(T5, T0, T3, T1, T7, T6, T4), 7) + HAVAL_ROTR32(T2, 11) + M[ 1] + 0x04C006BAu;
    state[2] = E2 + T2;
    T1 = HAVAL_ROTR32(HAVAL_F5(T4, T7, T2, T0, T6, T5, T3), 7) + HAVAL_ROTR32(T1, 11) + M[25] + 0xC1A94FB6u;
    state[1] = E1 + T1;
    T0 = HAVAL_ROTR32(HAVAL_F5(T3, T6, T1, T7, T5, T4, T2), 7) + HAVAL_ROTR32(T0, 11) + M[15] + 0x409F60C4u;
    state[0] = E0 + T0;
}


// Phase 5b Tier 4 sub-phase 5b.4a.1 (2026-05-27): lift snefru_block from
// RHash-master/librhash/snefru.c rhash_snefru_process_block (lines 765-841)
// + the 16 KB rhash_snefru_sbox table (snefru.c:50). librhash is the LIVE
// CPU oracle for e175/e177 (snefru.o in librhash.a, linked at Makefile:109).
//
// ONE PARAMETERISED block (D18.1.a): snefru_block(state, block, is256)
// handles BOTH Snefru-128 (is256=0) and Snefru-256 (is256=1). The donor's
// single rhash_snefru_process_block has the SAME 8-round S-box transform
// for both widths; only 3 sites differ on width and they collapse to
// compile-time `if (is256)` branches the JIT folds (is256 is a literal
// per emit, so the kernel stays fully unrolled per variant):
//   1. W[] fill: SNE256 loads hash[4..7] into W[4..7] then reads 8 message
//      words (32-byte data block); SNE128 loads only hash[0..3] then reads
//      12 message words (48-byte data block) -- the extra 4 words come
//      from the message for SNE128 (donor :779-791).
//   2. Final state XOR-back: SNE256 also writes hash[4..7] (donor :835-840).
//   3. Block size consumed: 48 bytes (SNE128) vs 32 bytes (SNE256) -- the
//      EMIT HELPER handles the per-width padding + length placement; this
//      block function just reads the bytes it needs via SNEFRU_BE32.
//
// Round count is FIXED at 8 (SNEFRU_NUMBER_OF_ROUNDS; the standard hardened
// 8-pass Snefru). There is NO configurable security/pass parameter.
//
// Schedule + state output are BIG-ENDIAN (donor be2me_32 on message load;
// be32_copy on state output). The message words are assembled here via
// SNEFRU_BE32 (BE byte order). The state output BE-vs-LE concern is handled
// in the emit helper per feedback_be_state_primitives_need_byteswap_in_-
// codegen.md: the emit helper byte-swaps the BE state words into the LE-uint
// frame the compact_fp probe expects.
//
// Constant memory budget: SNEFRU_SBOX = 4096 * 4 = 16 KB `__constant`.
// Cumulative post-Tier-3 (~26-27 KB) + 16 KB = ~42-43 KB of the 64 KB
// Pascal / Apple Silicon CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE budget; fits
// with ~21 KB headroom (the GOST 4 KB derived tables in 5b.4b bring it to
// ~46-47 KB, still within budget).
//
// R6 noinline per feedback_md5_block_noinline_pascal.md (Pascal register
// budget). R8 no nested block comments per feedback_no_nested_block_-
// comments_in_cl.md: donor /* */ blocks stripped; only // line comments.
__constant uint SNEFRU_SBOX[4096] = {
    0x64f9001bu, 0xfeddcdf6u, 0x7c8ff1e2u, 0x11d71514u, 0x8b8c18d3u, 0xdddf881eu,
    0x6eab5056u, 0x88ced8e1u, 0x49148959u, 0x69c56fd5u, 0xb7994f03u, 0x0fbcee3eu,
    0x3c264940u, 0x21557e58u, 0xe14b3fc2u, 0x2e5cf591u, 0xdceff8ceu, 0x092a1648u,
    0xbe812936u, 0xff7b0c6au, 0xd5251037u, 0xafa448f1u, 0x7dafc95au, 0x1ea69c3fu,
    0xa417abe7u, 0x5890e423u, 0xb0cb70c0u, 0xc85025f7u, 0x244d97e3u, 0x1ff3595fu,
    0xc4ec6396u, 0x59181e17u, 0xe635b477u, 0x354e7dbfu, 0x796f7753u, 0x66eb52ccu,
    0x77c3f995u, 0x32e3a927u, 0x80ccaed6u, 0x4e2be89du, 0x375bbd28u, 0xad1a3d05u,
    0x2b1b42b3u, 0x16c44c71u, 0x4d54bfa8u, 0xe57ddc7au, 0xec6d8144u, 0x5a71046bu,
    0xd8229650u, 0x87fc8f24u, 0xcbc60e09u, 0xb6390366u, 0xd9f76092u, 0xd393a70bu,
    0x1d31a08au, 0x9cd971c9u, 0x5c1ef445u, 0x86fab694u, 0xfdb44165u, 0x8eaafcbeu,
    0x4bcac6ebu, 0xfb7a94e5u, 0x5789d04eu, 0xfa13cf35u, 0x236b8da9u, 0x4133f000u,
    0x6224261cu, 0xf412f23bu, 0xe75e56a4u, 0x30022116u, 0xbaf17f1fu, 0xd09872f9u,
    0xc1a3699cu, 0xf1e802aau, 0x0dd145dcu, 0x4fdce093u, 0x8d8412f0u, 0x6cd0f376u,
    0x3de6b73du, 0x84ba737fu, 0xb43a30f2u, 0x44569f69u, 0x00e4eacau, 0xb58de3b0u,
    0x959113c8u, 0xd62efee9u, 0x90861f83u, 0xced69874u, 0x2f793ceeu, 0xe8571c30u,
    0x483665d1u, 0xab07b031u, 0x914c844fu, 0x15bf3be8u, 0x2c3f2a9au, 0x9eb95fd4u,
    0x92e7472du, 0x2297cc5bu, 0xee5f2782u, 0x5377b562u, 0xdb8ebbcfu, 0xf961deddu,
    0xc59b5c60u, 0x1bd3910du, 0x26d206adu, 0xb28514d8u, 0x5ecf6b52u, 0x7fea78bbu,
    0x504879acu, 0xed34a884u, 0x36e51d3cu, 0x1753741du, 0x8c47caedu, 0x9d0a40efu,
    0x3145e221u, 0xda27eb70u, 0xdf730ba3u, 0x183c8789u, 0x739ac0a6u, 0x9a58dfc6u,
    0x54b134c1u, 0xac3e242eu, 0xcc493902u, 0x7b2dda99u, 0x8f15bc01u, 0x29fd38c7u,
    0x27d5318fu, 0x604aaff5u, 0xf29c6818u, 0xc38aa2ecu, 0x1019d4c3u, 0xa8fb936eu,
    0x20ed7b39u, 0x0b686119u, 0x89a0906fu, 0x1cc7829eu, 0x9952ef4bu, 0x850e9e8cu,
    0xcd063a90u, 0x67002f8eu, 0xcfac8cb7u, 0xeaa24b11u, 0x988b4e6cu, 0x46f066dfu,
    0xca7eec08u, 0xc7bba664u, 0x831d17bdu, 0x63f575e6u, 0x9764350eu, 0x47870d42u,
    0x026ca4a2u, 0x8167d587u, 0x61b6adabu, 0xaa6564d2u, 0x70da237bu, 0x25e1c74au,
    0xa1c901a0u, 0x0eb0a5dau, 0x7670f741u, 0x51c05aeau, 0x933dfa32u, 0x0759ff1au,
    0x56010ab8u, 0x5fdecb78u, 0x3f32edf8u, 0xaebedbb9u, 0x39f8326du, 0xd20858c5u,
    0x9b638be4u, 0xa572c80au, 0x28e0a19fu, 0x432099fcu, 0x3a37c3cdu, 0xbf95c585u,
    0xb392c12au, 0x6aa707d7u, 0x52f66a61u, 0x12d483b1u, 0x96435b5eu, 0x3e75802bu,
    0x3ba52b33u, 0xa99f51a5u, 0xbda1e157u, 0x78c2e70cu, 0xfcae7ce0u, 0xd1602267u,
    0x2affac4du, 0x4a510947u, 0x0ab2b83au, 0x7a04e579u, 0x340dfd80u, 0xb916e922u,
    0xe29d5e9bu, 0xf5624af4u, 0x4ca9d9afu, 0x6bbd2cfeu, 0xe3b7f620u, 0xc2746e07u,
    0x5b42b9b6u, 0xa06919bcu, 0xf0f2c40fu, 0x72217ab5u, 0x14c19df3u, 0xf3802daeu,
    0xe094beb4u, 0xa2101affu, 0x0529575du, 0x55cdb27cu, 0xa33bddb2u, 0x6528b37du,
    0x740c05dbu, 0xe96a62c4u, 0x40782846u, 0x6d30d706u, 0xbbf48e2cu, 0xbce2d3deu,
    0x049e37fau, 0x01b5e634u, 0x2d886d8du, 0x7e5a2e7eu, 0xd7412013u, 0x06e90f97u,
    0xe45d3ebau, 0xb8ad3386u, 0x13051b25u, 0x0c035354u, 0x71c89b75u, 0xc638fbd0u,
    0x197f11a1u, 0xef0f08fbu, 0xf8448651u, 0x38409563u, 0x452f4443u, 0x5d464d55u,
    0x03d8764cu, 0xb1b8d638u, 0xa70bba2fu, 0x94b3d210u, 0xeb6692a7u, 0xd409c2d9u,
    0x68838526u, 0xa6db8a15u, 0x751f6c98u, 0xde769a88u, 0xc9ee4668u, 0x1a82a373u,
    0x0896aa49u, 0x42233681u, 0xf62c55cbu, 0x9f1c5404u, 0xf74fb15cu, 0xc06e4312u,
    0x6ffe5d72u, 0x8aa8678bu, 0x337cd129u, 0x8211cefdu, 0x074a1d09u, 0x52a10e5au,
    0x9275a3f8u, 0x4b82506cu, 0x37df7e1bu, 0x4c78b3c5u, 0xcefab1dau, 0xf472267eu,
    0xb63045f6u, 0xd66a1fc0u, 0x400298e3u, 0x27e60c94u, 0x87d2f1b8u, 0xdf9e56ccu,
    0x45cd1803u, 0x1d35e098u, 0xcce7c736u, 0x03483bf1u, 0x1f7307d7u, 0xc6e8f948u,
    0xe613c111u, 0x3955c6ffu, 0x1170ed7cu, 0x8e95da41u, 0x99c31bf4u, 0xa4da8021u,
    0x7b5f94fbu, 0xdd0da51fu, 0x6562aa77u, 0x556bcb23u, 0xdb1bacc6u, 0x798040b9u,
    0xbfe5378fu, 0x731d55e6u, 0xdaa5bfeeu, 0x389bbc60u, 0x1b33fba4u, 0x9c567204u,
    0x36c26c68u, 0x77ee9d69u, 0x8aeb3e88u, 0x2d50b5ceu, 0x9579e790u, 0x42b13cfcu,
    0x33fbd32bu, 0xee0503a7u, 0xb5862824u, 0x15e41eadu, 0xc8412ef7u, 0x9d441275u,
    0x2fcec582u, 0x5ff483b7u, 0x8f3931dfu, 0x2e5d2a7bu, 0x49467bf9u, 0x0653dea9u,
    0x2684ce35u, 0x7e655e5cu, 0xf12771d8u, 0xbb15cc67u, 0xab097ca1u, 0x983dcf52u,
    0x10ddf026u, 0x21267f57u, 0x2c58f6b4u, 0x31043265u, 0x0bab8c01u, 0xd5492099u,
    0xacaae619u, 0x944ce54au, 0xf2d13d39u, 0xadd3fc32u, 0xcda08a40u, 0xe2b0d451u,
    0x9efe08aeu, 0xb9d50fd2u, 0xea5cd7fdu, 0xc9a749ddu, 0x13ea2253u, 0x832debaau,
    0x24be640fu, 0xe03e926au, 0x29e01cdeu, 0x8bf59f18u, 0x0f9d00b6u, 0xe1238b46u,
    0x1e7d8e34u, 0x93619adbu, 0x76b32f9fu, 0xbd972cecu, 0xe31fa976u, 0xa68fbb10u,
    0xfb3ba49du, 0x8587c41du, 0xa5add1d0u, 0xf3cf84bfu, 0xd4e11150u, 0xd9ffa6bcu,
    0xc3f6018cu, 0xaef10572u, 0x74a64b2fu, 0xe7dc9559u, 0x2aae35d5u, 0x5b6f587fu,
    0xa9e353feu, 0xca4fb674u, 0x04ba24a8u, 0xe5c6875fu, 0xdcbc6266u, 0x6bc5c03fu,
    0x661eef02u, 0xed740babu, 0x058e34e4u, 0xb7e946cfu, 0x88698125u, 0x72ec48edu,
    0xb11073a3u, 0xa13485ebu, 0xa2a2429cu, 0xfa407547u, 0x50b76713u, 0x5418c37du,
    0x96192da5u, 0x170bb04bu, 0x518a021eu, 0xb0ac13d1u, 0x0963fa2au, 0x4a6e10e1u,
    0x58472bdcu, 0xf7f8d962u, 0x979139eau, 0x8d856538u, 0xc0997042u, 0x48324d7au,
    0x447623cbu, 0x8cbbe364u, 0x6e0c6b0eu, 0xd36d63b0u, 0x3f244c84u, 0x3542c971u,
    0x2b228dc1u, 0xcb0325bbu, 0xf8c0d6e9u, 0xde11066bu, 0xa8649327u, 0xfc31f83eu,
    0x7dd80406u, 0xf916dd61u, 0xd89f79d3u, 0x615144c2u, 0xebb45d31u, 0x28002958u,
    0x56890a37u, 0xf05b3808u, 0x123ae844u, 0x86839e16u, 0x914b0d83u, 0xc506b43cu,
    0xcf3cba5eu, 0x7c60f5c9u, 0x22deb2a0u, 0x5d9c2715u, 0xc77ba0efu, 0x4f45360bu,
    0xc1017d8bu, 0xe45adc29u, 0xa759909bu, 0x412cd293u, 0xd7d796b1u, 0x00c8ff30u,
    0x23a34a80u, 0x4ec15c91u, 0x714e78b5u, 0x47b9e42eu, 0x78f3ea4du, 0x7f078f5bu,
    0x346c593au, 0xa3a87a1au, 0x9bcbfe12u, 0x3d439963u, 0xb2ef6d8eu, 0xb8d46028u,
    0x6c2fd5cau, 0x62675256u, 0x01f2a2f3u, 0xbc96ae0au, 0x709a8920u, 0xb4146e87u,
    0x6308b9e2u, 0x64bda7bau, 0xafed6892u, 0x6037f2a2u, 0xf52969e0u, 0x0adb43a6u,
    0x82811400u, 0x90d0bdf0u, 0x19c9549eu, 0x203f6a73u, 0x1accaf4fu, 0x89714e6du,
    0x164d4705u, 0x67665f07u, 0xec206170u, 0x0c2182b2u, 0xa02b9c81u, 0x53289722u,
    0xf6a97686u, 0x140e4179u, 0x9f778849u, 0x9a88e15du, 0x25cadb54u, 0xd157f36fu,
    0x32a421c3u, 0xb368e98au, 0x5a92cd0du, 0x757aa8d4u, 0xc20ac278u, 0x08b551c7u,
    0x849491e8u, 0x4dc75ad6u, 0x697c33beu, 0xbaf0ca33u, 0x46125b4eu, 0x59d677b3u,
    0x30d9c8f2u, 0xd0af860cu, 0x1c7fd0fau, 0xfe0ff72cu, 0x5c8d6f43u, 0x57fdec3bu,
    0x6ab6ad97u, 0xd22adf89u, 0x18171785u, 0x02bfe22du, 0x6db80917u, 0x80b216afu,
    0xe85e4f9au, 0x7a1c306eu, 0x6fc49bf5u, 0x3af7a11cu, 0x81e215e7u, 0x68363fcdu,
    0x3e9357c8u, 0xef52fd55u, 0x3b8bab4cu, 0x3c8cf495u, 0xbefceebdu, 0xfd25b714u,
    0xc498d83du, 0x0d2e1a8du, 0xe9f966acu, 0x0e387445u, 0x435419e5u, 0x5e7ebec4u,
    0xaa90b8d9u, 0xff1a3a96u, 0x4a8fe4e3u, 0xf27d99cdu, 0xd04a40cau, 0xcb5ff194u,
    0x3668275au, 0xff4816beu, 0xa78b394cu, 0x4c6be9dbu, 0x4eec38d2u, 0x4296ec80u,
    0xcdce96f8u, 0x888c2f38u, 0xe75508f5u, 0x7b916414u, 0x060aa14au, 0xa214f327u,
    0xbe608dafu, 0x1ebbdec2u, 0x61f98ce9u, 0xe92156feu, 0x4f22d7a3u, 0x3f76a8d9u,
    0x559a4b33u, 0x38ad2959u, 0xf3f17e9eu, 0x85e1ba91u, 0xe5eba6fbu, 0x73dcd48cu,
    0xf5c3ff78u, 0x481b6058u, 0x8a3297f7u, 0x8f1f3bf4u, 0x93785ab2u, 0x477a4a5bu,
    0x6334eb5du, 0x6d251b2eu, 0x74a9102du, 0x07e38ffau, 0x915c9c62u, 0xccc275eau,
    0x6be273ecu, 0x3ebddd70u, 0xd895796cu, 0xdc54a91bu, 0xc9afdf81u, 0x23633f73u,
    0x275119b4u, 0xb19f6b67u, 0x50756e22u, 0x2bb152e2u, 0x76ea46a2u, 0xa353e232u,
    0x2f596ad6u, 0x0b1edb0bu, 0x02d3d9a4u, 0x78b47843u, 0x64893e90u, 0x40f0caadu,
    0xf68d3ad7u, 0x46fd1707u, 0x1c9c67efu, 0xb5e086deu, 0x96ee6ca6u, 0x9aa34774u,
    0x1ba4f48au, 0x8d01abfdu, 0x183ee1f6u, 0x5ff8aa7au, 0x17e4faaeu, 0x303983b0u,
    0x6c08668bu, 0xd4ac4382u, 0xe6c5849fu, 0x92fefb53u, 0xc1cac4ceu, 0x43501388u,
    0x441118cfu, 0xec4fb308u, 0x53a08e86u, 0x9e0fe0c5u, 0xf91c1525u, 0xac45be05u,
    0xd7987cb5u, 0x49ba1487u, 0x57938940u, 0xd5877648u, 0xa958727fu, 0x58dfe3c3u,
    0xf436cf77u, 0x399e4d11u, 0xf0a5bfa9u, 0xef61a33bu, 0xa64cac60u, 0x04a8d0bau,
    0x030dd572u, 0xb83d320fu, 0xcab23045u, 0xe366f2f0u, 0x815d008du, 0xc897a43au,
    0x1d352df3u, 0xb9cc571du, 0x8bf38744u, 0x72209092u, 0xeba124ebu, 0xfb99ce5eu,
    0x3bb94293u, 0x28da549cu, 0xaab8a228u, 0xa4197785u, 0x33c70296u, 0x25f6259bu,
    0x5c85da21u, 0xdf15bdeeu, 0x15b7c7e8u, 0xe2abef75u, 0xfcc19bc1u, 0x417ff868u,
    0x14884434u, 0x62825179u, 0xc6d5c11cu, 0x0e4705dcu, 0x22700de0u, 0xd3d2af18u,
    0x9be822a0u, 0x35b669f1u, 0xc42bb55cu, 0x0a801252u, 0x115bf0fcu, 0x3cd7d856u,
    0xb43f5f9du, 0xc2306516u, 0xa1231c47u, 0xf149207eu, 0x5209a795u, 0x34b3ccd8u,
    0x67aefe54u, 0x2c83924eu, 0x6662cbacu, 0x5eedd161u, 0x84e681aau, 0x5d57d26bu,
    0xfa465cc4u, 0x7e3ac3a8u, 0xbf7c0cc6u, 0xe18a9aa1u, 0xc32f0a6fu, 0xb22cc00du,
    0x3d280369u, 0x994e554fu, 0x68f480d3u, 0xadcff5e6u, 0x3a8eb265u, 0x83269831u,
    0xbd568a09u, 0x4bc8ae6au, 0x69f56d2bu, 0x0f17eac8u, 0x772eb6c7u, 0x9f41343cu,
    0xab1d0742u, 0x826a6f50u, 0xfea2097cu, 0x1912c283u, 0xce185899u, 0xe4444839u,
    0x2d8635d5u, 0x65d0b1ffu, 0x865a7f17u, 0x326d9fb1u, 0x59e52820u, 0x0090ade1u,
    0x753c7149u, 0x9ddd8b98u, 0xa5a691dau, 0x0d0382bbu, 0x8904c930u, 0x086cb000u,
    0x6e69d3bdu, 0x24d4e7a7u, 0x05244fd0u, 0x101a5e0cu, 0x6a947dcbu, 0xe840f77bu,
    0x7d0c5003u, 0x7c370f1fu, 0x805245edu, 0xe05e3d3fu, 0x7906880eu, 0xbabfcd35u,
    0x1a7ec697u, 0x8c052324u, 0x0c6ec8dfu, 0xd129a589u, 0xc7a75b02u, 0x12d81de7u,
    0xd9be2a66u, 0x1f4263abu, 0xde73fdb6u, 0x2a00680au, 0x56649e36u, 0x3133ed55u,
    0x90fa0bf2u, 0x2910a02au, 0x949d9d46u, 0xa0d1dcddu, 0xcfc9b7d4u, 0xd2677be5u,
    0x95cb36b3u, 0x13cd9410u, 0xdbf73313u, 0xb7c6e8c0u, 0xf781414bu, 0x510b016du,
    0xb0de1157u, 0xd6b0f62cu, 0xbb074eccu, 0x7f1395b7u, 0xee792cf9u, 0xea6fd63eu,
    0x5bd6938eu, 0xaf02fc64u, 0xdab57ab8u, 0x8edb3784u, 0x8716318fu, 0x164d1a01u,
    0x26f26141u, 0xb372e6b9u, 0xf8fc2b06u, 0x7ac00e04u, 0x3727b89au, 0x97e9bca5u,
    0x9c2a742fu, 0xbc3b1f7du, 0x7165b471u, 0x609b4c29u, 0x20925351u, 0x5ae72112u,
    0x454be5d1u, 0xc0ffb95fu, 0xdd0ef919u, 0x6f2d70c9u, 0x0974c5bfu, 0x98aa6263u,
    0x01d91e4du, 0x2184bb6eu, 0x70c43c1eu, 0x4d435915u, 0xae7b8523u, 0xb6fb06bcu,
    0x5431ee76u, 0xfdbc5d26u, 0xed77493du, 0xc5712ee4u, 0xa8380437u, 0x2eef261au,
    0x5a79392bu, 0xb8af32c2u, 0x41f7720au, 0x833a61ecu, 0x13dfedacu, 0xc4990bc4u,
    0xdc0f54bcu, 0xfedd5e88u, 0x80da1881u, 0x4dea1afdu, 0xfd402cc6u, 0xae67cc7au,
    0xc5238525u, 0x8ea01254u, 0xb56b9bd5u, 0x862fbd6du, 0xac8575d3u, 0x6fba3714u,
    0xda7ebf46u, 0x59cd5238u, 0x8ac9dbfeu, 0x353729fcu, 0xe497d7f2u, 0xc3ab84e0u,
    0xf05a114bu, 0x7b887a75u, 0xedc603ddu, 0x5e6fe680u, 0x2c84b399u, 0x884eb1dau,
    0x1cb8c8bfu, 0xaa51098au, 0xc862231cu, 0x8bac2221u, 0x21b387e5u, 0x208a430du,
    0x2a3f0f8bu, 0xa5ff9cd2u, 0x6012a2eau, 0x147a9ee7u, 0xf62a501du, 0xb4b2e51au,
    0x3ef3484cu, 0xc0253c59u, 0x2b82b536u, 0x0aa9696bu, 0xbe0c109bu, 0xc70b7929u,
    0xce3e8a19u, 0x2f66950eu, 0x459f1c2cu, 0xe68fb93du, 0xa3c3ff3eu, 0x62b45c62u,
    0x300991cbu, 0x01914c57u, 0x7f7bc06au, 0x182831f5u, 0xe7b74bcau, 0xfa50f6d0u,
    0x523caa61u, 0xe3a7cf05u, 0xe9e41311u, 0x280a21d1u, 0x6a4297e1u, 0xf24dc67eu,
    0xfc3189e6u, 0xb72bf34fu, 0x4b1e67afu, 0x543402ceu, 0x79a59867u, 0x0648e02au,
    0x00a3ac17u, 0xc6208d35u, 0x6e7f5f76u, 0xa45bb4beu, 0xf168fa63u, 0x3f4125f3u,
    0xf311406fu, 0x02706565u, 0xbfe58022u, 0x0cfcfdd9u, 0x0735a7f7u, 0x8f049092u,
    0xd98edc27u, 0xf5c5d55cu, 0xe0f201dbu, 0x0dcafc9au, 0x7727fb79u, 0xaf43abf4u,
    0x26e938c1u, 0x401b26a6u, 0x900720fau, 0x2752d97bu, 0xcff1d1b3u, 0xa9d9e424u,
    0x42db99abu, 0x6cf8be5fu, 0xe82cebe3u, 0x3afb733bu, 0x6b734eb6u, 0x1036414au,
    0x975f667cu, 0x049d6377u, 0xba587c60u, 0xb1d10483u, 0xde1aefccu, 0x1129d055u,
    0x72051e91u, 0x6946d623u, 0xf9e86ea7u, 0x48768c00u, 0xb0166c93u, 0x9956bbf0u,
    0x1f1f6d84u, 0xfb15e18eu, 0x033b495du, 0x56e3362eu, 0x4f44c53cu, 0x747cba51u,
    0x89d37872u, 0x5d9c331bu, 0xd2ef9fa8u, 0x254917f8u, 0x1b106f47u, 0x37d75553u,
    0xb3f053b0u, 0x7dccd8efu, 0xd30eb802u, 0x5889f42du, 0x610206d7u, 0x1a7d34a1u,
    0x92d87dd8u, 0xe5f4a315u, 0xd1cf0e71u, 0xb22dfe45u, 0xb901e8ebu, 0x0fc0ce5eu,
    0x2efa60c9u, 0x2de74290u, 0x36d0c906u, 0x381c70e4u, 0x4c6da5b5u, 0x3d81a682u,
    0x7e381f34u, 0x396c4f52u, 0x95ad5901u, 0x1db50c5au, 0x29982e9eu, 0x1557689fu,
    0x3471ee42u, 0xd7e2f7c0u, 0x8795a1e2u, 0xbc324d8du, 0xe224c3c8u, 0x12837e39u,
    0xcdee3d74u, 0x7ad2143fu, 0x0e13d40cu, 0x78bd4a68u, 0xa2eb194du, 0xdb9451f9u,
    0x859b71dcu, 0x5c4f5b89u, 0xca14a8a4u, 0xef92f003u, 0x16741d98u, 0x33aa4444u,
    0x9e967fbbu, 0x092e3020u, 0xd86a35b8u, 0x8cc17b10u, 0xe1bf08aeu, 0x55693fc5u,
    0x7680ad13u, 0x1e6546e8u, 0x23b6e7b9u, 0xee77a4b2u, 0x08ed0533u, 0x44fd2895u,
    0xb6393b69u, 0x05d6cacfu, 0x9819b209u, 0xecbbb72fu, 0x9a75779cu, 0xeaec0749u,
    0x94a65aeeu, 0xbdf52dc3u, 0xd6a25d04u, 0x82008e4eu, 0xa6de160fu, 0x9b036afbu,
    0x228b3a66u, 0x5fb10a70u, 0xcc338b58u, 0x5378a9dfu, 0xc908bca9u, 0x4959e25bu,
    0x46909a97u, 0x66ae8f6eu, 0xdd0683e9u, 0x65f994b4u, 0x6426cda5u, 0xc24b8840u,
    0x32539da0u, 0x63175650u, 0xd0c815ffu, 0x50cbc41eu, 0xf7c774a3u, 0x31b0c231u,
    0x8d0d8116u, 0x24bef16cu, 0xd555d256u, 0xdf47ea8cu, 0x6d21eccdu, 0xa887a012u,
    0x84542aedu, 0xa7b9c1bdu, 0x914c1bb1u, 0xa0d5b67du, 0x438ce937u, 0x7030f873u,
    0x71f6b0c7u, 0x574576bau, 0xf8bc4541u, 0x9c61d348u, 0x1960579du, 0x17c4daadu,
    0x96a4cb0bu, 0xc193f2f6u, 0x756eafa2u, 0x7c1d2f94u, 0xf4fe2b43u, 0xcb86e33au,
    0xebd4c728u, 0x9d18ae64u, 0x9fe13e30u, 0x3ce0f5deu, 0xaba1f985u, 0xaddc2718u,
    0x68ce6278u, 0xd45e241fu, 0xa15c82b7u, 0x3b2293d4u, 0x739edd32u, 0x674a6bf1u,
    0x5b5d587fu, 0x4772deaau, 0x4a63968fu, 0x0be68686u, 0x513d6426u, 0x939a4787u,
    0xbba89296u, 0x4ec20007u, 0x818d0d08u, 0xff64dfd6u, 0xcb2297cbu, 0xdb48a144u,
    0xa16cbe4bu, 0xbbea1d6cu, 0x5af6b6b7u, 0x8a8110b6u, 0xf9236ef9u, 0xc98f83e6u,
    0x0f9c65b8u, 0x252d4a89u, 0xa497f068u, 0xa5d7ed2du, 0x94c22845u, 0x9da1c8c4u,
    0xe27c2e2eu, 0x6e8ba2b4u, 0xc3dd17fbu, 0x498cd482u, 0x0dfe6a9fu, 0xb0705829u,
    0x9a1e6dc1u, 0xf829717cu, 0x07bb8e3au, 0xda3c0b02u, 0x1af82fc7u, 0x73b70955u,
    0x7a04379cu, 0x5ee20a28u, 0x83712ae5u, 0xf4c47c6du, 0xdf72ba56u, 0xd794858du,
    0x8c0cf709u, 0x18f0f390u, 0xb6c69b35u, 0xbf2f01dbu, 0x2fa74dcau, 0xd0cd9127u,
    0xbde66cecu, 0x3deebd46u, 0x57c88fc3u, 0xcee1406fu, 0x0066385au, 0xf3c3444fu,
    0x3a79d5d5u, 0x75751eb9u, 0x3e7f8185u, 0x521c2605u, 0xe1aaab6eu, 0x38ebb80fu,
    0xbee7e904u, 0x61cb9647u, 0xea54904eu, 0x05ae00e4u, 0x2d7ac65fu, 0x087751a1u,
    0xdcd82915u, 0x0921ee16u, 0xdd86d33bu, 0xd6bd491au, 0x40fbadf0u, 0x4232cbd2u,
    0x33808d10u, 0x39098c42u, 0x193f3199u, 0x0bc1e47au, 0x4a82b149u, 0x02b65a8au,
    0x104cdc8eu, 0x24a8f52cu, 0x685c6077u, 0xc79f95c9u, 0x1d11fe50u, 0xc08dafcdu,
    0x7b1a9a03u, 0x1c1f11d8u, 0x84250e7fu, 0x979db248u, 0xebdc0501u, 0xb9553395u,
    0xe3c05ea8u, 0xb1e51c4cu, 0x13b0e681u, 0x3b407766u, 0x36db3087u, 0xee17c9fcu,
    0x6c53ecf2u, 0xadccc58fu, 0xc427660bu, 0xefd5867du, 0x9b6d54a5u, 0x6ff1aeffu,
    0x8e787952u, 0x9e2bffe0u, 0x8761d034u, 0xe00bdbadu, 0xae99a8d3u, 0xcc03f6e2u,
    0xfd0ed807u, 0x0e508ae3u, 0xb74182abu, 0x4349245du, 0xd120a465u, 0xb246a641u,
    0xaf3b7ab0u, 0x2a6488bbu, 0x4b3a0d1fu, 0xe7c7e58cu, 0x3faff2ebu, 0x90445ffdu,
    0xcf38c393u, 0x995d07e7u, 0xf24f1b36u, 0x356f6891u, 0x6d6ebcbeu, 0x8da9e262u,
    0x50fd520eu, 0x5bca9e1eu, 0x37472cf3u, 0x69075057u, 0x7ec5fdedu, 0x0cab892au,
    0xfb2412bau, 0x1728debfu, 0xa000a988u, 0xd843ce79u, 0x042e20ddu, 0x4fe8f853u,
    0x56659c3cu, 0x2739d119u, 0xa78a6120u, 0x80960375u, 0x70420611u, 0x85e09f78u,
    0xabd17e96u, 0x1b513eafu, 0x1e01eb63u, 0x26ad2133u, 0xa890c094u, 0x7613cf60u,
    0x817e781bu, 0xa39113d7u, 0xe957fa58u, 0x4131b99eu, 0x28b1efdau, 0x66acfba7u,
    0xff68944au, 0x77a44fd1u, 0x7f331522u, 0x59ffb3fau, 0xa6df935bu, 0xfa12d9dfu,
    0xc6bf6f3fu, 0x89520cf6u, 0x659edd6au, 0x544da739u, 0x8b052538u, 0x7c30ea21u,
    0xc2345525u, 0x15927fb2u, 0x144a436bu, 0xba107b8bu, 0x1219ac97u, 0x06730432u,
    0x31831ab3u, 0xc55a5c24u, 0xaa0fcd3eu, 0xe5606be8u, 0x5c88f19bu, 0x4c0841eeu,
    0x1fe37267u, 0x11f9c4f4u, 0x9f1b9daeu, 0x864e76d0u, 0xe637c731u, 0xd97d23a6u,
    0x32f53d5cu, 0xb8161980u, 0x93fa0f84u, 0xcaef0870u, 0x8874487eu, 0x98f2cc73u,
    0x645fb5c6u, 0xcd853659u, 0x2062470du, 0x16ede8e9u, 0x6b06dab5u, 0x78b43900u,
    0xfc95b786u, 0x5d8e7de1u, 0x465b5954u, 0xfe7ba014u, 0xf7d23f7bu, 0x92bc8b18u,
    0x03593592u, 0x55cef4f7u, 0x74b27317u, 0x79de1fc2u, 0xc8a0bfbdu, 0x229398ccu,
    0x62a602ceu, 0xbcb94661u, 0x5336d206u, 0xd2a375feu, 0x6a6ab483u, 0x4702a5a4u,
    0xa2e9d73du, 0x23a2e0f1u, 0x9189140au, 0x581d18dcu, 0xb39a922bu, 0x82356212u,
    0xd5f432a9u, 0xd356c2a3u, 0x5f765b4du, 0x450afcc8u, 0x4415e137u, 0xe8ecdfbcu,
    0xed0de3eau, 0x60d42b13u, 0xf13df971u, 0x71fc5da2u, 0xc1455340u, 0xf087742fu,
    0xf55e5751u, 0x67b3c1f8u, 0xac6b8774u, 0x7dcfaaacu, 0x95983bc0u, 0x489bb0b1u,
    0x2c184223u, 0x964b6726u, 0x2bd3271cu, 0x72266472u, 0xded64530u, 0x0a2aa343u,
    0xd4f716a0u, 0xb4dad6d9u, 0x2184345eu, 0x512c990cu, 0x29d92d08u, 0x2ebe709au,
    0x01144c69u, 0x34584b9du, 0xe4634ed6u, 0xecc963cfu, 0x3c6984aau, 0x4ed056efu,
    0x9ca56976u, 0x8f3e80d4u, 0xb5bae7c5u, 0x30b5caf5u, 0x63f33a64u, 0xa9e4bbdeu,
    0xf6b82298u, 0x4d673c1du, 0x4b4f1121u, 0xba183081u, 0xc784f41fu, 0xd17d0bacu,
    0x083d2267u, 0x37b1361eu, 0x3581ad05u, 0xfda2f6bcu, 0x1e892cddu, 0xb56d3c3au,
    0x32140e46u, 0x138d8aabu, 0xe14773d4u, 0x5b0e71dfu, 0x5d1fe055u, 0x3fb991d3u,
    0xf1f46c71u, 0xa325988cu, 0x10f66e80u, 0xb1006348u, 0x726a9f60u, 0x3b67f8bau,
    0x4e114ef4u, 0x05c52115u, 0x4c5ca11cu, 0x99e1efd8u, 0x471b83b3u, 0xcbf7e524u,
    0x43ad82f5u, 0x690ca93bu, 0xfaa61bb2u, 0x12a832b5u, 0xb734f943u, 0xbd22aea7u,
    0x88fec626u, 0x5e80c3e7u, 0xbe3eaf5eu, 0x44617652u, 0xa5724475u, 0xbb3b9695u,
    0x7f3fee8fu, 0x964e7debu, 0x518c052du, 0x2a0bbc2bu, 0xc2175f5cu, 0x9a7b3889u,
    0xa70d8d0cu, 0xeaccdd29u, 0xcccd6658u, 0x34bb25e6u, 0xb8391090u, 0xf651356fu,
    0x52987c9eu, 0x0c16c1cdu, 0x8e372d3cu, 0x2fc6ebbdu, 0x6e5da3e3u, 0xb0e27239u,
    0x5f685738u, 0x45411786u, 0x067f65f8u, 0x61778b40u, 0x81ab2e65u, 0x14c8f0f9u,
    0xa6b7b4ceu, 0x4036eaecu, 0xbf62b00au, 0xecfd5e02u, 0x045449a6u, 0xb20afd28u,
    0x2166d273u, 0x0d13a863u, 0x89508756u, 0xd51a7530u, 0x2d653f7au, 0x3cdbdbc3u,
    0x80c9df4fu, 0x3d5812d9u, 0x53fbb1f3u, 0xc0f185c0u, 0x7a3c3d7eu, 0x68646410u,
    0x857607a0u, 0x1d12622eu, 0x97f33466u, 0xdb4c9917u, 0x6469607cu, 0x566e043du,
    0x79ef1edbu, 0x2c05898du, 0xc9578e25u, 0xcd380101u, 0x46e04377u, 0x7d1cc7a9u,
    0x6552b837u, 0x20192608u, 0xb97500c5u, 0xed296b44u, 0x368648b4u, 0x62995cd5u,
    0x82731400u, 0xf9aebd8bu, 0x3844c0c7u, 0x7c2de794u, 0x33a1a770u, 0x8ae528c2u,
    0x5a2be812u, 0x1f8f4a07u, 0x2b5ed7cau, 0x937eb564u, 0x6fda7e11u, 0xe49b5d6cu,
    0xb4b3244eu, 0x18aa53a4u, 0x3a061334u, 0x4d6067a3u, 0x83ba5868u, 0x9bdf4dfeu,
    0x7449f261u, 0x709f8450u, 0xcad133cbu, 0xde941c3fu, 0xf52ae484u, 0x781d77edu,
    0x7e4395f0u, 0xae103b59u, 0x922331bbu, 0x42ce50c8u, 0xe6f08153u, 0xe7d941d0u,
    0x5028ed6bu, 0xb3d2c49bu, 0xad4d9c3eu, 0xd201fb6eu, 0xa45bd5beu, 0xffcb7f4bu,
    0x579d7806u, 0xf821bb5bu, 0x59d592adu, 0xd0be0c31u, 0xd4e3b676u, 0x0107165au,
    0x0fe939d2u, 0x49bcaafdu, 0x55ffcfe5u, 0x2ec1f783u, 0xf39a09a5u, 0x3eb42772u,
    0x19b55a5du, 0x024a0679u, 0x8c83b3f7u, 0x8642ba1du, 0xacacd9eau, 0x87d352c4u,
    0x60931f45u, 0xa05f97d7u, 0x1cecd42cu, 0xe2fcc87bu, 0xb60f94e2u, 0x67a34b0bu,
    0xfcdd40c9u, 0x0b150a27u, 0xd3ee9e04u, 0x582e29e9u, 0x4ac22b41u, 0x6ac4e1b8u,
    0xbccaa51au, 0x237af30eu, 0xebc3b709u, 0xc4a59d19u, 0x284bc98au, 0xe9d41a93u,
    0x6bfa2018u, 0x73b2d651u, 0x11f9a2fau, 0xce09bff1u, 0x41a470aau, 0x25888f22u,
    0x77e754e8u, 0xf7330d8eu, 0x158eab16u, 0xc5d68842u, 0xc685a6f6u, 0xe5b82fdeu,
    0x09ea3a96u, 0x6dde1536u, 0x4fa919dau, 0x26c0be9fu, 0x9eed6f69u, 0xf05555f2u,
    0xe06fc285u, 0x9cd76d23u, 0xaf452a92u, 0xefc74cb7u, 0x9d6b4732u, 0x8be408eeu,
    0x22401d0du, 0xee6c459du, 0x7587cb82u, 0xe8746862u, 0x5cbdde87u, 0x98794278u,
    0x31afb94du, 0xc11e0f2fu, 0x30e8fc2au, 0xcf3261efu, 0x1a3023e1u, 0xaa2f86cfu,
    0xf202e24au, 0x8d08dcffu, 0x764837c6u, 0xa26374ccu, 0x9f7c3e88u, 0x949cc57du,
    0xdd26a07fu, 0xc39efab0u, 0xc8f879a1u, 0xdce67bb9u, 0xf4b0a435u, 0x912c9ae0u,
    0xd85603e4u, 0x953a9bbfu, 0xfb8290d6u, 0x0aebcd5fu, 0x16206a9au, 0x6c787a14u,
    0xd9a0f16au, 0x29bf4f74u, 0x8f8bce91u, 0x0e5a9354u, 0xab038cb1u, 0x1b8ad11bu,
    0xe327ff49u, 0x0053da20u, 0x90cf51dcu, 0xda92fe6du, 0x0390ca47u, 0xa8958097u,
    0xa9dc5bafu, 0x3931e3c1u, 0x840446b6u, 0x63d069fbu, 0xd7460299u, 0x7124ecd1u,
    0x0791e613u, 0x485918fcu, 0xd635d04cu, 0xdf96ac33u, 0x66f2d303u, 0x247056aeu,
    0xa1a7b2a8u, 0x27d8cc9cu, 0x17b6e998u, 0x7bf5590fu, 0xfe97f557u, 0x5471d8a2u,
    0x83a327a1u, 0x9f379f51u, 0x40a7d007u, 0x11307423u, 0x224587c1u, 0xac27d63bu,
    0x3b7e64eau, 0x2e1cbfa6u, 0x09996000u, 0x03bc0e2cu, 0xd4c4478au, 0x4542e0abu,
    0xfeda26d4u, 0xc1d10fcbu, 0x8252f596u, 0x4494eb5cu, 0xa362f314u, 0xf5ba81fdu,
    0x75c3a376u, 0x4ca214cau, 0xe164deddu, 0x5088fa97u, 0x4b0930e0u, 0x2fcfb7e8u,
    0x33a6f4b2u, 0xc7e94211u, 0x2d66c774u, 0x43be8baeu, 0xc663d445u, 0x908eb130u,
    0xf4e3be15u, 0x63b9d566u, 0x529396b5u, 0x1e1be743u, 0x4d5ff63fu, 0x985e4a83u,
    0x71ab9df7u, 0xc516c6f5u, 0x85c19ab4u, 0x1f4daee4u, 0xf2973431u, 0xb713dc5eu,
    0x3f2e159au, 0xc824da16u, 0x06bf376au, 0xb2fe23ecu, 0xe39b1c22u, 0xf1eecb5fu,
    0x08e82d52u, 0x565686c2u, 0xab0aea93u, 0xfd47219fu, 0xebdbabd7u, 0x2404a185u,
    0x8c7312b9u, 0xa8f2d828u, 0x0c8902dau, 0x65b42b63u, 0xc0bbef62u, 0x4e3e4cefu,
    0x788f8018u, 0xee1ebab7u, 0x93928f9du, 0x683d2903u, 0xd3b60689u, 0xafcb0ddcu,
    0x88a4c47au, 0xf6dd9c3du, 0x7ea5fca0u, 0x8a6d7244u, 0xbe11f120u, 0x04ff91b8u,
    0x8d2dc8c0u, 0x27f97fdbu, 0x7f9e1f47u, 0x1734f0c7u, 0x26f3ed8eu, 0x0df8f2bfu,
    0xb0833d9eu, 0xe420a4e5u, 0xa423cae6u, 0x95616772u, 0x9ae6c049u, 0x075941f2u,
    0xd8e12812u, 0x000f6f4fu, 0x3c0d6b05u, 0x6cef921cu, 0xb82bc264u, 0x396cb008u,
    0x5d608a6fu, 0x6d7782c8u, 0x186550aau, 0x6b6fec09u, 0x28e70b13u, 0x57ce5688u,
    0xecd3af84u, 0x23335a95u, 0x91f40cd2u, 0x7b6a3b26u, 0xbd32b3b6u, 0x3754a6fbu,
    0x8ed088f0u, 0xf867e87cu, 0x20851746u, 0x6410f9c6u, 0x35380442u, 0xc2ca10a7u,
    0x1adea27fu, 0x76bddd79u, 0x92742cf4u, 0x0e98f7eeu, 0x164e931du, 0xb9c835b3u,
    0x69060a99u, 0xb44c531eu, 0xfa7b66feu, 0xc98a5b53u, 0x7d95aae9u, 0x302f467bu,
    0x74b811deu, 0xf3866abdu, 0xb5b3d32du, 0xfc3157a4u, 0xd251fe19u, 0x0b5d8eacu,
    0xda71ffd5u, 0x47ea05a3u, 0x05c6a9e1u, 0xca0ee958u, 0x9939034du, 0x25dc5edfu,
    0x79083cb1u, 0x86768450u, 0xcf757d6du, 0x5972b6bcu, 0xa78d59c9u, 0xc4ad8d41u,
    0x2a362ad3u, 0xd1179991u, 0x601407ffu, 0xdcf50917u, 0x587069d0u, 0xe0821ed6u,
    0xdbb59427u, 0x73911a4bu, 0x7c904fc3u, 0x844afb92u, 0x6f8c955du, 0xe8c0c5bbu,
    0xb67ab987u, 0xa529d96cu, 0xf91f7181u, 0x618b1b06u, 0xe718bb0cu, 0x8bd7615bu,
    0xd5a93a59u, 0x54aef81bu, 0x772136e3u, 0xce44fd9cu, 0x10cda57eu, 0x87d66e0bu,
    0x3d798967u, 0x1b2c1804u, 0x3edfbd68u, 0x15f6e62bu, 0xef68b854u, 0x3896db35u,
    0x12b7b5e2u, 0xcb489029u, 0x9e4f98a5u, 0x62eb77a8u, 0x217c24a2u, 0x964152f6u,
    0x49b2080au, 0x53d23ee7u, 0x48fb6d69u, 0x1903d190u, 0x9449e494u, 0xbf6e7886u,
    0xfb356cfau, 0x3a261365u, 0x424bc1ebu, 0xa1192570u, 0x019ca782u, 0x9d3f7e0eu,
    0x9c127575u, 0xedf02039u, 0xad57bcceu, 0x5c153277u, 0x81a84540u, 0xbcaa7356u,
    0xccd59b60u, 0xa62a629bu, 0xa25ccd10u, 0x2b5b65cfu, 0x1c535832u, 0x55fd4e3au,
    0x31d9790du, 0xf06bc37du, 0x4afc1d71u, 0xaeed5533u, 0xba461634u, 0xbb694b78u,
    0x5f3a5c73u, 0x6a3c764au, 0x8fb0cca9u, 0xf725684cu, 0x4fe5382fu, 0x1d0163afu,
    0x5aa07a8fu, 0xe205a8edu, 0xc30bad38u, 0xff22cf1fu, 0x72432e2eu, 0x32c2518bu,
    0x3487ce4eu, 0x7ae0ac02u, 0x709fa098u, 0x0a3b395au, 0x5b4043f8u, 0xa9e48c36u,
    0x149a8521u, 0xd07dee6bu, 0x46acd2f3u, 0x8958dffcu, 0xb3a1223cu, 0xb11d31c4u,
    0xcd7f4d3eu, 0x0f28e3adu, 0xe5b100beu, 0xaac54824u, 0xe9c9d7bau, 0x9bd47001u,
    0x80f149b0u, 0x66022f0fu, 0x020c4048u, 0x6efa192au, 0x67073f8du, 0x13ec7bf9u,
    0x3655011au, 0xe6afe157u, 0xd9845f6eu, 0xdecc4425u, 0x511ae2ccu, 0xdf81b4d8u,
    0xd7809e55u, 0xd6d883d9u, 0x2cc7978cu, 0x5e787cc5u, 0xdd0033d1u, 0xa050c937u,
    0x97f75dcdu, 0x299de580u, 0x41e2b261u, 0xea5a54f1u, 0x7e672590u, 0xbea513bbu,
    0x2c906fe6u, 0x86029c2bu, 0x55dc4f74u, 0x0553398eu, 0x63e09647u, 0xcafd0babu,
    0x264c37dfu, 0x8272210fu, 0x67afa669u, 0x12d98a5fu, 0x8cab23c4u, 0x75c68bd1u,
    0xc3370470u, 0x33f37f4eu, 0x283992ffu, 0xe73a3a67u, 0x1032f283u, 0xf5ad9fc2u,
    0x963f0c5du, 0x664fbc45u, 0x202ba41cu, 0xc7c02d80u, 0x54731e84u, 0x8a1085f5u,
    0x601d80fbu, 0x2f968e55u, 0x35e96812u, 0xe45a8f78u, 0xbd7de662u, 0x3b6e6eadu,
    0x8097c5efu, 0x070b6781u, 0xb1e508f3u, 0x24e4fae3u, 0xb81a7805u, 0xec0fc918u,
    0x43c8774bu, 0x9b2512a9u, 0x2b05ad04u, 0x32c2536fu, 0xedf236e0u, 0x8bc4b0cfu,
    0xbaceb837u, 0x4535b289u, 0x0d0e94c3u, 0xa5a371d0u, 0xad695a58u, 0x39e3437du,
    0x9186bffcu, 0x21038c3bu, 0x0aa9dff9u, 0x5d1f06ceu, 0x62def8a4u, 0xf740a2b4u,
    0xa2575868u, 0x682683c1u, 0xdbb30facu, 0x61fe1928u, 0x468a6511u, 0xc61cd5f4u,
    0xe54d9800u, 0x6b98d7f7u, 0x8418b6a5u, 0x5f09a5d2u, 0x90b4e80bu, 0x49b2c852u,
    0x69f11c77u, 0x17412b7eu, 0x7f6fc0edu, 0x56838dccu, 0x6e9546a2u, 0xd0758619u,
    0x087b9b9au, 0xd231a01du, 0xaf46d415u, 0x097060fdu, 0xd920f657u, 0x882d3f9fu,
    0x3ae7c3c9u, 0xe8a00d9bu, 0x4fe67ebeu, 0x2ef80eb2u, 0xc1916b0cu, 0xf4dffea0u,
    0xb97eb3ebu, 0xfdff84ddu, 0xff8b14f1u, 0xe96b0572u, 0xf64b508cu, 0xae220a6eu,
    0x4423ae5au, 0xc2bece5eu, 0xde27567cu, 0xfc935c63u, 0x47075573u, 0xe65b27f0u,
    0xe121fd22u, 0xf2668753u, 0x2debf5d7u, 0x8347e08du, 0xac5eda03u, 0x2a7cebe9u,
    0x3fe8d92eu, 0x23542fe4u, 0x1fa7bd50u, 0xcf9b4102u, 0x9d0dba39u, 0x9cb8902au,
    0xa7249d8bu, 0x0f6d667au, 0x5ebfa9ecu, 0x6a594df2u, 0x79600938u, 0x023b7591u,
    0xea2c79c8u, 0xc99d07eau, 0x64cb5ee1u, 0x1a9cab3du, 0x76db9527u, 0xc08e012fu,
    0x3dfb481au, 0x872f22e7u, 0x2948d15cu, 0xa4782c79u, 0x6f50d232u, 0x78f0728au,
    0x5a87aab1u, 0xc4e2c19cu, 0xee767387u, 0x1b2a1864u, 0x7b8d10d3u, 0xd1713161u,
    0x0eeac456u, 0xd8799e06u, 0xb645b548u, 0x4043cb65u, 0xa874fb29u, 0x4b12d030u,
    0x7d687413u, 0x18ef9a1fu, 0xd7631d4cu, 0x5829c7dau, 0xcdfa30fau, 0xc5084bb0u,
    0x92cd20e2u, 0xd4c16940u, 0x03283ec0u, 0xa917813fu, 0x9a587d01u, 0x70041f8fu,
    0xdc6ab1dcu, 0xddaee3d5u, 0x31829742u, 0x198c022du, 0x1c9eafcbu, 0x5bbc6c49u,
    0xd3d3293au, 0x16d50007u, 0x04bb8820u, 0x3c5c2a41u, 0x37ee7af8u, 0x8eb04025u,
    0x9313ecbau, 0xbffc4799u, 0x8955a744u, 0xef85d633u, 0x504499a7u, 0xa6ca6a86u,
    0xbb3d3297u, 0xb34a8236u, 0x6dccbe4fu, 0x06143394u, 0xce19fc7bu, 0xccc3c6c6u,
    0xe36254aeu, 0x77b7eda1u, 0xa133dd9eu, 0xebf9356au, 0x513ccf88u, 0xe2a1b417u,
    0x972ee5bdu, 0x853824cdu, 0x5752f4eeu, 0x6c1142e8u, 0x3ea4f309u, 0xb2b5934au,
    0xdfd628aau, 0x59acea3eu, 0xa01eb92cu, 0x389964bcu, 0xda305dd4u, 0x019a59b7u,
    0x11d2ca93u, 0xfaa6d3b9u, 0x4e772ecau, 0x72651776u, 0xfb4e5b0eu, 0xa38f91a8u,
    0x1d0663b5u, 0x30f4f192u, 0xb50051b6u, 0xb716ccb3u, 0x4abd1b59u, 0x146c5f26u,
    0xf134e2deu, 0x00f67c6cu, 0xb0e1b795u, 0x98aa4ec7u, 0x0cc73b34u, 0x654276a3u,
    0x8d1ba871u, 0x740a5216u, 0xe0d01a23u, 0x9ed161d6u, 0x9f36a324u, 0x993ebb7fu,
    0xfeb9491bu, 0x365ddcdbu, 0x810cffc5u, 0x71ec0382u, 0x2249e7bfu, 0x48817046u,
    0xf3a24a5bu, 0x4288e4d9u, 0x0bf5c243u, 0x257fe151u, 0x95b64c0du, 0x4164f066u,
    0xaaf7db08u, 0x73b1119du, 0x8f9f7bb8u, 0xd6844596u, 0xf07a34a6u, 0x53943d0au,
    0xf9dd166du, 0x7a8957afu, 0xf8ba3ce5u, 0x27c9621eu, 0x5cdae910u, 0xc8518998u,
    0x941538feu, 0x136115d8u, 0xaba8443cu, 0x4d01f931u, 0x34edf760u, 0xb45f266bu,
    0xd5d4de14u, 0x52d8ac35u, 0x15cfd885u, 0xcbc5cd21u, 0x4cd76d4du, 0x7c80ef54u,
    0xbc92ee75u, 0x1e56a1f6u, 0xbaa20b6cu, 0x9ffbad26u, 0xe1f7d738u, 0x794aec8du,
    0xc9e9cf3cu, 0x8a9a7846u, 0xc57c4685u, 0xb9a92fedu, 0x29cb141fu, 0x52f9ddb7u,
    0xf68ba6bcu, 0x19ccc020u, 0x4f584aaau, 0x3bf6a596u, 0x003b7cf7u, 0x54f0ce9au,
    0xa7ec4303u, 0x46cf0077u, 0x78d33aa1u, 0x215247d9u, 0x74bcdf91u, 0x08381d30u,
    0xdac43e40u, 0x64872531u, 0x0beffe5fu, 0xb317f457u, 0xaebb12dau, 0xd5d0d67bu,
    0x7d75c6b4u, 0x42a6d241u, 0x1502d0a9u, 0x3fd97fffu, 0xc6c3ed28u, 0x81868d0au,
    0x92628bc5u, 0x86679544u, 0xfd1867afu, 0x5ca3ea61u, 0x568d5578u, 0x4a2d71f4u,
    0x43c9d549u, 0x8d95de2bu, 0x6e5c74a0u, 0x9120ffc7u, 0x0d05d14au, 0xa93049d3u,
    0xbfa80e17u, 0xf4096810u, 0x043f5ef5u, 0xa673b4f1u, 0x6d780298u, 0xa4847783u,
    0x5ee726fbu, 0x9934c281u, 0x220a588cu, 0x384e240fu, 0x933d5c69u, 0x39e5ef47u,
    0x26e8b8f3u, 0x4c1c6212u, 0x8040f75du, 0x074b7093u, 0x6625a8d7u, 0x36298945u,
    0x76285088u, 0x651d37c3u, 0x24f5274du, 0xdbca3dabu, 0x186b7ee1u, 0xd80f8182u,
    0x14210c89u, 0x943a3075u, 0x4e6e11c4u, 0x4d7e6badu, 0xf05064c8u, 0x025dcd97u,
    0x4bc10302u, 0x7cede572u, 0x8f90a970u, 0xab88eebau, 0xb5998029u, 0x5124d839u,
    0xb0eeb6a3u, 0x89ddabdcu, 0xe8074d76u, 0xa1465223u, 0x32518cf2u, 0x9d39d4ebu,
    0xc0d84524u, 0xe35e6ea8u, 0x7abf3804u, 0x113e2348u, 0x9ae6069du, 0xb4dfdabbu,
    0xa8c5313fu, 0x23ea3f79u, 0x530e36a2u, 0xa5fd228bu, 0x95d1d350u, 0x2b14cc09u,
    0x40042956u, 0x879d05ccu, 0x2064b9cau, 0xacaca40eu, 0xb29c846eu, 0x9676c9e3u,
    0x752b7b8au, 0x7be2bcc2u, 0x6bd58f5eu, 0xd48f4c32u, 0x606835e4u, 0x9cd7c364u,
    0x2c269b7au, 0x3a0d079cu, 0x73b683feu, 0x45374f1eu, 0x10afa242u, 0x577f8666u,
    0xddaa10f6u, 0xf34f561cu, 0x3d355d6bu, 0xe47048aeu, 0xaa13c492u, 0x050344fdu,
    0x2aab5151u, 0xf5b26ae5u, 0xed919a59u, 0x5ac67900u, 0xf1cde380u, 0x0c79a11bu,
    0x351533fcu, 0xcd4d8e36u, 0x1f856005u, 0x690b9fddu, 0xe736dccfu, 0x1d47bf6au,
    0x7f66c72au, 0x85f21b7fu, 0x983cbdb6u, 0x01ebbebfu, 0x035f3b99u, 0xeb111f34u,
    0x28cefdc6u, 0x5bfc9ecdu, 0xf22eacb0u, 0x9e41cbb2u, 0xe0f8327cu, 0x82e3e26fu,
    0xfc43fc86u, 0xd0ba66dfu, 0x489ef2a7u, 0xd9e0c81du, 0x68690d52u, 0xcc451367u,
    0xc2232e16u, 0xe95a7335u, 0x0fdae19bu, 0xff5b962cu, 0x97596527u, 0xc46db333u,
    0x3ed4c562u, 0xc14c9d9eu, 0x5d6faa21u, 0x638e940du, 0xf9316d58u, 0x47b3b0eau,
    0x30ffcad2u, 0xce1bba7du, 0x1e6108e6u, 0x2e1ea33du, 0x507bf05bu, 0xfafef94bu,
    0xd17de8e2u, 0x5598b214u, 0x1663f813u, 0x17d25a2du, 0xeefa5ff9u, 0x582f4e37u,
    0x12128773u, 0xfef17ab8u, 0x06005322u, 0xbb32bbc9u, 0x8c898508u, 0x592c15f0u,
    0xd38a4054u, 0x4957b7d6u, 0xd2b891dbu, 0x37bd2d3eu, 0x34ad20cbu, 0x622288e9u,
    0x2dc7345au, 0xafb416c0u, 0x1cf459b1u, 0xdc7739fau, 0x0a711a25u, 0x13e18a0cu,
    0x5f72af4cu, 0x6ac8db11u, 0xbe53c18eu, 0x1aa569b9u, 0xef551ea4u, 0xa02a429fu,
    0xbd16e790u, 0x7eb9171au, 0x77d693d8u, 0x8e06993au, 0x9bde7560u, 0xe5801987u,
    0xc37a09beu, 0xb8db76acu, 0xe2087294u, 0x6c81616du, 0xb7f30fe7u, 0xbc9b82bdu,
    0xfba4e4d4u, 0xc7b1012fu, 0xa20c043bu, 0xde9febd0u, 0x2f9297ceu, 0xe610aef8u,
    0x70b06f19u, 0xc86ae00bu, 0x0e01988fu, 0x41192ae0u, 0x448c1cb5u, 0xadbe92eeu,
    0x7293a007u, 0x1b54b5b3u, 0xd61f63d1u, 0xeae40a74u, 0x61a72b55u, 0xec83a7d5u,
    0x88942806u, 0x90a07da5u, 0xd7424b95u, 0x67745b4eu, 0xa31a1853u, 0xca6021efu,
    0xdfb56c4fu, 0xcbc2d915u, 0x3c48e918u, 0x8bae3c63u, 0x6f659c71u, 0xf8b754c1u,
    0x2782f3deu, 0xf796f168u, 0x71492c84u, 0x33c0f5a6u, 0x3144f6ecu, 0x25dc412eu,
    0xb16c5743u, 0x83a1fa7eu, 0x0997b101u, 0xb627e6e8u, 0xcf33905cu, 0x8456fb65u,
    0xb29bea74u, 0xc35da605u, 0x305c1ca3u, 0xd2e9f5bcu, 0x6fd5bff4u, 0xff347703u,
    0xfc45b163u, 0xf498e068u, 0xb71229fcu, 0x81acc3fbu, 0x78538a8bu, 0x984ecf81u,
    0xa5da47a4u, 0x8f259eefu, 0x6475dc65u, 0x081865b9u, 0x49e14a3cu, 0x19e66079u,
    0xd382e91bu, 0x5b109794u, 0x3f9f81e1u, 0x4470a388u, 0x41601abeu, 0xaaf9f407u,
    0x8e175ef6u, 0xed842297u, 0x893a4271u, 0x1790839au, 0xd566a99eu, 0x6b417deeu,
    0x75c90d23u, 0x715edb31u, 0x723553f7u, 0x9afb50c9u, 0xfbc5f600u, 0xcd3b6a4eu,
    0x97ed0fbau, 0x29689aecu, 0x63135c8eu, 0xf0e26c7eu, 0x0692ae7fu, 0xdbb208ffu,
    0x2ede3e9bu, 0x6a65bebdu, 0xd40867e9u, 0xc954afc5u, 0x73b08201u, 0x7ffdf809u,
    0x1195c24fu, 0x1ca5adcau, 0x74bd6d1fu, 0xb393c455u, 0xcadfd3fau, 0x99f13011u,
    0x0ebca813u, 0x60e791b8u, 0x6597ac7au, 0x18a7e46bu, 0x09cb49d3u, 0x0b27df6du,
    0xcfe52f87u, 0xcef66837u, 0xe6328035u, 0xfa87c592u, 0x37baff93u, 0xd71fcc99u,
    0xdcab205cu, 0x4d7a5638u, 0x48012510u, 0x62797558u, 0xb6cf1fe5u, 0xbc311834u,
    0x9c2373acu, 0x14ec6175u, 0xa439cbdfu, 0x54afb0eau, 0xd686960bu, 0xfdd0d47bu,
    0x7b063902u, 0x8b78bac3u, 0x26c6a4d5u, 0x5c0055b6u, 0x2376102eu, 0x0411783eu,
    0x2aa3f1cdu, 0x51fc6ea8u, 0x701ce243u, 0x9b2a0abbu, 0x0ad93733u, 0x6e80d03du,
    0xaf6295d1u, 0xf629896fu, 0xa30b0648u, 0x463d8dd4u, 0x963f84cbu, 0x01ff94f8u,
    0x8d7fefdcu, 0x553611c0u, 0xa97c1719u, 0xb96af759u, 0xe0e3c95eu, 0x0528335bu,
    0x21fe5925u, 0x821a5245u, 0x807238b1u, 0x67f23db5u, 0xea6b4eabu, 0x0da6f985u,
    0xab1bc85au, 0xef8c90e4u, 0x4526230eu, 0x38eb8b1cu, 0x1b91cd91u, 0x9fce5f0cu,
    0xf72cc72bu, 0xc64f2617u, 0xdaf7857du, 0x7d373cf1u, 0x28eaedd7u, 0x203887d0u,
    0xc49a155fu, 0xa251b3b0u, 0xf2d47ae3u, 0x3d9ef267u, 0x4a94ab2fu, 0x7755a222u,
    0x0205e329u, 0xc28fa7a7u, 0xaec1fe51u, 0x270f164cu, 0x8c6d01bfu, 0x53b5bc98u,
    0xc09d3febu, 0x834986ccu, 0x4309a12cu, 0x578b2a96u, 0x3bb74b86u, 0x69561b4au,
    0x037e32f3u, 0xde335b08u, 0xc5156be0u, 0xe7ef09adu, 0x93b834c7u, 0xa7719352u,
    0x59302821u, 0xe3529d26u, 0xf961da76u, 0xcb142c44u, 0xa0f3b98du, 0x76502457u,
    0x945a414bu, 0x078eeb12u, 0xdff8de69u, 0xeb6c8c2du, 0xbda90c4du, 0xe9c44d16u,
    0x168dfd66u, 0xad64763bu, 0xa65fd764u, 0x95a29c06u, 0x32d7713fu, 0x40f0b277u,
    0x224af08fu, 0x004cb5e8u, 0x92574814u, 0x8877d827u, 0x3e5b2d04u, 0x68c2d5f2u,
    0x86966273u, 0x1d433adau, 0x8774988au, 0x3c0e0bfeu, 0xddad581du, 0x2fd654edu,
    0x0f4769fdu, 0xc181ee9du, 0x5fd88f61u, 0x341dbb3au, 0x528543f9u, 0xd92235cfu,
    0x1ea82eb4u, 0xb5cd790fu, 0x91d24f1eu, 0xa869e6c2u, 0x61f474d2u, 0xcc205addu,
    0x0c7bfba9u, 0xbf2b0489u, 0xb02d72d8u, 0x2b46ece6u, 0xe4dcd90au, 0xb8a11440u,
    0xee8a63b7u, 0x854dd1a1u, 0xd1e00583u, 0x42b40e24u, 0x9e8964deu, 0xb4b35d78u,
    0xbec76f6eu, 0x24b9c620u, 0xd8d399a6u, 0x5adb2190u, 0x2db12730u, 0x3a5866afu,
    0x58c8fadbu, 0x5d8844e7u, 0x8a4bf380u, 0x15a01d70u, 0x79f5c028u, 0x66be3b8cu,
    0xf3e42b53u, 0x56990039u, 0x2c0c3182u, 0x5e16407cu, 0xecc04515u, 0x6c440284u,
    0x4cb6701au, 0x13bfc142u, 0x9d039f6au, 0x4f6e92c8u, 0xa1407c62u, 0x8483a095u,
    0xc70ae1c4u, 0xe20213a2u, 0xbacafc41u, 0x4ecc12b3u, 0x4bee3646u, 0x1fe807aeu,
    0x25217f9cu, 0x35dde5f5u, 0x7a7dd6ceu, 0xf89cce50u, 0xac07b718u, 0x7e73d2c6u,
    0xe563e76cu, 0x123ca536u, 0x3948ca56u, 0x9019dd49u, 0x10aa88d9u, 0xc82451e2u,
    0x473eb6d6u, 0x506fe854u, 0xe8bb03a5u, 0x332f4c32u, 0xfe1e1e72u, 0xb1ae572au,
    0x7c0d7bc1u, 0xe1c37eb2u, 0xf542aa60u, 0xf1a48ea0u, 0xd067b89fu, 0xbbfa195du,
    0x1a049b0du, 0x315946aau, 0x36d1b447u, 0x6d2ebdf0u, 0x0d188a6du, 0x12cea0dbu,
    0x7e63740eu, 0x6a444821u, 0x253d234fu, 0x6ffc6597u, 0x94a6bdefu, 0x33ee1b2fu,
    0x0a6c00c0u, 0x3aa336b1u, 0x5af55d17u, 0x265fb3dcu, 0x0e89cf4du, 0x0786b008u,
    0xc80055b8u, 0x6b17c3ceu, 0x72b05a74u, 0xd21a8d78u, 0xa6b70840u, 0xfe8eae77u,
    0xed69565cu, 0x55e1bcf4u, 0x585c2f60u, 0xe06f1a62u, 0xad67c0cdu, 0x7712af88u,
    0x9cc26acau, 0x1888053du, 0x37eb853eu, 0x9215abd7u, 0xde30adfcu, 0x1f1038e6u,
    0x70c51c8au, 0x8d586c26u, 0xf72bdd90u, 0x4dc3ce15u, 0x68eaeefau, 0xd0e9c8b9u,
    0x200f9c44u, 0xddd141bau, 0x024bf1d3u, 0x0f64c9d4u, 0xc421e9e9u, 0x9d11c14cu,
    0x9a0dd9e4u, 0x5f92ec19u, 0x1b980df0u, 0x1dcc4542u, 0xb8fe8c56u, 0x0c9c9167u,
    0x4e81eb49u, 0xca368f27u, 0xe3603b37u, 0xea08acccu, 0xac516992u, 0xc34f513bu,
    0x804d100du, 0x6edca4c4u, 0xfc912939u, 0x29d219b0u, 0x278aaa3cu, 0x4868da7du,
    0x54e890b7u, 0xb46d735au, 0x514589aau, 0xd6c630afu, 0x4980dfe8u, 0xbe3ccc55u,
    0x59d41202u, 0x650c078bu, 0xaf3a9e7bu, 0x3ed9827au, 0x9e79fc6eu, 0xaadbfbaeu,
    0xc5f7d803u, 0x3daf7f50u, 0x67b4f465u, 0x73406e11u, 0x39313f8cu, 0x8a6e6686u,
    0xd8075f1fu, 0xd3cbfed1u, 0x69c7e49cu, 0x930581e0u, 0xe4b1a5a8u, 0xbbc45472u,
    0x09ddbf58u, 0xc91d687eu, 0xbdbffda5u, 0x88c08735u, 0xe9e36bf9u, 0xdb5ea9b6u,
    0x95559404u, 0x08f432fbu, 0xe24ea281u, 0x64663579u, 0x000b8010u, 0x7914e7d5u,
    0x32fd0473u, 0xd1a7f0a4u, 0x445ab98eu, 0xec72993fu, 0xa29a4d32u, 0xb77306d8u,
    0xc7c97cf6u, 0x7b6ab645u, 0xf5ef7adfu, 0xfb2e15f7u, 0xe747f757u, 0x5e944354u,
    0x234a2669u, 0x47e46359u, 0x9b9d11a9u, 0x40762cedu, 0x56f1de98u, 0x11334668u,
    0x890a9a70u, 0x1a296113u, 0xb3bd4af5u, 0x163b7548u, 0xd51b4f84u, 0xb99b2abcu,
    0x3cc1dc30u, 0xa9f0b56cu, 0x812272b2u, 0x0b233a5fu, 0xb650dbf2u, 0xf1a0771bu,
    0x36562b76u, 0xdc037b0fu, 0x104c97ffu, 0xc2ec98d2u, 0x90596f22u, 0x28b6620bu,
    0xdf42b212u, 0xfdbc4243u, 0xf3fb175eu, 0x4a2d8b00u, 0xe8f3869bu, 0x30d69bc3u,
    0x853714c8u, 0xa7751d2eu, 0x31e56deau, 0xd4840b0cu, 0x9685d783u, 0x068c9333u,
    0x8fba032cu, 0x76d7bb47u, 0x6d0ee22bu, 0xb546794bu, 0xd971b894u, 0x8b09d253u,
    0xa0ad5761u, 0xee77ba06u, 0x46359f31u, 0x577cc7ecu, 0x52825efdu, 0xa4beed95u,
    0x9825c52au, 0xeb48029au, 0xbaae59f8u, 0xcf490ee1u, 0xbc990164u, 0x8ca49dfeu,
    0x4f38a6e7u, 0x2ba98389u, 0x8228f538u, 0x199f64acu, 0x01a1cac5u, 0xa8b51641u,
    0x5ce72d01u, 0x8e5df26bu, 0x60f28e1eu, 0xcd5be125u, 0xe5b376bfu, 0x1c8d3116u,
    0x7132cbb3u, 0xcb7ae320u, 0xc0fa5366u, 0xd7653e34u, 0x971c88c2u, 0xc62c7dd0u,
    0x34d0a3dau, 0x868f6709u, 0x7ae6fa8fu, 0x22bbd523u, 0x66cd3d5bu, 0x1ef9288du,
    0xf9cf58c1u, 0x5b784e80u, 0x7439a191u, 0xae134c36u, 0x9116c463u, 0x2e9e1396u,
    0xf8611f3au, 0x2d2f3307u, 0x247f37ddu, 0xc1e2ff9du, 0x43c821e5u, 0x05ed5cabu,
    0xef74e80au, 0x4cca6028u, 0xf0ac3cbdu, 0x5d874b29u, 0x6c62f6a6u, 0x4b2a2ef3u,
    0xb1aa2087u, 0x62a5d0a3u, 0x0327221cu, 0xb096b4c6u, 0x417ec693u, 0xaba840d6u,
    0x789725ebu, 0xf4b9e02du, 0xe6e00975u, 0xcc04961au, 0x63f624bbu, 0x7fa21ecbu,
    0x2c01ea7fu, 0xb2415005u, 0x2a8bbeb5u, 0x83b2b14eu, 0xa383d1a7u, 0x5352f96au,
    0x043ecdadu, 0xce1918a1u, 0xfa6be6c9u, 0x50def36fu, 0xf6b80ce2u, 0x4543ef7cu,
    0x9953d651u, 0xf257955du, 0x87244914u, 0xda1e0a24u, 0xffda4785u, 0x14d327a2u,
    0x3b93c29fu, 0x840684b4u, 0x61ab71a0u, 0x9f7b784au, 0x2fd570cfu, 0x15955bdeu,
    0x38f8d471u, 0x3534a718u, 0x133fb71du, 0x3fd80f52u, 0x4290a8beu, 0x75ff44c7u,
    0xa554e546u, 0xe1023499u, 0xbf2652e3u, 0x7d20399eu, 0xa1df7e82u, 0x177092eeu,
    0x217dd3f1u, 0x7c1ff8d9u, 0x12113f2eu, 0xbfbd0785u, 0xf11793fbu, 0xa5bff566u,
    0x83c7b0e5u, 0x72fb316bu, 0x75526a9au, 0x41e0e612u, 0x7156ba09u, 0x53ce7deeu,
    0x0aa26881u, 0xa43e0d7du, 0x3da73ca3u, 0x182761edu, 0xbd5077ffu, 0x56db4aa0u,
    0xe792711cu, 0xf0a4eb1du, 0x7f878237u, 0xec65c4e8u, 0x08dc8d43u, 0x0f8ce142u,
    0x8258abdau, 0xf4154e16u, 0x49dec2fdu, 0xcd8d5705u, 0x6c2c3a0fu, 0x5c12bb88u,
    0xeff3cdb6u, 0x2c89ed8cu, 0x7beba967u, 0x2a142157u, 0xc6d0836fu, 0xb4f97e96u,
    0x6931e969u, 0x514e6c7cu, 0xa7792600u, 0x0bbbf780u, 0x59671bbdu, 0x0707b676u,
    0x37482d93u, 0x80af1479u, 0x3805a60du, 0xe1f4cac1u, 0x580b3074u, 0x30b8d6ceu,
    0x05a304beu, 0xd176626du, 0xebca97f3u, 0xbb201f11u, 0x6a1afe23u, 0xffaa86e4u,
    0x62b4da49u, 0x1b6629f5u, 0xf5d9e092u, 0xf37f3dd1u, 0x619bd45bu, 0xa6ec8e4fu,
    0x29c80939u, 0x0c7c0c34u, 0x9cfe6e48u, 0xe65fd3acu, 0x73613b65u, 0xb3c669f9u,
    0xbe2e8a9eu, 0x286f9678u, 0x5797fd13u, 0x99805d75u, 0xcfb641c5u, 0xa91074bau,
    0x6343af47u, 0x6403cb46u, 0x8894c8dbu, 0x2663034cu, 0x3c40dc5eu, 0x00995231u,
    0x96789aa2u, 0x2efde4b9u, 0x7dc195e1u, 0x547dadd5u, 0x06a8ea04u, 0xf2347a63u,
    0x5e0dc6f7u, 0x8462dfc2u, 0x1e6b2c3cu, 0x9bd275b3u, 0x91d419e2u, 0xbcefd17eu,
    0xb9003924u, 0xd07e7320u, 0xdef0495cu, 0xc36ad00eu, 0x1785b1abu, 0x92e20bcfu,
    0xb139f0e9u, 0x675bb9a1u, 0xaecfa4afu, 0x132376cbu, 0xe84589d3u, 0x79a05456u,
    0xa2f860bcu, 0x1ae4f8b5u, 0x20df4db4u, 0xa1e1428bu, 0x3bf60a1au, 0x27ff7bf1u,
    0xcb44c0e7u, 0xf7f587c4u, 0x1f3b9b21u, 0x94368f01u, 0x856e23a4u, 0x6f93de3fu,
    0x773f5bbfu, 0x8b22056eu, 0xdf41f654u, 0xb8246ff4u, 0x8d57bff2u, 0xd57167eau,
    0xc5699f22u, 0x40734ba7u, 0x5d5c2772u, 0x033020a8u, 0xe30a7c4du, 0xadc40fd6u,
    0x76353441u, 0x5aa5229bu, 0x81516590u, 0xda49f14eu, 0x4fa672a5u, 0x4d9fac5fu,
    0x154be230u, 0x8a7a5cc0u, 0xce3d2f84u, 0xcca15514u, 0x5221360cu, 0xaf0fb81eu,
    0x5bdd5873u, 0xf6825f8fu, 0x1113d228u, 0x70ad996cu, 0x93320051u, 0x60471c53u,
    0xe9ba567bu, 0x3a462ae3u, 0x5f55e72du, 0x1d3c5ad7u, 0xdcfc45ecu, 0x34d812efu,
    0xfa96ee1bu, 0x369d1ef8u, 0xc9b1a189u, 0x7c1d3555u, 0x50845edcu, 0x4bb31877u,
    0x8764a060u, 0x8c9a9415u, 0x230e1a3au, 0xb05e9133u, 0x242b9e03u, 0xa3b99db7u,
    0xc2d7fb0au, 0x3333849du, 0xd27278d4u, 0xb5d3efa6u, 0x78ac28adu, 0xc7b2c135u,
    0x0926ecf0u, 0xc1374c91u, 0x74f16d98u, 0x2274084au, 0x3f6d9cfau, 0x7ac0a383u,
    0xb73aff1fu, 0x3909a23du, 0x9f1653aeu, 0x4e2f3e71u, 0xca5ab22au, 0xe01e3858u,
    0x90c5a7ebu, 0x3e4a17dfu, 0xaa987fb0u, 0x488bbd62u, 0xb625062bu, 0x2d776bb8u,
    0x43b5fc08u, 0x1490d532u, 0xd6d12495u, 0x44e89845u, 0x2fe60118u, 0x9d9ef950u,
    0xac38133eu, 0xd3864329u, 0x017b255au, 0xfdc2dd26u, 0x256851e6u, 0x318e7086u,
    0x2bfa4861u, 0x89eac706u, 0xee5940c6u, 0x68c3bc2fu, 0xe260334bu, 0x98da90bbu,
    0xf818f270u, 0x4706d897u, 0x212d3799u, 0x4cf7e5d0u, 0xd9c9649fu, 0xa85db5cdu,
    0x35e90e82u, 0x6b881152u, 0xab1c02c7u, 0x46752b02u, 0x664f598eu, 0x45ab2e64u,
    0xc4cdb4b2u, 0xba42107fu, 0xea2a808au, 0x971bf3deu, 0x4a54a836u, 0x4253aeccu,
    0x1029be68u, 0x6dcc9225u, 0xe4bca56au, 0xc0ae50b1u, 0x7e011d94u, 0xe59c162cu,
    0xd8e5c340u, 0xd470fa0bu, 0xb2be79ddu, 0xd783889cu, 0x1cede8f6u, 0x8f4c817au,
    0xddb785c9u, 0x860232d8u, 0x198aaad9u, 0xa0814738u, 0x3219cffcu, 0x169546d2u,
    0xfc0cb759u, 0x55911510u, 0x04d5cec3u, 0xed08cc3bu, 0x0d6cf427u, 0xc8e38ccau,
    0x0eeee3feu, 0x9ee7d7c8u, 0xf9f24fa9u, 0xdb04b35du, 0x9ab0c9e0u, 0x651f4417u,
    0x028f8b07u, 0x6e28d9aau, 0xfba96319u, 0x8ed66687u, 0xfecbc58du, 0x954ddb44u,
    0x7b0bdffeu, 0x865d16b1u, 0x49a058c0u, 0x97abaa3fu, 0xcaacc75du, 0xaba6c17du,
    0xf8746f92u, 0x6f48aeedu, 0x8841d4b5u, 0xf36a146au, 0x73c390abu, 0xe6fb558fu,
    0x87b1019eu, 0x26970252u, 0x246377b2u, 0xcbf676aeu, 0xf923db06u, 0xf7389116u,
    0x14c81a90u, 0x83114eb4u, 0x8b137559u, 0x95a86a7au, 0xd5b8da8cu, 0xc4df780eu,
    0x5a9cb3e2u, 0xe44d4062u, 0xe8dc8ef6u, 0x9d180845u, 0x817ad18bu, 0xc286c85bu,
    0x251f20deu, 0xee6d5933u, 0xf6edef81u, 0xd4d16c1eu, 0xc94a0c32u, 0x8437fd22u,
    0x3271ee43u, 0x42572aeeu, 0x5f91962au, 0x1c522d98u, 0x59b23f0cu, 0xd86b8804u,
    0x08c63531u, 0x2c0d7a40u, 0xb97c4729u, 0x04964df9u, 0x13c74a17u, 0x5878362fu,
    0x4c808cd6u, 0x092cb1e0u, 0x6df02885u, 0xa0c2105eu, 0x8aba9e68u, 0x64e03057u,
    0xe5d61325u, 0x0e43a628u, 0x16dbd62bu, 0x2733d90bu, 0x3ae57283u, 0xc0c1052cu,
    0x4b6fb620u, 0x37513953u, 0xfc898bb3u, 0x471b179fu, 0xdf6e66b8u, 0xd32142f5u,
    0x9b30fafcu, 0x4ed92549u, 0x105c6d99u, 0x4acd69ffu, 0x2b1a27d3u, 0x6bfcc067u,
    0x6301a278u, 0xad36e6f2u, 0xef3ff64eu, 0x56b3cadbu, 0x0184bb61u, 0x17beb9fdu,
    0xfaec6109u, 0xa2e1ffa1u, 0x2fd224f8u, 0x238f5be6u, 0x8f8570cfu, 0xaeb5f25au,
    0x4f1d3e64u, 0x4377eb24u, 0x1fa45346u, 0xb2056386u, 0x52095e76u, 0xbb7b5adcu,
    0x3514e472u, 0xdde81e6eu, 0x7acea9c4u, 0xac15cc48u, 0x71c97d93u, 0x767f941cu,
    0x911052a2u, 0xffea09bfu, 0xfe3ddcf0u, 0x15ebf3aau, 0x9235b8bcu, 0x75408615u,
    0x9a723437u, 0xe1a1bd38u, 0x33541b7eu, 0x1bdd6856u, 0xb307e13eu, 0x90814bb0u,
    0x51d7217bu, 0x0bb92219u, 0x689f4500u, 0xc568b01fu, 0x5df3d2d7u, 0x3c0ecd0du,
    0x2a0244c8u, 0x852574e8u, 0xe72f23a9u, 0x8e26ed02u, 0x2d92cbddu, 0xdabc0458u,
    0xcdf5feb6u, 0x9e4e8dccu, 0xf4f1e344u, 0x0d8c436du, 0x4427603bu, 0xbdd37fdau,
    0x80505f26u, 0x8c7d2b8eu, 0xb73273c5u, 0x397362eau, 0x618a3811u, 0x608bfb88u,
    0x06f7d714u, 0x212e4677u, 0x28efceadu, 0x076c0371u, 0x36a3a4d9u, 0x5487b455u,
    0x3429a365u, 0x65d467acu, 0x78ee7eebu, 0x99bf12b7u, 0x4d129896u, 0x772a5601u,
    0xcce284c7u, 0x2ed85c21u, 0xd099e8a4u, 0xa179158au, 0x6ac0ab1au, 0x299a4807u,
    0xbe67a58du, 0xdc19544au, 0xb8949b54u, 0x8d315779u, 0xb6f849c1u, 0x53c5ac34u,
    0x66de92a5u, 0xf195dd13u, 0x318d3a73u, 0x301ec542u, 0x0cc40da6u, 0xf253ade4u,
    0x467ee566u, 0xea5585ecu, 0x3baf19bbu, 0x7de9f480u, 0x79006e7cu, 0xa9b7a197u,
    0xa44bd8f1u, 0xfb2ba739u, 0xec342fd4u, 0xed4fd32du, 0x3d1789bau, 0x400f5d7fu,
    0xc798f594u, 0x4506a847u, 0x034c0a95u, 0xe2162c9du, 0x55a9cfd0u, 0x692d832eu,
    0xcf9db2cau, 0x5e2287e9u, 0xd2610ef3u, 0x1ae7ecc2u, 0x48399ca0u, 0xa7e4269bu,
    0x6ee3a0afu, 0x7065bfe1u, 0xa6ffe708u, 0x2256804cu, 0x7476e21bu, 0x41b0796cu,
    0x7c243b05u, 0x000a950fu, 0x1858416bu, 0xf5a53c89u, 0xe9fef823u, 0x3f443275u,
    0xe0cbf091u, 0x0af27b84u, 0x3ebb0f27u, 0x1de6f7f4u, 0xc31c29f7u, 0xb166de3du,
    0x12932ec3u, 0x9c0c0674u, 0x5cda81b9u, 0xd1bd9d12u, 0xaffd7c82u, 0x8962bca7u,
    0xa342c4a8u, 0x62457151u, 0x82089f03u, 0xeb49c670u, 0x5b5f6530u, 0x7e28bad2u,
    0x20880ba3u, 0xf0faafcdu, 0xce82b56fu, 0x0275335cu, 0xc18e8afbu, 0xde601d69u,
    0xba9b820au, 0xc8a2be4fu, 0xd7cac335u, 0xd9a73741u, 0x115e974du, 0x7f5ac21du,
    0x383bf9c6u, 0xbcaeb75fu, 0xfd0350ceu, 0xb5d06b87u, 0x9820e03cu, 0x72d5f163u,
    0xe3644fc9u, 0xa5464c4bu, 0x57048fcbu, 0x9690c9dfu, 0xdbf9eafau, 0xbff4649au,
    0x053c00e3u, 0xb4b61136u, 0x67593dd1u, 0x503ee960u, 0x9fb4993au, 0x19831810u,
    0xc670d518u, 0xb05b51d8u, 0x0f3a1ce5u, 0x6caa1f9cu, 0xaacc31beu, 0x949ed050u,
    0x1ead07e7u, 0xa8479abdu, 0xd6cffcd5u, 0x936993efu, 0x472e91cbu, 0x5444b5b6u,
    0x62be5861u, 0x1be102c7u, 0x63e4b31eu, 0xe81f71b7u, 0x9e2317c9u, 0x39a408aeu,
    0x518024f4u, 0x1731c66fu, 0x68cbc918u, 0x71fb0c9eu, 0xd03b7fddu, 0x7d6222ebu,
    0x9057eda3u, 0x1a34a407u, 0x8cc2253du, 0xb6f6979du, 0x835675dcu, 0xf319be9fu,
    0xbe1cd743u, 0x4d32fee4u, 0x77e7d887u, 0x37e9ebfdu, 0x15f851e8u, 0x23dc3706u,
    0x19d78385u, 0xbd506933u, 0xa13ad4a6u, 0x913f1a0eu, 0xdde560b9u, 0x9a5f0996u,
    0xa65a0435u, 0x48d34c4du, 0xe90839a7u, 0x8abba54eu, 0x6fd13ce1u, 0xc7eebd3cu,
    0x0e297602u, 0x58b9bbb4u, 0xef7901e6u, 0x64a28a62u, 0xa509875au, 0xf8834442u,
    0x2702c709u, 0x07353f31u, 0x3b39f665u, 0xf5b18b49u, 0x4010ae37u, 0x784de00bu,
    0x7a1121e9u, 0xde918ed3u, 0xc8529dcdu, 0x816a5d05u, 0x02ed8298u, 0x04e3dd84u,
    0xfd2bc3e2u, 0xaf167089u, 0x96af367eu, 0xa4da6232u, 0x18ff7325u, 0x05f9a9f1u,
    0x4fefb9f9u, 0xcd94eaa5u, 0xbfaa5069u, 0xa0b8c077u, 0x60d86f57u, 0xfe71c813u,
    0x29ebd2c8u, 0x4ca86538u, 0x6bf1a030u, 0xa237b88au, 0xaa8af41du, 0xe1f7b6ecu,
    0xe214d953u, 0x33057879u, 0x49caa736u, 0xfa45cff3u, 0xc063b411u, 0xba7e27d0u,
    0x31533819u, 0x2a004ac1u, 0x210efc3fu, 0x2646885eu, 0x66727dcfu, 0x9d7fbf54u,
    0xa8dd0ea8u, 0x3447caceu, 0x3f0c14dbu, 0xb8382aacu, 0x4ace3539u, 0x0a518d51u,
    0x95178981u, 0x35aee2cau, 0x73f0f7e3u, 0x94281140u, 0x59d0e523u, 0xd292cb88u,
    0x565d1b27u, 0x7ec8fbafu, 0x069af08du, 0xc127fd24u, 0x0bc77b10u, 0x5f03e7efu,
    0x453e99bau, 0xeed9ff7fu, 0x87b55215u, 0x7915ab4cu, 0xd389a358u, 0x5e75ce6du,
    0x28d655c0u, 0xdad26c73u, 0x2e2510ffu, 0x9fa7eeccu, 0x1d0629c3u, 0xdc9c9c46u,
    0x2d67ecd7u, 0xe75e94bdu, 0x3d649e2au, 0x6c413a2bu, 0x706f0d7cu, 0xdfb0127bu,
    0x4e366b55u, 0x2c825650u, 0x24205720u, 0xb5c998f7u, 0x3e95462cu, 0x756e5c72u,
    0x3259488fu, 0x11e8771au, 0xa7c0a617u, 0x577663e5u, 0x089b6401u, 0x8eab1941u,
    0xae55ef8cu, 0x3aac5460u, 0xd4e6262fu, 0x5d979a47u, 0xb19823b0u, 0x7f8d6a0cu,
    0xffa08683u, 0x0170cd0fu, 0x858cd5d8u, 0x53961c90u, 0xc4c61556u, 0x41f2f226u,
    0xcfcd062du, 0xf24c03b8u, 0xea81df5bu, 0x7be2fa52u, 0xb361f98bu, 0xc2901316u,
    0x55ba4bbcu, 0x93b234a9u, 0x0fbc6603u, 0x80a96822u, 0x6d60491fu, 0x22bd00f8u,
    0xbcad5aadu, 0x52f3f13bu, 0x42fd2b28u, 0xb41dd01cu, 0xc52c93bfu, 0xfc663094u,
    0x8f58d100u, 0x43fecc08u, 0xc6331e5du, 0xe6480f66u, 0xca847204u, 0x4bdf1da0u,
    0x30cc2efbu, 0x13e02deau, 0xfb49ac45u, 0xf9d4434fu, 0xf47c5b9cu, 0x148879c2u,
    0x039fc234u, 0xa3db9bfcu, 0xd1a1dc5cu, 0x763d7cd4u, 0xed6d2f93u, 0xab13af6eu,
    0x1e8e054au, 0xd68f4f9au, 0xc30484b3u, 0xd7d50afau, 0x6930855fu, 0xcc07db95u,
    0xce746db1u, 0x744e967du, 0xf16cf575u, 0x8643e8b5u, 0xf0eae38eu, 0xe52de1d1u,
    0x6587dae0u, 0x0c4b8121u, 0x1c7ac567u, 0xac0db20au, 0x36c3a812u, 0x5b1a4514u,
    0xa9a3f868u, 0xb9263baau, 0xcb3ce9d2u, 0xe44fb1a4u, 0x9221bc82u, 0xb29390feu,
    0x6ab41863u, 0x974a3e2eu, 0x89f531c5u, 0x255ca13eu, 0x8b65d348u, 0xec248f78u,
    0xd8fc16f0u, 0x50ecdeeeu, 0x09010792u, 0x3c7d1fb2u, 0xeba5426bu, 0x847b417au,
    0x468b40d9u, 0x8dc4e680u, 0x7cc1f391u, 0x2f1eb086u, 0x6e5baa6au, 0xe0b395dau,
    0xe31b2cf6u, 0xd9690b0du, 0x729ec464u, 0x38403ddeu, 0x610b80a2u, 0x5cf433abu,
    0xb0785fc4u, 0xd512e4c6u, 0xbbb7d699u, 0x5a86591bu, 0x10cf5376u, 0x12bf9f4bu,
    0x980fbaa1u, 0x992a4e70u, 0x20fa7ae7u, 0xf7996ebbu, 0xc918a2beu, 0x82de74f2u,
    0xad54209bu, 0xf66b4d74u, 0x1fc5b771u, 0x169d9229u, 0x887761dfu, 0x00b667d5u,
    0xdb425e59u, 0xb72f2844u, 0x9b0ac1f5u, 0x9c737e3au, 0x2b85476cu, 0x6722add6u,
    0x44a63297u, 0x0d688cedu, 0xabc59484u, 0x4107778au, 0x8ad94c6fu, 0xfe83df90u,
    0x0f64053fu, 0xd1292e9du, 0xc5744356u, 0x8dd1abb4u, 0x4c4e7667u, 0xfb4a7fc1u,
    0x74f402cbu, 0x70f06afdu, 0xa82286f2u, 0x918dd076u, 0x7a97c5ceu, 0x48f7bde3u,
    0x6a04d11du, 0xac243ef7u, 0x33ac10cau, 0x2f7a341eu, 0x5f75157au, 0xf4773381u,
    0x591c870eu, 0x78df8cc8u, 0x22f3adb0u, 0x251a5993u, 0x09fbef66u, 0x796942a8u,
    0x97541d2eu, 0x2373daa9u, 0x1bd2f142u, 0xb57e8eb2u, 0xe1a5bfdbu, 0x7d0efa92u,
    0xb3442c94u, 0xd2cb6447u, 0x386ac97eu, 0x66d61805u, 0xbdada15eu, 0x11bc1aa7u,
    0x14e9f6eau, 0xe533a0c0u, 0xf935ee0au, 0x8fee8a04u, 0x810d6d85u, 0x7c68b6d6u,
    0x4edc9aa2u, 0x956e897du, 0xed87581au, 0x264be9d7u, 0xff4ddb29u, 0x823857c2u,
    0xe005a9a0u, 0xf1cc2450u, 0x6f9951e1u, 0xaade2310u, 0xe70c75f5u, 0x83e1a31fu,
    0x4f7dde8eu, 0xf723b563u, 0x368e0928u, 0x86362b71u, 0x21e8982du, 0xdfb3f92bu,
    0x44676352u, 0x99efba31u, 0x2eab4e1cu, 0xfc6ca5e7u, 0x0ebe5d4eu, 0xa0717d0cu,
    0xb64f8199u, 0x946b31a1u, 0x5656cbc6u, 0xcffec3efu, 0x622766c9u, 0xfa211e35u,
    0x52f98b89u, 0x6d01674bu, 0x4978a802u, 0xf651f701u, 0x15b0d43du, 0xd6ff4683u,
    0x3463855fu, 0x672ba29cu, 0xbc128312u, 0x4626a70du, 0xc8927a5au, 0xb8481cf9u,
    0x1c962262u, 0xa21196bau, 0xbaba5ee9u, 0x5bb162d0u, 0x69943bd1u, 0x0c47e35cu,
    0x8cc9619au, 0xe284d948u, 0x271bf264u, 0xc27fb398u, 0x4bc70897u, 0x60cf202cu,
    0x7f42d6aau, 0xa5a13506u, 0x5d3e8860u, 0xcea63d3cu, 0x63bf0a8fu, 0xf02e9efau,
    0xb17b0674u, 0xb072b1d3u, 0x06e5723bu, 0x3737e436u, 0x24aa49c7u, 0x0ded0d18u,
    0xdb256b14u, 0x58b27877u, 0xecb49f54u, 0x6c40256au, 0x6ea92ffbu, 0x3906aa4cu,
    0xc9866fd5u, 0x4549323eu, 0xa7b85fabu, 0x1918cc27u, 0x7308d7b5u, 0x1e16c7adu,
    0x71850b37u, 0x3095fd78u, 0xa63b70e6u, 0xd880e2aeu, 0x3e282769u, 0xa39ba6bcu,
    0x98700fa3u, 0xf34c53e8u, 0x288af426u, 0xb99d930fu, 0xf5b99df1u, 0xe9d0c8cfu,
    0x5ac8405du, 0x50e7217bu, 0x511fbbbeu, 0x2ca2e639u, 0xc020301bu, 0x356dbc00u,
    0x8e43ddb9u, 0x4d327b4au, 0xf20ff3edu, 0x1dbb29bdu, 0x43d44779u, 0xa1b68f70u,
    0x6114455bu, 0xe63d280bu, 0x6bf6ff65u, 0x10fc39e5u, 0x3dae126eu, 0xc1d7cf11u,
    0xcb60b795u, 0x1789d5b3u, 0x9bca36b7u, 0x08306075u, 0x84615608u, 0x8b3a0186u,
    0xe88fbecdu, 0x7ba47c4du, 0x2de44dacu, 0x653fe58du, 0xcca0b968u, 0xd7fa0e72u,
    0x93901780u, 0x1f2c26ccu, 0xae595b6bu, 0xa9ecea9bu, 0xe3dbf8c4u, 0x319cc130u,
    0x12981196u, 0x01a3a4deu, 0x32c454b6u, 0x755bd817u, 0x3cd871e4u, 0xa48bb8dau,
    0x02fdec09u, 0xfd2dc2e2u, 0x9e578088u, 0x9a9f916du, 0x4065fe6cu, 0x1853999eu,
    0xc7793f23u, 0xdc1016bbu, 0x969355ffu, 0x7ef292f6u, 0xcdce4adcu, 0x05e24416u,
    0x85c16c46u, 0xd441d37fu, 0x57bd6855u, 0x8746f54fu, 0x9ca773dfu, 0x770bae22u,
    0x54828413u, 0xb75e4b19u, 0x04c35c03u, 0xbf7cca07u, 0x2955c4ddu, 0x721db041u,
    0xb2394f33u, 0x03f51387u, 0x89b73c9fu, 0x0b1737f3u, 0x07e69024u, 0x9231d245u,
    0x76193861u, 0x88159c15u, 0xdeb552d9u, 0xd9767e40u, 0x20c6c0c3u, 0x4281977cu,
    0xf8afe1e0u, 0xd32a0751u, 0x3fc27432u, 0xddf1dcc5u, 0x68581f34u, 0x3bcd5025u,
    0x0091b2eeu, 0x4aeb6944u, 0x1602e743u, 0xea09eb58u, 0xef0a2a8bu, 0x641e03a5u,
    0xeb50e021u, 0x5c8ccef8u, 0x802ff0b8u, 0xd5e3edfeu, 0xc4dd1b49u, 0x5334cd2au,
    0x13f82d2fu, 0x47450c20u, 0x55dafbd2u, 0xbec0c6f4u, 0xb45d7959u, 0x3ad36e8cu,
    0x0aa8ac57u, 0x1a3c8d73u, 0xe45aafb1u, 0x9f664838u, 0xc6880053u, 0xd0039bbfu,
    0xee5f19ebu, 0xca0041d8u, 0xbbea3aafu, 0xda628291u, 0x9d5c95d4u, 0xadd504a6u,
    0xc39ab482u, 0x5e9e14a4u, 0x2be065f0u, 0x2a13fc3au, 0x9052e8ecu, 0xaf6f5afcu,
    0x519aa8b5u, 0xbb303da9u, 0xe00e2b10u, 0xdfa6c1dbu, 0x2e6b952eu, 0xee10dc23u,
    0x37936d09u, 0x1fc42e92u, 0x39b25a9fu, 0x13ff89f4u, 0xc8f53feau, 0x18500bc7u,
    0x95a0379du, 0x98f751c2u, 0x2289c42fu, 0xa21e4098u, 0x6f391f41u, 0xf27e7e58u,
    0x0d0df887u, 0x4b79d540u, 0x8e8409aau, 0x71fe46f8u, 0x688a9b29u, 0x3f08b548u,
    0x84abe03au, 0x5e91b6c1u, 0xfde4c2aeu, 0x251d0e72u, 0x92d4fee5u, 0xf9371967u,
    0x9175108fu, 0xe6e81835u, 0x8c8cb8eeu, 0xb55a67b3u, 0xcef138ccu, 0x8b256268u,
    0x00d815f5u, 0xe8810812u, 0x77826189u, 0xea73267du, 0x19b90f8du, 0x45c33bb4u,
    0x82477056u, 0xe1770075u, 0x09467aa6u, 0xa7c6f54au, 0x79768742u, 0x61b86bcau,
    0xd6644a44u, 0xe33f0171u, 0xc229fbcdu, 0x41b08febu, 0xd1903e30u, 0x65ec9080u,
    0x563d6fbdu, 0xf56da488u, 0xebf64cd8u, 0x4934426bu, 0x7c8592fcu, 0x6aca8cf2u,
    0x1cea111bu, 0x3a57ee7au, 0xace11c0du, 0x9942d85eu, 0xc4613407u, 0xfa8e643bu,
    0x327fc701u, 0x4ca9be82u, 0x3352526du, 0x2c047f63u, 0xf3a8f7ddu, 0x1a4a98a8u,
    0x762ed4d1u, 0x27c75008u, 0xbdf497c0u, 0x7a7b84dfu, 0x315c28abu, 0x801f93e3u,
    0xf19b0ca1u, 0x8f14e46au, 0xe48ba333u, 0x9605e625u, 0xf03ecb60u, 0x60385f2du,
    0x902845bau, 0x7f96d66fu, 0x24bff05cu, 0x2820730bu, 0x947133cbu, 0xd444828au,
    0xb343f6f1u, 0x0bef4705u, 0x8da574f9u, 0x01e25d6cu, 0x1732793eu, 0x4f0f7b27u,
    0x364b7117u, 0xb2d1da77u, 0xa6c5f1e9u, 0x574ca5b1u, 0x386a3076u, 0xad6894d6u,
    0x1156d7fau, 0xa48d1d9au, 0x4794c0afu, 0x150c0aa0u, 0x26d348acu, 0x29fdeabeu,
    0xa5dede53u, 0x81671e8eu, 0x594ee3bfu, 0xa96c56e6u, 0x3426a726u, 0xc5976579u,
    0xbc22e5e4u, 0xc1006319u, 0xdaafdd2au, 0xa1a1aa83u, 0x3badd0e7u, 0xc3b14981u,
    0xd770b155u, 0xccd7c693u, 0x42e944c5u, 0x03e0064fu, 0xca95b4efu, 0x3dee81c3u,
    0xfbbcd98cu, 0x1e07e15bu, 0x667ce949u, 0xe7d6773fu, 0x21b6124bu, 0x6b2a6ef7u,
    0xd3278a9cu, 0x9a988304u, 0x75d2ae9bu, 0xfe49e2ffu, 0x9bc24f46u, 0x74cc2cf6u,
    0xa3139f36u, 0x6c9ef35au, 0x9fc1dffeu, 0x9e5facdcu, 0xaadc8bbbu, 0x5abdbc5fu,
    0x44b3b390u, 0xf754efa7u, 0x5fe3bdb7u, 0x4e59c886u, 0x06a4c984u, 0xa0338878u,
    0xcd513cd7u, 0x63ebd27eu, 0x8aba80adu, 0x50da144eu, 0x5d9f4e97u, 0x025b751cu,
    0x2d580200u, 0xb6c05837u, 0x580aa15du, 0x54022a6eu, 0xb41a5415u, 0x4863fab6u,
    0xb0b79957u, 0x46d0d159u, 0xdc2b8650u, 0x20a7bb0cu, 0x4a032974u, 0xec8636a2u,
    0x8548f24cu, 0xf6a2bf16u, 0x1088f4b0u, 0x0c2f3a94u, 0x525dc396u, 0x14065785u,
    0x2b4dca52u, 0x08aeed39u, 0xabedfc99u, 0xb1dbcf18u, 0x87f85bbcu, 0xae3aff61u,
    0x433ccd70u, 0x5b23cc64u, 0x7b453213u, 0x5355c545u, 0x9318ec0au, 0x78692d31u,
    0x0a21693du, 0xd5666814u, 0x05fb59d9u, 0xc71985b2u, 0x2abb8e0eu, 0xcf6e6c91u,
    0xd9cfe7c6u, 0xefe7132cu, 0x9711ab28u, 0x3ce52732u, 0x12d516d2u, 0x7209a0d0u,
    0xd278d306u, 0x70fa4b7bu, 0x1d407dd3u, 0xdb0beba4u, 0xbfd97621u, 0xa8be21e1u,
    0x1b6f1b66u, 0x30650ddau, 0xba7ddbb9u, 0x7df953fbu, 0x9d1c3902u, 0xedf0e8d5u,
    0xb8741ae0u, 0x0f240565u, 0x62cd438bu, 0xc616a924u, 0xaf7a96a3u, 0x35365538u,
    0xe583af4du, 0x73415eb8u, 0x23176a47u, 0xfc9ccee8u, 0x7efc9de2u, 0x695e03cfu,
    0xf8ce66d4u, 0x88b4781du, 0x67dd9c03u, 0x3e8f9e73u, 0xc0c95c51u, 0xbe314d22u,
    0x55aa0795u, 0xcb1bb011u, 0xe980fdc8u, 0x9c62b7ceu, 0xde2d239eu, 0x042cadf3u,
    0xffdf04deu, 0x5ce6a60fu, 0xd8c831edu, 0xb7b5b9ecu, 0xb9cbf962u, 0xe253b254u,
    0x0735ba1fu, 0x16ac917fu, 0xdd607c2bu, 0x64a335c4u, 0x40159a7cu, 0x869222f0u,
    0x6ef21769u, 0x839d20a5u, 0xd03b24c9u, 0xf412601eu, 0x6d72a243u, 0x0e018dfdu,
    0x89f3721au, 0xc94f4134u, 0x2f992f20u, 0x4d87253cu
};

// SNEFRU_BE32: assemble a big-endian uint32 from 4 message bytes at p+4*i.
// Mirrors the donor be2me_32(block[i]) (message words are big-endian).
#define SNEFRU_BE32(p, i) \
    ( ((uint)(p)[4*(i)+0] << 24) | ((uint)(p)[4*(i)+1] << 16) \
    | ((uint)(p)[4*(i)+2] <<  8) |  (uint)(p)[4*(i)+3] )

// snefru_block: Snefru core transformation on one data block. `state[8]`
// is the 512-bit hashing state (caller zero-inits before the first block;
// Snefru's IV is all-zero). `block` is the raw data-block bytes: 48 bytes
// for SNE128 (is256=0), 32 bytes for SNE256 (is256=1). Mutates state in
// place. Donor lineage rhash_snefru_process_block snefru.c:765-841.
__attribute__((noinline)) void snefru_block(uint *state, const uchar *block,
                                            int is256) {
    uint W[16];
    uint rot;
    int sbi;

    // Fill W[] (donor :774-792). W[0..3] always come from state[0..3].
    W[0] = state[0]; W[1] = state[1];
    W[2] = state[2]; W[3] = state[3];
    if (is256) {
        // SNE256: W[4..7] from state[4..7]; 8 message words into W[8..15].
        W[4] = state[4]; W[5] = state[5];
        W[6] = state[6]; W[7] = state[7];
        W[ 8] = SNEFRU_BE32(block, 0); W[ 9] = SNEFRU_BE32(block, 1);
        W[10] = SNEFRU_BE32(block, 2); W[11] = SNEFRU_BE32(block, 3);
        W[12] = SNEFRU_BE32(block, 4); W[13] = SNEFRU_BE32(block, 5);
        W[14] = SNEFRU_BE32(block, 6); W[15] = SNEFRU_BE32(block, 7);
    } else {
        // SNE128: 12 message words. First 4 into W[4..7] (donor :783-785),
        // then 8 more into W[8..15] (donor advanced `block += 4`; here we
        // index from word 4 for the W[8..15] group).
        W[4] = SNEFRU_BE32(block, 0); W[5] = SNEFRU_BE32(block, 1);
        W[6] = SNEFRU_BE32(block, 2); W[7] = SNEFRU_BE32(block, 3);
        W[ 8] = SNEFRU_BE32(block, 4); W[ 9] = SNEFRU_BE32(block, 5);
        W[10] = SNEFRU_BE32(block, 6); W[11] = SNEFRU_BE32(block, 7);
        W[12] = SNEFRU_BE32(block, 8); W[13] = SNEFRU_BE32(block, 9);
        W[14] = SNEFRU_BE32(block,10); W[15] = SNEFRU_BE32(block,11);
    }

    // 8 S-box rounds (donor :794-828). sbox advances by 512 each round.
    for (sbi = 0; sbi < 8; sbi++) {
        __constant uint *sbox = SNEFRU_SBOX + 512 * sbi;
        // cycle 4 times: rot = 0x18100810 >> (8*cycle), low byte = shift.
        for (rot = 0x18100810u; rot; rot >>= 8) {
            uint x;
            // SNEFERU_UPDATE_W(i): x = sbox[(i<<7 & 0x100) + (W[i]&0xff)];
            //   W[(i-1)&15] ^= x; if (i>=2) ROTR W[(i-1)&15] by rot;
            //   W[(i+1)&15] ^= x.  (donor macro :802-807)
#define SNEFRU_UPD(i) \
            x = sbox[((((i) << 7) & 0x100) + (W[(i)] & 0xff))]; \
            W[((i) - 1) & 0x0f] ^= x; \
            if ((i) >= 2) W[((i) - 1) & 0x0f] = \
                rotate(W[((i) - 1) & 0x0f], (uint)(32 - ((uchar)rot))); \
            W[((i) + 1) & 0x0f] ^= x;
            SNEFRU_UPD(0);  SNEFRU_UPD(1);  SNEFRU_UPD(2);  SNEFRU_UPD(3);
            SNEFRU_UPD(4);  SNEFRU_UPD(5);  SNEFRU_UPD(6);  SNEFRU_UPD(7);
            SNEFRU_UPD(8);  SNEFRU_UPD(9);  SNEFRU_UPD(10); SNEFRU_UPD(11);
            SNEFRU_UPD(12); SNEFRU_UPD(13); SNEFRU_UPD(14); SNEFRU_UPD(15);
#undef SNEFRU_UPD
            // ROTR W[0] and W[15] by rot (donor :825-826).
            W[ 0] = rotate(W[ 0], (uint)(32 - ((uchar)rot)));
            W[15] = rotate(W[15], (uint)(32 - ((uchar)rot)));
        }
    }

    // Store hashing state (donor :830-840). XOR-back, reversed W order.
    state[0] ^= W[15];
    state[1] ^= W[14];
    state[2] ^= W[13];
    state[3] ^= W[12];
    if (is256) {
        state[4] ^= W[11];
        state[5] ^= W[10];
        state[6] ^= W[ 9];
        state[7] ^= W[ 8];
    }
}
#undef SNEFRU_BE32

/* ---- GOST R 34.11-94 block function (legacy, TEST S-box set) ----
// Phase 5b Tier 4 sub-phase 5b.4b.2 (2026-05-27): lift gost_block from the
// in-tree gosthash/gosthash.c gosthash_compress (the LIVE CPU oracle for
// e125 GOSTMD5PASS via gosthash.o; gosthash() called at mdxfind.c:29076).
// This is the GOST 28147-89 TEST S-box set (Saarinen 1998 / RFC 4357), NOT
// CryptoPro (RHASH_GOST_CRYPTOPRO / e14 GOST-CRYPTO is a DIFFERENT job, out
// of scope). R-Tier4-gost-sbox HIGH: wrong S-box -> silently wrong digests.
// Confirmed via test_gost_vectors.c (4 published TEST-set vectors + 22-len
// cross-check vs rhash RHASH_GOST, zero CryptoPro collisions).
//
// gost_block(state[8], M[8]) mirrors gosthash_compress: it runs the chi
// compression (8-iter U/V key-schedule loop + GOST 28147-89 32-round Feistel
// encipher via the 4 derived S-box tables + the 3 LFSR product-matrix mixing
// stages, gosthash.c:191-260). state and M are both LE-packed uint[8] (the
// message-byte -> LE-word conversion is gosthash_bytes:285-297, done in the
// emit helper). The running mod-2^256 checksum sum[8] AND the dual
// finalization (compress(bit-length) then compress(checksum)) are carried by
// the emit helper, NOT here -- gost_block is a single compression only.
//
// Constant memory budget: GOST_SBOX_1..4 = 4 * 256 * 4 = 4 KB `__constant`.
// Cumulative post-Snefru ~42-43 KB + 4 KB = ~46-47 KB of 64 KB Pascal /
// Apple Silicon budget (fits, ~17 KB headroom). The 4 derived tables are
// precomputed host-side from gosthash_init()'s 8x16 TEST S-box (the standard
// speedup) and baked as literals; byte-exact vs the donor's gost_sbox_1..4.
//
// R6 noinline per feedback_md5_block_noinline_pascal.md (Pascal register
// budget). R8 no nested block comments -- donor stripped, line comments. */
__constant uint GOST_SBOX_1[256] = {
    0x00072000u, 0x00075000u, 0x00074800u, 0x00071000u, 0x00076800u, 0x00074000u,
    0x00070000u, 0x00077000u, 0x00073000u, 0x00075800u, 0x00070800u, 0x00076000u,
    0x00073800u, 0x00077800u, 0x00072800u, 0x00071800u, 0x0005a000u, 0x0005d000u,
    0x0005c800u, 0x00059000u, 0x0005e800u, 0x0005c000u, 0x00058000u, 0x0005f000u,
    0x0005b000u, 0x0005d800u, 0x00058800u, 0x0005e000u, 0x0005b800u, 0x0005f800u,
    0x0005a800u, 0x00059800u, 0x00022000u, 0x00025000u, 0x00024800u, 0x00021000u,
    0x00026800u, 0x00024000u, 0x00020000u, 0x00027000u, 0x00023000u, 0x00025800u,
    0x00020800u, 0x00026000u, 0x00023800u, 0x00027800u, 0x00022800u, 0x00021800u,
    0x00062000u, 0x00065000u, 0x00064800u, 0x00061000u, 0x00066800u, 0x00064000u,
    0x00060000u, 0x00067000u, 0x00063000u, 0x00065800u, 0x00060800u, 0x00066000u,
    0x00063800u, 0x00067800u, 0x00062800u, 0x00061800u, 0x00032000u, 0x00035000u,
    0x00034800u, 0x00031000u, 0x00036800u, 0x00034000u, 0x00030000u, 0x00037000u,
    0x00033000u, 0x00035800u, 0x00030800u, 0x00036000u, 0x00033800u, 0x00037800u,
    0x00032800u, 0x00031800u, 0x0006a000u, 0x0006d000u, 0x0006c800u, 0x00069000u,
    0x0006e800u, 0x0006c000u, 0x00068000u, 0x0006f000u, 0x0006b000u, 0x0006d800u,
    0x00068800u, 0x0006e000u, 0x0006b800u, 0x0006f800u, 0x0006a800u, 0x00069800u,
    0x0007a000u, 0x0007d000u, 0x0007c800u, 0x00079000u, 0x0007e800u, 0x0007c000u,
    0x00078000u, 0x0007f000u, 0x0007b000u, 0x0007d800u, 0x00078800u, 0x0007e000u,
    0x0007b800u, 0x0007f800u, 0x0007a800u, 0x00079800u, 0x00052000u, 0x00055000u,
    0x00054800u, 0x00051000u, 0x00056800u, 0x00054000u, 0x00050000u, 0x00057000u,
    0x00053000u, 0x00055800u, 0x00050800u, 0x00056000u, 0x00053800u, 0x00057800u,
    0x00052800u, 0x00051800u, 0x00012000u, 0x00015000u, 0x00014800u, 0x00011000u,
    0x00016800u, 0x00014000u, 0x00010000u, 0x00017000u, 0x00013000u, 0x00015800u,
    0x00010800u, 0x00016000u, 0x00013800u, 0x00017800u, 0x00012800u, 0x00011800u,
    0x0001a000u, 0x0001d000u, 0x0001c800u, 0x00019000u, 0x0001e800u, 0x0001c000u,
    0x00018000u, 0x0001f000u, 0x0001b000u, 0x0001d800u, 0x00018800u, 0x0001e000u,
    0x0001b800u, 0x0001f800u, 0x0001a800u, 0x00019800u, 0x00042000u, 0x00045000u,
    0x00044800u, 0x00041000u, 0x00046800u, 0x00044000u, 0x00040000u, 0x00047000u,
    0x00043000u, 0x00045800u, 0x00040800u, 0x00046000u, 0x00043800u, 0x00047800u,
    0x00042800u, 0x00041800u, 0x0000a000u, 0x0000d000u, 0x0000c800u, 0x00009000u,
    0x0000e800u, 0x0000c000u, 0x00008000u, 0x0000f000u, 0x0000b000u, 0x0000d800u,
    0x00008800u, 0x0000e000u, 0x0000b800u, 0x0000f800u, 0x0000a800u, 0x00009800u,
    0x00002000u, 0x00005000u, 0x00004800u, 0x00001000u, 0x00006800u, 0x00004000u,
    0x00000000u, 0x00007000u, 0x00003000u, 0x00005800u, 0x00000800u, 0x00006000u,
    0x00003800u, 0x00007800u, 0x00002800u, 0x00001800u, 0x0003a000u, 0x0003d000u,
    0x0003c800u, 0x00039000u, 0x0003e800u, 0x0003c000u, 0x00038000u, 0x0003f000u,
    0x0003b000u, 0x0003d800u, 0x00038800u, 0x0003e000u, 0x0003b800u, 0x0003f800u,
    0x0003a800u, 0x00039800u, 0x0002a000u, 0x0002d000u, 0x0002c800u, 0x00029000u,
    0x0002e800u, 0x0002c000u, 0x00028000u, 0x0002f000u, 0x0002b000u, 0x0002d800u,
    0x00028800u, 0x0002e000u, 0x0002b800u, 0x0002f800u, 0x0002a800u, 0x00029800u,
    0x0004a000u, 0x0004d000u, 0x0004c800u, 0x00049000u, 0x0004e800u, 0x0004c000u,
    0x00048000u, 0x0004f000u, 0x0004b000u, 0x0004d800u, 0x00048800u, 0x0004e000u,
    0x0004b800u, 0x0004f800u, 0x0004a800u, 0x00049800u
};
__constant uint GOST_SBOX_2[256] = {
    0x03a80000u, 0x03c00000u, 0x03880000u, 0x03e80000u, 0x03d00000u, 0x03980000u,
    0x03a00000u, 0x03900000u, 0x03f00000u, 0x03f80000u, 0x03e00000u, 0x03b80000u,
    0x03b00000u, 0x03800000u, 0x03c80000u, 0x03d80000u, 0x06a80000u, 0x06c00000u,
    0x06880000u, 0x06e80000u, 0x06d00000u, 0x06980000u, 0x06a00000u, 0x06900000u,
    0x06f00000u, 0x06f80000u, 0x06e00000u, 0x06b80000u, 0x06b00000u, 0x06800000u,
    0x06c80000u, 0x06d80000u, 0x05280000u, 0x05400000u, 0x05080000u, 0x05680000u,
    0x05500000u, 0x05180000u, 0x05200000u, 0x05100000u, 0x05700000u, 0x05780000u,
    0x05600000u, 0x05380000u, 0x05300000u, 0x05000000u, 0x05480000u, 0x05580000u,
    0x00a80000u, 0x00c00000u, 0x00880000u, 0x00e80000u, 0x00d00000u, 0x00980000u,
    0x00a00000u, 0x00900000u, 0x00f00000u, 0x00f80000u, 0x00e00000u, 0x00b80000u,
    0x00b00000u, 0x00800000u, 0x00c80000u, 0x00d80000u, 0x00280000u, 0x00400000u,
    0x00080000u, 0x00680000u, 0x00500000u, 0x00180000u, 0x00200000u, 0x00100000u,
    0x00700000u, 0x00780000u, 0x00600000u, 0x00380000u, 0x00300000u, 0x00000000u,
    0x00480000u, 0x00580000u, 0x04280000u, 0x04400000u, 0x04080000u, 0x04680000u,
    0x04500000u, 0x04180000u, 0x04200000u, 0x04100000u, 0x04700000u, 0x04780000u,
    0x04600000u, 0x04380000u, 0x04300000u, 0x04000000u, 0x04480000u, 0x04580000u,
    0x04a80000u, 0x04c00000u, 0x04880000u, 0x04e80000u, 0x04d00000u, 0x04980000u,
    0x04a00000u, 0x04900000u, 0x04f00000u, 0x04f80000u, 0x04e00000u, 0x04b80000u,
    0x04b00000u, 0x04800000u, 0x04c80000u, 0x04d80000u, 0x07a80000u, 0x07c00000u,
    0x07880000u, 0x07e80000u, 0x07d00000u, 0x07980000u, 0x07a00000u, 0x07900000u,
    0x07f00000u, 0x07f80000u, 0x07e00000u, 0x07b80000u, 0x07b00000u, 0x07800000u,
    0x07c80000u, 0x07d80000u, 0x07280000u, 0x07400000u, 0x07080000u, 0x07680000u,
    0x07500000u, 0x07180000u, 0x07200000u, 0x07100000u, 0x07700000u, 0x07780000u,
    0x07600000u, 0x07380000u, 0x07300000u, 0x07000000u, 0x07480000u, 0x07580000u,
    0x02280000u, 0x02400000u, 0x02080000u, 0x02680000u, 0x02500000u, 0x02180000u,
    0x02200000u, 0x02100000u, 0x02700000u, 0x02780000u, 0x02600000u, 0x02380000u,
    0x02300000u, 0x02000000u, 0x02480000u, 0x02580000u, 0x03280000u, 0x03400000u,
    0x03080000u, 0x03680000u, 0x03500000u, 0x03180000u, 0x03200000u, 0x03100000u,
    0x03700000u, 0x03780000u, 0x03600000u, 0x03380000u, 0x03300000u, 0x03000000u,
    0x03480000u, 0x03580000u, 0x06280000u, 0x06400000u, 0x06080000u, 0x06680000u,
    0x06500000u, 0x06180000u, 0x06200000u, 0x06100000u, 0x06700000u, 0x06780000u,
    0x06600000u, 0x06380000u, 0x06300000u, 0x06000000u, 0x06480000u, 0x06580000u,
    0x05a80000u, 0x05c00000u, 0x05880000u, 0x05e80000u, 0x05d00000u, 0x05980000u,
    0x05a00000u, 0x05900000u, 0x05f00000u, 0x05f80000u, 0x05e00000u, 0x05b80000u,
    0x05b00000u, 0x05800000u, 0x05c80000u, 0x05d80000u, 0x01280000u, 0x01400000u,
    0x01080000u, 0x01680000u, 0x01500000u, 0x01180000u, 0x01200000u, 0x01100000u,
    0x01700000u, 0x01780000u, 0x01600000u, 0x01380000u, 0x01300000u, 0x01000000u,
    0x01480000u, 0x01580000u, 0x02a80000u, 0x02c00000u, 0x02880000u, 0x02e80000u,
    0x02d00000u, 0x02980000u, 0x02a00000u, 0x02900000u, 0x02f00000u, 0x02f80000u,
    0x02e00000u, 0x02b80000u, 0x02b00000u, 0x02800000u, 0x02c80000u, 0x02d80000u,
    0x01a80000u, 0x01c00000u, 0x01880000u, 0x01e80000u, 0x01d00000u, 0x01980000u,
    0x01a00000u, 0x01900000u, 0x01f00000u, 0x01f80000u, 0x01e00000u, 0x01b80000u,
    0x01b00000u, 0x01800000u, 0x01c80000u, 0x01d80000u
};
__constant uint GOST_SBOX_3[256] = {
    0x30000002u, 0x60000002u, 0x38000002u, 0x08000002u, 0x28000002u, 0x78000002u,
    0x68000002u, 0x40000002u, 0x20000002u, 0x50000002u, 0x48000002u, 0x70000002u,
    0x00000002u, 0x18000002u, 0x58000002u, 0x10000002u, 0xb0000005u, 0xe0000005u,
    0xb8000005u, 0x88000005u, 0xa8000005u, 0xf8000005u, 0xe8000005u, 0xc0000005u,
    0xa0000005u, 0xd0000005u, 0xc8000005u, 0xf0000005u, 0x80000005u, 0x98000005u,
    0xd8000005u, 0x90000005u, 0x30000005u, 0x60000005u, 0x38000005u, 0x08000005u,
    0x28000005u, 0x78000005u, 0x68000005u, 0x40000005u, 0x20000005u, 0x50000005u,
    0x48000005u, 0x70000005u, 0x00000005u, 0x18000005u, 0x58000005u, 0x10000005u,
    0x30000000u, 0x60000000u, 0x38000000u, 0x08000000u, 0x28000000u, 0x78000000u,
    0x68000000u, 0x40000000u, 0x20000000u, 0x50000000u, 0x48000000u, 0x70000000u,
    0x00000000u, 0x18000000u, 0x58000000u, 0x10000000u, 0xb0000003u, 0xe0000003u,
    0xb8000003u, 0x88000003u, 0xa8000003u, 0xf8000003u, 0xe8000003u, 0xc0000003u,
    0xa0000003u, 0xd0000003u, 0xc8000003u, 0xf0000003u, 0x80000003u, 0x98000003u,
    0xd8000003u, 0x90000003u, 0x30000001u, 0x60000001u, 0x38000001u, 0x08000001u,
    0x28000001u, 0x78000001u, 0x68000001u, 0x40000001u, 0x20000001u, 0x50000001u,
    0x48000001u, 0x70000001u, 0x00000001u, 0x18000001u, 0x58000001u, 0x10000001u,
    0xb0000000u, 0xe0000000u, 0xb8000000u, 0x88000000u, 0xa8000000u, 0xf8000000u,
    0xe8000000u, 0xc0000000u, 0xa0000000u, 0xd0000000u, 0xc8000000u, 0xf0000000u,
    0x80000000u, 0x98000000u, 0xd8000000u, 0x90000000u, 0xb0000006u, 0xe0000006u,
    0xb8000006u, 0x88000006u, 0xa8000006u, 0xf8000006u, 0xe8000006u, 0xc0000006u,
    0xa0000006u, 0xd0000006u, 0xc8000006u, 0xf0000006u, 0x80000006u, 0x98000006u,
    0xd8000006u, 0x90000006u, 0xb0000001u, 0xe0000001u, 0xb8000001u, 0x88000001u,
    0xa8000001u, 0xf8000001u, 0xe8000001u, 0xc0000001u, 0xa0000001u, 0xd0000001u,
    0xc8000001u, 0xf0000001u, 0x80000001u, 0x98000001u, 0xd8000001u, 0x90000001u,
    0x30000003u, 0x60000003u, 0x38000003u, 0x08000003u, 0x28000003u, 0x78000003u,
    0x68000003u, 0x40000003u, 0x20000003u, 0x50000003u, 0x48000003u, 0x70000003u,
    0x00000003u, 0x18000003u, 0x58000003u, 0x10000003u, 0x30000004u, 0x60000004u,
    0x38000004u, 0x08000004u, 0x28000004u, 0x78000004u, 0x68000004u, 0x40000004u,
    0x20000004u, 0x50000004u, 0x48000004u, 0x70000004u, 0x00000004u, 0x18000004u,
    0x58000004u, 0x10000004u, 0xb0000002u, 0xe0000002u, 0xb8000002u, 0x88000002u,
    0xa8000002u, 0xf8000002u, 0xe8000002u, 0xc0000002u, 0xa0000002u, 0xd0000002u,
    0xc8000002u, 0xf0000002u, 0x80000002u, 0x98000002u, 0xd8000002u, 0x90000002u,
    0xb0000004u, 0xe0000004u, 0xb8000004u, 0x88000004u, 0xa8000004u, 0xf8000004u,
    0xe8000004u, 0xc0000004u, 0xa0000004u, 0xd0000004u, 0xc8000004u, 0xf0000004u,
    0x80000004u, 0x98000004u, 0xd8000004u, 0x90000004u, 0x30000006u, 0x60000006u,
    0x38000006u, 0x08000006u, 0x28000006u, 0x78000006u, 0x68000006u, 0x40000006u,
    0x20000006u, 0x50000006u, 0x48000006u, 0x70000006u, 0x00000006u, 0x18000006u,
    0x58000006u, 0x10000006u, 0xb0000007u, 0xe0000007u, 0xb8000007u, 0x88000007u,
    0xa8000007u, 0xf8000007u, 0xe8000007u, 0xc0000007u, 0xa0000007u, 0xd0000007u,
    0xc8000007u, 0xf0000007u, 0x80000007u, 0x98000007u, 0xd8000007u, 0x90000007u,
    0x30000007u, 0x60000007u, 0x38000007u, 0x08000007u, 0x28000007u, 0x78000007u,
    0x68000007u, 0x40000007u, 0x20000007u, 0x50000007u, 0x48000007u, 0x70000007u,
    0x00000007u, 0x18000007u, 0x58000007u, 0x10000007u
};
__constant uint GOST_SBOX_4[256] = {
    0x000000e8u, 0x000000d8u, 0x000000a0u, 0x00000088u, 0x00000098u, 0x000000f8u,
    0x000000a8u, 0x000000c8u, 0x00000080u, 0x000000d0u, 0x000000f0u, 0x000000b8u,
    0x000000b0u, 0x000000c0u, 0x00000090u, 0x000000e0u, 0x000007e8u, 0x000007d8u,
    0x000007a0u, 0x00000788u, 0x00000798u, 0x000007f8u, 0x000007a8u, 0x000007c8u,
    0x00000780u, 0x000007d0u, 0x000007f0u, 0x000007b8u, 0x000007b0u, 0x000007c0u,
    0x00000790u, 0x000007e0u, 0x000006e8u, 0x000006d8u, 0x000006a0u, 0x00000688u,
    0x00000698u, 0x000006f8u, 0x000006a8u, 0x000006c8u, 0x00000680u, 0x000006d0u,
    0x000006f0u, 0x000006b8u, 0x000006b0u, 0x000006c0u, 0x00000690u, 0x000006e0u,
    0x00000068u, 0x00000058u, 0x00000020u, 0x00000008u, 0x00000018u, 0x00000078u,
    0x00000028u, 0x00000048u, 0x00000000u, 0x00000050u, 0x00000070u, 0x00000038u,
    0x00000030u, 0x00000040u, 0x00000010u, 0x00000060u, 0x000002e8u, 0x000002d8u,
    0x000002a0u, 0x00000288u, 0x00000298u, 0x000002f8u, 0x000002a8u, 0x000002c8u,
    0x00000280u, 0x000002d0u, 0x000002f0u, 0x000002b8u, 0x000002b0u, 0x000002c0u,
    0x00000290u, 0x000002e0u, 0x000003e8u, 0x000003d8u, 0x000003a0u, 0x00000388u,
    0x00000398u, 0x000003f8u, 0x000003a8u, 0x000003c8u, 0x00000380u, 0x000003d0u,
    0x000003f0u, 0x000003b8u, 0x000003b0u, 0x000003c0u, 0x00000390u, 0x000003e0u,
    0x00000568u, 0x00000558u, 0x00000520u, 0x00000508u, 0x00000518u, 0x00000578u,
    0x00000528u, 0x00000548u, 0x00000500u, 0x00000550u, 0x00000570u, 0x00000538u,
    0x00000530u, 0x00000540u, 0x00000510u, 0x00000560u, 0x00000268u, 0x00000258u,
    0x00000220u, 0x00000208u, 0x00000218u, 0x00000278u, 0x00000228u, 0x00000248u,
    0x00000200u, 0x00000250u, 0x00000270u, 0x00000238u, 0x00000230u, 0x00000240u,
    0x00000210u, 0x00000260u, 0x000004e8u, 0x000004d8u, 0x000004a0u, 0x00000488u,
    0x00000498u, 0x000004f8u, 0x000004a8u, 0x000004c8u, 0x00000480u, 0x000004d0u,
    0x000004f0u, 0x000004b8u, 0x000004b0u, 0x000004c0u, 0x00000490u, 0x000004e0u,
    0x00000168u, 0x00000158u, 0x00000120u, 0x00000108u, 0x00000118u, 0x00000178u,
    0x00000128u, 0x00000148u, 0x00000100u, 0x00000150u, 0x00000170u, 0x00000138u,
    0x00000130u, 0x00000140u, 0x00000110u, 0x00000160u, 0x000001e8u, 0x000001d8u,
    0x000001a0u, 0x00000188u, 0x00000198u, 0x000001f8u, 0x000001a8u, 0x000001c8u,
    0x00000180u, 0x000001d0u, 0x000001f0u, 0x000001b8u, 0x000001b0u, 0x000001c0u,
    0x00000190u, 0x000001e0u, 0x00000768u, 0x00000758u, 0x00000720u, 0x00000708u,
    0x00000718u, 0x00000778u, 0x00000728u, 0x00000748u, 0x00000700u, 0x00000750u,
    0x00000770u, 0x00000738u, 0x00000730u, 0x00000740u, 0x00000710u, 0x00000760u,
    0x00000368u, 0x00000358u, 0x00000320u, 0x00000308u, 0x00000318u, 0x00000378u,
    0x00000328u, 0x00000348u, 0x00000300u, 0x00000350u, 0x00000370u, 0x00000338u,
    0x00000330u, 0x00000340u, 0x00000310u, 0x00000360u, 0x000005e8u, 0x000005d8u,
    0x000005a0u, 0x00000588u, 0x00000598u, 0x000005f8u, 0x000005a8u, 0x000005c8u,
    0x00000580u, 0x000005d0u, 0x000005f0u, 0x000005b8u, 0x000005b0u, 0x000005c0u,
    0x00000590u, 0x000005e0u, 0x00000468u, 0x00000458u, 0x00000420u, 0x00000408u,
    0x00000418u, 0x00000478u, 0x00000428u, 0x00000448u, 0x00000400u, 0x00000450u,
    0x00000470u, 0x00000438u, 0x00000430u, 0x00000440u, 0x00000410u, 0x00000460u,
    0x00000668u, 0x00000658u, 0x00000620u, 0x00000608u, 0x00000618u, 0x00000678u,
    0x00000628u, 0x00000648u, 0x00000600u, 0x00000650u, 0x00000670u, 0x00000638u,
    0x00000630u, 0x00000640u, 0x00000610u, 0x00000660u
};

// GOST_ENCRYPT_ROUND inlined (donor gosthash.c:71-77): one Feistel round pair.
#define GOST_GE_ROUND(k1, k2) \
    t = (k1) + r; \
    l ^= GOST_SBOX_1[t & 0xff] ^ GOST_SBOX_2[(t >> 8) & 0xff] ^ \
         GOST_SBOX_3[(t >> 16) & 0xff] ^ GOST_SBOX_4[t >> 24]; \
    t = (k2) + l; \
    r ^= GOST_SBOX_1[t & 0xff] ^ GOST_SBOX_2[(t >> 8) & 0xff] ^ \
         GOST_SBOX_3[(t >> 16) & 0xff] ^ GOST_SBOX_4[t >> 24];

// GOST_ENCRYPT (donor :81-100): 32 rounds = 3x forward key order + 1x reverse,
// then swap r/l.
#define GOST_GE_ENCRYPT(key) \
    GOST_GE_ROUND(key[0], key[1]) GOST_GE_ROUND(key[2], key[3]) \
    GOST_GE_ROUND(key[4], key[5]) GOST_GE_ROUND(key[6], key[7]) \
    GOST_GE_ROUND(key[0], key[1]) GOST_GE_ROUND(key[2], key[3]) \
    GOST_GE_ROUND(key[4], key[5]) GOST_GE_ROUND(key[6], key[7]) \
    GOST_GE_ROUND(key[0], key[1]) GOST_GE_ROUND(key[2], key[3]) \
    GOST_GE_ROUND(key[4], key[5]) GOST_GE_ROUND(key[6], key[7]) \
    GOST_GE_ROUND(key[7], key[6]) GOST_GE_ROUND(key[5], key[4]) \
    GOST_GE_ROUND(key[3], key[2]) GOST_GE_ROUND(key[1], key[0]) \
    t = r; r = l; l = t;

__attribute__((noinline)) void gost_block(uint *h, const uint *m) {
    int i;
    uint l, r, t, key[8], u[8], v[8], w[8], s[8];

    for (i = 0; i < 8; i++) u[i] = h[i];
    for (i = 0; i < 8; i++) v[i] = m[i];

    // chi compression: 8-iter U/V key-schedule loop (donor :114-189).
    for (i = 0; i < 8; i += 2) {
        w[0] = u[0] ^ v[0]; w[1] = u[1] ^ v[1];
        w[2] = u[2] ^ v[2]; w[3] = u[3] ^ v[3];
        w[4] = u[4] ^ v[4]; w[5] = u[5] ^ v[5];
        w[6] = u[6] ^ v[6]; w[7] = u[7] ^ v[7];

        // P-Transformation (donor :127-142).
        key[0] = (w[0]  & 0x000000ff) | ((w[2] & 0x000000ff) << 8) |
            ((w[4] & 0x000000ff) << 16) | ((w[6] & 0x000000ff) << 24);
        key[1] = ((w[0] & 0x0000ff00) >> 8)  | (w[2]  & 0x0000ff00) |
            ((w[4] & 0x0000ff00) << 8) | ((w[6] & 0x0000ff00) << 16);
        key[2] = ((w[0] & 0x00ff0000) >> 16) | ((w[2] & 0x00ff0000) >> 8) |
            (w[4] & 0x00ff0000) | ((w[6] & 0x00ff0000) << 8);
        key[3] = ((w[0] & 0xff000000) >> 24) | ((w[2] & 0xff000000) >> 16) |
            ((w[4] & 0xff000000) >> 8) | (w[6] & 0xff000000);
        key[4] = (w[1] & 0x000000ff) | ((w[3] & 0x000000ff) << 8) |
            ((w[5] & 0x000000ff) << 16) | ((w[7] & 0x000000ff) << 24);
        key[5] = ((w[1] & 0x0000ff00) >> 8) | (w[3]  & 0x0000ff00) |
            ((w[5] & 0x0000ff00) << 8) | ((w[7] & 0x0000ff00) << 16);
        key[6] = ((w[1] & 0x00ff0000) >> 16) | ((w[3] & 0x00ff0000) >> 8) |
            (w[5] & 0x00ff0000) | ((w[7] & 0x00ff0000) << 8);
        key[7] = ((w[1] & 0xff000000) >> 24) | ((w[3] & 0xff000000) >> 16) |
            ((w[5] & 0xff000000) >> 8) | (w[7] & 0xff000000);

        // enciphering transformation (donor :144-149).
        r = h[i]; l = h[i + 1];
        GOST_GE_ENCRYPT(key);
        s[i] = r; s[i + 1] = l;

        if (i == 6) break;

        // U = A(U) (donor :154-163).
        l = u[0] ^ u[2]; r = u[1] ^ u[3];
        u[0] = u[2]; u[1] = u[3]; u[2] = u[4]; u[3] = u[5];
        u[4] = u[6]; u[5] = u[7]; u[6] = l; u[7] = r;

        // Constant C_3 (donor :165-175), applied only after the i==2 iter.
        if (i == 2) {
            u[0] ^= 0xff00ff00; u[1] ^= 0xff00ff00;
            u[2] ^= 0x00ff00ff; u[3] ^= 0x00ff00ff;
            u[4] ^= 0x00ffff00; u[5] ^= 0xff0000ff;
            u[6] ^= 0x000000ff; u[7] ^= 0xff00ffff;
        }

        // V = A(A(V)) (donor :177-188).
        l = v[0]; r = v[2];
        v[0] = v[4]; v[2] = v[6];
        v[4] = l ^ r; v[6] = v[0] ^ r;
        l = v[1]; r = v[3];
        v[1] = v[5]; v[3] = v[7];
        v[5] = l ^ r; v[7] = v[1] ^ r;
    }

    // 12 rounds of the LFSR (product matrix) and xor in M (donor :193-217).
    u[0] = m[0] ^ s[6];
    u[1] = m[1] ^ s[7];
    u[2] = m[2] ^ (s[0] << 16) ^ (s[0] >> 16) ^ (s[0] & 0xffff) ^
        (s[1] & 0xffff) ^ (s[1] >> 16) ^ (s[2] << 16) ^ s[6] ^ (s[6] << 16) ^
        (s[7] & 0xffff0000) ^ (s[7] >> 16);
    u[3] = m[3] ^ (s[0] & 0xffff) ^ (s[0] << 16) ^ (s[1] & 0xffff) ^
        (s[1] << 16) ^ (s[1] >> 16) ^ (s[2] << 16) ^ (s[2] >> 16) ^
        (s[3] << 16) ^ s[6] ^ (s[6] << 16) ^ (s[6] >> 16) ^ (s[7] & 0xffff) ^
        (s[7] << 16) ^ (s[7] >> 16);
    u[4] = m[4] ^
        (s[0] & 0xffff0000) ^ (s[0] << 16) ^ (s[0] >> 16) ^
        (s[1] & 0xffff0000) ^ (s[1] >> 16) ^ (s[2] << 16) ^ (s[2] >> 16) ^
        (s[3] << 16) ^ (s[3] >> 16) ^ (s[4] << 16) ^ (s[6] << 16) ^
        (s[6] >> 16) ^(s[7] & 0xffff) ^ (s[7] << 16) ^ (s[7] >> 16);
    u[5] = m[5] ^ (s[0] << 16) ^ (s[0] >> 16) ^ (s[0] & 0xffff0000) ^
        (s[1] & 0xffff) ^ s[2] ^ (s[2] >> 16) ^ (s[3] << 16) ^ (s[3] >> 16) ^
        (s[4] << 16) ^ (s[4] >> 16) ^ (s[5] << 16) ^  (s[6] << 16) ^
        (s[6] >> 16) ^ (s[7] & 0xffff0000) ^ (s[7] << 16) ^ (s[7] >> 16);
    u[6] = m[6] ^ s[0] ^ (s[1] >> 16) ^ (s[2] << 16) ^ s[3] ^ (s[3] >> 16) ^
        (s[4] << 16) ^ (s[4] >> 16) ^ (s[5] << 16) ^ (s[5] >> 16) ^ s[6] ^
        (s[6] << 16) ^ (s[6] >> 16) ^ (s[7] << 16);
    u[7] = m[7] ^ (s[0] & 0xffff0000) ^ (s[0] << 16) ^ (s[1] & 0xffff) ^
        (s[1] << 16) ^ (s[2] >> 16) ^ (s[3] << 16) ^ s[4] ^ (s[4] >> 16) ^
        (s[5] << 16) ^ (s[5] >> 16) ^ (s[6] >> 16) ^ (s[7] & 0xffff) ^
        (s[7] << 16) ^ (s[7] >> 16);

    // 16 * 1 round of the LFSR and xor in H (donor :221-229).
    v[0] = h[0] ^ (u[1] << 16) ^ (u[0] >> 16);
    v[1] = h[1] ^ (u[2] << 16) ^ (u[1] >> 16);
    v[2] = h[2] ^ (u[3] << 16) ^ (u[2] >> 16);
    v[3] = h[3] ^ (u[4] << 16) ^ (u[3] >> 16);
    v[4] = h[4] ^ (u[5] << 16) ^ (u[4] >> 16);
    v[5] = h[5] ^ (u[6] << 16) ^ (u[5] >> 16);
    v[6] = h[6] ^ (u[7] << 16) ^ (u[6] >> 16);
    v[7] = h[7] ^ (u[0] & 0xffff0000) ^ (u[0] << 16) ^ (u[7] >> 16) ^
        (u[1] & 0xffff0000) ^ (u[1] << 16) ^ (u[6] << 16) ^ (u[7] & 0xffff0000);

    // 61 rounds of LFSR, mixing up h (product matrix) (donor :233-260).
    h[0] = (v[0] & 0xffff0000) ^ (v[0] << 16) ^ (v[0] >> 16) ^ (v[1] >> 16) ^
        (v[1] & 0xffff0000) ^ (v[2] << 16) ^ (v[3] >> 16) ^ (v[4] << 16) ^
        (v[5] >> 16) ^ v[5] ^ (v[6] >> 16) ^ (v[7] << 16) ^ (v[7] >> 16) ^
        (v[7] & 0xffff);
    h[1] = (v[0] << 16) ^ (v[0] >> 16) ^ (v[0] & 0xffff0000) ^ (v[1] & 0xffff) ^
        v[2] ^ (v[2] >> 16) ^ (v[3] << 16) ^ (v[4] >> 16) ^ (v[5] << 16) ^
        (v[6] << 16) ^ v[6] ^ (v[7] & 0xffff0000) ^ (v[7] >> 16);
    h[2] = (v[0] & 0xffff) ^ (v[0] << 16) ^ (v[1] << 16) ^ (v[1] >> 16) ^
        (v[1] & 0xffff0000) ^ (v[2] << 16) ^ (v[3] >> 16) ^ v[3] ^ (v[4] << 16) ^
        (v[5] >> 16) ^ v[6] ^ (v[6] >> 16) ^ (v[7] & 0xffff) ^ (v[7] << 16) ^
        (v[7] >> 16);
    h[3] = (v[0] << 16) ^ (v[0] >> 16) ^ (v[0] & 0xffff0000) ^
        (v[1] & 0xffff0000) ^ (v[1] >> 16) ^ (v[2] << 16) ^ (v[2] >> 16) ^ v[2] ^
        (v[3] << 16) ^ (v[4] >> 16) ^ v[4] ^ (v[5] << 16) ^ (v[6] << 16) ^
        (v[7] & 0xffff) ^ (v[7] >> 16);
    h[4] = (v[0] >> 16) ^ (v[1] << 16) ^ v[1] ^ (v[2] >> 16) ^ v[2] ^
        (v[3] << 16) ^ (v[3] >> 16) ^ v[3] ^ (v[4] << 16) ^ (v[5] >> 16) ^
        v[5] ^ (v[6] << 16) ^ (v[6] >> 16) ^ (v[7] << 16);
    h[5] = (v[0] << 16) ^ (v[0] & 0xffff0000) ^ (v[1] << 16) ^ (v[1] >> 16) ^
        (v[1] & 0xffff0000) ^ (v[2] << 16) ^ v[2] ^ (v[3] >> 16) ^ v[3] ^
        (v[4] << 16) ^ (v[4] >> 16) ^ v[4] ^ (v[5] << 16) ^ (v[6] << 16) ^
        (v[6] >> 16) ^ v[6] ^ (v[7] << 16) ^ (v[7] >> 16) ^ (v[7] & 0xffff0000);
    h[6] = v[0] ^ v[2] ^ (v[2] >> 16) ^ v[3] ^ (v[3] << 16) ^ v[4] ^
        (v[4] >> 16) ^ (v[5] << 16) ^ (v[5] >> 16) ^ v[5] ^ (v[6] << 16) ^
        (v[6] >> 16) ^ v[6] ^ (v[7] << 16) ^ v[7];
    h[7] = v[0] ^ (v[0] >> 16) ^ (v[1] << 16) ^ (v[1] >> 16) ^ (v[2] << 16) ^
        (v[3] >> 16) ^ v[3] ^ (v[4] << 16) ^ v[4] ^ (v[5] >> 16) ^ v[5] ^
        (v[6] << 16) ^ (v[6] >> 16) ^ (v[7] << 16) ^ v[7];
}
#undef GOST_GE_ROUND
#undef GOST_GE_ENCRYPT

/* ---- RIPEMD-160 block function ---- */

#define RMD_F1(x, y, z) ((x) ^ (y) ^ (z))
#define RMD_F2(x, y, z) ((((y) ^ (z)) & (x)) ^ (z))
#define RMD_F3(x, y, z) (((x) | ~(y)) ^ (z))
#define RMD_F4(x, y, z) ((((x) ^ (y)) & (z)) ^ (y))
#define RMD_F5(x, y, z) ((x) ^ ((y) | ~(z)))

#define RMD_STEP(FUNC, A, B, C, D, E, X, S, K) \
    (A) += FUNC((B), (C), (D)) + (X) + K; \
    (A) = rotate((A), (uint)(S)) + (E); \
    (C) = rotate((C), (uint)10);

#define L1(A,B,C,D,E,X,S) RMD_STEP(RMD_F1,A,B,C,D,E,X,S,0u)
#define L2(A,B,C,D,E,X,S) RMD_STEP(RMD_F2,A,B,C,D,E,X,S,0x5a827999u)
#define L3(A,B,C,D,E,X,S) RMD_STEP(RMD_F3,A,B,C,D,E,X,S,0x6ed9eba1u)
#define L4(A,B,C,D,E,X,S) RMD_STEP(RMD_F4,A,B,C,D,E,X,S,0x8f1bbcdcu)
#define L5(A,B,C,D,E,X,S) RMD_STEP(RMD_F5,A,B,C,D,E,X,S,0xa953fd4eu)
#define R1(A,B,C,D,E,X,S) RMD_STEP(RMD_F5,A,B,C,D,E,X,S,0x50a28be6u)
#define R2(A,B,C,D,E,X,S) RMD_STEP(RMD_F4,A,B,C,D,E,X,S,0x5c4dd124u)
#define R3(A,B,C,D,E,X,S) RMD_STEP(RMD_F3,A,B,C,D,E,X,S,0x6d703ef3u)
#define R4(A,B,C,D,E,X,S) RMD_STEP(RMD_F2,A,B,C,D,E,X,S,0x7a6d76e9u)
#define R5(A,B,C,D,E,X,S) RMD_STEP(RMD_F1,A,B,C,D,E,X,S,0u)

void rmd160_block(uint *hash, const uint *X) {
    uint A = hash[0], B = hash[1], C = hash[2], D = hash[3], E = hash[4];
    uint a1, b1, c1, d1, e1;
    L1(A,B,C,D,E,X[0],11);L1(E,A,B,C,D,X[1],14);L1(D,E,A,B,C,X[2],15);L1(C,D,E,A,B,X[3],12);
    L1(B,C,D,E,A,X[4],5);L1(A,B,C,D,E,X[5],8);L1(E,A,B,C,D,X[6],7);L1(D,E,A,B,C,X[7],9);
    L1(C,D,E,A,B,X[8],11);L1(B,C,D,E,A,X[9],13);L1(A,B,C,D,E,X[10],14);L1(E,A,B,C,D,X[11],15);
    L1(D,E,A,B,C,X[12],6);L1(C,D,E,A,B,X[13],7);L1(B,C,D,E,A,X[14],9);L1(A,B,C,D,E,X[15],8);
    L2(E,A,B,C,D,X[7],7);L2(D,E,A,B,C,X[4],6);L2(C,D,E,A,B,X[13],8);L2(B,C,D,E,A,X[1],13);
    L2(A,B,C,D,E,X[10],11);L2(E,A,B,C,D,X[6],9);L2(D,E,A,B,C,X[15],7);L2(C,D,E,A,B,X[3],15);
    L2(B,C,D,E,A,X[12],7);L2(A,B,C,D,E,X[0],12);L2(E,A,B,C,D,X[9],15);L2(D,E,A,B,C,X[5],9);
    L2(C,D,E,A,B,X[2],11);L2(B,C,D,E,A,X[14],7);L2(A,B,C,D,E,X[11],13);L2(E,A,B,C,D,X[8],12);
    L3(D,E,A,B,C,X[3],11);L3(C,D,E,A,B,X[10],13);L3(B,C,D,E,A,X[14],6);L3(A,B,C,D,E,X[4],7);
    L3(E,A,B,C,D,X[9],14);L3(D,E,A,B,C,X[15],9);L3(C,D,E,A,B,X[8],13);L3(B,C,D,E,A,X[1],15);
    L3(A,B,C,D,E,X[2],14);L3(E,A,B,C,D,X[7],8);L3(D,E,A,B,C,X[0],13);L3(C,D,E,A,B,X[6],6);
    L3(B,C,D,E,A,X[13],5);L3(A,B,C,D,E,X[11],12);L3(E,A,B,C,D,X[5],7);L3(D,E,A,B,C,X[12],5);
    L4(C,D,E,A,B,X[1],11);L4(B,C,D,E,A,X[9],12);L4(A,B,C,D,E,X[11],14);L4(E,A,B,C,D,X[10],15);
    L4(D,E,A,B,C,X[0],14);L4(C,D,E,A,B,X[8],15);L4(B,C,D,E,A,X[12],9);L4(A,B,C,D,E,X[4],8);
    L4(E,A,B,C,D,X[13],9);L4(D,E,A,B,C,X[3],14);L4(C,D,E,A,B,X[7],5);L4(B,C,D,E,A,X[15],6);
    L4(A,B,C,D,E,X[14],8);L4(E,A,B,C,D,X[5],6);L4(D,E,A,B,C,X[6],5);L4(C,D,E,A,B,X[2],12);
    L5(B,C,D,E,A,X[4],9);L5(A,B,C,D,E,X[0],15);L5(E,A,B,C,D,X[5],5);L5(D,E,A,B,C,X[9],11);
    L5(C,D,E,A,B,X[7],6);L5(B,C,D,E,A,X[12],8);L5(A,B,C,D,E,X[2],13);L5(E,A,B,C,D,X[10],12);
    L5(D,E,A,B,C,X[14],5);L5(C,D,E,A,B,X[1],12);L5(B,C,D,E,A,X[3],13);L5(A,B,C,D,E,X[8],14);
    L5(E,A,B,C,D,X[11],11);L5(D,E,A,B,C,X[6],8);L5(C,D,E,A,B,X[15],5);L5(B,C,D,E,A,X[13],6);
    a1 = A; b1 = B; c1 = C; d1 = D; e1 = E;
    A = hash[0]; B = hash[1]; C = hash[2]; D = hash[3]; E = hash[4];
    R1(A,B,C,D,E,X[5],8);R1(E,A,B,C,D,X[14],9);R1(D,E,A,B,C,X[7],9);R1(C,D,E,A,B,X[0],11);
    R1(B,C,D,E,A,X[9],13);R1(A,B,C,D,E,X[2],15);R1(E,A,B,C,D,X[11],15);R1(D,E,A,B,C,X[4],5);
    R1(C,D,E,A,B,X[13],7);R1(B,C,D,E,A,X[6],7);R1(A,B,C,D,E,X[15],8);R1(E,A,B,C,D,X[8],11);
    R1(D,E,A,B,C,X[1],14);R1(C,D,E,A,B,X[10],14);R1(B,C,D,E,A,X[3],12);R1(A,B,C,D,E,X[12],6);
    R2(E,A,B,C,D,X[6],9);R2(D,E,A,B,C,X[11],13);R2(C,D,E,A,B,X[3],15);R2(B,C,D,E,A,X[7],7);
    R2(A,B,C,D,E,X[0],12);R2(E,A,B,C,D,X[13],8);R2(D,E,A,B,C,X[5],9);R2(C,D,E,A,B,X[10],11);
    R2(B,C,D,E,A,X[14],7);R2(A,B,C,D,E,X[15],7);R2(E,A,B,C,D,X[8],12);R2(D,E,A,B,C,X[12],7);
    R2(C,D,E,A,B,X[4],6);R2(B,C,D,E,A,X[9],15);R2(A,B,C,D,E,X[1],13);R2(E,A,B,C,D,X[2],11);
    R3(D,E,A,B,C,X[15],9);R3(C,D,E,A,B,X[5],7);R3(B,C,D,E,A,X[1],15);R3(A,B,C,D,E,X[3],11);
    R3(E,A,B,C,D,X[7],8);R3(D,E,A,B,C,X[14],6);R3(C,D,E,A,B,X[6],6);R3(B,C,D,E,A,X[9],14);
    R3(A,B,C,D,E,X[11],12);R3(E,A,B,C,D,X[8],13);R3(D,E,A,B,C,X[12],5);R3(C,D,E,A,B,X[2],14);
    R3(B,C,D,E,A,X[10],13);R3(A,B,C,D,E,X[0],13);R3(E,A,B,C,D,X[4],7);R3(D,E,A,B,C,X[13],5);
    R4(C,D,E,A,B,X[8],15);R4(B,C,D,E,A,X[6],5);R4(A,B,C,D,E,X[4],8);R4(E,A,B,C,D,X[1],11);
    R4(D,E,A,B,C,X[3],14);R4(C,D,E,A,B,X[11],14);R4(B,C,D,E,A,X[15],6);R4(A,B,C,D,E,X[0],14);
    R4(E,A,B,C,D,X[5],6);R4(D,E,A,B,C,X[12],9);R4(C,D,E,A,B,X[2],12);R4(B,C,D,E,A,X[13],9);
    R4(A,B,C,D,E,X[9],12);R4(E,A,B,C,D,X[7],5);R4(D,E,A,B,C,X[10],15);R4(C,D,E,A,B,X[14],8);
    R5(B,C,D,E,A,X[12],8);R5(A,B,C,D,E,X[15],5);R5(E,A,B,C,D,X[10],12);R5(D,E,A,B,C,X[4],9);
    R5(C,D,E,A,B,X[1],12);R5(B,C,D,E,A,X[5],5);R5(A,B,C,D,E,X[8],14);R5(E,A,B,C,D,X[7],6);
    R5(D,E,A,B,C,X[6],8);R5(C,D,E,A,B,X[2],13);R5(B,C,D,E,A,X[13],6);R5(A,B,C,D,E,X[14],5);
    R5(E,A,B,C,D,X[0],15);R5(D,E,A,B,C,X[3],13);R5(C,D,E,A,B,X[9],11);R5(B,C,D,E,A,X[11],11);
    D += c1 + hash[1]; hash[1] = hash[2] + d1 + E; hash[2] = hash[3] + e1 + A;
    hash[3] = hash[4] + a1 + B; hash[4] = hash[0] + b1 + C; hash[0] = D;
}

/* ---- RIPEMD-128 block function ---- */
/* RIPEMD-128 single 64-byte block compression. Ported from rmd128.c
 * compress() (Bosselaers 1996, lines 39-196). 4-uint state (vs the
 * 5-uint state of rmd160_block). Dual pipeline: left line uses
 * F1->F2->F3->F4; right line uses F4->F3->F2->F1 -- per Bosselaers
 * Table 4 (RMD-128 right line is F4 F3 F2 F1, NOT F5 F4 F3 F2 F1 as
 * in RMD-160). Anyone porting "by analogy to rmd160_block" will get
 * the right-line ordering wrong silently -- spec R2 callout.
 *
 * Reuses RMD_F1..RMD_F4 from gpu_common.cl rmd160 section above
 * (RMD_F5 is unused by RMD-128). Defines local RMD128_STEP (4-arg
 * variant without the +E and without the C-rotation that
 * rmd160_block's RMD_STEP carries) so this body matches rmd128.h's
 * FF/GG/HH/II macro shape directly. Right-line K constants:
 *   III  = RMD_F4 with K=0x50a28be6
 *   HHH  = RMD_F3 with K=0x5c4dd124
 *   GGG  = RMD_F2 with K=0x6d703ef3
 *   FFF  = RMD_F1 with K=0
 *
 * Schedule packing is LE (rmd128.h BYTES_TO_DWORD is LE); caller
 * packs the 64-byte block into X[0..15] LE. Output: hash[0..3] is
 * 4 LE uint32 chaining values; caller writes them as 16 LE bytes
 * to form the 128-bit digest. CPU oracle RIPEMD128() writes bytes
 * LE too -- byte-exact match.
 *
 * noinline per feedback_md5_block_noinline_pascal.md (spec R5) --
 * empirically required for Pascal register-budget; same discipline
 * as md5_block / md4_block / rmd160_block. */
#define RMD128_STEP(FUNC, A, B, C, D, X, S, K) \
    (A) += FUNC((B), (C), (D)) + (X) + K; \
    (A) = rotate((A), (uint)(S));

#define LL1(A,B,C,D,X,S) RMD128_STEP(RMD_F1,A,B,C,D,X,S,0u)
#define LL2(A,B,C,D,X,S) RMD128_STEP(RMD_F2,A,B,C,D,X,S,0x5a827999u)
#define LL3(A,B,C,D,X,S) RMD128_STEP(RMD_F3,A,B,C,D,X,S,0x6ed9eba1u)
#define LL4(A,B,C,D,X,S) RMD128_STEP(RMD_F4,A,B,C,D,X,S,0x8f1bbcdcu)
/* Right line: F4 F3 F2 F1 ordering -- Bosselaers Table 4. */
#define RR1(A,B,C,D,X,S) RMD128_STEP(RMD_F4,A,B,C,D,X,S,0x50a28be6u)
#define RR2(A,B,C,D,X,S) RMD128_STEP(RMD_F3,A,B,C,D,X,S,0x5c4dd124u)
#define RR3(A,B,C,D,X,S) RMD128_STEP(RMD_F2,A,B,C,D,X,S,0x6d703ef3u)
#define RR4(A,B,C,D,X,S) RMD128_STEP(RMD_F1,A,B,C,D,X,S,0u)

__attribute__((noinline))
void rmd128_block(uint *hash, const uint *X) {
    uint A = hash[0], B = hash[1], C = hash[2], D = hash[3];
    uint a1, b1, c1, d1;
    /* left round 1 (F1) */
    LL1(A,B,C,D,X[ 0],11); LL1(D,A,B,C,X[ 1],14); LL1(C,D,A,B,X[ 2],15); LL1(B,C,D,A,X[ 3],12);
    LL1(A,B,C,D,X[ 4], 5); LL1(D,A,B,C,X[ 5], 8); LL1(C,D,A,B,X[ 6], 7); LL1(B,C,D,A,X[ 7], 9);
    LL1(A,B,C,D,X[ 8],11); LL1(D,A,B,C,X[ 9],13); LL1(C,D,A,B,X[10],14); LL1(B,C,D,A,X[11],15);
    LL1(A,B,C,D,X[12], 6); LL1(D,A,B,C,X[13], 7); LL1(C,D,A,B,X[14], 9); LL1(B,C,D,A,X[15], 8);
    /* left round 2 (F2) */
    LL2(A,B,C,D,X[ 7], 7); LL2(D,A,B,C,X[ 4], 6); LL2(C,D,A,B,X[13], 8); LL2(B,C,D,A,X[ 1],13);
    LL2(A,B,C,D,X[10],11); LL2(D,A,B,C,X[ 6], 9); LL2(C,D,A,B,X[15], 7); LL2(B,C,D,A,X[ 3],15);
    LL2(A,B,C,D,X[12], 7); LL2(D,A,B,C,X[ 0],12); LL2(C,D,A,B,X[ 9],15); LL2(B,C,D,A,X[ 5], 9);
    LL2(A,B,C,D,X[ 2],11); LL2(D,A,B,C,X[14], 7); LL2(C,D,A,B,X[11],13); LL2(B,C,D,A,X[ 8],12);
    /* left round 3 (F3) */
    LL3(A,B,C,D,X[ 3],11); LL3(D,A,B,C,X[10],13); LL3(C,D,A,B,X[14], 6); LL3(B,C,D,A,X[ 4], 7);
    LL3(A,B,C,D,X[ 9],14); LL3(D,A,B,C,X[15], 9); LL3(C,D,A,B,X[ 8],13); LL3(B,C,D,A,X[ 1],15);
    LL3(A,B,C,D,X[ 2],14); LL3(D,A,B,C,X[ 7], 8); LL3(C,D,A,B,X[ 0],13); LL3(B,C,D,A,X[ 6], 6);
    LL3(A,B,C,D,X[13], 5); LL3(D,A,B,C,X[11],12); LL3(C,D,A,B,X[ 5], 7); LL3(B,C,D,A,X[12], 5);
    /* left round 4 (F4) */
    LL4(A,B,C,D,X[ 1],11); LL4(D,A,B,C,X[ 9],12); LL4(C,D,A,B,X[11],14); LL4(B,C,D,A,X[10],15);
    LL4(A,B,C,D,X[ 0],14); LL4(D,A,B,C,X[ 8],15); LL4(C,D,A,B,X[12], 9); LL4(B,C,D,A,X[ 4], 8);
    LL4(A,B,C,D,X[13], 9); LL4(D,A,B,C,X[ 3],14); LL4(C,D,A,B,X[ 7], 5); LL4(B,C,D,A,X[15], 6);
    LL4(A,B,C,D,X[14], 8); LL4(D,A,B,C,X[ 5], 6); LL4(C,D,A,B,X[ 6], 5); LL4(B,C,D,A,X[ 2],12);
    /* save left line, restart with IV for right line */
    a1 = A; b1 = B; c1 = C; d1 = D;
    A = hash[0]; B = hash[1]; C = hash[2]; D = hash[3];
    /* right round 1 (F4) -- Bosselaers Table 4 RMD-128 right line F4 F3 F2 F1 */
    RR1(A,B,C,D,X[ 5], 8); RR1(D,A,B,C,X[14], 9); RR1(C,D,A,B,X[ 7], 9); RR1(B,C,D,A,X[ 0],11);
    RR1(A,B,C,D,X[ 9],13); RR1(D,A,B,C,X[ 2],15); RR1(C,D,A,B,X[11],15); RR1(B,C,D,A,X[ 4], 5);
    RR1(A,B,C,D,X[13], 7); RR1(D,A,B,C,X[ 6], 7); RR1(C,D,A,B,X[15], 8); RR1(B,C,D,A,X[ 8],11);
    RR1(A,B,C,D,X[ 1],14); RR1(D,A,B,C,X[10],14); RR1(C,D,A,B,X[ 3],12); RR1(B,C,D,A,X[12], 6);
    /* right round 2 (F3) */
    RR2(A,B,C,D,X[ 6], 9); RR2(D,A,B,C,X[11],13); RR2(C,D,A,B,X[ 3],15); RR2(B,C,D,A,X[ 7], 7);
    RR2(A,B,C,D,X[ 0],12); RR2(D,A,B,C,X[13], 8); RR2(C,D,A,B,X[ 5], 9); RR2(B,C,D,A,X[10],11);
    RR2(A,B,C,D,X[14], 7); RR2(D,A,B,C,X[15], 7); RR2(C,D,A,B,X[ 8],12); RR2(B,C,D,A,X[12], 7);
    RR2(A,B,C,D,X[ 4], 6); RR2(D,A,B,C,X[ 9],15); RR2(C,D,A,B,X[ 1],13); RR2(B,C,D,A,X[ 2],11);
    /* right round 3 (F2) */
    RR3(A,B,C,D,X[15], 9); RR3(D,A,B,C,X[ 5], 7); RR3(C,D,A,B,X[ 1],15); RR3(B,C,D,A,X[ 3],11);
    RR3(A,B,C,D,X[ 7], 8); RR3(D,A,B,C,X[14], 6); RR3(C,D,A,B,X[ 6], 6); RR3(B,C,D,A,X[ 9],14);
    RR3(A,B,C,D,X[11],12); RR3(D,A,B,C,X[ 8],13); RR3(C,D,A,B,X[12], 5); RR3(B,C,D,A,X[ 2],14);
    RR3(A,B,C,D,X[10],13); RR3(D,A,B,C,X[ 0],13); RR3(C,D,A,B,X[ 4], 7); RR3(B,C,D,A,X[13], 5);
    /* right round 4 (F1) */
    RR4(A,B,C,D,X[ 8],15); RR4(D,A,B,C,X[ 6], 5); RR4(C,D,A,B,X[ 4], 8); RR4(B,C,D,A,X[ 1],11);
    RR4(A,B,C,D,X[ 3],14); RR4(D,A,B,C,X[11],14); RR4(C,D,A,B,X[15], 6); RR4(B,C,D,A,X[ 0],14);
    RR4(A,B,C,D,X[ 5], 6); RR4(D,A,B,C,X[12], 9); RR4(C,D,A,B,X[ 2],12); RR4(B,C,D,A,X[13], 9);
    RR4(A,B,C,D,X[ 9],12); RR4(D,A,B,C,X[ 7], 5); RR4(C,D,A,B,X[10],15); RR4(B,C,D,A,X[14], 8);
    /* combine: mirrors rmd128.c lines 188-193 with left = a1..d1 saved
     * vars and right = current A..D. Cross-mix pattern:
     *   ddd += cc + MDbuf[1];   -> D += c1 + hash[1]   (new hash[0])
     *   MDbuf[1] = MDbuf[2] + dd + aaa;  -> hash[1] = hash[2] + d1 + A
     *   MDbuf[2] = MDbuf[3] + aa + bbb;  -> hash[2] = hash[3] + a1 + B
     *   MDbuf[3] = MDbuf[0] + bb + ccc;  -> hash[3] = hash[0] + b1 + C
     *   MDbuf[0] = ddd;                  -> hash[0] = D
     * Note the ordering: hash[1..3] must be written before hash[0] is
     * overwritten -- the temporaries cover this. */
    D += c1 + hash[1];
    hash[1] = hash[2] + d1 + A;
    hash[2] = hash[3] + a1 + B;
    hash[3] = hash[0] + b1 + C;
    hash[0] = D;
}

/* RIPEMD-320 compression. Mirrors gpu_hmac_rmd320.cl rmd320_block exactly,
 * but uses RMD_F1..RMD_F5 (defined above) directly via RMD_STEP rather
 * than the local F1..F5 aliases — keeps gpu_common.cl self-contained.
 *
 * State convention: hash[0..9] are 10 × uint32 LE chaining values.
 * The dual pipeline (A..E vs AA..EE) is cross-swapped between rounds
 * (one of {A,B,C,D,E} swaps with the corresponding {AA,BB,CC,DD,EE}
 * after each 16-step round). At end-of-compression the lines do NOT
 * merge — they are added back to hash[] with the cross-mixed
 * accumulation:
 *   hash[0] += A;  hash[1] += B;  hash[2] += C;  hash[3] += D;  hash[4] += EE;
 *   hash[5] += AA; hash[6] += BB; hash[7] += CC; hash[8] += DD; hash[9] += E;
 *
 * Used by: gpu_ripemd320_core.cl (B5 sub-batch 2). */
void rmd320_block(uint *hash, const uint *X) {
    uint A = hash[0], B = hash[1], C = hash[2], D = hash[3], E = hash[4];
    uint AA = hash[5], BB = hash[6], CC = hash[7], DD = hash[8], EE = hash[9];

    /* j=0..15 */
    RMD_STEP(RMD_F1, A, B, C, D, E, X[0], 11, 0x00000000u);
    RMD_STEP(RMD_F1, E, A, B, C, D, X[1], 14, 0x00000000u);
    RMD_STEP(RMD_F1, D, E, A, B, C, X[2], 15, 0x00000000u);
    RMD_STEP(RMD_F1, C, D, E, A, B, X[3], 12, 0x00000000u);
    RMD_STEP(RMD_F1, B, C, D, E, A, X[4], 5, 0x00000000u);
    RMD_STEP(RMD_F1, A, B, C, D, E, X[5], 8, 0x00000000u);
    RMD_STEP(RMD_F1, E, A, B, C, D, X[6], 7, 0x00000000u);
    RMD_STEP(RMD_F1, D, E, A, B, C, X[7], 9, 0x00000000u);
    RMD_STEP(RMD_F1, C, D, E, A, B, X[8], 11, 0x00000000u);
    RMD_STEP(RMD_F1, B, C, D, E, A, X[9], 13, 0x00000000u);
    RMD_STEP(RMD_F1, A, B, C, D, E, X[10], 14, 0x00000000u);
    RMD_STEP(RMD_F1, E, A, B, C, D, X[11], 15, 0x00000000u);
    RMD_STEP(RMD_F1, D, E, A, B, C, X[12], 6, 0x00000000u);
    RMD_STEP(RMD_F1, C, D, E, A, B, X[13], 7, 0x00000000u);
    RMD_STEP(RMD_F1, B, C, D, E, A, X[14], 9, 0x00000000u);
    RMD_STEP(RMD_F1, A, B, C, D, E, X[15], 8, 0x00000000u);
    RMD_STEP(RMD_F5, AA, BB, CC, DD, EE, X[5], 8, 0x50A28BE6u);
    RMD_STEP(RMD_F5, EE, AA, BB, CC, DD, X[14], 9, 0x50A28BE6u);
    RMD_STEP(RMD_F5, DD, EE, AA, BB, CC, X[7], 9, 0x50A28BE6u);
    RMD_STEP(RMD_F5, CC, DD, EE, AA, BB, X[0], 11, 0x50A28BE6u);
    RMD_STEP(RMD_F5, BB, CC, DD, EE, AA, X[9], 13, 0x50A28BE6u);
    RMD_STEP(RMD_F5, AA, BB, CC, DD, EE, X[2], 15, 0x50A28BE6u);
    RMD_STEP(RMD_F5, EE, AA, BB, CC, DD, X[11], 15, 0x50A28BE6u);
    RMD_STEP(RMD_F5, DD, EE, AA, BB, CC, X[4], 5, 0x50A28BE6u);
    RMD_STEP(RMD_F5, CC, DD, EE, AA, BB, X[13], 7, 0x50A28BE6u);
    RMD_STEP(RMD_F5, BB, CC, DD, EE, AA, X[6], 7, 0x50A28BE6u);
    RMD_STEP(RMD_F5, AA, BB, CC, DD, EE, X[15], 8, 0x50A28BE6u);
    RMD_STEP(RMD_F5, EE, AA, BB, CC, DD, X[8], 11, 0x50A28BE6u);
    RMD_STEP(RMD_F5, DD, EE, AA, BB, CC, X[1], 14, 0x50A28BE6u);
    RMD_STEP(RMD_F5, CC, DD, EE, AA, BB, X[10], 14, 0x50A28BE6u);
    RMD_STEP(RMD_F5, BB, CC, DD, EE, AA, X[3], 12, 0x50A28BE6u);
    RMD_STEP(RMD_F5, AA, BB, CC, DD, EE, X[12], 6, 0x50A28BE6u);
    { uint T = A; A = AA; AA = T; }
    /* j=16..31 */
    RMD_STEP(RMD_F2, E, A, B, C, D, X[7], 7, 0x5A827999u);
    RMD_STEP(RMD_F2, D, E, A, B, C, X[4], 6, 0x5A827999u);
    RMD_STEP(RMD_F2, C, D, E, A, B, X[13], 8, 0x5A827999u);
    RMD_STEP(RMD_F2, B, C, D, E, A, X[1], 13, 0x5A827999u);
    RMD_STEP(RMD_F2, A, B, C, D, E, X[10], 11, 0x5A827999u);
    RMD_STEP(RMD_F2, E, A, B, C, D, X[6], 9, 0x5A827999u);
    RMD_STEP(RMD_F2, D, E, A, B, C, X[15], 7, 0x5A827999u);
    RMD_STEP(RMD_F2, C, D, E, A, B, X[3], 15, 0x5A827999u);
    RMD_STEP(RMD_F2, B, C, D, E, A, X[12], 7, 0x5A827999u);
    RMD_STEP(RMD_F2, A, B, C, D, E, X[0], 12, 0x5A827999u);
    RMD_STEP(RMD_F2, E, A, B, C, D, X[9], 15, 0x5A827999u);
    RMD_STEP(RMD_F2, D, E, A, B, C, X[5], 9, 0x5A827999u);
    RMD_STEP(RMD_F2, C, D, E, A, B, X[2], 11, 0x5A827999u);
    RMD_STEP(RMD_F2, B, C, D, E, A, X[14], 7, 0x5A827999u);
    RMD_STEP(RMD_F2, A, B, C, D, E, X[11], 13, 0x5A827999u);
    RMD_STEP(RMD_F2, E, A, B, C, D, X[8], 12, 0x5A827999u);
    RMD_STEP(RMD_F4, EE, AA, BB, CC, DD, X[6], 9, 0x5C4DD124u);
    RMD_STEP(RMD_F4, DD, EE, AA, BB, CC, X[11], 13, 0x5C4DD124u);
    RMD_STEP(RMD_F4, CC, DD, EE, AA, BB, X[3], 15, 0x5C4DD124u);
    RMD_STEP(RMD_F4, BB, CC, DD, EE, AA, X[7], 7, 0x5C4DD124u);
    RMD_STEP(RMD_F4, AA, BB, CC, DD, EE, X[0], 12, 0x5C4DD124u);
    RMD_STEP(RMD_F4, EE, AA, BB, CC, DD, X[13], 8, 0x5C4DD124u);
    RMD_STEP(RMD_F4, DD, EE, AA, BB, CC, X[5], 9, 0x5C4DD124u);
    RMD_STEP(RMD_F4, CC, DD, EE, AA, BB, X[10], 11, 0x5C4DD124u);
    RMD_STEP(RMD_F4, BB, CC, DD, EE, AA, X[14], 7, 0x5C4DD124u);
    RMD_STEP(RMD_F4, AA, BB, CC, DD, EE, X[15], 7, 0x5C4DD124u);
    RMD_STEP(RMD_F4, EE, AA, BB, CC, DD, X[8], 12, 0x5C4DD124u);
    RMD_STEP(RMD_F4, DD, EE, AA, BB, CC, X[12], 7, 0x5C4DD124u);
    RMD_STEP(RMD_F4, CC, DD, EE, AA, BB, X[4], 6, 0x5C4DD124u);
    RMD_STEP(RMD_F4, BB, CC, DD, EE, AA, X[9], 15, 0x5C4DD124u);
    RMD_STEP(RMD_F4, AA, BB, CC, DD, EE, X[1], 13, 0x5C4DD124u);
    RMD_STEP(RMD_F4, EE, AA, BB, CC, DD, X[2], 11, 0x5C4DD124u);
    { uint T = B; B = BB; BB = T; }
    /* j=32..47 */
    RMD_STEP(RMD_F3, D, E, A, B, C, X[3], 11, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, C, D, E, A, B, X[10], 13, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, B, C, D, E, A, X[14], 6, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, A, B, C, D, E, X[4], 7, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, E, A, B, C, D, X[9], 14, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, D, E, A, B, C, X[15], 9, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, C, D, E, A, B, X[8], 13, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, B, C, D, E, A, X[1], 15, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, A, B, C, D, E, X[2], 14, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, E, A, B, C, D, X[7], 8, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, D, E, A, B, C, X[0], 13, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, C, D, E, A, B, X[6], 6, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, B, C, D, E, A, X[13], 5, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, A, B, C, D, E, X[11], 12, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, E, A, B, C, D, X[5], 7, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, D, E, A, B, C, X[12], 5, 0x6ED9EBA1u);
    RMD_STEP(RMD_F3, DD, EE, AA, BB, CC, X[15], 9, 0x6D703EF3u);
    RMD_STEP(RMD_F3, CC, DD, EE, AA, BB, X[5], 7, 0x6D703EF3u);
    RMD_STEP(RMD_F3, BB, CC, DD, EE, AA, X[1], 15, 0x6D703EF3u);
    RMD_STEP(RMD_F3, AA, BB, CC, DD, EE, X[3], 11, 0x6D703EF3u);
    RMD_STEP(RMD_F3, EE, AA, BB, CC, DD, X[7], 8, 0x6D703EF3u);
    RMD_STEP(RMD_F3, DD, EE, AA, BB, CC, X[14], 6, 0x6D703EF3u);
    RMD_STEP(RMD_F3, CC, DD, EE, AA, BB, X[6], 6, 0x6D703EF3u);
    RMD_STEP(RMD_F3, BB, CC, DD, EE, AA, X[9], 14, 0x6D703EF3u);
    RMD_STEP(RMD_F3, AA, BB, CC, DD, EE, X[11], 12, 0x6D703EF3u);
    RMD_STEP(RMD_F3, EE, AA, BB, CC, DD, X[8], 13, 0x6D703EF3u);
    RMD_STEP(RMD_F3, DD, EE, AA, BB, CC, X[12], 5, 0x6D703EF3u);
    RMD_STEP(RMD_F3, CC, DD, EE, AA, BB, X[2], 14, 0x6D703EF3u);
    RMD_STEP(RMD_F3, BB, CC, DD, EE, AA, X[10], 13, 0x6D703EF3u);
    RMD_STEP(RMD_F3, AA, BB, CC, DD, EE, X[0], 13, 0x6D703EF3u);
    RMD_STEP(RMD_F3, EE, AA, BB, CC, DD, X[4], 7, 0x6D703EF3u);
    RMD_STEP(RMD_F3, DD, EE, AA, BB, CC, X[13], 5, 0x6D703EF3u);
    { uint T = C; C = CC; CC = T; }
    /* j=48..63 */
    RMD_STEP(RMD_F4, C, D, E, A, B, X[1], 11, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, B, C, D, E, A, X[9], 12, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, A, B, C, D, E, X[11], 14, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, E, A, B, C, D, X[10], 15, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, D, E, A, B, C, X[0], 14, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, C, D, E, A, B, X[8], 15, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, B, C, D, E, A, X[12], 9, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, A, B, C, D, E, X[4], 8, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, E, A, B, C, D, X[13], 9, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, D, E, A, B, C, X[3], 14, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, C, D, E, A, B, X[7], 5, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, B, C, D, E, A, X[15], 6, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, A, B, C, D, E, X[14], 8, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, E, A, B, C, D, X[5], 6, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, D, E, A, B, C, X[6], 5, 0x8F1BBCDCu);
    RMD_STEP(RMD_F4, C, D, E, A, B, X[2], 12, 0x8F1BBCDCu);
    RMD_STEP(RMD_F2, CC, DD, EE, AA, BB, X[8], 15, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, BB, CC, DD, EE, AA, X[6], 5, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, AA, BB, CC, DD, EE, X[4], 8, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, EE, AA, BB, CC, DD, X[1], 11, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, DD, EE, AA, BB, CC, X[3], 14, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, CC, DD, EE, AA, BB, X[11], 14, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, BB, CC, DD, EE, AA, X[15], 6, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, AA, BB, CC, DD, EE, X[0], 14, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, EE, AA, BB, CC, DD, X[5], 6, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, DD, EE, AA, BB, CC, X[12], 9, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, CC, DD, EE, AA, BB, X[2], 12, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, BB, CC, DD, EE, AA, X[13], 9, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, AA, BB, CC, DD, EE, X[9], 12, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, EE, AA, BB, CC, DD, X[7], 5, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, DD, EE, AA, BB, CC, X[10], 15, 0x7A6D76E9u);
    RMD_STEP(RMD_F2, CC, DD, EE, AA, BB, X[14], 8, 0x7A6D76E9u);
    { uint T = D; D = DD; DD = T; }
    /* j=64..79 */
    RMD_STEP(RMD_F5, B, C, D, E, A, X[4], 9, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, A, B, C, D, E, X[0], 15, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, E, A, B, C, D, X[5], 5, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, D, E, A, B, C, X[9], 11, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, C, D, E, A, B, X[7], 6, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, B, C, D, E, A, X[12], 8, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, A, B, C, D, E, X[2], 13, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, E, A, B, C, D, X[10], 12, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, D, E, A, B, C, X[14], 5, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, C, D, E, A, B, X[1], 12, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, B, C, D, E, A, X[3], 13, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, A, B, C, D, E, X[8], 14, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, E, A, B, C, D, X[11], 11, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, D, E, A, B, C, X[6], 8, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, C, D, E, A, B, X[15], 5, 0xA953FD4Eu);
    RMD_STEP(RMD_F5, B, C, D, E, A, X[13], 6, 0xA953FD4Eu);
    RMD_STEP(RMD_F1, BB, CC, DD, EE, AA, X[12], 8, 0x00000000u);
    RMD_STEP(RMD_F1, AA, BB, CC, DD, EE, X[15], 5, 0x00000000u);
    RMD_STEP(RMD_F1, EE, AA, BB, CC, DD, X[10], 12, 0x00000000u);
    RMD_STEP(RMD_F1, DD, EE, AA, BB, CC, X[4], 9, 0x00000000u);
    RMD_STEP(RMD_F1, CC, DD, EE, AA, BB, X[1], 12, 0x00000000u);
    RMD_STEP(RMD_F1, BB, CC, DD, EE, AA, X[5], 5, 0x00000000u);
    RMD_STEP(RMD_F1, AA, BB, CC, DD, EE, X[8], 14, 0x00000000u);
    RMD_STEP(RMD_F1, EE, AA, BB, CC, DD, X[7], 6, 0x00000000u);
    RMD_STEP(RMD_F1, DD, EE, AA, BB, CC, X[6], 8, 0x00000000u);
    RMD_STEP(RMD_F1, CC, DD, EE, AA, BB, X[2], 13, 0x00000000u);
    RMD_STEP(RMD_F1, BB, CC, DD, EE, AA, X[13], 6, 0x00000000u);
    RMD_STEP(RMD_F1, AA, BB, CC, DD, EE, X[14], 5, 0x00000000u);
    RMD_STEP(RMD_F1, EE, AA, BB, CC, DD, X[0], 15, 0x00000000u);
    RMD_STEP(RMD_F1, DD, EE, AA, BB, CC, X[3], 13, 0x00000000u);
    RMD_STEP(RMD_F1, CC, DD, EE, AA, BB, X[9], 11, 0x00000000u);
    RMD_STEP(RMD_F1, BB, CC, DD, EE, AA, X[11], 11, 0x00000000u);
    { uint T = E; E = EE; EE = T; }

    /* RIPEMD-320 accumulation (Bosselaers reference, MDcompress combine):
     *   MDbuf[0..4] += (aa,bb,cc,dd,ee)     [left-line registers, post-swap]
     *   MDbuf[5..9] += (aaa,bbb,ccc,ddd,eee) [right-line, post-swap]
     *
     * IMPORTANT: the EARLIER gpu_hmac_rmd320.cl version had hash[4]+=EE and
     * hash[9]+=E — a typo that produced wrong bytes for state[4] and state[9].
     * That bug was undetected in production because HMAC-RMD320 probes only
     * the first 16 bytes (state[0..3]), so state[4]/state[9] never matter
     * to the compact-table probe. The B5 sub-batch 2 RIPEMD-320 template
     * dispatches the FULL 40-byte digest through the host hit-replay path
     * which does a full byte-compare — exposing the bug. Fix preserved here. */
    hash[0] += A;  hash[1] += B;  hash[2] += C;  hash[3] += D;  hash[4] += E;
    hash[5] += AA; hash[6] += BB; hash[7] += CC; hash[8] += DD; hash[9] += EE;
}

/* ---- BLAKE2S compress ---- */

__constant uint B2S_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

__constant uchar B2S_SIGMA[10][16] = {
    { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 },
    { 14,10,4,8,9,15,13,6,1,12,0,2,11,7,5,3 },
    { 11,8,12,0,5,2,15,13,10,14,3,6,7,1,9,4 },
    { 7,9,3,1,13,12,11,14,2,6,5,10,4,0,15,8 },
    { 9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13 },
    { 2,12,6,10,0,11,8,3,4,13,7,5,15,14,1,9 },
    { 12,5,1,15,14,13,4,10,0,7,6,3,9,2,8,11 },
    { 13,11,7,14,12,1,3,9,5,0,15,4,8,6,2,10 },
    { 6,15,14,9,11,3,0,8,12,2,13,7,1,4,10,5 },
    { 10,2,8,4,7,6,1,5,15,11,9,14,3,12,13,0 }
};

void b2s_compress(uint *h, const uchar *block, ulong counter, int last) {
    uint v[16], m[16];
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i+8] = B2S_IV[i]; }
    v[12] ^= (uint)counter;
    v[13] ^= (uint)(counter >> 32);
    if (last) v[14] = ~v[14];
    for (int i = 0; i < 16; i++)
        m[i] = ((uint)block[i*4]) | ((uint)block[i*4+1]<<8) |
               ((uint)block[i*4+2]<<16) | ((uint)block[i*4+3]<<24);
    for (int r = 0; r < 10; r++) {
        __constant const uchar *s = B2S_SIGMA[r];
        v[0]+=v[4]+m[s[0]]; v[12]=rotate(v[12]^v[0],(uint)16); v[8]+=v[12]; v[4]=rotate(v[4]^v[8],(uint)20);
        v[0]+=v[4]+m[s[1]]; v[12]=rotate(v[12]^v[0],(uint)24); v[8]+=v[12]; v[4]=rotate(v[4]^v[8],(uint)25);
        v[1]+=v[5]+m[s[2]]; v[13]=rotate(v[13]^v[1],(uint)16); v[9]+=v[13]; v[5]=rotate(v[5]^v[9],(uint)20);
        v[1]+=v[5]+m[s[3]]; v[13]=rotate(v[13]^v[1],(uint)24); v[9]+=v[13]; v[5]=rotate(v[5]^v[9],(uint)25);
        v[2]+=v[6]+m[s[4]]; v[14]=rotate(v[14]^v[2],(uint)16); v[10]+=v[14]; v[6]=rotate(v[6]^v[10],(uint)20);
        v[2]+=v[6]+m[s[5]]; v[14]=rotate(v[14]^v[2],(uint)24); v[10]+=v[14]; v[6]=rotate(v[6]^v[10],(uint)25);
        v[3]+=v[7]+m[s[6]]; v[15]=rotate(v[15]^v[3],(uint)16); v[11]+=v[15]; v[7]=rotate(v[7]^v[11],(uint)20);
        v[3]+=v[7]+m[s[7]]; v[15]=rotate(v[15]^v[3],(uint)24); v[11]+=v[15]; v[7]=rotate(v[7]^v[11],(uint)25);
        v[0]+=v[5]+m[s[8]]; v[15]=rotate(v[15]^v[0],(uint)16); v[10]+=v[15]; v[5]=rotate(v[5]^v[10],(uint)20);
        v[0]+=v[5]+m[s[9]]; v[15]=rotate(v[15]^v[0],(uint)24); v[10]+=v[15]; v[5]=rotate(v[5]^v[10],(uint)25);
        v[1]+=v[6]+m[s[10]]; v[12]=rotate(v[12]^v[1],(uint)16); v[11]+=v[12]; v[6]=rotate(v[6]^v[11],(uint)20);
        v[1]+=v[6]+m[s[11]]; v[12]=rotate(v[12]^v[1],(uint)24); v[11]+=v[12]; v[6]=rotate(v[6]^v[11],(uint)25);
        v[2]+=v[7]+m[s[12]]; v[13]=rotate(v[13]^v[2],(uint)16); v[8]+=v[13]; v[7]=rotate(v[7]^v[8],(uint)20);
        v[2]+=v[7]+m[s[13]]; v[13]=rotate(v[13]^v[2],(uint)24); v[8]+=v[13]; v[7]=rotate(v[7]^v[8],(uint)25);
        v[3]+=v[4]+m[s[14]]; v[14]=rotate(v[14]^v[3],(uint)16); v[9]+=v[14]; v[4]=rotate(v[4]^v[9],(uint)20);
        v[3]+=v[4]+m[s[15]]; v[14]=rotate(v[14]^v[3],(uint)24); v[9]+=v[14]; v[4]=rotate(v[4]^v[9],(uint)25);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i+8];
}

/* ---- BLAKE2B compress (Memo B Phase B5 sub-batch 3) ----
 *
 * RFC 7693 BLAKE2b: 128-byte block, 64-bit lanes, 12-round G compression.
 * Used by BLAKE2B-256 (32-byte digest) and BLAKE2B-512 (64-byte digest).
 *
 * Identical structure to b2s_compress above, scaled to ulong:
 *   h[8]    : digest chaining state (in/out)
 *   block   : 128-byte input message block
 *   t0, t1  : split 128-bit byte counter (BLAKE2b spec; for our wordlist
 *             inputs t1 always = 0 because total bytes < 2^64)
 *   last    : 1 = final block (sets v[14] ^= 0xFFFF...; matches host
 *             blake2b_compress() in mdxfind.c using last==1 -> v[14]=~IV[6])
 *
 * R1 mitigation: function takes __private uchar* by ABI but does only
 * byte reads via simple [i*8..i*8+7] indexing — no addrspace-cast helpers,
 * no pointer-to-pointer arguments. Same shape as b2s_compress. */

__constant ulong B2B_IV[8] = {
    0x6A09E667F3BCC908UL, 0xBB67AE8584CAA73BUL,
    0x3C6EF372FE94F82BUL, 0xA54FF53A5F1D36F1UL,
    0x510E527FADE682D1UL, 0x9B05688C2B3E6C1FUL,
    0x1F83D9ABFB41BD6BUL, 0x5BE0CD19137E2179UL
};

/* BLAKE2b uses 12 rounds; SIGMA wraps modulo 10 (rounds 10 and 11 reuse
 * SIGMA[0] and SIGMA[1] respectively per RFC 7693). Stored as 12 rows for
 * direct round-index lookup; rows 10/11 are duplicates of 0/1. */
__constant uchar B2B_SIGMA[12][16] = {
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
    { 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
    {  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
    {  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
    {  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
    { 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
    { 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
    {  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
    { 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
    {  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
    { 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 }
};

/* BLAKE2b G mixing: rotates by (32, 24, 16, 63). The OpenCL `rotate` builtin
 * works on 64-bit values for ulong vectors; the rotr64-as-rotate-left
 * conversion: rotr64(x, n) == rotate(x, 64-n). */
#define B2B_G(a, b, c, d, x, y) do { \
    a = a + b + (x); \
    d = rotate(d ^ a, (ulong)(64 - 32)); \
    c = c + d; \
    b = rotate(b ^ c, (ulong)(64 - 24)); \
    a = a + b + (y); \
    d = rotate(d ^ a, (ulong)(64 - 16)); \
    c = c + d; \
    b = rotate(b ^ c, (ulong)(64 - 63)); \
} while (0)

void b2b_compress(ulong *h, const uchar *block, ulong t0, ulong t1, int last) {
    ulong v[16], m[16];
    /* Load 128-byte block as 16 ulong LITTLE-ENDIAN (BLAKE2b spec). */
    for (int i = 0; i < 16; i++) {
        int b = i * 8;
        m[i] = ((ulong)block[b])
             | ((ulong)block[b + 1] << 8)
             | ((ulong)block[b + 2] << 16)
             | ((ulong)block[b + 3] << 24)
             | ((ulong)block[b + 4] << 32)
             | ((ulong)block[b + 5] << 40)
             | ((ulong)block[b + 6] << 48)
             | ((ulong)block[b + 7] << 56);
    }
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i + 8] = B2B_IV[i]; }
    v[12] ^= t0;
    v[13] ^= t1;
    if (last) v[14] = ~v[14];
    for (int r = 0; r < 12; r++) {
        __constant const uchar *s = B2B_SIGMA[r];
        B2B_G(v[ 0], v[ 4], v[ 8], v[12], m[s[ 0]], m[s[ 1]]);
        B2B_G(v[ 1], v[ 5], v[ 9], v[13], m[s[ 2]], m[s[ 3]]);
        B2B_G(v[ 2], v[ 6], v[10], v[14], m[s[ 4]], m[s[ 5]]);
        B2B_G(v[ 3], v[ 7], v[11], v[15], m[s[ 6]], m[s[ 7]]);
        B2B_G(v[ 0], v[ 5], v[10], v[15], m[s[ 8]], m[s[ 9]]);
        B2B_G(v[ 1], v[ 6], v[11], v[12], m[s[10]], m[s[11]]);
        B2B_G(v[ 2], v[ 7], v[ 8], v[13], m[s[12]], m[s[13]]);
        B2B_G(v[ 3], v[ 4], v[ 9], v[14], m[s[14]], m[s[15]]);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i + 8];
}

/* ---- Keccak-f[1600] permutation (Memo B Phase B5 sub-batch 4, 2026-05-03) -
 *
 * Shared by KECCAK-{224,256,384,512} (suffix=0x01) and SHA3-{224,256,384,512}
 * (suffix=0x06). Sponge construction: per-algo "rate" (bytes absorbed per
 * Keccak-f call) is 200 - 2*(output_bytes); state is fixed at 5x5 ulong = 1600 bits.
 *
 *   Keccak-224 / SHA3-224  -> rate=144, output=28 (HASH_WORDS=7)
 *   Keccak-256 / SHA3-256  -> rate=136, output=32 (HASH_WORDS=8)
 *   Keccak-384 / SHA3-384  -> rate=104, output=48 (HASH_WORDS=12)
 *   Keccak-512 / SHA3-512  -> rate= 72, output=64 (HASH_WORDS=16)
 *
 * Promoted from gpu/gpu_keccakunsalted.cl (slab kernel) so all 8 per-algo
 * cores share one cryptographic surface. The slab kernel is left in place
 * as the legacy short-circuit until B8 retires it.
 *
 * Naming: the slab file used `keccak_f1600`; the function symbol here is
 * `keccakf1600` (no underscore between f and 1600) to avoid any ABI clash if
 * both source files end up in the same compile unit. The slab-path .cl is
 * compiled to a separate cl_program so symbol-set isolation holds either
 * way; the rename is a defensive belt-and-suspenders.
 *
 * The 24-round permutation is the spec-canonical implementation (theta /
 * rho+pi / chi / iota). Round constants RC[24] and rotation offsets ROTC[25]
 * are __constant for cross-kernel reuse. */
__constant ulong KECCAK_RC[24] = {
    0x0000000000000001UL, 0x0000000000008082UL, 0x800000000000808aUL,
    0x8000000080008000UL, 0x000000000000808bUL, 0x0000000080000001UL,
    0x8000000080008081UL, 0x8000000000008009UL, 0x000000000000008aUL,
    0x0000000000000088UL, 0x0000000080008009UL, 0x000000008000000aUL,
    0x000000008000808bUL, 0x800000000000008bUL, 0x8000000000008089UL,
    0x8000000000008003UL, 0x8000000000008002UL, 0x8000000000000080UL,
    0x000000000000800aUL, 0x800000008000000aUL, 0x8000000080008081UL,
    0x8000000000008080UL, 0x0000000080000001UL, 0x8000000080008008UL
};

__constant uint KECCAK_ROTC[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

void keccakf1600(ulong *st) {
    for (int round = 0; round < 24; round++) {
        ulong C[5], D[5];
        for (int x = 0; x < 5; x++)
            C[x] = st[x] ^ st[x+5] ^ st[x+10] ^ st[x+15] ^ st[x+20];
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x+4) % 5] ^ rotate(C[(x+1) % 5], (ulong)1);
            for (int y = 0; y < 25; y += 5)
                st[x+y] ^= D[x];
        }
        ulong B[25];
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 5; y++)
                B[x + 5 * ((2*y + 3*x) % 5)] = rotate(st[x*5+y], (ulong)KECCAK_ROTC[x*5+y]);
        for (int x = 0; x < 5; x++)
            for (int y = 0; y < 25; y += 5)
                st[x+y] = B[x+y] ^ (~B[((x+1)%5)+y] & B[((x+2)%5)+y]);
        st[0] ^= KECCAK_RC[round];
    }
}

__kernel void gpu_selftest(__global uint *results) {
    uint tid = get_global_id(0);
    uint M[16];
    M[0] = 0x74736574u;  /* "test" LE */
    M[1] = 0x00000080u;  /* padding byte */
    for (int i = 2; i < 14; i++) M[i] = 0;
    M[14] = 32u;         /* 4 bytes * 8 bits */
    M[15] = 0;
    uint hx = 0x67452301u, hy = 0xEFCDAB89u, hz = 0x98BADCFEu, hw = 0x10325476u;
    md5_block(&hx, &hy, &hz, &hw, M);
    /* MD5("test") = 098f6bcd... -> LE word0 = 0xcd6b8f09 */
    results[tid] = (hx == 0xcd6b8f09u && hy == 0x73d32146u) ? 1u : 0u;
}
