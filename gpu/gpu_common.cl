/*
 * $Revision: 1.20 $
 * $Log: gpu_common.cl,v $
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
 * locked contract; B3 wires read+CAS-min on overflow). reserved32[0..1] at
 * offsets 80-87 are still claimed by gpu_md5_packed.cl etc. for word_start
 * and packed_size (see gpu_opencl.c gpu_opencl_dispatch_packed).
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
    uint  reserved32[2];      /* 80-87: reserved (packed kernels: word_start, packed_size) */
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
    uint  overflow_first_rule;/* 108: B3 rule_idx CAS-min target */
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
 * The OCLParams.overflow_first_rule field is unused in B3 (kept for
 * forward-compat; host writes 0). Kernel CAS-min is on
 * overflow_first_word interpreted as the lane gid.
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
    ulong W[80];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 80; i++) {
        ulong s0 = rotr64(W[i-15], 1) ^ rotr64(W[i-15], 8) ^ (W[i-15] >> 7);
        ulong s1 = rotr64(W[i-2], 19) ^ rotr64(W[i-2], 61) ^ (W[i-2] >> 6);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    ulong a = state[0], b = state[1], c = state[2], d = state[3];
    ulong e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 80; i++) {
        ulong S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        ulong ch = (e & f) ^ (~e & g);
        ulong t1 = h + S1 + ch + K512[i] + W[i];
        ulong S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        ulong maj = (a & b) ^ (a & c) ^ (b & c);
        ulong t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
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
