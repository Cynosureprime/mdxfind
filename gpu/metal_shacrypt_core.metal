/*
 * $Revision: $
 * $Log: $
 *
 * Hand-edited (Steps A+C port) from gpu/codegen/cl2metal.py base generated
 * from gpu_shacrypt_core.cl (RCS rev 1.3). Steps A+C ported 2026-05-19:
 *   Step A: metal_common.metal sha512_block flat-unrolled (MTL_SHA512_STEP_S).
 *   Step C: STEP 5 inner loop wpc boolean-cube (#if HASH_WORDS==16 gate).
 * Mirrors gpu_shacrypt_core.cl rev 1.4 optimizations for SHA512CRYPT.
 * SHA256CRYPT (HASH_WORDS=8) keeps original sc_update path unchanged.
 *
 * Address-space port from OpenCL:
 *   __global  -> device
 *   __private -> thread
 *   barrier   -> threadgroup_barrier
 *   typedef struct ... T;  ->  struct T { ... };
 *   T *st     -> thread T &st
 *
 * Patterns 1/3 enforced: every pointer arg is address-space
 * qualified; every helper is static inline.
 */

#ifndef HASH_WORDS
#define HASH_WORDS 8
#endif
#ifndef HASH_BLOCK_BYTES
#define HASH_BLOCK_BYTES 64
#endif

/* SHA-256 IV constants. */
#define SC_IV0_32 0x6a09e667u
#define SC_IV1_32 0xbb67ae85u
#define SC_IV2_32 0x3c6ef372u
#define SC_IV3_32 0xa54ff53au
#define SC_IV4_32 0x510e527fu
#define SC_IV5_32 0x9b05688cu
#define SC_IV6_32 0x1f83d9abu
#define SC_IV7_32 0x5be0cd19u

/* SHA-512 IV constants (used only at HASH_WORDS=16 in Phases 3 / 4). */
#define SC_IV0_64 0x6a09e667f3bcc908UL
#define SC_IV1_64 0xbb67ae8584caa73bUL
#define SC_IV2_64 0x3c6ef372fe94f82bUL
#define SC_IV3_64 0xa54ff53a5f1d36f1UL
#define SC_IV4_64 0x510e527fade682d1UL
#define SC_IV5_64 0x9b05688c2b3e6c1fUL
#define SC_IV6_64 0x1f83d9abfb41bd6bUL
#define SC_IV7_64 0x5be0cd19137e2179UL

/* Per-lane state struct. The HASH_WORDS gate is RUNTIME-evaluated, but
 * the struct must hold whichever digest width is active at compile time.
 *
 * For SHA-256 path (HASH_WORDS=8): use uint state[8], uint buf[16],
 *   uint counter (each compiled instance picks the right width via the
 *   #ifdef inside the helpers below — but ONLY at the storage layer; the
 *   logic gates are runtime).
 *
 * For SHA-512 path (HASH_WORDS=16): use ulong state[8], ulong buf[16],
 *   ulong counter.
 *
 * Storage IS gated by `#ifdef` here (NOT `#if HASH_WORDS == N`) -- this is
 * a STORAGE choice (which TYPE compiles), not an EXECUTION branch (which
 * BODY runs). The lesson from feedback_runtime_gate_for_template_branches.md
 * is about EXECUTION branches in shared multi-width fragments. Storage-
 * only #ifdef on the TYPE is fine and standard practice (see e.g.
 * gpu_sha256crypt.cl + gpu_sha512crypt.cl which use this pattern). */
#if HASH_WORDS == 16
/* SHA-512 storage path */
typedef ulong sc_word_t;
typedef ulong sc_counter_t;
#define SC_BLOCK_BYTES 128
#define SC_DIGEST_BYTES 64
#define SC_PAD_LEN_OFF  112      /* pos at which 16-byte length begins */
#define SC_LEN_HI_IDX   14
#define SC_LEN_LO_IDX   15
#else
/* SHA-256 storage path (default; HASH_WORDS=8) */
typedef uint sc_word_t;
typedef uint sc_counter_t;
#define SC_BLOCK_BYTES 64
#define SC_DIGEST_BYTES 32
#define SC_PAD_LEN_OFF  56       /* pos at which 8-byte length begins */
#define SC_LEN_HI_IDX   14
#define SC_LEN_LO_IDX   15
#endif

struct template_state {
    sc_word_t h[HASH_WORDS];
};

/* template_init: install IV. The IV constants are TYPE-dependent so we
 * use `#if` here (same discipline as the typedefs above). */
static inline void template_init(thread template_state &st)
{
#if HASH_WORDS == 16
    st.h[0] = SC_IV0_64; st.h[1] = SC_IV1_64;
    st.h[2] = SC_IV2_64; st.h[3] = SC_IV3_64;
    st.h[4] = SC_IV4_64; st.h[5] = SC_IV5_64;
    st.h[6] = SC_IV6_64; st.h[7] = SC_IV7_64;
    /* st.h[8..15] used as scratch for the final BE-extracted digest
     * words / LE-packed output (see template_finalize tail). */
    for (int i = 8; i < 16; i++) st.h[i] = 0;
#else
    st.h[0] = SC_IV0_32; st.h[1] = SC_IV1_32;
    st.h[2] = SC_IV2_32; st.h[3] = SC_IV3_32;
    st.h[4] = SC_IV4_32; st.h[5] = SC_IV5_32;
    st.h[6] = SC_IV6_32; st.h[7] = SC_IV7_32;
#endif
}

/* template_transform: stub. SHACRYPT manages multi-block absorption inline
 * via the buffered streaming helpers below; never routes through this. */
static inline void template_transform(thread template_state &st, thread const uchar *block)
{
    (void)st; (void)block;
}

/* ---- Buffered streaming helpers ----------------------------------------
 *
 * sc_init / sc_update / sc_update_g / sc_final mirror the slab oracles'
 * accumulator pattern (gpu_sha256crypt.cl + gpu_sha512crypt.cl).
 *
 * Per the discipline note above:
 *   - Width-specific TYPES (sc_word_t / sc_counter_t) and constants
 *     (SC_BLOCK_BYTES, SC_PAD_LEN_OFF, IV literals) are gated by `#if
 *     HASH_WORDS == 16`.
 *   - The COMPRESSION call site is gated by `#if HASH_WORDS == 16`
 *     because sha256_block (uint*) vs sha512_block (ulong*) have
 *     incompatible signatures; a runtime gate would require ill-typed
 *     dead code.
 *   - Pure-uint control flow / final-pack loops use runtime
 *     `if (HASH_WORDS == N)` since both branches type-check.
 */

static inline void sc_buf_set_byte(thread sc_word_t *buf, int pos, uchar val)
{
#if HASH_WORDS == 16
    int wi = pos >> 3;
    int bi = (7 - (pos & 7)) << 3;
    buf[wi] = (buf[wi] & ~((sc_word_t)0xffUL << bi))
              | ((sc_word_t)val << bi);
#else
    int wi = pos >> 2;
    int bi = (3 - (pos & 3)) << 3;
    buf[wi] = (buf[wi] & ~((sc_word_t)0xffu << bi))
              | ((sc_word_t)val << bi);
#endif
}

static inline void sc_init(thread sc_word_t *state)
{
#if HASH_WORDS == 16
    state[0] = SC_IV0_64; state[1] = SC_IV1_64;
    state[2] = SC_IV2_64; state[3] = SC_IV3_64;
    state[4] = SC_IV4_64; state[5] = SC_IV5_64;
    state[6] = SC_IV6_64; state[7] = SC_IV7_64;
#else
    state[0] = SC_IV0_32; state[1] = SC_IV1_32;
    state[2] = SC_IV2_32; state[3] = SC_IV3_32;
    state[4] = SC_IV4_32; state[5] = SC_IV5_32;
    state[6] = SC_IV6_32; state[7] = SC_IV7_32;
#endif
}

/* Accumulate `len` bytes from PRIVATE memory into the streaming buffer.
 * On block-boundary, compress + zero buf.
 *
 * The compression call site uses `#if HASH_WORDS == 16` because the two
 * primitives (sha256_block / sha512_block) take INCOMPATIBLE pointer
 * types (uint* / ulong*); a runtime gate would require ill-typed dead
 * code. Discipline: type-mismatched branches use `#if`; same-type
 * arithmetic / control flow uses runtime `if (HASH_WORDS == N)`. */
static inline void sc_update(thread sc_word_t *state, thread sc_word_t *buf, thread int &bufpos, thread sc_counter_t &counter, thread const uchar *data, int len)
{
    counter += (sc_counter_t)len;
    int bp = bufpos;
    for (int i = 0; i < len; i++) {
        sc_buf_set_byte(buf, bp, data[i]);
        bp++;
        if (bp == SC_BLOCK_BYTES) {
#if HASH_WORDS == 16
            sha512_block(state, buf);
#else
            sha256_block(state, buf);
#endif
            for (int j = 0; j < 16; j++) buf[j] = 0;
            bp = 0;
        }
    }
    bufpos = bp;
}

/* Accumulate from GLOBAL memory. Identical to sc_update but data ptr is
 * __global. Used for salt_bytes (which lives in __global salt_buf). */
static inline void sc_update_g(thread sc_word_t *state, thread sc_word_t *buf, thread int &bufpos, thread sc_counter_t &counter, device const uchar *data, int len)
{
    counter += (sc_counter_t)len;
    int bp = bufpos;
    for (int i = 0; i < len; i++) {
        sc_buf_set_byte(buf, bp, data[i]);
        bp++;
        if (bp == SC_BLOCK_BYTES) {
#if HASH_WORDS == 16
            sha512_block(state, buf);
#else
            sha256_block(state, buf);
#endif
            for (int j = 0; j < 16; j++) buf[j] = 0;
            bp = 0;
        }
    }
    bufpos = bp;
}

/* Pad + length-encode + final compress; extract digest as raw bytes
 * (BE state words). out[] receives SC_DIGEST_BYTES bytes. */
static inline void sc_final(thread sc_word_t *state,
                                thread sc_word_t *buf,
                                int bufpos,
                                sc_counter_t counter,
                                thread uchar *out)
{
    sc_buf_set_byte(buf, bufpos, 0x80);
    bufpos++;

    if (bufpos > SC_PAD_LEN_OFF) {
        for (int i = bufpos; i < SC_BLOCK_BYTES; i++)
            sc_buf_set_byte(buf, i, 0);
#if HASH_WORDS == 16
        sha512_block(state, buf);
#else
        sha256_block(state, buf);
#endif
        for (int j = 0; j < 16; j++) buf[j] = 0;
        bufpos = 0;
    }

    /* Zero between bufpos and the length field. */
    for (int i = bufpos; i < SC_PAD_LEN_OFF; i++)
        sc_buf_set_byte(buf, i, 0);

    /* Length in bits (BE). High word = 0 for our message sizes (well
     * under 2^32 bytes for SHA-256, well under 2^64 bytes for SHA-512). */
    buf[SC_LEN_HI_IDX] = 0;
    buf[SC_LEN_LO_IDX] = (sc_word_t)counter * (sc_word_t)8;

#if HASH_WORDS == 16
    sha512_block(state, buf);
    /* Extract 64 bytes BE from state[0..7] (ulong). */
    for (int i = 0; i < 8; i++) {
        ulong w = state[i];
        out[i*8 + 0] = (uchar)(w >> 56);
        out[i*8 + 1] = (uchar)(w >> 48);
        out[i*8 + 2] = (uchar)(w >> 40);
        out[i*8 + 3] = (uchar)(w >> 32);
        out[i*8 + 4] = (uchar)(w >> 24);
        out[i*8 + 5] = (uchar)(w >> 16);
        out[i*8 + 6] = (uchar)(w >> 8);
        out[i*8 + 7] = (uchar)(w);
    }
#else
    sha256_block(state, buf);
    /* Extract 32 bytes BE from state[0..7] (uint). */
    for (int i = 0; i < 8; i++) {
        uint w = state[i];
        out[i*4 + 0] = (uchar)(w >> 24);
        out[i*4 + 1] = (uchar)(w >> 16);
        out[i*4 + 2] = (uchar)(w >> 8);
        out[i*4 + 3] = (uchar)(w);
    }
#endif
}

/* One-shot wrapper: compute HASH(data, len) -> out[SC_DIGEST_BYTES] */
static inline void sc_oneshot(thread const uchar *data, int len, thread uchar *out)
{
    sc_word_t state[8];
    sc_word_t buf[16];
    int bp = 0;
    sc_counter_t counter = 0;
    sc_init(state);
    for (int j = 0; j < 16; j++) buf[j] = 0;
    sc_update(state, buf, bp, counter, data, len);
    sc_final(state, buf, bp, counter, out);
}

/* template_finalize: full SHACRYPT chain (5 steps + rounds-iter loop).
 *
 * algo_mode dispatch (Phase 4 will use this):
 *   0 = SHA256CRYPT / SHA512CRYPT — straight chain on `data` as the password
 *   1 = SHA512CRYPTMD5 (Phase 4)  — first MD5(data) -> hex32, THEN feed
 *                                    that hex into the SHA-512 chain
 *   In Phase 2 ship, only mode 0 is used.
 *
 * SALT prefix parsing:
 *   salt_bytes points at "$5$[rounds=N$]<salt>$" or
 *                       "$6$[rounds=N$]<salt>$"
 *   - skip the 3-byte "$5$" / "$6$" prefix
 *   - if "rounds=" follows, parse decimal up to '$' (clamp 1000..999999999)
 *   - then read raw_salt up to '$' or end (cap 16 bytes)
 *
 * Final state in st->h[0..HASH_WORDS-1] (BE-extracted digest as
 * BIG-ENDIAN uint32 words for SHA-256; for SHA-512, the digest is 8
 * ulongs but template_digest_compare needs UINT32 LE for the compact
 * probe; we pack curin[0..3] LE-from-BE into the state words to mirror
 * the slab oracle behavior).
 */
static inline void template_finalize(thread template_state &st,
                                device const uchar *data,
                                int len,
                                device const uchar *salt_bytes,
                                uint salt_len,
                                uint algo_mode)
{
    /* Defensive: clamp plen to a reasonable upper bound. mdxfind default
     * passwords <= 39 bytes; HASH_BLOCK_BYTES = 64 (SHA-256) or 128
     * (SHA-512). Working buffer inside the bit loop / step 1 grows with
     * plen; keep buf[256] for the worst-case temp construction. */
    int plen = len;
    if (plen > 64) plen = 64;  /* cap; won't happen in practice */

    /* Copy password into private memory (data is post-rule buffer). */
    uchar pw[64];
    for (int i = 0; i < plen; i++) pw[i] = data[i];

    /* SHA512CRYPTMD5 carrier (2026-05-08): algo_mode == 1u selects MD5-
     * preprocess of the password (data[0..plen]) BEFORE the SHA-crypt
     * chain runs. CPU semantics at mdxfind.c:12199-12212 substitutes
     * the password with the 32-char ASCII-hex representation of MD5
     * (original_password) -- the rest of crypt_round (which is the
     * cryptlen=64 / SHA-512 branch) runs on this 32-byte string.
     *
     * Mirror that exactly here, INSIDE template_finalize so rules-engine
     * post-rule plaintexts get the substitution AFTER rule application
     * (matches CPU semantics: rules apply to the original word, MD5 of
     * the post-rule word feeds the SHA-crypt chain).
     *
     * Using runtime `if` per feedback_runtime_gate_for_template_branches
     * .md (NEVER `#if`; rev 1.7 Pascal NVIDIA ABORT lesson). The branch
     * compiles unconditionally; on non-mode-1 paths NVCC's DCE strips
     * it. SHA256CRYPT instantiation (HASH_WORDS=8, BASE_ALGO=sha256-
     * crypt) hits a different cache key and never executes this branch
     * (host sets algo_mode=0u for it).
     *
     * MD5-of-pw (≤ 64 bytes input): 1 block if plen ≤ 55; 2 blocks if
     * 56 ≤ plen ≤ 64 (since MD5 padding requires 9 trailing bytes:
     * 0x80 + 8-byte length). md5_block from gpu_common takes the
     * h0..h3 state + M[16] message buffer. Output: 32-byte ASCII hex
     * written back into pw[]; plen reset to 32. */
    if (algo_mode == 1u) {
        /* Build MD5 message buffer with standard padding. msg_len = plen.
         * Use uchar mb[128] = 2 blocks, zero-init via local-scope var.
         * Single block path (plen <= 55):
         *   mb[plen] = 0x80; mb[56..63] = bit-length LE (uint64).
         * Two block path (56 <= plen <= 64):
         *   mb[plen] = 0x80 (could land in block 0 or block 1);
         *   mb[120..127] = bit-length LE (uint64).
         */
        uchar mb[128];
        for (int i = 0; i < 128; i++) mb[i] = 0;
        for (int i = 0; i < plen; i++) mb[i] = pw[i];
        mb[plen] = 0x80;
        ulong bitlen = (ulong)plen * 8u;
        int n_blocks = (plen <= 55) ? 1 : 2;
        int len_off = (n_blocks == 1) ? 56 : 120;
        mb[len_off + 0] = (uchar)(bitlen & 0xff);
        mb[len_off + 1] = (uchar)((bitlen >>  8) & 0xff);
        mb[len_off + 2] = (uchar)((bitlen >> 16) & 0xff);
        mb[len_off + 3] = (uchar)((bitlen >> 24) & 0xff);
        mb[len_off + 4] = (uchar)((bitlen >> 32) & 0xff);
        mb[len_off + 5] = (uchar)((bitlen >> 40) & 0xff);
        mb[len_off + 6] = (uchar)((bitlen >> 48) & 0xff);
        mb[len_off + 7] = (uchar)((bitlen >> 56) & 0xff);

        /* MD5 IV */
        uint h0 = 0x67452301u;
        uint h1 = 0xefcdab89u;
        uint h2 = 0x98badcfeu;
        uint h3 = 0x10325476u;

        /* Pack first block as 16 LE uint32 from mb[0..63]. */
        uint M[16];
        for (int j = 0; j < 16; j++) {
            int k = j * 4;
            M[j] = (uint)mb[k]
                 | ((uint)mb[k+1] << 8)
                 | ((uint)mb[k+2] << 16)
                 | ((uint)mb[k+3] << 24);
        }
        md5_block(h0, h1, h2, h3, M);
        if (n_blocks == 2) {
            for (int j = 0; j < 16; j++) {
                int k = 64 + j * 4;
                M[j] = (uint)mb[k]
                     | ((uint)mb[k+1] << 8)
                     | ((uint)mb[k+2] << 16)
                     | ((uint)mb[k+3] << 24);
            }
            md5_block(h0, h1, h2, h3, M);
        }

        /* Render MD5 digest as 32-char lowercase hex into pw[0..31]. The
         * digest in (h0,h1,h2,h3) is stored LE byte-per-byte; CPU
         * semantics (mymd5 + prmd5 in mdxfind.c) emits it as 32 lower-
         * case ASCII hex chars in the natural byte order h0[0..3] then
         * h1[0..3] then h2[0..3] then h3[0..3]. */
        const char hex_digits[] = "0123456789abcdef";
        uint hh[4]; hh[0] = h0; hh[1] = h1; hh[2] = h2; hh[3] = h3;
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                uchar by = (uchar)((hh[j] >> (k * 8)) & 0xff);
                pw[j * 8 + k * 2 + 0] = (uchar)hex_digits[(by >> 4) & 0x0f];
                pw[j * 8 + k * 2 + 1] = (uchar)hex_digits[by & 0x0f];
            }
        }
        plen = 32;
    }

    /* ---- Salt prefix parsing ---- */
    int slen_full = (int)salt_len;
    int spos = 3;       /* skip "$5$" or "$6$" */
    int rounds = 5000;

    /* Detect "rounds=" (10 bytes of "rounds=N$") */
    if (slen_full > 10
        && salt_bytes[3] == 'r' && salt_bytes[4] == 'o'
        && salt_bytes[5] == 'u' && salt_bytes[6] == 'n'
        && salt_bytes[7] == 'd' && salt_bytes[8] == 's'
        && salt_bytes[9] == '=')
    {
        rounds = 0;
        spos = 10;
        while (spos < slen_full
               && salt_bytes[spos] >= '0' && salt_bytes[spos] <= '9')
        {
            rounds = rounds * 10 + (salt_bytes[spos] - '0');
            spos++;
        }
        if (rounds < 1000) rounds = 1000;
        if (rounds > 999999999) rounds = 999999999;
        if (spos < slen_full && salt_bytes[spos] == '$') spos++;
    }

    /* Extract raw_salt up to 16 chars (terminated by '$' or end). */
    uchar raw_salt[16];
    int saltlen = 0;
    for (int i = spos; i < slen_full && saltlen < 16; i++) {
        uchar c = salt_bytes[i];
        if (c == 0 || c == '$') break;
        raw_salt[saltlen++] = c;
    }
    if (saltlen == 0) {
        /* Defensive: zero state and return — saltlen 0 is invalid. */
        for (int i = 0; i < HASH_WORDS; i++) st.h[i] = 0;
        return;
    }

    /* ---- Working buffers ---- */
    sc_word_t state[8];
    sc_word_t ctx_buf[16];
    int ctx_bufpos;
    sc_counter_t ctx_counter;
    uchar tmp[256];                    /* scratch for building messages */
    uchar digest_a[64];                /* alt_result (cryptlen <= 64) */
    uchar digest_b[64];                /* temporary (P-prep / S-prep) */
    uchar curin[64];                   /* main loop running digest */

    int cryptlen = SC_DIGEST_BYTES;    /* compile-time const per width */

    /* ---- STEP 1: digest_a = HASH(pass || raw_salt || pass) ---- */
    {
        int tlen = 0;
        for (int i = 0; i < plen; i++)    tmp[tlen++] = pw[i];
        for (int i = 0; i < saltlen; i++) tmp[tlen++] = raw_salt[i];
        for (int i = 0; i < plen; i++)    tmp[tlen++] = pw[i];
        sc_oneshot(tmp, tlen, digest_a);
    }

    /* ---- STEP 2: build curin ---- */
    sc_init(state);
    for (int j = 0; j < 16; j++) ctx_buf[j] = 0;
    ctx_bufpos = 0; ctx_counter = 0;

    /* update(pass + salt) */
    sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, pw, plen);
    sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, raw_salt, saltlen);

    /* For plen bytes, append digest_a in cryptlen-byte chunks. */
    {
        int x = plen;
        while (x > cryptlen) {
            sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, digest_a, cryptlen);
            x -= cryptlen;
        }
        sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, digest_a, x);
    }

    /* Bit loop: for each bit of plen (LSB to MSB) */
    for (int x = plen; x != 0; x >>= 1) {
        if (x & 1)
            sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, digest_a, cryptlen);
        else
            sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, pw, plen);
    }

    sc_final(state, ctx_buf, ctx_bufpos, ctx_counter, curin);

    /* ---- STEP 3 (P-prep): digest_p = HASH(pass repeated plen times) ---- */
    sc_init(state);
    for (int j = 0; j < 16; j++) ctx_buf[j] = 0;
    ctx_bufpos = 0; ctx_counter = 0;
    for (int x = 0; x < plen; x++)
        sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, pw, plen);
    sc_final(state, ctx_buf, ctx_bufpos, ctx_counter, digest_b);

    /* P-bytes: digest_b replicated to fill plen bytes. */
    uchar p_bytes[64];
    for (int i = 0; i < plen; i++)
        p_bytes[i] = digest_b[i % cryptlen];

    /* ---- STEP 4 (S-prep): digest_s = HASH(salt repeated 16+curin[0]
     *                                      times) ---- */
    sc_init(state);
    for (int j = 0; j < 16; j++) ctx_buf[j] = 0;
    ctx_bufpos = 0; ctx_counter = 0;
    int s_repeats = 16 + (int)(uint)curin[0];
    for (int x = 0; x < s_repeats; x++)
        sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, raw_salt, saltlen);
    sc_final(state, ctx_buf, ctx_bufpos, ctx_counter, digest_b);

    /* S-bytes: first saltlen bytes of digest_b. */
    uchar s_bytes[16];
    for (int i = 0; i < saltlen; i++)
        s_bytes[i] = digest_b[i];

    /* ---- STEP 5: rounds main loop ----
     *
     * Optimized inner loop for SHA512CRYPT (HASH_WORDS == 16):
     * Replaces per-iter sc_init/sc_update/sc_final (byte-RMW + state-
     * machine overhead) with a direct block-construction approach.
     * Mirrors hashcat m01800 wpc[8][16] pattern extended for plen > 15.
     * Metal port of gpu_shacrypt_core.cl rev 1.4 Step C boolean-cube.
     *
     * Metal translation notes vs OpenCL:
     *   - ulong blk[16] / bl2[16] / cu[8] / pu[8] / ns[8] are
     *     function-local (thread private stack); no qualifier needed.
     *   - sha512_block(ns, blk) passes thread ulong* implicitly (both
     *     arrays are local); no addr-space cast required.
     *   - Metal compiler inlines sha512_block (OPPOSITE of Pascal
     *     noinline -- per feedback_md5_block_noinline_pascal.md).
     *
     * SHA256CRYPT (HASH_WORDS == 8) keeps the original sc_update path.
     * The 2-block boundary logic differs (64B block vs 128B); no wpc. */
#if HASH_WORDS == 16
    {
        /* Pre-pack p_bytes[plen] into pu[8] BE ulong (zeros beyond plen). */
        ulong pu[8];
        for (int i = 0; i < 8; i++) pu[i] = 0UL;
        for (int i = 0; i < plen; i++) {
            int wi = i >> 3, bi = (7 - (i & 7)) << 3;
            pu[wi] |= (ulong)(uchar)p_bytes[i] << bi;
        }

        /* Pack initial curin[64] into cu[8] BE ulong. */
        ulong cu[8];
        for (int i = 0; i < 8; i++) {
            int b = i * 8;
            cu[i] = ((ulong)curin[b+0] << 56) | ((ulong)curin[b+1] << 48)
                  | ((ulong)curin[b+2] << 40) | ((ulong)curin[b+3] << 32)
                  | ((ulong)curin[b+4] << 24) | ((ulong)curin[b+5] << 16)
                  | ((ulong)curin[b+6] <<  8) | ((ulong)curin[b+7]);
        }

        for (int r = 0; r < rounds; r++) {
            int pc = (r & 1) + ((r % 3) ? 2 : 0) + ((r % 7) ? 4 : 0);

            ulong blk[16];
            ulong bl2[16];
            for (int i = 0; i < 16; i++) { blk[i] = 0UL; bl2[i] = 0UL; }

            int pos = 0;

/* Helper: append N bytes from uchar src[] at byte offset pos in blk/bl2. */
#define SC5_APPEND(src, len) \
            for (int _i = 0; _i < (len); _i++, pos++) { \
                int _wi = pos >> 3, _bi = (7 - (pos & 7)) << 3; \
                ulong _v = (ulong)(uchar)(src)[_i] << _bi; \
                if (_wi < 16) blk[_wi] |= _v; else bl2[_wi - 16] |= _v; \
            }

/* Helper: append cu[8] ulong BE array (64 bytes) at current pos. */
#define SC5_APPEND_CU() \
            if (pos == 0) { \
                for (int _i = 0; _i < 8; _i++) blk[_i] = cu[_i]; \
                pos = 64; \
            } else if ((pos & 7) == 0) { \
                int _bw = pos >> 3; \
                for (int _i = 0; _i < 8; _i++) { \
                    int _idx = _bw + _i; \
                    if (_idx < 16) blk[_idx] |= cu[_i]; \
                    else bl2[_idx - 16] |= cu[_i]; \
                } \
                pos += 64; \
            } else { \
                for (int _i = 0; _i < 64; _i++, pos++) { \
                    int _wi = pos >> 3, _bi = (7 - (pos & 7)) << 3; \
                    int _ci = _i >> 3, _cbi = (7 - (_i & 7)) * 8; \
                    ulong _v = ((cu[_ci] >> _cbi) & 0xffUL) << _bi; \
                    if (_wi < 16) blk[_wi] |= _v; else bl2[_wi-16] |= _v; \
                } \
            }

/* Helper: append pu[8] ulong BE array (plen bytes). */
#define SC5_APPEND_PU() \
            if ((pos & 7) == 0 && (plen & 7) == 0) { \
                int _bw = pos >> 3, _nw = plen >> 3; \
                for (int _i = 0; _i < _nw; _i++) { \
                    int _idx = _bw + _i; \
                    if (_idx < 16) blk[_idx] |= pu[_i]; \
                    else bl2[_idx - 16] |= pu[_i]; \
                } \
                pos += plen; \
            } else { SC5_APPEND(p_bytes, plen) }

            /* Build message segments according to pc index. */
            if (pc & 1) { SC5_APPEND_PU() } else { SC5_APPEND_CU() }
            if (pc & 2) { SC5_APPEND(s_bytes, saltlen) }
            if (pc & 4) { SC5_APPEND_PU() }
            if (pc & 1) { SC5_APPEND_CU() } else { SC5_APPEND_PU() }

#undef SC5_APPEND
#undef SC5_APPEND_CU
#undef SC5_APPEND_PU

            /* Insert SHA-512 Merkle-Damgard padding (128B block):
             * 0x80 at pos, zeros until length field, 128-bit BE length
             * at blk[14..15] (single block, pos <= 111) or
             * bl2[14..15] (two blocks, pos >= 112). */
            int totbytes = pos;
            {
                int _wi = pos >> 3, _bi = (7 - (pos & 7)) << 3;
                ulong _v = 0x80UL << _bi;
                if (_wi < 16) blk[_wi] |= _v; else bl2[_wi - 16] |= _v;
            }
            ulong bitlen = (ulong)totbytes * 8UL;

            ulong ns[8];
            ns[0] = SC_IV0_64; ns[1] = SC_IV1_64;
            ns[2] = SC_IV2_64; ns[3] = SC_IV3_64;
            ns[4] = SC_IV4_64; ns[5] = SC_IV5_64;
            ns[6] = SC_IV6_64; ns[7] = SC_IV7_64;

            if (totbytes < 112) {
                blk[15] = bitlen;
                sha512_block(ns, blk);
            } else {
                bl2[15] = bitlen;
                sha512_block(ns, blk);
                sha512_block(ns, bl2);
            }

            /* Update cu[] = new curin (already in ulong BE form). */
            for (int i = 0; i < 8; i++) cu[i] = ns[i];
        }

        /* Extract final curin from cu[8] back into curin[64] for pack step. */
        for (int i = 0; i < 8; i++) {
            ulong w = cu[i];
            int b = i * 8;
            curin[b+0] = (uchar)(w >> 56); curin[b+1] = (uchar)(w >> 48);
            curin[b+2] = (uchar)(w >> 40); curin[b+3] = (uchar)(w >> 32);
            curin[b+4] = (uchar)(w >> 24); curin[b+5] = (uchar)(w >> 16);
            curin[b+6] = (uchar)(w >>  8); curin[b+7] = (uchar)(w);
        }
    }
#else
    /* SHA256CRYPT (HASH_WORDS == 8): keep original sc_update path. */
    for (int r = 0; r < rounds; r++) {
        sc_init(state);
        for (int j = 0; j < 16; j++) ctx_buf[j] = 0;
        ctx_bufpos = 0; ctx_counter = 0;

        if (r & 1)
            sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, p_bytes, plen);
        else
            sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, curin, cryptlen);

        if (r % 3)
            sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, s_bytes, saltlen);

        if (r % 7)
            sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, p_bytes, plen);

        if (r & 1)
            sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, curin, cryptlen);
        else
            sc_update(state, ctx_buf, ctx_bufpos, ctx_counter, p_bytes, plen);

        sc_final(state, ctx_buf, ctx_bufpos, ctx_counter, curin);
    }
#endif

    /* ---- Pack final digest into st.h for compare/emit ----
     *
     * The compact-table probe + EMIT_HIT_8 (SHA-256) / EMIT_HIT_16
     * (SHA-512) read st.h[i] as LE uint32 words. The slab oracle packs
     * the first 4 uint32 words via LE-byte-read of curin[0..15]. We
     * mirror exactly: pack curin[i*4 .. i*4+3] LE -> st.h[i].
     *
     * Loop bound + st.h[] write is width-dependent (8 vs 16 iterations).
     * st.h has HASH_WORDS slots. A runtime gate would have a dead-branch
     * OOB write at HASH_WORDS=8 (st.h[15] doesn't exist). Use #if for
     * the same reason as the compression call site (TYPE/STORAGE
     * discipline). */
#if HASH_WORDS == 16
    for (int i = 0; i < 16; i++) {
        uint w = ((uint)curin[i*4 + 0])
               | ((uint)curin[i*4 + 1] << 8)
               | ((uint)curin[i*4 + 2] << 16)
               | ((uint)curin[i*4 + 3] << 24);
        /* Phase 3's EMIT_HIT_16 will re-extract 16 LE uint32 words via
         * (uint)st.h[i] (lower 32 bits of each ulong). */
        st.h[i] = (sc_word_t)w;
    }
#else
    for (int i = 0; i < 8; i++) {
        uint w = ((uint)curin[i*4 + 0])
               | ((uint)curin[i*4 + 1] << 8)
               | ((uint)curin[i*4 + 2] << 16)
               | ((uint)curin[i*4 + 3] << 24);
        st.h[i] = (sc_word_t)w;
    }
#endif
    return;
}

/* template_iterate: STUB. max_iter=1 forced host-side for SHACRYPT, so
 * the kernel's outer iter loop runs exactly once and never calls this.
 * Mirror PHPBB3 / SHA1DRU / MD5CRYPT pattern. */
static inline void template_iterate(thread template_state &st)
{
    (void)st;
}

/* template_digest_compare: probe the compact table with the first 4 LE
 * uint32 words of the digest. Identical to the slab oracle's probe. */
static inline int template_digest_compare(thread const template_state &st,
                                device const uint *compact_fp,
                                device const uint *compact_idx,
                                ulong compact_mask,
                                uint max_probe,
                                uint hash_data_count,
                                device const uchar *hash_data_buf,
                                device const ulong *hash_data_off,
                                device const ulong *overflow_keys,
                                device const uchar *overflow_hashes,
                                device const uint *overflow_offsets,
                                uint overflow_count,
                                thread uint *out_idx)
{
    return probe_compact_idx(
        (uint)st.h[0], (uint)st.h[1], (uint)st.h[2], (uint)st.h[3],
        compact_fp, compact_idx,
        compact_mask, max_probe, hash_data_count,
        hash_data_buf, hash_data_off,
        overflow_keys, overflow_hashes, overflow_offsets, overflow_count,
        out_idx);
}

/* template_emit_hit: emit the digest words as a hit.
 *
 * EMIT_HIT_N macros in gpu_common.cl take `h` as an array argument (not
 * individual word args; cf. gpu_sha256saltpass_core.cl rev 1.x style).
 * We construct a local uint[HASH_WORDS] array from st->h[i] (already
 * LE-packed from curin in template_finalize) and pass it. NO bswap32:
 * st->h[i] is already LE uint32; the slab oracle does the same.
 *
 * Width-conditional via #if (CONSTANT macros, not execution branches —
 * the runtime-gate rule applies to BODIES, not preprocessor selection of
 * which #define line is active per compile target). */
#if HASH_WORDS == 16
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    do { \
        uint _h[16]; \
        for (int _w = 0; _w < 16; _w++) _h[_w] = (uint)(st).h[_w]; \
        EMIT_HIT_16((hits), (hit_count), (max_hits), \
                    (widx), (sidx), (iter), _h) \
    } while (0)

#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, \
                                      widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _h[16]; \
        for (int _w = 0; _w < 16; _w++) _h[_w] = (uint)(st).h[_w]; \
        EMIT_HIT_16_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
                    (widx), (sidx), (iter), _h, \
                    (hashes_shown), (matched_idx), (dedup_mask), \
                    (ovr_set), (ovr_gid), (lane_gid)); \
    } while (0)
#else
#define template_emit_hit(hits, hit_count, max_hits, st, widx, sidx, iter) \
    do { \
        uint _h[8]; \
        for (int _w = 0; _w < 8; _w++) _h[_w] = (uint)(st).h[_w]; \
        EMIT_HIT_8((hits), (hit_count), (max_hits), \
                   (widx), (sidx), (iter), _h) \
    } while (0)

#define template_emit_hit_or_overflow(hits, hit_count, max_hits, st, \
                                      widx, sidx, iter, \
                                      hashes_shown, matched_idx, dedup_mask, \
                                      ovr_set, ovr_gid, lane_gid) \
    do { \
        uint _h[8]; \
        for (int _w = 0; _w < 8; _w++) _h[_w] = (uint)(st).h[_w]; \
        EMIT_HIT_8_DEDUP_OR_OVERFLOW((hits), (hit_count), (max_hits), \
                    (widx), (sidx), (iter), _h, \
                    (hashes_shown), (matched_idx), (dedup_mask), \
                    (ovr_set), (ovr_gid), (lane_gid)); \
    } while (0)
#endif
