/* metal_hmac_sha512.metal — HMAC-SHA512/SHA384 + SHA512PASSSALT/SALTPASS
 * Hit stride: 19 (3 + 16) for SHA512, 15 (3 + 12) for SHA384
 */

/* K512[], sha512_block, S512_copy_bytes, S512_copy_bytes_p, S512_set_byte,
 * bswap64, rotr64, SALTED_PARAMS, PROBE7_NOOVF
 * all provided by metal_common.metal */

#define SHA512_IV0 0x6a09e667f3bcc908UL
#define SHA512_IV1 0xbb67ae8584caa73bUL
#define SHA512_IV2 0x3c6ef372fe94f82bUL
#define SHA512_IV3 0xa54ff53a5f1d36f1UL
#define SHA512_IV4 0x510e527fade682d1UL
#define SHA512_IV5 0x9b05688c2b3e6c1fUL
#define SHA512_IV6 0x1f83d9abfb41bd6bUL
#define SHA512_IV7 0x5be0cd19137e2179UL

/* Probe and emit full 16-word SHA512 hash from ostate[8] */
#define PROBE_AND_HIT_SHA512(ostate_arr, word_idx, salt_idx) { \
    uint _h512[16]; \
    for (int _i = 0; _i < 8; _i++) { \
        ulong _s = bswap64(ostate_arr[_i]); \
        _h512[_i*2] = (uint)_s; _h512[_i*2+1] = (uint)(_s >> 32); } \
    ulong _key = (ulong(_h512[1]) << 32) | _h512[0]; \
    uint _fp = uint(_key >> 32); if (_fp == 0) _fp = 1; \
    ulong _pos = (_key ^ (_key >> 32)) & params.compact_mask; \
    for (uint _p = 0; _p < params.max_probe; _p++) { \
        uint _cfp = compact_fp[_pos]; if (_cfp == 0) break; \
        if (_cfp == _fp) { uint _idx = compact_idx[_pos]; \
            if (_idx < params.hash_data_count) { \
                ulong _off = hash_data_off[_idx]; \
                device const uint *_ref = (device const uint *)(hash_data_buf + _off); \
                if (_h512[0]==_ref[0] && _h512[1]==_ref[1] && _h512[2]==_ref[2] && _h512[3]==_ref[3]) { \
                    uint _slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed); \
                    if (_slot < params.max_hits) { uint _base = _slot * 19; \
                        hits[_base] = word_idx; hits[_base+1] = salt_idx; hits[_base+2] = 1; \
                        for (int _i2=0;_i2<16;_i2++) hits[_base+3+_i2]=_h512[_i2]; } return; } } } \
        _pos = (_pos + 1) & params.compact_mask; } }

/* Probe and emit full 12-word SHA384 hash from ostate[8] */
#define PROBE_AND_HIT_SHA384(ostate_arr, word_idx, salt_idx) { \
    uint _h384[12]; \
    for (int _i = 0; _i < 6; _i++) { \
        ulong _s = bswap64(ostate_arr[_i]); \
        _h384[_i*2] = (uint)_s; _h384[_i*2+1] = (uint)(_s >> 32); } \
    ulong _key = (ulong(_h384[1]) << 32) | _h384[0]; \
    uint _fp = uint(_key >> 32); if (_fp == 0) _fp = 1; \
    ulong _pos = (_key ^ (_key >> 32)) & params.compact_mask; \
    for (uint _p = 0; _p < params.max_probe; _p++) { \
        uint _cfp = compact_fp[_pos]; if (_cfp == 0) break; \
        if (_cfp == _fp) { uint _idx = compact_idx[_pos]; \
            if (_idx < params.hash_data_count) { \
                ulong _off = hash_data_off[_idx]; \
                device const uint *_ref = (device const uint *)(hash_data_buf + _off); \
                if (_h384[0]==_ref[0] && _h384[1]==_ref[1] && _h384[2]==_ref[2] && _h384[3]==_ref[3]) { \
                    uint _slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed); \
                    if (_slot < params.max_hits) { uint _base = _slot * 19; \
                        hits[_base] = word_idx; hits[_base+1] = salt_idx; hits[_base+2] = 1; \
                        for (int _i2=0;_i2<12;_i2++) hits[_base+3+_i2]=_h384[_i2]; \
                        for (uint _z=15;_z<19;_z++) hits[_base+_z]=0; } return; } } } \
        _pos = (_pos + 1) & params.compact_mask; } }

/* HMAC core: prepares key_block, runs inner+outer hash, probes.
 * VARIANT: 0=ksalt (key=salt,msg=pass), 1=kpass (key=pass,msg=salt)
 * IV_ARRAY: SHA512 or SHA384 IV values */
#define HMAC_SHA512_BODY(IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7,inner_bytes) \
    ulong key_block[16]; \
    for (int i=0;i<16;i++) key_block[i]=0; \
    if (klen > 128) { \
        ulong kst[8]={IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7}; \
        ulong M2[16]; for (int i=0;i<16;i++) M2[i]=0; \
        S512_copy_bytes(M2,0,kdata,klen<128?klen:128); \
        if (klen<=111) { S512_set_byte(M2,klen,0x80); M2[15]=(ulong)klen*8; } \
        sha512_block(kst,M2); \
        if (klen>111) { \
            for (int i=0;i<16;i++) M2[i]=0; \
            int rem=klen-128; if (rem>0) S512_copy_bytes(M2,0,kdata+128,rem); \
            S512_set_byte(M2,rem,0x80); M2[15]=(ulong)klen*8; sha512_block(kst,M2); \
        } \
        for (int i=0;i<8;i++) key_block[i]=bswap64(kst[i]); \
        klen=64; \
    } else { \
        for (int i=0;i<klen;i++) key_block[i>>3]|=((ulong)kdata[i])<<((i&7)<<3); \
    } \
    ulong ipad[16],M[16]; \
    for (int i=0;i<16;i++) ipad[i]=key_block[i]^0x3636363636363636UL; \
    for (int i=0;i<16;i++) M[i]=bswap64(ipad[i]); \
    ulong istate[8]={IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7}; \
    sha512_block(istate,M); \
    for (int i=0;i<16;i++) M[i]=0; \
    if (mlen<=111) { \
        S512_copy_bytes(M,0,mdata,mlen); S512_set_byte(M,mlen,0x80); \
        M[15]=(ulong)(128+mlen)*8; sha512_block(istate,M); \
    } else { \
        S512_copy_bytes(M,0,mdata,mlen<128?mlen:128); \
        if (mlen<128) S512_set_byte(M,mlen,0x80); \
        sha512_block(istate,M); for (int i=0;i<16;i++) M[i]=0; \
        int rem=mlen-128; if (rem>0) S512_copy_bytes(M,0,mdata+128,rem); \
        if (mlen>=128) S512_set_byte(M,rem>0?rem:0,0x80); \
        M[15]=(ulong)(128+mlen)*8; sha512_block(istate,M); \
    } \
    ulong opad2[16]; \
    for (int i=0;i<16;i++) opad2[i]=key_block[i]^0x5c5c5c5c5c5c5c5cUL; \
    for (int i=0;i<16;i++) M[i]=bswap64(opad2[i]); \
    ulong ostate[8]={IV0,IV1,IV2,IV3,IV4,IV5,IV6,IV7}; \
    sha512_block(ostate,M); \
    for (int i=0;i<(inner_bytes/8);i++) M[i]=istate[i]; \
    M[inner_bytes/8]=0x8000000000000000UL; \
    for (int i=(inner_bytes/8)+1;i<15;i++) M[i]=0; \
    M[15]=(ulong)(128+inner_bytes)*8; \
    sha512_block(ostate,M); \
    ulong ss0=bswap64(ostate[0]),ss1=bswap64(ostate[1]); \
    uint hx=(uint)ss0,hy=(uint)(ss0>>32),hz=(uint)ss1,hw=(uint)(ss1>>32);

kernel void hmac_sha512_ksalt_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if (word_idx>=params.num_words) return;
    int mlen=hexlens[word_idx]; device const uint8_t *mdata=hexhashes+word_idx*256;
    uint soff=salt_offsets[salt_idx]; int klen=salt_lens[salt_idx];
    device const uint8_t *kdata=salts+soff;
    HMAC_SHA512_BODY(SHA512_IV0,SHA512_IV1,SHA512_IV2,SHA512_IV3,SHA512_IV4,SHA512_IV5,SHA512_IV6,SHA512_IV7,64)
    PROBE_AND_HIT_SHA512(ostate,word_idx,salt_idx)
}

kernel void hmac_sha512_kpass_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if (word_idx>=params.num_words) return;
    int klen=hexlens[word_idx]; device const uint8_t *kdata=hexhashes+word_idx*256;
    uint soff=salt_offsets[salt_idx]; int mlen=salt_lens[salt_idx];
    device const uint8_t *mdata=salts+soff;
    HMAC_SHA512_BODY(SHA512_IV0,SHA512_IV1,SHA512_IV2,SHA512_IV3,SHA512_IV4,SHA512_IV5,SHA512_IV6,SHA512_IV7,64)
    PROBE_AND_HIT_SHA512(ostate,word_idx,salt_idx)
}

#define SHA384_IV0 0xcbbb9d5dc1059ed8UL
#define SHA384_IV1 0x629a292a367cd507UL
#define SHA384_IV2 0x9159015a3070dd17UL
#define SHA384_IV3 0x152fecd8f70e5939UL
#define SHA384_IV4 0x67332667ffc00b31UL
#define SHA384_IV5 0x8eb44a8768581511UL
#define SHA384_IV6 0xdb0c2e0d64f98fa7UL
#define SHA384_IV7 0x47b5481dbefa4fa4UL

kernel void hmac_sha384_ksalt_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if (word_idx>=params.num_words) return;
    int mlen=hexlens[word_idx]; device const uint8_t *mdata=hexhashes+word_idx*256;
    uint soff=salt_offsets[salt_idx]; int klen=salt_lens[salt_idx];
    device const uint8_t *kdata=salts+soff;
    HMAC_SHA512_BODY(SHA384_IV0,SHA384_IV1,SHA384_IV2,SHA384_IV3,SHA384_IV4,SHA384_IV5,SHA384_IV6,SHA384_IV7,48)
    PROBE_AND_HIT_SHA384(ostate,word_idx,salt_idx)
}

kernel void hmac_sha384_kpass_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if (word_idx>=params.num_words) return;
    int klen=hexlens[word_idx]; device const uint8_t *kdata=hexhashes+word_idx*256;
    uint soff=salt_offsets[salt_idx]; int mlen=salt_lens[salt_idx];
    device const uint8_t *mdata=salts+soff;
    HMAC_SHA512_BODY(SHA384_IV0,SHA384_IV1,SHA384_IV2,SHA384_IV3,SHA384_IV4,SHA384_IV5,SHA384_IV6,SHA384_IV7,48)
    PROBE_AND_HIT_SHA384(ostate,word_idx,salt_idx)
}

/* SHA512(pass + salt) */
kernel void sha512passsalt_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if (word_idx>=params.num_words) return;
    int plen=hexlens[word_idx]; device const uint8_t *pass=hexhashes+word_idx*256;
    uint soff=salt_offsets[salt_idx]; int slen=salt_lens[salt_idx];
    int total_len=plen+slen;
    ulong state[8]={SHA512_IV0,SHA512_IV1,SHA512_IV2,SHA512_IV3,SHA512_IV4,SHA512_IV5,SHA512_IV6,SHA512_IV7};
    ulong M[16]; for (int i=0;i<16;i++) M[i]=0;
    if (total_len<=111) {
        S512_copy_bytes(M,0,pass,plen); S512_copy_bytes(M,plen,salts+soff,slen);
        S512_set_byte(M,total_len,0x80); M[15]=(ulong)total_len*8;
        sha512_block(state,M);
    } else {
        int pb1=plen<128?plen:128; S512_copy_bytes(M,0,pass,pb1);
        int sb1=128-pb1; if (sb1>slen) sb1=slen;
        if (sb1>0) S512_copy_bytes(M,pb1,salts+soff,sb1);
        if (total_len<128) S512_set_byte(M,total_len,0x80);
        sha512_block(state,M); for (int i=0;i<16;i++) M[i]=0;
        int p2=0; int pb2=plen-pb1;
        if (pb2>0) { S512_copy_bytes(M,0,pass+pb1,pb2); p2=pb2; }
        int sb2=slen-sb1; if (sb2>0) { S512_copy_bytes(M,p2,salts+soff+sb1,sb2); p2+=sb2; }
        if (total_len>=128) S512_set_byte(M,p2,0x80);
        M[15]=(ulong)total_len*8; sha512_block(state,M);
    }
    PROBE_AND_HIT_SHA512(state,word_idx,salt_idx)
}

/* SHA512(salt + pass) */
kernel void sha512saltpass_batch(SALTED_PARAMS) {
    uint word_idx=tid/params.num_salts; uint salt_idx=params.salt_start+(tid%params.num_salts);
    if (word_idx>=params.num_words) return;
    int plen=hexlens[word_idx]; device const uint8_t *pass=hexhashes+word_idx*256;
    uint soff=salt_offsets[salt_idx]; int slen=salt_lens[salt_idx];
    int total_len=slen+plen;
    ulong state[8]={SHA512_IV0,SHA512_IV1,SHA512_IV2,SHA512_IV3,SHA512_IV4,SHA512_IV5,SHA512_IV6,SHA512_IV7};
    ulong M[16]; for (int i=0;i<16;i++) M[i]=0;
    if (total_len<=111) {
        S512_copy_bytes(M,0,salts+soff,slen); S512_copy_bytes(M,slen,pass,plen);
        S512_set_byte(M,total_len,0x80); M[15]=(ulong)total_len*8;
        sha512_block(state,M);
    } else {
        int sb1=slen<128?slen:128; S512_copy_bytes(M,0,salts+soff,sb1);
        int pb1=128-sb1; if (pb1>plen) pb1=plen;
        if (pb1>0) S512_copy_bytes(M,sb1,pass,pb1);
        if (total_len<128) S512_set_byte(M,total_len,0x80);
        sha512_block(state,M); for (int i=0;i<16;i++) M[i]=0;
        int p2=0; int sb2=slen-sb1;
        if (sb2>0) { S512_copy_bytes(M,0,salts+soff+sb1,sb2); p2=sb2; }
        int pb2=plen-pb1; if (pb2>0) { S512_copy_bytes(M,p2,pass+pb1,pb2); p2+=pb2; }
        if (total_len>=128) S512_set_byte(M,p2,0x80);
        M[15]=(ulong)total_len*8; sha512_block(state,M);
    }
    PROBE_AND_HIT_SHA512(state,word_idx,salt_idx)
}
