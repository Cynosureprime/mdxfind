#include <metal_stdlib>
using namespace metal;

/* MD5 constants */
constant uint K[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};
constant uint S[64] = {
    7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
    5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
    4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
    6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21
};
constant uint G[64] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    1,6,11,0,5,10,15,4,9,14,3,8,13,2,7,12,
    5,8,11,14,1,4,7,10,13,0,3,6,9,12,15,2,
    0,7,14,5,12,3,10,1,8,15,6,13,4,11,2,9
};

/* MD5 compress: one 64-byte block */
static void md5_block(thread uint4 &state, thread const uint *M) {
    uint a = state.x, b = state.y, c = state.z, d = state.w;
    for (int i = 0; i < 64; i++) {
        uint f, g = G[i];
        if (i < 16)      f = (b & c) | (~b & d);
        else if (i < 32) f = (d & b) | (~d & c);
        else if (i < 48) f = b ^ c ^ d;
        else              f = c ^ (~d | b);
        f = f + a + K[i] + M[g];
        a = d; d = c; c = b;
        b = b + ((f << S[i]) | (f >> (32 - S[i])));
    }
    state += uint4(a, b, c, d);
}

/* MD5 compress from round 8: fully unrolled, no branches.
 * Takes pre-computed (a,b,c,d) after rounds 0-7.
 * Adds IV to produce final hash. */
#define FF(a,b,c,d,m,s,k) { a += ((b&c)|(~b&d)) + m + k; a = b + ((a<<s)|(a>>(32-s))); }
#define GG(a,b,c,d,m,s,k) { a += ((d&b)|(~d&c)) + m + k; a = b + ((a<<s)|(a>>(32-s))); }
#define HH(a,b,c,d,m,s,k) { a += (b^c^d) + m + k; a = b + ((a<<s)|(a>>(32-s))); }
#define II(a,b,c,d,m,s,k) { a += (c^(~d|b)) + m + k; a = b + ((a<<s)|(a>>(32-s))); }

/* Fully unrolled MD5 compress — all 64 rounds with literal constants.
 * No branches, no array lookups, no variable shifts.
 * Critical for iterated algorithms (PHPBB3) where this runs 2048+ times. */
__attribute__((always_inline))
static void md5_block_full(thread uint4 &state, thread const uint *M) {
    uint a = state.x, b = state.y, c = state.z, d = state.w;
    /* Rounds 0-7: F function */
    FF(a,b,c,d, M[ 0], 7, 0xd76aa478)
    FF(d,a,b,c, M[ 1],12, 0xe8c7b756)
    FF(c,d,a,b, M[ 2],17, 0x242070db)
    FF(b,c,d,a, M[ 3],22, 0xc1bdceee)
    FF(a,b,c,d, M[ 4], 7, 0xf57c0faf)
    FF(d,a,b,c, M[ 5],12, 0x4787c62a)
    FF(c,d,a,b, M[ 6],17, 0xa8304613)
    FF(b,c,d,a, M[ 7],22, 0xfd469501)
    /* Rounds 8-15: F function */
    FF(a,b,c,d, M[ 8], 7, 0x698098d8)
    FF(d,a,b,c, M[ 9],12, 0x8b44f7af)
    FF(c,d,a,b, M[10],17, 0xffff5bb1)
    FF(b,c,d,a, M[11],22, 0x895cd7be)
    FF(a,b,c,d, M[12], 7, 0x6b901122)
    FF(d,a,b,c, M[13],12, 0xfd987193)
    FF(c,d,a,b, M[14],17, 0xa679438e)
    FF(b,c,d,a, M[15],22, 0x49b40821)
    /* Rounds 16-31: G function */
    GG(a,b,c,d, M[ 1], 5, 0xf61e2562)
    GG(d,a,b,c, M[ 6], 9, 0xc040b340)
    GG(c,d,a,b, M[11],14, 0x265e5a51)
    GG(b,c,d,a, M[ 0],20, 0xe9b6c7aa)
    GG(a,b,c,d, M[ 5], 5, 0xd62f105d)
    GG(d,a,b,c, M[10], 9, 0x02441453)
    GG(c,d,a,b, M[15],14, 0xd8a1e681)
    GG(b,c,d,a, M[ 4],20, 0xe7d3fbc8)
    GG(a,b,c,d, M[ 9], 5, 0x21e1cde6)
    GG(d,a,b,c, M[14], 9, 0xc33707d6)
    GG(c,d,a,b, M[ 3],14, 0xf4d50d87)
    GG(b,c,d,a, M[ 8],20, 0x455a14ed)
    GG(a,b,c,d, M[13], 5, 0xa9e3e905)
    GG(d,a,b,c, M[ 2], 9, 0xfcefa3f8)
    GG(c,d,a,b, M[ 7],14, 0x676f02d9)
    GG(b,c,d,a, M[12],20, 0x8d2a4c8a)
    /* Rounds 32-47: H function */
    HH(a,b,c,d, M[ 5], 4, 0xfffa3942)
    HH(d,a,b,c, M[ 8],11, 0x8771f681)
    HH(c,d,a,b, M[11],16, 0x6d9d6122)
    HH(b,c,d,a, M[14],23, 0xfde5380c)
    HH(a,b,c,d, M[ 1], 4, 0xa4beea44)
    HH(d,a,b,c, M[ 4],11, 0x4bdecfa9)
    HH(c,d,a,b, M[ 7],16, 0xf6bb4b60)
    HH(b,c,d,a, M[10],23, 0xbebfbc70)
    HH(a,b,c,d, M[13], 4, 0x289b7ec6)
    HH(d,a,b,c, M[ 0],11, 0xeaa127fa)
    HH(c,d,a,b, M[ 3],16, 0xd4ef3085)
    HH(b,c,d,a, M[ 6],23, 0x04881d05)
    HH(a,b,c,d, M[ 9], 4, 0xd9d4d039)
    HH(d,a,b,c, M[12],11, 0xe6db99e5)
    HH(c,d,a,b, M[15],16, 0x1fa27cf8)
    HH(b,c,d,a, M[ 2],23, 0xc4ac5665)
    /* Rounds 48-63: I function */
    II(a,b,c,d, M[ 0], 6, 0xf4292244)
    II(d,a,b,c, M[ 7],10, 0x432aff97)
    II(c,d,a,b, M[14],15, 0xab9423a7)
    II(b,c,d,a, M[ 5],21, 0xfc93a039)
    II(a,b,c,d, M[12], 6, 0x655b59c3)
    II(d,a,b,c, M[ 3],10, 0x8f0ccc92)
    II(c,d,a,b, M[10],15, 0xffeff47d)
    II(b,c,d,a, M[ 1],21, 0x85845dd1)
    II(a,b,c,d, M[ 8], 6, 0x6fa87e4f)
    II(d,a,b,c, M[15],10, 0xfe2ce6e0)
    II(c,d,a,b, M[ 6],15, 0xa3014314)
    II(b,c,d,a, M[13],21, 0x4e0811a1)
    II(a,b,c,d, M[ 4], 6, 0xf7537e82)
    II(d,a,b,c, M[11],10, 0xbd3af235)
    II(c,d,a,b, M[ 2],15, 0x2ad7d2bb)
    II(b,c,d,a, M[ 9],21, 0xeb86d391)
    state += uint4(a, b, c, d);
}

/* MD5 compress for the padding block of a 64-byte message.
 * M[] = {0x80, 0, 0, ..., 0, 512, 0} — all constants, zero memory access.
 * The compiler folds M[g] into each round constant. */
__attribute__((always_inline))
static void md5_block_pad64(thread uint4 &state) {
    uint a = state.x, b = state.y, c = state.z, d = state.w;
    /* M[0]=0x80, M[14]=512, all others=0 */
    FF(a,b,c,d, 0x80u,  7, 0xd76aa478)   /* r0:  M[0]  */
    FF(d,a,b,c, 0,     12, 0xe8c7b756)   /* r1:  M[1]  */
    FF(c,d,a,b, 0,     17, 0x242070db)   /* r2:  M[2]  */
    FF(b,c,d,a, 0,     22, 0xc1bdceee)   /* r3:  M[3]  */
    FF(a,b,c,d, 0,      7, 0xf57c0faf)   /* r4:  M[4]  */
    FF(d,a,b,c, 0,     12, 0x4787c62a)   /* r5:  M[5]  */
    FF(c,d,a,b, 0,     17, 0xa8304613)   /* r6:  M[6]  */
    FF(b,c,d,a, 0,     22, 0xfd469501)   /* r7:  M[7]  */
    FF(a,b,c,d, 0,      7, 0x698098d8)   /* r8:  M[8]  */
    FF(d,a,b,c, 0,     12, 0x8b44f7af)   /* r9:  M[9]  */
    FF(c,d,a,b, 0,     17, 0xffff5bb1)   /* r10: M[10] */
    FF(b,c,d,a, 0,     22, 0x895cd7be)   /* r11: M[11] */
    FF(a,b,c,d, 0,      7, 0x6b901122)   /* r12: M[12] */
    FF(d,a,b,c, 0,     12, 0xfd987193)   /* r13: M[13] */
    FF(c,d,a,b, 512u,  17, 0xa679438e)   /* r14: M[14] */
    FF(b,c,d,a, 0,     22, 0x49b40821)   /* r15: M[15] */
    GG(a,b,c,d, 0,      5, 0xf61e2562)   /* r16: M[1]  */
    GG(d,a,b,c, 0,      9, 0xc040b340)   /* r17: M[6]  */
    GG(c,d,a,b, 0,     14, 0x265e5a51)   /* r18: M[11] */
    GG(b,c,d,a, 0x80u, 20, 0xe9b6c7aa)   /* r19: M[0]  */
    GG(a,b,c,d, 0,      5, 0xd62f105d)   /* r20: M[5]  */
    GG(d,a,b,c, 0,      9, 0x02441453)   /* r21: M[10] */
    GG(c,d,a,b, 0,     14, 0xd8a1e681)   /* r22: M[15] */
    GG(b,c,d,a, 0,     20, 0xe7d3fbc8)   /* r23: M[4]  */
    GG(a,b,c,d, 0,      5, 0x21e1cde6)   /* r24: M[9]  */
    GG(d,a,b,c, 512u,   9, 0xc33707d6)   /* r25: M[14] */
    GG(c,d,a,b, 0,     14, 0xf4d50d87)   /* r26: M[3]  */
    GG(b,c,d,a, 0,     20, 0x455a14ed)   /* r27: M[8]  */
    GG(a,b,c,d, 0,      5, 0xa9e3e905)   /* r28: M[13] */
    GG(d,a,b,c, 0,      9, 0xfcefa3f8)   /* r29: M[2]  */
    GG(c,d,a,b, 0,     14, 0x676f02d9)   /* r30: M[7]  */
    GG(b,c,d,a, 0,     20, 0x8d2a4c8a)   /* r31: M[12] */
    HH(a,b,c,d, 0,      4, 0xfffa3942)   /* r32: M[5]  */
    HH(d,a,b,c, 0,     11, 0x8771f681)   /* r33: M[8]  */
    HH(c,d,a,b, 0,     16, 0x6d9d6122)   /* r34: M[11] */
    HH(b,c,d,a, 512u,  23, 0xfde5380c)   /* r35: M[14] */
    HH(a,b,c,d, 0,      4, 0xa4beea44)   /* r36: M[1]  */
    HH(d,a,b,c, 0,     11, 0x4bdecfa9)   /* r37: M[4]  */
    HH(c,d,a,b, 0,     16, 0xf6bb4b60)   /* r38: M[7]  */
    HH(b,c,d,a, 0,     23, 0xbebfbc70)   /* r39: M[10] */
    HH(a,b,c,d, 0,      4, 0x289b7ec6)   /* r40: M[13] */
    HH(d,a,b,c, 0x80u, 11, 0xeaa127fa)   /* r41: M[0]  */
    HH(c,d,a,b, 0,     16, 0xd4ef3085)   /* r42: M[3]  */
    HH(b,c,d,a, 0,     23, 0x04881d05)   /* r43: M[6]  */
    HH(a,b,c,d, 0,      4, 0xd9d4d039)   /* r44: M[9]  */
    HH(d,a,b,c, 0,     11, 0xe6db99e5)   /* r45: M[12] */
    HH(c,d,a,b, 0,     16, 0x1fa27cf8)   /* r46: M[15] */
    HH(b,c,d,a, 0,     23, 0xc4ac5665)   /* r47: M[2]  */
    II(a,b,c,d, 0x80u,  6, 0xf4292244)   /* r48: M[0]  */
    II(d,a,b,c, 0,     10, 0x432aff97)   /* r49: M[7]  */
    II(c,d,a,b, 512u,  15, 0xab9423a7)   /* r50: M[14] */
    II(b,c,d,a, 0,     21, 0xfc93a039)   /* r51: M[5]  */
    II(a,b,c,d, 0,      6, 0x655b59c3)   /* r52: M[12] */
    II(d,a,b,c, 0,     10, 0x8f0ccc92)   /* r53: M[3]  */
    II(c,d,a,b, 0,     15, 0xffeff47d)   /* r54: M[10] */
    II(b,c,d,a, 0,     21, 0x85845dd1)   /* r55: M[1]  */
    II(a,b,c,d, 0,      6, 0x6fa87e4f)   /* r56: M[8]  */
    II(d,a,b,c, 0,     10, 0xfe2ce6e0)   /* r57: M[15] */
    II(c,d,a,b, 0,     15, 0xa3014314)   /* r58: M[6]  */
    II(b,c,d,a, 0,     21, 0x4e0811a1)   /* r59: M[13] */
    II(a,b,c,d, 0,      6, 0xf7537e82)   /* r60: M[4]  */
    II(d,a,b,c, 0,     10, 0xbd3af235)   /* r61: M[11] */
    II(c,d,a,b, 0,     15, 0x2ad7d2bb)   /* r62: M[2]  */
    II(b,c,d,a, 0,     21, 0xeb86d391)   /* r63: M[9]  */
    state += uint4(a, b, c, d);
}

static void md5_block_from8(thread uint4 &state, thread const uint *M) {
    uint a = state.x, b = state.y, c = state.z, d = state.w;
    /* Rounds 8-15: F function */
    FF(a,b,c,d, M[ 8], 7, 0x698098d8)
    FF(d,a,b,c, M[ 9],12, 0x8b44f7af)
    FF(c,d,a,b, M[10],17, 0xffff5bb1)
    FF(b,c,d,a, M[11],22, 0x895cd7be)
    FF(a,b,c,d, M[12], 7, 0x6b901122)
    FF(d,a,b,c, M[13],12, 0xfd987193)
    FF(c,d,a,b, M[14],17, 0xa679438e)
    FF(b,c,d,a, M[15],22, 0x49b40821)
    /* Rounds 16-31: G function */
    GG(a,b,c,d, M[ 1], 5, 0xf61e2562)
    GG(d,a,b,c, M[ 6], 9, 0xc040b340)
    GG(c,d,a,b, M[11],14, 0x265e5a51)
    GG(b,c,d,a, M[ 0],20, 0xe9b6c7aa)
    GG(a,b,c,d, M[ 5], 5, 0xd62f105d)
    GG(d,a,b,c, M[10], 9, 0x02441453)
    GG(c,d,a,b, M[15],14, 0xd8a1e681)
    GG(b,c,d,a, M[ 4],20, 0xe7d3fbc8)
    GG(a,b,c,d, M[ 9], 5, 0x21e1cde6)
    GG(d,a,b,c, M[14], 9, 0xc33707d6)
    GG(c,d,a,b, M[ 3],14, 0xf4d50d87)
    GG(b,c,d,a, M[ 8],20, 0x455a14ed)
    GG(a,b,c,d, M[13], 5, 0xa9e3e905)
    GG(d,a,b,c, M[ 2], 9, 0xfcefa3f8)
    GG(c,d,a,b, M[ 7],14, 0x676f02d9)
    GG(b,c,d,a, M[12],20, 0x8d2a4c8a)
    /* Rounds 32-47: H function */
    HH(a,b,c,d, M[ 5], 4, 0xfffa3942)
    HH(d,a,b,c, M[ 8],11, 0x8771f681)
    HH(c,d,a,b, M[11],16, 0x6d9d6122)
    HH(b,c,d,a, M[14],23, 0xfde5380c)
    HH(a,b,c,d, M[ 1], 4, 0xa4beea44)
    HH(d,a,b,c, M[ 4],11, 0x4bdecfa9)
    HH(c,d,a,b, M[ 7],16, 0xf6bb4b60)
    HH(b,c,d,a, M[10],23, 0xbebfbc70)
    HH(a,b,c,d, M[13], 4, 0x289b7ec6)
    HH(d,a,b,c, M[ 0],11, 0xeaa127fa)
    HH(c,d,a,b, M[ 3],16, 0xd4ef3085)
    HH(b,c,d,a, M[ 6],23, 0x04881d05)
    HH(a,b,c,d, M[ 9], 4, 0xd9d4d039)
    HH(d,a,b,c, M[12],11, 0xe6db99e5)
    HH(c,d,a,b, M[15],16, 0x1fa27cf8)
    HH(b,c,d,a, M[ 2],23, 0xc4ac5665)
    /* Rounds 48-63: I function */
    II(a,b,c,d, M[ 0], 6, 0xf4292244)
    II(d,a,b,c, M[ 7],10, 0x432aff97)
    II(c,d,a,b, M[14],15, 0xab9423a7)
    II(b,c,d,a, M[ 5],21, 0xfc93a039)
    II(a,b,c,d, M[12], 6, 0x655b59c3)
    II(d,a,b,c, M[ 3],10, 0x8f0ccc92)
    II(c,d,a,b, M[10],15, 0xffeff47d)
    II(b,c,d,a, M[ 1],21, 0x85845dd1)
    II(a,b,c,d, M[ 8], 6, 0x6fa87e4f)
    II(d,a,b,c, M[15],10, 0xfe2ce6e0)
    II(c,d,a,b, M[ 6],15, 0xa3014314)
    II(b,c,d,a, M[13],21, 0x4e0811a1)
    II(a,b,c,d, M[ 4], 6, 0xf7537e82)
    II(d,a,b,c, M[11],10, 0xbd3af235)
    II(c,d,a,b, M[ 2],15, 0x2ad7d2bb)
    II(b,c,d,a, M[ 9],21, 0xeb86d391)
    state = uint4(0x67452301 + a, 0xEFCDAB89 + b, 0x98BADCFE + c, 0x10325476 + d);
}

/* Full MD5 hash for messages up to 55 bytes (single block) */
static void md5_short(thread const uint8_t *msg, int len, thread uint4 &hash) {
    uint M[16] = {0};
    for (int i = 0; i < len; i++)
        ((thread uint8_t *)M)[i] = msg[i];
    ((thread uint8_t *)M)[len] = 0x80;
    M[14] = len * 8;
    hash = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
    md5_block(hash, M);
}

/* MD5 for messages 56-119 bytes (two blocks) */
static void md5_two(thread const uint8_t *msg, int len, thread uint4 &hash) {
    uint M[16] = {0};
    /* first block */
    for (int i = 0; i < 64 && i < len; i++)
        ((thread uint8_t *)M)[i] = msg[i];
    if (len < 64) {
        ((thread uint8_t *)M)[len] = 0x80;
        if (len < 56) { M[14] = len * 8; }
    }
    hash = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
    md5_block(hash, M);
    if (len >= 56) {
        /* second block */
        for (int i = 0; i < 16; i++) M[i] = 0;
        for (int i = 64; i < len; i++)
            ((thread uint8_t *)M)[i - 64] = msg[i];
        if (len >= 64)
            ((thread uint8_t *)M)[len - 64] = 0x80;
        M[14] = len * 8;
        md5_block(hash, M);
    }
}

/* Hex-encode 16 bytes to 32 chars */
static void hex_encode(thread const uint8_t *bin, thread uint8_t *hex) {
    for (int i = 0; i < 16; i++) {
        uint8_t hi = bin[i] >> 4;
        uint8_t lo = bin[i] & 0x0f;
        hex[i*2]   = hi < 10 ? hi + '0' : hi - 10 + 'a';
        hex[i*2+1] = lo < 10 ? lo + '0' : lo - 10 + 'a';
    }
}

/* ---- Byte-swap utilities ---- */

static inline uint bswap32(uint x) {
    return ((x >> 24) & 0xffu) | ((x >> 8) & 0xff00u) |
           ((x << 8) & 0xff0000u) | ((x << 24) & 0xff000000u);
}

static inline ulong bswap64(ulong x) {
    return ((x >> 56) & 0xffUL) | ((x >> 40) & 0xff00UL) |
           ((x >> 24) & 0xff0000UL) | ((x >> 8) & 0xff000000UL) |
           ((x << 8) & 0xff00000000UL) | ((x << 24) & 0xff0000000000UL) |
           ((x << 40) & 0xff000000000000UL) | ((x << 56) & 0xff00000000000000UL);
}

static inline ulong rotr64(ulong x, uint n) { return (x >> n) | (x << (64 - n)); }

/* ---- Hex encoding helpers ---- */

/* Encode one byte as two hex chars packed into low 16 bits, little-endian order */
static uint hex_byte_lc(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    uint hc = hi + ((hi < 10) ? '0' : ('a' - 10));
    uint lc = lo + ((lo < 10) ? '0' : ('a' - 10));
    return hc | (lc << 8);
}

/* Encode one byte as two hex chars packed into low 16 bits, big-endian order */
static uint hex_byte_be(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    return ((hi + ((hi < 10) ? '0' : ('a' - 10))) << 8)
         |  (lo + ((lo < 10) ? '0' : ('a' - 10)));
}

/* 64-bit variant of hex_byte_be for SHA-512 hex encoding */
static ulong hex_byte_be64(uint b) {
    uint hi = (b >> 4) & 0xf;
    uint lo = b & 0xf;
    return ((ulong)(hi + ((hi < 10) ? '0' : ('a' - 10))) << 8)
         |  (ulong)(lo + ((lo < 10) ? '0' : ('a' - 10)));
}

/* Hex-encode 4 LE uint32 hash words to 32 bytes in M[0..7] */
static void md5_to_hex_lc(uint hx, uint hy, uint hz, uint hw, thread uint *M) {
    uint v[4] = {hx, hy, hz, hw};
    for (int i = 0; i < 4; i++) {
        uint b0 = v[i] & 0xff, b1 = (v[i]>>8) & 0xff;
        uint b2 = (v[i]>>16) & 0xff, b3 = (v[i]>>24) & 0xff;
        M[i*2]   = hex_byte_lc(b0) | (hex_byte_lc(b1) << 16);
        M[i*2+1] = hex_byte_lc(b2) | (hex_byte_lc(b3) << 16);
    }
}

/* ---- Big-endian byte manipulation for SHA1/SHA256/SHA512 M[] blocks ---- */

/* Copy bytes from device memory into big-endian uint32 M[] */
static void S_copy_bytes(thread uint *M, int byte_off, device const uint8_t *src, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        int wi = (byte_off + i) / 4;
        int bi = 3 - ((byte_off + i) % 4);
        M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)src[i] << (bi * 8));
    }
}

static void S_set_byte(thread uint *M, int byte_off, uint8_t val) {
    int wi = byte_off / 4;
    int bi = 3 - (byte_off % 4);
    M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)val << (bi * 8));
}

/* Copy bytes from thread-private memory into big-endian M[] */
static void S_copy_bytes_priv(thread uint *M, int byte_off, thread const uint8_t *src, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        int wi = (byte_off + i) / 4;
        int bi = 3 - ((byte_off + i) % 4);
        M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)src[i] << (bi * 8));
    }
}

static void S_set_byte_priv(thread uint *M, int byte_off, uint8_t val) {
    int wi = byte_off / 4;
    int bi = 3 - (byte_off % 4);
    M[wi] = (M[wi] & ~(0xffu << (bi * 8))) | ((uint)val << (bi * 8));
}

/* ---- SHA512 byte helpers (big-endian ulong[16]) ---- */

static void S512_copy_bytes(thread ulong *M, int byte_off, device const uint8_t *src, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        int wi = (byte_off + i) / 8;
        int bi = 7 - ((byte_off + i) % 8);
        M[wi] = (M[wi] & ~(0xffUL << (bi * 8))) | ((ulong)src[i] << (bi * 8));
    }
}

static void S512_copy_bytes_p(thread ulong *M, int byte_off, thread const uint8_t *src, int nbytes) {
    for (int i = 0; i < nbytes; i++) {
        int wi = (byte_off + i) / 8;
        int bi = 7 - ((byte_off + i) % 8);
        M[wi] = (M[wi] & ~(0xffUL << (bi * 8))) | ((ulong)src[i] << (bi * 8));
    }
}

static void S512_set_byte(thread ulong *M, int byte_off, uint8_t val) {
    int wi = byte_off / 8;
    int bi = 7 - (byte_off % 8);
    M[wi] = (M[wi] & ~(0xffUL << (bi * 8))) | ((ulong)val << (bi * 8));
}

/* ---- SHA512 block function ---- */

/* rotr64 already defined above (Whirlpool section) */

constant ulong K512[80] = {
    0x428a2f98d728ae22UL,0x7137449123ef65cdUL,0xb5c0fbcfec4d3b2fUL,0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL,0x59f111f1b605d019UL,0x923f82a4af194f9bUL,0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL,0x12835b0145706fbeUL,0x243185be4ee4b28cUL,0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL,0x80deb1fe3b1696b1UL,0x9bdc06a725c71235UL,0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL,0xefbe4786384f25e3UL,0x0fc19dc68b8cd5b5UL,0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL,0x4a7484aa6ea6e483UL,0x5cb0a9dcbd41fbd4UL,0x76f988da831153b5UL,
    0x983e5152ee66dfabUL,0xa831c66d2db43210UL,0xb00327c898fb213fUL,0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL,0xd5a79147930aa725UL,0x06ca6351e003826fUL,0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL,0x2e1b21385c26c926UL,0x4d2c6dfc5ac42aedUL,0x53380d139d95b3dfUL,
    0x650a73548baf63deUL,0x766a0abb3c77b2a8UL,0x81c2c92e47edaee6UL,0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL,0xa81a664bbc423001UL,0xc24b8b70d0f89791UL,0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL,0xd69906245565a910UL,0xf40e35855771202aUL,0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL,0x1e376c085141ab53UL,0x2748774cdf8eeb99UL,0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL,0x4ed8aa4ae3418acbUL,0x5b9cca4f7763e373UL,0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL,0x78a5636f43172f60UL,0x84c87814a1f0ab72UL,0x8cc702081a6439ecUL,
    0x90befffa23631e28UL,0xa4506cebde82bde9UL,0xbef9a3f7b2c67915UL,0xc67178f2e372532bUL,
    0xca273eceea26619cUL,0xd186b8c721c0c207UL,0xeada7dd6cde0eb1eUL,0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL,0x0a637dc5a2c898a6UL,0x113f9804bef90daeUL,0x1b710b35131c471bUL,
    0x28db77f523047d84UL,0x32caab7b40c72493UL,0x3c9ebe0a15c9bebcUL,0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL,0x597f299cfc657e2aUL,0x5fcb6fab3ad6faecUL,0x6c44198c4a475817UL
};

static void sha512_block(thread ulong *state, thread ulong *M) {
    ulong W[16];
    for (int i=0;i<16;i++) W[i]=M[i];
    ulong a=state[0],b=state[1],c=state[2],d=state[3],
          e=state[4],f=state[5],g=state[6],h=state[7];
    #define S5R(i,wi) { \
        ulong S1=rotr64(e,14)^rotr64(e,18)^rotr64(e,41); \
        ulong ch=(e&f)^(~e&g); ulong t1=h+S1+ch+K512[i]+(wi); \
        ulong S0=rotr64(a,28)^rotr64(a,34)^rotr64(a,39); \
        ulong maj=(a&b)^(a&c)^(b&c); ulong t2=S0+maj; \
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
    #define S5W(i) { int j=(i)&15; \
        ulong x=W[((i)-15)&15]; ulong s0=rotr64(x,1)^rotr64(x,8)^(x>>7); \
        x=W[((i)-2)&15]; ulong s1=rotr64(x,19)^rotr64(x,61)^(x>>6); \
        W[j]=W[j]+s0+W[((i)-7)&15]+s1; }
    S5R(0,W[0]);S5R(1,W[1]);S5R(2,W[2]);S5R(3,W[3]);
    S5R(4,W[4]);S5R(5,W[5]);S5R(6,W[6]);S5R(7,W[7]);
    S5R(8,W[8]);S5R(9,W[9]);S5R(10,W[10]);S5R(11,W[11]);
    S5R(12,W[12]);S5R(13,W[13]);S5R(14,W[14]);S5R(15,W[15]);
    S5W(16);S5R(16,W[0]);S5W(17);S5R(17,W[1]);S5W(18);S5R(18,W[2]);S5W(19);S5R(19,W[3]);
    S5W(20);S5R(20,W[4]);S5W(21);S5R(21,W[5]);S5W(22);S5R(22,W[6]);S5W(23);S5R(23,W[7]);
    S5W(24);S5R(24,W[8]);S5W(25);S5R(25,W[9]);S5W(26);S5R(26,W[10]);S5W(27);S5R(27,W[11]);
    S5W(28);S5R(28,W[12]);S5W(29);S5R(29,W[13]);S5W(30);S5R(30,W[14]);S5W(31);S5R(31,W[15]);
    S5W(32);S5R(32,W[0]);S5W(33);S5R(33,W[1]);S5W(34);S5R(34,W[2]);S5W(35);S5R(35,W[3]);
    S5W(36);S5R(36,W[4]);S5W(37);S5R(37,W[5]);S5W(38);S5R(38,W[6]);S5W(39);S5R(39,W[7]);
    S5W(40);S5R(40,W[8]);S5W(41);S5R(41,W[9]);S5W(42);S5R(42,W[10]);S5W(43);S5R(43,W[11]);
    S5W(44);S5R(44,W[12]);S5W(45);S5R(45,W[13]);S5W(46);S5R(46,W[14]);S5W(47);S5R(47,W[15]);
    S5W(48);S5R(48,W[0]);S5W(49);S5R(49,W[1]);S5W(50);S5R(50,W[2]);S5W(51);S5R(51,W[3]);
    S5W(52);S5R(52,W[4]);S5W(53);S5R(53,W[5]);S5W(54);S5R(54,W[6]);S5W(55);S5R(55,W[7]);
    S5W(56);S5R(56,W[8]);S5W(57);S5R(57,W[9]);S5W(58);S5R(58,W[10]);S5W(59);S5R(59,W[11]);
    S5W(60);S5R(60,W[12]);S5W(61);S5R(61,W[13]);S5W(62);S5R(62,W[14]);S5W(63);S5R(63,W[15]);
    S5W(64);S5R(64,W[0]);S5W(65);S5R(65,W[1]);S5W(66);S5R(66,W[2]);S5W(67);S5R(67,W[3]);
    S5W(68);S5R(68,W[4]);S5W(69);S5R(69,W[5]);S5W(70);S5R(70,W[6]);S5W(71);S5R(71,W[7]);
    S5W(72);S5R(72,W[8]);S5W(73);S5R(73,W[9]);S5W(74);S5R(74,W[10]);S5W(75);S5R(75,W[11]);
    S5W(76);S5R(76,W[12]);S5W(77);S5R(77,W[13]);S5W(78);S5R(78,W[14]);S5W(79);S5R(79,W[15]);
    #undef S5R
    #undef S5W
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

/* ---- SHA1 block function ---- */

/* SHA1 compress: M[] must be big-endian uint32 words */
static void sha1_compress(thread uint *state, thread uint *M) {
    uint W[80];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 80; i++)
        W[i] = rotate(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1u);

    uint a = state[0], b = state[1], c = state[2], d = state[3], e = state[4];
    uint t;
    for (int i = 0; i < 20; i++) {
        t = rotate(a, 5u) + ((b & c) | (~b & d)) + e + 0x5A827999u + W[i];
        e = d; d = c; c = rotate(b, 30u); b = a; a = t;
    }
    for (int i = 20; i < 40; i++) {
        t = rotate(a, 5u) + (b ^ c ^ d) + e + 0x6ED9EBA1u + W[i];
        e = d; d = c; c = rotate(b, 30u); b = a; a = t;
    }
    for (int i = 40; i < 60; i++) {
        t = rotate(a, 5u) + ((b & c) | (b & d) | (c & d)) + e + 0x8F1BBCDCu + W[i];
        e = d; d = c; c = rotate(b, 30u); b = a; a = t;
    }
    for (int i = 60; i < 80; i++) {
        t = rotate(a, 5u) + (b ^ c ^ d) + e + 0xCA62C1D6u + W[i];
        e = d; d = c; c = rotate(b, 30u); b = a; a = t;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d; state[4] += e;
}

/* ---- SHA1 hex encoding for iteration ---- */

static void sha1_to_hex_lc(thread uint *state, thread uint *M) {
    for (int i = 0; i < 5; i++) {
        uint v = state[i];
        uint b0 = (v >> 24) & 0xff, b1 = (v >> 16) & 0xff;
        uint b2 = (v >> 8) & 0xff, b3 = v & 0xff;
        M[i*2]   = (hex_byte_be(b0) << 16) | hex_byte_be(b1);
        M[i*2+1] = (hex_byte_be(b2) << 16) | hex_byte_be(b3);
    }
}

/* ---- Salted kernel parameter list macro ---- */

#define SALTED_PARAMS \
    device const uint8_t *hexhashes [[buffer(0)]], device const ushort *hexlens [[buffer(1)]], \
    device const ushort *unused2 [[buffer(2)]], device const uint8_t *salts [[buffer(3)]], \
    device const uint *salt_offsets [[buffer(4)]], device const ushort *salt_lens [[buffer(5)]], \
    device const uint *compact_fp [[buffer(6)]], device const uint *compact_idx [[buffer(7)]], \
    constant MetalParams &params [[buffer(8)]], device const uint8_t *hash_data_buf [[buffer(9)]], \
    device const uint64_t *hash_data_off [[buffer(10)]], device const ushort *hash_data_len [[buffer(11)]], \
    device uint *hits [[buffer(12)]], device atomic_uint *hit_count [[buffer(13)]], \
    device const uint64_t *overflow_keys [[buffer(14)]], device const uint8_t *overflow_hashes [[buffer(15)]], \
    device const uint *overflow_offsets [[buffer(16)]], device const ushort *overflow_lengths [[buffer(17)]], \
    uint tid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint tgsize [[threads_per_threadgroup]]

/* ---- Compact table probe + overflow search macros ---- */

/* Universal hit entry stride: [0]=widx [1]=sidx [2]=iter [3..18]=hash[0..15] */
constant uint HIT_STRIDE = 19;

/* PROBE6: 4-word hash comparison, stride 19 hit recording.
 * h = uint4 with hash words, widx/sidx = word/salt indices.
 * Searches compact table + overflow, emits hit as:
 *   hits[base+0]=widx, [1]=sidx, [2]=1, [3..6]=hash, [7..18]=0 */
#define PROBE6(h, widx, sidx) { \
    ulong key = (ulong(h.y) << 32) | h.x; \
    uint fp = uint(key >> 32); if (fp == 0) fp = 1; \
    ulong pos = (key ^ (key >> 32)) & params.compact_mask; \
    bool found = false; \
    for (uint p = 0; p < params.max_probe && !found; p++) { \
        uint cfp = compact_fp[pos]; if (cfp == 0) break; \
        if (cfp == fp) { uint idx = compact_idx[pos]; \
            if (idx < params.hash_data_count) { \
                ulong off = hash_data_off[idx]; \
                device const uint *ref = (device const uint *)(hash_data_buf + off); \
                if (h.x == ref[0] && h.y == ref[1] && h.z == ref[2] && h.w == ref[3]) \
                    found = true; \
            } } \
        pos = (pos + 1) & params.compact_mask; \
    } \
    if (!found && params.overflow_count > 0) { \
        int lo = 0, hi2 = int(params.overflow_count) - 1; \
        while (lo <= hi2 && !found) { \
            int mid = (lo + hi2) / 2; ulong mkey = overflow_keys[mid]; \
            if (key < mkey) hi2 = mid - 1; \
            else if (key > mkey) lo = mid + 1; \
            else { \
                for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) { \
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]); \
                    if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) \
                        found = true; } \
                for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key && !found; d++) { \
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]); \
                    if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) \
                        found = true; } \
                break; } } } \
    if (found) { \
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed); \
        if (slot < params.max_hits) { \
            uint base = slot * HIT_STRIDE; \
            hits[base] = widx; hits[base+1] = sidx; hits[base+2] = 1; \
            hits[base+3] = h.x; hits[base+4] = h.y; \
            hits[base+5] = h.z; hits[base+6] = h.w; \
            for (uint _z = 7; _z < HIT_STRIDE; _z++) hits[base+_z] = 0; } } }

/* PROBE6_NOOVF: Same as PROBE6 but without overflow search */
#define PROBE6_NOOVF(h, widx, sidx) { \
    ulong key = (ulong(h.y) << 32) | h.x; \
    uint fp = uint(key >> 32); if (fp == 0) fp = 1; \
    ulong pos = (key ^ (key >> 32)) & params.compact_mask; \
    bool found = false; \
    for (uint p = 0; p < params.max_probe && !found; p++) { \
        uint cfp = compact_fp[pos]; if (cfp == 0) break; \
        if (cfp == fp) { uint idx = compact_idx[pos]; \
            if (idx < params.hash_data_count) { \
                ulong off = hash_data_off[idx]; \
                device const uint *ref = (device const uint *)(hash_data_buf + off); \
                if (h.x == ref[0] && h.y == ref[1] && h.z == ref[2] && h.w == ref[3]) \
                    found = true; \
            } } \
        pos = (pos + 1) & params.compact_mask; \
    } \
    if (found) { \
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed); \
        if (slot < params.max_hits) { \
            uint base = slot * HIT_STRIDE; \
            hits[base] = widx; hits[base+1] = sidx; hits[base+2] = 1; \
            hits[base+3] = h.x; hits[base+4] = h.y; \
            hits[base+5] = h.z; hits[base+6] = h.w; \
            for (uint _z = 7; _z < HIT_STRIDE; _z++) hits[base+_z] = 0; } } }

/* PROBE7: stride 19, emits iter_num=1 at [2] then 4 hash words at [3..6], zero-pad rest */
#define PROBE7(h, widx, sidx) { \
    ulong key = (ulong(h.y) << 32) | h.x; \
    uint fp = uint(key >> 32); if (fp == 0) fp = 1; \
    ulong pos = (key ^ (key >> 32)) & params.compact_mask; \
    bool found = false; \
    for (uint p = 0; p < params.max_probe && !found; p++) { \
        uint cfp = compact_fp[pos]; if (cfp == 0) break; \
        if (cfp == fp) { uint idx = compact_idx[pos]; \
            if (idx < params.hash_data_count) { \
                ulong off = hash_data_off[idx]; \
                device const uint *ref = (device const uint *)(hash_data_buf + off); \
                if (h.x == ref[0] && h.y == ref[1] && h.z == ref[2] && h.w == ref[3]) \
                    found = true; \
            } } \
        pos = (pos + 1) & params.compact_mask; \
    } \
    if (!found && params.overflow_count > 0) { \
        int lo = 0, hi2 = int(params.overflow_count) - 1; \
        while (lo <= hi2 && !found) { \
            int mid = (lo + hi2) / 2; ulong mkey = overflow_keys[mid]; \
            if (key < mkey) hi2 = mid - 1; \
            else if (key > mkey) lo = mid + 1; \
            else { \
                for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) { \
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]); \
                    if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) \
                        found = true; } \
                for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key && !found; d++) { \
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]); \
                    if (h.x == oref[0] && h.y == oref[1] && h.z == oref[2] && h.w == oref[3]) \
                        found = true; } \
                break; } } } \
    if (found) { \
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed); \
        if (slot < params.max_hits) { \
            uint base = slot * HIT_STRIDE; \
            hits[base] = widx; hits[base+1] = sidx; hits[base+2] = 1; \
            hits[base+3] = h.x; hits[base+4] = h.y; \
            hits[base+5] = h.z; hits[base+6] = h.w; \
            for (uint _z = 7; _z < HIT_STRIDE; _z++) hits[base+_z] = 0; } } }

/* PROBE8: stride 19, for SHA1 (5-word hash in uint array h[]).
 * h[] must be 5-element array of bswapped (LE) hash words.
 * Comparison uses first 4 words; all 5 stored in hit, rest zero-padded. */
#define PROBE8(h, widx, sidx) { \
    ulong key = (ulong(h[1]) << 32) | h[0]; \
    uint fp = h[1]; if (fp == 0) fp = 1; \
    ulong pos = (key ^ (key >> 32)) & params.compact_mask; \
    bool found = false; \
    for (uint p = 0; p < params.max_probe && !found; p++) { \
        uint cfp = compact_fp[pos]; if (cfp == 0) break; \
        if (cfp == fp) { uint idx = compact_idx[pos]; \
            if (idx < params.hash_data_count) { \
                ulong off = hash_data_off[idx]; \
                device const uint *ref = (device const uint *)(hash_data_buf + off); \
                if (h[0] == ref[0] && h[1] == ref[1] && h[2] == ref[2] && h[3] == ref[3]) \
                    found = true; \
            } } \
        pos = (pos + 1) & params.compact_mask; \
    } \
    if (!found && params.overflow_count > 0) { \
        int lo = 0, hi2 = int(params.overflow_count) - 1; \
        while (lo <= hi2 && !found) { \
            int mid = (lo + hi2) / 2; ulong mkey = overflow_keys[mid]; \
            if (key < mkey) hi2 = mid - 1; \
            else if (key > mkey) lo = mid + 1; \
            else { \
                for (int d = mid; d >= 0 && overflow_keys[d] == key && !found; d--) { \
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]); \
                    if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3]) \
                        found = true; } \
                for (int d = mid+1; d < int(params.overflow_count) && overflow_keys[d] == key && !found; d++) { \
                    device const uint *oref = (device const uint *)(overflow_hashes + overflow_offsets[d]); \
                    if (h[0] == oref[0] && h[1] == oref[1] && h[2] == oref[2] && h[3] == oref[3]) \
                        found = true; } \
                break; } } } \
    if (found) { \
        uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed); \
        if (slot < params.max_hits) { \
            uint base = slot * HIT_STRIDE; \
            hits[base] = widx; hits[base+1] = sidx; hits[base+2] = 1; \
            for (int i = 0; i < 5; i++) hits[base+3+i] = h[i]; \
            for (uint _z = 8; _z < HIT_STRIDE; _z++) hits[base+_z] = 0; } } }

/* compact_mix: XOR-fold first 8 hash bytes */
static uint64_t compact_mix(uint64_t k) {
    return k ^ (k >> 32);
}

/* GPU Params: 128-byte uniform API. uint64 first, then uint32, then reserved.
 * Must match host-side MetalParams/OCLParams exactly. */
struct MetalParams {
    uint64_t compact_mask;    /*  0: hash table mask */
    uint64_t mask_start;      /*  8: mask keyspace offset */
    uint64_t mask_base0;      /* 16: pre-decomposed positions 0-7 */
    uint64_t mask_base1;      /* 24: pre-decomposed positions 8-15 */
    uint     num_words;       /* 32: words in batch */
    uint     num_salts;       /* 36: salts for dispatch */
    uint     salt_start;      /* 40: starting salt index */
    uint     max_probe;       /* 44: compact table probe depth */
    uint     hash_data_count; /* 48: hash_data entries */
    uint     max_hits;        /* 52: hit buffer capacity */
    uint     overflow_count;  /* 56: overflow table entries */
    uint     max_iter;        /* 60: iteration count (-i) */
    uint     num_masks;       /* 64: mask combinations per chunk */
    uint     n_prepend;       /* 68: prepend mask positions (-N) */
    uint     n_append;        /* 72: append mask positions (-n) */
    uint     iter_count;      /* 76: per-dispatch iteration (PHPBB3) */
    uint     reserved32[4];   /* 80-95: reserved */
    uint64_t reserved64[4];   /* 96-127: reserved */
};

/* Hex-encode 4 uint32 hash to 32 bytes in M[0..7] for iteration */
static inline void hash_to_hex_M(uint4 h, thread uint *M) {
    /* Copy to local array to ensure addressable byte layout */
    uint hwords[4] = { h.x, h.y, h.z, h.w };
    thread uint8_t *mb = (thread uint8_t *)M;
    thread const uint8_t *hb = (thread const uint8_t *)hwords;
    for (int i = 0; i < 16; i++) {
        uint8_t hi = hb[i] >> 4;
        uint8_t lo = hb[i] & 0xf;
        mb[i*2]   = hi + (hi < 10 ? '0' : 'a' - 10);
        mb[i*2+1] = lo + (lo < 10 ? '0' : 'a' - 10);
    }
}

/* ---- PROBE variants: no-overflow with early return ---- */

/* PROBE7_NOOVF: stride 19, compact-only probe with early return.
 * For HMAC kernels where overflow search is not needed. */
#define PROBE7_NOOVF(h, widx, sidx) { \
    ulong key = (ulong(h.y) << 32) | h.x; \
    uint fp = uint(key >> 32); if (fp == 0) fp = 1; \
    ulong pos = (key ^ (key >> 32)) & params.compact_mask; \
    for (uint p = 0; p < params.max_probe; p++) { \
        uint cfp = compact_fp[pos]; if (cfp == 0) break; \
        if (cfp == fp) { uint idx = compact_idx[pos]; \
            if (idx < params.hash_data_count) { \
                ulong off = hash_data_off[idx]; \
                device const uint *ref = (device const uint *)(hash_data_buf + off); \
                if (h.x == ref[0] && h.y == ref[1] && h.z == ref[2] && h.w == ref[3]) { \
                    uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed); \
                    if (slot < params.max_hits) { uint base = slot * HIT_STRIDE; \
                        hits[base] = widx; hits[base+1] = sidx; hits[base+2] = 1; \
                        hits[base+3] = h.x; hits[base+4] = h.y; \
                        hits[base+5] = h.z; hits[base+6] = h.w; \
                        for (uint _z = 7; _z < HIT_STRIDE; _z++) hits[base+_z] = 0; } return; \
                } } } \
        pos = (pos + 1) & params.compact_mask; } }

/* PROBE11_NOOVF: stride 19, compact-only probe with early return.
 * For SHA256-based HMAC that stores all 8 hash words in the hit. */
#define PROBE11_NOOVF(h0,h1,h2,h3,h_arr,widx,sidx) { \
    uint4 hh = uint4(h0,h1,h2,h3); ulong key = (ulong(hh.y) << 32) | hh.x; \
    uint fp = uint(key >> 32); if (fp == 0) fp = 1; \
    ulong pos = (key ^ (key >> 32)) & params.compact_mask; \
    for (uint p = 0; p < params.max_probe; p++) { uint cfp = compact_fp[pos]; if (cfp == 0) break; \
        if (cfp == fp) { uint idx = compact_idx[pos]; if (idx < params.hash_data_count) { \
            ulong off = hash_data_off[idx]; device const uint *ref = (device const uint *)(hash_data_buf + off); \
            if (hh.x == ref[0] && hh.y == ref[1] && hh.z == ref[2] && hh.w == ref[3]) { \
                uint slot = atomic_fetch_add_explicit(hit_count, 1, memory_order_relaxed); \
                if (slot < params.max_hits) { uint base = slot * HIT_STRIDE; \
                    hits[base] = widx; hits[base+1] = sidx; hits[base+2] = 1; \
                    for (int ii = 0; ii < 8; ii++) hits[base+3+ii] = h_arr[ii]; \
                    for (uint _z = 11; _z < HIT_STRIDE; _z++) hits[base+_z] = 0; } return; \
            } } } pos = (pos + 1) & params.compact_mask; } }

/* ---- SHA256 block function ---- */

constant uint K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static void sha256_block(thread uint *state, thread uint *M) {
    uint W[64];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 64; i++) {
        uint s0 = rotate(W[i-15], 25u) ^ rotate(W[i-15], 14u) ^ (W[i-15] >> 3);
        uint s1 = rotate(W[i-2], 15u) ^ rotate(W[i-2], 13u) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    uint a = state[0], b = state[1], c = state[2], d = state[3];
    uint e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; i++) {
        uint S1 = rotate(e, 26u) ^ rotate(e, 21u) ^ rotate(e, 7u);
        uint ch = (e & f) ^ (~e & g);
        uint t1 = h + S1 + ch + K256[i] + W[i];
        uint S0 = rotate(a, 30u) ^ rotate(a, 19u) ^ rotate(a, 10u);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

/* Convert 8 SHA256 BE state words to 16 BE M[] words of hex text (lowercase) */
static void sha256_to_hex_lc(thread uint *state, thread uint *M) {
    for (int i = 0; i < 8; i++) {
        uint s = state[i];
        uint b0 = (s >> 24) & 0xff, b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff, b3 = s & 0xff;
        M[i*2]   = (hex_byte_be(b0) << 16) | hex_byte_be(b1);
        M[i*2+1] = (hex_byte_be(b2) << 16) | hex_byte_be(b3);
    }
}

/* Convert 7 SHA224 BE state words to 14 BE M[] words of hex text (lowercase) */
static void sha224_to_hex_lc(thread uint *state, thread uint *M) {
    for (int i = 0; i < 7; i++) {
        uint s = state[i];
        uint b0 = (s >> 24) & 0xff, b1 = (s >> 16) & 0xff;
        uint b2 = (s >> 8)  & 0xff, b3 = s & 0xff;
        M[i*2]   = (hex_byte_be(b0) << 16) | hex_byte_be(b1);
        M[i*2+1] = (hex_byte_be(b2) << 16) | hex_byte_be(b3);
    }
}

/* SHA512 functions moved to individual kernel files (metal_sha512unsalted.metal,
 * metal_hmac_sha512.metal) to reduce combined source size for Metal JIT compiler.
 * The JIT's MTLCompilerService XPC process has memory limits that are exceeded
 * when compiling 50KB+ of combined source. */

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

static void rmd160_block(thread uint *hash, thread const uint *X) {
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

/* ---- BLAKE2S compress ---- */

constant uint B2S_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

constant uchar B2S_SIGMA[10][16] = {
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

static void b2s_compress(thread uint *h, thread const uchar *block, ulong counter, int last) {
    uint v[16], m[16];
    for (int i = 0; i < 8; i++) { v[i] = h[i]; v[i+8] = B2S_IV[i]; }
    v[12] ^= (uint)counter;
    v[13] ^= (uint)(counter >> 32);
    if (last) v[14] = ~v[14];
    for (int i = 0; i < 16; i++)
        m[i] = ((uint)block[i*4]) | ((uint)block[i*4+1]<<8) |
               ((uint)block[i*4+2]<<16) | ((uint)block[i*4+3]<<24);
    for (int r = 0; r < 10; r++) {
        constant const uchar *s = B2S_SIGMA[r];
        v[0]+=v[4]+m[s[0]]; v[12]=rotate(v[12]^v[0],16u); v[8]+=v[12]; v[4]=rotate(v[4]^v[8],20u);
        v[0]+=v[4]+m[s[1]]; v[12]=rotate(v[12]^v[0],24u); v[8]+=v[12]; v[4]=rotate(v[4]^v[8],25u);
        v[1]+=v[5]+m[s[2]]; v[13]=rotate(v[13]^v[1],16u); v[9]+=v[13]; v[5]=rotate(v[5]^v[9],20u);
        v[1]+=v[5]+m[s[3]]; v[13]=rotate(v[13]^v[1],24u); v[9]+=v[13]; v[5]=rotate(v[5]^v[9],25u);
        v[2]+=v[6]+m[s[4]]; v[14]=rotate(v[14]^v[2],16u); v[10]+=v[14]; v[6]=rotate(v[6]^v[10],20u);
        v[2]+=v[6]+m[s[5]]; v[14]=rotate(v[14]^v[2],24u); v[10]+=v[14]; v[6]=rotate(v[6]^v[10],25u);
        v[3]+=v[7]+m[s[6]]; v[15]=rotate(v[15]^v[3],16u); v[11]+=v[15]; v[7]=rotate(v[7]^v[11],20u);
        v[3]+=v[7]+m[s[7]]; v[15]=rotate(v[15]^v[3],24u); v[11]+=v[15]; v[7]=rotate(v[7]^v[11],25u);
        v[0]+=v[5]+m[s[8]]; v[15]=rotate(v[15]^v[0],16u); v[10]+=v[15]; v[5]=rotate(v[5]^v[10],20u);
        v[0]+=v[5]+m[s[9]]; v[15]=rotate(v[15]^v[0],24u); v[10]+=v[15]; v[5]=rotate(v[5]^v[10],25u);
        v[1]+=v[6]+m[s[10]]; v[12]=rotate(v[12]^v[1],16u); v[11]+=v[12]; v[6]=rotate(v[6]^v[11],20u);
        v[1]+=v[6]+m[s[11]]; v[12]=rotate(v[12]^v[1],24u); v[11]+=v[12]; v[6]=rotate(v[6]^v[11],25u);
        v[2]+=v[7]+m[s[12]]; v[13]=rotate(v[13]^v[2],16u); v[8]+=v[13]; v[7]=rotate(v[7]^v[8],20u);
        v[2]+=v[7]+m[s[13]]; v[13]=rotate(v[13]^v[2],24u); v[8]+=v[13]; v[7]=rotate(v[7]^v[8],25u);
        v[3]+=v[4]+m[s[14]]; v[14]=rotate(v[14]^v[3],16u); v[9]+=v[14]; v[4]=rotate(v[4]^v[9],20u);
        v[3]+=v[4]+m[s[15]]; v[14]=rotate(v[14]^v[3],24u); v[9]+=v[14]; v[4]=rotate(v[4]^v[9],25u);
    }
    for (int i = 0; i < 8; i++) h[i] ^= v[i] ^ v[i+8];
}

