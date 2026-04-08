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
void md5_block(thread uint4 &state, thread const uint *M) {
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
void md5_block_full(thread uint4 &state, thread const uint *M) {
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
void md5_block_pad64(thread uint4 &state) {
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

void md5_block_from8(thread uint4 &state, thread const uint *M) {
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
void md5_short(thread const uint8_t *msg, int len, thread uint4 &hash) {
    uint M[16] = {0};
    for (int i = 0; i < len; i++)
        ((thread uint8_t *)M)[i] = msg[i];
    ((thread uint8_t *)M)[len] = 0x80;
    M[14] = len * 8;
    hash = uint4(0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476);
    md5_block(hash, M);
}

/* MD5 for messages 56-119 bytes (two blocks) */
void md5_two(thread const uint8_t *msg, int len, thread uint4 &hash) {
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
void hex_encode(thread const uint8_t *bin, thread uint8_t *hex) {
    for (int i = 0; i < 16; i++) {
        uint8_t hi = bin[i] >> 4;
        uint8_t lo = bin[i] & 0x0f;
        hex[i*2]   = hi < 10 ? hi + '0' : hi - 10 + 'a';
        hex[i*2+1] = lo < 10 ? lo + '0' : lo - 10 + 'a';
    }
}

/* compact_mix: XOR-fold first 8 hash bytes */
uint64_t compact_mix(uint64_t k) {
    return k ^ (k >> 32);
}

struct MetalParams {
    uint64_t compact_mask;
    uint     num_words;
    uint     num_salts;
    uint     salt_start;
    uint     max_probe;
    uint     hash_data_count;
    uint     max_hits;
    uint     overflow_count;
    uint     max_iter;
    uint     num_masks;     /* mask combinations per chunk (0 = not mask mode) */
    uint     mask_start;    /* offset for mask chunking */
    uint     n_prepend;     /* number of prepend mask positions */
    uint     n_append;      /* number of append mask positions */
    uint     iter_count;    /* PHPBB3: uniform iteration count for this dispatch group */
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

