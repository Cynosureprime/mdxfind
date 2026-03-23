// streebog.c
// 28-Apr-14    Markku-Juhani O. Saarinen <mjos@iki.fi>
//              See LICENSE for Licensing and Warranty information.

#include "streebog.h"
#include <stdio.h>

// sbob_tab64.c

extern const uint64_t sbob_sl64[8][256];
extern const uint64_t sbob_rc64[12][8];

// LPS transform using 64-bit tables

#define SBOG_LPSti64 \
    (sbob_sl64[0][t.b[i]] ^     sbob_sl64[1][t.b[i + 8]] ^  \
    sbob_sl64[2][t.b[i + 16]] ^ sbob_sl64[3][t.b[i + 24]] ^ \
    sbob_sl64[4][t.b[i + 32]] ^ sbob_sl64[5][t.b[i + 40]] ^ \
    sbob_sl64[6][t.b[i + 48]] ^ sbob_sl64[7][t.b[i + 56]])

// The "g" compression function

static void streebog_g(w512_t *h, const const w512_t *m, uint64_t n)
{
    int i, r;
    w512_t k, s, t;


    // k = LPS(h ^ n)
    memcpy(&t, h, 64);
    for (i = 63; n > 0; i--) {
        t.b[i] ^= n & 0xFF;
        n >>= 8;
    }
    for (i = 0; i < 8; i++)
        k.q[i] = SBOG_LPSti64;

    // s = m
    memcpy(&s, m, 64);

    for (r = 0; r < 12; r++) {
        // s = LPS(s ^ k)
        for (i = 0; i < 8; i++)
            t.q[i] = s.q[i] ^ k.q[i];
        for (i = 0; i < 8; i++)
            s.q[i] = SBOG_LPSti64;

        // k = LPS(k ^ c[i])
        for (i = 0; i < 8; i++)
            t.q[i] = k.q[i] ^ sbob_rc64[r][i];
        for (i = 0; i < 8; i++)
            k.q[i] = SBOG_LPSti64;
    }

    for (i = 0; i < 8; i++)
        h->q[i] ^= s.q[i] ^ k.q[i] ^ m->q[i];
}

// initialzie the context

int streebog_init(streebog_t *sbx, int hlen)
{
    if (hlen != 32 && hlen != 64)
        return 0;

    // IV:: 01010.. for 256-bit hash, 0000... for 512-bit hash
    memset(&sbx->h, hlen == 32 ? 0x01 : 0x00, 64);
    memset(&sbx->e, 0x00, 64);

    sbx->hlen = hlen;
    sbx->pt = 63;
    sbx->n = 0;

    return 1;
}

// add data

int streebog_update(streebog_t *sbx, const void *data, size_t len)
{
    uint64_t i;
    int j, c;

    j = sbx->pt;

    for (i = 0; i < len; i++) {

        sbx->m.b[j--] = ((uint8_t *) data)[i];

        // compress
        if (j < 0) {

            streebog_g(&sbx->h, &sbx->m, sbx->n);
            sbx->n += 0x200;

            // epsilon summation
            c = 0;
            for (j = 63; j >= 0; j--) {
                c += sbx->e.b[j] + sbx->m.b[j];
                sbx->e.b[j] = c & 0xFF;
                c >>= 8;
            }
            j = 63;
        }
    }

    sbx->pt = j;

    return 1;
}

// finalization

int streebog_final(void *hash, streebog_t *sbx)
{
    int i, c;

    // pad the message and run final g
    i = sbx->pt;
    sbx->m.b[i--] = 0x01;
    while (i >= 0)
        sbx->m.b[i--] = 0x00;
    streebog_g(&sbx->h, &sbx->m, sbx->n);

    // epsilon summation
    c = 0;
    for (i = 63; i >= 0; i--) {
        c += sbx->e.b[i] + sbx->m.b[i];
        sbx->e.b[i] = c & 0xFF;
        c >>= 8;
    }

    // finalization n
    memset(&sbx->m, 0x00, 64);

    sbx->n += (63 - sbx->pt) << 3;      // total bits
    for (i = 63; sbx->n > 0; i--) {
        sbx->m.b[i] = sbx->n & 0xFF;
        sbx->n >>= 8;
    }

    streebog_g(&sbx->h, &sbx->m, 0);
    streebog_g(&sbx->h, &sbx->e, 0);

    // copy the result
    memcpy(hash, &sbx->h, sbx->hlen);

    // clear out sensitive stuff
    memset(sbx, 0x00, sizeof(streebog_t));

    return 1;
}

// Streebog (GOST34.11-2012). hlen = 32 (256-bit)) or hlen=64 (512-bit)

void *streebog(void *hash, int hlen, const void *data, size_t len)
{
    streebog_t sbx;

    if (streebog_init(&sbx, hlen) == 0)
        return NULL;
    if (streebog_update(&sbx, data, len) == 0)
        return NULL;
    if (streebog_final(hash, &sbx) == 0)
        return NULL;

    return hash;
}


void gost2012_32(void *data,size_t len,void *hash)
{
     streebog_t sbx;
     streebog_init(&sbx,32);
     streebog_update(&sbx,data,len);
     streebog_final(hash,&sbx);
     return;
}

void gost2012_64(void *data,size_t len, void *hash)
{
     
     streebog_t sbx;
     streebog_init(&sbx,64);
     streebog_update(&sbx,data,len);
     streebog_final(hash,&sbx);
     return;
}

