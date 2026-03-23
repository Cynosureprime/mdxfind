// streebog.h
// 28-Apr-14    Markku-Juhani O. Saarinen <mjos@iki.fi>
//              See LICENSE for Licensing and Warranty information.

#ifndef STREEBOG_H
#define STREEBOG_H

// just need the w512_t type and sbob_pi() prototypes from here
#include "stribob.h"

// context
typedef struct {
    w512_t h, m, e;
    int pt, hlen;
    uint64_t n;
} streebog_t;

// openssl-like api; return 1 on success and 0 on failure
// note that hash len hlen = 32 for 256-bit hash and 64 for 512-bit hash
int streebog_init(streebog_t *sbx, int hlen);
int streebog_update(streebog_t *sbx, const void *data, size_t len);
int streebog_final(void *hash, streebog_t *sbx);

// all-in-one version
void *streebog(void *hash, int hlen, const void *data, size_t len);

#endif
