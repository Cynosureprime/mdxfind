/*
 * hx_func.c - hash function registry for hx
 *
 * Two kinds of functions:
 * 1. Native hx functions (string transforms, encoding)
 * 2. Bridge functions that wrap hashpipe's compute_* via hx_bridge_hash()
 *
 * When compiled standalone (HX_STANDALONE), includes basic hash functions
 * via OpenSSL.  When linked into hashpipe, hash functions are registered
 * dynamically from hashpipe's type registry.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <openssl/des.h>
#include <openssl/aes.h>
#include <openssl/sha.h>
#include <openssl/md4.h>
#include <openssl/md5.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include "hx_vm.h"

/* ---- shared helpers ---- */

static int val_to_int(hx_val *v)
{
	if (v->is_int) return (int)v->ival;
	if (v->data) return atoi(v->data);
	return 0;
}

/* ---- hex encoding helper ---- */

static const char hex_lc[] = "0123456789abcdef";

static void to_hex(const unsigned char *bin, int binlen, char *out)
{
	int i;
	for (i = 0; i < binlen; i++) {
		out[i * 2]     = hex_lc[bin[i] >> 4];
		out[i * 2 + 1] = hex_lc[bin[i] & 0x0f];
	}
	out[binlen * 2] = '\0';
}

/* ================================================================
 * Base64 helper (RFC 4648 with `=` padding, used by bridge and KDFs)
 * ================================================================ */

static const char b64_std_alpha[] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/*
 * Encode binlen raw bytes from `bin` into `out` as RFC 4648 base64
 * with `=` padding.  Returns the number of bytes written (excluding
 * the terminating NUL).  `out` must have room for at least
 * ((binlen + 2) / 3) * 4 + 1 bytes.
 */
static int to_b64(const unsigned char *bin, int binlen, char *out)
{
	int i, o = 0;
	for (i = 0; i < binlen; i += 3) {
		unsigned int a = bin[i];
		unsigned int b = (i + 1 < binlen) ? bin[i + 1] : 0;
		unsigned int c = (i + 2 < binlen) ? bin[i + 2] : 0;
		unsigned int triple = (a << 16) | (b << 8) | c;
		out[o++] = b64_std_alpha[(triple >> 18) & 0x3f];
		out[o++] = b64_std_alpha[(triple >> 12) & 0x3f];
		out[o++] = (i + 1 < binlen) ? b64_std_alpha[(triple >> 6) & 0x3f] : '=';
		out[o++] = (i + 2 < binlen) ? b64_std_alpha[triple & 0x3f] : '=';
	}
	out[o] = '\0';
	return o;
}

/* ================================================================
 * Bridge wrapper: adapts hashpipe's compute_*(pass,passlen,salt,
 * saltlen,dest) to hx's function signature.  The hashfn_t and
 * digest size are stored in the func_entry's bridge fields.
 *
 * Supported roles for bridged types: ROLE_BIN, ROLE_HEX, ROLE_B64.
 * ROLE_DEFAULT canonicalizes to ROLE_HEX (registered explicitly via
 * hx_register_hashpipe_types in hashpipe.c).  ROLE_MCF is rejected
 * at compile time via supported_roles.
 * ================================================================ */

void hx_bridge_hash(hx_func_entry *self, hx_val *args, int nargs,
                    hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[128];
	const unsigned char *salt = NULL;
	int saltlen = 0;
	int n = self->bridge_bytes;

	(void)nargs;

	/* If 2 args, second is salt */
	if (nargs >= 2) {
		salt = (const unsigned char *)args[1].data;
		saltlen = args[1].len;
	}

	self->bridge((const unsigned char *)args[0].data, args[0].len,
	             salt, saltlen, digest);

	/* ROLE_DEFAULT for digests dispatches to ROLE_HEX */
	if (role == ROLE_DEFAULT) role = self->default_role;
	if (role == ROLE_DEFAULT) role = ROLE_HEX;  /* belt-and-braces */

	if (role == ROLE_BIN) {
		result->data = hx_arena_alloc(arena, n);
		memcpy(result->data, digest, n);
		result->len = n;
	} else if (role == ROLE_B64) {
		int max = ((n + 2) / 3) * 4 + 1;
		result->data = hx_arena_alloc(arena, max);
		result->len = to_b64(digest, n, result->data);
	} else {
		/* ROLE_HEX (default) */
		result->data = hx_arena_alloc(arena, n * 2 + 1);
		to_hex(digest, n, result->data);
		result->len = n * 2;
	}
}

/*
 * Shared helper: emit a fixed-size digest in the requested role.
 * ROLE_DEFAULT canonicalizes to ROLE_HEX for digests/HMAC/KDF outputs.
 */
static void emit_digest(unsigned char *digest, int dlen, hx_val *result,
                        hx_arena *arena, uint8_t role)
{
	if (role == ROLE_DEFAULT) role = ROLE_HEX;

	if (role == ROLE_BIN) {
		result->data = hx_arena_alloc(arena, dlen);
		memcpy(result->data, digest, dlen);
		result->len = dlen;
	} else if (role == ROLE_B64) {
		int max = ((dlen + 2) / 3) * 4 + 1;
		result->data = hx_arena_alloc(arena, max);
		result->len = to_b64(digest, dlen, result->data);
	} else {
		/* ROLE_HEX */
		result->data = hx_arena_alloc(arena, dlen * 2 + 1);
		to_hex(digest, dlen, result->data);
		result->len = dlen * 2;
	}
}

/* ================================================================
 * Standalone hash functions (used when HX_STANDALONE is defined)
 * ================================================================ */

#ifdef HX_STANDALONE
#include <openssl/md5.h>
#include <openssl/md4.h>
#include <openssl/sha.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>

static void fn_md5(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[16];
	(void)self; (void)nargs;
	MD5((unsigned char *)args[0].data, args[0].len, digest);
	emit_digest(digest, 16, result, arena, role);
}

static void fn_md4(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[16];
	(void)self; (void)nargs;
	MD4((unsigned char *)args[0].data, args[0].len, digest);
	emit_digest(digest, 16, result, arena, role);
}

static void fn_sha1(hx_func_entry *self, hx_val *args, int nargs,
                    hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[20];
	(void)self; (void)nargs;
	SHA1((unsigned char *)args[0].data, args[0].len, digest);
	emit_digest(digest, 20, result, arena, role);
}

static void fn_sha256(hx_func_entry *self, hx_val *args, int nargs,
                      hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[32];
	(void)self; (void)nargs;
	SHA256((unsigned char *)args[0].data, args[0].len, digest);
	emit_digest(digest, 32, result, arena, role);
}

static void fn_sha512(hx_func_entry *self, hx_val *args, int nargs,
                      hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[64];
	(void)self; (void)nargs;
	SHA512((unsigned char *)args[0].data, args[0].len, digest);
	emit_digest(digest, 64, result, arena, role);
}

static void fn_hmac_sha1(hx_func_entry *self, hx_val *args, int nargs,
                         hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[20];
	unsigned int dlen = 20;
	(void)self; (void)nargs;
	HMAC(EVP_sha1(), args[0].data, args[0].len,
	     (unsigned char *)args[1].data, args[1].len, digest, &dlen);
	emit_digest(digest, 20, result, arena, role);
}

static void fn_hmac_sha256(hx_func_entry *self, hx_val *args, int nargs,
                           hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[32];
	unsigned int dlen = 32;
	(void)self; (void)nargs;
	HMAC(EVP_sha256(), args[0].data, args[0].len,
	     (unsigned char *)args[1].data, args[1].len, digest, &dlen);
	emit_digest(digest, 32, result, arena, role);
}
#endif /* HX_STANDALONE */

/* ================================================================
 * PBKDF2 (available with OpenSSL — both standalone and hashpipe)
 * ================================================================ */

#include <openssl/evp.h>

/*
 * pbkdf2_sha1(pass, salt, iterations, dklen)
 * pbkdf2_sha256(pass, salt, iterations, dklen)
 * pbkdf2_sha512(pass, salt, iterations, dklen)
 * pbkdf2_md5(pass, salt, iterations, dklen)
 *
 * Returns raw key bytes (or hex if no _bin suffix).
 * iterations and dklen are integer arguments.
 */
static void fn_pbkdf2(hx_func_entry *self, hx_val *args, int nargs,
                      hx_val *result, hx_arena *arena, uint8_t role,
                      const EVP_MD *md)
{
	int iterations = 1000, dklen = 32;
	unsigned char *dk;

	if (nargs >= 3) iterations = val_to_int(&args[2]);
	if (nargs >= 4) dklen = val_to_int(&args[3]);
	if (iterations < 1) iterations = 1;
	if (dklen < 1) dklen = 1;
	if (dklen > 1024) dklen = 1024;

	dk = (unsigned char *)hx_arena_alloc(arena, dklen);
	PKCS5_PBKDF2_HMAC(args[0].data, args[0].len,
	                   (unsigned char *)args[1].data, args[1].len,
	                   iterations, md, dklen, dk);
	emit_digest(dk, dklen, result, arena, role);
	(void)self;
}

static void fn_pbkdf2_sha1(hx_func_entry *self, hx_val *args, int nargs,
                            hx_val *result, hx_arena *arena, uint8_t role)
{ fn_pbkdf2(self, args, nargs, result, arena, role, EVP_sha1()); }

static void fn_pbkdf2_sha256(hx_func_entry *self, hx_val *args, int nargs,
                              hx_val *result, hx_arena *arena, uint8_t role)
{ fn_pbkdf2(self, args, nargs, result, arena, role, EVP_sha256()); }

static void fn_pbkdf2_sha512(hx_func_entry *self, hx_val *args, int nargs,
                              hx_val *result, hx_arena *arena, uint8_t role)
{ fn_pbkdf2(self, args, nargs, result, arena, role, EVP_sha512()); }

static void fn_pbkdf2_md5(hx_func_entry *self, hx_val *args, int nargs,
                           hx_val *result, hx_arena *arena, uint8_t role)
{ fn_pbkdf2(self, args, nargs, result, arena, role, EVP_md5()); }

/* ================================================================
 * KDF built-ins (hashpipe-only — require bcrypt, yescrypt, argon2)
 * ================================================================ */

#ifndef HX_STANDALONE

#include "yescrypt/yescrypt.h"
#include "argon2/argon2.h"

extern char *crypt_rn(const char *key, const char *setting,
                      void *output, int size);
extern int crypto_scrypt(const uint8_t *passwd, size_t passwdlen,
                         const uint8_t *salt, size_t saltlen,
                         uint64_t N, uint32_t r, uint32_t p,
                         uint8_t *buf, size_t buflen);

/*
 * bcrypt-base64 inverse-decode: turn a 31-char bcrypt-base64 string
 * into 23 raw bytes.  bcrypt's output format encodes only 23 of the
 * 24 internal Blowfish bytes (the final byte is bug-compatibly
 * truncated; see crypt_blowfish.c, "only encode 23 of the 24 bytes").
 *
 * Returns 0 on success, -1 if a non-alphabet byte is encountered.
 *
 * Alphabet (matches `bcrypt_b64[]` defined later in this file):
 *   ./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
 */
static int bcrypt_b64_decode(const char *src, int srclen,
                             unsigned char *dst, int *dstlen)
{
	static signed char d[256];
	static int d_init = 0;
	int i, j;
	int outpos = 0;

	if (!d_init) {
		static const char alpha[] =
			"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
		memset(d, -1, sizeof(d));
		for (j = 0; j < 64; j++) d[(unsigned char)alpha[j]] = j;
		d_init = 1;
	}

	for (i = 0; i + 1 < srclen; ) {
		int c1 = d[(unsigned char)src[i++]];
		int c2 = (i < srclen) ? d[(unsigned char)src[i++]] : 0;
		int c3 = (i < srclen) ? d[(unsigned char)src[i++]] : -1;
		int c4 = (i < srclen) ? d[(unsigned char)src[i++]] : -1;
		if (c1 < 0 || c2 < 0) return -1;
		dst[outpos++] = (unsigned char)((c1 << 2) | ((c2 & 0x30) >> 4));
		if (c3 < 0) break;
		dst[outpos++] = (unsigned char)(((c2 & 0x0f) << 4) | ((c3 & 0x3c) >> 2));
		if (c4 < 0) break;
		dst[outpos++] = (unsigned char)(((c3 & 0x03) << 6) | c4);
	}
	*dstlen = outpos;
	return 0;
}

/*
 * bcrypt(pass, salt, cost)
 *
 * pass:  password string
 * salt:  16 bytes of raw salt (or 22-char bcrypt-base64 salt)
 * cost:  integer cost factor (4-31), default 12
 *
 * Output roles (per spec hx.1 §2.4 rev 1.5):
 *   ROLE_DEFAULT == ROLE_MCF — full $2b$CC$<22-salt><31-hash> string (60 chars)
 *   ROLE_BIN — 23 raw Blowfish output bytes (decoded from the 31-char hash)
 *              Note: bcrypt's wire format is bug-compatibly 23 bytes, NOT 24
 *              (the 24th byte is truncated by crypt_blowfish.c per the
 *              original OpenBSD implementation).  This is what mdxfind sees
 *              and what hashcat expects.
 *   ROLE_HEX — 46-char lowercase hex of the 23 raw bytes
 *   ROLE_B64 — 31-char bcrypt-base64 hash portion (the canonical "hash"
 *              field within an MCF entry — useful for splitting MCF lines)
 */
static void fn_bcrypt(hx_func_entry *self, hx_val *args, int nargs,
                      hx_val *result, hx_arena *arena, uint8_t role)
{
	char setting[64], output[128], *pass_z;
	int cost = 12;
	(void)self;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	if (nargs >= 3) cost = val_to_int(&args[2]);
	if (cost < 4) cost = 4;
	if (cost > 31) cost = 31;

	/* Build the setting string: $2b$CC$<22-char bcrypt-base64 salt> */
	if (args[1].len >= 22 && (args[1].data[0] == '.' ||
	    (args[1].data[0] >= 'A' && args[1].data[0] <= 'Z') ||
	    (args[1].data[0] >= 'a' && args[1].data[0] <= 'z') ||
	    (args[1].data[0] >= '0' && args[1].data[0] <= '9'))) {
		/* salt is already bcrypt-base64 encoded */
		snprintf(setting, sizeof(setting), "$2b$%02d$%.22s", cost, args[1].data);
	} else {
		/* raw salt: bcrypt-base64 encode the first 16 bytes */
		char b64salt[25];
		const unsigned char *sb = (const unsigned char *)args[1].data;
		int slen = args[1].len > 16 ? 16 : args[1].len;
		int i, o = 0;
		static const char bc[] =
			"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
		/* pad to 16 bytes */
		unsigned char sbuf[16];
		memset(sbuf, 0, 16);
		memcpy(sbuf, sb, slen);
		for (i = 0; i < 16; i += 3) {
			unsigned int v = (sbuf[i] << 16) |
			                 (sbuf[i+1] << 8) | sbuf[i+2];
			b64salt[o++] = bc[(v >> 18) & 0x3f];
			b64salt[o++] = bc[(v >> 12) & 0x3f];
			b64salt[o++] = bc[(v >> 6) & 0x3f];
			b64salt[o++] = bc[v & 0x3f];
		}
		/* bcrypt uses exactly 22 chars of base64 for 16 bytes */
		b64salt[22] = '\0';
		snprintf(setting, sizeof(setting), "$2b$%02d$%s", cost, b64salt);
	}

	/* null-terminate password for crypt_rn */
	pass_z = hx_arena_alloc(arena, args[0].len + 1);
	memcpy(pass_z, args[0].data, args[0].len);
	pass_z[args[0].len] = '\0';

	if (!crypt_rn(pass_z, setting, output, sizeof(output))) {
		/* bcrypt failed — return empty */
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}

	/*
	 * output now looks like "$2b$CC$<22-char salt><31-char hash>" (60 chars).
	 * The 31-char hash starts at output + 7 + 22 = output + 29.
	 */
	{
		char *hash_b64 = output + 7 + 22;
		int hlen = (int)strlen(hash_b64);

		if (role == ROLE_MCF) {
			int olen = (int)strlen(output);
			result->data = hx_arena_alloc(arena, olen + 1);
			memcpy(result->data, output, olen + 1);
			result->len = olen;
		} else if (role == ROLE_B64) {
			/* The bcrypt-base64 hash portion (canonical bcrypt hash field).
			 * Note: this is bcrypt-alphabet, NOT RFC 4648.  Documented as
			 * "the hash portion of the MCF" in the spec. */
			result->data = hx_arena_alloc(arena, hlen + 1);
			memcpy(result->data, hash_b64, hlen + 1);
			result->len = hlen;
		} else {
			/* ROLE_BIN or ROLE_HEX — decode the 31-char hash to raw bytes */
			unsigned char raw[24];
			int raw_len = 0;
			if (bcrypt_b64_decode(hash_b64, hlen, raw, &raw_len) != 0
			    || raw_len <= 0) {
				/* decode failed — return empty */
				result->data = hx_arena_alloc(arena, 1);
				result->data[0] = '\0';
				result->len = 0;
				return;
			}
			/* raw_len is 23 for a 31-char input (bcrypt's bug-compatible
			 * encoding only round-trips 23 bytes). */
			if (role == ROLE_BIN) {
				result->data = hx_arena_alloc(arena, raw_len);
				memcpy(result->data, raw, raw_len);
				result->len = raw_len;
			} else {
				/* ROLE_HEX */
				result->data = hx_arena_alloc(arena, raw_len * 2 + 1);
				to_hex(raw, raw_len, result->data);
				result->len = raw_len * 2;
			}
		}
	}
}

/*
 * scrypt(pass, salt, N, r, p, dklen)
 *
 * N:     CPU/memory cost (power of 2), default 16384
 * r:     block size, default 8
 * p:     parallelism, default 1
 * dklen: derived key length, default 32
 */
static void fn_scrypt(hx_func_entry *self, hx_val *args, int nargs,
                      hx_val *result, hx_arena *arena, uint8_t role)
{
	uint64_t N = 16384;
	uint32_t r = 8, p = 1;
	int dklen = 32;
	unsigned char *dk;
	(void)self;

	if (nargs >= 3) N = (uint64_t)val_to_int(&args[2]);
	if (nargs >= 4) r = (uint32_t)val_to_int(&args[3]);
	if (nargs >= 5) p = (uint32_t)val_to_int(&args[4]);
	if (nargs >= 6) dklen = val_to_int(&args[5]);
	if (dklen < 1) dklen = 1;
	if (dklen > 1024) dklen = 1024;

	dk = (unsigned char *)hx_arena_alloc(arena, dklen);

	if (crypto_scrypt((const uint8_t *)args[0].data, args[0].len,
	                   (const uint8_t *)args[1].data, args[1].len,
	                   N, r, p, dk, dklen) != 0) {
		memset(dk, 0, dklen);
	}

	emit_digest(dk, dklen, result, arena, role);
}

/*
 * argon2id(pass, salt, time_cost, memory_cost_kb, parallelism, dklen)
 * argon2i(pass, salt, time_cost, memory_cost_kb, parallelism, dklen)
 * argon2d(pass, salt, time_cost, memory_cost_kb, parallelism, dklen)
 *
 * time_cost:      iterations, default 3
 * memory_cost_kb: memory in KB, default 65536 (64 MB)
 * parallelism:    threads, default 4
 * dklen:          output length, default 32
 */
static void fn_argon2_generic(hx_func_entry *self, hx_val *args, int nargs,
                               hx_val *result, hx_arena *arena, uint8_t role,
                               argon2_type a2type)
{
	int t_cost = 3, m_cost = 65536, parallelism = 4, dklen = 32;
	unsigned char *dk;
	argon2_context ctx;
	(void)self;

	if (nargs >= 3) t_cost      = val_to_int(&args[2]);
	if (nargs >= 4) m_cost      = val_to_int(&args[3]);
	if (nargs >= 5) parallelism = val_to_int(&args[4]);
	if (nargs >= 6) dklen       = val_to_int(&args[5]);
	if (t_cost < 1) t_cost = 1;
	if (m_cost < 8) m_cost = 8;
	if (parallelism < 1) parallelism = 1;
	if (dklen < 1) dklen = 1;
	if (dklen > 1024) dklen = 1024;

	dk = (unsigned char *)hx_arena_alloc(arena, dklen);

	memset(&ctx, 0, sizeof(ctx));
	ctx.out      = dk;
	ctx.outlen   = dklen;
	ctx.pwd      = (uint8_t *)args[0].data;
	ctx.pwdlen   = args[0].len;
	ctx.salt     = (uint8_t *)args[1].data;
	ctx.saltlen  = args[1].len;
	ctx.t_cost   = t_cost;
	ctx.m_cost   = m_cost;
	ctx.lanes    = parallelism;
	ctx.threads  = 1;   /* single-threaded in hx */
	ctx.version  = ARGON2_VERSION_13;

	if (argon2_ctx(&ctx, a2type) != ARGON2_OK)
		memset(dk, 0, dklen);

	emit_digest(dk, dklen, result, arena, role);
}

static void fn_argon2id(hx_func_entry *self, hx_val *args, int nargs,
                        hx_val *result, hx_arena *arena, uint8_t role)
{ fn_argon2_generic(self, args, nargs, result, arena, role, Argon2_id); }

static void fn_argon2i(hx_func_entry *self, hx_val *args, int nargs,
                       hx_val *result, hx_arena *arena, uint8_t role)
{ fn_argon2_generic(self, args, nargs, result, arena, role, Argon2_i); }

static void fn_argon2d(hx_func_entry *self, hx_val *args, int nargs,
                       hx_val *result, hx_arena *arena, uint8_t role)
{ fn_argon2_generic(self, args, nargs, result, arena, role, Argon2_d); }

/*
 * yescrypt-base64 inverse-decode.  Alphabet:
 *   ./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
 *
 * Bit order is little-endian within each 24-bit group: 3 input bytes
 * are stuffed as byte[0] | (byte[1]<<8) | (byte[2]<<16), then 4 chars
 * are emitted low-6 first.  This matches yescrypt's `encode64()` in
 * yescrypt-common.c.
 *
 * Returns 0 on success, -1 on bad character.  *dstlen is set to the
 * number of bytes decoded (no trailing zero padding).
 */
static int yescrypt_b64_decode(const char *src, int srclen,
                               unsigned char *dst, int dstcap,
                               int *dstlen)
{
	static signed char d[256];
	static int d_init = 0;
	int i, j;
	int outpos = 0;

	if (!d_init) {
		static const char alpha[] =
			"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
		memset(d, -1, sizeof(d));
		for (j = 0; j < 64; j++) d[(unsigned char)alpha[j]] = j;
		d_init = 1;
	}

	for (i = 0; i < srclen; ) {
		uint32_t value = 0;
		int bits = 0;
		while (bits < 24 && i < srclen) {
			int v = d[(unsigned char)src[i++]];
			if (v < 0) return -1;
			value |= ((uint32_t)v) << bits;
			bits += 6;
		}
		/* emit floor(bits/8) bytes, little-endian */
		while (bits >= 8 && outpos < dstcap) {
			dst[outpos++] = (unsigned char)(value & 0xff);
			value >>= 8;
			bits -= 8;
		}
	}
	*dstlen = outpos;
	return 0;
}

/*
 * yescrypt(pass, salt)
 *
 * Uses yescrypt's crypt-style interface.
 * The salt argument should be a complete yescrypt setting string
 * (e.g., "$y$j9T$salt$").
 *
 * Output roles (per spec hx.1 §2.4 rev 1.5):
 *   ROLE_DEFAULT == ROLE_MCF — full $y$<flags>$<salt>$<hash> string
 *   ROLE_BIN — raw KDF bytes (decoded from the hash portion of the MCF)
 *   ROLE_HEX — lowercase hex of the raw KDF bytes
 *   ROLE_B64 — RFC 4648 base64 of the raw KDF bytes (NOT the embedded
 *              yescrypt-base64 hash portion — per user direction)
 */
static __thread yescrypt_local_t *hx_yescrypt_local;

static void fn_yescrypt(hx_func_entry *self, hx_val *args, int nargs,
                        hx_val *result, hx_arena *arena, uint8_t role)
{
	uint8_t result_buf[256];
	uint8_t *yresult;
	char *setting;
	(void)self; (void)nargs;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	if (!hx_yescrypt_local) {
		hx_yescrypt_local = calloc(1, sizeof(yescrypt_local_t));
		yescrypt_init_local(hx_yescrypt_local);
	}

	/* null-terminate salt/setting for yescrypt */
	setting = hx_arena_alloc(arena, args[1].len + 1);
	memcpy(setting, args[1].data, args[1].len);
	setting[args[1].len] = '\0';

	yresult = yescrypt_r(NULL, hx_yescrypt_local,
	                     (const uint8_t *)args[0].data, args[0].len,
	                     (const uint8_t *)setting, NULL,
	                     result_buf, sizeof(result_buf));

	if (!yresult) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}

	if (role == ROLE_MCF) {
		/* full $y$... string verbatim */
		int rlen = (int)strlen((char *)yresult);
		result->data = hx_arena_alloc(arena, rlen + 1);
		memcpy(result->data, yresult, rlen + 1);
		result->len = rlen;
		return;
	}

	/*
	 * For BIN/HEX/B64: extract the hash portion (the substring after
	 * the final '$') and decode it from yescrypt-base64 to raw bytes.
	 */
	{
		char *yend = (char *)yresult + strlen((char *)yresult);
		char *hash_start = strrchr((char *)yresult, '$');
		unsigned char raw[64];
		int raw_len = 0;

		if (!hash_start || hash_start + 1 >= yend) {
			result->data = hx_arena_alloc(arena, 1);
			result->data[0] = '\0';
			result->len = 0;
			return;
		}
		hash_start++;  /* step past the '$' */

		if (yescrypt_b64_decode(hash_start, (int)(yend - hash_start),
		                        raw, sizeof(raw), &raw_len) != 0
		    || raw_len <= 0) {
			result->data = hx_arena_alloc(arena, 1);
			result->data[0] = '\0';
			result->len = 0;
			return;
		}

		emit_digest(raw, raw_len, result, arena, role);
	}
}

/* ================================================================
 * Crypt-family functions (md5crypt, apr1, sha256crypt, sha512crypt,
 * descrypt, phpass).  These produce MCF output by default and
 * support _bin/_hex/_b64 for the raw hash portion.
 *
 * Implemented directly using OpenSSL rather than hashpipe's WS-based
 * compute functions, so they work in both standalone and linked modes.
 * ================================================================ */

/* ---- itoa64 alphabet for md5crypt/sha*crypt ---- */
static const char itoa64_md5[] =
	"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

static void crypt_to64(char *s, unsigned int v, int n)
{
	while (--n >= 0) {
		*s++ = itoa64_md5[v & 0x3f];
		v >>= 6;
	}
}

/* ---- md5crypt / apr1 ---- */

static void do_hx_md5crypt(const char *pass, int passlen,
	const char *salt, int saltlen,
	const char *magic, int magiclen,
	char *result)
{
	MD5_CTX ctx, ctx1;
	unsigned char digest[16];
	int i;
	char *p;

	/* alt = MD5(pass + salt + pass) */
	MD5_Init(&ctx1);
	MD5_Update(&ctx1, pass, passlen);
	MD5_Update(&ctx1, salt, saltlen);
	MD5_Update(&ctx1, pass, passlen);
	MD5_Final(digest, &ctx1);

	/* ctx = MD5(pass + magic + salt + alt_bytes + bit_pattern) */
	MD5_Init(&ctx);
	MD5_Update(&ctx, pass, passlen);
	MD5_Update(&ctx, magic, magiclen);
	MD5_Update(&ctx, salt, saltlen);

	for (i = passlen; i > 0; i -= 16)
		MD5_Update(&ctx, digest, (i > 16) ? 16 : i);

	for (i = passlen; i > 0; i >>= 1) {
		if (i & 1)
			MD5_Update(&ctx, "", 1);
		else
			MD5_Update(&ctx, pass, 1);
	}
	MD5_Final(digest, &ctx);

	/* 1000 iterations */
	for (i = 0; i < 1000; i++) {
		MD5_Init(&ctx);
		if (i & 1) MD5_Update(&ctx, pass, passlen);
		else MD5_Update(&ctx, digest, 16);
		if (i % 3) MD5_Update(&ctx, salt, saltlen);
		if (i % 7) MD5_Update(&ctx, pass, passlen);
		if (i & 1) MD5_Update(&ctx, digest, 16);
		else MD5_Update(&ctx, pass, passlen);
		MD5_Final(digest, &ctx);
	}

	/* Build MCF output */
	p = result;
	memcpy(p, magic, magiclen); p += magiclen;
	memcpy(p, salt, saltlen); p += saltlen;
	*p++ = '$';

	crypt_to64(p, (digest[ 0]<<16) | (digest[ 6]<<8) | digest[12], 4); p += 4;
	crypt_to64(p, (digest[ 1]<<16) | (digest[ 7]<<8) | digest[13], 4); p += 4;
	crypt_to64(p, (digest[ 2]<<16) | (digest[ 8]<<8) | digest[14], 4); p += 4;
	crypt_to64(p, (digest[ 3]<<16) | (digest[ 9]<<8) | digest[15], 4); p += 4;
	crypt_to64(p, (digest[ 4]<<16) | (digest[10]<<8) | digest[ 5], 4); p += 4;
	crypt_to64(p,                     digest[11]                  , 2); p += 2;
	*p = '\0';
}

static void fn_md5crypt(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	char output[128];
	char *salt_z;
	int saltlen;
	(void)self; (void)nargs;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	/* salt */
	saltlen = args[1].len;
	if (saltlen > 8) saltlen = 8;
	salt_z = hx_arena_alloc(arena, saltlen + 1);
	memcpy(salt_z, args[1].data, saltlen);
	salt_z[saltlen] = '\0';

	do_hx_md5crypt(args[0].data, args[0].len, salt_z, saltlen,
		"$1$", 3, output);

	if (role == ROLE_MCF) {
		int olen = (int)strlen(output);
		result->data = hx_arena_alloc(arena, olen + 1);
		memcpy(result->data, output, olen + 1);
		result->len = olen;
	} else {
		/* Extract the 22-char hash after the last $ and decode */
		char *hash = strrchr(output, '$');
		unsigned char raw[16];
		if (hash) hash++;
		else hash = output;
		/* md5crypt hash is 22 itoa64 chars encoding 16 bytes */
		/* For _hex/_b64/_bin, emit the raw 16-byte digest */
		/* Simpler: just re-derive and emit the digest directly */
		/* We already have the digest in the do_ function but it's local.
		 * For simplicity, emit the MCF hash portion as-is for _b64,
		 * and the raw digest for _bin/_hex by re-running. */
		/* Actually, let's just re-run and capture the digest */
		{
			MD5_CTX ctx, ctx1;
			unsigned char digest[16];
			int i;

			MD5_Init(&ctx1);
			MD5_Update(&ctx1, args[0].data, args[0].len);
			MD5_Update(&ctx1, salt_z, saltlen);
			MD5_Update(&ctx1, args[0].data, args[0].len);
			MD5_Final(digest, &ctx1);

			MD5_Init(&ctx);
			MD5_Update(&ctx, args[0].data, args[0].len);
			MD5_Update(&ctx, "$1$", 3);
			MD5_Update(&ctx, salt_z, saltlen);
			for (i = args[0].len; i > 0; i -= 16)
				MD5_Update(&ctx, digest, (i > 16) ? 16 : i);
			for (i = args[0].len; i > 0; i >>= 1) {
				if (i & 1) MD5_Update(&ctx, "", 1);
				else MD5_Update(&ctx, args[0].data, 1);
			}
			MD5_Final(digest, &ctx);

			for (i = 0; i < 1000; i++) {
				MD5_Init(&ctx);
				if (i & 1) MD5_Update(&ctx, args[0].data, args[0].len);
				else MD5_Update(&ctx, digest, 16);
				if (i % 3) MD5_Update(&ctx, salt_z, saltlen);
				if (i % 7) MD5_Update(&ctx, args[0].data, args[0].len);
				if (i & 1) MD5_Update(&ctx, digest, 16);
				else MD5_Update(&ctx, args[0].data, args[0].len);
				MD5_Final(digest, &ctx);
			}
			emit_digest(digest, 16, result, arena, role);
		}
	}
}

static void fn_apr1(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	char output[128];
	char *salt_z;
	int saltlen;
	(void)self; (void)nargs;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	saltlen = args[1].len;
	if (saltlen > 8) saltlen = 8;
	salt_z = hx_arena_alloc(arena, saltlen + 1);
	memcpy(salt_z, args[1].data, saltlen);
	salt_z[saltlen] = '\0';

	do_hx_md5crypt(args[0].data, args[0].len, salt_z, saltlen,
		"$apr1$", 6, output);

	if (role == ROLE_MCF) {
		int olen = (int)strlen(output);
		result->data = hx_arena_alloc(arena, olen + 1);
		memcpy(result->data, output, olen + 1);
		result->len = olen;
	} else {
		/* Same as md5crypt but with apr1 magic — raw digest is identical algo */
		MD5_CTX ctx, ctx1;
		unsigned char digest[16];
		int i;

		MD5_Init(&ctx1);
		MD5_Update(&ctx1, args[0].data, args[0].len);
		MD5_Update(&ctx1, salt_z, saltlen);
		MD5_Update(&ctx1, args[0].data, args[0].len);
		MD5_Final(digest, &ctx1);

		MD5_Init(&ctx);
		MD5_Update(&ctx, args[0].data, args[0].len);
		MD5_Update(&ctx, "$apr1$", 6);
		MD5_Update(&ctx, salt_z, saltlen);
		for (i = args[0].len; i > 0; i -= 16)
			MD5_Update(&ctx, digest, (i > 16) ? 16 : i);
		for (i = args[0].len; i > 0; i >>= 1) {
			if (i & 1) MD5_Update(&ctx, "", 1);
			else MD5_Update(&ctx, args[0].data, 1);
		}
		MD5_Final(digest, &ctx);

		for (i = 0; i < 1000; i++) {
			MD5_Init(&ctx);
			if (i & 1) MD5_Update(&ctx, args[0].data, args[0].len);
			else MD5_Update(&ctx, digest, 16);
			if (i % 3) MD5_Update(&ctx, salt_z, saltlen);
			if (i % 7) MD5_Update(&ctx, args[0].data, args[0].len);
			if (i & 1) MD5_Update(&ctx, digest, 16);
			else MD5_Update(&ctx, args[0].data, args[0].len);
			MD5_Final(digest, &ctx);
		}
		emit_digest(digest, 16, result, arena, role);
	}
}

/* ---- sha256crypt ($5$) ---- */

/* Drepper's SHA-256-crypt implemented directly with OpenSSL SHA256 */

static void fn_sha256crypt(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	SHA256_CTX ctx, alt_ctx;
	unsigned char digest[32], alt_digest[32], temp[32];
	char *pass_z, *salt_z;
	int passlen, saltlen, rounds = 5000;
	int i;
	static const char b64t[] =
		"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	(void)self;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	passlen = args[0].len;
	pass_z = args[0].data;

	saltlen = args[1].len;
	if (saltlen > 16) saltlen = 16;
	salt_z = hx_arena_alloc(arena, saltlen + 1);
	memcpy(salt_z, args[1].data, saltlen);
	salt_z[saltlen] = '\0';

	if (nargs >= 3) rounds = val_to_int(&args[2]);
	if (rounds < 1000) rounds = 1000;
	if (rounds > 999999999) rounds = 999999999;

	/* Step 1-3: alt_digest = SHA256(pass + salt + pass) */
	SHA256_Init(&alt_ctx);
	SHA256_Update(&alt_ctx, pass_z, passlen);
	SHA256_Update(&alt_ctx, salt_z, saltlen);
	SHA256_Update(&alt_ctx, pass_z, passlen);
	SHA256_Final(alt_digest, &alt_ctx);

	/* Step 4-8 */
	SHA256_Init(&ctx);
	SHA256_Update(&ctx, pass_z, passlen);
	SHA256_Update(&ctx, salt_z, saltlen);
	for (i = passlen; i > 32; i -= 32)
		SHA256_Update(&ctx, alt_digest, 32);
	SHA256_Update(&ctx, alt_digest, i);

	for (i = passlen; i > 0; i >>= 1) {
		if (i & 1)
			SHA256_Update(&ctx, alt_digest, 32);
		else
			SHA256_Update(&ctx, pass_z, passlen);
	}
	SHA256_Final(digest, &ctx);

	/* P_bytes */
	SHA256_Init(&alt_ctx);
	for (i = 0; i < passlen; i++)
		SHA256_Update(&alt_ctx, pass_z, passlen);
	SHA256_Final(temp, &alt_ctx);

	{
		unsigned char *p_bytes = hx_arena_alloc(arena, passlen);
		for (i = 0; i + 32 <= passlen; i += 32)
			memcpy(p_bytes + i, temp, 32);
		if (i < passlen)
			memcpy(p_bytes + i, temp, passlen - i);

		/* S_bytes */
		SHA256_Init(&alt_ctx);
		for (i = 0; i < 16 + (unsigned char)digest[0]; i++)
			SHA256_Update(&alt_ctx, salt_z, saltlen);
		SHA256_Final(temp, &alt_ctx);

		unsigned char *s_bytes = hx_arena_alloc(arena, saltlen);
		for (i = 0; i + 32 <= saltlen; i += 32)
			memcpy(s_bytes + i, temp, 32);
		if (i < saltlen)
			memcpy(s_bytes + i, temp, saltlen - i);

		/* Rounds */
		for (i = 0; i < rounds; i++) {
			SHA256_Init(&ctx);
			if (i & 1) SHA256_Update(&ctx, p_bytes, passlen);
			else SHA256_Update(&ctx, digest, 32);
			if (i % 3) SHA256_Update(&ctx, s_bytes, saltlen);
			if (i % 7) SHA256_Update(&ctx, p_bytes, passlen);
			if (i & 1) SHA256_Update(&ctx, digest, 32);
			else SHA256_Update(&ctx, p_bytes, passlen);
			SHA256_Final(digest, &ctx);
		}
	}

	if (role != ROLE_MCF) {
		emit_digest(digest, 32, result, arena, role);
		return;
	}

	/* Build MCF */
	{
		char mcf[256], *p = mcf;
		if (rounds == 5000)
			p += snprintf(p, sizeof(mcf), "$5$%s$", salt_z);
		else
			p += snprintf(p, sizeof(mcf), "$5$rounds=%d$%s$", rounds, salt_z);

#define B64_FROM_24BIT(b2, b1, b0, n) \
	{ unsigned int w = ((unsigned)(b2) << 16) | ((unsigned)(b1) << 8) | (b0); \
	  int nn = (n); while (nn-- > 0) { *p++ = b64t[w & 0x3f]; w >>= 6; } }

		B64_FROM_24BIT(digest[ 0], digest[10], digest[20], 4);
		B64_FROM_24BIT(digest[21], digest[ 1], digest[11], 4);
		B64_FROM_24BIT(digest[12], digest[22], digest[ 2], 4);
		B64_FROM_24BIT(digest[ 3], digest[13], digest[23], 4);
		B64_FROM_24BIT(digest[24], digest[ 4], digest[14], 4);
		B64_FROM_24BIT(digest[15], digest[25], digest[ 5], 4);
		B64_FROM_24BIT(digest[ 6], digest[16], digest[26], 4);
		B64_FROM_24BIT(digest[27], digest[ 7], digest[17], 4);
		B64_FROM_24BIT(digest[18], digest[28], digest[ 8], 4);
		B64_FROM_24BIT(digest[ 9], digest[19], digest[29], 4);
		B64_FROM_24BIT(0,          digest[31], digest[30], 3);
#undef B64_FROM_24BIT
		*p = '\0';

		int mlen = (int)(p - mcf);
		result->data = hx_arena_alloc(arena, mlen + 1);
		memcpy(result->data, mcf, mlen + 1);
		result->len = mlen;
	}
}

/* ---- sha512crypt ($6$) ---- */

/* Drepper's SHA-512-crypt — implement directly using OpenSSL SHA512 */

static void fn_sha512crypt(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	SHA512_CTX ctx, alt_ctx;
	unsigned char digest[64], alt_digest[64], temp[64];
	char *pass_z, *salt_z;
	int passlen, saltlen, rounds = 5000;
	int i;
	static const char b64t[] =
		"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	(void)self;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	passlen = args[0].len;
	pass_z = args[0].data;

	saltlen = args[1].len;
	if (saltlen > 16) saltlen = 16;
	salt_z = hx_arena_alloc(arena, saltlen + 1);
	memcpy(salt_z, args[1].data, saltlen);
	salt_z[saltlen] = '\0';

	if (nargs >= 3) rounds = val_to_int(&args[2]);
	if (rounds < 1000) rounds = 1000;
	if (rounds > 999999999) rounds = 999999999;

	/* Step 1-3: alt_digest = SHA512(pass + salt + pass) */
	SHA512_Init(&alt_ctx);
	SHA512_Update(&alt_ctx, pass_z, passlen);
	SHA512_Update(&alt_ctx, salt_z, saltlen);
	SHA512_Update(&alt_ctx, pass_z, passlen);
	SHA512_Final(alt_digest, &alt_ctx);

	/* Step 4-8: digest = SHA512(pass + salt + alt_bytes + bit_pattern) */
	SHA512_Init(&ctx);
	SHA512_Update(&ctx, pass_z, passlen);
	SHA512_Update(&ctx, salt_z, saltlen);
	for (i = passlen; i > 64; i -= 64)
		SHA512_Update(&ctx, alt_digest, 64);
	SHA512_Update(&ctx, alt_digest, i);

	for (i = passlen; i > 0; i >>= 1) {
		if (i & 1)
			SHA512_Update(&ctx, alt_digest, 64);
		else
			SHA512_Update(&ctx, pass_z, passlen);
	}
	SHA512_Final(digest, &ctx);

	/* Step 9-11: P_bytes = SHA512(pass repeated passlen times) */
	SHA512_Init(&alt_ctx);
	for (i = 0; i < passlen; i++)
		SHA512_Update(&alt_ctx, pass_z, passlen);
	SHA512_Final(temp, &alt_ctx);

	{
		unsigned char *p_bytes = hx_arena_alloc(arena, passlen);
		for (i = 0; i + 64 <= passlen; i += 64)
			memcpy(p_bytes + i, temp, 64);
		if (i < passlen)
			memcpy(p_bytes + i, temp, passlen - i);

		/* Step 12-14: S_bytes = SHA512(salt repeated 16+digest[0] times) */
		SHA512_Init(&alt_ctx);
		for (i = 0; i < 16 + (unsigned char)digest[0]; i++)
			SHA512_Update(&alt_ctx, salt_z, saltlen);
		SHA512_Final(temp, &alt_ctx);

		unsigned char *s_bytes = hx_arena_alloc(arena, saltlen);
		for (i = 0; i + 64 <= saltlen; i += 64)
			memcpy(s_bytes + i, temp, 64);
		if (i < saltlen)
			memcpy(s_bytes + i, temp, saltlen - i);

		/* Step 15-21: rounds iterations */
		for (i = 0; i < rounds; i++) {
			SHA512_Init(&ctx);
			if (i & 1) SHA512_Update(&ctx, p_bytes, passlen);
			else SHA512_Update(&ctx, digest, 64);
			if (i % 3) SHA512_Update(&ctx, s_bytes, saltlen);
			if (i % 7) SHA512_Update(&ctx, p_bytes, passlen);
			if (i & 1) SHA512_Update(&ctx, digest, 64);
			else SHA512_Update(&ctx, p_bytes, passlen);
			SHA512_Final(digest, &ctx);
		}
	}

	if (role != ROLE_MCF) {
		emit_digest(digest, 64, result, arena, role);
		return;
	}

	/* Build MCF: $6$[rounds=N$]salt$hash */
	{
		char mcf[256], *p = mcf;
		if (rounds == 5000)
			p += snprintf(p, sizeof(mcf), "$6$%s$", salt_z);
		else
			p += snprintf(p, sizeof(mcf), "$6$rounds=%d$%s$", rounds, salt_z);

		/* sha512crypt base64 encoding (permuted byte order) */
#define B64_FROM_24BIT(b2, b1, b0, n) \
	{ unsigned int w = ((unsigned)(b2) << 16) | ((unsigned)(b1) << 8) | (b0); \
	  int nn = (n); while (nn-- > 0) { *p++ = b64t[w & 0x3f]; w >>= 6; } }

		B64_FROM_24BIT(digest[ 0], digest[21], digest[42], 4);
		B64_FROM_24BIT(digest[22], digest[43], digest[ 1], 4);
		B64_FROM_24BIT(digest[44], digest[ 2], digest[23], 4);
		B64_FROM_24BIT(digest[ 3], digest[24], digest[45], 4);
		B64_FROM_24BIT(digest[25], digest[46], digest[ 4], 4);
		B64_FROM_24BIT(digest[47], digest[ 5], digest[26], 4);
		B64_FROM_24BIT(digest[ 6], digest[27], digest[48], 4);
		B64_FROM_24BIT(digest[28], digest[49], digest[ 7], 4);
		B64_FROM_24BIT(digest[50], digest[ 8], digest[29], 4);
		B64_FROM_24BIT(digest[ 9], digest[30], digest[51], 4);
		B64_FROM_24BIT(digest[31], digest[52], digest[10], 4);
		B64_FROM_24BIT(digest[53], digest[11], digest[32], 4);
		B64_FROM_24BIT(digest[12], digest[33], digest[54], 4);
		B64_FROM_24BIT(digest[34], digest[55], digest[13], 4);
		B64_FROM_24BIT(digest[56], digest[14], digest[35], 4);
		B64_FROM_24BIT(digest[15], digest[36], digest[57], 4);
		B64_FROM_24BIT(digest[37], digest[58], digest[16], 4);
		B64_FROM_24BIT(digest[59], digest[17], digest[38], 4);
		B64_FROM_24BIT(digest[18], digest[39], digest[60], 4);
		B64_FROM_24BIT(digest[40], digest[61], digest[19], 4);
		B64_FROM_24BIT(digest[62], digest[20], digest[41], 4);
		B64_FROM_24BIT(0,          0,          digest[63], 2);
#undef B64_FROM_24BIT
		*p = '\0';

		int mlen = (int)(p - mcf);
		result->data = hx_arena_alloc(arena, mlen + 1);
		memcpy(result->data, mcf, mlen + 1);
		result->len = mlen;
	}
}

/* ---- descrypt (traditional 13-char DES crypt) ---- */

struct des_info;  /* opaque, from crypt-des.c */
extern char *bsd_crypt_des(const char *key, const char *setting,
	char *output, struct des_info *di);

static void fn_descrypt(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	char output[128], setting[3], *pass_z;
	char passbuf[9];
	struct des_info *di;
	char *res;
	(void)self; (void)nargs;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	/* 2-char salt from first arg */
	setting[0] = (args[1].len > 0) ? args[1].data[0] : '.';
	setting[1] = (args[1].len > 1) ? args[1].data[1] : '.';
	setting[2] = '\0';

	/* DES crypt only uses first 8 bytes of password */
	memset(passbuf, 0, sizeof(passbuf));
	memcpy(passbuf, args[0].data,
		args[0].len > 8 ? 8 : args[0].len);

	di = calloc(1, 65536);  /* des_info is large */
	res = bsd_crypt_des(passbuf, setting, output, di);
	free(di);

	if (!res) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}

	if (role == ROLE_MCF) {
		int rlen = (int)strlen(res);
		result->data = hx_arena_alloc(arena, rlen + 1);
		memcpy(result->data, res, rlen + 1);
		result->len = rlen;
	} else {
		/* Hash portion is chars 2..12 (11 chars encoding 8 bytes) */
		char *hash = res + 2;
		int hlen = (int)strlen(hash);
		if (role == ROLE_B64) {
			result->data = hx_arena_alloc(arena, hlen + 1);
			memcpy(result->data, hash, hlen + 1);
			result->len = hlen;
		} else {
			result->data = hx_arena_alloc(arena, hlen * 2 + 1);
			to_hex((unsigned char *)hash, hlen, result->data);
			result->len = hlen * 2;
		}
	}
}

/* ---- phpass ($H$ / $P$) ---- */

static void fn_phpass(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	MD5_CTX ctx;
	unsigned char digest[16];
	char *salt_z, *pass_z;
	int saltlen, count, i;
	int log2_count = 11;  /* default: 2^11 = 2048 iterations */
	static const char itoa64_php[] =
		"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	(void)self;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	if (nargs >= 3) log2_count = val_to_int(&args[2]);
	if (log2_count < 7) log2_count = 7;
	if (log2_count > 30) log2_count = 30;
	count = 1 << log2_count;

	saltlen = args[1].len;
	if (saltlen > 8) saltlen = 8;
	salt_z = hx_arena_alloc(arena, saltlen + 1);
	memcpy(salt_z, args[1].data, saltlen);
	salt_z[saltlen] = '\0';

	pass_z = args[0].data;

	/* Initial: MD5(salt + password) */
	MD5_Init(&ctx);
	MD5_Update(&ctx, salt_z, saltlen);
	MD5_Update(&ctx, pass_z, args[0].len);
	MD5_Final(digest, &ctx);

	/* Iterate: MD5(digest + password) */
	for (i = 0; i < count; i++) {
		MD5_Init(&ctx);
		MD5_Update(&ctx, digest, 16);
		MD5_Update(&ctx, pass_z, args[0].len);
		MD5_Final(digest, &ctx);
	}

	if (role == ROLE_MCF) {
		char mcf[64], *p;
		/* $H$<log2char><8-char salt><22-char hash> */
		p = mcf;
		*p++ = '$'; *p++ = 'H'; *p++ = '$';
		*p++ = itoa64_php[log2_count];
		memcpy(p, salt_z, saltlen); p += saltlen;
		/* Encode 16 bytes as 22 phpass-base64 chars */
		for (i = 0; i < 16; i += 3) {
			unsigned int v = digest[i];
			*p++ = itoa64_php[v & 0x3f];
			v |= (i + 1 < 16) ? ((unsigned)digest[i+1] << 8) : 0;
			*p++ = itoa64_php[(v >> 6) & 0x3f];
			if (i + 1 >= 16) break;
			v |= (i + 2 < 16) ? ((unsigned)digest[i+2] << 16) : 0;
			*p++ = itoa64_php[(v >> 12) & 0x3f];
			if (i + 2 >= 16) break;
			*p++ = itoa64_php[(v >> 18) & 0x3f];
		}
		*p = '\0';
		result->data = hx_arena_alloc(arena, (int)(p - mcf) + 1);
		memcpy(result->data, mcf, (int)(p - mcf) + 1);
		result->len = (int)(p - mcf);
	} else {
		emit_digest(digest, 16, result, arena, role);
	}
}

/* ---- pomelo(pass, salt, t_cost, m_cost) ---- */

extern int pomelo_hash(void *out, size_t outlen, const void *in, size_t inlen,
	const void *salt, size_t saltlen,
	unsigned int t_cost, unsigned int m_cost);

static void fn_pomelo(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[32];
	int t_cost = 2, m_cost = 3;
	(void)self;

	if (role == ROLE_DEFAULT) role = ROLE_HEX;

	if (nargs >= 3) t_cost = val_to_int(&args[2]);
	if (nargs >= 4) m_cost = val_to_int(&args[3]);

	if (pomelo_hash(digest, 32, args[0].data, args[0].len,
	    args[1].data, args[1].len, t_cost, m_cost) != 0) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}

	emit_digest(digest, 32, result, arena, role);
}

/* ---- rc4_hmac_md5(pass, data) — Kerberos etype 23 core ----
 * Computes the RC4-HMAC-MD5 key derivation used in Kerberos etype 23:
 *   NTLM = MD4(UTF16LE(pass))
 *   K1 = HMAC-MD5(NTLM, usage_type_le32)
 *   K3 = HMAC-MD5(K1, checksum_from_data)
 *   plaintext = RC4(K3, encrypted_portion)
 * This is an opaque primitive for the KRB5PA23/KRB5TGS23 types.
 * In hx, returns the NTLM hash (the password-derived key); the
 * full protocol exchange (RC4 encrypt/decrypt) is verification-only.
 * For hx purposes, rc4_hmac_md5(pass) = HMAC-MD5(MD4(UTF16LE(pass)), 0x01000000)
 */
static void fn_rc4_hmac_md5(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char u16[2048], ntlm[16], k1[16];
	unsigned int hmac_len = 16;
	unsigned char usage[4] = {1, 0, 0, 0};
	int i, passlen = args[0].len;
	(void)self; (void)nargs;

	if (role == ROLE_DEFAULT) role = ROLE_HEX;

	/* UTF16LE encode the password (blind zero-extension) */
	if (passlen > 1024) passlen = 1024;
	for (i = 0; i < passlen; i++) {
		u16[i*2] = (unsigned char)args[0].data[i];
		u16[i*2+1] = 0;
	}

	/* NTLM = MD4(UTF16LE(pass)) */
	{ MD4_CTX ctx;
	  MD4_Init(&ctx);
	  MD4_Update(&ctx, u16, passlen * 2);
	  MD4_Final(ntlm, &ctx);
	}

	/* K1 = HMAC-MD5(NTLM, usage=1) */
	HMAC(EVP_md5(), ntlm, 16, usage, 4, k1, &hmac_len);

	emit_digest(k1, 16, result, arena, role);
}

/* ---- aes_cts_hmac_sha1(pass, salt, etype) — Kerberos etype 17/18 core ----
 * Kerberos string-to-key for AES128-CTS-HMAC-SHA1-96 (etype 17) and
 * AES256-CTS-HMAC-SHA1-96 (etype 18):
 *   key = PBKDF2-SHA1(pass, salt, 4096, keylen)
 *   where keylen = 16 (etype 17) or 32 (etype 18)
 *   then DK(key, usage) via AES-CBC (omitted here — just the base key)
 */
static void fn_aes128_cts_hmac_sha1(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char dk[16];
	(void)self; (void)nargs;

	if (role == ROLE_DEFAULT) role = ROLE_HEX;

	PKCS5_PBKDF2_HMAC_SHA1(args[0].data, args[0].len,
		(unsigned char *)args[1].data, args[1].len,
		4096, 16, dk);

	emit_digest(dk, 16, result, arena, role);
}

static void fn_aes256_cts_hmac_sha1(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char dk[32];
	(void)self; (void)nargs;

	if (role == ROLE_DEFAULT) role = ROLE_HEX;

	PKCS5_PBKDF2_HMAC_SHA1(args[0].data, args[0].len,
		(unsigned char *)args[1].data, args[1].len,
		4096, 32, dk);

	emit_digest(dk, 32, result, arena, role);
}

/* ---- sm3crypt ($sm3$) — Drepper-pattern crypt with SM3 (32-byte digest) ---- */

typedef struct { uint32_t h[8]; uint8_t buf[64]; uint64_t total; } hp_sm3_ctx;
extern void hp_sm3_init(hp_sm3_ctx *ctx);
extern void hp_sm3_update(hp_sm3_ctx *ctx, const uint8_t *data, size_t len);
extern void hp_sm3_final(hp_sm3_ctx *ctx, uint8_t *out);

static void fn_sm3crypt(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	hp_sm3_ctx ctx, alt_ctx;
	unsigned char digest[32], alt_digest[32], temp[32];
	char *pass_z, *salt_z;
	int passlen, saltlen, rounds = 5000;
	int i;
	static const char b64t[] =
		"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	(void)self;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	passlen = args[0].len;
	pass_z = args[0].data;
	saltlen = args[1].len;
	if (saltlen > 16) saltlen = 16;
	salt_z = hx_arena_alloc(arena, saltlen + 1);
	memcpy(salt_z, args[1].data, saltlen);
	salt_z[saltlen] = '\0';
	if (nargs >= 3) rounds = val_to_int(&args[2]);
	if (rounds < 1000) rounds = 1000;
	if (rounds > 999999999) rounds = 999999999;

	/* Drepper algorithm (same structure as sha256crypt, SM3 = 32 bytes) */
	hp_sm3_init(&alt_ctx);
	hp_sm3_update(&alt_ctx, (uint8_t *)pass_z, passlen);
	hp_sm3_update(&alt_ctx, (uint8_t *)salt_z, saltlen);
	hp_sm3_update(&alt_ctx, (uint8_t *)pass_z, passlen);
	hp_sm3_final(&alt_ctx, alt_digest);

	hp_sm3_init(&ctx);
	hp_sm3_update(&ctx, (uint8_t *)pass_z, passlen);
	hp_sm3_update(&ctx, (uint8_t *)salt_z, saltlen);
	for (i = passlen; i > 32; i -= 32)
		hp_sm3_update(&ctx, alt_digest, 32);
	hp_sm3_update(&ctx, alt_digest, i);
	for (i = passlen; i > 0; i >>= 1) {
		if (i & 1) hp_sm3_update(&ctx, alt_digest, 32);
		else hp_sm3_update(&ctx, (uint8_t *)pass_z, passlen);
	}
	hp_sm3_final(&ctx, digest);

	/* P_bytes */
	hp_sm3_init(&alt_ctx);
	for (i = 0; i < passlen; i++)
		hp_sm3_update(&alt_ctx, (uint8_t *)pass_z, passlen);
	hp_sm3_final(&alt_ctx, temp);
	{
		unsigned char *p_bytes = hx_arena_alloc(arena, passlen);
		for (i = 0; i + 32 <= passlen; i += 32) memcpy(p_bytes + i, temp, 32);
		if (i < passlen) memcpy(p_bytes + i, temp, passlen - i);

		/* S_bytes */
		hp_sm3_init(&alt_ctx);
		for (i = 0; i < 16 + (unsigned char)digest[0]; i++)
			hp_sm3_update(&alt_ctx, (uint8_t *)salt_z, saltlen);
		hp_sm3_final(&alt_ctx, temp);
		unsigned char *s_bytes = hx_arena_alloc(arena, saltlen);
		for (i = 0; i + 32 <= saltlen; i += 32) memcpy(s_bytes + i, temp, 32);
		if (i < saltlen) memcpy(s_bytes + i, temp, saltlen - i);

		for (i = 0; i < rounds; i++) {
			hp_sm3_init(&ctx);
			if (i & 1) hp_sm3_update(&ctx, p_bytes, passlen);
			else hp_sm3_update(&ctx, digest, 32);
			if (i % 3) hp_sm3_update(&ctx, s_bytes, saltlen);
			if (i % 7) hp_sm3_update(&ctx, p_bytes, passlen);
			if (i & 1) hp_sm3_update(&ctx, digest, 32);
			else hp_sm3_update(&ctx, p_bytes, passlen);
			hp_sm3_final(&ctx, digest);
		}
	}

	if (role != ROLE_MCF) {
		emit_digest(digest, 32, result, arena, role);
		return;
	}
	/* MCF: $sm3$[rounds=N$]salt$hash — same base64 permutation as sha256crypt */
	{
		char mcf[256], *p = mcf;
		if (rounds == 5000)
			p += snprintf(p, sizeof(mcf), "$sm3$%s$", salt_z);
		else
			p += snprintf(p, sizeof(mcf), "$sm3$rounds=%d$%s$", rounds, salt_z);
#define B64_FROM_24BIT(b2, b1, b0, n) \
	{ unsigned int w = ((unsigned)(b2) << 16) | ((unsigned)(b1) << 8) | (b0); \
	  int nn = (n); while (nn-- > 0) { *p++ = b64t[w & 0x3f]; w >>= 6; } }
		B64_FROM_24BIT(digest[ 0], digest[10], digest[20], 4);
		B64_FROM_24BIT(digest[21], digest[ 1], digest[11], 4);
		B64_FROM_24BIT(digest[12], digest[22], digest[ 2], 4);
		B64_FROM_24BIT(digest[ 3], digest[13], digest[23], 4);
		B64_FROM_24BIT(digest[24], digest[ 4], digest[14], 4);
		B64_FROM_24BIT(digest[15], digest[25], digest[ 5], 4);
		B64_FROM_24BIT(digest[ 6], digest[16], digest[26], 4);
		B64_FROM_24BIT(digest[27], digest[ 7], digest[17], 4);
		B64_FROM_24BIT(digest[18], digest[28], digest[ 8], 4);
		B64_FROM_24BIT(digest[ 9], digest[19], digest[29], 4);
		B64_FROM_24BIT(0,          digest[31], digest[30], 3);
#undef B64_FROM_24BIT
		*p = '\0';
		int mlen = (int)(p - mcf);
		result->data = hx_arena_alloc(arena, mlen + 1);
		memcpy(result->data, mcf, mlen + 1);
		result->len = mlen;
	}
}

/* ---- gost12_512crypt ($gost12512hash$) — Drepper-pattern with Streebog-512 ---- */

/* Streebog-512 from gosthash/gost2012.a */
typedef struct streebog_ctx streebog_ctx;
extern void streebog_init(streebog_ctx *ctx, int hashlen);
extern void streebog_update(streebog_ctx *ctx, const unsigned char *data, size_t len);
extern void streebog_final(unsigned char *out, streebog_ctx *ctx);

/* streebog_final outputs bytes in reversed order vs GOST standard;
 * this wrapper reverses to canonical byte order */
static void streebog_final_rev(unsigned char *out, streebog_ctx *ctx)
{
	unsigned char tmp[64];
	int i;
	streebog_final(tmp, ctx);
	for (i = 0; i < 64; i++) out[i] = tmp[63 - i];
}

static void fn_gost12_512crypt(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	/* Streebog context is large — allocate from arena */
	streebog_ctx *ctx = (streebog_ctx *)hx_arena_alloc(arena, 512);
	streebog_ctx *alt_ctx = (streebog_ctx *)hx_arena_alloc(arena, 512);
	unsigned char digest[64], alt_digest[64], temp[64];
	char *pass_z, *salt_z;
	int passlen, saltlen, rounds = 5000;
	int i;
	static const char b64t[] =
		"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	(void)self;

	if (role == ROLE_DEFAULT) role = ROLE_MCF;

	passlen = args[0].len;
	pass_z = args[0].data;
	saltlen = args[1].len;
	if (saltlen > 16) saltlen = 16;
	salt_z = hx_arena_alloc(arena, saltlen + 1);
	memcpy(salt_z, args[1].data, saltlen);
	salt_z[saltlen] = '\0';
	if (nargs >= 3) rounds = val_to_int(&args[2]);
	if (rounds < 1000) rounds = 1000;
	if (rounds > 999999999) rounds = 999999999;

	/* Drepper algorithm with Streebog-512 (64-byte digest) */
	streebog_init(alt_ctx, 64);
	streebog_update(alt_ctx, (unsigned char *)pass_z, passlen);
	streebog_update(alt_ctx, (unsigned char *)salt_z, saltlen);
	streebog_update(alt_ctx, (unsigned char *)pass_z, passlen);
	streebog_final_rev(alt_digest, alt_ctx);

	streebog_init(ctx, 64);
	streebog_update(ctx, (unsigned char *)pass_z, passlen);
	streebog_update(ctx, (unsigned char *)salt_z, saltlen);
	for (i = passlen; i > 64; i -= 64)
		streebog_update(ctx, alt_digest, 64);
	streebog_update(ctx, alt_digest, i);
	for (i = passlen; i > 0; i >>= 1) {
		if (i & 1) streebog_update(ctx, alt_digest, 64);
		else streebog_update(ctx, (unsigned char *)pass_z, passlen);
	}
	streebog_final_rev(digest, ctx);

	/* P_bytes */
	streebog_init(alt_ctx, 64);
	for (i = 0; i < passlen; i++)
		streebog_update(alt_ctx, (unsigned char *)pass_z, passlen);
	streebog_final_rev(temp, alt_ctx);
	{
		unsigned char *p_bytes = hx_arena_alloc(arena, passlen);
		for (i = 0; i + 64 <= passlen; i += 64) memcpy(p_bytes + i, temp, 64);
		if (i < passlen) memcpy(p_bytes + i, temp, passlen - i);

		/* S_bytes */
		streebog_init(alt_ctx, 64);
		for (i = 0; i < 16 + (unsigned char)digest[0]; i++)
			streebog_update(alt_ctx, (unsigned char *)salt_z, saltlen);
		streebog_final_rev(temp, alt_ctx);
		unsigned char *s_bytes = hx_arena_alloc(arena, saltlen);
		for (i = 0; i + 64 <= saltlen; i += 64) memcpy(s_bytes + i, temp, 64);
		if (i < saltlen) memcpy(s_bytes + i, temp, saltlen - i);

		for (i = 0; i < rounds; i++) {
			streebog_init(ctx, 64);
			if (i & 1) streebog_update(ctx, p_bytes, passlen);
			else streebog_update(ctx, digest, 64);
			if (i % 3) streebog_update(ctx, s_bytes, saltlen);
			if (i % 7) streebog_update(ctx, p_bytes, passlen);
			if (i & 1) streebog_update(ctx, digest, 64);
			else streebog_update(ctx, p_bytes, passlen);
			streebog_final_rev(digest, ctx);
		}
	}

	if (role != ROLE_MCF) {
		emit_digest(digest, 64, result, arena, role);
		return;
	}
	/* MCF: $gost12512hash$[rounds=N$]salt$hash — sha512crypt permutation */
	{
		char mcf[512], *p = mcf;
		if (rounds == 5000)
			p += snprintf(p, sizeof(mcf), "$gost12512hash$%s$", salt_z);
		else
			p += snprintf(p, sizeof(mcf), "$gost12512hash$rounds=%d$%s$", rounds, salt_z);
#define B64_FROM_24BIT(b2, b1, b0, n) \
	{ unsigned int w = ((unsigned)(b2) << 16) | ((unsigned)(b1) << 8) | (b0); \
	  int nn = (n); while (nn-- > 0) { *p++ = b64t[w & 0x3f]; w >>= 6; } }
		B64_FROM_24BIT(digest[ 0], digest[21], digest[42], 4);
		B64_FROM_24BIT(digest[22], digest[43], digest[ 1], 4);
		B64_FROM_24BIT(digest[44], digest[ 2], digest[23], 4);
		B64_FROM_24BIT(digest[ 3], digest[24], digest[45], 4);
		B64_FROM_24BIT(digest[25], digest[46], digest[ 4], 4);
		B64_FROM_24BIT(digest[47], digest[ 5], digest[26], 4);
		B64_FROM_24BIT(digest[ 6], digest[27], digest[48], 4);
		B64_FROM_24BIT(digest[28], digest[49], digest[ 7], 4);
		B64_FROM_24BIT(digest[50], digest[ 8], digest[29], 4);
		B64_FROM_24BIT(digest[ 9], digest[30], digest[51], 4);
		B64_FROM_24BIT(digest[31], digest[52], digest[10], 4);
		B64_FROM_24BIT(digest[53], digest[11], digest[32], 4);
		B64_FROM_24BIT(digest[12], digest[33], digest[54], 4);
		B64_FROM_24BIT(digest[34], digest[55], digest[13], 4);
		B64_FROM_24BIT(digest[56], digest[14], digest[35], 4);
		B64_FROM_24BIT(digest[15], digest[36], digest[57], 4);
		B64_FROM_24BIT(digest[37], digest[58], digest[16], 4);
		B64_FROM_24BIT(digest[59], digest[17], digest[38], 4);
		B64_FROM_24BIT(digest[18], digest[39], digest[60], 4);
		B64_FROM_24BIT(digest[40], digest[61], digest[19], 4);
		B64_FROM_24BIT(digest[62], digest[20], digest[41], 4);
		B64_FROM_24BIT(0,          0,          digest[63], 2);
#undef B64_FROM_24BIT
		*p = '\0';
		int mlen = (int)(p - mcf);
		result->data = hx_arena_alloc(arena, mlen + 1);
		memcpy(result->data, mcf, mlen + 1);
		result->len = mlen;
	}
}

#endif /* !HX_STANDALONE */

/* ================================================================
 * siphash(pass, key) — SipHash-2-4 with 16-byte key
 * murmur3(pass, seed) — MurmurHash3_x86_32
 * pbkdf1_sha1(pass, salt, iter, dklen) — PBKDF1 with SHA-1
 * These work in both standalone and linked modes.
 * ================================================================ */

/* ---- SipHash-2-4 ---- */

static uint64_t hx_siphash24(const unsigned char *in, int inlen,
	const unsigned char *key)
{
	uint64_t v0 = 0x736f6d6570736575ULL;
	uint64_t v1 = 0x646f72616e646f6dULL;
	uint64_t v2 = 0x6c7967656e657261ULL;
	uint64_t v3 = 0x7465646279746573ULL;
	uint64_t k0, k1, m;
	int i, blocks;

	memcpy(&k0, key, 8);
	memcpy(&k1, key + 8, 8);
	v3 ^= k1; v2 ^= k0; v1 ^= k1; v0 ^= k0;

	blocks = inlen / 8;
	for (i = 0; i < blocks; i++) {
		memcpy(&m, in + i * 8, 8);
		v3 ^= m;
#define SIPROUND \
		v0 += v1; v1 = (v1 << 13) | (v1 >> 51); v1 ^= v0; \
		v0 = (v0 << 32) | (v0 >> 32); \
		v2 += v3; v3 = (v3 << 16) | (v3 >> 48); v3 ^= v2; \
		v0 += v3; v3 = (v3 << 21) | (v3 >> 43); v3 ^= v0; \
		v2 += v1; v1 = (v1 << 17) | (v1 >> 47); v1 ^= v2; \
		v2 = (v2 << 32) | (v2 >> 32);
		SIPROUND; SIPROUND;
		v0 ^= m;
	}

	m = ((uint64_t)inlen) << 56;
	switch (inlen & 7) {
		case 7: m |= (uint64_t)in[blocks*8+6] << 48; /* fall through */
		case 6: m |= (uint64_t)in[blocks*8+5] << 40; /* fall through */
		case 5: m |= (uint64_t)in[blocks*8+4] << 32; /* fall through */
		case 4: m |= (uint64_t)in[blocks*8+3] << 24; /* fall through */
		case 3: m |= (uint64_t)in[blocks*8+2] << 16; /* fall through */
		case 2: m |= (uint64_t)in[blocks*8+1] << 8;  /* fall through */
		case 1: m |= (uint64_t)in[blocks*8+0];
	}
	v3 ^= m;
	SIPROUND; SIPROUND;
	v0 ^= m;

	v2 ^= 0xff;
	SIPROUND; SIPROUND; SIPROUND; SIPROUND;
#undef SIPROUND
	return v0 ^ v1 ^ v2 ^ v3;
}

static void fn_siphash(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char key[16];
	uint64_t hash;
	unsigned char raw[8];
	(void)self; (void)nargs;

	if (role == ROLE_DEFAULT) role = ROLE_HEX;

	/* Key is arg[1], padded/truncated to 16 bytes */
	memset(key, 0, 16);
	if (args[1].len >= 16) memcpy(key, args[1].data, 16);
	else memcpy(key, args[1].data, args[1].len);

	hash = hx_siphash24((unsigned char *)args[0].data, args[0].len, key);
	memcpy(raw, &hash, 8);

	emit_digest(raw, 8, result, arena, role);
}

/* ---- MurmurHash3_x86_32 ---- */

static void fn_murmur3(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	const unsigned char *data = (unsigned char *)args[0].data;
	int len = args[0].len;
	uint32_t seed = 0;
	const uint32_t c1 = 0xcc9e2d51, c2 = 0x1b873593;
	uint32_t h1;
	int nblocks, i;
	unsigned char raw[4];
	(void)self;

	if (role == ROLE_DEFAULT) role = ROLE_HEX;

	if (nargs >= 2) seed = (uint32_t)val_to_int(&args[1]);
	h1 = seed;
	nblocks = len / 4;

	for (i = 0; i < nblocks; i++) {
		uint32_t k1;
		memcpy(&k1, data + i * 4, 4);
		k1 *= c1; k1 = (k1 << 15) | (k1 >> 17); k1 *= c2;
		h1 ^= k1; h1 = (h1 << 13) | (h1 >> 19); h1 = h1 * 5 + 0xe6546b64;
	}
	{ const uint8_t *tail = data + nblocks * 4;
	  uint32_t k1 = 0;
	  switch (len & 3) {
		case 3: k1 ^= (uint32_t)tail[2] << 16; /* fall through */
		case 2: k1 ^= (uint32_t)tail[1] << 8;  /* fall through */
		case 1: k1 ^= tail[0];
		        k1 *= c1; k1 = (k1 << 15) | (k1 >> 17); k1 *= c2;
		        h1 ^= k1;
	  }
	}
	h1 ^= (uint32_t)len;
	h1 ^= h1 >> 16; h1 *= 0x85ebca6b;
	h1 ^= h1 >> 13; h1 *= 0xc2b2ae35;
	h1 ^= h1 >> 16;

	/* Big-endian output (matches hashpipe convention) */
	raw[0] = (h1 >> 24) & 0xff; raw[1] = (h1 >> 16) & 0xff;
	raw[2] = (h1 >> 8) & 0xff;  raw[3] = h1 & 0xff;

	emit_digest(raw, 4, result, arena, role);
}

/* ---- PBKDF1-SHA1 ---- */

static void fn_pbkdf1_sha1(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	SHA_CTX ctx;
	unsigned char digest[20];
	int iter = 1000, dklen = 20, i;
	(void)self;

	if (role == ROLE_DEFAULT) role = ROLE_HEX;

	if (nargs >= 3) iter = val_to_int(&args[2]);
	if (nargs >= 4) dklen = val_to_int(&args[3]);
	if (iter < 1) iter = 1;
	if (dklen < 1) dklen = 1;
	if (dklen > 20) dklen = 20;  /* PBKDF1 limited to hash output size */

	/* Step 1: SHA1(pass + salt) */
	SHA1_Init(&ctx);
	SHA1_Update(&ctx, args[0].data, args[0].len);
	SHA1_Update(&ctx, args[1].data, args[1].len);
	SHA1_Final(digest, &ctx);

	/* Steps 2..iter: SHA1(digest) */
	for (i = 1; i < iter; i++) {
		SHA1_Init(&ctx);
		SHA1_Update(&ctx, digest, 20);
		SHA1_Final(digest, &ctx);
	}

	emit_digest(digest, dklen, result, arena, role);
}

/* ---- argon2 (alias for argon2id) ---- */

#ifndef HX_STANDALONE
static void fn_argon2(hx_func_entry *self, hx_val *args, int nargs,
	hx_val *result, hx_arena *arena, uint8_t role)
{
	/* Dispatch to argon2id as the default variant */
	extern void fn_argon2id(hx_func_entry *, hx_val *, int,
		hx_val *, hx_arena *, uint8_t);
	fn_argon2id(self, args, nargs, result, arena, role);
}
#endif

/* ================================================================
 * Output function
 * ================================================================ */

/*
 * emit(x) — write value to stdout with a newline.
 * Returns the value unchanged (passthrough).
 * When a program contains emit() calls, the automatic
 * final-value output is suppressed.
 */
static void fn_emit(hx_func_entry *self, hx_val *args, int nargs,
                    hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs; (void)role;

	if (args[0].data && args[0].len > 0) {
		/*
		 * Auto-detect binary data: if any byte is not
		 * printable ASCII (0x20-0x7e), hex-encode the output.
		 * This handles the case where emit() receives the
		 * output of a _bin function.
		 */
		int i, is_binary = 0;
		for (i = 0; i < args[0].len; i++) {
			unsigned char c = (unsigned char)args[0].data[i];
			if (c < 0x20 || c > 0x7e) { is_binary = 1; break; }
		}
		if (is_binary) {
			char *hex = hx_arena_alloc(arena, args[0].len * 2 + 1);
			to_hex((unsigned char *)args[0].data,
			       args[0].len, hex);
			fwrite(hex, 1, args[0].len * 2, stdout);
		} else {
			fwrite(args[0].data, 1, args[0].len, stdout);
		}
	}
	putchar('\n');
	*result = args[0];
}

/* ================================================================
 * String transform functions (always available)
 * ================================================================ */

/*
 * hex(x) — encode raw bytes to lowercase hex.  This is hx's only
 * encoding function whose default IS hex (other functions like upper(),
 * lower() have no canonical encoding).
 *
 * `hex_bin(x)` is defined as the identity function (per spec hx.1
 * §2.4): it returns x unchanged.  This is the inverse of `hex_bin`'s
 * normal "hex of raw bytes" semantics: when applied to already-binary
 * data, returning the input unchanged means the chain `hex(x)` is
 * the canonical encoding step and `hex_bin(x)` is the no-op
 * passthrough that documents intent.
 *
 * supported_roles = ROLE_CAP_BIN, default_role = ROLE_HEX.
 */
static void fn_hex(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs;
	if (role == ROLE_BIN) { *result = args[0]; return; }
	/* ROLE_DEFAULT or ROLE_HEX: encode bytes to hex */
	result->data = hx_arena_alloc(arena, args[0].len * 2 + 1);
	to_hex((unsigned char *)args[0].data, args[0].len, result->data);
	result->len = args[0].len * 2;
}

static void fn_upper(hx_func_entry *self, hx_val *args, int nargs,
                     hx_val *result, hx_arena *arena, uint8_t role)
{
	int i;
	(void)self; (void)nargs; (void)role;
	result->data = hx_arena_alloc(arena, args[0].len + 1);
	for (i = 0; i < args[0].len; i++)
		result->data[i] = toupper((unsigned char)args[0].data[i]);
	result->data[args[0].len] = '\0';
	result->len = args[0].len;
}

static void fn_lower(hx_func_entry *self, hx_val *args, int nargs,
                     hx_val *result, hx_arena *arena, uint8_t role)
{
	int i;
	(void)self; (void)nargs; (void)role;
	result->data = hx_arena_alloc(arena, args[0].len + 1);
	for (i = 0; i < args[0].len; i++)
		result->data[i] = tolower((unsigned char)args[0].data[i]);
	result->data[args[0].len] = '\0';
	result->len = args[0].len;
}

static void fn_cut(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	int inlen = args[0].len;
	int start = 0, len;
	(void)self; (void)role;

	if (nargs >= 2) start = val_to_int(&args[1]);
	if (start < 0) { start = inlen + start; if (start < 0) start = 0; }
	if (start > inlen) start = inlen;
	if (nargs >= 3) { len = val_to_int(&args[2]); if (len < 0) len = 0; }
	else len = inlen - start;
	if (start + len > inlen) len = inlen - start;

	result->data = hx_arena_alloc(arena, len + 1);
	memcpy(result->data, args[0].data + start, len);
	result->data[len] = '\0';
	result->len = len;
}

/*
 * pad(x, length) — null-pad x to exactly length bytes.
 * If x is shorter, the result is x followed by zero bytes.
 * If x is longer, the result is truncated to length.
 * Used for fixed-width fields (e.g., RADMIN2 pads to 100 bytes).
 */
static void fn_pad(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	int padlen;
	(void)self; (void)nargs; (void)role;

	padlen = val_to_int(&args[1]);
	if (padlen < 0) padlen = 0;
	if (padlen > 65536) padlen = 65536;

	result->data = hx_arena_alloc(arena, padlen + 1);
	if (args[0].len >= padlen) {
		memcpy(result->data, args[0].data, padlen);
	} else {
		memcpy(result->data, args[0].data, args[0].len);
		memset(result->data + args[0].len, 0, padlen - args[0].len);
	}
	result->data[padlen] = '\0';
	result->len = padlen;
}

static void fn_trunc(hx_func_entry *self, hx_val *args, int nargs,
                     hx_val *result, hx_arena *arena, uint8_t role)
{
	hx_val cut_args[3];
	(void)nargs;
	cut_args[0] = args[0];
	cut_args[1].is_int = 1; cut_args[1].ival = 0;
	cut_args[2] = args[1];
	fn_cut(self, cut_args, 3, result, arena, role);
}

static void fn_rev(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	int i, len;
	(void)self; (void)nargs; (void)role;
	len = args[0].len;
	result->data = hx_arena_alloc(arena, len + 1);
	for (i = 0; i < len; i++)
		result->data[i] = args[0].data[len - 1 - i];
	result->data[len] = '\0';
	result->len = len;
}

/* ---- bswap32(x) — reverse byte order within each 4-byte group ---- */

static void fn_bswap32(hx_func_entry *self, hx_val *args, int nargs,
                       hx_val *result, hx_arena *arena, uint8_t role)
{
	int i, len;
	unsigned char *out;
	const unsigned char *in;
	(void)self; (void)nargs; (void)role;

	len = args[0].len;
	out = (unsigned char *)hx_arena_alloc(arena, len + 1);
	in = (const unsigned char *)args[0].data;

	/* Swap bytes within each 4-byte group */
	for (i = 0; i + 3 < len; i += 4) {
		out[i]     = in[i + 3];
		out[i + 1] = in[i + 2];
		out[i + 2] = in[i + 1];
		out[i + 3] = in[i];
	}
	/* Copy any trailing bytes that don't fill a full group */
	for (; i < len; i++)
		out[i] = in[i];
	out[len] = '\0';
	result->data = (char *)out;
	result->len = len;
}

/* ---- cap (capitalize) ---- */

/*
 * cap(x)     — capitalize the first lowercase letter [a-z] in x
 * cap(x, N)  — capitalize the character at position N (0-based)
 *
 * Only affects ASCII lowercase letters (a-z → A-Z).
 * Returns a copy; the original is not modified.
 */
/*
 * rotate(x, N) — rotate a string by N positions.
 * Positive N rotates right: last N chars move to front.
 * Negative N rotates left: first |N| chars move to end.
 * rotate("abcdef", 2) → "efabcd"
 * rotate("abcdef", -2) → "cdefab"
 */
static void fn_rotate(hx_func_entry *self, hx_val *args, int nargs,
                      hx_val *result, hx_arena *arena, uint8_t role)
{
	int len, n, split;
	(void)self; (void)nargs; (void)role;

	len = args[0].len;
	if (len == 0) { *result = args[0]; return; }

	n = val_to_int(&args[1]);
	/* normalize: positive = rotate right */
	n = n % len;
	if (n < 0) n += len;

	/* split point: for right rotation by n, split at len-n */
	split = len - n;

	result->data = hx_arena_alloc(arena, len + 1);
	memcpy(result->data, args[0].data + split, n);
	memcpy(result->data + n, args[0].data, split);
	result->data[len] = '\0';
	result->len = len;
}

static void fn_cap(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	int i, len, pos;
	(void)self; (void)role;

	len = args[0].len;
	result->data = hx_arena_alloc(arena, len + 1);
	memcpy(result->data, args[0].data, len);
	result->data[len] = '\0';
	result->len = len;

	if (nargs >= 2) {
		/* positional: capitalize at position N */
		pos = val_to_int(&args[1]);
		if (pos < 0) pos = len + pos;
		if (pos >= 0 && pos < len) {
			unsigned char c = (unsigned char)result->data[pos];
			if (c >= 'a' && c <= 'z')
				result->data[pos] = c - 32;
		}
	} else {
		/* default: first lowercase letter */
		for (i = 0; i < len; i++) {
			unsigned char c = (unsigned char)result->data[i];
			if (c >= 'a' && c <= 'z') {
				result->data[i] = c - 32;
				break;
			}
		}
	}
}

/* ---- rot13 ---- */

static void fn_rot13(hx_func_entry *self, hx_val *args, int nargs,
                     hx_val *result, hx_arena *arena, uint8_t role)
{
	int i, len;
	(void)self; (void)nargs; (void)role;
	len = args[0].len;
	result->data = hx_arena_alloc(arena, len + 1);
	for (i = 0; i < len; i++) {
		unsigned char c = (unsigned char)args[0].data[i];
		if (c >= 'A' && c <= 'Z')
			c = 'A' + (c - 'A' + 13) % 26;
		else if (c >= 'a' && c <= 'z')
			c = 'a' + (c - 'a' + 13) % 26;
		result->data[i] = c;
	}
	result->data[len] = '\0';
	result->len = len;
}

/* ---- base64 (standard, RFC 4648) ---- */

static const char b64_std[] =
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static void fn_base64(hx_func_entry *self, hx_val *args, int nargs,
                      hx_val *result, hx_arena *arena, uint8_t role)
{
	const unsigned char *in = (const unsigned char *)args[0].data;
	int inlen = args[0].len;
	int outlen = ((inlen + 2) / 3) * 4;
	char *out;
	int i, o;
	(void)self; (void)nargs; (void)role;

	out = hx_arena_alloc(arena, outlen + 1);
	o = 0;
	for (i = 0; i < inlen; i += 3) {
		unsigned int a = in[i];
		unsigned int b = (i + 1 < inlen) ? in[i + 1] : 0;
		unsigned int c = (i + 2 < inlen) ? in[i + 2] : 0;
		unsigned int triple = (a << 16) | (b << 8) | c;
		out[o++] = b64_std[(triple >> 18) & 0x3f];
		out[o++] = b64_std[(triple >> 12) & 0x3f];
		out[o++] = (i + 1 < inlen) ? b64_std[(triple >> 6) & 0x3f] : '=';
		out[o++] = (i + 2 < inlen) ? b64_std[triple & 0x3f] : '=';
	}
	out[o] = '\0';
	result->data = out;
	result->len  = o;
}

/* ---- frombase64 (base64 decode, RFC 4648) ---- */

static void fn_frombase64(hx_func_entry *self, hx_val *args, int nargs,
                           hx_val *result, hx_arena *arena, uint8_t role)
{
	static const signed char d64[256] = {
		[0 ... 255] = -1,
		['A'] = 0,  ['B'] = 1,  ['C'] = 2,  ['D'] = 3,
		['E'] = 4,  ['F'] = 5,  ['G'] = 6,  ['H'] = 7,
		['I'] = 8,  ['J'] = 9,  ['K'] = 10, ['L'] = 11,
		['M'] = 12, ['N'] = 13, ['O'] = 14, ['P'] = 15,
		['Q'] = 16, ['R'] = 17, ['S'] = 18, ['T'] = 19,
		['U'] = 20, ['V'] = 21, ['W'] = 22, ['X'] = 23,
		['Y'] = 24, ['Z'] = 25,
		['a'] = 26, ['b'] = 27, ['c'] = 28, ['d'] = 29,
		['e'] = 30, ['f'] = 31, ['g'] = 32, ['h'] = 33,
		['i'] = 34, ['j'] = 35, ['k'] = 36, ['l'] = 37,
		['m'] = 38, ['n'] = 39, ['o'] = 40, ['p'] = 41,
		['q'] = 42, ['r'] = 43, ['s'] = 44, ['t'] = 45,
		['u'] = 46, ['v'] = 47, ['w'] = 48, ['x'] = 49,
		['y'] = 50, ['z'] = 51,
		['0'] = 52, ['1'] = 53, ['2'] = 54, ['3'] = 55,
		['4'] = 56, ['5'] = 57, ['6'] = 58, ['7'] = 59,
		['8'] = 60, ['9'] = 61, ['+'] = 62, ['/'] = 63,
	};
	const unsigned char *in = (const unsigned char *)args[0].data;
	int inlen = args[0].len;
	int outmax = (inlen / 4) * 3 + 3;
	char *out;
	int i, o = 0;
	(void)self; (void)nargs; (void)role;

	out = hx_arena_alloc(arena, outmax);
	for (i = 0; i + 1 < inlen; ) {
		int a = d64[in[i++]];
		int b = (i < inlen) ? d64[in[i++]] : 0;
		int c = (i < inlen && in[i] != '=') ? d64[in[i++]] : -1;
		int d = (i < inlen && in[i] != '=') ? d64[in[i++]] : -1;
		if (a < 0 || b < 0) break;
		out[o++] = (a << 2) | (b >> 4);
		if (c >= 0) out[o++] = ((b & 0xf) << 4) | (c >> 2);
		if (d >= 0) out[o++] = ((c & 0x3) << 6) | d;
	}
	out[o] = '\0';
	result->data = out;
	result->len  = o;
}

/* ---- fromhex (hex decode) ---- */

static void fn_fromhex(hx_func_entry *self, hx_val *args, int nargs,
                        hx_val *result, hx_arena *arena, uint8_t role)
{
	const unsigned char *in = (const unsigned char *)args[0].data;
	int inlen = args[0].len;
	int outlen = inlen / 2;
	char *out;
	int i;
	(void)self; (void)nargs; (void)role;

	out = hx_arena_alloc(arena, outlen + 1);
	for (i = 0; i < outlen; i++) {
		int hi, lo;
		unsigned char c;
		c = in[i * 2];
		hi = (c >= '0' && c <= '9') ? c - '0' :
		     (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
		     (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
		c = in[i * 2 + 1];
		lo = (c >= '0' && c <= '9') ? c - '0' :
		     (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
		     (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
		out[i] = (hi << 4) | lo;
	}
	out[outlen] = '\0';
	result->data = out;
	result->len  = outlen;
}

/* ---- basic math (return integers with string representation) ---- */

static void fn_math_result(hx_val *result, hx_arena *arena, int64_t val)
{
	char *buf = hx_arena_alloc(arena, 24);
	result->is_int = 1;
	result->ival   = val;
	result->len    = sprintf(buf, "%lld", (long long)val);
	result->data   = buf;
}

/*
 * wperm(x, w0, w1, w2, w3) — permute 4-byte words within a byte sequence.
 * The input is divided into 4-byte words (padded with zero if needed).
 * The arguments w0-w3 are indices (0-based) specifying the new order.
 *
 * Example: MD5 produces 16 bytes = 4 words [a,b,c,d].
 *   wperm(md5_bin(pass), 3, 2, 0, 1)  → words [d,c,a,b] = "dcab"
 *   wperm(md5_bin(pass), 1, 2, 0, 3)  → words [b,c,a,d] = "bcad"
 */
static void fn_wperm(hx_func_entry *self, hx_val *args, int nargs,
                     hx_val *result, hx_arena *arena, uint8_t role)
{
	const unsigned char *in = (const unsigned char *)args[0].data;
	int inlen = args[0].len;
	int nwords = (inlen + 3) / 4;
	int outlen = nwords * 4;
	char *out;
	int i;
	(void)self; (void)role;

	/* need at least the data + one index argument */
	if (nargs < 2) {
		*result = args[0];
		return;
	}

	out = hx_arena_alloc(arena, outlen + 1);
	memset(out, 0, outlen);

	/* apply permutation: for each output position, copy the word
	   from the index specified by the corresponding argument */
	for (i = 0; i < nargs - 1 && i < nwords; i++) {
		int src = val_to_int(&args[i + 1]);
		if (src >= 0 && src < nwords) {
			int soff = src * 4;
			int doff = i * 4;
			int j;
			for (j = 0; j < 4 && soff + j < inlen; j++)
				out[doff + j] = in[soff + j];
		}
	}
	/* copy remaining words unchanged if not enough arguments */
	for (; i < nwords; i++) {
		int off = i * 4;
		int j;
		for (j = 0; j < 4 && off + j < inlen; j++)
			out[off + j] = in[off + j];
	}
	out[outlen] = '\0';
	result->data = out;
	result->len  = outlen;
}

static void fn_add(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs; (void)role;
	fn_math_result(result, arena, val_to_int(&args[0]) + val_to_int(&args[1]));
}

static void fn_sub(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs; (void)role;
	fn_math_result(result, arena, val_to_int(&args[0]) - val_to_int(&args[1]));
}

static void fn_mul(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs; (void)role;
	fn_math_result(result, arena, (int64_t)val_to_int(&args[0]) * val_to_int(&args[1]));
}

static void fn_mod(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	int b = val_to_int(&args[1]);
	(void)self; (void)nargs; (void)role;
	if (b == 0) { fn_math_result(result, arena, 0); return; }
	fn_math_result(result, arena, val_to_int(&args[0]) % b);
}

/* ---- phpass_encode (PHPBB3 / WordPress / phpass) ---- */

static const char itoa64[] =
	"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

static void fn_phpass_encode(hx_func_entry *self, hx_val *args, int nargs,
                             hx_val *result, hx_arena *arena, uint8_t role)
{
	const unsigned char *in = (const unsigned char *)args[0].data;
	int inlen = args[0].len;
	int outmax = ((inlen + 2) / 3) * 4 + 1;
	char *out;
	int i, o;
	unsigned int v;
	(void)self; (void)nargs; (void)role;

	out = hx_arena_alloc(arena, outmax);
	o = 0; i = 0;
	while (i < inlen) {
		v = in[i++];
		out[o++] = itoa64[v & 0x3f];
		if (i < inlen) v |= (unsigned int)in[i] << 8;
		out[o++] = itoa64[(v >> 6) & 0x3f];
		if (i++ >= inlen) break;
		if (i < inlen) v |= (unsigned int)in[i] << 16;
		out[o++] = itoa64[(v >> 12) & 0x3f];
		if (i++ >= inlen) break;
		out[o++] = itoa64[(v >> 18) & 0x3f];
	}
	out[o] = '\0';
	result->data = out;
	result->len  = o;
}

/* ---- des(key, plaintext) — single-DES ECB encrypt ----
 * key:       1..8 bytes (longer truncated, shorter zero-padded)
 * plaintext: 8 bytes (longer truncated, shorter zero-padded)
 * returns:   8 bytes; default rendering is 16-char lowercase hex
 */
static void fn_des(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	DES_cblock keyblk, ptblk, ctblk;
	DES_key_schedule ks;
	int klen = args[0].len, plen = args[1].len;
	(void)self; (void)nargs;

	memset(keyblk, 0, 8);
	memset(ptblk, 0, 8);
	memcpy(keyblk, args[0].data, klen < 8 ? klen : 8);
	memcpy(ptblk,  args[1].data, plen < 8 ? plen : 8);

	DES_set_key_unchecked(&keyblk, &ks);
	DES_ecb_encrypt(&ptblk, &ctblk, &ks, DES_ENCRYPT);

	emit_digest((unsigned char *)ctblk, 8, result, arena, role);
}

/* ---- des3(key, plaintext) — 3DES (EDE) ECB encrypt ----
 * key:       1..24 bytes — split into three 8-byte sub-keys k1|k2|k3
 *            (each sub-key zero-padded if the source segment is short)
 * plaintext: 8 bytes
 * returns:   8 bytes; default rendering is 16-char lowercase hex
 */
static void fn_des3(hx_func_entry *self, hx_val *args, int nargs,
                    hx_val *result, hx_arena *arena, uint8_t role)
{
	DES_cblock k1, k2, k3, ptblk, ctblk;
	DES_key_schedule ks1, ks2, ks3;
	int klen = args[0].len, plen = args[1].len;
	const char *kdata = args[0].data;
	(void)self; (void)nargs;

	memset(k1, 0, 8);
	memset(k2, 0, 8);
	memset(k3, 0, 8);
	memset(ptblk, 0, 8);

	if (klen >= 8)  memcpy(k1, kdata,    8);             else memcpy(k1, kdata, klen);
	if (klen > 8)  { if (klen >= 16) memcpy(k2, kdata+8,  8);  else memcpy(k2, kdata+8,  klen-8);  }
	if (klen > 16) { if (klen >= 24) memcpy(k3, kdata+16, 8);  else memcpy(k3, kdata+16, klen-16); }
	memcpy(ptblk, args[1].data, plen < 8 ? plen : 8);

	DES_set_key_unchecked(&k1, &ks1);
	DES_set_key_unchecked(&k2, &ks2);
	DES_set_key_unchecked(&k3, &ks3);
	DES_ecb3_encrypt(&ptblk, &ctblk, &ks1, &ks2, &ks3, DES_ENCRYPT);

	emit_digest((unsigned char *)ctblk, 8, result, arena, role);
}

/* ---- des_block(key7, plaintext) — DES with 7-byte key expansion ----
 * key7:      7 bytes — expanded to 8 bytes with DES parity bits
 *            (used by NTLMv1 challenge-response)
 * plaintext: 8 bytes
 * returns:   8 bytes; default rendering is 16-char lowercase hex
 */
static void fn_des_block(hx_func_entry *self, hx_val *args, int nargs,
                         hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char raw7[7], ptblk[8];
	DES_cblock key, ct;
	DES_key_schedule ks;
	int klen = args[0].len, plen = args[1].len;
	(void)self; (void)nargs;

	memset(raw7, 0, 7);
	memset(ptblk, 0, 8);
	memcpy(raw7,  args[0].data, klen < 7 ? klen : 7);
	memcpy(ptblk, args[1].data, plen < 8 ? plen : 8);

	/* 7→8 byte DES key expansion with parity */
	key[0] = (raw7[0] >> 1);
	key[1] = ((raw7[0] & 0x01) << 6) | (raw7[1] >> 2);
	key[2] = ((raw7[1] & 0x03) << 5) | (raw7[2] >> 3);
	key[3] = ((raw7[2] & 0x07) << 4) | (raw7[3] >> 4);
	key[4] = ((raw7[3] & 0x0F) << 3) | (raw7[4] >> 5);
	key[5] = ((raw7[4] & 0x1F) << 2) | (raw7[5] >> 6);
	key[6] = ((raw7[5] & 0x3F) << 1) | (raw7[6] >> 7);
	key[7] = raw7[6] & 0x7F;
	{ int i; for (i = 0; i < 8; i++) key[i] = (key[i] << 1) & 0xfe; }

	DES_set_key_unchecked((const_DES_cblock *)&key, &ks);
	DES_ecb_encrypt((const_DES_cblock *)ptblk, &ct, &ks, DES_ENCRYPT);

	emit_digest((unsigned char *)ct, 8, result, arena, role);
}

/* ---- juniper_encode (Juniper NetScreen / ScreenOS 30-char format) ----
 * Input:  16 raw bytes (typically an MD5 digest)
 * Output: 30-char ASCII string with 24 base64 data chars and 6 literal
 *         signature chars "nrcstn" interleaved at positions 0,6,12,17,23,29.
 * Bit layout: each 32-bit big-endian word splits into hi/lo 16-bit halves,
 *             each half encoded as 3 base64 chars at bit offsets {12,6,0}
 *             (groups of 4+6+6 = 16 bits).
 * Mirrors mdxfind.c:2140 (juniper_encode); keep in sync if that diverges.
 */
static void fn_juniper_encode(hx_func_entry *self, hx_val *args, int nargs,
                              hx_val *result, hx_arena *arena, uint8_t role)
{
	static const char b64alpha[] =
	    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	static const char sig[] = "nrcstn";
	static const int  sigpos[] = {0, 6, 12, 17, 23, 29};
	unsigned char buf[16];
	char data[24];
	char *out;
	int copylen = args[0].len < 16 ? args[0].len : 16;
	int w, p, si = 0, di = 0;
	unsigned int hi, lo;
	(void)self; (void)nargs; (void)role;

	memset(buf, 0, 16);
	memcpy(buf, args[0].data, copylen);

	for (w = 0; w < 4; w++) {
		hi = (buf[w*4+0] << 8) | buf[w*4+1];
		lo = (buf[w*4+2] << 8) | buf[w*4+3];
		data[w*6+0] = b64alpha[(hi >> 12) & 0x3f];
		data[w*6+1] = b64alpha[(hi >>  6) & 0x3f];
		data[w*6+2] = b64alpha[(hi      ) & 0x3f];
		data[w*6+3] = b64alpha[(lo >> 12) & 0x3f];
		data[w*6+4] = b64alpha[(lo >>  6) & 0x3f];
		data[w*6+5] = b64alpha[(lo      ) & 0x3f];
	}

	out = hx_arena_alloc(arena, 31);
	for (p = 0; p < 30; p++) {
		if (si < 6 && p == sigpos[si])
			out[p] = sig[si++];
		else
			out[p] = data[di++];
	}
	out[30] = '\0';

	result->data = out;
	result->len  = 30;
}

/* ---- Lotus/Domino algorithm support ----
 * Mirrors mdxfind.c:
 *   - lotus_magic_table[256]   at 2213
 *   - lotus64[] alphabet       at 2311
 *   - lotus64_encode           at 2556
 *   - lotus_mix                at 2579
 *   - lotus_transform_password at 2591
 *   - domino5_transform        at 2603
 * Keep in sync with mdxfind if any of those diverge.
 */
static const unsigned char lotus_magic_table[256] = {
  0xbd,0x56,0xea,0xf2,0xa2,0xf1,0xac,0x2a,0xb0,0x93,0xd1,0x9c,0x1b,0x33,0xfd,0xd0,
  0x30,0x04,0xb6,0xdc,0x7d,0xdf,0x32,0x4b,0xf7,0xcb,0x45,0x9b,0x31,0xbb,0x21,0x5a,
  0x41,0x9f,0xe1,0xd9,0x4a,0x4d,0x9e,0xda,0xa0,0x68,0x2c,0xc3,0x27,0x5f,0x80,0x36,
  0x3e,0xee,0xfb,0x95,0x1a,0xfe,0xce,0xa8,0x34,0xa9,0x13,0xf0,0xa6,0x3f,0xd8,0x0c,
  0x78,0x24,0xaf,0x23,0x52,0xc1,0x67,0x17,0xf5,0x66,0x90,0xe7,0xe8,0x07,0xb8,0x60,
  0x48,0xe6,0x1e,0x53,0xf3,0x92,0xa4,0x72,0x8c,0x08,0x15,0x6e,0x86,0x00,0x84,0xfa,
  0xf4,0x7f,0x8a,0x42,0x19,0xf6,0xdb,0xcd,0x14,0x8d,0x50,0x12,0xba,0x3c,0x06,0x4e,
  0xec,0xb3,0x35,0x11,0xa1,0x88,0x8e,0x2b,0x94,0x99,0xb7,0x71,0x74,0xd3,0xe4,0xbf,
  0x3a,0xde,0x96,0x0e,0xbc,0x0a,0xed,0x77,0xfc,0x37,0x6b,0x03,0x79,0x89,0x62,0xc6,
  0xd7,0xc0,0xd2,0x7c,0x6a,0x8b,0x22,0xa3,0x5b,0x05,0x5d,0x02,0x75,0xd5,0x61,0xe3,
  0x18,0x8f,0x55,0x51,0xad,0x1f,0x0b,0x5e,0x85,0xe5,0xc2,0x57,0x63,0xca,0x3d,0x6c,
  0xb4,0xc5,0xcc,0x70,0xb2,0x91,0x59,0x0d,0x47,0x20,0xc8,0x4f,0x58,0xe0,0x01,0xe2,
  0x16,0x38,0xc4,0x6f,0x3b,0x0f,0x65,0x46,0xbe,0x7e,0x2d,0x7b,0x82,0xf9,0x40,0xb5,
  0x1d,0x73,0xf8,0xeb,0x26,0xc7,0x87,0x97,0x25,0x54,0xb1,0x28,0xaa,0x98,0x9d,0xa5,
  0x64,0x6d,0x7a,0xd4,0x10,0x81,0x44,0xef,0x49,0xd6,0xae,0x2e,0xdd,0x76,0x5c,0x2f,
  0xa7,0x1c,0xc9,0x09,0x69,0x9a,0x83,0xcf,0x29,0x39,0xb9,0xe9,0x4c,0xff,0x43,0xab
};

static const char lotus64_alpha[] =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/";

static void hx_lotus_mix(unsigned char *state)
{
	int p = 0, i, k;
	for (i = 0; i < 18; i++) {
		for (k = 0; k < 48; k++) {
			p = (p + (48 - k)) & 0xff;
			p = state[k] ^ lotus_magic_table[p];
			state[k] = p;
		}
	}
}

static void hx_lotus_transform_password(const unsigned char *block,
                                        unsigned char *checksum)
{
	unsigned char t = checksum[15];
	int i;
	for (i = 0; i < 16; i++) {
		t = checksum[i] ^ lotus_magic_table[block[i] ^ t];
		checksum[i] = t;
	}
}

static void hx_domino5_transform(const unsigned char *input, int inlen,
                                 unsigned char *output)
{
	unsigned char state[48], padded[512], checksum[16];
	int padlen, nblocks, i, j;

	padlen = 16 - (inlen % 16);
	if (padlen == 0) padlen = 16;
	if (inlen + padlen > (int)sizeof(padded)) {
		memset(output, 0, 16);
		return;
	}
	memcpy(padded, input, inlen);
	memset(padded + inlen, padlen, padlen);
	nblocks = (inlen + padlen) / 16;

	memset(state, 0, 48);
	memset(checksum, 0, 16);

	for (i = 0; i < nblocks; i++) {
		unsigned char *block = padded + i * 16;
		for (j = 0; j < 16; j++) {
			state[16 + j] = block[j];
			state[32 + j] = block[j] ^ state[j];
		}
		hx_lotus_mix(state);
		hx_lotus_transform_password(block, checksum);
	}
	for (j = 0; j < 16; j++) {
		state[16 + j] = checksum[j];
		state[32 + j] = checksum[j] ^ state[j];
	}
	hx_lotus_mix(state);
	memcpy(output, state, 16);
}

static int hx_lotus64_encode(const unsigned char *in, int inlen, char *out)
{
	int i, j = 0;
	for (i = 0; i + 2 < inlen; i += 3) {
		out[j++] = lotus64_alpha[in[i] >> 2];
		out[j++] = lotus64_alpha[((in[i] & 3) << 4) | (in[i+1] >> 4)];
		out[j++] = lotus64_alpha[((in[i+1] & 0xf) << 2) | (in[i+2] >> 6)];
		out[j++] = lotus64_alpha[in[i+2] & 0x3f];
	}
	if (i < inlen) {
		out[j++] = lotus64_alpha[in[i] >> 2];
		if (i + 1 < inlen) {
			out[j++] = lotus64_alpha[((in[i] & 3) << 4) | (in[i+1] >> 4)];
			out[j++] = lotus64_alpha[(in[i+1] & 0xf) << 2];
		} else {
			out[j++] = lotus64_alpha[(in[i] & 3) << 4];
		}
	}
	out[j] = 0;
	return j;
}

/* ---- domino5(pass) — 16-byte Lotus/Domino 5 hash ---- */
static void fn_domino5(hx_func_entry *self, hx_val *args, int nargs,
                       hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char digest[16];
	(void)self; (void)nargs;

	if (args[0].len > 256) {
		memset(digest, 0, 16);
	} else {
		hx_domino5_transform((const unsigned char *)args[0].data,
		                     args[0].len, digest);
	}
	emit_digest(digest, 16, result, arena, role);
}

/* ---- lotus64_encode(bytes) — Lotus 0123...XYZabc...xyz+/ alphabet ----
 * Returns ASCII text (4 chars per 3 bytes; partial groups handled).
 */
static void fn_lotus64_encode(hx_func_entry *self, hx_val *args, int nargs,
                              hx_val *result, hx_arena *arena, uint8_t role)
{
	int outmax = ((args[0].len + 2) / 3) * 4 + 1;
	char *out;
	(void)self; (void)nargs; (void)role;

	out = hx_arena_alloc(arena, outmax);
	result->len  = hx_lotus64_encode((const unsigned char *)args[0].data,
	                                 args[0].len, out);
	result->data = out;
}

/* ---- domino6(pass, salt) — full Lotus/Domino 6 (G…) format ----
 * salt:  5 raw bytes (call as `domino6(pass, fromhex(salt))` if salt comes
 *        from mdxfind output as 10 hex chars).
 * Output: 22-char string "(G" + 19 lotus64 chars + ")".
 * Mirrors mdxfind.c:19754-19796.
 */
static void fn_domino6(hx_func_entry *self, hx_val *args, int nargs,
                       hx_val *result, hx_arena *arena, uint8_t role)
{
	static const char hexUC[] = "0123456789ABCDEF";
	unsigned char h1[16], buf[48], h2[16], raw[14];
	char *out;
	int x;
	(void)self; (void)nargs; (void)role;

	if (args[0].len > 256 || args[1].len < 5) {
		out = hx_arena_alloc(arena, 23);
		memcpy(out, "(G", 2);
		memset(out + 2, '0', 19);
		out[21] = ')';
		out[22] = '\0';
		result->data = out;
		result->len  = 22;
		return;
	}

	hx_domino5_transform((const unsigned char *)args[0].data,
	                     args[0].len, h1);

	memcpy(buf, args[1].data, 5);
	buf[5] = '(';
	for (x = 0; x < 14; x++) {
		buf[6 + x*2]     = hexUC[(h1[x] >> 4) & 0xf];
		buf[6 + x*2 + 1] = hexUC[ h1[x]       & 0xf];
	}
	hx_domino5_transform(buf, 34, h2);

	memcpy(raw, args[1].data, 5);
	memcpy(raw + 5, h2, 9);
	raw[3] += 4;                /* Lotus encoding quirk */

	out = hx_arena_alloc(arena, 23);
	out[0] = '(';
	out[1] = 'G';
	hx_lotus64_encode(raw, 14, out + 2);
	out[21] = ')';
	out[22] = '\0';
	result->data = out;
	result->len  = 22;
}

/* ---- a2e[]: ASCII → EBCDIC (CP037 / IBM-037 / EBCDIC-US) lookup ----
 * Mirrors mdxfind.c:2232 byte-for-byte; used by RACF, AS/400, and the
 * ebcdic() primitive.
 */
static const unsigned char hx_a2e[256] = {
  0x00,0x01,0x02,0x03,0x37,0x2d,0x2e,0x2f,0x16,0x05,0x25,0x0b,0x0c,0x0d,0x0e,0x0f,
  0x10,0x11,0x12,0x13,0x3c,0x3d,0x32,0x26,0x18,0x19,0x3f,0x27,0x1c,0x1d,0x1e,0x1f,
  0x40,0x5a,0x7f,0x7b,0x5b,0x6c,0x50,0x7d,0x4d,0x5d,0x5c,0x4e,0x6b,0x60,0x4b,0x61,
  0xf0,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0x7a,0x5e,0x4c,0x7e,0x6e,0x6f,
  0x7c,0xc1,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xd1,0xd2,0xd3,0xd4,0xd5,0xd6,
  0xd7,0xd8,0xd9,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xad,0xe0,0xbd,0x5f,0x6d,
  0x79,0x81,0x82,0x83,0x84,0x85,0x86,0x87,0x88,0x89,0x91,0x92,0x93,0x94,0x95,0x96,
  0x97,0x98,0x99,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xc0,0x4f,0xd0,0xa1,0x07,
  0x20,0x21,0x22,0x23,0x24,0x15,0x06,0x17,0x28,0x29,0x2a,0x2b,0x2c,0x09,0x0a,0x1b,
  0x30,0x31,0x1a,0x33,0x34,0x35,0x36,0x08,0x38,0x39,0x3a,0x3b,0x04,0x14,0x3e,0xe1,
  0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x51,0x52,0x53,0x54,0x55,0x56,0x57,
  0x58,0x59,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x70,0x71,0x72,0x73,0x74,0x75,
  0x76,0x77,0x78,0x80,0x8a,0x8b,0x8c,0x8d,0x8e,0x8f,0x90,0x9a,0x9b,0x9c,0x9d,0x9e,
  0x9f,0xa0,0xaa,0xab,0xac,0x4a,0xae,0xaf,0xb0,0xb1,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,
  0xb8,0xb9,0xba,0xbb,0xbc,0xa1,0xbe,0xbf,0xca,0xcb,0xcc,0xcd,0xce,0xcf,0xda,0xdb,
  0xdc,0xdd,0xde,0xdf,0xea,0xeb,0xec,0xed,0xee,0xef,0xfa,0xfb,0xfc,0xfd,0xfe,0xff
};

/* ---- a2e_pc[]: ASCII → EBCDIC-PC (CP1047 variant) ----
 * Mirrors mdxfind.c:2252; used by AS/400 DES password→key derivation.
 */
static const unsigned char hx_a2e_pc[256] = {
  0x2a,0xa8,0xae,0xad,0xc4,0xf1,0xf7,0xf4,0x86,0xa1,0xe0,0xbc,0xb3,0xb0,0xb6,0xb5,
  0x8a,0x89,0x8f,0x8c,0xd3,0xd0,0xce,0xe6,0x9b,0x98,0xd5,0xe5,0x92,0x91,0x97,0x94,
  0x2a,0x34,0x54,0x5d,0x1c,0x73,0x0b,0x51,0x31,0x10,0x13,0x37,0x7c,0x6b,0x3d,0x68,
  0x4a,0x49,0x4f,0x4c,0x43,0x40,0x46,0x45,0x5b,0x58,0x5e,0x16,0x32,0x57,0x76,0x75,
  0x52,0x29,0x2f,0x2c,0x23,0x20,0x26,0x25,0x3b,0x38,0x08,0x0e,0x0d,0x02,0x01,0x07,
  0x04,0x1a,0x19,0x6e,0x6d,0x62,0x61,0x67,0x64,0x7a,0x79,0x3e,0x6b,0x1f,0x15,0x70,
  0x58,0xa8,0xae,0xad,0xa2,0xa1,0xa7,0xa4,0xba,0xb9,0x89,0x8f,0x8c,0x83,0x80,0x86,
  0x85,0x9b,0x98,0xef,0xec,0xe3,0xe0,0xe6,0xe5,0xfb,0xf8,0x2a,0x7f,0x0b,0xe9,0xa4,
  0xea,0xe9,0xef,0xec,0xe3,0x80,0xa7,0x85,0xfb,0xf8,0xfe,0xfd,0xf2,0xb9,0xbf,0x9d,
  0xcb,0xc8,0x9e,0xcd,0xc2,0xc1,0xc7,0xba,0xda,0xd9,0xdf,0xdc,0xa2,0x83,0xd6,0x68,
  0x29,0x2f,0x2c,0x23,0x20,0x26,0x25,0x3b,0x38,0x08,0x0e,0x0d,0x02,0x01,0x07,0x04,
  0x1a,0x19,0x6e,0x6d,0x62,0x61,0x67,0x64,0x7a,0x79,0x4a,0x49,0x4f,0x4c,0x43,0x40,
  0x46,0x45,0x5b,0xab,0xbf,0xbc,0xb3,0xb0,0xb6,0xb5,0x8a,0x9e,0x9d,0x92,0x91,0x97,
  0x94,0xea,0xfe,0xfd,0xf2,0xf1,0xf7,0xf4,0xcb,0xc8,0xce,0xcd,0xc2,0xc1,0xc7,0xc4,
  0xda,0xd9,0xdf,0xdc,0xd3,0xd0,0xd6,0xd5,0x3e,0x3d,0x32,0x31,0x37,0x34,0x1f,0x1c,
  0x13,0x10,0x16,0x15,0x7f,0x7c,0x73,0x70,0x76,0x75,0x5e,0x5d,0x52,0x51,0x57,0x54
};

/* ---- SAP BCODE tables (mdxfind.c:2272 sapb_trans_tbl, 2291 bcodeArray) ---- */
static const unsigned char hx_sapb_trans_tbl[256] = {
    0x00,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0x3f,0x40,0x41,0x50,0x43,0x44,0x45,0x4b,0x47,0x48,0x4d,0x4e,0x54,0x51,0x53,0x46,
    0x35,0x36,0x37,0x38,0x39,0x3a,0x3b,0x3c,0x3d,0x3e,0x56,0x55,0x5c,0x49,0x5d,0x4a,
    0x42,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x58,0x5b,0x59,0xff,0x52,
    0x4c,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,0x0d,0x0e,0x0f,
    0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x1a,0x57,0x5e,0x5a,0x4f,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff
};
static const unsigned char hx_bcodeArray[48] = {
    0x14,0x77,0xf3,0xd4,0xbb,0x71,0x23,0xd0,0x03,0xff,0x47,0x93,0x55,0xaa,0x66,0x91,
    0xf2,0x88,0x6b,0x99,0xbf,0xcb,0x32,0x1a,0x19,0xd9,0xa7,0x82,0x22,0x49,0xa2,0x51,
    0xe2,0xb7,0x33,0x71,0x8b,0x9f,0x5d,0x01,0x44,0x70,0xae,0x11,0xef,0x28,0xf0,0x0d
};

/* ---- sap_bcode(pass, salt) — SAP BCODE 8-byte hash ----
 * Mirrors mdxfind.c:33128-33223 (the 8-byte path, pre-XOR-fold output).
 * For BCODE4 (e923), wrap the result as
 *   cut(upper(sap_bcode(pass, salt)), 0, 8) . "00000000"
 * since BCODE4 zero-fills the trailing 8 hex chars.
 */
static void fn_sap_bcode(hx_func_entry *self, hx_val *args, int nargs,
                         hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char tpass[8] = {0}, tsalt[12], digest[16], tbuf[64];
	unsigned char folded[8];
	int plen = args[0].len < 8 ? args[0].len : 8;
	int slen = args[1].len < 12 ? args[1].len : 12;
	int x, mlen;
	(void)self; (void)nargs;

	for (x = 0; x < 8; x++) {
		if (x < plen) {
			unsigned char c = (unsigned char)args[0].data[x];
			if (c >= 'a' && c <= 'z') c -= 32;
			tpass[x] = hx_sapb_trans_tbl[c];
		} else tpass[x] = 0;
	}
	for (x = 0; x < slen; x++) {
		unsigned char c = (unsigned char)args[1].data[x];
		if (c >= 'a' && c <= 'z') c -= 32;
		tsalt[x] = hx_sapb_trans_tbl[c];
	}

	{
		unsigned char buf[24];
		memcpy(buf, tpass, plen);
		memcpy(buf + plen, tsalt, slen);
		mlen = plen + slen;
		MD5(buf, mlen, digest);
	}

	{
		unsigned int sum20 = ((digest[0] >> 0) & 3)
		                   + ((digest[0] >> 2) & 3)
		                   + ((digest[1] >> 0) & 3)
		                   + ((digest[1] >> 2) & 3)
		                   + ((digest[2] >> 2) & 3);
		unsigned int i1 = 0, i2 = 0, i3 = 0;
		sum20 |= 0x20;
		memset(tbuf, 0, 64);
		while (i2 < sum20) {
			if (i1 < (unsigned int)plen) {
				if (digest[15 - i1] & 1) {
					tbuf[i2++] = hx_bcodeArray[48 - 1 - i1];
					if (i2 == sum20) break;
				}
				tbuf[i2++] = tpass[i1];
				if (i2 == sum20) break;
				i1++;
			}
			if (i3 < (unsigned int)slen) {
				tbuf[i2++] = tsalt[i3];
				if (i2 == sum20) break;
				i3++;
			}
			tbuf[i2] = hx_bcodeArray[i2 - i1 - i3];
			i2++;
			i2++;
		}
		MD5(tbuf, sum20, digest);
	}

	for (x = 0; x < 4; x++) {
		folded[x]     = digest[x]     ^ digest[x + 8];
		folded[x + 4] = digest[x + 4] ^ digest[x + 12];
	}
	emit_digest(folded, 8, result, arena, role);
}

/* ---- as400_des(pass, user) — IBM AS/400 DES password hash ----
 * Mirrors mdxfind.c:33246-33293.
 *   pass → ebcdic_pc (pad to 8 with 0x2a) → DES key
 *   user → upper, ebcdic (pad to 8 with 0x40 ASCII space). If user > 8 chars,
 *          fold remaining bytes via XOR back into bytes 0..(user.len-9).
 *   ciphertext = DES_ECB(plaintext, key)
 * Returns 8 bytes; default rendering is 16-char hex.
 */
static void fn_as400_des(hx_func_entry *self, hx_val *args, int nargs,
                         hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char deskey[8], plain[8], ctblk[8];
	DES_key_schedule ks;
	int plen = args[0].len, ulen = args[1].len;
	int x;
	(void)self; (void)nargs;

	if (plen > 8) plen = 8;
	for (x = 0; x < 8; x++)
		deskey[x] = (x < plen)
		    ? hx_a2e_pc[(unsigned char)args[0].data[x]]
		    : 0x2a;
	DES_set_key_unchecked((const_DES_cblock *)deskey, &ks);

	for (x = 0; x < 8; x++) {
		unsigned char c = (x < ulen) ? (unsigned char)args[1].data[x] : ' ';
		if (c >= 'a' && c <= 'z') c -= 32;
		plain[x] = hx_a2e[c];
	}
	for (x = 8; x < ulen; x++) {
		unsigned char c = (unsigned char)args[1].data[x];
		if (c >= 'a' && c <= 'z') c -= 32;
		plain[x - 8] ^= hx_a2e[c];
	}
	DES_ecb_encrypt((const_DES_cblock *)plain, (DES_cblock *)ctblk,
	                &ks, DES_ENCRYPT);

	emit_digest(ctblk, 8, result, arena, role);
}

/* ---- besder_encode(md5bytes) — Besder Auth 8-char encode ----
 * Algorithm per mdxfind.c:32992 (JOB_BESDER_AUTH). For each of 4
 * 4-byte groups in the 16-byte input, emits two alphanumeric chars:
 *   v0 = ((b0 + b1) & 0xff) % 62
 *   v1 = ((b2 + b3) & 0xff) % 62
 * indexed into "0-9A-Za-z". Inputs shorter than 16 bytes return
 * the empty string.
 */
static void fn_besder_encode(hx_func_entry *self, hx_val *args, int nargs,
                             hx_val *result, hx_arena *arena, uint8_t role)
{
	static const char besder_table[] =
	  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	const unsigned char *d = (const unsigned char *)args[0].data;
	char *out;
	int i;
	(void)self; (void)nargs; (void)role;

	if (args[0].len < 16) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}
	out = hx_arena_alloc(arena, 9);
	for (i = 0; i < 4; i++) {
		int v0 = ((d[i*4] + d[i*4+1]) & 0xff) % 62;
		int v1 = ((d[i*4+2] + d[i*4+3]) & 0xff) % 62;
		out[i*2]   = besder_table[v0];
		out[i*2+1] = besder_table[v1];
	}
	out[8] = '\0';
	result->data = out;
	result->len  = 8;
}

/* ---- dahua_encode(md5bytes) — Dahua Auth 8-char encode ----
 * Algorithm per mdxfind.c:32992 (JOB_DAHUA_AUTH; same case as
 * BESDER but without the & 0xff truncation):
 *   v0 = (b0 + b1) % 62
 *   v1 = (b2 + b3) % 62
 * indexed into "0-9A-Za-z". Inputs shorter than 16 bytes return
 * the empty string.
 */
static void fn_dahua_encode(hx_func_entry *self, hx_val *args, int nargs,
                            hx_val *result, hx_arena *arena, uint8_t role)
{
	static const char dahua_table[] =
	  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	const unsigned char *d = (const unsigned char *)args[0].data;
	char *out;
	int i;
	(void)self; (void)nargs; (void)role;

	if (args[0].len < 16) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}
	out = hx_arena_alloc(arena, 9);
	for (i = 0; i < 4; i++) {
		int v0 = (d[i*4] + d[i*4+1]) % 62;
		int v1 = (d[i*4+2] + d[i*4+3]) % 62;
		out[i*2]   = dahua_table[v0];
		out[i*2+1] = dahua_table[v1];
	}
	out[8] = '\0';
	result->data = out;
	result->len  = 8;
}

/* ---- aes_ecb_encrypt(key, plaintext) — single-block AES-ECB encrypt ----
 * Algorithm per mdxfind.c:33438 (JOB_AES{128,192,256}_NOKDF).
 * key:       16 / 24 / 32 bytes — selects AES-128/192/256.
 * plaintext: must be a positive multiple of 16 bytes.
 * Other lengths return the empty string.
 * Returns the ciphertext bytes, length == plaintext length.
 */
static void fn_aes_ecb_encrypt(hx_func_entry *self, hx_val *args, int nargs,
                               hx_val *result, hx_arena *arena, uint8_t role)
{
	int klen = args[0].len, ptlen = args[1].len;
	const unsigned char *key = (const unsigned char *)args[0].data;
	const unsigned char *pt  = (const unsigned char *)args[1].data;
	AES_KEY ak;
	unsigned char *out;
	int blocks, i;
	(void)self; (void)nargs;

	if ((klen != 16 && klen != 24 && klen != 32) ||
	    ptlen <= 0 || (ptlen & 15) != 0) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}

	AES_set_encrypt_key(key, klen * 8, &ak);
	blocks = ptlen / 16;
	out = (unsigned char *)hx_arena_alloc(arena, ptlen + 1);
	for (i = 0; i < blocks; i++)
		AES_ecb_encrypt(pt + i * 16, out + i * 16, &ak, AES_ENCRYPT);
	out[ptlen] = '\0';
	emit_digest((char *)out, ptlen, result, arena, role);
}

/* ---- aes_cbc_decrypt(key, iv, ct) — AES-CBC decrypt (no padding strip) ----
 * key: 16/24/32 bytes (AES-128/192/256).
 * iv:  16 bytes.
 * ct:  positive multiple of 16 bytes.
 * Returns the raw decrypted bytes (caller is responsible for any
 * padding handling). Bad lengths return the empty string.
 *
 * Algorithm per mdxfind.c:34340 (JOB_VEEAM_VBK) and elsewhere.
 */
static void fn_aes_cbc_decrypt(hx_func_entry *self, hx_val *args, int nargs,
                               hx_val *result, hx_arena *arena, uint8_t role)
{
	int klen = args[0].len, ivlen = args[1].len, ctlen = args[2].len;
	const unsigned char *key = (const unsigned char *)args[0].data;
	const unsigned char *ct  = (const unsigned char *)args[2].data;
	unsigned char ivcopy[16];
	AES_KEY ak;
	unsigned char *out;
	(void)self; (void)nargs;

	if ((klen != 16 && klen != 24 && klen != 32) ||
	    ivlen != 16 || ctlen <= 0 || (ctlen & 15) != 0) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}

	memcpy(ivcopy, args[1].data, 16);
	AES_set_decrypt_key(key, klen * 8, &ak);
	out = (unsigned char *)hx_arena_alloc(arena, ctlen + 1);
	AES_cbc_encrypt(ct, out, ctlen, &ak, ivcopy, AES_DECRYPT);
	out[ctlen] = '\0';
	emit_digest((char *)out, ctlen, result, arena, role);
}

/* ---- aes_cbc_encrypt(key, iv, pt) — AES-CBC encrypt (no padding) ----
 * Same length rules as aes_cbc_decrypt; caller pre-pads.
 */
static void fn_aes_cbc_encrypt(hx_func_entry *self, hx_val *args, int nargs,
                               hx_val *result, hx_arena *arena, uint8_t role)
{
	int klen = args[0].len, ivlen = args[1].len, ptlen = args[2].len;
	const unsigned char *key = (const unsigned char *)args[0].data;
	const unsigned char *pt  = (const unsigned char *)args[2].data;
	unsigned char ivcopy[16];
	AES_KEY ak;
	unsigned char *out;
	(void)self; (void)nargs;

	if ((klen != 16 && klen != 24 && klen != 32) ||
	    ivlen != 16 || ptlen <= 0 || (ptlen & 15) != 0) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}

	memcpy(ivcopy, args[1].data, 16);
	AES_set_encrypt_key(key, klen * 8, &ak);
	out = (unsigned char *)hx_arena_alloc(arena, ptlen + 1);
	AES_cbc_encrypt(pt, out, ptlen, &ak, ivcopy, AES_ENCRYPT);
	out[ptlen] = '\0';
	emit_digest((char *)out, ptlen, result, arena, role);
}

/* ---- aes_unwrap(kek, wrapped) — RFC 3394 AES Key Unwrap ----
 * kek:     16 bytes (AES-128) or 32 bytes (AES-256). Other lengths
 *          return empty.
 * wrapped: must be 8 + 8n bytes for n >= 1. Other lengths return empty.
 * Returns: 8n bytes of plaintext on success (when the unwrap-IV check
 *          matches 0xA6A6A6A6A6A6A6A6), or empty string on failure.
 */
static void fn_aes_unwrap(hx_func_entry *self, hx_val *args, int nargs,
                          hx_val *result, hx_arena *arena, uint8_t role)
{
	int klen = args[0].len, wlen = args[1].len;
	const unsigned char *kek = (const unsigned char *)args[0].data;
	const unsigned char *wrapped = (const unsigned char *)args[1].data;
	unsigned char A[8];
	unsigned char (*R)[8];
	unsigned char B[16];
	AES_KEY ak;
	int n, j, ii;
	(void)self; (void)nargs; (void)role;

	if ((klen != 16 && klen != 32) || wlen < 16 || (wlen & 7) != 0) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}
	n = wlen / 8 - 1;

	AES_set_decrypt_key(kek, klen * 8, &ak);
	memcpy(A, wrapped, 8);
	R = (unsigned char (*)[8])hx_arena_alloc(arena, n * 8);
	memcpy(R, wrapped + 8, n * 8);

	for (j = 5; j >= 0; j--) {
		for (ii = n - 1; ii >= 0; ii--) {
			unsigned int t = (unsigned int)(n * j + ii + 1);
			A[7] ^= t        & 0xff;
			A[6] ^= (t >> 8) & 0xff;
			A[5] ^= (t >> 16)& 0xff;
			A[4] ^= (t >> 24)& 0xff;
			memcpy(B,     A,    8);
			memcpy(B + 8, R[ii], 8);
			AES_ecb_encrypt(B, B, &ak, AES_DECRYPT);
			memcpy(A,     B,     8);
			memcpy(R[ii], B + 8, 8);
		}
	}

	{
		static const unsigned char iv[8] =
		    {0xa6,0xa6,0xa6,0xa6,0xa6,0xa6,0xa6,0xa6};
		if (memcmp(A, iv, 8) != 0) {
			result->data = hx_arena_alloc(arena, 1);
			result->data[0] = '\0';
			result->len = 0;
			return;
		}
	}

	{
		unsigned char *out = (unsigned char *)hx_arena_alloc(arena, n * 8 + 1);
		memcpy(out, R, n * 8);
		out[n * 8] = 0;
		result->data = (char *)out;
		result->len  = n * 8;
	}
}

/* ---- racf_kdfaes(pass, user, header, salt) — IBM RACF KDFAES ----
 * Algorithm per mdxfind.c:30390-30547. Five phases:
 *   1) DES key from EBCDIC-PC password (pad to 8 with 0x2a)
 *   2) DES-ECB encrypt EBCDIC username (pad to 8 with 0x40)
 *   3) Memory-filling PBKDF2-HMAC-SHA256 loop:
 *        mem_fac iterations producing 32-byte blocks in a memory buffer;
 *        salt updated each round to (prelast || out).
 *   4) Transposition pass: index-permuted PBKDF2-HMAC-SHA256(iter=1).
 *   5) Final PBKDF2-HMAC-SHA256 over (mem_fac-1)*32-byte buffer to derive
 *      a 32-byte AES-256 key; AES-256-ECB encrypt EBCDIC user (padded to
 *      16 with zeros) → 16-byte hash.
 *
 * Header (16 bytes):
 *   [8..9]   mem_fac_raw (BE u16, 1..15)  → mem_fac = (1<<raw)/32
 *   [10..11] rep_fac     (BE u16, >=1)    → pbkdf_iters = rep_fac * 100
 * Other header bytes are ignored by the KDF (carried in the format).
 *
 * Returns 16 bytes; default rendering is 32-char lowercase hex.
 */
static void fn_racf_kdfaes(hx_func_entry *self, hx_val *args, int nargs,
                           hx_val *result, hx_arena *arena, uint8_t role)
{
	const char *pdata = args[0].data;
	const char *udata = args[1].data;
	const char *hdata = args[2].data;
	const char *sdata = args[3].data;
	int plen = args[0].len, ulen = args[1].len;
	unsigned char deskey[8], euser[16], desout[8];
	unsigned char header[16], salt16[16];
	unsigned char key[32], outbuf[32], U[32], prelast[32];
	unsigned char pbsalt[64], stmp[128];
	unsigned char *membuf;
	unsigned char aeskey[32], aesout[16];
	DES_key_schedule ks;
	AES_KEY ak;
	int mem_fac, rep_fac, pbkdf_iters, mem_fac_raw;
	int i, j, k, slen;
	unsigned int hmlen = 32;
	(void)self; (void)nargs;

	if (plen > 8) plen = 8;
	if (args[2].len < 16 || args[3].len < 16) {
		memset(aesout, 0, 16);
		emit_digest(aesout, 16, result, arena, role);
		return;
	}
	memcpy(header, hdata, 16);
	memcpy(salt16, sdata, 16);

	/* Header parameters */
	mem_fac_raw = ((int)header[8] << 8) | header[9];
	rep_fac     = ((int)header[10] << 8) | header[11];
	if (mem_fac_raw < 1 || mem_fac_raw > 15 || rep_fac < 1) {
		memset(aesout, 0, 16);
		emit_digest(aesout, 16, result, arena, role);
		return;
	}
	mem_fac = (1 << mem_fac_raw) / 32;
	if (mem_fac < 1) {
		memset(aesout, 0, 16);
		emit_digest(aesout, 16, result, arena, role);
		return;
	}
	pbkdf_iters = rep_fac * 100;

	/* Phase 1: DES key from EBCDIC-PC password */
	for (i = 0; i < 8; i++)
		deskey[i] = (i < plen) ? hx_a2e_pc[(unsigned char)pdata[i]] : 0x2a;
	DES_set_key_unchecked((const_DES_cblock *)deskey, &ks);

	/* Username → standard EBCDIC, pad to 8 with 0x40 */
	for (i = 0; i < 8; i++) {
		unsigned char c = (i < ulen) ? (unsigned char)udata[i] : ' ';
		if (c >= 'a' && c <= 'z') c -= 32;
		euser[i] = hx_a2e[c];
	}

	/* Phase 2: DES encrypt username */
	DES_ecb_encrypt((const_DES_cblock *)euser, (DES_cblock *)desout,
	                &ks, DES_ENCRYPT);

	/* Phase 3: memory-filling PBKDF2-HMAC-SHA256 loop */
	membuf = (unsigned char *)hx_arena_alloc(arena, (size_t)mem_fac * 32);
	memset(key, 0, 32);
	memcpy(key, desout, 8);
	memcpy(pbsalt, salt16, 16);
	pbsalt[16] = (mem_fac >> 24) & 0xff;
	pbsalt[17] = (mem_fac >> 16) & 0xff;
	pbsalt[18] = (mem_fac >>  8) & 0xff;
	pbsalt[19] = (mem_fac      ) & 0xff;
	slen = 20;

	for (i = 0; i < mem_fac; i++) {
		memcpy(stmp, pbsalt, slen);
		stmp[slen]   = 0;
		stmp[slen+1] = 0;
		stmp[slen+2] = 0;
		stmp[slen+3] = 1;
		HMAC(EVP_sha256(), key, 8, stmp, slen + 4, U, &hmlen);
		memcpy(outbuf, U, 32);
		for (j = 1; j < pbkdf_iters; j++) {
			if (j == pbkdf_iters - 1) memcpy(prelast, U, 32);
			HMAC(EVP_sha256(), key, 8, U, 32, U, &hmlen);
			for (k = 0; k < 32; k++) outbuf[k] ^= U[k];
		}
		memcpy(membuf + (size_t)i * 32, outbuf, 32);
		memcpy(pbsalt, prelast, 16);
		memcpy(pbsalt + 16, outbuf, 32);
		slen = 48;
	}

	/* Phase 4: transposition pass */
	memcpy(key, outbuf, 32);
	for (i = 0; i < mem_fac; i++) {
		uint32_t n_val = ((uint32_t)key[28] << 24)
		               | ((uint32_t)key[29] << 16)
		               | ((uint32_t)key[30] << 8)
		               |  (uint32_t)key[31];
		int n_key = n_val & (mem_fac - 1);
		memcpy(stmp, membuf + (size_t)n_key * 32, 32);
		stmp[32] = 0; stmp[33] = 0; stmp[34] = 0; stmp[35] = 1;
		HMAC(EVP_sha256(), key, 32, stmp, 36, outbuf, &hmlen);
		memcpy(membuf + (size_t)i * 32, outbuf, 32);
		memcpy(key, outbuf, 32);
	}

	/* Phase 5: final PBKDF2 over (mem_fac-1)*32 bytes */
	{
		int final_slen = (mem_fac - 1) * 32;
		unsigned char *fs;
		memset(membuf + (size_t)(mem_fac - 1) * 32, 0, 32);
		fs = (unsigned char *)hx_arena_alloc(arena, final_slen + 8);
		memcpy(fs, membuf, final_slen);
		fs[final_slen]   = 0;
		fs[final_slen+1] = 0;
		fs[final_slen+2] = 0;
		fs[final_slen+3] = 1;
		HMAC(EVP_sha256(), key, 32, fs, final_slen + 4, U, &hmlen);
		memcpy(outbuf, U, 32);
		for (j = 1; j < pbkdf_iters; j++) {
			HMAC(EVP_sha256(), key, 32, U, 32, U, &hmlen);
			for (k = 0; k < 32; k++) outbuf[k] ^= U[k];
		}
		memcpy(aeskey, outbuf, 32);
	}

	/* Phase 6: AES-256-ECB encrypt EBCDIC user (pad to 16 with 0x00) */
	{
		unsigned char pt[16];
		memset(pt, 0, 16);
		memcpy(pt, euser, 8);
		AES_set_encrypt_key(aeskey, 256, &ak);
		AES_encrypt(pt, aesout, &ak);
	}

	emit_digest(aesout, 16, result, arena, role);
}

/* ---- axcrypt(pass, salt, iter) — AxCrypt AES Key Wrap KDF ----
 * Algorithm per mdxfind.c:19274-19345 (forward path):
 *   KEK = (sha1(pass)[0..15]) XOR salt[0..15]
 *   A   = 0xA6 * 8;  R1 = R2 = 0
 *   for j in 0..iter-1:
 *     A,R1 = AES_enc(A||R1, KEK);  A ^= LE32(2*j+1) at A[0..3]
 *     A,R2 = AES_enc(A||R2, KEK);  A ^= LE32(2*j+2) at A[0..3]
 *   output = A || R1 || R2  (24 bytes, default 48 lower-hex)
 */
static void fn_axcrypt(hx_func_entry *self, hx_val *args, int nargs,
                       hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char sha[20], salt16[16], kek[16];
	unsigned char A[8], R1[8], R2[8];
	unsigned char block[16], outblk[16];
	AES_KEY ks;
	int iter, j, i, slen;
	(void)self; (void)nargs;

	SHA1((const unsigned char *)args[0].data, args[0].len, sha);

	memset(salt16, 0, 16);
	slen = args[1].len < 16 ? args[1].len : 16;
	memcpy(salt16, args[1].data, slen);

	for (i = 0; i < 16; i++) kek[i] = sha[i] ^ salt16[i];

	iter = val_to_int(&args[2]);
	if (iter < 1) iter = 1;

	memset(A, 0xa6, 8);
	memset(R1, 0, 8);
	memset(R2, 0, 8);

	AES_set_encrypt_key(kek, 128, &ks);

	for (j = 0; j < iter; j++) {
		unsigned long tv;

		memcpy(block,     A, 8);
		memcpy(block + 8, R1, 8);
		AES_encrypt(block, outblk, &ks);
		memcpy(A,  outblk,     8);
		memcpy(R1, outblk + 8, 8);
		tv = (unsigned long)(2 * j + 1);
		A[0] ^= tv         & 0xff;
		A[1] ^= (tv >> 8)  & 0xff;
		A[2] ^= (tv >> 16) & 0xff;
		A[3] ^= (tv >> 24) & 0xff;

		memcpy(block,     A, 8);
		memcpy(block + 8, R2, 8);
		AES_encrypt(block, outblk, &ks);
		memcpy(A,  outblk,     8);
		memcpy(R2, outblk + 8, 8);
		tv = (unsigned long)(2 * j + 2);
		A[0] ^= tv         & 0xff;
		A[1] ^= (tv >> 8)  & 0xff;
		A[2] ^= (tv >> 16) & 0xff;
		A[3] ^= (tv >> 24) & 0xff;
	}

	{
		unsigned char wrapped[24];
		memcpy(wrapped,      A,  8);
		memcpy(wrapped + 8,  R1, 8);
		memcpy(wrapped + 16, R2, 8);
		emit_digest(wrapped, 24, result, arena, role);
	}
}

/* ---- oracle7(pass, user) — Oracle 7 H: Type DES-CBC password hash ----
 * Algorithm per mdxfind.c:17729-17793:
 *   wbuf = utf16be( upper(user) . upper(pass) ) zero-padded so that
 *          (user.len + pass.len) rounds up to a multiple of 4 characters.
 *   key1 = 0x0123456789ABCDEF (constant)
 *   pass1 = DES-CBC(wbuf, key=key1, iv=0)
 *   key2 = last 8 bytes (= IV state) from pass1
 *   pass2 = DES-CBC(wbuf, key=key2, iv=0)
 *   result = last 8 bytes from pass2
 * Returns 8 bytes; default rendering is 16-char hex. (mdxfind reports
 * the digest in uppercase hex; wrap with upper() if that's wanted.)
 */
static void fn_oracle7(hx_func_entry *self, hx_val *args, int nargs,
                       hx_val *result, hx_arena *arena, uint8_t role)
{
	static const unsigned char key1[8] =
	    {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF};
	const char *pdata = args[0].data;
	const char *udata = args[1].data;
	int plen = args[0].len, ulen = args[1].len;
	int catlen, padlen, widelen, max_w;
	unsigned char *wbuf;
	DES_key_schedule ks;
	DES_cblock iv, blk;
	int y, i;
	(void)self; (void)nargs;

	catlen = ulen + plen;
	padlen = (catlen + 3) & ~3;
	max_w  = padlen * 2;
	wbuf   = (unsigned char *)hx_arena_alloc(arena, max_w + 8);
	widelen = 0;

	for (i = 0; i < ulen; i++) {
		unsigned char c = (unsigned char)udata[i];
		if (c >= 'a' && c <= 'z') c -= 32;
		wbuf[widelen++] = 0;
		wbuf[widelen++] = c;
	}
	for (i = 0; i < plen; i++) {
		unsigned char c = (unsigned char)pdata[i];
		if (c >= 'a' && c <= 'z') c -= 32;
		wbuf[widelen++] = 0;
		wbuf[widelen++] = c;
	}
	while (widelen < max_w) {
		wbuf[widelen++] = 0;
		wbuf[widelen++] = 0;
	}

	/* Pass 1: DES-CBC with fixed key */
	DES_set_key_unchecked((const_DES_cblock *)key1, &ks);
	memset(iv, 0, 8);
	for (y = 0; y < widelen; y += 8) {
		blk[0] = wbuf[y+0] ^ iv[0]; blk[1] = wbuf[y+1] ^ iv[1];
		blk[2] = wbuf[y+2] ^ iv[2]; blk[3] = wbuf[y+3] ^ iv[3];
		blk[4] = wbuf[y+4] ^ iv[4]; blk[5] = wbuf[y+5] ^ iv[5];
		blk[6] = wbuf[y+6] ^ iv[6]; blk[7] = wbuf[y+7] ^ iv[7];
		DES_ecb_encrypt((const_DES_cblock *)blk, (DES_cblock *)iv,
		                &ks, DES_ENCRYPT);
	}

	/* Pass 2: DES-CBC with key derived from last block of pass 1 */
	DES_set_key_unchecked((const_DES_cblock *)iv, &ks);
	memset(iv, 0, 8);
	for (y = 0; y < widelen; y += 8) {
		blk[0] = wbuf[y+0] ^ iv[0]; blk[1] = wbuf[y+1] ^ iv[1];
		blk[2] = wbuf[y+2] ^ iv[2]; blk[3] = wbuf[y+3] ^ iv[3];
		blk[4] = wbuf[y+4] ^ iv[4]; blk[5] = wbuf[y+5] ^ iv[5];
		blk[6] = wbuf[y+6] ^ iv[6]; blk[7] = wbuf[y+7] ^ iv[7];
		DES_ecb_encrypt((const_DES_cblock *)blk, (DES_cblock *)iv,
		                &ks, DES_ENCRYPT);
	}

	emit_digest((unsigned char *)iv, 8, result, arena, role);
}

/* ---- ebcdic(s) — ASCII → EBCDIC byte-by-byte conversion ---- */
static void fn_ebcdic(hx_func_entry *self, hx_val *args, int nargs,
                     hx_val *result, hx_arena *arena, uint8_t role)
{
	int n = args[0].len;
	const unsigned char *in = (const unsigned char *)args[0].data;
	unsigned char *out;
	int i;
	(void)self; (void)nargs; (void)role;

	out = (unsigned char *)hx_arena_alloc(arena, n + 1);
	for (i = 0; i < n; i++) out[i] = hx_a2e[in[i]];
	out[n] = '\0';
	result->data = (char *)out;
	result->len  = n;
}

/* ---- racf_encrypt(pass, user) — IBM RACF DES hash ----
 * Algorithm per mdxfind.c:19619-19662:
 *   Pass → DES key: ebcdic(c) ^ 0x55, shift left 1 bit, set low bit for odd parity.
 *   User → plaintext: upper, pad-to-8 with ASCII space, then ebcdic.
 *   ciphertext = DES_ECB_encrypt(plaintext, key)
 * Returns the 8-byte ciphertext; default rendering is 16-char lowercase hex.
 * For the canonical $racf$*USER*HEX form, wrap with upper() and prepend.
 */
static void fn_racf_encrypt(hx_func_entry *self, hx_val *args, int nargs,
                            hx_val *result, hx_arena *arena, uint8_t role)
{
	unsigned char deskey[8], plain[8], ctblk[8];
	DES_key_schedule ks;
	int plen = args[0].len < 8 ? args[0].len : 8;
	int ulen = args[1].len < 8 ? args[1].len : 8;
	int x, bits;
	unsigned char val, t;
	(void)self; (void)nargs;

	for (x = 0; x < 8; x++) {
		unsigned char e = (x < plen)
		    ? hx_a2e[(unsigned char)args[0].data[x]]
		    : 0x40;             /* EBCDIC space */
		val = (e ^ 0x55);
		val = (val << 1) & 0xfe;
		bits = 0; t = val;
		while (t) { bits += t & 1; t >>= 1; }
		if ((bits & 1) == 0) val |= 1;
		deskey[x] = val;
	}
	for (x = 0; x < 8; x++) {
		unsigned char c = (x < ulen) ? (unsigned char)args[1].data[x] : ' ';
		if (c >= 'a' && c <= 'z') c -= 32;
		plain[x] = hx_a2e[c];
	}
	DES_set_key_unchecked((DES_cblock *)deskey, &ks);
	DES_ecb_encrypt((DES_cblock *)plain, (DES_cblock *)ctblk, &ks, DES_ENCRYPT);

	emit_digest(ctblk, 8, result, arena, role);
}

/* ---- cisco_pix_encode (Cisco PIX / ASA 16-char format) ----
 * Input:  16 raw bytes (typically an MD5 digest)
 * Output: 16-char ASCII string in the phpitoa64 alphabet
 *         (./0-9A-Za-z), encoding 4 chunks of 3 bytes each
 *         (every 4th byte of the digest is dropped).
 * Mirrors mdxfind.c:2176 (cisco_pix_encode); keep in sync if that diverges.
 */
static void fn_cisco_pix_encode(hx_func_entry *self, hx_val *args, int nargs,
                                hx_val *result, hx_arena *arena, uint8_t role)
{
	static const char phpitoa64[] =
	    "./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	unsigned char buf[16];
	char *out;
	int copylen = args[0].len < 16 ? args[0].len : 16;
	int i, j = 0;
	(void)self; (void)nargs; (void)role;

	memset(buf, 0, 16);
	memcpy(buf, args[0].data, copylen);

	out = hx_arena_alloc(arena, 17);
	for (i = 0; i < 16; i += 4) {
		unsigned int v = buf[i] | (buf[i+1] << 8) | (buf[i+2] << 16);
		out[j++] = phpitoa64[v & 0x3f];
		out[j++] = phpitoa64[(v >> 6) & 0x3f];
		out[j++] = phpitoa64[(v >> 12) & 0x3f];
		out[j++] = phpitoa64[(v >> 18) & 0x3f];
	}
	out[16] = '\0';
	result->data = out;
	result->len  = 16;
}

/* ---- progress_encode (Progress OpenEdge ENCODE) ---- */

extern void myprogress(char *cur, int len, char *dest);

static void fn_progress_encode(hx_func_entry *self, hx_val *args, int nargs,
                               hx_val *result, hx_arena *arena, uint8_t role)
{
	int inlen = args[0].len;
	int padlen = ((inlen + 15) / 16) * 16;
	char *padded;
	char *out;
	(void)self; (void)nargs; (void)role;

	if (padlen < 16) padlen = 16;
	padded = hx_arena_alloc(arena, padlen);
	out = hx_arena_alloc(arena, 17);

	memcpy(padded, args[0].data, inlen);
	memset(padded + inlen, 0, padlen - inlen);
	myprogress(padded, inlen, out);
	out[16] = '\0';

	result->data = out;
	result->len = 16;
}

/* ---- sha512crypt_encode ---- */

/*
 * SHA-512 crypt uses a custom base64 with alphabet ./0-9A-Za-z
 * and a specific byte permutation of the 64-byte digest.
 * The encoding groups bytes in a specific order (not sequential)
 * and packs 3 bytes into 4 characters, similar to phpass but
 * with big-endian bit extraction.
 */
static const char crypt_b64[] =
	"./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

static void crypt_b64_encode3(const unsigned char *in, int n,
                               char *out, int *opos)
{
	unsigned int v = 0;
	int i, o = *opos;

	/* big-endian packing: first byte is high bits */
	if (n > 0) v  = (unsigned int)in[0] << 16;
	if (n > 1) v |= (unsigned int)in[1] << 8;
	if (n > 2) v |= (unsigned int)in[2];

	for (i = 0; i < n + 1; i++) {
		out[o++] = crypt_b64[v & 0x3f];
		v >>= 6;
	}
	*opos = o;
}

static void fn_sha512crypt_encode(hx_func_entry *self, hx_val *args,
                                   int nargs, hx_val *result,
                                   hx_arena *arena, uint8_t role)
{
	/*
	 * SHA-512 crypt byte permutation: the 64-byte digest is
	 * rearranged into groups of 3 bytes before base64 encoding.
	 * Permutation order per crypt(3) specification.
	 */
	static const int perm[] = {
		42, 21,  0,  1, 43, 22, 23,  2, 44, 45, 24,  3,
		 4, 46, 25, 26,  5, 47, 48, 27,  6,  7, 49, 28,
		29,  8, 50, 51, 30,  9, 10, 52, 31, 32, 11, 53,
		54, 33, 12, 13, 55, 34, 35, 14, 56, 57, 36, 15,
		16, 58, 37, 38, 17, 59, 60, 39, 18, 19, 61, 40,
		41, 20, 62, 63
	};
	const unsigned char *in = (const unsigned char *)args[0].data;
	int inlen = args[0].len;
	char *out;
	int opos = 0, i;
	unsigned char permuted[64];
	(void)self; (void)nargs; (void)role;

	/* permute the input bytes (pad with 0 if short) */
	for (i = 0; i < 64; i++)
		permuted[i] = (perm[i] < inlen) ? in[perm[i]] : 0;

	/* encode: 21 groups of 3 bytes + 1 trailing byte = 86 chars */
	out = hx_arena_alloc(arena, 90);
	for (i = 0; i < 63; i += 3)
		crypt_b64_encode3(&permuted[i], 3, out, &opos);
	/* last byte: 1 byte → 2 chars */
	crypt_b64_encode3(&permuted[63], 1, out, &opos);

	out[opos] = '\0';
	result->data = out;
	result->len  = opos;
}

/* ---- bcrypt_encode ---- */

/*
 * bcrypt uses a custom base64 alphabet:
 *   ./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789
 * Note: different order from standard base64 and crypt.
 * Encoding is big-endian, 4 chars per 3 bytes, no padding.
 */
static const char bcrypt_b64[] =
	"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

static void fn_bcrypt_encode(hx_func_entry *self, hx_val *args,
                              int nargs, hx_val *result,
                              hx_arena *arena, uint8_t role)
{
	const unsigned char *in = (const unsigned char *)args[0].data;
	int inlen = args[0].len;
	int outmax = ((inlen + 2) / 3) * 4 + 1;
	char *out;
	int i, o;
	(void)self; (void)nargs; (void)role;

	out = hx_arena_alloc(arena, outmax);
	o = 0;
	for (i = 0; i < inlen; i += 3) {
		unsigned int a = in[i];
		unsigned int b = (i + 1 < inlen) ? in[i + 1] : 0;
		unsigned int c = (i + 2 < inlen) ? in[i + 2] : 0;
		unsigned int triple = (a << 16) | (b << 8) | c;

		out[o++] = bcrypt_b64[(triple >> 18) & 0x3f];
		out[o++] = bcrypt_b64[(triple >> 12) & 0x3f];
		if (i + 1 < inlen)
			out[o++] = bcrypt_b64[(triple >> 6) & 0x3f];
		if (i + 2 < inlen)
			out[o++] = bcrypt_b64[triple & 0x3f];
	}
	out[o] = '\0';
	result->data = out;
	result->len  = o;
}

/* ---- iconv-based encoding functions ---- */

#include <iconv.h>

/*
 * Generic iconv wrapper.  Converts args[0] from one encoding to
 * another using iconv.  Uses //IGNORE to silently drop characters
 * that cannot be represented in the target encoding.
 *
 * If iconv_open fails (unsupported encoding on this platform),
 * the input is returned unchanged with a warning on first use.
 */
static void hx_iconv_convert(const char *from, const char *to,
                              hx_val *args, hx_val *result,
                              hx_arena *arena)
{
	iconv_t cd;
	char *inbuf, *outbuf, *out;
	size_t inleft, outleft, outmax;

	outmax = args[0].len * 4 + 4;  /* generous: UTF-8 → UTF-16 at most 2x */
	out = hx_arena_alloc(arena, outmax);

	cd = iconv_open(to, from);
	if (cd == (iconv_t)-1) {
		/* unsupported encoding: return input unchanged */
		static int warned = 0;
		if (!warned) {
			fprintf(stderr, "hx: iconv_open(\"%s\",\"%s\") failed\n",
			        to, from);
			warned = 1;
		}
		memcpy(out, args[0].data, args[0].len);
		result->data = out;
		result->len  = args[0].len;
		return;
	}

	inbuf = args[0].data;
	inleft = args[0].len;
	outbuf = out;
	outleft = outmax;

	iconv(cd, &inbuf, &inleft, &outbuf, &outleft);
	iconv(cd, NULL, NULL, NULL, NULL);  /* flush */
	iconv_close(cd);

	result->data = out;
	result->len  = outmax - outleft;
}

/*
 * utf16le(x) — UTF-8 to UTF-16LE via iconv.
 * Uses //IGNORE: invalid UTF-8 sequences are silently dropped.
 * This matches mdxfind's -b behavior exactly.
 */
static void fn_utf16le(hx_func_entry *self, hx_val *args, int nargs,
                       hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs; (void)role;
	hx_iconv_convert("UTF-8", "UTF-16LE//IGNORE", args, result, arena);
}

/*
 * utf16be(x) — UTF-8 to UTF-16BE via iconv.
 */
static void fn_utf16be(hx_func_entry *self, hx_val *args, int nargs,
                       hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs; (void)role;
	hx_iconv_convert("UTF-8", "UTF-16BE//IGNORE", args, result, arena);
}

/*
 * utf7(x) — UTF-8 to UTF-7 via iconv.
 */
static void fn_utf7(hx_func_entry *self, hx_val *args, int nargs,
                    hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs; (void)role;
	hx_iconv_convert("UTF-8", "UTF-7//IGNORE", args, result, arena);
}

/*
 * zext16(x) — zero-extend each byte to a 16-bit word (little-endian).
 * This is the hashcat-compatible "UTF-16LE" that does NOT perform
 * real UTF-8 decoding: each input byte becomes (byte, 0x00).
 * Matches hashcat's NTLM implementation.
 */
static void fn_zext16(hx_func_entry *self, hx_val *args, int nargs,
                      hx_val *result, hx_arena *arena, uint8_t role)
{
	const unsigned char *in = (const unsigned char *)args[0].data;
	int inlen = args[0].len;
	char *out;
	int i;
	(void)self; (void)nargs; (void)role;

	out = hx_arena_alloc(arena, inlen * 2 + 2);
	for (i = 0; i < inlen; i++) {
		out[i * 2]     = in[i];
		out[i * 2 + 1] = 0;
	}
	result->data = out;
	result->len  = inlen * 2;
}

/*
 * from_cp1252(x) — reinterpret bytes as CP1252, convert to UTF-8.
 * from_cp1251(x) — reinterpret bytes as CP1251, convert to UTF-8.
 */
static void fn_from_cp1252(hx_func_entry *self, hx_val *args, int nargs,
                            hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs; (void)role;
	hx_iconv_convert("CP1252", "UTF-8//IGNORE", args, result, arena);
}

static void fn_from_cp1251(hx_func_entry *self, hx_val *args, int nargs,
                            hx_val *result, hx_arena *arena, uint8_t role)
{
	(void)self; (void)nargs; (void)role;
	hx_iconv_convert("CP1251", "UTF-8//IGNORE", args, result, arena);
}

/* ---- length ---- */

/*
 * length(x) — returns the byte length of x as an integer.
 * Useful in cut() arguments, loop bounds, and conditions.
 */
static void fn_length(hx_func_entry *self, hx_val *args, int nargs,
                      hx_val *result, hx_arena *arena, uint8_t role)
{
	char *buf;
	(void)self; (void)nargs; (void)role;

	result->is_int = 1;
	result->ival   = args[0].len;
	/* also produce string representation for use in string context */
	buf = hx_arena_alloc(arena, 20);
	result->len = sprintf(buf, "%d", args[0].len);
	result->data = buf;
}

/* ---- xor ---- */

/*
 * xor(a, b) — byte-wise exclusive OR.
 * If the arguments are of unequal length, the shorter is
 * cycled (repeated) to match the length of the longer.
 */
static void fn_xor(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	int alen = args[0].len, blen = args[1].len;
	int outlen = alen > blen ? alen : blen;
	char *out;
	int i;
	(void)self; (void)nargs; (void)role;

	if (outlen == 0 || alen == 0 || blen == 0) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}

	out = hx_arena_alloc(arena, outlen + 1);
	for (i = 0; i < outlen; i++)
		out[i] = args[0].data[i % alen] ^ args[1].data[i % blen];
	out[outlen] = '\0';
	result->data = out;
	result->len  = outlen;
}

/* ---- and / or (bitwise) ---- */

/*
 * and(a, b) — byte-wise AND.  Shorter argument is cycled.
 * or(a, b)  — byte-wise OR.   Shorter argument is cycled.
 */
static void fn_and(hx_func_entry *self, hx_val *args, int nargs,
                   hx_val *result, hx_arena *arena, uint8_t role)
{
	int alen = args[0].len, blen = args[1].len;
	int outlen = alen > blen ? alen : blen;
	char *out;
	int i;
	(void)self; (void)nargs; (void)role;

	if (outlen == 0 || alen == 0 || blen == 0) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}
	out = hx_arena_alloc(arena, outlen + 1);
	for (i = 0; i < outlen; i++)
		out[i] = args[0].data[i % alen] & args[1].data[i % blen];
	out[outlen] = '\0';
	result->data = out;
	result->len  = outlen;
}

static void fn_or(hx_func_entry *self, hx_val *args, int nargs,
                  hx_val *result, hx_arena *arena, uint8_t role)
{
	int alen = args[0].len, blen = args[1].len;
	int outlen = alen > blen ? alen : blen;
	char *out;
	int i;
	(void)self; (void)nargs; (void)role;

	if (outlen == 0 || alen == 0 || blen == 0) {
		result->data = hx_arena_alloc(arena, 1);
		result->data[0] = '\0';
		result->len = 0;
		return;
	}
	out = hx_arena_alloc(arena, outlen + 1);
	for (i = 0; i < outlen; i++)
		out[i] = args[0].data[i % alen] | args[1].data[i % blen];
	out[outlen] = '\0';
	result->data = out;
	result->len  = outlen;
}

/* ================================================================
 * Registry
 * ================================================================ */

/*
 * Role-capability presets:
 *   ROLES_DIGEST = BIN|HEX|B64       — digests, HMAC, KDF (raw-bytes output)
 *   ROLES_NONE   = 0                 — string transforms, encodings (no suffix
 *                                      applies; bare name only)
 *   ROLES_HEXBIN = BIN               — `hex` (special: identity for _bin)
 *   ROLES_CRYPT  = BIN|HEX|B64|MCF   — bcrypt, yescrypt (full MCF available)
 */
#define ROLES_DIGEST  (ROLE_CAP_BIN | ROLE_CAP_HEX | ROLE_CAP_B64)
#define ROLES_NONE    (0)
#define ROLES_HEXBIN  (ROLE_CAP_BIN)
#define ROLES_CRYPT   (ROLE_CAP_BIN | ROLE_CAP_HEX | ROLE_CAP_B64 | ROLE_CAP_MCF)

hx_func_entry hx_func_table[HX_MAX_FUNCS] = {
#ifdef HX_STANDALONE
	{ "md5",           fn_md5,           16, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "md4",           fn_md4,           16, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "sha1",          fn_sha1,          20, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "sha256",        fn_sha256,        32, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "sha512",        fn_sha512,        64, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "hmac_sha1",     fn_hmac_sha1,     20, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "hmac_sha256",   fn_hmac_sha256,   32, NULL, 0, ROLES_DIGEST, ROLE_HEX },
#endif
	{ "hex",           fn_hex,            0, NULL, 0, ROLES_HEXBIN, ROLE_HEX },
	{ "upper",         fn_upper,          0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "lower",         fn_lower,          0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "cut",           fn_cut,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "pad",           fn_pad,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "trunc",         fn_trunc,          0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "rev",           fn_rev,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "bswap32",       fn_bswap32,        0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "rotate",        fn_rotate,         0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "cap",           fn_cap,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "rot13",         fn_rot13,          0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "base64",        fn_base64,         0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "frombase64",    fn_frombase64,     0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "fromhex",       fn_fromhex,        0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "wperm",         fn_wperm,          0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "add",           fn_add,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "sub",           fn_sub,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "mul",           fn_mul,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "mod",           fn_mod,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "phpass_encode", fn_phpass_encode,  0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "progress_encode", fn_progress_encode, 0, NULL, 0, ROLES_NONE, ROLE_DEFAULT },
	{ "juniper_encode",  fn_juniper_encode,  0, NULL, 0, ROLES_NONE, ROLE_DEFAULT },
	{ "cisco_pix_encode", fn_cisco_pix_encode, 0, NULL, 0, ROLES_NONE, ROLE_DEFAULT },
	{ "ebcdic",        fn_ebcdic,         0, NULL, 0, ROLES_NONE, ROLE_DEFAULT },
	{ "racf_encrypt",  fn_racf_encrypt,   0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "domino5",       fn_domino5,        0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "lotus64_encode",fn_lotus64_encode, 0, NULL, 0, ROLES_NONE, ROLE_DEFAULT },
	{ "domino6",       fn_domino6,        0, NULL, 0, ROLES_NONE, ROLE_DEFAULT },
	{ "oracle7",       fn_oracle7,        0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "axcrypt",       fn_axcrypt,        0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "sap_bcode",     fn_sap_bcode,      0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "as400_des",     fn_as400_des,      0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "racf_kdfaes",   fn_racf_kdfaes,    0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "aes_unwrap",    fn_aes_unwrap,     0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "besder_encode", fn_besder_encode,  0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "dahua_encode",  fn_dahua_encode,   0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "aes_ecb_encrypt", fn_aes_ecb_encrypt, 0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "aes_cbc_encrypt", fn_aes_cbc_encrypt, 0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "aes_cbc_decrypt", fn_aes_cbc_decrypt, 0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "des",           fn_des,            0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "des_block",     fn_des_block,      0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "des3",          fn_des3,           0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "sha512crypt_encode", fn_sha512crypt_encode, 0, NULL, 0, ROLES_NONE, ROLE_DEFAULT },
	{ "bcrypt_encode", fn_bcrypt_encode,  0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "utf16le",       fn_utf16le,        0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "utf16be",       fn_utf16be,        0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "utf7",          fn_utf7,           0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "zext16",        fn_zext16,         0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "from_cp1252",   fn_from_cp1252,    0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "from_cp1251",   fn_from_cp1251,    0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "length",        fn_length,         0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "xor",           fn_xor,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "emit",          fn_emit,           0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "and",           fn_and,            0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	{ "or",            fn_or,             0, NULL, 0, ROLES_NONE,   ROLE_DEFAULT },
	/* PBKDF2 (always available via OpenSSL) */
	{ "pbkdf2_sha1",   fn_pbkdf2_sha1,    0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "pbkdf2_sha256", fn_pbkdf2_sha256,  0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "pbkdf2_sha512", fn_pbkdf2_sha512,  0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "pbkdf2_md5",    fn_pbkdf2_md5,     0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
#ifndef HX_STANDALONE
	/* KDF built-ins (hashpipe-only) */
	{ "bcrypt",        fn_bcrypt,         0, NULL, 0, ROLES_CRYPT,  ROLE_MCF },
	{ "scrypt",        fn_scrypt,         0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "argon2id",      fn_argon2id,       0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "argon2i",       fn_argon2i,        0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "argon2d",       fn_argon2d,        0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "yescrypt",      fn_yescrypt,       0, NULL, 0, ROLES_CRYPT,  ROLE_MCF },
	/* crypt-family built-ins */
	{ "md5crypt",      fn_md5crypt,       0, NULL, 0, ROLES_CRYPT,  ROLE_MCF },
	{ "apr1",          fn_apr1,           0, NULL, 0, ROLES_CRYPT,  ROLE_MCF },
	{ "sha256crypt",   fn_sha256crypt,    0, NULL, 0, ROLES_CRYPT,  ROLE_MCF },
	{ "sha512crypt",   fn_sha512crypt,    0, NULL, 0, ROLES_CRYPT,  ROLE_MCF },
	{ "descrypt",      fn_descrypt,       0, NULL, 0, ROLES_CRYPT,  ROLE_MCF },
	{ "phpass",        fn_phpass,         0, NULL, 0, ROLES_CRYPT,  ROLE_MCF },
	{ "argon2",        fn_argon2,         0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "pomelo",        fn_pomelo,         0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "rc4_hmac_md5",  fn_rc4_hmac_md5,   0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "aes128_cts_hmac_sha1", fn_aes128_cts_hmac_sha1, 0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "aes256_cts_hmac_sha1", fn_aes256_cts_hmac_sha1, 0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "sm3crypt",      fn_sm3crypt,       0, NULL, 0, ROLES_CRYPT,  ROLE_MCF },
	{ "gost12_512crypt", fn_gost12_512crypt, 0, NULL, 0, ROLES_CRYPT, ROLE_MCF },
#endif
	/* Always-available functions (no hashpipe dependency) */
	{ "siphash",       fn_siphash,        8, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "murmur3",       fn_murmur3,        4, NULL, 0, ROLES_DIGEST, ROLE_HEX },
	{ "pbkdf1_sha1",   fn_pbkdf1_sha1,    0, NULL, 0, ROLES_DIGEST, ROLE_HEX },
};

/* Count initial static entries */
#ifdef HX_STANDALONE
int hx_func_count = 7 + 38 + 21 + 3;  /* 7 hash + 38 common (incl des_block, bswap32) + 21 custom + 3 always */
#else
int hx_func_count = 38 + 21 + 19 + 3;  /* 38 common (incl bswap32) + 21 custom + 6 kdf + 13 crypt/krb + 3 always */
#endif

hx_func_entry *hx_func_lookup(const char *name)
{
	int i;
	for (i = 0; i < hx_func_count; i++)
		if (hx_func_table[i].name &&
		    strcmp(hx_func_table[i].name, name) == 0)
			return &hx_func_table[i];
	return NULL;
}

void hx_func_register(const char *name, hx_hashfn fn, int max_out,
                       hx_bridge_fn bridge, int bridge_bytes,
                       uint8_t supported_roles, uint8_t default_role)
{
	if (hx_func_count >= HX_MAX_FUNCS) {
		fprintf(stderr, "hx: function registry full\n");
		return;
	}
	/* don't register duplicates */
	if (hx_func_lookup(name))
		return;
	hx_func_table[hx_func_count].name            = strdup(name);
	hx_func_table[hx_func_count].fn              = fn;
	hx_func_table[hx_func_count].max_out         = max_out;
	hx_func_table[hx_func_count].bridge          = bridge;
	hx_func_table[hx_func_count].bridge_bytes    = bridge_bytes;
	hx_func_table[hx_func_count].supported_roles = supported_roles;
	hx_func_table[hx_func_count].default_role    = default_role;
	hx_func_count++;
}
