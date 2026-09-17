/*
 * classify_utf8 -- classify a byte string as ASCII-only, valid UTF-8, or
 * high-bit-but-not-UTF-8.  Self-contained; no dependencies beyond <stddef.h>.
 *
 * $Revision: $
 * $Log: $
 */
#ifndef CLASSIFY_UTF8_H
#define CLASSIFY_UTF8_H

#include <stddef.h>

/* Returned bits.  HI and UTF8 are deliberately bits 0 and 1 so that the
 * "use the UTF-32 engine" test is a single mask-and-compare:
 *
 *     if ((res & U8C_WIDE) == U8C_WIDE)  -> UTF-32 path
 *     else                               -> byte path
 *
 * Everything that is not provably wide text -- empty input, ASCII, and
 * high-bit-but-invalid -- falls to the byte path WITHOUT enumeration, which is
 * what makes U8C_NONE safe rather than a trap.  Never write dispatch as a
 * negative test on U8C_ASCII. */
#define U8C_NONE   0x00u   /* no bytes examined: n == 0, or p == NULL        */
#define U8C_HI     0x01u   /* at least one byte >= 0x80                      */
#define U8C_UTF8   0x02u   /* structurally valid UTF-8 throughout            */
#define U8C_ASCII  0x04u   /* every byte < 0x80 (implies !U8C_HI)            */
#define U8C_WIDE   (U8C_HI | U8C_UTF8)   /* the dispatch predicate           */

/* Bits 3-7 are reserved for future boundary encodings (e.g. CP1252-decodable).
 * A caller must not assume unknown bits are zero in future versions. */

/*
 * Classify p[0 .. n-1].
 *
 * Returns a bitwise OR of the U8C_* values above.  Exactly one of U8C_ASCII or
 * U8C_HI is set for n > 0; U8C_UTF8 is set only in company with U8C_HI.
 *
 *   n == 0, or p == NULL  ->  U8C_NONE (0).  An empty string is vacuously both
 *                            ASCII and valid UTF-8, but it carries no evidence,
 *                            so no flag is asserted.  Callers using the
 *                            U8C_WIDE test above handle this correctly for
 *                            free; see the note on that macro.
 *
 * "Valid UTF-8" here means, per the definition this was built to:
 *   - not ASCII-only, AND
 *   - at least one valid multibyte sequence, AND
 *   - EXACTLY ZERO invalid sequences.
 * The third clause is why the walk cannot stop at the first good sequence: a
 * string carrying good multibyte text and one corrupt byte is NOT UTF-8.
 *
 * Rejects, as RFC 3629 requires: overlong forms (C0/C1 leads, E0 80..9F,
 * F0 80..8F), UTF-16 surrogates encoded as UTF-8 (ED A0..BF), scalars above
 * U+10FFFF (F4 90.. and F5..FF), truncated sequences, and stray continuation
 * bytes.  An embedded NUL is an ordinary ASCII byte and is NOT a terminator --
 * the length governs.
 */
unsigned classify_utf8(const void *p, size_t n);

/* Same contract, but REQUIRES at least 16 readable bytes at p + n.  Inside
 * mdxfind that holds by construction for every candidate word; for any other
 * caller it does not, and violating it is undefined (ASan will abort).  On
 * targets with no padded win this is an alias for classify_utf8. */
unsigned classify_utf8_padded(const void *p, size_t n);

#endif /* CLASSIFY_UTF8_H */
