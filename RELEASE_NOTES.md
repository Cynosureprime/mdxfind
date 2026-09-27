# mdxfind v1.606: seventeen new hash types, and rules combined with a mask

Source: mdxfind.c 1.596 -> 1.606.

## New hash types (e1030 - e1046)

| index | name | construction |
|---|---|---|
| e1030 | `MD5BASE64MD5SHA1` | `md5(base64(md5(sha1(pass))))` |
| e1031 | `WRLSHA1` | `wrl(sha1(pass))` |
| e1032 | `MD5sub8-24MD5sub8-24MD5MD5MD5` | `md5(cut(md5(cut(md5(md5(md5(pass))), 8, 16)), 8, 16))` |
| e1033 | `MD5SHA1SHA1MD5SHA1MD5` | `md5(sha1(sha1(md5(sha1(md5(pass))))))` |
| e1034 | `MD5SHA1SHA1SHA1` | `md5(sha1(sha1(sha1(pass))))` |
| e1035 | `MD5SHA1MD5SHA1MD5SHA1` | `md5(sha1(md5(sha1(md5(sha1(pass))))))` |
| e1036 | `MD5SHA512MD5` | `md5(sha512(md5(pass)))` |
| e1037 | `MD5sub1-16MD5` | `md5(cut(md5(pass), 0, 16))` |
| e1038 | `MD5sub1-28MD5` | `md5(cut(md5(pass), 0, 28))` |
| e1039 | `MD5MD5sub1-30MD5` | `md5(md5(cut(md5(pass), 0, 30)))` |
| e1040 | `APACHE-SHA-TRUNC16` | `"{SHA}" . base64(trunc(sha1_bin(pass), 16))` |
| e1041 | `MD5SALTLAST16` | `cut(md5(md5(pass) . salt), -16)` |
| e1042 | `MD5SALTMD5PASS-PASS` | `md5(salt . md5(pass) . ":" . pass)` |
| e1043 | `MD5-1xMD5SHA1pSHA1p` | `md5(md5(sha1(pass)) . sha1(pass))` |
| e1044 | `MD5-1xMD5SHA256pSHA256p` | `md5(md5(sha256(pass)) . sha256(pass))` |
| e1045 | `MD5-1xMD5SHA512pSHA512p` | `md5(md5(sha512(pass)) . sha512(pass))` |
| e1046 | `MD5-1xMD5MD5pMD5p` | `md5(md5(md5(pass)) . md5(pass))` |

All seventeen are unsalted or single-salt constructions with no hashcat mode.
Each is catalogued in `hx.8` and covered by the `hx_dedup_check` gate, and each
was verified against an independently supplied hash before implementation.

Three carry a stored form: `APACHE-SHA-TRUNC16` is the RFC 2307 `{SHA}` scheme
of e457 with a 16-byte payload instead of 20 and is distinguished from it by
decoded payload length; `MD5SALTLAST16` stores only the last 16 hex of its
digest, so it is single-depth and `-i` does not iterate it; `MD5SALTMD5PASS-PASS`
carries its site prefix as the salt rather than a literal, so one type covers
every installation.

## Rules combined with a mask lost candidates

`-r` together with `-n`/`-N` silently dropped work on large jobs. The mask
fan-out sized the mask axis alone while the host computed in `size_t`, so a
job whose product of words, rules and mask size exceeded 2^32 wrapped, and the
chunking that avoided the overflow restarted the mask cursor at the wrong
position on each rule advance. A production case went from 1 of 23 recovered to
23 of 23, matching hashcat exactly on both digests and plaintexts.

## Brute-force mask fixes

`?b` generated one candidate per position instead of 256. Separately, the
brute-force bootstrap could deadlock when the wait counted enumerated devices
rather than dispatching ones.
