# 7-Zip Archives (7zAES) — `-m e1000`

mdxfind type `e1000` recovers passwords for AES-256 encrypted `.7z` archives. It maps to
hashcat mode `11600` and consumes the same `$7z$` hash format that `7z2john.pl` produces.

It exists because the other tools do not merely struggle with some 7-Zip archives — they
report **"exhausted, 0 cracked" while the correct password sits in the wordlist**, with no
error and no warning. That failure is silent, it is not rare, and nothing in the hash line
reveals it. This document explains the trap, how `e1000` sidesteps it, and how to tell a
real negative from a manufactured one.

## Why this type is important

### The silent false negative

Both hashcat and john verify a 7-Zip candidate the obvious way: derive the key, decrypt,
**decompress**, then compare CRC32. That requires a decompressor for whatever codec the
archive used, and it requires knowing which codec that was.

The hash line is supposed to say. It frequently does not.

`7z2john.pl` recognises a fixed table of codec IDs. Anything outside that table leaves its
internal `type_of_compression` at the initial value, and the archive is emitted as
**`type 0`, meaning "no decompression needed" — indistinguishable from a genuinely stored
archive**. A tool that trusts the field then CRC32s data that is still compressed. The
comparison fails for every candidate, forever, and the run ends looking exactly like an
honest exhaustion.

Deflate64 is the common case: its codec ID is one digit away from Deflate's, and it is not
in the table. A controlled test — same file, same password, same two-word wordlist
containing that password, only the codec differing:

| archive | codec | hashcat `-m 11600` | john `--format=7z` | mdxfind `-m e1000` |
|---|---|---|---|---|
| `lzma.7z` | LZMA | Cracked 1/1 | Cracked 1/1 | Cracked 1/1 |
| `d64.7z` | Deflate64 | **Exhausted 0/1** | **0 cracked, 1 left** | **Cracked 1/1** |

The same erasure happens on the *filter* axis. The ARM64 and Delta filters have no codec ID
in the table either, so an `ARM64 + LZMA2` archive is emitted as plain `type 2`. Run the
full verification path by hand with the **correct** password:

```
no filter      crc_len=154352  stored=a9394311  computed=a9394311  MATCH
ARM64 filter   crc_len=154352  stored=a9394311  computed=fcba50f3  MISMATCH
```

LZMA2 decompresses cleanly in both cases. The CRC fails only because the inverse ARM64
filter is never applied — and the hash line gives no hint that a filter exists at all.

The rule to carry away:

> Any codec or filter the extractor does not recognise is silently erased from the hash
> line. Every tool that trusts the type field then fails on that archive with no
> diagnostic.

### What e1000 does instead

7-Zip zero-pads its AES stream up to a 16-byte boundary. So the last `packedlen −
unpackedlen` bytes of the *decrypted* stream must be zero. Checking that costs **one
16-byte CBC decrypt** — the final ciphertext block, using the preceding block as its IV.

No decompression. No codec support. No knowledge of the filter chain. That is what makes
`e1000` work on archives the other tools cannot verify at all, and it is why it is
immune to the misclassification above.

The cost is that the check is only as strong as the padding is long, which is the subject
of the rest of this document.

## The hash format

```
$7z$ type $ log2(iter) $ saltlen $ salt $ ivlen $ iv $ CRC32 $ packedlen $ unpackedlen $ data [ $ crc_len $ coder_attributes ]
      0         1           2        3       4      5     6        7            8         9         10           11
```

Field indices are counted after the `$7z$` prefix, so the count is **10** without the
trailing pair and **12** with it.

### The type field

It does not name the codec directly, and reading it as if it did is the source of the
problem above.

| value | meaning | trailing pair |
|---|---|---|
| `0` | no decompression needed to check the CRC32 — genuinely stored, **or** a codec the extractor failed to recognise | absent |
| `1..127` | low nibble = decompressor, high nibble = preprocessor | **present** |
| `128` | data truncated to its final block; the padding check is the only possible verification | absent |

Low nibble (`type & 0x0f`):

| 1 | 2 | 3 | 6 | 7 |
|---|---|---|---|---|
| LZMA1 | LZMA2 | PPMd | BZip2 | Deflate |

High nibble (`(type >> 4) & 0x07`):

| 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| BCJ (x86) | BCJ2 | PPC | IA64 | ARM | ARMT | SPARC |

Not representable, and therefore silently erased: **Deflate64, ARM64, RISCV, Delta,
Swap2/4**, and anything added to 7-Zip later.

For `type 128`, the extractor moves the *penultimate* ciphertext block into the `iv` field
and leaves only the final block in `data`. mdxfind reads `iv || data` as the 32-byte tail.

### Key derivation

```
if log2_iter == 0x3F:                 # 63 — the "no hashing" special case
    key = (salt || password_utf16le) zero-padded / truncated to 32 bytes
else:
    ctx = SHA256_init()
    for i in 0 .. (1 << log2_iter) - 1:
        SHA256_update(ctx, salt)                  # usually empty
        SHA256_update(ctx, password_utf16le)
        SHA256_update(ctx, uint64_le(i))
    key = SHA256_final(ctx)                       # 32 bytes, AES-256
```

Two things trip implementations. It is **one streaming SHA-256 context updated `2^log2`
times and finalized once**, not `2^log2` successive hashes. And the password is
**UTF-16LE**, so a non-ASCII password is unreachable to an ASCII-only implementation.
mdxfind converts through iconv and reports non-ASCII plaintext `$HEX[]`-encoded.

Because the KDF consumes only `(salt, log2_iter)` and the password, every archive sharing
those derives the *same* key. 7-Zip writes `saltlen = 0`, so in practice all archives in a
run collapse into one group and the `2^19`-round derivation is paid **once per candidate**
however many archives are loaded. Tools that treat them as distinct salts pay it per
archive — measured on a GTX 1080, five archives in one file ran at 25.6 candidates/s
against 129 candidates/s for a single archive. mdxfind gets all five at the single-archive
rate.

## Verification tiers

A hit is not a boolean. mdxfind accumulates evidence and reports how much it has.

| tier | check | cost | strength | needs |
|---|---|---|---|---|
| 0 | trailing zero-pad | 1 AES block | `8 × padsize` bits, 0–120 | the final two ciphertext blocks |
| 1 | head-block structure | 1 AES block | 8–112 bits | the **first** 32 bytes of `data` |
| 2 | decompress + CRC32 | whole stream | decisive | the complete stream |
| 3 | `7zz t -p<password>` | a process | definitive | the archive itself |

Tiers 1 and 2 run only after tier 0 passes — about once per `2^(8×padsize)` candidates —
so they cost nothing against the KDF.

### Tier 0 and the padsize lottery

`padsize` is `compressed_size mod 16`, which is uniformly distributed. Over 240 generated
archives:

```
padsize 0      ~6%    no padding at all — tier 0 cannot decide
padsize 1..3   ~19%   8 to 24 bits — usable but noisy
padsize >= 4   ~75%   32 bits or better
```

At `padsize 1` the check is **8 bits wide**, so roughly one candidate in 256 passes it.
A thousand-word run against such an archive yields about four "cracks", none of them real.
This is the single most important thing to understand about the type, and mdxfind now says
so at load time rather than letting you discover it in the results.

### Tier 1

The first decrypted block, checked against what the type promises. The strongest case is a
header-encrypted (`-mhe=on`) archive, whose plaintext *is* a raw 7z header — measured
across Copy, LZMA2, Deflate64 and BZip2 archives, 14 of the first 16 bytes are constant.
mdxfind checks only the structural prefix (`0x01` kHeader followed by kMainStreamsInfo or
kFilesInfo), which stays correct for archive shapes not sampled. Also checked: the LZMA1
range-coder init byte, the LZMA2 control byte, the BZip2 `BZh` signature, and the Deflate
BTYPE field.

Tier 1 is the only evidence available when `padsize` is 0, and it requires the head of the
stream — see the note on truncation below.

### Tier 2 and the codec oracle

Decrypt the whole stream, decompress, compare CRC32. Because the type field lies, the
codec is **brute-forced rather than trusted**: stored, LZMA2, LZMA1, Deflate and BZip2 are
each tried, and each is also tried under the inverse ARM64 and Delta filters. A wrong key
fails every decompressor's header validation in microseconds, so the search is cheap.

A tier-2 **match** confirms the password outright. A tier-2 **miss** is treated as a
refutation only when the line names a codec mdxfind implements and declares no
preprocessor; otherwise it is "no opinion", because a miss cannot be distinguished from a
codec that cannot be decompressed.

## Load-time triage

Everything needed to decide whether an archive can *ever* be solved reliably is in the hash
line, so mdxfind decides it while loading — before spending a single KDF.

Rejected outright, each with a reason on stderr:

| condition | why |
|---|---|
| field count is not 10 or 12 | malformed |
| `saltlen` / `ivlen` disagree with their fields | malformed |
| non-hex salt, iv or data | malformed |
| `log2 > 40` | `2^N` iterations unreachable |
| `log2 == 63` | the special KDF is not implemented |
| `padsize` outside 0..16 | malformed |
| fewer than two ciphertext blocks, and the stream is not complete | nothing to check |
| **line exceeds the input line limit** | it arrived truncated, so its trailing bytes are not the stream's — the pad check would run against the wrong ciphertext and could never pass |
| `padsize 0` with no complete stream | undecidable by any implemented tier |

Warned about but still searched:

```
7ZIP: padsize 1 gives only 8 bits of evidence; expect ~1 false hit per 2^8 candidates.
Supply the complete stream (do not run tools/7z2mdx.py on a line that already fits) to
enable decompress+CRC verification.
```

At the end of loading:

```
7ZIP triage: 12 loaded (12 verifiable by decompress+CRC, 0 pad-check only), 3 rejected, 0 below 32 bits
```

Rejected archives are **excluded from the "Searching through N unique 7ZIP archives"
count**, so that figure only ever describes work that could actually produce a result. If
everything was rejected, mdxfind says so instead of running a pointless search:

```
7ZIP: all 1 archive(s) were rejected at load; nothing to search
```

Per-line reject messages are capped at ten, followed by a suppression note; the summary is
always printed in full.

### Tuning

| variable | default | effect |
|---|---|---|
| `MDXFIND_7Z_MIN_BITS` | `32` | evidence below this is warned about |
| `MDXFIND_7Z_STRICT` | unset | reject rather than warn below the threshold |

`32` keeps the expected false-positive count below one for any run up to 4.3 billion
candidates.

## Support tools

### `7z2john.pl` — the extractor

Ships with john the ripper. It reads `.7z` files (including split `.7z.001` sets and
non-packed `.sfx`) and emits the `$7z$` line. Strip the leading `filename:` it prefixes:

```sh
7z2john.pl archive.7z | sed 's/^[^:]*://' > archive.hash
```

Read its warnings. It will tell you when it believes the archive needs a decompressor the
cracker lacks — though, per the opening section, it cannot warn about a codec it failed to
recognise in the first place, because it does not know it failed.

### `tools/7z2mdx.py` — the fitter

mdxfind reads hash lines into a fixed buffer (40 KB). A `type 0` line is not shortened by
the extractor, so a large archive produces a line far past that, which would arrive
truncated — and a truncated line's trailing bytes are not the stream's trailing bytes, so
the pad check could never pass.

`7z2mdx.py` reduces such a line to the final two ciphertext blocks.

```sh
7z2john.pl archive.7z | 7z2mdx.py > archive.mdx
7z2mdx.py file1.hash file2.hash > combined.mdx      # also accepts filenames
```

It only truncates a line that will not otherwise fit. **A record that fits is passed
through untouched**, because truncating destroys tier 2 — the only thing that can turn a
pad-check hit into a confirmed crack. When it does truncate it says so:

```
NOTE: truncating a 49073-byte line to the final two blocks. This drops decompress+CRC
verification, leaving 40 bits of pad check; expect ~1 false hit per 2^40 candidates.
```

and it refuses outright when truncation would leave nothing usable:

```
skip: padsize 0 and the line needs truncating to fit mdxfind's 40960-byte limit;
nothing would be left to verify with.
```

Do not run it defensively over lines that already fit — that is exactly how a decidable
archive becomes an undecidable one.

### `tools/7z_validate.sh` — the acceptance test

Builds a matrix of archives covering every codec, the branch filters, both
header-encryption modes, a multi-file archive, a single-block archive and padsizes 0–3;
validates each fixture against the native oracle *before* the attack; extracts; cracks with
a short mostly-invalid wordlist; then re-verifies every reported crack with `7zz t`.

```sh
tools/7z_validate.sh [workdir]
```

It exists because **a tool returning zero is not evidence of absence until it has returned
non-zero on a known positive** — which is the whole subject of this document. Run it after
any change to the type.

### `7zz` — the oracle

7-Zip's own binary is the final authority, and the only one that can settle an archive
whose codec mdxfind cannot decompress:

```sh
7zz t -p'<password>' archive.7z      # rc=0 correct, rc=2 wrong
```

Use it to confirm any hit that mdxfind reports as resting on thin evidence:

```
7ZIP: reporting a hit backed by only 8 bits (no decompress+CRC available).
Confirm with: 7zz t -p<password> <archive>
```

## Running it

The hash text contains `$`, so it goes on the **`-F`** channel with **`-M`** for the type,
and the type selector must come **before** the file — `-F` loads at the moment getopt
reaches it, and a selector arriving later means the file is parsed with only the default
type live.

```sh
mdxfind -M e1000 -F archive.mdx wordlist.txt              # correct
mdxfind -F archive.mdx -M e1000 wordlist.txt              # WRONG — loads nothing usable
```

The confirmation to look for is `N 7ZIP hashes read from <file>`.

Mixed with other types in one pass:

```sh
mdxfind -M e1000 -F archives.mdx -m e1,e30 -f hexhashes.txt /path/to/wordlist
```

## Diagnostics

`-z` reports every tier for each candidate, so a wrong KDF or a misread type is
diagnosable without a debugger:

```
7ZIP <hash>:KEY=<64 hex>:LASTBLOCK=<32 hex>:PADSIZE=11:PADOK=1:HEADBITS=16:TIER2=1:CODEC=LZMA2:BITS=104
```

| field | meaning |
|---|---|
| `KEY` | the derived AES-256 key |
| `LASTBLOCK` | the decrypted final ciphertext block |
| `PADSIZE` / `PADOK` | tier 0 |
| `HEADBITS` | tier 1 evidence, or 0 |
| `TIER2` | `1` confirmed, `0` no opinion, `-1` refuted |
| `CODEC` | which combination reproduced the CRC |
| `BITS` | total cheap-tier evidence |

Note that `-z` is a **generate** mode. A line it prints is mdxfind calculating, not
finding.

## Coverage

hashcat's parser accepts only `data_type` in `{0, 1, 2, 7}` and rejects a non-empty salt
outright. john handles more, but its padding check is off by default and must be enabled
with `TrustPadding=Y` in `john.conf`.

| case | hashcat | john | mdxfind `e1000` |
|---|---|---|---|
| PPMd (3), BZip2 (6) | rejects at parse | no decompressor | **cracks** |
| BCJ / BCJ2 / PPC / IA64 / ARM / ARMT / SPARC | rejects at parse | decodes most | **cracks** |
| Deflate64, ARM64, Delta, RISCV | accepted, silently unverifiable | same | **cracks** |
| `saltlen != 0` | rejects | handles | handles |
| stream over 8 MB | rejects | handles | needs only 32 bytes |
| `type 128` (truncated) | **rejects** | handles | **handles** |
| archives sharing a KDF group | one derivation each | one derivation each | **one derivation for all** |

## Limits

- **BCJ2** needs four separate streams that the hash format cannot carry. Tier 0 still
  cracks it; only `7zz t` can confirm it.
- **PPMd** and **true Deflate64** have no tier-2 decompressor yet, so those archives rest
  on tiers 0 and 1. A small Deflate64 stream that never uses the extended window or length
  codes does verify through the Deflate path, but that is luck, not support.
- **BCJ, PPC, IA64, ARM, ARMT and SPARC** inverse filters are not implemented, so filtered
  archives get tier 0 only. ARM64 and Delta are implemented.
- **`log2 == 0x3F`** is rejected rather than cracked.
- An archive that is both `padsize 0` and too large to fit the line buffer cannot be
  attacked at all; mdxfind and `7z2mdx.py` both say so rather than failing quietly.
