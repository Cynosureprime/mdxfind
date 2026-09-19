# mdxfind v1.596: DESCRYPT on the GPU is 10.8x faster, and GPU brute force emits its hits again

Source: mdxfind.c 1.590 -> 1.596, gpu/gpu_opencl.c 1.215 -> 1.216, gpu_metal.m 1.140 -> 1.141,
gpu/gpujob_opencl.c 1.164 -> 1.165, gpu/gpu_descrypt_core.cl 1.1 -> 1.2,
gpu/gpu_template.cl 1.22 -> 1.24, gpu/gpu_common.cl 1.36 -> 1.37,
gpu/metal_descrypt_core.metal -> 1.3, gpu/metal_template.metal 1.3 -> 1.13,
gpu/metal_common.metal 1.38 -> 1.39, bench_rates.h -> 1.11.

## Read this first if you use GPU brute force

**Brute-force hits were computed and then discarded before emission.** A run reported a
non-zero hit count and a correct found-count summary while emitting no cracked lines at
all. The chunk producer set the synthetic job's `packed_pos` to 1, but the plaintext length
prefix is a 2-byte little-endian header, so the hit replay bounds check evaluated
`0 + 2 + 0 > 1` and rejected every hit unconditionally.

This affects **every algorithm** whose brute-force chunks reach the GPU rules-engine
replay, not only DESCRYPT. Confirmed on `e1` MD5 and `e500` DESCRYPT: 0 emitted lines
against 2 counted hits before, 2 lines after.

It is not a regression from recent work. The producer was written when the header was a
single byte and was not revisited when the header widened.

If an earlier version gave you a non-zero hit count with no cracked lines, that is this
bug, and the candidates are recoverable by re-running. Validated on a GTX 1080: GPU
brute-force output now equals the `-G none` CPU oracle for both types with byte-identical
plaintexts, and the non-brute-force paths are unchanged against the pre-fix binary.

## DESCRYPT `e500` on the GPU: 10.79x on a GTX 1080, and 1.40x ahead of hashcat

Three increments to the existing scalar carrier. No new kernel and no new family:

1. The SP tables staged into local and threadgroup memory, as one shared 2 KB workgroup
   copy rather than a per-lane writable slab.
2. A nibble-table key setup built on the device from the same `DESCRYPT_pc2` walk the bit
   loop used.
3. The key schedule hoisted out of the salt loop, where it had been recomputed for every
   one of 4,096 salts.

**The two backends ship at different levels, because the measurements disagree about which
increment pays and the reason is structural.**

OpenCL ships level 1. Increment 1 alone takes a GTX 1080 from 13.5 to 146.1 M crypt per
second at 3,471 salts, a factor of 10.79, and 1.40x ahead of `hashcat -a 0` measured on the
same non-cracking fixture in the same hour at 104.6. Increment 2 is neutral there, and
increment 3 costs 75.9 percent at 16 salts because it divides the NDRange by the salt batch.

Metal ships level 3 with the key tables off. Increment 1 gives only 10 to 11 percent,
because Apple has no constant-cache broadcast penalty to relieve; increment 2 is a loss;
increment 3 is the whole win, at plus 32.8 percent on an M1 and plus 44.2 percent on an
M2 Max. The Metal generic-family grid is `num_words` alone, with rule, mask and salt as
inner loops, so the hoist removes work without removing a grid axis. Both Apple GPUs
independently agree level 3 is fastest at every salt count, so no per-GPU selection is
warranted.

Net shipped gain over the previous release at 3,471 salts: **10.79x on the GTX 1080, 35.8
percent on an M1, 42.3 percent on an M2 Max.**

Every knob here is compile-time. No environment variable is read, consistent with v1.590.

**Correctness was validated against an implementation by different authors**, glibc and
Darwin `crypt(3)`, rather than against mdxfind's own `crypt-des.c`: 8 passwords by all
4,096 salts, 32,768 of 32,768 recovered, zero differences on both backends and zero CPU
versus GPU divergence, plus CPU equals GPU on a real 47,366-hash list. Byte identity of the
shared templates was proven rather than assumed, by preprocessing 111 OpenCL and 162 Metal
program variants against the prior revision with zero lines differing.

Two assumptions were measured wrong and are recorded rather than quietly dropped: the key
schedule does not become 25 to 35 percent of the kernel once the tables move, and increment
1 returns no constant-bank headroom, because the constant table is the source the local
copy initialises from.

## A crash: out-of-bounds access in the DES-crypt salt compaction

The four DES-crypt salt-compaction loops (MD4DESCRYPT, MD5DESCRYPT, DESCRYPT, BSDICRYPT)
each decremented the index inside a `while` whose condition re-reads that same slot, so
removing the salt at index 0 left the index at -1 and the condition dereferenced element
-1. Confirmed SIGSEGV, with the fault on the condition itself.

A null dereference was the lucky outcome: the loop body writes through that pointer, so a
heap layout where those bytes read as a valid writable pointer gives a wild write instead
of a stop.

It is reachable through `-F` with a DES type and a plain-hex type both selected, which is
what a broad sweep over a mixed list does. Of eleven validation cases, nine segfault before
and pass after; the two that pass on both are the controls. Coverage includes salt lengths
1, 2, 3, 5, 8, 9, 255, 300, 4096 and 5000, `$HEX[]` salts that decode to a different length
than written, and several hashes sharing one salt, which also proves the compaction does not
retire a shared salt early.

## Brute-force progress could exceed 100 percent

Display only. Nothing computed changes. For an iterated type the GPU accounting has already
multiplied by iterations times salts, while the display divided by the live salt count
alone, leaving the iteration factor in.

Measured on a 10M 7-digit mask over 3,471 DESCRYPT salts, the previous build printed
157.3M of 10.0M at 1572.9 percent and 209.7M at 2097.2 percent, with the sample candidate
clamped to the final mask value and the ETA reporting "done", while the run was 62.9 and
83.9 percent complete. The ratio is exactly the 25 rounds of Unix DES crypt. Three
symptoms, one cause: the percentage, the clamped position sample and the false completion
signal all follow from progress exceeding the keyspace.

The same runs now print 62.9 and 83.9 percent with real positions and a falling ETA. Types
whose round count is encoded per salt (bcrypt cost, the phpBB3 iteration character, and a
`rounds=` prefix on `$5$` and `$6$`) cannot have their divisor reconstructed at that site,
so their percentage is capped at 100 and the ETA prints "unknown" rather than naming a
wrong time. Non-iterated types are unaffected, and MD5 brute force was already correct.

## Two hashcat mode mappings pointed at the no-equivalent sentinel

- **13900 OpenCart** now selects `e438 SHA1SALTSHA1SALTSHA1PASS`, which is
  `sha1(salt . sha1(salt . sha1(pass)))` and has been implemented all along. This is not
  cosmetic: 65535 in that field means "no equivalent" and the `-M` selection loop assigned
  it straight through, so `-M 13900` was selecting a nonexistent operation.
- **15000 FileZilla Server** now selects `e386 SHA512PASSSALT`, which is
  `sha512(pass . salt)`. `e386` keeps its existing 1710 mapping and answers to both, which
  the table already does elsewhere.

Both were verified against hashcat's own published vectors rather than self-generated ones,
with a wrong-password control. No algorithm is added or altered, so no stored result
changes.

They were found by auditing all 70 sentinel-mapped modes against hashcat's per-module test
vectors. For the record, that audit produced one real mis-mapping, one false positive
(1Password agilekeychain, where a container split on colons made the iteration count look
like a hash, and which still "verified" under a deliberately wrong password and was
therefore rejected), and 68 genuine gaps, being container, archive, cipher and document
formats such as WPA, the TrueCrypt and VeraCrypt families, MS Office, PDF and the wallet
formats. A wrong-password control is mandatory on this kind of sweep; without it the
1Password artefact reads as a find.

## Two new types: `e1028 CRYPTOPPLEGACY` and `e1029 CRYPTOPPDEFAULT`

Crypto++ `DataEncryptor` stored forms. **These are key recovery, not digest comparison:**
the stored record is salt, then a key check, then ciphertext, so judging one means
decrypting it, and the recovered plaintext is the encrypted body.

The site key is an ordinary salt whose cardinality happens to be one, supplied with `-s` or
`-S` like any other salt. A found line is therefore self-contained:

    TYPE <record>:<sitekey>:<plaintext>

which is the same shape every salted type already emits, so `mdxfind | hashpipe` works
unchanged and `mdsplit` and `getpass` need no new handling. Records load structurally
through `-F`.

`e1028` is the 1998-era form, DES-EDE2-CBC keyed by a SHA-1 mash, and is what `Dynu.dll`
writes. `e1029` is the modern SHA-256 variant, verified bidirectionally against an upstream
Crypto++ build. The two are separate types because they are different constructions, and a
hit on one must never be read as a record from the other's application.

## Also in this release

- `Totrules_gpu` over-counted brute-force chunks by the chunk ratio, because the accounting
  multiplier used the whole keyspace where a chunk covers only part of it. That value feeds
  the periodic rate line and a final summary line, so it was not cosmetic. It is inert with
  no rule file, and the no-rule brute-force case was verified unchanged.
- Benchmark rate entries for the two new types, measured natively rather than scaled.
