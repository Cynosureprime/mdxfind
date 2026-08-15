# mdxfind v1.536 — Four types were computing the wrong hash; one had stopped using the password entirely

Source: mdxfind.c rev 1.534 → 1.536.

Every hash type was cross-checked against hashpipe, and where the two disagreed, against live cracks and an independent implementation. **If you have run `-m e535`, `-m e584`, `-m e603`, `-m e715` or `-m e440`, re-run them** — those types were not computing what they claimed, so a negative result from them means nothing.

## Fixes

**SHA1-CUSTOMUSERSALT (`-m e535`) ignored the password.** It hashed a constant per username, so `password123`, `hunter2` and `abc` all produced the same value. Two lines deriving `sha1(pass)` and `sha1(sha1(pass))` lived inside the Judy-walk preamble that the per-job salt-snapshot conversion replaced wholesale in rev 1.174; the replacement carried the loop but not the computation. Nothing warned, because both variables stayed declared and read — the type went on emitting stable, plausible hex for two and a half years. Restored to its original construction, including the pepper that a separate change had dropped two revisions earlier.

**`-m e584` computed a different type's hash.** The uppercase test omitted `SHA1SHA256UCSHA256SHA256`, so it produced the non-uppercase variant. Anything it found was really `-m e614`. Twenty-one crack files were mislabelled as a result and have been renamed.

**Three types truncated a staged digest.** `prmd5()` writes its hex *and* a NUL at `out[len]`, so staging one digest next to another in the same buffer destroys the first byte of the neighbour. `-m e603` (a regression — it could no longer find hashes it had cracked before), `-m e715`, and `-m e440`, where only the non-SSE branch was affected, meaning x86 and ARM builds had been computing different digests for the same type.

**MD5AM, MD5AM2 and MD5SPECAM emitted a constant.** The SSE batch packs a lane only when the assembled string fits, but the readback tested the *password* length. A lane that took the scalar path was never packed, so reading it back returned MD5 of the zeroed buffer — one fixed value, emitted for every password, that would "crack" any list containing it.

**Salts longer than 255 bytes were silently truncated** rather than rejected, quietly corrupting any type whose salt exceeded the buffer. Salts now use a 4096-byte limit with a heap fallback, and anything beyond it is refused with a warning naming the type rather than mangled. This made a latent stack overflow in the SSHA paths reachable, so those are now bounds-checked.

**Four hashcat modes mapped to the wrong type and could never match.** `-m 3500` pointed at `md5(h . h)` instead of MD5 at iteration 3. `-m 4521` (Redmine) and `-m 4522` (PunBB) pointed at `sha1(sha1(pass) . salt)` — the operands the wrong way round. `-m 6000` ran RIPEMD-128 against RIPEMD-160 hashes. All four verified against hashcat's own published vectors.

**`-m 8600` (DOMINO5) admitted no hashes when selected alone.** It was flagged for the Judy loader, but no loader references it, so the hash table came up empty and the run reported "None found". It only appeared to work when some other type happened to be selected alongside it. The algorithm was never wrong.

Also: SAP BCODE checksum and PASSCODE compare width, BCRYPT256's final round, SYMFONY256's iteration count, and a KRB5PA23 salt read that ran off the front of the array.

## Documentation

`docs/HASH_TYPES.md` now covers all 1001 types, 998 with a verified example. Each example was fed back through hashpipe pinned to its own type, so the table cannot list a value that no longer computes. The three without one are the two evaluation instruments and the MD5SPECAM parsing wrapper, for which no example is possible.

# mdxfind v1.534 — Fixes an 11x dispatch regression in v1.531/v1.532, and gives SHA-256 a hardware path on every architecture

Source: mdxfind.c rev 1.532 → 1.534; mymd5.c rev 1.34 → 1.35; mdxfind.h rev 1.24 → 1.25; sha1_shani.c rev 1.1 → 1.2.

**If you are running v1.531 or v1.532, upgrade.** Both carry a dispatch regression that makes salted types with a GPU path several times slower on large wordlists. Results were always correct — this cost time, not cracks.

## Fixes

**Salted GPU-hybrid types ran up to 11x slower (regression, v1.531).** v1.531 changed how many wordlist lines go into one dispatch, switching the test from a type's raw rate to its cost per word. That was right for CPU-only types — SCRYPT against 1,800 salts is about two words per second and genuinely wants one word per dispatch so the work spreads across cores. It was wrong for anything with a GPU kernel, because `gpu_try_pack` accumulates words and sends them to the device together, and one word per dispatch starves it.

Cost per word cannot tell the two apart. PHPBB3 (`-m e455`, hashcat 400) at 2,931 h/s against 5,000 salts is 0.59 words/sec — *more* expensive per word than SCRYPT — yet it needs the opposite treatment. The test is now whether the type can reach a GPU kernel at all.

Measured on a GTX 1080, 5,000 hashes against 20,000 words: **4.9 s before the regression, 55.9 s with it, 4.9 s restored** — identical crack counts throughout. The SCRYPT fan-out that v1.531 added is preserved and verified at 278% CPU against 99% before it.

The affected set is wider than PHPBB3: any type with a GPU kernel whose bench rate is below its salt count, which at a few thousand salts is most of the salted catalog. The gate now carries a prominent FRAGILE header naming both regression fixtures, since it has failed in opposite directions twice.

## New

**SHA-256 now has a hardware path on x86 as well as ARM.** `mysha256()` is one-shot, so any type needing to hash incrementally had nowhere to go inside mymd5.c and escaped to portable-C fallbacks or to OpenSSL — meaning those types ran at reference speed no matter what the processor could do.

A streaming interface now sits alongside the one-shot: `mysha256_begin/add/end` over a caller-owned fixed-size workspace carved from the existing per-thread buffers. No allocation on any path, nothing to free, and thread-safe by construction. Behind it are three arms chosen by one gate — x86 SHA-NI, ARM crypto extensions, portable C — with the pointer starting NULL so a CPU that lacks the feature degrades to correct-and-slow rather than crashing. `MDXFIND_SHA256_DEBUG=1` reports which arm won.

**`-m e1000` (7-Zip) uses it.** The KDF streams roughly 12 MB per candidate; it previously did so through OpenSSL EVP. End-to-end on an Apple M1: **131 → 646 candidates/sec**. Gains elsewhere depend on how OpenSSL was built for that platform — Windows and macOS x86 builds ship a no-asm OpenSSL and benefit most, while Linux x86-64 already had assembly and is at parity, but is now independent of it. mdxfind no longer calls the OpenSSL EVP interface anywhere.

The x86 routine was validated on an AMD Ryzen against published SHA-256 vectors and differentially against the portable implementation; the same binary was then confirmed to produce identical output on a CPU with the feature and one without.

---

# mdxfind v1.532 — Two silent-miss fixes: WRL found nothing on any OpenCL GPU, and short hashes were dropped at load

Source: mdxfind.c rev 1.531 → 1.532; gpu/gpu_wrl_core.cl rev 1.2 → 1.3; gpu/gpu_wrl_core_str.h rev 1.1 → 1.2.

Both defects in this release were **silent**: the affected hashes were listed in "Working on hash types", the run exited normally, and nothing was reported as wrong. Anyone who ran WRL on a GPU, or a short-digest type alongside a salted type, got a clean "None found" that was not the truth. If either applies to past work, it is worth re-running.

## Fixes

**WRL (`-m e5`) found zero hashes on every OpenCL device.** `gpu_wrl_core.cl` re-defined `WRL_RC`, `WRL_SBOX` and `WRL_OP`, which `gpu_common.cl` has also supplied since 2026-05-27, when the `wrl_block` primitive was lifted in from librhash for the WRLMD5PASS family helper. Both files are concatenated into the same program, so every WRL template build failed with `CL_BUILD_PROGRAM_FAILURE` — "redefinition of `WRL_RC`". The kernel resolver then returned NULL and the work was **not** re-routed to the CPU, so WRL simply found nothing. This was not a loss of acceleration; it was a loss of results, on every OpenCL GPU, for roughly ten weeks. Metal was never affected — its equivalent lift namespaced the symbols as `MTL_WRL_*`.

The duplicate definitions are removed; the tables were verified value-identical first (10 of 10 round constants, 2048 of 2048 S-box entries).

**Hashes shorter than 32 hex characters were silently discarded at load.** `load_hash_file()` has two hex-loading paths. The fast path admits digests of 8 bytes and up; the slow-path fallthrough required 16. Selecting any salted or user-bearing type together with `-F` forces the slow path, so an 8-byte digest — MYSQL3 (`-m e456`) is the type in the shipped catalog — was dropped with no warning and no count whenever it shared a run with, say, MD5PASSSALT. Alone it worked; in company it vanished.

The slow-path gate now matches the fast path and the storage layer beneath it. Eight bytes is the true floor: the compact-table key is a single `uint64_t` read off the hash, `addhash()` rejects anything shorter, and sub-16-byte entries are zero-padded to 16 so the GPU's 4x`uint32` probe cannot read into the next entry.

## Validation

On an NVIDIA GTX 1080, kernel cache cleared so every kernel JIT-compiled fresh: `-h ALL` over a fixture generated by `-z` for all 22 types in that set now loads 330 hashes and finds **22 of 22 types, 15 of 15 each**, with GPU and CPU output byte-identical and no build errors. Before the fixes the same fixture found 20 of 22 (no WRL, no MYSQL3). WRL was additionally checked at iteration counts 1 and 3, byte-identical against CPU at both.

Coverage is one GPU vendor. AMD and Apple Silicon were not re-tested for this release.

---

# mdxfind v1.531 — Four new hash types; fixes a heap corruption that aborted multi-type runs, and a dispatch bug that pinned expensive types to one core

Source: mdxfind.c rev 1.530 → 1.531; gpu/gpu_opencl.c rev 1.209 → 1.210; gpu/gpu_kernel_cache.{c,h} rev 1.7 → 1.8 / 1.4 → 1.5; userdef.{c,h} rev 1.7 → 1.8 / 1.6 → 1.7; bench_rates.h rev 1.2 → 1.3; new mdxhost.h (companion release: hashpipe v1.100).

## Fixes

**Heap corruption during hash loading.** `build_compact_table()` allocated compact-table overflow chain entries from the `mymalloc` arena, while `compact_resize()` releases them with `free()`. Arena slices are not individually freeable, so the first time the compact table doubled, `free()` was handed a pointer into the middle of a slab and the process aborted with `free(): invalid pointer`. A second defect in the same block wrote one element past the end of the hash entry arrays, because it checked only the byte-buffer capacity and not the entry-array capacity.

Chain entries are now allocated with `malloc`, matching the two other chain-allocation sites. New `hashdata_reserve()` / `hashdatabuf_reserve()` helpers grow both the entry arrays and the byte buffer, and every append site uses them. A hardcoded `HashDataCap = 65536` that claimed capacity the arrays might not have was removed. AddressSanitizer is clean on a 55,404-hash corpus.

**Dispatch: expensive types with many salts ran on a single core.** The one-word-per-dispatch fan-out gated on a type's raw hashes/sec rather than its cost per wordlist line. SCRYPT at 3,403 h/s against 1,800 salts is about one word per *second*, but 3,403 cleared the threshold, so 512 words were batched into a single dispatch. The gate now uses `rate / Livesalts`. Measured on 1,800 scrypt hashes: 1× → 4.3× parallel (scrypt is memory-bound, so it does not reach core count; compute-bound types gain more). SHA1CRYPT was affected the same way. Output is unchanged — this is a throughput fix only.

## New hash types

**`-m e999` SHA1CRYPT** — NetBSD/Juniper sha1crypt, `$sha1$` (hashcat 15100). Iterated HMAC-SHA1 with the password as the persistent key. Input is liberal and output is conformant: the 28-character digest field encodes 21 bytes where SHA-1 produces only 20, and NetBSD pads the final group with `digest[0]` wrapped around. hashcat's published `-m 15100` example hash uses `0` for that byte and is therefore **nonconformant**; mdxfind accepts that spelling and reports the correct NetBSD/corpus one. The wrap convention was confirmed against 8,262 real hashes, 8,262 of 8,262.

**`-m e998` GOST-YESCRYPT** — `$gy$`. `HMAC-Streebog256(HMAC-Streebog256(Streebog256(K), M), yescrypt(K, S))`, where the inner HMAC message is the setting **without** its trailing `$`. Validated 25 of 25 against libxcrypt 4.4.27 across distinct real salts, plus salt lengths 4/8/12/16 and empty, and against an independently written third-party implementation.

**`-m e1001` CMIYC** — `$cmiyc$`, a contest-local type recovered from a stripped AIX PowerPC binary. Memory-hard: 64 MiB working set and 9,437,184 SHA-512 operations per candidate per salt. Validated against real hashes with known plaintexts.

**`-m e884` SCRYPT now also accepts the `$7$` crypt spelling** in addition to `SCRYPT:N:r:p:b64salt:b64hash`. The `$7$` form packs N/r/p as crypt64 characters and uses the salt as raw ASCII rather than base64, so it is normalised at load into the canonical form; both spellings are emitted on a crack.

**`-m e1000` 7ZIP** — 7-Zip AES, `$7z$` (hashcat 11600). Verifies by checking the AES zero-padding on the final ciphertext block rather than decrypting and decompressing, so **Deflate64 archives crack** — hashcat and john both report "exhausted" on those even when the password is in the wordlist, because neither implements a Deflate64 decompressor in its verify path. Stock `7z2john` output exceeds mdxfind's line limit — the entire encrypted stream sits in the final field, roughly 113 KB even for a small archive — so `tools/7z2mdx.py` truncates that field to the two ciphertext blocks stage 1 actually needs.

## Internal

**Built-in and user-defined hash types now have separate address spaces.** The user-defined op base was a hand-maintained constant that the built-in range had already grown into — built-ins reached 999 against a base of 1,000, leaving zero free slots, so the next built-in type would have silently aliased a user-defined op and shared its per-op state. The base is now derived from the `Types[]` length at startup, all 2,417 per-op array accesses route through accessors, and built-in and user state live in genuinely separate storage. Aliasing is structurally impossible rather than arithmetically avoided, and adding a built-in type needs no constant bumped.

This also fixed six sites that indexed `Types[]` by op across the whole range, reading past the end of the array for any user-defined type that had loaded hashes.

**Windows: the hx codegen source dump no longer targets `/tmp`.** `/tmp` is a POSIX convention that does not exist on Windows, so every dump on a Windows rig failed with `ENOENT` and logged a warning once per device per job. The dump is now written under the `MDXFIND_CACHE` directory, and when the cache is disabled it is skipped **silently** — a missing debug artifact is a non-event and does not deserve a message per device per job.

**Windows: diagnostics reported the host as `unknown`.** `gethostname()` is a Winsock call that fails with `WSANOTINITIALISED` unless `WSAStartup()` has run, and mdxfind never calls it, so the hostname in every FATAL and warning line was the fallback string. All ten call sites now use a portable helper (`GetComputerNameA` on Windows).

---

# mdxfind v1.530 — KRB5TGS23 / KRB5PA23: definitive HMAC checksum verification (fixes false positives)

Source: mdxfind.c rev 1.529 → 1.530 (companion hashpipe fix released as hashpipe v1.99).

The Kerberos RC4-HMAC (etype 23) verifiers — **KRB5TGS23** (`-m 13100`, e914) and **KRB5PA23** (e874) — validated a candidate password by checking the RC4-decrypted plaintext's ASN.1 / timestamp byte structure instead of recomputing the cryptographic checksum. A wrong key can hit that loose byte pattern by chance, producing a **false positive** — a reported crack that is not actually the password. (The KRB5TGS23 ASN.1 check additionally mishandled the DER `0x82` long-length encoding, testing the SEQUENCE tag at the wrong offset.)

Fix: both verifiers now recompute `HMAC-MD5(K1, decrypted)` and require all 16 bytes to equal the stored checksum — the definitive test, matching hashcat mode 13100. Verified: a reported false positive is now correctly rejected, valid test vectors still crack, and the full self-test suite passes. The AES etype 17/18 Kerberos types (KRB5PA-17/18, KRB5DB17/18) already performed proper HMAC-SHA1 / key verification and are unchanged.

If you have collected KRB5TGS23 / KRB5PA23 results from earlier builds, re-verify them — some may be spurious.

---

# mdxfind v1.529 — GPU: fix `CL_INVALID_GLOBAL_WORK_SIZE` (-63) on large salted hash lists with many rules

Source: gpu/gpu_opencl.c rev 1.208 → 1.209; mdxfind.c rev 1.528 → 1.529 (documentation / version-anchor only, no functional change in mdxfind.c).

On OpenCL AMD GPUs, `mdxfind` with a salted hash type plus a rule file could fail with `FATAL: GPU error: md5_rules dispatch error -63` (`CL_INVALID_GLOBAL_WORK_SIZE`). It reproduced with a small wordlist, a large salted hash list (≥ 8192 unique salts), and more than ~1024 rules; a ≤ 1k-rule run was unaffected.

Cause: the salted rules dispatch packs the salt axis into a single 1-D OpenCL global work size, `global = num_words × n_rules × mask_size × salt_axis`, where `salt_axis` follows the per-page salt count (default 8192). On AMD GPUs a 1-D global work size is bounded by the 32-bit global-ID range (2³²), and the product crosses it once `num_words × n_rules × salts_per_page > 2³²` — with the common 512-word batch and an 8192-salt page, that is exactly `n_rules > 1024`.

Fix: the per-page salt count is now capped so the resulting global work size stays under the device limit (`salts_per_page ≤ MAX_GLOBAL / (num_words × n_rules × mask_size)`). The cap is applied before the salt-page count is computed, so a smaller page just yields more pages and every salt is still processed — no cracks are lost, and low-rule / unsalted / small-salt workloads are unchanged (the cap is inert unless the product would overflow). The env var `MDXFIND_GPU_MAX_GLOBAL` overrides the ceiling for devices whose limit differs.

---

# mdxfind v1.528 — Fix `-M e537` (PHPBB3MD5): `$P$`/`$H$` hashes now load when e537 is selected without also selecting e455

Source: mdxfind.c rev 1.526 → 1.528. The release tag is realigned to the mdxfind.c `$Header` revision (prior release tags ran one ahead of the file revision; this release restores tag == rev).

`mdxfind -M e537` (PHPBB3MD5) silently failed to load any `$P$`/`$H$` hash: the run reported zero hash calculations and "None found", even when the hash's password was present in the wordlist. hashpipe identified the same hash correctly, which localised the fault to mdxfind's input path.

The `$P$`/`$H$` loader block gated on `lf[JOB_PHPBB3]` (e455) alone. PHPBB3MD5 (e537) and PHPBB3 (e455) consume the identical hash format — they differ only in whether the candidate is the password or `md5(password)` — and the e537 compute arm falls through to the e455 arm. Selecting e537 without e455 left the loader gate closed, so the hash was never parsed, never salt-loaded, and never entered the table. The post-load pass that copies the salt into `Typesalt[JOB_PHPBB3MD5]` was already present; only the loader gate was missing e537.

Fix: the loader gate now admits either op (`lf[JOB_PHPBB3] || lf[JOB_PHPBB3MD5]`). Verified: `-M e537` alone now cracks a phpBB3-MD5 hash; `-M e455` and `-M e455,e537` are unchanged; a `$1$` (MD5CRYPT) sanity load confirms neighbouring `$…$` parsing is undisturbed. An audit of the other shared-format families found bcrypt (`$2$`) and the `$5$`/`$6$` crypt family already admit all their fall-through variants — phpBB3 was the only gate missing its `*MD5` op.

No other behaviour changes vs v1.527.

---

# mdxfind v1.527 — User-defined hash types: accept `salt` + `user` slots; bcrypt + phpass available to mdxfind userdef expressions

Source: userdef.h rev 1.5 → 1.6, userdef.c rev 1.6 → 1.7, mdxfind.c rev 1.525 → 1.526, hx_func.c rev 1.4 → 1.5; iMac Makefile rev 1.46 → 1.47 (adds `-DHX_HAS_KDF` to the `hx_func_sa.o` build rule).

Window: 2026-06-10. Released 2026-06-10.

v1.527 unblocks salted user-defined hash types in mdxfind for the common case of one salt + one repurposed-as-second-salt slot. The userdef v1 loader rejected ANY reference to `salt` / `salt2` / `pepper` / `user` to keep the M1 scope unsalted; v1.527 lifts that for `salt` and `user` (loader + dispatch infrastructure already in place via `Typesalt[]` / `Typeuser[]` and `hx_vm_run`'s 5-slot signature). `salt2` + `pepper` stay rejected until the v2 load grammar (`load=` / `fields=` / `sep=` / `*.enc=`) lands, since the existing per-line file loader has no way to parse those out of the hash file.

## What unlocks

`bcrypt(phpbb3(pass))` and similar two-salt compositions can now run as `-m u<id>` in mdxfind:

```ini
# userdef.txt
[BcryptPhpbb3]
id = 800
hx = bcrypt_hex(phpass_encode(phpass_bin(pass, salt, 9)), user, 10)
```

Invocation against external salt pools:
```
mdxfind -m u800 \
    -s phpbb3_salts.txt \
    -u bcrypt_salts.txt \
    -F hashes.txt \
    wordlist.gz
```

End-to-end test vector (verified): `pass=123456`, phpbb3 salt = `RsqOrLNk` (8 char + log2=9), bcrypt salt = `vw31ldi5VPlyG2t5HxqIKe` (22 char + cost=10), expected bcrypt-hex digest `d07c7b7f140772676d0480c3136ab484a20e426505ddfb`.

## userdef.c changes

- `program_uses_salt(prog)` (bool) replaced by `program_slot_mask(prog)` (USERDEF_SLOT_* bitmask).
- `finalize_stanza()` rejects only when `SALT2 | PEPPER` bits are set; `SALT | USER` are accepted and propagated to the new `slot_mask` field of `struct userdef_type`.
- Load report adds `, uses salt` / `, uses user` annotations.
- Legacy `uses_salt` field preserved (now nonzero iff `slot_mask != 0`).

## mdxfind.c changes

- TypeOpts auto-assignment for user ops now consults `slot_mask`: `TYPEOPT_NEEDSALT|SALTJUDY` if `salt` is referenced; `TYPEOPT_NEEDUSER|USERJUDY` if `user` is referenced. The standard generic-file salt+user loaders + the SaltArray→Typesalt and UserArray→Typeuser post-load copies then populate per-type Judys without additional code.
- The userdef dispatch arm now iterates `Typesalt[op]` (via `build_salt_snapshot`) and `Typeuser[op]` (via `JSLF`/`JSLN`) when the corresponding slot is referenced; nested salt × user product when both are present. Unused slots take a single empty-value visit (unsalted path is byte-identical to v1.526).
- `-i N>1` iteration feedback for salted types: the per-iter `vpass` re-fed digest is the LAST seen digest across the salt × user product, matching the per-built-in convention.

## hx_func.c + Makefile change

`bcrypt` + `phpass` (plus the rest of the KDF / crypt-family entries that link into mdxfind: scrypt, argon2*, yescrypt, md5crypt, apr1, sha256/512crypt, descrypt) are now registered in mdxfind's `hx_func_sa.o` table under the new compile flag `HX_HAS_KDF`. The build-only `tools/hx8_to_c` standalone tool (HX_STANDALONE, no HX_HAS_KDF) skips them — it doesn't link the crypt libs and never could. The 6 truly hashpipe-only entries (pomelo, rc4_hmac_md5, aes128/256_cts_hmac_sha1, sm3crypt, gost12_512crypt) keep a nested `#ifndef HX_STANDALONE` guard because mdxfind doesn't link `pomelo_hash` / `hp_sm3_*` / KRB5 CTS.

iMac Makefile rule `hx_func_sa.o:` adds `-DHX_HAS_KDF`. Per-host Makefiles on remote build hosts (.205, .206, .209, .206/.209 cross-compile, ubpower8, dev1, hpi7, firefly) must add the same flag — they are NOT auto-synced.

## What's still NOT supported

- `salt2` + `pepper` userdef slot references (need v2 load grammar). Document workaround: repurpose `user` as the "second salt."
- Per-hash variable iteration counts in hx expressions (phpbb3 log2, bcrypt cost). The hx `^N` operator is compile-time only; runtime iter would need an `HX_SLOT_ITER` language extension. Workaround: one userdef stanza per distinct `(log2, cost)` pair.
- Per-line `hash:salt:user` binding. The generic loader at `mdxfind.c:46195` stores the SAME `:suffix` into BOTH `Typesalt[Dosalt[x]]` AND `Typeuser[Douser[x]]` — it doesn't split on a second colon. v1.527 supports salts and users only via the pool-file flags `-s SALT_FILE` and `-u USER_FILE` (combinatorial N × M per pass). Per-line binding is a future loader change.

---

# mdxfind v1.526 — Makefile hot-fix: stop `make clean` from deleting checked-in metal `_str.h` + `mdxfind_metallib.h`

Source: Makefile only (clean target). Binary content identical to v1.525 — mdxfind.c `$Header` stays at rev 1.525.

The v1.525 `clean` target indiscriminately deleted `gpu/mdxfind_metallib.h` and `gpu/metal_*_str.h` files even though they are CHECKED IN as the pre-generated ground-truth artifacts shipped to Linux/Windows builds (which do not run `metal2str.py`). Running `make clean` on a fresh Linux clone produced a dirty working tree with phantom modified-file noise.

This release moves those deletions into a new explicit `make metalclean` target for the macOS workflow that genuinely wants to force the metal artifacts to regenerate. On macOS the existing pattern rules at the top of the Makefile already auto-regenerate via timestamp dependency on `.metal` source files; no manual cleanup is required.

After upgrading to v1.526:
- `make clean && make all` on a fresh clone leaves `git status` clean.
- macOS devs who want to force a metal regen: `make metalclean` (or delete the files by hand).

No source code, no library, no platform behavior changes vs v1.525.

---

# mdxfind v1.525 — BSDICRYPT split (e997), MD5SALT padding fix sweep (GPU OpenCL + Metal + ARM NEON + PowerPC slen 24..31)

Source: mdxfind.c rev 1.524 → 1.525, mymd5.c rev 1.33 → 1.34, gpu/gpu_md5salt_core.cl rev 1.7 → 1.8 (+ `_str.h` 1.5 → 1.6), gpu/metal_md5salt_core.metal rev 1.3 → 1.4 (+ `_str.h` -ko 1.2 → 1.3 NEW to github).

Window: 2026-06-01 (post-freeze-lift) through 2026-06-04. Released 2026-06-04.

Two independent fixes shipping together: (1) BSDi extended-DES (725-round, 20-char `_CCCCSSSS` format) is moved out of `JOB_DESCRYPT` into its own `JOB_BSDICRYPT` (e997, hashcat mode 12400), restoring `-z descrypt` to the 4096 standard salts only; (2) a multi-platform MD5 padding bug in the MD5SALT path on GPU OpenCL, GPU Metal, CPU ARM NEON, and CPU PowerPC is corrected for the `total_len ∈ [56..63]` boundary case. CPU x86 SSE path was unaffected.

## BSDICRYPT split (e997 — hashcat mode 12400)

Since mdxfind.c rev 1.187 the loader recognized both 13-char standard `crypt(3)` (`saltchar2 + 11 hash`) and the 20-char BSDi extended format (`_CCCCSSSShashhash...`, single `_` prefix + 4 round-count chars + 4 salt chars + 11 hash chars) and routed both to `JOB_DESCRYPT`. The `-z` generator at `mdxfind.c:48156-48174` was emitting 8192 candidate salts (4096 standard + 4096 extended `_J9..XX..`) per password — twice the actual standard-DES keyspace, and conflating two wire-format-distinct algorithms behind one mode number.

This release introduces `JOB_BSDICRYPT = 997` (next available after `JOB_PEERCOIN_WALLET = 996`) with the BSDi extended path lifted into a sibling case mirroring `JOB_DESCRYPT`. Wire-format demultiplexing in the loader narrows `JOB_DESCRYPT` to len==13 only; the len==20 + `_` prefix variant routes to `JOB_BSDICRYPT`. Counters split: `DEScryptcnt` for standard, `BSDIcryptcnt` for extended.

Behavior change (intentional):
- `echo password | ./mdxfind -m e500 -z stdin` now emits **4096** lines (4096 standard salts × 1 password); previously emitted 8192.
- `echo password | ./mdxfind -m e997 -z stdin` emits the 4096 `_J9..XX..` extended-DES salts.
- `hashcat -m 12400` reference (the canonical BSDi-extended example vector `_J9..SDizh.vll5aL`) now matches against `-m e997`, not `-m e500`. `Maphashcat[12400]` repointed from 500 to 997.

Validation: `-z e500` keyspace audited (4096 lines, zero `_` prefix); `-z e997` keyspace audited (4096 lines, all `_` prefix, all 20-char); round-trip crack of the hashcat reference vector `_GW..8841inaTltazRsQ` succeeds via `-m e997` and `-m 12400`; table-parity diff against the hashcat type catalog clean at indices 997 and 12400.

## MD5SALT padding fix sweep — `total_len ∈ [56..63]`

The MD5 message-padding step requires the 0x80 end-of-message marker to land in block 1 when `total_len ∈ [56..63]` (i.e. the salt+pass length leaves <8 bytes of slack in the first 64-byte block, so the 8-byte length suffix forces a second block, but the EOM marker itself still belongs in block 1 at byte `total_len`). Four implementations were silently mis-placing the marker into block 2 at byte 0 (or in some cases skipping it entirely on ARM NEON), producing wrong digests for the corresponding salt-length range.

| Implementation | Status pre-v1.525 | Affected slen (salt-length) range | Fix |
|---|---|---|---|
| CPU x86 SSE (`mymd5.c:1575` mymd5salt2 Intel path) | CORRECT (reference) | — | unchanged |
| CPU ARM NEON (`mymd5.c:1077` mymd5salt2 NEON path) | wrong digests slen 24..31; collisions slen 32..35 | 24..31 | single→two-block compression, eom-in-first-block |
| CPU PowerPC (`mymd5.c:622` mymd5salt2 PowerPC path) | wrong digests slen 24..31 | 24..31 | single→two-block compression, eom-in-first-block |
| GPU OpenCL (`gpu/gpu_md5salt_core.cl` slow-path) | wrong digests slen 24..31 | 24..31 | `eom_in_first` guard before block-2 padding |
| GPU Metal (`gpu/metal_md5salt_core.metal` slow-path, both kernel-A and kernel-B) | wrong digests slen 24..31 | 24..31 | `eom_in_first` guard (twin of OpenCL) |

Reference fixture: user-supplied `work2711.txt` (all 30-char salts derived from the production `50m/50m.MD5SALT` dataset) had been silently producing 626,852 GPU cracks vs 626,863 CPU x86 SSE cracks on a 626,863-hash universe; the GPU shortfall + the ARM NEON CPU also producing wrong digests on Apple-host references were the two ends of the same boundary-padding bug. Post-fix, all five implementations produce byte-identical e31 sweep output across slen 22..40 — sha-of-sorted-output `003f7c00bcdbe9c342355eea4f60a65e`. Sorted-and-deduped crack counts match across CPU x86 SSE, GPU OpenCL (Pascal + Ada + RDNA4), GPU Metal (M1 + M2 Max).

The standard `sm-saltfull` benchmark fixture uses salts ≤ 23 chars, so this boundary was outside the regression sweep. Production workloads that use a 30-char salt format (e.g. wallet-derived KDFs, custom enterprise schemas with hash-truncated nonce headers) would have silently produced wrong-digest cracks before this release.

Reference invariance check (recommended for downstream consumers): on Apple Silicon hosts, do NOT use `-G none` as the MD5SALT correctness reference for pre-v1.525 binaries — the ARM NEON CPU path was the wrong-digest peer of the GPU bug. Use the x86 SSE CPU (or Python `hashlib.md5(salt+pass)`) as the cross-architecture oracle.

## Build / sync

Source-tree size, build matrix, and platform set unchanged from v1.524. The Metal `_str.h` companion file pattern shipped in v1.524 gains one more pair (`metal_md5salt_core_str.h` — `-ko` mode, regenerated by the standard `cl2str.py` toolchain). The `mdxfind-release` `sync_metal` step has been extended to include `metal_*_core_str.h` so future companion-pair additions ship automatically.

## Acknowledgments / cross-references

ARM NEON + PowerPC fix was a collateral catch during the Metal validation: the agent establishing the CPU oracle on Apple ARM dev1/dev3 hosts surfaced the slen 24..31 digest divergence that pointed at the same boundary-padding class as the GPU bug. The ARM NEON slen 32..35 collision symptom (different inputs → same digest) is resolved by the same two-block conversion — the single-block path was structurally dropping block-2 input bytes from the digest.

---

# mdxfind v1.524 — codegen kernel-B iteration (`-i N>1`), Metal kernel-A chunking, codegen auto-dispatcher (env-flag retirement), MD4/SHA1/SHA1RAW/SHA256/SHA256RAW codegen admission, per-stage Metal DISPATCH_TRACE, Gate-8 cosmetic widening

Source: mdxfind.c rev 1.523 → 1.524, gpu/gpu_opencl.c rev 1.205 → 1.208, gpu/gpu_opencl.h rev 1.42 → 1.43, gpu/gpujob_opencl.c rev 1.157 → 1.160, gpu/codegen_auto_dispatch.h rev 1.1 → 1.2 (NEW), gpu/codegen_auto_dispatch.c rev 1.1 → 1.2 (NEW), gpu/metal_kernel_a_rules.metal rev 1.5 → 1.6 (+ `_str.h` -ko 1.5 → 1.6), gpu_metal.m rev 1.125 → 1.131, gpu_metal.h rev 1.64 → 1.65, gpu/gpujob_metal.m rev 1.41 → 1.44, gpujob.h rev 1.47 → 1.48, codegen/hx_emit_opencl.c rev 1.20 → 1.21, codegen/hx_emit_metal.c rev 1.19 → 1.20, codegen/hx_emit_primitives.c rev 1.13 → 1.14.

Window: 2026-05-31 through 2026-06-01. Released 2026-06-01 (freeze-lift day).

This release extends the two-engine GPU codegen architecture introduced in v1.523 along three orthogonal axes: kernel-B gains a runtime iteration count (`-i N>1` works inside the two-kernel pipeline, hex-feedback semantic), Metal kernel-A gains the rule-axis chunking already shipped on OpenCL (closes the 99K-rule × 1M-word OOM ceiling on Apple), and the user-visible env-flag opt-in is retired in favor of an in-engine capability+perf matrix that auto-selects backend per (op, iter, rules, mask, bf, backend_kind). Codegen now admits MD4 + SHA1 + SHA1RAW + SHA256 + SHA256RAW alongside the original MD5. A latent Apple Metal correctness gap (legacy `template_phase0` returns zero cracks at `-m e1 -i N>1`) is closed by the auto-dispatcher routing those cells through codegen automatically.

## User-facing change summary

**New env vars (auto-dispatcher control):**
- `MDXFIND_GPU_BACKEND={auto|legacy|codegen}` — auto-dispatch (default) or FORCE one backend; developer/test only.
- `MDXFIND_GPU_BACKEND_QUIET=1` — suppress the one-shot per-(op, backend) advisory line.
- `MDXFIND_METAL_GPU_CAND_CAP_MB=N` — Metal kernel-A candidate buffer cap in MB; default 0.15 × `[device recommendedMaxWorkingSetSize]` clamped `[64..1024]`. Stress with `=64` for chunk-correctness verification.
- `MDXFIND_DISPATCH_TRACE=1` (Metal) — per-chunk `[disp-metal] kernel_a_us / host_gap_us / kernel_b_us / span_us` emission; field-name parity with the OpenCL `[disp]` twin.

**Deprecated env var (still honored as warning, ignored as decision):**
- `MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5` — superseded by the auto-dispatcher. If set, prints a one-shot stderr WARNING and is ignored; matrix decision fires instead. One-version retention; will be removed in v1.526+.

**New flag-effective behavior:**
- Apple Metal `-m e1 -i N>1` with rules now produces correct cracks WITHOUT any env flag (previously legacy `template_phase0` returned zero cracks; user had to know about the env-flag opt-in to get codegen routing).
- Metal `-m e1` with 99K+ rules × 1M+ words no longer hits the `mtl_buf_proto_packed` 415 GB single-buffer OOM; chunking now mirrors the OpenCL #343 implementation.
- `-i N>1` works through codegen for MD5, MD4, SHA1, SHA256 (hex-feedback). SHA1RAW and SHA256RAW (binary-feedback) admit at iter=1 only and fall through to legacy `template_iterate` at iter>1 with a per-op advisory.

## Codegen kernel-B iteration support (`-i N>1`)

Codegen kernel B now supports runtime iteration via the existing `OCLParams.max_iter` field (zero ABI change — the offset-60 slot was reserved for this in the v1.502 layout). Per-iteration probe mask is `1u << (iter & 31u)`; feed semantic is hex-encoded digest (matches the CPU JOB_MD5 / JOB_MD4 / JOB_SHA1 / JOB_SHA256 loop bodies at `mdxfind.c:28666-28679` and `:29088-29097`). The kernel B body now emits a `for (iter=1u; iter <= mi; iter++) { probe; if (iter<mi) feed; }` wrapper around the per-primitive `usp_{md5,md4,sha1,sha256}_iter_hex32_feed` helpers (`codegen/hx_emit_opencl.c:4191-4480`, Metal twin at `codegen/hx_emit_metal.c:3829-4140`).

Validated byte-exact cross-arch at iter ∈ {1, 2, 5, 10, 100} on Pascal (fpga GTX 1080) + Maxwell (hpi7 GTX 960) + Apple M1 (dev1) + Apple M2 Max (dev3). Sorted-md5 byte-identical to legacy `template_iterate` on hosts where legacy is functional. Apple M1 legacy `template_phase0` returns 0 cracks at iter>1 per a pre-existing template-iterate gap (`gpu/metal_template.metal:684` "Phase 1: template_iterate() is intentionally NOT called"); codegen is the only correct path on M1 for this cell. Apple M2 Max legacy WORKS at iter>1 (returns correct cracks); the gap is M1-specific. Auto-dispatcher routes both to codegen regardless (M1 = correctness; M2 Max = perf, 1.24× faster than legacy).

Cross-arch perf at 99,074 rules × `-i 10` × rockyou-1m × profile_md5_10k_hits.txt (50 cracks each, sorted-md5 `eac0169d…`):

| Host (arch) | Legacy `-m e1 -i 10` | Codegen `-m e1 -i 10` | Codegen / Legacy |
|---|---:|---:|---:|
| Ada mmt (RTX 4070 Ti SUPER) | 88.7 s | 108.3 s | 1.22× slower |
| Pascal fpga (GTX 1080) | 305.42 s | 425.56 s | 1.39× slower |
| Apple M1 dev1 | BROKEN (3 garbage cracks, 7–12 s) | 1950.3 s | n/a — legacy unusable |
| Apple M2 Max dev3 | 298.9 s | 240.2 s | **0.80× FASTER** |

The gap closes meaningfully at higher iter on NVIDIA (Ada 2.28× → 1.22×, Pascal 1.46× → 1.39×); on M2 Max codegen INVERTS the OpenCL pattern and beats legacy at both `-i 1` (2.73× faster) and `-i 10` (1.24× faster). M1 codegen wall measurements at `-i 1` show large thermal-state dependence (~318 s cold-from-idle vs ~620 s on a heated chassis); future M1 perf measurement should use ≥5-minute cool-down or report cold + warm separately.

## Metal kernel-A rule-axis chunking

Metal `gpu_metal_kernelb_dispatch_proto_chunked` now mirrors the OpenCL #343 chunked path. The single-buffer `mtl_buf_proto_packed = newBuffer(num_words × n_rules × 256)` allocation that FATAL'd at 415 GB on dev1 M1 (99,074 rules × 1,000,000 words) is replaced by a per-chunk loop with cap-bounded allocations. Apple-specific divergences from OpenCL:

- Cap fraction 0.15 × `[device recommendedMaxWorkingSetSize]` (vs OpenCL 0.25 × `CL_DEVICE_GLOBAL_MEM_SIZE`) — accounts for Apple unified memory shared with the OS and other processes.
- Env override `MDXFIND_METAL_GPU_CAND_CAP_MB` clamped `[64..1024]` MB.
- SHAPE-2 file-static host arena keyed by global ordinal `G` for hit accumulation; the `gpu_metal_kernelb_proto_plaintext()` accessor (installed by v1.523 hit-merge fix) branches on `mtl_proto_resolved_active` — UMA zero-copy read is no longer safe under chunking since each chunk reuses the device candidate buffer.
- `num_rule_chunks ≤ 1` (small-rule workloads) takes the verbatim single-dispatch path; production `-m e347` and the seven Phase-5a family algos are byte-identical to v1.523.

Latent bug deleted in the same commit: `gpu/metal_kernel_a_rules.metal:989-993` had a B3-cursor early-return that would have dropped the entire grid for `chunk_base > 0` (`rule_idx_local < rule_cursor_start` true for all lanes at non-zero chunk base). Inert pre-chunking; would have silently produced 0 cracks if chunking shipped without removal. Caught in spec D5.a.

Validated on dev1 M1: 99,074 rules × 1,000,000 words succeeds at K=204, 486 chunks, 50 cracks `eac0169d…`. dev3 M2 Max same workload at K=256, 388 chunks (larger `recommendedMaxWorkingSetSize`), byte-identical cracks. Cap-stress `=64` MB produces 6,193 chunks (12.7× more), wall 847 s, no FATAL, no overflow. Cross-backend crack-parity confirmed against fpga Pascal OpenCL chunked.

## Codegen auto-dispatcher (env-flag retirement)

The `MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5=1` opt-in shipped 2026-05-29 is retired in favor of a hardcoded capability+perf matrix consulted at the existing route-gate sites (`gpu/gpujob_opencl.c:1160-1199`, `gpu/gpujob_metal.m:1248-1313`, `gpu_metal.m:4861`). New file `gpu/codegen_auto_dispatch.{h,c}` (~465 LOC) carries `codegen_auto_dispatch_pick(op, iter, rules, mask, bf, backend_kind) → {LEGACY, CODEGEN, FATAL}`. Lazy-cached after first call; one-shot stderr advisory per `(op, backend_pick)` tuple via dedup set.

Two correctness-critical cells flip behavior vs the prior env-unset default:
1. **Apple Metal × MD5/MD4 × iter>1 × rules** → CODEGEN automatically (flagship user-facing correctness fix; legacy `template_phase0` returned 0 cracks).
2. **OpenCL × MD5/MD4 × any-iter × rules** → LEGACY (today's env-unset default preserved; legacy is 1.22–2.28× faster than codegen on NVIDIA per the perf table above).

Deprecation shim: if a user still sets `MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5`, a one-shot stderr WARNING fires and the value is IGNORED — matrix decision fires anyway. One-version retention.

Developer FORCE override: `MDXFIND_GPU_BACKEND={auto|legacy|codegen}`. Useful for A/B perf testing and reproducing the broken-legacy cell intentionally.

Validated on 5 hosts (Pascal fpga, Maxwell hpi7, Ada mmt, Apple M1 dev1, Apple M2 Max dev3) across 60+ cells against CPU oracle; sorted-md5 byte-identical on every correctness cell.

## Codegen admission widened — MD4 + SHA1 + SHA1RAW + SHA256 + SHA256RAW

The original v1.503 codegen admitted only JOB_MD5. v1.524 widens to the full unsalted single-hash set:

| Job | iter range | Backend pick (default Apple Metal) | Backend pick (default OpenCL) |
|---|---|---|---|
| JOB_MD5 (e1) | any | CODEGEN if iter>1 else LEGACY | LEGACY |
| JOB_MD4 (e3) | any | CODEGEN if iter>1 else LEGACY | LEGACY |
| JOB_SHA1 (e8) | any | CODEGEN if iter>1 else LEGACY | LEGACY |
| JOB_SHA256 (e10) | any | CODEGEN if iter>1 else LEGACY | LEGACY |
| JOB_SHA1RAW (binary-feedback) | iter=1 | LEGACY | LEGACY |
| JOB_SHA256RAW (binary-feedback) | iter=1 | LEGACY | LEGACY |

The hex-feedback ops (JOB_SHA1, JOB_SHA256) ride the same iter-v1 codegen path as MD5/MD4; their CPU paths at `mdxfind.c:28666-28679` and `:29088-29097` use lower-case hex feedback (`prmd5(curin.h, mdbuf, len)`) between iters, identical to MD5's chain. The binary-feedback RAW siblings (JOB_SHA1RAW, JOB_SHA256RAW; `mdxfind.c:27994`, `:29077`) admit at iter=1 only; codegen's hex-feedback would produce wrong cracks at iter>1, so the route gate falls through to legacy `template_iterate` with a per-op advisory.

A latent iter-aware recompute bug was caught + fixed during validation: the SHA1/SHA256 hit-replay's full-digest recompute branch was iter=1-only; broke immediately at iter=2 for the new ops. Fix walks N iters of `mysha1 / mysha256` + `prmd5` hex-feedback chain mirroring the CPU loop, applied to both backends.

## Per-stage Metal DISPATCH_TRACE instrumentation

The `MDXFIND_DISPATCH_TRACE=1` env var (OpenCL since v1.504) now emits per-chunk lines on Metal too, with matching field names so cross-backend log scrapers don't need to branch:

```
[disp-metal] dev=0 op=1 path=kernelB_proto_chunked chunk=0 num_words=16384 n_rules=99075
             rules_in_chunk=204 hits=11 kernel_a_us=13380 host_gap_us=49 kernel_b_us=8315 span_us=87539
```

Uses `[cb GPUStartTime]` / `[cb GPUEndTime]` for per-kernel GPU-side timing (after `waitUntilCompleted`); `mach_absolute_time` for host-side gap. Coexists with the pre-existing `MDXFIND_METAL_KERNEL_A_TIMING=1` Knob G timing (different env, different scope). Default-off byte-identical to pre-edit. Distinct `[disp-metal]` prefix (vs OpenCL `[disp]`) signals the per-chunk vs per-batch granularity difference.

Closes the instrumentation gap surfaced during the M1 thermal-throttling investigation: prior to this Metal users had no per-stage breakdown to distinguish kernel-A vs host-gap vs kernel-B contribution, making thermal/contention/host-bottleneck diagnosis impossible without instrumenting the build.

## Gate-8 cosmetic widening

The pre-existing experimental-codegen one-shot advisory emitted by `gpu_experiment_rules_codegen_md5_enabled()` (OpenCL) and `gpu_metal_experiment_rules_codegen_md5_enabled()` (Metal) — now retired by the auto-dispatcher — had hardcoded "JOB_MD5 (-m e1)" wording; the v1.1 widen ship had extended the admit set to MD4 + SHA1RAW + SHA256RAW without updating the advisory string. v1.524 retires both accessors entirely as part of the auto-dispatcher; the new advisory machinery in `gpu/codegen_auto_dispatch.c` enumerates the actual op picked correctly.

---

# mdxfind v1.523 — A2/A3 long-mask GPU dispatch (paired OpenCL+Metal), Knob G uint4 coalesced candidate writes, rule-axis chunking for 100K+ rule sets, A4 brute-force engine, critical Metal e347 hit-merge fix

Source: mdxfind.c rev 1.522 → 1.523, gpu/gpu_opencl.c rev 1.195 → 1.202, gpu/gpu_opencl.h rev 1.42, gpu/gpujob_opencl.c rev 1.156, gpu/gpu_kernel_a_rules.cl rev 1.6 → 1.7 (+ paired `_str.h` -ko), gpu/gpu_kernel_a_masks.cl rev 1.4 (NEW + `_str.h`), gpu/gpu_kernel_a_rules_masks.cl rev 1.2 (NEW + `_str.h`), gpu/gpu_kernel_a_bruteforce.cl rev 1.2 (NEW + `_str.h`), gpu/metal_kernel_a_rules.metal rev 1.5 (+ `_str.h`), gpu/metal_kernel_a_masks.metal rev 1.4 (NEW + `_str.h`), gpu/metal_kernel_a_rules_masks.metal rev 1.2 (NEW + `_str.h`), gpu/metal_kernel_a_bruteforce.metal rev 1.2 (NEW + `_str.h`), gpu_metal.m rev 1.119 → 1.122, gpu_metal.h rev 1.59 → 1.61, gpu/gpujob_metal.m rev 1.39 → 1.40, gpujob.h rev 1.47.

Window: 2026-05-28 through 2026-05-31. Released 2026-06-XX (week of 2026-06-01 freeze lift).

This release bundles the two-engine GPU codegen architecture's first non-rules engines (A2 masks-only, A3 rules+masks composition, A4 brute-force), the first measured-perf kernel-A optimization (Knob G uint4 coalesced writes — Ada -21.6 % / Pascal -11.3 % wall), rule-axis chunking that unlocks 100K-plus rule sets on the codegen GPU path, and a critical correctness fix for Metal users running `-m e347` and the Phase-5a MAKE_MD5PASS family algorithms.

## Critical correctness fix — Metal e347 + Phase-5a family hit-merge

**Metal users running `-m e347` (MD5MD5SALT) or any of the seven Phase-5a MAKE_MD5PASS family algorithms (e122, e159, e161, e163, e165, e167, e169) at `n_rules >= 16` were silently dropping cracks.** The kernel-B codegen wrote `entry[0]` as a slot index into the kernel-A SCRATCH packed buffer, but the host hit-replay loop in `gpu/gpujob_metal.m` treated it as a `widx` index into the host input-word buffer. At low `n_rules` (1–8) this produced partial drops with structurally-untrustworthy plaintext labels; at `n_rules >= 16` it dropped every crack. The defect was Metal-only; the OpenCL twin has had a dedicated proto-replay block since the Phase-4 codegen ship.

Fix: a dedicated `if (_proto_fired)` replay block in `gpu/gpujob_metal.m`, mirroring the OpenCL twin (`gpu/gpujob_opencl.c:1488-1647`), backed by a new `gpu_metal_kernelb_proto_plaintext()` accessor in `gpu_metal.m` (mirror of `gpu_opencl_kernelb_proto_plaintext()` at `gpu_opencl.c:14483`). The block routes e347 hits through `checkhashsalt()` with `pack_map` salt resolve and the seven family ops through `checkhash()` with `oracle_compute_md5pass_family` full-digest recompute. Pure addition (~150 LOC); the standard rules-engine replay loop is unchanged and OpenCL is unaffected.

Validation gates: 14-cell repro grid on dev1 M1 (all stdout cracks now equal kernel-B hit count, was 0 at `n_rules >= 16`); 28-cell Phase-5a family-matrix (7 ops × 4 rule counts) byte-exact vs CPU oracle; dev3 M2 Max cross-check identical to dev1; OpenCL Pascal smoke confirms zero bleed-over; degenerate `n_rules=1` preserves the original 10/10 path.

**Metal users on prior releases should upgrade as soon as the binary ships.** The bug was latent from the Phase-4 codegen introduction (v1.502) through v1.509.

## Long-mask GPU dispatch — A2 (masks-only) + A3 (rules + masks composition)

Two new kernel-A engines are paired across OpenCL and Apple Metal: A2 generates candidates from `-n` / `-N` mask expansion; A3 composes A1's rule walker with A2's mask expander for the rules-plus-masks product. Both wire into the existing codegen kernel-B (engine B is unchanged — the swap is purely on the candidate-generation side, per the two-engine architecture).

The user-visible change: **long literal prefixes in masks are now GPU-eligible.** Pre-amendment the GPU rules-engine path fell back to CPU whenever a mask had more than 16 total positions (literal bytes plus variable placeholders combined), which excluded the common pattern of a long prefix sentence followed by a short variable tail. Example invocation:

```
mdxfind -f hashes.txt -m e1 -N "TestPrefix2026-05-30: ?a?a?a?a" wordlist.txt
```

The 22-character literal prefix plus 4 `?a` placeholders (34-byte mask) now stays on the GPU. The two caps are decoupled: `GPU_MASK_VAR_CAP=16` placeholder positions per side (mask-iteration domain, unchanged) and `GPU_MASK_LIT_BYTES_CAP=224` literal bytes per side (the new headroom). The total candidate length is still bounded by the 255-byte `RULE_BUF_LIMIT`.

Wire format: an interleaved run-descriptor stream (LIT length-prefixed bytes / VAR class-id / END) replaces the prior pure-position stream. `n_prepend` / `n_append` shift from position count to descriptor-stream byte length. A single unified walker handles both short and long masks on the same path (no dual-path drift); backward-compatible short-mask workloads were re-verified GPU-routed on all four perf hosts.

Cross-architecture validation: the same sorted-crack md5 `211342b917d22593465c4c6e3cca6a08` was produced byte-identically on five architectures running the same 8.145-billion-candidate workload — NVIDIA Ada (mmt, RTX 4070 Ti SUPER, OpenCL), NVIDIA Pascal (fpga, GTX 1080, OpenCL), AMD RDNA2 (ioblade, gfx1036, OpenCL `-G 3`), Apple M1 (dev1, Metal), and Apple M2 Max (dev3, Metal). A3 rules-plus-masks production tests (prepend-only and mixed prepend+append) were byte-identical across the two tested backends (Pascal OpenCL and M1 Metal).

Production e347 byte-identical when no `-n` / `-N` flag is set.

## A4 brute-force engine (developer-preview)

A fourth kernel-A variant ships as the brute-force candidate generator (`gpu/gpu_kernel_a_bruteforce.cl` + `gpu/metal_kernel_a_bruteforce.metal`), completing the kernel-A variant set {rules, masks, rules+masks, brute-force}. The A4 engine plugs into the same candidate-generation slot as A1/A2/A3.

A4 is gated behind two env flags and is **not yet on the production dispatch path**:

```
MDXFIND_KERNEL_A_PROTO=1 MDXFIND_KERNEL_A_VARIANT=4 ./mdxfind ...
```

Production runs with the env flags unset are byte-identical to v1.509. The harness has been validated on a 5-host fleet (3 NVIDIA OpenCL + 1 AMD blocked by compact-table gate in env-mode + 1 Apple Metal) across four fixtures (bf_smoke, bf_single, bf_dual, bf_custom) — all 12 fixture × host cells pass state-counter parity and candidate-multiset byte-identity vs the Python oracle. Promotion to default dispatch is a future sub-phase.

## Performance — Knob G uint4 coalesced candidate writes (paired OpenCL+Metal)

Kernel A1's per-byte candidate-write loop was replaced with 16-byte `uint4` stores from a private 16-aligned staging buffer. Per-slot byte claim is rounded up via `need_aligned = (need_bytes + 15) & ~15`. The kernel-B consumer, accessor, and decode paths are bit-exact.

OpenCL measured deltas at the 100K-rule workload (`dive.rule` 99,075 rules × `rockyou-1m.txt` × MD5):

| Host | Architecture | Kernel A | Wall |
|------|--------------|---------:|-----:|
| mmt | Ada RTX 4070 Ti SUPER | -32.4 % | -21.6 % |
| fpga | Pascal GTX 1080 | -24.7 % | -11.3 % |

Opt-in:

```
MDXFIND_EXPERIMENT_RULES_CODEGEN_VEC_WRITE=1 ./mdxfind ...
```

(composes with `MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5=1` for the unsalted-MD5 codegen route fork).

The Metal twin (`MDXFIND_METAL_EXPERIMENT_KNOBG_VEC_WRITE=1`, in `gpu/metal_kernel_a_rules.metal`) ships at parity but is **null on Apple AGX**: -0.06 % on M1, +0.41 % on M2 Max. Decomposition evidence (PROFILE_VARIANT V0–V6, see below) shows the apply_rule switch is the dominant component on AGX (77 % of kernel A on M1, 68 % on M2 Max) — the candidate-write loop simply isn't where Apple's time goes. The Metal Knob G is preserved as a future Apple-compiler-evolution monitoring substrate and to keep the OpenCL/Metal parity discipline intact; users should keep it unset.

Production `-m e347` byte-identical with the flag unset (it gates only the `-m e1` MD5 codegen experiment route).

## Rule-axis chunking — 100K+ rule sets on the codegen GPU path

The codegen kernel-B dispatcher (`kernelb_dispatch_proto`) now chunks across the rule axis when the candidate buffer would exceed a per-device cap. Cap helper: `clamp(0.25 × device_global_mem, 64 MB, 1 GB)`; override with `MDXFIND_GPU_CAND_CAP_MB=<N>`. Cross-chunk hit handling resolves the matched plaintext before clobber.

Single-dispatch path (`num_rule_chunks <= 1`) takes the verbatim pre-chunking body — production e347 is bit-identical to v1.509 when the workload fits in one dispatch. Multi-chunk path has a bounded halve-K retry on overflow.

This lifts the prior implicit ceiling that capped the codegen-GPU rules-engine at ~16K active rules. `dive.rule` (99,075 rules) now dispatches end-to-end on the codegen path with `MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5=1`; the legacy `-m e347` hand-tuned rules-engine path is unaffected.

## Experimental — codegen-MD5 rules route fork (`-m e1`)

`-m e1` rules can be routed through the two-engine codegen pipeline (A1 → codegen kernel-B unsalted MD5) behind:

```
MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5=1 ./mdxfind -m e1 -r rules wordlist.txt
```

Flag-unset is byte-identical to the production `-m e1` path. This is the carrier for ongoing kernel-A perf knob measurement (PROFILE_VARIANT, Knob G); the production `-m e1` path remains the hand-tuned rules engine. Head-to-head perf evaluation is in progress.

Validated bit-identical crack sets vs the legacy path on Ada / Pascal / RDNA4 at `-i1`.

## Observability

Three diagnostic facilities for kernel-A perf investigation; none are intended for production use, all default off, all are env-flag gated.

- `MDXFIND_DISPATCH_TRACE=1` extended to report `host_gap_us` plus per-kernel A and B timing through the chunked-codegen multi-chunk path. Prior to this release the multi-chunk path read 0 for these counters.
- `MDXFIND_PROFILE_VARIANT=N` (N in 0..6) selects one of seven kernel-A component-attribution variants. V0 is the production baseline byte-identical; V1–V6 are diagnostic stubs that isolate atomic-claim, candidate-write, and apply_rule cost shares. Used to bound predicted gains for future kernel-A work.
- `MDXFIND_METAL_KERNEL_A_TIMING=1` enables per-dispatch kernel-A wall timing on Metal (cb_a.GPUStartTime / GPUEndTime).
- `MDXFIND_METAL_PROFILE_VARIANT=N` is the Metal twin of the OpenCL PROFILE_VARIANT.

## Experimental, reverted

Two perf knobs prototyped during this window were reverted from the working tree after measurement showed no gain. They are not in the shipped binary; they are mentioned here so readers comparing source archives between v1.509 and v1.523 understand the diff.

- **Knob F** — workgroup-local atomic claim coalescing. Predicted +5 to +15 % kernel A on Pascal; measured -1.21 % Ada and +1.28 % Pascal regression. Architectural lesson: predictions need component-share microbench evidence, not just a decomposition upper bound.
- **Knob H** — apply_rule monolithic switch refactored to a per-opcode static-inline LUT (paired OpenCL+Metal). Predicted -5 / -15 / -38 / -34 % kernel A on Ada / Pascal / M1 / M2 Max; measured -0.96 % to +1.12 % median across all five hosts (null). Working theory: the OpenCL and Metal compilers already lower the V0 monolithic switch to a jump-table-equivalent shape; the LUT refactor reaches the same compiled shape. Architectural lesson: REFACTOR predictions need an AIR / PTX diff vs the baseline, not just confirmation that the proposed shape inlines.

## Architectural framing

Two project disciplines established during this window govern future kernel-A and kernel-B work:

- **Two-engine codegen architecture.** Kernel A is the candidate-generation engine (variants: A1 rules, A2 masks, A3 rules+masks, A4 brute-force); kernel B is the hash engine (per-algorithm codegen). The engines are independently swappable. Future perf knobs scope to one engine at a time.
- **OpenCL / Metal parity.** Kernel-A variants and host-side dispatch infrastructure ship as paired OpenCL+Metal efforts from inception. Backend-asymmetric perf knobs (e.g., Knob G's null result on AGX) ship at parity for code-shape consistency and future compiler-evolution monitoring.

## v1.510 through v1.522

Intermediate source-ready revisions (1.510–1.521) are the User-Defined Hash Types v1 ship described in the v1.509 entry above; that work was completed in the same freeze window and ships in the same release artifact. Rev 1.522 is the Knob G + observability bundle; rev 1.523 is the A2/A3 long-mask amendment ship.

## Known issues / scope

- A4 brute-force is developer-preview only; production dispatch promotion is a future sub-phase.
- The codegen-MD5 rules route fork (`MDXFIND_EXPERIMENT_RULES_CODEGEN_MD5=1`) is experimental; the production `-m e1` rules path is the hand-tuned engine.
- GPU dispatch for user-defined types (the v1.509 USER_<name> feature) is still planned future work; user types continue to run on the CPU through the hx interpreter in this release.
- Mali (firefly) correctness spot-check on the long-mask path was not run this session; the small-mask backward-compat path is presumed unaffected. Long-mask spot-check on Mali is a backlog item.

This is a SOURCE-READY ship: all source is checked in to RCS on the iMac authoritative tree and cross-arch-validated across five architectures. Version bump, GitHub release, and `mdxfind-release` run are deferred per the release freeze in effect until the week of 2026-06-01.

---

# mdxfind v1.509 — User-defined hash types: custom hx-expression algorithms via userdef.txt, selected with -m u<id>, output USER_<name>

Source: userdef.c rev 1.4, userdef.h rev 1.4 (NEW shared loader module), mdxfind.c rev 1.521, hashpipe.c rev 1.94, Makefile rev 1.46.

Window: 2026-05-28.

A skilled user can now define a custom hash algorithm as an hx expression in a configuration file and use it in mdxfind and hashpipe exactly like a built-in catalog type — no C code, no recompile. The hx language already interprets arbitrary expressions (this is what `hashpipe -X` runs); this release makes that a first-class extension point. Author and test an expression with `hashpipe -X 'expr'`, add a stanza, and run it.

## Definition file — stanza format

User types are read from `userdef.txt`, located in the directory that contains the `MDXFIND_CACHE` path (the same directory as the SQLite line-count cache). The file is INI-style: a `[Name]` header (the type's display name), an `id` key (a free-form identifier — number, UUID, or content-hash, the user's choice), and an `hx` key (the expression). The `hx` value is read verbatim to end-of-line because hx expressions contain `. ( ) " ' $ : ^` that the hx parser owns; the stanza reader never tokenizes it. `#` comments and blank lines are allowed. Unknown keys are warned-and-ignored, so a file written for a future version still loads under the current one.

```
[Cust1]
id = 47
hx = sha1(md5(pass) . "register")
```

## Selector and output

A user type is selected by its identifier: `mdxfind -m u47 -f hashes.txt wordlist.txt`. The `u<id>` selector is an exact string-keyed lookup (not the `-h` regex path), so an identifier may safely contain regex metacharacters. Matches are reported with the stanza name carrying a `USER_` prefix and the standard iteration suffix, e.g. `USER_Cust1x01 e0d195a682ae3f8a9edbd111d4ae28a3095beb87:password123`. User types ride the same `-i` iteration machinery as built-in types (`-i 2` produces `x02`).

## hashpipe pipe-mode identification

hashpipe loads the same `userdef.txt` and identifies user types in pipe mode: `echo <digest>:<password> | hashpipe` reports `USER_<name>x01` on a match. Identification is additive — hashpipe still reports any matching built-in type alongside the user type. A non-matching line produces no `USER_` output.

## Load-time advisories

When `userdef.txt` loads, each type prints a status line summarizing what was registered (name, selector, internal op, digest length). Three further advisories surface through the same load-time and invoke-time output, with no new command-line flags:

- **Dedup advisory** — when a user expression is byte-for-byte equivalent to a built-in catalog type, mdxfind prints a non-fatal note suggesting the equivalent `-m e<N>`, which has a hand-tuned GPU path. The advisory never blocks; the user's type still runs.
- **Content-hash suggestion** — mdxfind computes and displays a stable content-hash of the compiled expression as a suggested shared identifier. It is a suggestion only, never enforced; a content-hash id is self-describing, which mitigates the dependency that a `USER_*` solved-hash file is only interpretable with its `userdef.txt` present.
- **Skip and reject messages** — a malformed stanza or unparseable expression is skipped with a loud per-entry warning naming the file and line, so a typo in one entry does not abort a run that selects another. If the *selected* type failed to load, the run is fatal with the specific diagnostic.

## v1 scope

User types run on the CPU through the hx interpreter. At invoke, an honest status line reports GPU eligibility: a GPU-supported expression shape reports that the shape is eligible but that GPU dispatch for user-defined types is not yet enabled and runs on CPU; an unsupported shape reports that the GPU is not available for the shape and runs on CPU. GPU dispatch for user-defined types is planned future work.

User types are unsalted and password-only in this release. An expression referencing salt, second salt, pepper, or a user/userid value is rejected at load with a clear message. Salted and structured user types — carried by a per-type hash-line load grammar — are a planned future direction.

The dedup advisory and the GPU-eligibility status are mdxfind-only (hashpipe has no GPU path and no catalog comparator); hashpipe still loads, identifies, and skips salted entries at load. No new command-line flags were added — all of the above surfaces through normal load-time and invoke output.

This is a SOURCE-READY ship: all source is checked in to RCS on the authoritative tree. Version bump, GitHub release, and `mdxfind-release` are deferred per the release freeze in effect until the week of 2026-06-01.

---

# mdxfind v1.508 — Phase 5c: e123 MD5MD5PASS multi-emit GPU codegen — the FIRST multi-emit algorithm — family 30/30 on BOTH backends (OpenCL + Metal)

Source: mdxfind.c rev 1.519, codegen/hx_emit_opencl.c rev 1.19, **codegen/hx_emit_metal.c rev 1.18 (Metal twin, 5c.3)**, codegen/hx_emit_primitives.c rev 1.11, codegen/hx_spec_entry.h rev 1.3, tools/hx8_to_c.c rev 1.4, gpu/gpu_codegen_eligible.c rev 1.7 (comment-only), codegen/tests/run_validation_family_md5pass.sh rev 1.12, codegen/tests/family_md5pass/e123_smoke.txt rev 1.1 (NEW), codegen/tests/family_md5pass/e123_multiemit_canary.txt rev 1.1 (NEW G1b canary). Regenerated artifact (not RCS-tracked): codegen/hx_specs_data.c (e123 row flips is_outlier=1→0, program NULL→&_hx_program_122, emit_class=HX_EMIT_MULTI, note_ref=24). UNCHANGED: gpu/gpu_common.cl + gpu/metal_common.metal (md5_block already present on both — multi-emit lives in the emit helper, not a new primitive), gpu/gpu_common_str.h + gpu/metal_common_str.h (no str regen needed), gpu/gpujob_metal.m + gpu/gpujob_opencl.c (4-arg oracle signature preserved; e123 16-byte digest never takes the hit-replay recompute branch; metal_gpu_hash_words / gpu_hash_words resolve e123 to 4 words via the default arm on both backends).

Window: 2026-05-27.

**e123 MD5MD5PASS is the FIRST multi-emit algorithm** in the hx codegen pipeline: ONE password produces TWO outer-hash digests, each probed against the loaded hash table as an independent found-hash candidate (matching the CPU oracle at mdxfind.c:25181-25204, which calls `checkhash()` once per variant):
- variant 0 (canonical): `md5( hex32(md5(pass)) . pass )`
- variant 1 (colon):     `md5( hex32(md5(pass)) . ':' . pass )`

## Sub-phase 5c.1 — emit_class plumbing + markup-strip + regenerate

`enum hx_emit_class { HX_EMIT_SINGLE=0, HX_EMIT_MULTI=1 }` + `int emit_class` + `int note_ref` added to `struct hx_spec_entry` (codegen/hx_spec_entry.h rev 1.3). C zero-init defaults every existing entry to HX_EMIT_SINGLE / note_ref=0 — the 29 prior family members + all non-family entries take the single-emit body unchanged.

The generator `tools/hx8_to_c.c` rev 1.4 gains `extract_note_ref()` + `strip_note_markup()`. For e123 ONLY (gated on `type==123 && note_ref==24`, per the minimal-change R4 rule), it strips the troff-italicised `(see Note [24])` annotation so the leading `md5(md5(pass).pass)` compiles to a real 6-op FAMILY_MD5PASS program and flags `emit_class=HX_EMIT_MULTI`, `note_ref=24`. `note_ref` is recorded for every Note-annotated row (e.g. e687 carries note_ref=24 but stays an outlier / SINGLE). Regenerated `codegen/hx_specs_data.c`: a normalized structural diff confirms ONLY the e123 row changed (is_outlier 1→0, program NULL→&_hx_program_122) — exactly +1 program static; every other row's structural fields byte-identical. Pre-flight grep for stderr-merge corruption = 0 (per feedback_hx8_to_c_no_stderr_merge.md).

## Sub-phase 5c.2 — OpenCL multi-emit body + MD5-as-outer helper + oracle

**New MD5-as-outer emit helper** `emit_outer_md5_concat_then_hash` (codegen/hx_emit_opencl.c rev 1.19): MD5 was always the INNER hash in the shipped family; e123 needs MD5 as the OUTER too. The helper mirrors the MD4 helper (LE schedule, 4-uint state, 16-byte digest, single-block fast path + multi-block first_has_pad tail) plus a `sep` parameter: sep=0 emits `hex32 || pass`; sep=1 injects a `':'` at logical position 32 and shifts pass to position 33 (total_len = 33 + plen).

**Multi-emit kernel body** `emit_family_md5pass_kernel_multiemit`: computes `md5(pass)` ONCE (shared inner state across both variants — natural hoist), then a compile-time-N=2 UNROLLED set of probe+emit blocks. Each block calls the outer helper with its `sep`, probes `compact_fp` to resolve its own `matched_idx`, and calls the **EXISTING `EMIT_HIT_4_DEDUP_OR_OVERFLOW` macro UNCHANGED**. The dedup key is already `(matched_idx, iter_bit)` keyed on the matched loaded-hash slot — ALREADY the correct multi-emit key: two variants hitting two different loaded hashes land in two different dedup cells and BOTH emit. NO new field, NO buffer resize, NO key widening (recompute-per-variant per the agreed design; the single-emit body is untouched). The hit record stays 16 bytes with NO variant tag — the emitted fingerprint self-identifies the matched hash on hit-replay, matching CPU semantics (mdxfind prints the matched hash FORM, not the variant).

**Eligibility gate** is the SINGLE `job_to_prim_table` row `{ 123, HX_PRIM_MD5 }` (codegen/hx_emit_primitives.c rev 1.11), NOT a global `supported_5a` flag. HX_PRIM_MD5 was already `supported_5a=1` (it is the inner hash used by every family member), so adding this row admits ONLY job 123 (no other family member maps to an MD5 outer) and does NOT wrongly admit other MD5-inner algos. The multi-emit per-variant body is selected downstream by the spec entry's `emit_class == HX_EMIT_MULTI`, not by the prim id. The admit predicate, harness gates, chokepoint, init-gate, and OpenCL `_proto_hexlen` all auto-propagate via D17.4.b. e123 closes the MAKE_MD5PASS family at **30/30** GPU-eligible on OpenCL.

**Oracle + dlen** (mdxfind.c rev 1.519): `oracle_compute_md5pass_family` replaces the e123 FATAL arm with the canonical variant-0 return (16 bytes) — the gpujob hit-replay recompute branch is not reached for a 16-byte digest, so the 4-arg signature stays unchanged for all callers; `dlen` switch adds JOB_MD5MD5PASS=16. The validation harness `hx_family_md5pass_validate_run_shared` is made multi-emit-aware: `n_variants=2` for e123 (1 otherwise), plants BOTH variant digests per pass as separate loaded-hash rows, sizes the compact table + hits buffer for `n_rows = n_pass × n_variants`, and the diff matches each hit's digest against the pass's variant rows (per-row seen bitmap), requiring `vn_hits == n_rows`.

## OpenCL validation (Pascal GTX 1080, fpga.local) — all GREEN

- **G1b dual-hash canary (THE multi-emit gate): PASS.** 4 passwords → GPU emitted **8 cracks** (BOTH variants per password), byte-exact vs the CPU oracle. The plaintext `password` appears twice with distinct digests (canonical `d3f6f4e6…` + colon `0c7f57e9…`, each verified against openssl and the CPU `mdxfind -m e123`).
- **e123 5-fixture matrix: PASS** — e123_smoke + family_smoke (n_rows=16), medium (n_rows=2048), edge_minlen / edge_maxlen (n_rows=256), and large at half-fixture (524,288 pw → **1,048,576 rows / 1,048,576 hits** byte-exact). The full 2,097,152-row family_large exceeds the NVIDIA single-NDRange watchdog (a harness fixture-scale ceiling at the 2× multi-emit hit volume — NOT a correctness defect; the e122 single-emit control PASSES at 1,048,576 rows on the same binary).
- **G2 single-emit regression: PASS (15/15 cells)** — all prior family members show `n_variants=1`, one hit per pass, byte-exact; the emit_class plumbing defaults them to SINGLE with ZERO behavior change.
- **G3 e347 regression: PASS** — smoke (32 pairs) + medium (1024 pairs) byte-exact; no collateral from the shared family emitter / harness changes.

The dumped e123 kernel on fpga shows both `sep=0` and `sep=1` probe+emit blocks and JIT-compiles cleanly on Pascal.

## Sub-phase 5c.3 — Metal twin (codegen/hx_emit_metal.c rev 1.18)

Hand-ported, structural mirror of the OpenCL twin (no translator):

**`emit_outer_md5_concat_then_hash_metal`** — MD5-as-OUTER helper with the `sep` parameter (sep=0 canonical `hex32 || pass`; sep=1 colon `hex32 || ':' || pass`, total_len=33+plen). Metal idioms: `device const uchar *pass` + `thread uint *` outputs; `md5_block(a,b,c,d,M)` from metal_common.metal takes `thread uint &` references and accumulates into a..d, so a..d are pre-seeded with the MD5 IV (same accumulate convention as the OpenCL `&a..&d` pointer form). LE schedule, single-block fast path + multi-block `first_has_pad` tail, NO state byte-swap. R11 verified: MD5 uses XOR/add/rotate only — no scalar `bitselect()` in play.

**`emit_family_md5pass_kernel_metal_multiemit`** — computes `md5(pass)` ONCE, then the compile-time-N=2 unrolled loop: per variant the outer-helper call (`sep`) → `probe_compact_idx` → the **EXISTING Metal `EMIT_HIT_4_DEDUP_OR_OVERFLOW` macro UNCHANGED**, dedup keyed on per-variant `matched_idx`. 16-byte hit record, no variant tag (fingerprint self-identifies). Selected by `entry->emit_class == HX_EMIT_MULTI` at the top of `emit_family_md5pass_kernel_metal`; the single-emit body is fully isolated (G2 no-op). The Metal HX_PRIM_MD5 FATAL is replaced with the emit_class gate (MD5-outer admitted only when MULTI), and MD5 is wired into the per-primitive emit dispatch + FATAL filter. `gpu/gpujob_metal.m` caller untouched (4-arg oracle signature preserved; verified zero changes needed). `metal_gpu_hash_words(JOB_MD5MD5PASS)` resolves to 4 words via the default arm — exact parity with the OpenCL `gpu_hash_words` default arm.

## Metal cross-arch validation (5c.4 — Apple M2 Max, dev3.local, Metal) — all GREEN

Built on dev1 (Apple Silicon Metal) per the iMac↔dev1 scp-staging truncation memo; test binary staged to dev3 via chunked-scp local-hop; production binaries on fpga + dev3 UNTOUCHED; /tmp-direct VALIDATE invocation.

- **G1b dual-hash canary (THE mandatory multi-emit gate): PASS on Metal.** 4 passwords → Metal emitted **8 cracks** (BOTH variants per password): `n_pass=4 n_variants=2 n_rows=8 vn_hits=8 matched=8 missing=0 extras=0 digest_mismatches=0`, byte-exact vs the CPU oracle. (Also confirmed PASS on dev1 Apple Silicon as a fail-fast pre-stage.)
- **e123 5-fixture matrix: PASS on Metal** — e123_smoke + family_smoke (n_rows=16), medium (n_rows=2048), edge_minlen / edge_maxlen (n_rows=256), and large at HALF-fixture (524,288 pw → **1,048,576 rows / 1,048,576 hits** byte-exact). Half-large per the 5c.2 NVIDIA-TDR finding (safe default on Apple Silicon too).
- **G2 single-emit regression: PASS (29/29 cells) on Metal** — every prior family member (e120, e122, e125, e127–e155 HAVAL ×15, e157, e159, e161, e163, e165, e167, e169, e171, e173, e175, e177) shows `n_variants=1`, byte-exact; the emit_class plumbing is a confirmed no-op on Metal.
- **G3 e347 regression: PASS on Metal** — smoke (32 pairs) + medium (1024 pairs) byte-exact; the shared family-emitter change causes zero collateral to the e347 path.

**OpenCL non-regression:** the only file changed in 5c.3/5c.4 is codegen/hx_emit_metal.c (a Metal-only TU, gated out of the OpenCL build). `rcsdiff` confirms the OpenCL-compiled codegen sources (hx_emit_opencl.c, hx_emit_primitives.c, hx_spec_entry.h) and mdxfind.c (1.519) are byte-identical to their checked-in 5c.2 state — the OpenCL multi-emit path proven in 5c.2 is unperturbed.

**e123 COMPLETE — first multi-emit milestone.** The MAKE_MD5PASS family is now **30/30 GPU-eligible on BOTH backends** (OpenCL Pascal + Apple Silicon Metal). e123 is the FIRST multi-emit algorithm in the hx codegen pipeline; `enum hx_emit_class` + `note_ref` are the extension seam for the remaining ~23 Note-[24] multi-emit entries (deferred — each its own future sub-phase).

This is a SOURCE-READY ship: all source is checked in to RCS on the iMac authoritative tree and cross-arch-validated on both backends. Version bump, GitHub release, and `mdxfind-release` are deferred per the release freeze in effect until the week of 2026-06-01.

---

# mdxfind v1.507 — Phase 5b Tier 4 COMPLETE (sub-phases 5b.4a + 5b.4b): Snefru-128/256 + GOST R 34.11-94 MAKE_MD5PASS family GPU acceleration; family 29/30 (source-ready)

Source: mdxfind.c rev 1.518, gpu/gpu_common.cl rev 1.34, gpu/gpu_common_str.h rev 1.26, gpu/metal_common.metal rev 1.33, gpu/metal_common_str.h rev 1.12, gpu/gpujob_metal.m rev 1.39, codegen/hx_emit_primitives.c rev 1.10, codegen/hx_emit_opencl.c rev 1.18, codegen/hx_emit_metal.c rev 1.17, codegen/tests/run_validation_family_md5pass.sh rev 1.11, codegen/tests/test_snefru_vectors.c rev 1.1 (NEW 5b.4a R15 regression canary), codegen/tests/test_gost_vectors.c rev 1.1 (NEW 5b.4b R15 regression canary), codegen/tests/family_md5pass/{e175,e177}_smoke.txt rev 1.1 (NEW 5b.4a), codegen/tests/family_md5pass/e125_smoke.txt rev 1.1 (NEW 5b.4b). UNCHANGED: gpu/gpu_opencl.c rev 1.196, gpu_metal.m rev 1.119, gpu/gpu_codegen_eligible.c rev 1.6, gpu/gpujob_opencl.c rev 1.155, codegen/hx_emit_primitives.h rev 1.2 — the D17.4.b table-driven admit + OpenCL `_proto_hexlen` + harness OR-chains + chokepoint + init-gate + listing path all auto-propagate; slot-map cap already 64.

Window: 2026-05-27.

Phase 5b Tier 4 is the FOURTH and final tier of the MAKE_MD5PASS primitive-lift roadmap. Per the architect's D18.6.b recommendation Tier 4 shipped in two sub-phases: **5b.4a shipped the Snefru pair** (`e175` SNE128MD5PASS, `e177` SNE256MD5PASS) — the clean pair sharing one parameterized block; **5b.4b ships `e125` GOSTMD5PASS** (the structurally-divergent block-cipher primitive — the highest-transcription-risk primitive in all of Phase 5b — isolated for dedicated attention). With Tier 4 COMPLETE the MAKE_MD5PASS family reaches **29/30 GPU-eligible** (96.7%), leaving only the `e123` MD5MD5PASS multi-emit outlier (deferred to its own future sub-phase).

This is a SOURCE-READY ship: all source is checked in to RCS on the iMac authoritative tree and validated on both backends. The version bump, GitHub release, and `mdxfind-release` run are deferred per the release freeze in effect until the week of 2026-06-01.

## Sub-phase 5b.4a — Snefru-128 (e175) + Snefru-256 (e177) GPU acceleration

`snefru_block` (Snefru core transformation, the standard hardened 8-pass variant) is now resident in the shared GPU helper sources (`gpu/gpu_common.cl` rev 1.33 and `gpu/metal_common.metal` rev 1.32) as a 512-bit (8-uint) state primitive, plus the 16 KB `SNEFRU_SBOX` / `MTL_SNEFRU_SBOX` lookup table (4096 uint32). The donor is the in-tree `RHash-master/librhash/snefru.c` `rhash_snefru_process_block` — which is also the LIVE CPU oracle for e175/e177 (librhash.a is in mdxfind's link list).

Per the architect's D18.1.a recommendation, Snefru is implemented as **ONE parameterized block** `snefru_block(state, block, is256)` handling both widths, NOT two separate functions. The donor's single `rhash_snefru_process_block` has the SAME 8-round S-box transform for both widths; only three sites differ on width, and they collapse to compile-time `if (is256)` branches that the JIT folds (is256 is baked as a literal per emit, so the kernel stays fully unrolled per variant):

- **W[] fill** — Snefru-256 loads `state[4..7]` into `W[4..7]` then reads 8 message words (a 32-byte data block); Snefru-128 loads only `state[0..3]` then reads 12 message words (a 48-byte data block).
- **Final state XOR-back** — Snefru-256 additionally writes `state[4..7]`.
- **Data-block size** — handled in the emit helper.

The round count is FIXED at 8 (`SNEFRU_NUMBER_OF_ROUNDS`); there is NO configurable security/pass parameter. The S-box and core transform are width-independent. Snefru's schedule and state output are BIG-ENDIAN (donor `be2me_32` on message load, `be32_copy` on state output); the message words are assembled via a `SNEFRU_BE32` byte-order helper, and the emit helper byte-swaps the BE state words into the LE-uint frame the `compact_fp` probe expects (per `feedback_be_state_primitives_need_byteswap_in_codegen.md`).

### Block-size asymmetry — the key Tier-4 Snefru risk (R-Tier4-snefru-blocksize)

Unlike HAVAL's uniform 128-byte block, **Snefru-128 processes 48-byte data blocks and Snefru-256 processes 32-byte data blocks** (`data_block_size = 64 - digest_length`). The emit helper's padding and length-field placement therefore differ per width. A single parameterized emit helper `emit_outer_snefru_concat_then_hash(is256, digest_bytes)` (and its Metal twin) bakes the per-width data-block size (DBLK = 48 vs 32) and the length-field byte offsets into two distinct emitted GPU functions (`outer_snefru128_…` / `outer_snefru256_…`). The finalization mirrors `rhash_snefru_final`: zero-pad the last partial block and compress it, then build a length block placing `be2me_32(length >> 29)` at byte offset `DBLK-8` and `be2me_32(length << 3)` at `DBLK-4` (the message length is in BYTES). In the emitted kernels this resolves to `block[40]`/`block[44]` for SNE128 and `block[24]`/`block[28]` for SNE256 — verified directly in the dumped kernel sources.

### Pre-flight + pre-port verification

- **R15 pre-flight** (`codegen/tests/test_snefru_vectors.c` rev 1.1): the librhash Snefru donor was cross-checked against the canonical empty-string Snefru-128 + Snefru-256 vectors (`8617f366…` prefix) AND 228 split-update vs single-update self-consistency cells straddling both data-block boundaries (lengths 31/32/33/47/48/49/63/64/65/95/96/97/127/128/129/200 at 7 chunk sizes per width). **230/230 cells PASS** — confirms the donor's multi-block + partial-fill paths byte-exact for the GPU port oracle.
- **Pre-port C-mirror** (`/tmp/test_snefru_port.c`, one-time harness, not committed): a standalone C reimplementation of the exact GPU `snefru_block` plus the emit-helper full-block-walk + partial-block-pad + length-block placement was cross-checked against the librhash donor for both widths across 28 lengths including all block boundaries: **56/56 cells PASS** byte-exact BEFORE any hardware round-trip. This isolated both the block transform AND the per-width padding/length math (R-Tier4-snefru-blocksize) before the kernel ever ran on a GPU.

### Constant-memory budget (R11)

The 16 KB `SNEFRU_SBOX` brings the cumulative `__constant` footprint to ~42-43 KB of the 64 KB Pascal / Apple Silicon `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE` budget (post-Tier-3 ~26-27 KB + 16 KB), with ~21 KB headroom. The GOST 4 KB derived tables in 5b.4b will bring it to ~46-47 KB, still comfortably within budget.

### Admit-path edits (NOT pure flag-flip — the Tier-4 divergence from HAVAL)

Unlike the HAVAL tiers, the GOST/Snefru `job_to_prim_table[]` rows were NOT pre-staged in 5b.3a. Per the architect's D18.4.a, 5b.4a adds all 3 Tier-4 rows in `codegen/hx_emit_primitives.c` (the one-time non-flag-flip admit edit): `{ 125, HX_PRIM_GOST }` (pre-staged numeric-sorted, HARMLESS until 5b.4b flips its `supported_5a`), `{ 175, HX_PRIM_SNE128 }`, `{ 177, HX_PRIM_SNE256 }`. After these rows land plus the `supported_5a` flips for SNE128/SNE256, the admit predicate, OpenCL `_proto_hexlen`, harness OR-chains, chokepoint, init-gate, and listing path all auto-propagate via D17.4.b with ZERO edits at those sites. The one OTHER non-auto-propagating site is the Metal `metal_gpu_hash_words()` hand-switch (NOT table-driven, unlike the OpenCL `_proto_hexlen`): 5b.4a adds SNE256 to the `return 8` group (32 bytes) and an explicit SNE128 case to the default-equivalent `return 4` arm (16 bytes), per `feedback_metal_hash_words_width_helper.md`.

### Validation matrix (5b.4a Snefru pair)

| Cell group | OpenCL (Pascal GTX 1080, fpga.local) | Metal (Apple M2 Max, dev3.local) |
|------------|--------------------------------------|----------------------------------|
| 2 algos × smoke (8 pw)               | 2/2 PASS | 2/2 PASS |
| 2 algos × large (1,048,576 pw)       | 2/2 PASS | 2/2 PASS |
| 2 algos × edge_maxlen (plen→131)     | 2/2 PASS | 2/2 PASS |

**12/12 Snefru cells PASS** (2 algos × 3 fixtures × 2 backends); over 4 million password-digest verifications byte-exact vs the CPU oracle across both backends, zero missing, zero extras, zero digest mismatches. R-Tier4-snefru-blocksize verified in the dumped kernel sources (DBLK=48/length@block[40,44] for SNE128; DBLK=32/length@block[24,28] for SNE256). D17.4.b auto-propagation confirmed: `mdxfind -h` shows e175 + e177 `[GPU]`-tagged with zero listing-path edits; e125 GOSTMD5PASS correctly remains UNtagged (its row is pre-staged but `supported_5a=0` until 5b.4b).

### Aggregate regression (re-run for 5b.4a)

Prior-tier MAKE_MD5PASS family members (HAVAL 3-pass e127, 4-pass e129, 5-pass e131, Tiger e171, Whirlpool e173, SHA1 e161) × smoke × 2 backends: **12/12 cells PASS** — zero regression. Phase 4 e347 production-dispatcher regression (smoke n_pairs=32 + medium n_pairs=1024 × OpenCL + Metal): **4/4 cells PASS**; 5b.4a does not perturb the e347 codegen path.

## Sub-phase 5b.4b — GOST R 34.11-94 (e125) GPU acceleration — Tier 4 COMPLETE, family 29/30

`gost_block` (the GOST R 34.11-94 "chi" compression function, the legacy Russian standard hash) is now resident in the shared GPU helper sources (`gpu/gpu_common.cl` rev 1.34 and `gpu/metal_common.metal` rev 1.33) as a 256-bit (8-uint) state primitive, plus the 4 KB derived S-box tables (`GOST_SBOX_1..4` / `MTL_GOST_SBOX_1..4`, 4 × 256 uint32). The donor is the in-tree `gosthash/gosthash.c` `gosthash_compress` — which is also the LIVE CPU oracle for e125 (`gosthash()` is called directly from the JOB_GOSTMD5PASS case body; `gosthash.o` is in mdxfind's link list). `e125` GOSTMD5PASS is the FINAL GPU-eligible MAKE_MD5PASS family member.

### TEST S-box set, NOT CryptoPro (R-Tier4-gost-sbox, HIGH)

GOST 28147-89 uses an application-specified 8×16 4-bit S-box. e125 uses the **TEST set** (Saarinen 1998 / RFC 4357; the `sbox[8][16]` at gosthash.c:32-42), NOT the CryptoPro set. The CryptoPro set (`RHASH_GOST_CRYPTOPRO`, the separate non-family e14 GOST-CRYPTO job) produces DIFFERENT digests; a wrong-S-box kernel would silently corrupt every digest. The 4 derived speed-up tables are precomputed host-side from `gosthash_init()`'s logic and baked as `__constant` literals (byte-exact vs the donor's `gost_sbox_1..4`, verified in the C-mirror with 0 mismatches). The MANDATORY R15 pre-flight (`codegen/tests/test_gost_vectors.c` rev 1.1) cross-checks `gosthash()` against the 4 published GOST R 34.11-94 TEST-set vectors (empty/"a"/"abc"/"message digest") AND against rhash `RHASH_GOST` (the librhash default, also the test set) across 22 multi-block lengths — **26/26 cells PASS, ZERO CryptoPro collisions**, distinguishing the TEST set from CryptoPro across the full sweep.

### The most structurally-divergent primitive in Phase 5b (R-Tier4-gost-blockcipher, HIGH)

GOST is the ONLY MAKE_MD5PASS primitive based on a block cipher. `gost_block` mirrors `gosthash_compress`: an 8-iteration "chi" key-schedule loop (the U/V state rotation + the P-transformation building 8×32-bit subkeys) wrapping the GOST 28147-89 32-round Feistel encipher (via the 4 derived S-box tables), followed by the three LFSR product-matrix mixing stages (the 12-round / 16-round / 61-round comments at gosthash.c:191-260 — the single highest typo risk, transcribed verbatim). Beyond the block, GOST carries a **running mod-2^256 checksum `sum[8]`** accumulated across every data block (with the donor's `c = (c<a)||(c<b)` carry propagation) AND a **dual finalization**: after the data blocks, compress the 256-bit bit-length block, then compress the accumulated `sum[8]` checksum block (`gosthash_final:358-359`). The state output is little-endian byte order, so the probe `h0..h3 = state[0..3]` directly (no byte-swap). These cross-block-state mechanics are unique among the family primitives; the bespoke emit helper `emit_outer_gost_concat_then_hash` (and its Metal twin) carries the `sum[8]` accumulation and emits both finalization compressions.

### C-mirror before hardware (the highest-transcription-risk primitive)

Per the C-mirror discipline, the EXACT `gost_block` + `sum[8]` carry + dual finalization were reimplemented in plain C (`/tmp/test_gost_port.c`, uncommitted) and validated **27/27 byte-exact vs `gosthash()`** across lengths straddling the 32-byte block boundary (31/32/33/63/64/65/…) BEFORE any GPU code was written; the precise unified-loop control flow the emitted kernel uses (full + partial blocks in one loop) was separately validated **20/20 byte-exact**. The inserted OpenCL `gost_block` body + macros were then differentially confirmed byte-identical to the validated C-mirror.

### Const budget

The GOST 4 KB derived tables bring the cumulative `__constant` footprint to ~46-47 KB of the 64 KB Pascal / Apple Silicon `CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE` budget (post-Snefru ~42-43 KB + 4 KB), with ~17 KB headroom. JIT-compiled clean on both backends.

### Admit wiring (5b.4b)

The `{ 125, HX_PRIM_GOST }` `job_to_prim_table` row was pre-staged in 5b.4a, so 5b.4b's admit edit is the single `supported_5a` 0→1 flip for `HX_PRIM_GOST` in `codegen/hx_emit_primitives.c`. The admit predicate, OpenCL `_proto_hexlen`, harness OR-chains, chokepoint, init-gate, and listing path all auto-propagate via D17.4.b with ZERO edits. The one non-auto-propagating site is the Metal `metal_gpu_hash_words()` hand-switch: 5b.4b adds JOB_GOSTMD5PASS to the `return 8` group (32 bytes). The bespoke `emit_outer_gost_concat_then_hash` helpers were added to `codegen/hx_emit_opencl.c` (rev 1.18) + `hx_emit_metal.c` (rev 1.17), each wired at 4 sites (helper-name switch, call-line tree, FATAL filter widened to 29, dispatch switch); the oracle (`gosthash()` direct, return 32) + dlen (32) arms were added to `mdxfind.c` (rev 1.518).

### Validation matrix (5b.4b GOST)

| Fixture | n_pass | OpenCL (fpga GTX 1080) | Metal (dev3 M2 Max) |
|---|---|---|---|
| e125_smoke | 8 | PASS | PASS |
| e125_medium | 1,024 | PASS | PASS |
| e125_large | 1,048,576 | PASS | PASS |
| e125_edge_minlen (1–4) | 128 | PASS | PASS |
| e125_edge_maxlen (56–128) | 128 | PASS | PASS |

**10/10 GOST cells PASS** (5 fixtures × 2 backends); over 4.2 million password-digest verifications byte-exact vs the `gosthash()` CPU oracle across both backends, zero missing, zero extras, zero digest mismatches. R-Tier4-gost-sbox verified: the byte-exact match against the TEST-set oracle (a CryptoPro-encoded kernel would have produced 100% digest mismatches) confirms the correct S-box. The edge_maxlen fixture (pass 56–128 → total 88–160 bytes = up to 5 GOST blocks) exercises the `sum[8]` checksum carry across multiple data blocks plus the dual finalization. D17.4.b auto-propagation confirmed: `mdxfind -h` shows e125 `[GPU]`-tagged with zero listing-path edits.

### Aggregate regression (re-run for 5b.4b)

Prior-tier MAKE_MD5PASS family members (Snefru-128 e175, Snefru-256 e177, HAVAL 5-pass e131, Tiger e171, Whirlpool e173, SHA1 e161) × smoke × 2 backends: **12/12 cells PASS** — zero regression. Phase 4 e347 production-dispatcher regression (smoke n_pairs=32 + medium n_pairs=1024 × OpenCL + Metal): **4/4 cells PASS**; 5b.4b does not perturb the e347 codegen path.

### Tier 4 COMPLETE — family milestone 29/30

With e125 GOST shipped, all 29 GPU-eligible MAKE_MD5PASS members are `[GPU]`-tagged: e120, e122, **e125**, e127, e129, e131, e133, e135, e137, e139, e141, e143, e145, e147, e149, e151, e153, e155, e157, e159, e161, e163, e165, e167, e169, e171, e173, e175, e177. The ONLY remaining CPU-only family member is `e123` MD5MD5PASS — the multi-emit outlier (canonical + colon variant), deferred to its own future sub-phase. The MAKE_MD5PASS primitive-lift roadmap (Tiers 1-4) is now complete.

---

# mdxfind v1.506 — Phase 5b Tier 3 COMPLETE (sub-phases 5b.3a + 5b.3b + 5b.3c): full 15-variant HAVAL MAKE_MD5PASS family GPU acceleration (source-ready)

Source: mdxfind.c rev 1.516, gpu/gpu_common.cl rev 1.32, gpu/gpu_common_str.h rev 1.24, gpu/metal_common.metal rev 1.31, gpu/metal_common_str.h rev 1.10, gpu/gpu_opencl.c rev 1.196, gpu_metal.m rev 1.119, gpu/gpu_codegen_eligible.c rev 1.6, gpu/gpujob_opencl.c rev 1.155, gpu/gpujob_metal.m rev 1.37, codegen/hx_emit_primitives.c rev 1.8, codegen/hx_emit_primitives.h rev 1.2, codegen/hx_emit_opencl.c rev 1.16, codegen/hx_emit_metal.c rev 1.15, codegen/tests/run_validation_family_md5pass.sh rev 1.9, codegen/tests/test_haval_paper_vectors.c rev 1.1 (NEW 5b.3a R15 regression canary), codegen/tests/family_md5pass/{e127,e133,e139,e145,e151}_smoke.txt rev 1.1 (NEW 5b.3a), codegen/tests/family_md5pass/{e129,e135,e141,e147,e153}_smoke.txt rev 1.1 (NEW 5b.3b), codegen/tests/family_md5pass/{e131,e137,e143,e149,e155}_smoke.txt rev 1.1 (NEW 5b.3c).

Window: 2026-05-27.

Phase 5b Tier 3 lifts the HAVAL primitive family — a parameterized family of 15 variants (5 digest widths × 3 pass counts) — into the shared GPU helper sources and wires it into the MAKE_MD5PASS family codegen pipeline. Tier 3 ships in three pass-count sub-phases per the architect's D17.6.b recommendation: **5b.3a ships the 5 three-pass variants** (`e127` HAV128, `e133` HAV160/3, `e139` HAV192/3, `e145` HAV224/3, `e151` HAV256); **5b.3b ships the 5 four-pass variants** (`e129` HAV128/4, `e135` HAV160/4, `e141` HAV192/4, `e147` HAV224/4, `e153` HAV256/4); **5b.3c ships the 5 five-pass variants** (`e131` HAV128/5, `e137` HAV160/5, `e143` HAV192/5, `e149` HAV224/5, `e155` HAV256/5). With 5b.3c, Tier 3 is COMPLETE: all 15 HAVAL variants (e127 through e155) are GPU-eligible. The MAKE_MD5PASS family now has 26 GPU-eligible members (86.7% family coverage); 4 remain CPU-only — the `e123` MD5MD5PASS multi-emit outlier and the 3 Tier 4 primitives (snefru × 2, gost).

This is a SOURCE-READY ship: all source is checked in to RCS on the iMac authoritative tree and validated on both backends. The version bump, GitHub release, and `mdxfind-release` run are deferred to a single Tier 3 release event after this 5b.3c source check-in.

## Sub-phase 5b.3a — 3-pass HAVAL (e127 / e133 / e139 / e145 / e151) GPU acceleration

`haval3_block` (HAVAL 3-pass compression, Zheng-Pieprzyk-Seberry 1993) is now resident in the shared GPU helper sources (`gpu/gpu_common.cl` and `gpu/metal_common.metal`) as an 8-uint state primitive over a 128-byte block. HAVAL's `__constant` footprint is minimal: the 32-byte 8-uint initial value (`HAVAL_IV` / `MTL_HAVAL_IV`); the 96 round constants are inlined as compile-time hex literals at each round step (matching the donor's structure and letting the JIT compiler fold them into immediate operands). The donor implementation is the public-domain `mhash-0.9.9.9/lib/haval.c` `havalTransform3` (Paulo S.L.M. Barreto, 1998). Pre-flight R15 cross-verification against `sph_haval` (mdxfind's CPU oracle) confirmed standard-conformance across all 15 published empty-input HAVAL paper vectors (5 widths × 3 passes) plus 105 split-update vs single-update self-consistency cells over multi-block boundary inputs (`codegen/tests/test_haval_paper_vectors.c` rev 1.1, **120/120 cells PASS**).

A single parameterized emit helper `emit_outer_haval_concat_then_hash` (and its Metal twin) handles all 15 HAVAL variants: it bakes the per-variant `(passes, digest_bytes)` tuple into one specialized GPU function at C-emit time. This is the architect's D17.1.a recommendation — a single source of truth for the entire HAVAL family rather than 15 bespoke helpers. HAVAL's structural divergence from the prior MD-family primitives is intrinsic and was carefully transcribed:

- **128-byte block** (twice the conventional 64-byte block), 32 LE-packed uint32 message words.
- **Pad-toggle byte is `0x01`, NOT `0x80`** (donor `havalFinal:760`, explicitly cited in the emit code). Every other primitive in `gpu_common.cl` uses the conventional `0x80` MD/SHA padding bit; HAVAL is the exception, and a careless copy would silently corrupt every digest.
- **Variant-parameter encoding at `block[118..119]`** (donor `havalFinal:786-790`): each `(width, passes)` tuple produces a distinct 2-byte encoding `(version=1) | (passes<<3) | (digest_bits<<6)` and `digest_bits>>2`. Computed at C-emit time and baked as literal constants. For the five 3-pass variants `block[118]=0x19` uniformly; `block[119]` is `0x20/0x28/0x30/0x38/0x40` for the 128/160/192/224/256-bit widths respectively (verified across all 5 emitted kernels).
- **Post-compression digest fold**, JIT-specialized per width so each emitted kernel carries exactly one fold branch (no runtime conditional): the heavy 128-bit byte-redistribution fold, the 160-bit ROTR fold, the 192-bit 5-bit-slice fold, the 224-bit byte-slot-shift fold (donor `havalFinal:816-902`), and the trivial 256-bit direct output (no fold).

HAVAL state is little-endian-native (matching the donor and the published test vectors), so the digest extract is a direct `state[0..3]` read with no byte-swap epilogue. The 4-uint probe carries the first 16 bytes to the hit record; the CPU recompute path (already wired since the SHA-512/Whirlpool tiers) supplies the full digest for the 20/24/28/32-byte widths on hit.

### Pre-port C-mirror validation

A standalone C reimplementation of the exact GPU `haval3_block` + emit-helper padding/fold logic was cross-checked against `sph_haval` before any hardware round-trip: **60/60 cells PASS** (5 widths × 12 inputs including the multi-block boundary cases at 84/85/86/118/119/120-byte plaintexts). This isolated the compression-body transcription, the `0x01` pad toggle, the `block[118..119]` parameter encoding, and all five per-width folds as byte-exact before the kernel ever ran on a GPU.

### Load-bearing prerequisites shipped in 5b.3a

5b.3a also lands four prerequisites that benefit Tier 3 + Tier 4 + beyond:

1. **D17.4.b table-driven admit refactor** — a new `hx_primitive_for_job()` lookup helper in `codegen/hx_emit_primitives.c` (hand-built JOB-enum → outer-primitive map) collapses four previously-hand-coded widening sites onto one truth source: the `gpu_codegen_kernelb_family_md5pass_eligible()` admit predicate, the OpenCL + Metal harness OR-chains in `mdxfind.c`, and the `_proto_hexlen` digest-width switch in `gpu/gpujob_opencl.c`. After this refactor each site is a one-liner querying `hx_primitive_for_job()` + `hx_primitive_is_supported_5a()`; future Tier ships (5b.3b, 5b.3c, Tier 4) flip a single `supported_5a` flag in `prim_table[]` and all four sites auto-propagate with zero further edits.
2. **Slot-map cap bump 16 → 64** — the per-JOB codegen program/PSO slot map (`hx_codegen_slots[]` in `gpu/gpu_opencl.c`, `mtl_codegen_slots[]` in `gpu_metal.m`) is bumped from 16 to 64. The post-Tier-3 active codegen entry count (11 prior + 15 HAVAL + e347 = 27) comfortably fits, with headroom through Tier 4.
3. **prim_table alias rows** — bare callnames `hav128` and `hav256` (used by the e127/e151 catalog entries, without the `_3` suffix) are aliased to the canonical `HX_PRIM_HAV128_3` / `HX_PRIM_HAV256_3` ids, resolving a catalog-callname-vs-prim-table mismatch that would otherwise FATAL the emit for those two algorithms.
4. **JIT-only dump-harness `_with_common` fix** — a latent bug in the non-VALIDATE dump harness path (which routed only e347 through the gpu_common-prepending JIT helper) is fixed to route all MAKE_MD5PASS family members through `_with_common` too. The bug was masked through Tiers 1-2 because family validation always used `MDXFIND_HX_CODEGEN_VALIDATE=1` (which uses the correct `_with_common_keep` dispatch path).

### Validation matrix (5b.3a 3-pass HAVAL ship)

| Cell group | OpenCL (Pascal GTX 1080, fpga.local) | Metal (Apple M2 Max, dev3.local) |
|------------|--------------------------------------|----------------------------------|
| 5 algos × smoke (8 pw)               | 5/5 PASS | 5/5 PASS |
| 5 algos × large (1,048,576 pw)       | 5/5 PASS | 5/5 PASS |
| 5 algos × edge_maxlen (plen→131)     | 5/5 PASS | 5/5 PASS |

**30/30 HAVAL cells PASS** (5 algos × 3 fixtures × 2 backends); ~10.5 million password-digest verifications byte-exact vs the CPU oracle across both backends. The e127 (128-bit fold, the most complex) full 5-fixture validation (smoke + medium + large + edge_minlen + edge_maxlen × 2 backends = 10 cells) all PASS, sanity-checking the parameterized helper across the full fixture range.

## Aggregate Phase 5a + Tier 1 + Tier 2 regression (re-run for 5b.3a)

11 prior GPU-eligible MAKE_MD5PASS family members (`e120`, `e122`, `e157`, `e159`, `e161`, `e163`, `e165`, `e167`, `e169`, `e171`, `e173`) × smoke fixture × 2 backends: **22/22 cells PASS** — the D17.4.b table-driven admit refactor introduced zero regression. Phase 4 e347 production-dispatcher regression: **4/4 cells PASS** (smoke + large × OpenCL + Metal); 5b.3a does not perturb the e347 codegen path.

## Sub-phase 5b.3b — 4-pass HAVAL (e129 / e135 / e141 / e147 / e153) GPU acceleration

`haval4_block` (HAVAL 4-pass compression) is now resident in the shared GPU helper sources (`gpu/gpu_common.cl` rev 1.31, `gpu/metal_common.metal` rev 1.30) as an 8-uint state primitive over a 128-byte block, positioned adjacent to `haval3_block`. The donor is the same public-domain `mhash-0.9.9.9/lib/haval.c` `havalTransform4` (Paulo S.L.M. Barreto, 1998). A critical structural point: HAVAL's per-step F-function argument orderings and message-word permutation schedule in passes 1–3 of the 4-pass core DIFFER from the 3-pass core — the schedule is pass-count specific — so `haval4_block` is a verbatim transcription of `havalTransform4`, not a reuse of the `haval3_block` passes plus an extra pass. Pass 4 adds the F4 round function with 32 more round constants (`0x7A325381 … 0x137A3BE4`) and the feedforward in its final 8 steps. The `MTL_HAVAL_F4` macro (added in 5b.3a) is now exercised; HAVAL's F1–F5 are pure XOR/AND/OR/NOT compositions, so no scalar `bitselect` is involved (R11 not in play).

The 5 four-pass variants reuse the existing parameterized emit helper `emit_outer_haval_concat_then_hash` unchanged — 5b.3b simply wires the 5 `HX_PRIM_HAV*_4` ids into the dispatch / FATAL-filter / helper-name switches with `passes=4`. The helper already bakes the pass count into both the compression-function call (`haval4_block`) and the `block[118]` variant-parameter byte. For the five 4-pass variants `block[118]=0x21` uniformly (vs the 3-pass `0x19`, because the `(passes<<3)` field changes from `0x18` to `0x20`); `block[119]` is `0x20/0x28/0x30/0x38/0x40` for the 128/160/192/224/256-bit widths respectively (verified in the emitted kernel source for e129 and e153). The per-width digest folds are pass-count-independent (HAVAL's final fold is parameterized only by digest width) and were validated unchanged.

Pre-port R15 cross-verification: a standalone C-mirror of `haval4_block` plus the family padding/fold was validated against `sph_haval` for all 5 widths × 12 inputs (including the multi-block boundary cases at 84/85/86/118/119/120 bytes), **60/60 cells PASS** before any GPU code shipped.

### D17.4.b auto-propagation (5b.3b is flag-flip-only)

5b.3b is the first sub-phase to fully exercise the D17.4.b table-driven admit refactor shipped in 5b.3a. The `job_to_prim_table` already carried the 5 `HAV*_4` JOB-enum rows, so flipping the 5 `prim_table.supported_5a` flags from 0 to 1 auto-propagated to ALL downstream gates with zero edits: the admit predicate, the OpenCL + Metal harness OR-chains, the OpenCL `_proto_hexlen` digest-width switch, and the `mdxfind -h` `[GPU]` listing. The 5 new entries (`e129`, `e135`, `e141`, `e147`, `e153`) appear `[GPU]`-tagged in `mdxfind -h` with no edit to the listing path. The only hand edits were: lift `haval4_block` + its Metal twin, flip the 5 flags, wire the emit-dispatch arms, add the 5 oracle + dlen arms, add the 4 `metal_gpu_hash_words` arms (HAV128/4 reuses the 4-word default), and the fixtures.

### Validation matrix (5b.3b 4-pass HAVAL ship)

**30/30 HAVAL 4-pass cells PASS** (5 algos × 3 fixtures [smoke + large + edge_maxlen] × 2 backends); ~31.5 million password-digest verifications byte-exact vs the CPU oracle (Pascal GTX 1080 OpenCL + Apple M2 Max Metal), zero digest mismatches, zero extras, zero missing. Prior-tier regression (3-pass HAVAL + MD2 + SHA1 + Tiger + Whirlpool × smoke × 2 backends): **18/18 cells PASS**. Phase 4 e347 production-dispatcher regression: **4/4 cells PASS** (smoke + large × OpenCL + Metal); 5b.3b does not perturb the e347 path.

## Sub-phase 5b.3c — 5-pass HAVAL (e131 / e137 / e143 / e149 / e155) GPU acceleration — Tier 3 COMPLETE

`haval5_block` (HAVAL 5-pass compression) is now resident in the shared GPU helper sources (`gpu/gpu_common.cl` rev 1.32, `gpu/metal_common.metal` rev 1.31) as an 8-uint state primitive over a 128-byte block, positioned adjacent to `haval4_block`. The donor is the same public-domain `mhash-0.9.9.9/lib/haval.c` `havalTransform5` (Paulo S.L.M. Barreto, 1998) — the longest of the three compression cores at ~200 lines. A critical structural point: HAVAL's per-step F-function argument orderings and message-word permutation schedule in passes 1–4 of the 5-pass core DIFFER from BOTH the 3-pass AND 4-pass cores — the schedule is pass-count specific across all passes — so `haval5_block` is a verbatim transcription of `havalTransform5`, not a reuse of the earlier passes plus extra rounds. Pass 5 adds the F5 round function with 32 more round constants (`0xBA3BF050 … 0x409F60C4`) and the feedforward in its final 8 steps. The `MTL_HAVAL_F5` macro (added in 5b.3a) is now exercised; HAVAL's F1–F5 are pure XOR/AND/OR/NOT compositions, so no scalar `bitselect` is involved (R11 not in play).

The 5 five-pass variants reuse the existing parameterized emit helper `emit_outer_haval_concat_then_hash` unchanged — 5b.3c simply wires the 5 `HX_PRIM_HAV*_5` ids into the dispatch / FATAL-filter / helper-name switches with `passes=5`. The helper already bakes the pass count into both the compression-function call (`haval5_block`) and the `block[118]` variant-parameter byte. For the five 5-pass variants `block[118]=0x29` uniformly (vs the 4-pass `0x21` and 3-pass `0x19`, because the `(passes<<3)` field is now `0x28`); `block[119]` is `0x20/0x28/0x30/0x38/0x40` for the 128/160/192/224/256-bit widths respectively (verified in the emitted kernel source for e131 and e155). The per-width digest folds are pass-count-independent (HAVAL's final fold is parameterized only by digest width) and were validated unchanged.

A pre-port C-mirror of `haval5_block` + the padding/fold logic (the longest transcription target in Tier 3) was validated byte-exact against `sph_haval` for all 5 widths × 12 inputs (including the multi-block boundary cases at 84/85/86/118/119/120 bytes): **60/60 cells PASS** before any GPU hardware ran, catching any 5-pass-schedule transcription error in the plain-C mirror first.

### D17.4.b auto-propagation (5b.3c is flag-flip-only)

5b.3c is flag-flip-only on the admit path, like 5b.3b. The `job_to_prim_table` already carried the 5 `HAV*_5` JOB-enum rows, so flipping the 5 `prim_table.supported_5a` flags from 0 to 1 auto-propagated to ALL downstream gates with zero edits: the admit predicate, the OpenCL + Metal harness OR-chains, the OpenCL `_proto_hexlen` digest-width switch, and the `mdxfind -h` `[GPU]` listing. The 5 new entries (`e131`, `e137`, `e143`, `e149`, `e155`) appear `[GPU]`-tagged in `mdxfind -h` with no edit to the listing path. The only hand edits were: lift `haval5_block` + its Metal twin, flip the 5 flags, wire the emit-dispatch arms, add the 5 oracle + dlen arms, add the 4 `metal_gpu_hash_words` arms (HAV128/5 reuses the 4-word default), and the fixtures. Files left UNCHANGED in 5b.3c: `gpu/gpu_opencl.c`, `gpu_metal.m`, `gpu/gpu_codegen_eligible.c`, `gpu/gpujob_opencl.c`, `codegen/hx_emit_primitives.h` (slot cap already 64; admit/_proto_hexlen/harness all auto-propagate via D17.4.b).

### Validation matrix (5b.3c 5-pass HAVAL ship)

**30/30 HAVAL 5-pass cells PASS** (5 algos × 3 fixtures [smoke + large + edge_maxlen] × 2 backends); ~10.5 million password-digest verifications byte-exact vs the CPU oracle (Pascal GTX 1080 OpenCL + Apple M2 Max Metal), zero digest mismatches, zero extras, zero missing. Prior-tier regression (3-pass + 4-pass HAVAL + MD2 + SHA1 + Tiger + Whirlpool × smoke × 2 backends): **22/22 cells PASS**. Phase 4 e347 production-dispatcher regression: **4/4 cells PASS** (smoke + large × OpenCL + Metal); 5b.3c does not perturb the e347 path.

### Tier 3 COMPLETE milestone

With 5b.3c shipped, all 15 HAVAL MAKE_MD5PASS variants (e127 through e155) are `[GPU]`-tagged and validated on both backends. The MAKE_MD5PASS family is now 26/30 GPU-eligible (86.7%): MD2, MD4, 15 × HAVAL, RMD128, RMD160, SHA1, SHA224, SHA256, SHA384, SHA512, Tiger, Whirlpool. The 4 remaining CPU-only members are the `e123` MD5MD5PASS multi-emit outlier (deferred to a separate multi-emit sub-phase) and the 3 Tier 4 primitives (snefru × 2, gost). The single Tier 3 release event (version bump + GitHub release + `mdxfind-release`) ships on top of this source-ready check-in.

---

# mdxfind v1.505 — Phase 5b Tier 2: Whirlpool / Tiger MAKE_MD5PASS family GPU acceleration

Source: mdxfind.c rev 1.510, gpu/gpu_common.cl rev 1.29, gpu/gpu_common_str.h rev 1.21, gpu/metal_common.metal rev 1.28, gpu/metal_common_str.h rev 1.7, gpu/gpu_codegen_eligible.c rev 1.5, gpu/gpujob_opencl.c rev 1.154, gpu/gpujob_metal.m rev 1.34, codegen/hx_emit_primitives.c rev 1.5, codegen/hx_emit_opencl.c rev 1.13, codegen/hx_emit_metal.c rev 1.12, codegen/tests/run_validation_family_md5pass.sh rev 1.6, codegen/tests/family_md5pass/e173_smoke.txt rev 1.1 (NEW 5b.2a), codegen/tests/family_md5pass/e171_smoke.txt rev 1.1 (NEW 5b.2b), codegen/tests/test_wrl_nessie.c rev 1.1 (NEW 5b.2a regression), codegen/tests/test_tiger_nessie.c rev 1.1 (NEW 5b.2b regression).

Window: 2026-05-27.

Phase 5b Tier 2 lifts two MAKE_MD5PASS outer primitives — Whirlpool (`wrl`) and Tiger — into the shared GPU helper sources and wires them into the family codegen pipeline. After Tier 2 the family has 11 GPU-eligible members; 18 remain CPU-only pending Tier 3 (haval × 15) and Tier 4 (snefru + gost) primitive lifts.

## Sub-phase 5b.2a — WRLMD5PASS (e173) GPU acceleration

`wrl_block` (Whirlpool, ISO/IEC 10118-3) is now resident in the shared GPU helper sources (`gpu/gpu_common.cl` and `gpu/metal_common.metal`) as an 8-ulong state primitive with 16 KB of `__constant` S-box tables and 80 bytes of round constants (total Tier 2a constant memory budget: 16.4 KB; comfortable headroom within the 64 KB CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE on both Pascal and Apple Silicon). The donor implementation is `RHash-master/librhash/whirlpool.c` (Aleksey Kravchenko, 2009-2012) and its companion S-box `RHash-master/librhash/whirlpool_sbox.c`; pre-flight R12 cross-verification against the OpenSSL `WHIRLPOOL()` implementation (mdxfind's CPU oracle) confirmed both donors agree byte-for-byte on all 8 published NESSIE test vectors (`codegen/tests/test_wrl_nessie.c` rev 1.1, 16/16 cells PASS).

A bespoke per-primitive emit helper `emit_outer_wrl_concat_then_hash` (and its Metal twin) is added to the codegen pipeline. Whirlpool's structural divergence from SHA-512 is intrinsic: 64-byte block (not 128), 32-byte BE length suffix at M[4..7] (not 16-byte at M[14..15]), all-zero IV (not the SHA-512 IV), and ALWAYS-multi-block (single-block fast path elided — the threshold `32 + plen + 1 + 32 <= 64` never holds for the family use case). The state output is byte-swap-as-ulong then split into LE-uint pair, identical epilogue to the existing SHA-2 64-bit family helper.

### Validation matrix (Whirlpool ship)

| Fixture / Backend       | OpenCL (Pascal GTX 1080) | Metal (Apple Silicon M2 Max) |
|-------------------------|--------------------------|------------------------------|
| family_smoke (8)        | e173 PASS                | e173 PASS                    |
| family_medium (1,024)   | e173 PASS                | e173 PASS                    |
| family_large (1,048,576)| e173 PASS                | e173 PASS                    |
| family_edge_minlen (128)| e173 PASS                | e173 PASS                    |
| family_edge_maxlen (128)| e173 PASS                | e173 PASS                    |

**10/10 cells PASS**; 2,099,712 password-digest verifications byte-exact vs the CPU oracle across both backends.

## Sub-phase 5b.2b — TIGERMD5PASS (e171) GPU acceleration

`tiger_block` (Tiger/192, Anderson + Biham 1996) is now resident in the shared GPU helper sources as a 3-ulong state primitive with 8 KB of `__constant` S-box tables (4 × 256 ulong). Combined with Whirlpool's 16 KB the total Tier 2 constant memory footprint is ~24 KB; comfortable headroom remains on both Pascal and Apple Silicon. The donor implementation is `RHash-master/librhash/tiger.c` (Aleksey Kravchenko, 2007-2012) and its companion S-box `RHash-master/librhash/tiger_sbox.c`; pre-flight R12 cross-verification against `sph_tiger` from libsph (mdxfind's CPU oracle) confirmed both donors agree byte-for-byte on all 7 published NESSIE test vectors plus the 1-million-`a` stress vector (`codegen/tests/test_tiger_nessie.c` rev 1.1, 16/16 cells PASS).

A bespoke per-primitive emit helper `emit_outer_tiger_concat_then_hash` (and its Metal twin) is added to the codegen pipeline. Tiger's structural divergence from both SHA-512 and Whirlpool is intrinsic: LE schedule (M packed lo-byte-first, matching MD-family convention; the opposite of Whirlpool/SHA-2 BE), 8-byte LE length suffix at M[7] only, padding byte `0x01` (legacy Tiger, NOT Tiger2's `0x80`), Tiger initial chaining value (`0x0123456789abcdef`, `0xfedcba9876543210`, `0xf096a5b4c3b2e187`), and a single-block fast path that IS applicable for short passwords (threshold `32 + plen + 1 + 8 <= 64` holds for `plen <= 23`). The state output is the direct LE-uint split from `state[0..1]` — no byte-swap epilogue is needed because Tiger's spec output is LE state direct. The 3-pass round structure (pass1 mul 5 → KeySchedule → pass2 mul 7 with rotated arg order (c, a, b) → KeySchedule → pass3 mul 9 with (b, c, a)) is transcribed verbatim from the donor source, with explicit line-number citations at each pass boundary.

### Validation matrix (Tiger ship)

| Fixture / Backend       | OpenCL (Pascal GTX 1080) | Metal (Apple Silicon M2 Max) |
|-------------------------|--------------------------|------------------------------|
| family_smoke (8)        | e171 PASS                | e171 PASS                    |
| family_medium (1,024)   | e171 PASS                | e171 PASS                    |
| family_large (1,048,576)| e171 PASS                | e171 PASS                    |
| family_edge_minlen (128)| e171 PASS                | e171 PASS                    |
| family_edge_maxlen (128)| e171 PASS                | e171 PASS                    |

**10/10 cells PASS**; 2,099,728 password-digest verifications byte-exact vs the CPU oracle across both backends. Pascal large fixture (1,048,576) walltime = 1s (~1M verifications/s; Tiger is roughly 4-5× faster than Whirlpool on Pascal due to fewer rounds and smaller S-box footprint). Apple Silicon M2 Max large fixture walltime = sub-second.

## Aggregate Phase 5a + Tier 1 + Tier 2 regression

11 GPU-eligible MAKE_MD5PASS family members (`e120`, `e122`, `e157`, `e159`, `e161`, `e163`, `e165`, `e167`, `e169`, `e171`, `e173`) × 5 fixtures (smoke, medium, large, edge_minlen, edge_maxlen) × 2 backends (OpenCL on Pascal GTX 1080, Metal on Apple Silicon M2 Max): **110/110 cells PASS**, ~23 million password-digest verifications byte-exact. Phase 4 e347 production-dispatcher regression: **4/4 cells PASS** (smoke + large × OpenCL + Metal); Tier 2 does not perturb the e347 codegen path.

## `mdxfind -h` GPU tag display now predicate-based

The `[GPU]` suffix on the `mdxfind -h` hash-type listing is now driven by a single composite predicate (`gpu_op_advertise_for_h_listing`) that ORs the actual runtime GPU admit predicates instead of a hand-maintained if-chain of JOB literals. The composite predicate currently calls (1) a linear scan of the `gpu_ops[]` legacy template-path table, (2) an explicit `JOB_MD5MD5SALT` (e347) check covering the Phase 4 codegen production path, and (3) `gpu_codegen_kernelb_family_md5pass_eligible()` covering the Phase 5a/5b MAKE_MD5PASS family. Prior to this release the if-chain had drifted out of sync with the runtime path: the 12 codegen-eligible algorithms shipped in v1.502 and v1.504/v1.505 (`e120`, `e122`, `e157`, `e159`, `e161`, `e163`, `e165`, `e167`, `e169`, `e171`, `e173`, `e347`) were silently un-tagged in the `-h` listing despite being live on the runtime GPU dispatch path. Post-change the listing emits 90 `[GPU]` tags on the OpenCL build (was 78); all 78 pre-existing tags preserved byte-identically. Future Tier 3 and Tier 4 family ships (haval × 15, gost, gost_crypto, sne128, sne256 outer primitives) will auto-update the listing once the family eligibility predicate widens — no further edits to the listing site are needed.

---

# mdxfind v1.504 — Phase 5b Tier 1: MD2 / RMD128 MAKE_MD5PASS family GPU acceleration + RIPEMD-128 / RIPEMD-160 standard-conformance bug fix

Source: mdxfind.c rev 1.507, rmd128.c rev 1.1 (NEW; bug fix), rmd160.c rev 1.1 (NEW; bug fix), gpu/gpu_common.cl rev 1.27, gpu/gpu_common_str.h rev 1.19, gpu/metal_common.metal rev 1.26, gpu/metal_common_str.h rev 1.5, gpu/gpu_codegen_eligible.c rev 1.3, codegen/hx_emit_primitives.c rev 1.3, codegen/hx_emit_opencl.c rev 1.11, codegen/hx_emit_metal.c rev 1.10, codegen/tests/run_validation_family_md5pass.sh rev 1.4, codegen/tests/family_md5pass/e120_smoke.txt rev 1.1 (NEW 5b.1a), codegen/tests/family_md5pass/e157_smoke.txt rev 1.1 (NEW 5b.1b), codegen/tests/test_rmd128_bosselaers.c rev 1.1 (NEW; regression), codegen/tests/test_rmd160_bosselaers.c rev 1.1 (NEW; regression).

Window: 2026-05-27.

Phase 5b Tier 1 lifts two MAKE_MD5PASS outer primitives — MD2 and RMD128 — into the shared GPU helper sources and wires them into the family codegen pipeline. After Tier 1 the family has 9 GPU-eligible members; 20 remain CPU-only pending Tier 2-4 primitive lifts (tiger, wrl, haval×15, gost, gost_crypto, sne128, sne256). During Tier 1 validation a long-standing standard-conformance bug in the in-tree RIPEMD-128 implementation was discovered and fixed; subsequent inspection found the identical bug pattern in the in-tree RIPEMD-160 implementation and that is fixed in the same release. See the "Bug fix" section below.

## Bug fix — RIPEMD-128 / RIPEMD-160 length-encoding standard-conformance

A pair of long-standing length-encoding bugs in the in-tree RIPEMD-128 (`rmd128.c`) and RIPEMD-160 (`rmd160.c`) implementations (both Bosselaers ESAT-COSIC 1996 donor lineage) are fixed this release. The bug pattern is identical in both files: the `RIPEMDxxx()` wrapper's per-block consumer loop modifies the local `len` parameter, and the final `MDfinish()` call was passing the post-loop residual byte count instead of the original total length. Because `MDfinish()` encodes its `lswlen` argument as the message bit-length in the standard MD-family length suffix, the digest of any message longer than 63 bytes was non-standard — the encoded length was `(total_bytes mod 64) * 8` instead of `total_bytes * 8`. The `MDfinish()` header docstring in `rmd128.h` and `rmd160.h` had always documented the intended semantics (`lswlen` is the TOTAL byte length; only `lswlen mod 64` bytes remain in `strptr`), so the bug was solely in the callers.

After the fix, the in-tree RIPEMD-128 and RIPEMD-160 implementations match Bosselaers's 1996 reference values byte-for-byte, including the canonical test vectors:

### RIPEMD-128 (`rmd128.c` rev 1.1)

| Input                                              | Expected (Bosselaers)              | Pre-fix mdxfind                    | Post-fix mdxfind                   |
|----------------------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| empty string                                       | `cdf26213a150dc3ecb610f18f6b38b46` | (unchanged — single-block)         | `cdf26213a150dc3ecb610f18f6b38b46` |
| `"a"`                                              | `86be7afa339d0fc7cfc785e72f578d33` | (unchanged — single-block)         | `86be7afa339d0fc7cfc785e72f578d33` |
| `"abc"`                                            | `c14a12199c66e4ba84636b0f69144c77` | (unchanged — single-block)         | `c14a12199c66e4ba84636b0f69144c77` |
| `"message digest"`                                 | `9e327b3d6e523062afc1132d7df9d1b8` | (unchanged — single-block)         | `9e327b3d6e523062afc1132d7df9d1b8` |
| `a..z` (26 bytes)                                  | `fd2aa607f71dc8f510714922b371834e` | (unchanged — single-block)         | `fd2aa607f71dc8f510714922b371834e` |
| `A..Z+a..z+0..9` (62 bytes)                        | `d1e959eb179c911faea4624c60c5c702` | (unchanged — single-block)         | `d1e959eb179c911faea4624c60c5c702` |
| `"1234567890"` × 8 (80 bytes)                      | `3f45ef194732c2dbb2c4a2c769795fa3` | `1959258deca4645654950534f3537250` | `3f45ef194732c2dbb2c4a2c769795fa3` |
| `"a"` × 1,000,000                                  | `4a7f5723f954eba1216c9d8f6320431f` | (non-conformant)                   | `4a7f5723f954eba1216c9d8f6320431f` |

All 8 RIPEMD-128 vectors PASS post-fix on the standalone regression test `codegen/tests/test_rmd128_bosselaers.c` (rev 1.1 NEW). After the fix, the in-tree `RIPEMD128()` agrees with `sph_ripemd128` (sphlib-3.0), and with every other standard implementation.

### RIPEMD-160 (`rmd160.c` rev 1.1)

| Input                                              | Expected (Bosselaers)                        | Pre-fix in-tree                              | Post-fix in-tree                             |
|----------------------------------------------------|----------------------------------------------|----------------------------------------------|----------------------------------------------|
| empty string                                       | `9c1185a5c5e9fc54612808977ee8f548b2258d31`   | (unchanged — single-block)                   | `9c1185a5c5e9fc54612808977ee8f548b2258d31`   |
| `"a"`                                              | `0bdc9d2d256b3ee9daae347be6f4dc835a467ffe`   | (unchanged — single-block)                   | `0bdc9d2d256b3ee9daae347be6f4dc835a467ffe`   |
| `"abc"`                                            | `8eb208f7e05d987a9b044a8e98c6b087f15a0bfc`   | (unchanged — single-block)                   | `8eb208f7e05d987a9b044a8e98c6b087f15a0bfc`   |
| `"message digest"`                                 | `5d0689ef49d2fae572b881b123a85ffa21595f36`   | (unchanged — single-block)                   | `5d0689ef49d2fae572b881b123a85ffa21595f36`   |
| `a..z` (26 bytes)                                  | `f71c27109c692c1b56bbdceb5b9d2865b3708dbc`   | (unchanged — single-block)                   | `f71c27109c692c1b56bbdceb5b9d2865b3708dbc`   |
| `A..Z+a..z+0..9` (62 bytes)                        | `b0e20b6e3116640286ed3a87a5713079b21f5189`   | (unchanged — single-block)                   | `b0e20b6e3116640286ed3a87a5713079b21f5189`   |
| `"1234567890"` × 8 (80 bytes)                      | `9b752e45573d4b39f4dbd3323cab82bf63326bfb`   | `5f7ffbbfd70ae0b9ad611b7961a32a7646f9c384`   | `9b752e45573d4b39f4dbd3323cab82bf63326bfb`   |
| `"a"` × 1,000,000                                  | `52783243c1697bdbe16d37f97f68f08325dc1528`   | `eb56500397007b3e6e07fe58db85a7ceaa78d37f`   | `52783243c1697bdbe16d37f97f68f08325dc1528`   |

All 8 RIPEMD-160 vectors PASS post-fix on the standalone regression test `codegen/tests/test_rmd160_bosselaers.c` (rev 1.1 NEW). After the fix, the in-tree `RIPEMD160()` agrees with `sph_ripemd160` (sphlib-3.0) and OpenSSL's `RIPEMD160`, and with every other standard implementation.

### Scope and fix shape

Both fixes are confined to the `RIPEMDxxx()` wrapper functions in `rmd128.c` and `rmd160.c`: each saves the original message length into local `total_lswlen` and `total_mswlen` before the per-block consumer loop, then passes those totals to `MDfinish()`. No signature changes; `MDfinish()` is unchanged in either file.

**Scope of RIPEMD-128 behavior change**: every `mdxfind` catalog entry that calls `RIPEMD128()` (e16 RMD128, e156 RMD128MD5, e157 RMD128MD5PASS, e210 HMAC-RMD128, e231 RMD128MD5MD5, e498 RMD128MD4, e714 SHA1RMD128) produces different output post-fix for any RIPEMD-128 input longer than 63 bytes. For inputs ≤ 63 bytes (single-block, no consumer-loop iterations) the digest is byte-identical to pre-fix.

**Scope of RIPEMD-160 behavior change**: the in-tree `RIPEMD160()` symbol is shadowed at link time by OpenSSL's `libcrypto.a` `RIPEMD160` in the `mdxfind` binary (the in-tree `rmd160.o` is only linked into the separate `mdxocl` target). The `mdxfind` binary therefore was already using a standard-conformant RIPEMD-160; the fix to `rmd160.c` restores the in-tree implementation to standard conformance for any caller that links it directly (including `mdxocl`, the standalone Bosselaers regression test, and any future build target that links `rmd160.o`). The five `mdxfind` catalog entries that mention `RIPEMD160()` in their case bodies (e17 RMD160, e158 RMD160MD5, e159 RMD160MD5PASS, e196 MD5RMD160, e746 SHA1RMD160TRUNC) and the `oracle_compute_md5pass_family()` arm for e159 all resolve `RIPEMD160` to the OpenSSL implementation in the production binary, so their behavior is unchanged. HMAC paths (e211 HMAC-RMD160, e798 HMAC-RMD160 KPASS) use the `mhash` library's `MHASH_RIPEMD160` and are likewise unchanged.

The user has confirmed (2026-05-27) that no production solved-hash archives include RIPEMD-128 or RIPEMD-160 computations on inputs longer than 60 bytes for any catalog entry, so neither fix orphans any existing cracking workflow.

The GPU codegen emit helpers (`emit_outer_rmd128_concat_then_hash` in `hx_emit_opencl.c` rev 1.11 and `emit_outer_rmd128_concat_then_hash_metal` in `hx_emit_metal.c` rev 1.10) had carried a `bug_lswlen` workaround introduced during the initial 5b.1b validation pass to keep the GPU output byte-exact with the (then-buggy) CPU oracle. That workaround is reverted this release: the length suffix in both backends now uses `total_len * 8` unconditionally, matching the now-conformant CPU oracle.

The corresponding RMD-160 GPU codegen emit helpers (`emit_outer_rmd160_concat_then_hash` / `emit_outer_rmd160_concat_then_hash_metal`) never carried a `bug_lswlen` workaround — they were always standard-correct because the production `mdxfind` binary's `RIPEMD160` was already coming from OpenSSL. No emit-helper revert is required for RMD-160; no GPU file changes for the RMD-160 fix.

## Sub-phase 5b.1a — MD2MD5PASS (e120) GPU acceleration

`e120 MD2MD5PASS` (`md2(md5_hex(pass) . pass)`) joins the 7 Phase 5a family members on GPU. Eighth GPU-eligible member of the 30-entry MAKE_MD5PASS family.

The MD2 outer-hash primitive (`md2_block`) is new to both `gpu/gpu_common.cl` and `gpu/metal_common.metal` this sub-phase. The 256-byte MD2 S-box (RFC 1319 Table T) lives in `__constant` address space on OpenCL and `constant` address space on Metal. The block compression matches B-Con and sph_md2 reference byte-for-byte (18 rounds × 48-byte state + 16-byte checksum). The emit helper (`emit_outer_md2_concat_then_hash`) is bespoke because MD2's structure diverges from the MD4/MD5 family (16-byte block, PKCS padding, checksum-as-final-block).

## Sub-phase 5b.1b — RMD128MD5PASS (e157) GPU acceleration

`e157 RMD128MD5PASS` (`rmd128(md5_hex(pass) . pass)`) becomes the ninth GPU-eligible member of the family.

The RMD-128 outer-hash primitive (`rmd128_block`) is new to both `gpu/gpu_common.cl` (4-uint state, dual pipeline, reuses RMD_F1..F4 macros from the existing RMD-160 helper, defines local RMD128_STEP / LL1..LL4 / RR1..RR4 round-K macros) and `gpu/metal_common.metal` (Metal twin via hand-port; same structure with RMD128_STEP_METAL / LL1M..LL4M / RR1M..RR4M). The dual pipeline runs **left line F1->F2->F3->F4 and right line F4->F3->F2->F1** per Bosselaers Table 4 — the right-line ordering is inverted relative to RMD-160 (which uses F5->F4->F3->F2->F1) and is the highest-risk transcription point per spec R2.

The emit helper (`emit_outer_rmd128_concat_then_hash`) clones the existing RMD-160 helper with state width adjusted from 5 uints to 4 uints. The standard-conformant length suffix (`total_len * 8`) is encoded directly; see the "Bug fix" section above for the resolution of the temporary `bug_lswlen` workaround that was carried during the initial 5b.1b validation pass.

## Validation matrix

Sub-phase 5b.1a (MD2) 10-cell Tier 1 matrix: 5 fixtures × 2 backends (OpenCL Pascal GTX 1080 + Metal Apple M2 Max) × e120 → 10/10 PASS, 2,097,792 password-digest verifications byte-exact vs CPU oracle.

Sub-phase 5b.1b (RMD128) 10-cell Tier 1 matrix: same shape × e157 → 10/10 PASS, 2,099,728 password-digest verifications byte-exact vs CPU oracle (verified with the rmd128.c bug fix and the matching emit-helper revert in place; the previously-failing `family_edge_maxlen` cell — 128 plaintexts with plens 56-128 exercising the multi-block path — now passes on both backends with zero diffs).

Aggregate 90-cell post-Tier-1 family regression (9 family members × 5 fixtures × 2 backends): 90/90 PASS. Zero regressions on the 7 Phase 5a members (e122 e159 e161 e163 e165 e167 e169); e120 and e157 admit cleanly without disturbing any of them. None of the non-RMD128 family entries are affected by the RIPEMD-128 bug fix.

Phase 4 e347 production-dispatcher regression (2 fixtures × 2 backends): 4/4 PASS, byte-exact unchanged. e347 (MD5MD5SALT) does not use RIPEMD-128, and the e347 regression confirms no collateral damage from the rmd128.c edit.

| Fixture / Backend     | OpenCL e120 | OpenCL e157 | Metal e120 | Metal e157 |
|-----------------------|-------------|-------------|------------|------------|
| family_smoke (8)      | PASS        | PASS        | PASS       | PASS       |
| family_medium (1024)  | PASS        | PASS        | PASS       | PASS       |
| family_large (1048576)| PASS        | PASS        | PASS       | PASS       |
| family_edge_minlen    | PASS        | PASS        | PASS       | PASS       |
| family_edge_maxlen    | PASS        | PASS        | PASS       | PASS       |

The `gpu_codegen_kernelb_family_md5pass_eligible()` admit-predicate widens from 7 arms (Phase 5a) to 9 arms (adding case 120 and case 157). The aggregate runner `run_validation_family_md5pass.sh` widens its `FAMILY_JOBS` list from 7 entries to 9 (numeric-sorted: `120 122 157 159 161 163 165 167 169`).

## Coming in Tier 2-4 (future releases)

Tier 2 (tiger + wrl), Tier 3 (haval × 15 variants), Tier 4 (snefru + gost + gost_crypto) ship per their corresponding `<primitive>_block` lifts into `gpu_common.cl` and `metal_common.metal`. The Phase 5b scoping memo lays out per-tier priority and per-family architect spec discipline.

---

# mdxfind v1.503 — Mask iteration scale fixes + hx.8 catalog doc corrections

Source: mdxfind.c rev 1.505, gpu/gpu_opencl.c rev 1.195, gpu_metal.m rev 1.118, hx.8 rev 1.12, hashpipe.c rev 1.90.

Window: 2026-05-26.

## Mask iterator scale fixes

The mask iterator's progress counter and per-thread loop bound were both 32-bit signed. With keyspaces commonly exceeding 2^31 (e.g., `?d?d?d?d?d?d?d?d?d?d` = 10^10 ≈ 2.3 × 2^32), the iterator would silently wrap, exit early, or produce incorrect candidate counts. Both `number_iter` and `loop_bound` are now `uint64_t` end-to-end; full 10^10+ keyspaces traverse correctly.

Multi-thread mask attacks now actually use the requested thread count. The previous chunking math computed the per-thread range using the (overflow-prone) 32-bit counter, which on large keyspaces collapsed all work onto thread 0 while the other workers spun idle. The chunker has been rewritten over the widened 64-bit counter so each worker takes a proportional non-overlapping slice; `-T N` scales as expected on long mask runs.

## MAX_MASK_POS raised (CPU 16 → 256; GPU stays at 16 with explicit cap symbol)

`MAX_MASK_POS` (CPU-side per-side mask position cap) raised from 16 to 256. The GPU per-side cap is now an explicit, separately named symbol `MAX_MASK_POS_GPU_SIDE = 16` documenting what the bundled GPU kernels actually support (the kernels' unrolled per-position state still hard-codes 16, and raising that requires a coordinated multi-file kernel edit — see Known issues below).

Previously, masks longer than 16 positions on either side were silently truncated by the parser, producing a much smaller keyspace than the user asked for with no diagnostic. The parser now emits a **FATAL** message naming the offending mask, its position count, and the active per-side cap, then exits non-zero.

GPU upload sites in both OpenCL (`gpu/gpu_opencl.c`) and Metal (`gpu_metal.m`) gained a runtime guard: when a mask attack on a GPU target exceeds `MAX_MASK_POS_GPU_SIDE`, the run is gracefully demoted to CPU with a one-line warning naming the mask and the per-side cap. No crash, no silent truncation.

## hx.8 catalog doc corrections

Three entries in the hx algorithm catalog (manual page `hx(8)`) carried stale expressions that didn't match what mdxfind actually computes. Corrected this release:

- `e232 MD5BASE64` — was `base64(md5_bin(pass))`, now `md5(base64(pass))`
- `e539 MYSQL5MD5` — was `"*" . upper(sha1(sha1_bin(md5(pass))))`, now `sha1(sha1_bin(md5(pass)))`
- `e993 WPBCRYPT` — was `bcrypt(pass, salt, 10)`, now `bcrypt(base64(hmac_sha384_bin(pass, "wp-sha384")), salt, N)` (matches hashcat m35500)

No compute changes — only the documented expressions were drifted. CPU and GPU dispatch were always computing the corrected forms above; the manual now reflects reality.

The canonical hx language manual is published at <https://www.mdxfind.com/hx.pdf>. The hx parser and algorithm catalog source live in the upstream hashpipe project: <https://github.com/Cynosureprime/hashpipe>.

## Known issues / scope

- The GPU per-side mask cap stays at 16 this release. Raising it requires coordinated edits across eight kernel sources, their header companions, the `cl2str.py` / `cl2metal.py` string-literal serializers, and the host upload enums. The work is straightforward but mechanical, and not blocking; deferred to a future release with a real driver. Customers needing mask attacks with more than 16 positions per side run on CPU — the parser and iterator now support up to 256 CPU-side positions.
- `hashpipe.c` ships with no net source-functionality change vs v1.502; the file's revision counter advanced because intermediate experimental work was added and then reverted. Behavior is identical to v1.502 hashpipe.

# mdxfind v1.502 — Phase 4 + Phase 5a hx codegen (e347 + 7 MAKE_MD5PASS family GPU acceleration)

Source: mdxfind.c rev 1.501, gpu/gpu_opencl.c rev 1.194, gpu/gpu_opencl.h rev 1.41, gpu/gpujob_opencl.c rev 1.152, gpu_metal.m rev 1.117, gpu_metal.h rev 1.59, gpu/gpujob_metal.m rev 1.32, gpu/gpu_common.cl rev 1.25, gpu/metal_common.metal rev 1.24, gpu/gpu_codegen_eligible.{c,h} rev 1.1 (NEW), codegen/ tree (NEW: ~15 files, in-process hx P4 state-machine codegen), tools/hx8_to_c (NEW: build-time hx.8 → C-literal serializer).

Window: 2026-05-20 through 2026-05-23.

## Phase 4 — Production GPU dispatch for e347 (MD5(MD5(MD5(pass)).salt)) via codegen

A new in-process hx codegen pipeline produces JIT-compiled kernel B for `JOB_MD5MD5SALT` (e347) on both OpenCL (Pascal and newer) and Apple Metal. The codegen walker reads the `hx_program` bytecode for the algorithm, applies a pattern detector (`HX_PATTERN_E347_MD5MD5MD5SALT`) that recognizes the hand-tunable shape, and emits a specialized kernel source tuned to that shape (per-thread serial SALT_BATCH=64 inner loop, register-held pre-state, salt-axis amortization — the tp0 pattern that empirically wins on Pascal salted-MD5).

Cross-arch byte-exact validated against the CPU oracle on 1,048,576-pair fixtures (smoke / medium / large × 2 backends):

- Apple M2 Max: 0.25 s end-to-end on the large fixture; zero diff
- Pascal GTX 1080: 1.05 s end-to-end on the large fixture; zero diff

The hand-written `gpu_kernelb_md5md5salt_nocache.cl` (previously the e347 production path) is **retired**. The hand-written kernel carried a long-standing chain-drift bug for non-trivial salt cardinalities; codegen output is byte-exact against both the CPU implementation and hashpipe (independent reference). The legacy file has been deleted from the working tree.

`MDXFIND_HX_CODEGEN=0` is the (now-deprecated) opt-out env var. Setting it on a Phase-4+ build is **FATAL** with a deprecation message — the legacy hand-written code path is gone, there is nothing to fall back to.

## Phase 5a — MAKE_MD5PASS family GPU acceleration via codegen

Seven new algorithms GPU-accelerated by the same codegen pipeline, each computing `outer_hash(md5_hex(pass) . pass)`:

| eN  | Name           | Outer    | Digest width |
|----:|----------------|----------|-------------:|
| 122 | MD4MD5PASS     | MD4      |     16 bytes |
| 159 | RMD160MD5PASS  | RIPEMD160 |    20 bytes |
| 161 | SHA1MD5PASS    | SHA-1    |     20 bytes |
| 163 | SHA224MD5PASS  | SHA-224  |     28 bytes |
| 165 | SHA256MD5PASS  | SHA-256  |     32 bytes |
| 167 | SHA384MD5PASS  | SHA-384  |     48 bytes |
| 169 | SHA512MD5PASS  | SHA-512  |     64 bytes |

Codegen uses a per-primitive emit dispatch table (`codegen/hx_emit_primitives.c`, new in sub-phase 5a.2) so each family member shares the inner-hash + concat scaffolding and differs only in the outer-hash primitive selection.

Full 70-cell cross-arch validation matrix (7 algorithms × 5 fixtures × 2 backends) — all PASS, all byte-exact.

Usage: `./mdxfind -m e<N> -G 0 -F hashes -M <NAME> wordlist` selects GPU dispatch automatically for any of the 7 family members above. The `gpu_codegen_kernelb_family_md5pass_eligible()` admit-predicate helper (new file `gpu/gpu_codegen_eligible.{c,h}`) widens the chokepoint OR-chain to admit these JOBs.

Twenty-two additional MAKE_MD5PASS family members (MD2, GOST family, Haval ×15, RMD128, Tiger, Whirlpool, Snefru-256/512) remain CPU-only pending Phase 5b — their block primitives need lifting into `gpu/gpu_common.cl` first. e123 MD5MD5PASS (multi-emit canonical + colon variant) stays CPU-only until multi-emit codegen lands in a future sub-phase.

## Documentation

- The hx algorithm spec (Appendix A of the hx manual) audit and 32 doc-fix corrections: MAKE_MD5PASS family had missing concat operators in the canonical expressions; MD5MD5USER had user/pass argument transposition. Multi-emit families now annotated with Note [24] markers — 28 entries identifying mdxfind's multi-output emission patterns.
- The canonical hx language manual is published at <https://www.mdxfind.com/hx.pdf>. The hx parser + algorithm catalog source live in the upstream hashpipe project: <https://github.com/Cynosureprime/hashpipe>.

## Build / infrastructure

- New `codegen/` directory containing the in-process P4 state-machine codegen — `hx_walker.c` (state machine + bytecode dispatch), `hx_emit_opencl.c` + `hx_emit_metal.c` (per-backend emit helpers), `hx_patterns.c` (pattern detector for hand-tunable shapes), `hx_emit_primitives.c` (per-primitive outer-hash dispatch), `hx_dump.c` (env-flag source dump), and `hx_specs_data.c` (the compiled `hx_program` table, ~28 KLOC of generated C literals).
- New `tools/hx8_to_c.c` build-time tool that converts the hx algorithm catalog to `codegen/hx_specs_data.c`. Shipped here for completeness, but requires the upstream hashpipe source tree to compile (it links the hx parser library). External users build directly against the pre-generated `codegen/hx_specs_data.c` checked in here; that file is regenerated upstream when the hx catalog changes.
- New `gpu/gpu_codegen_eligible.{c,h}` — pure-C admit-predicate helper used by both OpenCL and Metal builds.
- New `hx_vm.h` and `hx_ast.h` — header-only types shared by codegen and the upstream hx VM (no implementation files; codegen consumes the compiled `hx_program` data).
- New kernel A variants under `gpu/` — `gpu_kernel_a_{rules,masks,rules_masks,bruteforce}.{cl,_str.h}` plus their Metal twins `metal_kernel_a_*.{metal,_str.h}` — hand-written rule / mask / brute-force producers from Phase 1a, used by the two-kernel pipeline for e347 and the family ops.
- `MDXFIND_HX_CODEGEN_DUMP=/tmp/x.cl` env var dumps the emitted codegen kernel source to the named path for post-mortem inspection.
- `MDXFIND_HX_CODEGEN_VALIDATE=1` + `MDXFIND_HX_CODEGEN_FIXTURE=<path>` env vars exercise the byte-exact validation harness against a fixture file (developer mode; exits 0 / 1 based on diff vs CPU oracle).

## Known issues / scope

- Codegen `kernelb_hx_codegen_phase0` ships the canonical e347 + 7 MAKE_MD5PASS family members. All other GPU ops continue to use the existing template-kernel infrastructure (unchanged from v1.485).
- e123 MD5MD5PASS remains CPU-only — multi-emit codegen is a future sub-phase.
- bcrypt, yescrypt, argon2, descrypt all remain hand-written kernels (out of codegen scope by design — they don't fit the hx expression model).
- The hand-port kernel A variants are unchanged from Phase 1a; Phase 1b "template kernel migration to codegen" is a low-priority follow-on (no perf or correctness motivation to rush it).
# mdxfind v1.475 — Metal coverage expansion + shared-loader refactor

Source: mdxfind.c rev 1.475, gpu_metal.m rev 1.100, gpu_metal.h rev 1.49, gpu/gpu_opencl.c rev 1.171, gpu/gpujob_metal.m rev 1.25, gpu/gpujob_opencl.c rev 1.138, gpu/codegen/cl2metal.py rev 1.9.

Window: 2026-05-16 through 2026-05-17.

## Metal coverage: 25 → 52 families

Apple Metal GPU acceleration extended from 25 to 52 algorithm families. New this release, grouped by phase:

- **Phase 2d.6** — RIPEMD: ripemd160 (e17), ripemd320 (e816)
- **Phase 2d.7a** — Blake2: blake2s256 (e844), blake2b256 (e845), blake2b512 (e841)
- **Phase 2d.7b** — Keccak + SHA-3: keccak{224,256,384,512} (e84-e87), sha3_{224,256,384,512} (e88-e91)
- **Phase 2d.7c** — Streebog: streebog256 (e430), streebog512 (e431)
- **Phase 2d.7d** — HMAC siblings: hmac_blake2s (e828), hmac_streebog256 (e837 KPASS / e838 KSALT), hmac_streebog512 (e839 KPASS / e840 KSALT)
- **Phase 2d.8a** — Iter-loop: phpbb3 (e455), md5crypt (e511)
- **Phase 2d.8b** — SHACRYPT: sha256crypt (e512), sha512crypt (e513), sha512cryptmd5 (e538)
- **Phase 2d.9a** — Feistel hand-port: descrypt (e500)
- **Phase 2d.9b** — Eksblowfish hand-port: bcrypt (e450)

All 52 families verified byte-exact CPU/Metal parity on Apple M1 and M2 Max. Admission summary: 52 families admitted, 208/208 variants admitted, 0 prunes, 0 CPU-only.

## Shared-loader refactor (gpu_metal.m −79.6%)

Per-family hand-cloned scaffolding replaced with a generic loader driven by an extended `struct gpu_metal_family`. New fields: `core_str`, `base_macros`, `dispatch_tg_size`, `fam_idx`. Parallel arrays `metal_family_libs[CAP][8]` + `metal_family_psos[CAP][8]` hidden in the .m so the .h stays pure C.

| Component                | Pre-refactor | Post-refactor | Delta            |
|--------------------------|-------------:|--------------:|-----------------:|
| gpu_metal.m              |       20,764 |         4,242 | −16,522 (−79.6%) |
| gpu_metal.h              |        1,002 |           432 |    −570 (−56.9%) |

47 families use `metal_pso_for_variant_default` (generic). 5 families keep custom resolvers:

- **md5salt** — PRESALT V_S|V_R fold; V_S, V_S|V_M, V_S|V_R|V_M route through generic
- **sha512cryptmd5** — aliases sha512crypt's compiled PSO at dispatch
- **hmac_streebog256_kpass / _ksalt** — dual-struct entries share one PSO via canonical fam_idx
- **hmac_streebog512_kpass / _ksalt** — same dual-struct pattern
- **bcrypt** — `dispatch_tg_size=8` struct override (replaces hardcoded `op == JOB_BCRYPT` check)

## External-failure-fatal discipline

New headers `gpu/gpu_fatal.h` + `gpu/gpu_debug.h`. Approximately 57 silent-failure sites converted across gpu_metal.m, gpu/gpujob_metal.m, gpu/gpu_opencl.c, gpu/gpujob_opencl.c. Runtime failures (PSO create, buffer alloc, dispatch error, clEnqueue errors) now call `GPU_FATAL` / `MTL_FATAL_NSERR` with file:line + op + error string and `_Exit(1)`.

Init-time eager-compile failures route through the admission-prune path (capability check, not runtime failure). Query `gpu_metal_op_variant_admitted()` exposed for host-side gpu_ops[] consumption (deferred).

## Debug emissions compile-time gated

`MDXFIND_GPU_DEBUG` macro in `gpu/gpu_debug.h`. Default builds omit:

- Per-(family, variant) JIT-compiled / PSO-created-lazily markers
- Per-run trace (`salts uploaded`, `first dispatch issued`, `buf_scratch_pool allocated`, `salt-chunked dispatch`)
- Init chatter
- `STDERR: GPU admission` summary line

Debug builds restore all markers:

```
make CFLAGS_EXTRA="-DMDXFIND_GPU_DEBUG=1" mdxfind
```

Verified via `strings | grep` that the production binary physically omits the debug strings from `.rodata`. Kept unconditional in production: `GPU_FATAL` / `MTL_FATAL_NSERR` runtime errors, device identity line, per-pruned-combo prune lines, end-of-job per-device stats.

## Breaking changes (operator-facing)

1. **Stderr format**. Operators previously grepping per-(family, variant) markers (`Metal: sha512-variant library JIT-compiled` and similar) must either rebuild with `-DMDXFIND_GPU_DEBUG=1` or re-target their grep to end-of-job per-device stats. `STDERR: GPU admission: N families admitted ...` is also debug-only now.
2. **Marker format**. When a debug build is active, generic-loader markers use the form `(generic vbits=0x... rules=N mask=N salt=N)` rather than the pre-refactor per-family fixed format. Backward-compatible substring `library JIT-compiled` still matches both.

## Platform notes

A pre-existing Apple Metal compiler bug in macOS Ventura 13.7.x affected SHA-2/512-family PSO creation on M2 Max. Fixed upstream in macOS 26.5 / Xcode 26.5. M2 Max users should upgrade for SHA-2/512-family GPU acceleration. M1 hosts on macOS 14+ are unaffected throughout.

### Intel Mac (macOS)

Metal GPU acceleration is **disabled at compile time** on Intel Mac. Apple's `MTLCompilerService` XPC daemon hangs on JIT PSO creation for AMD GCN GPUs (e.g., Radeon Pro 580X) on macOS Sequoia 15.x — confirmed on iMac and nutshack at rev 1.475. The hang is in Apple's driver and not fixable from mdxfind. Intel Mac users get CPU mode (or OpenCL if their Makefile enables it). Apple Silicon Macs (M-series) are unaffected; Metal GPU acceleration is fully supported on ARM macOS.

## Deferred

- Host-side `gpu_ops[]` consumption of the new `gpu_metal_op_variant_admitted()` query (defense-in-depth)
- mdxfind source-side `-h REGEX` runaway guard
- OpenCL shared-loader refactor parallel to Metal (estimated ~3K LOC savings)
- Phase 2e Dynsize Task #226 (long-standing backlog)
- OpenCL init-tier debug emissions (~285 ambiguous sites left as production pending adjudication)

## Files

Added:

- `gpu/gpu_fatal.h` (rev 1.1)
- `gpu/gpu_debug.h` (rev 1.1)
- `gpu/metal_*_core.metal` + paired `_str.h` for each Phase 2d.6–2d.9 family
- `gpu/codegen/cl2metal_overrides/*.yaml` for each translated family

Heavily modified:

- `gpu_metal.m` rev 1.93 → 1.100 (refactor + debug gating + admission summary)
- `gpu_metal.h` rev 1.47 → 1.49
- `gpu/gpujob_metal.m` rev 1.22 → 1.25
- `gpu/gpu_opencl.c` rev 1.169 → 1.171 (D5b FATAL conversion + debug gating)
- `gpu/gpujob_opencl.c` rev 1.137 → 1.138
- `mdxfind.c` rev 1.473 → 1.475 (Phase 2d.5.7 + 2d.7a host-wiring fixes)
- `gpu/codegen/cl2metal.py` rev 1.5 → 1.9 (pointer-state helpers, dual_addr_space_helpers overlay, _rewrite_local_uchar_casts, _FN_HEAD_RE widening, extra_scalar_ref_types overlay)
