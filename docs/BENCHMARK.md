# Benchmarks

## Standard Benchmark Test Set

A reproducible benchmark suite is available for download from [www.mdxfind.com](https://www.mdxfind.com):

| File | Contents | Size |
|------|----------|------|
| [rockyou.txt.gz](https://www.mdxfind.com/rockyou.txt.gz) | Rockyou wordlist (14.3M passwords) | 49MB |
| [mdxfind-benchmark-full.zip](https://www.mdxfind.com/mdxfind-benchmark-full.zip) | Full hash test files (14.3M hashes each, unsalted + salted) | 1.7GB |
| [mdxfind-benchmark-small.zip](https://www.mdxfind.com/mdxfind-benchmark-small.zip) | Small hash test files (1M hashes each, unsalted + salted) | 122MB |
| [mdxfind-benchmark-2811.zip](https://www.mdxfind.com/mdxfind-benchmark-2811.zip) | Mode 2811 salted hash files (MyBB md5(md5($salt).md5($pass)), 5-char salts) | 374MB |

### Test files

The benchmark hash files were generated from rockyou.txt using mdxfind itself:

```bash
mdxfind -z -h '^MD5$' -f /dev/null rockyou.txt | cut -d' ' -f2 | cut -d: -f1 > testfull.txt
```

### Unsalted MD5 test files

| File | Hashes | Solvable | Description |
|------|--------|----------|-------------|
| `testfull.txt` | 14,341,564 | 100% | All hashes are solvable with rockyou.txt |
| `test50.txt` | 14,341,564 | ~50% | 50% of hashes randomly reversed (unsolvable noise) |
| `test10.txt` | 14,341,564 | ~10% | 90% of hashes randomly reversed (mostly noise) |
| `sm-testfull.txt` | 1,000,000 | 100% | First 1M lines of testfull.txt (for small/ARM hosts) |
| `sm-test50.txt` | 1,000,000 | ~50% | First 1M lines of test50.txt |
| `sm-test10.txt` | 1,000,000 | ~10% | First 1M lines of test10.txt |

### Salted MD5 test files

Each line is `hash:salt` format, where the hash is MD5(original\_MD5\_hash + salt) and the salt is a random 3-character string. This creates a double-hash-plus-salt scheme: solving requires computing MD5(candidate), appending the salt, then computing MD5 again — effectively MD5(MD5($pass) + $salt). This is internal type e31 (MD5SALT with MD5 pre-hash).

This creates two classes of solvable hashes within each salted file:

- **Non-reversed entries** (from the non-reversed hashes in the source test file): solvable as **MD5SALT (e31)** — standard MD5(MD5($pass) + $salt)
- **Reversed entries** (from the reversed hashes): solvable as **MD5revMD5SALT (e541)** — MD5(reverse(MD5($pass)) + $salt)

The standard salted benchmark uses e31 (MD5SALT) only, for fair comparison with other tools:

```bash
mdxfind -M e31 -F saltfull.txt rockyou.txt
```

However, mdxfind can solve *both* types in a single run by adding e541:

```bash
mdxfind -M e31,e541 -F saltfull.txt rockyou.txt
```

To our knowledge, no other tool supports the reversed-MD5-salted variant (e541).

| File | Hashes | Solvable | Description |
|------|--------|----------|-------------|
| `saltfull.txt` | 14,341,564 | 100% | Salted version of testfull.txt |
| `salt50.txt` | 14,341,564 | ~50% | Salted version of test50.txt |
| `salt10.txt` | 14,341,564 | ~10% | Salted version of test10.txt |
| `sm-saltfull.txt` | 1,000,000 | 100% | First 1M lines (for small/ARM hosts) |
| `sm-salt50.txt` | 1,000,000 | ~50% | First 1M lines |
| `sm-salt10.txt` | 1,000,000 | ~10% | First 1M lines |

The reversed hashes simulate real-world conditions where only a fraction of the hash list is solvable with a given wordlist. The `test10.txt` scenario (10% solvable) is typical of working with large leaked hash collections.

### Mode 2811 (MyBB) salted test files

Hashcat mode 2811 / mdxfind type e367 (MD5-MD5SALTMD5PASS): `md5(md5($salt).md5($pass))` with 5-character random printable ASCII salts. Generated from rockyou.txt using `gen2811` (source in `bench/`).

| File | Hashes | Solvable | Description |
|------|--------|----------|-------------|
| `salt2811.txt` | 14,341,564 | 100% | Full rockyou set, compound salted |
| `sm-salt2811.txt` | 1,000,000 | 100% | First 1M lines (for small/ARM hosts) |

```bash
# Full 2811 benchmark
time mdxfind -M 2811 -F salt2811.txt rockyou.txt > /dev/null

# Small 2811 benchmark
time mdxfind -M 2811 -F sm-salt2811.txt rockyou.txt > /dev/null
```

### Running the benchmark

```bash
# Full unsalted benchmark
gunzip rockyou.txt.gz
time mdxfind -f testfull.txt rockyou.txt > /dev/null
time mdxfind -f test50.txt rockyou.txt > /dev/null
time mdxfind -f test10.txt rockyou.txt > /dev/null

# Full salted benchmark
time mdxfind -M e31 -F saltfull.txt rockyou.txt > /dev/null
time mdxfind -M e31 -F salt50.txt rockyou.txt > /dev/null
time mdxfind -M e31 -F salt10.txt rockyou.txt > /dev/null

# Small unsalted benchmark (for Raspberry Pi, etc.)
time mdxfind -f sm-testfull.txt rockyou.txt > /dev/null
time mdxfind -f sm-test50.txt rockyou.txt > /dev/null
time mdxfind -f sm-test10.txt rockyou.txt > /dev/null

# Small salted benchmark
time mdxfind -M e31 -F sm-saltfull.txt rockyou.txt > /dev/null
time mdxfind -M e31 -F sm-salt50.txt rockyou.txt > /dev/null
time mdxfind -M e31 -F sm-salt10.txt rockyou.txt > /dev/null
```

Report: CPU model, OS, thread count, wall-clock time, and hashes found for each test file.

## Standard Benchmark Results

### Unsalted MD5 (rockyou.txt wordlist, sorted by testfull time)

#### Full test (14.3M hashes)

Expected finds: testfull=14,341,564, test50=7,169,180, test10=1,434,116.

| Machine | CPU | Clock | Full | 50% | 10% | Rate (full) |
|---------|-----|-------|------|-----|-----|-------------|
| dev3 | Apple M2 Max (12 cores) | 3.5 GHz | 3.0s | 2.0s | 1.0s | 4.7M/s |
| dev1 | Apple M1 (8 cores) | 3.2 GHz | 3.0s | 2.0s | 1.0s | 4.7M/s |
| mmt | 2x Xeon E5-2697 v4 (72T) | 2.3 GHz | 8.0s | 4.0s | 1.0s | 1.8M/s |
| firefly | AArch64 RK3399 (6 cores) | 2.0 GHz | 9.0s | 7.0s | 4.0s | 1.6M/s |
| ubpower8 | POWER8 (8 cores) | 3.4 GHz | 29.0s | 14.0s | 3.0s | 0.5M/s |

#### Small test (1M hashes)

Expected finds: sm-testfull=1,000,000, sm-test50=500,583, sm-test10=100,203.

| Machine | CPU | Clock | Full | 50% | 10% | Rate (full) |
|---------|-----|-------|------|-----|-----|-------------|
| firefly | AArch64 RK3399 (6 cores) | 2.0 GHz | 3.0s | 3.0s | 3.0s | 4.8M/s |
| pi3 | ARMv7 BCM2837 (4 cores) | 1.2 GHz | 7.0s | 6.0s | 6.0s | 2.0M/s |
| pi1a | ARMv6 BCM2835 (1 core) | 700 MHz | 87.0s | 76.0s | 69.0s | 0.16M/s |

### Salted MD5SALT (e31) — sm-saltfull (1M hashes, 1M unique salts, rockyou.txt wordlist)

| Machine | CPU/GPU | Clock | Found | Time | Hash calcs | Rate |
|---------|---------|-------|-------|------|-----------|------|
| -- | 12x NVIDIA RTX 4090 OpenCL (64 cores) | -- | 1,000,000 | 14s | 1,981B | 328G/s |
| ioblade | 5-GPU OpenCL (2x AMD gfx1201 + RTX 4070 Ti + RTX 3080 + AMD iGPU) | -- | 1,000,000 | 31s | 1,831B | 62.9G/s |
| dev3 | Apple M2 Max Metal (12 cores) | 3.5 GHz | 1,000,000 | 133s | 790B | 5.92G/s |
| dev1 | Apple M1 Metal (8 cores) | 3.2 GHz | 1,000,000 | 569s | 783B | 1.38G/s |
| fpga | NVIDIA GTX 1080 OpenCL | -- | 1,000,000 | 728s | 1,703B | 2.34G/s |
| hpi7 | NVIDIA GTX 960 OpenCL | -- | 1,000,000 | 988s | 1,019B | 1.03G/s |
| -- | 12x NVIDIA RTX 4090 hashcat 7.1 (Pure Kernel) | -- | 1,000,000 | 996s | 8,461B | 14.3G/s |
| mmt | 2x Xeon E5-2697 v4 (72T) | 2.3 GHz | 1,000,000 | 1916s | 960B | 501M/s |
| gp2 | AMD Radeon HD 7950 (Tahiti) OpenCL | -- | 1,000,000 | 1516s | 2,503B | 1.65G/s |
| gp | NVIDIA Tesla M2070 OpenCL | -- | 1,000,000 | 4730s | 2,818B | 595.7M/s |
| fpga | NVIDIA GTX 1080 hashcat (Pure Kernel) | -- | 1,000,000 | 4404s | -- | 175.9M/s |
| dev3 | Apple M2 Max CPU (12 cores) | 3.5 GHz | 1,000,000 | 4532s | 352B | 77.7M/s |
| dev1 | Apple M1 CPU (8 cores) | 3.2 GHz | 1,000,000 | 8403s | 352B | 41.9M/s |
| ubpower8 | POWER8 (80T) | 3.4 GHz | 1,000,000 | 12483s | 961B | 77.0M/s |

### Salted MD5-MD5SALTMD5PASS (e367, mode 2811) — sm-salt2811 (1M hashes, 1M unique salts, rockyou.txt wordlist)

Algorithm: `md5(md5($salt).md5($pass))` — compound salted type requiring three MD5 operations per candidate per salt.

| Machine | CPU/GPU | Clock | Found | Time | Hash calcs | Rate |
|---------|---------|-------|-------|------|-----------|------|
| fpga | NVIDIA GTX 1080 OpenCL | -- | 1,000,000 | 1451s | 4,073B | 2.82G/s |
| gp1 | AMD Radeon HD 7950 OpenCL | -- | 1,000,000 | 1977s | 3,790B | 1.92G/s |
| hpi7 | NVIDIA GTX 960 OpenCL | -- | 1,000,000 | 5513s | 3,597B | 652.6M/s |

The salted benchmark is dramatically more expensive than unsalted because each candidate must be tested against every unique salt. With 1M unique salts and 14.3M passwords, this requires hundreds of billions of hash computations.

**Notes:**
- The M1 processes the full 14.3M hash set in 3 seconds — hash loading time dominates.
- Lower solvability (test10) runs faster because fewer hash matches trigger output processing.
- The 72-core Xeon's per-thread rate is modest, but thread count gives it strong absolute throughput on salted workloads.
- The ARMv6 Pi 1 is ~30x slower than the M1 but still functional for smaller hash sets.

## Community Benchmarks

### mdxfind vs hashcat vs john — 25M hashes, 10 hash types

Contributed by @A1131. 25,000,000 hashes, 200MB wordlist, Ubuntu 24.04. mdxfind tested all 10 types (MD5 through SHA256) simultaneously via `-m e1-e10`.

```
time ./mdxfind -m e1-e10 -f hashes.txt wordlist.txt
```

| Tool | Hardware | Time |
|------|----------|------|
| **mdxfind** | Intel Core i5-9300H (CPU) | **26.0s** |
| john | RTX 1050 Ti (GPU) | 31.1s |
| hashcat | RTX 1050 Ti (GPU) | 56.9s |

mdxfind on a laptop CPU outperformed both GPU-accelerated tools on a mid-range GPU — while simultaneously testing 10 hash types. This reflects mdxfind's architecture: it loads all hashes into a Judy array and tests every candidate against the entire hash set in a single pass, whereas hashcat and john are optimized for smaller hash lists with deeper iteration counts.

mdxfind's advantage grows with hash list size — the Judy array lookup is O(1) regardless of whether there are 1,000 or 100,000,000 hashes loaded.

## Adding Your Benchmarks

If you have benchmark results comparing mdxfind to other tools, please open an issue or pull request. Include:

- Hash count and type
- Wordlist size
- Hardware (CPU model, GPU if applicable)
- OS and version
- Wall-clock time and hashes found
- Command lines used
