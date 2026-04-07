# Benchmarks

## Standard Benchmark Test Set

A reproducible benchmark suite is available for download from [www.mdxfind.com](https://www.mdxfind.com):

| File | Contents | Size |
|------|----------|------|
| [rockyou.txt.gz](https://www.mdxfind.com/rockyou.txt.gz) | Rockyou wordlist (14.3M passwords) | 49MB |
| [mdxfind-benchmark-full.zip](https://www.mdxfind.com/mdxfind-benchmark-full.zip) | Full hash test files (14.3M hashes each, unsalted + salted) | 1.7GB |
| [mdxfind-benchmark-small.zip](https://www.mdxfind.com/mdxfind-benchmark-small.zip) | Small hash test files (1M hashes each, unsalted + salted) | 122MB |
| [mdxfind-benchmark-2811.zip](https://www.mdxfind.com/mdxfind-benchmark-2811.zip) | Mode 2811 salted hash files (MyBB md5(md5($salt).md5($pass)), 5-char salts) | 374MB |
| [mdxfind-benchmark-400.zip](https://www.mdxfind.com/mdxfind-benchmark-400.zip) | Mode 400 PHPBB3/phpass hash files (100K hashes, variable iterations) | 2.3MB |

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

**Note:** The 2811.zip was updated 2026-04-05 — the previous version was missing the first salted password from rockyou.txt.

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
| mmt | NVIDIA RTX 4070 Ti SUPER OpenCL + CPU | -- | 1,000,000 | 45s | 353B | 8.03G/s |
| dev3 | Apple M2 Max Metal (12 cores) | 3.5 GHz | 1,000,000 | 262s | 697B | 2.65G/s |
| fpga | NVIDIA GTX 1080 OpenCL | -- | 1,000,000 | 376s | 353B | 940M/s |
| gp1 | AMD Radeon HD 7950 OpenCL + CPU | -- | 1,000,000 | 603s | 353B | 586M/s |
| mmt | 2x Xeon E5-2697 v4 (72T) | 2.3 GHz | 1,000,000 | 686s | 351B | 513M/s |
| hpi7 | NVIDIA GTX 960 OpenCL | -- | 1,000,000 | 902s | 353B | 391M/s |
| -- | 12x NVIDIA RTX 4090 hashcat 7.1 (Pure Kernel) | -- | 1,000,000 | 996s | 8,461B | 14.3G/s |
| dev1 | Apple M1 Metal (8 cores) | 3.2 GHz | 1,000,000 | 1035s | 695B | 672M/s |
| gp | NVIDIA Tesla M2070 OpenCL + CPU | -- | 1,000,000 | 1616s | 353B | 219M/s |
| mmt | NVIDIA RTX 4070 Ti SUPER hashcat 6.2.6 (Pure Kernel) | -- | 1,000,000 | 1661s | -- | 1.53G/s |
| dev3 | Apple M2 Max CPU (12 cores) | 3.5 GHz | 1,000,000 | 2635s | 351B | 133M/s |
| fpga | NVIDIA GTX 1080 hashcat (Pure Kernel) | -- | 1,000,000 | 4404s | -- | 175.9M/s |
| dev1 | Apple M1 CPU (8 cores) | 3.2 GHz | 1,000,000 | 4859s | 351B | 72.3M/s |
| hpi7 | NVIDIA GTX 960 hashcat 6.2.6 (Pure Kernel) | -- | 1,000,000 | 8191s | -- | 60.8M/s |
| ubpower8 | POWER8 (80T) | 3.4 GHz | 1,000,000 | 12483s | 961B | 77.0M/s |

### Salted MD5-MD5SALTMD5PASS (e367, mode 2811) — sm-salt2811 (1M hashes, 1M unique salts, rockyou.txt wordlist)

Algorithm: `md5(md5($salt).md5($pass))` — compound salted type requiring three MD5 operations per candidate per salt.

```bash
mdxfind -M 2811 -F bench/sm-salt2811.txt rockyou.txt > /dev/null
hashcat -a 0 -m 2811 -o /dev/null --potfile-disable sm-salt2811.txt rockyou.txt
```

| Machine | CPU/GPU | Clock | Found | Time | Hash calcs | Rate |
|---------|---------|-------|-------|------|-----------|------|
| mmt | NVIDIA RTX 4070 Ti SUPER OpenCL + CPU | -- | 1,000,000 | 83s | 503B | 6.25G/s |
| dev3 | Apple M2 Max Metal + CPU | 3.5 GHz | 1,000,000 | 191s | 503B | 2.65G/s |
| fpga | NVIDIA GTX 1080 OpenCL + CPU | -- | 1,000,000 | 447s | 503B | 1.13G/s |
| gp1 | AMD Radeon HD 7950 OpenCL + CPU | -- | 1,000,000 | 458s | 503B | 1.10G/s |
| dev1 | Apple M1 Metal + CPU | 3.2 GHz | 1,000,000 | 757s | 503B | 665M/s |
| hpi7 | NVIDIA GTX 960 OpenCL + CPU | -- | 1,000,000 | 1140s | 503B | 441M/s |
| mmt | NVIDIA RTX 4070 Ti SUPER hashcat 6.2.6 (Pure Kernel) | -- | 1,000,000 | 2131s | -- | 2.03G/s |
| gp | NVIDIA Tesla M2070 OpenCL + CPU | -- | 1,000,000 | 2713s | 503B | 186M/s |
| fpga | NVIDIA GTX 1080 hashcat 6.2.6 (Pure Kernel) | -- | 1,000,000 | 7740s | -- | 169.1M/s |
| hpi7 | NVIDIA GTX 960 hashcat 6.2.6 (Pure Kernel) | -- | 1,000,000 | 10020s | -- | 76.5M/s |

### PHPBB3/phpass (e455, mode 400) — sm-salt400 (100K hashes, 100K passwords)

Algorithm: iterated MD5 with per-hash salt and variable iteration count (512-2048 rounds). Due to the high per-hash computational cost, this benchmark uses 100,000 hashes and the first 100,000 passwords from rockyou.txt rather than the full 14.3M wordlist.

```bash
mdxfind -M 400 -F bench/sm-salt400.txt bench/sm-salt400.pass > /dev/null
hashcat -a 0 -m 400 -o /dev/null --potfile-disable sm-salt400.txt sm-salt400.pass
```

| Machine | CPU/GPU | Clock | Found | Time | Hash calcs | Rate |
|---------|---------|-------|-------|------|-----------|------|
| mmt | NVIDIA RTX 4070 Ti SUPER OpenCL + CPU | -- | 100,000 | 64s | 2,678B | 42.5G/s |
| fpga | NVIDIA GTX 1080 OpenCL + CPU | -- | 100,000 | 180s | 2,676B | 14.9G/s |
| dev3 | Apple M2 Max Metal + CPU | 3.5 GHz | 100,000 | 304s | 2,795B | 9.20G/s |
| mmt | NVIDIA RTX 4070 Ti SUPER hashcat 6.2.6 (Pure Kernel) | -- | 100,000 | 324s | -- | 19.4M/s |
| fpga | NVIDIA GTX 1080 hashcat 6.2.6 (Pure Kernel) | -- | 100,000 | 578s | -- | 5.98M/s |
| hpi7 | NVIDIA GTX 960 OpenCL + CPU | -- | 100,000 | 738s | 2,675B | 3.63G/s |
| dev1 | Apple M1 Metal + CPU | 3.2 GHz | 100,000 | 1196s | 2,792B | 2.33G/s |
| hpi7 | NVIDIA GTX 960 hashcat 6.2.6 (Pure Kernel) | -- | 100,000 | 1863s | -- | 2.31M/s |

### DESCRYPT (e500, mode 1500) — sm-salt1500 (1M hashes, 4096 salts, 1M passwords)

Algorithm: traditional DES `crypt()` with 12-bit salt and 8-character maximum password length.

```bash
mdxfind -M 1500 -F bench/sm-salt1500.txt bench/sm-salt1500.pass > /dev/null
hashcat -a 0 -m 1500 -o /dev/null --potfile-disable sm-salt1500.txt sm-salt1500.pass
```

| Machine | CPU/GPU | Clock | Found | Time | Hash calcs | Rate |
|---------|---------|-------|-------|------|-----------|------|
| mmt | NVIDIA RTX 4070 Ti SUPER OpenCL + CPU | -- | 1,000,000 | 11s | 4B | 425M/s |
| fpga | NVIDIA GTX 1080 OpenCL + CPU | -- | 1,000,000 | 26s | 4B | 170M/s |
| dev3 | Apple M2 Max Metal + CPU | 3.5 GHz | 1,000,000 | 54s | 4B | 75.9M/s |
| fpga | NVIDIA GTX 1080 hashcat 6.2.6 (Pure Kernel) | -- | 1,000,000 | 84s | -- | 34.0M/s |
| hpi7 | NVIDIA GTX 960 OpenCL + CPU | -- | 1,000,000 | 103s | 4B | 40.4M/s |
| mmt | NVIDIA RTX 4070 Ti SUPER hashcat 6.2.6 (Pure Kernel) | -- | 1,000,000 | 135s | -- | 126.6M/s |
| dev1 | Apple M1 Metal + CPU | 3.2 GHz | 1,000,000 | 141s | 4B | 29.0M/s |
| hpi7 | NVIDIA GTX 960 hashcat 6.2.6 (Pure Kernel) | -- | 1,000,000 | 221s | -- | 4.36M/s |
| fpga | AMD Ryzen 7 1800X CPU (16T) | 3.6 GHz | 1,000,000 | 1378s | 4B | 2.97M/s |
| dev3 | Apple M2 Max CPU (12 cores) | 3.5 GHz | 1,000,000 | 1849s | 4B | 2.21M/s |
| hpi7 | Intel Xeon E3-1230 v5 CPU (8T) | 3.4 GHz | 1,000,000 | 2390s | 4B | 1.71M/s |

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
