# Benchmarks

## Standard Benchmark Test Set

A reproducible benchmark suite is available for download from [www.mdxfind.com](https://www.mdxfind.com):

| File | Contents | Size |
|------|----------|------|
| [rockyou.txt.gz](https://www.mdxfind.com/rockyou.txt.gz) | Rockyou wordlist (14.3M passwords) | 49MB |
| [mdxfind-benchmark-full.zip](https://www.mdxfind.com/mdxfind-benchmark-full.zip) | Full hash test files (14.3M hashes each, unsalted + salted) | 1.7GB |
| [mdxfind-benchmark-small.zip](https://www.mdxfind.com/mdxfind-benchmark-small.zip) | Small hash test files (1M hashes each, unsalted + salted) | 122MB |

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

### Unsalted MD5 — testfull (14.3M hashes, rockyou.txt wordlist)

| Machine | CPU | Clock | Test | Found | Time | Rate |
|---------|-----|-------|------|-------|------|------|
| dev1 | Apple M1 | 3.2 GHz | testfull (14.3M) | 14,341,564 | 3.0s | 4.7M/s |
| dev1 | Apple M1 | 3.2 GHz | test50 (14.3M) | 7,169,180 | 2.0s | 7.1M/s |
| dev1 | Apple M1 | 3.2 GHz | test10 (14.3M) | 1,434,116 | 1.0s | 14.2M/s |
| ubpower8 | POWER8 | 3.4 GHz | testfull (14.3M) | 14,341,564 | 29.0s | 0.5M/s |
| ubpower8 | POWER8 | 3.4 GHz | test50 (14.3M) | 7,169,180 | 14.0s | 1.0M/s |
| ubpower8 | POWER8 | 3.4 GHz | test10 (14.3M) | 1,434,116 | 3.0s | 4.8M/s |

### Unsalted MD5 — small test (1M hashes, rockyou.txt wordlist)

| Machine | CPU | Clock | Test | Found | Time | Rate |
|---------|-----|-------|------|-------|------|------|
| firefly | AArch64 (RK3399) | 2.0 GHz | sm-testfull (1M) | 1,000,000 | 3.0s | 4.8M/s |
| firefly | AArch64 (RK3399) | 2.0 GHz | sm-test50 (1M) | 500,583 | 3.0s | 4.8M/s |
| firefly | AArch64 (RK3399) | 2.0 GHz | sm-test10 (1M) | 100,203 | 3.0s | 4.8M/s |
| pi3 | ARMv7 (BCM2837) | 1.2 GHz | sm-testfull (1M) | 1,000,000 | 7.0s | 2.0M/s |
| pi3 | ARMv7 (BCM2837) | 1.2 GHz | sm-test50 (1M) | 500,583 | 6.0s | 2.4M/s |
| pi3 | ARMv7 (BCM2837) | 1.2 GHz | sm-test10 (1M) | 100,203 | 6.0s | 2.4M/s |
| pi1a | ARMv6 (BCM2835) | 700 MHz | sm-testfull (1M) | 1,000,000 | 87.0s | 0.16M/s |
| pi1a | ARMv6 (BCM2835) | 700 MHz | sm-test50 (1M) | 500,583 | 76.0s | 0.19M/s |
| pi1a | ARMv6 (BCM2835) | 700 MHz | sm-test10 (1M) | 100,203 | 69.0s | 0.21M/s |

**Notes:**
- The M1 processes the full 14.3M hash set in 3 seconds — dominated by hash loading time, not computation.
- Lower solvability (test10) runs faster because fewer hash matches trigger output processing.
- The ARMv6 Pi 1 is ~30x slower than the M1 but still functional for smaller hash sets.

## Community Benchmarks

### mdxfind vs hashcat vs john — 2.5M MD5 hashes

Contributed by @A1131. 2,500,000 32-character hex hashes, 200MB wordlist, Ubuntu 24.04.

| Tool | Hardware | Time |
|------|----------|------|
| **mdxfind** | Intel Core i5-9300H (CPU) | **26.0s** |
| john | RTX 1050 Ti (GPU) | 31.1s |
| hashcat | RTX 1050 Ti (GPU) | 56.9s |

mdxfind on a laptop CPU outperformed both GPU-accelerated tools on a mid-range GPU. This reflects mdxfind's architecture: it loads all hashes into a Judy array and tests every candidate against the entire hash set in a single pass, whereas hashcat and john are optimized for smaller hash lists with deeper iteration counts.

mdxfind's advantage grows with hash list size — the Judy array lookup is O(1) regardless of whether there are 1,000 or 100,000,000 hashes loaded.

## Adding Your Benchmarks

If you have benchmark results comparing mdxfind to other tools, please open an issue or pull request. Include:

- Hash count and type
- Wordlist size
- Hardware (CPU model, GPU if applicable)
- OS and version
- Wall-clock time and hashes found
- Command lines used
