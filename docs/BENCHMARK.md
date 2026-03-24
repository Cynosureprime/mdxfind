# Benchmarks

## Standard Benchmark Test Set

A reproducible benchmark suite is available as release assets on the [v1.214 release page](https://github.com/Cynosureprime/mdxfind/releases/tag/v1.214):

| File | Contents | Size |
|------|----------|------|
| `rockyou.txt.gz` | Rockyou wordlist (14.3M passwords) | 49MB |
| `mdxfind-benchmark-full.zip` | Full hash test files (14.3M hashes each) | 784MB |
| `mdxfind-benchmark-small.zip` | Small hash test files (1M hashes each) | 55MB |

### Test files

The benchmark hash files were generated from rockyou.txt using mdxfind itself:

```bash
mdxfind -z -h '^MD5$' -f /dev/null rockyou.txt | cut -d' ' -f2 | cut -d: -f1 > testfull.txt
```

| File | Hashes | Solvable | Description |
|------|--------|----------|-------------|
| `testfull.txt` | 14,341,564 | 100% | All hashes are solvable with rockyou.txt |
| `test50.txt` | 14,341,564 | ~50% | 50% of hashes randomly reversed (unsolvable noise) |
| `test10.txt` | 14,341,564 | ~10% | 90% of hashes randomly reversed (mostly noise) |
| `sm-testfull.txt` | 1,000,000 | 100% | First 1M lines of testfull.txt (for small/ARM hosts) |
| `sm-test50.txt` | 1,000,000 | ~50% | First 1M lines of test50.txt |
| `sm-test10.txt` | 1,000,000 | ~10% | First 1M lines of test10.txt |

The reversed hashes simulate real-world conditions where only a fraction of the hash list is solvable with a given wordlist. The `test10.txt` scenario (10% solvable) is typical of working with large leaked hash collections.

### Running the benchmark

```bash
# Full benchmark (needs ~14M-hash test files)
gunzip rockyou.txt.gz
time mdxfind -f testfull.txt rockyou.txt > /dev/null
time mdxfind -f test50.txt rockyou.txt > /dev/null
time mdxfind -f test10.txt rockyou.txt > /dev/null

# Small benchmark (for Raspberry Pi, etc.)
time mdxfind -f sm-testfull.txt rockyou.txt > /dev/null
time mdxfind -f sm-test50.txt rockyou.txt > /dev/null
time mdxfind -f sm-test10.txt rockyou.txt > /dev/null
```

Report: CPU model, OS, thread count, wall-clock time, and hashes found for each test file.

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
