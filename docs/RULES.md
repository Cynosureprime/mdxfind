# Rule Reference

mdxfind (via `-r` and `-R`) and [procrule](https://github.com/Cynosureprime/procrule) implement hashcat/JtR-compatible password mutation rules.

Rules are written one per line in a rule file. Each rule is a sequence of operations applied left to right. Multiple operations can be combined on a single line to create complex transformations.

Positions are encoded as `0`–`9` for positions 0–9, `A`–`Z` for positions 10–35, and `a`–`z` for positions 36–61. This allows rules to address characters up to position 61 in very long passwords.

## Case Rules

| Rule | Description | Example |
|------|-------------|---------|
| `l` | Lowercase all characters | `PaSsWoRd` → `password` |
| `u` | Uppercase all characters | `password` → `PASSWORD` |
| `c` | Capitalize first letter, lowercase rest | `password` → `Password` |
| `C` | Lowercase first letter, uppercase rest | `Password` → `pASSWORD` |
| `t` | Toggle case of all characters | `PaSsWoRd` → `pAsSwOrD` |
| `TN` | Toggle case at position N | `password` → `pAssword` (T1) |
| `E` | Title case (capitalize after each space) | `hello world` → `Hello World` |
| `eX` | Title case with custom separator X | `hello-world` → `Hello-World` (e-) |

## Insertion and Deletion

| Rule | Description | Example |
|------|-------------|---------|
| `$X` | Append character X | `pass` → `pass1` ($1) |
| `^X` | Prepend character X | `pass` → `1pass` (^1) |
| `[` | Delete first character | `password` → `assword` |
| `]` | Delete last character | `password` → `passwor` |
| `DN` | Delete character at position N | `password` → `pasword` (D3) |
| `iNX` | Insert character X at position N | `password` → `pas-sword` (i3-) |
| `oNX` | Overwrite character at position N with X | `password` → `pas@word` (o3@) |
| `'N` | Truncate word at length N | `password` → `pass` ('4) |
| `xNM` | Extract M characters starting at position N | `password` → `swor` (x3 4) |
| `ONM` | Delete M characters starting at position N | `password` → `pasd` (O3 4) |

## Duplication

| Rule | Description | Example |
|------|-------------|---------|
| `d` | Duplicate entire word | `pass` → `passpass` |
| `f` | Reflect — append reversed copy | `abc` → `abccba` |
| `pN` | Append duplicated word N times | `ab` → `ababab` (p2) |
| `q` | Duplicate every character | `abc` → `aabbcc` |
| `zN` | Duplicate first character N times | `pass` → `pppass` (z2) |
| `ZN` | Duplicate last character N times | `pass` → `passss` (Z2) |
| `yN` | Duplicate first N characters, prepend them | `password` → `papassword` (y2) |
| `YN` | Duplicate last N characters, append them | `password` → `passwordrd` (Y2) |

## Rearrangement

| Rule | Description | Example |
|------|-------------|---------|
| `r` | Reverse the word | `password` → `drowssap` |
| `{` | Rotate left — move first character to end | `password` → `asswordp` |
| `}` | Rotate right — move last character to front | `password` → `dpasswor` |
| `k` | Swap first two characters | `password` → `apssword` |
| `K` | Swap last two characters | `password` → `passwodr` |
| `*NM` | Swap characters at positions N and M | `password` → `psasword` (*1 2) |

## Character Manipulation

| Rule | Description | Example |
|------|-------------|---------|
| `sXY` | Replace all occurrences of X with Y | `password` → `p@ssword` (sa@) |
| `@X` | Purge — remove all occurrences of X | `password` → `pssword` (@a) |
| `+N` | Increment ASCII value at position N | `password` → `qassword` (+0) |
| `-N` | Decrement ASCII value at position N | `password` → `oassword` (-0) |
| `.N` | Replace character at N with character at N+1 | `password` → `paasword` (.1) |
| `,N` | Replace character at N with character at N-1 | `password` → `ppssword` (,1) |
| `LN` | Bit-shift left character at position N | |
| `RN` | Bit-shift right character at position N | |
| `vNX` | Insert character X every N characters | `password` → `pa-ss-wo-rd` (v2-) |

## Encoding

| Rule | Description | Example |
|------|-------------|---------|
| Ctrl-B (`\x02`) | Base64 encode the word | `test` → `dGVzdA==` |
| `h` | Hex-encode each byte (lowercase) | `test` → `74657374` |
| `H` | Hex-encode each byte (uppercase) | `test` → `74657374` |

## Memory

Memory rules allow saving and recalling the word state, enabling complex multi-step transformations.

| Rule | Description |
|------|-------------|
| `M` | Memorize current word state |
| `4` | Append memorized word |
| `6` | Prepend memorized word |
| `Q` | Reject word if it equals the memorized word (use after `M` to ensure the rule changed something) |
| `XNMI` | Insert M characters from memorized word at offset N, at position I in current word |

Example: `Mc$1Q` — memorize original, capitalize, append "1", reject if unchanged. This ensures only words that were actually modified are emitted.

## Rejection and Control

Rejection rules filter candidates based on conditions. If the condition is met, the candidate is rejected (not emitted).

| Rule | Description |
|------|-------------|
| `<N` | Reject if word length is less than N |
| `>N` | Reject if word length is greater than N |
| `_N` | Reject unless original word length equals N |
| `!X` | Reject if word contains character X |
| `/X` | Reject if word does not contain character X |
| `(X` | Reject if first character is not X |
| `)X` | Reject if last character is not X |

## Combining Rules

Multiple operations on a single line are applied left to right. This allows powerful combinations:

| Rule | Effect | Example with "password" |
|------|--------|------------------------|
| `c$1` | Capitalize + append 1 | `Password1` |
| `u$!$!` | Uppercase + append !! | `PASSWORD!!` |
| `sa@so0` | a→@ and o→0 (leetspeak) | `p@ssw0rd` |
| `^!c` | Prepend ! + capitalize | `!Password` |
| `r$1` | Reverse + append 1 | `drowssap1` |
| `d'8` | Duplicate + truncate to 8 | `password` (passpass → passp... truncated) |
| `Mc$1Q` | Capitalize + append 1 (only if changed) | `Password1` (rejects if already capitalized with 1) |

## Using Rules with mdxfind

```bash
# Apply rules from a file (concatenated)
mdxfind -r rules.txt -f hashes.txt wordlist.txt

# Apply multiple rule files (concatenated — rules are additive)
mdxfind -r best64.rule -r toggles.rule -f hashes.txt wordlist.txt

# Apply multiple rule files (dot-product — every combination)
mdxfind -R transforms.rule -R suffixes.rule -f hashes.txt wordlist.txt

# Show which rules were most effective
mdxfind -r rules.txt -Z -f hashes.txt wordlist.txt
```

See [EXAMPLES.md](EXAMPLES.md) for detailed examples with sample output.

## How mdxfind Processes Rules

### Compilation

When mdxfind reads a rule file, it goes through three phases:

1. **Parse** — each line is read and comments (lines starting with `#`) and blank lines are discarded.
2. **Validate** — each rule is checked for correct syntax. Invalid rules are reported on stderr and discarded. This means you can safely use hashcat rule files that may contain rules mdxfind doesn't support — they'll be skipped with a warning.
3. **Compile** — valid rules are converted into an internal bytecode format for fast application during the search loop. This avoids re-parsing the rule string for every candidate.

```
$ mdxfind -r best64.rule -f hashes.txt wordlist.txt
103 rules read from best64.rule
77 total rules in use
```

In this example, best64.rule has 103 lines. After removing comments, blank lines, and the no-op `:` rule (which reproduces the original word and is always tried implicitly), 77 rules remain.

### SIMD Rule Batching

For certain algorithms — notably MD5 — mdxfind can apply multiple rules to the same input word and compute the hashes in parallel using SIMD instructions. On x86_64, the MD5 implementation processes 4 candidates simultaneously in 128-bit SSE registers. This means that applying 4 rules costs approximately the same as computing a single MD5 hash.

This dramatically reduces the cost of rules. Here is a real benchmark using 1 million MD5 hashes, 13.5 million candidate words, and the best64 rule set (77 effective rules):

```
$ mdxfind -f 1m.txt 10m.pass                          # no rules
13,484,773 lines processed
1.00 seconds hashing, 13,484,773 total hash calculations
13.45M hashes per second
16,898 MD5x01 hashes found

$ mdxfind -r best64.rule -f 1m.txt 10m.pass           # with 77 rules
103 rules read from best64.rule
77 total rules in use
13,484,773 lines processed
1,038,192,924 total rule-generated passwords tested
53.22 seconds hashing, 1,038,192,924 total hash calculations
19.51M hashes per second
77,093 MD5x01 hashes found
```

**Analysis:**

| | No rules | With 77 rules |
|---|---|---|
| Candidates tested | 13.5M | 1,038M (77x) |
| Time | 1.0s | 53.2s |
| Throughput | 13.5M/s | 19.5M/s |
| Hashes found | 16,898 | 77,093 |

Without SIMD batching, 77 rules would take 77 × 1.0s = **77 seconds**. The actual time is **53 seconds** — a 1.45x speedup from the 4-wide SIMD rule batching. The effective throughput of 19.5M/s (higher than the no-rule 13.5M/s) reflects the amortization of per-word overhead across multiple rule applications.

The 77,093 hashes found (vs 16,898 without rules) show the dramatic improvement in coverage: 4.6x more hashes solved by trying common password mutations.

### Internal Rules vs External Pipeline

For comparison, the same candidates can be generated externally using procrule and piped into mdxfind. Since procrule suppresses rules that don't change the word, we also pass the original wordlist to ensure identical coverage:

```
$ procrule -r best64.rule 10m.pass | mdxfind -f 1m.txt stdin 10m.pass
948,460,875 lines processed
32.12 seconds hashing, 948,460,875 total hash calculations
29.53M hashes per second
77,093 MD5x01 hashes found
```

| Approach | Candidates | Wall time | CPU time | Throughput | Hashes found |
|----------|-----------|-----------|----------|------------|-------------|
| Internal rules (`-r`) | 1,038M | 53s | 1m32s | 19.5M/s | 77,093 |
| External pipe (procrule) | 948M | 32s | 4m56s | 29.5M/s | 77,093 |

Several things to note:

- **Candidate count differs.** procrule suppresses rules that produce no change (e.g., `l` on an already-lowercase word), generating 948M candidates. mdxfind's internal `-r` applies every rule to every word, generating 1,038M — about 9% more work.
- **Internal rules use 3.2x less total CPU** despite processing more candidates. The SIMD rule batching and avoidance of I/O overhead between processes make internal rules far more CPU-efficient.
- **External pipe is faster in wall time** on an idle multi-core system because procrule and mdxfind run on separate cores in parallel, and mdxfind's no-rule inner loop has higher per-candidate throughput (no rule bytecode interpretation overhead).

Both approaches find the same 77,093 hashes. Use internal rules (`-r`) when CPU efficiency matters or cores are limited; use procrule when you need the candidate list for other tools (hashcat, etc.) or when wall-clock time on an idle multi-core system is the priority.
