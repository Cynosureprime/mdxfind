# mdxfind Reference and Examples

## Supported Hash Algorithms

mdxfind supports several hundred algorithms, with billions of variations of each algorithm. There are several different variations available. See [HASH_TYPES.md](../HASH_TYPES.md) for the complete list.

## Iterations (-i switch)

The most basic variation is to "iterate" the algorithm. For mdxfind, this means to take the result of the last hash, express it in ASCII hex, and with the length set to the native length of the hash, re-compute the hash the specified number of times.

For example, consider various MD5 encodings of "test":

| Iterations | Hash |
|------------|------|
| MD5x01 | `098f6bcd4621d373cade4e832627b4f6` |
| MD5x02 | `fb469d7ef430b0baf0cab6c436e70375` |
| MD5x03 | `25ab3b38f7afc116f18fa9821e44d561` |
| MD5x1000 | `bd02d934c758f72e81f70a56f3b8c575` |

If the file `/tmp/foo` contains the password "test", you could use the command line:

```
mdxfind -i 1000 -f hash.txt /tmp/foo
```

Which would find all 4 of the encodings of "test" in less than a second.

The iterations use the native length of the encoding; thus SHA-1 would iterate the 40-character ASCII hex representations, SHA-256 would be the 64-character ASCII hex, and SHA-512 would be the 128-character strings.

Iterations always refer to the outermost type of hash function for more complex types. MD5SHA1MD5SHA1SHA1, for example, would iterate the MD5 function.

You can iterate any hash function up to 2^32 times (4,294,967,295).

## Interior Iterations (-q switch)

Some hash functions use so-called interior iterations. These iterate a part of the hash interior to the function, separate from the outer iterations. A good example is MD5SHA1MD5x, where the rightmost MD5 function can be iterated many times. These are denoted in the hash function type by the lowercase "x" in the string.

| Function | Hash |
|----------|------|
| MD5SHA1MD5x100 | `426cedad6a7161e7de6f20f08a24ef28` |

You can solve this hash with the command:

```
mdxfind -q 1000 -f hash.txt -h ^md5sha1md5x$ /tmp/foo
```

## Rotations (-g switch)

Rotations are performed after the hash is computed, operating on the ASCII hex string representation (not the internal binary representation). This avoids endianness issues and represents what hash function creators do to further manipulate hash functions.

For example, the MD5 of "test" is `098f6bcd4621d373cade4e832627b4f6`. All 31 rotations:

| Function | Hash |
|----------|------|
| rot1\_MD5 | `6098f6bcd4621d373cade4e832627b4f` |
| rot2\_MD5 | `f6098f6bcd4621d373cade4e832627b4` |
| rot3\_MD5 | `4f6098f6bcd4621d373cade4e832627b` |
| rot4\_MD5 | `b4f6098f6bcd4621d373cade4e832627` |
| rot5\_MD5 | `7b4f6098f6bcd4621d373cade4e83262` |
| ... | ... |
| rot31\_MD5 | `98f6bcd4621d373cade4e832627b4f60` |

To solve rotated hashes:

```
mdxfind -g 32 -f hash.txt /tmp/foo
```

The `-g` switch is somewhat special. If the lengths of the input hashes are all the same (for example, 32 characters), `-g 32` will try all 31 rotations of the basic hash. This works the same for SHA-1 (`-g 40`) and all other lengths. But if there are multiple different-length hashes on the input, mdxfind will not try all the variations, and you will need to do each one as a separate pass (`-g 1`, then `-g 2`, etc.).

## Intrinsic Variations (no switch)

By default, mdxfind will try reasonable variations of the specified algorithms against all input hashes. For example, SHA1 produces a 40-character ASCII hex hash. mdxfind will attempt to match a 32-character input hash against both the beginning and end of the string, which will find truncations of SHA1 hashes in 32-character "MD5-like" input hashes.

The same algorithm is used for any algorithm which can produce a longer hash than the input hash list. For example, four 32-character hashes could match a single SHA-512 candidate.

## Which Algorithm Is Right?

mdxfind supports hundreds of algorithms, and can try candidate passwords against the input hash list using any or all of them simultaneously. This can and does result in multiple solutions for a single input hash. So, which algorithm is right?

This goes to the core of how mdxfind operates, and why it doesn't have a `--remove` switch. mdxfind will never alter any input file — it simply produces solutions on stdout for post-processing.

Consider the hash `626ce222351b68b259f73de172f11249`, which is the MD5 of the password "10test". Given that a wordlist contains both "10test" and "test":

```
mdxfind -s 2-salt -f hash.txt -h ^md5saltpass$,^md5$ wordlist.txt
```

will produce:

```
MD5x01 626ce222351b68b259f73de172f11249:10test
MD5SALTPASSx01 626ce222351b68b259f73de172f11249:10:test
```

Which is correct? Well, they both are. The decision as to which should be used is deferred to the post-processor. The user can also elect to process one group of input hashes one way, and a second group a different way, simply by selecting the appropriate solutions.

This is one of the great strengths — and great weaknesses — of mdxfind.

## Unprintable Characters and $HEX[]

mdxfind always tries to produce output which can be reused. This is particularly important given the number of special characters which can be used in passwords today. To do this, mdxfind will convert any password which contains characters outside the range of space through tilde (0x20-0x7E) — specifically control characters, the colon character (since it's the delimiter), DEL, and all high-bit characters — to the `$HEX[]` notation.

For example, the hash `d8e8fca2dc0f896fd7cb4cb0031ba249` (which is "test" followed by a linefeed) will be shown as:

```
MD5x01 d8e8fca2dc0f896fd7cb4cb0031ba249:$HEX[746573740a]
```

mdxfind fully understands the hex format, and will happily use this as a candidate password. There are no limitations on the characters which can appear in a `$HEX[]` representation — strings of NULs, UTF-8, UTF-16, and UTF-32 can all be used.

## Appending Characters (-n switch)

The `-n` switch appends characters to every candidate password. This is one of the most powerful features for finding passwords with common suffixes like year numbers, PINs, or short character sequences.

### Appending digits

`-n 2` appends all 2-digit numbers (00-99) to each candidate:

```
mdxfind -n 2 -f hashes.txt wordlist.txt
```

If your wordlist contains "test", mdxfind will try: test00, test01, test02, ... test99.

`-n 3` appends all 3-digit numbers (000-999), and so on up to `-n 16`.

### Appending hex digits

Add an `x` suffix to use hexadecimal digits (0-9, a-f):

```
mdxfind -n 3x -f hashes.txt wordlist.txt
```

This tries 000 through fff (4096 combinations) appended to each candidate. Use `X` for uppercase hex (0-9, A-F).

### Appending with masks

For more control, use mask syntax with character classes:

```
mdxfind -n '?l?d' -f hashes.txt wordlist.txt
```

This appends one lowercase letter followed by one digit (260 combinations: a0, a1, ... z9).

Available mask character classes:

| Class | Meaning | Count |
|-------|---------|-------|
| `?d` | Digits (0-9) | 10 |
| `?l` | Lowercase letters (a-z) | 26 |
| `?u` | Uppercase letters (A-Z) | 26 |
| `?s` | Special characters | 33 |
| `?a` | All printable ASCII | 95 |
| `?b` | All bytes (0x00-0xff) | 256 |
| `[...]` | Custom character set | varies |

Custom character sets use bracket notation:

```
# Append two hex digits (custom set)
mdxfind -n '?[0-9a-f]?[0-9a-f]' -f hashes.txt wordlist.txt

# Append a year from 2000-2029
mdxfind -n '20?[0-2]?d' -f hashes.txt wordlist.txt
```

### Multiplier effect

The `-n` switch multiplies your wordlist by the number of combinations. A wordlist of 10 million words with `-n 2` becomes 1 billion candidates (10M x 100). With `-n 3`, it becomes 10 billion. Choose carefully — `-n 4` on a large wordlist may take a very long time.

| Switch | Combinations | 10M wordlist becomes |
|--------|-------------|---------------------|
| `-n 1` | 10 | 100M |
| `-n 2` | 100 | 1B |
| `-n 3` | 1,000 | 10B |
| `-n 4` | 10,000 | 100B |
| `-n 2x` | 256 | 2.56B |
| `-n 3x` | 4,096 | 40.96B |
| `-n '?l?d'` | 260 | 2.6B |
| `-n '?d?d?d?d'` | 10,000 | 100B |

## Prepending Characters (-N switch)

The `-N` switch prepends characters to every candidate password, using mask syntax:

```
mdxfind -N '?d?d' -f hashes.txt wordlist.txt
```

If your wordlist contains "test", mdxfind will try: 00test, 01test, 02test, ... 99test.

The mask syntax for `-N` is the same as for `-n`. This is useful for passwords where users prefix with digits or short codes:

```
# Prepend 1-3 digits
mdxfind -N '?d' -f hashes.txt wordlist.txt
mdxfind -N '?d?d' -f hashes.txt wordlist.txt
mdxfind -N '?d?d?d' -f hashes.txt wordlist.txt
```

### Combining -n and -N

You can use both switches together to try modifications on both ends:

```
mdxfind -N '?d' -n '?d?d' -f hashes.txt wordlist.txt
```

This prepends one digit and appends two digits to each candidate, for 10 x 100 = 1,000 combinations per word.

## Selecting Hash Types (-h and -m switches)

### By name with regex (-h)

The `-h` switch selects hash types using regular expressions (case-insensitive). Multiple types can be comma-separated, and `-h` can be used multiple times:

```bash
# All MD5 variants
mdxfind -h MD5 -f hashes.txt wordlist.txt

# Only exact MD5 (no variants)
mdxfind -h ^MD5$ -f hashes.txt wordlist.txt

# MD5 and SHA1
mdxfind -h MD5,SHA1 -f hashes.txt wordlist.txt

# All salted types
mdxfind -h SALT -f hashes.txt wordlist.txt

# Everything except salted and user types
mdxfind -h ALL -h '!salt,!user' -f hashes.txt wordlist.txt
```

### By hashcat mode or internal index (-m)

The `-m` switch selects by hashcat mode number or mdxfind's internal type index:

```bash
# hashcat mode 0 (MD5)
mdxfind -m 0 -f hashes.txt wordlist.txt

# hashcat modes 0 and 100 (MD5 + SHA1)
mdxfind -m 0,100 -f hashes.txt wordlist.txt

# Internal types e1 through e10 (MD5 through SHA256)
mdxfind -m e1-e10 -f hashes.txt wordlist.txt

# Mix hashcat and internal: MD5 + NTLM
mdxfind -m e1,1000 -f hashes.txt wordlist.txt
```

Use `-h` with no other arguments to display the full list of supported types with their hashcat mode mappings.

## Salt Files (-s switch)

The `-s` switch loads salts from a file. This is essential when you have salted hashes but the salt is not embedded in the hash line. mdxfind will try every salt against every hash for the selected salted algorithms:

```bash
# Extract salts from hash:salt format (characters 34 onward)
cut -c 34- salted_hashes.txt > salts.txt

# Run with salts
mdxfind -s salts.txt -h "^MD5SALT$" -f salted_hashes.txt wordlist.txt
```

You can use `-s` multiple times. A special form generates numeric salts:

```bash
# Generate 2-digit salts (00-99)
mdxfind -s 2-salt -f hashes.txt wordlist.txt

# Generate 3-digit salts (000-999)
mdxfind -s 3-salt -f hashes.txt wordlist.txt

# Generate 6-digit salts (000000-999999)
mdxfind -s 6-salt -f hashes.txt wordlist.txt
```

## Loading Hashes with Embedded Salts (-F and -M switches)

When hashes have salts embedded in the line (e.g., `hash:salt`), use `-F` to load them directly and `-M` to specify the hash type:

```bash
# Load salted hashes from stdin, type MD5SALT (internal e31)
cat salted/*.txt | mdxfind -M e31 -F stdin wordlist.txt

# Load from a file
mdxfind -M e31 -F salted_hashes.txt wordlist.txt
```

## Username/Userid Files (-u switch)

Some hash types incorporate a username into the hash computation (e.g., Oracle, MSCACHE). Load usernames from a file:

```bash
mdxfind -u users.txt -h USER -f hashes.txt wordlist.txt
```

## Suffix Files (-k switch)

The `-k` switch reads literal strings from a file and appends each one to every candidate word. Unlike `-n` (which generates digit/mask combinations), `-k` uses arbitrary strings — domain names, common suffixes, year strings, etc.:

```bash
# suffixes.txt contains: @gmail.com, @yahoo.com, 2024, 2025, !!, ...
mdxfind -k suffixes.txt -f hashes.txt wordlist.txt
```

Each line in the suffix file is appended to each word, so a 10-line suffix file multiplies your wordlist by 10.

## Email Address Munging (-a switch)

The `-a` switch performs intelligent email-style mutations on candidate words. For each word containing an `@` sign:

1. The domain portion is identified and preserved
2. The username portion has all `.` separators stripped
3. All possible positions for inserting `.` between characters are tried

This is useful because email providers (especially Gmail) treat dots in usernames as optional — `john.doe@gmail.com`, `johndoe@gmail.com`, and `j.o.h.n.d.o.e@gmail.com` all deliver to the same mailbox. Some sites store the user-entered version as a "password" or identifier.

```bash
mdxfind -a -f hashes.txt email_wordlist.txt
```

Words without an `@` sign are skipped by the munging (but still hashed normally).

## Unicode Expansion (-b switch)

The `-b` switch expands each word into UTF-16LE encoding (best effort), which is required for hash types like NTLM that operate on Unicode input:

```bash
mdxfind -b -h NTLM -f ntlm_hashes.txt wordlist.txt
```

## XML Entity Replacement (-c switch)

The `-c` switch replaces special characters (`<`, `>`, `&`, etc.) with their XML entity equivalents (`&lt;`, `&gt;`, `&amp;`). Useful for web application hashes where the password was HTML-encoded before hashing:

```bash
mdxfind -c -f hashes.txt wordlist.txt
```

## De-duplication (-d switch)

The `-d` switch attempts to de-duplicate candidate passwords across wordlists. If the same word appears in multiple files (or multiple times in one file), it is only hashed once:

```bash
mdxfind -d -f hashes.txt wordlist1.txt wordlist2.txt wordlist3.txt
```

This saves time when using overlapping wordlists but uses additional memory for tracking.

## Extended Truncation Search (-e switch)

By default, mdxfind matches shorter input hashes against common boundaries of longer computed hashes (e.g., the first 32 characters of a SHA-1 hex output). The `-e` switch extends this to check every byte boundary:

```bash
mdxfind -e -f hashes.txt wordlist.txt
```

This is slower but finds hashes that were truncated at unusual positions.

## CR/LF Variations (-l switch)

The `-l` switch tries each candidate with various line-ending characters appended — NUL, CR (`\r`), LF (`\n`), and CR/LF (`\r\n`). Results containing unprintable characters are shown in `$HEX[]` format:

```bash
mdxfind -l -f hashes.txt wordlist.txt
```

This catches cases where a password was hashed with a trailing newline or other control character still attached.

## Thread Count (-t switch)

Control the number of worker threads:

```bash
# Use 4 threads
mdxfind -t 4 -f hashes.txt wordlist.txt

# Use 1 thread (useful for debugging or low-memory systems)
mdxfind -t 1 -f hashes.txt wordlist.txt
```

By default, mdxfind uses all available CPU cores.

## Skip Lines (-w switch)

Skip a number of lines from the beginning of the first wordlist. Useful for resuming an interrupted session:

```bash
# Skip the first 5 million lines
mdxfind -w 5000000 -f hashes.txt large_wordlist.txt
```

## Directory Recursion (-y switch)

Treat wordlist arguments as directories and recurse into them:

```bash
mdxfind -y -f hashes.txt /path/to/wordlist_directory/
```

## Print Source (-p switch)

Print the source filename for each found password. Useful when processing multiple wordlists to know which file contained the winning candidate:

```bash
mdxfind -p -f hashes.txt wordlist1.txt wordlist2.txt wordlist3.txt
```

## Preserve Salts (-v switch)

Normally, when a salted hash is solved, the matching salt is removed from consideration to avoid redundant work. The `-v` switch disables this behavior, keeping all salts active throughout the run. This is important when:

- Salts are reused across multiple hashes in the input
- The salt count may be inaccurate due to combined salt files or earlier processing errors

```bash
mdxfind -v -s salts.txt -h SALT -f hashes.txt wordlist.txt
```

## Hash Generation / Debug Output (-z switch)

The `-z` switch generates hashes for every supported algorithm for each input word. Rather than searching for matches against an input hash list, it outputs every computed hash. This makes mdxfind a universal hash generator for any of its 994+ supported types.

Use `-f /dev/null` since no input hashes are needed:

```
$ echo -e "test\npassword" | mdxfind -z -h '^MD5$,^SHA1$,^SHA256$,^NTLM$' -f /dev/null stdin
MD5x01 098f6bcd4621d373cade4e832627b4f6:test
SHA1x01 a94a8fe5ccb19ba61c4c0873d391e987982fbbd3:test
SHA256x01 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08:test
NTLMx01 0cb6948805f797bf2a82807973b89537:test
MD5x01 5f4dcc3b5aa765d61d8327deb882cf99:password
SHA1x01 5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8:password
SHA256x01 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8:password
NTLMx01 8846f7eaee8fb117ad06bdd830b7586c:password
```

This is useful for:

- **Creating test hash files** for validating mdxfind or other tools
- **Verifying hash computations** — confirm that a known password produces the expected hash
- **Exploring complex types** — see what MD5SHA1MD5 or SHA1SALTPASS actually compute
- **Building reference sets** — generate hashes for a known wordlist across all types

Combined with `-i` (iterations), `-s` (salts), and `-h` (type selection), you can generate hashes for virtually any scheme:

```bash
# Generate iterated MD5 hashes
echo "test" | mdxfind -z -i 1000 -h '^MD5$' -f /dev/null stdin

# Generate salted hashes with a specific salt
echo "test" | mdxfind -z -s salt_file.txt -h MD5SALT -f /dev/null stdin

# Generate all supported hash types for a word
echo "test" | mdxfind -z -h ALL -f /dev/null stdin
```

## Rule Hit Histogram (-Z switch)

After a run with rules, print a histogram showing which rules produced the most matches:

```bash
mdxfind -r best64.rule -Z -f hashes.txt wordlist.txt
```

This helps identify the most effective rules for your target hash lists, so you can build optimized rule files for future sessions.

## Test Vector Generation ($TESTVEC[])

`$TESTVEC[bytes x count]` creates repeating byte patterns as candidate passwords, up to 2,000,000 bytes. This is useful for algorithm validation, FIPS compliance testing, and regression testing.

```bash
# Single NUL byte
echo '$TESTVEC[00 x 1]' | mdxfind -h '^SHA1$' -z -f /dev/null stdin 2>/dev/null
SHA1x01 5ba93c9db0cff93f52b521d7420e43f6eda2784f:$TESTVEC[00 x 1]

# FIPS 140-2 test: byte 0x61 ('a') repeated 1 million times
echo '$TESTVEC[61 x 1000000]' | mdxfind -h '^SHA1$' -z -f /dev/null stdin 2>/dev/null
SHA1x01 34aa973cd4c4daa4f61eeb2bdbad27316534016f:$TESTVEC[61 x 1000000]

# Multiple bytes repeated
echo '$TESTVEC[41 42 43 x 100]' | mdxfind -h '^MD5$' -z -f /dev/null stdin 2>/dev/null
```

## Understanding the Status Line

During a run, mdxfind periodically displays a progress line on stderr:

```
Working on mybigwordlist w=62, line 2771, Found=2809, 0.54Mh/s, 184.73c/s
```

| Field | Meaning |
|-------|---------|
| `w=62` | Work units queued — how "busy" mdxfind is. Low values mean disk I/O is the bottleneck; high values mean the algorithms are the bottleneck. |
| `line 2771` | Current line number in the current wordlist file. |
| `Found=2809` | Running count of hashes solved so far. |
| `0.54Mh/s` | Hash operations per second (millions). Each candidate may produce multiple hash operations across different algorithm types. |
| `184.73c/s` | Candidates per second — useful for slow algorithms like bcrypt. |

The `w=` value is particularly informative: for fast algorithms like MD5, it stays low (mdxfind outpaces disk I/O). For expensive algorithms like bcrypt or PBKDF2, it climbs high as the disk fills the queue faster than the algorithms can consume it.

The `line` number can be used with `-w` to resume an interrupted run: note the line number, subtract a few thousand for safety margin, and use `-w` on the next run.

## Piping mdxfind Output to mdsplit

mdxfind output can be piped directly to mdsplit for real-time organization of results:

```bash
# Crack and organize in one pipeline
cat *.txt | mdxfind -h ALL wordlist.txt | mdsplit *.txt

# With tee to keep a copy of raw results
cat *.txt | mdxfind -h ALL wordlist.txt | tee -a session.res | mdsplit *.txt
```

mdsplit also accepts result files via `-f`:

```bash
# Process previously saved results
mdsplit -f session.res *.txt

# If results lack hash-type prefixes (e.g., hashcat potfile), specify a default type
mdsplit -t MD5 -f hashcat.potfile *.txt
```

## Windows Notes

On the Windows command line, use double quotes instead of single quotes for regex:

```cmd
mdxfind -h "^MD5$" -f hashes.txt wordlist.txt
mdxfind -h "ALL" -h "!salt,!user" -f hashes.txt wordlist.txt
```

## Antivirus False Positives

Some antivirus vendors on VirusTotal occasionally flag mdxfind as a coin miner, due to its hashing features. This is a false positive — mdxfind does not mine cryptocurrency. It is a hash-cracking tool for security research.

## Rule-Based Password Mutations (-r and -R switches)

mdxfind has built-in support for hashcat/JtR-compatible rules, applied directly during hash search. This is far more efficient than generating candidates externally and piping them in, because mdxfind applies the rules inside its inner loop and avoids I/O overhead. See [RULES.md](RULES.md) for the complete rule reference.

### Basic rule usage (-r)

Create a rule file with one rule per line:

```
# rules.txt
l          # lowercase
u          # uppercase
c          # capitalize first letter
d          # duplicate word
$1         # append "1"
^1         # prepend "1"
r          # reverse
```

Apply it:

```
mdxfind -r rules.txt -f hashes.txt wordlist.txt
```

Given a wordlist containing "password", "test", and "hello", mdxfind will try every rule against every word: PASSWORD, TEST, HELLO, Password, Test, Hello, passwordpassword, testtest, hellohello, password1, test1, hello1, 1password, 1test, 1hello, drowssap, tset, olleh — in addition to the original words.

```
$ mdxfind -r rules.txt -f hashes.txt wordlist.txt
7 rules read from rules.txt
MD5x01 01ee9547a3f708f8fd986216bffd1eb7:1password
MD5x01 b497dd1a701a33026f7211533620780d:drowssap
MD5x01 033bd94b1168d7e4f0d644c3c95e35bf:TEST
MD5x01 23b431acfeb41e15d466d75de822307c:hellohello
23 total rule-generated passwords tested
4 Total hashes found
```

### Multiple rule files — concatenated (-r)

Using `-r` multiple times concatenates the rule sets:

```
# r1.txt: c, $1, $!
# r2.txt: $2, $3
mdxfind -r r1.txt -r r2.txt -f hashes.txt wordlist.txt
# Result: 5 rules (c, $1, $!, $2, $3)
```

Each rule is applied independently to each word.

### Multiple rule files — dot-product (-R)

Using `-R` creates a cross-product of the rule sets. Each rule from the first file is combined with each rule from the second file:

```
# r1.txt: c, $1, $!
# r2.txt: $2, $3
mdxfind -R r1.txt -R r2.txt -f hashes.txt wordlist.txt
# Result: 6 combined rules: c$2, c$3, $1$2, $1$3, $!$2, $!$3
```

For the word "password", this produces: Password2, Password3, password12, password13, password!2, password!3.

```
$ mdxfind -R r1.txt -R r2.txt -f hashes.txt wordlist.txt
6 total rules in use
MD5x01 c24a542f884e144451f9063b79e7994e:password12
MD5x01 ee684912c7e588d03ccb40f17ed080c9:password13
MD5x01 6f9dff5af05096ea9f23cc7bedd65683:Password2
MD5x01 874fcc6e14275dde5a23319c9ce5f8e4:Password3
4 Total hashes found
```

The dot-product is powerful for combining transformation rules (capitalize, toggle case) with suffix/prefix rules (append digits, special characters). Two rule files of 100 rules each produce 10,000 combined rules with `-R`.

### Commonly used rules

Popular hashcat rule files work directly with mdxfind:

- `best64.rule` — 64 most effective rules from hashcat
- `rockyou-30000.rule` — top 30,000 rules derived from the Rockyou leak
- `toggles.rule` — case-toggling variations
- `leetspeak.rule` — leet substitutions (a→@, e→3, etc.)

```
mdxfind -r best64.rule -f hashes.txt wordlist.txt >> results.res
```

### Rule statistics (-Z)

Use `-Z` to see which rules produce the most hits:

```
mdxfind -r rules.txt -Z -f hashes.txt wordlist.txt
```

This prints a histogram of rule effectiveness after the run, helping you identify which rules are worth keeping for future sessions.

### procrule: standalone rule processor

For cases where you need to generate a candidate wordlist (e.g., for feeding into hashcat or other tools), [procrule](https://github.com/Cynosureprime/procrule) applies the same rule syntax as a standalone tool:

```
procrule -r rules.txt wordlist.txt > candidates.txt
```

procrule only emits words that were actually changed by a rule — if a rule produces the same output as the input, it is suppressed. This makes the output strictly additive to the original wordlist.
