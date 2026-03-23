# mdxfind

Multi-threaded, multi-algorithm hash search engine. Searches wordlists against large hash collections across 994 hash types simultaneously, using Judy arrays for memory-efficient hash storage and SIMD acceleration on supported platforms. Includes **mdsplit**, a companion tool that separates solved hashes by type into organized output files.

Uses [yarn.c](https://github.com/madler/pigz) for threading, [libJudy](https://judy.sourceforge.net/) for compressed hash lookup, and [hashpipe](https://github.com/Cynosureprime/hashpipe) for hash verification.

## Overview

mdxfind is designed for processing very large hash collections (100+ million hashes) against wordlists, with optional rules, salts, usernames, peppers, and hybrid mask attacks. It can:

- Test every word against every loaded hash across all selected algorithms in a single pass
- Apply password mangling rules in concatenated or dot-product form
- Append or prepend character masks to each candidate
- Handle salts, usernames, peppers, and suffixes from separate files or embedded in the hash file
- Deduplicate wordlists on the fly
- Expand passwords to Unicode, XML-escape special characters, or munge email addresses
- Rotate calculated hashes to match truncated or manipulated input hashes
- Output results in a standardized `TYPE hash[:salt]:password` format consumed by mdsplit

### Typical workflow

```
                 wordlists
                    |
                    v
hash file ---> mdxfind ---> stdout (solved) ---> mdsplit ---> per-type .txt files
                    |
                    v
               stderr (progress/stats)
```

1. Load hashes from stdin or `-f` file
2. Load salts (`-s`), usernames (`-u`), peppers (`-j`), suffixes (`-k`) if needed
3. Process wordlists with optional rules (`-r`, `-R`) and masks (`-n`, `-N`)
4. Pipe solved results through mdsplit to organize by hash type

## Usage

```
mdxfind [options] [wordlist ...] < hashfile
mdxfind -f hashfile [options] [wordlist ...]
```

### Options

**Hash selection:**

| Option | Description |
|--------|-------------|
| `-h REGEX` | Select hash types by regex, comma-separated. Use `!` to negate, `.` for all. Multiple `-h` allowed |
| `-m MODE` | Select by hashcat mode or internal index: `-m 0` (MD5), `-m e1-e10` (range), `-m 0,100,e369` (mixed) |
| `-M TYPE` | Select type for `-F` embedded-salt loading (e.g., `-M e373`) |

**Hash input:**

| Option | Description |
|--------|-------------|
| `-f FILE` | Read hashes from file (instead of stdin). Allows stdin to be used for wordlists |
| `-F FILE` | Read hashes with embedded salts in `hash:salt` format. Requires `-M` to select type |
| `-s FILE` | Read salts from file (one per line) |
| `-u FILE` | Read usernames from file |
| `-j FILE` | Read peppers (prefixes) from file |
| `-k FILE` | Read suffixes from file |
| `-i N` | Iteration count for iterated hash types |

**Password manipulation:**

| Option | Description |
|--------|-------------|
| `-r FILE` | Apply rules (concatenated form) |
| `-R FILE` | Apply rules (dot-product form) |
| `-n SPEC` | Append mask/digits: `-n 2` (2 digits), `-n 3x` (3 hex), `-n '?l?d'` (letter+digit), `-n '?[0-9a-f]?[0-9a-f]'` |
| `-N SPEC` | Prepend mask/digits (same syntax as `-n`) |
| `-a` | Email address munging (try local part, domain, variations) |
| `-b` | Expand each word to Unicode (UTF-16LE), best effort |
| `-c` | Replace special chars (`<>&`, etc.) with XML equivalents |
| `-d` | Deduplicate wordlists, best effort |

**Search behavior:**

| Option | Description |
|--------|-------------|
| `-e` | Extended search for truncated hashes |
| `-g` | Rotate calculated hashes to attempt match against input |
| `-q N` | Internal iteration count for composed types (SHA1MD5x, etc.) |
| `-v` | Do not mark salts as found (continue searching all salts) |
| `-w N` | Skip N lines from first wordlist |
| `-y` | Enable directory recursion for wordlists |

**Output and control:**

| Option | Description |
|--------|-------------|
| `-t N` | Number of threads (default: number of CPUs) |
| `-p` | Print source filename of found plaintexts |
| `-l` | Append CR/LF/CRLF and print in hex |
| `-z` | Debug mode: print all computed hash results |
| `-Z` | Histogram of rule hits |
| `-V` | Display version and exit |

### Hash Type Selection

The `-h` option accepts Perl-compatible regular expressions to filter hash types by name:

```bash
# All MD5 variants
mdxfind -h MD5 -f hashes.txt wordlist.txt

# SHA1 and SHA256 only
mdxfind -h 'SHA1$,SHA256$' -f hashes.txt wordlist.txt

# Everything except NTLM
mdxfind -h '!NTLM' -f hashes.txt wordlist.txt

# All types
mdxfind -h '.' -f hashes.txt wordlist.txt
```

The `-m` option accepts hashcat mode numbers, internal `eN` indices, or ranges:

```bash
# Hashcat mode 0 = MD5
mdxfind -m 0 -f hashes.txt wordlist.txt

# Multiple hashcat modes
mdxfind -m 0,100,1000 -f hashes.txt wordlist.txt

# Internal index range
mdxfind -m e1-e12 -f hashes.txt wordlist.txt

# Mixed
mdxfind -m 0,e369,3200 -f hashes.txt wordlist.txt
```

### Input Format

Hash files contain one hash per line:

```
5f4dcc3b5aa765d61d8327deb882cf99
e10adc3949ba59abbe56e057f20f883e
```

For salted types with `-s`, salts are in a separate file. For `-F` embedded salts, the format is `hash:salt`:

```
d06be999220ca97b73b14db78492be76:Kp
8e1856406c4f9f18ae1717b6f88fde35:a
```

### Output Format

Solved hashes go to stdout in the format:

```
TYPExNN hash[:salt]:password
```

Examples:
```
MD5x01 5f4dcc3b5aa765d61d8327deb882cf99:password
MD5PASSSALTx01 d06be999220ca97b73b14db78492be76:Kp:password123
SHA256x01 5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8:password
BCRYPTx01 $2a$05$LhayLxezLhK1LhWvKxCyLOj0j1u.Kj0jZ0pEmm134uzrQlFvQJLF6:password
```

The `xNN` suffix indicates the iteration count (x01 = single hash, x02 = double, etc.).

Progress and statistics go to stderr.

## Examples

### Basic hash cracking

```bash
# Simple MD5 search
mdxfind -h MD5 -f md5hashes.txt wordlist.txt

# All types, multiple wordlists
mdxfind -h '.' -f hashes.txt dict1.txt dict2.txt dict3.txt

# Read hashes from stdin
cat hashes.txt | mdxfind -h '.' wordlist.txt
```

### Salted hashes

```bash
# External salt file
mdxfind -h MD5SALT -f hashes.txt -s salts.txt wordlist.txt

# Embedded salts with -F
mdxfind -M e373 -F hash_salt_file.txt wordlist.txt

# Salts + usernames
mdxfind -h '.' -f hashes.txt -s salts.txt -u users.txt wordlist.txt
```

### Rules and masks

```bash
# Apply hashcat-style rules
mdxfind -h '.' -f hashes.txt -r rules.txt wordlist.txt

# Append 2 digits
mdxfind -h '.' -f hashes.txt -n 2 wordlist.txt

# Append custom mask (letter + digit)
mdxfind -h '.' -f hashes.txt -n '?l?d' wordlist.txt

# Prepend 4 digits
mdxfind -h '.' -f hashes.txt -N 4 wordlist.txt

# Combine rules with mask
mdxfind -h '.' -f hashes.txt -r rules.txt -n 2 wordlist.txt
```

### Iterated hashes

```bash
# Up to 5 iterations
mdxfind -h '.' -f hashes.txt -i 5 wordlist.txt

# SHA256CRYPT with specific iterations
mdxfind -h SHA256CRYPT -f hashes.txt -i 5000 wordlist.txt
```

### Full pipeline with mdsplit

```bash
# Crack and sort results by type
cat *.txt | mdxfind -h '.' -i 5 -s salts.txt wordlist.txt | mdsplit *.txt

# Multi-pass until convergence
for pass in 1 2 3; do
  mdxfind -f unsolved.txt -h '.' -i 5 -s salts.txt wordlist.txt | mdsplit unsolved.txt
done
```

### Performance tuning

```bash
# Limit to 8 threads
mdxfind -h '.' -f hashes.txt -t 8 wordlist.txt

# Deduplicate wordlists on the fly
mdxfind -h '.' -f hashes.txt -d wordlist.txt

# Recurse into directory of wordlists
mdxfind -h '.' -f hashes.txt -y /path/to/wordlists/

# Skip first 1M lines of wordlist (resume)
mdxfind -h '.' -f hashes.txt -w 1000000 wordlist.txt
```

## Architecture

### Threading Model

mdxfind uses a producer-consumer architecture with [yarn.c](https://github.com/madler/pigz) (Mark Adler's thread pool abstraction over pthreads):

- **Main thread** reads wordlist lines into fixed-size job buffers and dispatches them to worker threads
- **Worker threads** (one per CPU by default) process each word against all loaded hashes across all selected types
- Job buffers are recycled through a pipeline to minimize allocation overhead

### Hash Storage

Hash lookup uses [Judy arrays](https://judy.sourceforge.net/) (HP's compressed associative data structure), providing O(1) lookup with dramatically lower memory usage than conventional hash tables:

- ~100 million MD5 hashes fit in ~1.5 GB (vs. ~7.2 GB in a flat hash table)
- Hash values are split into prefix (Judy key) and suffix (compact table) for efficient storage
- Per-type salt arrays (`Typesalt[]`) store salt values associated with each hash

### Compact Hash Table

For hex hashes loaded via `-f` or `-F`, mdxfind uses a compact table that stores only the hash bytes needed for comparison, indexed by a prefix extracted from the hash value. This reduces memory usage while maintaining O(1) amortized lookup with overflow chaining.

### SIMD Acceleration

On x86_64 platforms, mdxfind uses SSE/SSE2 intrinsics for:
- Bulk hash comparison against the compact table
- MD5 computation (custom implementation in `mymd5.c`)
- SHA-1 computation (Intel SHA-NI via `sha1_shani.c`, fallback assembly in `sha1_block.s`)

ARM platforms use NEON equivalents where available. PowerPC uses AltiVec. Non-Intel platforms use portable C fallbacks.

### Rule Processing

mdxfind supports hashcat-compatible password mangling rules in two modes:

- **Concatenated rules** (`-r`): Each rule is a sequence of operations applied in order to each candidate word. One rule per line.
- **Dot-product rules** (`-R`): Multiple rule files whose operations are combined as a cross-product, generating all combinations.

The rule engine (`ruleproc.c`) supports the standard hashcat rule set including case manipulation, character insertion/deletion/replacement, rotation, duplication, truncation, and memory operations.

### Hybrid Mask Attacks

The `-n` and `-N` options append or prepend character patterns to each candidate:

| Spec | Description |
|------|-------------|
| `2` | Two decimal digits (00-99) |
| `3x` | Three hex digits (000-fff) |
| `?l` | One lowercase letter (a-z) |
| `?u` | One uppercase letter (A-Z) |
| `?d` | One digit (0-9) |
| `?s` | One special character |
| `?a` | Any printable character |
| `?[charset]` | Custom character class |

Masks are composable: `-n '?u?l?d'` appends uppercase+lowercase+digit (e.g., `passwordAb3`).

### Hash Rotation

The `-g` flag enables hash rotation: after computing a hash, mdxfind rotates the hash value byte-by-byte and checks each rotation against the loaded hashes. This catches truncated or byte-swapped hashes that appear in some databases.

## Supported Hash Types

mdxfind supports 994 hash types, covering 297 hashcat mode mappings. Run `mdxfind -h` (with no other arguments) to see the complete list.

### Options column key

| Code | Meaning |
|------|---------|
| `f` | Supports `-f` (hex hash file loading) |
| `f,s` | Supports `-f` and `-s` (hex hash + external salt) |
| `J` | JSON/structured format (parsed internally) |
| `J,s` | JSON format with salt support |

### Unsalted types (selected)

| Index | Type | Algorithm | hashcat |
|-------|------|-----------|---------|
| e1 | MD5 | `md5($pass)` | 0 |
| e2 | MD5UC | `md5($pass)` uppercase hex | 4300 |
| e3 | MD4 | `md4($pass)` | 900 |
| e8 | SHA1 | `sha1($pass)` | 100 |
| e10 | SHA256 | `sha256($pass)` | 1400 |
| e12 | SHA512 | `sha512($pass)` | 1700 |
| e13 | GOST | `gost($pass)` | 6900 |
| e17 | RMD160 | `ripemd160($pass)` | -- |
| e18 | TIGER | `tiger($pass)` | -- |
| e84-e87 | KECCAK | Keccak-224/256/384/512 | 17700-18000 |
| e88-e91 | SHA3 | SHA3-224/256/384/512 | 17300-17600 |
| e369 | NTLM | `md4(utf16le($pass))` | 1000 |
| e988 | SM3 | `sm3($pass)` | 31100 |

### Salted types (selected)

| Index | Type | Algorithm | hashcat |
|-------|------|-----------|---------|
| e31 | MD5SALT | `md5(hex(md5($pass)).$salt)` | 2611 |
| e373 | MD5PASSSALT | `md5($pass.$salt)` | -- |
| e394 | MD5SALTPASS | `md5($salt.$pass)` | -- |
| e385 | SHA1SALTPASS | `sha1($salt.$pass)` | -- |
| e405 | SHA1PASSSALT | `sha1($pass.$salt)` | -- |
| e412 | SHA256SALTPASS | `sha256($salt.$pass)` | -- |
| e413 | SHA256PASSSALT | `sha256($pass.$salt)` | -- |
| e386 | SHA512PASSSALT | `sha512($pass.$salt)` | -- |
| e388 | SHA512SALTPASS | `sha512($salt.$pass)` | -- |
| e411 | MD5SALTPASSSALT | `md5($salt.$pass.$salt)` | 3800 |
| e439 | MSCACHE | `md4(md4(utf16le($pass)).$salt)` | 1100 |
| e857 | SKYPE | `md5($pass.\|$salt)` | 23500 |
| e896 | ORACLE7 | Oracle 7 DES | 3100 |
| e897 | NETNTLMV1 | NTLMv1 challenge-response | 5500 |
| e898 | NETNTLMV2 | NTLMv2 challenge-response | 5600 |

### Composed/chained types (selected)

| Index | Type | Algorithm | hashcat |
|-------|------|-----------|---------|
| e123 | MD5MD5PASS | `md5(hex(md5($pass)).$pass)` | 2810 |
| e160 | SHA1MD5 | `sha1(hex(md5($pass)))` | 4400 |
| e178 | MD5SHA1 | `md5(hex(sha1($pass)))` | 4700 |
| e188 | MD5SHA1MD5 | `md5(hex(sha1(hex(md5($pass)))))` | -- |
| e251 | SHA256SHA1 | `sha256(hex(sha1($pass)))` | -- |
| e368 | MD5NTLM | `md5(hex(md4(utf16le($pass))))` | -- |
| e497 | MD4UTF16MD5 | `md4(utf16le(hex(md5($pass))))` | -- |

### Crypt types

| Index | Type | Algorithm | hashcat |
|-------|------|-----------|---------|
| e500 | DESCRYPT | DES `crypt()` | 1500 |
| e511 | MD5CRYPT | `$1$` md5crypt | 500 |
| e512 | SHA256CRYPT | `$5$` sha256crypt | 7400 |
| e513 | SHA512CRYPT | `$6$` sha512crypt | 1800 |
| e450 | BCRYPT | `$2a$`/`$2b$` bcrypt | 3200 |
| e884 | SCRYPT | `$7$` scrypt | 8900 |

### PBKDF2 / KDF types

| Index | Type | Algorithm | hashcat |
|-------|------|-----------|---------|
| e529 | CISCO8 | `$8$` PBKDF2-SHA256 | 9200 |
| e530 | PBKDF2-SHA256 | PBKDF2-HMAC-SHA256 | 10900 |
| e531 | PBKDF2-MD5 | PBKDF2-HMAC-MD5 | -- |
| e532 | PBKDF2-SHA1 | PBKDF2-HMAC-SHA1 | 12001 |
| e533 | PBKDF2-SHA512 | PBKDF2-HMAC-SHA512 | 12100 |
| e899 | LASTPASS | PBKDF2+AES | 6800 |
| e987 | ARGON2 | Argon2id | 34000 |

### LDAP SSHA types

| Index | Type | Algorithm | hashcat |
|-------|------|-----------|---------|
| e833 | SSHA1BASE64 | `{SSHA}base64(sha1($p.$s).$s)` | 111 |
| e835 | SSHA256BASE64 | `{SSHA256}base64(sha256($p.$s).$s)` | 1411 |
| e836 | SSHA512BASE64 | `{SSHA512}base64(sha512($p.$s).$s)` | 1711 |

### Additional algorithm families

mdxfind also supports GOST-CRYPTO, Streebog-256/512, gost12512crypt, RIPEMD-128/320, HAVAL (128/160/192/224/256-bit with 3/4/5 rounds), BLAKE-224/256/384/512, BMW, CubeHash, ECHO, Fugue, Groestl, Hamsi, JH, Luffa, Panama, RadioGatun, Shabal, SHAvite, SIMD, Skein, Whirlpool, MD6 (128/256/512), MDC2, EDON-256/512, Snefru-128/256, HAS-160, BLAKE2B/2S, SM3, and hundreds of composed/chained variants of these algorithms.

Application-specific types include WPA-PMKID/EAPOL, Kerberos 5 Pre-Auth and DB (etype 17/18), TACACS+, JWT, Apple Secure Notes/Keychain/iWork/APFS, Ansible Vault, Bitwarden, MongoDB SCRAM, SolarWinds Orion, VMware VMX, SQLCipher, PostgreSQL SCRAM, AWS Signature v4, QNX shadow, SAP BCODE/PASSCODE, Cisco PIX/ASA/IOS, FortiGate, Drupal 7, PHPBB3, WordPress, and many more.

## mdsplit

**mdsplit** is a companion tool that processes mdxfind output, splitting solved hashes by type into separate files and removing them from unsolved hash files.

### Usage

```
mdsplit [options] [hashfile ...]
```

### Options

| Option | Description |
|--------|-------------|
| `-V` | Display version and exit |
| `-h` or `-?` | Show help |
| `-a` | Process all files, not just `.txt` |
| `-l` | No file locking (for filesystems that don't support it) |
| `-r` | No reverse hash scanning (default: scan for reverse matches) |
| `-f FILE` | Read results from file instead of stdin |
| `-t TYPE` | Default type for untyped results (e.g., `-t MD5x01`) |
| `-b SIZE` | Buffer size: `500K` (default), `1M`, `1G` |
| `-p STR` | Prepend string to solution filename |
| `-i STR` | Insert string after `.` in solution filename |
| `-s STR` | Append string to solution filename |

### Examples

```bash
# Pipe mdxfind output directly
cat *.txt | mdxfind -i 3 /tmp/words | mdsplit *.txt

# Process saved results
mdsplit -f results.txt unsolved_files/

# Add prefix to output filenames
mdsplit -p solved_ -f results.txt *.txt
```

### How it works

1. Reads mdxfind-format results from stdin or `-f`
2. For each result line (`TYPExNN hash:password`), writes to a type-specific output file (e.g., `MD5x01.txt`, `SHA1x01.txt`)
3. Scans the specified hash files for matching hashes and removes them (the hash is now solved)
4. Uses Judy arrays for O(1) hash lookups, handling large files efficiently

## Memory Efficiency

mdxfind is designed to handle massive hash collections. Judy array compression provides dramatically better memory efficiency than flat hash tables:

| Hashes | mdxfind (Judy) | Flat table | Savings |
|--------|---------------|------------|---------|
| 1M | ~15 MB | ~72 MB | 4.8x |
| 10M | ~150 MB | ~720 MB | 4.8x |
| 100M | ~1.5 GB | ~7.2 GB | 4.8x |

The compact hash table adds a small per-hash overhead for fast prefix matching, but the dominant memory cost is the Judy array. This makes it practical to load hash databases that would exhaust memory with conventional approaches.

## Building

```bash
make deps      # pull and build all dependencies from source
make all       # build mdxfind and mdsplit
```

`make deps` clones each dependency from its authoritative GitHub repository, pins it to a verified commit hash, and builds a static library. This requires git, a C compiler, make, autotools (for libmhash and libJudy), and yasm (for x86_64 SHA-1 assembly).

If you already have the required static libraries (from a previous `make deps` or a manual build), `make all` is sufficient.

To remove downloaded dependency sources:
```bash
make distclean
```

### Docker

Docker can be used to build and run mdxfind without installing dependencies locally:

```bash
docker build . -t csp/mdxfind
```

```bash
# Run mdxfind
docker run -v ${PWD}:/data -it --rm csp/mdxfind -h MD5 -f /data/hashes.txt /data/wordlist.txt

# Run mdsplit
docker run -v ${PWD}:/data -it --rm --entrypoint mdsplit csp/mdxfind -f /data/results.txt /data/*.txt
```

The `/data` directory inside the container is used as the working directory.

### Dependencies

mdxfind requires the following static libraries (all built automatically by `make deps`):

| Library | Archive | Purpose |
|---------|---------|---------|
| [OpenSSL](https://github.com/openssl/openssl) 1.1.1w | `libssl.a`, `libcrypto.a` | MD5, SHA, DES, AES, HMAC, EVP, PBKDF2 |
| [sphlib](https://github.com/pornin/sphlib) | `libsph.a` | 50+ hash algorithms (BLAKE, Keccak, Groestl, Skein, etc.) |
| [libmhash](https://github.com/Distrotech/mhash) | `libmhash.a` | Tiger, Haval, Snefru, CRC32 |
| [RHash](https://github.com/rhash/RHash) | `librhash.a` | GOST, EDON-R, HAS-160 |
| [MD6](https://github.com/brandondahler/retter) | `md6.a` | MD6 (Ron Rivest reference implementation) |
| [Streebog](https://github.com/mjosaarinen/brutus) | `gosthash/gost2012/gost2012.a` | GOST R 34.11-2012 (Streebog) |
| [bcrypt](https://github.com/openwall/crypt_blowfish) | `bcrypt-master/bcrypt.a` | bcrypt password hashing |
| [Argon2](https://github.com/P-H-C/phc-winner-argon2) | `argon2/argon2.a` | Argon2id (PHC winner, RFC 9106) |
| [libJudy](https://github.com/netdata/libjudy) | `libJudy.a` | Judy arrays (compressed associative data structure) |
| [yescrypt](https://github.com/openwall/yescrypt) | `yescrypt/*.o` | yescrypt/scrypt KDF |
| [PCRE](https://github.com/luvit/pcre) 8.45 | `libpcre.a` | Perl-compatible regular expressions (for `-h` type selection) |
| LM hash | `lm/lm.a` | LM/NTLM hash support (bundled) |

### Known Build Issues

**yasm required (x86_64 only)**: The Intel SHA-1 assembly (`sha1_block.s`) requires [yasm](https://yasm.tortall.net/). On macOS: `port install yasm` or `brew install yasm`. On Debian/Ubuntu: `apt install yasm`. On non-x86_64 platforms, this file is not used.

**sphlib BMW strict aliasing bug (GCC 12+)**: sphlib's `bmw.c` contains strict aliasing violations that cause GCC to generate incorrect code for BMW-224 and BMW-256 at `-O2` and above. Apple clang is unaffected. The `make deps` target applies the workaround (`-fno-strict-aliasing`). See [sphlib#3](https://github.com/pornin/sphlib/issues/3).

### Supported Platforms

The Makefile detects the build platform automatically. Tested on:

- macOS x86\_64 and arm64 (requires libiconv from MacPorts)
- Linux x86\_64 (Ubuntu 18.04, 22.04)
- Linux i386 (32-bit)
- Linux ppc64le (PowerPC 8)
- FreeBSD 13.2 x86\_64 (uses gmake)
- Windows x86, x64, and ARM64 (cross-compiled via mingw-w64 / llvm-mingw)

## Type Indices and Hashcat Modes

mdxfind uses internal type indices (`e1` through `e994`). The `-m` option accepts both internal indices (with `e` prefix) and hashcat mode numbers (bare numbers):

```bash
# Internal indices
mdxfind -m e1,e8,e369 -f hashes.txt wordlist.txt

# Hashcat mode numbers
mdxfind -m 0 -f hashes.txt wordlist.txt          # MD5 (hashcat mode 0)
mdxfind -m 1000 -f hashes.txt wordlist.txt       # NTLM (hashcat mode 1000)
mdxfind -m 3200 -f hashes.txt wordlist.txt       # bcrypt (hashcat mode 3200)

# Mixed: hashcat modes and internal indices together
mdxfind -m 1000,e1,3200 -f hashes.txt wordlist.txt

# Ranges (internal indices only)
mdxfind -m e1-e12 -f hashes.txt wordlist.txt
```

Run `mdxfind -h` (with no other options) to see the full type list with hashcat mode mappings.

## $HEX[] Encoding

mdxfind uses `$HEX[]` encoding for passwords containing non-printable or binary characters. When a password contains bytes outside the printable ASCII range, it is output as `$HEX[hexbytes]`:

```
MD5x01 abc123:$HEX[c3a96e67737472c3b66d]
```

This encoding was invented by Cynosure Prime and is now a de facto standard across hash cracking tools. It enables lossless representation of arbitrary binary passwords in text-based output formats.

In wordlists, `$HEX[]` encoded passwords are processed natively -- mdxfind decodes them before hashing.

## Acknowledgments

mdxfind depends on the following libraries:

- [OpenSSL](https://github.com/openssl/openssl) -- OpenSSL Project
- [yarn.c](https://github.com/madler/pigz) -- Mark Adler (from pigz)
- [libJudy](https://judy.sourceforge.net/) -- Doug Baskins (Hewlett-Packard)
- [sphlib](https://github.com/pornin/sphlib) -- Thomas Pornin (Projet RNRT SAPHIR)
- [RHash](https://github.com/rhash/RHash) -- RHash Project
- [libmhash](https://mhash.sourceforge.net/) -- Nikos Mavroyanopoulos, Sascha Schumann
- [PCRE](https://www.pcre.org/) -- Philip Hazel (University of Cambridge)
- [bcrypt](https://www.openwall.com/crypt/) -- Niels Provos, David Mazieres (via Openwall crypt_blowfish)
- [yescrypt](https://www.openwall.com/yescrypt/) -- Alexander Peslyak (via Openwall)
- [Argon2](https://github.com/P-H-C/phc-winner-argon2) -- Alex Biryukov, Daniel Dinu, Dmitry Khovratovich (University of Luxembourg; PHC winner, RFC 9106)
- [stribob](https://github.com/mjosaarinen/brutus) -- Markku-Juhani O. Saarinen (Streebog/GOST R 34.11-2012 primitives; standalone wrapper from stricat bundled with permission)

## License

MIT
