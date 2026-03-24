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
