# How To: Hash Recovery Workflow with mdxfind

This guide describes a practical workflow for managing and solving large collections of password hashes using mdxfind and its companion tools.

## Overview

The workflow has four stages:

1. **Receive** — a hash list arrives in some format
2. **Prepare** — clean and normalize it into a standard input file
3. **Solve** — run mdxfind against wordlists to find passwords
4. **Organize** — use mdsplit to sort results by hash type back into the original lists

## File Naming Convention

Each hash list gets a name (or number), and files associated with it follow a consistent naming scheme:

| File | Purpose |
|------|---------|
| `listname.orig` | The original file, exactly as received. Never modified after initial save. |
| `listname.txt` | The cleaned hash list — one hash per line, ready for mdxfind. |
| `listname.MD5x01` | Solved MD5 hashes (created by mdsplit). |
| `listname.SHA1x02` | Solved SHA1 hashes, and so on for each hash type discovered. |
| `listname.res` | Raw result file from mdxfind (before mdsplit processing). |

The `.orig` file is your insurance policy — the untouched original. The `.txt` file is what you actually work with.

## Stage 1: Receiving a Hash List

Hash lists come from many sources and in many formats. Save the raw file as-is:

```bash
cp incoming_hashes.txt 50m.orig
```

## Stage 2: Preparing the Input

Real-world hash lists are frequently messy: they may contain comments, blank lines, headers, corrupted characters, mixed formats (some lines with usernames, some without), or encoding issues. The goal is to produce a clean `.txt` file with one hash per line.

Common cleanup tasks:

```bash
# Remove comment lines and blank lines
grep -v '^#' 50m.orig | grep -v '^$' > 50m.txt

# Extract hashes from username:hash format
cut -d: -f2 50m.orig > 50m.txt

# For salted hashes, keep the salt — mdxfind understands hash:salt format
# Just remove non-hash lines
grep -v '^#' salted_list.orig > salted_list.txt
```

The exact cleanup depends on the source. The key principle: `listname.txt` should contain only hash lines that mdxfind can parse.

## Stage 3: Solving with mdxfind

### Basic usage

Feed one or more hash files to mdxfind with `-f`, and provide wordlists as arguments:

```bash
mdxfind -f hashes.txt wordlist.txt
```

### Working on multiple lists simultaneously

A major strength of mdxfind is its ability to search across many hash lists at once. By default, mdxfind reads hashes from stdin, so you can concatenate all your `.txt` files and pipe them in:

```bash
cat *.txt | mdxfind wordlist.txt
```

Or use `-f` to load hashes from a file, which frees up stdin for other uses:

```bash
mdxfind -f hashes.txt wordlist.txt
```

mdxfind handles millions of hashes efficiently using Judy arrays for compressed storage.

In practice, a typical session looks like:

```bash
cat *.txt big/*.txt salted/*.txt | mdxfind -h ALL wordlist.txt > results.res
```

The `-h ALL` flag tells mdxfind to try all hash types. You can also restrict to specific types:

```bash
# Only MD5 through SHA256
mdxfind -m e1-e10 -f hashes.txt wordlist.txt

# Only salted MD5
mdxfind -h MD5SALT -f salted.txt wordlist.txt
```

### Saving results

mdxfind outputs results to stdout in `HASHTYPE hash:password` format:

```
MD5x01 028b5d7e02583176dc8b5123e1300481:mich1ael
MD5x01 1ab9c7052cd5defb4c930f831d670eb7:cabo11
SHA1x02 a94a8fe5ccb19ba61c4c0873d391e987982fbbd3:test
```

Redirect to a `.res` file:

```bash
cat *.txt | mdxfind -h ALL wordlist.txt >> session.res
```

Use `>>` (append) so you can run multiple sessions without losing earlier results.

### Salt files

For salted hash types, provide salt files with `-s`:

```bash
cat *.txt salted/*.txt | mdxfind -s salts.txt -h MD5SALT wordlist.txt >> session.res
```

### Iterative solving

Hash recovery is iterative. After each run, extract the found passwords and use them as a new wordlist (passwords from one list often work on others):

```bash
# Extract passwords from results
getpass session.res > found_passwords.txt

# Remove duplicates
sort -u found_passwords.txt > unique_passwords.txt

# Feed them back
cat *.txt | mdxfind -h ALL unique_passwords.txt >> session.res
```

## Stage 4: Organizing Results with mdsplit

After accumulating results, use mdsplit to sort them back into per-list, per-type files:

```bash
cat session.res | mdsplit *.txt big/*.txt salted/*.txt
```

mdsplit reads the result lines, matches each hash back to its source `.txt` file, and writes it to the appropriate `listname.HASHTYPE` file. For example, if `50m.txt` contained hash `028b5d...` and it was solved as MD5, mdsplit creates or appends to `50m.MD5x01`.

This is how your directory evolves over time:

```
50m.orig          # Original, untouched
50m.txt           # Input hashes (gets shorter as you remove solved ones)
50m.MD5x01        # Solved MD5 hashes
50m.SHA1x02       # Solved SHA1 hashes
50m.MD5SALTx01    # Solved salted MD5 hashes
```

### Filtering solved hashes from input

Optionally, after running mdsplit, you can remove already-solved hashes from `.txt` files to speed up future runs. Use rling to subtract found hashes:

```bash
# Extract just the hashes from solved results
cut -d' ' -f2 50m.MD5x01 | cut -d: -f1 > solved_hashes.txt

# Remove them from the input
rling 50m.txt solved_hashes.txt > 50m_remaining.txt
mv 50m_remaining.txt 50m.txt
```

## Companion Tools

mdxfind is part of an ecosystem of tools that work together:

| Tool | Purpose |
|------|---------|
| **mdxfind** | Multi-algorithm hash search engine |
| **mdsplit** | Sorts mdxfind results by hash type and source list |
| **getpass** | Extracts passwords from result files, handles `$HEX[]` encoding |
| **hashpipe** | Hash verification — confirms found passwords are correct |
| **procrule** | Rule-based password mutation engine |
| **rling** | High-speed line deduplication and subtraction |

*Detailed guides for getpass, procrule, and other tools will be added in future updates.*

## Tips

- **Start broad, narrow later.** Use `-h ALL` first to discover what hash types are present, then target specific types in follow-up runs.
- **Append results.** Always use `>>` to append to result files. You'll run many sessions over days or weeks.
- **Run mdsplit periodically.** It's cheap to run and keeps your solved hashes organized.
- **Keep `.orig` files forever.** You may need to re-extract hashes if you discover the format was parsed incorrectly.
- **Use getpass to recycle passwords.** Passwords found in one list are often reused in others. Extract them and feed them back through mdxfind.
