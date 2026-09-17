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

A major strength of mdxfind is its ability to search across many hash lists at once. By default, mdxfind reads hashes from stdin, so you can concatenate all your `.txt` files and pipe them in.  Unlike virtually all other hash cracking programs, mdxfind does not care about exact hash lengths, or specific formats (with some exceptions for complex hashes).  Basically, as long as it is mostly there, mdxfind can probably parse it.

```bash
cat *.txt | mdxfind wordlist.txt
```

Or use `-f` to load hashes from a file, which frees up stdin for other uses:

```bash
mdxfind -f hashes.txt wordlist.txt
```

mdxfind handles millions, or billions of hashes efficiently using Judy arrays for compressed storage.

In practice, a typical session looks like:

```bash
cat *.txt big/*.txt salted/*.txt | mdxfind -h ALL wordlist.txt > results.res
```

The `-h ALL` flag tells mdxfind to try many common hash types. You can also restrict to specific types:

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

For salted hash types, provide salt files with `-s`.  This is *great* when you have salted hashes without the salts present. For several salted algorithms, mdxfind can actually generate the missing salts.  But if you have them, you can use them:

```bash
cut -c 34- salted/*.txt >salts.txt
cat *.txt salted/*.txt | mdxfind -s salts.txt -h "^MD5SALT$" wordlist.txt >> session.res
```

If you know the type of algorithms, and the salts are present in the file, mdxfind can use them directly, too, using the -M and -F options

```bash
cat *.txt salted/*.txt | mdxfind -M e31 -F stdin wordlist.txt >> session.res
```

That last form is also the one to prefer when a salt is shared by more than one
hash. A salt supplied by `-S` is deduplicated and retires after a single hit, so
three hashes sharing one salt recover only one of the three; carried next to its
own hash it recovers all three. The same applies to usernames via `-U`. See
[SALTS_AND_USERS.md](SALTS_AND_USERS.md).

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
50m.txt           # Input hashes (fewer entries as you remove solved ones)
50m.MD5x01        # Solved MD5 hashes
50m.SHA1x02       # Solved SHA1 hashes
50m.MD5SALTx01    # Solved salted MD5 hashes
```

### Automatic solved-hash removal

mdsplit automatically removes solved hashes from the `.txt` files as it processes results. After running mdsplit, each `.txt` file contains only the remaining unsolved hashes — no manual filtering step is needed. This is why keeping the `.orig` file is important: the `.txt` file shrinks over time as hashes are solved and moved to their `.HASHTYPE` output files.

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
| **mdxpause** | Pause and resume running mdxfind instances |

*Detailed guides for getpass, procrule, and other tools will be added in future updates.*

### Pausing and Resuming with mdxpause

Long-running mdxfind jobs can be paused and resumed without losing progress. This is useful when you need the CPU for something else temporarily.

**Start a job:**

```bash
mdxfind -M e31 -F sm-saltfull.txt rockyou.txt > results.res 2>progress.err &
```

**Pause it** — with the PID if known, or interactively if not:

```bash
# If you know the PID:
mdxpause pause 117550

# If you don't, mdxpause finds running instances automatically:
mdxpause pause
  pause PID 117550 (./mdxfind -M e31 -F sm-saltfull.txt rockyou.txt)... OK
```

mdxfind prints a pause notice to stderr with a restart hint:

```
MDXfind process (117550) paused at 2026-03-30 10:04:29, restart with -w 3073 on rockyou.txt
```

While paused, progress stops — the line count and hash rate freeze. The process remains in memory and can be resumed at any time.

**Resume it:**

```bash
mdxpause resume 117550
```

mdxfind logs the resume and how long it was paused:

```
MDXfind process (117550) resumed (paused 0:00:00:15)
Working on rockyou.txt, w=9, line 5449530, Found=25728, 704.68Mh/s, 580.67c/s
```

Processing continues exactly where it left off. On Linux/macOS/FreeBSD, mdxpause uses SIGUSR1 (pause) and SIGUSR2 (resume). On Windows, it uses named events.

## Tips

- **Start broad, narrow later.** Use `-h ALL` first to discover what hash types are present, then target specific types in follow-up runs.
- **Append results.** Always use `>>` to append to result files. You'll run many sessions over days or weeks.
- **Run mdsplit periodically.** It's cheap to run and keeps your solved hashes organized.
- **Keep `.orig` files forever.** You may need to re-extract hashes if you discover the format was parsed incorrectly.
- **Use getpass to recycle passwords.** Passwords found in one list are often reused in others. Extract them and feed them back through mdxfind.
