# Progress Output

While mdxfind runs, the `ReportStats()` thread writes a one-line status update to **stderr every 15 seconds**. Each line is a comfort message: it confirms the run is alive and reports where it is. Hash matches are still emitted to **stdout** as they occur; the progress line never carries match data.

## Line Formats

The format selected depends on (a) whether a wordlist total is available — set explicitly with `-W <count>` or computed in the background by `-W auto` — and (b) whether at least one GPU job slot is active (`gpujob_available()`).

### 1. GPU + wordlist total available

```
Working on rockyou.txt, w=39, gq=129/0, 4.5Gh/71.7Gh (6.3%), Found=7284, 298.54Mh/s, 401.71Mc/s, ETA 3m45s
```

### 2. CPU only + wordlist total available

```
Working on rockyou.txt, w=39, 4.5Gh/71.7Gh (6.3%), Found=7284, 298.54Mh/s, 401.71Mc/s, ETA 3m45s
```

### 3. GPU + no wordlist total

```
Working on rockyou.txt, w=39, gq=129/0, line 1234567, Found=7284, 298.54Mh/s, 401.71Mc/s
```

### 4. CPU only + no wordlist total

```
Working on rockyou.txt, w=39, line 1234567, Found=7284, 298.54Mh/s, 401.71Mc/s
```

### 5. Brute-force mode

```
Brute ?l?l?l?l?l?l?l?l?l, 3.9T/5.4T (72.7%), Found=224732, 5.37Gh/s, 'sxegdtkfb', ETA 4m36s
```

Brute-force mode is selected positionally — when exactly one non-option argument is supplied and `stat(2)` on it fails, mdxfind parses it as a mask and runs against the full keyspace. The progress line carries:

- `Brute` literal prefix (replaces `Working on`).
- The mask string itself in the `Curfile` slot (`Curfile = argv[0]` in this mode — there is no real input file).
- Keyspace progress `dprog/dtotal (pct%)`, where the denominator is `BruteForceTotal = MaskTotal` and the numerator is `Tothash` divided by the active multiplier (algorithm count × salt count).
- A sample candidate at the current progress index — see the field reference below.

The brute-force line has **no `gq=` field and no `w=` field**. Multi-GPU coordination in brute-force mode uses an atomic mask cursor rather than the work-queue model used in wordlist mode, so the queue-depth diagnostics that gate `gq=` and `w=` do not apply. The CPU/GPU split that produces the four wordlist variants above does not apply either: there is one brute-force format. See [BRUTE_FORCE.md](BRUTE_FORCE.md) for activation rules and dispatch details.

## Field Reference

| Field | Meaning |
|-------|---------|
| `Working on FILE` | Current input wordlist (`Curfile`). |
| `w=N` | Depth of the `WorkWaiting` queue — line-buffers produced by the reader and not yet claimed by a worker thread. High values mean workers are saturated; values near zero mean the reader is the bottleneck. Same definition in all four formats. |
| `gq=PENDING/FREE` | Present only when at least one GPU job slot is available. `PENDING` is `gpujob_queue_depth()` (jobs queued for GPU dispatch); `FREE` is `gpujob_free_count()` (unused jobg buffers). High `PENDING` + low `FREE` = GPU saturated; low `PENDING` + high `FREE` = CPU producer is starving the GPU. |
| `Xh/Yh (P%)` | Progress as `Tothash / expected_total`, where `expected_total = total_lines × salt_mult × rules × iterations`. The `h` suffix is hashes computed; the SI prefix (`K`/`M`/`G`/`T`) is autoscaled by `format_rate()`. |
| `line N` | Lowest line position currently being processed (`Lowline + LowSkip`), reported when no wordlist total is known. |
| `Found=N` | Total successful hash matches across all algorithms in the run (`Totfound`), cumulative since start. |
| `X.YY[K\|M\|G\|T]h/s` | Hash rate. `Tothash / wtime` — total hashes computed divided by elapsed wallclock seconds since the run started. Long-run average, not an instantaneous reading. |
| `X.YY[K\|M\|G\|T]c/s` | Candidate rate over the most recent 15-second window. When rules are loaded this is `(Totrules - lastline)/15`; otherwise it is `(TotLines - lastline)/15`. **A 15s sliding window, not a long-run average** — short-term changes show up here first. |
| `ETA TIME` or `ETA~TIME` | Estimated time remaining, formatted by `format_eta()`. See ETA prefix below. |
| `Brute MASK` | Brute-force-mode prefix. Replaces `Working on FILE`. The mask string itself is shown in place of `Curfile` because there is no input file in this mode. |
| `'sample'` | Brute-force-mode only. A concrete candidate string drawn from the current keyspace progress index via `mask_expand_into()`. **It is the candidate at the completed-progress position, not the most recently tried candidate** — a fast GPU may already be hashing values past the position counted into `Tothash`. The two are very close in practice. |

### ETA prefix: space vs tilde

```
ETA 3m45s     <- space prefix: total line count is final
ETA~3m45s     <- tilde prefix: total line count is still being computed
```

The tilde appears when `-W auto` was requested but the background line-count thread has not yet finished. The estimate uses whatever lines have been counted so far and is refined each tick. Once counting completes, the prefix becomes a space and the figure stabilizes.

### ETA `finishing`

```
... ETA finishing
```

Emitted when `remaining <= 0` (progress fraction has hit 1.0 or saturated) but the work queue still has entries (`wq > 0`). Indicates the producer has read all input and the workers — or queued GPU jobs — are draining.

### ETA scale

`format_eta()` autoscales:

| Range | Format |
|-------|--------|
| < 60 s | `45s` |
| < 1 h | `3m45s` |
| < 1 d | `1h22m` |
| < 30 d | `4d3h` |
| < 1 y | `~2mo15d` |
| ≥ 1 y | `~1y3mo` |

The leading `~` on month/year forms is part of the unit display (months and years are approximate calendar units), independent of the line-count tilde described above.

## Cadence and Silence

The reporting loop is `for (x = 0; x < 15; x++) { sleep(1); ... }` — exactly 15 one-second naps per tick. There is **no per-progress filter**: a `Working on` line is emitted on every tick during the running state, even if `Tothash` did not change. The only cases where a tick does **not** produce a `Working on` line:

- The run has finished (`HashWaiting` lock signals exit). `ReportStats()` returns.
- `MDXpause` is set. Pause-state messages (`pausing...`, `paused at ...`, `paused for N minutes`) are emitted in place of the progress line. On resume, a `resumed (paused d:hh:mm:ss)` line is emitted before normal ticks resume.
- Brute-force mode emits a `Brute ...` line instead. See [BRUTE_FORCE.md](BRUTE_FORCE.md) and the format-5 entry above.

A single hit (match) printed to stdout never suppresses the next stderr tick — they are independent streams.

## Interpretation Tips

- **`h/s` falling steadily over a long run with `c/s` flat or rising** — the CPU producer is no longer keeping the workers fed (e.g., a slow rule expansion or a regression in the hot path). Rev 1.339 surfaced this pattern as a regression signal.
- **`h/s ≈ c/s`** — no mask expansion in flight; one candidate produces one hash.
- **`h/s` much greater than `c/s`** — masks are multiplying each candidate. With `-n '?d?d'`, expect roughly `h/s ≈ 100 × c/s` per active algorithm.
- **`gq=0/N` persistently with GPU enabled** — the GPU is starved. The producer (CPU rule/mask expansion or wordlist read) is the bottleneck, not the GPU.
- **`gq=N/0` persistently** — the GPU job queue is saturated and the free-buffer pool is empty. Workers will block waiting for a GPU job slot until the GPU drains.
- **`w=0` for long stretches** — the reader has run out of input or is blocked on I/O. With multiple wordlists, this is normal at file boundaries.
- **`ETA~` not converging on a stable figure** — `-W auto` line counting is still in progress. Wait for the prefix to flip to a space; the estimate stabilizes once the count is final.
