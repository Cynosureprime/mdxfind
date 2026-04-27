# Running mdxfind on Windows

mdxfind has a Windows build (x86-64, 32-bit, ARM64), but Windows treats password-recovery tools defensively. This page is about getting it running without doing anything you'll regret.

## TL;DR — pick a path

Three layers will fight you. Pick one of these and stop reading the others:

| Path | Best for | Effort |
|------|----------|--------|
| **A. Dedicated cracking rig** | Anyone with hardware whose main job is cracking | Set Windows up once with SAC off; one Defender exclusion |
| **B. WSL2 with the Linux build** | Mixed-use desktop, casual / occasional cracking | 5 min, no Windows defenses to fight |
| **C. Native .exe on consumer Windows** | Need native GPU on a daily-driver Windows machine | 15 min, ongoing maintenance, two layers to manage |

## What's actually blocking you

Three independent defense layers, in the order you encounter them:

### 1. SmartScreen — at download

Edge and the Defender SmartScreen filter check downloads against Microsoft's cloud reputation. A fresh mdxfind release nobody has downloaded yet has no reputation and gets warned about ("Windows protected your PC"). Click *More info* → *Run anyway*, or `Unblock-File` it from PowerShell. This one is harmless and goes away once enough people have downloaded the same file.

### 2. Smart App Control (SAC) — at launch

SAC is what actually stops the .exe from running on Windows 11. It allows a binary if **either**:

- it has an Authenticode signature from a publisher Windows trusts, **or**
- it has positive cloud reputation (Microsoft's ISG service has seen this exact file widely and has classified it benign).

An unsigned, freshly-built mdxfind has neither. You'll see Event IDs 3077 / 3118 ("Smart App Control Block") in the Event Viewer under *Microsoft-Windows-CodeIntegrity/Operational*. The launch silently fails — no error dialog, just nothing happens, and a CodeIntegrity event in the log.

SAC is enabled by default on new Windows 11 installs in `VerifiedAndReputable` enforcement mode, and **once it's enforced you can't turn it off** without reinstalling Windows. So the time to opt out is during initial setup.

### 3. Windows Defender behavioral scan — while running

This is the layer most people don't expect. Even after a binary is signed and SAC lets it through, Defender's behavioral engine watches what it *does* — and what mdxfind does (mass cryptographic hashing, mask-based keyspace enumeration, hex-fingerprint lookups against a hash table) is exactly the on-disk and runtime fingerprint that Microsoft trains as `Hacktool:Win32/*`. There is no signature you can apply that will change this — code-signing solves SAC, not Defender. Defender will quarantine the binary mid-run, sometimes hours into a job, sometimes during the next Windows Update reboot scan.

The good news: Defender's exclusion list overrides this for any path or process you specify. The exclusion is the actual fix.

## Path A — dedicated cracking rig (recommended)

If you have a rig whose main job is cracking, this is the realistic path.

**At Windows install time**, during OOBE: **decline Smart App Control**. SAC's `VerifiedAndReputable` mode is one-way; you can't reverse it later without reinstalling. Pick wisely on the first boot.

**Install mdxfind** in a path you control, e.g., `C:\Tools\mdxfind\`.

**Add a Defender exclusion**, in an elevated PowerShell:

```powershell
Add-MpPreference -ExclusionPath 'C:\Tools\mdxfind'
Add-MpPreference -ExclusionProcess 'mdxfind.exe'
```

Done. The path exclusion stops Defender from scanning hash files (which can themselves trigger flags depending on contents). The process exclusion stops behavioral monitoring of the running binary. No reboot needed.

This is the same approach hashcat and John the Ripper users take. mdxfind isn't doing anything different from those tools as far as Windows is concerned.

## Path B — WSL2 with the Linux build

If your Windows machine is also your daily driver, the cleanest answer is to run the Linux build under WSL2. WSL2 runs Linux in a managed VM; Defender doesn't behavioral-scan inside WSL2's filesystem, and SAC doesn't apply to ELF binaries.

```powershell
wsl --install -d Ubuntu-24.04
```

Inside the resulting Ubuntu shell:

```bash
sudo apt install libjudy-dev libssl-dev libpcre3-dev libmhash-dev librhash-dev zlib1g-dev
# Then build mdxfind from source, or fetch a Linux release tarball.
```

Your Windows-side hash files are visible from WSL at `/mnt/c/...`.

**GPU caveats:**
- NVIDIA: works fine. Install the NVIDIA WSL CUDA driver on the Windows host (one .exe from NVIDIA's site); WSL gets full CUDA access.
- AMD: limited. ROCm-on-WSL is partial and not all GPUs are supported.
- Intel/integrated: not supported in WSL.

If you have AMD or Intel GPUs and you need GPU dispatch, Path A or C is the better answer.

## Path C — native .exe on consumer Windows

You need GPU dispatch, and you're not willing or able to dedicate the machine. Here's the path.

### Get a signed release

Releases on the GitHub project (when shipped signed) carry an Authenticode signature. Verify before trusting:

```powershell
Get-AuthenticodeSignature .\mdxfind.exe | Format-List
```

`Status` should be `Valid`. `NotSigned` means it's an unsigned dev build (use Path A or B). `HashMismatch` means the file has been altered after signing — don't run it.

### Get past SmartScreen

First download will trigger a SmartScreen warning. Either *More info → Run anyway*, or:

```powershell
Unblock-File .\mdxfind.exe
```

This clears the Mark-of-the-Web zone identifier so SmartScreen treats the file as locally-sourced.

### Get past SAC

A signed release with reasonable distribution should pass SAC. If it doesn't, the publisher's cert hasn't built up Microsoft ISG reputation yet — you have a few options:

- Wait a few weeks for community reputation to build (passive).
- Switch to Path B (WSL2) until then.
- Use a self-signed cert imported as a trusted publisher (advanced — see *Self-signed builds* below).

### Add the Defender exclusion (still required)

Signing doesn't help with Defender's behavioral classifier. You still need:

```powershell
Add-MpPreference -ExclusionPath 'C:\path\to\mdxfind\directory'
Add-MpPreference -ExclusionProcess 'mdxfind.exe'
```

Without this, Defender will quarantine the binary the first time its behavioral scan fires on a running mdxfind process — usually within minutes-to-hours of first launch, sometimes during a routine Windows Update.

### If Defender already quarantined the binary

```powershell
Get-MpThreat                                    # see what was quarantined
Get-MpThreatDetection | Format-List              # detail per detection
```

To restore from quarantine:

```powershell
$mp = "$env:ProgramData\Microsoft\Windows Defender\Platform"
$exe = Get-ChildItem $mp -Recurse -Filter MpCmdRun.exe | Sort-Object LastWriteTime -Descending | Select-Object -First 1
& $exe.FullName -Restore -Name "Hacktool:Win32/<exact-name-from-Get-MpThreat>"
```

Then immediately add the exclusions above so it doesn't happen again.

## What not to do

We've seen people try to "fix" the Windows defense layers in ways that brick the machine. Each of these has bitten somebody we know:

- **Disable SAC via registry** (`HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy:VerifiedAndReputablePolicyState`). It's one-way. You will be reinstalling Windows.
- **Deploy a custom WDAC enforcement policy** to allowlist mdxfind by hash. WDAC base policies stack with AND logic — a policy that allows only `mdxfind.exe` implicitly blocks everything else, including the per-session sshd handler. If you've reached the box over SSH, you've just locked yourself out and recovery requires console / KVM / Windows Recovery Environment access.
- **Stop the Defender service** (e.g., `Set-MpPreference -DisableRealtimeMonitoring $true`). Modern Windows resets this on update or reboot, and Tamper Protection on Windows 11 will simply refuse the change. Use exclusions instead — they're the supported mechanism.
- **Disable Tamper Protection** to do any of the above. Tamper Protection is the layer that stops malware from doing exactly what you're about to do. Turning it off makes the machine substantially less safe; the exclusion-based approach in Path A doesn't require turning it off.

The exclusion-based approach is reversible, supported, and what every legitimate recovery-tool operator on Windows uses.

## Cleaning up afterwards

Remove exclusions:

```powershell
Remove-MpPreference -ExclusionPath 'C:\Tools\mdxfind'
Remove-MpPreference -ExclusionProcess 'mdxfind.exe'
```

If you imported a self-signed publisher cert (see below), remove it from both stores:

```powershell
Get-ChildItem Cert:\LocalMachine\TrustedPublisher\ |
    Where-Object Subject -match "mdxfind" | Remove-Item
Get-ChildItem Cert:\LocalMachine\Root\ |
    Where-Object Subject -match "mdxfind" | Remove-Item
```

Verify the exclusions are gone:

```powershell
Get-MpPreference | Select-Object ExclusionPath, ExclusionProcess
```

## Self-signed builds (for people building from source)

If you're cross-compiling mdxfind yourself (mingw on Linux, MSYS2 on Windows, etc.), the resulting binary is unsigned. The realistic answer is Path A or Path B for running it.

If you specifically need SAC to accept a self-built binary on a SAC-enforced machine that isn't your dedicated rig, you can sign with your own cert and import that cert as a trusted publisher on the target machine. The signature won't be trusted on anyone else's machine — it's only good on the machines where you've explicitly imported your cert.

The basics:

1. Generate a code-signing certificate (RSA 4096, SHA-256, EKU = `codeSigning`, valid 5 years). OpenSSL works fine for this.
2. Sign the .exe with `osslsigncode` (Linux/macOS) or `signtool.exe` (Windows). Include a free Authenticode timestamp (e.g. `http://timestamp.sectigo.com`) so the signature outlives the cert's expiry.
3. Export the public part of your cert as `.cer` (DER format).
4. On each target machine, import that `.cer` into both `Cert:\LocalMachine\Root` (so the chain validates) and `Cert:\LocalMachine\TrustedPublisher` (so SAC trusts the publisher).

The full cross-build + signing recipe lives in the project's build documentation; this page is about *running* mdxfind, not building it.

## Antivirus false positives on VirusTotal

mdxfind shows up on VirusTotal with a few hits, typically classified as `Hacktool`, `Riskware`, or occasionally `Coinminer`. None of these are malicious behavior — mdxfind doesn't mine, doesn't phone home, doesn't escalate privileges, and doesn't touch anything on the system besides what you explicitly point it at. The classifications fire on the same on-disk pattern that triggers Defender's behavioral scan: extensive cryptographic code, hash-table lookup machinery, and mask-based candidate enumeration. That pattern overlaps with how a credential-recovery tool legitimately works.

If your environment is regulated or your security team triages VT hits, the practical answer is the same as on the running-it side: an exclusion / allowlist for the path or hash, applied through your normal endpoint management process.
