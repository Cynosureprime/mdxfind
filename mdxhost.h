/* $Revision: $
 *
 * $Log: $
 */

/* mdxhost.h -- portable "what host am I on?" for diagnostic messages.
 *
 * Every FATAL/warning line in mdxfind names the host it came from, so a
 * log pasted from a remote rig is attributable without asking. The old
 * idiom at each site was:
 *
 *     char hostname[256] = "unknown";
 *     gethostname(hostname, sizeof(hostname) - 1);
 *
 * which is silently broken on Windows: there, gethostname() is a Winsock
 * call that fails with WSANOTINITIALISED unless WSAStartup() has run
 * first, and mdxfind never calls WSAStartup(). The buffer is left
 * untouched, so every diagnostic on a Windows rig reported the host as
 * "unknown" -- observed 2026-08-05 on a 12-GPU OpenCL run where all 35
 * log lines were unattributable.
 *
 * GetComputerNameA() is the native equivalent and needs no init. It is
 * already the idiom used by rule-bench.c and pcie-microbench.c; this
 * header exists so the other call sites stop hand-rolling it.
 *
 * Header-only on purpose: a new .c would mean a new object on the link
 * line of every per-machine Makefile across all ten build hosts, and
 * those Makefiles are deliberately never copied between machines.
 */

#ifndef MDXHOST_H
#define MDXHOST_H

#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/* Fill `buf` with the local host name, NUL-terminated, never empty.
 * Falls back to the literal "unknown" if the platform call fails, so
 * callers can print it unconditionally with %s. */
static inline void mdx_gethostname(char *buf, size_t buflen)
{
    if (!buf || buflen == 0) return;
    buf[0] = '\0';

#ifdef _WIN32
    {
        /* GetComputerNameA wants the buffer size in/out. It fails with
         * ERROR_BUFFER_OVERFLOW rather than truncating, so a short
         * buffer lands in the "unknown" path below -- correct, if
         * uninformative. MAX_COMPUTERNAME_LENGTH is 15, so any sane
         * caller buffer is large enough. */
        DWORD n = (DWORD)(buflen - 1);
        if (!GetComputerNameA(buf, &n)) buf[0] = '\0';
    }
#else
    if (gethostname(buf, buflen - 1) != 0) buf[0] = '\0';
    buf[buflen - 1] = '\0';
#endif

    if (buf[0] == '\0') {
        /* Open-coded copy: keeps this header free of <string.h>/<stdio.h>
         * so it can be included anywhere without dragging in more. */
        static const char unk[] = "unknown";
        size_t i = 0;
        while (unk[i] != '\0' && i + 1 < buflen) { buf[i] = unk[i]; i++; }
        buf[i] = '\0';
    }
}

#endif /* MDXHOST_H */
