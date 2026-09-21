#!/usr/bin/env python3
"""Assemble Collatz in MIPS, recompile it to C, and check three independent
implementations agree.

Three records of one fact, kept separately: plain Python (what the answer IS),
a fetch/decode/dispatch interpreter (what the MACHINE does), and gcc-compiled
output of my recompiler (what the RECOMPILER thinks the machine does). Any two
agreeing proves less than people think; all three agreeing across 500 inputs is
a real check.
"""
import subprocess
import sys

from mips import addiu, addu, andi, beq, jr, nop, pretty, srl
from recomp import PRELUDE, emit, interp

# Collatz step count. $a0 = n on entry, $v0 = steps on return.
#
#        0x00  addiu $v0, $zero, 0     steps = 0
#        0x04  addiu $t0, $zero, 1     t0 = 1
# loop:  0x08  beq   $a0, $t0, done    while n != 1
#        0x0c  nop                       (delay slot)
#        0x10  andi  $t1, $a0, 1       t1 = n & 1
#        0x14  beq   $t1, $zero, even
#        0x18  nop
#        0x1c  addu  $t2, $a0, $a0     odd: t2 = 2n
#        0x20  addu  $a0, $t2, $a0           n  = 3n
#        0x24  addiu $a0, $a0, 1             n  = 3n+1
#        0x28  beq   $zero, $zero, next
#        0x2c  nop
# even:  0x30  srl   $a0, $a0, 1       even: n = n/2
# next:  0x34  addiu $v0, $v0, 1       steps++
#        0x38  beq   $zero, $zero, loop
#        0x3c  nop
# done:  0x40  jr    $ra
#        0x44  nop
#
# Branch offsets are in INSTRUCTIONS from the instruction after the branch:
#   off = (target - pc - 4) / 4
COLLATZ = [
    addiu("v0", "zero", 0),
    addiu("t0", "zero", 1),
    beq("a0", "t0", 13),        # -> 0x40 done
    nop(),
    andi("t1", "a0", 1),
    beq("t1", "zero", 6),       # -> 0x30 even
    nop(),
    addu("t2", "a0", "a0"),
    addu("a0", "t2", "a0"),
    addiu("a0", "a0", 1),
    beq("zero", "zero", 2),     # -> 0x34 next
    nop(),
    srl("a0", "a0", 1),
    addiu("v0", "v0", 1),
    beq("zero", "zero", -13),   # -> 0x08 loop
    nop(),
    jr("ra"),
    nop(),
]

# The delay-slot probe, second attempt.
#
# The FIRST probe clobbered $a0 in the delay slot — and failed to discriminate,
# because the function returns $v0, which is assigned a constant further down.
# The delay slot ran (or didn't) and the difference never reached the output.
# A probe whose effect cannot reach the observable is not a probe; it is a test
# that passes for the wrong reason. My own script caught it and refused the
# conclusion, which is the only reason I noticed.
#
# This version puts the delay slot's effect DIRECTLY on the return value:
#
#   0x00  addiu $t0, $zero, 5
#   0x04  addiu $v0, $zero, 0
#   0x08  beq   $a0, $t0, hit      -> 0x18
#   0x0c  addiu $v0, $v0, 7        delay slot: runs EITHER WAY, hits the result
#   0x10  jr    $ra                miss path returns 7
#   0x14  nop
#   hit: 0x18  addiu $v0, $v0, 100 hit path returns 7 + 100 = 107
#   0x1c  jr    $ra
#   0x20  nop
#
# Correct : n=5 -> 107 (slot ran, then branched), n=7 -> 7
# Naive   : n=5 -> 100 (branch taken, slot skipped), n=7 -> 7
DELAY_PROBE = [
    addiu("t0", "zero", 5),
    addiu("v0", "zero", 0),
    beq("a0", "t0", 3),         # -> 0x18
    addiu("v0", "v0", 7),       # delay slot, reaches the return value
    jr("ra"),
    nop(),
    addiu("v0", "v0", 100),
    jr("ra"),
    nop(),
]


def py_collatz(n):
    s = 0
    while n != 1:
        n = 3 * n + 1 if n & 1 else n // 2
        s += 1
    return s


def build_and_run(words, name, inputs, delay_slots=True):
    c = PRELUDE + "\n" + emit(words, name=name, delay_slots=delay_slots) + f"""
int main(int argc, char **argv) {{
    for (int i = 1; i < argc; i++) {{
        unsigned long v = strtoul(argv[i], 0, 10);
        printf("%u\\n", {name}((uint32_t)v));
    }}
    return 0;
}}
"""
    c = c.replace("#include <stdio.h>", "#include <stdio.h>\n#include <stdlib.h>")
    src, exe = f"{name}.c", f"./{name}"
    open(src, "w").write(c)
    p = subprocess.run(["gcc", "-O2", "-Wall", "-o", exe, src],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stderr)
        raise SystemExit(f"gcc failed for {name}")
    out = subprocess.run([exe] + [str(i) for i in inputs],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


if __name__ == "__main__":
    print("  --- the program, as the disassembler sees it back ---")
    for i, w in enumerate(COLLATZ):
        print(f"    0x{i*4:04x}  {w:08x}   {pretty(w, i*4)}")

    ns = list(range(1, 501))
    print("\n  --- three independent implementations, 500 inputs ---")
    a = [py_collatz(n) for n in ns]
    b = [interp(COLLATZ, n) for n in ns]
    c = build_and_run(COLLATZ, "collatz", ns)
    print(f"    python reference : {a[:8]} …")
    print(f"    interpreter      : {b[:8]} …")
    print(f"    recompiled C     : {c[:8]} …")
    ok = (a == b == c)
    print(f"    all three agree on all {len(ns)}: {ok}")
    if not ok:
        bad = [(n, x, y, z) for n, x, y, z in zip(ns, a, b, c) if not x == y == z][:5]
        print(f"    MISMATCHES: {bad}")
        raise SystemExit(1)
    print(f"    longest chain under 500: n={ns[a.index(max(a))]} takes {max(a)} steps")

    print("\n  --- the delay-slot probe ---")
    probe_in = [5, 7]
    corr_i = [interp(DELAY_PROBE, n) for n in probe_in]
    corr_c = build_and_run(DELAY_PROBE, "probe_ok", probe_in, delay_slots=True)
    naive = build_and_run(DELAY_PROBE, "probe_naive", probe_in, delay_slots=False)
    print(f"    input                 : {probe_in}")
    print(f"    interpreter (correct) : {corr_i}")
    print(f"    recompiled, slots ON  : {corr_c}")
    print(f"    recompiled, slots OFF : {naive}")
    print()
    if corr_i == corr_c and naive != corr_c:
        print("    The two recompilations DISAGREE, and the delay-slot-aware one")
        print("    matches the interpreter. The naive version is wrong in a way")
        print("    that no nop-only program could ever have revealed.")
    elif naive == corr_c:
        print("    Both agree — the probe failed to discriminate. Fix the probe,")
        print("    not the conclusion.")
