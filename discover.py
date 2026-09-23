#!/usr/bin/env python3
"""Where do the indirect-jump targets come from?

Yesterday I gave my dispatch table three addresses by hand and wrote in the
queue that this was the dishonest part: a real recompiler has to FIND them.
This is that, and the answer turns out to be uncomfortable in a way worth
writing down.

There are three ways to discover the targets of `jr $t3`:

  1. PATTERN-MATCH the jump table. Works when the compiler emitted a shape you
     recognise — `sll; addiu base; addu; jr` is a dense switch. Fails on any
     idiom you haven't enumerated, and fails silently, because an unrecognised
     table looks exactly like an unrecognised table.

  2. PROVE it, by tracing every value the register can hold. This is the
     general case and it is undecidable. In practice it means abstract
     interpretation with a lattice, and it gives you a SUPERSET on good days
     and gives up on bad ones.

  3. RUN IT and write down where it actually went.

Option 3 is what this file does, and it is what real projects fall back on. It
always works, it needs no cleverness, and it has one property that ought to be
stated louder than it is:

    **you recompile what you observed.**

A target reachable only on an input you never traced is not in your dispatch
table, and the recompiled program will fail at RUNTIME on that input, with
nothing at compile time to warn you. Which means the coverage of your tracing
inputs stops being a testing statistic and becomes a CORRECTNESS property of
the artifact you ship.

I only recognised the shape because of a project I read up on this week: the
Ogre Battle 64 recompilation reports `99.05%` of the ROM's code span
recompiled AND, separately, that a hand-played mission enters `44.4%` of
registered functions, with two modules never entered at all. Two numbers,
and the second is the one that bounds what you can trust.
"""
import subprocess

from mips import addiu, addu, jr, nop, sll
from recomp import PRELUDE, disasm, interp, run_one
from indirect import build_and_run, emit_indirect

# A four-way jump table. Same shape as yesterday's, one more handler.
#   0x00  sll   $t1, $a0, 4        index * 16
#   0x04  addiu $t2, $zero, 0x20   handler base
#   0x08  addu  $t3, $t2, $t1
#   0x0c  jr    $t3                <- indirect
#   0x10..0x1c  nop padding
#   0x20  H0 -> 100      0x30  H1 -> 200
#   0x40  H2 -> 300      0x50  H3 -> 400   <- only reachable with index 3
TABLE4 = [
    sll("t1", "a0", 4),
    addiu("t2", "zero", 0x20),
    addu("t3", "t2", "t1"),
    jr("t3"),
    nop(), nop(), nop(), nop(),
    addiu("v0", "zero", 100), jr("ra"), nop(), nop(),
    addiu("v0", "zero", 200), jr("ra"), nop(), nop(),
    addiu("v0", "zero", 300), jr("ra"), nop(), nop(),
    addiu("v0", "zero", 400), jr("ra"), nop(), nop(),
]
ALL_HANDLERS = [0x20, 0x30, 0x40, 0x50]


def trace(words, inputs, base=0):
    """Run the interpreter over `inputs` and record every indirect-jump target
    actually taken. This is discovery-by-observation: no analysis, no pattern
    library, just a note of where control went."""
    seen = set()
    for a0 in inputs:
        reg = [0] * 32
        reg[4] = a0
        pc, steps = base, 0
        while steps < 10000:
            steps += 1
            w = words[(pc - base) // 4]
            m, o, tgt = disasm(w, pc)
            if m == "jr":
                run_one(words, reg, base, pc + 4)      # delay slot
                if o[0] == 31:
                    break
                seen.add(reg[o[0]])                    # <- the observation
                pc = reg[o[0]]
                continue
            if m in ("beq", "bne"):
                take = (reg[o[0]] == reg[o[1]]) if m == "beq" else (reg[o[0]] != reg[o[1]])
                run_one(words, reg, base, pc + 4)
                pc = tgt if take else pc + 8
                continue
            run_one(words, reg, base, pc)
            pc += 4
    return sorted(seen)


def recognise(words, base=0):
    """Option 1: find the table by its SHAPE, without running anything.

    A dense switch compiles to a recognisable idiom:
        sll   idx, src, k          index * 2^k
        addiu b,  $zero, BASE      table base
        addu  tgt, b, idx
        jr    tgt
    Walk back from the `jr` and recover BASE and the stride.
    """
    for i, w in enumerate(words):
        m, o, _ = disasm(w, base + i * 4)
        if m != "jr" or o[0] == 31 or i < 3:
            continue
        m1, o1, _ = disasm(words[i - 1], 0)     # addu tgt, b, idx
        m2, o2, _ = disasm(words[i - 2], 0)     # addiu b, $zero, BASE
        m3, o3, _ = disasm(words[i - 3], 0)     # sll idx, src, k
        if (m1, m2, m3) != ("addu", "addiu", "sll"):
            continue
        if o1[0] != o[0]:
            continue
        return dict(base_addr=o2[2], stride=1 << o3[2], at=base + i * 4)
    return None

if __name__ == "__main__":
    print("  A 4-way jump table. Handler 3 is reachable only with $a0 == 3.\n")

    traced_inputs = [0, 1, 2]          # deliberately incomplete
    found = trace(TABLE4, traced_inputs)
    print(f"  1. Trace the interpreter over inputs {traced_inputs}:")
    print(f"       discovered targets: {[hex(t) for t in found]}")
    print(f"       actually present:   {[hex(t) for t in ALL_HANDLERS]}")
    print(f"       missed: {[hex(t) for t in ALL_HANDLERS if t not in found]}")
    print("     Nothing announced the miss. The trace is complete for what it ran.\n")

    print("  2. Recompile using ONLY the discovered targets, then test all four:")
    exe = build_and_run(PRELUDE + "\n" + emit_indirect(TABLE4, name="jt_traced",
                                                       known=found),
                        "jt_traced", [0, 1, 2, 3])
    want = [100, 200, 300, 400]
    print(f"       inputs  {[0,1,2,3]}")
    print(f"       want    {want}")
    print(f"       got     {exe}")
    bad = [i for i, (g, w) in enumerate(zip(exe, want)) if g != w]
    for i in bad:
        print(f"     ** input {i}: expected {want[i]}, got {exe[i]} (0x{exe[i]:X}) — "
              f"target not in the table, failing at RUNTIME")
    print()

    print("  3. Now trace the input we left out, and rebuild:")
    found2 = trace(TABLE4, [0, 1, 2, 3])
    print(f"       discovered targets: {[hex(t) for t in found2]}")
    exe2 = build_and_run(PRELUDE + "\n" + emit_indirect(TABLE4, name="jt_traced2",
                                                        known=found2),
                         "jt_traced2", [0, 1, 2, 3])
    print(f"       got     {exe2}   correct: {exe2 == want}\n")

    print("  The recompiler did not get better between step 2 and step 3.")
    print("  The TEST INPUTS did. Coverage of the tracing run is not a quality")
    print("  metric here — it is the completeness bound on the artifact.")

    print()
    print("  4. The other option: recognise the table by SHAPE, running nothing.")
    r = recognise(TABLE4)
    print(f"       recovered: base 0x{r['base_addr']:02x}, stride {r['stride']} bytes,"
          f" at the jr @ 0x{r['at']:04x}")
    print(f"       implied targets: 0x{r['base_addr']:02x}, "
          f"0x{r['base_addr']+r['stride']:02x}, 0x{r['base_addr']+2*r['stride']:02x}, …")
    print()
    print("     It found the STRUCTURE without executing a single instruction —")
    print("     and it cannot tell you the EXTENT. There is no bounds check in")
    print("     this program, so the recovered pattern describes an infinite")
    print("     family. Three handlers, four, four hundred: identical shape.")
    print()
    print("  So the two methods fail in opposite directions:")
    print("     tracing   — sound, never invents a target, silently INCOMPLETE")
    print("     pattern   — gives base and stride without running, UNBOUNDED extent")
    print("  and neither failure is visible in its output. Both hand you a table.")
