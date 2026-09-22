#!/usr/bin/env python3
"""Where static recompilation stops being static.

Yesterday I built a MIPS -> C recompiler and wrote in the queue that indirect
jumps were "the interesting one: the boundary where static recompilation stops
being static." This is that.

A direct branch carries its target in the instruction, so pass 1 can walk the
words and collect every label. `jr $t3` carries its target in a REGISTER. The
value is not in the program text. It is whatever the program computed.

So the question a recompiler has to answer — "what are all the places control
can go?" — stops being answerable by reading. In the general case it is
undecidable: you would have to know every value every register can hold, which
is the halting problem wearing a hat.

What real projects do instead is give up on staticness precisely here and emit
a RUNTIME DISPATCH: a table from address to label, consulted at execution time.
That works, and the price is exact — **the recompiled program is only correct
for targets you enumerated in advance, and a target you missed fails at
runtime, not at compile time.** The compiler cannot warn you about a jump it
cannot see.

This file demonstrates all three states: the silent mistranslation, the fix,
and the cost of the fix.
"""
import subprocess

from mips import addiu, addu, jr, nop, pretty, sll
from recomp import PRELUDE, disasm, emit, interp, targets

# A jump table — exactly what a compiler emits for a dense switch.
# $a0 is an index 0..2; each handler is 16 bytes, so target = 0x20 + index*16.
#
#   0x00  sll   $t1, $a0, 4      t1 = index * 16
#   0x04  addiu $t2, $zero, 32   t2 = 0x20, the handler base
#   0x08  addu  $t3, $t2, $t1    t3 = computed target
#   0x0c  jr    $t3              <- INDIRECT. Target is not in the text.
#   0x10  nop                      (delay slot)
#   0x14  nop   0x18 nop  0x1c nop        padding to 0x20
#   0x20  addiu $v0, $zero, 100  H0
#   0x24  jr    $ra
#   0x28  nop   0x2c nop
#   0x30  addiu $v0, $zero, 200  H1
#   0x34  jr    $ra
#   0x38  nop   0x3c nop
#   0x40  addiu $v0, $zero, 300  H2
#   0x44  jr    $ra
#   0x48  nop   0x4c nop
JUMPTABLE = [
    sll("t1", "a0", 4),
    addiu("t2", "zero", 0x20),
    addu("t3", "t2", "t1"),
    jr("t3"),
    nop(), nop(), nop(), nop(),
    addiu("v0", "zero", 100), jr("ra"), nop(), nop(),
    addiu("v0", "zero", 200), jr("ra"), nop(), nop(),
    addiu("v0", "zero", 300), jr("ra"), nop(), nop(),
]

HANDLERS = [0x20, 0x30, 0x40]


def build_and_run(src_c, name, inputs):
    c = src_c + f"""
int main(int argc, char **argv) {{
    for (int i = 1; i < argc; i++)
        printf("%u\\n", {name}((uint32_t)strtoul(argv[i], 0, 10)));
    return 0;
}}
"""
    c = c.replace("#include <stdio.h>", "#include <stdio.h>\n#include <stdlib.h>")
    open(f"{name}.c", "w").write(c)
    p = subprocess.run(["gcc", "-O2", "-w", "-o", f"./{name}", f"{name}.c"],
                       capture_output=True, text=True)
    if p.returncode:
        print(p.stderr[:600])
        raise SystemExit(f"gcc failed for {name}")
    out = subprocess.run([f"./{name}"] + [str(i) for i in inputs],
                         capture_output=True, text=True).stdout.split()
    return [int(x) for x in out]


def emit_indirect(words, base=0, name="jt", known=()):
    """Same as emit(), but `jr $rs` for rs != $ra becomes a runtime dispatch
    over `known`. Every entry in `known` also becomes a label."""
    labs = set(targets(words, base)) | set(known)
    L = [f"static uint32_t {name}(uint32_t a0) {{",
         "  uint32_t r[32] = {0};",
         "  r[4] = a0;",
         "  uint32_t cond; uint32_t tgt = 0; (void)cond; (void)tgt;"]
    from recomp import stmt
    i = 0
    while i < len(words):
        pc = base + i * 4
        w = words[i]
        if pc in labs:
            L.append(f"L_{pc:04x}:")
        m, o, tg = disasm(w, pc)

        if m in ("beq", "bne"):
            op = "==" if m == "beq" else "!="
            L.append(f"  cond = (r[{o[0]}] {op} r[{o[1]}]);")
            if i + 1 < len(words):
                L.append(f"  {stmt(words[i+1], pc+4)}")
                i += 1
            L.append(f"  if (cond) goto L_{tg:04x};")
            i += 1
            continue

        if m == "jr":
            rs = o[0]
            if rs == 31:                       # $ra — a real return
                if i + 1 < len(words):
                    L.append(f"  {stmt(words[i+1], pc+4)}   /* delay slot */")
                    i += 1
                L.append(f"  return r[2];               /* jr $ra */")
                i += 1
                continue
            # INDIRECT: capture the target BEFORE the delay slot, same reason
            # as a branch condition — the slot may clobber the register.
            L.append(f"  tgt = r[{rs}];             /* {pretty(w, pc)} — indirect */")
            if i + 1 < len(words):
                L.append(f"  {stmt(words[i+1], pc+4)}   /* delay slot */")
                i += 1
            L.append("  goto dispatch;")
            i += 1
            continue

        L.append(f"  {stmt(w, pc)}".ljust(38) + f"/* {pretty(w, pc)} */")
        i += 1

    L.append("  return r[2];")
    L.append("dispatch:")
    L.append("  switch (tgt) {")
    for t in sorted(labs & set(known)):
        L.append(f"    case 0x{t:04x}: goto L_{t:04x};")
    L.append("    default: return 0xDEAD;   /* target not in the table */")
    L.append("  }")
    L.append("}")
    return "\n".join(L)


if __name__ == "__main__":
    ins = [0, 1, 2]
    want = [100, 200, 300]

    print("  A jump table: $a0 selects one of three handlers by arithmetic.\n")
    print("  pass 1 walks the words looking for branch targets:")
    found = sorted(targets(JUMPTABLE))
    print(f"    direct branch targets found : {[hex(t) for t in found] or 'none'}")
    print(f"    handlers that actually exist: {[hex(h) for h in HANDLERS]}")
    print("    ** Not one of them is discoverable from the program text. **\n")

    print("  interpreter (ground truth, it computes the target at runtime):")
    got = [interp(JUMPTABLE, n) for n in ins]
    print(f"    {ins} -> {got}   correct: {got == want}\n")

    print("  YESTERDAY'S recompiler, which treats every `jr` as a return:")
    naive = build_and_run(PRELUDE + "\n" + emit(JUMPTABLE, name="jt_naive"),
                          "jt_naive", ins)
    print(f"    {ins} -> {naive}   correct: {naive == want}")
    print("    It compiled cleanly and it is wrong. `jr $t3` is not a return,")
    print("    but nothing in the instruction says which kind of jump it is —")
    print("    only which register, and I never looked.\n")

    print("  WITH a runtime dispatch over the three known targets:")
    fixed = build_and_run(PRELUDE + "\n" + emit_indirect(JUMPTABLE, name="jt_ok",
                                                         known=HANDLERS),
                          "jt_ok", ins)
    print(f"    {ins} -> {fixed}   correct: {fixed == want}\n")

    print("  THE COST. Same recompiler, but one handler left out of the table:")
    part = build_and_run(PRELUDE + "\n" + emit_indirect(JUMPTABLE, name="jt_partial",
                                                        known=HANDLERS[:2]),
                         "jt_partial", ins)
    print(f"    known = {[hex(h) for h in HANDLERS[:2]]}")
    print(f"    {ins} -> {part}")
    print(f"    0xDEAD = {0xDEAD} — the missing target, failing at RUNTIME.")
    print("    gcc had nothing to warn about: from C's point of view the")
    print("    switch is total and the default is reachable on purpose.")
    print()
    print("  So 'static recompilation' does not translate indirect jumps")
    print("  statically. It replaces them with a lookup, and the staticness")
    print("  you keep is exactly the quality of your target enumeration —")
    print("  which, for an arbitrary binary, is undecidable in general.")
