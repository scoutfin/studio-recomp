#!/usr/bin/env python3
"""Branches into the middle of a function, and calls through a function pointer.

Studio, 2026-11-05. callgraph.py found that a jal target is where a call lands,
not a wall, and grouped functions that branch into each other into REGIONS, one C
function apiece. It claimed that also covers a branch into the MIDDLE of another
function (a shared epilogue), but nothing tested it. Test one here.

Then jalr: a call whose target is a register. Function discovery so far is
"entry 0 plus every jal target", which cannot see a function that is only ever
reached through a pointer. Same blindness as the jump tables in September: the
target is data, not an operand.
"""
import subprocess

import callgraph as cg
from mips import addiu, addu, beq, disasm, jal, jalr, jr, lw, nop, pretty, slt, sw

SENTINEL, STACK_TOP = cg.SENTINEL, cg.STACK_TOP


def assemble(prog):
    labels, pc = {}, 0
    for item in prog:
        if isinstance(item, str):
            labels[item.rstrip(":")] = pc
        else:
            pc += 4
    words, pc = [], 0
    for item in prog:
        if isinstance(item, str):
            continue
        op, *a = item
        if op == "beq":
            words.append(beq(a[0], a[1], (labels[a[2]] - (pc + 4)) // 4))
        elif op == "jal":
            words.append(jal(labels[a[0]]))
        elif op == "la":                      # load a label's address: addiu rt, $zero, addr
            words.append(addiu(a[0], "zero", labels[a[1]]))
        elif op == "jalr":
            words.append(jalr(a[0]))
        else:
            words.append({"addiu": addiu, "addu": addu, "slt": slt, "lw": lw, "sw": sw,
                          "jr": jr, "nop": nop}[op](*a))
        pc += 4
    return words, labels


B = lambda l: ("beq", "zero", "zero", l)

# main(n) = f(n) + g(n), where f(n) = n + 1 and g(n) = n + 2 -> 2n + 3.
# g has no epilogue of its own: it branches into the MIDDLE of f, to f's epilogue,
# which restores $ra and $sp and returns. Compilers do this to save bytes.
SHARED = [
    "main:",
    ("addiu", "sp", "sp", -12), ("sw", "ra", 8, "sp"), ("sw", "s0", 4, "sp"), ("sw", "a0", 0, "sp"),
    ("jal", "f"), ("nop",),
    ("addu", "s0", "v0", "zero"),
    ("lw", "a0", 0, "sp"),
    ("jal", "g"), ("nop",),
    ("addu", "v0", "v0", "s0"),
    ("lw", "ra", 8, "sp"), ("lw", "s0", 4, "sp"), ("addiu", "sp", "sp", 12),
    ("jr", "ra"), ("nop",),
    "f:",
    ("addiu", "sp", "sp", -8), ("sw", "ra", 4, "sp"),
    ("addiu", "v0", "a0", 1),
    "f_epilogue:",
    ("lw", "ra", 4, "sp"), ("addiu", "sp", "sp", 8),
    ("jr", "ra"), ("nop",),
    "g:",
    ("addiu", "sp", "sp", -8), ("sw", "ra", 4, "sp"),
    ("addiu", "v0", "a0", 2),
    B("f_epilogue"), ("nop",),               # into the middle of f
]

# main(n) = h(n) + 1, but h is only ever reached through a register.
# h(n) = 2n. Nothing jals to h, so discovery-by-jal never learns it is a function.
FNPTR = [
    "main:",
    ("addiu", "sp", "sp", -4), ("sw", "ra", 0, "sp"),
    ("la", "t9", "h"),
    ("jalr", "t9"), ("nop",),                 # call through a function pointer
    ("addiu", "v0", "v0", 1),
    ("lw", "ra", 0, "sp"), ("addiu", "sp", "sp", 4),
    ("jr", "ra"), ("nop",),
    "h:",
    ("addu", "v0", "a0", "a0"),
    ("jr", "ra"), ("nop",),
]


# The same call, but the pointer goes through memory first (a struct field, a
# vtable slot, a callback stored for later). Same h, same answer.
FNPTR_MEM = [
    "main:",
    ("addiu", "sp", "sp", -8), ("sw", "ra", 4, "sp"),
    ("la", "t0", "h"),
    ("sw", "t0", 0, "sp"),                    # store the pointer
    ("lw", "t9", 0, "sp"),                    # load it back
    ("jalr", "t9"), ("nop",),
    ("addiu", "v0", "v0", 1),
    ("lw", "ra", 4, "sp"), ("addiu", "sp", "sp", 8),
    ("jr", "ra"), ("nop",),
    "h:",
    ("addu", "v0", "a0", "a0"),
    ("jr", "ra"), ("nop",),
]


# No pointers at all: main(n) = n + 8. The 8 is just a number. It also happens
# to be the address of the third instruction, which is what the crude scan sees.
COINCIDENCE = [
    "main:",
    ("addiu", "t1", "zero", 8),               # an ordinary constant...
    ("addiu", "v0", "zero", 0),
    ("addu", "v0", "a0", "t1"),               # ...and this instruction sits at 0x8
    ("jr", "ra"), ("nop",),
]


# ---- reference: callgraph's interpreter plus jalr -------------------------
def interp(words, a0, max_steps=2_000_000):
    reg, mem = [0] * 32, bytearray(cg.MEMSZ)
    reg[4], reg[29], reg[31] = a0, STACK_TOP, SENTINEL
    pc = 0
    for _ in range(max_steps):
        if pc == SENTINEL:
            return reg[2]
        m, o, tgt = disasm(words[pc // 4], pc)
        if m == "jal":
            reg[31] = pc + 8; cg._exec(words, reg, mem, pc + 4); pc = tgt
        elif m == "jalr":
            dest = reg[o[1]]                  # read the target BEFORE the delay slot runs
            reg[o[0]] = pc + 8; cg._exec(words, reg, mem, pc + 4); pc = dest
        elif m == "jr":
            cg._exec(words, reg, mem, pc + 4); pc = reg[o[0]]
        elif m == "beq":
            take = reg[o[0]] == reg[o[1]]
            cg._exec(words, reg, mem, pc + 4); pc = tgt if take else pc + 8
        else:
            cg._exec(words, reg, mem, pc); pc += 4
    raise RuntimeError("did not halt")


# ---- discovery --------------------------------------------------------------
def jal_entries(words):
    return {0} | {disasm(w, i * 4)[2] for i, w in enumerate(words) if disasm(w, i * 4)[0] == "jal"}


def pointer_entries(words):
    """Constant propagation, the cheapest kind: if the instruction loading a jalr's
    register earlier in the same straight-line run is `addiu rX, $zero, imm`, then
    imm is a function entry. Sees a pointer built from a constant; cannot see one
    loaded from memory, passed in, or computed."""
    found = set()
    for i, w in enumerate(words):
        m, o, _ = disasm(w, i * 4)
        if m != "jalr":
            continue
        rs = o[1]
        for j in range(i - 1, -1, -1):
            mj, oj, tj = disasm(words[j], j * 4)
            if mj in ("beq", "bne", "jal", "jalr", "jr") or tj is not None:
                break                         # left the straight line; give up
            if mj == "addiu" and oj[0] == rs:
                if oj[1] == 0:
                    found.add(oj[2])
                break
            if mj in ("addu", "lw", "slt") and oj[0] == rs:
                break                         # rs set some other way: not a constant
    return found


def address_constants(words):
    """The crude scan: any `addiu rX, $zero, imm` whose imm is an instruction
    address is a candidate entry. Finds pointers however they travel later, and
    also mistakes an ordinary number for a pointer when the two happen to match."""
    n = len(words) * 4
    return {o[2] for i, w in enumerate(words)
            for m, o, _ in [disasm(w, i * 4)]
            if m == "addiu" and o[1] == 0 and 0 < o[2] < n and o[2] % 4 == 0}


def functions(words, entries):
    es = sorted(entries)
    return [(e, es[i + 1] if i + 1 < len(es) else len(words) * 4) for i, e in enumerate(es)]


def _exits(words, s, e):
    """Does the function ending at e end in an unconditional transfer (jr, or an
    always-taken beq), so that control cannot run off its end into the next one?"""
    if e - s < 8:
        return False
    m, o, _ = disasm(words[(e - 8) // 4], e - 8)
    return m == "jr" or (m == "beq" and o[0] == o[1] == 0)


def regions(words, entries, fallthrough=False):
    fns = functions(words, entries)
    parent = list(range(len(fns)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owner = lambda a: next(i for i, (s, e) in enumerate(fns) if s <= a < e)
    for i, (s, e) in enumerate(fns):
        for pc in range(s, e, 4):
            m, _, t = disasm(words[pc // 4], pc)
            if m in ("beq", "bne") and not (s <= t < e):
                parent[find(i)] = find(owner(t))
    if fallthrough:
        # A "function" that does not end in an exit runs straight on into the next
        # one, so the two must be one C function or the first will simply return.
        for i in range(len(fns) - 1):
            if not _exits(words, *fns[i]):
                parent[find(i)] = find(i + 1)
    groups = {}
    for i, f in enumerate(fns):
        groups.setdefault(find(i), []).append(f)
    return [sorted(g) for g in groups.values()]


def emit(words, entries, fallthrough=False):
    groups = regions(words, entries, fallthrough)
    out = []
    for k, g in enumerate(groups):
        out.append(f"static void region_{k}(int entry);")
        for idx, (s, _) in enumerate(g):
            out.append(f"static void fn_{s:04x}(void) {{ region_{k}({idx}); }}")
    # A jalr target is a runtime value, so the recompiled call is a lookup over
    # every entry discovery knows about. An address it does not know is the one
    # failure no compiler can warn about: it can only be reported at runtime.
    out.append("static void dispatch(uint32_t a) {\n  switch (a) {")
    for e in sorted(entries):
        out.append(f"  case 0x{e:04x}: fn_{e:04x}(); return;")
    out.append('  default: fprintf(stderr, "jalr to 0x%04x: not a known function\\n", a); exit(3);\n  }\n}')
    for k, g in enumerate(groups):
        labs = {s for s, _ in g}
        for s, e in g:
            for pc in range(s, e, 4):
                m, _, t = disasm(words[pc // 4], pc)
                if m in ("beq", "bne"):
                    labs.add(t)
        out.append(f"static void region_{k}(int entry) {{\n  uint32_t *r = RF, cond, dest; (void)cond; (void)dest;")
        out.append("  switch (entry) {" + " ".join(f"case {i}: goto L_{s:04x};" for i, (s, _) in enumerate(g)) + " }")
        for s, e in g:
            pc = s
            while pc < e:
                w = words[pc // 4]
                m, o, t = disasm(w, pc)
                if pc in labs:
                    out.append(f"L_{pc:04x}:")
                # the delay slot is only meaningful after a control instruction
                nxt = cg._stmt(words[pc // 4 + 1], pc + 4) if m in ("beq", "jal", "jalr", "jr") else None
                if m == "beq":
                    out += [f"  cond = (r[{o[0]}] == r[{o[1]}]);   /* {pretty(w, pc)} */",
                            f"  {nxt}   /* delay slot */", f"  if (cond) goto L_{t:04x};"]
                    pc += 8; continue
                if m == "jal":
                    out += [f"  {nxt}   /* delay slot */", f"  fn_{t:04x}();   /* {pretty(w, pc)} */"]
                    pc += 8; continue
                if m == "jalr":
                    out += [f"  dest = r[{o[1]}];", f"  {nxt}   /* delay slot */",
                            f"  dispatch(dest);   /* {pretty(w, pc)} */"]
                    pc += 8; continue
                if m == "jr":
                    out += [f"  {nxt}   /* delay slot */", "  return;"]
                    pc += 8; continue
                out.append(f"  {cg._stmt(w, pc)}".ljust(40) + f"/* {pretty(w, pc)} */")
                pc += 4
        out.append("}")
    return "\n".join(out)


def build_run(words, entries, name, ns, opt="-O2", fallthrough=False):
    open(f"{name}.c", "w").write(cg.PRELUDE + emit(words, entries, fallthrough) + cg.MAIN_C)
    p = subprocess.run(["gcc", opt, "-o", name, f"{name}.c"], capture_output=True, text=True)
    if p.returncode:
        return None, p.stderr.strip().splitlines()[-1]
    r = subprocess.run([f"./{name}", *map(str, ns)], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()], (r.stderr.strip() or f"exit {r.returncode}")


if __name__ == "__main__" and __import__("sys").argv[1:] != ["--table"]:
    ns = list(range(0, 21))
    # 1. shared epilogue
    words, L = assemble(SHARED)
    ref = [interp(words, n) for n in ns]
    assert ref == [2 * n + 3 for n in ns]
    ents = jal_entries(words)
    print("SHARED: entries", [hex(e) for e in sorted(ents)], "regions",
          [[hex(s) for s, _ in g] for g in regions(words, ents)])
    try:
        cg.emit(words, tailcalls=True)
        print("   callgraph.py tail-call emitter: accepted it (unexpected)")
    except ValueError as e:
        print("   callgraph.py tail-call emitter: refuses ->", e)
    for opt in ("-O0", "-O2"):
        out, msg = build_run(words, ents, "cg2_shared", ns, opt)
        print(f"   region emitter {opt}: {'matches interp n=0..20' if out == ref else ('MISMATCH ' + str(out) + ' ' + msg)}")
    # 2. function pointer
    words, L = assemble(FNPTR)
    ref = [interp(words, n) for n in ns]
    assert ref == [2 * n + 1 for n in ns]
    ents = jal_entries(words)
    print(f"FNPTR: h is at {L['h']:#x}; jal-discovery finds", [hex(e) for e in sorted(ents)])
    out, msg = build_run(words, ents, "cg2_fnptr", ns)
    print("   region emitter, jal-only discovery:", "matches" if out == ref else f"output {out[:3] if out else out} | {msg}")
    extra = pointer_entries(words)
    print("   constant propagation finds pointer targets", [hex(e) for e in sorted(extra)])
    out, msg = build_run(words, ents | extra, "cg2_fnptr_cp", ns)
    print("   region emitter, + pointer discovery:", "matches interp n=0..20" if out == ref else f"MISMATCH {out} {msg}")
    # 3. the pointer passes through memory
    words, L = assemble(FNPTR_MEM)
    ref = [interp(words, n) for n in ns]
    assert ref == [2 * n + 1 for n in ns]
    ents = jal_entries(words); extra = pointer_entries(words)
    print(f"FNPTR_MEM: h at {L['h']:#x}; constant propagation finds", [hex(e) for e in sorted(extra)])
    out, msg = build_run(words, ents | extra, "cg2_fnptr_mem", ns)
    print("   region emitter, jal + pointer discovery:", "matches" if out == ref else f"output {out} | {msg}")
    # 4. the crude scan: catches the memory-borne pointer...
    extra = address_constants(words)
    out, msg = build_run(words, ents | extra, "cg2_fnptr_mem_scan", ns)
    print("   + crude address-constant scan finds", [hex(e) for e in sorted(extra)], "->",
          "matches" if out == ref else f"{out} {msg}")
    # ...and invents a function out of an ordinary number
    words, L = assemble(COINCIDENCE)
    ref = [interp(words, n) for n in ns]
    assert ref == [n + 8 for n in ns]
    scan = address_constants(words)
    print(f"COINCIDENCE: crude scan finds {[hex(e) for e in sorted(scan)]} (no pointers exist)")
    out, msg = build_run(words, jal_entries(words) | scan, "cg2_coinc", ns)
    print("   region emitter with that false entry:", "matches" if out == ref else f"WRONG, first outputs {out[:4]} want {ref[:4]}")
    out, msg = build_run(words, jal_entries(words) | scan, "cg2_coinc_ft", ns, fallthrough=True)
    print("   + fallthrough merging:", "matches interp n=0..20" if out == ref else f"WRONG {out[:4]}")
    for name, prog, want in (("SHARED", SHARED, lambda n: 2*n+3), ("FNPTR", FNPTR, lambda n: 2*n+1), ("FNPTR_MEM", FNPTR_MEM, lambda n: 2*n+1)):
        w2, _ = assemble(prog)
        e2 = jal_entries(w2) | address_constants(w2)
        out, msg = build_run(w2, e2, "cg2_all", ns, fallthrough=True)
        print(f"   crude scan + fallthrough on {name}:", "matches" if out == [want(n) for n in ns] else f"WRONG {out} {msg}")



def table(ns=range(21)):
    """Every cell of the table in the 2026-11-05 studio log, actually run."""
    ns = list(ns)
    progs = {"fnptr": (FNPTR, lambda n: 2 * n + 1), "fnptr via memory": (FNPTR_MEM, lambda n: 2 * n + 1),
             "coincidence": (COINCIDENCE, lambda n: n + 8)}
    rules = [("jal targets only", jal_entries, False),
             ("+ constant propagation", lambda w: jal_entries(w) | pointer_entries(w), False),
             ("+ crude scan", lambda w: jal_entries(w) | address_constants(w), False),
             ("+ crude scan + fallthrough merge", lambda w: jal_entries(w) | address_constants(w), True)]
    print(" " * 34 + "".join(f"{p:18}" for p in progs))
    for name, rule, ft in rules:
        cells = []
        for p, want in progs.values():
            w, _ = assemble(p)
            out, msg = build_run(w, rule(w), "cg2_tab", ns, fallthrough=ft)
            cells.append("ok" if out == [want(n) for n in ns] else ("ABORT" if "not a known function" in msg else "WRONG"))
        print(f"{name:34}" + "".join(f"{c:18}" for c in cells))


if __name__ == "__main__" and __import__("sys").argv[1:] == ["--table"]:
    table()
