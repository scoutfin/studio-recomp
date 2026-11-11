#!/usr/bin/env python3
"""A real pointer costs two instructions, and that breaks both of my scanners.

Studio, 2026-11-11. callgraph2.py (11-05) found functions reachable only through a
pointer in two ways: constant propagation anchored at the `jalr`, walking backwards
to whatever set its register; and a crude scan for any `addiu rX, $zero, imm` whose
imm looks like an instruction address. Both match on that one instruction form.

That form can only express an address that fits in a signed 16-bit immediate. Every
program I have assembled so far sits at address 0, so every pointer fit. **Real MIPS
text starts at 0x00400000.** A real pointer is built in two instructions:

    lui   $t9, 0x0040        # high half into the top 16 bits
    addiu $t9, $t9, 0x0018   # low half -- SIGN-EXTENDED, which is the trap
    jalr  $t9

So neither scanner can see a single pointer in a program linked at a realistic
address, and the fix is not a wider pattern. The two halves need not be adjacent,
need not share a register with the jalr, and one `lui` can serve many pointers.

Run: python3 addrpair.py
"""
import subprocess

import callgraph as cg
import callgraph2 as cg2
from mips import (addiu, addu, beq, disasm, jal, jalr, jr, lui, lw, nop, ori,
                  pretty, slt, sw)

SENTINEL, STACK_TOP = cg.SENTINEL, cg.STACK_TOP
TEXT = 0x00400000          # where a real MIPS program's text segment starts
# A base whose low half sits just under 0x8000, so that a function a few
# instructions in has a low half with bit 15 SET. Nothing special about the
# number: in any real binary roughly half the functions land on that side.
TEXT_HI = 0x00407FF0


# ---- splitting an address into a lui/lo pair --------------------------------
def split_addiu(a):
    """`lui rt,hi` then `addiu rt,rt,lo`. addiu sign-extends its immediate, so if
    the low half has bit 15 set the addiu SUBTRACTS 0x10000 worth, and hi has to be
    incremented to compensate. Every real assembler does this; GNU as calls the
    unincremented version a relocation truncation."""
    lo = a & 0xFFFF
    hi = ((a >> 16) + (1 if lo & 0x8000 else 0)) & 0xFFFF
    return hi, lo


def split_ori(a):
    """`lui rt,hi` then `ori rt,rt,lo`. ori zero-extends, so no carry. Same pointer,
    different hi field whenever bit 15 of the low half is set -- which is the whole
    reason a scanner cannot treat the two the same way."""
    return (a >> 16) & 0xFFFF, a & 0xFFFF


# ---- assembler, now with a base address -------------------------------------
# `la` is a two-word macro unconditionally. A pseudo-instruction whose WIDTH
# depends on the value it is loading would make label addresses depend on label
# addresses; real assemblers iterate to a fixpoint (relaxation) to allow it. Fixed
# width costs one wasted lui at base 0 and removes the whole problem.
LA_WIDTH = 2


def assemble(prog, base=TEXT, la="addiu"):
    labels, pc = {}, base
    for item in prog:
        if isinstance(item, str):
            labels[item.rstrip(":")] = pc
        else:
            pc += 4 * (LA_WIDTH if item[0] == "la" else 1)
    words, pc = [], base
    for item in prog:
        if isinstance(item, str):
            continue
        op, *a = item
        if op == "beq":
            words.append(beq(a[0], a[1], (labels[a[2]] - (pc + 4)) // 4))
        elif op == "jal":
            words.append(jal(labels[a[0]]))
        elif op == "la":
            addr = labels[a[1]] if a[1] in labels else a[1]
            hi, lo = (split_addiu if la == "addiu" else split_ori)(addr)
            words.append(lui(a[0], hi))
            words.append((addiu if la == "addiu" else ori)(a[0], a[0], lo))
            pc += 4
        elif op == "lui":
            words.append(lui(a[0], a[1]))
        elif op == "ori":
            words.append(ori(a[0], a[1], a[2]))
        elif op == "jalr":
            words.append(jalr(a[0]))
        else:
            words.append({"addiu": addiu, "addu": addu, "slt": slt, "lw": lw,
                          "sw": sw, "jr": jr, "nop": nop}[op](*a))
        pc += 4
    return words, labels


B = lambda l: ("beq", "zero", "zero", l)


# ---- fixtures ---------------------------------------------------------------
# main(n) = h(n) + 1, h(n) = 2n. h is reached ONLY through a register, and the
# register is loaded with a two-instruction address pair. This is FNPTR from
# 11-05, relocated, and that is the only difference.
PAIR = [
    "main:",
    ("addiu", "sp", "sp", -4), ("sw", "ra", 0, "sp"),
    ("la", "t9", "h"),
    ("jalr", "t9"), ("nop",),
    ("addiu", "v0", "v0", 1),
    ("lw", "ra", 0, "sp"), ("addiu", "sp", "sp", 4),
    ("jr", "ra"), ("nop",),
    "h:",
    ("addu", "v0", "a0", "a0"),
    ("jr", "ra"), ("nop",),
]

# The two halves are not adjacent. A compiler schedules whatever it likes between
# them; the lui is just a value-producing instruction with no special status.
PAIR_SPLIT = [
    "main:",
    ("addiu", "sp", "sp", -4), ("sw", "ra", 0, "sp"),
    ("lui", "t9", 0),                         # patched below: hi half of h
    ("addiu", "t1", "zero", 7),               # three unrelated instructions
    ("addu", "t2", "a0", "t1"),
    ("slt", "t3", "t1", "t2"),
    ("addiu", "t9", "t9", 0),                 # patched below: lo half of h
    ("jalr", "t9"), ("nop",),
    ("addiu", "v0", "v0", 1),
    ("lw", "ra", 0, "sp"), ("addiu", "sp", "sp", 4),
    ("jr", "ra"), ("nop",),
    "h:",
    ("addu", "v0", "a0", "a0"),
    ("jr", "ra"), ("nop",),
]

# One lui, two pointers. h and k share a high half because they are 24 bytes
# apart, which is what a real compiler does for any two functions in the same
# 64K page. A matcher that pairs a lui with ONE following low-half instruction
# finds whichever comes first and stops.
PAIR_SHARED = [
    "main:",
    ("addiu", "sp", "sp", -12), ("sw", "ra", 8, "sp"), ("sw", "s0", 4, "sp"),
    ("lui", "s0", 0),                         # patched: hi of h (== hi of k)
    ("addiu", "t9", "s0", 0),                 # patched: lo of h
    ("jalr", "t9"), ("nop",),
    ("sw", "v0", 0, "sp"),
    ("addiu", "t9", "s0", 0),                 # patched: lo of k
    ("addiu", "a0", "v0", 0),
    ("jalr", "t9"), ("nop",),
    ("lw", "t1", 0, "sp"),
    ("addu", "v0", "v0", "t1"),
    ("lw", "ra", 8, "sp"), ("lw", "s0", 4, "sp"), ("addiu", "sp", "sp", 12),
    ("jr", "ra"), ("nop",),
    "h:",
    ("addu", "v0", "a0", "a0"),
    ("jr", "ra"), ("nop",),
    "k:",
    ("addiu", "v0", "a0", 3),
    ("jr", "ra"), ("nop",),
]

# The same thing in $t8. By the MIPS ABI a callee may clobber $t8, so a compiler
# would not keep a live value there across a call -- but hand-written assembly
# does, and a scanner that trusts the ABI is trusting a convention, not the code.
PAIR_SHARED_T8 = [
    "main:",
    ("addiu", "sp", "sp", -8), ("sw", "ra", 4, "sp"),
    ("lui", "t8", 0),
    ("addiu", "t9", "t8", 0),
    ("jalr", "t9"), ("nop",),
    ("sw", "v0", 0, "sp"),
    ("addiu", "t9", "t8", 0),
    ("addiu", "a0", "v0", 0),
    ("jalr", "t9"), ("nop",),
    ("lw", "t1", 0, "sp"),
    ("addu", "v0", "v0", "t1"),
    ("lw", "ra", 4, "sp"), ("addiu", "sp", "sp", 8),
    ("jr", "ra"), ("nop",),
    "h:",
    ("addu", "v0", "a0", "a0"),
    ("jr", "ra"), ("nop",),
    "k:",
    ("addiu", "v0", "a0", 3),
    ("jr", "ra"), ("nop",),
]

# The pointer goes through memory before it is called, so nothing in the
# straight-line run before the jalr sets its register from a constant.
PAIR_MEM = [
    "main:",
    ("addiu", "sp", "sp", -8), ("sw", "ra", 4, "sp"),
    ("la", "t0", "h"),
    ("sw", "t0", 0, "sp"),
    ("lw", "t9", 0, "sp"),
    ("jalr", "t9"), ("nop",),
    ("addiu", "v0", "v0", 1),
    ("lw", "ra", 4, "sp"), ("addiu", "sp", "sp", 8),
    ("jr", "ra"), ("nop",),
    "h:",
    ("addu", "v0", "a0", "a0"),
    ("jr", "ra"), ("nop",),
]

# No pointers at all: main(n) = n + 0x400018. The constant is built the same way
# a pointer is, and at TEXT it collides with h's address in PAIR. This is the
# coincidence fixture from 11-05, moved up to 32 bits.
COINCIDENCE32 = [
    "main:",
    ("lui", "t1", 0x0040),
    ("ori", "t1", "t1", 0x0018),
    ("addu", "v0", "a0", "t1"),
    ("jr", "ra"), ("nop",),
]


def patch_pair(words, base, idx_hi, idx_lo, addr, lo_rs=None, lo_rt=None):
    """Rewrite a placeholder lui/addiu pair to carry `addr`. idx_* are word indices."""
    hi, lo = split_addiu(addr)
    _, o_hi, _ = disasm(words[idx_hi])
    _, o_lo, _ = disasm(words[idx_lo])
    words[idx_hi] = lui(o_hi[0], hi)
    words[idx_lo] = addiu(lo_rt if lo_rt is not None else o_lo[0],
                          lo_rs if lo_rs is not None else o_lo[1], lo)


# ---- interpreter, base-aware ------------------------------------------------
def _ex(words, base, reg, mem, pc):
    m, o, _ = disasm(words[(pc - base) // 4], pc)

    def wr(d, v):
        if d:
            reg[d] = v & 0xFFFFFFFF
    if m == "nop":
        return
    if m == "addu":    wr(o[0], reg[o[1]] + reg[o[2]])
    elif m == "addiu": wr(o[0], reg[o[1]] + o[2])
    elif m == "lui":   wr(o[0], o[1] << 16)
    elif m == "ori":   wr(o[0], reg[o[1]] | o[2])
    elif m == "slt":   wr(o[0], int((reg[o[1]] ^ 0x80000000) < (reg[o[2]] ^ 0x80000000)))
    elif m == "lw":    wr(o[0], cg._be_lw(mem, reg[o[2]] + o[1]))
    elif m == "sw":    cg._be_sw(mem, reg[o[2]] + o[1], reg[o[0]])
    else: raise ValueError(m)


def interp(words, a0, base=TEXT, max_steps=2_000_000):
    reg, mem = [0] * 32, bytearray(cg.MEMSZ)
    reg[4], reg[29], reg[31] = a0, STACK_TOP, SENTINEL
    pc = base
    for _ in range(max_steps):
        if pc == SENTINEL:
            return reg[2]
        m, o, tgt = disasm(words[(pc - base) // 4], pc)
        if m == "jal":
            reg[31] = pc + 8; _ex(words, base, reg, mem, pc + 4); pc = tgt
        elif m == "jalr":
            dest = reg[o[1]]
            reg[o[0]] = pc + 8; _ex(words, base, reg, mem, pc + 4); pc = dest
        elif m == "jr":
            _ex(words, base, reg, mem, pc + 4); pc = reg[o[0]]
        elif m in ("beq", "bne"):
            take = (reg[o[0]] == reg[o[1]]) == (m == "beq")
            _ex(words, base, reg, mem, pc + 4); pc = tgt if take else pc + 8
        else:
            _ex(words, base, reg, mem, pc); pc += 4
    raise RuntimeError("did not halt")


# ---- discovery --------------------------------------------------------------
def jal_entries(words, base=TEXT):
    return {base} | {disasm(w, base + i * 4)[2] for i, w in enumerate(words)
                     if disasm(w, base + i * 4)[0] == "jal"}


def in_text(a, words, base):
    return base <= a < base + len(words) * 4 and a % 4 == 0


def pair_entries_naive(words, base=TEXT):
    """The pattern matcher: a `lui rt, hi`, then the first later instruction in the
    same straight-line run that writes rt from rt. Combines the halves with
    (hi << 16) | lo, which is what the two instructions LOOK like they do."""
    found = set()
    for i, w in enumerate(words):
        m, o, _ = disasm(w, base + i * 4)
        if m != "lui":
            continue
        hi, rt = o[1], o[0]
        for j in range(i + 1, len(words)):
            mj, oj, tj = disasm(words[j], base + j * 4)
            if mj in ("beq", "bne", "jal", "jalr", "jr") or tj is not None:
                break
            if mj in ("addiu", "ori") and oj[0] == rt and oj[1] == rt:
                a = (hi << 16) | (oj[2] & 0xFFFF)
                if in_text(a, words, base):
                    found.add(a)
                break
            if oj and oj[0] == rt:
                break
    return found


CALLER_SAVED = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 31}


def pair_entries(words, base=TEXT, across="abi", bare_lui=True):
    """Constant propagation, collecting AT CONSTRUCTION rather than at a use.

    The difference from pair_entries_naive is not precision, it is the anchor.
    There is no anchor: every register value that can be shown constant and lands
    in the text range is a candidate, whatever it is later used for. A split pair
    and a pointer that travels through memory need no case of their own.

    `across` decides what a control transfer does to the known constants, and it
    is the whole measurement:
      "block"  reset everything, the way a straight-line run is usually defined
      "abi"    a call clobbers only the caller-saved registers; $s0-$s7 survive
      "never"  keep everything, and let a stale constant produce a false entry

    `bare_lui=False` drops a value that came straight from a lui with no low half
    added yet. That value is always a 64K-aligned address, which in a real binary
    is the start of a page and almost never a function."""
    found, const, from_lui = set(), {}, set()

    def kill(d):
        const.pop(d, None); from_lui.discard(d)
    for i, w in enumerate(words):
        pc = base + i * 4
        m, o, _ = disasm(w, pc)
        if m in ("jal", "jalr"):
            if across == "block":
                const.clear(); from_lui.clear()
            elif across == "abi":
                for d in list(const):
                    if d in CALLER_SAVED:
                        kill(d)
            continue
        if m in ("beq", "bne", "jr"):
            if across == "block":
                const.clear(); from_lui.clear()
            continue
        if m == "lui":
            const[o[0]] = o[1] << 16; from_lui.add(o[0])
        elif m == "addiu" and (o[1] == 0 or o[1] in const):
            # o[2] is already the SIGNED immediate, so addiu's sign-extension
            # carry is handled by arithmetic rather than by a rule about bit 15.
            const[o[0]] = (const.get(o[1], 0) + o[2]) & 0xFFFFFFFF
            from_lui.discard(o[0])
        elif m == "ori" and (o[1] == 0 or o[1] in const):
            const[o[0]] = const.get(o[1], 0) | o[2]
            from_lui.discard(o[0])
        elif m == "nop":
            pass
        elif o:
            kill(o[0])                        # written some other way: not constant
        for d, v in const.items():
            if in_text(v, words, base) and (bare_lui or d not in from_lui):
                found.add(v)
    return found


# ---- emitter, base-aware ----------------------------------------------------
def _stmt(w, pc):
    m, o, _ = disasm(w, pc)
    if m == "lui":
        return f"W({o[0]}, {o[1]}u << 16);"
    if m == "ori":
        return f"W({o[0]}, r[{o[1]}] | {o[2]}u);"
    return cg._stmt(w, pc)


def functions(words, entries, base):
    es = sorted(entries)
    return [(e, es[i + 1] if i + 1 < len(es) else base + len(words) * 4)
            for i, e in enumerate(es)]


def _exits(words, base, s, e):
    if e - s < 8:
        return False
    m, o, _ = disasm(words[(e - 8 - base) // 4], e - 8)
    return m == "jr" or (m == "beq" and o[0] == o[1] == 0)


def regions(words, entries, base=TEXT, fallthrough=True):
    fns = functions(words, entries, base)
    parent = list(range(len(fns)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owner = lambda a: next(i for i, (s, e) in enumerate(fns) if s <= a < e)
    for i, (s, e) in enumerate(fns):
        for pc in range(s, e, 4):
            m, _, t = disasm(words[(pc - base) // 4], pc)
            if m in ("beq", "bne") and not (s <= t < e):
                parent[find(i)] = find(owner(t))
    if fallthrough:
        for i in range(len(fns) - 1):
            if not _exits(words, base, *fns[i]):
                parent[find(i)] = find(i + 1)
    groups = {}
    for i, f in enumerate(fns):
        groups.setdefault(find(i), []).append(f)
    return [sorted(g) for g in groups.values()]


def emit(words, entries, base=TEXT, fallthrough=True):
    groups = regions(words, entries, base, fallthrough)
    out = []
    for k, g in enumerate(groups):
        out.append(f"static void region_{k}(int entry);")
        for idx, (s, _) in enumerate(g):
            out.append(f"static void fn_{s:08x}(void) {{ region_{k}({idx}); }}")
    out.append("static void dispatch(uint32_t a) {\n  switch (a) {")
    for e in sorted(entries):
        out.append(f"  case 0x{e:08x}: fn_{e:08x}(); return;")
    out.append('  default: fprintf(stderr, "jalr to 0x%08x: not a known function\\n", a);'
               ' exit(3);\n  }\n}')
    for k, g in enumerate(groups):
        labs = {s for s, _ in g}   # every group start is a goto target of the entry switch
        for s, e in g:
            for pc in range(s, e, 4):
                m, _, t = disasm(words[(pc - base) // 4], pc)
                if m in ("beq", "bne"):
                    labs.add(t)
        out.append(f"static void region_{k}(int entry) {{\n"
                   "  uint32_t *r = RF, cond, dest; (void)cond; (void)dest;")
        out.append("  switch (entry) {"
                   + " ".join(f"case {i}: goto L_{s:08x};" for i, (s, _) in enumerate(g))
                   + " }")
        for s, e in g:
            pc = s
            while pc < e:
                w = words[(pc - base) // 4]
                m, o, t = disasm(w, pc)
                if pc in labs:
                    out.append(f"L_{pc:08x}:")
                nxt = (_stmt(words[(pc - base) // 4 + 1], pc + 4)
                       if m in ("beq", "bne", "jal", "jalr", "jr") else None)
                if m in ("beq", "bne"):
                    op = "==" if m == "beq" else "!="
                    out += [f"  cond = (r[{o[0]}] {op} r[{o[1]}]);   /* {pretty(w, pc)} */",
                            f"  {nxt}   /* delay slot */", f"  if (cond) goto L_{t:08x};"]
                    pc += 8; continue
                if m == "jal":
                    out += [f"  {nxt}   /* delay slot */", f"  fn_{t:08x}();   /* {pretty(w, pc)} */"]
                    pc += 8; continue
                if m == "jalr":
                    out += [f"  dest = r[{o[1]}];", f"  {nxt}   /* delay slot */",
                            f"  dispatch(dest);   /* {pretty(w, pc)} */"]
                    pc += 8; continue
                if m == "jr":
                    out += [f"  {nxt}   /* delay slot */", "  return;"]
                    pc += 8; continue
                out.append(f"  {_stmt(w, pc)}".ljust(44) + f"/* {pretty(w, pc)} */")
                pc += 4
        out.append("}")
    return "\n".join(out)


MAIN_C = """
int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    memset(mem, 0, sizeof mem); memset(RF, 0, sizeof RF);
    RF[29] = 0x8000; RF[4] = (uint32_t)strtoul(argv[i], 0, 10);
    fn_%08x();
    printf("%%u\\n", RF[2]);
  }
  return 0;
}
"""


def build_run(words, entries, name, ns, base=TEXT, opt="-O2", fallthrough=True):
    src = cg.PRELUDE + emit(words, entries, base, fallthrough) + (MAIN_C % base)
    open(f"{name}.c", "w").write(src)
    p = subprocess.run(["gcc", opt, "-o", name, f"{name}.c"], capture_output=True, text=True)
    if p.returncode:
        return None, p.stderr.strip().splitlines()[-1]
    r = subprocess.run([f"./{name}", *map(str, ns)], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split()], (r.stderr.strip() or f"exit {r.returncode}")


def verdict(out, want, msg):
    if out == want:
        return "ok"
    if msg and "not a known function" in msg:
        return "ABORT"
    return f"WRONG {out[:3] if out else out}"


def prepare(name, base=TEXT, la="addiu"):
    """Assemble a fixture, patching the hand-written placeholder pairs."""
    if name == "PAIR":
        return assemble(PAIR, base, la)
    if name == "PAIR_MEM":
        return assemble(PAIR_MEM, base, la)
    if name == "COINCIDENCE32":
        w, L = assemble(COINCIDENCE32, base, la)
        hi, lo = split_ori(base + 16)
        w[0], w[1] = lui("t1", hi), ori("t1", "t1", lo)
        return w, L
    if name == "PAIR_SPLIT":
        w, L = assemble(PAIR_SPLIT, base, la)
        patch_pair(w, base, 2, 6, L["h"])
        return w, L
    if name in ("PAIR_SHARED", "PAIR_SHARED_T8"):
        reg, i_lui = ("s0", 3) if name == "PAIR_SHARED" else ("t8", 2)
        w, L = assemble(globals()[name], base, la)
        hi_h, lo_h = split_addiu(L["h"])
        hi_k, lo_k = split_addiu(L["k"])
        assert hi_h == hi_k, (hex(L["h"]), hex(L["k"]))   # the point of the fixture
        w[i_lui] = lui(reg, hi_h)
        w[i_lui + 1] = addiu("t9", reg, lo_h)
        w[i_lui + 5] = addiu("t9", reg, lo_k)
        return w, L
    raise KeyError(name)


def prepare_big(prog, base=TEXT):
    """Assemble a padded program. The la macro is still two words, so the only
    thing that changes is how far apart the pair's halves can be in ADDRESS."""
    return assemble(prog, base)


WANT = {"PAIR": lambda n: 2 * n + 1, "PAIR_SPLIT": lambda n: 2 * n + 1,
        "PAIR_SHARED": lambda n: 4 * n + 3, "PAIR_SHARED_T8": lambda n: 4 * n + 3, "PAIR_MEM": lambda n: 2 * n + 1,
        "COINCIDENCE32": lambda n: n + TEXT + 16}

if __name__ == "__main__":
    ns = list(range(0, 11))
    names = ["PAIR", "PAIR_SPLIT", "PAIR_SHARED", "PAIR_SHARED_T8", "PAIR_MEM",
             "COINCIDENCE32"]

    print("=== 0. the interpreter agrees with the arithmetic, so the fixtures are real ===")
    for nm in names:
        w, L = prepare(nm)
        got = [interp(w, n) for n in ns]
        want = [WANT[nm](n) for n in ns]
        print(f"  {nm:14} {len(w):3} words  interp == closed form: {got == want}"
              f"   {'h=' + hex(L['h']) if 'h' in L else ''}"
              f"{'  k=' + hex(L['k']) if 'k' in L else ''}")

    print("\n=== 1. what the 11-05 scanners see in a program linked at 0x00400000 ===")
    for nm in names:
        w, _ = prepare(nm)
        old_cp = cg2.pointer_entries(w)
        old_scan = cg2.address_constants(w)
        print(f"  {nm:14} jalr-anchored const-prop: {len(old_cp)} found"
              f"   crude addiu scan: {len(old_scan)} found")

    print("\n=== 2. the pointer disassembled, and the two splits ===")
    w, L = prepare("PAIR")
    for i in (2, 3, 4):
        print(f"  {TEXT + i * 4:#010x}  {pretty(w[i], TEXT + i * 4)}")
    print(f"  h is at {L['h']:#010x}")
    for a in (0x00400044, 0x00408008):
        print(f"  {a:#010x}:  addiu split hi={split_addiu(a)[0]:#06x} lo={split_addiu(a)[1]:#06x}"
              f"    ori split hi={split_ori(a)[0]:#06x} lo={split_ori(a)[1]:#06x}")

    print("\n=== 3. recompile and RUN: three discovery rules x five programs ===")
    rules = [("jal targets only", lambda w, b: jal_entries(w, b)),
             ("+ naive lui/lo pattern", lambda w, b: jal_entries(w, b) | pair_entries_naive(w, b)),
             ("+ constant propagation", lambda w, b: jal_entries(w, b) | pair_entries(w, b))]
    print(" " * 26 + "".join(f"{n:16}" for n in names))
    for label, rule in rules:
        cells = []
        for nm in names:
            w, _ = prepare(nm)
            out, msg = build_run(w, rule(w, TEXT), f"ap_{nm.lower()}", ns)
            cells.append(verdict(out, [WANT[nm](n) for n in ns], msg))
        print(f"{label:26}" + "".join(f"{c:16}" for c in cells))

    print("\n=== 4. the same program at two bases: where the naive combiner breaks ===")
    for base in (TEXT, TEXT_HI):
        w, L = prepare("PAIR", base)
        want = [2 * n + 1 for n in ns]
        assert [interp(w, n, base) for n in ns] == want
        naive = pair_entries_naive(w, base)
        cp = pair_entries(w, base)
        lo = L["h"] & 0xFFFF
        print(f"  base {base:#010x}  h={L['h']:#010x}  low half {lo:#06x}"
              f"  bit15={'SET' if lo & 0x8000 else 'clear'}")
        # "finds h" is the question, not set equality: 0x00400000 is the bare
        # lui's value AND main's real entry, so counting it as an error would be
        # a scoring bug rather than a finding. Section 7 separates the two.
        print(f"     naive (hi<<16)|lo -> {[hex(x) for x in sorted(naive)]}"
              f"   finds h? {L['h'] in naive}")
        print(f"     constant prop     -> {[hex(x) for x in sorted(cp)]}"
              f"   finds h? {L['h'] in cp}")
        for nmn, ents in (("naive", naive), ("constprop", cp)):
            out, msg = build_run(w, jal_entries(w, base) | ents,
                                 f"ap_hi_{nmn}", ns, base=base)
            print(f"     emitted C with {nmn:9} entries: {verdict(out, want, msg)}"
                  f"   {msg if out != want else ''}")

    print("\n=== 5. false positives: every immediate constant my fixtures contain ===")
    allfix = {**{n: globals()[n] for n in ("PAIR_SPLIT", "PAIR_SHARED", "PAIR_SHARED_T8", "COINCIDENCE32")},
              "PAIR": PAIR, "PAIR_MEM": PAIR_MEM,
              "SHARED": cg2.SHARED, "FNPTR": cg2.FNPTR, "FNPTR_MEM": cg2.FNPTR_MEM,
              "COINCIDENCE": cg2.COINCIDENCE}
    consts, hits0, hitsT = [], [], []
    for nm, prog in allfix.items():
        for item in prog:
            if isinstance(item, str):
                continue
            for a in item[1:]:
                if isinstance(a, int):
                    consts.append((nm, a))
    for nm, c in consts:
        v = c & 0xFFFFFFFF
        if 0 < v < 0x100 and v % 4 == 0:
            hits0.append((nm, c))
        if TEXT <= v < TEXT + 0x100 and v % 4 == 0:
            hitsT.append((nm, c))
    print(f"  {len(consts)} immediate operands across {len(allfix)} fixtures")
    print(f"  land in a 256-byte text range at base 0x00000000: {len(hits0)}"
          f"  ({100 * len(hits0) / len(consts):.1f}%)  e.g. {sorted({c for _, c in hits0})[:8]}")
    print(f"  land in a 256-byte text range at base 0x00400000: {len(hitsT)}"
          f"  ({100 * len(hitsT) / len(consts):.1f}%)")

    print("\n=== 6. what a control transfer should do to a known constant ===")
    print("    real = entries that exist; found = candidates; extra = false entries")
    hdr = f"{'rule':34}" + "".join(f"{n:16}" for n in names)
    for across in ("block", "abi", "never"):
        for bare in (True, False):
            cells = []
            for nm in names:
                w, L = prepare(nm)
                real = {TEXT} | {v for k, v in L.items() if k != "main"}
                ents = jal_entries(w) | pair_entries(w, across=across, bare_lui=bare)
                out, msg = build_run(w, ents, "ap_sweep", ns)
                v = verdict(out, [WANT[nm](n) for n in ns], msg)
                cells.append(f"{v}({len(ents)}/{len(real)})")
            label = f"across={across:6} bare_lui={'yes' if bare else 'no ':3}"
            if not cells:
                continue
            if across == "block" and bare:
                print(hdr)
            print(f"{label:34}" + "".join(f"{c:16}" for c in cells))

    print("\n=== 7. the bare lui, on a text segment bigger than one 64K page ===")
    # Above, bare_lui changed nothing, because the only bare-lui value was
    # 0x00400000 -- which IS main. Give the program more than 64K of text and
    # the intermediate lands in the middle of it instead, where no function is.
    pad = [("nop",)] * 16400
    BIG = PAIR[:3] + pad + PAIR[3:]
    w, L = prepare_big(BIG)
    real = {TEXT, L["h"]}
    for bare in (True, False):
        got = jal_entries(w) | pair_entries(w, bare_lui=bare)
        print(f"  text {len(w) * 4 / 1024:.0f}K, h={L['h']:#010x}  bare_lui={'yes' if bare else 'no'}:"
              f" {len(got)} candidates, {len(got - real)} of them not functions"
              f" {sorted(hex(x) for x in got - real)}")

    print("\n=== 8. the same two instruction SHAPES, two different meanings ===")
    for la in ("addiu", "ori"):
        w, L = prepare("PAIR", TEXT_HI, la)
        want = [2 * n + 1 for n in ns]
        assert [interp(w, n, TEXT_HI) for n in ns] == want
        naive, cp = pair_entries_naive(w, TEXT_HI), pair_entries(w, TEXT_HI)
        print(f"  lui + {la:5} -> {pretty(w[2], TEXT_HI + 8)} ; {pretty(w[3], TEXT_HI + 12)}")
        print(f"     h={L['h']:#010x}   naive (hi<<16)|lo: {'right' if naive == {L['h']} else 'WRONG ' + str([hex(x) for x in sorted(naive)])}"
              f"   const prop: {'right' if L['h'] in cp else 'WRONG'}")

    print("\n=== 9. when the +0x10000 error survives the range filter ===")
    # In section 8 the naive combiner returned nothing, not a wrong address: its
    # answer was 0x10000 too high and the text-range test threw it away. The
    # error only becomes a FALSE ENTRY when the segment reaches 64K past the
    # real target, so pad AFTER h rather than before it.
    w, L = prepare_big(PAIR + [("nop",)] * 16400, TEXT_HI)
    real, jals = {TEXT_HI, L["h"]}, jal_entries(w, TEXT_HI)
    naive, cp = pair_entries_naive(w, TEXT_HI), pair_entries(w, TEXT_HI, bare_lui=False)
    print(f"  text {len(w) * 4 / 1024:.0f}K, h={L['h']:#010x} (low half {L['h'] & 0xFFFF:#06x})")
    for nm, got in (("naive", naive), ("const prop", cp)):
        print(f"  {nm:10} -> {sorted(hex(x) for x in got)}"
              f"   invented {len(got - real)}, missed {len(real - got - jals)}")
