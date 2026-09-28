#!/usr/bin/env python3
"""Option 2: bound the indirect-jump targets from ABOVE, without running anything.

Tracing (discover.py) is sound and silently incomplete — it bounds the jump
table from BELOW. Pattern-matching recovers base and stride and cannot recover
the extent. This is the third method: track every value a register CAN hold,
and read the answer off the register feeding `jr`.

WHY A STRIDED INTERVAL AND NOT AN INTERVAL
------------------------------------------
A jump table computes `target = BASE + (index << k)`. Under a plain interval
domain the best you can say is `[0x20, 0x50]`, which contains 0x21..0x2F — 45
addresses that are not handlers and never could be. The abstraction has thrown
away the only structure that mattered.

A *strided* interval `stride[lo, hi]` = {lo, lo+stride, ..., hi} survives a
left shift exactly: shifting `1[0,3]` by 4 gives `16[0,48]`, and adding the
base gives `16[0x20, 0x50]` — the four real handlers and nothing else. The
domain was chosen to have the same shape as the thing being analysed, which is
the whole art of picking one.

WHERE THE BOUND COMES FROM, AND WHY IT USUALLY ISN'T THERE
----------------------------------------------------------
`$a0` arrives unconstrained, so it starts at TOP and the analysis honestly
answers UNBOUNDED. To get a finite table you need a bounds check in the
program — and then the analysis has to do the part that makes it an abstract
interpreter rather than a calculator: when a conditional branch is taken, it
must push the condition BACKWARDS onto the register the comparison read.

That is why the guarded and unguarded versions of the same table give
completely different answers here, and it is the concrete form of something I
wrote in the last log: the bounds check you would dismiss as defensive
programming is the thing that makes the binary analysable.
"""
import sys
from dataclasses import dataclass

from mips import addiu, addu, beq, jr, lw, nop, sll, slt
from recomp import disasm

MASK = 0xFFFFFFFF


# --- the abstract domain ------------------------------------------------------

@dataclass(frozen=True)
class SI:
    """Strided interval: {lo, lo+stride, ..., hi}. stride 0 means the single
    value lo. `top` means 'could be anything' — the honest shrug."""
    stride: int = 0
    lo: int = 0
    hi: int = 0
    top: bool = False

    @staticmethod
    def const(v):
        return SI(0, v, v)

    @staticmethod
    def TOP():
        return SI(top=True)

    def values(self, cap=4096):
        if self.top:
            return None
        if self.stride == 0:
            return [self.lo]
        n = (self.hi - self.lo) // self.stride + 1
        if n > cap:
            return None
        return [self.lo + i * self.stride for i in range(n)]

    def __str__(self):
        if self.top:
            return "TOP (unbounded)"
        if self.stride == 0:
            return f"{{0x{self.lo:x}}}"
        return f"0x{self.lo:x} + {self.stride}k, k in [0,{(self.hi-self.lo)//self.stride}]"


def si_add(a, b):
    if a.top or b.top:
        return SI.TOP()
    if a.stride == 0 and b.stride == 0:
        return SI.const((a.lo + b.lo) & MASK)
    if a.stride == 0:
        return SI(b.stride, (b.lo + a.lo) & MASK, (b.hi + a.lo) & MASK)
    if b.stride == 0:
        return SI(a.stride, (a.lo + b.lo) & MASK, (a.hi + b.lo) & MASK)
    # Two non-constant strides: only exact when the strides agree. Otherwise
    # the join of the two lattices is not a strided interval and I refuse to
    # pretend it is.
    if a.stride == b.stride:
        return SI(a.stride, (a.lo + b.lo) & MASK, (a.hi + b.hi) & MASK)
    return SI.TOP()


def si_shl(a, k):
    if a.top:
        return SI.TOP()
    if a.stride == 0:
        return SI.const((a.lo << k) & MASK)
    return SI(a.stride << k, (a.lo << k) & MASK, (a.hi << k) & MASK)


def si_join(a, b):
    """Least upper bound — used where control flow merges."""
    if a is None:
        return b
    if b is None:
        return a
    if a.top or b.top:
        return SI.TOP()
    if a == b:
        return a
    if a.stride == 0 and b.stride == 0:
        lo, hi = min(a.lo, b.lo), max(a.lo, b.lo)
        return SI.const(lo) if lo == hi else SI(hi - lo, lo, hi)
    if a.stride == b.stride and (a.lo - b.lo) % max(1, a.stride) == 0:
        return SI(a.stride, min(a.lo, b.lo), max(a.hi, b.hi))
    return SI.TOP()


# --- a second domain, because the load changes what shape the answer has -------
#
# Added 2026-09-28, and the reason is the whole lesson of this session.
#
# A strided interval is exactly right for the ADDRESS arithmetic: `index << 2`
# plus a base is a stride, and SI tracks it with no loss. But a real jump table
# holds ARBITRARY handler addresses — 0x400120, 0x4001a8, 0x400214, 0x400188 —
# and those have no stride at all. Joining them with si_join collapses to TOP or,
# worse, to an interval containing thousands of addresses that are not handlers.
#
# So the domain that is perfect on one side of the load is wrong on the other.
# `lw` is the point where the abstraction has to CHANGE SHAPE: strided interval
# in, finite value set out. I did not see that until I tried to write the load.

VS_CAP = 64


@dataclass(frozen=True)
class VS:
    """A finite set of concrete values, or TOP above VS_CAP. Used for anything
    read out of memory, where structure is not expected."""
    vals: frozenset = frozenset()
    top: bool = False

    @staticmethod
    def of(it):
        v = frozenset(it)
        return VS(top=True) if len(v) > VS_CAP else VS(v)

    @staticmethod
    def TOP():
        return VS(top=True)

    def values(self, cap=VS_CAP):
        return None if self.top else sorted(self.vals)

    def __str__(self):
        if self.top:
            return "TOP (unbounded)"
        v = sorted(self.vals)
        return "{" + ", ".join(f"0x{x:x}" for x in v) + "}"


def is_top(a):
    return a.top


def enumerate_av(a, cap=VS_CAP):
    """Concrete values of either domain, or None if unbounded/too many."""
    return None if a.top else a.values(cap)


def av_join(a, b):
    """Join across the two domains. Same-kind joins keep their kind; mixed ones
    fall to VS, because a set is the only thing that can hold both without
    inventing structure."""
    if a is None:
        return b
    if b is None:
        return a
    if isinstance(a, SI) and isinstance(b, SI):
        return si_join(a, b)
    if a.top or b.top:
        return VS.TOP()
    xa, xb = enumerate_av(a), enumerate_av(b)
    if xa is None or xb is None:
        return VS.TOP()
    return VS.of(xa + xb)


def av_add(a, b):
    if isinstance(a, SI) and isinstance(b, SI):
        return si_add(a, b)
    if a.top or b.top:
        return VS.TOP()
    xa, xb = enumerate_av(a), enumerate_av(b)
    if xa is None or xb is None or len(xa) * len(xb) > VS_CAP:
        return VS.TOP()
    return VS.of(((x + y) & MASK) for x in xa for y in xb)


def av_shl(a, k):
    if isinstance(a, SI):
        return si_shl(a, k)
    if a.top:
        return VS.TOP()
    return VS.of(((x << k) & MASK) for x in a.vals)


def av_load(addr, ro):
    """The load. `addr` is an abstract address; `ro` maps concrete word-aligned
    addresses to concrete words (the binary's read-only data).

    Honest failure modes, in order:
      - address unbounded        -> TOP, obviously
      - too many addresses       -> TOP rather than a huge set
      - ANY address not in `ro`  -> TOP, because one unknown word poisons the set
                                    and pretending otherwise is how an analysis
                                    reports a bound it has not got
    """
    xs = enumerate_av(addr)
    if xs is None:
        return VS.TOP(), "address is unbounded"
    if len(xs) > VS_CAP:
        return VS.TOP(), f"{len(xs)} possible addresses exceeds cap"
    out = []
    for a in xs:
        if a % 4:
            return VS.TOP(), f"unaligned address 0x{a:x}"
        if a not in ro:
            return VS.TOP(), f"0x{a:x} is not in known read-only memory"
        out.append(ro[a])
    return VS.of(out), None


def si_below(a, limit):
    """Refine `a` with the fact a < limit. This is the backwards step."""
    if a.top:
        # An unconstrained value learning an upper bound becomes 1[0, limit-1]:
        # unsigned comparison, so the lower end is 0.
        return SI(1, 0, limit - 1) if limit > 0 else SI.TOP()
    if a.stride == 0:
        return a if a.lo < limit else None            # None = infeasible path
    hi = min(a.hi, a.lo + ((limit - 1 - a.lo) // a.stride) * a.stride)
    return SI(a.stride, a.lo, hi) if hi >= a.lo else None


# --- the analysis -------------------------------------------------------------

def analyse(words, base=0, verbose=False, ro=None):
    """Walk the instruction stream, tracking an SI per register, and report the
    set of targets `jr` could reach.

    Straight-line plus forward branches only — enough for a dispatch stub, and
    I am not pretending it is a general CFG fixpoint. See the limits in the
    studio log.
    """
    ro = ro or {}
    why = None                                         # why a load gave up
    regs = {i: SI.TOP() for i in range(32)}
    regs[0] = SI.const(0)                              # $zero really is zero
    facts = {}                                         # rd -> ('lt', rs, limit)
    pc = base
    steps = 0
    while steps < 4000:
        steps += 1
        idx = (pc - base) // 4
        if idx < 0 or idx >= len(words):
            break
        m, o, tgt = disasm(words[idx], pc)
        if verbose:
            print(f"    0x{pc:04x}  {m:<6} regs[t3]={regs[11]}")

        if m == "jr":
            return regs[o[0]], facts, why
        elif m == "sll":
            regs[o[0]] = av_shl(regs[o[1]], o[2])
        elif m == "addu":
            regs[o[0]] = av_add(regs[o[1]], regs[o[2]])
        elif m == "addiu":
            regs[o[0]] = av_add(regs[o[1]], SI.const(o[2] & MASK))
        elif m == "slt":
            # Record the comparison rather than its boolean value: the boolean
            # is worthless, the CONSTRAINT it stands for is the whole point.
            lim = regs[o[2]]
            if not lim.top and lim.stride == 0:
                facts[o[0]] = ("lt", o[1], lim.lo)
            regs[o[0]] = SI(1, 0, 1)
        elif m == "beq":
            # `beq rd, $zero, else` — fall-through is the case where the
            # comparison held. Push that fact onto the compared register.
            rd, rt = o[0], o[1]
            if rt == 0 and rd in facts:
                kind, src, lim = facts[rd]
                refined = si_below(regs[src], lim)
                if refined is None:
                    break
                regs[src] = refined
            pc += 8                                    # skip the delay slot
            continue
        elif m == "lw":
            # rt <- mem[rs + offset]. The abstraction changes shape here: a
            # strided interval goes in and a value set comes out, because table
            # contents have no stride.
            rt, off, rs = o
            addr = av_add(regs[rs], SI.const(off & MASK))
            loaded, reason = av_load(addr, ro)
            if reason and why is None:
                why = reason
            regs[rt] = loaded
        elif m == "sw":
            # Not modelled. A store to an address I cannot pin could clobber any
            # word I later load, so the only sound thing is to forget read-only
            # memory entirely — and since I only ever load from .rodata here,
            # saying so out loud is better than silently keeping a stale map.
            if why is None:
                why = "a store was seen; writable memory is not modelled"
            ro = {}
        elif m in ("nop", "bne", "srl", "subu", "andi", "ori"):
            if m in ("srl", "subu", "andi", "ori"):
                regs[o[0]] = SI.TOP()                  # not modelled; be honest
        pc += 4
    return SI.TOP(), facts, why


# --- the two programs ---------------------------------------------------------

#   sll   $t1, $a0, 4          index * 16
#   addiu $t2, $zero, 0x20     handler base
#   addu  $t3, $t2, $t1
#   jr    $t3
UNGUARDED = [
    sll("t1", "a0", 4),
    addiu("t2", "zero", 0x20),
    addu("t3", "t2", "t1"),
    jr("t3"),
]

#   addiu $t0, $zero, 4        the limit
#   slt   $t1, $a0, $t0        t1 = (a0 < 4)
#   beq   $t1, $zero, +4       if NOT less, branch away to a default
#   nop                        (delay slot)
#   sll   $t1, $a0, 4
#   addiu $t2, $zero, 0x20
#   addu  $t3, $t2, $t1
#   jr    $t3
GUARDED = [
    addiu("t0", "zero", 4),
    slt("t1", "a0", "t0"),
    beq("t1", "zero", 4),
    nop(),
    sll("t1", "a0", 4),
    addiu("t2", "zero", 0x20),
    addu("t3", "t2", "t1"),
    jr("t3"),
]

TRUE_HANDLERS = [0x20, 0x30, 0x40, 0x50]


# --- a realistic dispatch: the table is READ, not computed --------------------
#
# Everything above computes its target arithmetically (BASE + index*16), which is
# a compiler pattern but not the common one. A real switch emits a table of
# addresses in .rodata and loads from it:
#
#     addiu $t4, $zero, 4          limit
#     slt   $t0, $a0, $t4          index < limit ?
#     beq   $t0, $zero, bail
#     nop                          (delay slot)
#     sll   $t1, $a0, 2            index * 4   (word-sized entries)
#     addiu $t2, $zero, 0x40       table base
#     addu  $t2, $t2, $t1
#     lw    $t3, 0($t2)            <-- the load this session added
#     jr    $t3
#
# The handler addresses are deliberately NOT strided. That is the point: a
# strided interval cannot hold them, so if the analysis reports them exactly it
# is because the load changed domains.

TABLE_BASE = 0x40
TABLE = [0x400120, 0x4001a8, 0x400214, 0x400188]
RO = {TABLE_BASE + 4 * i: v for i, v in enumerate(TABLE)}

LOADED = [
    addiu("$t4", "$zero", len(TABLE)),
    slt("$t0", "$a0", "$t4"),
    beq("$t0", "$zero", 8),
    0,
    sll("$t1", "$a0", 2),
    addiu("$t2", "$zero", TABLE_BASE),
    addu("$t2", "$t2", "$t1"),
    lw("$t3", 0, "$t2"),
    jr("$t3"),
]


def report(name, words, verbose=False, ro=None, expect=None):
    si, facts, why = analyse(words, verbose=verbose, ro=ro)
    vals = si.values()
    print(f"\n  {name}")
    print(f"    jr register  : {si}")
    if vals is None:
        print("    targets      : cannot bound — the analysis gives up, correctly")
    else:
        print(f"    targets      : {[hex(v) for v in vals]}  ({len(vals)})")
        truth = expect if expect is not None else TRUE_HANDLERS
        missing = [h for h in truth if h not in vals]
        extra = [v for v in vals if v not in truth]
        print(f"    sound?       : {'YES — every real handler is inside' if not missing else 'NO, MISSED ' + str(missing)}")
        print(f"    precise?     : {'exact' if not extra else f'{len(extra)} spurious'}")
    if why:
        print(f"    memory       : gave up — {why}")
    return vals


def selftest():
    # --- memory cases, added 2026-09-28 --------------------------------------
    v = analyse(LOADED, ro=RO)[0].values()
    assert v == sorted(TABLE), f"loaded dispatch should recover the table exactly, got {v}"
    # No memory map: must give up rather than invent a bound.
    assert analyse(LOADED, ro=None)[0].top, "no memory map must mean TOP"
    # One entry past the end of the table: an off-by-one bounds check must NOT be
    # silently absorbed. This is the case I most wanted to fail loudly.
    short = {k: x for k, x in RO.items() if k != TABLE_BASE + 12}
    assert analyse(LOADED, ro=short)[0].top, "a table hole must poison the load"
    # A store anywhere forgets writable memory and says so.
    from mips import sw as _sw
    with_store = LOADED[:7] + [_sw("$t0", 0, "$t2")] + LOADED[7:]
    r, _f, why = analyse(with_store, ro=RO)
    assert r.top and why and "store" in why, f"a store must invalidate the map, got {r} / {why}"
    # And the justification for having two domains at all: SI cannot hold this.
    from functools import reduce as _reduce
    assert _reduce(si_join, [SI.const(x) for x in TABLE]).top, \
        "if si_join could hold arbitrary handlers, VS would be unnecessary"


    """Four cases. The analysis is allowed to be imprecise and is not allowed
    to be unsound, so every check below is about the DIRECTION of the error."""
    from mips import bne
    ok = True
    print("  absint selftest\n")

    # 1. over-approximation: guard permits 8, only 4 handlers exist
    wide = [addiu("t0", "zero", 8), slt("t1", "a0", "t0"), beq("t1", "zero", 4),
            nop(), sll("t1", "a0", 4), addiu("t2", "zero", 0x20),
            addu("t3", "t2", "t1"), jr("t3")]
    v = analyse(wide)[0].values()
    missed = [h for h in TRUE_HANDLERS if h not in v]
    spur = [x for x in v if x not in TRUE_HANDLERS]
    good = not missed and len(spur) == 4
    ok &= good
    print(f"    [{'ok' if good else 'FAIL'}] superset when the guard is loose: "
          f"{len(v)} targets, {len(missed)} missed, {len(spur)} spurious")

    # 2. RED: break the shift and confirm the soundness check actually fires.
    #    A check that has never reported a miss has not been shown to be a check.
    global si_shl
    orig, si_shl = si_shl, (lambda a, k: orig_shl(a, max(0, k - 1)))
    v2 = analyse(GUARDED)[0].values()
    si_shl = orig
    caught = [h for h in TRUE_HANDLERS if h not in v2]
    ok &= bool(caught)
    print(f"    [{'ok' if caught else 'FAIL'}] with a deliberately wrong stride it "
          f"reports MISSED {[hex(m) for m in caught]}")

    # 3. no bounds check in the program -> no bound in the answer
    top = analyse(UNGUARDED)[0].top
    ok &= top
    print(f"    [{'ok' if top else 'FAIL'}] unguarded table stays TOP rather than "
          f"inventing a bound")

    # 4. a branch polarity I never implemented must degrade to TOP, not to a
    #    confident wrong answer
    g = [addiu("t0", "zero", 4), slt("t1", "a0", "t0"), bne("t1", "zero", 1),
         nop(), sll("t1", "a0", 4), addiu("t2", "zero", 0x20),
         addu("t3", "t2", "t1"), jr("t3")]
    t4 = analyse(g)[0].top
    ok &= t4
    print(f"    [{'ok' if t4 else 'FAIL'}] unmodelled `bne` guard degrades to TOP, "
          f"not to a bound it cannot justify")

    print(f"\n  {'all 9 pass (4 arithmetic + 5 memory)' if ok else '*** FAILURES ***'} — imprecise is fine, "
          f"unsound is not.")
    return 0 if ok else 1


orig_shl = si_shl

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    v = "-v" in sys.argv
    print("  Bounding a jump table from ABOVE, by abstract interpretation.")
    report("UNGUARDED — no bounds check in the program", UNGUARDED, v)
    report("GUARDED — the same table behind `slt`/`beq`", GUARDED, v)

    print("\n  The point of having both methods:")
    from discover import TABLE4, trace
    lower = trace(TABLE4, [0, 1, 2])
    upper = report("(upper bound, guarded)", GUARDED)
    print(f"\n    tracing inputs [0,1,2] (lower bound): {[hex(t) for t in lower]}")
    print(f"    abstract interpretation (upper bound): {[hex(t) for t in upper]}")
    if upper and set(lower) != set(upper):
        gap = [hex(t) for t in upper if t not in lower]
        print(f"    DISAGREEMENT -> {gap} is reachable and was never traced.")
        print("    That is the signal. One method alone reports nothing here.")
    full = trace(TABLE4, [0, 1, 2, 3])
    print(f"\n    tracing inputs [0,1,2,3]:             {[hex(t) for t in full]}")
    if upper and set(full) == set(upper):
        print("    bounds MEET — every traced target is permitted, every")
        print("    permitted target was traced. As close to a proof as this gets.")
