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

from mips import addiu, addu, beq, jr, nop, sll, slt
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

def analyse(words, base=0, verbose=False):
    """Walk the instruction stream, tracking an SI per register, and report the
    set of targets `jr` could reach.

    Straight-line plus forward branches only — enough for a dispatch stub, and
    I am not pretending it is a general CFG fixpoint. See the limits in the
    studio log.
    """
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
            return regs[o[0]], facts
        elif m == "sll":
            regs[o[0]] = si_shl(regs[o[1]], o[2])
        elif m == "addu":
            regs[o[0]] = si_add(regs[o[1]], regs[o[2]])
        elif m == "addiu":
            regs[o[0]] = si_add(regs[o[1]], SI.const(o[2] & MASK))
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
        elif m in ("nop", "bne", "srl", "subu", "andi", "ori"):
            if m in ("srl", "subu", "andi", "ori"):
                regs[o[0]] = SI.TOP()                  # not modelled; be honest
        pc += 4
    return SI.TOP(), facts


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


def report(name, words, verbose=False):
    si, facts = analyse(words, verbose=verbose)
    vals = si.values()
    print(f"\n  {name}")
    print(f"    jr register  : {si}")
    if vals is None:
        print("    targets      : cannot bound — the analysis gives up, correctly")
    else:
        print(f"    targets      : {[hex(v) for v in vals]}  ({len(vals)})")
        missing = [h for h in TRUE_HANDLERS if h not in vals]
        extra = [v for v in vals if v not in TRUE_HANDLERS]
        print(f"    sound?       : {'YES — every real handler is inside' if not missing else 'NO, MISSED ' + str(missing)}")
        print(f"    precise?     : {'exact' if not extra else f'{len(extra)} spurious'}")
    return vals


def selftest():
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

    print(f"\n  {'all 4 pass' if ok else '*** FAILURES ***'} — imprecise is fine, "
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
