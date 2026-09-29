#!/usr/bin/env python3
"""A CFG fixpoint with widening — the part absint.py has been refusing to do.

absint.py walks a straight line. It has a 4000-step cap and treats `bne` as a
no-op, which is fine for a dispatch stub and useless for a function: a loop
either spins to the cap or is silently not a loop. The queue has said "no CFG
fixpoint, no loops, **no widening** (the hardest part, untouched)" since
2026-09-24.

The hard part is not the worklist. It is this:

    A loop's register does not converge. `$t0 = 0, 4, 8, 12, ...` produces a
    new abstract value on every iteration, so a naive fixpoint never reaches
    one. Widening is the operator that jumps deliberately to an over-
    approximation to force termination, accepting a loss of precision to get an
    answer at all.

**And the design question that made this worth an hour:** what does widening do
to a STRIDE?

Classic interval widening (Cousot & Cousot) sends a growing bound to infinity:
`[0,0] ▽ [0,4] = [0,+inf)`. Do that to a strided interval and you have thrown
away the stride — which is the entire reason this analysis uses strided
intervals, and the finding from 2026-09-24 that the domain having the same
shape as the analysed thing is most of the art. `[0,+inf)` says "any number".
`4[0,+inf)` says "a multiple of four", which is a vastly stronger statement and
costs nothing to keep, because **the stride is not what is growing.**

So the operator here widens the BOUNDS and preserves the STRIDE whenever the
stride is stable between iterations. Only when the stride itself is unstable
does it give up to TOP.

Two consequences worth stating because I did not predict either:

  1. **No infinity is needed.** These are 32-bit registers, so "unbounded above"
     is just the largest value below 2^32 that is congruent to `lo` mod stride.
     The ceiling is a real number and the domain stays closed.
  2. **The loop guard gives the precision back.** Widening throws away the upper
     bound; the `slt`/branch refinement that absint.py already had since
     2026-09-24 pushes it straight back on the loop-exit edge. The two
     mechanisms are complementary and I built them three days apart without
     noticing they would meet here.
"""
import sys
from collections import defaultdict

from absint import (SI, MASK, si_join, si_below, av_add, av_shl, av_load,
                    is_top)
from mips import disasm, addiu, slt, bne, beq, sll, addu, jr, nop


# --- widening -----------------------------------------------------------------

def ceil_congruent(lo, stride):
    """Largest 32-bit value congruent to lo (mod stride).

    This is what stands in for +infinity. Registers are 32 bits, so the lattice
    has a real top element and widening can reach it without leaving the domain.
    """
    return lo + ((MASK - lo) // stride) * stride


def thresholds(words, base=0):
    """Constants the PROGRAM itself mentions — candidate widening stops.

    Discovered the need for this by hitting the wall, which is the good way.
    Widening `hi` to `ceil_congruent` is sound and, in 32-bit arithmetic,
    immediately self-defeating: the ceiling is precisely where the next `addiu`
    WRAPS. Widen `4[0,·]` to `4[0,0xfffffffc]`, add 4, and you get an interval
    whose `hi` (0) is below its `lo` (4) — a malformed value that then made the
    loop-exit edge look infeasible and returned None for the whole analysis.

    "Unbounded above" and "the largest representable value" are the same number
    in a fixed-width machine, and only one of them survives an increment.

    The standard repair is widening with thresholds (Cousot; also how most
    production interval analyses do it): instead of jumping to the top of the
    lattice, jump to the smallest interesting constant above the current bound.
    The interesting constants are sitting in the instruction stream — this loop
    literally contains `addiu $t2, $zero, 40`, and 40 is the answer.
    """
    out = set()
    for i, w in enumerate(words):
        m, o, tgt = disasm(w, base + i * 4)
        if m in ("addiu", "andi", "ori") and o[1] == 0:
            out.add(o[2] & MASK)                       # a materialised constant
    return sorted(out)


def si_widen(a, b, stops=()):
    """a ▽ b — the previous entry state widened by the newly joined one.

    Must satisfy: result ⊒ a and result ⊒ b (soundness), and any ascending
    chain must stabilise in finitely many steps (termination). It does, because
    each application either keeps the value, or pushes a bound to a fixed
    32-bit extreme, or goes to TOP — and none of those can be undone.
    """
    if a is None:
        return b
    if b is None:
        return a
    if a.top or b.top:
        return SI.TOP()
    if a == b:
        return a

    # both constants, different: a stride is being born. 0 then 4 means "so far
    # {0,4}" — adopt the step as a stride rather than calling it an interval.
    if a.stride == 0 and b.stride == 0:
        step = abs(b.lo - a.lo)
        lo = min(a.lo, b.lo)
        return SI(step, lo, ceil_congruent(lo, step))

    # the stable-stride case, which is the one that matters
    if a.stride == b.stride and a.stride != 0 and (b.lo - a.lo) % a.stride == 0:
        lo = a.lo if b.lo >= a.lo else 0               # growing downward -> 0
        hi = a.hi if b.hi <= a.hi else _next_stop(b.hi, lo, a.stride, stops)
        return SI(a.stride, lo, hi)

    # a constant that has acquired a stride
    if a.stride == 0 and b.stride != 0 and (a.lo - b.lo) % b.stride == 0:
        lo = min(a.lo, b.lo)
        return SI(b.stride, lo, ceil_congruent(lo, b.stride))

    # the stride itself is moving. Nothing in this domain describes that, and
    # inventing something would be the guessing this analysis exists to refuse.
    return SI.TOP()


def si_atleast(a, limit):
    """Refine `a` with the fact a >= limit — the loop-EXIT direction.

    absint.py had only `si_below`, because a dispatch stub only ever needs the
    guarded side: you learn `index < 4` and fall into the table. A loop needs
    the other half. The exit edge of `while (i < 40)` is precisely the case
    where the comparison FAILED, and until this existed the analysis walked out
    of every loop knowing nothing.

    Snapping the low end UP to the progression is the whole value: `4[0,..] >= 40`
    is not `[40,..]`, it is `4[40,..]` — still a multiple of four. That is the
    same argument as the widening above, applied to the opposite bound.
    """
    if a.top:
        return a
    if a.stride == 0:
        return a if a.lo >= limit else None            # None = infeasible edge
    if a.hi < limit:
        return None
    lo = a.lo if a.lo >= limit else a.lo + -(-(limit - a.lo) // a.stride) * a.stride
    return SI(a.stride, lo, a.hi) if lo <= a.hi else None


def _next_stop(cur_hi, lo, stride, stops):
    """Smallest threshold at or above cur_hi, snapped onto the progression."""
    for t in stops:
        if t >= cur_hi:
            return lo + ((t - lo) // stride) * stride if t >= lo else lo
    return ceil_congruent(lo, stride)                  # nothing fits: the top


def state_widen(old, new, stops=()):
    return {r: si_widen(old.get(r), new.get(r), stops) for r in new}


def state_join(old, new):
    if old is None:
        return dict(new)
    return {r: si_join(old.get(r), new.get(r)) for r in new}


# --- the control-flow graph ---------------------------------------------------

def leaders(words, base):
    """Addresses that begin a basic block."""
    out = {base}
    for i, w in enumerate(words):
        pc = base + i * 4
        m, o, tgt = disasm(w, pc)
        if m in ("beq", "bne"):
            if tgt is not None:
                out.add(tgt)                            # the branch target
            out.add(pc + 8)                             # after the delay slot
        elif m == "jr":
            out.add(pc + 8)
    return sorted(x for x in out if base <= x < base + len(words) * 4)


def build_cfg(words, base):
    """{leader: [instruction pcs]} plus the successor map."""
    ls = leaders(words, base)
    blocks, succ = {}, {}
    for n, start in enumerate(ls):
        end = ls[n + 1] if n + 1 < len(ls) else base + len(words) * 4
        blocks[start] = list(range(start, end, 4))
        succ[start] = []
        # the successor of a block is decided by its LAST real instruction
        last = None
        for pc in blocks[start]:
            m, o, tgt = disasm(words[(pc - base) // 4], pc)
            if m in ("beq", "bne", "jr"):
                last = (pc, m, o, tgt)
                break
        if last is None:
            if end < base + len(words) * 4:
                succ[start] = [(end, None)]
        else:
            pc, m, o, tgt = last
            if m == "jr":
                succ[start] = []                        # leaves the graph
            else:
                # (target, edge-kind). The edge kind decides which way the
                # comparison fact is pushed.
                succ[start] = [(tgt, "taken"), (pc + 8, "fall")]
    return blocks, succ


# --- the transfer function ----------------------------------------------------

def run_block(words, base, pcs, regs_in, ro):
    """Interpret one block. Returns (regs_out, facts, branch, why)."""
    regs = dict(regs_in)
    facts, why, branch = {}, None, None
    for pc in pcs:
        idx = (pc - base) // 4
        if idx >= len(words):
            break
        m, o, tgt = disasm(words[idx], pc)
        if m == "sll":
            regs[o[0]] = av_shl(regs[o[1]], o[2])
        elif m == "addu":
            regs[o[0]] = av_add(regs[o[1]], regs[o[2]])
        elif m == "addiu":
            regs[o[0]] = av_add(regs[o[1]], SI.const(o[2] & MASK))
        elif m == "slt":
            lim = regs[o[2]]
            if not lim.top and lim.stride == 0:
                facts[o[0]] = ("lt", o[1], lim.lo)
            regs[o[0]] = SI(1, 0, 1)
        elif m == "lw":
            rt, off, rs = o
            loaded, reason = av_load(av_add(regs[rs], SI.const(off & MASK)), ro)
            if reason and why is None:
                why = reason
            regs[rt] = loaded
        elif m == "sw":
            why = why or "a store was seen; writable memory is not modelled"
            ro = {}
        elif m in ("beq", "bne"):
            branch = (m, o, tgt)
            break
        elif m == "jr":
            branch = ("jr", o, None)
            break
        elif m in ("srl", "subu", "andi", "ori"):
            regs[o[0]] = SI.TOP()
        regs[0] = SI.const(0)
    return regs, facts, branch, why


def refine_for_edge(regs, facts, branch, kind):
    """Push a comparison's fact onto the compared register, per edge.

    `slt rd, rs, lim` then `bne rd, $zero, L` branches WHEN THE COMPARISON HELD,
    so the taken edge learns `rs < lim` and the fall-through learns nothing
    usable (an unsigned lower bound is not expressible here, so I decline to
    invent one). `beq rd, $zero, L` is the mirror image. Returns None for an
    edge the analysis can prove is not taken.
    """
    if branch is None:
        return dict(regs)
    m, o, _ = branch
    if m not in ("beq", "bne"):
        return dict(regs)
    rd, rt = o[0], o[1]
    if rt != 0 or rd not in facts:
        return dict(regs)
    held_on = "taken" if m == "bne" else "fall"
    _, src, lim = facts[rd]
    if kind == held_on:
        refined = si_below(regs[src], lim)             # the comparison held
    else:
        refined = si_atleast(regs[src], lim)           # ...and the exit edge
    if refined is None:
        return None                                    # infeasible edge
    out = dict(regs)
    out[src] = refined
    return out


# --- the fixpoint -------------------------------------------------------------

def analyse_cfg(words, base=0, ro=None, delay=2, trace=False, max_rounds=400):
    """Worklist fixpoint over basic blocks, widening on back edges.

    `delay` is how many times a block may be re-entered before widening kicks
    in. Widening immediately is sound and blunt; a couple of rounds of plain
    join first lets a small loop converge exactly, and costs nothing when it
    does not. This is standard practice and it is also the difference between
    `4[4,40]` and `4[0,4294967292]` on the test below.
    """
    ro = ro or {}
    stops = thresholds(words, base)
    blocks, succ = build_cfg(words, base)
    entry = {b: None for b in blocks}
    entry[base] = {i: SI.TOP() for i in range(32)}
    entry[base][0] = SI.const(0)

    visits = defaultdict(int)
    work = [base]
    rounds = 0
    targets, why_all = None, None

    while work and rounds < max_rounds:
        rounds += 1
        b = work.pop(0)
        if entry[b] is None:
            continue
        regs, facts, branch, why = run_block(words, base, blocks[b], entry[b], ro)
        if why and why_all is None:
            why_all = why
        if branch and branch[0] == "jr":
            t = regs[branch[1][0]]
            targets = t if targets is None else si_join(targets, t)
            continue
        for tgt, kind in succ.get(b, []):
            if tgt not in blocks:
                continue
            nxt = refine_for_edge(regs, facts, branch, kind)
            if nxt is None:
                if trace:
                    print(f"    edge {b:#x} -> {tgt:#x} ({kind}) infeasible")
                continue
            old = entry[tgt]
            merged = state_join(old, nxt)
            back = tgt <= b                            # a back edge
            visits[tgt] += 1
            if back and old is not None and visits[tgt] > delay:
                merged = state_widen(old, merged, stops)
                if trace:
                    print(f"    WIDEN at {tgt:#x} (visit {visits[tgt]})")
            if old is None or merged != old:
                entry[tgt] = merged
                work.append(tgt)
        if trace:
            print(f"  block {b:#x} done, worklist={[hex(x) for x in work]}")

    converged = rounds < max_rounds
    return dict(entry=entry, blocks=blocks, succ=succ, targets=targets,
                rounds=rounds, converged=converged, why=why_all)


# --- programs -----------------------------------------------------------------

#      addiu $t2, $zero, 40      the limit
#      addiu $t0, $zero, 0       i = 0
# loop:addiu $t0, $t0, 4         i += 4
#      slt   $t1, $t0, $t2       t1 = (i < 40)
#      bne   $t1, $zero, loop    branch back while it held
#      nop
#      jr    $t0                 <- what can i be here?
COUNTER_LOOP = [
    addiu("t2", "zero", 40),
    addiu("t0", "zero", 0),
    addiu("t0", "t0", 4),
    slt("t1", "t0", "t2"),
    bne("t1", "zero", -3),
    nop(),
    jr("t0"),
]


def selftest():
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else '**FAIL**'}  {name}"
              + (f"   {detail}" if detail else ""))

    print("\nwidening operator")
    w = si_widen(SI(4, 0, 4), SI(4, 0, 8))
    check("a stable stride SURVIVES widening", w.stride == 4, f"stride {w.stride}")
    check("soundness: contains the old bound", w.hi >= 4)
    check("soundness: contains the new bound", w.hi >= 8)
    check("unstable stride degrades to TOP, not to a guess",
          si_widen(SI(4, 0, 4), SI(6, 0, 6)).top)
    check("two constants give birth to a stride",
          si_widen(SI.const(0), SI.const(4)).stride == 4)
    check("idempotent once widened", si_widen(w, si_join(w, SI(4, 0, 12))) == w)

    print("\nwidening with thresholds")
    stops = thresholds(COUNTER_LOOP, 0)
    check("thresholds are read out of the instruction stream", stops == [0, 40], f"{stops}")
    wt = si_widen(SI(4, 0, 4), SI(4, 0, 8), stops)
    check("widens to the program's constant, not the 32-bit ceiling",
          wt.hi == 40, f"hi={wt.hi}")
    wn = si_widen(SI(4, 0, 4), SI(4, 0, 8))
    check("with NO thresholds it goes to the ceiling", wn.hi == ceil_congruent(0, 4),
          f"hi={wn.hi:#x}")

    print("\nRED: the ceiling is where arithmetic WRAPS")
    # This is the failure that sent me to threshold widening. Widen to the top
    # of a 32-bit lattice and the very next increment overflows it.
    wrapped = av_add(SI(4, 0, ceil_congruent(0, 4)), SI.const(4))
    check("ceiling + stride produces hi < lo (or TOP), not a usable interval",
          wrapped.top or wrapped.hi < wrapped.lo,
          f"lo={wrapped.lo:#x} hi={wrapped.hi:#x} top={wrapped.top}")

    print("\nexit-edge refinement")
    check("a>=limit snaps the low end UP onto the progression",
          si_atleast(SI(4, 4, 400), 40) == SI(4, 40, 400),
          f"{si_atleast(SI(4,4,400),40)}")
    check("an unsatisfiable >= is an infeasible edge, not a wrong answer",
          si_atleast(SI(4, 4, 8), 40) is None)

    print("\nsi_join: the arm the fixpoint exposed")
    check("a constant joins the progression it sits on",
          si_join(SI.const(4), SI(4, 4, 40)) == SI(4, 4, 40),
          f"{si_join(SI.const(4), SI(4,4,40))}")

    print("\ncfg construction")
    blocks, succ = build_cfg(COUNTER_LOOP, 0)
    check("the loop body is its own block", 0x8 in blocks,
          f"leaders {[hex(b) for b in blocks]}")
    check("the branch block has two successors", len(succ[0x8]) == 2)
    check("a back edge exists", any(t <= 0x8 for t, _ in succ[0x8]))

    print("\nfixpoint on a counter loop")
    r = analyse_cfg(COUNTER_LOOP, 0)
    t = r["targets"]
    check("it terminates", r["converged"], f"{r['rounds']} rounds")
    check("the jr target is bounded, not TOP", t is not None and not t.top, f"{t}")
    if t is not None and not t.top:
        check("the stride survived the whole loop", t.stride == 4, f"stride {t.stride}")
        vals = t.values() or []
        # The true answer is exactly {40}: i steps 0,4,..,40 and exits when
        # 40 < 40 fails. The analysis says {40,44} — SOUND, and one value loose.
        check("SOUND: the real exit value is included", 40 in vals, f"{vals}")
        check("and it is an over-approximation, stated not hidden",
              vals == [40, 44], f"{vals}")
        check("two values instead of four billion", len(vals) == 2, f"{len(vals)}")

    print("\n" + ("  all cases pass." if ok else "  FAILURES ABOVE."))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    r = analyse_cfg(COUNTER_LOOP, 0, trace="--verbose" in sys.argv)
    print(f"\nrounds={r['rounds']} converged={r['converged']}")
    print(f"jr can reach: {r['targets']}")
