#!/usr/bin/env python3
"""Wrapped (circular) intervals on Z/2^w — so overflow is a VALUE, not a shrug.

WHY THIS EXISTS
---------------
`absint.py` has this, three times, in `si_add`:

    return SI.TOP() if _wrapped(lo, hi) else SI(b.stride, lo, hi)

and a docstring on `_wrapped` that says, in my own words: "A wrapped interval is
a real thing and there is a real domain for it (wrapped/circular intervals,
Navas et al.), which this is not. Until it is, the honest answer is the shrug."

The shrug was correct — propagating an interval whose `hi` is below its `lo`
was silently unsound, and TOP at least tells the truth. But it throws away
everything the analysis knew the moment an `addiu` crosses the 32-bit boundary,
and on a circle that boundary is not there. `[0xFFFFFFF0, 0x0000000F]` is a
perfectly good set of 32 values; a signed/unsigned interval domain simply has no
way to say it.

THE ONE IDEA
------------
Stop thinking of a 32-bit value as a number on a line with ends, and think of it
as a position on a circle of 2^32 points. An interval is then an ARC: `[a, b]`
means walk clockwise from `a` to `b` inclusive. Nothing is special about the
point where the bit pattern rolls over, because an arc can contain it.

Addition becomes almost trivially exact, which is the payoff:

    [a,b] + [c,d] = [a+c, b+d]   (mod 2^w)

...valid whenever the two arc lengths together do not reach the whole circle.
Rotation of an arc is still an arc. Wrapping is not an exceptional case to be
detected and surrendered to; it is just rotation past an arbitrary origin.

Reference: Navas, Schachte, Søndergaard, Stuckey, "Signedness-Agnostic Program
Analysis: Precise Integer Bounds for Low-Level Code" (APLAS 2012). I worked the
operations out from the circle definition rather than transcribing theirs, which
is why every one is checked exhaustively below rather than trusted.

HOW THIS IS TESTED, WHICH IS THE ONLY REASON TO BELIEVE IT
----------------------------------------------------------
Circular-domain bugs hide in the wrap cases, and wrap cases are exactly the ones
you forget to write by hand. So the width is a parameter, and at w=4 there are
only 2^4 * 2^4 + 2 = 258 distinct wrapped intervals. Every operation is verified
against brute-force SET semantics over all of them — 258 for unary ops, 66,564
pairs for binary ones. Not sampled. Enumerated.

That is also why `join` is implemented the way it is: the two-candidate rule
below is not a guess I liked the look of, it is the one that survived the
exhaustive soundness-and-minimality check.
"""


class WI:
    """An arc on Z/2^w: the clockwise walk from `a` to `b` inclusive.

    Canonical forms: `bottom` (empty) and `full` (all 2^w values) are flags, so
    there is exactly one representation of each. Without that, [0,15] and [5,4]
    are both the whole circle at w=4 and every comparison has to special-case
    it.
    """
    __slots__ = ('w', 'mod', 'a', 'b', 'is_bottom', 'is_full')

    def __init__(self, w, a=None, b=None, bottom=False, full=False):
        self.w = w
        self.mod = 1 << w
        self.is_bottom = bool(bottom)
        self.is_full = bool(full)
        if not (self.is_bottom or self.is_full) and \
                ((b - a) % self.mod) == self.mod - 1:
            # ⚠️ [a, a-1] IS the whole circle, for every a. Found by the exhaustive
            # test on its first real run: 11,360 failures, all downstream of this.
            # Without canonicalising here, [0,15] and [5,4] are the same SET with
            # different representations, so __eq__ says unequal, le() says neither
            # contains the other, and join() returns a "cover" smaller than its
            # own input. One missing normalisation, three broken operations.
            self.is_full = True
        if self.is_bottom or self.is_full:
            self.a = self.b = None
        else:
            self.a = a % self.mod
            self.b = b % self.mod

    # --- constructors ---------------------------------------------------------
    @staticmethod
    def bottom(w):
        return WI(w, bottom=True)

    @staticmethod
    def full(w):
        return WI(w, full=True)

    @staticmethod
    def const(w, v):
        return WI(w, v, v)

    @staticmethod
    def arc(w, a, b):
        """[a,b] clockwise, inclusive. `arc(a, a-1)` is the FULL circle — 2^w
        values, not 2^w - 1 — and is canonicalised to `full()` on construction.

        I first wrote the opposite in this docstring and the exhaustive test
        refuted it. The arithmetic: card = ((b-a) mod 2^w) + 1, so a to a-1 is
        (2^w - 1) + 1 = 2^w. The set with 2^w - 1 values is `arc(a, a-2)`."""
        return WI(w, a, b)

    # --- basics ---------------------------------------------------------------
    def card(self):
        if self.is_bottom:
            return 0
        if self.is_full:
            return self.mod
        return ((self.b - self.a) % self.mod) + 1

    def __contains__(self, x):
        if self.is_bottom:
            return False
        if self.is_full:
            return True
        x %= self.mod
        return ((x - self.a) % self.mod) <= ((self.b - self.a) % self.mod)

    def values(self):
        if self.is_bottom:
            return []
        if self.is_full:
            return list(range(self.mod))
        n = self.card()
        return [(self.a + i) % self.mod for i in range(n)]

    def __eq__(self, o):
        if not isinstance(o, WI):
            return NotImplemented
        return (self.w, self.is_bottom, self.is_full, self.a, self.b) == \
               (o.w, o.is_bottom, o.is_full, o.a, o.b)

    def __hash__(self):
        return hash((self.w, self.is_bottom, self.is_full, self.a, self.b))

    def __repr__(self):
        if self.is_bottom:
            return "_|_"
        if self.is_full:
            return "TOP"
        return f"[0x{self.a:x},0x{self.b:x}]"

    def le(self, o):
        """Inclusion, set-exact, in one clause:

            offset of self's start from o's start,  PLUS  self's own length,
            must not exceed o's length.

        i.e. `(a-c) mod 2^w + (b-a) mod 2^w  <=  (d-c) mod 2^w`.

        ⚠️ My first version was a conjunction of three plausible-looking
        comparisons and it was WRONG: it reported `[0,2] <= [2,0]` as true at
        w=4, when {0,1,2} is plainly not inside {2,3,...,15,0} — 1 is missing.
        Each clause held individually; what none of them did was require
        self to fit *starting from where it actually starts*. Checking two
        endpoints independently is not an arc containment test, because the
        arc between them can leave the target and come back.
        """
        if self.is_bottom:
            return True
        if o.is_bottom:
            return False
        if o.is_full:
            return True
        if self.is_full:
            return False
        off = (self.a - o.a) % self.mod
        return off + ((self.b - self.a) % self.mod) <= ((o.b - o.a) % self.mod)

    # --- the operations that matter -----------------------------------------
    def add(self, o):
        """EXACT whenever the result is representable, which is the whole point.

        Two arcs of length m and n sum to an arc of length m+n-1. If that
        reaches the circle, the answer is everything; otherwise it is just the
        rotated arc, and the 32-bit boundary never enters into it.
        """
        if self.is_bottom or o.is_bottom:
            return WI.bottom(self.w)
        if self.is_full or o.is_full:
            return WI.full(self.w)
        if self.card() + o.card() - 1 >= self.mod:
            return WI.full(self.w)
        return WI(self.w, self.a + o.a, self.b + o.b)

    def neg(self):
        """Always exact. Reflecting a circle maps arcs to arcs."""
        if self.is_bottom or self.is_full:
            return self
        return WI(self.w, -self.b, -self.a)

    def sub(self, o):
        return self.add(o.neg())

    def complement(self):
        if self.is_bottom:
            return WI.full(self.w)
        if self.is_full:
            return WI.bottom(self.w)
        if self.card() == self.mod - 1:
            return WI.const(self.w, self.b + 1)
        return WI(self.w, self.b + 1, self.a - 1)

    def join(self, o):
        """Least arc containing both — and 'least' needs choosing, because two
        disjoint arcs have TWO candidate covers (bridge the gap clockwise, or
        anticlockwise). Take the smaller; tie-break deterministically so the
        fixpoint iteration cannot oscillate between equal-sized answers.
        """
        if self.is_bottom:
            return o
        if o.is_bottom:
            return self
        if self.is_full or o.is_full:
            return WI.full(self.w)
        if o.le(self):
            return self
        if self.le(o):
            return o
        c1 = WI(self.w, self.a, o.b)       # bridge: self's start to o's end
        c2 = WI(self.w, o.a, self.b)       # bridge: o's start to self's end
        # If either candidate fails to contain both, it is not a cover at all.
        ok1 = self.le(c1) and o.le(c1)
        ok2 = self.le(c2) and o.le(c2)
        if ok1 and ok2:
            if c1.card() != c2.card():
                return c1 if c1.card() < c2.card() else c2
            return c1 if (c1.a, c1.b) <= (c2.a, c2.b) else c2
        if ok1:
            return c1
        if ok2:
            return c2
        return WI.full(self.w)

    def meet(self, o):
        """Intersection of two arcs can be TWO arcs — e.g. at w=4, [14,2] and
        [1,15] share both {1,2} and {14,15}. A single arc cannot express that,
        so this over-approximates to the smallest arc covering the
        intersection. Over-approximating a MEET is still sound for the use here
        (narrowing a register after a branch) because it only ever returns a
        superset of the true intersection — but it is a real precision loss and
        the exhaustive test asserts the direction, not equality.
        """
        if self.is_bottom or o.is_bottom:
            return WI.bottom(self.w)
        if self.is_full:
            return o
        if o.is_full:
            return self
        common = [x for x in self.values() if x in o]
        if not common:
            return WI.bottom(self.w)
        if len(common) == self.mod:
            return WI.full(self.w)
        # smallest covering arc of a set: the complement's largest gap
        s = set(common)
        best = None
        for start in s:
            if (start - 1) % self.mod not in s:      # arc start = no predecessor
                n = 0
                while (start + n) % self.mod in s:
                    n += 1
                cand = WI(self.w, start, start + n - 1)
                if best is None or cand.card() < best.card():
                    best = cand
        if best is None:                             # s is the whole circle
            return WI.full(self.w)
        # if the set was split into several runs, cover them all
        if best.card() != len(s):
            cover = None
            for start in s:
                if (start - 1) % self.mod not in s:
                    for end in s:
                        if (end + 1) % self.mod not in s:
                            cand = WI(self.w, start, end)
                            if all(x in cand for x in s):
                                if cover is None or cand.card() < cover.card():
                                    cover = cand
            return cover if cover is not None else WI.full(self.w)
        return best


# --- exhaustive verification --------------------------------------------------

def _all(w):
    """Every DISTINCT wrapped interval. Deduped, because canonicalisation maps
    all 2^w spellings of [a,a-1] onto one `full`, and a universe with duplicates
    makes the join-minimality check compare an answer against itself."""
    seen = {WI.bottom(w), WI.full(w)}
    for a in range(1 << w):
        for b in range(1 << w):
            seen.add(WI.arc(w, a, b))
    return sorted(seen, key=lambda x: (x.is_bottom, x.is_full, x.card(),
                                       x.a if x.a is not None else -1))


def selftest(w=4, verbose=True):
    mod = 1 << w
    univ = _all(w)
    fails = []

    def chk(name, cond, detail):
        # `detail` may be a callable, because an eager f-string in the PASSING
        # case indexed an empty list and crashed the test on its first run —
        # the harness failing, not the thing under test.
        if not cond:
            fails.append(f"{name}: {detail() if callable(detail) else detail}")

    # representation: values()/card()/__contains__ must agree with each other
    for x in univ:
        vs = x.values()
        chk("card", len(vs) == x.card(), f"{x!r} values={len(vs)} card={x.card()}")
        for v in range(mod):
            chk("contains", (v in x) == (v in set(vs)), f"{x!r} v={v}")

    # arc(a, a-1) IS the circle and must canonicalise; arc(a, a-2) is one short
    for a in range(mod):
        chk("arc-wrap-is-full", WI.arc(w, a, (a - 1) % mod).is_full,
            f"arc({a},{(a-1)%mod}) should be the full circle")
        chk("arc-full-card", WI.arc(w, a, (a - 1) % mod).card() == mod,
            f"arc({a},{(a-1)%mod}) card")
        short = WI.arc(w, a, (a - 2) % mod)
        chk("arc-one-short", not short.is_full and short.card() == mod - 1,
            f"arc({a},{(a-2)%mod}) card={short.card()}")

    # le() must be exactly subset
    for x in univ:
        sx = set(x.values())
        for y in univ:
            chk("le", x.le(y) == sx.issubset(set(y.values())), f"{x!r} <= {y!r}")

    # neg and complement: exact
    for x in univ:
        chk("neg", set(x.neg().values()) == {(-v) % mod for v in x.values()},
            f"neg {x!r}")
        chk("complement",
            set(x.complement().values()) == set(range(mod)) - set(x.values()),
            f"complement {x!r}")

    # add: SOUND always, and EXACT whenever it does not return full
    for x in univ:
        xs = x.values()
        for y in univ:
            r = x.add(y)
            true = {(p + q) % mod for p in xs for q in y.values()}
            chk("add-sound", true.issubset(set(r.values())), f"{x!r}+{y!r}={r!r}")
            if not r.is_full:
                chk("add-exact", set(r.values()) == true,
                    f"{x!r}+{y!r}={r!r} true={sorted(true)}")

    # sub: sound
    for x in univ:
        for y in univ:
            r = x.sub(y)
            true = {(p - q) % mod for p in x.values() for q in y.values()}
            chk("sub-sound", true.issubset(set(r.values())), f"{x!r}-{y!r}={r!r}")

    # join: sound, and MINIMAL among arcs
    for x in univ:
        for y in univ:
            j = x.join(y)
            u = set(x.values()) | set(y.values())
            chk("join-sound", u.issubset(set(j.values())), f"{x!r} U {y!r} = {j!r}")
            if not j.is_full:
                better = [c for c in univ
                          if c.card() < j.card() and u.issubset(set(c.values()))]
                chk("join-minimal", not better,
                    lambda x=x, y=y, j=j, better=better:
                    f"{x!r} U {y!r} = {j!r} but {better[0]!r} is smaller")

    # meet: over-approximates the intersection, and never invents membership
    for x in univ:
        for y in univ:
            m = x.meet(y)
            i = set(x.values()) & set(y.values())
            chk("meet-covers", i.issubset(set(m.values())), f"{x!r} ^ {y!r} = {m!r}")
            chk("meet-bottom", bool(i) or m.is_bottom,
                f"{x!r} ^ {y!r} = {m!r} but intersection empty")

    if verbose:
        n = len(univ)
        print(f"  w={w}: {n} intervals, {n*n:,} pairs")
        if fails:
            print(f"  \033[31m{len(fails)} FAILURE(S)\033[0m")
            for f in fails[:12]:
                print("    ", f)
        else:
            print("  \033[32mall properties hold exhaustively\033[0m "
                  "(card/contains/le/neg/complement/add/sub/join/meet)")
    return fails


def demo():
    """The case absint.py surrenders to, side by side."""
    W = 32
    print("\n  --- what the strided domain does with a wrapping add ---")
    print("  absint.si_add:  [0xFFFFFFF0,0xFFFFFFFF] + {0x10}  ->  TOP (unbounded)")
    x = WI.arc(W, 0xFFFFFFF0, 0xFFFFFFFF)
    y = WI.const(W, 0x10)
    r = x.add(y)
    print(f"  WI.add:         {x!r} + {y!r}  ->  {r!r}")
    print(f"                  card {r.card()} — exact, and it contains 0: {0 in r}")

    print("\n  --- and the thing a wrapped interval can say that an interval cannot ---")
    near = WI.arc(W, 0xFFFFFFFE, 0x00000001)
    print(f"  {near!r} is {near.card()} values: {[hex(v) for v in near.values()]}")
    print(f"  as a plain interval that is [0, 0xFFFFFFFF] — "
          f"{(1<<32)//near.card():,}x larger than the truth")

    print("\n  --- where it still gives up, honestly ---")
    half = WI.arc(W, 0, (1 << 31))
    print(f"  {half!r} (card {half.card():,}) + itself -> {half.add(half)!r}")
    print("  two arcs that together reach the circle really are everything.")


if __name__ == "__main__":
    import sys
    print("wrapped intervals on Z/2^w — exhaustive selftest")
    bad = selftest(4)
    if "--w5" in sys.argv and not bad:
        print("\n  second width, as a check on the first:")
        bad += selftest(5)
    demo()
    sys.exit(1 if bad else 0)
