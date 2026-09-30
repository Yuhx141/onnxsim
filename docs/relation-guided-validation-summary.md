# Relation-guided optimizer validation

This branch records a focused evaluation of relation-guided validation as a
complement to onnxsim's existing whole-graph fuzzing.

## Scope and evidence

The broader research corpus combines traceable public ONNX models from Hugging
Face, targeted reduced cases, and NNSmith-generated models. The controlled
comparison reported here was narrower: it used 120 NNSmith seeds (0--119),
with seeds 0--59 used for the initial run and the disjoint range 60--119 used
as a held-out batch. Each generated model was evaluated in three forms: the
baseline, a variant that exposed one intermediate as a graph output, and a
variant that added an external consumer. This produced 360 ONNX-checker-valid
variants in total.

Of those 360 variants, 213 completed simplification and the runtime equivalence
check, 138 reached backend admission or execution errors during checking (for
example, unsupported operator/type combinations), and nine produced numerical
check failures. The nine failures came from three underlying NNSmith seeds,
each repeated across the baseline and both relation variants; disabling
constant folding restored those three cases. No observed failure occurred only
in a relation variant: every failed relation variant had the same failure in
its corresponding baseline. This is a result for this bounded batch, not a
general claim that the method has no false positives.

The wider investigation produced seven reduced issue candidates (#1995--#2001)
covering target validity, a native crash, graph lifetime and scope, and semantic
boundaries. It also produced two contract questions (#2002 and #2003) about
mask interpretation and formula-matching tolerance. ReferenceEvaluator and
ONNX Runtime differences were measured separately so backend limitations were
not reported as optimizer defects.

## Fork fixes

The branch now has fork-local fixes for the reported cases. These changes have
not been merged upstream.

| Report | Fork behavior |
| --- | --- |
| #1995 | reshape a singleton-rank Conv output scale to rank one before folding it into the bias |
| #1996 | decline the reshape-family fusion when `allowzero=1` would combine `0` and `-1` |
| #1997 | normalize the `Shape` range before reading input dimensions |
| #1998 | keep matched RoPE values that are also graph outputs |
| #1999 | decline constant-trip unrolling when the body has nested graph attributes |
| #2000 | decline Attention/GQA fusion when the source scale would become the target operator's zero sentinel |
| #2001 | decline normalization fusion when a DOUBLE epsilon is not exactly representable by the float attribute |
| #2002 | only treat negative infinity, rather than a finite penalty, as an exact hard causal mask |
| #2003 | match GELU and LayerNorm formula constants exactly in their source tensor precision |

The #1997 change is in
[`21d635f6`](https://github.com/Yuhx141/onnxsim/commit/21d635f601768047aa1ed97a27a199c88e7d7ced).
The remaining changes and their regression tests are in
[`fd4951d0`](https://github.com/Yuhx141/onnxsim/commit/fd4951d07e12cd2621111a647c8aa4ccb1ae049c).

The eight public reproducer directories for #1995, #1996, and #1998--#2003
were replayed against this build. Every optimized graph passed the ONNX checker
and executed. The numerical cases were exact after the affected fusion
declined. The RoPE case no longer aborts and preserves its public embedding;
its fused output differs from the decomposed output by at most
`1.1920929e-7`, the existing floating-point rounding of that valid rewrite.

Ten new regression tests and 41 neighboring existing tests pass on the regular
build. The ten new regressions also pass with the repository's
address/alignment sanitizer flags, with leak checking disabled to exclude the
known Python/NumPy shutdown allocation.

## Ongoing work

The relation-guided method is still being extended across model sources and
optimizer families. The original 120-seed comparison used a regular build; it
was not an AddressSanitizer or Valgrind run. After the comparison, the #1997
regression and 18 existing related tests were run locally with the repository's
address/alignment sanitizer instrumentation. The ten later issue regressions
were checked the same way. These focused checks produced no invalid-access or
alignment report; they are not sanitizer coverage of the full 120-seed batch.

For future experiments, sanitizer diagnostics are a useful additional signal
alongside graph validity, pass isolation, and output comparison. We plan to
start with sanitizer replay of high-risk cases and evaluate its runtime cost
and diagnostic noise before using it more broadly.

This public branch is an interim record rather than a complete artifact
release. It intentionally contains aggregate results and independently
testable fixes, but not the full mutation strategy or research tooling.
The method and experimental design are still being developed as part of a
paper. We plan to release a fuller implementation and reproducibility package
with a preprint or paper once the method and evaluation have stabilized.

We would welcome maintainer review when that fuller release is ready.
Continued evaluation may also uncover additional optimizer issues; any such
findings will be reported separately with a small reproducer and a clearly
stated support boundary.
