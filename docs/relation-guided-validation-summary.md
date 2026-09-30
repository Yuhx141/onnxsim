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

## Included fix

This branch includes one independently reviewable change for #1997. A legal
`Shape` range beyond the input rank previously reached invalid iterator
arithmetic in `eliminate_shape_op` and terminated the process. The replacement
pass normalizes the range before reading the input dimensions. The original
reproducer now produces a checker-valid empty `int64` initializer, and the 19
focused optimizer tests pass.

## Ongoing work

The relation-guided method is still being extended across model sources and
optimizer families. The original 120-seed comparison used a regular build; it
was not an AddressSanitizer or Valgrind run. After the comparison, the included
fix and the 19 related optimizer tests were also run locally with the
repository's address/alignment sanitizer instrumentation and produced no
invalid-access or alignment report. This focused check is not sanitizer
coverage of the full 120-seed batch.

This public branch is an interim record rather than a complete artifact
release. It intentionally contains aggregate results and the independently
testable crash fix, but not the full mutation strategy or research tooling.
The method and experimental design are still being developed as part of a
paper. We plan to release a fuller implementation and reproducibility package
with a preprint or paper once the method and evaluation have stabilized.

We would welcome maintainer review when that fuller release is ready.
Continued evaluation may also uncover additional optimizer issues; any such
findings will be reported separately with a small reproducer and a clearly
stated support boundary.
