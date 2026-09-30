# Relation-guided optimizer validation

This branch records a focused evaluation of relation-guided validation as a
complement to onnxsim's existing whole-graph fuzzing.

## Scope and evidence

The broader corpus combines traceable public ONNX models from Hugging Face,
targeted reduced cases, and NNSmith-generated models. A controlled comparison
against #1274 covered 120 models and 360 checker-valid variants. The added
validation layer introduced no relation-specific false positives and separated
three pre-existing constant-folding numerical cases from optimizer-pass
findings.

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
optimizer families. This public branch intentionally limits itself to the
measured summary and the independently testable crash fix while the research
method continues to mature.

Once the method and evaluation are mature, we intend to prepare a fuller
methodological description and paper, and would welcome maintainer review of
that work. Continued evaluation may also uncover additional optimizer issues;
any such findings will be reported separately with a small reproducer and a
clearly stated support boundary.
