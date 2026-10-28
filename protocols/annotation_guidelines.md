# Annotation Guidelines (v0)

**Status:** DRAFT. Do not begin gold annotation until this document is frozen
and versioned as `gold-v1`.

## Unit of annotation
One record = one research question + one or more answer propositions +
supporting/refuting/qualifying evidence spans + relation judgments.

## Label set
| Label | Definition |
|---|---|
| supports | Evidence directly supports the proposition |
| refutes | Evidence is incompatible with the proposition under comparable conditions |
| qualifies | Evidence narrows or conditions the proposition |
| context_differs | Findings differ, but conditions do not establish a genuine contradiction |
| insufficient | Evidence cannot establish support or refutation |
| irrelevant | Span does not materially bear on the proposition |

## Offset discipline
All spans must be recorded as code-point offsets into the normalized text
produced by `evidencegraph.ingestion.normalization.normalize_text`. Quoted text
must equal the slice exactly.

## Independence rule
Annotators must not see system-generated claim graphs or answers for the
question being annotated. Annotations are frozen and hashed before confirmatory
runs.

## Known edge cases
- Non-significant does not mean no effect.
- Different effect sizes are not automatically contradictory.
- A hypothesis stated by authors is not a result.
