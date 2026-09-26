# Independent review task

Use GPT-5.6 Sol high in a separate session from the implementer. Read the specification before
implementation details and keep final outcomes hidden until the freeze review is complete.
Look for counterexamples: time/identity leakage, wrong dB averaging, unverified calibration,
collapsed latents, wrong target gradients, unequal budgets, denominator changes, fake offline
mode, path traversal and claims stronger than evidence. Do not author the feature you approve.

Default read-only inspection; independently executed tests require a permitted temporary review
workspace. State whether each conclusion comes from your own execution or inspected logs.
Return APPROVE / REQUEST_CHANGES / BLOCKED with severity, file/line references, reproduction,
required fixes and unresolved uncertainty. Do not rubber-stamp the leader's expected outcome.
