## 2026-06-21 - [Log Bombing via Unvalidated Input Dependency]
**Vulnerability:** A log-bombing vulnerability existed where validating and logging a failure for one input (e.g., URL length) utilized the raw, unvalidated value of another input (e.g., remote IP) in the log message.
**Learning:** Sequentially validating multiple inputs is dangerous if the validation failure of a later input logs the unvalidated state of a prior input.
**Prevention:** Always validate and safely truncate all untrusted inputs simultaneously at the beginning of the request cycle before any logging or business logic occurs.
