## 2026-06-21 - [Log Bombing via Unvalidated Input Dependency]
**Vulnerability:** A log-bombing vulnerability existed where validating and logging a failure for one input (e.g., URL length) utilized the raw, unvalidated value of another input (e.g., remote IP) in the log message.
**Learning:** Sequentially validating multiple inputs is dangerous if the validation failure of a later input logs the unvalidated state of a prior input.
**Prevention:** Always validate and safely truncate all untrusted inputs simultaneously at the beginning of the request cycle before any logging or business logic occurs.

## 2026-06-21 - [Log Bombing via Steady-Rate Rate Limit Evasion]
**Vulnerability:** A log-bombing vulnerability existed in the rate limiter because it suppressed repeated logs by appending a dummy request timestamp to the queue. An attacker sending requests at a slow, steady rate (e.g., 1 request every 0.5s for a 100-request/60s limit) could cause older requests to expire, dropping the queue size back down to 100 and repeatedly triggering the log message every time the limit was breached anew.
**Learning:** Suppressing logs based on the length of a sliding window queue is fundamentally flawed because the queue size shrinks as time passes.
**Prevention:** To reliably suppress logs in a rate limiter, explicitly track the `last_logged` time for each IP (e.g., `{'requests': deque(), 'last_logged': 0.0}`) and only log a violation if `current_time - last_logged > RATE_LIMIT_WINDOW`.
