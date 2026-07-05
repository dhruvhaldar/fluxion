## 2026-06-21 - [Log Bombing via Unvalidated Input Dependency]
**Vulnerability:** A log-bombing vulnerability existed where validating and logging a failure for one input (e.g., URL length) utilized the raw, unvalidated value of another input (e.g., remote IP) in the log message.
**Learning:** Sequentially validating multiple inputs is dangerous if the validation failure of a later input logs the unvalidated state of a prior input.
**Prevention:** Always validate and safely truncate all untrusted inputs simultaneously at the beginning of the request cycle before any logging or business logic occurs.

## 2026-06-21 - [Log Bombing via Steady-Rate Rate Limit Evasion]
**Vulnerability:** A log-bombing vulnerability existed in the rate limiter because it suppressed repeated logs by appending a dummy request timestamp to the queue. An attacker sending requests at a slow, steady rate (e.g., 1 request every 0.5s for a 100-request/60s limit) could cause older requests to expire, dropping the queue size back down to 100 and repeatedly triggering the log message every time the limit was breached anew.
**Learning:** Suppressing logs based on the length of a sliding window queue is fundamentally flawed because the queue size shrinks as time passes.
**Prevention:** To reliably suppress logs in a rate limiter, explicitly track the `last_logged` time for each IP (e.g., `{'requests': deque(), 'last_logged': 0.0}`) and only log a violation if `current_time - last_logged > RATE_LIMIT_WINDOW`.

## 2026-06-21 - [Log Bombing via Improper String Truncation Condition]
**Vulnerability:** A log-bombing vulnerability existed where the logging logic attempted to truncate large URLs using `[:256]` but the condition to trigger truncation was erroneously set to `len(raw_url) > 2048`. This allowed payloads between 257 and 2048 characters to bypass truncation entirely and be logged in full, risking a Disk DoS attack.
**Learning:** Truncation bounds must exactly match their triggering conditions. A mismatch creates a window where massive, unsanitized inputs bypass formatting constraints.
**Prevention:** Always ensure the condition for truncation directly matches the length of the truncation slice (e.g., `if len(val) > LIMIT: val = val[:LIMIT]`).
## 2026-07-01 - [Log Bombing via Un-Rate-Limited Early Returns]
**Vulnerability:** A log-bombing (Disk DoS) vulnerability existed where early request rejections (e.g., excessively long URIs, missing IPs, directory traversal attempts) logged a warning synchronously and returned a 400/414 response *before* hitting the main application rate limiter. This allowed attackers to flood the server with malformed requests to bypass the rate limiter entirely and infinitely spam the application logs.
**Learning:** Security validations that return early and log warnings must themselves be protected by rate-limiting or log-suppression logic.
**Prevention:** Implement a separate, dedicated rate-limiting mechanism (like an LRU tracker dictionary) specifically to suppress duplicate log entries for early-block security events.

## 2024-07-02 - IPv6 Subnet Parsing and Tracker Key DoS Mitigation
**Vulnerability:** A Denial of Service (DoS) vulnerability existed in `api/index.py` due to using unnormalized `request.remote_addr` inputs as dictionary keys for early security blocks. Attackers could rotate IPv6 addresses within a /64 subnet, generating an endless stream of unique keys (e.g., `long_ip_2001:db8::1`, `long_ip_2001:db8::2`), bypassing the rate-limiter and causing unbounded memory consumption.
**Learning:** Even when early logic truncates long strings to a max length (e.g., 45 chars), if the input is fundamentally variable (like raw IPv6 addresses), treating it as a dictionary key causes memory leaks and log flooding. We also learned that global tracking objects must be cleared in Pytest fixtures to avoid test-case state leakage.
**Prevention:** Consolidate IP normalization into a single robust helper function (stripping scope IDs, formatting IPv4-mapped IPv6, and normalizing IPv6 to /64). For completely invalid IPs or pre-parsing failures, fallback to static tracking keys (e.g., `"invalid_ip_global"`) to permanently bound the memory signature in the tracker dict.

## 2026-07-05 - [IP Parser Bypass via Whitespace Padding]
**Vulnerability:** A vulnerability existed where an attacker could bypass IP tracking, rate-limiting, and blocklists by padding the X-Forwarded-For or remote address with leading or trailing whitespaces (e.g., ` 192.168.1.1 `). Python's `ipaddress` module strictly validates IP strings and rejects padded strings with a `ValueError`. This resulted in the IP being classified as invalid and circumventing tracking mechanisms dependent on `normalize_ip`.
**Learning:** External libraries such as `ipaddress` often have strict parsing rules and do not automatically sanitize whitespace. Any unstripped user input passed directly to these libraries could trigger false-negative validations.
**Prevention:** Always sanitize input strings by invoking `.strip()` before passing them to strict parsers like `ipaddress.ip_address`.
