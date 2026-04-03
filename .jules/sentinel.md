## 2026-03-30 - [Flawed Path Sanitization]
**Vulnerability:** `secure_filename()` from `werkzeug.utils` was incorrectly used to sanitize requested asset paths (`/assets/<path:path>`).
**Learning:** `secure_filename()` replaces slashes with underscores to safely sanitize *uploaded* filenames. When used on a requested path string, it flattens the directory structure, functionally breaking the serving of any files stored in subdirectories (e.g., `nested/test.png` becomes `nested_test.png`).
**Prevention:** Do not use `secure_filename()` to sanitize paths containing subdirectories. Instead, rely on `send_from_directory()`'s native traversal protection or explicitly validate path resolution using `os.path.abspath()` combined with a `startswith()` boundary check.

## 2026-04-03 - [Information Leakage via Default Error Pages]
**Vulnerability:** Unhandled exceptions in the Flask application could bubble up and potentially leak internal stack traces or application state depending on the server environment.
**Learning:** By default, if a generic Python exception occurs (like a `ValueError` during path processing or array bounds error), Flask can render a default 500 error page that might expose framework details, or worse, full stack traces if debug mode is accidentally enabled or misconfigured in production.
**Prevention:** Implement a global `@app.errorhandler(Exception)` that passes through intentional `werkzeug.exceptions.HTTPException` errors (like 404s), but catches all unexpected exceptions, logs them safely internally, and returns a sanitized, generic 500 message to the client.
