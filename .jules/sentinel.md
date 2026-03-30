## 2026-03-30 - [Flawed Path Sanitization]
**Vulnerability:** `secure_filename()` from `werkzeug.utils` was incorrectly used to sanitize requested asset paths (`/assets/<path:path>`).
**Learning:** `secure_filename()` replaces slashes with underscores to safely sanitize *uploaded* filenames. When used on a requested path string, it flattens the directory structure, functionally breaking the serving of any files stored in subdirectories (e.g., `nested/test.png` becomes `nested_test.png`).
**Prevention:** Do not use `secure_filename()` to sanitize paths containing subdirectories. Instead, rely on `send_from_directory()`'s native traversal protection or explicitly validate path resolution using `os.path.abspath()` combined with a `startswith()` boundary check.
