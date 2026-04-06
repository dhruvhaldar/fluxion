## 2026-03-30 - [Flawed Path Sanitization]
**Vulnerability:** `secure_filename()` from `werkzeug.utils` was incorrectly used to sanitize requested asset paths (`/assets/<path:path>`).
**Learning:** `secure_filename()` replaces slashes with underscores to safely sanitize *uploaded* filenames. When used on a requested path string, it flattens the directory structure, functionally breaking the serving of any files stored in subdirectories (e.g., `nested/test.png` becomes `nested_test.png`).
**Prevention:** Do not use `secure_filename()` to sanitize paths containing subdirectories. Instead, rely on `send_from_directory()`'s native traversal protection or explicitly validate path resolution using `os.path.abspath()` combined with a `startswith()` boundary check.
## 2026-04-06 - [Workspace Pollution]
**Vulnerability:** Leftover temporary debugging scripts (e.g., `test_error_handler.py`, `test_app.py`) were left in the project root after validating a security fix.
**Learning:** These arbitrary, unmaintained scripts pollute the codebase, add confusion for future developers, and might be accidentally committed, violating coding standard boundaries. Proper verification should use existing or new unit test files (e.g., within the `tests/` directory) rather than root-level scratchpads.
**Prevention:** Always delete any temporary diagnostic files (`rm temp_script.py`) created during validation before finalizing work, or explicitly build tests within the designated test suite structure.
