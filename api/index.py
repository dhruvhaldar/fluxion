from flask import Flask, send_from_directory, request
import os
import re
import logging
import time
import threading
import ipaddress
from collections import deque
from werkzeug.exceptions import HTTPException

import werkzeug.serving
from werkzeug.middleware.proxy_fix import ProxyFix

# Security Enhancement: Prevent Werkzeug from disclosing server version
werkzeug.serving.WSGIRequestHandler.server_version = ""
werkzeug.serving.WSGIRequestHandler.sys_version = ""

app = Flask(__name__)

# Security Enhancement: Accurately resolve client IP behind reverse proxy (e.g., Vercel)
# to ensure security audit logs capture the real attacker IP instead of the proxy IP.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Security Enhancement: Prevent leaking stack traces or sensitive internal state on unexpected errors
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        # Security Enhancement: Return safe plain text response for HTTPExceptions
        # rather than the default Werkzeug HTML templates to prevent framework
        # fingerprinting and potential XSS issues.
        body = f"{e.code} {e.name}: {e.description}"
        headers = {'Content-Type': 'text/plain; charset=utf-8'}
        # Security Enhancement: Explicitly close connection on 413 to prevent DoS via keep-alive streaming
        if e.code == 413:
            headers['Connection'] = 'close'
        return body, e.code, headers
    app.logger.error("Unexpected error", exc_info=True)
    return "Internal Server Error", 500, {"Content-Type": "text/plain; charset=utf-8"}

# Security Enhancement: Ensure secure defaults for sessions
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)

# Security Enhancement: Configure SECRET_KEY from environment to avoid dynamic generation
# which causes issues across multi-worker environments. Do not use a random fallback!
secret_key = os.environ.get('SECRET_KEY')
is_testing = 'pytest' in __import__('sys').modules
if not secret_key and not os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "t"] and not is_testing:
    raise RuntimeError("SECRET_KEY must be set in production.")
if secret_key:
    app.config['SECRET_KEY'] = secret_key

# Security Enhancement: Restrict max content length to mitigate DoS (Denial of Service) via large payloads
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

@app.before_request
def check_content_length():
    # Security Enhancement: To mitigate resource exhaustion (DoS) attacks from excessively large payloads,
    # explicitly check request.content_length against a configured maximum (e.g., app.config['MAX_CONTENT_LENGTH'])
    # early in the request lifecycle, such as within an @app.before_request hook. Reject oversized payloads immediately
    # with a 413 Payload Too Large error, ensuring the connection is closed before Werkzeug consumes the stream.
    if request.content_length is not None and request.content_length > app.config.get('MAX_CONTENT_LENGTH', float('inf')):
        return "Payload Too Large", 413, {"Content-Type": "text/plain; charset=utf-8", "Connection": "close"}

# Security Enhancement: Rate Limiting
MAX_TRACKED_IPS = 10000
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 100

# Track IPs in memory. Bounded by MAX_TRACKED_IPS to prevent memory exhaustion DoS.
ip_tracker = {}
ip_tracker_lock = threading.Lock()

# Track early blocks to prevent log bombing
early_block_tracker = {}
early_block_lock = threading.Lock()

def normalize_ip(raw_ip):
    """
    Normalizes an IP address. Strips scope IDs, converts IPv4-mapped IPv6,
    and normalizes IPv6 to its /64 subnet. Returns None if invalid.
    """
    if not raw_ip:
        return None
    try:
        clean_ip = raw_ip.split('%')[0].strip()
        ip_obj = ipaddress.ip_address(clean_ip)
        if getattr(ip_obj, 'ipv4_mapped', None):
            return ip_obj.ipv4_mapped.compressed
        if ip_obj.version == 6:
            net = ipaddress.ip_network(f"{ip_obj.compressed}/64", strict=False)
            return net.network_address.compressed
        return ip_obj.compressed
    except ValueError:
        return None


def log_early_block(key, message):
    current_time = time.monotonic()
    with early_block_lock:
        if key not in early_block_tracker:
            if len(early_block_tracker) >= MAX_TRACKED_IPS:
                oldest_key = next(iter(early_block_tracker))
                del early_block_tracker[oldest_key]
            early_block_tracker[key] = current_time
            app.logger.warning(message)
        else:
            early_block_tracker[key] = early_block_tracker.pop(key)
            if current_time - early_block_tracker[key] > RATE_LIMIT_WINDOW:
                early_block_tracker[key] = current_time
                app.logger.warning(message)


@app.before_request
def rate_limit():
    raw_ip = request.remote_addr
    raw_url = request.url

    # Explicitly validate and truncate all inputs BEFORE using any of them in log messages.
    # Logging a failure for one input using the unvalidated, raw value of another creates a log-bombing vulnerability.
    safe_ip = raw_ip[:45] + '...[TRUNCATED]' if raw_ip and len(raw_ip) > 45 else raw_ip
    safe_url = raw_url[:256] + '...[TRUNCATED]' if raw_url and len(raw_url) > 256 else raw_url
    safe_method = request.method[:20] + '...[TRUNCATED]' if request.method and len(request.method) > 20 else request.method

    # Security Enhancement: Global strict HTTP method restriction.
    # Reject methods we don't use to prevent HTTP verb tampering and save resources.
    allowed_methods = {"GET", "HEAD", "OPTIONS"}
    if request.method not in allowed_methods:
        log_early_block(f"invalid_method_global", f"Security Event: Blocked request using unsupported method {repr(safe_method)}. url: {repr(safe_url)}")
        return "Method Not Allowed", 405, {"Content-Type": "text/plain; charset=utf-8"}

    if not raw_ip:
        log_early_block("missing_ip", f"Security Event: Blocked request with missing remote address. url: {repr(safe_url)}")
        return "Bad Request", 400, {"Content-Type": "text/plain; charset=utf-8"}

    # Security Enhancement: Limit the length of the remote address to mitigate DoS
    # via memory exhaustion or log bombing using extremely long spoofed IP headers.
    if len(raw_ip) > 45:
        log_early_block("long_ip_global", f"Security Event: Blocked request due to excessively long remote address: {repr(safe_ip)}. url: {repr(safe_url)}")
        return "Bad Request", 400, {"Content-Type": "text/plain; charset=utf-8"}

    # Normalize IP address to prevent bypass of IP-based controls
    ip = normalize_ip(raw_ip)

    # Security Enhancement: Reject bodies in read-only API to prevent HTTP Request Smuggling,
    # Cache Poisoning, and resource exhaustion.
    if (request.content_length is not None and request.content_length > 0) or 'Transfer-Encoding' in request.headers:
        ip_key = ip if ip else "missing_ip"
        log_early_block(f"unexpected_body_{ip_key}", f"Security Event: Blocked request from {repr(safe_ip)} due to unexpected request body in read-only API.")
        return "Payload Too Large", 413, {"Content-Type": "text/plain; charset=utf-8", "Connection": "close"}

    # Security Enhancement: Normalize IP address to prevent bypass of IP-based controls
    # via multiple representations of the same IPv6 address (e.g., 2001:db8::1 vs 2001:db8:0:0:0:0:0:1).
    # Also handles IPv4-mapped IPv6 addresses (e.g. ::ffff:192.168.0.1) to prevent rate-limit bypasses
    # Groups IPv6 addresses by /64 subnet to prevent rate-limit bypasses by using different addresses in the same subnet
    if not ip:
        log_early_block("invalid_ip_global", f"Security Event: Blocked request from {repr(request.remote_addr)} with invalid IP address format.")
        return "Bad Request", 400, {"Content-Type": "text/plain; charset=utf-8"}

    # Security Enhancement: Restrict the maximum length of the entire URL (including query strings)
    # to mitigate DoS (Denial of Service) attacks via memory exhaustion and buffer overflows.
    if raw_url and len(raw_url) > 2048:
        log_early_block(f"long_uri_{ip}", f"Security Event: Blocked request from {repr(safe_ip)} due to URI length > 2048. url: {repr(safe_url)}")
        return "URI Too Long", 414, {"Content-Type": "text/plain; charset=utf-8"}

    # Security Enhancement: Use monotonic time for rate limiting to prevent
    # bypasses or lockouts caused by system clock adjustments (e.g., NTP sync).
    current_time = time.monotonic()

    with ip_tracker_lock:
        if ip not in ip_tracker:
            # Enforce maximum size on the tracker dictionary
            if len(ip_tracker) >= MAX_TRACKED_IPS:
                # Security Enhancement: Prevent Algorithmic Complexity DoS (CPU exhaustion)
                # by avoiding an O(N) scan of all tracked IPs. We only check the oldest IP
                # and evict it, which is O(1) and safely bounds memory without CPU lag.
                oldest_ip = next(iter(ip_tracker))
                del ip_tracker[oldest_ip]

            ip_tracker[ip] = {'requests': deque(), 'last_logged': float('-inf')}
        else:
            # Security Enhancement: Implement LRU eviction policy to prevent eviction bypass
            # By moving the accessed IP to the end of the dictionary, we ensure that
            # active IPs are not evicted when MAX_TRACKED_IPS is reached.
            ip_tracker[ip] = ip_tracker.pop(ip)

        tracker = ip_tracker[ip]
        req_queue = tracker['requests']

        # Prune old requests for this IP
        while req_queue and req_queue[0] < current_time - RATE_LIMIT_WINDOW:
            req_queue.popleft()

        if len(req_queue) >= RATE_LIMIT_MAX_REQUESTS:
            # Security Enhancement: Prevent log-bombing / Disk DoS by only logging
            # the rate limit violation once per burst. By explicitly tracking the last
            # log time, we reliably suppress logs even against steady-rate attackers.
            if current_time - tracker['last_logged'] > RATE_LIMIT_WINDOW:
                app.logger.warning(f"Security Event: Rate limit exceeded for {repr(request.remote_addr)} (normalized to {repr(ip)})")
                tracker['last_logged'] = current_time
            return "Too Many Requests", 429, {"Content-Type": "text/plain; charset=utf-8", "Retry-After": str(RATE_LIMIT_WINDOW)}

        req_queue.append(current_time)


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '0'
    # Security Enhancement: Restrict Content-Security-Policy to block base-uri injection, form submissions, frame embedding, and plugin execution
    response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'; upgrade-insecure-requests;"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'
    # Security Enhancement: Prevent leaking referrer information cross-origin and disable sensitive browser features
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    # Security Enhancement: Prevent cross-origin resource embedding/reading
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    if response.status_code >= 400:
        response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers.pop('Server', None)
    return response

@app.route('/', methods=['GET'])
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="theme-color" content="#ffffff" media="(prefers-color-scheme: light)">
        <meta name="theme-color" content="#1a202c" media="(prefers-color-scheme: dark)">
        <title>Fluxion CFD</title>
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🌊</text></svg>">
        <link rel="stylesheet" href="/assets/style.css">
    </head>
    <body id="top" tabindex="-1">
        <a href="#main" class="skip-link">Skip to main content</a>
        <main id="main" tabindex="-1">
            <h1>Fluxion</h1>
            <p>A structured <abbr title="Finite Volume Method" tabindex="0">FVM</abbr> solver for SG2212 <abbr title="Computational Fluid Dynamics" tabindex="0">CFD</abbr>.</p>
            <p>This page demonstrates the generated artifacts from the solver.</p>

            <h2 id="lid-driven-cavity-streamlines" tabindex="-1">Lid Driven Cavity Streamlines (<abbr title="Reynolds number" tabindex="0">Re</abbr>=100) <a href="#lid-driven-cavity-streamlines" class="heading-anchor" aria-label="Permalink to Lid Driven Cavity Streamlines (Re=100)" title="Permalink to Lid Driven Cavity Streamlines (Re=100)">#</a></h2>
            <figure>
                <a href="/assets/lid_driven_streamlines.png" target="_blank" rel="noopener noreferrer" title="View full size" aria-describedby="fig1-caption">
                    <img src="/assets/lid_driven_streamlines.png" width="800" height="600" alt="A color contour map showing a large primary vortex centered at coordinates (0.6, 0.5) with smaller secondary vortices in the bottom corners." loading="lazy" decoding="async" />
                    <span class="sr-only">(opens in a new tab)</span>
                </a>
                <figcaption id="fig1-caption">Figure 1: Streamlines and velocity magnitude contours for <abbr title="Reynolds number" tabindex="0">Re</abbr>=100</figcaption>
            </figure>

            <h2 id="grid-convergence-study" tabindex="-1">Grid Convergence Study <a href="#grid-convergence-study" class="heading-anchor" aria-label="Permalink to Grid Convergence Study" title="Permalink to Grid Convergence Study">#</a></h2>
            <figure>
                <a href="/assets/grid_convergence.png" target="_blank" rel="noopener noreferrer" title="View full size" aria-describedby="fig2-caption">
                    <img src="/assets/grid_convergence.png" width="800" height="600" alt="A log-log line graph plotting L2-Error Norm against Grid Spacing. A straight line is fitted to the data points showing a downward slope of approximately 2.0." loading="lazy" decoding="async" />
                    <span class="sr-only">(opens in a new tab)</span>
                </a>
                <figcaption id="fig2-caption">Figure 2: Grid Convergence Study demonstrating second-order spatial accuracy</figcaption>
            </figure>

            <h2 id="convection-scheme-comparison" tabindex="-1">Convection Scheme Comparison <a href="#convection-scheme-comparison" class="heading-anchor" aria-label="Permalink to Convection Scheme Comparison" title="Permalink to Convection Scheme Comparison">#</a></h2>
            <figure>
                <a href="/assets/scheme_comparison.png" target="_blank" rel="noopener noreferrer" title="View full size" aria-describedby="fig3-caption">
                    <img src="/assets/scheme_comparison.png" width="1000" height="600" alt="A line graph comparing three lines against a sharp step function. The upwind line is heavily smoothed out, while the central difference line has sharp wiggles near the step." loading="lazy" decoding="async" />
                    <span class="sr-only">(opens in a new tab)</span>
                </a>
                <figcaption id="fig3-caption">Figure 3: Comparison of Convection Schemes (Upwind, Central, <abbr title="Quadratic Upstream Interpolation for Convective Kinematics" tabindex="0">QUICK</abbr>)</figcaption>
            </figure>

        </main>
        <footer>
            <p>For code and documentation, visit the <a href="https://github.com/dhruvhaldar/fluxion" target="_blank" rel="noopener noreferrer" aria-label="View the Fluxion CFD GitHub Repository (opens in a new tab)">GitHub Repository <span class="sr-only">(opens in a new tab)</span><svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="external-link-icon" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg></a>.</p>
            <p><a href="#top" aria-label="Back to top of page"><svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="back-to-top-icon" aria-hidden="true"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg> Back to top</a></p>
        </footer>
    </body>
    </html>
    """

@app.route('/assets/<path:path>', methods=['GET'])
def send_assets(path):
    raw_ip = request.remote_addr
    norm_ip = normalize_ip(raw_ip)
    ip_key = norm_ip if norm_ip else "global"

    # Security Enhancement: Restrict input length to mitigate DoS attacks
    if len(path) > 256:
        # Security Enhancement: Truncate excessively long path payload before logging
        # to mitigate log bombing/Disk DoS.
        truncated_path = path[:256] + '...[TRUNCATED]'
        log_early_block(f"long_asset_path_{ip_key}", f"Security Event: Blocked request from {repr(request.remote_addr)} due to URI length > 256. path: {repr(truncated_path)}")
        return "URI Too Long", 414, {"Content-Type": "text/plain; charset=utf-8"}

    # Prevent directory traversal attacks
    # explicitly checking is good defense in depth
    if '..' in path or path.startswith('/') or '%' in path:
        log_early_block(f"dir_traversal_{ip_key}", f"Security Event: Blocked request from {repr(request.remote_addr)} due to potential directory traversal. path: {repr(path)}")
        return "Bad Request", 400, {"Content-Type": "text/plain; charset=utf-8"}

    # Security Enhancement: Block requests for hidden files or directories
    # to prevent accidental exposure of sensitive internal metadata (e.g., .git/, .env)
    # even if allowed_extensions is later relaxed to include generic formats.
    if path.startswith('.') or '/.' in path:
        log_early_block(f"hidden_file_{ip_key}", f"Security Event: Blocked request from {repr(request.remote_addr)} for hidden file/directory. path: {repr(path)}")
        return "Bad Request", 400, {"Content-Type": "text/plain; charset=utf-8"}

    # Security Enhancement: Strict allowed characters for file paths to prevent log injection or unexpected parser behavior
    # Using \Z and re.fullmatch to ensure trailing newlines are correctly blocked
    if not re.fullmatch(r'^[a-zA-Z0-9_./-]+\Z', path):
        log_early_block(f"invalid_chars_{ip_key}", f"Security Event: Blocked request from {repr(request.remote_addr)} due to invalid characters in path. path: {repr(path)}")
        return "Bad Request", 400, {"Content-Type": "text/plain; charset=utf-8"}

    # Security Enhancement: Only allow serving known safe media extensions
    allowed_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico',
        '.css', '.js', '.woff', '.woff2', '.ttf', '.eot'
    }
    _, ext = os.path.splitext(path)
    if ext.lower() not in allowed_extensions:
        log_early_block(f"unsupported_media_{ip_key}", f"Security Event: Blocked request from {repr(request.remote_addr)} due to unsupported media type. ext: {repr(ext)}")
        return "Unsupported Media Type", 415, {"Content-Type": "text/plain; charset=utf-8"}

    # Determine the absolute path to the assets directory
    # Assumes api/index.py is one level deeper than root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.abspath(os.path.join(base_dir, 'assets'))

    # Security Enhancement: Ensure resolved path stays within the intended assets directory
    # This acts as a robust defense against any bypass of previous string checks
    requested_path = os.path.abspath(os.path.join(assets_dir, path))
    if not requested_path.startswith(assets_dir + os.sep):
        log_early_block(f"out_of_bounds_{ip_key}", f"Security Event: Blocked request from {repr(request.remote_addr)} due to out-of-bounds resolved path. path: {repr(path)}")
        return "Bad Request", 400, {"Content-Type": "text/plain; charset=utf-8"}

    return send_from_directory(assets_dir, path)

# For local testing
if __name__ == '__main__':
    # SECURE: Do not run with debug=True in production, as it exposes the Werkzeug debugger.
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "t"]
    app.run(debug=debug_mode)
