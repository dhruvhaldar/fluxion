from flask import Flask, send_from_directory, request
import os
import re
import logging
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

import werkzeug.serving
# Security Enhancement: Prevent Werkzeug from disclosing server version
werkzeug.serving.WSGIRequestHandler.server_version = ""
werkzeug.serving.WSGIRequestHandler.sys_version = ""

app = Flask(__name__)
# Security Enhancement: Accurately resolve client IPs behind reverse proxies (like Vercel) for audit logging
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Security Enhancement: Prevent leaking stack traces or sensitive internal state on unexpected errors
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        # Security Enhancement: Return safe plain text response for HTTPExceptions
        # rather than the default Werkzeug HTML templates to prevent framework
        # fingerprinting and potential XSS issues.
        response = e.get_response()
        response.data = f"{e.code} {e.name}: {e.description}"
        response.content_type = "text/plain"
        return response
    app.logger.error("Unexpected error", exc_info=True)
    return "Internal Server Error", 500

# Security Enhancement: Restrict max content length to mitigate DoS (Denial of Service) via large payloads
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024


@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    # Security Enhancement: Restrict Content-Security-Policy to block base-uri injection, form submissions, frame embedding, and plugin execution
    response.headers['Content-Security-Policy'] = "default-src 'none'; style-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'; upgrade-insecure-requests;"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    # Security Enhancement: Prevent leaking referrer information cross-origin and disable sensitive browser features
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    # Security Enhancement: Prevent cross-origin resource embedding/reading
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
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
    <body id="top">
        <a href="#main" class="skip-link">Skip to main content</a>
        <main id="main" tabindex="-1">
            <h1>Fluxion</h1>
            <p>A structured Finite Volume Method (FVM) solver for SG2212 Computational Fluid Dynamics.</p>
            <p>This page demonstrates the generated artifacts from the solver.</p>

            <h2 id="lid-driven-cavity-streamlines">Lid Driven Cavity Streamlines (Re=100) <a href="#lid-driven-cavity-streamlines" class="heading-anchor" aria-label="Permalink to this section">#</a></h2>
            <figure>
                <a href="/assets/lid_driven_streamlines.png" target="_blank" rel="noopener noreferrer" title="Click to view full size">
                    <img src="/assets/lid_driven_streamlines.png" width="800" height="600" alt="Streamlines and velocity magnitude contours for Re=100. The primary vortex is centered at (0.6, 0.5), matching benchmark data." />
                    <span class="sr-only">(opens in a new tab)</span>
                </a>
                <figcaption>Figure 1: Streamlines and velocity magnitude contours for Re=100</figcaption>
            </figure>

            <h2 id="grid-convergence-study">Grid Convergence Study <a href="#grid-convergence-study" class="heading-anchor" aria-label="Permalink to this section">#</a></h2>
            <figure>
                <a href="/assets/grid_convergence.png" target="_blank" rel="noopener noreferrer" title="Click to view full size">
                    <img src="/assets/grid_convergence.png" width="800" height="600" alt="Log-Log plot of L2-Error Norm vs. Grid Spacing (dx). The slope of the line is approximately 2.0, confirming the solver is Second-Order Accurate in space." />
                    <span class="sr-only">(opens in a new tab)</span>
                </a>
                <figcaption>Figure 2: Grid Convergence Study demonstrating second-order spatial accuracy</figcaption>
            </figure>

            <h2 id="convection-scheme-comparison">Convection Scheme Comparison <a href="#convection-scheme-comparison" class="heading-anchor" aria-label="Permalink to this section">#</a></h2>
            <figure>
                <a href="/assets/scheme_comparison.png" target="_blank" rel="noopener noreferrer" title="Click to view full size">
                    <img src="/assets/scheme_comparison.png" width="1000" height="600" alt="Convection of a step profile comparing Upwind, Central Difference, and QUICK schemes. Upwind shows diffusion, Central shows dispersion." />
                    <span class="sr-only">(opens in a new tab)</span>
                </a>
                <figcaption>Figure 3: Comparison of Convection Schemes (Upwind, Central, QUICK)</figcaption>
            </figure>

        </main>
        <footer>
            <p>For code and documentation, visit the <a href="https://github.com/dhruvhaldar/fluxion" target="_blank" rel="noopener noreferrer">GitHub Repository <span class="sr-only">(opens in a new tab)</span><svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="external-link-icon" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg></a>.</p>
            <p><a href="#top"><span aria-hidden="true">↑</span> Back to top</a></p>
        </footer>
    </body>
    </html>
    """

@app.route('/assets/<path:path>', methods=['GET'])
def send_assets(path):
    # Security Enhancement: Restrict input length to mitigate DoS attacks
    if len(path) > 256:
        # Security Enhancement: Truncate excessively long path payload before logging
        # to mitigate log bombing/Disk DoS.
        truncated_path = path[:256] + '...[TRUNCATED]'
        app.logger.warning(f"Security Event: Blocked request from {request.remote_addr} due to URI length > 256. path: {repr(truncated_path)}")
        return "URI Too Long", 414

    # Prevent directory traversal attacks
    # explicitly checking is good defense in depth
    if '..' in path or path.startswith('/') or '%' in path:
        app.logger.warning(f"Security Event: Blocked request from {request.remote_addr} due to potential directory traversal. path: {repr(path)}")
        return "Bad Request", 400

    # Security Enhancement: Strict allowed characters for file paths to prevent log injection or unexpected parser behavior
    # Using \Z and re.fullmatch to ensure trailing newlines are correctly blocked
    if not re.fullmatch(r'^[a-zA-Z0-9_./-]+\Z', path):
        app.logger.warning(f"Security Event: Blocked request from {request.remote_addr} due to invalid characters in path. path: {repr(path)}")
        return "Bad Request", 400

    # Security Enhancement: Only allow serving known safe media extensions
    allowed_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico',
        '.css', '.js', '.woff', '.woff2', '.ttf', '.eot'
    }
    _, ext = os.path.splitext(path)
    if ext.lower() not in allowed_extensions:
        app.logger.warning(f"Security Event: Blocked request from {request.remote_addr} due to unsupported media type. ext: {repr(ext)}")
        return "Unsupported Media Type", 415

    # Determine the absolute path to the assets directory
    # Assumes api/index.py is one level deeper than root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.abspath(os.path.join(base_dir, 'assets'))

    # Security Enhancement: Ensure resolved path stays within the intended assets directory
    # This acts as a robust defense against any bypass of previous string checks
    requested_path = os.path.abspath(os.path.join(assets_dir, path))
    if not requested_path.startswith(assets_dir + os.sep):
        app.logger.warning(f"Security Event: Blocked request from {request.remote_addr} due to out-of-bounds resolved path. path: {repr(path)}")
        return "Bad Request", 400

    return send_from_directory(assets_dir, path)

# For local testing
if __name__ == '__main__':
    # SECURE: Do not run with debug=True in production, as it exposes the Werkzeug debugger.
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "t"]
    app.run(debug=debug_mode)
