from flask import Flask, send_from_directory
import os
import logging
from werkzeug.exceptions import HTTPException

import werkzeug.serving
# Security Enhancement: Prevent Werkzeug from disclosing server version
werkzeug.serving.WSGIRequestHandler.server_version = ""
werkzeug.serving.WSGIRequestHandler.sys_version = ""

app = Flask(__name__)

# Security Enhancement: Restrict max content length to mitigate DoS (Denial of Service) via large payloads
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    # Security Enhancement: Restrict Content-Security-Policy to block base-uri injection, form submissions, frame embedding, and plugin execution
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'; upgrade-insecure-requests;"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    # Security Enhancement: Prevent leaking referrer information cross-origin and disable sensitive browser features
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    # Security Enhancement: Prevent cross-origin resource embedding/reading
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers.pop('Server', None)
    return response

# Security Enhancement: Global exception handler to prevent leaking stack traces
@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through standard HTTP errors (like 404, 400, etc.)
    if isinstance(e, HTTPException):
        return e
    # Log unexpected errors internally
    logging.error("Unhandled Exception: %s", str(e), exc_info=True)
    # Return generic 500 error to the client
    return "Internal Server Error", 500

@app.route('/', methods=['GET'])
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fluxion CFD</title>
        <style>

            :root {
                color-scheme: light dark;
                --bg-color: #ffffff;
                --text-color: #333333;
                --text-muted: #595959;
                --heading-color: #1a202c;
                --link-color: #2b6cb0;
                --link-hover-color: #1e40af;
                --link-focus-outline: #3182ce;
                --img-border: #e2e8f0;
                --img-shadow: rgba(0,0,0,0.1);
            }
            @media (prefers-color-scheme: dark) {
                :root {
                    --bg-color: #1a202c;
                    --text-color: #e2e8f0;
                    --text-muted: #a0aec0;
                    --heading-color: #f7fafc;
                    --link-color: #63b3ed;
                    --link-hover-color: #90cdf4;
                    --link-focus-outline: #90cdf4;
                    --img-border: #4a5568;
                    --img-shadow: rgba(0,0,0,0.5);
                }
            }
            body { font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; background-color: var(--bg-color); color: var(--text-color); transition: background-color 0.3s, color 0.3s; }
            figure { margin: 0 0 24px 0; padding: 0; }
            img { max-width: 100%; height: auto; border: 1px solid var(--img-border); border-radius: 4px; display: block; box-shadow: 0 1px 3px var(--img-shadow); transition: border-color 0.3s, box-shadow 0.3s; }
            figcaption { margin-top: 8px; font-size: 0.9em; color: var(--text-muted); text-align: center; font-style: italic; }
            h1, h2 { color: var(--heading-color); transition: color 0.3s; }
            a { color: var(--link-color); text-decoration: underline; text-underline-offset: 3px; text-decoration-color: var(--link-color); transition: text-decoration-color 0.2s, color 0.3s; }
            a:hover { color: var(--link-hover-color); text-decoration-color: var(--link-hover-color); text-decoration-thickness: 2px; }
            a:focus-visible { outline: 3px solid var(--link-focus-outline); outline-offset: 2px; border-radius: 2px; }
            .skip-link { text-decoration: none; position: absolute; top: -40px; left: 0; background: var(--bg-color); color: var(--link-color); padding: 8px; z-index: 100; transition: top 0.2s; font-weight: bold; border-bottom: none; border-right: 1px solid var(--img-border); border-bottom: 1px solid var(--img-border); border-bottom-right-radius: 4px; }
            .skip-link:focus { top: 0; outline: none; border-bottom: 1px solid var(--link-focus-outline); border-right: 1px solid var(--link-focus-outline); }
            .skip-link:focus-visible { outline: 3px solid var(--link-focus-outline); outline-offset: 0px; }
            main:focus { outline: none; }
        </style>
    </head>
    <body>
        <a href="#main" class="skip-link">Skip to main content</a>
        <main id="main" tabindex="-1">
            <h1>Fluxion</h1>
            <p>A structured Finite Volume Method (FVM) solver for SG2212 Computational Fluid Dynamics.</p>
            <p>This page demonstrates the generated artifacts from the solver.</p>

            <h2>Lid Driven Cavity Streamlines (Re=100)</h2>
            <figure>
                <img src="/assets/lid_driven_streamlines.png" alt="Streamlines and velocity magnitude contours for Re=100. The primary vortex is centered at (0.6, 0.5), matching benchmark data." />
                <figcaption>Figure 1: Streamlines and velocity magnitude contours for Re=100</figcaption>
            </figure>

            <h2>Grid Convergence Study</h2>
            <figure>
                <img src="/assets/grid_convergence.png" alt="Log-Log plot of L2-Error Norm vs. Grid Spacing (dx). The slope of the line is approximately 2.0, confirming the solver is Second-Order Accurate in space." />
                <figcaption>Figure 2: Grid Convergence Study demonstrating second-order spatial accuracy</figcaption>
            </figure>

            <h2>Convection Scheme Comparison</h2>
            <figure>
                <img src="/assets/scheme_comparison.png" alt="Convection of a step profile comparing Upwind, Central Difference, and QUICK schemes. Upwind shows diffusion, Central shows dispersion." />
                <figcaption>Figure 3: Comparison of Convection Schemes (Upwind, Central, QUICK)</figcaption>
            </figure>

            <p>For code and documentation, visit the <a href="https://github.com/dhruvhaldar/fluxion" target="_blank" rel="noopener noreferrer" aria-label="GitHub Repository (opens in a new tab)">GitHub Repository <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="display:inline-block; vertical-align:middle; margin-left:4px; margin-bottom:2px;" aria-hidden="true"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg></a>.</p>
        </main>
    </body>
    </html>
    """

@app.route('/assets/<path:path>', methods=['GET'])
def send_assets(path):
    # Prevent directory traversal attacks
    # explicitly checking is good defense in depth
    if '..' in path or path.startswith('/') or '%' in path:
        return "Bad Request", 400

    # Security Enhancement: Only allow serving known safe media extensions
    allowed_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico',
        '.css', '.js', '.woff', '.woff2', '.ttf', '.eot'
    }
    _, ext = os.path.splitext(path)
    if ext.lower() not in allowed_extensions:
        return "Unsupported Media Type", 415

    # Determine the absolute path to the assets directory
    # Assumes api/index.py is one level deeper than root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.abspath(os.path.join(base_dir, 'assets'))

    # Security Enhancement: Ensure resolved path stays within the intended assets directory
    # This acts as a robust defense against any bypass of previous string checks
    requested_path = os.path.abspath(os.path.join(assets_dir, path))
    if not requested_path.startswith(assets_dir + os.sep):
        return "Bad Request", 400

    return send_from_directory(assets_dir, path)

# For local testing
if __name__ == '__main__':
    # SECURE: Do not run with debug=True in production, as it exposes the Werkzeug debugger.
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "t"]
    app.run(debug=debug_mode)
