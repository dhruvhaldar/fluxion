from flask import Flask, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self';"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    # Security Enhancement: Prevent leaking referrer information cross-origin and disable sensitive browser features
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    # Security Enhancement: Prevent cross-origin resource embedding/reading
    response.headers['Cross-Origin-Resource-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    return response

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
                --bg-color: #ffffff;
                --text-color: #333333;
                --text-muted: #666666;
                --heading-color: #1a202c;
                --link-color: #3182ce;
                --link-hover-border: #3182ce;
                --link-focus-outline: #63b3ed;
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
                    --link-hover-border: #63b3ed;
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
            a { color: var(--link-color); text-decoration: none; border-bottom: 1px solid transparent; transition: border-color 0.2s, color 0.3s; }
            a:hover { border-bottom-color: var(--link-hover-border); }
            a:focus-visible { outline: 3px solid var(--link-focus-outline); outline-offset: 2px; border-radius: 2px; }
            .skip-link { position: absolute; top: -40px; left: 0; background: var(--bg-color); color: var(--link-color); padding: 8px; z-index: 100; transition: top 0.2s; font-weight: bold; border-bottom: none; border-right: 1px solid var(--img-border); border-bottom: 1px solid var(--img-border); border-bottom-right-radius: 4px; }
            .skip-link:focus { top: 0; outline: none; border-bottom: 1px solid var(--link-focus-outline); border-right: 1px solid var(--link-focus-outline); }
            .skip-link:focus-visible { outline: 3px solid var(--link-focus-outline); outline-offset: 0px; }
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

            <p>For code and documentation, visit the <a href="https://github.com/dhruvhaldar/fluxion" target="_blank" rel="noopener noreferrer" aria-label="GitHub Repository (opens in a new tab)">GitHub Repository</a>.</p>
        </main>
    </body>
    </html>
    """

@app.route('/assets/<path:path>', methods=['GET'])
def send_assets(path):
    # Prevent directory traversal attacks
    # werkzeug's send_from_directory does this securely, but explicitly checking is good defense in depth
    if '..' in path or path.startswith('/') or '%' in path:
        return "Bad Request", 400

    # Sanitize the filename to prevent traversal via modified path segments
    filename = secure_filename(path)

    # Security Enhancement: Only allow serving known safe media extensions
    allowed_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico',
        '.css', '.js', '.woff', '.woff2', '.ttf', '.eot'
    }
    _, ext = os.path.splitext(filename)
    if ext.lower() not in allowed_extensions:
        return "Unsupported Media Type", 415

    # Determine the absolute path to the assets directory
    # Assumes api/index.py is one level deeper than root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, 'assets')
    return send_from_directory(assets_dir, filename)

# For local testing
if __name__ == '__main__':
    # SECURE: Do not run with debug=True in production, as it exposes the Werkzeug debugger.
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1", "t"]
    app.run(debug=debug_mode)
