from flask import Flask, send_from_directory
import os

app = Flask(__name__)

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self';"
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Fluxion CFD</title>
        <style>
            body { font-family: system-ui, -apple-system, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; color: #333; }
            img { max-width: 100%; height: auto; border: 1px solid #e2e8f0; border-radius: 4px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            h1, h2 { color: #1a202c; }
            a { color: #3182ce; text-decoration: none; border-bottom: 1px solid transparent; transition: border-color 0.2s; }
            a:hover { border-bottom-color: #3182ce; }
            a:focus-visible { outline: 3px solid #63b3ed; outline-offset: 2px; border-radius: 2px; }
        </style>
    </head>
    <body>
        <main>
            <h1>Fluxion</h1>
            <p>A structured Finite Volume Method (FVM) solver for SG2212 Computational Fluid Dynamics.</p>
            <p>This page demonstrates the generated artifacts from the solver.</p>

            <h2>Lid Driven Cavity Streamlines (Re=100)</h2>
            <img src="/assets/lid_driven_streamlines.png" alt="Lid Driven Cavity Streamlines" />

            <h2>Grid Convergence Study</h2>
            <img src="/assets/grid_convergence.png" alt="Grid Convergence" />

            <h2>Convection Scheme Comparison</h2>
            <img src="/assets/scheme_comparison.png" alt="Scheme Comparison" />

            <p>For code and documentation, visit the <a href="https://github.com/dhruvhaldar/fluxion" target="_blank" rel="noopener noreferrer" aria-label="GitHub Repository (opens in a new tab)">GitHub Repository</a>.</p>
        </main>
    </body>
    </html>
    """

@app.route('/assets/<path:path>')
def send_assets(path):
    # Prevent directory traversal attacks
    # werkzeug's send_from_directory does this securely, but explicitly checking is good defense in depth
    if '..' in path or path.startswith('/') or '%' in path:
        return "Bad Request", 400

    # Determine the absolute path to the assets directory
    # Assumes api/index.py is one level deeper than root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, 'assets')
    return send_from_directory(assets_dir, path)

# For local testing
if __name__ == '__main__':
    app.run(debug=True)
