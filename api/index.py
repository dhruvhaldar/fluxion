from flask import Flask, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fluxion CFD</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <h1>Fluxion</h1>
        <p>A structured Finite Volume Method (FVM) solver for SG2212 Computational Fluid Dynamics.</p>
        <p>This page demonstrates the generated artifacts from the solver.</p>

        <h2>Lid Driven Cavity Streamlines (Re=100)</h2>
        <img src="/assets/lid_driven_streamlines.png" alt="Lid Driven Cavity Streamlines" />

        <h2>Grid Convergence Study</h2>
        <img src="/assets/grid_convergence.png" alt="Grid Convergence" />

        <h2>Convection Scheme Comparison</h2>
        <img src="/assets/scheme_comparison.png" alt="Scheme Comparison" />

        <p>For code and documentation, visit the <a href="https://github.com/your-username/fluxion">GitHub Repository</a>.</p>
    </body>
    </html>
    """

@app.route('/assets/<path:path>')
def send_assets(path):
    # Determine the absolute path to the assets directory
    # Assumes api/index.py is one level deeper than root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = os.path.join(base_dir, 'assets')
    return send_from_directory(assets_dir, path)

# For local testing
if __name__ == '__main__':
    app.run(debug=True)
