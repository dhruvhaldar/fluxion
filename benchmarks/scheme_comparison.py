from fluxion.models import AdvectionDiffusion
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    os.makedirs('assets', exist_ok=True)
    print("Running Scheme Comparison Benchmark...")
    AdvectionDiffusion.compare_schemes(
        schemes=['upwind', 'central', 'quick'],
        save_path='assets/scheme_comparison.png'
    )
    print("Scheme comparison plot saved to assets/scheme_comparison.png")
