import re

def fix():
    with open('fluxion/solvers.py', 'r') as f:
        content = f.read()

    # Remove the unused buf allocations
    search = """        # ⚡ Bolt: Pre-allocate contiguous buffers for strided slices to avoid implicit non-contiguous allocations
        buf1 = np.empty(p_slice[s1].shape)
        buf2 = np.empty(p_slice[s2].shape)
        buf3 = np.empty(p_slice[s3].shape)
        buf4 = np.empty(p_slice[s4].shape)"""
    content = content.replace(search, "")

    # Fix the contradictory comments
    old_comment = """                # ⚡ Bolt: Use contiguous pre-allocated buffers and chained in-place operations
                # to eliminate non-contiguous implicit intermediate arrays during strided slicing."""
    new_comment = """                # ⚡ Bolt: Evaluate standard vectorized expressions for strided sub-grids.
                # Python wrapper overhead from chained in-place operations (e.g. np.add(..., out=buf))
                # is slower than implicit non-contiguous array allocations in this context."""

    content = content.replace(old_comment, new_comment)

    with open('fluxion/solvers.py', 'w') as f:
        f.write(content)

fix()
