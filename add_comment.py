with open('fluxion/models/navier_stokes.py', 'r') as f:
    content = f.read()

content = content.replace("nu_inv_dx2 = self.nu * inv_dx2", "# ⚡ Bolt: Algebraically factor nu into grid constants to save full-array multiplications later\n        nu_inv_dx2 = self.nu * inv_dx2")

with open('fluxion/models/navier_stokes.py', 'w') as f:
    f.write(content)
