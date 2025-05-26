import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io.vasp import write_vasp
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from mace.calculators import mace_mp

def main():
    # Create calculator
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device='cpu')

    # Build a 6x6x6 supercell of silicon
    unit_cell = bulk('Si', 'diamond', a=5.43)
    supercell = unit_cell * (1, 1, 1)

    # Assign calculator and perform relaxation
    supercell.calc = calc
    dyn = BFGS(supercell)
    dyn.run(fmax=0.05)

    # Run vibrational analysis
    vib = Vibrations(supercell, name="vib_pristine")
    vib.clean()
    vib.run()

    # Extract and save eigenvalues
    eigenvalues = vib.get_energies()
    np.savetxt("pristine_eigenvalues.txt", eigenvalues)

    print("Phonon DOS eigenvalues have been saved to 'pristine_eigenvalues.txt'.")
    vib.summary()

if __name__ == "__main__":
    main()
