import pandas as pd
import re
import argparse
# Load the CSV file into a DataFrame
df = pd.read_csv("../../Defect_combinations.csv")
MAX_STEPS = 200

atomic_numbers = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56
}

# Function to extract elements and filter based on atomic number
def filter_by_atomic_number(crystal):
    # Extract the chemical formula (part before the underscore)
    formula = crystal.split('_')[0]
    # Find all element symbols using regex
    elements = re.findall(r'[A-Z][a-z]?', formula)
    # Check if any element has an atomic number > 53
    return all(element in atomic_numbers for element in elements)

def extract_element(species_string):
    # Use regex to find the element symbol
    return re.findall(r"Species ([A-Z][a-z]?)", species_string)

# Filter rows where doping columns do not contain "None"
df = df[
    ~df["n_dopant"].str.contains("None") &
    ~df["n_original"].str.contains("None") &
    ~df["p_dopant"].str.contains("None") &
    ~df["p_original"].str.contains("None")
]
df["n_dopant"] = df["n_dopant"].apply(extract_element)
df["n_original"] = df["n_original"].apply(extract_element)
df["p_dopant"] = df["p_dopant"].apply(extract_element)
df["p_original"] = df["p_original"].apply(extract_element)

# Filter rows based on atomic number
filtered_df = df[df["Crystal"].apply(filter_by_atomic_number)]

def find_closest_factors(N, target_range=(333, 500)):
    target_min, target_max = target_range
    # Start with cube root approximation for balanced factors
    base = int((target_max / N) ** (1/3)) + 1
    best_combination = None
    smallest_deviation = float('inf')

    for x in range(base, 0, -1):
        for y in range(x, 0, -1):  # Ensure y ≤ x
            for z in range(y, 0, -1):  # Ensure z ≤ y
                product = N * x * y * z
                if target_min <= product <= target_max:
                    # Compute deviation as the sum of absolute differences
                    deviation = abs(x - y) + abs(y - z) + abs(z - x)
                    if deviation < smallest_deviation:
                        smallest_deviation = deviation
                        best_combination = (x, y, z, product)

    return best_combination

def combinations(input_file):
    row = filtered_df[filtered_df["Crystal"] == input_file]
    
    if row.empty:
        raise ValueError(f"No entry found for crystal structure {input_file}.")
    
    # Extract dopants and originals
    n_dopant = row["n_dopant"].iloc[0]
    n_original = row["n_original"].iloc[0]
    p_dopant = row["p_dopant"].iloc[0]
    p_original = row["p_original"].iloc[0]
    actions = {}
    for i in range(len(n_dopant)):
        if n_original[i] not in actions:
            actions[n_original[i]] = []
        actions[n_original[i]].append(n_dopant[i])
        if p_original[i] not in actions:
            actions[p_original[i]] = []
        actions[p_original[i]].append(p_dopant[i])
    return actions

import numpy as np
import json
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from mace.calculators import mace_mp
from ase.io import read

def run_MACE(input_file, job_id, vacancy_ratio=0.01, doping_ratio=0.01):
    # Create calculator
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda')
    
    # Load the JSON file and extract the POSCAR string
    with open('../../SM_dataset/'+input_file, 'r') as f:
        data = json.load(f)
    
    poscar_content = data.get("Structure_rlx")
    with open("temp_poscar.vasp", 'w') as temp_file:
        temp_file.write(poscar_content)

    try:
        atoms = read("temp_poscar.vasp")
    except Exception: # error reading?
        return None
    # pre-run
    atoms.calc = calc
    dyn_pre = BFGS(atoms)
    dyn_pre.run(fmax=0.005, steps=MAX_STEPS)
    if dyn_pre.nsteps >= MAX_STEPS - 1:
        # not converged!
        print(f'{input_file} not converged.')
        return 'GEEE'

    # Calculate the number of atoms and determine scaling factor for ~400-500 atoms
    num_atoms = len(atoms)
    x, y, z, product = find_closest_factors(num_atoms, target_range=(333, 500))
    assert 333 <= product <= 500, 'Invalid supercell!'
    supercell = atoms * (x, y, z)
    chem_formula_pristine = supercell.get_chemical_formula()

    # Then dope it
    # Apply vacancies
    num_to_remove = int(vacancy_ratio * len(supercell))
    num_to_dope = int(doping_ratio * len(supercell))

    indices_to_remove = np.random.choice(len(supercell), num_to_remove, replace=False)
    indices_to_remove = sorted(indices_to_remove, reverse=True)
    for index in indices_to_remove:
        del supercell[index]
    
    actions = combinations(input_file)

    # Apply doping with one-to-one mapping for substitutions
    indices_to_dope = np.random.choice(len(supercell), num_to_dope, replace=False)
    doping_pairs = []
    for i in indices_to_dope:
        if supercell[i].symbol in actions:
            doping_pairs.append((supercell[i].symbol, np.random.choice(actions[supercell[i].symbol])))

    for index, (original, substitute) in zip(indices_to_dope, doping_pairs):
        if substitute not in atomic_numbers:
            continue
        # print(f"Substituting {original} at index {index} with {substitute}")
        supercell[index].symbol = substitute

    print(f"Number of atoms after doping: {len(supercell)}")
    chem_formula_defect = supercell.get_chemical_formula()
    print(f"Chemical Formula: {chem_formula_defect}")

    # Assign calculator and perform relaxation
    supercell.calc = calc
    dyn = BFGS(supercell)
    dyn.run(fmax=0.05, steps=MAX_STEPS)
    if dyn.nsteps >= MAX_STEPS - 1:
        # not converged!
        print(f'{chem_formula_defect} not converged.')
        return 'GEEE'

    # Run vibrational analysis
    vib = Vibrations(supercell, name="vib_defects")
    vib.clean()
    vib.run()

    # Extract and save eigenvalues
    eigenvalues = vib.get_energies()
    np.save(f"eigvals_{job_id}_{chem_formula_defect}.npy", \
            eigenvalues)

    print("Phonon DOS eigenvalues have been saved.")
    vib.summary()
    vib.clean()


parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, required=True)
args = parser.parse_args()
indices_total = np.load('../../Simulation_batch_indices.npy')
indices = indices_total[args.batch - 1]
print(indices)
for i in indices:
    vacancy_ratio = 0
    doping_ratio = 0
    run_MACE(filtered_df['Crystal'].iloc[i], job_id=f'{i+1}_0', \
                vacancy_ratio=vacancy_ratio, doping_ratio=doping_ratio)