from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_prediction.dopant_predictor import (
    get_dopants_from_shannon_radii,
    get_dopants_from_substitution_probabilities
)
from pymatgen.ext.matproj import MPRester
from pymatgen.io.vasp import Poscar

num_dopants = 5 # n, p doping each 5

import os
import json
from tqdm import tqdm
import pandas as pd

directory = 'SM_dataset'  # Directory containing JSON files

def give_doping_recommendation(structure):
    structure.add_oxidation_state_by_guess()

    try: # try substitution probability first
        threshold = 1e-3  # probability threshold for substitution/structure predictions
        dopants = get_dopants_from_substitution_probabilities(
            structure, num_dopants=num_dopants, threshold=threshold
        )
    except ValueError: # if doesn't work here, e.g. alloys, try shannon radii matching
        try:
            cnn = CrystalNN()
            bonded_structure = cnn.get_bonded_structure(structure)
            dopants = get_dopants_from_shannon_radii(bonded_structure, num_dopants=num_dopants)
        except Exception as e:
            # if still doesn't work, for example single element compound have no oxidation number
            # then return nothing
            dopants = None
    return dopants

data_list = []

for filename in tqdm(os.listdir(directory)):
    # Construct the full path to the file
    file_path = os.path.join(directory, filename)
    
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)
        structure_string = data["Structure_rlx"]
        ICSD = data["ICSD_number"]

    structure_string = '\n'+structure_string.split('\n', 1)[1]

    # Use pymatgen to parse the structure
    poscar = Poscar.from_str(structure_string, read_velocities=False)
    # poscar.write_file('./POSCAR')
    structure = poscar.structure
    if sum(poscar.natoms) >= 25:
        dopants = None
    else: # return a dictionary from pymatgen format
        dopants = give_doping_recommendation(structure)
    
    # convert to unified dataframe
    if dopants is not None:
        # Extract the first 5 dopants for n_doping and p_doping, if available
        n_dopants = [dopants['n_type'][i]['dopant_species'] if i < len(dopants['n_type']) else None for i in range(5)]
        n_originals = [dopants['n_type'][i]['original_species'] if i < len(dopants['n_type']) else None for i in range(5)]
        p_dopants = [dopants['p_type'][i]['dopant_species'] if i < len(dopants['p_type']) else None for i in range(5)]
        p_originals = [dopants['p_type'][i]['original_species'] if i < len(dopants['p_type']) else None for i in range(5)]
    else:
        n_dopants, n_originals = [None] * 5, [None] * 5
        p_dopants, p_originals = [None] * 5, [None] * 5
    
    data_list.append({
        "ICSD": ICSD,
        "Crystal": filename,
        "n_dopant": n_dopants,
        "n_original": n_originals,
        "p_dopant": p_dopants,
        "p_original": p_originals,
    })

df = pd.DataFrame(data_list)

df.to_csv('Defect_combinations.csv')
df.to_pickle('Defect_combinations.pkl')