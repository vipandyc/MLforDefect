import re
import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# Atomic number dictionary
atomic_numbers = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56
}
num_to_symbol = {v: k for k, v in atomic_numbers.items()}
num_to_symbol[0] = 'Vac'

MAX_LENGTH = 100
VECTOR_LENGTH = len(atomic_numbers) + 1  # 56+1 for vacancy
DOS_FACTOR = 28
DOPE_FACTOR = 10
Difficulty = 0.5 # higher means higher probability of adding more distractive dopants

# Function to parse formula into vector representation
def parse_formula(formula):
    element_counts = {}
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    for element, count in matches:
        count = int(count) if count else 1
        element_counts[element] = element_counts.get(element, 0) + count

    return element_counts

# Function to convert formula to vector
def formula_to_vector(formula_counts):
    vector = np.zeros(VECTOR_LENGTH, dtype=np.float64)
    total_atoms = sum(formula_counts.values())

    for element, count in formula_counts.items():
        if element in atomic_numbers:
            vector[atomic_numbers[element]] = count / total_atoms

    return vector, total_atoms

# Function to process dataset
def process_entry(row):
    # Process formula_j0
    formula_j0_counts = parse_formula(row["formula_j0"])
    formula_j0_vector, total_atoms_j0 = formula_to_vector(formula_j0_counts)

    # Process formula_j
    formula_j_counts = parse_formula(row["formula_j"])
    formula_j_vector, total_atoms_j = formula_to_vector(formula_j_counts)

    # Compute elemental difference and vacancy ratio
    diff_vector = np.zeros(VECTOR_LENGTH, dtype=np.float64)
    
    for element, count in formula_j_counts.items():
        if element in atomic_numbers and element not in formula_j0_counts:
            diff_vector[atomic_numbers[element]] = count / total_atoms_j0
    #print(diff_vector)
    
    vacancy_count = total_atoms_j0 - total_atoms_j
    diff_vector[0] = vacancy_count / total_atoms_j0
    diff_vector = diff_vector * DOPE_FACTOR

    # Process eigvals_j0 and eigvals_j
    dos_j0, _ = np.histogram(np.real(row["eigvals_j0"]), bins=100, density=True)
    # align dos_j0 with dos_j, facilitate substraction
    dos_j, _ = np.histogram(np.real(row["eigvals_j"]), bins=100, range=(np.real(row["eigvals_j0"]).min(), \
                                                                np.real(row["eigvals_j0"]).max()), density=True)
    
    # (Optional) Smooth the DOS using Gaussian filter
    # dos_j0 = gaussian_filter1d(dos_j0, sigma=1.0)
    # dos_j = gaussian_filter1d(dos_j, sigma=1.0)

    dos_j0 = torch.tensor(dos_j0, dtype=torch.float64) / DOS_FACTOR
    # dos_j = torch.tensor(dos_j, dtype=torch.float64) / DOS_FACTOR
    dos_j = torch.tensor(np.log10((dos_j / DOS_FACTOR + 1e-2) / (dos_j0 + 1e-2)), dtype=torch.float64)

    # Process dopant_options
    dopant_vector = np.zeros(VECTOR_LENGTH, dtype=np.float64)
    dopant_vector[0] = 1 # vacancy always possible
    dopants = row["dopant_options_j0"]
    for dopant in dopants:
        if dopant in atomic_numbers:
            if diff_vector[atomic_numbers[dopant]] == 0:
                if np.random.rand() < Difficulty:
                    dopant_vector[atomic_numbers[dopant]] = 1
            else:
                dopant_vector[atomic_numbers[dopant]] = 1

    return dos_j0, dos_j, torch.tensor(dopant_vector), F.pad(torch.tensor(formula_j0_vector, dtype=torch.float64),\
                                               (0, MAX_LENGTH-VECTOR_LENGTH), value=0.0), torch.tensor(diff_vector)

class MaterialsDataset(Dataset):
    def __init__(self, dataframe):
        self.data = [process_entry(row) for _, row in tqdm(dataframe.iterrows(), total=len(dataframe))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dos_j0, dos_j, dopant_vector, formula_j0_vector, diff_vector = self.data[idx]

        # Convert to tensors
        dos_j0 = torch.tensor(dos_j0, dtype=torch.float32)  # Original DOS
        dos_j = torch.tensor(dos_j, dtype=torch.float32)  # Modified DOS
        dopant_vector = torch.tensor(dopant_vector, dtype=torch.float32)  # Binary dopant selection
        formula_j0_vector = torch.tensor(formula_j0_vector, dtype=torch.float32)  # Extra stacked channel
        diff_vector = torch.tensor(diff_vector, dtype=torch.float32)  # Output concentration

        return dos_j0, dos_j, dopant_vector, formula_j0_vector, diff_vector