import glob
import os
import pickle
import re
from os.path import splitext

import numpy as np
import pandas as pd
from openbabel import pybel

import isicle
from isicle.interfaces import FileParserInterface


class ORCAParser(FileParserInterface):
    """Extract information from an ORCA simulation output files."""

    def __init__(self, data=None):
        self.data = data

        self.result = {}

    def load(self, path):
        self.data = isicle.io.load_pickle(path)

    def _find_output_by_header(self, header):
        # Fat regex
        pattern = (
            r"(-{2,})\n\s{0,}("
            + re.escape(header)
            + ")\s{0,}\n-{2,}\n([\s\S]*?)(?=-{2,}\n[^\n]*\n-{2,}\n|$)"
        )

        # Search ORCA output file
        matches = re.findall(pattern, self.data["out"])

        # Return "body" of each match
        return [x[2].strip() for x in matches]

    def _parse_protocol(self):
        return self.data["inp"]

    def _parse_geom(self):
        return self.data["xyz"]

    def _parse_energy(self):
        # Split text
        lines = self.data["property"].split("\n")

        # Search for energy values
        elines = [x for x in lines if "Total DFT Energy" in x]

        # Energy values not found
        if len(elines) == 0:
            return None

        # Map float over values
        evals = [float(x.split()[-1].strip()) for x in elines]

        # Return last energy value
        return evals[-1]

    def _parse_frequency(self):
        # Define columns
        columns = ["wavenumber", "eps", "intensity", "TX", "TY", "TZ"]

        # Split sections by delimiter
        blocks = self.data["hess"].split("$")

        # Search for frequency values
        freq_block = [x for x in blocks if x.startswith("ir_spectrum")]

        # Frequency values not found
        if len(freq_block) == 0:
            return None

        # Grab last frequency block
        # Doubtful if more than one, but previous results in list
        freq_block = freq_block[-1]

        # Split block into lines
        lines = freq_block.split("\n")

        # Map float over values
        vals = np.array(
            [
                list(map(float, x.split()))
                for x in lines
                if len(x.split()) == len(columns)
            ]
        )

        # Zip columns and values
        return dict(zip(columns, vals.T))

    def _parse_timing(self):
        # Grab only last few lines
        lines = self.data["out"].split("\n")[-100:]

        # Find start of timing section
        parts = []
        start_idx = None
        for i, line in enumerate(lines):
            if line.startswith("Timings for individual modules"):
                start_idx = i + 2

            # Strip out extraneous info
            parts.append(
                [x.strip() for x in line.split("  ") if x and x.strip() != "..."]
            )

        # Timing not found
        if start_idx is None:
            return None

        # Split out timing section
        tlines = lines[start_idx:]
        tparts = parts[start_idx:]

        # Individual timings
        timings = [x for x in tparts if any([" sec " in y for y in x])]
        timings = {x[0].strip("..."): float(x[1].split()[0]) for x in timings}

        # Boolean indication of success
        success = len([x for x in tlines if "ORCA TERMINATED NORMALLY" in x]) > 0
        timings["success"] = success

        # Total time
        total_time = [x for x in tlines if "TOTAL RUN TIME" in x]

        if len(total_time) > 0:
            total_time = total_time[-1].split(":")[-1].strip()
            times = list(map(int, total_time.split()[::2]))
            units = total_time.split()[1::2]
        else:
            total_time = None

        timings["Total run time"] = dict(zip(units, times))

        return timings

    def _parse_shielding(self):
        # Filter comments
        property = [
            x.strip()
            for x in self.data["property"].split("\n")
            if not x.startswith("#")
        ]
        property = "\n".join(property)

        # Split sections by delimiter
        blocks = property.split("$ ")

        # Search for shielding values
        shielding_block = [x for x in blocks if x.startswith("EPRNMR_OrbitalShielding")]

        # Shielding values not found
        if len(shielding_block) == 0:
            return None

        # Grab last shielding block
        # Doubtful if more than one, but previous results in list
        shielding_block = shielding_block[-1]

        # Define a pattern for extracting relevant information
        pattern = re.compile(
            r"Nucleus: (\d+) (\w+)\n(Shielding tensor.*?P\(iso\) \s*[-+]?\d*\.\d+)",
            re.DOTALL,
        )

        # Match against pattern
        matches = pattern.findall(shielding_block)

        # Result container
        shielding = {}

        # Enumerate matches
        for match in matches:
            # Per-nucleus info
            nucleus_index = match[0]
            nucleus_name = match[1]
            nucleus_data = match[2]

            # Extracting values using regex
            tensors = re.findall(r"(-?\d+\.\d+|-?\d+.\d+e[+-]\d+)", nucleus_data)
            tensors = [float(val) for val in tensors]

            # Creating arrays from extracted values
            shielding_tensor = np.array(tensors[:9]).reshape(3, 3)
            p_tensor_eigenvectors = np.array(tensors[9:18]).reshape(3, 3)
            p_eigenvalues = np.array(tensors[18:21])
            p_iso = float(tensors[21])

            # Constructing the dictionary with nuclei index and name
            shielding[f"{nucleus_index}{nucleus_name}"] = {
                "shielding tensor": shielding_tensor,
                "P tensor eigenvectors": p_tensor_eigenvectors,
                "P eigenvalues": p_eigenvalues,
                "P(iso)": p_iso,
            }

        # Add shielding summary
        shielding["shielding_summary"] = self._parse_shielding_summary()

        return shielding

    def _parse_orbital_energies(self):
        header = "ORBITAL ENERGIES"
        text = self._find_output_by_header(header)

        # Orbital energies not found
        if len(text) == 0:
            return None

        # Get last relevant output
        text = text[-1].split("\n")

        # Parse table
        text = [x.strip() for x in text if x.strip() != "" and "*" not in x]
        columns = text[0].split()
        body = [x.split() for x in text[1:]]

        # Construct data frame
        df = pd.DataFrame(body, columns=columns, dtype=float)

        # Map correct types
        df["NO"] = df["NO"].astype(int)

        # Drop unoccupied orbitals?
        return df

    def _parse_spin(self):
        header = "SUMMARY OF ISOTROPIC COUPLING CONSTANTS (Hz)"
        text = self._find_output_by_header(header)

        # Spin couplings not found
        if len(text) == 0:
            return None

        # Get last relevant output
        text = text[-1].split("\n")

        # Parse table
        text = [x.strip() for x in text if x.strip() != "" and "*" not in x]
        columns = [x.replace(" ", "") for x in re.split("\s{2,}", text[0])]
        body = [re.split("\s{2,}", x)[1:] for x in text[1:-1]]

        # Construct data frame
        return pd.DataFrame(body, dtype=float, columns=columns, index=columns)

    def _parse_shielding_summary(self):
        header = "CHEMICAL SHIELDING SUMMARY (ppm)"
        text = self._find_output_by_header(header)

        # Shielding values not found
        if len(text) == 0:
            return None

        # Get last relevant output
        text = text[-1].split("\n")

        # Parse table
        text = [x.strip() for x in text if x.strip() != ""]

        # Find stop index
        stop_idx = -1
        for i, row in enumerate(text):
            if all([x == "-" for x in row]):
                stop_idx = i
                break

        # Split columns and body
        columns = text[0].split()
        body = [x.split() for x in text[2:stop_idx]]

        # Construct data frame
        df = pd.DataFrame(body, columns=columns)

        # Map correct types
        for col, dtype in zip(df.columns, (int, str, float, float)):
            df[col] = df[col].astype(dtype)
        return df

    def _parse_thermo(self):
        # In hessian file
        header = "THERMOCHEMISTRY_Energies"

    def _parse_molden(self):
        return None

    def _parse_charge(self):
        return None

    def _parse_connectivity(self):
        return None

    def parse(self):
        result = {
            "protocol": self._parse_protocol(),
            "geom": self._parse_geom(),
            "total_dft_energy": self._parse_energy(),
            "orbital_energies": self._parse_orbital_energies(),
            "shielding": self._parse_shielding(),
            "spin": self._parse_spin(),
            "frequency": self._parse_frequency(),
            "molden": self._parse_molden(),
            "charge": self._parse_charge(),
            "timing": self._parse_timing(),
            "connectivity": self._parse_connectivity(),
        }

        # Pop success from timing
        if result["timing"] is not None:
            result["success"] = result["timing"].pop("success")
        else:
            result["success"] = False

        # Filter empty fields
        result = {k: v for k, v in result.items() if v is not None}

        # Store attribute
        self.result = result

        return result

    def save(self, path):
        isicle.io.save_pickle(path, self.result)

class GaussianParser(FileParserInterface):
    '''Extract text from a Gaussian simulation output file.'''

    def __init__(self):
        self.contents = None
        self.result = None
        self.path = None

    def load(self, path: str):
        '''Load in the data file'''
        with open(path, 'r') as f:
            self.contents = f.readlines()
        self.path = path
        return self.contents
    
    def _parse_protocol(self):
        protocol = []
        for idx, line in enumerate(self.contents):
            if line.startswith(' # '):
                for idx2, line2 in enumerate(self.contents[idx:]):
                    if line2.startswith(' ---'):
                        break
                    else:
                        protocol.append(line2)
                break
        protocol = ''.join([x.strip() for x in protocol])
        
        return protocol

    def _parse_geometry(self):
        #Compile list of indices associated with empty lines in the file
        #Will help grab block of text at end of file.
        line_list = [idx for idx, line in enumerate(self.contents) if 'Input orientation' in line]
        first_geom = line_list[0] + 5
        last_geom = line_list[-1] + 5
   
        atomic_masses = isicle.utils.atomic_masses()

        def get_geom(idx):
            geom = []
            for line in self.contents[idx:]:
                if line.startswith(' -----'):
                    break
                else:
                    geom.append(line)

            #text editing to make readable
            for idx, line in enumerate(geom):
                split_line = line.split()
                geom[idx]=[atomic_masses.Symbol[int(split_line[1])-1], float(split_line[3]), float(split_line[4]), float(split_line[5])]

            return geom

        self.first = get_geom(first_geom)
        self.last = get_geom(last_geom)

        xyz = [[x[0],str(x[1]), str(x[2]), str(x[3])] for x in self.last]
        xyz = ['\t'.join(x) for x in xyz]

        xyz_block = str(len(xyz))+'\n\n'
        xyz = '\n'.join(xyz)
        xyz_block = xyz_block + xyz

        xyz_path = self.path.split('.')[0] + '.xyz'
        with open(xyz_path, 'w+') as f:
            f.write(xyz_block)
        f.close()

        geom = isicle.load(xyz_path)
        os.remove(xyz_path)

        return geom

    def _parse_rms(self):
        xyz1 = [x[1:] for x in self.first]
        xyz2 = [y[1:] for y in self.last]

        rms = np.sqrt(np.mean((np.array(xyz1) - np.array(xyz2)) ** 2))

        return rms

    def _parse_forces(self):
        line_list = [idx for idx, line in enumerate(self.contents) if 'Forces (Hartrees/Bohr)' in line]
        line_list = [idx+3 for idx in line_list]
        atomic_masses = isicle.utils.atomic_masses()
   
        def get_forces(idx):
            forces = []
            for line in self.contents[idx:]:
                if line.startswith(' -----'):
                    break
                else:
                    forces.append(line)

            #text editing to make readable
            for idx, line in enumerate(forces):
                split_line = line.split()
                forces[idx]=[atomic_masses.Symbol[int(split_line[1])-1], float(split_line[2]), float(split_line[3]), float(split_line[4])]

            return forces

        forces = []
        for idx in line_list:
            forces.append(get_forces(idx))

        return forces

    def _parse_frequency(self):
        
        frequency_list = []

        for line in self.contents:
            if 'Frequencies' in line:
                freq = line.split()[2:]
                for i in freq:
                    frequency_list.append(i)        

        return frequency_list

    def _parse_energy(self):

        scf = None
        zpe = None
        internal = None
        enthalpy = None
        gibbs_free = None

        for line in self.contents:

            if ' SCF Done: ' in line:
                scf = float(line.split()[4])

            if 'Sum of electronic and zero-point' in line:
                zpe = float(line.split()[6])

            if "Sum of electronic and thermal Energies" in line:
                internal = float(line.split()[6])

            if "Sum of electronic and thermal Enthalpies" in line:
                enthalpy = float(line.split()[6])

            if "Sum of electronic and thermal Free Energies" in line:
                gibbs_free = float(line.split()[7])

        d = {'SCF': scf,
             'ZPE': zpe,
             'Interal': internal,
             'Enthalpy': enthalpy,
             'Gibbs': gibbs_free}

        return d

    def _parse_shielding(self):
        return

    def _parse_spin(self):
        return

    def _parse_charge(self):

        mull_str = 'Mulliken charges:'
        charges = []

        for idx, line in enumerate(self.contents):

            if mull_str in line:
                start_idx = idx + 2
                for id, chrge in enumerate(self.contents[start_idx:]):
                    charge = chrge.split()
                    if len(charge) > 3:
                        break
                    else:
                        charges.append(charge[-1])
        return charges

    def _parse_polarity(self):
        # Dipole
        dipole_str = 'Electric dipole moment'
        for idx,line in enumerate(self.contents):
            if dipole_str in line:
                total_dipole = float(self.contents[idx+3].split()[2].replace("D", "E"))
                x_dipole = self.contents[idx+4].split()[2]
                y_dipole = self.contents[idx+5].split()[2]
                z_dipole = self.contents[idx+6].split()[2]

                isotropic_polar = float(self.contents[idx+12].split()[2].replace("D", "E"))
                anisotropic_polar = float(self.contents[idx+13].split()[2].replace("D", "E"))
                break
        
        d = {'Dipole (Debye)': total_dipole,
             'Isotropic Polarizability (A^3)': isotropic_polar,
             'Anisotropic Polarizability (A^3)': anisotropic_polar}
        
        return d

    def _parse_timing(self):
        timing = []
        for idx, line in enumerate(self.contents):
            if 'Job cpu time' in line:
                t = line.split()
                time = ':'.join([t[3], t[5], t[7], t[9]])
                timing.append(time)
            if 'Elapsed time' in line:
                t = line.split()
                time = ':'.join([t[3], t[5], t[7], t[9]])
                timing.append(time)
        if len(timing) == 4:
            d = {'geometry optimization time': {'Job cpu time': timing[0],
                                           'Elapsed time': timing[1]},
                 'frequency time': {'Job cpu time': timing[2],
                               'Elapsed time': timing[3]}}
        else:
            d = {'geometry optimization time': {'Job cpu time': timing[0],
                                           'Elapsed time': timing[1]}}
        return d

    def parse(self):
        '''
        Extract relevant information from Gaussian output

        Parameters
        ----------
        to_parse : list of str
            geometry, energy, shielding, spin, frequency, charge, timing 
        '''

        # Check that the file is valid first
        if len(self.contents) == 0:
            raise RuntimeError('No contents to parse: {}'.format(self.path))
        if 'Normal termination' not in self.contents[-1]:
            raise RuntimeError('Incomplete Gaussian run: {}'.format(self.path))

        # Initialize result object to store info
        result = {}

        try:
            result['protocol'] = self._parse_protocol()
        except:
            pass
        
        try:
            result['geom'] = self._parse_geometry()
        except:
            pass
        
        try:
            result['rms'] = self._parse_rms()
        except:
            pass    

        try:
            result['forces'] = self._parse_forces()
        except:
            pass
        try:
            result['energy'] = self._parse_energy()
        except:
            pass

        try:
            result['shielding'] = self._parse_shielding()
        except:  
            pass

        try:
            result['spin'] = self._parse_spin()
        except:
            pass

        try:
            result['frequency'] = self._parse_frequency()
        except:
            pass

        try:
            result['charge'] = self._parse_charge()
        except:
            pass

        try:
            result['polarity'] = self._parse_polarity()
        except:
            pass

        try:
            result['timing'] = self._parse_timing()
        except:
            pass
        
        return result

    def save(self):
        pass

class NWChemParser(FileParserInterface):
    """
    Extract text from an NWChem simulation output file.
    """

    def __init__(self):
        self.contents = None
        self.result = None
        self.path = None

    def load(self, path: str):
        """
        Load in the data file
        """
        with open(path, "r") as f:
            self.contents = f.readlines()
        self.path = path
        return self.contents

    def _parse_geometry(self):
        """
        Parse geometry either from XYZ files generated by geometry optimization or 
        from within the NWChem output file.
        """
        search = os.path.dirname(self.path)
        geoms = sorted(glob.glob(os.path.join(search, "*.xyz")))

        if len(geoms) > 1:

            def to_xyz_list(fname):
                geom = isicle.io.load(fname)
                xyz = geom.to_xyzblock().split('\n')
                xyz = [x.split() for x in xyz]
                xyz = [[x[0], float(x[1]), float(x[2]), float(x[3])] for x in xyz]
                return xyz
                
            self.first = to_xyz_list(geoms[0])
            self.last = to_xyz_list(geoms[-1])
            return isicle.io.load(geoms[-1])

        else:
            line_list = [idx for idx, line in enumerate(self.contents) if "Geometry \"geometry\" -> " in line]
            first_geom = line_list[0] + 7
            last_geom = line_list[-1] + 7

            def get_geom(idx):
                geom = []
                for line in self.contents[idx:]:
                    if line == '\n':
                        break
                    else:
                        geom.append(line)

                #text editing to make readable
                for idx, line in enumerate(geom):
                    split_line = line.split()
                    geom[idx]=[split_line[1], float(split_line[3]), float(split_line[4]), float(split_line[5])]

                return geom

            self.first = get_geom(first_geom)
            self.last = get_geom(last_geom)

            xyz = [[x[0],str(x[1]), str(x[2]), str(x[3])] for x in self.last]
            xyz = ['\t'.join(x) for x in xyz]

            xyz_block = str(len(xyz))+'\n\n'
            xyz = '\n'.join(xyz)
            xyz_block = xyz_block + xyz

            xyz_path = self.path.split('.')[0] + '.xyz'
            with open(xyz_path, 'w+') as f:
                f.write(xyz_block)
            f.close()

            geom = isicle.load(xyz_path)
            os.remove(xyz_path)

    def _parse_rms(self):
        xyz1 = [x[1:] for x in self.first]
        xyz2 = [y[1:] for y in self.last]

        rms = np.sqrt(np.mean((np.array(xyz1) - np.array(xyz2)) ** 2))

        return rms

    def _parse_energy(self):
        """
        Parse DFT energies from each step in the geometry optimization.
        """

        # Init
        energy = []

        # Cycle through file
        for line in self.contents:
            if "Total DFT energy" in line:
                # Overwrite last saved energy
                energy.append(float(line.split()[-1]))

        return energy


    def _parse_forces(self):
        line_list = [idx for idx, line in enumerate(self.contents) if 'GRADIENTS' in line]
        line_list = [idx+4 for idx in line_list]
        atomic_masses = isicle.utils.atomic_masses()
   
        def get_forces(idx):
            forces = []
            for line in self.contents[idx:]:
                if line == '\n':
                    break
                else:
                    forces.append(line)

            #text editing to make readable
            for idx, line in enumerate(forces):
                split_line = line.split()
                forces[idx]=[split_line[1], float(split_line[5]), float(split_line[6]), float(split_line[7])]

            return forces

        forces = []
        for idx in line_list:
            forces.append(get_forces(idx))

        return forces


    def _parse_shielding(self):
        """
        Add docstring
        """
        # Init
        ready = False
        shield_idxs = []
        shield_atoms = []
        shields = []

        for line in self.contents:
            if " SHIELDING" in line:
                shield_idxs = [int(x) for x in line.split()[2:]]
                if len(shield_idxs) == 0:
                    collect_idx = True

            if "Atom:" in line:
                atom = line.split()[2]
                idx = line.split()[1]
                ready = True

            elif "isotropic" in line and ready is True:
                shield = float(line.split()[-1])
                shield_atoms.append(atom)
                shields.append(shield)
                if collect_idx is True:
                    shield_idxs.append(int(idx))

        if len(shields) > 1:
            return {"index": shield_idxs, "atom": shield_atoms, "shielding": shields}

        raise Exception

    def _parse_spin(self):
        """
        Add docstring
        """
        # TO DO: Add g-factors

        # Declaring couplings
        coup_pairs = []
        coup = []
        index = []
        g_factor = []
        ready = False

        for line in self.contents:
            if "Atom  " in line:
                line = line.split()
                idx1 = int((line[1].split(":"))[0])
                idx2 = int((line[5].split(":"))[0])
                ready = True
            elif "Isotropic Spin-Spin Coupling =" in line and ready is True:
                coupling = float(line.split()[4])
                coup_pairs.append([idx1, idx2])
                coup.append(coup)
                ready = False
            elif "Respective Nuclear g-factors:" in line:
                line = line.split()
                if idx1 not in index:
                    index.append(idx1)
                    g = float(line[3])
                    g_factor.append(g)
                if idx2 not in index:
                    index.append(idx2)
                    g = float(line[5])
                    g_factor.append(g)

        return {
            "pair indices": coup_pairs,
            "spin couplings": coup,
            "index": index,
            "g-tensors": g_factor,
        }

    def _parse_frequency(self):
        """
        Add docstring
        """
        # TO DO: Add freq intensities
        # TO DO: Add rotational/translational/vibrational Cv and entropy
        freq = None
        zpe = None
        enthalpies = None
        entropies = None
        capacities = None
        temp = None
        scaling = None
        natoms = None
        has_frequency = None

        for i, line in enumerate(self.contents):
            if ("geometry" in line) and (natoms is None):
                atom_start = i + 7
            if ("Atomic Mass" in line) and (natoms is None):
                atom_stop = i - 2
                natoms = atom_stop - atom_start + 1
            if "Normal Eigenvalue" in line:
                has_frequency = True
                freq_start = i + 3
                freq_stop = i + 2 + 3 * natoms

            # Get values
            if "Zero-Point correction to Energy" in line:
                zpe = line.rstrip().split("=")[-1]

            if "Thermal correction to Enthalpy" in line:
                enthalpies = line.rstrip().split("=")[-1]

            if "Total Entropy" in line:
                entropies = line.rstrip().split("=")[-1]

            if "constant volume heat capacity" in line:
                capacities = line.rstrip().split("=    ")[-1]

        if has_frequency is True:
            freq = np.array(
                [float(x.split()[1]) for x in self.contents[freq_start : freq_stop + 1]]
            )
            intensity_au = np.array(
                [float(x.split()[3]) for x in self.contents[freq_start : freq_stop + 1]]
            )
            intensity_debyeangs = np.array(
                [float(x.split()[4]) for x in self.contents[freq_start : freq_stop + 1]]
            )
            intensity_KMmol = np.array(
                [float(x.split()[5]) for x in self.contents[freq_start : freq_stop + 1]]
            )
            intensity_arbitrary = np.array(
                [float(x.split()[6]) for x in self.contents[freq_start : freq_stop + 1]]
            )

            return {
                "frequencies": freq,
                "intensity atomic units": intensity_au,
                "intensity (debye/angs)**2": intensity_debyeangs,
                "intensity (KM/mol)": intensity_KMmol,
                "intensity arbitrary": intensity_arbitrary,
                "correction to enthalpy": enthalpies,
                "total entropy": entropies,
                "constant volume heat capacity": capacities,
                "zero-point correction": zpe,
            }

        raise Exception

    def _parse_charge(self):
        """
        Add docstring
        """
        # TO DO: Parse molecular charge and atomic charges
        # TO DO: Add type of charge
        # TO DO: Multiple instances of charge analysis seen (two Mulliken and one Lowdin, difference?)
        charges = []
        ready = False

        for line in self.contents:
            # Load charges from table
            if "Atom       Charge   Shell Charges" in line:
                # Table header found. Overwrite anything saved previously
                ready = True
                charges = []
            elif ready is True and line.strip() in ["", "Line search:"]:
                # Table end found
                ready = False
            elif ready is True:
                # Still reading from charges table
                charges.append(line)

            # Include? Commented or from past files
            # elif ready is True:
            #     lowdinIdx.append(i + 2)
            #     ready = False
            # elif 'Shell Charges' in line and ready is True:  # Shell Charges
            #     lowdinIdx.append(i + 2)
            #     ready = False
            # elif 'Lowdin Population Analysis' in line:
            #     ready = True

        # Process table if one was found
        if len(charges) > 0:
            # return charges

            # Remove blank line in charges (table edge)
            charges = charges[1:]

            # Process charge information
            df = pd.DataFrame(
                [x.split()[0:4] for x in charges],
                columns=["idx", "Atom", "Number", "Charge"],
            )
            df.Number = df.Number.astype("int")
            df.Charge = df.Number - df.Charge.astype("float")

            return df.Charge.values

        raise Exception

    def _parse_timing(self):
        """
        Add docstring
        """
        # Init
        indices = []
        preoptTime = 0
        geomoptTime = 0
        freqTime = 0
        cpuTime = 0
        # wallTime = 0
        # ready = False
        opt = False
        freq = False

        for i, line in enumerate(self.contents):
            # ?
            if "No." in line and len(indices) == 0:
                indices.append(i + 2)  # 0
            elif "Atomic Mass" in line and len(indices) == 1:
                indices.append(i - 1)  # 1
                indices.append(i + 3)  # 2
            elif "Effective nuclear repulsion energy" in line and len(indices) == 3:
                indices.append(i - 2)  # 3

            # Check for optimization and frequency calcs
            if "NWChem geometry Optimization" in line:
                opt = True
            elif "NWChem Nuclear Hessian and Frequency Analysis" in line:
                freq = True

            # Get timing
            if "Total iterative time" in line and opt is False:
                preoptTime += float(line.rstrip().split("=")[1].split("s")[0])
            elif "Total iterative time" in line and opt is True and freq is False:
                geomoptTime += float(line.rstrip().split("=")[1].split("s")[0])
            elif "Total iterative time" in line and freq is True:
                freqTime += float(line.rstrip().split("=")[1].split("s")[0])

            if "Total times" in line:
                cpuTime = float(line.rstrip().split(":")[1].split("s")[0])
                # wallTime = float(line.rstrip().split(':')[2].split('s')[0])
                freqTime = cpuTime - geomoptTime - preoptTime

        # natoms = int(self.contents[indices[1] - 1].split()[0])

        return {
            "single point": preoptTime,
            "geometry optimization": geomoptTime,
            "frequency": freqTime,
            "total": cpuTime,
        }

    def _parse_molden(self):
        """
        Add docstring
        """
        search = splitext(self.path)[0]
        m = glob.glob(search + "*.molden")

        if len(m) != 1:
            raise Exception

        return m[0]

    def _parse_protocol(self):
        """
        Parse out dft protocol
        """
        functional = []
        basis_set = []
        solvation = []
        tasks = []
        basis = None
        func = None
        solvent = None

        for line in self.contents:
            if "* library" in line:
                basis = line.split()[-1]
            if " xc " in line:
                func = line.split(" xc ")[-1].strip()
            if "solvent " in line:
                solvent = line.split()[-1]
            if "task dft optimize" in line:
                tasks.append("optimize")
                basis_set.append(basis)
                functional.append(func)
                solvation.append(solvent)
            if "SHIELDING" in line:
                tasks.append("shielding")
                basis_set.append(basis)
                functional.append(func)
                solvation.append(solvent)
            if "SPINSPIN" in line:
                tasks.append("spin")
                basis_set.append(basis)
                functional.append(func)
                solvation.append(solvent)
            if "freq " in line:
                tasks.append("frequency")
                basis_set.append(basis)
                functional.append(func)
                solvation.append(solvent)

        return {
            "functional": functional,
            "basis set": basis_set,
            "solvation": solvation,
            "tasks": tasks,
        }

    def _parse_connectivity(self):
        """
        Add docstring
        """
        coor_substr = "internuclear distances"

        # Extracting Atoms & Coordinates
        ii = [i for i in range(len(self.contents)) if coor_substr in self.contents[i]]
        ii.sort()

        g = ii[0] + 4
        connectivity = []
        while g <= len(self.contents) - 1:
            if "-" not in self.contents[g]:
                line = self.contents[g].split()
                pair = [line[1], line[4], int(line[0]), int(line[3])]
                connectivity.append(pair)

            else:
                break
            g += 1

        return connectivity

    def parse(self):
        """
        Extract relevant information from NWChem output.

        Parameters
        ----------
        to_parse : list of str
            geometry, energy, shielding, spin, frequency, molden, charge, timing

        """

        # Check that the file is valid first
        if len(self.contents) == 0:
            raise RuntimeError("No contents to parse: {}".format(self.path))
        if "Total times  cpu" not in self.contents[-1]:
            raise RuntimeError("Incomplete NWChem run: {}".format(self.path))

        # Initialize result object to store info
        result = {}

        try:
            result["protocol"] = self._parse_protocol()
        except:
            pass

        try:
            result["geom"] = self._parse_geometry()

        except:
            pass
        try:
            result["rms"] = self._parse_rms()

        except:
            pass

        try:
            result["forces"] = self._parse_forces()

        except:
            pass
        try:
            result["energy"] = self._parse_energy()
        except:
            pass

        try:
            result["shielding"] = self._parse_shielding()
        except:  # Must be no shielding info
            pass

        try:
            result["spin"] = self._parse_spin()
        except:
            pass

        try:
            result["frequency"] = self._parse_frequency()
        except:
            pass

        try:
            result["molden"] = self._parse_molden()
        except:
            pass

        try:
            result["charge"] = self._parse_charge()
        except:
            pass

        try:
            result["timing"] = self._parse_timing()
        except:
            pass

        try:
            result["connectivity"] = self._parse_connectivity()
        except:
            pass

        return result

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return


class ImpactParser(FileParserInterface):
    """
    Extract text from an Impact mobility calculation output file.
    """

    def __init__(self):
        """
        Add docstring
        """
        self.contents = None
        self.result = None

    def load(self, path: str):
        """
        Load in the data file
        """
        with open(path, "rb") as f:
            self.contents = f.readlines()

        return self.contents

    def parse(self):
        """
        Extract relevant information from data
        """

        # Check CCS results == 1
        count = 0
        for line in self.contents:
            l = line.split(" ")
            if "CCS" in l[0]:
                count += 1
        if count != 1:
            return self.result

        # Assume values in second line
        l = self.contents[1].split(" ")
        l = [x for x in l if len(x) > 0]

        # Pull values of interest - may be error prone
        values = []
        try:
            values.append(float(l[-5]))
            values.append(float(l[-3][:-1]))
            values.append(float(l[-2]))
            values.append(int(l[-1]))
        except (ValueError, IndexError) as e:
            print("Could not parse file: ", e)
            return None

        # Add to dictionary to return
        result = {}
        keys = ["CCS_PA", "SEM_rel", "CCS_TJM", "n_iter"]
        for key, val in zip(keys, values):
            result[key] = [val]

        # Save and return results
        self.result = result
        return result  # TODO: return CCS?

    def save(self, path: str, sep="\t"):
        """
        Write parsed object to file
        """
        pd.DataFrame(self.result).to_csv(path, sep=sep, index=False)
        return


class MobcalParser(FileParserInterface):
    """
    Extract text from a MOBCAL mobility calculation output file.
    """

    def __init__(self):
        """
        Add docstring
        """
        self.contents = None
        self.result = {}

    def load(self, path: str):
        """
        Load in the data file
        """
        with open(path, "r") as f:
            self.contents = f.readlines()

        return self.contents

    def parse(self):
        """
        Extract relevant information from data
        """
        done = False
        for line in self.contents:
            # if "average (second order) TM mobility" in line:
            #     m_mn = float(line.split('=')[-1])
            if "average TM cross section" in line:
                ccs_mn = float(line.split("=")[-1])
            elif "standard deviation TM cross section" in line:
                ccs_std = float(line.split("=")[-1])
            elif "standard deviation (percent)" in line:
                done = True
        if done is True:
            self.result["ccs"] = {"mean": ccs_mn, "std": ccs_std}

        return self.result

    def save(self, path: str, sep="\t"):
        """
        Write parsed object to file
        """
        pd.DataFrame(self.result).to_csv(path, sep=sep, index=False)
        return


class SanderParser(FileParserInterface):
    """
    Extract text from an Sander simulated annealing simulation output file.
    """

    def load(self, path: str):
        """
        Load in the data file
        """
        raise NotImplementedError

    def parse(self):
        """
        Extract relevant information from data
        """
        raise NotImplementedError

    def save(self, path: str):
        """
        Write parsed object to file
        """
        raise NotImplementedError


class XTBParser(FileParserInterface):
    """
    Add docstring
    """

    def __init__(self):
        """
        Add docstring
        """
        self.contents = None
        self.result = None
        self.path = None

    def load(self, path: str):
        """
        Load in the data file
        """
        with open(path, "r") as f:
            self.contents = f.readlines()
        self.path = path
        # return self.contents

    def _crest_energy(self):
        """
        Add docstring
        """
        relative_energy = []
        total_energy = []
        population = []

        ready = False
        for h in range(len(self.contents), 0, -1):
            if "Erel/kcal" in self.contents[h]:
                g = h + 1
                for j in range(g, len(self.contents)):
                    line = self.contents[j].split()
                    if len(line) == 8:
                        relative_energy.append(float(line[1]))
                        total_energy.append(float(line[2]))
                        population.append(float(line[4]))
                        ready = True

                    if "/K" in line[1]:
                        break
            if ready == True:
                break

        return {
            "relative energies": relative_energy,
            "total energies": total_energy,
            "population": population,
        }

    def _crest_timing(self):
        """
        Add docstring
        """

        def grab_time(line):
            line = line.replace(" ", "")
            line = line.split(":")

            return ":".join(line[1:]).strip("\n")

        ready = False
        for line in self.contents:
            if "test MD wall time" in line:
                test_MD = grab_time(line)
                ready = True

            if "MTD wall time" in line:
                MTD_time = grab_time(line)

            if "multilevel OPT wall time" in line:
                multilevel_OPT = grab_time(line)

            if "MD wall time" in line and ready == True:
                MD = grab_time(line)
                ready = False

            if "GC wall time" in line:
                GC = grab_time(line)

            if "Overall wall time" in line:
                overall = grab_time(line)

        return {
            "test MD wall time": test_MD,
            "metadynamics wall time": MTD_time,
            "multilevel opt wall time": multilevel_OPT,
            "molecular dynamics wall time": MD,
            "genetic z-matrix crossing wall time": GC,
            "overall wall time": overall,
        }

    def _isomer_energy(self):
        """
        Add docstring
        """
        complete = False
        relative_energies = []
        total_energies = []
        for i in range(len(self.contents), 0, -1):
            if "structure    ΔE(kcal/mol)   Etot(Eh)" in self.contents[i]:
                h = i + 1
                for j in range(h, len(self.contents)):
                    if self.contents[j] != " \n":
                        line = self.contents[j].split()
                        relative_energies.append(float(line[1]))
                        total_energies.append(float(line[2]))
                    else:
                        complete = True
                        break

            if complete == True:
                break

        return {"relative energy": relative_energies, "total energy": total_energies}

    def _isomer_timing(self):
        """
        Add docstring
        """

        def grab_time(line):
            line = line.replace(" ", "")
            line = line.split(":")

            return ":".join(line[1:]).strip("\n")

        for line in self.contents:
            if "LMO calc. wall time" in line:
                LMO_time = grab_time(line)

            if "multilevel OPT wall time" in line:
                OPT_time = grab_time(line)

            if "Overall wall time" in line:
                OVERALL_time = grab_time(line)

        return {
            "local molecular orbital wall time": LMO_time,
            "multilevel opt wall time": OPT_time,
            "overall wall time": OVERALL_time,
        }

    def _opt_energy(self):
        """
        Add docstring
        """
        energies = []
        for line in self.contents:
            if "* total energy" in line:
                energies.append(float(line.split()[4]))

        return energies

    def _opt_timing(self):
        """
        Add docstring
        """

        def grab_time(line):
            line = line.replace(" ", "")
            line = line.split(":")

            return ":".join(line[1:]).strip("\n")

        tot = False
        scf = False
        anc = False

        for line in self.contents:
            if "wall-time" in line and tot is False:
                total_time = grab_time(line)
                tot = True

            elif "wall-time" in line and scf is False:
                scf_time = grab_time(line)
                scf = True

            if "wall-time" in line and anc is False:
                anc_time = grab_time(line)
                anc = True

        return {
            "Total wall time": total_time,
            "SCF wall time": scf_time,
            "ANC optimizer wall time": anc_time,
        }

    def _parse_energy(self):
        """
        Add docstring
        """
        if self.parse_crest == True:
            return self._crest_energy()
        if self.parse_opt == True:
            return self._opt_energy()
        if self.parse_isomer == True:
            return self._isomer_energy()

    def _parse_timing(self):
        """
        Add docstring
        """
        if self.parse_crest == True:
            return self._crest_timing()
        if self.parse_opt == True:
            return self._opt_timing()
        if self.parse_isomer == True:
            return self._isomer_timing()

    def _parse_protocol(self):
        """
        Add docstring
        """
        protocol = None

        for line in self.contents:
            if " > " in line:
                protocol = line.strip("\n")
            if "program call" in line:
                protocol = (line.split(":")[1]).strip("\n")
        return protocol

    def _parse_geometry(self):
        """
        Split .xyz into separate XYZGeometry instances
        """

        FILE = self.xyz_path
        if len(list(pybel.readfile("xyz", FILE))) > 1:
            geom_list = []
            count = 1
            XYZ = FILE.split(".")[0]

            x = []
            for geom in pybel.readfile("xyz", FILE):
                geom.write("xyz", "%s_%d.xyz" % (XYZ, count))
                x.append(isicle.io.load("%s_%d.xyz" % (XYZ, count)))
                os.remove("%s_%d.xyz" % (XYZ, count))
                count += 1

        else:
            x = [isicle.io.load(self.xyz_path)]


        # Establishing first and last geometries for RMS
        def xyz_block_to_list(geom):
            xyz = geom.to_xyzblock().split('\n')[2:]
            xyz = [x.split() for x in xyz]
            xyz = [[x[0], float(x[1]), float(x[2]), float(x[3])] for x in xyz]
            return xyz

        self.first = xyz_block_to_list(x[0])
        self.last = xyz_block_to_list(x[-1])
 
        return isicle.conformers.ConformationalEnsemble(x)

    def _parse_rms(self):
        xyz1 = [x[1:] for x in self.first]
        xyz2 = [y[1:] for y in self.last]

        rms = np.sqrt(np.mean((np.array(xyz1) - np.array(xyz2)) ** 2))

        return rms

    def _parse_forces(self):

        with open('gradient', 'r') as f:
            contents = f.readlines()

        atoms = []
        forces = []
        for line in contents:
            if len(line.split()) == 4:
                atoms.append(line.split()[-1])
            if len(line.split()) == 3:
                f = line.split()
                forces.append([float(x) for x in f])

        forces = [[atoms[idx], forces[idx][0], forces[idx][1], forces[idx][2]] for idx, line in enumerate(forces)]

        return forces

    def parse(self):
        """
        Extract relevant information from data
        """

        # Check that the file is valid first
        if len(self.contents) == 0:
            raise RuntimeError("No contents to parse: {}".format(self.path))

        last_lines = "".join(self.contents[-10:])
        if (
            ("terminat" not in last_lines)
            & ("normal" not in last_lines)
            & ("ratio" not in last_lines)
        ):
            raise RuntimeError("XTB job failed: {}".format(self.path))

        self.parse_crest = False
        self.parse_opt = False 
        self.parse_isomer = False

        # Initialize result object to store info
        result = {}
        result["protocol"] = self._parse_protocol()

        # Parse geometry from assoc. XYZ file
        try:
            if self.path.endswith("xyz"):
                try:
                    self.xyz_path = self.path
                    result["geom"] = self._parse_geometry()

                except:
                    pass

            if self.path.endswith("out") or self.path.endswith("log"):
                # try geometry parsing
                try:
                    XYZ = None
                    if result["protocol"].split()[0] == "xtb":
                        self.parse_opt = True
                        XYZ = "xtbopt.log"
                    if result["protocol"].split()[1] == "crest":
                        if "-deprotonate" in result["protocol"]:
                            self.parse_isomer = True
                            XYZ = "deprotonated.xyz"
                        elif "-protonate" in result["protocol"]:
                            self.parse_isomer = True
                            XYZ = "protonated.xyz"
                        elif "-tautomer" in result["protocol"]:
                            self.parse_isomer = True
                            XYZ = "tautomers.xyz"
                        else:
                            self.parse_crest = True
                            XYZ = "crest_conformers.xyz"

                    if XYZ is None:
                        raise RuntimeError(
                            "XYZ file associated with XTB job not available,\
                                        please parse separately."
                        )

                    else:
                        temp_dir = os.path.dirname(self.path)
                        self.xyz_path = os.path.join(temp_dir, XYZ)

                        result["geom"] = self._parse_geometry()
                except:
                    pass
        except:
            pass

        try:
            result["timing"] = self._parse_timing()
        except:
            pass

        try:
            result["energy"] = self._parse_energy()
        except:
            pass

        try:
            result['rms'] = self._parse_rms()
        except:
            pass

        try:
            result['forces'] = self._parse_forces()
        except:
            pass

        return result

    def save(self, path):
        """
        Add docstring
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return


class TINKERParser(FileParserInterface):
    """
    Add docstring
    """

    def __init__(self):
        """
        Add docstring
        """
        self.contents = None
        self.result = None
        self.path = None

    def load(self, path: str):
        """
        Load in the data file
        """
        with open(path, "r") as f:
            self.contents = f.readlines()
        self.path = path

    def _parse_energy(self):
        """
        Add docstring
        """
        inp = self.contents
        if len(inp) < 13:
            quit()

        # Get the conformer energies from the file
        energies = []
        for line in inp[13:]:
            data = line[:-1].split("  ")
            data = [_f for _f in data if _f]
            if len(data) >= 3:
                if "Map" in data[0] and "Minimum" in data[1]:
                    energies.append(float(data[-1]))

        return energies

    def _parse_conformers(self):
        """
        Add docstring
        """

        def parse_atom_symbol(AtomNum):
            # TODO: modify lookup to use resources/atomic_masses.tsv
            Lookup = [
                "H",
                "He",
                "Li",
                "Be",
                "B",
                "C",
                "N",
                "O",
                "F",
                "Ne",
                "Na",
                "Mg",
                "Al",
                "Si",
                "P",
                "S",
                "Cl",
                "Ar",
                "K",
                "Ca",
                "Sc",
                "Ti",
                "V",
                "Cr",
                "Mn",
                "Fe",
                "Co",
                "Ni",
                "Cu",
                "Zn",
                "Ga",
                "Ge",
                "As",
                "Se",
                "Br",
                "Kr",
                "Rb",
                "Sr",
                "Y",
                "Zr",
                "Nb",
                "Mo",
                "Tc",
                "Ru",
                "Rh",
                "Pd",
                "Ag",
                "Cd",
                "In",
                "Sn",
                "Sb",
                "Te",
                "I",
                "Xe",
                "Cs",
                "Ba",
                "La",
                "Ce",
                "Pr",
                "Nd",
                "Pm",
                "Sm",
                "Eu",
                "Gd",
                "Tb",
                "Dy",
                "Ho",
                "Er",
                "Tm",
                "Yb",
                "Lu",
                "Hf",
                "Ta",
                "W",
                "Re",
                "Os",
                "Ir",
                "Pt",
                "Au",
                "Hg",
                "Tl",
                "Pb",
                "Bi",
                "Po",
                "At",
                "Rn",
            ]

            if AtomNum > 0 and AtomNum < len(Lookup):
                return Lookup[AtomNum - 1]
            else:
                print("No such element with atomic number " + str(AtomNum))
                return 0

        conffile = open(self.path.split(".")[0] + ".arc", "r")
        confdata = conffile.readlines()
        conffile.close()
        conformers = []
        atoms = []
        atomtypes = isicle.utils.tinker_lookup()["atomtypes"].to_list()
        anums = isicle.utils.tinker_lookup()["anums"].to_list()
        atypes = [x[:3] for x in atomtypes]

        # Parse data from arc file, extract coordinates and atom type
        for line in confdata:
            data = [_f for _f in line.split("  ") if _f]
            if len(data) < 3:
                conformers.append([])
            else:
                if len(conformers) == 1:
                    anum = anums[atypes.index(data[1][:3])]
                    atoms.append(parse_atom_symbol(anum))
                conformers[-1].append([x for x in data[2:5]])

        # Convert from TINKER xyz format to standard xyz format
        xyz_file = []
        for conf in conformers:
            xyz_file.append(" {}\n".format(len(conf)))
            for idx, line in enumerate(conf):
                s = " {}\t{}\t{}\t{}".format(atoms[idx], line[0], line[1], line[2])
                xyz_file.append(s)

        # Write xyzs to file
        FILE = "conformers.xyz"
        f = open(FILE, "w+")
        for i in xyz_file:
            f.write(i + "\n")
        f.close()

        # Read in file by
        if len(list(pybel.readfile("xyz", FILE))) > 1:
            geom_list = []
            count = 1
            XYZ = FILE.split(".")[0]
            for geom in pybel.readfile("xyz", FILE):
                geom.write("xyz", "%s_%d.xyz" % (XYZ, count), overwrite=True)
                geom_list.append("%s_%d.xyz" % (XYZ, count))
                count += 1

            x = [isicle.io.load(i) for i in geom_list]

        else:
            x = [isicle.io.load(self.xyz_path)]

        return isicle.conformers.ConformationalEnsemble(x)

    def parse(self):
        """
        Extract relevant information from data
        """

        # Check that the file is valid first
        if len(self.contents) == 0:
            raise RuntimeError("No contents to parse: {}".format(self.path))

        # Initialize result object to store info
        result = {}

        try:
            result["geom"] = self._parse_conformers()
        except:
            pass

        try:
            result["energy"] = self._parse_energy()
        except:
            pass

        try:
            result["charge"] = self._parse_charge()
        except:
            pass

        return result

    def save(self):
        """
        Add docstring
        """
        return
