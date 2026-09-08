# -*- coding = utf-8 -*-
"""
author: yunda_si@ucac.ac.cn
"""

from Bio.PDB import PDBParser, MMCIFParser, PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
from openmm.app import *
from openmm import *
from simtk.unit import *
from Bio.PDB.mmcifio import MMCIFIO
from pathlib import Path

def configure_cpu_threads(ncpu: int) -> None:
    """Configure CPU thread counts used by OpenMM/BLAS libraries."""
    if ncpu < 1:
        raise ValueError(f"ncpu must be >= 1, got {ncpu}")

    value = str(ncpu)
    os.environ["OPENMM_CPU_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["OMP_NUM_THREADS"] = value


def load_structure(infile):
    if infile.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure('struct', infile)


def save_structure(structure, outfile, ftype='pdb'):
    if ftype == 'pdb':
        io = PDBIO()
    elif ftype == 'cif':
        io = MMCIFIO()
    else:
        raise ValueError(f"Unsupported output format: {ftype}")

    io.set_structure(structure)
    io.save(outfile)


def clean_structure(infile, outfile, ftype='pdb', keep_h=False):
    structure = load_structure(infile)

    builder = StructureBuilder()
    builder.init_structure('new')
    builder.init_model(0)

    for model in structure:
        for chain in model:
            builder.init_chain(chain.id)
            for res in chain:
                if res.id[0] != ' ':
                    continue
                res_id = res.id[1]
                resname = res.resname

                builder.init_residue(resname, ' ', res_id, ' ')

                for atom in res:
                    name = atom.get_name()
                    element = (atom.element or "").strip().upper()
                    is_h = element == "H" or name.upper().startswith("H")
                    if is_h and not keep_h:
                        continue

                    builder.init_atom(name,
                                      atom.coord,
                                      atom.bfactor,
                                      atom.occupancy,
                                      atom.altloc,
                                      name,
                                      atom.serial_number,
                                      element=atom.element)

    save_structure(builder.get_structure(), outfile, ftype)

def parse_device(device, ncpu):

    value = str(device).strip().lower()

    if value == 'cpu':
        platform = Platform.getPlatformByName("CPU")
        return platform, {"Threads": str(ncpu)}

    if value.startswith("cuda:"):
        index = value.split(":", 1)[1]
    elif value == "cuda":
        index = "0"
    else:
        raise ValueError(f"Unsupported device '{device}'. Use cpu, cuda, cuda:<index>.")

    try:
        platform = Platform.getPlatformByName("CUDA")
    except Exception as exc:
        raise RuntimeError(
            "CUDA was requested, but the OpenMM CUDA platform is unavailable. "
            "Use --device cpu or install/configure CUDA-enabled OpenMM."
        ) from exc

    return platform, {"DeviceIndex": index, "Precision": "mixed"}

def optimize_structure(inpdb, outpdb, steps, device, ncpu):
    configure_cpu_threads(ncpu)

    pdb = PDBFile(inpdb)
    modeller = Modeller(pdb.topology, pdb.positions)
    forcefield = ForceField('amber14-all.xml', 'implicit/gbn2.xml')
    modeller.addHydrogens(forcefield)

    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=CutoffNonPeriodic,
                                     nonbondedCutoff=1 * nanometer,
                                     constraints=HBonds,
                                     soluteDielectric=1.0,
                                     solventDielectric=78.5,
                                     implicitSolventKappa=0.93)

    k = 500 * kilojoules_per_mole / nanometer**2
    rest = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    rest.addGlobalParameter("k", k)
    rest.addPerParticleParameter("x0")
    rest.addPerParticleParameter("y0")
    rest.addPerParticleParameter("z0")

    system.addForce(rest)

    integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.001 * picoseconds)
    platform, properties = parse_device(device, ncpu)

    simulation = Simulation(modeller.topology, system, integrator, platform, properties)
    simulation.context.setPositions(modeller.positions)

    simulation.minimizeEnergy(tolerance=10 * kilojoules_per_mole / nanometer, maxIterations=steps)

    position = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(simulation.topology, position, open(outpdb, 'w'))


def refine(inpdb_file, refined_file, device, ncpu):

    count = len(open(inpdb_file).readlines())
    steps = (int(count / 5.0)) * 1
    steps = max(steps, 200)
    steps = min(steps, 2000)
    print('steps:', steps)

    suffix = Path(refined_file).suffix.lower()
    output_format = "cif" if suffix in {".cif", ".mmcif"} else "pdb"

    clean_structure(inpdb_file, refined_file+'amber_tmp.pdb', ftype='pdb', keep_h=False)
    optimize_structure(refined_file + 'amber_tmp.pdb', refined_file, steps, device, ncpu)
    clean_structure(refined_file, refined_file, ftype=output_format, keep_h=True)

    try:
        os.remove(refined_file + 'amber_tmp.pdb')
    except:
        pass

