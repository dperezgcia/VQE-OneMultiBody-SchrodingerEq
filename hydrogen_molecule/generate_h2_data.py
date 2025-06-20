import numpy as np

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

import pickle

def generate_h2_data(geometry: str, dist: float, generate_ansatz: bool = False):
    """
    Genera y guarda el Hamiltoniano, la energía de repulsión nuclear y,
    opcionalmente, el ansatz UCCSD para la molécula de H₂.

    Args:
        geometry (str): Geometría molecular en formato string para PySCFDriver.
        dist (float): Distancia entre los núcleos (en angstroms), usada para nombrar los archivos.
        generate_ansatz (bool): Si es True, también genera y guarda el ansatz UCCSD.
    """

    # Configuramos el driver de PySCF
    driver = PySCFDriver(atom=geometry, basis='sto3g')
    es_problem = driver.run()

    # Construimos el Hamiltoniano y lo mapeamos
    hamiltonian = es_problem.second_q_ops()[0]
    mapper = JordanWignerMapper()
    hamiltonian = mapper.map(hamiltonian)

    # Guardamos el Hamiltoniano
    with open(f"hydrogen_molecule/data/hamiltonian{dist:.3f}.pkl", "wb") as f:
        pickle.dump(hamiltonian, f)

    # Guardamos la energía de repulsión nuclear
    nuclear_repulsion = es_problem.nuclear_repulsion_energy
    with open(f"hydrogen_molecule/data/nuclear_repulsion{dist:.3f}.pkl", "wb") as f:
        pickle.dump(nuclear_repulsion, f)

    # Opcionalmente generamos el ansatz UCCSD
    if generate_ansatz:
        num_spatial_orbitals = es_problem.num_spin_orbitals // 2  
        num_particles = es_problem.num_particles  

        hf_initial_state = HartreeFock(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper
        )

        ansatz = UCCSD(
            num_spatial_orbitals=num_spatial_orbitals,
            num_particles=num_particles,
            qubit_mapper=mapper,
            initial_state=hf_initial_state
        )

        with open("hydrogen_molecule/data/ansatz.pkl", "wb") as f:
            pickle.dump(ansatz, f)


# Generamos los datos para el punto de equilibrio R = 0.74 Å
geometry = f"H 0.0 0.0 {-0.74/2}; H 0.0 0.0 {0.74/2}"
generate_h2_data(geometry, 0.74, generate_ansatz=True)


# Generamos los datos del H₂ para 20 distancias (aplicaremos el VQE a cada uno de los Hamiltonianos)
distances = np.linspace(0.25, 2.5, 20)
for dist in distances:
    geometry = f"H 0.0 0.0 {-dist/2}; H 0.0 0.0 {dist/2}"
    generate_h2_data(geometry, dist, generate_ansatz=False) # Solo necesitamos generar el ansatz una vez

# Generamos los datos del H₂ para 100 distancias (necesario para tener una curva de referencia)
distances = np.linspace(0.25, 2.5, 100)
for dist in distances:
    geometry = f"H 0.0 0.0 {-dist/2}; H 0.0 0.0 {dist/2}"
    generate_h2_data(geometry, dist, generate_ansatz=False)

