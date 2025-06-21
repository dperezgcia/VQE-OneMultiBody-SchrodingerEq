import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit.primitives import Estimator, Sampler
from qiskit import ClassicalRegister, transpile
from qiskit_algorithms import VQD
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.algorithms.optimizers import COBYLA
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


def run_vqe(ansatz, hamiltonian, backend=AerSimulator(), optimizer_method="COBYLA", initial_params=None, maxiter=None):
    """
    Ejecuta el algoritmo VQE para un ansatz y Hamiltoniano dados.

    Args:
        ansatz (QuantumCircuit): Circuito cuántico parametrizado.
        hamiltonian (SparsePauliOp): Hamiltoniano mapeado.
        backend (Backend, optional): Backend de Qiskit. Por defecto usa AerSimulator.
        optimizer_method (str, optional): Método del optimizador. Por defecto usa COBYLA.
        initial_params (np.ndarray, optional): Vector inicial de parámetros.
        maxiter (int, optional): Máximo número de iteraciones del optimizador.

    Returns:
        result (OptimizeResult): Resultado de la optimización.
        cost_history (list): Lista de energías por iteración.
    """

    num_params = ansatz.num_parameters
    if initial_params is None:
        initial_params = np.zeros(num_params)

    estimator = Estimator()
    cost_history = []

    def cost_function(params):
        job = estimator.run([ansatz], [hamiltonian], [params])
        energy = job.result().values[0]
        cost_history.append(energy)
        return energy

    result = minimize(
        cost_function,
        initial_params,
        method=optimizer_method,
        options={"maxiter": maxiter} if maxiter else None
    )

    return result, cost_history



def plot_cost_history(cost_history, filename="cost_history.png"):
    """
    Grafica la evolución de la energía durante la optimización y guarda la figura.

    Args:
        cost_history (list): Lista de energías por iteración.
        filename (str): Nombre del archivo donde se guardará la figura.
    """

    plt.figure()
    plt.plot(range(len(cost_history)), cost_history)
    plt.xlabel("Iteraciones")
    plt.ylabel("Energía")
    plt.title("Convergencia del VQE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"images/{filename}", dpi=300)
    plt.show()



def get_statevector(ansatz, optimal_params):
    """
    Asigna parámetros óptimos al ansatz y devuelve el estado cuántico.

    Args:
        ansatz (QuantumCircuit): Circuito cuántico parametrizado.
        optimal_params (np.ndarray): Vector de parámetros optimizados.

    Returns:
        Statevector: Estado cuántico resultante.
    """
    param_dict = {param: optimal_params[i] for i, param in enumerate(ansatz.parameters)}
    qc_bound = ansatz.assign_parameters(param_dict)
    return Statevector(qc_bound)


def simulate_measurement(ansatz, optimal_params, backend=AerSimulator()):
    """
    Simula la medición del circuito con parámetros óptimos.

    Args:
        ansatz (QuantumCircuit): Ansatz parametrizado.
        optimal_params (np.ndarray): Parámetros optimizados.
        backend (Backend, optional): Backend para la simulación. Por defecto usa AerSimulator.

    Returns:
        counts (dict): Distribución de conteos de medida.
    """

    param_dict = {param: optimal_params[i] for i, param in enumerate(ansatz.parameters)}
    qc = ansatz.assign_parameters(param_dict)
    qc.add_register(ClassicalRegister(qc.num_qubits))
    qc.measure(range(qc.num_qubits), range(qc.num_qubits))

    transpiled = transpile(qc, backend)
    result = backend.run(transpiled).result()
    counts = result.get_counts()
    return counts


def run_vqd(hamiltonian, ansatz, k, optimizer_method=COBYLA, maxiter=1000):
    """
    Ejecuta el algoritmo VQD para encontrar múltiples autovalores de un Hamiltoniano.

    Args:
        hamiltonian (PauliSumOp): Hamiltoniano del sistema.
        ansatz (QuantumCircuit): Circuito ansatz parametrizado.
        optimizer_method (Optimizer, optional): Instancia del optimizador a utilizar. Por defecto usa COBYLA.
        k (int): Número de autovalores a calcular.
        maxiter (int): Máximo de iteraciones del optimizador.

    Returns:
        VQEResult: Resultado con energías y parámetros óptimos.
    """
    estimator = Estimator()
    sampler = Sampler()
    fidelity = ComputeUncompute(sampler=sampler)

    vqd = VQD(
        estimator=estimator,
        fidelity=fidelity,
        ansatz=ansatz,
        optimizer=optimizer_method(maxiter=maxiter),
        k=k
    )

    result = vqd.compute_eigenvalues(hamiltonian)
    return result


