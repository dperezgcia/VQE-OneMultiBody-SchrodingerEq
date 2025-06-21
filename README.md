# TFG: Aplicación de algoritmos cuánticos variacionales a la resolución numérica de la ecuación de Schrödinger para sistemas de uno y varios cuerpos

En este trabajo se aplica el algoritmo *Variational Quantum Eigensolver* para la obtención de los autovalores y autovectores de distintos sistemas cuánticos utilizando Qiskit.

# Descripción

Se han aplicado técnicas variacionales sobre distintos Hamiltonianos cuánticos: un Hamiltoniano molecular (*H₂*), un oscilador anarmónico unidimensional y un oscilador armónico tridimensional. Para ello se han utilizado diferentes *ansatzs* según el sistema:

- En el caso del Hamiltoniano molecular, se ha usado un *ansatz* **UCCSD**.
- En los demás sistemas, se ha utilizado el *ansatz* **EfficientSU2**.

Los experimentos se han ejecutado sobre el simulador **AerSimulator** de Qiskit.

---

## Preprocesamiento con PySCF

Para el sistema molecular, se ha utilizado **PySCF** en Linux para generar los datos físicos del sistema. Esto crea automáticamente una carpeta `data/` que contiene:

- Archivos `hamiltonian*.pkl`: Hamiltonianos moleculares en formato `SparsePauliOp`
- Archivos `nuclear_repulsion*.pkl`: Energías de repulsión nuclear
- Archivo `ansatz.pkl`: Circuito parametrizado correspondiente al *ansatz* UCCSD

---

## Dependencias principales

- `qiskit`
- `qiskit_nature`
- `pyscf`
- `numpy`
- `matplotlib`
