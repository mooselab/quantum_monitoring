**QMon: Monitoring the Execution of Quantum Programs with Mid-Circuit Measurement and Reset**

The repository contains the detailed results and replication package for the paper "QMon: Monitoring the Execution of Quantum
Programs with Mid-Circuit Measurement and Reset".

Quantum computing has advanced rapidly, transitioning from experimental physics to accessible cloud-based platforms, marking the early stages of commercialization. 
Effective monitoring is crucial for testing and debugging quantum programs, enabling the detection and management of errors that compromise computational accuracy.
This paper presents QMon, a methodology for monitoring quantum programs using mid-circuit measurement and reset to detect bit-flip errors. QMon monitors quantum circuits by inserting mid-circuit measurement points (monitoring nodes) and reconstructing the measured qubits to their original states.
We evaluate our approach based on its impact on circuit behavior, monitoring coverage, and error detection effectiveness. Results show that most circuits maintain their original behavior after adding monitoring nodes. However, preserving functionality limits monitoring coverage (e.g., an average qubit coverage of 39.8\%). Despite this, our method effectively detects and localizes errors, identifying 34 out of 99 error-inducing mutations across 33 quantum circuits, with 33 mutations accurately localized.
Our methodology enhances the robustness of quantum programs, contributing to improved quantum error detection. As quantum hardware evolves, this approach could play a key role in ensuring the accuracy and reliability of quantum computations for real-world applications.


## Repository Structure
This repository is organized into the following folders:

- **reconstruction/**: Contains the code for reconstructing quantum circuits.
- **quantum circuits/**: Contains the the quantum circuits for experiments.
- **data/**: Contains processed data and final results.

## Dependencies
We recommend using an Anaconda environment with Python version 3.9, and following Python requirement should be met.

* Numpy 1.23.5
* Pandas 2.2.2
* Qiskit 0.44.2

## Data Source

The quantum circuits dataset used in our project is sourced from the MQTBench, hosted by the Chair of Quantum Technologies at the Technical University of Munich (TUM). More information can be found on their website: [MQTBench](https://www.cda.cit.tum.de/mqtbench/).


## Methodology
![image](https://github.com/user-attachments/assets/036f817e-fb00-4d88-8eae-822b63c246ea)



## Acknowledgements

Our implimentation bases on or contains many references to following repositories:

* [mqt-bench](https://github.com/cda-tum/mqt-bench)
* [mqt-predictor](https://github.com/cda-tum/mqt-predictor)



