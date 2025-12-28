**QMon: Monitoring the Execution of Quantum Programs with Mid-Circuit Measurement and Reset**

The repository contains the detailed results and replication package for the paper "QMon: Monitoring the Execution of Quantum
Programs with Mid-Circuit Measurement and Reset".

Unlike classical software, where logging and runtime tracing can effectively reveal internal execution status, quantum circuits possess unique properties, such as the no-cloning theorem and measurement-induced collapse, that prevent direct observation or duplication of their states. These characteristics make it especially challenging to monitor the execution of quantum circuits, complicating essential tasks such as debugging and runtime monitoring.
This paper presents **QMon**, a practical methodology that leverages mid-circuit measurements and reset operations to monitor the internal states of quantum circuits while preserving their original runtime behavior.  
QMon enables the instrumentation of monitoring operators at developer-specified locations within the circuit, allowing comparisons between expected and observed quantum-state probabilities at those locations. 
We evaluated QMon by analyzing its impact on circuit behavior, monitoring coverage, and effectiveness in bug localization. Experimental results involving **154** quantum circuits show that all circuits preserve their intended functionality after instrumentation and that QMon successfully detects and localizes various programming errors. Although monitoring coverage is limited by the need to preserve delicate quantum properties, such as entanglement, QMon effectively detects errors while introducing no or negligible disturbance to the original quantum states. QMon facilitates the development of more robust and reliable quantum software as the field continues to mature.


## Repository Structure
This repository is organized into the following folders:

- **reconstruction/**: Contains the code for reconstructing quantum circuits.
- **quantum circuits/**: Contains the the quantum circuits for experiments.

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



