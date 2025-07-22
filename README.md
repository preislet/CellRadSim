# CellRadSim
An Extended Mathematical Model of the p53 Network. Simulating DNA Damage Responses Across Diverse Cellular Phenotypes.


## Overview

This project presents an extended mathematical model of the p53 network, a critical pathway in tumor suppression. The model, based on ordinary differential equations (ODEs), simulates the complex dynamics of cellular responses to DNA damage. A key innovation of this work is the systematic parameterization of the model to represent a wide array of different cell types, providing a versatile platform to study cell-specific responses to genotoxic stress.

The core of the model revolves around the negative feedback loop between the p53 protein and its inhibitor, Mdm2. Upon DNA damage, this feedback loop is modulated by upstream kinases, leading to different cellular outcomes like cell cycle arrest or apoptosis (programmed cell death).

## Key Features

*   **Extended ODE Model:** Expands upon previous models by incorporating additional key components of the p53 pathway.
*   **Distinct Damage Pathways:** Explicitly models the activation of upstream kinases, ATM and ATR, which respond to different types of DNA damage (double-strand breaks and single-strand breaks, respectively).
*   **Downstream Effectors:** Includes critical downstream targets of p53 that determine cell fate:
    *   `p21`: A protein that can lead to cell cycle arrest, providing time for DNA repair.
    *   `Wip1`: A phosphatase that acts as a negative regulator in the feedback loop.
    *   `ApoptFactor`: A generic representation of pro-apoptotic proteins that can trigger cell death.
*   **Diverse Cellular Phenotypes:** The model's main novelty is its adaptation to simulate various cell types by adjusting kinetic parameters based on known biological characteristics.

## Simulated Cell Types

The model has been parameterized to simulate the following diverse cellular phenotypes, allowing for comparative analysis of their DNA damage responses:

*   **AdvancedCell_Baseline:** A generic, healthy mammalian cell.
*   **CancerCell_p53Mutant:** Represents cancer cells with a mutated, non-functional p53.
*   **CancerCell_Restored_p53:** Simulates the therapeutic restoration of p53 function in cancer cells.
*   **RadioresistantCancerCell_p53wt:** Models cancer cells that are resistant to radiation therapy despite having normal p53.
*   **FibroblastCell:** Quiescent stromal cells.
*   **HepatocyteCell:** Metabolically active liver cells with regenerative capacity.
*   **NeuronCell:** Post-mitotic nerve cells.
*   **MelanocyteCell:** Skin cells with specialized responses to UV radiation.
*   **MonocyteCell:** Immune cells that must respond rapidly to damage.
*   **UVResistantCell:** Cells with enhanced capacity to repair UV damage.
*   **StemCell_Embryonic:** Embryonic stem cells with a low tolerance for DNA damage.
*   **SenescentCell:** Cells in a state of permanent cell cycle arrest, which are often resistant to apoptosis.

## Implementation

The model is constructed as a system of coupled ordinary differential equations (ODEs) that describe the temporal dynamics of key protein concentrations. Simulations and analysis are performed using Python, with the `solve_ivp` function from the SciPy library used to integrate the ODEs.

## Conclusion

This framework provides a flexible and powerful tool for in silico investigation of the p53 network. It allows for generating hypotheses about cell-type-specific sensitivities to DNA damaging agents, potential mechanisms of drug resistance in cancer, and the fundamental principles that govern cell fate decisions. The work underscores the critical importance of cellular context in shaping the outcome of p53 activation.

This model builds upon foundational work in the field, particularly the 2005 model by Ciliberto, Novak, and Tyson which first detailed the oscillatory potential of the p53/Mdm2 network.