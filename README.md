# ECAN-PLN Integration Experiment (Real Data)

This project demonstrates the performance of integration between **ECAN** (Economic Cognitive Attention Network) and **PLN** (Probabilistic Logic Network) on massive, real-world knowledge graphs using `incident_final.pl` formats.

## Overview

In traditional configurations, logic systems like PLN suffer from combinatorial explosion when reasoning over huge knowledge bases containing tens of thousands of facts. ECAN mitigates this by applying a "cognitive filter", identifying and prioritizing the most relevant facts via contextual diffusion.

### The Real Data Architecture

Rather than using simulated graphs, this configuration runs on thousands of nodes directly imported from WordNet/ConceptNet through the `incident_final.pl` distribution graph:

1. **Diffusion**: ECAN uses the vast `incident_final.pl` (Incident-Atom Space) to natively map graph neighborhoods. Stimulating `ant` naturally walks the real WordNet topology, activating neighborhood synsets (`ant_n_01`, `insect_n_01`, etc.).
2. **Attentional Focus Extraction**: ECAN identifies the top high-importance relationships (via `getAfAtoms`), successfully finding paths between core biology terms through the noise.
3. **Bridge to PLN**: The highly relevant subset of `isa` relations are converted into strict Probabilistic Logic Network `Sentence` structures (`Inheritance` links).
4. **Reasoning**: PLN efficiently calculates transitive relations (e.g., `ant_n_01` → `animal_n_01` via a 4-hop chain) on the filtered subset of facts.

## Architecture

* `ecan_pln_experiment.metta`: The main pipeline applying ECAN filtering upon the real graph, passing the refined output to PLN queries.
* `vanilla_pln_experiment.metta`: The performance baseline doing brute-force evaluation. PLN receives all WordNet relations directly, suffering exponential branching factors.

## Execution

Ensure you are located at the root of the repo (or running via PeTTa):

```bash
# Execute ECAN + PLN
time sh ../PeTTa/run.sh ecan_pln_experiment.metta -s

# Execute Vanilla PLN comparison
time sh ../PeTTa/run.sh vanilla_pln_experiment.metta -s
```
