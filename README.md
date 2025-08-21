# Supervised Learning of Protein Melting Temperature: Cross‐Species vs. Species‐Specific Prediction

This repository contains the code to reproduce the results reported in the following article:

- _Supervised Learning of Protein Melting Temperature: Cross‐Species vs. Species‐Specific Prediction_ , Sebastián García López, Jesper Salomon, Wouter Boomsma - **Proteins: Structure, Function, and Bioinformatics**

The work also can be found in the bioRxiv preprint doi: https://doi.org/10.1101/2024.10.12.61797 

---

# 📝 Installation

In order to The work depends on a virtual environment 

```bash
git clone https://github.com/deltadedirac/thermocontrast_tm.git
cd devcontainer
conda env create --file thermocontrast_tm_env_end.yaml
conda activate thermocontrast_tm_env_end
```

---

# 📝 Weights from Pretrained Models 

The trained models for this work will be released in the coming days or weeks, as the repository is currently being reorganized to make it clearer and more user-friendly for the community.

---

## Notes: 

- The main branch is currently being rewritten and restructured to make it easier for users to understand. <br><br>

- The results of the study are located in the tests_10_05_2024 folder, where the main notebook for running the experiments is LAMLP_Benchmark_all_taxonomy_08_April_2024.ipynb. This notebook depends on configuration files stored in the config subfolder within the same directory. Please note that the notebook name may be updated during future restructuring.<br><br>

- To run the notebooks from the console (as was necessary in this study), you can use tools such as papermill, or execute them with ipython as follows:

```bash
ipython -c "%run NOTEBOOK_NAME.ipynb"
```
That said, upcoming updates to the README.md file will provide clearer and more concise instructions for this process.

- The complete results of the study can be found in the results_exp10_05_2024_with_predsvals folder, located inside tests_10_05_2024.

---

## Cite this repository!

If you find this code and work useful, feel free to cite it!

```bibtex
@article{lopez2025supervised,
  title={Supervised Learning of Protein Melting Temperature: Cross-Species vs. Species-Specific Prediction},
  author={L{\'o}pez, Sebasti{\'a}n Garc{\'\i}a and Salomon, Jesper and Boomsma, Wouter},
  journal={Proteins: Structure, Function, and Bioinformatics},
  publisher={Wiley Online Library}
}
```








