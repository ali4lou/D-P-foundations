# D-P foundations
Source code, datasets, and instructions for reproducing the results of our paper 'Cell atlases and the Developmental Foundations of the Phenotype'.

### Developmental and phenotypic gene expression matrices 
`matrix_creation_codes` folder - Code to create the developmental gene expression matrix (using the scRNA-seq data from Packer et al. [2019]) and the gene-phenotype association matrix (using the Worm Phenotype Ontology from the WormBase). 

### Reproduction of results 
`result_codes` folder - Code to reproduce each section of results and their associated supplementary figures.  

### GO enrichment, pleiotropy comparison and NMF analysis
`other_analysis` folder - Code to create reproduce the GO enrichment for each subset of genes, the pleiotropy comparison with the following papers: Xiao et al. [2022], Green et al. [2024],
Zou et al. [2008] and the NMF analysis. 

### Files to Download

The required files to execute the code are contained in a compressed file **`files.zip`**. Additionally, larger files can be found on Zenodo. You can download them using the following link: [Zenodo link](<https://zenodo.org/records/14629057>).

In Zenodo, we have included the Packer et al. [2019] scRNA-seq data and the data downloaded from WormBase, which are necessary for creating the developmental and phenotypic gene expression matrices.


## Setup and Script Dependencies

Make sure to update the paths at the beginning of each script to match your local setup. Specifically:  
- Set the path to the location where you want to save your data.  
- Specify the path to where you have downloaded the files from the **`files`** folder and **Zenodo**.  

This adjustment is necessary to ensure the code runs properly in your environment.  

The scripts **`phen_matrix.py`** and **`GO_assotiation.py`** require that the file **`great_library_phenotypes.py`** has been executed beforehand and is located in the same directory as these scripts. Ensure this step is completed before running them.   


## References

This repository references the following studies:

1. Packer, J. S., Zhu, Q., Huynh, C., Sivaramakrishnan, P., Preston, E., Dueck, H., Stefanik, D., Tan, K., Trapnell, C., Kim, J., Waterston, R. H., & Murray, J. I. (2019). [A lineage-resolved molecular atlas of *C. elegans* embryogenesis at single-cell resolution](https://doi.org/10.1126/science.aax1971). *Science*, 365(6459), eaax1971. https://doi.org/10.1126/science.aax1971

2. Xiao, L., Fan, D., Qi, H., Cong, Y., & Du, Z. (2022). [Defect-buffering cellular plasticity increases robustness of metazoan embryogenesis](https://doi.org/10.1016/j.cels.2022.07.001). *Cell Systems*, 13(8), 615–630.e9. https://doi.org/10.1016/j.cels.2022.07.001

3. Green, R. A., Khaliullin, R. N., Zhao, Z., Ochoa, S. D., Hendel, J. M., Chow, T.-L., Moon, H., Biggs, R. J., Desai, A., & Oegema, K. (2024). [Automated profiling of gene function during embryonic development](https://doi.org/10.1016/j.cell.2024.04.012). *Cell*, 187(12), 3141–3160.e23. https://doi.org/10.1016/j.cell.2024.04.012

4. Zou, L., Sriswasdi, S., Ross, B., Missiuro, P. V., Liu, J., & Ge, H. (2008). [Systematic analysis of pleiotropy in *C. elegans* early embryogenesis](https://doi.org/10.1371/journal.pcbi.1000003). *PLoS Computational Biology*, 4(2), e1000003. https://doi.org/10.1371/journal.pcbi.1000003

5. WormBase. [Comprehensive database for *C. elegans* research](https://wormbase.org). Retrieved from [https://wormbase.org](https://wormbase.org).


