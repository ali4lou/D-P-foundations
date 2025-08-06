# D-P foundations
Source code, datasets, and instructions for reproducing the results of our paper 'Cell atlases and the Developmental Foundations of the Phenotype'.

### Developmental and phenotypic gene expression matrices 
`matrix_creation_codes` folder - Code to create the developmental gene expression matrix (using the scRNA-seq data from Packer et al. [2019]) and the gene-phenotype association matrix (using the Worm Phenotype Ontology from the WormBase). 
Required files are included in `files.zip` and on Zenodo.

### Reproduction of results 
`result_codes` folder - Code to reproduce each section of results and their associated supplementary figures. 
Required files are included in `files.zip` and on Zenodo.

### GO enrichment, pleiotropy comparison and NMF analysis
`other_analysis` folder - Code to reproduce the GO enrichment for each subset of genes, the pleiotropy comparison with the following papers: Xiao et al. [2022], Green et al. [2024], Zou et al. [2008] and the NMF analysis. 
Required files are included in `files.zip` and on Zenodo.

### Verification of the D-P rule in other data sheets - (Supplementary material)

`tested_DP_rule` folder – This directory includes different scripts and datasets used to verify the D-P rule in multiple contexts:

- **Systemic RNAi Studies**  
  The script `dp_rule_systemic_RNAi_data.py` tests the D-P rule using only genes whose phenotype associations originate from publications with more than 100 genes annotated via RNAi (i.e., systemic studies).  
  The required data files are included in `files.zip` and on Zenodo.

- **Xiao et al. (2022) Dataset**  
  The script `xiao_dp_rule.py` constructs a new phenotypic space using gene-phenotype associations from Xiao et al. [2022], and evaluates the D-P rule in this context.  
  The necessary data files are also available in `files.zip` and in Zenodo.

- **Zebrafish Analyses**  
  The `zebrafish/` subfolder contains scripts to verify the D-P rule and a coarse interpretation of the rule in *Danio rerio* (zebrafish). We used the scRNA-seq data from Lange et al. [2024] and the gene-phenotype data from the Zebrafish Information Network (ZFIN).
  Corresponding data files are located in `files_zf.zip` and on Zenodo. The scRNA-seq data from Lange et al. [2024] can be found in their web page: [Zebrahub](https://zebrahub.sf.czbiohub.org/data).

### Files to Download

The required files to execute the code are contained in a compressed file **`files.zip`**. Also, in **`files_zf.zip`** contains the datasets needed for running the zebrafish analyses.

Additionally, larger files can be found on Zenodo. You can download them using the following link: [Zenodo link](<https://zenodo.org/records/16753814>).

- `larger_files.zip` includes the Packer et al. [2019] scRNA-seq data and files downloaded from WormBase, which are necessary for generating the developmental and phenotypic gene expression matrices.

- `larger_files_zf.zip` contains additional files required for the zebrafish analyses.

- `xiao.zip` includes Supplementary Tables S3 and S4 from Xiao et al. [2022], used in the verification of the D-P rule.


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

6. Zebrafish Information Network (ZFIN). [Comprehensive knowledgebase for *Danio rerio* research](https://zfin.org). Retrieved from [https://zfin.org](https://zfin.org)
  
7. Lange, M., Granados, A., VijayKumar, S., Bragantini, J., Ancheta, S., Kim, Y. J., Santhosh, S., Borja, M., Kobayashi, H., McGeever, E., Solak, A. C., Yang, B., Zhao, X., Liu, Y., Detweiler, A. M., Paul, S., Theodoro, I., Mekonen, H., Charlton, C., Lao, T., Banks, R., Xiao, S., Jacobo, A., Balla, K., Awayan, K., D'Souza, S., Haase, R., Dizeux, A., Pourquie, O., Gómez-Sjöberg, R., Huber, G., Serra, M., Neff, N., Pisco, A. O., & Royer, L. A. (2024). *A multimodal zebrafish developmental atlas reveals the state-transition dynamics of late-vertebrate pluripotent axial progenitors*. *Cell*, 187(23), 6742–6759.e17. ([https://doi.org/10.1016/j.cell.2024.09.047](https://doi.org/10.1016/j.cell.2024.09.047))

8. ZebraHub. [Multimodal zebrafish developmental atlas – Data Portal](https://zebrahub.sf.czbiohub.org/data). Retrieved from https://zebrahub.sf.czbiohub.org/data




