# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:41:25 2024

Comparison of pleiotropy with paper: Systematic Anlaysis of Pleiotropy in C. elegans 


@author: logslab
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import cm 
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from scipy.stats import fisher_exact
from sklearn.metrics.pairwise import pairwise_distances


path_save_data='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\pleio_comparison\\pleio_comp_systematic_analysis_pleio\\'


#1.) We are going to read all the data
path_dev='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\matrix_construction\\dev_space\\'
path_phen='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\matrix_construction\\phen_space\\'
path_pleio='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\NNMF_justification\\'
path_pleio_genes='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\section2\\'
path_deviations='C:\\Users\\logslab\\Desktop\\D-P_rule_paper\\section1_deviations\\'


#1.1.) We read commmon genes
f=open(path_dev+'genes_id.txt', 'r')
txt = f.read()
genes = txt.split('\n')
del txt, f
genes=np.delete(genes, len(genes)-1)
genes=np.array(genes)

#1.2.) We read NMF pleio score
f=open(path_pleio+'nmf_pleiotropy.txt', 'r')
txt = f.read()
pleio_score_nnmf = txt.split('\n')
del txt, f
pleio_score_nnmf=np.delete(pleio_score_nnmf, len(pleio_score_nnmf)-1)
pleio_score_nnmf=np.array(pleio_score_nnmf, dtype='f')


#2.) We read our pleio genes 
f=open(path_pleio_genes+'pleio_genes.txt', 'r')
txt = f.read()
pleio_genes = txt.split('\n')
del txt, f
pleio_genes=np.delete(pleio_genes, len(pleio_genes)-1)
pleio_genes=np.array(pleio_genes)


#3.) We read the pleio genes of the paper
df = pd.read_excel(path_save_data+'pleio_genes_paper_systematic_analysis.xlsx', engine='openpyxl')

pleio_genes_paper=np.array(df['WormBase ID'], dtype=str)
pleiotropy_genes_paper=np.array(df['Pleiotropy index'])


#4.) We search how many genes we have analyzed and if they are pleio genes
intesec_pleio=np.intersect1d(pleio_genes, pleio_genes_paper)

intesec_genes, ind_genes_1, ind_genes_2 = np.intersect1d(genes, pleio_genes_paper, return_indices=True)
our_pleiotropy_intersec=pleio_score_nnmf[ind_genes_1]
paper_pleiotropy_intersec=pleiotropy_genes_paper[ind_genes_2]

plt.scatter(paper_pleiotropy_intersec, our_pleiotropy_intersec)
pearsonr(paper_pleiotropy_intersec, our_pleiotropy_intersec)


#5.) Genes of the D-P rule
f=open(path_deviations+'genes_dp_rule.txt', 'r')
txt = f.read()
genes_dp_rule = txt.split('\n')
del txt, f
genes_dp_rule=np.delete(genes_dp_rule, len(genes_dp_rule)-1)
genes_dp_rule=np.array(genes_dp_rule)

f=open(path_deviations+'genes_Dp_phen_deviation.txt', 'r')
txt = f.read()
genes_Dp_phen_deviation = txt.split('\n')
del txt, f
genes_Dp_phen_deviation=np.delete(genes_Dp_phen_deviation, len(genes_Dp_phen_deviation)-1)
genes_Dp_phen_deviation=np.array(genes_Dp_phen_deviation)

f=open(path_deviations+'genes_dP_dev_deviation.txt', 'r')
txt = f.read()
genes_dP_dev_deviation = txt.split('\n')
del txt, f
genes_dP_dev_deviation=np.delete(genes_dP_dev_deviation, len(genes_dP_dev_deviation)-1)
genes_dP_dev_deviation=np.array(genes_dP_dev_deviation)



intesec_DP_rule=np.intersect1d(genes_dp_rule, pleio_genes_paper)
intesec_genes_Dp_phen_deviation=np.intersect1d(genes_Dp_phen_deviation, pleio_genes_paper)
intesec_genes_Dp_phen_deviation=np.intersect1d(genes_Dp_phen_deviation, pleio_genes_paper)







