import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import warnings; warnings.simplefilter('ignore')
from scipy.stats import mstats, kstest, ttest_ind, fisher_exact
import csv
import math
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import NMF


path_save_data='YOUR_PATH_TO_SAVE_DATA'

#1.2.) We read commmon phenotypes
f=open(path_save_data+'phen_stage_analize.txt', 'r')
txt = f.read()
phen = txt.split('\n')
del txt, f
phen=np.delete(phen, len(phen)-1)
phen=np.array(phen)

final_stage=[]
for i in range(len(phen)):
    part = phen[i].split('\t')
    final_stage.append(part[3])
    
final_stage=np.array(final_stage)

stage=['ZDB-STAGE-050211-1', 'ZDB-STAGE-010723-4', 
'ZDB-STAGE-010723-15','ZDB-STAGE-010723-11', 'ZDB-STAGE-010723-33', 
'ZDB-STAGE-010723-20', 'ZDB-STAGE-010723-25', 'ZDB-STAGE-010723-28',
'ZDB-STAGE-010723-7', 'ZDB-STAGE-010723-1', 'ZDB-STAGE-010723-14', 
'ZDB-STAGE-010723-22', 'ZDB-STAGE-010723-27', 'ZDB-STAGE-010723-24', 
'ZDB-STAGE-010723-21', 'ZDB-STAGE-010723-19', 'ZDB-STAGE-010723-17']

stage=np.array(stage)

stage_common, ind_final_stage, j=np.intersect1d(final_stage, stage, return_indices=True)


check_phen=phen[ind_final_stage]


