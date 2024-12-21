# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:55:17 2024

@author: Alicia

Great library of phenotypes
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib import cm 
from mpl_toolkits.mplot3d.axes3d import Axes3D, get_test_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def convert_dictionary_to_array(dictionary):
    # pairs in the dictionary
    result = dictionary.items()
    # Convert object to a list
    data = list(result)
    # Convert list to an array
    array = np.array(data)
    return array

def find_path(graph, inicio, name_to_id): 
    paths = nx.all_simple_paths(
        graph,
        source=name_to_id[inicio],
        target=name_to_id['nematode phenotype']
    )
    innerlist = []
    for path in paths:
        innerlist.append(path)
        
    return innerlist
        
def find_path_mouse(graph, inicio, name_to_id): 
    paths = nx.all_simple_paths(
        graph,
        source=name_to_id[inicio],
        target=name_to_id['mammalian phenotype']
    )
    innerlist = []
    for path in paths:
        innerlist.append(path)
        
    return innerlist


def create_gene_phenotype_matrix(genes_id_type, phenotypes_sorted, phen_type, pathlist_sorted):
    matrix=np.zeros((len(genes_id_type), len(phenotypes_sorted)))
    for i in range(len(genes_id_type)):
        for j in range(len(phen_type[i])):
            phen_ind=np.where(phenotypes_sorted==phen_type[i][j])
            phen_ind=int(phen_ind[0])
            #ahora buscamos todos los fenotypes de la jerarquia
            for n in range(len(pathlist_sorted[phen_ind])):
                for k in range(len(pathlist_sorted[phen_ind][n])):
                    ind_matrix=np.where(phenotypes_sorted==pathlist_sorted[phen_ind][n][k])
                    ind_matrix=int(ind_matrix[0])
                    matrix[i][ind_matrix]=1
    return matrix



def find_parent(phen, graph):
    father=[]
    for i in range(len(phen)):
        father_inner=[]
        node = phen[i]
        for child, parent, key in graph.out_edges(node, keys=True):
            father_inner.append(parent)
        father.append(father_inner)
    del father_inner
    return father






