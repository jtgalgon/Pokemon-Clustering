import csv
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from  scipy.cluster.hierarchy import linkage

class Cluster:
  def __init__(self, id, pokemon):
    self.id = id
    self.pokemon = pokemon

def load_data(filepath):
  csv_list = []
  with open(filepath, mode='r') as file:
    csv_file = csv.DictReader(file)
    
    for row in csv_file:
      csv_list.append(dict(row))
  return csv_list

def calc_features(row):
  array = np.array([int(row['Attack']), int(row['Sp. Atk']), int(row['Speed']), int(row['Defense']), int(row['Sp. Def']), int(row['HP'])], dtype=np.int64)
  return array


def hac(features):
  clusters = list()
  for i in range(len(features)):
    cluster = Cluster(i, np.array([i]))
    clusters.append(cluster)
  comp_3 = np.zeros([len(features), 6])
  for i in range(len(features)):
    comp_3[i] = features[i]
  
  A = np.zeros([len(features), 1])
  for i in range(len(comp_3)):
    for x in range(6):
      y = comp_3[i][x]
      A[i] += pow(y, 2)
  B = np.ones((len(features), 1))
  comp_1 = np.dot(A, np.transpose(B))
  comp_2 = np.dot(B, np.transpose(A))
  comp_3 = (np.dot(comp_3, np.transpose(comp_3)))
  D = comp_1 + comp_2 - ( 2 * comp_3)
  D = np.sqrt(D)
  z = np.zeros([len(features) - 1, 4])
  for i in range(len(features) - 1):
    min_distance = np.inf
    for cluster_x in clusters:    
      for cluster_y in clusters:    
        dis = clusterDist(cluster_x , cluster_y, D)
        if (dis < min_distance):
          if(cluster_x.id != cluster_y.id): 
            min_distance = dis
            cluster_1_id = cluster_x
            cluster_2_id = cluster_y
    new_cluster = Cluster((len(features) + i), np.concatenate((cluster_1_id.pokemon,cluster_2_id.pokemon)))            
    clusters.append(new_cluster)
    z[i, 0] = cluster_1_id.id
    z[i, 1] = cluster_2_id.id
    z[i, 2] = min_distance
    z[i, 3] = len(cluster_1_id.pokemon) + len(cluster_2_id.pokemon)
    clusters.remove(cluster_1_id)
    clusters.remove(cluster_2_id)
  return z

def imshow_hac(Z, names):
  plt.figure(figsize=(100,100))
  dendrogram(Z, labels=names, leaf_rotation=90)
  plt.show()



def clusterDist(cluster_x, cluster_y, D):
  max = 0
  for i in cluster_x.pokemon:
    for j in cluster_y.pokemon:
      if(D[i][j] > max): 
        max = D[i][j]
  return max

def hac_linkage(features):
  return linkage(features)


if __name__=="__main__":
  features_and_names = [(calc_features(row), row['Name']) for row in load_data('Pokemon.csv')[:800]]
  Z = hac([row[0] for row in features_and_names])
  names = [row[1] for row in features_and_names]
  imshow_hac(Z,names)