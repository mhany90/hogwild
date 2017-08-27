import csv
from numpy.random import rand
import numpy
from sklearn.datasets import make_classification

def generateData(n):
 xb = (rand(n)*2-1)/2-0.5
 yb = (rand(n)*2-1)/2+0.5
 xr = (rand(n)*2-1)/2+0.5
 yr = (rand(n)*2-1)/2-0.5
 inputs = []
 for i in range(len(xb)):
    inputs.append([xb[i],yb[i],'A'])
    inputs.append([xr[i],yr[i],'B'])
 return inputs


inputs = []

X, y = make_classification(n_samples=70000, n_features=300, n_informative=30, n_redundant=50, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
for data, label in zip(X, y):
    x = list(data)
    print(x)
    x.append(label)
    inputs.append(x)




#data = generateData(500000)
#print(inputs)

with open('data.csv', 'w') as myfile:
    wr = csv.writer(myfile, lineterminator='\n')
    wr.writerows(inputs)