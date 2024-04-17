import time
import os
import csv
import random

files = ['cristalino','evaporacao','moinho']
vetores_dados = []

output_file = open('dataset.csv','w') 
writer = csv.writer(output_file)

header = []

for i in range(1081):
	header.append('V%04d'%(i))
header.append('Class')

writer.writerow(header)

#reading originals

index = 0
for file in files:
	data = open(file,'r')
	lines = data.readlines()
	v_data = []
	for line in lines:
		raw_data = line.split(' ')
		v_data.append(float(raw_data[1]))
	vetores_dados.append(v_data)
	index += 1

#Generating dataset

random.seed()

popsize = 2000 #number of elements for each class

index = 0
for clss in files:
	for i in range(popsize):
		line = []
		for d in vetores_dados[index]:
			line.append(d * random.uniform(0.6,1.4))
		line.append(index)
		writer.writerow(line)
	index +=1

