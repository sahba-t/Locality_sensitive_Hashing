import numpy as np


def parse_data():
	input_file = 'data.csv'
	#adding 1 to 61067 and 1 to 1000 because data is 1 indexed!
	result = np.zeros(shape=(1001,61068), dtype=np.int16)
	with open(input_file, 'r') as inStream:
		for line in inStream:
			tokens = (line.strip()).split(',')
			result[int(tokens[0]), int(tokens[1])] = int(tokens[2])
	return result

def parse_lables():
	input_file = 'label.csv'
	lables = np.zeros(1001)
	with open(input_file, 'r') as inStream:
		i =1
		for line in inStream:
			tokens = (line.strip()).split(',')
			lables[i] = tokens[0]
			i = i+1
	return lables



def cos_similarity(v1, v2):
	return np.dot(v1,v2) / (np.linalg.norm(v1,None) * np.linalg.norm(v2, None))

def nearest_neighbor(v,articles):
	#v is the index of the article in question
	#returns the index of the nearest nneighbor
	nearest = -1
	nearestindex = -1
	for i in range(1,1001):
		if(i != v):
			similarity = cos_similarity(articles[i,:],articles[v,:])
			if(similarity > nearest):
				nearest =  similarity
				nearestindex = i
	return nearestindex



if __name__ == '__main__':
	result = parse_data();
	lables = parse_lables();
	# test
	print(lables)
	print(nearest_neighbor(1,result))
	print(lables[nearest_neighbor(1,result)])