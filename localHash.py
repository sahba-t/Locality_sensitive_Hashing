import parser
import numpy as np
import math
import bruteForce
import sys

hash_tables = None
hash_matrices = None
dimension = 61068
input_lines = 1000
line_break = '-----------------------------------------------------\n'
# turn debug output on and off!
debug = False
# gets a vector which is the result of our hash (with k has functions) and maps that to a index
def tuple_to_idx(v):
    result = 0
    mask = 1
    for x in v:
        # if the element is +, we translate that as 1. otherwise 0 this way we map 
        # from k elements in the vector to a 2^k number
        if x > 0:
            result |= mask
        mask <<= 1
    return result

def init_hash_matrices(l, k):
    # #xxx change this
    # global dimension
    # dimension = 5

    global hash_tables, hash_matrices
    hash_tables = []
    hash_matrices = []
    
    for i in range(0, l):
        hash_table = []
        for j in range(0, 2**k):
            hash_table.append([])
        
        hash_tables.append(hash_table)
        hash_matrix = np.random.randn(k, dimension)
        # normalizing each row of the hash matrix
        for j in range(0, k):
            norm = np.linalg.norm(hash_matrix[j], None)
            hash_matrix[j] /= norm

        hash_matrices.append(hash_matrix)

def hash_elements(input_data, l):
    
    for i in range(1, input_lines + 1):
        for j in range(0, l):
            hash_results = np.matmul(hash_matrices[j], np.transpose(input_data[i]))
            # print(hash_results)
            idx = tuple_to_idx(hash_results)
            hash_tables[j][idx].append(i)


def brute_force_comparison(v, articles, input_data):
	#v is the index of the query article is the list of the index of target articles
    # input_data is the actual input 
	#returns the index of the nearest nneighbor
	nearest = -1
	nearestindex = -1
	for x in articles:
		if(x != v):
			similarity = bruteForce.cos_similarity(input_data[x,:],input_data[v,:])
			if(similarity > nearest):
				nearest =  similarity
				nearestindex = x
	return nearestindex


def classify_articles(input_data):
    comparisons = 0
    corrects = 0
    wrongs = 0
    for i in range(1, input_lines + 1):
        # so that it works both with python2 and python3 :)
        true_class = 1 + math.floor((i-1) / 50)
        candidates = []
        for j in range(0, l):
            hash_results = np.matmul(hash_matrices[j], np.transpose(input_data[i]))
            idx = tuple_to_idx(hash_results)
            for x in hash_tables[j][idx]:
                if x != i:
                    candidates.append(x)
                    comparisons += 1
        nearest_index = brute_force_comparison(i, candidates, input_data)
        near_n_class = 1 + math.floor((nearest_index-1) / 50)
        
        if debug:
        	print('the nearest neighbor of %d is %d real class: %d, nearest neighbor class %d' %(i, nearest_index, true_class, near_n_class ))
        
        if  near_n_class == true_class:
            corrects += 1
        else:
            wrongs += 1
    # end of for loop
    print('average comparisins is %.2f ' %(comparisons / 1000)) 
    print('correct: %d incorrect: %d ' %(corrects, wrongs))

# this function uses the bruteforce routine to test the performance of the brute force 
def test_brute_force(input_data):
    corrects = 0
    wrongs = 0
    for i in range(1, input_lines + 1):
        true_class = 1 + math.floor((i-1) / 50)
        nidx = bruteForce.nearest_neighbor(i, input_data)
        nn_class  = 1 + math.floor((nidx - 1)/50)
        if debug:
        	print('the nearest neighbor of %d is %d real classes: %d, %d' %(i, nidx, true_class, nn_class ))
        
        if  nn_class == true_class:
            corrects += 1
        else:
            wrongs += 1
    print('correct: %d incorrect: %d ' %(corrects, wrongs))  

if __name__ == '__main__':


	input_data = parser.parse_data()
	for l in [4,8,16]:
		for k in [4, 8, 16]:
			print('Using local sensitivity hashing. l= %d k= %d' %(l, k))
			init_hash_matrices(l=l, k=k)
			hash_elements(input_data, l)
			classify_articles(input_data)
			print(line_break)
			# to force output if redirecting to a file
			sys.stdout.flush()

	print('Now doing brute force')
	test_brute_force(input_data)

    # print(hash_tables[1])
