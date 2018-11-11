import parser
import numpy as np

hash_tables = []
hash_matrices = []
dimension = 61068
input_lines = 1000
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
    for i in range(0, l):
        hash_table = []
        for j in range(0, 2**k):
            hash_table.append([])
        
        hash_tables.append(hash_table)
        hash_matrix = np.random.randn(k, dimension)
        # normalizing each row of the hash matrix
        for j in range(0, k):
            norm = np.linalg.norm(hash_matrix, None)
            hash_matrix[i] /= norm

        hash_matrices.append(hash_matrix)

def hash_elements(input_data, l):
    
    for i in range(1, input_lines + 1):
        for j in range(0, l):
            hash_results = np.matmul(hash_matrices[j], np.transpose(input_data[i]))
            # print(hash_results)
            idx = tuple_to_idx(hash_results)
            hash_tables[j][idx].append(i)


if __name__ == '__main__':
    print(tuple_to_idx([1,2,3,4]))
    print(tuple_to_idx([-1,-2,3,4]))
    print(tuple_to_idx([1,2,-3,-4]))

    l = 4
    k = 4
    init_hash_matrices(l, k)
    input_data = parser.parse_data()
    hash_elements(input_data, l)
    print(hash_tables[1])