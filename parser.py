import numpy as np


def parse_data():
	input_file = 'data.csv'
	#adding 1 to 61067 adn1 1000 because data is 1 indexed!
	result = np.zeros(shape=(1001,61068), dtype=np.int16)
	with open(input_file, 'r') as inStream:
		for line in inStream:
			tokens = (line.strip()).split(',')
			result[int(tokens[0]), int(tokens[1])] = int(tokens[2])
	return result


if __name__ == '__main__':
	result = parse_data();
	# test
	print(result[1,3:17])
	print(result[1000,61030:])