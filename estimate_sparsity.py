import sys

import numpy as np
import math



#sparsity estimation of matrix based on Prof. Saday and Pandey analysis
def read_matrix_sparse(filename, has_header):


	lines = []

	with open(filename) as file:

		lines = file.readlines()

	if (has_header):
		lines = lines[1:]

	#cast into full _matrix

	print("N lines {}".format(len(lines)))

	array_lines = []

	for line in lines:


		line_data = [int(i) for i in line.split(" ")[:-1]]

		# if (line_data[-1] != 0):
		array_lines.append(line_data)

		

	print("N array lines {}: 0: {}".format(len(array_lines), array_lines[0]))

	return np.array(array_lines)


# Lower bound on the sparsity

#for each l in L

#can np apply this in parallel via np.apply?
#wrap in lambda and pass.
def linearize_row(row, max_dims, contraction_dims):

	#contraction dims is bitarray
	#1 signals increment of contraction dim

	#0 signals increment of non_contraction_dims


	contraction_sum = 0
	l_sum = 0

	for i in range(len(row)):

		if (contraction_dims[i]):

			contraction_sum *= max_dims[i]

			contraction_sum += row[i]

		else:

			l_sum *= max_dims[i]
			l_sum += row[i]

	return [l_sum, contraction_sum]



def linearize(array, contraction_dims):

	#clip the last column.
	# array =  array[:,  0:-1]


	max_dims = np.zeros(array.shape[1])

	for i in range(len(max_dims)):

		# print(array[:, i].shape)
		max_dims[i] = np.max(array[:, i])

	# print("Max dims of clipped.")
	# print(max_dims)
	# print(array[0])

	empty_contraction_dims = np.zeros(array.shape[1])

	for i in contraction_dims:
		empty_contraction_dims[i] = 1

	# print("contraction_dims")
	# print(empty_contraction_dims)

	output = []

	for row in array:

		output.append(linearize_row(row, max_dims, empty_contraction_dims))

	# print(output[0])
	return output

def sparsity_lower_bound(array_left, contraction_left, array_right, contraction_right):


	#right hash table - maps c to #r present
	#linearize spaces first

	left_linearized = linearize(array_left, contraction_left)

	right_linearized = linearize(array_right, contraction_right)

	#convert right into a table

	right_table = {}

	#linearize outputs [r_sum, contraction_sum] for each pair
	#sum r_sum for each contraction.

	for line in right_linearized:

		#print(line)

		if (right_table.get(line[1])):
			right_table[line[1]] += 1
		else:
			right_table[line[1]] = 1

		

	#right table is initialized

	#left table must be a mapping from L -> C
	left_table = {}

	for line in left_linearized:

		if (left_table.get(line[0])):
			left_table[line[0]].append(line[1])
		else:
			left_table[line[0]] = [line[1]]


	nnz_lower = 0
	for key in left_table.keys():
		
		nnz_l = len(left_table[key])

		local_sum = 0
		for c in left_table[key]:

			if (right_table.get(c)):

				local_sum += right_table[c]
		nnz_lower += local_sum/nnz_l

	return nnz_lower

def sparsity_upper_bound(array_left, contraction_left, array_right, contraction_right):


	#right hash table - maps c to #r present
	#linearize spaces first

	left_linearized = linearize(array_left, contraction_left)

	right_linearized = linearize(array_right, contraction_right)

	#convert right into a table

	right_table = {}

	#linearize outputs [r_sum, contraction_sum] for each pair
	#sum r_sum for each contraction.

	for line in right_linearized:

		#print(line)

		if (right_table.get(line[1])):
			right_table[line[1]] += 1
		else:
			right_table[line[1]] = 1

		

	#right table is initialized

	#left table must be a mapping from L -> C
	left_table = {}

	for line in left_linearized:

		if (left_table.get(line[0])):
			left_table[line[0]].append(line[1])
		else:
			left_table[line[0]] = [line[1]]


	nnz_lower = 0

	nnz_l_total = 0
	for key in left_table.keys():
		
		nnz_l = len(left_table[key])

		nnz_l_total += nnz_l

		local_sum = 0
		for c in left_table[key]:

			if (right_table.get(c)):

				local_sum += right_table[c]
		nnz_lower += local_sum


	return nnz_lower




#start of main.

# Access the arguments
arguments = sys.argv[1:]  # Exclude the script name

has_header = True

array_left = read_matrix_sparse(arguments[0], has_header)

array_left.sort()

array_right = read_matrix_sparse(arguments[1], has_header)

left_n_contract = int(arguments[2])



left_dims = [int(arguments[i+3]) for i in range(left_n_contract)]


remainder = 3+left_n_contract

right_n_contract = int(arguments[remainder])

right_dims = [int(arguments[i+1+remainder]) for i in range(right_n_contract)]



print("Left has shape {} dims {}, right has shape {} dims {}".format(array_left.shape, left_dims, array_right.shape, right_dims))


lb = sparsity_lower_bound(array_left, left_dims, array_right, right_dims)

ub = sparsity_upper_bound(array_left, left_dims, array_right, right_dims)

print("Lower bound on sparsity: {}, upper bound {}".format(lb, ub))


#estimate_lower_bound(array_left, array_right)

# frob_norm = frobenius_norm(array_left, array_right)

# print("Frobenius norm of {} and {}: {}".format(arguments[0], arguments[1], frob_norm))