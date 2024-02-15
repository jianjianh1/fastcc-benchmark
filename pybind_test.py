import sparse_opcnt
t1 = sparse_opcnt.Tensor("bcsstk17.mtx", True)
t1.write("bcsstk17_pybind.mtx")
left_contr = sparse_opcnt.CoOrdinate([1])
right_contr = sparse_opcnt.CoOrdinate([0])
num_ops = t1.count_ops(t1, left_contr, right_contr)
print(num_ops)
