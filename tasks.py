import invoke
extension_name = "sparse_opcnt"

invoke.run(
    "clang++  -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -I/home/saurabh/taskflow -I/home/saurabh/hopscotch-map/include -I/opt/intel/oneapi/mpi/2021.9.0/include -O3 -mtune=native -march=native -shared -std=c++2a -fPIC -L /usr/lib/gcc/x86_64-linux-gnu/11 -L /opt/intel/oneapi/mkl/latest/lib/intel64 driver.cc -lmkl_rt "
    "-o libspcnt.so "
)
#invoke.run(
#    "g++ -I/home/saurabh/taskflow -I/home/saurabh/hopscotch-map/include -g -fsanitize=address -shared -std=c++2a -fPIC driver.cc "
#    "-o libspcnt.so "
#)
invoke.run(
    "clang++  -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 -O3 -mtune=native -march=native -shared -std=c++2a -fPIC "
    "`python3 -m pybind11 --includes` "
    "-I /usr/include/python3.7 -I . -I/home/saurabh/taskflow -I/home/saurabh/hopscotch-map/include "
    "pybind_wrapper.cpp "
    "-o {0}`python3.10-config --extension-suffix` "
    "-L. -L /usr/lib/gcc/x86_64-linux-gnu/11 -lspcnt -Wl,-rpath,.".format(extension_name)
)

#invoke.run(
#    "g++ -g -shared -std=c++2a -fPIC -fsanitize=address "
#    "`python3 -m pybind11 --includes` "
#    "-I /usr/include/python3.7 -I . -I/home/saurabh/taskflow -I/home/saurabh/hopscotch-map/include "
#    "pybind_wrapper.cpp "
#    "-o {0}`python3.10-config --extension-suffix` "
#    "-L. -lspcnt -Wl,-rpath,.".format(extension_name)
#)
