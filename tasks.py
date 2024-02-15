import invoke
extension_name = "sparse_opcnt"
invoke.run(
    "g++ -O3 -Wall -shared -std=c++11 -fPIC driver.cc "
    "-o libspcnt.so "
)
invoke.run(
    "g++ -O3 -Wall -shared -std=c++11 -fPIC "
    "`python3 -m pybind11 --includes` "
    "-I /usr/include/python3.7 -I .  "
    "pybind_wrapper.cpp "
    "-o {0}`python3.10-config --extension-suffix` "
    "-L. -lspcnt -Wl,-rpath,.".format(extension_name)
)
