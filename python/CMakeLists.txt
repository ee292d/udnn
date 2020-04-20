set(PYBIND11_CPP_STANDARD -std=c++14)

pybind11_add_module(_udnn udnn.cc)
target_link_libraries(_udnn PRIVATE udnn)