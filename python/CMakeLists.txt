# only enable the python build when we have enough physical memory
cmake_host_system_information(RESULT MEM_SIZE QUERY TOTAL_PHYSICAL_MEMORY)

if (${MEM_SIZE} GREATER_EQUAL 1024)
    set(PYBIND11_CPP_STANDARD -std=c++17)
    pybind11_add_module(_udnn udnn.cc tensor.cc model.cc quantize.cc)
    target_link_libraries(_udnn PRIVATE udnn)
endif()
