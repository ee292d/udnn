add_library(udnn STATIC model.hh tensor.hh udnn.hh layer.hh model.cc)

# -fpic
set_property(TARGET udnn PROPERTY POSITION_INDEPENDENT_CODE ON)

# simd library
target_include_directories(udnn PUBLIC ../extern/xsimd/include)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
else()
    message(FATAL_ERROR "Unable to use native machine flag")
endif()
