add_executable(benchmark main.cc)
target_link_libraries(benchmark udnn)
target_include_directories(benchmark PRIVATE ../src)