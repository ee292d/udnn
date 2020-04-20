add_executable(test_layer test_layer.cc)
target_link_libraries(test_layer gtest gtest_main udnn)
gtest_discover_tests(test_layer)

add_executable(test_simd test_simd.cc)
target_link_libraries(test_simd gtest gtest_main udnn)
gtest_discover_tests(test_simd)

add_executable(test_tensor test_tensor.cc)
target_link_libraries(test_tensor gtest gtest_main udnn)
gtest_discover_tests(test_tensor)

add_executable(test_model test_model.cc)
target_link_libraries(test_model gtest gtest_main udnn)
gtest_discover_tests(test_model)