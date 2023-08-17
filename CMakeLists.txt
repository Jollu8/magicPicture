

cmake_minimum_required(VERSION 3.23)
project(project)
cmake_policy(SET CMP0042 NEW)


find_package(OpenCV REQUIRED)

option(WITH_GTEST "Whether to build google test" ON)


add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME}  ${OpenCV_LIBS})
set_target_properties(${PROJECT_NAME} PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        )

if(WITH_GTEST)
    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(my_test "tests.cpp")

    target_include_directories(my_test PRIVATE
            ${GTEST_INCLUDE_DIRS}
            ${CMAKE_SOURCE_DIR}
            )

    set_target_properties(my_test PROPERTIES
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
            )

    target_link_libraries(my_test PRIVATE
            gtest
            gtest_main
            ${OpenCV_LIBS}
            )
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE
            /W4 /await
            )
    if(WITH_GTEST)
        target_compile_options(my_test PRIVATE
                /W4 /await
                )
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE
            "-Wall" "-Wextra" "-Werror" "-pedantic" "-fcoroutines"
            )
    if(WITH_GTEST)
        target_compile_options(my_test PRIVATE
                "-Wall" "-Wextra" "-Werror" "-pedantic" "-fcoroutines"
                )
    endif()
endif()

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT zholu008@gmail.com)
include(CPack)

if(WITH_GTEST)
    include(CTest)
    enable_testing()
    add_test(project my_test)
endif()
