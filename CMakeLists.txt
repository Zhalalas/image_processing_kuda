cmake_minimum_required(VERSION 3.10)

project(ImageProcessingCPP VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find system OpenCV
find_package(OpenCV REQUIRED)

add_executable(ImageProcessingCPP main.cpp
    include/resizer.cpp
    include/scale.cpp
    include/validation.cpp
    include/estimation.cpp
    include/transform.cpp
    include/normcrop.cpp
    include/align.cpp
    include/cropandalign.cpp
    include/alignface.cpp
    include/writeimage.cpp
)
target_link_libraries(ImageProcessingCPP PRIVATE ${OpenCV_LIBS})

# Include JSON headers if you have it locally
add_subdirectory(external/json)
target_link_libraries(ImageProcessingCPP PRIVATE nlohmann_json::nlohmann_json)

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)