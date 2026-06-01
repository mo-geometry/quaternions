cmake_minimum_required(VERSION 3.20)
project(quaternions LANGUAGES C CXX) 

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# GLFW — windowing & input
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# GLM — header-only maths library (vec3, mat4, quaternions)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
)
FetchContent_MakeAvailable(glm)

# Dear ImGui — immediate-mode GUI
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.8
)
FetchContent_MakeAvailable(imgui)

# Build ImGui as a static library with GLFW + OpenGL3 backends
add_library(imgui_impl STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_impl PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_impl PUBLIC glfw)
# ImGui OpenGL3 backend needs to know we're on Apple GL
if(APPLE)
    target_compile_definitions(imgui_impl PUBLIC GL_SILENCE_DEPRECATION)
endif()

# tinyobjloader — header-only OBJ file parser
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG        v1.0.6
)
FetchContent_MakeAvailable(tinyobjloader)

# tinyobjloader v1 may not export a modern CMake target, so create one
if(NOT TARGET tinyobjloader)
    add_library(tinyobjloader INTERFACE)
    target_include_directories(tinyobjloader INTERFACE
        ${tinyobjloader_SOURCE_DIR}
    )
endif()

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(quaternions
    src/main.cpp
    src/shader.cpp
    src/mesh.cpp
    src/camera.cpp
    src/quaternion.cpp
)

target_include_directories(quaternions PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(quaternions PRIVATE
    glfw
    glm::glm
    imgui_impl
    tinyobjloader
)

# macOS: use system OpenGL framework directly (no glad needed)
if(APPLE)
    target_compile_definitions(quaternions PRIVATE GL_SILENCE_DEPRECATION)
    target_link_libraries(quaternions PRIVATE
        "-framework OpenGL"
        "-framework Cocoa"
        "-framework IOKit"
        "-framework CoreVideo"
    )
endif()

# Copy assets to build directory
file(COPY ${CMAKE_SOURCE_DIR}/assets DESTINATION ${CMAKE_BINARY_DIR})
