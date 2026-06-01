# Quaternion Bunny Viewer

Interactive Stanford bunny renderer with quaternion-based rotation controls
and real-time matrix display. Built with C++17, OpenGL 3.3, GLFW, and
Dear ImGui.

## Quick Start

### Prerequisites

- CMake >= 3.20
- C++17 compiler (Apple Clang from Xcode Command Line Tools)
- Git (for FetchContent dependency downloads)

On macOS:
```bash
xcode-select --install   # if not already installed
brew install cmake        # if cmake not already installed
```

### Download the Stanford Bunny

```bash
bash scripts/download_bunny.sh
```

This downloads the Stanford bunny OBJ (2503 vertices, 4968 faces) into
`assets/bunny.obj`. If the download fails, the project includes a
fallback `assets/cube.obj` — just change the path in `main.cpp`.

### Build

```bash
mkdir build && cd build
cmake ..
cmake --build . -j$(sysctl -n hw.ncpu)
```

The first build takes a minute or two while CMake fetches dependencies
(GLFW, glad, GLM, Dear ImGui, tinyobjloader).

### Run

```bash
./quaternions
```

## Controls

| Input | Action |
|---|---|
| Left-drag | Orbit camera |
| Scroll | Zoom in/out |
| ImGui sliders | Rotate/translate bunny |
| Reset button | Return to identity transform |
| Esc | Quit |

## Project Structure

```
quaternions/
  CMakeLists.txt          — Build system (fetches all dependencies)
  include/
    camera.h              — Orbit camera (view matrix)
    mesh.h                — OBJ loader → GPU upload
    quaternion.h          — Quaternion algebra API (YOUR WORK)
    shader.h              — Shader compilation helper
  src/
    main.cpp              — Application entry point, render loop, ImGui panel
    camera.cpp            — Orbit camera implementation
    mesh.cpp              — OBJ loading via tinyobjloader
    quaternion.cpp        — Quaternion implementations (TODO stubs)
    shader.cpp            — Shader compile/link
  assets/
    bunny.obj             — Stanford bunny (download via script)
    cube.obj              — Fallback cube mesh
  scripts/
    download_bunny.sh     — Downloads the bunny OBJ
```

## Your Task: Implement quaternion.cpp

The file `src/quaternion.cpp` contains six functions with TODO stubs.
Implement them in order:

1. **`quat_from_axis_angle`** — Half-angle formula. Simplest starting point.
2. **`quat_multiply`** — Hamilton product. Write out all four components.
3. **`quat_normalize`** — Length and divide. Guard against zero.
4. **`quat_to_rotation_matrix`** — The 9-element formula. Watch the
   column-major indexing (GLM stores `mat3[col][row]`).
5. **`quat_from_euler`** — Compose three axis-angle quaternions.
6. **`build_model_matrix`** — Assemble the 4x4 from rotation + translation.

The placeholders return identity values, so the app will build and run
immediately — the bunny just won't rotate until you implement the functions.

### Reference

Your Python implementation in `parallel_transport/core/quaternion.py`
uses identical formulas. The key difference is GLM's column-major
matrix storage:

```
Python (NumPy, row-major):     C++ (GLM, column-major):
  mat[row][col]                  mat[col][row]
```

### Key C++ Concepts to Notice

As you implement, pay attention to:

- **`float` vs `double`**: GPU pipelines use single precision. All our
  types are `float` / `glm::vec3` / `glm::mat3`.
- **`const&` parameters**: Read-only references — no copying, no mutation.
  Same intent as Python's convention of not modifying input arrays.
- **`sizeof(float)` = 4 bytes**: A `float[16]` (one 4x4 matrix) is
  exactly 64 bytes. GPU uniform buffers have strict alignment rules
  based on these sizes.
- **Column-major layout**: `glm::value_ptr(mat4)` returns a `float*`
  to 16 contiguous floats in column-major order. OpenGL's
  `glUniformMatrix4fv` expects this layout.

## Dependencies (fetched automatically by CMake)

| Library | Purpose | Equivalent in conic_sections |
|---|---|---|
| GLFW 3.4 | Windowing, input, GL context | `glfw` (Python binding) |
| glad | OpenGL function loader | `moderngl` handles this |
| GLM 1.0 | Vector/matrix math | `numpy` |
| Dear ImGui 1.91 | Immediate-mode GUI | `imgui-bundle` |
| tinyobjloader | OBJ file parsing | `trimesh` |
