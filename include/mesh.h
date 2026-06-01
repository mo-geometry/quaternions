#pragma once

#include "gl_header.h"
#include <string>
#include <vector>

/// A simple triangle mesh loaded from an OBJ file.
///
/// Vertex data is interleaved: [px, py, pz, nx, ny, nz] per vertex.
/// If the OBJ has no normals, flat-shading normals are generated
/// from the face geometry.
struct Mesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    int index_count = 0;

    /// Load an OBJ file and upload geometry to the GPU.
    /// Returns true on success.
    bool load(const std::string& path);

    /// Draw the mesh (assumes shader is already bound).
    void draw() const;

    /// Release GPU resources.
    void cleanup();
};
