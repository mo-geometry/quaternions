#include "mesh.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <unordered_map>

/// Hash for de-duplicating vertex/normal index pairs.
struct PairHash {
    std::size_t operator()(const std::pair<int, int>& p) const {
        return std::hash<int>()(p.first) ^ (std::hash<int>()(p.second) << 16);
    }
};

bool Mesh::load(const std::string& path) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &err,
                          path.c_str())) {
        std::cerr << "OBJ load error: " << err << "\n";
        return false;
    }
    if (!err.empty()) {
        std::cerr << "OBJ warning: " << err << "\n";
    }

    bool has_normals = !attrib.normals.empty();

    // Interleaved vertex data: position (3) + normal (3)
    std::vector<float> vertices;
    std::vector<unsigned int> indices;

    // De-duplicate vertices by (vertex_index, normal_index) pair
    std::unordered_map<std::pair<int, int>, unsigned int, PairHash> unique;

    for (const auto& shape : shapes) {
        size_t offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            int fv = shape.mesh.num_face_vertices[f];

            // Compute flat-shading normal if OBJ has none
            float fn[3] = {0.0f, 1.0f, 0.0f};
            if (!has_normals && fv >= 3) {
                const auto& i0 = shape.mesh.indices[offset + 0];
                const auto& i1 = shape.mesh.indices[offset + 1];
                const auto& i2 = shape.mesh.indices[offset + 2];
                float ax = attrib.vertices[3 * i1.vertex_index + 0]
                         - attrib.vertices[3 * i0.vertex_index + 0];
                float ay = attrib.vertices[3 * i1.vertex_index + 1]
                         - attrib.vertices[3 * i0.vertex_index + 1];
                float az = attrib.vertices[3 * i1.vertex_index + 2]
                         - attrib.vertices[3 * i0.vertex_index + 2];
                float bx = attrib.vertices[3 * i2.vertex_index + 0]
                         - attrib.vertices[3 * i0.vertex_index + 0];
                float by = attrib.vertices[3 * i2.vertex_index + 1]
                         - attrib.vertices[3 * i0.vertex_index + 1];
                float bz = attrib.vertices[3 * i2.vertex_index + 2]
                         - attrib.vertices[3 * i0.vertex_index + 2];
                fn[0] = ay * bz - az * by;
                fn[1] = az * bx - ax * bz;
                fn[2] = ax * by - ay * bx;
                float len = std::sqrt(fn[0]*fn[0] + fn[1]*fn[1] + fn[2]*fn[2]);
                if (len > 1e-8f) {
                    fn[0] /= len; fn[1] /= len; fn[2] /= len;
                }
            }

            for (int v = 0; v < fv; ++v) {
                const auto& idx = shape.mesh.indices[offset + v];
                auto key = std::make_pair(idx.vertex_index, idx.normal_index);

                auto it = unique.find(key);
                if (it != unique.end()) {
                    indices.push_back(it->second);
                } else {
                    unsigned int new_idx =
                        static_cast<unsigned int>(vertices.size() / 6);

                    // Position
                    vertices.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
                    vertices.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
                    vertices.push_back(attrib.vertices[3 * idx.vertex_index + 2]);

                    // Normal
                    if (has_normals && idx.normal_index >= 0) {
                        vertices.push_back(
                            attrib.normals[3 * idx.normal_index + 0]);
                        vertices.push_back(
                            attrib.normals[3 * idx.normal_index + 1]);
                        vertices.push_back(
                            attrib.normals[3 * idx.normal_index + 2]);
                    } else {
                        vertices.push_back(fn[0]);
                        vertices.push_back(fn[1]);
                        vertices.push_back(fn[2]);
                    }

                    unique[key] = new_idx;
                    indices.push_back(new_idx);
                }
            }
            offset += fv;
        }
    }

    index_count = static_cast<int>(indices.size());

    std::cout << "Loaded " << path << ": "
              << (vertices.size() / 6) << " vertices, "
              << (indices.size() / 3) << " triangles\n";

    // --- Upload to GPU ---
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glGenBuffers(1, &ebo);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 vertices.size() * sizeof(float),
                 vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 indices.size() * sizeof(unsigned int),
                 indices.data(), GL_STATIC_DRAW);

    // Position attribute (location = 0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute (location = 1)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    return true;
}

void Mesh::draw() const {
    glBindVertexArray(vao);
    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Mesh::cleanup() {
    if (ebo) glDeleteBuffers(1, &ebo);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (vao) glDeleteVertexArrays(1, &vao);
    ebo = vbo = vao = 0;
}
