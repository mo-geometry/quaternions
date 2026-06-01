/// =========================================================================
/// SO(3) Rotation & Quaternion Extraction — Brian's implementation file
///
/// Convention: quaternion stored as glm::vec4(x, y, z, w)
///   - q.x, q.y, q.z = vector part (imaginary)
///   - q.w             = scalar part (real)
///
/// GLM stores matrices in COLUMN-MAJOR order:
///   mat3 m;
///   m[col][row]   — e.g. m[0][1] is column 0, row 1
///
/// This is the opposite of NumPy's row-major default, so be careful
/// when translating.  The mathematical formulas are the same; only
/// the indexing convention differs.
///
/// Reference: parallel_transport/core/quaternion.py
/// =========================================================================

#include "quaternion.h"
#include <iostream>
#include <glm/gtc/quaternion.hpp>  // for ground truth
#include <cmath>

// -------------------------------------------------------------------------
// Step 1: Elementary rotation matrices
//
// These are PROVIDED (not TODO) — they are straightforward and let Brian
// focus on the interesting parts (composition and quaternion extraction).
//
// IMPORTANT GLM column-major storage:
//   To build the matrix Rx = | 1    0       0    |
//                            | 0  cos(a)  -sin(a)|
//                            | 0  sin(a)   cos(a)|
//
//   We set:  m[0] = column 0 = (1, 0, 0)
//            m[1] = column 1 = (0, cos, sin)
//            m[2] = column 2 = (0, -sin, cos)
// -------------------------------------------------------------------------

inline void print_mat3(const std::string& name, const glm::mat3& m) {
    std::cout << name << ":\n";
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 3; ++col) {
            std::cout << m[col][row] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

glm::mat3 rotation_x(float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    glm::mat3 m(1.0f);  // identity
    // Column 1
    m[1][1] = c;
    m[1][2] = s;
    // Column 2
    m[2][1] = -s;
    m[2][2] = c;
    return m;
}

glm::mat3 rotation_y(float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    glm::mat3 m(1.0f);
    // Column 0
    m[0][0] = c;
    m[0][2] = -s;
    // Column 2
    m[2][0] = s;
    m[2][2] = c;
    return m;
}

glm::mat3 rotation_z(float angle) {
    float c = std::cos(angle);
    float s = std::sin(angle);
    glm::mat3 m(1.0f);
    // Column 0
    m[0][0] = c;
    m[0][1] = s;
    // Column 1
    m[1][0] = -s;
    m[1][1] = c;
    return m;
}

// -------------------------------------------------------------------------
// Step 2: Compose rotation matrices in selected order
//
// TODO(Brian): Implement this function.
//
// For INTRINSIC rotation order XYZ, the body-fixed axes rotate with it.
// The equivalent extrinsic (world-fixed) multiplication is:
//   R = Rz * Ry * Rx
//
// So "first rotate about X, then Y, then Z" in the body frame
// becomes "rightmost matrix applied first" in matrix multiplication.
//
// All 6 orders:
//   XYZ: R = Rz * Ry * Rx    (intrinsic X, then Y, then Z)
//   XZY: R = Ry * Rz * Rx    (intrinsic X, then Z, then Y)
//   YXZ: R = Rz * Rx * Ry    (intrinsic Y, then X, then Z)
//   YZX: R = Rx * Rz * Ry    (intrinsic Y, then Z, then X)
//   ZXY: R = Ry * Rx * Rz    (intrinsic Z, then X, then Y)
//   ZYX: R = Rx * Ry * Rz    (intrinsic Z, then Y, then X)
// -------------------------------------------------------------------------

glm::mat3 compose_rotation(float pitch, float yaw, float roll,
                            RotationOrder order) {
    // TODO(Brian): Implement
    //
    // 1. Build the three elementary matrices:
    glm::mat3 Rx = rotation_x(pitch);
    glm::mat3 Ry = rotation_y(yaw);
    glm::mat3 Rz = rotation_z(roll);
    
    // // Debug print the matrices
    // print_mat3("Rx", Rx);
    // print_mat3("Ry", Ry);
    // print_mat3("Rz", Rz);

    // // test the rotation matrix multiplication and indexing
    // glm::mat3 test = Rz * Ry * Rx;
    // glm::mat3 unity = test * glm::transpose(test);
    // print_mat3("Rz * Ry * Rx", test);
    // print_mat3("test * test^T (should be identity)", unity);

    // 2. Multiply them in the correct order based on 'order'.
    //    Use the table above. GLM uses operator* for matrix multiply.
    //    Example for XYZ:  return Rz * Ry * Rx;
    //
    // 3. Switch on the RotationOrder enum:
    switch (order) {
        case RotationOrder::XYZ: return Rz * Ry * Rx;
        case RotationOrder::XZY: return Ry * Rz * Rx;
        case RotationOrder::YXZ: return Rz * Rx * Ry;
        case RotationOrder::YZX: return Rx * Rz * Ry;
        case RotationOrder::ZXY: return Ry * Rx * Rz;
        case RotationOrder::ZYX: return Rx * Ry * Rz;
    }

    // Placeholder — identity (no rotation)
    (void)pitch; (void)yaw; (void)roll; (void)order; 

    return glm::mat3(1.0f);
}
 

glm::vec4 quat_from_rotation_matrix(const glm::mat3& R) {
    float trace = R[0][0] + R[1][1] + R[2][2];
    float x, y, z, w;
    float s;
    
    if (trace > 0.0f) {
        // Case 1: trace > 0
        s = 2.0f * std::sqrt(trace + 1.0f);
        w = s / 4.0f;
        x = (R[1][2] - R[2][1]) / s;
        y = (R[2][0] - R[0][2]) / s;
        z = (R[0][1] - R[1][0]) / s;
        // std::cout << "Case 1: trace > 0, s=" << s << "\n";
    }
    else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        // Case 2: R[0][0] is largest
        s = 2.0f * std::sqrt(1.0f + R[0][0] - R[1][1] - R[2][2]);
        x = s / 4.0f;
        w = (R[1][2] - R[2][1]) / s;
        y = (R[0][1] + R[1][0]) / s;
        z = (R[0][2] + R[2][0]) / s;
        // std::cout << "Case 2: R[0][0] largest, s=" << s << "\n";
    }
    else if (R[1][1] > R[2][2]) {
        // Case 3: R[1][1] is largest
        s = 2.0f * std::sqrt(1.0f + R[1][1] - R[0][0] - R[2][2]);
        y = s / 4.0f;
        w = (R[2][0] - R[0][2]) / s;
        x = (R[0][1] + R[1][0]) / s;
        z = (R[1][2] + R[2][1]) / s;
        // std::cout << "Case 3: R[1][1] largest, s=" << s << "\n";
    }
    else {
        // Case 4: R[2][2] is largest
        s = 2.0f * std::sqrt(1.0f + R[2][2] - R[0][0] - R[1][1]);
        z = s / 4.0f;
        w = (R[0][1] - R[1][0]) / s;
        x = (R[0][2] + R[2][0]) / s;
        y = (R[1][2] + R[2][1]) / s;
        // std::cout << "Case 4: R[2][2] largest, s=" << s << "\n";
    } 
    // Normalize and return
    glm::vec4 q(x, y, z, w);
    return glm::normalize(q);
}

// -------------------------------------------------------------------------
// Step 4: Ground truth — uses GLM's built-in quaternion library
// -------------------------------------------------------------------------

glm::vec4 quat_ground_truth(float pitch, float yaw, float roll,
                             RotationOrder order) {
    // Build quaternions from axis-angle using GLM
    glm::quat qx = glm::angleAxis(pitch, glm::vec3(1, 0, 0));
    glm::quat qy = glm::angleAxis(yaw,   glm::vec3(0, 1, 0));
    glm::quat qz = glm::angleAxis(roll,  glm::vec3(0, 0, 1));

    // Compose in the same order as compose_rotation
    // GLM quaternion multiply: q1 * q2 applies q2 first, then q1
    // This matches the matrix convention: M1 * M2 applies M2 first
    glm::quat result;
    switch (order) {
        case RotationOrder::XYZ: result = qz * qy * qx; break;
        case RotationOrder::XZY: result = qy * qz * qx; break;
        case RotationOrder::YXZ: result = qz * qx * qy; break;
        case RotationOrder::YZX: result = qx * qz * qy; break;
        case RotationOrder::ZXY: result = qy * qx * qz; break;
        case RotationOrder::ZYX: result = qx * qy * qz; break;
    }

    // GLM quat stores as (w, x, y, z) internally, but we use vec4(x,y,z,w)
    return glm::vec4(result.x, result.y, result.z, result.w);
}

// -------------------------------------------------------------------------
// Model matrix builder
// -------------------------------------------------------------------------

glm::mat4 build_model_matrix(const glm::mat3& rot,
                              const glm::vec3& translation) {
    glm::mat4 m(1.0f);

    // Copy 3x3 rotation into upper-left block
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
            m[col][row] = rot[col][row];
        }
    }

    // Set translation in column 3
    m[3][0] = translation.x;
    m[3][1] = translation.y;
    m[3][2] = translation.z;

    return m;
}
 
glm::mat3 quaternion_to_matrix(const glm::quat& q) {
    // Convert quaternion to rotation matrix using GLM
    return glm::mat3_cast(q); 
}