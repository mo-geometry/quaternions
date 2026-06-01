#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

/// =========================================================================
/// SO(3) Rotation & Quaternion Extraction — Brian's implementation area
///
/// Approach:
///   1. Build individual rotation matrices Rx, Ry, Rz from Euler angles
///   2. Compose them in a user-selected order (6 options)
///   3. Extract the unit quaternion from the resulting SO(3) matrix
///   4. Compare against GLM's ground-truth quaternion
///
/// This demonstrates understanding of the relationship between rotation
/// representations: Euler angles -> SO(3) matrix -> unit quaternion.
///
/// A quaternion q = w + xi + yj + zk is stored as glm::vec4(x, y, z, w)
/// to match GLM's convention.
///
/// GLM stores matrices in COLUMN-MAJOR order:
///   mat3 m;
///   m[col][row]   — e.g. m[0][1] is column 0, row 1
///
/// TODO(Brian): Implement the functions marked with TODO below.
///              Reference: your parallel_transport quaternion module.
/// =========================================================================

/// Rotation order for composing Euler angle matrices.
///
/// E.g. XYZ means: R = Rz * Ry * Rx  (rightmost applied first).
/// The convention is INTRINSIC rotations: the axes rotate with the body.
/// For intrinsic XYZ, the equivalent extrinsic order is ZYX, hence
/// R_total = R_last * R_middle * R_first.
enum class RotationOrder {
    XYZ,  // R = Rz * Ry * Rx
    XZY,  // R = Ry * Rz * Rx
    YXZ,  // R = Rz * Rx * Ry
    YZX,  // R = Rx * Rz * Ry
    ZXY,  // R = Ry * Rx * Rz
    ZYX   // R = Rx * Ry * Rz
};

/// Labels for the rotation order combobox.
inline const char* rotation_order_names[] = {
    "Pitch-Yaw-Roll", "Pitch-Roll-Yaw", "Yaw-Pitch-Roll", "Yaw-Roll-Pitch", "Roll-Pitch-Yaw", "Roll-Yaw-Pitch"
};

// -------------------------------------------------------------------------
// Step 1: Elementary rotation matrices
// -------------------------------------------------------------------------

/// Build a 3x3 rotation matrix about the X axis.
///
/// @param angle  Rotation angle in radians.
/// @return       3x3 rotation matrix (GLM column-major).
///
///         | 1     0       0    |
///  Rx =   | 0   cos(a)  -sin(a)|
///         | 0   sin(a)   cos(a)|
glm::mat3 rotation_x(float angle);

/// Build a 3x3 rotation matrix about the Y axis.
///
/// @param angle  Rotation angle in radians.
/// @return       3x3 rotation matrix (GLM column-major).
///
///         | cos(a)   0   sin(a)|
///  Ry =   |   0      1     0  |
///         |-sin(a)   0   cos(a)|
glm::mat3 rotation_y(float angle);

/// Build a 3x3 rotation matrix about the Z axis.
///
/// @param angle  Rotation angle in radians.
/// @return       3x3 rotation matrix (GLM column-major).
///
///         | cos(a)  -sin(a)  0|
///  Rz =   | sin(a)   cos(a)  0|
///         |   0        0     1|
glm::mat3 rotation_z(float angle);

// -------------------------------------------------------------------------
// Step 2: Compose rotation matrices in selected order
// -------------------------------------------------------------------------

/// Compose Rx, Ry, Rz in the specified intrinsic rotation order.
///
/// @param pitch  Rotation about X axis (radians).
/// @param yaw    Rotation about Y axis (radians).
/// @param roll   Rotation about Z axis (radians).
/// @param order  One of the 6 Euler rotation orders.
/// @return       Combined 3x3 rotation matrix.
glm::mat3 compose_rotation(float pitch, float yaw, float roll,
                            RotationOrder order);

// -------------------------------------------------------------------------
// Step 3: Extract quaternion from rotation matrix
// -------------------------------------------------------------------------

/// Extract a unit quaternion from a 3x3 rotation matrix.
///
/// @param R  A valid SO(3) rotation matrix.
/// @return   Unit quaternion as vec4(x, y, z, w).
///
/// Uses the Shepperd method (numerically stable):
///   trace = R[0][0] + R[1][1] + R[2][2]
///
///   if trace > 0:
///     s = 0.5 / sqrt(trace + 1)
///     w = 0.25 / s
///     x = (R[2][1] - R[1][2]) * s
///     y = (R[0][2] - R[2][0]) * s
///     z = (R[1][0] - R[0][1]) * s
///   else:
///     find the largest diagonal element and branch accordingly
///
/// NOTE: GLM is column-major, so R[col][row].
///   R[0][0] = row 0, col 0
///   R[1][0] = row 0, col 1
///   R[0][1] = row 1, col 0
///   etc.
glm::vec4 quat_from_rotation_matrix(const glm::mat3& R);

/// Convert a GLM quaternion back to a 3x3 rotation matrix.
///
/// Uses glm::mat3_cast internally. This lets us round-trip:
///   Euler angles -> SO(3) matrix -> quaternion -> matrix (via this function)
/// and verify the reconstructed matrix matches the original.
///
/// @param q  Unit quaternion.
/// @return   3x3 rotation matrix (GLM column-major).
glm::mat3 quaternion_to_matrix(const glm::quat& q);

// -------------------------------------------------------------------------
// Step 4: Ground truth comparison
// -------------------------------------------------------------------------

/// Compute the GLM ground-truth quaternion from Euler angles.
///
/// Uses glm::angleAxis and quaternion multiplication internally.
/// Returns vec4(x, y, z, w) for direct comparison with your result.
///
/// @param pitch  Rotation about X axis (radians).
/// @param yaw    Rotation about Y axis (radians).
/// @param roll   Rotation about Z axis (radians).
/// @param order  Rotation order (must match compose_rotation).
/// @return       Unit quaternion as vec4(x, y, z, w).
glm::vec4 quat_ground_truth(float pitch, float yaw, float roll,
                             RotationOrder order);

// -------------------------------------------------------------------------
// Model matrix builder (unchanged)
// -------------------------------------------------------------------------

/// Build a 4x4 model matrix from a 3x3 rotation and translation.
///
/// The upper-left 3x3 is the rotation matrix.
/// Column 3 (indices [3][0..2]) holds the translation.
///
/// @param rot         3x3 rotation matrix.
/// @param translation World-space translation (x, y, z).
/// @return            4x4 model matrix (column-major, GLM convention).
glm::mat4 build_model_matrix(const glm::mat3& rot,
                              const glm::vec3& translation);
