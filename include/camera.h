#pragma once

#include <glm/glm.hpp>

/// Simple orbit camera that produces a view matrix.
///
/// The camera orbits around a target point.  Drag to rotate,
/// scroll to zoom.  The view matrix is recomputed each frame
/// from azimuth, elevation, and distance.
struct OrbitCamera {
    glm::vec3 target   = glm::vec3(0.0f, 0.08f, 0.0f);  // bunny centre approx
    float     distance = 0.35f;
    float     azimuth  = 0.8f;   // radians
    float     elevation = 0.4f;  // radians

    float     sensitivity = 0.005f;
    float     zoom_speed  = 0.02f;
    float     min_dist    = 0.05f;
    float     max_dist    = 5.0f;

    /// Compute the 4x4 view matrix from current orbit parameters.
    glm::mat4 view_matrix() const;

    /// Compute the camera's world-space eye position.
    glm::vec3 eye_position() const;

    /// Apply a mouse drag delta (in pixels) to rotate the orbit.
    void on_mouse_drag(float dx, float dy);

    /// Apply a scroll delta to zoom in/out.
    void on_scroll(float delta);
};
