#include "camera.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_transform.hpp>

glm::vec3 OrbitCamera::eye_position() const {
    float x = target.x + distance * std::cos(elevation) * std::sin(azimuth);
    float y = target.y + distance * std::sin(elevation);
    float z = target.z + distance * std::cos(elevation) * std::cos(azimuth);
    return glm::vec3(x, y, z);
}

glm::mat4 OrbitCamera::view_matrix() const {
    glm::vec3 eye = eye_position();
    return glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

void OrbitCamera::on_mouse_drag(float dx, float dy) {
    azimuth  -= dx * sensitivity;
    elevation += dy * sensitivity;

    // Clamp elevation to avoid gimbal lock at poles
    const float limit = glm::radians(89.0f);
    elevation = std::clamp(elevation, -limit, limit);
}

void OrbitCamera::on_scroll(float delta) {
    distance -= delta * zoom_speed;
    distance  = std::clamp(distance, min_dist, max_dist);
}
