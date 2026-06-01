/// =========================================================================
/// Quaternion Bunny Viewer
///
/// A C++ OpenGL application that loads the Stanford bunny and provides
/// interactive rotation via quaternion algebra, with a real-time matrix
/// display in a Dear ImGui side panel.
///
/// Architecture:
///   - GLFW:    windowing, input, OpenGL context
///   - GLM:     vector/matrix maths (camera, ground truth quaternions)
///   - ImGui:   immediate-mode GUI for the control panel
///   - tinyobjloader: OBJ mesh loading
///
/// The render loop:
///   1. Process input (orbit camera drag, scroll)
///   2. Update model matrix (ImGui sliders → Euler angles → SO(3)
///      rotation matrix → model matrix; extract quaternion for display)
///   3. Upload uniforms (model, view, projection, light)
///   4. Draw mesh
///   5. Draw ground grid
///   6. Render ImGui panel
///   7. Swap buffers
/// =========================================================================

#include "gl_header.h"
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "camera.h"
#include "mesh.h"
#include "quaternion.h"
#include "shader.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// ---------------------------------------------------------------------------
// Shader sources (embedded)
// ---------------------------------------------------------------------------

static const char* blinn_phong_vs = R"(
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform mat3 normal_matrix;

out vec3 frag_pos;
out vec3 frag_normal;

void main() {
    vec4 world_pos = model * vec4(in_position, 1.0);
    frag_pos    = world_pos.xyz;
    frag_normal = normalize(normal_matrix * in_normal);
    gl_Position = projection * view * world_pos;
}
)";

static const char* blinn_phong_fs = R"(
#version 330 core
in vec3 frag_pos;
in vec3 frag_normal;

uniform vec3 light_dir;
uniform vec3 eye_pos;
uniform vec3 object_color;

out vec4 out_color;

void main() {
    vec3 N = normalize(frag_normal);
    vec3 L = normalize(-light_dir);
    vec3 V = normalize(eye_pos - frag_pos);
    vec3 H = normalize(L + V);

    float ambient  = 0.15;
    float diffuse  = max(dot(N, L), 0.0);
    float specular = pow(max(dot(N, H), 0.0), 64.0);

    vec3 color = object_color * (ambient + 0.7 * diffuse) + vec3(0.3) * specular;
    out_color = vec4(color, 1.0);
}
)";

static const char* grid_vs = R"(
#version 330 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;

uniform mat4 view;
uniform mat4 projection;

out vec3 frag_color;

void main() {
    frag_color  = in_color;
    gl_Position = projection * view * vec4(in_position, 1.0);
}
)";

static const char* grid_fs = R"(
#version 330 core
in vec3 frag_color;
out vec4 out_color;

void main() {
    out_color = vec4(frag_color, 1.0);
}
)";

// ---------------------------------------------------------------------------
// Ground grid geometry
// ---------------------------------------------------------------------------

struct GridData {
    GLuint vao = 0, vbo = 0;
    int vertex_count = 0;
};

static GridData create_grid(float size = 1.0f, int divisions = 20) {
    std::vector<float> data;  // interleaved: pos(3) + color(3)

    float step = (2.0f * size) / static_cast<float>(divisions);
    float grey[3]  = {0.35f, 0.35f, 0.35f};
    float red[3]   = {0.8f, 0.2f, 0.2f};
    float blue[3]  = {0.2f, 0.2f, 0.8f};

    for (int i = 0; i <= divisions; ++i) {
        float coord = -size + static_cast<float>(i) * step;
        bool is_axis = (std::abs(coord) < step * 0.01f);

        // Line parallel to X axis (varies Z)
        float* c1 = is_axis ? red : grey;
        data.insert(data.end(), {-size, 0.0f, coord, c1[0], c1[1], c1[2]});
        data.insert(data.end(), { size, 0.0f, coord, c1[0], c1[1], c1[2]});

        // Line parallel to Z axis (varies X)
        float* c2 = is_axis ? blue : grey;
        data.insert(data.end(), {coord, 0.0f, -size, c2[0], c2[1], c2[2]});
        data.insert(data.end(), {coord, 0.0f,  size, c2[0], c2[1], c2[2]});
    }

    GridData g;
    g.vertex_count = static_cast<int>(data.size() / 6);

    glGenVertexArrays(1, &g.vao);
    glGenBuffers(1, &g.vbo);
    glBindVertexArray(g.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 data.size() * sizeof(float), data.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
                          (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    return g;
}

// ---------------------------------------------------------------------------
// Globals for GLFW callbacks
// ---------------------------------------------------------------------------

static OrbitCamera g_camera;
static bool        g_dragging     = false;
static double      g_last_mouse_x = 0.0;
static double      g_last_mouse_y = 0.0;

static void mouse_button_callback(GLFWwindow* window, int button,
                                  int action, int /*mods*/) {
    // Don't capture input when ImGui wants it
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_dragging = true;
            glfwGetCursorPos(window, &g_last_mouse_x, &g_last_mouse_y);
        } else if (action == GLFW_RELEASE) {
            g_dragging = false;
        }
    }
}

static void cursor_pos_callback(GLFWwindow* /*window*/, double xpos,
                                double ypos) {
    if (ImGui::GetIO().WantCaptureMouse) return;

    if (g_dragging) {
        float dx = static_cast<float>(xpos - g_last_mouse_x);
        float dy = static_cast<float>(ypos - g_last_mouse_y);
        g_camera.on_mouse_drag(dx, dy);
    }
    g_last_mouse_x = xpos;
    g_last_mouse_y = ypos;
}

static void scroll_callback(GLFWwindow* /*window*/, double /*xoffset*/,
                             double yoffset) {
    if (ImGui::GetIO().WantCaptureMouse) return;
    g_camera.on_scroll(static_cast<float>(yoffset));
}

static void key_callback(GLFWwindow* window, int key, int /*scancode*/,
                          int action, int /*mods*/) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
}

// ---------------------------------------------------------------------------
// ImGui panel — control panel with matrix display
// ---------------------------------------------------------------------------

struct ModelState {
    float pitch = 0.0f;  // radians
    float yaw   = 0.0f;
    float roll  = 0.0f;
    float tx    = 0.0f;  // translation
    float ty    = 0.0f;
    float tz    = 0.0f;
    int   rotation_order = 0;  // index into RotationOrder enum / names
};

static void render_imgui_panel(ModelState& state, const glm::mat4& model_mat) {
    // Fixed-width side panel on the right
    ImGuiIO& io = ImGui::GetIO();
    float panel_width = 340.0f;
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - panel_width, 0.0f));
    ImGui::SetNextWindowSize(ImVec2(panel_width, io.DisplaySize.y));

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoMove
                           | ImGuiWindowFlags_NoResize
                           | ImGuiWindowFlags_NoCollapse;

    ImGui::Begin("Quaternion Controls", nullptr, flags);

    // --- Euler angle sliders ---
    ImGui::SeparatorText("Euler Angles");
    float deg_pitch = glm::degrees(state.pitch);
    float deg_yaw   = glm::degrees(state.yaw);
    float deg_roll  = glm::degrees(state.roll);

    if (ImGui::SliderFloat("Pitch (X)", &deg_pitch, -180.0f, 180.0f, "%.1f"))
        state.pitch = glm::radians(deg_pitch);
    if (ImGui::SliderFloat("Yaw (Y)",   &deg_yaw,   -180.0f, 180.0f, "%.1f"))
        state.yaw = glm::radians(deg_yaw);
    if (ImGui::SliderFloat("Roll (Z)",  &deg_roll,  -180.0f, 180.0f, "%.1f"))
        state.roll = glm::radians(deg_roll);

    // --- Rotation order combobox ---
    ImGui::Combo("Order", &state.rotation_order,
                 rotation_order_names,
                 IM_ARRAYSIZE(rotation_order_names));

    // --- Translation sliders ---
    ImGui::SeparatorText("Translation");
    ImGui::SliderFloat("X##tx", &state.tx, -1.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Y##ty", &state.ty, -0.5f, 0.5f, "%.3f");
    ImGui::SliderFloat("Z##tz", &state.tz, -1.0f, 1.0f, "%.3f");

    if (ImGui::Button("Reset")) {
        state = ModelState{};
    }

    // --- Compute rotation matrix from Euler angles ---
    RotationOrder order = static_cast<RotationOrder>(state.rotation_order);
    glm::mat3 rot = compose_rotation(state.pitch, state.yaw, state.roll, order);

    // --- SO(3) Rotation Matrix display ---
    ImGui::SeparatorText("SO(3) Rotation Matrix");
    const float* r = glm::value_ptr(rot);

    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2, 2));
    ImGui::BeginTable("rot_matrix", 3, ImGuiTableFlags_SizingFixedFit);
    for (int row = 0; row < 3; ++row) {
        ImGui::TableNextRow();
        for (int col = 0; col < 3; ++col) {
            ImGui::TableSetColumnIndex(col);
            ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f),
                               "%+.4f", r[col * 3 + row]);
        }
    }
    ImGui::EndTable();
    ImGui::PopStyleVar();

    // --- Extracted quaternion (Brian's implementation) ---
    ImGui::SeparatorText("Extracted Quaternion");
    glm::vec4 q = quat_from_rotation_matrix(rot);
    float norm = std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    ImGui::Text("w: %+.4f", q.w);
    ImGui::Text("x: %+.4f  y: %+.4f  z: %+.4f", q.x, q.y, q.z);
    ImGui::Text("||q|| = %.6f", norm);

    // --- GLM ground truth quaternion ---
    ImGui::SeparatorText("Ground Truth (GLM)");
    glm::vec4 gt = quat_ground_truth(state.pitch, state.yaw, state.roll, order);
    ImGui::Text("w: %+.4f", gt.w);
    ImGui::Text("x: %+.4f  y: %+.4f  z: %+.4f", gt.x, gt.y, gt.z);

    // --- Error metric ---
    // Quaternion distance: q and -q represent the same rotation,
    // so we compare min(||q - gt||, ||q + gt||)
    float d1 = glm::length(glm::vec4(q.x - gt.x, q.y - gt.y,
                                      q.z - gt.z, q.w - gt.w));
    float d2 = glm::length(glm::vec4(q.x + gt.x, q.y + gt.y,
                                      q.z + gt.z, q.w + gt.w));
    float err = std::min(d1, d2);

    if (err < 1e-4f) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.3f, 1.0f),
                           "Error: %.2e  PASS", err);
    } else {
        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                           "Error: %.2e", err);
    }

    // --- Round-trip: quaternion -> rotation matrix ---
    // Build the model matrix from the EXTRACTED quaternion's rotation,
    // not the original compose_rotation. This proves the round-trip works:
    //   Euler -> SO(3) -> quaternion -> SO(3) -> model matrix
    glm::quat q_glm(q.w, q.x, q.y, q.z);  // GLM quat constructor: (w,x,y,z)
    glm::mat3 rot_from_quat = quaternion_to_matrix(q_glm);
    glm::mat4 model_roundtrip = build_model_matrix(rot_from_quat,
        glm::vec3(state.tx, state.ty, state.tz));

    // --- 4x3 Model matrix display (from round-tripped quaternion) ---
    ImGui::SeparatorText("Model Matrix (4x3) [q -> R]");
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(2, 2));

    const float* m = glm::value_ptr(model_roundtrip);

    ImGui::BeginTable("matrix", 4, ImGuiTableFlags_SizingFixedFit);
    for (int row = 0; row < 3; ++row) {
        ImGui::TableNextRow();
        for (int col = 0; col < 4; ++col) {
            ImGui::TableSetColumnIndex(col);
            float val = m[col * 4 + row];
            if (col < 3) {
                ImGui::TextColored(ImVec4(0.6f, 0.8f, 1.0f, 1.0f),
                                   "%+.3f", val);
            } else {
                ImGui::TextColored(ImVec4(0.6f, 1.0f, 0.6f, 1.0f),
                                   "%+.3f", val);
            }
        }
    }
    ImGui::EndTable();
    ImGui::PopStyleVar();

    ImGui::TextWrapped("Round-trip: Euler -> R -> q -> R -> model");

    // --- Help ---
    ImGui::SeparatorText("Controls");
    ImGui::TextWrapped(
        "Left-drag: orbit camera\n"
        "Scroll: zoom\n"
        "Sliders: rotate/translate bunny\n"
        "Esc: quit"
    );

    ImGui::End();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    // --- GLFW init ---
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialise GLFW\n");
        return EXIT_FAILURE;
    }

    // macOS requires forward-compatible core profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    int win_w = 1280, win_h = 800;
    GLFWwindow* window = glfwCreateWindow(win_w, win_h,
                                           "Quaternion Bunny Viewer",
                                           nullptr, nullptr);
    if (!window) {
        std::fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // vsync

    std::printf("OpenGL %s, GLSL %s\n",
                glGetString(GL_VERSION),
                glGetString(GL_SHADING_LANGUAGE_VERSION));

    // --- GLFW callbacks ---
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

    // --- ImGui init ---
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
    ImGui::StyleColorsDark();

    // --- Shaders ---
    GLuint mesh_prog = create_shader_program(blinn_phong_vs, blinn_phong_fs);
    GLuint grid_prog = create_shader_program(grid_vs, grid_fs);
    if (!mesh_prog || !grid_prog) {
        std::fprintf(stderr, "Shader compilation failed\n");
        return EXIT_FAILURE;
    }

    // --- Load bunny mesh ---
    Mesh bunny;
    if (!bunny.load("assets/bunny.obj")) {
        std::fprintf(stderr, "Failed to load bunny mesh\n");
        return EXIT_FAILURE;
    }

    // --- Grid ---
    GridData grid = create_grid(1.0f, 20);

    // --- OpenGL state ---
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glClearColor(0.12f, 0.12f, 0.14f, 1.0f);

    // --- Light direction (fixed, world space) ---
    glm::vec3 light_dir = glm::normalize(glm::vec3(-0.5f, -1.0f, -0.3f));

    // --- Model state (controlled by ImGui) ---
    ModelState model_state;

    // --- Render loop ---
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Framebuffer size (handles Retina scaling)
        int fb_w, fb_h;
        glfwGetFramebufferSize(window, &fb_w, &fb_h);
        glViewport(0, 0, fb_w, fb_h);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --- Matrices ---
        float aspect = static_cast<float>(fb_w) / static_cast<float>(fb_h);
        glm::mat4 projection = glm::perspective(
            glm::radians(45.0f), aspect, 0.01f, 100.0f);
        glm::mat4 view = g_camera.view_matrix();
        glm::vec3 eye  = g_camera.eye_position();

        // --- Model matrix from SO(3) rotation ---
        RotationOrder order = static_cast<RotationOrder>(
            model_state.rotation_order);
        glm::mat3 rot = compose_rotation(
            model_state.pitch, model_state.yaw, model_state.roll, order);
        glm::vec3 translation(model_state.tx, model_state.ty, model_state.tz);
        glm::mat4 model = build_model_matrix(rot, translation);

        // Normal matrix: transpose of inverse of upper-left 3x3
        glm::mat3 normal_mat = glm::transpose(glm::inverse(glm::mat3(model)));

        // --- Draw bunny ---
        glUseProgram(mesh_prog);
        glUniformMatrix4fv(glGetUniformLocation(mesh_prog, "model"),
                           1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(mesh_prog, "view"),
                           1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(mesh_prog, "projection"),
                           1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix3fv(glGetUniformLocation(mesh_prog, "normal_matrix"),
                           1, GL_FALSE, glm::value_ptr(normal_mat));
        glUniform3fv(glGetUniformLocation(mesh_prog, "light_dir"),
                     1, glm::value_ptr(light_dir));
        glUniform3fv(glGetUniformLocation(mesh_prog, "eye_pos"),
                     1, glm::value_ptr(eye));
        glUniform3f(glGetUniformLocation(mesh_prog, "object_color"),
                    0.75f, 0.55f, 0.35f);  // warm clay colour

        bunny.draw();

        // --- Draw grid ---
        glUseProgram(grid_prog);
        glUniformMatrix4fv(glGetUniformLocation(grid_prog, "view"),
                           1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(grid_prog, "projection"),
                           1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(grid.vao);
        glDrawArrays(GL_LINES, 0, grid.vertex_count);
        glBindVertexArray(0);

        // --- ImGui ---
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        render_imgui_panel(model_state, model);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // --- Cleanup ---
    bunny.cleanup();
    glDeleteBuffers(1, &grid.vbo);
    glDeleteVertexArrays(1, &grid.vao);
    glDeleteProgram(mesh_prog);
    glDeleteProgram(grid_prog);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return EXIT_SUCCESS;
}
