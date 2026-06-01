#include "shader.h"

#include <iostream>
#include <vector>

/// Check for compile errors on a single shader stage.
static bool check_compile(GLuint shader, const char* label) {
    GLint success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(shader, sizeof(info), nullptr, info);
        std::cerr << label << " compile error:\n" << info << "\n";
        return false;
    }
    return true;
}

/// Check for link errors on a shader program.
static bool check_link(GLuint program) {
    GLint success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char info[512];
        glGetProgramInfoLog(program, sizeof(info), nullptr, info);
        std::cerr << "Program link error:\n" << info << "\n";
        return false;
    }
    return true;
}

GLuint create_shader_program(const std::string& vert_src,
                             const std::string& frag_src) {
    // --- Vertex shader ---
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    const char* vs_c = vert_src.c_str();
    glShaderSource(vs, 1, &vs_c, nullptr);
    glCompileShader(vs);
    if (!check_compile(vs, "Vertex shader")) {
        glDeleteShader(vs);
        return 0;
    }

    // --- Fragment shader ---
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    const char* fs_c = frag_src.c_str();
    glShaderSource(fs, 1, &fs_c, nullptr);
    glCompileShader(fs);
    if (!check_compile(fs, "Fragment shader")) {
        glDeleteShader(vs);
        glDeleteShader(fs);
        return 0;
    }

    // --- Link program ---
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    // Shaders can be deleted after linking
    glDeleteShader(vs);
    glDeleteShader(fs);

    if (!check_link(prog)) {
        glDeleteProgram(prog);
        return 0;
    }

    return prog;
}
