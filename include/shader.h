#pragma once

#include "gl_header.h"
#include <string>

/// Compile a vertex + fragment shader pair and return the program ID.
/// Prints compile/link errors to stderr and returns 0 on failure.
GLuint create_shader_program(const std::string& vert_src,
                             const std::string& frag_src);
