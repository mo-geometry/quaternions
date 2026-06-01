#pragma once

/// Platform-specific OpenGL header.
///
/// On macOS, the system OpenGL.framework provides all GL 3.3 core
/// functions directly — no loader library (glad/GLEW) needed.
/// On other platforms, add glad or GLEW here.

#ifdef __APPLE__
    #ifndef GL_SILENCE_DEPRECATION
        #define GL_SILENCE_DEPRECATION
    #endif
    #include <OpenGL/gl3.h>
#else
    // For cross-platform support, add glad here:
    // #include <glad/glad.h>
    #error "Non-Apple platforms need a GL loader (glad or GLEW)"
#endif
