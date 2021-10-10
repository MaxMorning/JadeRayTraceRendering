cmake_minimum_required(VERSION 3.9)
project(Clion)

set(CMAKE_CXX_STANDARD 11)

# 添加头文件
set(GLEW_H /usr/local/Cellar/glew/2.2.0_1/include)
set(GLFW_H /usr/local/Cellar/glfw/3.3.3/include/GLFW)
set(Freeglut_H /usr/local/Cellar/freeglut/3.2.1_1/include/GL)
set(GLM_H .)
include_directories(${GLEW_H} ${GLFW_H} ${GLM_H} ${Freeglut_H})

# 添加目标链接
set(GLEW_LINK /usr/local/Cellar/glew/2.2.0_1/lib/libGLEW.2.2.0.dylib)
set(GLFW_LINK /usr/local/Cellar/glfw/3.3.3/lib/libglfw.3.3.dylib)
set(Freeglut_LINK /usr/local/Cellar/freeglut/3.2.1_1/lib/libglut.3.11.0.dylib)
link_libraries(${OPENGL} ${GLEW_LINK} ${GLFW_LINK} ${Freeglut_LINK})

# 执行编译命令
add_executable(Clion main2.cpp lib/hdrloader.cpp glad.c)

if (APPLE)
    target_link_libraries(Clion "-framework OpenGL")
    target_link_libraries(Clion "-framework GLUT")
endif()