cmake_minimum_required(VERSION 3.9)
project(JadeRender)

set(CMAKE_CXX_STANDARD 11)

if (APPLE)
else()
    find_package(CUDA REQUIRED)
endif()
# 添加头文件
if (APPLE)
    set(GLEW_H /usr/local/Cellar/glew/2.2.0_1/include)
    set(GLFW_H /usr/local/Cellar/glfw/3.3.3/include/GLFW)
else()
    set(GLEW_H E:\\Download\\glew-2.1.0-win32\\include)
    set(GLFW_H F:\\TestCode\\CPP\\JadeRayTraceRendering\\lib\\GLFW\\includes\\GLFW)
endif()

#set(Freeglut_H /usr/local/Cellar/freeglut/3.2.1_1/include/GL)
set(GLM_H .)
include_directories(${GLEW_H} ${GLFW_H} ${GLM_H})

# 添加目标链接
if (APPLE)
    set(GLEW_LINK /usr/local/Cellar/glew/2.2.0_1/lib/libGLEW.2.2.0.dylib)
    set(GLFW_LINK /usr/local/Cellar/glfw/3.3.3/lib/libglfw.3.3.dylib)
else()
    set(OPENGL "C:\\Program Files (x86)\\Windows Kits\\10\\Lib\\10.0.18362.0\\um\\x64\\OpenGL32.Lib")
    set(GLEW_LINK E:\\Download\\glew-2.1.0-win32\\lib\\Release\\x64\\glew32.lib)
    set(GLFW_LINK F:\\TestCode\\CPP\\JadeRayTraceRendering\\lib\\GLFW\\libs\\glfw3.lib)
#set(Freeglut_LINK /usr/local/Cellar/freeglut/3.2.1_1/lib/libglut.3.11.0.dylib)
endif()
link_libraries(${OPENGL} ${GLEW_LINK} ${GLFW_LINK})

# 执行编译命令
add_executable(JadeRender PathTrace.cpp lib/hdrloader.cpp)

if (APPLE)
    target_link_libraries(JadeRender "-framework OpenGL")
    target_link_libraries(JadeRender "-framework GLUT")
endif()