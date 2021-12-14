//
// Project: Clion
// File Name: PathTrace.cpp
// Author: Morning
// Description:
//
// Create Date: 2021/10/9
//

#include <GL/glew.h>
#include <glfw3.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <lib/hdrloader.h>

using namespace glm;

#define INF 2147483647
#define WIDTH 1024
#define HEIGHT 1024

#define DIFFUSE 0
#define MIRROR 1

#define NO_REFRACT 0
#define SUB_SURFACE 1
#define DIR_REFRACT 2
// ----------------------------------------------------------------------------- //

// 物体表面材质定义
struct Material {
    vec3 emissive = vec3(0, 0, 0);  // 作为光源时的发光颜色
    vec3 brdf = vec3(0.8, 0.8, 0.8); // BRDF
    int reflex_mode;           // 反射模式，漫反射0 / 镜面反射1
    int refract_mode;           // 折射模式，无透射0 / 次表面散射1 / 直接折射2
    vec3 refract_rate = vec3(0.8, 0.8, 0.8); // 折射吸光率
    vec3 refract_albedo = vec3(0.8, 0.8, 0.8); // 折射反照率
    float refract_index;     // 折射率
};

// 三角形定义
struct Triangle {
    vec3 p1, p2, p3;    // 顶点坐标
    vec3 norm;    // 顶点法线
    Material material;  // 材质
};

// BVH 树节点
struct BVHNode {
    int left, right;    // 左右子树索引
    int n, index;       // 叶子节点信息
    vec3 AA, BB;        // 碰撞盒
};

// ----------------------------------------------------------------------------- //

struct Triangle_encoded {
    vec3 p1, p2, p3;    // 顶点坐标
    vec3 norm;          // 法线
    vec3 emissive;      // 自发光参数
    vec3 brdf;          // BRDF
};

struct BVHNode_encoded {
    vec3 childs;        // (left, right, 保留)
    vec3 leafInfo;      // (n, index, 保留)
    vec3 AA, BB;
};

// ----------------------------------------------------------------------------- //
// BMP Operation
// 文件信息头结构体
typedef struct
{
    unsigned int   bfSize;        // 文件大小 以字节为单位(2-5字节)
    unsigned short bfReserved1;   // 保留，必须设置为0 (6-7字节)
    unsigned short bfReserved2;   // 保留，必须设置为0 (8-9字节)
    unsigned int   bfOffBits;     // 从文件头到像素数据的偏移  (10-13字节)
} _BITMAPFILEHEADER;

//图像信息头结构体
typedef struct
{
    unsigned int    biSize;          // 此结构体的大小 (14-17字节)
    int             biWidth;         // 图像的宽  (18-21字节)
    int             biHeight;        // 图像的高  (22-25字节)
    unsigned short  biPlanes;        // 表示bmp图片的平面属，显然显示器只有一个平面，所以恒等于1 (26-27字节)
    unsigned short  biBitCount;      // 一像素所占的位数，一般为24   (28-29字节)
    unsigned int    biCompression;   // 说明图象数据压缩的类型，0为不压缩。 (30-33字节)
    unsigned int    biSizeImage;     // 像素数据所占大小, 这个值应该等于上面文件头结构中bfSize-bfOffBits (34-37字节)
    int             biXPelsPerMeter; // 说明水平分辨率，用象素/米表示。一般为0 (38-41字节)
    int             biYPelsPerMeter; // 说明垂直分辨率，用象素/米表示。一般为0 (42-45字节)
    unsigned int    biClrUsed;       // 说明位图实际使用的彩色表中的颜色索引数（设为0的话，则说明使用所有调色板项）。 (46-49字节)
    unsigned int    biClrImportant;  // 说明对图象显示有重要影响的颜色索引的数目，如果是0，表示都重要。(50-53字节)
} _BITMAPINFOHEADER;

void save_image(unsigned char* target_img, int width, int height)
{
    FILE* file_ptr = fopen("RenderResultGL.bmp", "wb+");

    unsigned short fileType = 0x4d42;
    _BITMAPFILEHEADER fileHeader;
    _BITMAPINFOHEADER infoHeader;

    fileHeader.bfSize = (width) * (height) * 3 + 54;
    fileHeader.bfReserved1 = 0;
    fileHeader.bfReserved2 = 0;
    fileHeader.bfOffBits = 54;

    infoHeader.biSize = 40;
    infoHeader.biWidth = width;
    infoHeader.biHeight = height;
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0;
    infoHeader.biSizeImage = (width) * (height) * 3;
    infoHeader.biXPelsPerMeter = 0;
    infoHeader.biYPelsPerMeter = 0;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;

    fwrite(&fileType, sizeof(unsigned short), 1, file_ptr);
    fwrite(&fileHeader, sizeof(_BITMAPFILEHEADER), 1, file_ptr);
    fwrite(&infoHeader, sizeof(_BITMAPINFOHEADER), 1, file_ptr);

    fwrite(target_img, sizeof(unsigned char), (height) * (width) * 3, file_ptr);

    fclose(file_ptr);
}

// ----------------------------------------------------------------------------- //

class RenderPass {
public:
    GLuint FBO = 0;
    GLuint vao, vbo;
    std::vector<GLuint> colorAttachments;
    GLuint program;
    int width = WIDTH;
    int height = HEIGHT;
    void bindData(bool finalPass = false) {
        if (!finalPass) glGenFramebuffers(1, &FBO);
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        std::vector<vec3> square = { vec3(-1, -1, 0), vec3(1, -1, 0), vec3(-1, 1, 0), vec3(1, 1, 0), vec3(-1, 1, 0), vec3(1, -1, 0) };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * square.size(), NULL, GL_STATIC_DRAW);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3) * square.size(), &square[0]);

        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);   // layout (location = 0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
        // 不是 finalPass 则生成帧缓冲的颜色附件
        if (!finalPass) {
            std::vector<GLuint> attachments;
            for (int i = 0; i < colorAttachments.size(); i++) {
                glBindTexture(GL_TEXTURE_2D, colorAttachments[i]);
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorAttachments[i], 0);// 将颜色纹理绑定到 i 号颜色附件
                attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
            }
            glDrawBuffers(attachments.size(), &attachments[0]);
        }

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    void draw(std::vector<GLuint> texPassArray = {}) {
        glUseProgram(program);
        glBindFramebuffer(GL_FRAMEBUFFER, FBO);
        glBindVertexArray(vao);
        // 传上一帧的帧缓冲颜色附件
        for (int i = 0; i < texPassArray.size(); i++) {
            glActiveTexture(GL_TEXTURE0+i);
            glBindTexture(GL_TEXTURE_2D, texPassArray[i]);
            std::string uName = "texPass" + std::to_string(i);
            glUniform1i(glGetUniformLocation(program, uName.c_str()), i);
        }
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glUseProgram(0);
    }
};

// ----------------------------------------------------------------------------- //

GLuint trianglesTextureBuffer;
GLuint nodesTextureBuffer;
GLuint emitTrianglesIndices;
GLuint lastFrame;
GLuint hdrMap;

RenderPass pass1;
RenderPass pass2;
RenderPass pass3;

// 相机参数
float upAngle = 0.0;
float rotateAngle = 0.0;
float r = 4.0;

// ----------------------------------------------------------------------------- //
void check_error(int i)
{
    printf("%d : ", i);
    GLenum error;
    if((error = glGetError()) != GL_NO_ERROR)
    {
        switch (error)
        {
            case GL_INVALID_ENUM:
                printf("GL Error: GL_INVALID_ENUM \n");
                break;
            case GL_INVALID_VALUE:
                printf("GL Error: GL_INVALID_VALUE \n");
                break;
            case GL_INVALID_OPERATION:
                printf("GL Error: GL_INVALID_OPERATION \n");
                break;
            case GL_OUT_OF_MEMORY:
                printf("GL Error: GL_OUT_OF_MEMORY \n");
                break;
            default:
                printf("GL Error: 0x%x\n",error);
                break;
        }
    } else {
        printf("\n");
    }
}

// 按照三角形中心排序 -- 比较函数
bool cmpx(const Triangle& t1, const Triangle& t2) {
    vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
    vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
    return center1.x < center2.x;
}
bool cmpy(const Triangle& t1, const Triangle& t2) {
    vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
    vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
    return center1.y < center2.y;
}
bool cmpz(const Triangle& t1, const Triangle& t2) {
    vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
    vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
    return center1.z < center2.z;
}

// ----------------------------------------------------------------------------- //

// 读取文件并且返回一个长字符串表示文件内容
std::string readShaderFile(std::string filepath) {
    std::string res, line;
    std::ifstream fin(filepath);
    if (!fin.is_open())
    {
        std::cout << "File  " << filepath << " open failed" << std::endl;
        exit(-1);
    }
    while (std::getline(fin, line))
    {
        res += line + '\n';
    }
    fin.close();
    return res;
}

GLuint getTextureRGB32F(int width, int height) {
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    return tex;
}

// 获取着色器对象
GLuint getShaderProgram(std::string fshader, std::string vshader) {
    // 读取shader源文件
    std::string vSource = readShaderFile(vshader);
    std::string fSource = readShaderFile(fshader);
    const char* vpointer = vSource.c_str();
    const char* fpointer = fSource.c_str();

    // 容错
    GLint success;
    GLchar infoLog[512];

    // 创建并编译顶点着色器
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, (const GLchar**)(&vpointer), NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);   // 错误检测
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "Vertex Shader compile error.\n" << infoLog << std::endl;
        exit(-1);
    }

    // 创建并且编译片段着色器
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, (const GLchar**)(&fpointer), NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);   // 错误检测
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "Fragment shader compile error.\n" << infoLog << std::endl;
        exit(-1);
    }

    // 链接两个着色器到program对象
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // 删除着色器对象
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    return shaderProgram;
}

// ----------------------------------------------------------------------------- //

// 模型变换矩阵
mat4 getTransformMatrix(vec3 rotateCtrl, vec3 translateCtrl, vec3 scaleCtrl) {
    glm::mat4 unit(    // 单位矩阵
            glm::vec4(1, 0, 0, 0),
            glm::vec4(0, 1, 0, 0),
            glm::vec4(0, 0, 1, 0),
            glm::vec4(0, 0, 0, 1)
    );
    mat4 scale = glm::scale(unit, scaleCtrl);
    mat4 translate = glm::translate(unit, translateCtrl);
    mat4 rotate = unit;
    rotate = glm::rotate(rotate, glm::radians(rotateCtrl.x), glm::vec3(1, 0, 0));
    rotate = glm::rotate(rotate, glm::radians(rotateCtrl.y), glm::vec3(0, 1, 0));
    rotate = glm::rotate(rotate, glm::radians(rotateCtrl.z), glm::vec3(0, 0, 1));

    mat4 model = translate * rotate * scale;
    return model;
}

// 读取 obj
std::vector<std::string> obj_file_name;
std::vector<mat4> obj_trans_mats;
std::vector<Material> obj_materials;
std::vector<bool> obj_normalize;
void readObj(const std::string& filepath, std::vector<Triangle>& triangles, Material material, mat4 trans, bool normal_transform) {
    // 记录
    obj_file_name.push_back(filepath);
    obj_trans_mats.push_back(trans);
    obj_materials.push_back(material);
    obj_normalize.push_back(normal_transform);
    
    // 顶点位置，索引
    std::vector<vec3> vertices;
    std::vector<GLuint> indices;

    // 打开文件流
    std::ifstream fin(filepath);
    std::string line;
    if (!fin.is_open()) {
        std::cout << "File " << filepath << " open failed." << std::endl;
        exit(-1);
    }

    // 计算 AABB 盒，归一化模型大小
    float maxx = -11451419.19;
    float maxy = -11451419.19;
    float maxz = -11451419.19;
    float minx = 11451419.19;
    float miny = 11451419.19;
    float minz = 11451419.19;

    // 按行读取
    while (std::getline(fin, line)) {
        std::istringstream sin(line);   // 以一行的数据作为 string stream 解析并且读取
        std::string type;
        GLfloat x, y, z;
        int v0, v1, v2;

        if (line.length() > 0 && line[0] == '#') {
            continue;
        }
        // 统计斜杆数目，用不同格式读取
        for (int i = 0; i < line.length(); i++) {
            if (line[i] == '/') {
                line[i] = ' ';
            }
        }

        // 读取obj文件
        sin >> type;
        if (type == "v") {
            sin >> x >> y >> z;
            vertices.emplace_back(x, y, z);
            maxx = max(maxx, x); maxy = max(maxx, y); maxz = max(maxx, z);
            minx = min(minx, x); miny = min(minx, y); minz = min(minx, z);
        }
        if (type == "f") {
            sin >> v0 >> v1 >> v2;
            indices.push_back(v0 - 1);
            indices.push_back(v1 - 1);
            indices.push_back(v2 - 1);
        }
    }

    // 模型大小归一化
    if (normal_transform) {
        float lenx = maxx - minx;
        float leny = maxy - miny;
        float lenz = maxz - minz;
        float maxaxis = max(lenx, max(leny, lenz));
        vec3 center = vec3((maxx + minx) / 2, (maxy + miny) / 2, (maxz + minz) / 2);
        for (auto& v : vertices) {
            v -= center;
            v.x /= maxaxis;
            v.y /= maxaxis;
            v.z /= maxaxis;
        }
    }


    // 通过矩阵进行坐标变换
    for (auto& v : vertices) {
        vec4 vv = vec4(v.x, v.y, v.z, 1);
        vv = trans * vv;
        v = vec3(vv.x, vv.y, vv.z);
    }

    // 构建 Triangle 对象数组
    int offset = triangles.size();  // 增量更新

    triangles.resize(offset + indices.size() / 3);

    for (int i = 0; i < indices.size(); i += 3) {
        Triangle& t = triangles[offset + i / 3];
        // 传顶点属性
        t.p1 = vertices[indices[i]];
        t.p2 = vertices[indices[i + 1]];
        t.p3 = vertices[indices[i + 2]];
        // 计算法线
        t.norm = normalize(cross(t.p2 - t.p1, t.p3 - t.p1));

        // 传材质
        t.material = material;
    }
}

// 构建 BVH
int buildBVH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n) {
    if (l > r) return 0;

    // 注：
    // 此处不可通过指针，引用等方式操作，必须用 nodes[id] 来操作
    // 因为 std::vector<> 扩容时会拷贝到更大的内存，那么地址就改变了
    // 而指针，引用均指向原来的内存，所以会发生错误
    nodes.emplace_back();
    int id = nodes.size() - 1;   // 注意： 先保存索引
    nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
    nodes[id].AA = vec3(1145141919, 1145141919, 1145141919);
    nodes[id].BB = vec3(-1145141919, -1145141919, -1145141919);

    // 计算 AABB
    for (int i = l; i <= r; i++) {
        // 最小点 AA
        float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
        float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
        float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].AA.x = min(nodes[id].AA.x, minx);
        nodes[id].AA.y = min(nodes[id].AA.y, miny);
        nodes[id].AA.z = min(nodes[id].AA.z, minz);
        // 最大点 BB
        float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
        float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
        float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].BB.x = max(nodes[id].BB.x, maxx);
        nodes[id].BB.y = max(nodes[id].BB.y, maxy);
        nodes[id].BB.z = max(nodes[id].BB.z, maxz);
    }

    // 不多于 n 个三角形 返回叶子节点
    if ((r - l + 1) <= n) {
        nodes[id].n = r - l + 1;
        nodes[id].index = l;
        return id;
    }

    // 否则递归建树
    float lenx = nodes[id].BB.x - nodes[id].AA.x;
    float leny = nodes[id].BB.y - nodes[id].AA.y;
    float lenz = nodes[id].BB.z - nodes[id].AA.z;
    // 按 x 划分
    if (lenx >= leny && lenx >= lenz)
        std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpx);
    // 按 y 划分
    if (leny >= lenx && leny >= lenz)
        std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpy);
    // 按 z 划分
    if (lenz >= lenx && lenz >= leny)
        std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpz);
    // 递归
    int mid = (l + r) / 2;
    int left = buildBVH(triangles, nodes, l, mid, n);
    int right = buildBVH(triangles, nodes, mid + 1, r, n);

    nodes[id].left = left;
    nodes[id].right = right;

    return id;
}

// SAH 优化构建 BVH
int buildBVHwithSAH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n) {
    if (l > r) return 0;

    nodes.emplace_back();
    int id = nodes.size() - 1;
    nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
    nodes[id].AA = vec3(1145141919, 1145141919, 1145141919);
    nodes[id].BB = vec3(-1145141919, -1145141919, -1145141919);

    // 计算 AABB
    for (int i = l; i <= r; i++) {
        // 最小点 AA
        float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
        float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
        float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].AA.x = min(nodes[id].AA.x, minx);
        nodes[id].AA.y = min(nodes[id].AA.y, miny);
        nodes[id].AA.z = min(nodes[id].AA.z, minz);
        // 最大点 BB
        float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
        float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
        float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
        nodes[id].BB.x = max(nodes[id].BB.x, maxx);
        nodes[id].BB.y = max(nodes[id].BB.y, maxy);
        nodes[id].BB.z = max(nodes[id].BB.z, maxz);
    }

    // 不多于 n 个三角形 返回叶子节点
    if ((r - l + 1) <= n) {
        nodes[id].n = r - l + 1;
        nodes[id].index = l;
        return id;
    }

    // 否则递归建树
    float Cost = INF;
    int Axis = 0;
    int Split = (l + r) / 2;
    for (int axis = 0; axis < 3; axis++) {
        // 分别按 x，y，z 轴排序
        if (axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
        if (axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
        if (axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);

        // leftMax[i]: [l, i] 中最大的 xyz 值
        // leftMin[i]: [l, i] 中最小的 xyz 值
        std::vector<vec3> leftMax(r - l + 1, vec3(-INF, -INF, -INF));
        std::vector<vec3> leftMin(r - l + 1, vec3(INF, INF, INF));
        // 计算前缀 注意 i-l 以对齐到下标 0
        for (int i = l; i <= r; i++) {
            Triangle& t = triangles[i];
            int bias = (i == l) ? 0 : 1;  // 第一个元素特殊处理

            leftMax[i - l].x = max(leftMax[i - l - bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
            leftMax[i - l].y = max(leftMax[i - l - bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
            leftMax[i - l].z = max(leftMax[i - l - bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

            leftMin[i - l].x = min(leftMin[i - l - bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
            leftMin[i - l].y = min(leftMin[i - l - bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
            leftMin[i - l].z = min(leftMin[i - l - bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
        }

        // rightMax[i]: [i, r] 中最大的 xyz 值
        // rightMin[i]: [i, r] 中最小的 xyz 值
        std::vector<vec3> rightMax(r - l + 1, vec3(-INF, -INF, -INF));
        std::vector<vec3> rightMin(r - l + 1, vec3(INF, INF, INF));
        // 计算后缀 注意 i-l 以对齐到下标 0
        for (int i = r; i >= l; i--) {
            Triangle& t = triangles[i];
            int bias = (i == r) ? 0 : 1;  // 第一个元素特殊处理

            rightMax[i - l].x = max(rightMax[i - l + bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
            rightMax[i - l].y = max(rightMax[i - l + bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
            rightMax[i - l].z = max(rightMax[i - l + bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

            rightMin[i - l].x = min(rightMin[i - l + bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
            rightMin[i - l].y = min(rightMin[i - l + bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
            rightMin[i - l].z = min(rightMin[i - l + bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
        }

        // 遍历寻找分割
        float cost = INF;
        int split = l;
        for (int i = l; i <= r - 1; i++) {
            float lenx, leny, lenz;
            // 左侧 [l, i]
            vec3 leftAA = leftMin[i - l];
            vec3 leftBB = leftMax[i - l];
            lenx = leftBB.x - leftAA.x;
            leny = leftBB.y - leftAA.y;
            lenz = leftBB.z - leftAA.z;
            float leftS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
            float leftCost = leftS * (i - l + 1);

            // 右侧 [i+1, r]
            vec3 rightAA = rightMin[i + 1 - l];
            vec3 rightBB = rightMax[i + 1 - l];
            lenx = rightBB.x - rightAA.x;
            leny = rightBB.y - rightAA.y;
            lenz = rightBB.z - rightAA.z;
            float rightS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
            float rightCost = rightS * (r - i);

            // 记录每个分割的最小答案
            float totalCost = leftCost + rightCost;
            if (totalCost < cost) {
                cost = totalCost;
                split = i;
            }
        }
        // 记录每个轴的最佳答案
        if (cost < Cost) {
            Cost = cost;
            Axis = axis;
            Split = split;
        }
    }

    // 按最佳轴分割
    if (Axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
    if (Axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
    if (Axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);

    // 递归
    int left  = buildBVHwithSAH(triangles, nodes, l, Split, n);
    int right = buildBVHwithSAH(triangles, nodes, Split + 1, r, n);

    nodes[id].left = left;
    nodes[id].right = right;

    return id;
}

// ----------------------------------------------------------------------------- //

// 绘制
clock_t t1, t2;
double dt, fps;
unsigned int frameCounter = 0;
vec3 eye_center = vec3(0);
vec3 eye = vec3(0);
mat4 cameraRotate = mat4(1);
void display(GLFWwindow* window, bool swap_buffer) {

    // 帧计时
    t2 = clock();
    dt = (double)(t2 - t1) / CLOCKS_PER_SEC;
    fps = 1.0 / dt;
    std::cout << std::fixed << std::setprecision(2) << "FPS : " << fps << "    Iter time: " << frameCounter << std::endl;
    t1 = t2;

    // 相机参数
    eye = vec3(-sin(radians(rotateAngle)) * cos(radians(upAngle)), sin(radians(upAngle)), cos(radians(rotateAngle)) * cos(radians(upAngle)));
    eye.x *= r; eye.y *= r; eye.z *= r;
    cameraRotate = lookAt(eye, eye_center, vec3(0, 1, 0));  // 相机注视着原点
    cameraRotate = inverse(cameraRotate);   // lookat 的逆矩阵将光线方向进行转换

    // 传 uniform 给 pass1
    glUseProgram(pass1.program);
    glUniform3fv(glGetUniformLocation(pass1.program, "eye"), 1, value_ptr(eye));
    glUniformMatrix4fv(glGetUniformLocation(pass1.program, "cameraRotate"), 1, GL_FALSE, value_ptr(cameraRotate));
    glUniform1ui(glGetUniformLocation(pass1.program, "frameCounter"), frameCounter++);// 传计数器用作随机种子

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);
    glUniform1i(glGetUniformLocation(pass1.program, "triangles"), 0);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
    glUniform1i(glGetUniformLocation(pass1.program, "nodes"), 1);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, lastFrame);
    glUniform1i(glGetUniformLocation(pass1.program, "lastFrame"), 2);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, hdrMap);
    glUniform1i(glGetUniformLocation(pass1.program, "hdrMap"), 3);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_BUFFER, emitTrianglesIndices);
    glUniform1i(glGetUniformLocation(pass1.program, "emitTrianglesIndices"), 4);
    // 绘制
    pass1.draw();
    pass2.draw(pass1.colorAttachments);
    pass3.draw(pass2.colorAttachments);

    if (swap_buffer) {
        glfwSwapBuffers(window);
    }
}

// 每一帧
void frameFunc() {
    glfwPollEvents();
}

#define WASD_DELTA 2
#define ROTATE_DELTA 20
#define KEY_STATUS_SIZE 349

bool key_status[KEY_STATUS_SIZE] = {false};
GLfloat deltaTime = 0.0f;
GLfloat prevFrameTime = 0.0f;

void move_camera(GLFWwindow* window) {
    GLfloat currentFrame = glfwGetTime();
    deltaTime = currentFrame - prevFrameTime;
    prevFrameTime = currentFrame;

    if (key_status[GLFW_KEY_DOWN]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        upAngle -= ROTATE_DELTA * deltaTime;
    }

    if (key_status[GLFW_KEY_UP]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        upAngle += ROTATE_DELTA * deltaTime;
    }

    if (key_status[GLFW_KEY_LEFT]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        rotateAngle += ROTATE_DELTA * deltaTime;
    }

    if (key_status[GLFW_KEY_RIGHT]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        rotateAngle -= ROTATE_DELTA * deltaTime;
    }

    if (key_status[GLFW_KEY_W]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        eye_center.y += WASD_DELTA * deltaTime;
    }

    if (key_status[GLFW_KEY_S]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        eye_center.y -= WASD_DELTA * deltaTime;
    }

    if (key_status[GLFW_KEY_A]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        eye_center.x -= WASD_DELTA * deltaTime;
    }

    if (key_status[GLFW_KEY_D]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        eye_center.x += WASD_DELTA * deltaTime;
    }

    if (key_status[GLFW_KEY_H]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        r -= WASD_DELTA * deltaTime;
    }

    if (key_status[GLFW_KEY_N]) {
        frameCounter = 0;
        glfwRestoreWindow(window);
        r += WASD_DELTA * deltaTime;
    }
}
void generate_arguments();
void offline_render(int spp, GLFWwindow* window);

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode)
{
    //如果按下ESC，把windowShouldClose设置为True，外面的循环会关闭应用
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
            {
                glfwSetWindowShouldClose(window, GL_TRUE);
                std::cout << "ESC" << std::endl;
            }
                break;

            case GLFW_KEY_C:
            {
                std::cout << "SAVE" << std::endl;
                unsigned char* image = new unsigned char[WIDTH * HEIGHT * 3];
                glReadPixels(0, 0, WIDTH, HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, image);
                save_image(image, WIDTH, HEIGHT);
                delete[] image;
            }
                break;

            case GLFW_KEY_R:
            {
                int spp;
                std::cout << "Sample per Pixel : " << std::endl;
                std::cin >> spp;
                offline_render(spp, window);
            }
                break;

            case GLFW_KEY_F:
            {
                generate_arguments();
                glfwSetWindowShouldClose(window, GL_TRUE);
                std::cout << "Saving Cuda Render Args" << std::endl;
            }
                break;

            default:
                key_status[key] = true;
        }
    }
    else if (action == GLFW_RELEASE) {
        key_status[key] = false;
    }
}

std::vector<Triangle> triangles;
std::vector<BVHNode> nodes;
int nEmitTriangles = 0;

void set_shader(std::string fshader_path, int spp)
{
    pass1.program = getShaderProgram(fshader_path, "./shaders/vshader.vsh");
    pass1.colorAttachments.push_back(getTextureRGB32F(pass1.width, pass1.height));
    pass1.bindData();

    glUseProgram(pass1.program);
    glUniform1i(glGetUniformLocation(pass1.program, "nTriangles"), triangles.size());
    glUniform1i(glGetUniformLocation(pass1.program, "nNodes"), nodes.size());
    glUniform1i(glGetUniformLocation(pass1.program, "width"), pass1.width);
    glUniform1i(glGetUniformLocation(pass1.program, "height"), pass1.height);
    glUniform1i(glGetUniformLocation(pass1.program, "nEmitTriangles"), nEmitTriangles);
    if (spp > 0) {
        glUniform1i(glGetUniformLocation(pass1.program, "spp"), spp);
    }
    glUseProgram(0);

    pass2.program = getShaderProgram("./shaders/pass2.fsh", "./shaders/vshader.vsh");
    lastFrame = getTextureRGB32F(pass2.width, pass2.height);
    pass2.colorAttachments.push_back(lastFrame);
    pass2.bindData();

    pass3.program = getShaderProgram("./shaders/pass3.fsh", "./shaders/vshader.vsh");
    pass3.bindData(true);
}
// save args
void generate_arguments()
{
    std::ofstream fout("render_args.txt");
    fout << eye.x << ' ' << eye.y << ' ' << eye.z << std::endl;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            fout << cameraRotate[row][col] << ' ';
        }
        fout << std::endl;
    }

    fout << obj_file_name.size() << std::endl;

    for (int i = 0; i < obj_file_name.size(); ++i) {
        fout << obj_file_name[i] << std::endl;

        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                fout << obj_trans_mats[i][row][col] << ' ';
            }
            fout << std::endl;
        }

        fout << obj_materials[i].emissive.x << ' ' << obj_materials[i].emissive.y << ' ' << obj_materials[i].emissive.z << std::endl;
        fout << obj_materials[i].brdf.x << ' ' << obj_materials[i].brdf.y << ' ' << obj_materials[i].brdf.z << std::endl;
        fout << obj_materials[i].reflex_mode << std::endl;
        fout << obj_materials[i].refract_mode << std::endl;
        fout << obj_materials[i].refract_rate.x << ' ' << obj_materials[i].refract_rate.y << ' ' << obj_materials[i].refract_rate.z << std::endl;
        fout << obj_materials[i].refract_albedo.x << ' ' << obj_materials[i].refract_albedo.y << ' ' << obj_materials[i].refract_albedo.z << std::endl;
        fout << obj_materials[i].refract_index << std::endl;

        fout << (obj_normalize[i] ? 1 : 0) << std::endl;
    }

    fout.close();
}

void offline_render(int spp, GLFWwindow* window)
{
    // pipeline settings
    set_shader("./shaders/fshader_render.fsh", spp);

    // start render
    std::cout << "Start Rendering..." << std::endl << std::endl;

    glEnable(GL_DEPTH_TEST);  // 开启深度测试
    glClearColor(0.0, 0.0, 0.0, 1.0);   // 背景颜色 -- 黑
    frameCounter = 0;
    display(window, false);

    // save
    std::cout << "BEGIN SAVE" << std::endl;
    unsigned char* image = new unsigned char[WIDTH * HEIGHT * 3];
    std::cout << "ALLOCATE" << std::endl;
    glPixelStorei(GL_UNPACK_ALIGNMENT,1);
    glReadPixels(0, 0, WIDTH, HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, image);
    std::cout << "READ PIXELS" << std::endl;
    check_error(0);
    save_image(image, WIDTH, HEIGHT);
    std::cout << "STORE" << std::endl;
    delete[] image;
    std::cout << "RENDER DONE" << std::endl;

    frameCounter = 0;
    // switch to preview shader
    set_shader("./shaders/fshader_preview.fsh", -1);
    glEnable(GL_DEPTH_TEST);  // 开启深度测试
    glClearColor(0.0, 0.0, 0.0, 1.0);   // 背景颜色 -- 黑
    display(window, true);
}

int main()
{
    //初始化GLFW库
    if(!glfwInit())
        return -1;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    //创建窗口以及上下文
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Path Tracing", nullptr, nullptr);
    if(!window)
    {
        //创建失败会返回NULL
        glfwTerminate();
    }
    //建立当前窗口的上下文
    glfwMakeContextCurrent(window);

    glfwSetKeyCallback(window, key_callback); //注册回调函数
    glewInit();

    const GLubyte* version_string = glGetString(GL_VERSION);
    printf("GL Version : %s\n", version_string);

    Material m;
    m.brdf = vec3(0.02, 0.02, 0.02);
    m.reflex_mode = MIRROR;
    m.refract_mode = SUB_SURFACE;
//    m.refract_rate = vec3(0.3, 0.05, 0.3);
    m.refract_rate = vec3(0.1, 0.1, 0.1);
//    m.refract_albedo = vec3(0.0035, 0.0129, 0.0048);
    m.refract_albedo = vec3(0.3, 0.3, 0.3);
    m.refract_index = 2.66;
//    readObj("Master.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0, -0.5, 0), vec3(1, 1, 1)),true);
//    readObj("model.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0, -0.4, 0.3), vec3(0.8, 0.8, 0.8)),true);
//    readObj("loong.obj", triangles, m, getTransformMatrix(vec3(0, -0, 0), vec3(0.1, -0.5, 0.0), vec3(0.7, 0.7, 0.7)),true);
//    readObj("vase.obj", triangles, m, getTransformMatrix(vec3(0, -0, 0), vec3(0.1, -0.5, 0.0), vec3(0.3, 0.3, 0.3)),true);
//    readObj("box.obj", triangles, m, getTransformMatrix(vec3(0, 40, 0), vec3(0, -0.5, 0), vec3(1, 1, 0.21)), true);

//    m.brdf = vec3(0.7, 0.7, 0.7);
//    readObj("loong.obj", triangles, m, getTransformMatrix(vec3(0, -40, 0), vec3(-0.2, -0.5, 0.2), vec3(0.5, 0.5, 0.5)),true);
//    readObj("marble.obj", triangles, m, getTransformMatrix(vec3(-0, 0, 0), vec3(0, -0.49, 0.1), vec3(0.3, 0.3, 0.3)),true);
//    readObj("box.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0, -0.5, 0), vec3(0.2, 0.025, 0.2)), true);
//    m.refract_rate = vec3(0.05, 0.2, 0.05);
//    readObj("bunny.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(-0.5, -0.5, 0.3), vec3(0.5, 0.5, 0.5)),true);
    readObj("happyBuddha.obj", triangles, m, getTransformMatrix(vec3(-90, 0, 0), vec3(0, -0.52, 0.5), vec3(0.3, 0.3, 0.3)),true);

    m.brdf = vec3(0.3, 0.3, 0.3);
    m.emissive = vec3(1000, 1000, 1000);
    m.reflex_mode = DIFFUSE;
    m.refract_mode = NO_REFRACT;
    m.refract_index = 1.1;
//    readObj("light.obj", triangles, m, getTransformMatrix(vec3(90, 0, 90), vec3(-0.5, -0.5, -0.0), vec3(1.5, 0.5, 1.5)), true);
    readObj("light.obj", triangles, m, getTransformMatrix(vec3(0, 90, 90), vec3(-0.2, 1.2, 1.0), vec3(1.5, 0.5, 1.5)), true);

//    m.emissive = vec3(200, 200, 200);
//    readObj("light.obj", triangles, m, getTransformMatrix(vec3(90, 90, 0), vec3(-2.2, -0.2, 2.0), vec3(1.5, 0.5, 1.5)), true);
    m.emissive = vec3(100, 100, 100);
//    readObj("light.obj", triangles, m, getTransformMatrix(vec3(0, 90, 90), vec3(1.5, 0, 1.0), vec3(1.5, 0.5, 1.5)), true);
//    readObj("Master_eye.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0.05, -0.24, 0.17), vec3(0.4, 0.4, 0.4)),true);
//    readObj("Master_light.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0.05, -0.05, 0.35), vec3(0.2, 0.2, 0.2)),true);
//    readObj("box.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0, -0.09, 0), vec3(0.018, 0.002, 0.018)), true);

//    m.brdf = vec3(0.01, 0.01, 0.01);
//    m.emissive = vec3(0, 0, 0);
//    readObj("flashlightShell.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0, -0.39, -0.01), vec3(0.02, 0.20, 0.02)), true);
//    m.emissive = vec3(1000, 1000, 1000);
//    readObj("flashlightLight.obj", triangles, m, getTransformMatrix(vec3(90, 0, 0), vec3(0, -0.48, 0.0), vec3(0.019, 0.019, 0.019)), true);

    // Cornell Box
//    r = 8;
//    mat4 trans_mat = getTransformMatrix(vec3(0, 0, 0), vec3(-2.796, -2.796, 0), vec3(0.01, 0.01, 0.01));
//    m.brdf = vec3(0, 0, 0);
    m.reflex_mode = MIRROR;
    m.refract_mode = NO_REFRACT;
    m.refract_rate = vec3(0.7, 0.7, 0.7);
    m.emissive = vec3(0);
//    m.refract_index = 1.44;
    mat4 trans_mat2 = getTransformMatrix(vec3(0, 0, 0), vec3(0, -0.5625, 0), vec3(12, 0.125, 12));
////    readObj("cornell_ball.obj", triangles, m, trans_mat2, true);
    readObj("box.obj", triangles, m, trans_mat2, true);

//    m.brdf = vec3(0.72, 0.72, 0.72);
//    m.reflex_mode = DIFFUSE;
//    m.refract_mode = NO_REFRACT;
//    mat4 trans_mat3 = getTransformMatrix(vec3(0, 0, 0), vec3(0, 0, -4), vec3(0.5, 4, 0.5));
//    readObj("box.obj", triangles, m, trans_mat3, true);

//    readObj("cornell_short.obj", triangles, m, trans_mat, false);

//    m.brdf = vec3(0.72, 0.72, 0.72);
//    m.reflex_mode = DIFFUSE;
//    m.refract_mode = NO_REFRACT;
//    readObj("cornell_tall.obj", triangles, m, trans_mat, false);
//
//    readObj("cornell_white_wall.obj", triangles, m, trans_mat, false);
//
//    m.reflex_mode = DIFFUSE;
//    readObj("cornell_floor.obj", triangles, m, trans_mat, false);
//
//    m.brdf = vec3(0.72, 0, 0);
//    m.reflex_mode = DIFFUSE;
//    readObj("cornell_left.obj", triangles, m, trans_mat, false);
//
//    m.brdf = vec3(0, 0.72, 0);
//    m.reflex_mode = DIFFUSE;
//    readObj("cornell_right.obj", triangles, m, trans_mat, false);
//
//    m.brdf = vec3(0.78, 0.78, 0.78);
//    m.emissive = vec3(40, 40, 40);
//    m.reflex_mode = DIFFUSE;
//    readObj("light.obj", triangles, m, trans_mat, false);

    size_t nTriangles = triangles.size();

    std::cout << "Model load done:  " << nTriangles << " Triangles." << std::endl;
    std::cout << "First Triangle: " << triangles[0].p1.x << ' ' << triangles[0].p1.y << ' ' << triangles[0].p1.z << std::endl;
    std::cout << "First Triangle: " << triangles[0].p2.x << ' ' << triangles[0].p2.y << ' ' << triangles[0].p2.z << std::endl;
    std::cout << "First Triangle: " << triangles[0].p3.x << ' ' << triangles[0].p3.y << ' ' << triangles[0].p3.z << std::endl;

    // 建立 bvh
    BVHNode testNode;
    testNode.left = 255;
    testNode.right = 128;
    testNode.n = 30;
    testNode.AA = vec3(1, 1, 0);
    testNode.BB = vec3(0, 1, 0);
    nodes.push_back(testNode);
    //buildBVH(triangles, nodes, 0, triangles.size() - 1, 8);
    buildBVHwithSAH(triangles, nodes, 0, triangles.size() - 1, 8);
    int nNodes = nodes.size();
    std::cout << "BVH Build done: " << nNodes << " nodes." << std::endl;

    // 编码 三角形, 材质
    std::vector<Triangle_encoded> triangles_encoded(nTriangles);
    std::vector<int> emit_triangles_indices;
    for (int i = 0; i < nTriangles; i++) {
        Triangle& t = triangles[i];
        Material& m_ = t.material;
        // 顶点位置
        triangles_encoded[i].p1 = t.p1;
        triangles_encoded[i].p2 = t.p2;
        triangles_encoded[i].p3 = t.p3;
        // 顶点法线
        triangles_encoded[i].norm = t.norm;
        // 材质
        triangles_encoded[i].emissive = m_.emissive;
        triangles_encoded[i].brdf = m_.brdf;

        // 统计发光三角形
        if (m_.emissive.x > 1.5e-4 || m_.emissive.y > 1.5e-4 || m_.emissive.z > 1.5e-4) {
            emit_triangles_indices.push_back(i);
            ++nEmitTriangles;
        }
    }

    // 编码 BVHNode, aabb
    std::vector<BVHNode_encoded> nodes_encoded(nNodes);
    for (int i = 0; i < nNodes; i++) {
        nodes_encoded[i].childs = vec3(nodes[i].left, nodes[i].right, 0);
        nodes_encoded[i].leafInfo = vec3(nodes[i].n, nodes[i].index, 0);
        nodes_encoded[i].AA = nodes[i].AA;
        nodes_encoded[i].BB = nodes[i].BB;
    }
    std::cout << "Code BVH Done." << std::endl;

    // ----------------------------------------------------------------------------- //

    // 生成纹理

    // 三角形数组
    GLuint tbo0;
    glGenBuffers(1, &tbo0);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo0);
    glBufferData(GL_TEXTURE_BUFFER, nTriangles * sizeof(Triangle_encoded), &triangles_encoded[0], GL_STATIC_DRAW);
    glGenTextures(1, &trianglesTextureBuffer);

    glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);

    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo0);
    std::cout << "GL Triangle Set." << std::endl;

    // BVHNode 数组
    GLuint tbo1;
    glGenBuffers(1, &tbo1);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo1);
    glBufferData(GL_TEXTURE_BUFFER, nodes_encoded.size() * sizeof(BVHNode_encoded), &nodes_encoded[0], GL_STATIC_DRAW);
    glGenTextures(1, &nodesTextureBuffer);
    glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo1);
    std::cout << "GL BVH Set." << std::endl;

    // hdr 全景图
    HDRLoaderResult hdrRes;
    HDRLoader::load("background.hdr", hdrRes);
    hdrMap = getTextureRGB32F(hdrRes.width, hdrRes.height);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, hdrRes.cols);
    std::cout << "HDR load done." << std::endl;

    // 发光三角形索引
    GLuint tbo2;
    glGenBuffers(1, &tbo2);
    glBindBuffer(GL_TEXTURE_BUFFER, tbo2);

    glBufferData(GL_TEXTURE_BUFFER, nEmitTriangles * sizeof(int), &emit_triangles_indices[0], GL_STATIC_DRAW);
    glGenTextures(1, &emitTrianglesIndices);
    glBindTexture(GL_TEXTURE_BUFFER, emitTrianglesIndices);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_R32I, tbo2);
    std::cout << "Emit triangle indices done." << std::endl;
    // ----------------------------------------------------------------------------- //

    // 管线配置

    set_shader("./shaders/fshader_preview.fsh", -1);

    // ----------------------------------------------------------------------------- //

    std::cout << "Start..." << std::endl << std::endl;

    glEnable(GL_DEPTH_TEST);  // 开启深度测试
    glClearColor(0.0, 0.0, 0.0, 1.0);   // 背景颜色 -- 黑

    //循环，直到用户关闭窗口
    while(!glfwWindowShouldClose(window))
    {
        /*******轮询事件*******/
        glfwPollEvents();

        move_camera(window);
        display(window, true);
    }
    glfwTerminate();
    return 0;
}