//
// Project: JadeRayTraceRendering
// File Name: PathTrace.cu
// Author: Morning
// Description:
//
// Create Date: 2021/10/13
//

#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <lib/hdrloader.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

using namespace std;

#define INF 2147483647.0
#ifdef LARGE
#define RENDER_WIDTH 1024
#define RENDER_HEIGHT 1024
#else
#define RENDER_WIDTH 128
#define RENDER_HEIGHT 128
#endif

#define TILE_SIZE 16
#define STACK_CAPACITY 128
#define BVH_STACK_CAPACITY 128
#define RR_RATE 0.9
#define PI 3.1415926
#define RAND_SIZE 31

#define DIFFUSE 0
#define MIRROR 1

#define NO_REFRACT 1.4e-5
#define SUB_SURFACE 0.9999
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

__host__ void save_image(unsigned char* target_img, int width, int height)
{
    FILE* file_ptr = fopen("RenderResultCuda.bmp", "wb+");

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

// 3D resources
// use in host
struct vec3_hs {
    float3 data;

    __host__ vec3_hs(const float3& ori) {
        data = ori;
    }

    __host__ vec3_hs() {
    }

    __host__ vec3_hs(float x, float y, float z) {
        data.x = x;
        data.y = y;
        data.z = z;
    }

    __host__ inline vec3_hs operator+(const vec3_hs& opr2) const {
        return vec3_hs(make_float3(data.x + opr2.data.x, data.y + opr2.data.y, data.z + opr2.data.z));
    }

    __host__ inline vec3_hs operator-(const vec3_hs& opr2) const {
        return vec3_hs(make_float3(data.x - opr2.data.x, data.y - opr2.data.y, data.z - opr2.data.z));
    }

    __host__ inline vec3_hs operator*(const vec3_hs& opr2) const {
        return vec3_hs(make_float3(data.x * opr2.data.x, data.y * opr2.data.y, data.z * opr2.data.z));
    }

    __host__ inline vec3_hs operator*(float scalar) const {
        return vec3_hs(make_float3(data.x * scalar, data.y * scalar, data.z * scalar));
    }

    __host__ inline vec3_hs operator/(const vec3_hs& opr2) const {
        return vec3_hs(make_float3(data.x / opr2.data.x, data.y / opr2.data.y, data.z / opr2.data.z));
    }
};

__host__ inline float dot(const vec3_hs& opr1, const vec3_hs& opr2) {
    return opr1.data.x * opr2.data.x + opr1.data.y * opr2.data.y + opr1.data.z * opr2.data.z;
}

__host__ inline float mixed_product(const vec3_hs& vec_a, const vec3_hs& vec_b, const vec3_hs& vec_c)
{
    return vec_a.data.x * (vec_b.data.y * vec_c.data.z - vec_b.data.z * vec_c.data.y) + 
        vec_a.data.y * (vec_b.data.z * vec_c.data.x - vec_b.data.x * vec_c.data.z) + 
        vec_a.data.z * (vec_b.data.x * vec_c.data.y - vec_b.data.y * vec_c.data.x);
}

__host__ vec3_hs transform(const vec3_hs& vec3, float f4, float mat4[4][4])
{
    vec3_hs v3(0, 0, 0);
    v3.data.x = mat4[0][0] * vec3.data.x + mat4[1][0] * vec3.data.y + mat4[2][0] * vec3.data.z + mat4[3][0] * f4;
    v3.data.y = mat4[0][1] * vec3.data.x + mat4[1][1] * vec3.data.y + mat4[2][1] * vec3.data.z + mat4[3][1] * f4;
    v3.data.z = mat4[0][2] * vec3.data.x + mat4[1][2] * vec3.data.y + mat4[2][2] * vec3.data.z + mat4[3][2] * f4;

    return v3;
}

__host__ inline vec3_hs normalize(const vec3_hs& opr) {
    float length_rev = 1.0 / sqrt(opr.data.x * opr.data.x + opr.data.y * opr.data.y + opr.data.z * opr.data.z);
    return vec3_hs(make_float3(opr.data.x * length_rev, opr.data.y * length_rev, opr.data.z * length_rev));
}

__host__ vec3_hs cross(const vec3_hs& vec_b, const vec3_hs& vec_c) {
    vec3_hs v3(0, 0, 0);
    v3.data.x = vec_b.data.y * vec_c.data.z - vec_b.data.z * vec_c.data.y;
    v3.data.y = vec_b.data.z * vec_c.data.x - vec_b.data.x * vec_c.data.z;
    v3.data.z = vec_b.data.x * vec_c.data.y - vec_b.data.y * vec_c.data.x;
    return v3;
}

// use in device
struct vec3_dv {
    float3 data;

    __device__ vec3_dv() {

    }

    __host__ vec3_dv(const vec3_hs& ori) {
        data = ori.data;
    }

    __device__ vec3_dv(const float3& ori) {
        data = ori;
    }

    __device__ vec3_dv(float x, float y, float z) {
        data.x = x;
        data.y = y;
        data.z = z;
    }

    __device__ inline vec3_dv operator+(const vec3_dv& opr2) {
        return vec3_dv(make_float3(data.x + opr2.data.x, data.y + opr2.data.y, data.z + opr2.data.z));
    }

    __device__ inline vec3_dv& operator+=(const vec3_dv& opr2) {
        data.x += opr2.data.x;
        data.y += opr2.data.y;
        data.z += opr2.data.z;
        return (*this);
    }

    __device__ inline vec3_dv operator-(const vec3_dv& opr2) {
        return vec3_dv(make_float3(data.x - opr2.data.x, data.y - opr2.data.y, data.z - opr2.data.z));
    }

    __device__ inline vec3_dv operator*(const vec3_dv& opr2) {
        return vec3_dv(make_float3(data.x * opr2.data.x, data.y * opr2.data.y, data.z * opr2.data.z));
    }

    __device__ inline vec3_dv& operator*=(const vec3_dv& opr2) {
        data.x *= opr2.data.x;
        data.y *= opr2.data.y;
        data.z *= opr2.data.z;
        return (*this);    }

    __device__ inline vec3_dv operator*(float scalar) {
        return vec3_dv(make_float3(data.x * scalar, data.y * scalar, data.z * scalar));
    }

    __device__ inline vec3_dv& operator*=(float scalar) {
        data.x *= scalar;
        data.y *= scalar;
        data.z *= scalar;
        return (*this);
    }

    __device__ inline vec3_dv operator/(const vec3_dv& opr2) {
        return vec3_dv(make_float3(data.x / opr2.data.x, data.y / opr2.data.y, data.z / opr2.data.z));
    }

    __device__ inline vec3_dv operator/(float opr2) {
        return vec3_dv(make_float3(data.x / opr2, data.y / opr2, data.z / opr2));
    }

    __device__ inline vec3_dv normalize() {
        float length_rev = 1.0 / norm3df(data.x, data.y, data.z);
        return vec3_dv(make_float3(data.x * length_rev, data.y * length_rev, data.z * length_rev));
    }
};

__device__ inline float dot(const vec3_dv& opr1, const vec3_dv& opr2) {
    return opr1.data.x * opr2.data.x + opr1.data.y * opr2.data.y + opr1.data.z * opr2.data.z;
}

__device__ inline float mixed_product(const vec3_dv& vec_a, const vec3_dv& vec_b, const vec3_dv& vec_c)
{
    return vec_a.data.x * (vec_b.data.y * vec_c.data.z - vec_b.data.z * vec_c.data.y) + 
        vec_a.data.y * (vec_b.data.z * vec_c.data.x - vec_b.data.x * vec_c.data.z) + 
        vec_a.data.z * (vec_b.data.x * vec_c.data.y - vec_b.data.y * vec_c.data.x);
}

__device__ vec3_dv transform(const vec3_dv& vec3, float f4, float mat4[4][4])
{
    vec3_dv v3(0, 0, 0);
    v3.data.x = mat4[0][0] * vec3.data.x + mat4[1][0] * vec3.data.y + mat4[2][0] * vec3.data.z + mat4[3][0] * f4;
    v3.data.y = mat4[0][1] * vec3.data.x + mat4[1][1] * vec3.data.y + mat4[2][1] * vec3.data.z + mat4[3][1] * f4;
    v3.data.z = mat4[0][2] * vec3.data.x + mat4[1][2] * vec3.data.y + mat4[2][2] * vec3.data.z + mat4[3][2] * f4;

    return v3;
}

__device__ static inline vec3_dv normalize(const vec3_dv& opr) {
    float length_rev = 1.0 / norm3df(opr.data.x, opr.data.y, opr.data.z);
    return vec3_dv(make_float3(opr.data.x * length_rev, opr.data.y * length_rev, opr.data.z * length_rev));
}

__device__ vec3_dv cross(const vec3_dv& vec_b, const vec3_dv& vec_c) {
    vec3_dv v3(0, 0, 0);
    v3.data.x = vec_b.data.y * vec_c.data.z - vec_b.data.z * vec_c.data.y;
    v3.data.y = vec_b.data.z * vec_c.data.x - vec_b.data.x * vec_c.data.z;
    v3.data.z = vec_b.data.x * vec_c.data.y - vec_b.data.y * vec_c.data.x;
    return v3;
}

// 物体表面材质定义
// complex calculated in device, edit in host
struct Material {
    vec3_dv emissive = vec3_dv(0, 0, 0);  // 作为光源时的发光颜色
    vec3_dv brdf = vec3_dv(0.8, 0.8, 0.8); // BRDF
    int reflex_mode;           // 反射模式，漫反射0 / 镜面反射1
    float refract_mode;           // 折射模式，无透射0 / 次表面散射0.5 / 直接折射 - 折射率
    vec3_dv refract_rate = vec3_dv(0.8, 0.8, 0.8); // 折射吸光率
    float refract_dec_rate;     // 折射衰减率
};

// 三角形定义
// used in host
struct Triangle {
    int index;              // 三角形的编号
    int obj_idx;            // 所属物件id
    vec3_hs p1, p2, p3;    // 顶点坐标
    vec3_hs norm;    // 顶点法线
    Material material;  // 材质
};

// BVH 树节点
// used in host
struct BVHNode {
    int left, right;    // 左右子树索引
    int n, index;       // 叶子节点信息
    vec3_hs AA, BB;        // 碰撞盒

    BVHNode() {
        AA = vec3_hs();
        BB = vec3_hs();
    }
};

// used in device
struct Triangle_cu {
    int obj_idx;           // 所属物件id
    vec3_dv p1, p2, p3;    // 顶点坐标
    vec3_dv norm;          // 法线
    vec3_dv emissive;      // 自发光参数
    vec3_dv brdf;          // BRDF
    int reflex_mode;      // 反射模式，漫反射0 / 镜面反射1
    float refract_mode;           // 折射模式，无透射0 / 次表面散射0.5 / 直接折射 - 折射率
    vec3_dv refract_rate = vec3_dv(0.8, 0.8, 0.8); // 折射吸光率
    float refract_dec_rate;     // 折射衰减率
};

// used in device
struct BVHNode_cu {
    int left, right;    // 左右子树索引
    int n, index;       // 叶子节点信息
    vec3_dv AA, BB;        // 碰撞盒
};

// 读取 obj
struct Obj_seg {
    int begin_idx;
    int end_idx;
};

int obj_idx_cnt = 0;
Obj_seg* obj_segs;
__host__ void readObj(const string& filepath, vector<Triangle>& triangles, Material material, float trans[4][4], bool normal_transform) {

    // 顶点位置，索引
    vector<vec3_hs> vertices;
    vector<int> indices;

    // 打开文件流
    ifstream fin(filepath);
    string line;
    if (!fin.is_open()) {
        cout << "File " << filepath << " open failed." << endl;
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
    while (getline(fin, line)) {
        istringstream sin(line);   // 以一行的数据作为 string stream 解析并且读取
        string type;
        float x, y, z;
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
        vec3_hs center = vec3_hs((maxx + minx) / 2, (maxy + miny) / 2, (maxz + minz) / 2);
        for (auto& v : vertices) {
            v = v - center;
            v.data.x /= maxaxis;
            v.data.y /= maxaxis;
            v.data.z /= maxaxis;
        }
    }


    // 通过矩阵进行坐标变换
    for (auto& v : vertices) {
        v = transform(v, 1, trans);
    }

    // 构建 Triangle 对象数组
    int offset = triangles.size();  // 增量更新

    triangles.resize(offset + indices.size() / 3);
    obj_segs[obj_idx_cnt].begin_idx = offset;
    obj_segs[obj_idx_cnt].end_idx = triangles.size() - 1;

    int triangle_index = 0;
    for (int i = 0; i < indices.size(); i += 3) {
        Triangle& t = triangles[offset + i / 3];
        t.index = triangle_index;
        t.obj_idx = obj_idx_cnt;
        // 传顶点属性
        t.p1 = vertices[indices[i]];
        t.p2 = vertices[indices[i + 1]];
        t.p3 = vertices[indices[i + 2]];
        // 计算法线
        t.norm = normalize(cross(t.p2 - t.p1, t.p3 - t.p1));

        // 传材质
        t.material = material;

        ++triangle_index;
    }

    ++obj_idx_cnt;
}

__host__ float size(Triangle triangle)
{
    vec3_hs v_1 = triangle.p2 - triangle.p1;
    vec3_hs v_2 = triangle.p3 - triangle.p1;
    vec3_hs cross_product = vec3_hs(v_1.data.y * v_2.data.z - v_1.data.z * v_2.data.y, v_1.data.z * v_2.data.x - v_1.data.x * v_2.data.z, v_1.data.x * v_2.data.y - v_1.data.y * v_2.data.x);
    return 0.5 * sqrt(dot(cross_product, cross_product));
}

// 按照三角形中心排序 -- 比较函数
__host__ bool cmpx(const Triangle& t1, const Triangle& t2) {
    vec3_hs center1 = (t1.p1 + t1.p2 + t1.p3) / vec3_hs(3, 3, 3);
    vec3_hs center2 = (t2.p1 + t2.p2 + t2.p3) / vec3_hs(3, 3, 3);
    return center1.data.x < center2.data.x;
}
__host__ bool cmpy(const Triangle& t1, const Triangle& t2) {
    vec3_hs center1 = (t1.p1 + t1.p2 + t1.p3) / vec3_hs(3, 3, 3);
    vec3_hs center2 = (t2.p1 + t2.p2 + t2.p3) / vec3_hs(3, 3, 3);
    return center1.data.y < center2.data.y;
}
__host__ bool cmpz(const Triangle& t1, const Triangle& t2) {
    vec3_hs center1 = (t1.p1 + t1.p2 + t1.p3) / vec3_hs(3, 3, 3);
    vec3_hs center2 = (t2.p1 + t2.p2 + t2.p3) / vec3_hs(3, 3, 3);
    return center1.data.z < center2.data.z;
}

__device__ vec3_dv max(const vec3_dv& opr1, const vec3_dv& opr2) {
    return vec3_dv(opr1.data.x > opr2.data.x ? opr1.data.x : opr2.data.x,
        opr1.data.y > opr2.data.y ? opr1.data.y : opr2.data.y,
        opr1.data.z > opr2.data.z ? opr1.data.z : opr2.data.z);
}

__device__ vec3_dv min(const vec3_dv& opr1, const vec3_dv& opr2) {
    return vec3_dv(opr1.data.x < opr2.data.x ? opr1.data.x : opr2.data.x,
        opr1.data.y < opr2.data.y ? opr1.data.y : opr2.data.y,
        opr1.data.z < opr2.data.z ? opr1.data.z : opr2.data.z);
}

// SAH 优化构建 BVH
__host__ int buildBVHwithSAH(vector<Triangle>& triangles, vector<BVHNode>& nodes, int l, int r, int n) {
    if (l > r) return 0;

    nodes.emplace_back();
    int id = nodes.size() - 1;
    nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
    nodes[id].AA = vec3_hs(1145141919, 1145141919, 1145141919);
    nodes[id].BB = vec3_hs(-1145141919, -1145141919, -1145141919);

    // 计算 AABB
    for (int i = l; i <= r; i++) {
        // 最小点 AA
        float minx = min(triangles[i].p1.data.x, min(triangles[i].p2.data.x, triangles[i].p3.data.x));
        float miny = min(triangles[i].p1.data.y, min(triangles[i].p2.data.y, triangles[i].p3.data.y));
        float minz = min(triangles[i].p1.data.z, min(triangles[i].p2.data.z, triangles[i].p3.data.z));
        nodes[id].AA.data.x = min(nodes[id].AA.data.x, minx);
        nodes[id].AA.data.y = min(nodes[id].AA.data.y, miny);
        nodes[id].AA.data.z = min(nodes[id].AA.data.z, minz);
        // 最大点 BB
        float maxx = max(triangles[i].p1.data.x, max(triangles[i].p2.data.x, triangles[i].p3.data.x));
        float maxy = max(triangles[i].p1.data.y, max(triangles[i].p2.data.y, triangles[i].p3.data.y));
        float maxz = max(triangles[i].p1.data.z, max(triangles[i].p2.data.z, triangles[i].p3.data.z));
        nodes[id].BB.data.x = max(nodes[id].BB.data.x, maxx);
        nodes[id].BB.data.y = max(nodes[id].BB.data.y, maxy);
        nodes[id].BB.data.z = max(nodes[id].BB.data.z, maxz);
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
        if (axis == 0) sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
        if (axis == 1) sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
        if (axis == 2) sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);

        // leftMax[i]: [l, i] 中最大的 xyz 值
        // leftMin[i]: [l, i] 中最小的 xyz 值
        vector<vec3_hs> leftMax(r - l + 1, vec3_hs(-INF, -INF, -INF));
        vector<vec3_hs> leftMin(r - l + 1, vec3_hs(INF, INF, INF));
        // 计算前缀 注意 i-l 以对齐到下标 0
        for (int i = l; i <= r; i++) {
            Triangle& t = triangles[i];
            int bias = (i == l) ? 0 : 1;  // 第一个元素特殊处理

            leftMax[i - l].data.x = max(leftMax[i - l - bias].data.x, max(t.p1.data.x, max(t.p2.data.x, t.p3.data.x)));
            leftMax[i - l].data.y = max(leftMax[i - l - bias].data.y, max(t.p1.data.y, max(t.p2.data.y, t.p3.data.y)));
            leftMax[i - l].data.z = max(leftMax[i - l - bias].data.z, max(t.p1.data.z, max(t.p2.data.z, t.p3.data.z)));

            leftMin[i - l].data.x = min(leftMin[i - l - bias].data.x, min(t.p1.data.x, min(t.p2.data.x, t.p3.data.x)));
            leftMin[i - l].data.y = min(leftMin[i - l - bias].data.y, min(t.p1.data.y, min(t.p2.data.y, t.p3.data.y)));
            leftMin[i - l].data.z = min(leftMin[i - l - bias].data.z, min(t.p1.data.z, min(t.p2.data.z, t.p3.data.z)));
        }

        // rightMax[i]: [i, r] 中最大的 xyz 值
        // rightMin[i]: [i, r] 中最小的 xyz 值
        vector<vec3_hs> rightMax(r - l + 1, vec3_hs(-INF, -INF, -INF));
        vector<vec3_hs> rightMin(r - l + 1, vec3_hs(INF, INF, INF));
        // 计算后缀 注意 i-l 以对齐到下标 0
        for (int i = r; i >= l; i--) {
            Triangle& t = triangles[i];
            int bias = (i == r) ? 0 : 1;  // 第一个元素特殊处理

            rightMax[i - l].data.x = max(rightMax[i - l + bias].data.x, max(t.p1.data.x, max(t.p2.data.x, t.p3.data.x)));
            rightMax[i - l].data.y = max(rightMax[i - l + bias].data.y, max(t.p1.data.y, max(t.p2.data.y, t.p3.data.y)));
            rightMax[i - l].data.z = max(rightMax[i - l + bias].data.z, max(t.p1.data.z, max(t.p2.data.z, t.p3.data.z)));

            rightMin[i - l].data.x = min(rightMin[i - l + bias].data.x, min(t.p1.data.x, min(t.p2.data.x, t.p3.data.x)));
            rightMin[i - l].data.y = min(rightMin[i - l + bias].data.y, min(t.p1.data.y, min(t.p2.data.y, t.p3.data.y)));
            rightMin[i - l].data.z = min(rightMin[i - l + bias].data.z, min(t.p1.data.z, min(t.p2.data.z, t.p3.data.z)));
        }

        // 遍历寻找分割
        float cost = INF;
        int split = l;
        for (int i = l; i <= r - 1; i++) {
            float lenx, leny, lenz;
            // 左侧 [l, i]
            vec3_hs leftAA = leftMin[i - l];
            vec3_hs leftBB = leftMax[i - l];
            lenx = leftBB.data.x - leftAA.data.x;
            leny = leftBB.data.y - leftAA.data.y;
            lenz = leftBB.data.z - leftAA.data.z;
            float leftS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
            float leftCost = leftS * (i - l + 1);

            // 右侧 [i+1, r]
            vec3_hs rightAA = rightMin[i + 1 - l];
            vec3_hs rightBB = rightMax[i + 1 - l];
            lenx = rightBB.data.x - rightAA.data.x;
            leny = rightBB.data.y - rightAA.data.y;
            lenz = rightBB.data.z - rightAA.data.z;
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
    if (Axis == 0) sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
    if (Axis == 1) sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
    if (Axis == 2) sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);

    // 递归
    int left  = buildBVHwithSAH(triangles, nodes, l, Split, n);
    int right = buildBVHwithSAH(triangles, nodes, Split + 1, r, n);

    nodes[id].left = left;
    nodes[id].right = right;

    return id;
}

// ----------------------------------------------------------------------------- //
vec3_dv eye_center = vec3_dv(0, 0, 0);
float camera_transform[4][4];

vector<BVHNode> nodes;
int nEmitTriangles = 0;


// HDR贴图
texture<float, 2, cudaReadModeElementType> text_ref_r;
texture<float, 2, cudaReadModeElementType> text_ref_g;
texture<float, 2, cudaReadModeElementType> text_ref_b;

__constant__ int nTriangles_dv;
__constant__ int nEmitTriangles_dv;
__constant__ int nNodes_dv;
__constant__ int spp;
__constant__ float3 eye_dv;
__constant__ float camera_transform_dv[4][4];

// 光线
struct Ray {
    vec3_dv startPoint;
    vec3_dv direction;
};

// 光线求交结果
struct HitResult {
    bool isHit;             // 是否命中
    int index;              // 命中三角形坐标
    float distance;         // 与交点的距离
    vec3_dv hitPoint;       // 光线命中点
};

__global__ void init_curand(curandState* curand_states, int seed)
{
    curand_init(seed, threadIdx.x, 0, &(curand_states[threadIdx.x]));
}

__device__ vec3_dv toneMapping(vec3_dv c, float limit) {
    float luminance = 0.3 * c.data.x + 0.6 * c.data.y + 0.1 * c.data.z;
    return c * float(1.0 / (1.0 + luminance / limit));
}

// ----------------------------------------------------------------------------- //
// 将三维向量 v 转为 HDR map 的纹理坐标 uv
__device__ float2 SampleSphericalMap(float3 v) {
    float2 uv = make_float2(atan2f(v.z, v.x), asinf(v.y));
    uv.x /= 2.0 * PI;
    uv.y /= PI;
    uv.x += 0.5;
    uv.y += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

// 获取 HDR 环境颜色
__device__ vec3_dv sampleHdr(vec3_dv v) {
    float2 uv = SampleSphericalMap(normalize(v).data);
    vec3_dv color = vec3_dv(tex2D(text_ref_r, uv.x, uv.y), tex2D(text_ref_g, uv.x, uv.y), tex2D(text_ref_b, uv.x, uv.y));
    color = min(color, vec3_dv(10, 10, 10));
    return color;
}

// 光线和三角形求交
__device__ HitResult hitTriangle(Triangle_cu triangle, Ray ray, int index) {
    HitResult res{false, 0, INF, vec3_dv(0, 0, 0)};
    res.distance = INF;
    res.isHit = false;

    vec3_dv normal_direction = normalize(ray.direction);
    vec3_dv src_point = ray.startPoint;
    // make shadow
    vec3_dv shadow_tri_a = triangle.p1 - normal_direction * dot(normal_direction, triangle.p1 - src_point);
    vec3_dv shadow_tri_b = triangle.p2 - normal_direction * dot(normal_direction, triangle.p2 - src_point);
    vec3_dv shadow_tri_c = triangle.p3 - normal_direction * dot(normal_direction, triangle.p3 - src_point);

    // check in center
    vec3_dv vec_pa = shadow_tri_a - src_point;
    vec3_dv vec_pb = shadow_tri_b - src_point;
    vec3_dv vec_pc = shadow_tri_c - src_point;

    float papb = mixed_product(normal_direction, vec_pa, vec_pb);
    float pbpc = mixed_product(normal_direction, vec_pb, vec_pc);
    float pcpa = mixed_product(normal_direction, vec_pc, vec_pa);
    if ((papb > 0 && pbpc > 0 && pcpa > 0) || (papb < 0 && pbpc < 0 && pcpa < 0)) {
        // in center
        // get hit point
        // get coordinary, reuse vec_pb ,vec_pc
        vec_pb = shadow_tri_b - shadow_tri_a;
        vec_pc = shadow_tri_c - shadow_tri_a;
        vec_pa = src_point - shadow_tri_a;
        float divider = vec_pb.data.x * vec_pc.data.y - vec_pb.data.y * vec_pc.data.x;
        float rate_a = (vec_pc.data.y * vec_pa.data.x - vec_pc.data.x * vec_pa.data.y) / divider;
        float rate_b = (-vec_pb.data.y * vec_pa.data.x + vec_pb.data.x * vec_pa.data.y) / divider;

        vec_pb = triangle.p2 - triangle.p1;
        vec_pc = triangle.p3 - triangle.p1;
        vec_pa = triangle.p1 + vec_pb * rate_a + vec_pc * rate_b;

        float distance = dot(vec_pa - src_point, normal_direction);
        if (distance > 0) {
            // ray will hit object
            // package result
            res.isHit = true;
            res.hitPoint = vec_pa;
            res.distance = distance;
            // res.normal = normalize(cross(triangle.p2 - triangle.p1, triangle.p3 - triangle.p1));
            // res.viewDir = ray.direction;
            res.index = index;
        }
    }

    return res;
}

// 和 aabb 盒子求交，没有交点则返回 -1

__device__ float hitAABB(Ray r, vec3_dv AA, vec3_dv BB) {
    vec3_dv invdir = vec3_dv(1.0 / r.direction.data.x, 1.0 / r.direction.data.y, 1.0 / r.direction.data.z);

    vec3_dv f = (BB - r.startPoint) * invdir;
    vec3_dv n = (AA - r.startPoint) * invdir;

    vec3_dv tmax = max(f, n);
    vec3_dv tmin = min(f, n);

    float t1 = min(tmax.data.x, min(tmax.data.y, tmax.data.z));
    float t0 = max(tmin.data.x, max(tmin.data.y, tmin.data.z));

    return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1);
}

// ----------------------------------------------------------------------------- //

// 暴力遍历数组下标范围 [l, r] 求最近交点
__device__ HitResult hitArray(Ray ray, int l, int r, int src_object_idx, Triangle_cu* triangles_cu) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;

    for(int i = l; i <= r; i++) {
        if (i == src_object_idx) {
            continue;
        }
        Triangle_cu triangle = triangles_cu[i];
        HitResult new_hit = hitTriangle(triangle, ray, i);
        if(new_hit.isHit && new_hit.distance < res.distance) {
            res = new_hit;
        }
    }
    return res;
}

// 遍历 BVH 求交
__device__ HitResult hitBVH(Ray ray, int src_object_idx, Triangle_cu* triangles_cu, BVHNode_cu* node_cu) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;

    // 栈
    int stack[BVH_STACK_CAPACITY];
    int sp = 0;

    stack[sp] = 1;
    sp++;
    while(sp > 0) {
        --sp;
        int top = stack[sp];
        BVHNode_cu node = node_cu[top];

        // 是叶子节点，遍历三角形，求最近交点
        if(node.n > 0) {
            int L = node.index;
            int R = node.index + node.n - 1;
            HitResult r = hitArray(ray, L, R, src_object_idx, triangles_cu);
            if(r.isHit && r.distance < res.distance) {
                res = r;
            }
            continue;
        }

        // 和左右盒子 AABB 求交
        float d1 = -1; // 左盒子距离
        float d2 = -1; // 右盒子距离
        if(node.left > 0) {
            BVHNode_cu leftNode = node_cu[node.left];
            d1 = hitAABB(ray, leftNode.AA, leftNode.BB);
        }
        if(node.right > 0) {
            BVHNode_cu rightNode = node_cu[node.right];
            d2 = hitAABB(ray, rightNode.AA, rightNode.BB);
        }

        // 在最近的盒子中搜索
        if(d1 > 0 && d2 > 0) {
            if(d1<d2) { // d1 < d2, 左边先
                stack[sp] = node.right;
                sp++;

                stack[sp] = node.left;
                sp++;
            } else {    // d2 < d1, 右边先
                stack[sp] = node.left;
                sp++;

                stack[sp] = node.right;
                sp++;
            }
        } else if(d1 > 0) {   // 仅命中左边
            stack[sp] = node.left;
            sp++;
        } else if(d2 > 0) {   // 仅命中右边
            stack[sp] = node.right;
            sp++;
        }
    }

    return res;
}

// ----------------------------------------------------------------------------- //

// 路径追踪
__device__ float size(Triangle_cu triangle)
{
    vec3_dv v_1 = triangle.p2 - triangle.p1;
    vec3_dv v_2 = triangle.p3 - triangle.p1;
    vec3_dv cross_product = vec3_dv(v_1.data.y * v_2.data.z - v_1.data.z * v_2.data.y, v_1.data.z * v_2.data.x - v_1.data.x * v_2.data.z, v_1.data.x * v_2.data.y - v_1.data.y * v_2.data.x);
    return 0.5 * sqrt(dot(cross_product, cross_product));
}

__device__ vec3_dv pathTracing(HitResult hit, vec3_dv direction, curandState* curand_state, Triangle_cu* triangles_cu, BVHNode_cu* node_cu, int* emitTrianglesIndices_cu, int* triangle_index_mapping_cu, float* prefix_size_sum_cu, Obj_seg* obj_segs_cu) {
    vec3_dv l_dir = vec3_dv(0, 0, 0);
    int stack_offset = 0;
    vec3_dv stack_dir[STACK_CAPACITY];
    vec3_dv stack_indir_rate[STACK_CAPACITY];
    // vec3_dv out_direction = -hit.viewDir;
    vec3_dv out_direction = direction;
    vec3_dv ray_src = hit.hitPoint;
    HitResult obj_hit = hit;
    vec3_dv obj_hit_normal = triangles_cu[obj_hit.index].norm;
    while (stack_offset < STACK_CAPACITY) {
        l_dir = vec3_dv(0, 0, 0);
        vec3_dv obj_hit_fr = triangles_cu[obj_hit.index].brdf * (1.0 / PI);
        int reflex_refract_select_rate = triangles_cu[obj_hit.index].refract_mode > NO_REFRACT ? 2 : 1;
        float select_reflex_refract = curand_uniform(curand_state);
        if (select_reflex_refract < 0.5 && triangles_cu[obj_hit.index].refract_mode > NO_REFRACT) {
            // process refract
            if (triangles_cu[obj_hit.index].refract_mode < SUB_SURFACE) {
                // process sub surface
                // indir light
                // test RR
                float rr_result = curand_uniform(curand_state);
                if (rr_result < RR_RATE) {
                    // sample a random point on current object
                    float random_idx = curand_uniform(curand_state) * prefix_size_sum_cu[obj_segs_cu[triangles_cu[obj_hit.index].obj_idx].end_idx];
                    // find target triangle index
                    int left = obj_segs_cu[triangles_cu[obj_hit.index].obj_idx].begin_idx;
                    int right = obj_segs_cu[triangles_cu[obj_hit.index].obj_idx].end_idx;
                    int middle = 0;
                    while (left < right - 1) {
                        middle = (left + right) / 2;
                        if (random_idx < prefix_size_sum_cu[triangle_index_mapping_cu[middle]]) {
                            right = middle;
                        }
                        else if (random_idx > prefix_size_sum_cu[triangle_index_mapping_cu[middle]]) {
                            left = middle;
                        }
                        // middle = (left + right) / 2;
                    }
                    // printf("Hit\n");
                    // middle is the target triangle
                    middle = triangle_index_mapping_cu[middle];

                    // sample a random point on selected triangle
                    float rand_x = curand_uniform(curand_state);
                    float rand_y = curand_uniform(curand_state);
                    if (rand_x + rand_y > 1) {
                        rand_x = 1 - rand_x;
                        rand_y = 1 - rand_y;
                    }

                    Triangle_cu& t_i = triangles_cu[middle];
                    vec3_dv random_point = t_i.p1 + (t_i.p2 - t_i.p1) * rand_x + (t_i.p3 - t_i.p1) * rand_y;

                    // random select a ray from random_point
                    float cosine_theta = 2 * (curand_uniform(curand_state) - 0.5);
                    float sine_theta = sqrt(1 - cosine_theta * cosine_theta);
                    float fai_value = 2 * PI * curand_uniform(curand_state);
                    vec3_dv ray_direction = vec3_dv(sine_theta * cos(fai_value), sine_theta * sin(fai_value), cosine_theta);
                    vec3_dv inner_direction = random_point - ray_src;
                    if (dot(ray_direction, t_i.norm) * dot(inner_direction, t_i.norm) > 0) {
                        ray_direction *= -1;
                    }

                    Ray new_ray;
                    new_ray.startPoint = random_point;
                    new_ray.direction = ray_direction;
                    HitResult new_hit = hitBVH(new_ray, middle, triangles_cu, node_cu);
                    float3 new_hit_emissive = triangles_cu[new_hit.index].emissive.data;
                    if (new_hit.isHit && (new_hit_emissive.x < 1.5e-4 && new_hit_emissive.y < 1.5e-4 && new_hit_emissive.z < 1.5e-4)) {
                        // Hit something
                        ray_direction *= -1;
                        // vec3_dv distance = random_point - obj_hit.hitPoint;
                        vec3_dv indir_rate = triangles_cu[obj_hit.index].refract_rate * triangles_cu[obj_hit.index].refract_dec_rate / RR_RATE; // todo bssrdf
                        ray_src = new_hit.hitPoint;
                        out_direction = ray_direction;


                        stack_dir[stack_offset] = l_dir; // here l_dir should be 0, 0, 0
                        stack_indir_rate[stack_offset] = indir_rate * reflex_refract_select_rate;
                        ++stack_offset;
                        obj_hit = new_hit;
                        obj_hit_normal = triangles_cu[obj_hit.index].norm;
                    }
                    else {
                        // sample from HDR
                        l_dir = sampleHdr(ray_direction) / RR_RATE * reflex_refract_select_rate;
                        break;
                    }
                }
                else {
                    break;
                }
            }
            else {
                // process direct refract
                float triangle_miu = triangles_cu[obj_hit.index].refract_mode;
                float R0 = (1 - triangle_miu) / (1 + triangle_miu) * (1 - triangle_miu) / (1 + triangle_miu);

                // Schlick approximation
                float cosine_i = abs(dot(obj_hit_normal, out_direction));
                // todo implement
                break;
            }
        }
        else {
            // process reflex
            if (triangles_cu[obj_hit.index].reflex_mode == DIFFUSE) {
                // Diffuse process
                // direct light
                // sample from emit triangles
                for (int i = 0; i < nEmitTriangles_dv; ++i) {
                    // random select a point on light triangle
                    float rand_x = curand_uniform(curand_state);
                    float rand_y = curand_uniform(curand_state);
                    if (rand_x + rand_y > 1) {
                        rand_x = 1 - rand_x;
                        rand_y = 1 - rand_y;
                    }
    
                    int emit_tri_idx = emitTrianglesIndices_cu[i];
                    Triangle_cu& t_i = triangles_cu[emit_tri_idx];
                    vec3_dv random_point = t_i.p1 + (t_i.p2 - t_i.p1) * rand_x + (t_i.p3 - t_i.p1) * rand_y;
    
                    // test block
                    vec3_dv obj_light_direction = random_point - ray_src;
                    // check obj_light_direction and out_direction are in the same semi-sphere
                    if (dot(obj_light_direction, obj_hit_normal) * dot(out_direction, obj_hit_normal) < 0) {
                        continue;
                    }
                    Ray new_ray;
                    new_ray.startPoint = ray_src;
                    new_ray.direction = obj_light_direction;
                    HitResult hit_result = hitBVH(new_ray, obj_hit.index, triangles_cu, node_cu);
    
                    if (hit_result.isHit && hit_result.index == emit_tri_idx) {
                        float direction_length_square = obj_light_direction.data.x * obj_light_direction.data.x + obj_light_direction.data.y * obj_light_direction.data.y + obj_light_direction.data.z * obj_light_direction.data.z;
                        l_dir += triangles_cu[hit_result.index].emissive * obj_hit_fr * abs(dot(obj_hit_normal, obj_light_direction) * dot(triangles_cu[hit_result.index].norm, obj_light_direction)) 
                                    / direction_length_square / direction_length_square * size(t_i);
                    }
                }
    
    
                // sample from HDR
                // random select a point on HDR Texture
                float cosine_theta = 2 * (curand_uniform(curand_state) - 0.5);
                float sine_theta = sqrt(1 - cosine_theta * cosine_theta);
                float fai_value = 2 * PI * curand_uniform(curand_state);
                vec3_dv ray_direction = vec3_dv(sine_theta * cos(fai_value), sine_theta * sin(fai_value), cosine_theta);
                if (dot(ray_direction, obj_hit_normal) * dot(out_direction, obj_hit_normal) < 0) {
                    ray_direction *= -1;
                }
    
                // test block
                Ray new_ray;
                new_ray.startPoint = ray_src;
                new_ray.direction = ray_direction;
                HitResult hit_result = hitBVH(new_ray, obj_hit.index, triangles_cu, node_cu);
                if (!hit_result.isHit) {
                    vec3_dv skyColor = sampleHdr(ray_direction);
                    l_dir += skyColor * obj_hit_fr * abs(dot(obj_hit_normal, ray_direction)) * 2 * PI;
                }

                l_dir *= reflex_refract_select_rate;
    
                float rr_result = curand_uniform(curand_state);
                if (rr_result < RR_RATE) {
                    vec3_dv indir_rate = vec3_dv(0, 0, 0);
                    // random select a ray from src_point
                    float cosine_theta = 2 * (curand_uniform(curand_state) - 0.5);
                    float sine_theta = sqrt(1 - cosine_theta * cosine_theta);
                    float fai_value = 2 * PI * curand_uniform(curand_state);
                    vec3_dv ray_direction = vec3_dv(sine_theta * cos(fai_value), sine_theta * sin(fai_value), cosine_theta);
                    if (dot(ray_direction, obj_hit_normal) * dot(out_direction, obj_hit_normal) < 0) {
                        ray_direction *= -1;
                    }
    
                    Ray new_ray;
                    new_ray.startPoint = ray_src;
                    new_ray.direction = ray_direction;
                    HitResult new_hit = hitBVH(new_ray, obj_hit.index, triangles_cu, node_cu);
                    float3 new_hit_emissive = triangles_cu[new_hit.index].emissive.data;
                    if (new_hit.isHit && (new_hit_emissive.x < 1.5e-4 && new_hit_emissive.y < 1.5e-4 && new_hit_emissive.z < 1.5e-4)) {
                        // Hit something
                        ray_direction *= -1;
                        indir_rate = obj_hit_fr * abs(dot(ray_direction, obj_hit_normal)) / RR_RATE;
                        ray_src = new_hit.hitPoint;
                        out_direction = ray_direction;
    
                        // if (stack_offset >= STACK_CAPACITY) {
                        //     return vec3_dv(1);
                        // }
                        stack_dir[stack_offset] = l_dir;
                        stack_indir_rate[stack_offset] = indir_rate * reflex_refract_select_rate;
                        ++stack_offset;
                        obj_hit = new_hit;
                        obj_hit_normal = triangles_cu[obj_hit.index].norm;
                    }
                    else {
                        break;
                    }
                }
                else {
                    break;
                }
            }
            else {
                // Mirror process
                vec3_dv obj_emissive = triangles_cu[obj_hit.index].emissive;
                if (obj_emissive.data.x > 1.5e-4 || obj_emissive.data.y > 1.5e-4 || obj_emissive.data.x > 1.5e-4) {
                    // hit a emit triangle
                    l_dir = obj_emissive * obj_hit_fr * reflex_refract_select_rate;
                    break;
                }
                else {
                    // hit a normal triangle
                    // RR to decide issue a light
                    float rr_result = curand_uniform(curand_state);
                    if (rr_result < RR_RATE) {
                        out_direction = obj_hit_normal * (2 * dot(out_direction, triangles_cu[obj_hit.index].norm)) - out_direction;
                        // check hit
                        Ray new_ray;
                        new_ray.startPoint = ray_src;
                        new_ray.direction = out_direction;
                        HitResult new_hit = hitBVH(new_ray, obj_hit.index, triangles_cu, node_cu);
                        if (new_hit.isHit) {
                            // hit some triangle (include emit one)
                            out_direction *= -1;
                            ray_src = new_hit.hitPoint;
                            obj_hit = new_hit;
                            obj_hit_normal = triangles_cu[obj_hit.index].norm;
                            stack_dir[stack_offset] = l_dir; // here l_dir should be 0, 0, 0
                            stack_indir_rate[stack_offset] = obj_hit_fr / RR_RATE * reflex_refract_select_rate;
                            ++stack_offset;
                        }
                        else {
                            // sample from HDR
                            l_dir = sampleHdr(out_direction) * obj_hit_fr * reflex_refract_select_rate;
                            break;
                        }                   
                    }
                    else {
                        // RR failed
                        break;
                    }
                }
            }
        }
    }

    // calc final irradiance
    for (int i = stack_offset - 1; i >= 0; --i) {
        l_dir *= stack_indir_rate[i];
        l_dir += stack_dir[i];
    }
    
    return l_dir;
}

__global__ void render_pixel(unsigned char* target_img, curandState* curand_states, Triangle_cu* triangles_cu, BVHNode_cu* node_cu, int* emitTrianglesIndices_cu, int* triangle_index_mapping_cu, float* prefix_size_sum_cu, Obj_seg* obj_segs_cu)
{
    int target_pixel_width = blockIdx.x * TILE_SIZE + threadIdx.x;
    int target_pixel_height = blockIdx.y * TILE_SIZE + threadIdx.y;

    vec3_dv final_result(0, 0, 0);

    // 投射光线
    Ray ray;

    ray.startPoint = eye_dv;
    for (int i = 0; i < spp; ++i) {
        float left_offset = -1.0 + 2.0 / RENDER_WIDTH * (target_pixel_width + curand_uniform(&curand_states[(threadIdx.x + 7 * threadIdx.y) % RAND_SIZE]) - 0.5);
        float up_offset = -1.0 + 2.0 / RENDER_HEIGHT * (target_pixel_height + curand_uniform(&curand_states[(threadIdx.x + 7 * threadIdx.y) % RAND_SIZE]) - 0.5);
        
        // translate
        vec3_dv dir = vec3_dv(left_offset, up_offset, -1.5);
        dir = transform(dir, 0, camera_transform_dv);

        ray.direction = normalize(dir);

        // primary hit
        HitResult firstHit = hitBVH(ray, -1, triangles_cu, node_cu);
        vec3_dv color;

        if(!firstHit.isHit) {
            color = vec3_dv(0, 0, 0);
            color = sampleHdr(ray.direction);
        } else {
            // printf("Hit!\n");
            vec3_dv Le = triangles_cu[firstHit.index].emissive;
            vec3_dv Li = pathTracing(firstHit, ray.direction * -1, &curand_states[(threadIdx.x + 7 * threadIdx.y) % RAND_SIZE], triangles_cu, node_cu, emitTrianglesIndices_cu, triangle_index_mapping_cu, prefix_size_sum_cu, obj_segs_cu);
            // vec3_dv Li = vec3_dv(1, 1, 1);
            color = Le + Li;
        }

        final_result = final_result + color;
    }

    final_result = final_result * vec3_dv(1.0 / spp, 1.0 / spp, 1.0 / spp);
    
    // tone mapping
    final_result = toneMapping(final_result, 1.5);

    // Gamma correction
    final_result.data.x = powf(final_result.data.x, 1.0 / 2.2);
    final_result.data.y = powf(final_result.data.y, 1.0 / 2.2);
    final_result.data.z = powf(final_result.data.z, 1.0 / 2.2);

    final_result *= 255.0;
    
    int base_idx = 3 * (target_pixel_height * RENDER_WIDTH + target_pixel_width);
    target_img[base_idx] = (unsigned char)(final_result.data.z > 255 ? 255 : final_result.data.z);
    target_img[base_idx + 1] = (unsigned char)(final_result.data.y > 255 ? 255 : final_result.data.y);
    target_img[base_idx + 2] = (unsigned char)(final_result.data.x > 255 ? 255 : final_result.data.x);
}

void check_error(string pass_name) {
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, (pass_name + " failed: %s\n").c_str(), cudaGetErrorString(cudaStatus));
    }
}

int main()
{
    // 读取render_args.txt
    ifstream fin("render_args.txt");
    fin >> eye_center.data.x >> eye_center.data.y >> eye_center.data.z;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            fin >> camera_transform[row][col];
        }
    }

    int obj_cnt;
    
    fin >> obj_cnt;

    vector<string> obj_file_name(obj_cnt);
    float* obj_trans_mats = new float[obj_cnt * 4 * 4];
    vector<Material> obj_materials(obj_cnt);
    vector<bool> obj_normalize(obj_cnt);
    for (int i = 0; i < obj_cnt; ++i) {
        fin >> obj_file_name[i];

        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                fin >> obj_trans_mats[i * 16 + row * 4 + col];
            }
        }

        fin >> obj_materials[i].emissive.data.x >> obj_materials[i].emissive.data.y >> obj_materials[i].emissive.data.z;
        fin >> obj_materials[i].brdf.data.x >> obj_materials[i].brdf.data.y >> obj_materials[i].brdf.data.z;
        fin >> obj_materials[i].reflex_mode;
        fin >> obj_materials[i].refract_mode;
        fin >> obj_materials[i].refract_rate.data.x >> obj_materials[i].refract_rate.data.y >> obj_materials[i].refract_rate.data.z;
        fin >> obj_materials[i].refract_dec_rate;

        int is_normalize;
        fin >> is_normalize;
        obj_normalize[i] = (is_normalize != 0);
    }

    fin.close();

    vector<Triangle> triangles;
    obj_segs = new Obj_seg[obj_cnt];

    for (int i = 0; i < obj_cnt; ++i) {
        readObj(obj_file_name[i], triangles, obj_materials[i], (float (*)[4])&obj_trans_mats[i * 16], obj_normalize[i]);
    }

    size_t nTriangles = triangles.size();

    cout << "Model load done:  " << nTriangles << " Triangles." << endl;

    // 计算前缀和
    float* prefix_size_sum = new float[nTriangles];
    for (int i = 0; i < obj_cnt; ++i) {
        float size_sum = 0;
        for (int idx = obj_segs[i].begin_idx; idx <= obj_segs[i].end_idx; ++idx) {
            size_sum += size(triangles[idx]);
            prefix_size_sum[idx] = size_sum;
        }
    }

    // 建立 bvh
    BVHNode testNode;
    testNode.left = 255;
    testNode.right = 128;
    testNode.n = 30;
    testNode.AA = vec3_hs(1, 1, 0);
    testNode.BB = vec3_hs(0, 1, 0);
    nodes.push_back(testNode);

    buildBVHwithSAH(triangles, nodes, 0, triangles.size() - 1, 8);
    int nNodes = nodes.size();
    cout << "BVH Build done: " << nNodes << " nodes." << endl;

    // 编码 三角形, 材质
    vector<Triangle_cu> triangles_encoded(nTriangles);
    vector<int> emit_triangles_indices;

    vector<int> triangle_index_mapping(nTriangles);
    for (int i = 0; i < nTriangles; i++) {
        Triangle& t = triangles[i];
        Material& m_ = t.material;

        triangles_encoded[i].obj_idx = t.obj_idx;
        triangle_index_mapping[t.obj_idx] = i;

        // 顶点位置
        triangles_encoded[i].p1 = vec3_dv(t.p1);
        triangles_encoded[i].p2 = vec3_dv(t.p2);
        triangles_encoded[i].p3 = vec3_dv(t.p3);
        // 顶点法线
        triangles_encoded[i].norm = vec3_dv(t.norm);
        // 材质
        triangles_encoded[i].emissive = m_.emissive;
        triangles_encoded[i].brdf = m_.brdf;
        triangles_encoded[i].reflex_mode = m_.reflex_mode;
        triangles_encoded[i].refract_mode = m_.refract_mode;
        triangles_encoded[i].refract_rate = m_.refract_rate;
        triangles_encoded[i].refract_dec_rate = m_.refract_dec_rate;

        // 统计发光三角形
        if (m_.emissive.data.x > 1.5e-4 || m_.emissive.data.y > 1.5e-4 || m_.emissive.data.z > 1.5e-4) {
            emit_triangles_indices.push_back(i);
            ++nEmitTriangles;
        }
    }

    // 编码 BVHNode, aabb
    vector<BVHNode_cu> nodes_encoded(nNodes);
    for (int i = 0; i < nNodes; i++) {
        nodes_encoded[i].left = nodes[i].left;
        nodes_encoded[i].right = nodes[i].right;
        nodes_encoded[i].n = nodes[i].n;
        nodes_encoded[i].index = nodes[i].index;
        nodes_encoded[i].AA = vec3_dv(nodes[i].AA);
        nodes_encoded[i].BB = vec3_dv(nodes[i].BB);
    }
    cout << "Code BVH Done." << endl;

    // ----------------------------------------------------------------------------- //
    // 传入显存
    // Graphic Memories pointer
    Triangle_cu* triangles_cu;
    BVHNode_cu* node_cu;
    int* emitTrianglesIndices_cu;
    int* triangle_index_mapping_cu;
    float* prefix_size_sum_cu;
    Obj_seg* obj_segs_cu;

    // 三角形数组
    cudaMalloc(&triangles_cu, nTriangles * sizeof(Triangle_cu));
    cudaMemcpy(triangles_cu, &triangles_encoded[0], nTriangles * sizeof(Triangle_cu), cudaMemcpyHostToDevice);
    cout << "Triangle Set." << endl;
    check_error("set triangles");

    // 折射相关数据
    cudaMalloc(&triangle_index_mapping_cu, nTriangles * sizeof(int));
    cudaMemcpy(triangle_index_mapping_cu, &triangle_index_mapping[0], nTriangles * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&prefix_size_sum_cu, nTriangles * sizeof(float));
    cudaMemcpy(prefix_size_sum_cu, prefix_size_sum, nTriangles * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&obj_segs_cu, obj_cnt * sizeof(Obj_seg));
    cudaMemcpy(obj_segs_cu, obj_segs, obj_cnt * sizeof(Obj_seg), cudaMemcpyHostToDevice);

    // BVHNode 数组
    cudaMalloc(&node_cu, nodes_encoded.size() * sizeof(BVHNode_cu));
    cudaMemcpy(node_cu, &nodes_encoded[0], nodes_encoded.size() * sizeof(BVHNode_cu), cudaMemcpyHostToDevice);
    cout << "BVH Set." << endl;
    check_error("set bvh");

    // hdr 全景图
    HDRLoaderResult hdrRes;
    HDRLoader::load("background.hdr", hdrRes);

    cudaChannelFormatDesc h_channel_desc = cudaCreateChannelDesc<float>();
    text_ref_r.addressMode[0] = cudaAddressModeMirror;
    text_ref_r.addressMode[1] = cudaAddressModeMirror;
    text_ref_r.normalized = true;
    text_ref_r.filterMode = cudaFilterModeLinear;

    text_ref_g.addressMode[0] = cudaAddressModeMirror;
    text_ref_g.addressMode[1] = cudaAddressModeMirror;
    text_ref_g.normalized = true;
    text_ref_g.filterMode = cudaFilterModeLinear;

    text_ref_b.addressMode[0] = cudaAddressModeMirror;
    text_ref_b.addressMode[1] = cudaAddressModeMirror;
    text_ref_b.normalized = true;
    text_ref_b.filterMode = cudaFilterModeLinear;

    float* h_hdr_img_r = new float[hdrRes.width * hdrRes.height];
    float* h_hdr_img_g = new float[hdrRes.width * hdrRes.height];
    float* h_hdr_img_b = new float[hdrRes.width * hdrRes.height];
    for (int i = 0; i < hdrRes.width * hdrRes.height; ++i) {
        h_hdr_img_r[i] = hdrRes.cols[i * 3];
        h_hdr_img_g[i] = hdrRes.cols[i * 3 + 1];
        h_hdr_img_b[i] = hdrRes.cols[i * 3 + 2];
    }

    float* d_hdr_img_r;
    float* d_hdr_img_g;
    float* d_hdr_img_b;
    cudaMalloc(&d_hdr_img_r, hdrRes.width * hdrRes.height * sizeof(float));
    cudaMemcpy(d_hdr_img_r, h_hdr_img_r, hdrRes.width * hdrRes.height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_hdr_img_g, hdrRes.width * hdrRes.height * sizeof(float));
    cudaMemcpy(d_hdr_img_g, h_hdr_img_g, hdrRes.width * hdrRes.height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_hdr_img_b, hdrRes.width * hdrRes.height * sizeof(float));
    cudaMemcpy(d_hdr_img_b, h_hdr_img_b, hdrRes.width * hdrRes.height * sizeof(float), cudaMemcpyHostToDevice);

    size_t offset;
    cudaBindTexture2D(&offset, &text_ref_r, d_hdr_img_r, &h_channel_desc, hdrRes.width, hdrRes.height, hdrRes.width * sizeof(float));
    cudaBindTexture2D(&offset, &text_ref_g, d_hdr_img_g, &h_channel_desc, hdrRes.width, hdrRes.height, hdrRes.width * sizeof(float));
    cudaBindTexture2D(&offset, &text_ref_b, d_hdr_img_b, &h_channel_desc, hdrRes.width, hdrRes.height, hdrRes.width * sizeof(float));
    cout << "HDR load done." << endl;
    check_error("set HDR");

    // 发光三角形索引
    cudaMalloc(&emitTrianglesIndices_cu, emit_triangles_indices.size() * sizeof(int));
    cudaMemcpy(emitTrianglesIndices_cu, &emit_triangles_indices[0], emit_triangles_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cout << "Emit triangle indices done." << endl;

    check_error("set emit triangle");
    // 渲染参数设置
    int spp_hs;
    cout << "Sample per Pixel: " << endl;
    cin >> spp_hs;

    cudaMemcpyToSymbol(nTriangles_dv, &nTriangles, sizeof(int), 0, cudaMemcpyHostToDevice);
    check_error("copy symbol nTriangle");
    cudaMemcpyToSymbol(nEmitTriangles_dv, &nEmitTriangles, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(nNodes_dv, &nNodes, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(spp, &spp_hs, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(eye_dv, &eye_center, sizeof(float3), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(camera_transform_dv, camera_transform, sizeof(float) * 4 * 4, 0, cudaMemcpyHostToDevice);
    check_error("copy symbol");
// ----------------------------------------------------------------------------- //

    cout << "Start..." << endl << endl;

    // initial random
    curandState* curand_states;
    cudaMalloc(&curand_states, RAND_SIZE * sizeof(curandState));

    init_curand <<<1, RAND_SIZE>>> (curand_states, 0);
    cudaDeviceSynchronize();
    check_error("curand init");

    // start render
    dim3 grid{RENDER_WIDTH / TILE_SIZE, RENDER_HEIGHT / TILE_SIZE, 1};
    dim3 block{TILE_SIZE, TILE_SIZE, 1};
    
    unsigned char* d_target_img;
    cudaMalloc(&d_target_img, RENDER_WIDTH * RENDER_HEIGHT * 3);

    render_pixel <<<grid, block>>> (d_target_img, curand_states, triangles_cu, node_cu, emitTrianglesIndices_cu, triangle_index_mapping_cu, prefix_size_sum_cu, obj_segs_cu);

    unsigned char* h_target_img = (unsigned char*)malloc(RENDER_WIDTH * RENDER_HEIGHT * 3);

    cudaDeviceSynchronize();

    check_error("Render");
    
    cudaMemcpy(h_target_img, d_target_img, RENDER_WIDTH * RENDER_HEIGHT * 3, cudaMemcpyDeviceToHost);
    check_error("copy");
    save_image(h_target_img, RENDER_WIDTH, RENDER_HEIGHT);
    free(h_target_img);
    free(obj_trans_mats);

    cudaUnbindTexture(text_ref_b);
    cudaUnbindTexture(text_ref_g);
    cudaUnbindTexture(text_ref_r);
    cudaFree(d_hdr_img_r);
    cudaFree(d_hdr_img_g);
    cudaFree(d_hdr_img_b);
    free(h_hdr_img_b);
    free(h_hdr_img_g);
    free(h_hdr_img_r);

    cudaFree(d_target_img);
    cudaFree(curand_states);

    cudaFree(emitTrianglesIndices_cu);
    cudaFree(node_cu);
    cudaFree(triangles_cu);
    cudaDeviceReset();

    return 0;
}