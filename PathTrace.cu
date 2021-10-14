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

using namespace std;

#define INF 2147483647
#ifdef LARGE
#define RENDER_WIDTH 1024
#define RENDER_HEIGHT 1024
#else
#define RENDER_WIDTH 128
#define RENDER_HEIGHT 128
#endif

#define TILE_SIZE 16
#define STACK_CAPACITY 128
#define SHARED_MEM_CAP STACK_CAPACITY * RENDER_WIDTH * RENDER_HEIGHT
#define RR_RATE 0.9
#define PI 3.1415926

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
    FILE* file_ptr = fopen("RenderResult.bmp", "wb+");

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

    __host__ vec3_hs(float x, float y, float z) {
        data.x = x;
        data.y = y;
        data.z = z;
    }

    __host__ inline vec3_hs operator+(const vec3_hs& opr2) {
        return vec3_hs(make_float3(data.x + opr2.data.x, data.y + opr2.data.y, data.z + opr2.data.z));
    }

    __host__ inline vec3_hs operator-(const vec3_hs& opr2) {
        return vec3_hs(make_float3(data.x - opr2.data.x, data.y - opr2.data.y, data.z - opr2.data.z));
    }

    __host__ inline vec3_hs operator*(const vec3_hs& opr2) {
        return vec3_hs(make_float3(data.x * opr2.data.x, data.y * opr2.data.y, data.z * opr2.data.z));
    }

    __host__ inline vec3_hs operator*(float scalar) {
        return vec3_hs(make_float3(data.x * scalar, data.y * scalar, data.z * scalar));
    }

    __host__ inline vec3_hs operator/(const vec3_hs& opr2) {
        return vec3_hs(make_float3(data.x / opr2.data.x, data.y / opr2.data.y, data.z / opr2.data.z));
    }

    __host__ inline vec3_hs normalize() {
        float length_rev = 1.0 / norm3df(data.x, data.y, data.z);
        return vec3_hs(make_float3(data.x * length_rev, data.y * length_rev, data.z * length_rev));
    }

    inline vec3_hs normalize_host() {
        float length_rev = 1.0 / norm3df(data.x, data.y, data.z);
        return vec3_hs(make_float3(data.x * length_rev, data.y * length_rev, data.z * length_rev));
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

vec3_hs transform(const vec3_hs& vec3, float f4, float mat4[4][4])
{
    vec3_hs v3(0, 0, 0);
    v3.data.x = mat4[0][0] * vec3.data.x + mat4[0][1] * vec3.data.y + mat4[0][2] * vec3.data.z + mat4[0][3] * f4;
    v3.data.y = mat4[1][0] * vec3.data.x + mat4[1][1] * vec3.data.y + mat4[1][2] * vec3.data.z + mat4[1][3] * f4;
    v3.data.z = mat4[2][0] * vec3.data.x + mat4[2][1] * vec3.data.y + mat4[2][2] * vec3.data.z + mat4[2][3] * f4;

    return v3;
}


// use in device
struct vec3_dv {
    float3 data;

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

    __device__ inline vec3_dv operator-(const vec3_dv& opr2) {
        return vec3_dv(make_float3(data.x - opr2.data.x, data.y - opr2.data.y, data.z - opr2.data.z));
    }

    __device__ inline vec3_dv operator*(const vec3_dv& opr2) {
        return vec3_dv(make_float3(data.x * opr2.data.x, data.y * opr2.data.y, data.z * opr2.data.z));
    }

    __device__ inline vec3_dv operator*(float scalar) {
        return vec3_dv(make_float3(data.x * scalar, data.y * scalar, data.z * scalar));
    }

    __device__ inline vec3_dv operator/(const vec3_dv& opr2) {
        return vec3_dv(make_float3(data.x / opr2.data.x, data.y / opr2.data.y, data.z / opr2.data.z));
    }

    __device__ inline vec3_dv normalize() {
        float length_rev = 1.0 / norm3df(data.x, data.y, data.z);
        return vec3_dv(make_float3(data.x * length_rev, data.y * length_rev, data.z * length_rev));
    }

    inline vec3_dv normalize_host() {
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
    v3.data.x = mat4[0][0] * vec3.data.x + mat4[0][1] * vec3.data.y + mat4[0][2] * vec3.data.z + mat4[0][3] * f4;
    v3.data.y = mat4[1][0] * vec3.data.x + mat4[1][1] * vec3.data.y + mat4[1][2] * vec3.data.z + mat4[1][3] * f4;
    v3.data.z = mat4[2][0] * vec3.data.x + mat4[2][1] * vec3.data.y + mat4[2][2] * vec3.data.z + mat4[2][3] * f4;

    return v3;
}

// 物体表面材质定义
// complex calculated in device, edit in host
struct Material {
    vec3_dv emissive = vec3_dv(0, 0, 0);  // 作为光源时的发光颜色
    vec3_dv brdf = vec3_dv(0.8, 0.8, 0.8); // BRDF
};

// 三角形定义
// used in host
struct Triangle {
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
};

// used in device
struct Triangle_cu {
    vec3_dv p1, p2, p3;    // 顶点坐标
    vec3_dv norm;          // 法线
    vec3_dv emissive;      // 自发光参数
    vec3_dv brdf;          // BRDF
};

// used in device
struct BVHNode_cu {
    int left, right;    // 左右子树索引
    int n, index;       // 叶子节点信息
    vec3_dv AA, BB;        // 碰撞盒
};

// 读取 obj
void readObj(const string& filepath, vector<Triangle>& triangles, Material material, float trans[4][4], bool normal_transform) {

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
            v -= center;
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

    for (int i = 0; i < indices.size(); i += 3) {
        Triangle& t = triangles[offset + i / 3];
        // 传顶点属性
        t.p1 = vertices[indices[i]];
        t.p2 = vertices[indices[i + 1]];
        t.p3 = vertices[indices[i + 2]];
        // 计算法线
        t.norm = normalize_host(cross(t.p2 - t.p1, t.p3 - t.p1));

        // 传材质
        t.material = material;
    }
}

// 按照三角形中心排序 -- 比较函数
bool cmpx(const Triangle& t1, const Triangle& t2) {
    vec3_hs center1 = (t1.p1 + t1.p2 + t1.p3) / vec3_hs(3, 3, 3);
    vec3_hs center2 = (t2.p1 + t2.p2 + t2.p3) / vec3_hs(3, 3, 3);
    return center1.data.x < center2.data.x;
}
bool cmpy(const Triangle& t1, const Triangle& t2) {
    vec3_hs center1 = (t1.p1 + t1.p2 + t1.p3) / vec3_hs(3, 3, 3);
    vec3_hs center2 = (t2.p1 + t2.p2 + t2.p3) / vec3_hs(3, 3, 3);
    return center1.data.y < center2.data.y;
}
bool cmpz(const Triangle& t1, const Triangle& t2) {
    vec3_hs center1 = (t1.p1 + t1.p2 + t1.p3) / vec3_hs(3, 3, 3);
    vec3_hs center2 = (t2.p1 + t2.p2 + t2.p3) / vec3_hs(3, 3, 3);
    return center1.data.z < center2.data.z;
}

// SAH 优化构建 BVH
int buildBVHwithSAH(vector<Triangle>& triangles, vector<BVHNode>& nodes, int l, int r, int n) {
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
int spp = 128;

vector<BVHNode> nodes;
int nEmitTriangles = 0;


// Graphic Memories pointer
Triangle_cu* triangles_cu;
BVHNode_cu* node_cu;
int* emitTrianglesIndices_cu;
// HDR贴图
texture<float3, cudaTextureType2D, cudaReadModeElementType> texRef;


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
    float obj_trans_mats[obj_cnt][4][4];
    vector<Material> obj_materials(obj_cnt);
    vector<bool> obj_normalize(obj_cnt);
    for (int i = 0; i < obj_cnt; ++i) {
        fin >> obj_file_name[i];

        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                fin >> obj_trans_mats[i][row][col];
            }
        }

        fin >> obj_materials[i].emissive.data.x >> obj_materials[i].emissive.data.y >> obj_materials[i].emissive.data.z;
        fin >> obj_materials[i].brdf.data.x >> obj_materials[i].brdf.data.y >> obj_materials[i].brdf.data.z;

        fin >> obj_normalize[i];
    }

    fin.close();

    vector<Triangle> triangles;
    for (int i = 0; i < obj_cnt; ++i) {
        readObj(obj_file_name[i], triangles, obj_materials[i], obj_trans_mats[i], obj_normalize[i]);
    }

    size_t nTriangles = triangles.size();

    cout << "Model load done:  " << nTriangles << " Triangles." << endl;

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
    for (int i = 0; i < nTriangles; i++) {
        Triangle& t = triangles[i];
        Material& m_ = t.material;
        // 顶点位置
        triangles_encoded[i].p1 = vec3_dv(t.p1);
        triangles_encoded[i].p2 = vec3_dv(t.p2);
        triangles_encoded[i].p3 = vec3_dv(t.p3);
        // 顶点法线
        triangles_encoded[i].norm = vec3_dv(t.norm);
        // 材质
        triangles_encoded[i].emissive = m_.emissive;
        triangles_encoded[i].brdf = m_.brdf;

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
    // 三角形数组
    cudaMalloc(&triangles_cu, nTriangles * sizeof(Triangle_cu));
    cudaMemcpy(triangles_cu, &triangles_encoded[0], nTriangles * sizeof(Triangle_cu), cudaMemcpyHostToDevice);
    cout << "GL Triangle Set." << endl;

    // BVHNode 数组
    cudaMalloc(&node_cu, nodes_encoded.size() * sizeof(BVHNode_cu));
    cudaMemcpy(triangles_cu, &nodes_encoded[0], nodes_encoded.size() * sizeof(BVHNode_cu), cudaMemcpyHostToDevice);
    cout << "GL BVH Set." << endl;

    // hdr 全景图
    HDRLoaderResult hdrRes;
    HDRLoader::load("background.hdr", hdrRes);
    
    cout << "HDR load done." << endl;

    // 发光三角形索引
    
    cout << "Emit triangle indices done." << endl;

    // 渲染参数设置

// ----------------------------------------------------------------------------- //

    cout << "Start..." << endl << endl;

    return 0;
}