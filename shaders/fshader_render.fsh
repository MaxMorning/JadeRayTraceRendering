#version 330 core

in vec3 pix;
out vec4 fragColor;

// ----------------------------------------------------------------------------- //

uniform uint frameCounter;
uniform int nTriangles;
uniform int nEmitTriangles;
uniform int nNodes;
uniform int width;
uniform int height;
uniform int spp;

uniform samplerBuffer triangles;
uniform samplerBuffer nodes;

uniform sampler2D lastFrame;
uniform sampler2D hdrMap;
uniform isamplerBuffer emitTrianglesIndices;

uniform vec3 eye;
uniform mat4 cameraRotate;

// ----------------------------------------------------------------------------- //

#define PI              3.1415926
#define INF             114514.0
#define SIZE_TRIANGLE   6
#define SIZE_BVHNODE    4
#define STACK_CAPACITY 128
#define RR_RATE 0.8

// ----------------------------------------------------------------------------- //

// Triangle 数据格式
struct Triangle {
    vec3 p1, p2, p3;    // 顶点坐标
    vec3 norm;    // 顶点法线
};

// BVH 树节点
struct BVHNode {
    int left;           // 左子树
    int right;          // 右子树
    int n;              // 包含三角形数目
    int index;          // 三角形索引
    vec3 AA, BB;        // 碰撞盒
};

// 物体表面材质定义
struct Material {
    vec3 emissive;          // 作为光源时的发光颜色
    vec3 brdf;
};

// 光线
struct Ray {
    vec3 startPoint;
    vec3 direction;
};

// 光线求交结果
struct HitResult {
    bool isHit;             // 是否命中
    int index;              // 命中三角形坐标
    float distance;         // 与交点的距离
    vec3 hitPoint;          // 光线命中点
    vec3 normal;            // 命中点法线
    vec3 viewDir;           // 击中该点的光线的方向
    Material material;      // 命中点的表面材质
};

// ----------------------------------------------------------------------------- //

/*
 * 生成随机向量，依赖于 frameCounter 帧计数器
 * 代码来源：https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
*/

uint seed = uint(
    uint((pix.x * 0.5 + 0.5) * width)  * uint(1973) +
    uint((pix.y * 0.5 + 0.5) * height) * uint(9277) +
    uint(frameCounter) * uint(26699)) | uint(1);

uint wang_hash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}

float rand() {
    return float(wang_hash(seed)) / 4294967296.0;
}

// 将三维向量 v 转为 HDR map 的纹理坐标 uv
vec2 SampleSphericalMap(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv /= vec2(2.0 * PI, PI);
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

// 获取 HDR 环境颜色
vec3 sampleHdr(vec3 v) {
    vec2 uv = SampleSphericalMap(normalize(v));
    vec3 color = texture(hdrMap, uv).rgb;
    color = min(color, vec3(10));
    return color;
}

// ----------------------------------------------------------------------------- //

// 获取第 i 下标的三角形
Triangle getTriangle(int i) {
    int offset = i * SIZE_TRIANGLE;
    Triangle t;

    // 顶点坐标
    t.p1 = texelFetch(triangles, offset + 0).xyz;
    t.p2 = texelFetch(triangles, offset + 1).xyz;
    t.p3 = texelFetch(triangles, offset + 2).xyz;
    // 法线
    t.norm = texelFetch(triangles, offset + 3).xyz;

    return t;
}

// 获取第 i 下标的三角形的材质
Material getMaterial(int i) {
    Material m;

    int offset = i * SIZE_TRIANGLE;
    m.emissive = texelFetch(triangles, offset + 4).xyz;
    m.brdf = texelFetch(triangles, offset + 5).xyz;

    return m;
}

// 获取第 i 下标的 BVHNode 对象
BVHNode getBVHNode(int i) {
    BVHNode node;

    // 左右子树
    int offset = i * SIZE_BVHNODE;
    ivec3 childs = ivec3(texelFetch(nodes, offset + 0).xyz);
    ivec3 leafInfo = ivec3(texelFetch(nodes, offset + 1).xyz);
    node.left = int(childs.x);
    node.right = int(childs.y);
    node.n = int(leafInfo.x);
    node.index = int(leafInfo.y);

    // 包围盒
    node.AA = texelFetch(nodes, offset + 2).xyz;
    node.BB = texelFetch(nodes, offset + 3).xyz;

    return node;
}

// ----------------------------------------------------------------------------- //

// 光线和三角形求交
float mixed_product(vec3 vec_a, vec3 vec_b, vec3 vec_c)
{
    return vec_a.x * (vec_b.y * vec_c.z - vec_b.z * vec_c.y) + 
        vec_a.y * (vec_b.z * vec_c.x - vec_b.x * vec_c.z) + 
        vec_a.z * (vec_b.x * vec_c.y - vec_b.y * vec_c.x);
}

HitResult hitTriangle(Triangle triangle, Ray ray, int index) {
    HitResult res;
    res.distance = INF;
    res.isHit = false;

    vec3 normal_direction = normalize(ray.direction);
    vec3 src_point = ray.startPoint;
    // make shadow
    vec3 shadow_tri_a = triangle.p1 - normal_direction * dot(normal_direction, triangle.p1 - src_point);
    vec3 shadow_tri_b = triangle.p2 - normal_direction * dot(normal_direction, triangle.p2 - src_point);
    vec3 shadow_tri_c = triangle.p3 - normal_direction * dot(normal_direction, triangle.p3 - src_point);

    // check in center
    vec3 vec_pa = shadow_tri_a - src_point;
    vec3 vec_pb = shadow_tri_b - src_point;
    vec3 vec_pc = shadow_tri_c - src_point;

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
        float divider = vec_pb.x * vec_pc.y - vec_pb.y * vec_pc.x;
        float rate_a = (vec_pc.y * vec_pa.x - vec_pc.x * vec_pa.y) / divider;
        float rate_b = (-vec_pb.y * vec_pa.x + vec_pb.x * vec_pa.y) / divider;

        vec_pb = triangle.p2 - triangle.p1;
        vec_pc = triangle.p3 - triangle.p1;
        vec_pa = triangle.p1 + rate_a * vec_pb + rate_b * vec_pc;

        float distance = dot(vec_pa - src_point, normal_direction);
        if (distance > 0) {
            // ray will hit object
            // package result
            res.isHit = true;
            res.hitPoint = vec_pa;
            res.distance = distance;
            res.normal = normalize(cross(triangle.p2 - triangle.p1, triangle.p3 - triangle.p1));
            res.viewDir = ray.direction;
            res.index = index;
        }
    }

    return res;
}

// 和 aabb 盒子求交，没有交点则返回 -1
float hitAABB(Ray r, vec3 AA, vec3 BB) {
    vec3 invdir = 1.0 / r.direction;

    vec3 f = (BB - r.startPoint) * invdir;
    vec3 n = (AA - r.startPoint) * invdir;

    vec3 tmax = max(f, n);
    vec3 tmin = min(f, n);

    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    float t0 = max(tmin.x, max(tmin.y, tmin.z));

    return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1);
}

// ----------------------------------------------------------------------------- //

// 暴力遍历数组下标范围 [l, r] 求最近交点
HitResult hitArray(Ray ray, int l, int r, int src_object_idx) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;
    int min_i = l;
    for(int i = l; i <= r; i++) {
        if (i == src_object_idx) {
            continue;
        }
        Triangle triangle = getTriangle(i);
        HitResult new_hit = hitTriangle(triangle, ray, i);
        if(new_hit.isHit && new_hit.distance < res.distance) {
            res = new_hit;
            min_i = i;
        }
    }
    res.material = getMaterial(min_i);
    return res;
}

// 遍历 BVH 求交
HitResult hitBVH(Ray ray, int src_object_idx) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;

    // 栈
    int stack[256];
    int sp = 0;

    stack[sp] = 1;
    sp++;
    while(sp > 0) {
        --sp;
        int top = stack[sp];
        BVHNode node = getBVHNode(top);

        // 是叶子节点，遍历三角形，求最近交点
        if(node.n > 0) {
            int L = node.index;
            int R = node.index + node.n - 1;
            HitResult r = hitArray(ray, L, R, src_object_idx);
            if(r.isHit && r.distance<res.distance) res = r;
            continue;
        }

        // 和左右盒子 AABB 求交
        float d1 = INF; // 左盒子距离
        float d2 = INF; // 右盒子距离
        if(node.left > 0) {
            BVHNode leftNode = getBVHNode(node.left);
            d1 = hitAABB(ray, leftNode.AA, leftNode.BB);
        }
        if(node.right > 0) {
            BVHNode rightNode = getBVHNode(node.right);
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
vec3 pathTracing_(HitResult hit, int maxBounce) {

    vec3 Lo = vec3(0);      // 最终的颜色
    vec3 history = vec3(1); // 递归积累的颜色

    for(int bounce = 0; bounce < maxBounce; bounce++) {
        // 随机出射方向 wi
        float cosine_theta = 2 * (rand() - 0.5);
        float sine_theta = sqrt(1 - cosine_theta * cosine_theta);
        float fai_value = 2 * PI * rand();
        vec3 wi = vec3(sine_theta * cos(fai_value), sine_theta * sin(fai_value), cosine_theta);
        if (dot(wi, hit.normal) * dot(hit.viewDir, hit.normal) > 0) {
            wi *= -1;
        }

        // 漫反射: 随机发射光线
        Ray randomRay;
        randomRay.startPoint = hit.hitPoint;
        randomRay.direction = wi;
        HitResult newHit = hitBVH(randomRay, hit.index);

        float pdf = 1.0 / (2.0 * PI);                                   // 半球均匀采样概率密度
        // float cosine_o = abs(dot(-hit.viewDir, hit.normal));         // 入射光和法线夹角余弦
        float cosine_i = abs(dot(randomRay.direction, hit.normal));  // 出射光和法线夹角余弦
        vec3 f_r = hit.material.brdf / PI;                         // 漫反射 BRDF

        // 未命中
        if(!newHit.isHit) {
            vec3 skyColor = sampleHdr(randomRay.direction);
            Lo += history * skyColor * f_r * cosine_i / pdf;
            break;
        }

        // 命中光源积累颜色
        vec3 Le = newHit.material.emissive;
        Lo += history * Le * f_r * cosine_i / pdf;

        // 递归(步进)
        hit = newHit;
        history *= f_r * cosine_i / pdf;  // 累积颜色
    }

    return Lo;
}

float size(Triangle triangle)
{
    vec3 v_1 = triangle.p2 - triangle.p1;
    vec3 v_2 = triangle.p3 - triangle.p1;
    vec3 cross_product = vec3(v_1.y * v_2.z - v_1.z * v_2.y, v_1.z * v_2.x - v_1.x * v_2.z, v_1.x * v_2.y - v_1.y * v_2.x);
    return 0.5 * sqrt(dot(cross_product, cross_product));
}

vec3 pathTracing(HitResult hit) {
    vec3 l_dir = vec3(0);
    int stack_offset = 0;
    vec3 stack_dir[STACK_CAPACITY];
    vec3 stack_indir_rate[STACK_CAPACITY];
    vec3 out_direction = -hit.viewDir;
    vec3 ray_src = hit.hitPoint;
    HitResult obj_hit = hit;
    while (stack_offset < STACK_CAPACITY) {
        // direct light
        // sample from emit triangles
        l_dir = vec3(0);
        vec3 obj_hit_fr = obj_hit.material.brdf / PI;
        for (int i = 0; i < nEmitTriangles; ++i) {
            // random select a point on light triangle
            float rand_x = rand();
            float rand_y = rand();
            if (rand_x + rand_y > 1) {
                rand_x = 1 - rand_x;
                rand_y = 1 - rand_y;
            }

            int emit_tri_idx = texelFetch(emitTrianglesIndices, i).x;
            Triangle t_i = getTriangle(emit_tri_idx);
            vec3 random_point = t_i.p1 + (t_i.p2 - t_i.p1) * rand_x + (t_i.p3 - t_i.p1) * rand_y;

            // test block
            vec3 obj_light_direction = random_point - ray_src;
            // check obj_light_direction and out_direction are in the same semi-sphere
            if (dot(obj_light_direction, obj_hit.normal) * dot(out_direction, obj_hit.normal) < 0) {
                continue;
            }
            Ray new_ray;
            new_ray.startPoint = ray_src;
            new_ray.direction = obj_light_direction;
            HitResult hit_result = hitBVH(new_ray, obj_hit.index);

            if (hit_result.isHit && hit_result.index == emit_tri_idx) {
                float direction_length_square = obj_light_direction.x * obj_light_direction.x + obj_light_direction.y * obj_light_direction.y + obj_light_direction.z * obj_light_direction.z;
                l_dir += hit_result.material.emissive * obj_hit_fr * abs(dot(obj_hit.normal, obj_light_direction) * dot(hit_result.normal, obj_light_direction)) 
                            / direction_length_square / direction_length_square * size(t_i);
            }
        }


        // sample from HDR
        // random select a point on HDR Texture
        float cosine_theta = 2 * (rand() - 0.5);
        float sine_theta = sqrt(1 - cosine_theta * cosine_theta);
        float fai_value = 2 * PI * rand();
        vec3 ray_direction = vec3(sine_theta * cos(fai_value), sine_theta * sin(fai_value), cosine_theta);
        if (dot(ray_direction, obj_hit.normal) * dot(out_direction, obj_hit.normal) < 0) {
            ray_direction *= -1;
        }

        // test block
        Ray new_ray;
        new_ray.startPoint = ray_src;
        new_ray.direction = ray_direction;
        HitResult hit_result = hitBVH(new_ray, obj_hit.index);
        if (!hit_result.isHit) {
            vec3 skyColor = sampleHdr(ray_direction);
            l_dir += skyColor * obj_hit_fr * abs(dot(obj_hit.normal, ray_direction)) * 2 * PI;
        }

        float rr_result = rand();
        if (rr_result < RR_RATE) {
            vec3 indir_rate = vec3(0);
            // random select a ray from src_point
            float cosine_theta = 2 * (rand() - 0.5);
            float sine_theta = sqrt(1 - cosine_theta * cosine_theta);
            float fai_value = 2 * PI * rand();
            vec3 ray_direction = vec3(sine_theta * cos(fai_value), sine_theta * sin(fai_value), cosine_theta);
            if (dot(ray_direction, obj_hit.normal) * dot(out_direction, obj_hit.normal) < 0) {
                ray_direction *= -1;
            }

            Ray new_ray;
            new_ray.startPoint = ray_src;
            new_ray.direction = ray_direction;
            HitResult new_hit = hitBVH(new_ray, obj_hit.index);
            if (new_hit.isHit && (new_hit.material.emissive.x < 0.01 && new_hit.material.emissive.y < 0.01 && new_hit.material.emissive.z < 0.01)) {
                // Hit something
                ray_direction *= -1;
                indir_rate = obj_hit_fr * abs(dot(ray_direction, obj_hit.normal)) / RR_RATE;
                ray_src = new_hit.hitPoint;
                out_direction = ray_direction;

                stack_dir[stack_offset] = l_dir;
                stack_indir_rate[stack_offset] = indir_rate;
                ++stack_offset;
                obj_hit = new_hit;
            }
        }
        else {
            break;
        }
    }

    // calc final irradiance
    for (int i = stack_offset - 1; i >= 0; --i) {
        l_dir *= stack_indir_rate[i];
        l_dir += stack_dir[i];
    }
    
    return l_dir;
}
// ----------------------------------------------------------------------------- //

void main() {
    vec3 final_result = vec3(0);
    // 投射光线
    Ray ray;

    ray.startPoint = eye;
    for (int i = 0; i < spp; ++i) {
        vec2 AA = vec2((rand() - 0.5) / float(width), (rand() - 0.5) / float(height));
        vec4 dir = cameraRotate * vec4(pix.xy + AA, -1.5, 0.0);
        ray.direction = normalize(dir.xyz);

        // primary hit
        HitResult firstHit = hitBVH(ray, -1);
        vec3 color;

        if(!firstHit.isHit) {
            color = vec3(0);
            color = sampleHdr(ray.direction);
        } else {
            vec3 Le = firstHit.material.emissive;
            vec3 Li = pathTracing(firstHit);
            // vec3 Li = pathTracing_(firstHit, 2);
            color = Le + Li;
        }

        final_result += color;
    }

    final_result /= spp;
    fragColor = vec4(final_result, 1.0);
}


