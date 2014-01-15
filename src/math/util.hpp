#pragma once

#include <iostream>
#include <cmath>
#include <vector_types.h>
#include <vector_functions.h>
using std::ostream;

/**
 * Float3 Operators
 */

inline ostream& operator<<(ostream& out, const float3& f) {
    out << "float3(" << f.x << ", " << f.y << ", " << f.z << ")";
    return out;
}

inline __device__ __host__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ __host__ void operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

inline __device__ __host__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ __host__ void operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

inline __device__ __host__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ __host__ void operator*=(float3& a, const float3& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

inline __device__ __host__ float3 operator/(const float3& a, const float3& b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __device__ __host__ void operator/=(float3& a, const float3& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
}

// Scalar Operators
inline __device__ __host__ float3 operator+(const float3& a, const float& b) {
    return a + make_float3(b,b,b);
}

inline __device__ __host__ void operator+=(float3& a, const float& b) {
    a += make_float3(b,b,b);
}

inline __device__ __host__ float3 operator-(const float3& a, const float& b) {
    return a - make_float3(b,b,b);
}

inline __device__ __host__ void operator-=(float3& a, const float& b) {
    a -= make_float3(b,b,b);
}

inline __device__ __host__ float3 operator*(const float3& a, const float& b) {
    return a * make_float3(b,b,b);
}

inline __device__ __host__ void operator*=(float3& a, const float& b) {
    a *= make_float3(b,b,b);
}

inline __device__ __host__ float3 operator/(const float3& a, const float& b) {
    return a / make_float3(b,b,b);
}

inline __device__ __host__ void operator/=(float3& a, const float& b) {
    a /= make_float3(b,b,b);
}

inline __device__ __host__ float3 operator+(const float& b, const float3& a) {
    return make_float3(b,b,b) + a;
}

inline __device__ __host__ float3 operator-(const float& b, const float3& a) {
    return make_float3(b,b,b) - a;
}

inline __device__ __host__ float3 operator*(const float& b, const float3& a) {
    return make_float3(b,b,b) * a;
}

inline __device__ __host__ float3 operator/(const float& b, const float3& a) {
    return make_float3(b,b,b) / a;
}

/**
 * Float4 Operators
 */

inline __device__ __host__ float4 operator+(const float4& a, const float4& b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline __device__ __host__ void operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline __device__ __host__ float4 operator-(const float4& a, const float4& b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __device__ __host__ void operator-=(float4& a, const float4& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline __device__ __host__ float4 operator*(const float4& a, const float4& b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

inline __device__ __host__ void operator*=(float4& a, const float4& b) {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}

inline __device__ __host__ float4 operator/(const float4& a, const float4& b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __device__ __host__ void operator/=(float4& a, const float4& b) {
    a.x /= b.x;
    a.y /= b.y;
    a.z /= b.z;
    a.w /= b.w;
}

// Scalar Operators
inline __device__ __host__ float4 operator+(const float4& a, const float& b) {
    return a + make_float4(b,b,b,b);
}

inline __device__ __host__ void operator+=(float4& a, const float& b) {
    a += make_float4(b,b,b,b);
}

inline __device__ __host__ float4 operator-(const float4& a, const float& b) {
    return a - make_float4(b,b,b,b);
}

inline __device__ __host__ void operator-=(float4& a, const float& b) {
    a -= make_float4(b,b,b,b);
}

inline __device__ __host__ float4 operator*(const float4& a, const float& b) {
    return a * make_float4(b,b,b,b);
}

inline __device__ __host__ void operator*=(float4& a, const float& b) {
    a *= make_float4(b,b,b,b);
}

inline __device__ __host__ float4 operator/(const float4& a, const float& b) {
    return a / make_float4(b,b,b,b);
}

inline __device__ __host__ void operator/=(float4& a, const float& b) {
    a /= make_float4(b,b,b,b);
}

inline __device__ __host__ float4 operator+(const float& b, const float4& a) {
    return make_float4(b,b,b,b) + a;
}

inline __device__ __host__ float4 operator-(const float& b, const float4& a) {
    return make_float4(b,b,b,b) - a;
}

inline __device__ __host__ float4 operator*(const float& b, const float4& a) {
    return make_float4(b,b,b,b) * a;
}

inline __device__ __host__ float4 operator/(const float& b, const float4& a) {
    return make_float4(b,b,b,b) / a;
}

// Vector Math Functions

inline __device__ __host__ float dot(const float3& u, const float3& v) {
    return u.x*v.x + u.y*v.y + u.z*v.z;
}

inline __device__ __host__ float dot(const float4& u, const float4& v) {
    return u.x*v.x + u.y*v.y + u.z*v.z + u.w*v.w;
}

inline __device__ __host__ float norm(const float3& v) {
    return sqrt(dot(v,v));
}

inline __device__ __host__ float norm(const float4& v) {
    return sqrt(dot(v,v));
}

inline __device__ __host__ float norm2(const float3& v) {
    return dot(v,v);
}

inline __device__ __host__ float norm2(const float4& v) {
    return dot(v,v);
}

inline __device__ __host__ float3 normalize(const float3& v) {
    return v / sqrt(norm2(v));
}

inline __device__ __host__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
            (a.y * b.z) - (a.z * b.y),
            (a.z * b.x) - (a.x * b.z),
            (a.x * b.y) - (a.y * b.x)
        );
}
