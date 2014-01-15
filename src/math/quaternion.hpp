#ifndef __QUATERNION__
#define __QUATERNION__

#include <vector_types.h>
#include "util.hpp"

class quaternion{
	public:
		quaternion(const float& w_, const float& x_, const float& y_, const float& z_){
			w = w_;
			x = x_;
			y = y_;
			z = z_;
		}

		quaternion(const float& w_, const float3& v){
			w = w_;
			x = v.x;
			y = v.y;
			z = v.z;
		}

		float w;
		float x;
		float y;
		float z;

		inline quaternion inverse(){
			const float d = w*w + x*x + y*y + z*z;
			return quaternion(w/d, -x/d, -y/d, -z/d);
		}
};

quaternion qmult(const quaternion& a, const quaternion& b){
	const float3 q = make_float3(a.x, a.y, a.z);
	const float3 r = make_float3(b.x, b.y, b.z);
    return quaternion(
            (a.w * b.w) - dot(q, r),
            (r * a.w) + (q * b.w) + cross(q, r)
            );
}

#endif
