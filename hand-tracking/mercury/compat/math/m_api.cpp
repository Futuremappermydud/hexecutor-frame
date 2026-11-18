// Copyright 2019-2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief The m_api.h subset, following Monado's math/m_base.cpp
 *        (3acbda6b7dacbe014749020e8543fded54ef076b).
 */
#include "math/m_api.h"
#include "math/m_eigen_interop.hpp"

#include <Eigen/Geometry>

using xrt::auxiliary::math::map_quat;
using xrt::auxiliary::math::map_vec3;

extern "C" void
math_vec3_cross(const struct xrt_vec3 *l, const struct xrt_vec3 *r, struct xrt_vec3 *result)
{
	map_vec3(*result) = map_vec3(*l).cross(map_vec3(*r));
}

extern "C" void
math_vec3_normalize(struct xrt_vec3 *in)
{
	map_vec3(*in) = map_vec3(*in).normalized();
}

extern "C" void
math_quat_normalize(struct xrt_quat *inout)
{
	map_quat(*inout).normalize();
}

extern "C" void
math_quat_invert(const struct xrt_quat *quat, struct xrt_quat *out_quat)
{
	map_quat(*out_quat) = map_quat(*quat).conjugate();
}

extern "C" void
math_quat_rotate_vec3(const struct xrt_quat *left, const struct xrt_vec3 *right, struct xrt_vec3 *result)
{
	Eigen::Quaternionf l = map_quat(*left);
	Eigen::Vector3f r = map_vec3(*right);
	map_vec3(*result) = l * r;
}

extern "C" void
math_quat_from_matrix_3x3(const struct xrt_matrix_3x3 *mat, struct xrt_quat *result)
{
	Eigen::Matrix3f m;
	m << mat->v[0], mat->v[1], mat->v[2], mat->v[3], mat->v[4], mat->v[5], mat->v[6], mat->v[7], mat->v[8];
	map_quat(*result) = Eigen::Quaternionf(m);
}

extern "C" void
math_quat_from_plus_x_z(const struct xrt_vec3 *plus_x, const struct xrt_vec3 *plus_z, struct xrt_quat *result)
{
	xrt_vec3 plus_y;
	math_vec3_cross(plus_z, plus_x, &plus_y);
	xrt_matrix_3x3 m = {{
	    plus_x->x, plus_y.x, plus_z->x, //
	    plus_x->y, plus_y.y, plus_z->y, //
	    plus_x->z, plus_y.z, plus_z->z, //
	}};
	math_quat_from_matrix_3x3(&m, result);
}

extern "C" void
math_pose_invert(const struct xrt_pose *pose, struct xrt_pose *outPose)
{
	Eigen::Isometry3f transform{Eigen::Translation3f{map_vec3(pose->position)} *
	                            Eigen::Quaternionf(map_quat(pose->orientation))};
	Eigen::Isometry3f inverse = transform.inverse();
	map_vec3(outPose->position) = inverse.translation();
	map_quat(outPose->orientation) = Eigen::Quaternionf(inverse.rotation());
}

extern "C" void
math_pose_transform_point(const struct xrt_pose *transform, const struct xrt_vec3 *point, struct xrt_vec3 *out_point)
{
	Eigen::Vector3f p = Eigen::Quaternionf(map_quat(transform->orientation)) * map_vec3(*point) +
	                    map_vec3(transform->position);
	map_vec3(*out_point) = p;
}
