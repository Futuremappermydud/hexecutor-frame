// Copyright 2019-2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Subset of Monado's math/m_api.h used by the vendored Mercury sources.
 *        Implemented in m_api.cpp with the semantics of Monado's m_base.cpp.
 */
#pragma once

#include "xrt/xrt_defines.h"
#include "math/m_mathinclude.h"

#ifdef __cplusplus
extern "C" {
#endif

void
math_vec3_cross(const struct xrt_vec3 *l, const struct xrt_vec3 *r, struct xrt_vec3 *result);

void
math_vec3_normalize(struct xrt_vec3 *in);

void
math_quat_normalize(struct xrt_quat *inout);

void
math_quat_invert(const struct xrt_quat *quat, struct xrt_quat *out_quat);

void
math_quat_rotate_vec3(const struct xrt_quat *left, const struct xrt_vec3 *right, struct xrt_vec3 *result);

void
math_quat_from_matrix_3x3(const struct xrt_matrix_3x3 *mat, struct xrt_quat *result);

void
math_quat_from_plus_x_z(const struct xrt_vec3 *plus_x, const struct xrt_vec3 *plus_z, struct xrt_quat *result);

void
math_pose_invert(const struct xrt_pose *pose, struct xrt_pose *outPose);

void
math_pose_transform_point(const struct xrt_pose *transform, const struct xrt_vec3 *point, struct xrt_vec3 *out_point);

#ifdef __cplusplus
}
#endif
