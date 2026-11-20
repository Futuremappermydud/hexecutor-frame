// Copyright 2019-2026, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
/*!
 * @file
 * @brief Subset of Monado's xrt/xrt_defines.h needed by the vendored Mercury
 *        sources. Layouts and values are copied unchanged from Monado
 *        3acbda6b7dacbe014749020e8543fded54ef076b.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __GNUC__
#define XRT_MAYBE_UNUSED __attribute__((unused))
#else
#define XRT_MAYBE_UNUSED
#endif

#ifdef __cplusplus
#define XRT_C11_COMPOUND(X)
#else
#define XRT_C11_COMPOUND(X) (X)
#endif

#if defined(__linux__)
#define XRT_OS_LINUX
#endif

struct xrt_quat
{
	float x;
	float y;
	float z;
	float w;
};

#define XRT_QUAT_IDENTITY {0.f, 0.f, 0.f, 1.f}

struct xrt_vec2
{
	float x;
	float y;
};

struct xrt_vec3
{
	float x;
	float y;
	float z;
};

struct xrt_vec3_f64
{
	double x;
	double y;
	double z;
};

#define XRT_VEC3_ZERO {0.f, 0.f, 0.f}
#define XRT_VEC3_UNIT_X {1.f, 0.f, 0.f}
#define XRT_VEC3_UNIT_Y {0.f, 1.f, 0.f}
#define XRT_VEC3_UNIT_Z {0.f, 0.f, 1.f}

struct xrt_size
{
	int w;
	int h;
};

struct xrt_pose
{
	struct xrt_quat orientation;
	struct xrt_vec3 position;
};

#define XRT_POSE_IDENTITY {XRT_QUAT_IDENTITY, XRT_VEC3_ZERO}

#define XRT_MATRIX_3X3_ELEMENTS 9

struct xrt_matrix_3x3
{
	float v[XRT_MATRIX_3X3_ELEMENTS];
};

struct xrt_matrix_3x3_f64
{
	double v[XRT_MATRIX_3X3_ELEMENTS];
};

struct xrt_matrix_4x4
{
	float v[16];
};

struct xrt_matrix_4x4_f64
{
	double v[16];
};

enum xrt_space_relation_flags
{
	XRT_SPACE_RELATION_ORIENTATION_VALID_BIT = (1u << 0u),
	XRT_SPACE_RELATION_POSITION_VALID_BIT = (1u << 1u),
	XRT_SPACE_RELATION_LINEAR_VELOCITY_VALID_BIT = (1u << 2u),
	XRT_SPACE_RELATION_ANGULAR_VELOCITY_VALID_BIT = (1u << 3u),
	XRT_SPACE_RELATION_ORIENTATION_TRACKED_BIT = (1u << 4u),
	XRT_SPACE_RELATION_POSITION_TRACKED_BIT = (1u << 5u),
	XRT_SPACE_RELATION_BITMASK_NONE = 0,
};

struct xrt_space_relation
{
	enum xrt_space_relation_flags relation_flags;
	struct xrt_pose pose;
	struct xrt_vec3 linear_velocity;
	struct xrt_vec3 angular_velocity;
};

enum xrt_hand_joint
{
	XRT_HAND_JOINT_PALM = 0,
	XRT_HAND_JOINT_WRIST = 1,
	XRT_HAND_JOINT_THUMB_METACARPAL = 2,
	XRT_HAND_JOINT_THUMB_PROXIMAL = 3,
	XRT_HAND_JOINT_THUMB_DISTAL = 4,
	XRT_HAND_JOINT_THUMB_TIP = 5,
	XRT_HAND_JOINT_INDEX_METACARPAL = 6,
	XRT_HAND_JOINT_INDEX_PROXIMAL = 7,
	XRT_HAND_JOINT_INDEX_INTERMEDIATE = 8,
	XRT_HAND_JOINT_INDEX_DISTAL = 9,
	XRT_HAND_JOINT_INDEX_TIP = 10,
	XRT_HAND_JOINT_MIDDLE_METACARPAL = 11,
	XRT_HAND_JOINT_MIDDLE_PROXIMAL = 12,
	XRT_HAND_JOINT_MIDDLE_INTERMEDIATE = 13,
	XRT_HAND_JOINT_MIDDLE_DISTAL = 14,
	XRT_HAND_JOINT_MIDDLE_TIP = 15,
	XRT_HAND_JOINT_RING_METACARPAL = 16,
	XRT_HAND_JOINT_RING_PROXIMAL = 17,
	XRT_HAND_JOINT_RING_INTERMEDIATE = 18,
	XRT_HAND_JOINT_RING_DISTAL = 19,
	XRT_HAND_JOINT_RING_TIP = 20,
	XRT_HAND_JOINT_LITTLE_METACARPAL = 21,
	XRT_HAND_JOINT_LITTLE_PROXIMAL = 22,
	XRT_HAND_JOINT_LITTLE_INTERMEDIATE = 23,
	XRT_HAND_JOINT_LITTLE_DISTAL = 24,
	XRT_HAND_JOINT_LITTLE_TIP = 25,
	XRT_HAND_JOINT_MAX_ENUM = 0x7FFFFFFF
};

#define XRT_HAND_JOINT_COUNT 26

struct xrt_hand_joint_value
{
	struct xrt_space_relation relation;
	float radius;
};

struct xrt_hand_joint_set
{
	union {
		struct xrt_hand_joint_value hand_joint_set_default[XRT_HAND_JOINT_COUNT];
	} values;
	struct xrt_space_relation hand_pose;
	bool is_active;
};
