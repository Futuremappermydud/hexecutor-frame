"""Offline physical KB4 lens ablation for natural stereo self-calibration.

Original virtual pinhole pixels are mapped through the frozen physical KB4
camera. Changes to physical right focal scale and either physical k1 coefficient
are estimated with relative rotation. Baseline direction/length and both optical
axes remain supplied geometry. This is not an arbitrary image correction field.
Training epipolar agreement does not establish horizontal accuracy.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares


class CameraRayModel:
    def __init__(self, intrinsic, physical_from_virtual, lens, actual_spline_intrinsics=None):
        self.actual_spline_intrinsics = actual_spline_intrinsics
        self.actual_spline = None
        if actual_spline_intrinsics is not None:
            from stereo_kb_spline import KBSpline
            self.actual_spline = KBSpline(actual_spline_intrinsics, interpolation="bilinear")
        self.intrinsic = np.asarray(intrinsic, float)
        self.rotation = np.asarray(physical_from_virtual, float)
        self.lens = np.asarray(lens, float)  # fx fy cx cy k1 k2 k3 k4
        if (self.intrinsic.shape != (3, 3) or self.rotation.shape != (3, 3) or self.lens.shape != (8,) or
                not all(np.isfinite(a).all() for a in (self.intrinsic, self.rotation, self.lens)) or
                np.any(self.lens[:2] <= 0) or not np.allclose(self.rotation.T @ self.rotation, np.eye(3), atol=1e-5) or
                not np.isclose(np.linalg.det(self.rotation), 1., atol=1e-5)):
            raise ValueError('finite intrinsic, physical camera rotation and KB4 lens required')
        if (not np.allclose(self.intrinsic[2], [0, 0, 1]) or self.intrinsic[0, 0] <= 0 or self.intrinsic[1, 1] <= 0):
            raise ValueError('positive pinhole intrinsic required')
        self.inverse_intrinsic = np.linalg.inv(self.intrinsic)
        self.inverse_rotation = np.linalg.inv(self.rotation)

    def _coefficients(self, delta_k1):
        if not np.isfinite(delta_k1):
            raise ValueError('finite KB4 coefficient change required')
        k = self.lens[4:].copy(); k[0] += delta_k1
        return k

    @staticmethod
    def _radial(theta, coefficients):
        t = theta*theta
        k1, k2, k3, k4 = coefficients
        return theta*(1+t*(k1+t*(k2+t*(k3+t*k4))))

    @staticmethod
    def _derivative(theta, coefficients):
        t = theta*theta
        k1, k2, k3, k4 = coefficients
        return 1+t*(3*k1+t*(5*k2+t*(7*k3+t*9*k4)))

    def _project_physical(self, points, delta_k1=0., focal_scale=1.):
        p = np.asarray(points, float)
        if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or
                np.any(np.linalg.norm(p, axis=1) == 0) or not np.isfinite(focal_scale) or focal_scale <= 0):
            raise ValueError('finite nonzero 3-D rays and positive focal scale required')
        radius = np.linalg.norm(p[:, :2], axis=1)
        theta = np.arctan2(radius, p[:, 2])
        if np.any(theta >= 1.5):
            raise ValueError('ray outside validated monotonic KB4 inversion interval')
        coefficients = self._coefficients(delta_k1)
        if np.any(self._derivative(theta, coefficients) <= 0):
            raise ValueError('nonmonotonic KB4 camera')
        distorted = self._radial(theta, coefficients)
        scale = np.divide(distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
        return p[:, :2]*scale[:, None]*self.lens[:2]*focal_scale + self.lens[2:4]

    def _unproject_physical(self, pixels, delta_k1=0., focal_scale=1.):
        pixels = np.asarray(pixels, float)
        if (pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all() or
                not np.isfinite(focal_scale) or focal_scale <= 0):
            raise ValueError('finite image pixels and positive focal scale required')
        q = (pixels-self.lens[2:4])/(self.lens[:2]*focal_scale)
        radius = np.linalg.norm(q, axis=1)
        coefficients = self._coefficients(delta_k1)
        # The evaluated IR field is <1.23 radians in both physical cameras.
        # Reject outside a monotonic interval rather than choosing another root.
        grid = np.linspace(0., 1.5, 64)
        if np.any(self._derivative(grid, coefficients) <= 0) or np.any(radius >= self._radial(np.array([1.5]), coefficients)[0]):
            raise ValueError('KB4 inverse outside validated interval')
        theta = np.minimum(radius, 1.49)
        for _ in range(15):
            theta = np.clip(theta-(self._radial(theta, coefficients)-radius)/self._derivative(theta, coefficients), 0., 1.5)
        if np.max(np.abs(self._radial(theta, coefficients)-radius), initial=0.) > 1e-9:
            raise ValueError('KB4 inverse did not converge')
        scale = np.divide(np.sin(theta), radius, out=np.ones_like(radius), where=radius > 1e-12)
        return np.column_stack((q*scale[:, None], np.cos(theta)))

    def _actual_camera(self, delta_k1, focal_scale):
        import copy
        if not np.isfinite(delta_k1) or not np.isfinite(focal_scale) or focal_scale <= 0:
            raise ValueError('finite lens change and positive focal scale required')
        camera = copy.copy(self.actual_spline)
        camera.k = camera.k.copy(); camera.k[0] += delta_k1
        camera.fx *= focal_scale; camera.fy *= focal_scale
        return camera

    def _project_actual(self, points, delta_k1=0., focal_scale=1.):
        if self.actual_spline is None:
            return self._project_physical(points, delta_k1, focal_scale)
        camera = self._actual_camera(delta_k1, focal_scale)
        points = np.asarray(points,float)
        theta = np.arctan2(np.linalg.norm(points[:,:2],axis=1),points[:,2])
        if np.any(theta >= 1.5) or np.any(self._derivative(theta,camera.k) <= 0):
            raise ValueError('actual spline projection outside monotonic forward domain')
        return np.array(camera.project(points.T)).T

    def _unproject_actual(self, pixels, delta_k1=0., focal_scale=1.):
        if self.actual_spline is None:
            return self._unproject_physical(pixels, delta_k1, focal_scale)
        return self._actual_camera(delta_k1, focal_scale).unproject(pixels)

    def rays(self, pixels, delta_k1=0., focal_scale=1., principal_shift=(0.,0.)):
        pixels = np.asarray(pixels, float).reshape(-1, 2)
        nominal = np.column_stack((pixels, np.ones(len(pixels)))) @ self.inverse_intrinsic.T
        raw = self._project_physical(nominal @ self.rotation.T)
        actual = self._unproject_actual(raw-np.asarray(principal_shift), delta_k1, focal_scale) @ self.inverse_rotation.T
        if np.any(actual[:, 2] <= 0):
            raise ValueError('corrected ray behind virtual camera')
        return actual/actual[:, 2, None]

    def project(self, points, delta_k1=0., focal_scale=1., principal_shift=(0.,0.)):
        raw = self._project_actual(np.asarray(points, float) @ self.rotation.T, delta_k1, focal_scale)+np.asarray(principal_shift)
        nominal = self._unproject_physical(raw) @ self.inverse_rotation.T
        if np.any(nominal[:, 2] <= 0):
            raise ValueError('projected point behind original virtual camera')
        q = nominal @ self.intrinsic.T
        return q[:, :2]/q[:, 2, None]

    def project_for_optimization(self, points, delta_k1=0., focal_scale=1., principal_shift=(0.,0.)):
        """Finite trial-step extension; accepted fits must have every active ray valid.

        On the supported field this is exactly project(). Outside it, clipped
        directions and penalties let a trust-region solver recover instead of
        throwing on a trial step. The returned validity mask is mandatory:
        these extended projections must never be published as calibrated rays.
        """
        points = np.asarray(points, float)
        physical = points @ self.rotation.T
        radius = np.linalg.norm(physical[:, :2], axis=1)
        theta = np.arctan2(radius, physical[:, 2])
        clipped = np.minimum(theta, 1.35)
        axis = np.divide(physical[:, :2], radius[:, None], out=np.zeros_like(physical[:, :2]),
                         where=radius[:, None] > 1e-12)
        safe = np.column_stack((axis*np.sin(clipped)[:, None], np.cos(clipped)))
        raw = self._project_actual(safe, delta_k1, focal_scale)+np.asarray(principal_shift)
        nominal = self._unproject_physical(raw) @ self.inverse_rotation.T
        valid = ((theta <= 1.35) & (nominal[:, 2] > .05) & (points[:, 2] > 0) &
                 (np.linalg.norm(points, axis=1) > 1e-12))
        q = nominal @ self.intrinsic.T
        predicted = q[:, :2]/np.maximum(nominal[:, 2, None], .05)
        penalty = (1000*np.maximum(theta-1.35, 0.) +
                   1000*np.maximum(.05-nominal[:, 2], 0.) + 1000*np.maximum(-points[:, 2], 0.))
        return predicted+penalty[:, None], valid


def epipolar_residual(matches, left, right, parameters):
    p = np.asarray(parameters, float)
    l = calibrated_left_camera(left,p).rays(matches[:, :2], p[4])
    if p.shape in ((10,), (11,)):right=shifted_principal_camera(right,p[8:10])
    r = right.rays(matches[:, 2:], p[5], np.exp(p[3])) @ cv2.Rodrigues(p[:3])[0].T
    plane=np.cross(np.broadcast_to(baseline_direction(p),l.shape),l)
    if np.any(np.abs(plane[:,1])<1e-8):raise ValueError('near-vertical epipolar line has no stable row residual')
    return np.sum(r*plane,axis=1)/r[:,2]/plane[:,1]*left.intrinsic[1,1]


def baseline_direction(parameters):
    """Layouts: physical6; +baseline slopes (8); +right center (10); +left log focal (11)."""
    p = np.asarray(parameters, float)
    if p.shape not in ((6,), (8,), (10,), (11,)) or not np.isfinite(p).all():
        raise ValueError('physical layout of six, eight, ten or eleven finite values required')
    d = np.r_[1., p[6:8]] if len(p) >= 8 else np.array([1., 0., 0.])
    return d/np.linalg.norm(d)


class _PrincipalShiftCamera(CameraRayModel):
    """Shift actual optical center while keeping the original nominal rectifier."""
    def __init__(self,camera,shift):
        super().__init__(camera.intrinsic,camera.rotation,camera.lens,camera.actual_spline_intrinsics)
        self.shift=np.asarray(shift,float)
        if self.shift.shape!=(2,) or not np.isfinite(self.shift).all():raise ValueError('finite physical principal-point shift required')
    def project(self,points,delta_k1=0.,focal_scale=1.):
        return super().project(points,delta_k1,focal_scale,self.shift)
    def rays(self,pixels,delta_k1=0.,focal_scale=1.):
        return super().rays(pixels,delta_k1,focal_scale,self.shift)
    def project_for_optimization(self,points,delta_k1=0.,focal_scale=1.):
        return super().project_for_optimization(points,delta_k1,focal_scale,self.shift)


def shifted_principal_camera(camera,shift):
    return _PrincipalShiftCamera(camera,shift)


class _FocalScaleCamera(CameraRayModel):
    def __init__(self,camera,scale):
        super().__init__(camera.intrinsic,camera.rotation,camera.lens,camera.actual_spline_intrinsics)
        self.scale=float(scale)
        if not np.isfinite(self.scale) or self.scale<=0:raise ValueError('positive finite focal scale required')
    def project(self,points,delta_k1=0.,focal_scale=1.):
        return super().project(points,delta_k1,focal_scale*self.scale)
    def rays(self,pixels,delta_k1=0.,focal_scale=1.):
        return super().rays(pixels,delta_k1,focal_scale*self.scale)
    def project_for_optimization(self,points,delta_k1=0.,focal_scale=1.):
        return super().project_for_optimization(points,delta_k1,focal_scale*self.scale)


def calibrated_left_camera(camera,parameters):
    """Eleven-value physical layout appends left log focal after prior ten."""
    p=np.asarray(parameters,float)
    return _FocalScaleCamera(camera,np.exp(p[10])) if p.shape==(11,) else camera


def project_right(points, right, parameters, baseline):
    p = np.asarray(parameters, float)
    if p.shape in ((10,), (11,)):right=shifted_principal_camera(right,p[8:10])
    rays = (np.asarray(points, float)-baseline*baseline_direction(p)) @ cv2.Rodrigues(p[:3])[0]
    return right.project(rays, p[5], np.exp(p[3]))


def fit_lens_calibration(matches, left, right, estimate_lens=True):
    matches = np.asarray(matches, float)
    if matches.ndim != 2 or matches.shape[1] != 4 or not np.isfinite(matches).all():
        raise ValueError('finite N-by-4 matches required')
    bins = np.floor(matches[:, :2]/2.).astype(np.int64)
    _, inverse, counts = np.unique(bins, axis=0, return_inverse=True, return_counts=True)
    weights = 1./counts[inverse] if len(matches) else np.empty(0)
    ncal = 6 if estimate_lens else 4
    p = np.zeros(6); reasons = []; fit = None
    bounds = np.array([.15, .15, .15, .05, .02, .02])[:ncal]
    def expand(x):
        return np.r_[x, np.zeros(6-ncal)]
    if len(counts) < 100:
        reasons.append('insufficient_unique_support')
    else:
        try:
            fit = least_squares(lambda x: epipolar_residual(matches, left, right, expand(x))*np.sqrt(weights),
                                p[:ncal], bounds=(-bounds, bounds), loss='soft_l1', f_scale=.5,
                                x_scale='jac', ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100)
            p = expand(fit.x)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            reasons.append('solver_failure')
    residuals = epipolar_residual(matches, left, right, p)
    inliers = np.abs(residuals) <= 1.
    fraction = float(weights[inliers].sum()/len(counts)) if len(counts) else 0.
    if fraction < .6:
        reasons.append('insufficient_inliers')
    if fit is None or not fit.success:
        reasons.append('not_converged')
    if np.any(np.abs(p[:ncal]) >= .999*bounds):
        reasons.append('parameter_bound_reached')
    singular, ratio = [], 0.
    if fit is not None:
        singular = np.linalg.svd(fit.jac, compute_uv=False)
        ratio = float(singular[-1]/singular[0]) if singular[0] > 0 else 0.
        if ratio < 1e-5:
            reasons.append('rank_or_conditioning_failure')
    span = np.ptp(matches[:, :2], axis=0)/np.diag(left.intrinsic)[:2] if len(matches) else np.zeros(2)
    if np.min(span) < .15:
        reasons.append('insufficient_spatial_span')
    return {'model': 'physical_kb4_relative6' if estimate_lens else 'physical_kb4_relative4', 'parameters_vector': p.tolist(),
            'parameters': {'rotation_vector_rad': p[:3].tolist(), 'physical_right_focal_scale': float(np.exp(p[3])),
                           'left_delta_k1': float(p[4]), 'right_delta_k1': float(p[5])},
            'accepted': not reasons, 'metric_acceptance': 'unevaluated',
            'diagnostics': {'reasons': reasons, 'calibration_parameter_count': ncal, 'match_count': len(matches), 'unique_support': len(counts),
                            'inlier_fraction': fraction, 'spatial_span_normalized': span.tolist(),
                            'singular_values': list(map(float, singular)), 'singular_ratio': ratio,
                            'converged': bool(fit is not None and fit.success),
                            'median_abs_epipolar_px': float(np.median(np.abs(residuals))) if len(matches) else None,
                            'scope': 'Training epipolar checks only; lens model and horizontal predictions require independent validation.'}}
