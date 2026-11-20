"""Explicit offline KB spline projection, checked against the installed vendor.

Interoperability evidence/probe: temporal-selfcal-v53/spline-oracle in the local
experiment archive. No vendor code or binary is embedded. The 5x5 grids have
x as their first axis; zero padding extends them to 7x7. Catmull-Rom samples
populate a 101x101 LUT over the central 84% of the six-spacing domain. The
vendor's default projection uses nearest LUT lookup, including edge clamping.
This model is experimental and does not change any rig-loading defaults.
"""
import numpy as np


def _cubic(a, b, c, d, t):
    return b + .5*t*(c-a+t*(2*a-5*b+4*c-d+t*(3*(b-c)+d-a)))


def _sample_grid(grid, x, y):
    """Tensor cubic interpolation with replicated outer padded nodes."""
    ix, iy = np.floor(x).astype(int), np.floor(y).astype(int)
    rows = []
    for dx in (-1, 0, 1, 2):
        values = [grid[np.clip(ix+dx, 0, 6), np.clip(iy+dy, 0, 6)]
                  for dy in (-1, 0, 1, 2)]
        rows.append(_cubic(*values, y-iy))
    return _cubic(*rows, x-ix)


class KBSpline:
    """KB radial projection plus a fixed vendor-format spatial correction.

    ``project`` accepts (3, ...) rays and returns two arrays, matching KB4.
    Only unit image scaling and the archived 5x5 schema are supported.
    """
    def __init__(self, intr, interpolation="nearest"):
        if interpolation not in ("nearest", "bilinear"):
            raise ValueError("nearest or bilinear interpolation required")
        self.interpolation = interpolation
        if intr.get('cameraModel') != 'kb_spline' or intr.get('gridSize') != [5, 5]:
            raise ValueError('explicit 5x5 kb_spline intrinsics required')
        self.fx, self.fy, self.cx, self.cy = [float(np.float32(intr[k])) for k in ('fx','fy','cx','cy')]
        self.k = np.array([intr.get('k'+str(i), 0.) for i in range(1, 5)], np.float32).astype(float)
        self.spacing = np.asarray(intr['nodeSpacing'], np.float32).astype(float)
        grids = np.asarray([intr['sU'], intr['sV']], np.float32).astype(float)
        if (self.spacing.shape != (2,) or grids.shape != (2, 25) or
                not np.isfinite(np.r_[self.fx,self.fy,self.cx,self.cy,self.k,self.spacing,grids.ravel()]).all() or
                min(self.fx,self.fy) <= 0 or np.any(self.spacing <= 0)):
            raise ValueError('finite positive focal lengths/spacing and two 25-node grids required')
        if np.any(grids[:, [7,11,12]] != 0):
            raise ValueError('vendor fixed grid nodes 7,11,12 must be zero')
        grid = np.pad(grids.reshape(2,5,5), ((0,0),(1,1),(1,1)))
        # Preserve the documented LUT discretization, rather than applying a
        # smooth spline directly and silently changing the vendor projection.
        coords = 3 + (np.arange(101)-50)*(6*.84/100)
        x,y = np.meshgrid(coords,coords, indexing='xy')
        self.lut = np.stack([_sample_grid(g,x,y) for g in grid], axis=-1)

    def project(self, points):
        p = np.asarray(points, float)
        if p.ndim < 1 or p.shape[0] != 3 or not np.isfinite(p).all() or np.any(np.linalg.norm(p,axis=0)==0):
            raise ValueError('finite nonzero rays with first dimension 3 required')
        x,y,z = p
        r = np.hypot(x,y)
        t = np.arctan2(r,z)
        t2 = t*t
        rd = t*(1+t2*(self.k[0]+t2*(self.k[1]+t2*(self.k[2]+t2*self.k[3]))))
        scale = np.divide(rd,r,out=np.ones_like(r),where=r>1e-12)
        u,v = self.fx*x*scale+self.cx, self.fy*y*scale+self.cy
        correction = self.correction(np.stack((u,v),axis=-1))
        return u+correction[...,0], v+correction[...,1]

    def correction(self, pixels):
        pixels = np.asarray(pixels,float)
        coordinates = []
        for raw,center,spacing in ((pixels[...,0],self.cx,self.spacing[0]),(pixels[...,1],self.cy,self.spacing[1])):
            # Double coordinates keep bilinear inversion smooth. Their sampled
            # vendor discrepancy is checked separately from nearest LUT identity.
            if self.interpolation == 'nearest':
                inv_grid = np.float32(1.)/(np.float32(6.)*np.float32(spacing))
                coordinate = ((np.asarray(raw,dtype=np.float32)-np.float32(center))*inv_grid*np.float32(100.)/np.float32(.84)+np.float32(50.))
            else:
                coordinate = (raw-center)*100/(6*spacing*.84)+50
            coordinates.append(np.clip(coordinate,0,100))
        x,y = coordinates
        if self.interpolation == 'nearest':
            return self.lut[np.floor(y+.5).astype(int),np.floor(x+.5).astype(int)]
        ix,iy = np.minimum(np.floor(x).astype(int),99),np.minimum(np.floor(y).astype(int),99)
        tx,ty = (x-ix)[...,None],(y-iy)[...,None]
        return ((1-ty)*((1-tx)*self.lut[iy,ix]+tx*self.lut[iy,ix+1])+
                ty*((1-tx)*self.lut[iy+1,ix]+tx*self.lut[iy+1,ix+1]))

    def unproject(self, pixels):
        """Invert the explicit continuous bilinear LUT on the forward KB domain.

        Nearest LUT inversion is deliberately unsupported: it has gaps and
        overlapping image intervals, so a single smooth inverse is undefined.
        """
        if self.interpolation != 'bilinear':
            raise ValueError('nearest lookup has no unique continuous inverse; select bilinear explicitly')
        pixels=np.asarray(pixels,float)
        if pixels.ndim!=2 or pixels.shape[1]!=2 or not np.isfinite(pixels).all():
            raise ValueError('finite Nx2 image pixels required')
        raw=pixels.copy()
        for _ in range(30):
            raw=pixels-self.correction(raw)
        if np.max(np.abs(raw+self.correction(raw)-pixels),initial=0.)>1e-9:
            raise ValueError('spline inverse did not converge')
        q=(raw-[self.cx,self.cy])/[self.fx,self.fy]
        radius=np.linalg.norm(q,axis=1)
        def radial(t):
            t2=t*t
            return t*(1+t2*(self.k[0]+t2*(self.k[1]+t2*(self.k[2]+t2*self.k[3]))))
        def derivative(t):
            t2=t*t
            return 1+t2*(3*self.k[0]+t2*(5*self.k[1]+t2*(7*self.k[2]+t2*9*self.k[3])))
        if np.any(derivative(np.linspace(0,1.5,256))<=0) or np.any(radius>=radial(1.5)):
            raise ValueError('KB inverse outside validated monotonic forward domain')
        theta=np.minimum(radius,1.49)
        for _ in range(20):theta=np.clip(theta-(radial(theta)-radius)/derivative(theta),0,1.5)
        if np.max(np.abs(radial(theta)-radius),initial=0.)>1e-10:
            raise ValueError('KB inverse did not converge')
        scale=np.divide(np.sin(theta),radius,out=np.ones_like(radius),where=radius>1e-12)
        rays=np.column_stack((q*scale[:,None],np.cos(theta)))
        if np.max(np.abs(np.array(self.project(rays.T)).T-pixels),initial=0.)>1e-7:
            raise ValueError('spline inverse reprojection failed')
        return rays
