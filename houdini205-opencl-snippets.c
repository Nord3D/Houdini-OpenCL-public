/*** HOUDINI VOLUME functions ***/

size_t  /* voxel idx from ix iy iz, or from "ijk" */
ijk2idx ( uint i, uint j, uint k,
         uint offset, uint ystride, uint zstride ) {
    return offset + i + j*ystride + k*zstride;
}

float3  /* Gradient at voxel */
du_ixyz ( __global const float * restrict u, 
            uint ix, uint iy, uint iz,
            uint of, uint ys, uint zs,
            float inv_two_dx ) 
{
    return inv_two_dx * (float3)(
    /* X */     u[ijk2idx(ix+1,iy,iz,of,ys,zs)] - u[ijk2idx(ix-1,iy,iz,of,ys,zs)],
    /* Y */     u[ijk2idx(ix,iy+1,iz,of,ys,zs)] - u[ijk2idx(ix,iy-1,iz,of,ys,zs)],
    /* Z */     u[ijk2idx(ix,iy,iz+1,of,ys,zs)] - u[ijk2idx(ix,iy,iz-1,of,ys,zs)]);
}

float3  /* Gradient at pos in index-space */
du_ipos ( __global const float * restrict u, 
            float3 pos,
            uint of, uint ys, uint zs,
            uint resx, uint resy, uint resz,
            float inv_two_dx )
{
    float2 st = (float2)(1, 0);
    return inv_two_dx * (float3)(
    /* X+1 */       trilinear_interp_vol(pos+st.xyy,u,of,1,ys,zs,resx,resy,resz)
    /* X-1 */     - trilinear_interp_vol(pos-st.xyy,u,of,1,ys,zs,resx,resy,resz),
    /* Y+1 */       trilinear_interp_vol(pos+st.yxy,u,of,1,ys,zs,resx,resy,resz)
    /* Y-1 */     - trilinear_interp_vol(pos-st.yxy,u,of,1,ys,zs,resx,resy,resz),
    /* Z+1 */       trilinear_interp_vol(pos+st.yyx,u,of,1,ys,zs,resx,resy,resz)
    /* Z-1 */     - trilinear_interp_vol(pos-st.yyx,u,of,1,ys,zs,resx,resy,resz));
}

float  /* Mean value smooth (assumed use in iterations) */
smooth_vol( __global const float * restrict u, 
            float3 pos,
            uint of, uint ys, uint zs,
            uint resx, uint resy, uint resz ) {
    float2 st = (float2)(1, 0);
    return (
    /* X+1 */       trilinear_interp_vol(pos+st.xyy,u,of,1,ys,zs,resx,resy,resz)
    /* X-1 */     + trilinear_interp_vol(pos-st.xyy,u,of,1,ys,zs,resx,resy,resz)
    /* Y+1 */     + trilinear_interp_vol(pos+st.yxy,u,of,1,ys,zs,resx,resy,resz)
    /* Y-1 */     + trilinear_interp_vol(pos-st.yxy,u,of,1,ys,zs,resx,resy,resz)
    /* Z+1 */     + trilinear_interp_vol(pos+st.yyx,u,of,1,ys,zs,resx,resy,resz)
    /* Z-1 */     + trilinear_interp_vol(pos-st.yyx,u,of,1,ys,zs,resx,resy,resz)
                   )/6.f;
}

