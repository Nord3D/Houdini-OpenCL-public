/*** FIT, SMOOTH functions ****************************************************/

/*** fit, efit ***/
float 
fit( float x,
      float omin, float omax, 
      float nmin, float nmax ) {
    x = clamp(x, omin,omax);
    x -= omin;
    x *= (nmax-nmin)/(omax-omin);
    x += nmin;
    return x;
}

float
efit( float x,
       float omin, float omax, 
       float nmin, float nmax ) {
    x -= omin;
    x *= (nmax-nmin)/(omax-omin);
    x += nmin;
    return x;
}


/*** fit variants vith mad(), fma() ***/
float fit_mad(float x, float omin,float omax, float nmin,float nmax) {
    return clamp(mad(x-omin, (nmax-nmin)/(omax-omin), nmin), nmin,nmax);
}

float fit_fma(float x, float omin,float omax, float nmin,float nmax) {
    return clamp(fma(x-omin, (nmax-nmin)/(omax-omin), nmin), nmin,nmax);
}


/*** smooth function like in Houdini VEX ***/
float smooth (  float min, float max,
                float x,
                float rolloff
                ) {
    x = smoothstep(min,max, x);
    if ( rolloff == 1 ) return x;
    x = rolloff > 1 ? pow(x, rolloff) : 1.f-pow(1.f-x, 1.f/fmax(rolloff,0.f));
    return x;
}




/*** Length, distance ****************************************************************/
/*** Squared vector length ***/
float
length2( float3 v ) {
    return dot( v, v );
}






/*** HOUDINI VOLUME functions *********************************************************/

/* Assumes uniform scaled voxels, i.e. voxelsize_x == voxelsize_y == voxelsize_z      */

/* some use trilinear_interp_vol() from <interpolate.h>, which is included by default */

#define M_1_3_F .33333333f
#define M_1_6_F .16666667f

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


float  /* Mean value smooth at voxel (assumed use in iterations) */
smooth_vol_ixyz( __global const float * restrict u, 
                uint ix, uint iy, uint iz,
                uint of, uint ys, uint zs,
                uint resx, uint resy, uint resz ) {
//    ix = clamp((int)ix, (int)0, (int)(resx-1));
//    iy = clamp((int)iy, (int)0, (int)(resy-1));
//    iz = clamp((int)iz, (int)0, (int)(resz-1));
    float sC  = u[ijk2idx( ix,   iy,   iz,   of,ys,zs )];
    float sxL = u[ijk2idx( ix+1, iy,   iz,   of,ys,zs )]; 
    float sxR = u[ijk2idx( ix-1, iy,   iz,   of,ys,zs )];
    float syL = u[ijk2idx( ix,   iy+1, iz,   of,ys,zs )]; 
    float syR = u[ijk2idx( ix,   iy-1, iz,   of,ys,zs )];
    float szL = u[ijk2idx( ix,   iy,   iz+1, of,ys,zs )]; 
    float szR = u[ijk2idx( ix,   iy,   iz-1, of,ys,zs )];
    float sx = ix==0 ? sC : ix==(resx-1) ? sC : (sxL+sxR)*.5;
    float sy = iy==0 ? sC : iy==(resy-1) ? sC : (syL+syR)*.5;
    float sz = iz==0 ? sC : iz==(resz-1) ? sC : (szL+szR)*.5;
    return ( sx + sy + sz )*M_1_3_F; // /3.f;
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
                   )*M_1_6_F; // /6.f;
}





/*** VOLUME WORLD POSITION in kernell *************************************************/
#bind volume &sdf float name=0 xformtoworld

@KERNEL
{
    float3 P = (float3)(@ix,@iy,@iz);
    P += (float3).5f; /* Half-voxel compensate */
    P = mat4vec3mul(@sdf.xformtoworld, P);
    ...
}





/*** N O I S E ***********************************************************************/

/*** Simple xnoise for VDB example ***/
#include <xnoise.h>

#bind parm amp      float   val=.1
#bind parm freq     float   val=10.
          
#bind vdb &sdf      float   name=0

@KERNEL
{
    float outn; float outdndx; float outdndy; float outdndz;
    
    xnoise3d(_bound_theXNoise, @sdf.pos * @freq,
               &outn, &outdndx, &outdndy, &outdndz);
               
    @sdf.set(@sdf - outn * @amp);
}



/*** Fractal xnoise with gradient warp for VDB example ***/
#include <xnoise.h>

#bind parm amp          float   val=.1
#bind parm freq         float   val=10.
#bind parm noctaves     int     val=5
#bind parm lacunarity   float   val=2.
#bind parm roughness    float   val=.5
#bind parm gwarp        float   val=.1

#bind vdb &sdf          float   name=0

@KERNEL
{
    float4 outng = (float4)(0);
    float3 pos = @sdf.pos;
    float a = 1. - .75*@roughness;

    for(int i = 0; i < @noctaves; ++i)
    {
        float outn; float outdndx; float outdndy; float outdndz;
        xnoise3d(_bound_theXNoise, pos * @freq,
                   &outn, &outdndx, &outdndy, &outdndz);

        outng.s0 += outn    * a;
        outng.s1 += outdndx * a;
        outng.s2 += outdndy * a;
        outng.s3 += outdndz * a;
        
        pos += outng.s123*(float3)@gwarp;
        pos *= (float3)@lacunarity;
        a *= @roughness;
    }
               
    @sdf.set(@sdf - outng.s0*outng.s0 * @amp);
}





/*** Laplacian Smooth for geometry points ********************************/
/*** Thanks alexwheezy for explaining how to invoke @WRITEBACK kernel ***/
#bind point &P  float3
#bind point &P_ float3
#bind point nbs int[] name=topo:neighbours

#bind parm step

@KERNEL
{
    float3 L = 0;
    
    for(int i = 0; i < @nbs.entries; ++i)
    {
        L += @P(@nbs[i]) - @P;
    }
    L /= @nbs.entries;
    
    @P_.set(@P + L * @step);
}

@WRITEBACK
{
    @P.set(@P_);
}

