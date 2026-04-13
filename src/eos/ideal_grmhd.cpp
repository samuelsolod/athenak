//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file ideal_grmhd.cpp
//! \brief derived class that implements ideal gas EOS in general relativistic mhd

#include <float.h>

#include "athena.hpp"
#include "mhd/mhd.hpp"
#include "eos.hpp"
#include "eos/ideal_c2p_mhd.hpp"

#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls EOS base class constructor

IdealGRMHD::IdealGRMHD(MeshBlockPack *pp, ParameterInput *pin) :
    EquationOfState("mhd", pp, pin) {
  eos_data.is_ideal = true;
  eos_data.gamma = pin->GetReal("mhd","gamma");
  eos_data.iso_cs = 0.0;
  eos_data.use_e = true;  // ideal gas EOS always uses internal energy
  eos_data.use_t = false;
  eos_data.gamma_max = pin->GetOrAddReal("mhd","gamma_max",(FLT_MAX));  // gamma ceiling
  eos_data.sigma_max = pin->GetOrAddReal("mhd","sigma_max",(FLT_MAX));  // sigma ceiling
}

//----------------------------------------------------------------------------------------
//! \fn void ConsToPrim()
//! \brief Converts conserved into primitive variables.
//! Operates over range of cells given in argument list.

void IdealGRMHD::ConsToPrim(DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                            DvceArray5D<Real> &prim, DvceArray5D<Real> &bcc,
                            const bool only_testfloors,
                            const int il, const int iu, const int jl, const int ju,
                            const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  auto &fofc_ = pmy_pack->pmhd->fofc;
  auto eos = eos_data;
  // Real gm1 = eos_data.gamma - 1.0;

  // YK: for sigma capping
  const Real eos_gamma = eos_data.gamma;
  const Real gm1 = eos_gamma - 1.0;
  const Real sigma_max = eos_data.sigma_max;
  const Real efloor = eos.pfloor / gm1;

  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  auto &use_excise = pmy_pack->pcoord->coord_data.bh_excise;
  auto &excision_floor_ = pmy_pack->pcoord->excision_floor;
  auto &excision_flux_ = pmy_pack->pcoord->excision_flux;
  auto &dexcise_ = pmy_pack->pcoord->coord_data.dexcise;
  auto &pexcise_ = pmy_pack->pcoord->coord_data.pexcise;

  const int ni   = (iu - il + 1);
  const int nji  = (ju - jl + 1)*ni;
  const int nkji = (ku - kl + 1)*nji;
  const int nmkji = nmb*nkji;

  int nfloord_=0, nfloore_=0, nceilv_=0, nfail_=0, maxit_=0;
  Kokkos::parallel_reduce("grmhd_c2p",Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, int &sumd, int &sume, int &sumv, int &sumf, int &max_it) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/ni;
    int i = (idx - m*nkji - k*nji - j*ni) + il;
    j += jl;
    k += kl;

    // load single state conserved variables
    MHDCons1D u;
    u.d  = cons(m,IDN,k,j,i);
    u.mx = cons(m,IM1,k,j,i);
    u.my = cons(m,IM2,k,j,i);
    u.mz = cons(m,IM3,k,j,i);
    u.e  = cons(m,IEN,k,j,i);

    // load cell-centered fields into conserved state
    // use input CC fields if only testing floors with FOFC
    if (only_testfloors) {
      u.bx = bcc(m,IBX,k,j,i);
      u.by = bcc(m,IBY,k,j,i);
      u.bz = bcc(m,IBZ,k,j,i);
    // else use simple linear average of face-centered fields
    } else {
      u.bx = 0.5*(b.x1f(m,k,j,i) + b.x1f(m,k,j,i+1));
      u.by = 0.5*(b.x2f(m,k,j,i) + b.x2f(m,k,j+1,i));
      u.bz = 0.5*(b.x3f(m,k,j,i) + b.x3f(m,k+1,j,i));
    }

    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    HydPrim1D w;
    bool dfloor_used=false, efloor_used=false;
    bool vceiling_used=false, c2p_failure=false;
    int iter_used=0;

    // Only execute cons2prim if outside excised region
    bool excised = false;
    if (use_excise) {
      if (excision_floor_(m,k,j,i)) {
        w.d = dexcise_;
        w.vx = 0.0;
        w.vy = 0.0;
        w.vz = 0.0;
        w.e = pexcise_/gm1;
        excised = true;
      }
      if (only_testfloors) {
        if (excision_flux_(m,k,j,i)) {
          excised = true;
        }
      }
    }

    if (!(excised)) {
      // calculate SR conserved quantities
      MHDCons1D u_sr;
      Real s2, b2, rpar;
      TransformToSRMHD(u,glower,gupper,s2,b2,rpar,u_sr);

      // call c2p function
      // (inline function in ideal_c2p_mhd.hpp file)
      SingleC2P_IdealSRMHD(u_sr, eos, s2, b2, rpar, w,
                           dfloor_used, efloor_used, c2p_failure, iter_used);

      //
      // @YK : Drift floor : limiting sigma, while keeping parallel momentum
      // constant
      //
      //  - inject density for sigma (ceiling the magnetization)
      //  - rescale v_parallel to preserve parallel momentum
      //

      // Compute lorentz factor
      Real tmp = glower[1][1]*SQR(w.vx)
               + glower[2][2]*SQR(w.vy)
               + glower[3][3]*SQR(w.vz)
               + 2.0*glower[1][2]*w.vx*w.vy + 2.0*glower[1][3]*w.vx*w.vz
               + 2.0*glower[2][3]*w.vy*w.vz;

      Real lorentz_W = sqrt(1.0 + tmp);
      //
      // @YK : Compute sigma_cold
      //
      const Real alpha = sqrt(-1.0 / gupper[0][0]);
      const Real u0 = lorentz_W / alpha;
      const Real alpha_W = alpha * lorentz_W;

      const Real u1 = w.vx - alpha_W * gupper[0][1];
      const Real u2 = w.vy - alpha_W * gupper[0][2];
      const Real u3 = w.vz - alpha_W * gupper[0][3];

      const Real u_1 = glower[1][0] * u0 + glower[1][1] * u1 +
                       glower[1][2] * u2 + glower[1][3] * u3;
      const Real u_2 = glower[2][0] * u0 + glower[2][1] * u1 +
                       glower[2][2] * u2 + glower[2][3] * u3;
      const Real u_3 = glower[3][0] * u0 + glower[3][1] * u1 +
                       glower[3][2] * u2 + glower[3][3] * u3;

      // Calculate comoving magnetic field (b0 moved to outer scope for reuse)
      const Real b0 = u_1 * u.bx + u_2 * u.by + u_3 * u.bz;
      Real b_sq;
      {
        const Real inv_u0 = 1.0 / u0;
        const Real b1 = (u.bx + b0 * u1) * inv_u0;
        const Real b2 = (u.by + b0 * u2) * inv_u0;
        const Real b3 = (u.bz + b0 * u3) * inv_u0;

        // lower vector indices
        const Real b_0 = glower[0][0] * b0 + glower[0][1] * b1 +
                         glower[0][2] * b2 + glower[0][3] * b3;
        const Real b_1 = glower[1][0] * b0 + glower[1][1] * b1 +
                         glower[1][2] * b2 + glower[1][3] * b3;
        const Real b_2 = glower[2][0] * b0 + glower[2][1] * b1 +
                         glower[2][2] * b2 + glower[2][3] * b3;
        const Real b_3 = glower[3][0] * b0 + glower[3][1] * b1 +
                         glower[3][2] * b2 + glower[3][3] * b3;

        // b^2
        b_sq = b0 * b_0 + b1 * b_1 + b2 * b_2 + b3 * b_3;
      }

      const Real gamma_efloor = eos_gamma * efloor;
      Real rhoh = w.d + (eos_gamma * w.e);
      const Real rhoh_cold = w.d + gamma_efloor;
      const Real sigma_cold = b_sq / rhoh_cold;

      if (sigma_cold > sigma_max) {
        dfloor_used = true;

        // Compute v_parallel and Bnorm
        const Real B2 =
            glower[1][1] * SQR(u.bx) + glower[2][2] * SQR(u.by) +
            glower[3][3] * SQR(u.bz) + 2.0 * glower[1][2] * u.bx * u.by +
            2.0 * glower[1][3] * u.bx * u.bz + 2.0 * glower[2][3] * u.by * u.bz;
        const Real Bnorm = sqrt(B2);

        // v_par = B^i v_i / B
        // Note u_i = W v_i
        // Reusing b0 which is exactly (u_1 * u.bx + u_2 * u.by + u_3 * u.bz)
        const Real v_par = b0 / (lorentz_W * Bnorm);

        // Store the constant
        const Real parallel_momentum_over_B = rhoh * SQR(lorentz_W) * v_par;

        // Precompute normalized B components and W inverse
        const Real inv_W = 1.0 / lorentz_W;
        const Real bx_hat = u.bx / Bnorm;
        const Real by_hat = u.by / Bnorm;
        const Real bz_hat = u.bz / Bnorm;

        // Compute drift velocity
        const Real vd1 = w.vx * inv_W - v_par * bx_hat;
        const Real vd2 = w.vy * inv_W - v_par * by_hat;
        const Real vd3 = w.vz * inv_W - v_par * bz_hat;

        Real vd_sq = glower[1][1] * SQR(vd1) + glower[2][2] * SQR(vd2) +
                     glower[3][3] * SQR(vd3) + 2.0 * glower[1][2] * vd1 * vd2 +
                     2.0 * glower[1][3] * vd1 * vd3 +
                     2.0 * glower[2][3] * vd2 * vd3;
        vd_sq = fmax(0.0, fmin(vd_sq, 0.99999999));

        // then inject floors
        rhoh = b_sq / sigma_max;
        w.d = rhoh - gamma_efloor;
        w.e = efloor;

        const Real z = 2.0 * parallel_momentum_over_B / rhoh;
        const Real one_minus_vd_sq = 1.0 - vd_sq;
        const Real v_par_updated =
            z * one_minus_vd_sq / (1.0 + sqrt(1.0 + SQR(z) * one_minus_vd_sq));

        // Recompute W and Wv^i
        lorentz_W = 1.0 / (one_minus_vd_sq - SQR(v_par_updated));
        w.vx = lorentz_W * (vd1 + v_par_updated * bx_hat);
        w.vy = lorentz_W * (vd2 + v_par_updated * by_hat);
        w.vz = lorentz_W * (vd3 + v_par_updated * bz_hat);
      }

      // @YK : Apply Lorentz factor capping
      // Rescale (W v^i), maintaining its direction
      if (lorentz_W > eos.gamma_max) {
        vceiling_used = true;
        const Real factor =
            sqrt((SQR(eos.gamma_max) - 1.0) / (SQR(lorentz_W) - 1.0));
        w.vx *= factor;
        w.vy *= factor;
        w.vz *= factor;
      }
    }

    // set FOFC flag and quit loop if this function called only to check floors
    if (only_testfloors) {
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure) {
        fofc_(m,k,j,i) = true;
        sumd++;  // use dfloor as counter for when either is true
      }
    } else {
      if (dfloor_used) {sumd++;}
      if (efloor_used) {sume++;}
      if (vceiling_used) {sumv++;}
      if (c2p_failure) {sumf++;}
      max_it = (iter_used > max_it) ? iter_used : max_it;

      // store primitive state in 3D array
      prim(m,IDN,k,j,i) = w.d;
      prim(m,IVX,k,j,i) = w.vx;
      prim(m,IVY,k,j,i) = w.vy;
      prim(m,IVZ,k,j,i) = w.vz;
      prim(m,IEN,k,j,i) = w.e;

      // store cell-centered fields in 3D array
      bcc(m,IBX,k,j,i) = u.bx;
      bcc(m,IBY,k,j,i) = u.by;
      bcc(m,IBZ,k,j,i) = u.bz;

      // reset conserved variables if floor, ceiling, failure, or excision encountered
      if (dfloor_used || efloor_used || vceiling_used || c2p_failure || excised) {
        MHDPrim1D w_in;
        w_in.d  = w.d;
        w_in.vx = w.vx;
        w_in.vy = w.vy;
        w_in.vz = w.vz;
        w_in.e  = w.e;
        w_in.bx = u.bx;
        w_in.by = u.by;
        w_in.bz = u.bz;

        HydCons1D u_out;
        SingleP2C_IdealGRMHD(glower, gupper, w_in, eos.gamma, u_out);
        cons(m,IDN,k,j,i) = u_out.d;
        cons(m,IM1,k,j,i) = u_out.mx;
        cons(m,IM2,k,j,i) = u_out.my;
        cons(m,IM3,k,j,i) = u_out.mz;
        cons(m,IEN,k,j,i) = u_out.e;
        u.d = u_out.d;  // (needed if there are scalars below)
      }

      // convert scalars (if any)
      for (int n=nmhd; n<(nmhd+nscal); ++n) {
        prim(m,n,k,j,i) = cons(m,n,k,j,i)/u.d;
      }
    }
  }, Kokkos::Sum<int>(nfloord_), Kokkos::Sum<int>(nfloore_), Kokkos::Sum<int>(nceilv_),
     Kokkos::Sum<int>(nfail_), Kokkos::Max<int>(maxit_));

  // store appropriate counters
  if (only_testfloors) {
    pmy_pack->pmesh->ecounter.nfofc += nfloord_;
  } else {
    pmy_pack->pmesh->ecounter.neos_dfloor += nfloord_;
    pmy_pack->pmesh->ecounter.neos_efloor += nfloore_;
    pmy_pack->pmesh->ecounter.neos_vceil  += nceilv_;
    pmy_pack->pmesh->ecounter.neos_fail   += nfail_;
    pmy_pack->pmesh->ecounter.maxit_c2p = maxit_;
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void PrimToCons()
//! \brief Converts primitive into conserved variables.  Operates over range of cells
//! given in argument list.

void IdealGRMHD::PrimToCons(const DvceArray5D<Real> &prim, const DvceArray5D<Real> &bcc,
                            DvceArray5D<Real> &cons, const int il, const int iu,
                            const int jl, const int ju, const int kl, const int ku) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int &is = indcs.is, &js = indcs.js, &ks = indcs.ks;
  auto &size = pmy_pack->pmb->mb_size;
  auto &flat = pmy_pack->pcoord->coord_data.is_minkowski;
  auto &spin = pmy_pack->pcoord->coord_data.bh_spin;
  int &nmhd  = pmy_pack->pmhd->nmhd;
  int &nscal = pmy_pack->pmhd->nscalars;
  int &nmb = pmy_pack->nmb_thispack;
  Real &gamma = eos_data.gamma;

  par_for("grmhd_p2c", DevExeSpace(), 0, (nmb-1), kl, ku, jl, ju, il, iu,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    // Extract components of metric
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, flat, spin, glower, gupper);

    // Load single state of primitive variables
    MHDPrim1D w;
    w.d  = prim(m,IDN,k,j,i);
    w.vx = prim(m,IVX,k,j,i);
    w.vy = prim(m,IVY,k,j,i);
    w.vz = prim(m,IVZ,k,j,i);
    w.e  = prim(m,IEN,k,j,i);

    // load cell-centered fields into primitive state
    w.bx = bcc(m,IBX,k,j,i);
    w.by = bcc(m,IBY,k,j,i);
    w.bz = bcc(m,IBZ,k,j,i);

    // call p2c function
    HydCons1D u;
    SingleP2C_IdealGRMHD(glower, gupper, w, gamma, u);

    // store conserved quantities in 3D array
    cons(m,IDN,k,j,i) = u.d;
    cons(m,IM1,k,j,i) = u.mx;
    cons(m,IM2,k,j,i) = u.my;
    cons(m,IM3,k,j,i) = u.mz;
    cons(m,IEN,k,j,i) = u.e;

    // convert scalars (if any)
    for (int n=nmhd; n<(nmhd+nscal); ++n) {
      cons(m,n,k,j,i) = u.d*prim(m,n,k,j,i);
    }
  });

  return;
}
