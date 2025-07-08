//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file kh.cpp
//  \brief Problem generator for KH instability
//  Sets up different initial conditions selected by flag "iprob"
//    - iprob=1 : tanh profile with a single mode perturbation
//    - iprob=2 : double tanh profile with a single mode perturbation

#include <iostream>
#include <sstream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "coordinates/adm.hpp"
#include "pgen.hpp"

void RefinementCondition(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn
//  \brief Problem Generator for KHI tests

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  user_ref_func = RefinementCondition;

  if (restart) return;
  // read problem parameters from input file
  int iprob  = pin->GetReal("problem","iprob");
  Real amp   = pin->GetReal("problem","amp");
  Real sigma = pin->GetReal("problem","sigma");
  Real vshear= pin->GetReal("problem","vshear");
  Real a_char = pin->GetOrAddReal("problem","a_char", 0.01);
  Real rho0  = pin->GetOrAddReal("problem","rho0",1.0);
  Real rho1  = pin->GetOrAddReal("problem","rho1",1.0);
  Real y0    = pin->GetOrAddReal("problem","y0",0.0);
  Real y1    = pin->GetOrAddReal("problem","y1",1.0);
  Real p_in  = pin->GetOrAddReal("problem","press",1.0);
  Real drho_rho0 = pin->GetOrAddReal("problem", "drho_rho0", 0.0);

  //user_hist_func = KHHistory;

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size = pmbp->pmb->mb_size;

  // Select either Hydro or MHD
  Real gm1, p0;
  int nfluid, nscalars;
  if (pmbp->phydro != nullptr) {
    gm1 = (pmbp->phydro->peos->eos_data.gamma) - 1.0;
    nfluid = pmbp->phydro->nhydro;
    nscalars = pmbp->phydro->nscalars;
  } else if (pmbp->pmhd != nullptr) {
    gm1 = (pmbp->pmhd->peos->eos_data.gamma) - 1.0;
    nfluid = pmbp->pmhd->nmhd;
    nscalars = pmbp->pmhd->nscalars;
  }
  if (pmbp->padm != nullptr) {
    gm1 = 1.0;
  }
  auto &w0_ = (pmbp->phydro != nullptr)? pmbp->phydro->w0 : pmbp->pmhd->w0;

  bool is_relativistic = false;
  if (pmbp->pcoord->is_special_relativistic ||
      pmbp->pcoord->is_general_relativistic ||
      pmbp->pcoord->is_dynamical_relativistic) {
    is_relativistic = true;
  }

  if (nscalars == 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "KH test requires nscalars != 0" << std::endl;
    exit(EXIT_FAILURE);
  }

  // initialize primitive variables
  par_for("pgen_kh1", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    w0_(m,IEN,k,j,i) = 20.0/gm1;
    w0_(m,IVZ,k,j,i) = 0.0;

    // Lorentz factor (needed to initializve 4-velocity in SR)
    Real u00 = 1.0;

    Real dens,pres,vx,vy,vz,scal;

    if (iprob == 1) {
      pres = 20.0;
      dens = 1.0;
      vx = -vshear*tanh(x2v/sigma);
      vy = -amp*vshear*sin(2.*M_PI*x1v)*exp( -SQR(x2v/sigma) );
      vz = 0.0;
      scal = 0.0;
      if (x2v > 0.0) scal = 1.0;
    } else if (iprob == 2) {
      // pres = 1.0;
      pres = p_in;
      vz = 0.0;
      if(x2v <= 0.0) {
        dens = rho0 - rho1*tanh((x2v+0.5)/a_char);
        vx = -vshear*tanh((x2v+0.5)/a_char);
        vy = -amp*vshear*sin(2.*M_PI*x1v)*exp( -SQR((x2v+0.5)/sigma) );
        if (is_relativistic) {
          u00 = 1.0/sqrt(1.0 - vx*vx - vy*vy);
        }
        // scal = 0.0;
        // if (x2v < -0.5) scal = 1.0;
        scal = y0 - y1*tanh((x2v+0.5)/a_char);
      } else {
        dens = rho0 + rho1*tanh((x2v-0.5)/a_char);
        vx = vshear*tanh((x2v-0.5)/a_char);
        vy = amp*vshear*sin(2.*M_PI*x1v)*exp( -SQR((x2v-0.5)/sigma) );
        if (is_relativistic) {
          u00 = 1.0/sqrt(1.0 - vx*vx - vy*vy);
        }
        // scal = 0.0;
        // if (x2v > 0.5) scal = 1.0;
        scal = y0 + y1*tanh((x2v-0.5)/a_char);
      }
    // Lecoanet test ICs
    } else if (iprob == 4) {
      pres = 10.0;
      Real a = 0.05;
      dens = 1.0 + 0.5*drho_rho0*(tanh((x2v + 0.5)/a) - tanh((x2v - 0.5)/a));
      vx = vshear*(tanh((x2v + 0.5)/a) - tanh((x2v - 0.5)/a) - 1.0);
      Real ave_sine = sin(2.*M_PI*x1v);
      if (x1v > 0.0) {
        ave_sine -= sin(2.*M_PI*(-0.5 + x1v));
      } else {
        ave_sine -= sin(2.*M_PI*(0.5 + x1v));
      }
      ave_sine /= 2.0;

      // translated x1= x - 1/2 relative to Lecoanet (2015) shifts sine function by pi
      // (half-period) and introduces U_z sign change:
      vy = -amp*ave_sine*
            (exp(-(SQR(x2v + 0.5))/(sigma*sigma)) + exp(-(SQR(x2v - 0.5))/(sigma*sigma)));
      scal = 0.5*(tanh((x2v + 0.5)/a) - tanh((x2v - 0.5)/a) + 2.0);
      vz = 0.0;
    }

    // set primitives in both newtonian and SR hydro
    w0_(m,IDN,k,j,i) = dens;
    w0_(m,IEN,k,j,i) = pres/gm1;
    w0_(m,IVX,k,j,i) = u00*vx;
    w0_(m,IVY,k,j,i) = u00*vy;
    w0_(m,IVZ,k,j,i) = u00*vz;
    // add passive scalars
    for (int n=nfluid; n<(nfluid+nscalars); ++n) {
      w0_(m,n,k,j,i) = scal;
    }
  });

  // initialize magnetic fields if MHD
  if (pmbp->pmhd != nullptr) {
    // Read magnetic field strength
    Real bx = pin->GetReal("problem","b0");
    auto &b0 = pmbp->pmhd->b0;
    auto &bcc0 = pmbp->pmhd->bcc0;
    par_for("pgen_b0", DevExeSpace(), 0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) = bx;
      b0.x2f(m,k,j,i) = 0.0;
      b0.x3f(m,k,j,i) = 0.0;
      if (i==ie) b0.x1f(m,k,j,i+1) = bx;
      if (j==je) b0.x2f(m,k,j+1,i) = 0.0;
      if (k==ke) b0.x3f(m,k+1,j,i) = 0.0;
      bcc0(m,IBX,k,j,i) = bx;
      bcc0(m,IBY,k,j,i) = 0.0;
      bcc0(m,IBZ,k,j,i) = 0.0;
    });
  }

  // Initialize the ADM variables if enabled
  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  // Convert primitives to conserved
  if (pmbp->padm == nullptr) {
    if (pmbp->phydro != nullptr) {
      auto &u0_ = pmbp->phydro->u0;
      pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
    } else if (pmbp->pmhd != nullptr) {
      auto &u0_ = pmbp->pmhd->u0;
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
    }
  }

  return;
}

// Custom AMR refinement criteria
void RefinementCondition(MeshBlockPack *pmbp) {
  // capture variables for kernels
  Mesh *pm = pmbp->pmesh;
  auto &indcs = pm->mb_indcs;
  auto &multi_d = pm->multi_d;
  auto &three_d = pm->three_d;
  int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
  int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
  int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;

  // check (on device) Hydro/MHD refinement conditions over all MeshBlocks
  auto refine_flag_ = pm->pmr->refine_flag;
  int nmb = pmbp->nmb_thispack;
  int mbs = pm->gids_eachrank[global_variable::my_rank];
  // get curve_threshold from MeshRefinement via mesh pointer
  const Real curve_threshold = pm->pmr->GetCurveThreshold();

  // check if hydro or mhd is active for this MeshBlockPack
  if ((pmbp->phydro != nullptr) || (pmbp->pmhd != nullptr)) {
    // get conserved vairables and prinitive variables (see athena.hpp for array indices)
    auto &w0 = (pmbp->phydro != nullptr) ? pmbp->phydro->w0 : pmbp->pmhd->w0;

    // get grid cell size in relevant directions 
    const Real dx1 = pm->mesh_size.dx1;
    const Real dx2 = pm->mesh_size.dx2;

    // run each MeshBlock in the MeshBlockPack in parrallel 
    par_for_outer("ConsRefineCond", DevExeSpace(), 0, 0, 0, nmb - 1,
      KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
        if (curve_threshold != 0.0) {
          Real curve_indicator = 0.0;

          // loop over all of the cells in the MeshBlock in parallel
          Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
            [=](const int idx, Real &max_curve) {
              int k = (idx) / nji;
              int j = (idx - k * nji) / nx1;
              int i = (idx - k * nji - j * nx1) + is;
              j += js;
              k += ks;

              Real rho = w0(m, IDN, k, j, i);
              Real pres = w0(m, IPR, k, j, i);

              // Second derivative in x 
              Real d2dx_rho, d2dx_pres;
              if (is <= i && i <= ie) {
                // central difference in x
                d2dx_rho = (-2.0 * rho + w0(m, IDN, k, j, i + 1) + w0(m, IDN, k, j, i - 1)) / SQR(dx1);
                d2dx_pres = (-2.0 * pres + w0(m, IPR, k, j, i + 1) + w0(m, IPR, k, j, i - 1)) / SQR(dx1);
              }

              // Second derivative in y 
              Real d2dy_rho, d2dy_pres;
              if (js <= j && j <= je) {
                // central difference in y
                d2dy_rho = (-2.0 * rho + w0(m, IDN, k, j + 1, i) + w0(m, IDN, k, j - 1, i)) / SQR(dx2);
                d2dy_pres = (-2.0 * pres + w0(m, IPR, k, j + 1, i) + w0(m, IPR, k, j - 1, i)) / SQR(dx2);
              }

              // Compute curvature indicator (Matsumoto 2007 eq. 72)
              Real curvature_rho = fabs(SQR(dx1) * d2dx_rho + SQR(dx2) * d2dy_rho) / rho;
              Real curvature_pres = fabs(SQR(dx1) * d2dx_pres + SQR(dx2) * d2dy_pres) / pres;

              max_curve = fmax(max_curve, fmax(curvature_rho, curvature_pres));
              // max-curve is the local curvature value for the current cell in the parrallell loop
              // Kokkos::Max find the maximum curvature over the entire meshblock
            },Kokkos::Max<Real>(curve_indicator));

            // check if the curve_indicator exceeds the threshold
            if (curve_indicator > curve_threshold) {refine_flag_.d_view(m+mbs) = 1;}
            if (curve_indicator < curve_threshold) {refine_flag_.d_view(m+mbs) = -1;}
        }
      }
    );
  }
}