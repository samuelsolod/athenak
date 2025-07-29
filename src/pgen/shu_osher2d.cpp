//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shu_osher.cpp
//  \brief Problem generator for Shu-Osher shocktube test, involving interaction of a
//   Mach 3 shock with a sine wave density distribution.
//
// REFERENCE: C.W. Shu & S. Osher, "Efficient implementation of essentially
//   non-oscillatory shock-capturing schemes, II", JCP, 83, 32 (1998)

// C++ headers
#include <cmath>  // sin()
#include <iostream> // cout

// Athena++ headers
#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen.hpp"
#include "globals.hpp"

// void RefinementCondition(MeshBlockPack *pmbp);

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//  \brief Shu-Osher test problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // user_ref_func = RefinementCondition;
  
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Shu-Osher test can only be run in Hydro, but no <hydro> block "
              << "in input file" << std::endl;
    exit(EXIT_FAILURE);
  }

  // setup problem parameters
  Real dl = 3.857143;
  Real pl = 10.33333;
  Real ul = 2.629369;
  Real vl = 0.0;
  Real wl = 0.0;

  // capture variables for kernel
  Real gm1 = pmbp->phydro->peos->eos_data.gamma - 1.0;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is = indcs.is; int &ie = indcs.ie;
  int &js = indcs.js; int &je = indcs.je;
  int &ks = indcs.ks; int &ke = indcs.ke;
  auto &size = pmbp->pmb->mb_size;
  auto &u0 = pmbp->phydro->u0;

  par_for("pgen_shock1", DevExeSpace(),0,(pmbp->nmb_thispack-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m,int k, int j, int i) {
    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    int nx2 = indcs.nx2;
    Real x2v = CellCenterX(j-js, nx2, x2min, x2max);

    if (x1v < -0.8) {
      u0(m,IDN,k,j,i) = dl;
      u0(m,IM1,k,j,i) = ul*dl;
      u0(m,IM2,k,j,i) = vl*dl;
      u0(m,IM3,k,j,i) = wl*dl;
      u0(m,IEN,k,j,i) = pl/gm1 + 0.5*dl*(ul*ul + vl*vl + wl*wl);
    } else {
      u0(m,IDN,k,j,i) = 1.0 + 0.2*std::sin(5.0*M_PI*(x1v));
      u0(m,IM1,k,j,i) = 0.0;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
      u0(m,IEN,k,j,i) = 1.0/gm1;
    }
  });

  return;
}
//---------------------------------------------------------------------------------------------------------------------------------
// Custom AMR refinement criteria based on PPAO for GRMHD (Deppe 2023)
// void RefinementCondition(MeshBlockPack *pmbp) {
//   // capture variables for kernels
//   Mesh *pm = pmbp->pmesh;
//   auto &indcs = pm->mb_indcs;
//   auto &multi_d = pm->multi_d;
//   auto &three_d = pm->three_d;
//   int is = indcs.is, ie = indcs.ie, nx1 = indcs.nx1;
//   int js = indcs.js, je = indcs.je, nx2 = indcs.nx2;
//   int ks = indcs.ks, ke = indcs.ke, nx3 = indcs.nx3;
//   const int nkji = nx3 * nx2 * nx1;
//   const int nji = nx2 * nx1;

//   // check (on device) Hydro/MHD refinement conditions over all MeshBlocks
//   auto refine_flag_ = pm->pmr->refine_flag;
//   int nmb = pmbp->nmb_thispack;
//   int mbs = pm->gids_eachrank[global_variable::my_rank];

//   // get preferred stencil order from MeshRefinement via mesh pointer
//   const int stencil_ = pm->pmr->GetStencilOrder();
//   const Real alpha_refine_ = pm->pmr->GetAlphaRefine();
//   const Real alpha_coarsen_ = pm->pmr->GetAlphaCoarsen();
//   const int variable = pm->pmr->GetVariable();

//   // check if hydro or mhd is active for this MeshBlockPack
//   if ((pmbp->phydro != nullptr)) {
//     // get conserved vairables and prinitive variables (see athena.hpp for array indices)
//     auto &w0 = pmbp->phydro->w0;

//     // get grid cell size in relevant directions 
//     const Real dx1 = pm->mesh_size.dx1;
//     const Real dx2 = pm->mesh_size.dx2;

//     // run each MeshBlock in the MeshBlockPack in parrallel 
//     par_for_outer("ConsRefineCond", DevExeSpace(), 0, 0, 0, nmb - 1,
//       KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
//         Real cN = 0.0;
//         Real sum_cN = 0.0;
//         // loop over all of the cells in the MeshBlock in parallel
//         Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
//           [=](const int idx, Real &max_cN, Real &max_sum_cN) {
//             int k = (idx) / nji;
//             int j = (idx - k * nji) / nx1;
//             int i = (idx - k * nji - j * nx1) + is;
//             j += js;
//             k += ks;

//             if (stencil_ == 3) {
//               // solution values for cells of interest for 3-point stencil
//               Real u1, u0x, u2x, u0y, u2y;
//               if (variable == 1) {
//                 u1 = w0(m, IDN, k, j, i);

//                 u0x = w0(m, IDN, k, j, i - 1);
//                 u2x = w0(m, IDN, k, j, i + 1); 

//                 u0y = w0(m, IDN, k, j - 1, i);
//                 u2y = w0(m, IDN, k, j + 1, i); 
//               }
//               if (variable == 2) {
//                 u1 = std::sqrt(SQR(w0(m, IVX, k, j, i)) + SQR(w0(m, IVY, k, j, i)));

//                 u0x = std::sqrt(SQR(w0(m, IVX, k, j, i-1)) + SQR(w0(m, IVY, k, j, i-1)));
//                 u2x = std::sqrt(SQR(w0(m, IVX, k, j, i+1)) + SQR(w0(m, IVY, k, j, i+1))); 

//                 u0y = std::sqrt(SQR(w0(m, IVX, k, j-1, i)) + SQR(w0(m, IVY, k, j-1, i)));
//                 u2y = std::sqrt(SQR(w0(m, IVX, k, j+1, i)) + SQR(w0(m, IVY, k, j+1, i))); 
//               }

//               // create array of solution values and initialize modal coeffiecent array
//               Real ux[3], uy[3], cx[3], cy[3]; 
//               ux[0] = u0x; ux[1] = u1; ux[2] = u2x;
//               uy[0] = u0y; uy[1] = u1; uy[2] = u2y;

//               for (int ii = 0; ii<3; ii++) {cx[ii] = 0.0;}
//               for (int ii = 0; ii<3; ii++) {cy[ii] = 0.0;}

//               // 3x3 Legendre coefficent matrix A
//               const Real A[3][3] = {
//                 {3.0/8.0,     1.0/4.0,      3.0/8.0},
//                 {-3.0/4.0,    0.0,          3.0/4.0},
//                 {3.0/4.0,     -3.0/2.0,     3.0/4.0}
//               };

//               // A * u = c
//               for (int row = 0; row < 3; ++row) {
//                 for (int col = 0; col < 3; ++col) {
//                   cx[row] += A[row][col] * ux[col];
//                   cy[row] += A[row][col] * uy[col];
//                 }
//               }

//               // compute (c_N)^2 and sum_0^N((c_n)^2)... see equation (9) in Deppe 2023
//               Real kappa3x = 0.0;
//               Real kappa3y = 0.0;
//               for (int jj = 0; jj < 3; ++jj) {
//                 kappa3x += cx[jj] * cx[jj] / (2.0 * jj + 1);
//                 kappa3y += cy[jj] * cy[jj] / (2.0 * jj + 1);
//               }
//               Real kappa3x_hat = cx[2] * cx[2] / 5.0;
//               Real kappa3y_hat = cy[2] * cy[2] / 5.0;

//               Real kappa3 = fmax(kappa3x, kappa3y);
//               Real kappa3_hat = fmax(kappa3x_hat, kappa3y_hat);

//               // extract kappa3_hat and kappa3 from parallel reduction
//               max_cN = fmax(kappa3_hat, max_cN);
//               max_sum_cN = fmax(kappa3, max_sum_cN);
//             }

//             if (stencil_ == 5) {
//               Real u2, u0x, u1x, u3x, u4x, u0y, u1y, u3y, u4y;
//               if (variable == 1) {
//                 u2 = w0(m, IDN, k, j, i);

//                 u0x = w0(m, IDN, k, j, i - 2);
//                 u1x = w0(m, IDN, k, j, i - 1);
//                 u3x = w0(m, IDN, k, j, i + 1);
//                 u4x = w0(m, IDN, k, j, i + 2);

//                 u0y = w0(m, IDN, k, j - 2, i);
//                 u1y = w0(m, IDN, k, j - 1, i);
//                 u3y = w0(m, IDN, k, j + 1, i);
//                 u4y = w0(m, IDN, k, j + 2, i);
//               }
//               if (variable == 2) {
//                 u2 = std::sqrt(SQR(w0(m, IVX, k, j, i)) + SQR(w0(m, IVY, k, j, i)));

//                 u0x = std::sqrt(SQR(w0(m, IVX, k, j, i - 2)) + SQR(w0(m, IVY, k, j, i - 2)));
//                 u1x = std::sqrt(SQR(w0(m, IVX, k, j, i - 1)) + SQR(w0(m, IVY, k, j, i - 1)));
//                 u3x = std::sqrt(SQR(w0(m, IVX, k, j, i + 1)) + SQR(w0(m, IVY, k, j, i + 1)));
//                 u4x = std::sqrt(SQR(w0(m, IVX, k, j, i + 2)) + SQR(w0(m, IVY, k, j, i + 2)));

//                 u0y = std::sqrt(SQR(w0(m, IVX, k, j - 2, i)) + SQR(w0(m, IVY, k, j - 2,i)));
//                 u1y = std::sqrt(SQR(w0(m, IVX, k,j - 1,i)) + SQR(w0(m, IVY,k,j - 1,i)));
//                 u3y = std::sqrt(SQR(w0(m, IVX,k,j + 1,i)) + SQR(w0(m, IVY,k,j + 1,i)));
//                 u4y = std::sqrt(SQR(w0(m, IVX,k,j + 2,i)) + SQR(w0(m, IVY,k,j + 2,i)));
//               }

//               Real ux[5], uy[5], cx[5], cy[5]; 
//               ux[0] = u0x; ux[1] = u1x; ux[2] = u2; ux[3] = u3x; ux[4] = u4x;
//               uy[0] = u0y; uy[1] = u1y; uy[2] = u2; uy[3] = u3y; uy[4] = u4y;

//               for (int kk = 0; kk<5; kk++) {cx[kk] = 0.0;}
//               for (int kk = 0; kk<5; kk++) {cy[kk] = 0.0;}

//               const Real A[5][5] = {
//                   {275.0/115.0,     25.0/288.0,     67.0/192.0,     25.0/288.0,     275.0/1152.0},
//                   {-55.0/96.0,      -5.0/48.0,      0.0,            5.0/48.0,       55.0/96.0},
//                   {1525.0/2016.0,   -475.0/504.0,   125.0/336.0,    -475.0/504.0,   1525.0/2016.0},
//                   {-25.0/48.0,      25.0/24.0,      0.0,           -25.0/24.0,      25.0/48.0},
//                   {125.0/336.0,     -125.0/84.0,    125.0/56.0,     -125.0/84.0,    125.0/336.0}
//               };

//               for (int row = 0; row < 5; ++row) {
//                 for (int col = 0; col < 5; ++col) {
//                   cx[row] += A[row][col] * ux[col];
//                   cy[row] += A[row][col] * uy[col];
//                 }
//               }

//               Real kappa3x = 0.0;
//               Real kappa3y = 0.0;
//               for (int jj = 0; jj < 5; ++jj) {
//                 kappa3x += cx[jj] * cx[jj] / (2.0 * jj + 1);
//                 kappa3y += cy[jj] * cy[jj] / (2.0 * jj + 1);
//               }
//               Real kappa3x_hat = cx[4] * cx[4] / 9.0;
//               Real kappa3y_hat = cy[4] * cy[4] / 9.0;

//               Real kappa3 = fmax(kappa3x, kappa3y);
//               Real kappa3_hat = fmax(kappa3x_hat, kappa3y_hat);

//               max_cN = fmax(kappa3_hat, max_cN);
//               max_sum_cN = fmax(kappa3, max_sum_cN);
//             }
//           // Kokkos::Max finds the maximum values over the entire meshblock
//           }, Kokkos::Max<Real>(cN), Kokkos::Max<Real>(sum_cN));

//         // check if the Nth degree power exceeds the sum of powers

//         if (stencil_ == 3) {
//           Real N = 2.0;
//           Real threshold_refine = pow(N, 2.0 * alpha_refine_);
//           Real threshold_coarsen = pow(N, 2.0 * alpha_coarsen_);

//           if (cN * threshold_refine > sum_cN) {
//             refine_flag_.d_view(m + mbs) = 1;
//           }
//           if (cN * threshold_coarsen < sum_cN) {
//             refine_flag_.d_view(m + mbs) = -1;
//           }
//         }

//         if (stencil_ == 5) {
//           Real N = 4.0;
//           Real threshold_refine = pow(N, 2.0 * alpha_refine_);
//           Real threshold_coarsen = pow(N, 2.0 * alpha_coarsen_);

//           if (cN * threshold_refine > sum_cN) {
//             refine_flag_.d_view(m + mbs) = 1;
//           }
//           if (cN * threshold_coarsen < sum_cN) {
//             refine_flag_.d_view(m + mbs) = -1;
//           }
//         }
//       });
//   }
// }

// // ---------------------------------------------------------------------------------------------------------------------------------
