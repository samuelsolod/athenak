//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file refinement_criteria.cpp
//! \brief Implements constructor and functions in RefinementCriteria class.

#include <iostream>
#include <algorithm> // max
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh.hpp"
#include "mesh_refinement.hpp"

#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "radiation/radiation.hpp"
#include "refinement_criteria.hpp"
#include "utils/utils.hpp"

//----------------------------------------------------------------------------------------
// RefinementCriteria constructor:

RefinementCriteria::RefinementCriteria(Mesh *pm, ParameterInput *pin) :
    ncriteria(0),
    nderived(0),
    pmy_mesh(pm),
    dvars("derived_ref_vars",1,1,1,1,1) {
  // cycle through ParameterInput list and read each <amr_criterion> block
  for (auto it = pin->block.begin(); it != pin->block.end(); ++it) {
    if (it->block_name.compare(0, 13, "amr_criterion") == 0) {
      RefCritData rcrit0;
      std::string method = pin->GetString(it->block_name, "method");
      if (method.compare("min_max") == 0) {
        rcrit0.rmethod = RefCritMethod::min_max;
      } else if (method.compare("slope") == 0) {
        rcrit0.rmethod = RefCritMethod::slope;
      } else if (method.compare("second_deriv") == 0) {
        rcrit0.rmethod = RefCritMethod::second_deriv;
      } else if (method.compare("location") == 0) {
        rcrit0.rmethod = RefCritMethod::location;
      } else if (method.compare("spectral_norm") == 0) {
        rcrit0.rmethod = RefCritMethod::spectral_norm;
      } else if (method.compare("user") == 0) {
        rcrit0.rmethod = RefCritMethod::user;
      } else {
        std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
        Kokkos::abort("Unknown refinement criterion");
      }
      // read refinement variable only when needed
      if ((method.compare("location") != 0) &&
          (method.compare("spectral_norm") != 0) &&
          (method.compare("user") != 0)) {
        rcrit0.rvariable = pin->GetString(it->block_name, "variable");
      }
      rcrit0.rvalue_min = pin->GetOrAddReal(it->block_name,"value_min",(-FLT_MAX));
      rcrit0.rvalue_max = pin->GetOrAddReal(it->block_name,"value_max", (FLT_MAX));
      rcrit0.rloc_x1  = pin->GetOrAddReal(it->block_name,"location_x1", 0.0);
      rcrit0.rloc_x2  = pin->GetOrAddReal(it->block_name,"location_x2", 0.0);
      rcrit0.rloc_x3  = pin->GetOrAddReal(it->block_name,"location_x3", 0.0);
      rcrit0.rloc_rad = pin->GetOrAddReal(it->block_name,"location_rad", 0.0);

      // @yk : spectral norm method
      const bool use_error_policy_sum = pin->GetOrAddBoolean(
          it->block_name, "use_sum_for_error_policy", false);
      rcrit0.spectral_norm_error_policy = use_error_policy_sum
                                              ? error_policy_for_multi_dim::sum
                                              : error_policy_for_multi_dim::max;
      rcrit0.spectral_norm_alpha_refine =
          pin->GetOrAddReal(it->block_name, "alpha_refine", 4.0);
      rcrit0.spectral_norm_alpha_coarsen =
          pin->GetOrAddReal(it->block_name, "alpha_coarsen", 5.0);

      rcrit0.dfloor = pin->GetOrAddReal(it->block_name, "dfloor", 1.0e-15);
      rcrit0.efloor = pin->GetOrAddReal(it->block_name, "efloor", 1.0e-15);

      rcrit0.monitor_momentum =
          pin->GetOrAddBoolean(it->block_name, "monitor_momentum", false);
      rcrit0.monitor_energy =
          pin->GetOrAddBoolean(it->block_name, "monitor_energy", false);

      rcrit0.use_primitives =
          pin->GetOrAddBoolean(it->block_name, "use_primitives", false);

      rcrit.emplace_back(rcrit0);
    }
  }
  ncriteria = rcrit.size();

  // Error if there were no <amr_criterion> blocks
  if (ncriteria==0) {
    std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
    Kokkos::abort("No <amr_criterion> blocks were found in input file");
  }

  // Error if class containing variable requested has not been initialized
  for (auto it = rcrit.begin(); it != rcrit.end(); ++it) {
    if ((it->rvariable.compare(0, 5, "hydro") == 0) &&
        (pm->pmb_pack->phydro == nullptr)) {
      std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
      Kokkos::abort("Hydro refinement variable used but <hydro> not defined");
    }
    if ((it->rvariable.compare(0, 3, "mhd") == 0) &&
        (pm->pmb_pack->pmhd == nullptr)) {
      std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
      Kokkos::abort("MHD refinement variable used but <mhd> not defined");
    }
    if ((it->rvariable.compare(0, 3, "rad") == 0) &&
        (pm->pmb_pack->prad == nullptr)) {
      std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
      Kokkos::abort("radiation refinement variable used but <radiation> not defined");
    }
  }

  // count number of derived variables used for refinement
  // This is necessary to figure out dimensions needed for dvars array
  nderived = 0;
  SetRefinementData(pm->pmb_pack, true, false);

  if (nderived > 0) {
    auto &indcs = pmy_mesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    int nmb=std::max((pm->pmb_pack->nmb_thispack), (pm->pmb_pack->pmesh->nmb_maxperrank));
    Kokkos::realloc(dvars, nmb, nderived, ncells3, ncells2, ncells1);
  }

  // Set rdata array to shallow slice of target data
  SetRefinementData(pm->pmb_pack, false, false);
}

//----------------------------------------------------------------------------------------
// destructor

RefinementCriteria::~RefinementCriteria() {
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::SetRefinementData()
//! \brief Cycles through all criteria and load data

void RefinementCriteria::SetRefinementData(MeshBlockPack* pmbp, bool count_derived,
                                           bool load_derived) {
  int iderived = 0;  // current index of variable in dvars array
  for (auto it = rcrit.begin(); it != rcrit.end(); ++it) {
    // Only load data for methods that need it
    if ((it->rmethod != RefCritMethod::location) &&
        (it->rmethod != RefCritMethod::spectral_norm) &&
        (it->rmethod != RefCritMethod::user)) {
      using Kokkos::ALL;
      // hydro (lab-frame) density
      if (it->rvariable.compare("hydro_u_d") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IDN);
          it->rdata = Kokkos::subview(pmbp->phydro->u0, ALL, n, ALL, ALL, ALL);
        }
      // hydro (rest-frame) density
      } else if (it->rvariable.compare("hydro_w_d") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IDN);
          it->rdata = Kokkos::subview(pmbp->phydro->w0, ALL, n, ALL, ALL, ALL);
        }
      // mhd (lab-frame) density
      } else if (it->rvariable.compare("mhd_u_d") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IDN);
          it->rdata = Kokkos::subview(pmbp->pmhd->u0, ALL, n, ALL, ALL, ALL);
        }
      // mhd (rest-frame) density
      } else if (it->rvariable.compare("mhd_w_d") == 0) {
        if (!(count_derived) && !(load_derived)) {
          int n = static_cast<int>(IDN);
          it->rdata = Kokkos::subview(pmbp->pmhd->w0, ALL, n, ALL, ALL, ALL);
        }
      // radiation coordinate frame energy density R^0^0
      } else if (it->rvariable.compare("rad_coord_e") == 0) {
        if (count_derived) {
          nderived += 1;
        } else if (load_derived) {
          ComputeDerivedVariable(it->rvariable, iderived, pmbp, dvars);
          iderived += 1;
        } else {
          it->rdata = Kokkos::subview(dvars, ALL, iderived, ALL, ALL, ALL);
          iderived += 1;
        }
      } else {
        std::cout<<"### FATAL ERROR in "<<__FILE__<<" at line "<<__LINE__<<std::endl;
        Kokkos::abort("Unknown refinement variable requested in a <amr_criterion>");
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::CheckMinMax()
//! \brief Checks whether MeshBlock should be flagged for refinement/derefinement based on
//! min/max of selected variable on device.  Variable is set in SetRefinementData().

void RefinementCriteria::CheckMinMax(MeshBlockPack* pmbp, RefCritData crit) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  // capture variables for kernels
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int nmb = pmbp->nmb_thispack;

  auto &valmax = crit.rvalue_max;
  auto &q0 = crit.rdata;
  if (valmax < (FLT_MAX)) {  // user has set a max value to check
    par_for_outer("MaxRefCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_qmax= -(FLT_MAX);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& qmax) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        qmax = fmax(q0(m,k,j,i), qmax);
      },Kokkos::Max<Real>(team_qmax));
      // only derefine when flag has not been set by other criteria
      int &flag = refine_flag.d_view(m+mbs);
      if  (team_qmax > valmax)                 {flag = 1;}
      if ((team_qmax < valmax) && (flag == 0)) {flag = -1;}
    });
  }

  auto &valmin = crit.rvalue_min;
  if (valmin > -(FLT_MAX)) {  // user has set a min value to check
    par_for_outer("MaxRefCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_qmin= -(FLT_MAX);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& qmin) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        qmin = fmin(q0(m,k,j,i), qmin);
      },Kokkos::Min<Real>(team_qmin));
      // only derefine when flag has not been set by other criteria
      int &flag = refine_flag.d_view(m+mbs);
      if  (team_qmin < valmin)                 {flag = 1;}
      if ((team_qmin > valmin) && (flag == 0)) {flag = -1;}
    });
  }
  // sync device array with host
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::CheckSlope()
//! \brief Checks whether MeshBlock should be flagged for refinement/derefinement based on
//! magnitude of normalized slope (dq/q) of selected variable on device.  Variable is set
//! in SetRefinementData().

void RefinementCriteria::CheckSlope(MeshBlockPack* pmbp, RefCritData crit) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  // capture variables for kernels
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int nmb = pmbp->nmb_thispack;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;

  auto &valmax = crit.rvalue_max;
  auto &q0 = crit.rdata;
  if (valmax < (FLT_MAX)) {  // user has set a max value to check
    par_for_outer("MaxRefCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_dqmax= -(FLT_MAX);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& dqmax) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        Real d2 = SQR(q0(m,k,j,i+1) - q0(m,k,j,i-1));
        if (multi_d) {d2 += SQR(q0(m,k,j+1,i) - q0(m,k,j-1,i));}
        if (three_d) {d2 += SQR(q0(m,k+1,j,i) - q0(m,k-1,j,i));}
        dqmax = fmax((0.5*sqrt(d2)/q0(m,k,j,i)), dqmax);
      },Kokkos::Max<Real>(team_dqmax));
      // only derefine when flag has not been set by other criteria
      int &flag = refine_flag.d_view(m+mbs);
      if  (team_dqmax > valmax)                 {flag = 1;}
      if ((team_dqmax < valmax) && (flag == 0)) {flag = -1;}
    });
  }
  // sync device array with host
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::CheckSecondDeriv()
//! \brief Checks whether MeshBlock should be flagged for refinement/derefinement based on
//! magnitude of normalized second derivative (d2q/q) of selected variable on device.
//! Variable is set in SetRefinementData().

void RefinementCriteria::CheckSecondDeriv(MeshBlockPack* pmbp, RefCritData crit) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  // capture variables for kernels
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1;
  const int nji  = nx2*nx1;
  int nmb = pmbp->nmb_thispack;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;

  auto &valmax = crit.rvalue_max;
  auto &q0 = crit.rdata;
  if (valmax < (FLT_MAX)) {  // user has set a max value to check
    par_for_outer("MaxRefCond",DevExeSpace(), 0, 0, 0, (nmb-1),
    KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
      Real team_d2qmax= -(FLT_MAX);
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tmember, nkji),
      [=](const int idx, Real& d2qmax) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/nx1;
        int i = (idx - k*nji - j*nx1) + is;
        j += js;
        k += ks;
        Real d2q = q0(m,k,j,i+1) - 2.0*q0(m,k,j,i) + q0(m,k,j,i-1);
        if (multi_d) {d2q += (q0(m,k,j+1,i) - 2.0*q0(m,k,j,i) + q0(m,k,j-1,i));}
        if (three_d) {d2q += (q0(m,k+1,j,i) - 2.0*q0(m,k,j,i) + q0(m,k-1,j,i));}
        d2qmax = fmax((fabs(d2q)/q0(m,k,j,i)), d2qmax);
      },Kokkos::Max<Real>(team_d2qmax));
      // only derefine when flag has not been set by other criteria
      int &flag = refine_flag.d_view(m+mbs);
      if  (team_d2qmax > valmax)                 {flag = 1;}
      if ((team_d2qmax < valmax) && (flag == 0)) {flag = -1;}
    });
  }
  // sync device array with host
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void RefinementCriteria::CheckLocation()
//! \brief Checks whether MeshBlock should be flagged for refinement/derefinement based on
//! whether any part of MeshBlock is within given radius of a given position

void RefinementCriteria::CheckLocation(MeshBlockPack* pmbp, RefCritData crit) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;

  Real &x1 = crit.rloc_x1;
  Real &x2 = crit.rloc_x2;
  Real &x3 = crit.rloc_x3;
  Real &rad = crit.rloc_rad;
  for (int m = 0; m < nmb; ++m) {
    // extract MeshBlock bounds
    Real &x1min = size.h_view(m).x1min;
    Real &x1max = size.h_view(m).x1max;
    Real &x2min = size.h_view(m).x2min;
    Real &x2max = size.h_view(m).x2max;
    Real &x3min = size.h_view(m).x3min;
    Real &x3max = size.h_view(m).x3max;

    if (((x1min < (x1+rad)) && (x1min > (x1-rad))) ||
        ((x1max < (x1+rad)) && (x1max > (x1-rad))) ||
        ((x1max > (x1+rad)) && (x1min < (x1-rad)))) {
      if (!(multi_d) ||
          (((x2min < (x2+rad)) && (x2min > (x2-rad))) ||
           ((x2max < (x2+rad)) && (x2max > (x2-rad))) ||
           ((x2max > (x2+rad)) && (x2min < (x2-rad)))) ) {
        if (!(three_d) ||
            (((x3min < (x3+rad)) && (x3min > (x3-rad))) ||
             ((x3max < (x3+rad)) && (x3max > (x3-rad))) ||
             ((x3max > (x3+rad)) && (x3min < (x3-rad)))) ) {
          refine_flag.h_view(m + mbs) = 1;
        }
      }
    }
  }
  // sync host array with device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  return;
}

//----------------------------------------------------------------------------------------
void RefinementCriteria::CheckSpectralNorm(MeshBlockPack *pmbp,
                                           RefCritData crit) {
  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];

  // capture variables for kernels
  auto &indcs = pmbp->pmesh->mb_indcs;
  int &is = indcs.is, nx1 = indcs.nx1;
  int &js = indcs.js, nx2 = indcs.nx2;
  int &ks = indcs.ks, nx3 = indcs.nx3;
  const int nkji = nx3 * nx2 * nx1;
  const int nji = nx2 * nx1;
  int nmb = pmbp->nmb_thispack;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;

  // capture evolved/primitive variables
  auto &u0 = (pmbp->phydro != nullptr) ? pmbp->phydro->u0 : pmbp->pmhd->u0;
  auto &w0 = (pmbp->phydro != nullptr) ? pmbp->phydro->w0 : pmbp->pmhd->w0;

  auto &var = crit.use_primitives ? w0 : u0;

  // refinement thresholds for a 5-stencil
  const Real spectral_norm_refine = pow(4.0, -crit.spectral_norm_alpha_refine);
  const Real spectral_norm_coarsen =
      pow(4.0, -crit.spectral_norm_alpha_coarsen);

  auto &thres_refine = spectral_norm_refine;
  auto &thres_coarsen = spectral_norm_coarsen;
  auto &dfloor = crit.dfloor;
  auto &efloor = crit.efloor;

  auto &spectral_norm_error_policy = crit.spectral_norm_error_policy;
  auto &monitor_momentum = crit.monitor_momentum;
  auto &monitor_energy = crit.monitor_energy;

  par_for_outer(
      "SpectralNorm", DevExeSpace(), 0, 0, 0, (nmb - 1),
      KOKKOS_LAMBDA(TeamMember_t tmember, const int m) {
        // Some free functions
        //
        auto compute_d4u_x_axis = [&var](const int m, const VariableIndex IDX,
                                         const int k, const int j,
                                         const int i) {
          return var(m, IDX, k, j, i + 2) - 4.0 * var(m, IDX, k, j, i + 1) +
                 6.0 * var(m, IDX, k, j, i) - 4.0 * var(m, IDX, k, j, i - 1) +
                 var(m, IDX, k, j, i - 2);
        };
        auto compute_d4u_y_axis = [&var](const int m, const VariableIndex IDX,
                                         const int k, const int j,
                                         const int i) {
          return var(m, IDX, k, j + 2, i) - 4.0 * var(m, IDX, k, j + 1, i) +
                 6.0 * var(m, IDX, k, j, i) - 4.0 * var(m, IDX, k, j - 1, i) +
                 var(m, IDX, k, j - 2, i);
        };
        auto compute_d4u_z_axis = [&var](const int m, const VariableIndex IDX,
                                         const int k, const int j,
                                         const int i) {
          return var(m, IDX, k + 2, j, i) - 4.0 * var(m, IDX, k + 1, j, i) +
                 6.0 * var(m, IDX, k, j, i) - 4.0 * var(m, IDX, k - 1, j, i) +
                 var(m, IDX, k - 2, j, i);
        };
        // Another free function -- might need this for momentum criteria
        auto compute_d4u = [](const Real qp2, const Real qp1, const Real q,
                              const Real qm1, const Real qm2) {
          return qp2 - 4.0 * qp1 + 6.0 * q - 4.0 * qm1 + qm2;
        };

        // Grid refinement flag
        int &flag = refine_flag.d_view(m + mbs);
        bool flag_coarsen = true;

        // Error measure reduced within a single MB
        Real team_abs_d4u_over_u_max = 0.0;

        // Spectral norm method: check the evolved density variable
        Kokkos::parallel_reduce(
            Kokkos::TeamThreadRange(tmember, nkji),
            [=](const int idx, Real &max_error) {
              int k = (idx) / nji;
              int j = (idx - k * nji) / nx1;
              int i = (idx - k * nji - j * nx1) + is;
              j += js;
              k += ks;

              Real abs_d4u_over_u{0.0};

              // @yk: monitor the AMR criteria only when rho > rho_floor
              if (u0(m, IDN, k, j, i) > dfloor) {
                abs_d4u_over_u = fabs(compute_d4u_x_axis(m, IDN, k, j, i));

                if (multi_d) {
                  if (spectral_norm_error_policy ==
                      error_policy_for_multi_dim::sum) {
                    abs_d4u_over_u += fabs(compute_d4u_y_axis(m, IDN, k, j, i));
                  } else {
                    abs_d4u_over_u =
                        fmax(abs_d4u_over_u,
                             fabs(compute_d4u_y_axis(m, IDN, k, j, i)));
                  }
                }

                if (three_d) {
                  if (spectral_norm_error_policy ==
                      error_policy_for_multi_dim::sum) {
                    abs_d4u_over_u += fabs(compute_d4u_z_axis(m, IDN, k, j, i));
                  } else {
                    abs_d4u_over_u =
                        fmax(abs_d4u_over_u,
                             fabs(compute_d4u_z_axis(m, IDN, k, j, i)));
                  }
                }

                // normalize with the local value at (i,j,k)
                abs_d4u_over_u /= u0(m, IDN, k, j, i);
              }

              max_error = fmax(max_error, abs_d4u_over_u);
            },
            Kokkos::Max<Real>(team_abs_d4u_over_u_max));

        if (team_abs_d4u_over_u_max > thres_refine) {
          flag = 1;
          return;
        }
        flag_coarsen =
            flag_coarsen and (team_abs_d4u_over_u_max < thres_coarsen);

        // Spectral norm method: check the magnitude of the momentum
        //
        // @yk: this check is a bit expensive and has a less well-defined scale
        // of values (e.g. velocity can be very small and there is nothing wrong
        // with that). Will be good if we do not use this, or consider
        // monitoring pressure instead?
        if (monitor_momentum) {
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(tmember, nkji),
              [=](const int idx, Real &max_error) {
                int k = (idx) / nji;
                int j = (idx - k * nji) / nx1;
                int i = (idx - k * nji - j * nx1) + is;
                j += js;
                k += ks;

                Real abs_d4u_over_u{0.0};

                // @yk: monitor the AMR criteria only when rho > rho_floor
                if (u0(m, IDN, k, j, i) > dfloor) {

                  auto get_momentum_magnitude =
                      [&u0, &multi_d, &three_d](const int m, const int k,
                                                const int j, const int i) {
                        if (three_d) {
                          return std::sqrt(SQR(u0(m, IM1, k, j, i)) +
                                           SQR(u0(m, IM2, k, j, i)) +
                                           SQR(u0(m, IM3, k, j, i)));

                        } else if (multi_d) {
                          return std::sqrt(SQR(u0(m, IM1, k, j, i)) +
                                           SQR(u0(m, IM2, k, j, i)));
                        } else {
                          return fabs(u0(m, IM1, k, j, i));
                        }
                      };

                  abs_d4u_over_u =
                      fabs(compute_d4u(get_momentum_magnitude(m, k, j, i + 2),
                                       get_momentum_magnitude(m, k, j, i + 1),
                                       get_momentum_magnitude(m, k, j, i),
                                       get_momentum_magnitude(m, k, j, i - 1),
                                       get_momentum_magnitude(m, k, j, i - 2)));

                  if (multi_d) {
                    if (spectral_norm_error_policy ==
                        error_policy_for_multi_dim::sum) {
                      abs_d4u_over_u += fabs(
                          compute_d4u(get_momentum_magnitude(m, k, j + 2, i),
                                      get_momentum_magnitude(m, k, j + 1, i),
                                      get_momentum_magnitude(m, k, j, i),
                                      get_momentum_magnitude(m, k, j - 1, i),
                                      get_momentum_magnitude(m, k, j - 2, i)));
                    } else {
                      abs_d4u_over_u =
                          fmax(abs_d4u_over_u,
                               fabs(compute_d4u(
                                   get_momentum_magnitude(m, k, j + 2, i),
                                   get_momentum_magnitude(m, k, j + 1, i),
                                   get_momentum_magnitude(m, k, j, i),
                                   get_momentum_magnitude(m, k, j - 1, i),
                                   get_momentum_magnitude(m, k, j - 2, i))));
                    }
                  }

                  if (three_d) {
                    if (spectral_norm_error_policy ==
                        error_policy_for_multi_dim::sum) {
                      abs_d4u_over_u += fabs(
                          compute_d4u(get_momentum_magnitude(m, k + 2, j, i),
                                      get_momentum_magnitude(m, k + 1, j, i),
                                      get_momentum_magnitude(m, k, j, i),
                                      get_momentum_magnitude(m, k - 1, j, i),
                                      get_momentum_magnitude(m, k - 2, j, i)));
                    } else {
                      abs_d4u_over_u =
                          fmax(abs_d4u_over_u,
                               fabs(compute_d4u(
                                   get_momentum_magnitude(m, k + 2, j, i),
                                   get_momentum_magnitude(m, k + 1, j, i),
                                   get_momentum_magnitude(m, k, j, i),
                                   get_momentum_magnitude(m, k - 1, j, i),
                                   get_momentum_magnitude(m, k - 2, j, i))));
                    }
                  }

                  // normalize with the local value at (i,j,k)
                  abs_d4u_over_u /=
                      (get_momentum_magnitude(m, k, j, i) +
                       1e-15); // @yk: this epsilon value to avoid div-by-zero
                               // may be implemented in a better way
                }

                max_error = fmax(max_error, abs_d4u_over_u);
              },
              Kokkos::Max<Real>(team_abs_d4u_over_u_max));

          if (team_abs_d4u_over_u_max > thres_refine) {
            flag = 1;
            return;
          }
          flag_coarsen =
              flag_coarsen and (team_abs_d4u_over_u_max < thres_coarsen);
        }

        if (monitor_energy) {
          // Spectral norm method: check the evolved energy variable
          Kokkos::parallel_reduce(
              Kokkos::TeamThreadRange(tmember, nkji),
              [=](const int idx, Real &max_error) {
                int k = (idx) / nji;
                int j = (idx - k * nji) / nx1;
                int i = (idx - k * nji - j * nx1) + is;
                j += js;
                k += ks;

                Real abs_d4u_over_u{0.0};

                // @yk: monitor the AMR criteria only when e > e_floor
                if (u0(m, IEN, k, j, i) > efloor) {
                  abs_d4u_over_u = fabs(compute_d4u_x_axis(m, IEN, k, j, i));

                  if (multi_d) {
                    if (spectral_norm_error_policy ==
                        error_policy_for_multi_dim::sum) {
                      abs_d4u_over_u +=
                          fabs(compute_d4u_y_axis(m, IEN, k, j, i));
                    } else {
                      abs_d4u_over_u =
                          fmax(abs_d4u_over_u,
                               fabs(compute_d4u_y_axis(m, IEN, k, j, i)));
                    }
                  }

                  if (three_d) {
                    if (spectral_norm_error_policy ==
                        error_policy_for_multi_dim::sum) {
                      abs_d4u_over_u +=
                          fabs(compute_d4u_z_axis(m, IEN, k, j, i));
                    } else {
                      abs_d4u_over_u =
                          fmax(abs_d4u_over_u,
                               fabs(compute_d4u_z_axis(m, IEN, k, j, i)));
                    }
                  }

                  // normalize with the local value at (i,j,k)
                  abs_d4u_over_u /= u0(m, IEN, k, j, i);
                }

                max_error = fmax(max_error, abs_d4u_over_u);
              },
              Kokkos::Max<Real>(team_abs_d4u_over_u_max));

          if (team_abs_d4u_over_u_max > thres_refine) {
            flag = 1;
            return;
          }
          flag_coarsen =
              flag_coarsen and (team_abs_d4u_over_u_max < thres_coarsen);
        }

        // only derefine when flag has not been set by other criteria
        if (flag_coarsen && (flag == 0)) {
          flag = -1;
        }
      });

  // sync device array with host
  refine_flag.template modify<DevExeSpace>();
  refine_flag.template sync<HostMemSpace>();
  return;
}
