//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file gr_envelope.cpp
//! \brief Problem generator that initializes a GR hydro/MHD run in Cartesian Kerr-Schild
//! coordinates from tabulated data (density, velocity, specific internal energy) read from
//! a CSV file whose grid is a nested-box static-mesh-refinement (SMR) structure that
//! EXACTLY mirrors the AthenaK mesh's own <mesh_refinement> structure: a root box of
//! half-width root_half_width, with n_levels-1 refined levels nested inside it, each
//! covering half the linear extent of its parent at double the resolution (nx_per_level
//! cells across its own box, at every level). Because the grids are meant to coincide
//! exactly, cell values are looked up directly (nearest-cell), not interpolated -- see
//! make_athenak_ic_smr_grid.py (the reference generator this format is based on) for the
//! precise construction and the meshblock-alignment requirement this places on the
//! matching AthenaK <mesh>/<mesh_refinement> block. Reuses gr_torus.cpp's GR plumbing
//! (coordinate transforms, excision handling, MHD seed-field machinery).
//!
//! Expected CSV format:
//!   Line 1 (metadata, optionally "#"-prefixed):
//!     "bhpos_cm = .., .., .. ; r_sink_cm = .. ; bhmass_g = .. ;
//!      root_half_width_rsun = .. ; n_levels = .. ; nx_per_level = .."
//!   Line 2 (header): level,x_center_cm,y_center_cm,z_center_cm,rho_g_cm3,vx_cm_s,vy_cm_s,
//!                    vz_cm_s,eint_erg_g,n_cells
//!   Then one row per OCCUPIED cell (sparse -- empty cells are simply absent), in any
//!   order. "level" in [0, n_levels), where level L's box has half-width
//!   root_half_width/2^L and nx_per_level cells across it in each dimension, with a point
//!   belonging to the FINEST level whose box (by Chebyshev/L-infinity distance from the
//!   BH) contains it. Coordinates are already BH-centered. vx/vy/vz are physical Cartesian
//!   velocity components (cm/s) aligned with the same x/y/z axes as AthenaK's Cartesian-KS
//!   mesh coordinates (i.e. already in the mesh's coordinate basis, not a spherical
//!   orthonormal frame -- no BL rotation is needed to use them). n_cells is a source
//!   particle/cell count and is unused here (only occupied cells are ever written).
//!
//! Requires a <units> block with bhmass_msun and density_cgs (see units/units.hpp) to
//! convert the file's cgs quantities into code units.

#include <cstdlib>    // exit, EXIT_FAILURE
#include <cmath>      // pow, sqrt, fabs

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

#include <fstream>    // ifstream
#include <iostream>   // endl
#include <limits>     // numeric_limits::min()
#include <map>
#include <sstream>    // stringstream
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "coordinates/adm.hpp"
#include "coordinates/coordinates.hpp"
#include "coordinates/cartesian_ks.hpp"
#include "coordinates/cell_locations.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "dyn_grmhd/dyn_grmhd.hpp"
#include "units/units.hpp"

namespace {

// Solar radius in cm; must match the value used to generate the CSV (unit_rsun in
// make_athenak_ic_smr_grid.py), since only root_half_width_rsun is given in the header.
constexpr Real kRsunCgs = 6.958e10;

// device-resident tabulated data (bundled so it can be passed by value into device lambdas
// and helper functions the same way scalar pgen parameters are). Flat, dense storage over
// all levels: index = level*nx_per_level^3 + (i*nx_per_level + j)*nx_per_level + k.
struct EnvTables {
  DvceArray1D<Real> rho, vx, vy, vz, eint, valid;
};

// container for physical parameters of the envelope initial data
struct envelope_pgen {
  Real spin;                                    // black hole spin
  Real dexcise, pexcise;                        // excision parameters
  Real gamma_adi;                                // EOS parameter
  Real rho_min, rho_pow, pgas_min, pgas_pow;    // background/atmosphere power-law
  Real root_half_width;                          // root box half-width (code units)
  int n_levels, nx_per_level;                    // table's nested-box geometry
  Real potential_beta_min, potential_cutoff, potential_falloff;  // MHD seed-field params
  Real potential_r_pow, potential_rho_pow, potential_r_in;       // MHD seed-field params
  Real rho_max_for_norm;                         // max density, for seed-field normalization
};

// host-side staging area for the parsed CSV, before unit conversion / device upload
struct CsvTable {
  Real root_half_width_rsun = 0.0;
  int n_levels = 0, nx_per_level = 0;
  Real bhmass_g = 0.0;
  std::vector<int> level;
  std::vector<Real> x, y, z, rho, vx, vy, vz, eint;
  std::vector<int> ncells;
};

envelope_pgen egen;

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct envelope_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi);

KOKKOS_INLINE_FUNCTION
static void LookupEnvelopeCell(struct envelope_pgen pgen, struct EnvTables tab,
                               Real x1, Real x2, Real x3,
                               Real *prho, Real *pvx, Real *pvy, Real *pvz,
                               Real *peint, Real *pvalid);

KOKKOS_INLINE_FUNCTION
static void CalculateEnvelopeVectorPotential(struct envelope_pgen pgen, struct EnvTables tab,
                                             Real r, Real theta, Real x1, Real x2, Real x3,
                                             Real *patheta, Real *paphi);

KOKKOS_INLINE_FUNCTION
Real A1(struct envelope_pgen pgen, struct EnvTables tab, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A2(struct envelope_pgen pgen, struct EnvTables tab, Real x1, Real x2, Real x3);
KOKKOS_INLINE_FUNCTION
Real A3(struct envelope_pgen pgen, struct EnvTables tab, Real x1, Real x2, Real x3);

// host-only helper (file I/O), not a device function
bool ReadEnvelopeCsv(const std::string &fname, CsvTable *tab);

} // namespace

//----------------------------------------------------------------------------------------
//! \fn void ProblemGenerator::UserProblem()
//! \brief Sets initial conditions for a GR hydro/MHD run from a grid-matched tabulated
//! envelope data file.
//! Compile with '-D PROBLEM=gr_envelope' to enroll as user-specific problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!pmbp->pcoord->is_general_relativistic &&
      !pmbp->pcoord->is_dynamical_relativistic) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "gr_envelope problem can only be run when GR defined in <coord> block"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // capture variables for kernel
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int ie = indcs.ie, je = indcs.je, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  auto &coord = pmbp->pcoord->coord_data;
  bool use_dyngr = (pmbp->pdyngr != nullptr);

  egen.spin = coord.bh_spin;

  // return if restart
  if (restart) return;

  // Select either Hydro or MHD
  DvceArray5D<Real> u0_, w0_;
  if (pmbp->phydro != nullptr) {
    u0_ = pmbp->phydro->u0;
    w0_ = pmbp->phydro->w0;
  } else if (pmbp->pmhd != nullptr) {
    u0_ = pmbp->pmhd->u0;
    w0_ = pmbp->pmhd->w0;
  }

  // Get ideal gas EOS data
  if (pmbp->phydro != nullptr) {
    egen.gamma_adi = pmbp->phydro->peos->eos_data.gamma;
  } else if (pmbp->pmhd != nullptr) {
    egen.gamma_adi = pmbp->pmhd->peos->eos_data.gamma;
  }
  Real gm1 = egen.gamma_adi - 1.0;

  // excision parameters
  egen.dexcise = coord.dexcise;
  egen.pexcise = coord.pexcise;

  // Units are required to convert the file's cgs quantities into code units
  if (pmbp->punit == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "gr_envelope requires a <units> block (bhmass_msun, density_cgs) to "
              << "convert the tabulated cgs data into code units" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Read problem-specific parameters from input file
  std::string data_file = pin->GetString("problem", "initial_data_file");
  egen.rho_min  = pin->GetReal("problem", "rho_min");
  egen.rho_pow  = pin->GetReal("problem", "rho_pow");
  egen.pgas_min = pin->GetReal("problem", "pgas_min");
  egen.pgas_pow = pin->GetReal("problem", "pgas_pow");

  // Read the CSV (every rank reads independently, matching the precedent established by
  // kadath_bns.cpp/sgrid_bns.cpp/elliptica.cpp for external-ID pgens)
  CsvTable tab;
  ReadEnvelopeCsv(data_file, &tab);

  if (global_variable::my_rank == 0) {
    Real bhmass_msun_input = pin->GetReal("units", "bhmass_msun");
    Real bhmass_msun_file = tab.bhmass_g / units::Units::msun_cgs;
    if (fabs(bhmass_msun_input - bhmass_msun_file) > 0.01*bhmass_msun_file) {
      std::cout << "### WARNING in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<units>/bhmass_msun (" << bhmass_msun_input << ") differs from the "
                << "envelope file's bhmass_g-derived mass (" << bhmass_msun_file
                << " Msun) by more than 1%" << std::endl;
    }
    std::cout << "gr_envelope: read " << tab.x.size() << " occupied cells (n_levels="
              << tab.n_levels << ", nx_per_level=" << tab.nx_per_level
              << ", root_half_width_rsun=" << tab.root_half_width_rsun << ") from "
              << data_file << std::endl;
  }

  // Convert to code units (once, host-side)
  Real cu_length = pmbp->punit->cm();
  Real cu_dens   = pmbp->punit->g_cm3();
  Real cu_vel    = pmbp->punit->cm_s();   // velocity_cgs()==c exactly in GR mode, so this
                                          // converts cm/s directly into units of v/c
  Real cu_eint   = SQR(cu_vel);           // erg/g has units of (cm/s)^2

  egen.n_levels = tab.n_levels;
  egen.nx_per_level = tab.nx_per_level;
  egen.root_half_width = tab.root_half_width_rsun * kRsunCgs * cu_length;

  // Bin each occupied CSV row into its (level,i,j,k) dense-array slot. Index placement
  // uses raw cm coordinates (matching exactly how make_athenak_ic_smr_grid.py itself
  // located each cell within its level's box), while the stored VALUES are unit-converted.
  int nxp = tab.nx_per_level;
  long ntab = static_cast<long>(tab.n_levels) * nxp * nxp * nxp;
  HostArray1D<Real> h_rho("h_rho", ntab), h_vx("h_vx", ntab), h_vy("h_vy", ntab),
                     h_vz("h_vz", ntab), h_eint("h_eint", ntab), h_valid("h_valid", ntab);
  for (long n = 0; n < ntab; ++n) { h_valid(n) = 0.0; }

  Real root_hw_cm = tab.root_half_width_rsun * kRsunCgs;
  for (size_t row = 0; row < tab.x.size(); ++row) {
    int L = tab.level[row];
    if (L < 0 || L >= tab.n_levels) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Envelope data file row has level=" << L << ", outside [0," << tab.n_levels
                << ")" << std::endl;
      exit(EXIT_FAILURE);
    }
    Real hw_cm = root_hw_cm * pow(2.0, -static_cast<Real>(L));
    Real dcell_cm = 2.0*hw_cm / nxp;
    auto clamp_idx = [nxp](Real x, Real hw, Real dcell) {
      int idx = static_cast<int>(floor((x + hw)/dcell));
      return (idx < 0) ? 0 : ((idx > nxp-1) ? nxp-1 : idx);
    };
    int i = clamp_idx(tab.x[row], hw_cm, dcell_cm);
    int j = clamp_idx(tab.y[row], hw_cm, dcell_cm);
    int k = clamp_idx(tab.z[row], hw_cm, dcell_cm);
    long idx = static_cast<long>(L)*nxp*nxp*nxp + (static_cast<long>(i)*nxp + j)*nxp + k;

    h_rho(idx)  = tab.rho[row] * cu_dens;
    h_vx(idx)   = tab.vx[row]  * cu_vel;
    h_vy(idx)   = tab.vy[row]  * cu_vel;
    h_vz(idx)   = tab.vz[row]  * cu_vel;
    h_eint(idx) = tab.eint[row] * cu_eint;
    h_valid(idx) = 1.0;
  }

  EnvTables tabs;
  Kokkos::realloc(tabs.rho, ntab);
  Kokkos::realloc(tabs.vx, ntab);
  Kokkos::realloc(tabs.vy, ntab);
  Kokkos::realloc(tabs.vz, ntab);
  Kokkos::realloc(tabs.eint, ntab);
  Kokkos::realloc(tabs.valid, ntab);
  Kokkos::deep_copy(tabs.rho, h_rho);
  Kokkos::deep_copy(tabs.vx, h_vx);
  Kokkos::deep_copy(tabs.vy, h_vy);
  Kokkos::deep_copy(tabs.vz, h_vz);
  Kokkos::deep_copy(tabs.eint, h_eint);
  Kokkos::deep_copy(tabs.valid, h_valid);

  // initialize primitive variables for new run ---------------------------------------

  auto etrs = egen;
  auto &size = pmbp->pmb->mb_size;
  Real ptotmax = std::numeric_limits<float>::min();
  Real rhomax = std::numeric_limits<float>::min();
  const int nmkji = (pmbp->nmb_thispack)*indcs.nx3*indcs.nx2*indcs.nx1;
  const int nkji = indcs.nx3*indcs.nx2*indcs.nx1;
  const int nji  = indcs.nx2*indcs.nx1;

  Kokkos::parallel_reduce("pgen_envelope1", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &max_ptot, Real &max_rho) {
    // compute m,k,j,i indices of thread and call function
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/indcs.nx1;
    int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
    k += ks;
    j += js;

    Real &x1min = size.d_view(m).x1min;
    Real &x1max = size.d_view(m).x1max;
    Real x1v = CellCenterX(i-is, indcs.nx1, x1min, x1max);

    Real &x2min = size.d_view(m).x2min;
    Real &x2max = size.d_view(m).x2max;
    Real x2v = CellCenterX(j-js, indcs.nx2, x2min, x2max);

    Real &x3min = size.d_view(m).x3min;
    Real &x3max = size.d_view(m).x3max;
    Real x3v = CellCenterX(k-ks, indcs.nx3, x3min, x3max);

    Real &dx1 = size.d_view(m).dx1;
    Real &dx2 = size.d_view(m).dx2;
    Real &dx3 = size.d_view(m).dx3;

    Real glower[4][4], gupper[4][4];
    ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                            glower, gupper);

    Real r, theta, phi;
    GetBoyerLindquistCoordinates(etrs, x1v, x2v, x3v, &r, &theta, &phi);

    // Background/atmosphere, recomputed at a half-cell offset to stay consistent with
    // the excision algorithm (avoids leaving excised corners exposed)
    Real r_excise, theta_excise, phi_excise;
    GetBoyerLindquistCoordinates(etrs, x1v + copysign(0.5*dx1,x1v),
                                       x2v + copysign(0.5*dx2,x2v),
                                       x3v + copysign(0.5*dx3,x3v), &r_excise,
                                       &theta_excise, &phi_excise);
    Real rho_bg, pgas_bg;
    if (r_excise > 1.0) {
      rho_bg = etrs.rho_min * pow(r, etrs.rho_pow);
      pgas_bg = etrs.pgas_min * pow(r, etrs.pgas_pow);
    } else {
      rho_bg = etrs.dexcise;
      pgas_bg = etrs.pexcise;
    }

    Real rho, vx, vy, vz, eint, valid;
    LookupEnvelopeCell(etrs, tabs, x1v, x2v, x3v, &rho, &vx, &vy, &vz, &eint, &valid);

    Real uu1 = 0.0, uu2 = 0.0, uu3 = 0.0, pgas;
    if (valid > 0.0) {
      // vx,vy,vz are already physical Cartesian velocity components in the same basis as
      // x1,x2,x3 (no BL rotation needed, unlike a spherically-tabulated source) -- this is
      // a flat-space relation (exact in flat space), a good approximation here since the
      // tabulated velocities themselves carry no GR content (the source simulation did not
      // apply relativistic corrections either).
      Real vsq = fmin(SQR(vx) + SQR(vy) + SQR(vz), 1.0 - 1.0e-10);
      Real W = 1.0/sqrt(1.0 - vsq);
      Real u1 = W * vx;
      Real u2 = W * vy;
      Real u3 = W * vz;

      Real tmp = glower[1][1]*u1*u1 + 2.0*glower[1][2]*u1*u2 + 2.0*glower[1][3]*u1*u3
               + glower[2][2]*u2*u2 + 2.0*glower[2][3]*u2*u3 + glower[3][3]*u3*u3;
      Real gammasq = 1.0 + tmp;
      Real b = glower[0][1]*u1 + glower[0][2]*u2 + glower[0][3]*u3;
      Real u0 = (-b - sqrt(fmax(SQR(b) - glower[0][0]*gammasq, 0.0)))/glower[0][0];

      uu1 = u1 - gupper[0][1]/gupper[0][0] * u0;
      uu2 = u2 - gupper[0][2]/gupper[0][0] * u0;
      uu3 = u3 - gupper[0][3]/gupper[0][0] * u0;
      pgas = gm1 * rho * eint;
    } else {
      rho = rho_bg;
      pgas = pgas_bg;
    }

    // Set primitive values
    w0_(m,IDN,k,j,i) = fmax(rho, rho_bg);
    if (!use_dyngr) {
      w0_(m,IEN,k,j,i) = fmax(pgas, pgas_bg) / gm1;
    } else {
      w0_(m,IPR,k,j,i) = fmax(pgas, pgas_bg);
    }
    w0_(m,IVX,k,j,i) = uu1;
    w0_(m,IVY,k,j,i) = uu2;
    w0_(m,IVZ,k,j,i) = uu3;

    Real ptot;
    if (!use_dyngr) {
      ptot = gm1*w0_(m,IEN,k,j,i);
    } else {
      ptot = w0_(m,IPR,k,j,i);
    }
    max_ptot = fmax(ptot, max_ptot);
    max_rho = fmax(w0_(m,IDN,k,j,i), max_rho);
  }, Kokkos::Max<Real>(ptotmax), Kokkos::Max<Real>(rhomax));

  // rho_max_for_norm must be globally (not just rank-locally) reduced BEFORE the MHD
  // vector-potential kernel below consumes it, unlike ptotmax/bsqmax which are only
  // consumed later (after their own reductions further down)
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &rhomax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  egen.rho_max_for_norm = rhomax;

  // initialize ADM variables -----------------------------------------

  if (pmbp->padm != nullptr) {
    pmbp->padm->SetADMVariables(pmbp);
  }

  // initialize magnetic fields ---------------------------------------

  if (pmbp->pmhd != nullptr) {
    egen.potential_beta_min = pin->GetOrAddReal("problem", "potential_beta_min", 100.0);
    egen.potential_cutoff   = pin->GetOrAddReal("problem", "potential_cutoff", 0.2);
    egen.potential_falloff  = pin->GetOrAddReal("problem", "potential_falloff", 0.0);
    egen.potential_r_pow    = pin->GetOrAddReal("problem", "potential_r_pow", 0.0);
    egen.potential_rho_pow  = pin->GetOrAddReal("problem", "potential_rho_pow", 1.0);
    egen.potential_r_in     = pin->GetOrAddReal("problem", "potential_r_in", 5.0);
    etrs = egen;

    // compute vector potential over all faces
    int ncells1 = indcs.nx1 + 2*(indcs.ng);
    int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
    DvceArray4D<Real> a1, a2, a3;
    Kokkos::realloc(a1, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a2, nmb,ncells3,ncells2,ncells1);
    Kokkos::realloc(a3, nmb,ncells3,ncells2,ncells1);

    auto &nghbr = pmbp->pmb->nghbr;
    auto &mblev = pmbp->pmb->mb_lev;

    par_for("pgen_vector_potential", DevExeSpace(), 0,nmb-1,ks,ke+1,js,je+1,is,ie+1,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real &x1min = size.d_view(m).x1min;
      Real &x1max = size.d_view(m).x1max;
      int nx1 = indcs.nx1;
      Real x1v = CellCenterX(i-is, nx1, x1min, x1max);
      Real x1f   = LeftEdgeX(i  -is, nx1, x1min, x1max);

      Real &x2min = size.d_view(m).x2min;
      Real &x2max = size.d_view(m).x2max;
      int nx2 = indcs.nx2;
      Real x2v = CellCenterX(j-js, nx2, x2min, x2max);
      Real x2f   = LeftEdgeX(j  -js, nx2, x2min, x2max);

      Real &x3min = size.d_view(m).x3min;
      Real &x3max = size.d_view(m).x3max;
      int nx3 = indcs.nx3;
      Real x3v = CellCenterX(k-ks, nx3, x3min, x3max);
      Real x3f   = LeftEdgeX(k  -ks, nx3, x3min, x3max);

      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      a1(m,k,j,i) = A1(etrs, tabs, x1v, x2f, x3f);
      a2(m,k,j,i) = A2(etrs, tabs, x1f, x2v, x3f);
      a3(m,k,j,i) = A3(etrs, tabs, x1f, x2f, x3v);

      // When neighboring MeshBock is at finer level, compute vector potential as sum of
      // values at fine grid resolution.  This guarantees flux on shared fine/coarse
      // faces is identical.

      // Correct A1 at x2-faces, x3-faces, and x2x3-edges
      if ((nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,40).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,41).lev > mblev.d_view(m) && j==js && k==ks) ||
          (nghbr.d_view(m,42).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,43).lev > mblev.d_view(m) && j==je+1 && k==ks) ||
          (nghbr.d_view(m,44).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,45).lev > mblev.d_view(m) && j==js && k==ke+1) ||
          (nghbr.d_view(m,46).lev > mblev.d_view(m) && j==je+1 && k==ke+1) ||
          (nghbr.d_view(m,47).lev > mblev.d_view(m) && j==je+1 && k==ke+1)) {
        Real xl = x1v + 0.25*dx1;
        Real xr = x1v - 0.25*dx1;
        a1(m,k,j,i) = 0.5*(A1(etrs, tabs, xl,x2f,x3f) + A1(etrs, tabs, xr,x2f,x3f));
      }

      // Correct A2 at x1-faces, x3-faces, and x1x3-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,24).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,25).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,26).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,27).lev > mblev.d_view(m) && k==ks) ||
          (nghbr.d_view(m,28).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,29).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,30).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,31).lev > mblev.d_view(m) && k==ke+1) ||
          (nghbr.d_view(m,32).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,33).lev > mblev.d_view(m) && i==is && k==ks) ||
          (nghbr.d_view(m,34).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,35).lev > mblev.d_view(m) && i==ie+1 && k==ks) ||
          (nghbr.d_view(m,36).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,37).lev > mblev.d_view(m) && i==is && k==ke+1) ||
          (nghbr.d_view(m,38).lev > mblev.d_view(m) && i==ie+1 && k==ke+1) ||
          (nghbr.d_view(m,39).lev > mblev.d_view(m) && i==ie+1 && k==ke+1)) {
        Real xl = x2v + 0.25*dx2;
        Real xr = x2v - 0.25*dx2;
        a2(m,k,j,i) = 0.5*(A2(etrs, tabs, x1f,xl,x3f) + A2(etrs, tabs, x1f,xr,x3f));
      }

      // Correct A3 at x1-faces, x2-faces, and x1x2-edges
      if ((nghbr.d_view(m,0 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,1 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,2 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,3 ).lev > mblev.d_view(m) && i==is) ||
          (nghbr.d_view(m,4 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,5 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,6 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,7 ).lev > mblev.d_view(m) && i==ie+1) ||
          (nghbr.d_view(m,8 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,9 ).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,10).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,11).lev > mblev.d_view(m) && j==js) ||
          (nghbr.d_view(m,12).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,13).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,14).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,15).lev > mblev.d_view(m) && j==je+1) ||
          (nghbr.d_view(m,16).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,17).lev > mblev.d_view(m) && i==is && j==js) ||
          (nghbr.d_view(m,18).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,19).lev > mblev.d_view(m) && i==ie+1 && j==js) ||
          (nghbr.d_view(m,20).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,21).lev > mblev.d_view(m) && i==is && j==je+1) ||
          (nghbr.d_view(m,22).lev > mblev.d_view(m) && i==ie+1 && j==je+1) ||
          (nghbr.d_view(m,23).lev > mblev.d_view(m) && i==ie+1 && j==je+1)) {
        Real xl = x3v + 0.25*dx3;
        Real xr = x3v - 0.25*dx3;
        a3(m,k,j,i) = 0.5*(A3(etrs, tabs, x1f,x2f,xl) + A3(etrs, tabs, x1f,x2f,xr));
      }
    });

    auto &b0 = pmbp->pmhd->b0;
    par_for("pgen_b0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      // Compute face-centered fields from curl(A).
      Real dx1 = size.d_view(m).dx1;
      Real dx2 = size.d_view(m).dx2;
      Real dx3 = size.d_view(m).dx3;

      b0.x1f(m,k,j,i) = ((a3(m,k,j+1,i) - a3(m,k,j,i))/dx2 -
                         (a2(m,k+1,j,i) - a2(m,k,j,i))/dx3);
      b0.x2f(m,k,j,i) = ((a1(m,k+1,j,i) - a1(m,k,j,i))/dx3 -
                         (a3(m,k,j,i+1) - a3(m,k,j,i))/dx1);
      b0.x3f(m,k,j,i) = ((a2(m,k,j,i+1) - a2(m,k,j,i))/dx1 -
                         (a1(m,k,j+1,i) - a1(m,k,j,i))/dx2);

      // Include extra face-component at edge of block in each direction
      if (i==ie) {
        b0.x1f(m,k,j,i+1) = ((a3(m,k,j+1,i+1) - a3(m,k,j,i+1))/dx2 -
                             (a2(m,k+1,j,i+1) - a2(m,k,j,i+1))/dx3);
      }
      if (j==je) {
        b0.x2f(m,k,j+1,i) = ((a1(m,k+1,j+1,i) - a1(m,k,j+1,i))/dx3 -
                             (a3(m,k,j+1,i+1) - a3(m,k,j+1,i))/dx1);
      }
      if (k==ke) {
        b0.x3f(m,k+1,j,i) = ((a2(m,k+1,j,i+1) - a2(m,k+1,j,i))/dx1 -
                             (a1(m,k+1,j+1,i) - a1(m,k+1,j,i))/dx2);
      }
    });

    // Compute cell-centered fields
    auto &bcc_ = pmbp->pmhd->bcc0;
    par_for("pgen_bcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });

    // find maximum bsq
    Real bsqmax = std::numeric_limits<float>::min();
    Kokkos::parallel_reduce("envelope_beta", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(const int &idx, Real &max_bsq) {
      int m = (idx)/nkji;
      int k = (idx - m*nkji)/nji;
      int j = (idx - m*nkji - k*nji)/indcs.nx1;
      int i = (idx - m*nkji - k*nji - j*indcs.nx1) + is;
      k += ks;
      j += js;

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
      ComputeMetricAndInverse(x1v, x2v, x3v, coord.is_minkowski, coord.bh_spin,
                              glower, gupper);

      Real &wvx = w0_(m,IVX,k,j,i);
      Real &wvy = w0_(m,IVY,k,j,i);
      Real &wvz = w0_(m,IVZ,k,j,i);
      Real &wbx = bcc_(m,IBX,k,j,i);
      Real &wby = bcc_(m,IBY,k,j,i);
      Real &wbz = bcc_(m,IBZ,k,j,i);

      Real q = glower[1][1]*wvx*wvx +2.0*glower[1][2]*wvx*wvy +2.0*glower[1][3]*wvx*wvz
             + glower[2][2]*wvy*wvy +2.0*glower[2][3]*wvy*wvz
             + glower[3][3]*wvz*wvz;
      Real alpha = sqrt(-1.0/gupper[0][0]);
      Real lor = sqrt(1.0 + q);
      Real u0 = lor / alpha;
      Real u1 = wvx - alpha * lor * gupper[0][1];
      Real u2 = wvy - alpha * lor * gupper[0][2];
      Real u3 = wvz - alpha * lor * gupper[0][3];

      Real u_1 = glower[1][0]*u0 + glower[1][1]*u1 + glower[1][2]*u2 + glower[1][3]*u3;
      Real u_2 = glower[2][0]*u0 + glower[2][1]*u1 + glower[2][2]*u2 + glower[2][3]*u3;
      Real u_3 = glower[3][0]*u0 + glower[3][1]*u1 + glower[3][2]*u2 + glower[3][3]*u3;

      Real b0 = u_1*wbx + u_2*wby + u_3*wbz;
      Real b1 = (wbx + b0 * u1) / u0;
      Real b2 = (wby + b0 * u2) / u0;
      Real b3 = (wbz + b0 * u3) / u0;

      Real b_0 = glower[0][0]*b0 + glower[0][1]*b1 + glower[0][2]*b2 + glower[0][3]*b3;
      Real b_1 = glower[1][0]*b0 + glower[1][1]*b1 + glower[1][2]*b2 + glower[1][3]*b3;
      Real b_2 = glower[2][0]*b0 + glower[2][1]*b1 + glower[2][2]*b2 + glower[2][3]*b3;
      Real b_3 = glower[3][0]*b0 + glower[3][1]*b1 + glower[3][2]*b2 + glower[3][3]*b3;
      Real bsq = b0*b_0 + b1*b_1 + b2*b_2 + b3*b_3;

      max_bsq = fmax(bsq, max_bsq);
    }, Kokkos::Max<Real>(bsqmax));

#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &ptotmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bsqmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif

    // Apply renormalization of magnetic field
    Real bnorm = sqrt((ptotmax/(0.5*bsqmax))/egen.potential_beta_min);

    par_for("pgen_normb0", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      b0.x1f(m,k,j,i) *= bnorm;
      b0.x2f(m,k,j,i) *= bnorm;
      b0.x3f(m,k,j,i) *= bnorm;
      if (i==ie) { b0.x1f(m,k,j,i+1) *= bnorm; }
      if (j==je) { b0.x2f(m,k,j+1,i) *= bnorm; }
      if (k==ke) { b0.x3f(m,k+1,j,i) *= bnorm; }
    });

    // Recompute cell-centered magnetic field
    par_for("pgen_normbcc", DevExeSpace(), 0,nmb-1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real& w_bx = bcc_(m,IBX,k,j,i);
      Real& w_by = bcc_(m,IBY,k,j,i);
      Real& w_bz = bcc_(m,IBZ,k,j,i);
      w_bx = 0.5*(b0.x1f(m,k,j,i) + b0.x1f(m,k,j,i+1));
      w_by = 0.5*(b0.x2f(m,k,j,i) + b0.x2f(m,k,j+1,i));
      w_bz = 0.5*(b0.x3f(m,k,j,i) + b0.x3f(m,k+1,j,i));
    });
  }

  // Convert primitives to conserved
  if (pmbp->padm == nullptr) {
    if (pmbp->phydro != nullptr) {
      pmbp->phydro->peos->PrimToCons(w0_, u0_, is, ie, js, je, ks, ke);
    } else if (pmbp->pmhd != nullptr) {
      auto &bcc0_ = pmbp->pmhd->bcc0;
      pmbp->pmhd->peos->PrimToCons(w0_, bcc0_, u0_, is, ie, js, je, ks, ke);
    }
  } else {
    pmbp->pdyngr->PrimToConInit(is, ie, js, je, ks, ke);
  }

  return;
}

namespace {

//----------------------------------------------------------------------------------------
// Function for returning corresponding Boyer-Lindquist coordinates of point
// (identical to the transform used in gr_torus.cpp/gr_bondi.cpp; works for arbitrary spin)

KOKKOS_INLINE_FUNCTION
static void GetBoyerLindquistCoordinates(struct envelope_pgen pgen,
                                         Real x1, Real x2, Real x3,
                                         Real *pr, Real *ptheta, Real *pphi) {
  Real rad = sqrt(SQR(x1) + SQR(x2) + SQR(x3));
  Real r = fmax((sqrt( SQR(rad) - SQR(pgen.spin) + sqrt(SQR(SQR(rad)-SQR(pgen.spin))
                      + 4.0*SQR(pgen.spin)*SQR(x3)) ) / sqrt(2.0)), 1.0);
  *pr = r;
  *ptheta = (fabs(x3/r) < 1.0) ? acos(x3/r) : acos(copysign(1.0, x3));
  *pphi = atan2(r*x2-pgen.spin*x1, pgen.spin*x2+r*x1) -
          pgen.spin*r/(SQR(r)-2.0*r+SQR(pgen.spin));
  return;
}

//----------------------------------------------------------------------------------------
// Direct (nearest-cell) lookup of the tabulated envelope data at a Cartesian-KS point.
// Finds the finest table level whose box (Chebyshev/L-infinity distance from the BH)
// contains the point -- mirroring exactly how make_athenak_ic_smr_grid.py assigned each
// source particle to a level -- then indexes that level's dense nx_per_level^3 array.
// Outputs are left at zero and *pvalid=0 if the point falls outside every level's box, or
// in a level-box cell that had no occupied CSV row (in which case the caller should fall
// back to the atmosphere/background floor).

KOKKOS_INLINE_FUNCTION
static void LookupEnvelopeCell(struct envelope_pgen pgen, struct EnvTables tab,
                               Real x1, Real x2, Real x3,
                               Real *prho, Real *pvx, Real *pvy, Real *pvz,
                               Real *peint, Real *pvalid) {
  *prho = 0.0; *pvx = 0.0; *pvy = 0.0; *pvz = 0.0; *peint = 0.0; *pvalid = 0.0;

  Real cheby = fmax(fabs(x1), fmax(fabs(x2), fabs(x3)));
  int level = -1;
  for (int L = pgen.n_levels-1; L >= 0; --L) {
    Real hw = pgen.root_half_width * pow(2.0, -static_cast<Real>(L));
    if (cheby <= hw) { level = L; break; }
  }
  if (level < 0) return;

  int nxp = pgen.nx_per_level;
  Real hw = pgen.root_half_width * pow(2.0, -static_cast<Real>(level));
  Real dcell = 2.0*hw / nxp;
  int i = static_cast<int>(floor((x1 + hw)/dcell));
  int j = static_cast<int>(floor((x2 + hw)/dcell));
  int k = static_cast<int>(floor((x3 + hw)/dcell));
  i = (i < 0) ? 0 : ((i > nxp-1) ? nxp-1 : i);
  j = (j < 0) ? 0 : ((j > nxp-1) ? nxp-1 : j);
  k = (k < 0) ? 0 : ((k > nxp-1) ? nxp-1 : k);

  int idx = level*nxp*nxp*nxp + (i*nxp + j)*nxp + k;
  if (tab.valid(idx) > 0.5) {
    *prho = tab.rho(idx); *pvx = tab.vx(idx); *pvy = tab.vy(idx); *pvz = tab.vz(idx);
    *peint = tab.eint(idx); *pvalid = 1.0;
  }
  return;
}

//----------------------------------------------------------------------------------------
// Vector potential for an analytic seed field (poloidal loop), following gr_torus.cpp's
// approach but using the looked-up+floored density in place of an analytic torus profile,
// and pgen.potential_r_in in place of the torus's r_edge

KOKKOS_INLINE_FUNCTION
static void CalculateEnvelopeVectorPotential(struct envelope_pgen pgen, struct EnvTables tab,
                                             Real r, Real theta, Real x1, Real x2, Real x3,
                                             Real *patheta, Real *paphi) {
  Real atheta = 0.0, aphi = 0.0;
  if (r >= pgen.potential_r_in) {
    Real rho_i, vx, vy, vz, eint, valid;
    LookupEnvelopeCell(pgen, tab, x1, x2, x3, &rho_i, &vx, &vy, &vz, &eint, &valid);
    Real rho_bg = pgen.rho_min * pow(r, pgen.rho_pow);
    Real rho_here = (valid > 0.0) ? fmax(rho_i, rho_bg) : rho_bg;

    Real sin_theta = sin(theta);
    Real cyl_radius = r * sin_theta;
    Real scaling = pow(cyl_radius/pgen.potential_r_in, pgen.potential_r_pow);
    if (pgen.potential_falloff != 0.0) {
      scaling *= exp(-r/pgen.potential_falloff);
    }
    aphi = pow(rho_here/pgen.rho_max_for_norm, pgen.potential_rho_pow) * scaling;
    aphi -= pgen.potential_cutoff;
    aphi = fmax(aphi, 0.0);
  }
  *patheta = atheta;
  *paphi = aphi;
  return;
}

//----------------------------------------------------------------------------------------
// A1/A2/A3: components of vector potential in Cartesian KS, computed from the spherical-KS
// (theta,phi) potential above (formulas identical to gr_torus.cpp's, which are generic
// BL/spherical-KS -> Cartesian-KS transforms, not torus-specific)

KOKKOS_INLINE_FUNCTION
Real A1(struct envelope_pgen pgen, struct EnvTables tab, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  Real atheta, aphi;
  CalculateEnvelopeVectorPotential(pgen, tab, r, theta, x1, x2, x3, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x1*x3*isin_term/(r*sqrt_term)) +
         aphi*(-x2/(SQR(x1)+SQR(x2))+pgen.spin*x1*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

KOKKOS_INLINE_FUNCTION
Real A2(struct envelope_pgen pgen, struct EnvTables tab, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  Real atheta, aphi;
  CalculateEnvelopeVectorPotential(pgen, tab, r, theta, x1, x2, x3, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(x2*x3*isin_term/(r*sqrt_term)) +
         aphi*(x1/(SQR(x1)+SQR(x2))+pgen.spin*x2*r/((SQR(pgen.spin)+SQR(r))*sqrt_term));
}

KOKKOS_INLINE_FUNCTION
Real A3(struct envelope_pgen pgen, struct EnvTables tab, Real x1, Real x2, Real x3) {
  Real r, theta, phi;
  GetBoyerLindquistCoordinates(pgen, x1, x2, x3, &r, &theta, &phi);
  Real atheta, aphi;
  CalculateEnvelopeVectorPotential(pgen, tab, r, theta, x1, x2, x3, &atheta, &aphi);

  Real big_r = sqrt( SQR(x1) + SQR(x2) + SQR(x3) );
  Real sqrt_term =  2.0*SQR(r) - SQR(big_r) + SQR(pgen.spin);
  Real isin_term = sqrt((SQR(pgen.spin)+SQR(r))/fmax(SQR(x1)+SQR(x2),1.0e-12));

  return atheta*(((1.0+SQR(pgen.spin/r))*SQR(x3)-sqrt_term)*isin_term/(r*sqrt_term)) +
         aphi*(pgen.spin*x3/(r*sqrt_term));
}

//----------------------------------------------------------------------------------------
// Host-only helper: parse the envelope CSV. Every rank reads/parses the file
// independently (matches the precedent set by kadath_bns.cpp/sgrid_bns.cpp/elliptica.cpp
// for external initial-data pgens; no MPI broadcast of file contents in this codebase).

bool ReadEnvelopeCsv(const std::string &fname, CsvTable *tab) {
  std::ifstream fin(fname.c_str());
  if (!fin.is_open()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Could not open envelope initial_data_file: " << fname << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string meta_line, header_line;
  std::getline(fin, meta_line);
  std::getline(fin, header_line);
  // tolerate an optional leading "# " comment marker on the metadata line
  if (!meta_line.empty() && meta_line[0] == '#') {
    meta_line = meta_line.substr(1);
  }

  // parse "key = value [, value ...] ; key = value ; ..." metadata line
  std::map<std::string, std::string> kv;
  {
    std::stringstream ss(meta_line);
    std::string segment;
    while (std::getline(ss, segment, ';')) {
      auto eq = segment.find('=');
      if (eq == std::string::npos) { continue; }
      std::string key = segment.substr(0, eq);
      std::string val = segment.substr(eq+1);
      auto trim = [](std::string &s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        s = (a == std::string::npos) ? "" : s.substr(a, b-a+1);
      };
      trim(key); trim(val);
      kv[key] = val;
    }
  }
  auto require_key = [&](const std::string &key) -> std::string {
    auto it = kv.find(key);
    if (it == kv.end()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Envelope data file metadata line missing required key: " << key
                << std::endl;
      exit(EXIT_FAILURE);
    }
    return it->second;
  };
  try {
    tab->root_half_width_rsun = std::stod(require_key("root_half_width_rsun"));
    tab->n_levels = std::stoi(require_key("n_levels"));
    tab->nx_per_level = std::stoi(require_key("nx_per_level"));
    tab->bhmass_g = std::stod(require_key("bhmass_g"));
  } catch (const std::exception &e) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Failed to parse envelope data file metadata line: " << e.what()
              << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) { continue; }
    std::stringstream ls(line);
    std::string tok;
    std::vector<std::string> fields;
    while (std::getline(ls, tok, ',')) { fields.push_back(tok); }
    if (fields.size() != 10) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Envelope data file row has " << fields.size()
                << " fields, expected 10: " << line << std::endl;
      exit(EXIT_FAILURE);
    }
    tab->level.push_back(std::stoi(fields[0]));
    tab->x.push_back(std::stod(fields[1]));
    tab->y.push_back(std::stod(fields[2]));
    tab->z.push_back(std::stod(fields[3]));
    tab->rho.push_back(std::stod(fields[4]));
    tab->vx.push_back(std::stod(fields[5]));
    tab->vy.push_back(std::stod(fields[6]));
    tab->vz.push_back(std::stod(fields[7]));
    tab->eint.push_back(std::stod(fields[8]));
    tab->ncells.push_back(std::stoi(fields[9]));
  }
  fin.close();
  return true;
}

} // namespace
