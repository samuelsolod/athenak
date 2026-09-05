"""
Builds the tabulated stellar-envelope initial-data CSV consumed by AthenaK's
`gr_envelope` problem generator (see the co-located gr_envelope.athinput and
src/pgen/fluids/gr_envelope.cpp in the AthenaK source tree).

WHAT THIS SCRIPT DOES
    Reads a snapshot of a star from the moving-mesh code AREPO (an HDF5 file
    containing gas particles, PartType0, and a sink particle representing a
    central compact object, PartType5), rebins the gas particles' mass,
    momentum, and specific internal energy onto a nested-box static mesh
    refinement (SMR) grid centered on that compact object, and writes the
    result to a CSV file. gr_envelope.cpp reads this CSV directly at the start
    of an AthenaK run and requires the AthenaK <mesh>/<mesh_refinement> block
    in the paired .athinput file to reproduce this script's nested-box levels
    exactly (same ROOT_HALF_WIDTH_RG, n_levels, nx_per_level -- all recorded in
    the CSV header this script writes).

GRID DEFINITION VS. OUTPUT UNITS
    All grid geometry -- the nested SMR box half-widths, and which box each
    particle is binned into -- is computed in units of the gravitational
    radius r_g = G*M_BH/c^2 of the black hole mass defined below, not in cm.
    The values actually WRITTEN to the output CSV, however (cell-center
    positions, density, velocity, specific internal energy), are converted
    back to AREPO's original physical CGS units (cm, g/cm^3, cm/s, erg/g); the
    r_g-based units are used only internally, to define the grid. Velocity and
    specific internal energy additionally carry a dimensionless mass-ratio
    rescaling (see "VELOCITY AND INTERNAL-ENERGY SCALING" below), which does
    not change their units. The exact column order and units of every output
    column are given in the header line this script writes to the CSV.

KEY ASSUMPTIONS
    - Only the first sink particle in the snapshot (PartType5, index 0) is
      used, as the compact object the grid is centered on; if more than one
      sink particle is present, the others are ignored entirely.
    - Gas (PartType0) positions, velocities, and specific internal energies
      are assumed to already be in AREPO's standard CGS convention (cm, cm/s,
      erg/g respectively); masses are in g.
    - The physical constants defined below (G_cgs, c_cgs, msun_cgs) must match
      the corresponding constants in AthenaK's units::Units class exactly:
      AthenaK independently recomputes r_g from <units>/bhmass_msun in the
      run's input file, and a mismatch between this script's r_g and that
      value larger than 1% triggers a startup warning in AthenaK (see
      gr_envelope.cpp).

OUTPUT
    A CSV at `outfile` (see Settings, below) whose first line is a "# key =
    value ; ..." metadata header (every parameter defined in this script,
    plus a few values read from the snapshot for reference) and whose
    remaining lines are one row per occupied grid cell, in the column order
    given at the end of that same header line.
"""

import h5py
import numpy as np

# ============================================================================
# ---------------- Settings ----------------
# ============================================================================
snapfile = "../arepo/snapshot_015.hdf5"   # source AREPO snapshot (HDF5), path relative
                                           # to this script's working directory
outfile = "athenak_ic_rg.csv"             # output CSV, read directly by gr_envelope.cpp

# ---- Physical constants (must match units::Units in AthenaK) ----
G_cgs = 6.67408e-8
c_cgs = 2.99792458e10
msun_cgs = 1.98841586e33
unit_rsun = 6.958e10

# ============================================================================
# BLACK HOLE MASS
#   The mass used for everything below (grid geometry, unit conversions, and
#   the velocity/internal-energy rescaling further down) is solved for from a
#   target Schwarzschild radius, TARGET_RS_RSUN (in solar radii), rather than
#   being read from the snapshot's own sink-particle mass:
#     r_s = 2*G*M_BH/c^2  =>  M_BH = r_s*c^2 / (2*G).
#   This decouples the black hole's gravitational radius -- and therefore how
#   finely the SMR grid below must be refined to resolve its horizon -- from
#   whatever mass the snapshot's sink particle happens to record. The
#   snapshot's own sink-particle mass (bhmass_snapshot, read further down) is
#   not used in any calculation here; it is kept only for reference and is
#   recorded as such in the CSV header.
# ============================================================================
TARGET_RS_RSUN = 0.025
r_s_target_cm = TARGET_RS_RSUN * unit_rsun
M_BH_g = r_s_target_cm * c_cgs**2 / (2.0 * G_cgs)     # black hole mass used everywhere below
M_BH_msun = M_BH_g / msun_cgs

# ============================================================================
# GRAVITATIONAL-RADIUS UNITS
#   r_g = G*M_BH/c^2, using the black hole mass defined above (== r_s_target/2
#   exactly, checked by the assertion below as a sanity test on the
#   arithmetic above). The event horizon sits at r = 2 r_g. All grid geometry
#   and particle positions from here on -- SMR level half-widths, bin edges,
#   the Chebyshev-distance level assignment -- are expressed in units of THIS
#   r_g; cm units reappear only in the final CSV output.
# ============================================================================
r_g_cm = G_cgs * M_BH_g / c_cgs**2
assert abs(r_g_cm - r_s_target_cm / 2.0) / r_g_cm < 1e-10, \
    "r_g should equal r_s_target/2 exactly (both derived from the same M_BH)"

# ============================================================================
# ROOT SMR BOX SIZE
#   ROOT_HALF_WIDTH_RG is the half-width, in r_g, of the root (coarsest)
#   nested SMR box -- also the physical domain size the paired
#   gr_envelope.athinput's <mesh> block must cover. It is set here to exactly
#   match the AREPO snapshot's own simulation box half-width (BoxSize/2 =
#   1.499e13 cm = 215.443 R_sun for this black hole mass), so that the root
#   box covers the star's entire simulated volume: no gas particle falls
#   outside the grid and is left untabulated. Using a smaller half-width here
#   would still produce a valid CSV, but would silently discard whatever
#   fraction of the star's mass lies beyond it.
#
#   This is a Cartesian nested-box grid, not a radial/log-spaced spherical
#   one, so the root box already extends down to (and including) r=0 --
#   there is no "r cannot start at zero" issue the way there would be for a
#   log-radial grid.
#
#   n_levels and nx_per_level (below) are both DERIVED from this half-width
#   together with a target resolution at the horizon (see "RESOLUTION AT THE
#   HORIZON"), so changing ROOT_HALF_WIDTH_RG alone automatically adjusts how
#   many refinement levels are generated.
# ============================================================================
ROOT_HALF_WIDTH_RG = 17235.477520264445

# ============================================================================
# RESOLUTION AT THE HORIZON
#   nx_per_level is the number of cells per axis in every SMR level's own box
#   (each level covers a different physical volume -- see "SMR LEVEL
#   GEOMETRY" below -- but all levels share this same per-axis cell count).
#   It is chosen, together with n_levels, so that the finest level actually
#   generated resolves the horizon (r = 2 r_g) at a cell size of a small
#   fraction of an r_g -- i.e. several tens of cells across the horizon's
#   diameter. The derivation:
#
#     - Level L's box half-width is ROOT_HALF_WIDTH_RG / 2^L, so level L's
#       cell size is 2*(ROOT_HALF_WIDTH_RG / 2^L) / nx_per_level.
#     - _level_at_horizon solves for the (real-valued, then rounded) level
#       whose box half-width equals the horizon radius, giving the level
#       index at which the box size and the horizon radius are comparable.
#     - n_levels is set equal to _level_at_horizon, which means the levels
#       actually generated are L = 0 .. n_levels-1 -- one level coarser than
#       _level_at_horizon itself. The finest generated level, n_levels-1, has
#       half-width ROOT_HALF_WIDTH_RG / 2**(n_levels-1) = 4.208 r_g, which
#       comfortably contains the horizon (r = 2 r_g) without needing to
#       generate an additional, even finer level.
#     - nx_per_level must ALSO be a power-of-2 multiple of the AthenaK
#       meshblock size used in the paired gr_envelope.athinput (4 cells/axis
#       there) so that every refined-region boundary in that input file lands
#       exactly on a meshblock edge at every level -- otherwise AthenaK's
#       meshblock-granularity static refinement would refine a larger region
#       than this script actually populated with data. nx_per_level=128
#       satisfies this (128 = 4*32).
#
#   With these values, the finest level's cell size is printed below
#   (_cellsize_horizon_level) together with how many cells span the horizon's
#   diameter, as a direct check on the resolution actually achieved.
# ============================================================================
HORIZON_RG = 2.0
_level_at_horizon = int(round(np.log2(ROOT_HALF_WIDTH_RG / HORIZON_RG) - 0.5))
nx_per_level = 128       # cells per axis, in every SMR level's own box (output resolution)
n_levels = _level_at_horizon    # levels 0..n_levels-1 are generated; see derivation above

# ============================================================================
# TWO-STAGE MASS-CONSERVING BINNING
#   Binning particles directly at nx_per_level starves most cells (as few as
#   1-2 particles per cell at the finest levels), producing noisy,
#   non-representative mass-weighted averages. Instead, particles are first
#   binned onto a coarser BIN_NX_PER_LEVEL grid, where enough particles land
#   in each cell for the mass-weighted averages to be statistically
#   meaningful, and that coarse (rho, v, eint) field is then upsampled onto
#   the finer nx_per_level output grid by replicating each coarse cell's
#   value uniformly across the UPSAMPLE_FACTOR^3 fine sub-cells that exactly
#   tile it.
#
#   This upsampling exactly conserves mass: rho is an intrinsic (mass/volume)
#   quantity, and a coarse cell's volume equals exactly UPSAMPLE_FACTOR^3 fine
#   sub-cell volumes (since nx_per_level = UPSAMPLE_FACTOR * BIN_NX_PER_LEVEL),
#   so summing the fine sub-cells' mass reproduces the coarse cell's mass
#   exactly:
#     sum_subcells(rho_fine * V_fine) = rho_coarse * (UPSAMPLE_FACTOR^3 * V_fine)
#                                      = rho_coarse * V_coarse
#                                      = coarse cell's original binned mass.
#   No particles are ever re-binned at the fine resolution -- fine cells
#   simply inherit their parent coarse cell's already mass-conservative value.
# ============================================================================
BIN_NX_PER_LEVEL = 16
UPSAMPLE_FACTOR = nx_per_level // BIN_NX_PER_LEVEL
assert nx_per_level % BIN_NX_PER_LEVEL == 0, \
    "nx_per_level must be an integer multiple of BIN_NX_PER_LEVEL to upsample exactly"

_finest_level = n_levels - 1  # deepest level index actually generated
_hw_horizon_level = ROOT_HALF_WIDTH_RG / (2 ** _finest_level)
_cellsize_horizon_level = 2.0 * _hw_horizon_level / nx_per_level
print("Horizon (r=2 r_g) resolved by level %d (box half-width %.4f r_g, "
      "cell size %.4f r_g, %.1f cells across the horizon diameter)"
      % (_finest_level, _hw_horizon_level, _cellsize_horizon_level,
         (2 * HORIZON_RG) / _cellsize_horizon_level))

# ============================================================================
# VELOCITY AND INTERNAL-ENERGY SCALING
#   Because the black hole mass used here (M_BH_msun, above) generally differs
#   from the mass of whatever central object the AREPO snapshot was originally
#   simulated with, the binned velocity and specific internal energy are
#   rescaled by a dimensionless ratio of the new mass to a fixed reference
#   mass, M_BH_OLD_MSUN: velocity by sqrt(MASS_RATIO), specific internal
#   energy by MASS_RATIO (linear). This mirrors how a Keplerian/virial
#   velocity scales as sqrt(GM/r) and a specific energy as v^2 ~ GM/r at fixed
#   r/r_g: since all positions here are already expressed in units of r_g (a
#   quantity purely) rather than a fixed physical length, holding the
#   *shape* of the flow fixed in r_g units while changing M_BH requires
#   exactly this rescaling of velocity and specific energy. Note this is
#   applied AFTER the mass-weighted averaging further down, which is
#   equivalent to applying it per-particle beforehand, since the scale
#   factors are the same constant for every particle and averaging is linear.
# ============================================================================
M_BH_OLD_MSUN = 10.0    # fixed reference mass (not read from the snapshot); mass ratio
                          # and the resulting velocity/energy scale factors are relative
                          # to this value
MASS_RATIO = M_BH_msun / M_BH_OLD_MSUN     # dimensionless
VELOCITY_SCALE = np.sqrt(MASS_RATIO)
EINT_SCALE = MASS_RATIO

# ============================================================================
# INTERIOR ZEROING
#   When ZERO_INNER_REGION is True, density, velocity, and specific internal
#   energy are set to zero (not removed -- the row and its particle-count
#   column are left in the CSV) for every output cell whose center lies
#   within ZERO_INNER_R_RG of the black hole. Applied as a final
#   post-processing step on the fully assembled, binned, and scaled table
#   (see near the end of this script). On the AthenaK side, a zeroed cell is
#   treated as tabulated-but-empty gas rather than as "outside the table", so
#   it does not fall back to gr_envelope.cpp's power-law atmosphere floor the
#   way a genuinely absent cell would.
# ============================================================================
ZERO_INNER_REGION = True
ZERO_INNER_R_RG = 10.0

# ============================================================================
# ---------------- Read AREPO snapshot (still cgs at this point) ----------------
# ============================================================================
with h5py.File(snapfile, "r") as ff:
    BoxSize = np.array(ff["Header"].attrs["BoxSize"])          # cm

    pos = np.array(ff["PartType0"]["Coordinates"])             # cm, (N, 3)
    mass = np.array(ff["PartType0"]["Masses"])                 # g
    vel = np.array(ff["PartType0"]["Velocities"])              # cm/s
    eint = np.array(ff["PartType0"]["InternalEnergy"])         # erg/g (specific)

    bhpos = np.array(ff["PartType5"]["Coordinates"])[0]        # cm; first sink particle only
    bhmass_snapshot = np.array(ff["PartType5"]["Masses"])[0]   # g -- NOT used for physics
                                                                 # below; kept only for
                                                                 # reference/printing (see
                                                                 # "BLACK HOLE MASS" above)
    r_sink = np.array(ff["Parameters"].attrs["BlackHoleMaxAccretionRadius"])  # cm

# Recenter gas coordinates on the sink particle's actual (AREPO) position. Only the BLACK
# HOLE MASS used for physics/units is replaced by the target value above -- its position is
# still taken directly from the snapshot.
pos -= BoxSize / 2
bhpos -= BoxSize / 2
pos -= bhpos

# ---- convert recentered positions to r_g units (see "GRAVITATIONAL-RADIUS UNITS" above);
#      velocities are left in cm/s here and only acquire the dimensionless VELOCITY_SCALE
#      factor later, during binning ----
x, y, z = pos[:, 0] / r_g_cm, pos[:, 1] / r_g_cm, pos[:, 2] / r_g_cm   # now dimensionless (r_g)
vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]                            # still cm/s

# ============================================================================
# SMR LEVEL ASSIGNMENT
#   Each particle is assigned to the FINEST SMR level whose box contains it,
#   using Chebyshev (L-infinity) distance from the origin -- i.e. a particle
#   belongs to level L if it lies within the cube of half-width
#   level_half_width_rg[L] centered on the black hole. The loop below walks
#   from the finest level to the coarsest, claiming only particles not yet
#   assigned to a finer level, so that every particle's mass is counted
#   exactly once (in its finest containing level) rather than once per level
#   whose box happens to contain it.
# ============================================================================
cheby = np.maximum(np.abs(x), np.maximum(np.abs(y), np.abs(z)))   # r_g units now
level_of = -np.ones(len(x), dtype=np.int8)   # -1 = outside the root box entirely

level_half_width_rg = [ROOT_HALF_WIDTH_RG / (2**L) for L in range(n_levels)]
for L in range(n_levels - 1, -1, -1):
    mask = (cheby <= level_half_width_rg[L]) & (level_of == -1)
    level_of[mask] = L

# ============================================================================
# PER-LEVEL BINNING
#   For each level, particles assigned to it are binned onto the coarse
#   BIN_NX_PER_LEVEL grid (mass-weighted averages of velocity/eint, and total
#   mass per cell), the dimensionless VELOCITY_SCALE/EINT_SCALE factors are
#   applied to those averages, and the result is upsampled onto the fine
#   nx_per_level output grid -- see the "TWO-STAGE MASS-CONSERVING BINNING"
#   note above for why/how. Every populated (level, fine-cell) combination
#   becomes one row of the final output table.
# ============================================================================
rows = []
print("level  half_width[r_g]  bin_cell[r_g]  n_particles  n_nonempty/nbins(coarse)  mass[g]")
for L in range(n_levels):
    hw = level_half_width_rg[L]

    # ---- coarse binning grid: this is where particles are actually binned ----
    edges_bin = np.linspace(-hw, hw, BIN_NX_PER_LEVEL + 1)
    dcell_bin = edges_bin[1] - edges_bin[0]
    V_cell_bin = dcell_bin**3   # r_g^3 (NOT cm^3 -- grid geometry stays in r_g units)

    # ---- fine output grid: coarse values are upsampled onto this ----
    edges_fine = np.linspace(-hw, hw, nx_per_level + 1)
    centers_fine = 0.5 * (edges_fine[:-1] + edges_fine[1:])

    sel = (level_of == L)
    n_sel = sel.sum()
    if n_sel == 0:
        print("%5d  %16.4f  %16.4f  %11d  %8s  %10s" % (L, hw, dcell_bin, 0, "0/0", "0"))
        continue

    i_idx = np.clip(np.searchsorted(edges_bin, x[sel], side="right") - 1, 0, BIN_NX_PER_LEVEL - 1)
    j_idx = np.clip(np.searchsorted(edges_bin, y[sel], side="right") - 1, 0, BIN_NX_PER_LEVEL - 1)
    k_idx = np.clip(np.searchsorted(edges_bin, z[sel], side="right") - 1, 0, BIN_NX_PER_LEVEL - 1)
    flat = (i_idx * BIN_NX_PER_LEVEL + j_idx) * BIN_NX_PER_LEVEL + k_idx
    nbins = BIN_NX_PER_LEVEL**3

    m_sel = mass[sel]
    mass_sum = np.bincount(flat, weights=m_sel, minlength=nbins)
    vx_sum = np.bincount(flat, weights=m_sel * vx[sel], minlength=nbins)
    vy_sum = np.bincount(flat, weights=m_sel * vy[sel], minlength=nbins)
    vz_sum = np.bincount(flat, weights=m_sel * vz[sel], minlength=nbins)
    eint_sum = np.bincount(flat, weights=m_sel * eint[sel], minlength=nbins)
    n_cells_bin = np.bincount(flat, minlength=nbins)

    occ = np.flatnonzero(n_cells_bin > 0)
    print("%5d  %16.4f  %16.4f  %11d  %5d/%-6d  %10.4e"
          % (L, hw, dcell_bin, n_sel, occ.size, nbins, mass_sum[occ].sum()))
    if occ.size == 0:
        continue

    ii, jj, kk = np.unravel_index(occ, (BIN_NX_PER_LEVEL, BIN_NX_PER_LEVEL, BIN_NX_PER_LEVEL))
    m_occ = mass_sum[occ]

    # mass-weighted averages, still in raw AREPO units (cm/s, erg/g) here
    vx_bin = vx_sum[occ] / m_occ
    vy_bin = vy_sum[occ] / m_occ
    vz_bin = vz_sum[occ] / m_occ
    eint_bin = eint_sum[occ] / m_occ

    # ---- apply the dimensionless mass-ratio scaling (see "VELOCITY AND
    #      INTERNAL-ENERGY SCALING" above) ----
    vx_bin = vx_bin * VELOCITY_SCALE
    vy_bin = vy_bin * VELOCITY_SCALE
    vz_bin = vz_bin * VELOCITY_SCALE
    eint_bin = eint_bin * EINT_SCALE

    # density in AREPO's original CGS units (g/cm^3), one value per coarse cell
    rho_coarse = (m_occ / V_cell_bin) / r_g_cm**3

    # ---- upsample: replicate each coarse cell's value across the
    #      UPSAMPLE_FACTOR^3 fine sub-cells that exactly tile it ----
    F = UPSAMPLE_FACTOR
    sub = np.arange(F)
    so_i, so_j, so_k = (a.ravel() for a in np.meshgrid(sub, sub, sub, indexing="ij"))  # length F^3 each

    fi = (ii[:, None] * F + so_i[None, :]).ravel()
    fj = (jj[:, None] * F + so_j[None, :]).ravel()
    fk = (kk[:, None] * F + so_k[None, :]).ravel()

    n_fine = fi.size  # = occ.size * F**3
    rho_fine = np.repeat(rho_coarse, F**3)
    vx_fine = np.repeat(vx_bin, F**3)
    vy_fine = np.repeat(vy_bin, F**3)
    vz_fine = np.repeat(vz_bin, F**3)
    eint_fine = np.repeat(eint_bin, F**3)
    # n_cells column reports the PARENT coarse cell's particle count for
    # every one of its fine sub-cells (there is no independent fine-cell
    # particle count -- see the two-stage binning note above)
    n_cells_fine = np.repeat(n_cells_bin[occ], F**3)

    # ---- convert position back to AREPO's original CGS units for the
    #      OUTPUT table (grid geometry above stays in r_g; only the
    #      written-out values change format here) ----
    rows.append(np.column_stack([
        np.full(n_fine, L),
        centers_fine[fi] * r_g_cm, centers_fine[fj] * r_g_cm, centers_fine[fk] * r_g_cm,
        rho_fine, vx_fine, vy_fine, vz_fine, eint_fine, n_cells_fine,
    ]))

table = np.concatenate(rows, axis=0)

# ============================================================================
# INTERIOR ZEROING (see "INTERIOR ZEROING" above for what/why) -- applied here
# on the fully assembled table. Cell centers are in cm (output units) at this
# point, so they are converted back to r_g to compare against the threshold.
# ============================================================================
if ZERO_INNER_REGION:
    r_cell_rg = np.sqrt(table[:, 1]**2 + table[:, 2]**2 + table[:, 3]**2) / r_g_cm
    inner = r_cell_rg < ZERO_INNER_R_RG
    table[inner, 4] = 0.0   # rho
    table[inner, 5] = 0.0   # vx
    table[inner, 6] = 0.0   # vy
    table[inner, 7] = 0.0   # vz
    table[inner, 8] = 0.0   # eint
    print("ZERO_INNER_REGION: zeroed rho/v/eint for %d / %d cells with r < %.1f r_g"
          % (inner.sum(), table.shape[0], ZERO_INNER_R_RG))

n_binned_total = int((level_of >= 0).sum())
mass_total = mass[level_of >= 0].sum()
ppc = table[:, 9]
print("\nBinned %d / %d gas cells (%d outside the +/-%.1f r_g root box)"
      % (n_binned_total, len(x), len(x) - n_binned_total, ROOT_HALF_WIDTH_RG))
print("Total mass in grid: %.6e g (snapshot total: %.6e g)" % (mass_total, mass.sum()))
print("Particles per occupied cell: min=%d, median=%.1f, mean=%.1f, max=%d"
      % (ppc.min(), np.median(ppc), ppc.mean(), ppc.max()))
print("Fraction of occupied cells with only 1 particle: %.1f%%" % (100.0 * (ppc == 1).mean()))

header = (
    "bhpos_cm = %.8e, %.8e, %.8e ; r_sink_cm = %.8e ; "
    "bhmass_snapshot_g = %.8e (UNUSED, for reference only) ; "
    "M_BH_new_g = %.8e ; M_BH_new_msun = %.8e ; TARGET_RS_RSUN = %.8e ; r_g_cm = %.8e ; "
    "M_BH_old_msun_reference = %.8e ; MASS_RATIO = %.8e ; "
    "VELOCITY_SCALE_sqrt_ratio = %.8e ; EINT_SCALE_ratio = %.8e ; "
    "ROOT_HALF_WIDTH_RG = %.8e ; n_levels = %d ; nx_per_level = %d ; "
    "BIN_NX_PER_LEVEL = %d ; UPSAMPLE_FACTOR = %d ; "
    "ZERO_INNER_REGION = %s ; ZERO_INNER_R_RG = %.8e ; "
    "NOTE: particles are binned at BIN_NX_PER_LEVEL then mass-conservatively "
    "upsampled (replicated) onto the nx_per_level output grid -- see the "
    "two-stage binning note in the script; n_cells is the PARENT coarse "
    "cell's particle count, repeated for all its fine sub-cells. Grid "
    "geometry is built in r_g, but columns below are back in AREPO's "
    "original CGS units (cm, g/cm^3, cm/s, erg/g) -- v/eint still carry "
    "the (5) mass-ratio scaling\n"
    "level,x_center_cm,y_center_cm,z_center_cm,"
    "rho_g_cm3,vx_scaled_cm_s,vy_scaled_cm_s,vz_scaled_cm_s,eint_scaled_erg_g,n_cells"
    % (bhpos[0], bhpos[1], bhpos[2], r_sink, bhmass_snapshot,
       M_BH_g, M_BH_msun, TARGET_RS_RSUN, r_g_cm,
       M_BH_OLD_MSUN, MASS_RATIO, VELOCITY_SCALE, EINT_SCALE,
       ROOT_HALF_WIDTH_RG, n_levels, nx_per_level,
       BIN_NX_PER_LEVEL, UPSAMPLE_FACTOR,
       str(ZERO_INNER_REGION), ZERO_INNER_R_RG)
)
fmt = ["%d"] + ["%.8e"] * 3 + ["%.8e"] * 5 + ["%d"]
with open(outfile, "w") as fh:
    fh.write("# " + header + "\n")
    np.savetxt(fh, table, delimiter=",", fmt=fmt)

print("\nWrote %s" % outfile)
print("M_BH_new = %.6e g = %.4f M_sun (target r_s = %.4f R_sun -> r_g = %.6e cm = %.6e R_sun)"
      % (M_BH_g, M_BH_msun, TARGET_RS_RSUN, r_g_cm, r_g_cm / unit_rsun))
print("mass ratio M_BH_new/M_BH_old(%.1f Msun) = %.4f  ->  VELOCITY_SCALE=%.4f, EINT_SCALE=%.4f"
      % (M_BH_OLD_MSUN, MASS_RATIO, VELOCITY_SCALE, EINT_SCALE))
print("(AREPO snapshot's own BH mass was %.4f M_sun -- NOT used above, per requirement (1))"
      % (bhmass_snapshot / msun_cgs))
