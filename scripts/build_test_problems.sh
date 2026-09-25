#!/usr/bin/env bash
# Builds a fixed list of pgen test problems, each into its own build directory
# at the repo root (build_<name>/), using:
#   cmake -D Athena_ENABLE_MPI=ON -D PROBLEM=<pgen/path/without/.cpp> ../
#   make -j10
#
# When PROBLEM is set to anything other than the default "built_in_pgens",
# CMake sets USER_PROBLEM_ENABLED=1 and CallProblemGenerator() only ever
# invokes ProblemGenerator::UserProblem(). Any pgen file that instead defines
# its own named function (e.g. ProblemGenerator::ShockTube) has to be
# temporarily renamed to UserProblem to compile under that path, then renamed
# back afterward so the next build (or the built_in_pgens default build) sees
# the original name again.
#
# A failed configure/build for one problem does not abort the whole script:
# the rename is always reverted, an error is reported, and the script moves
# on to the next problem. Overall exit status is nonzero if any problem failed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# problem_id (PROBLEM=... / pgen/<id>.cpp) : build directory name
PROBLEMS=(
  "tests/shock_tube:shock_tube"
  "fluids/shu_osher:shu_osher"
  "fluids/kh:kh"
  "fluids/rt:rt"
  "tests/lw_implode:lw_implode"
  "fluids/blast:blast"
  "tests/orszag_tang:orszag_tang"
)

FAILED=()

build_one() {
  local problem_id="$1" build_name="$2"
  local src_file="src/pgen/${problem_id}.cpp"
  local build_dir="build_${build_name}"

  if [[ ! -f "$src_file" ]]; then
    echo "!! Skipping ${problem_id}: ${src_file} not found" >&2
    return 1
  fi

  # Find the ProblemGenerator member function actually defined in this file.
  local orig_fn
  orig_fn=$(grep -oP 'ProblemGenerator::\K\w+(?=\(ParameterInput)' "$src_file" | head -1)
  if [[ -z "$orig_fn" ]]; then
    echo "!! Could not find a ProblemGenerator::<fn>(ParameterInput...) definition in ${src_file}" >&2
    return 1
  fi

  local renamed=0
  if [[ "$orig_fn" != "UserProblem" ]]; then
    echo ">> Renaming ProblemGenerator::${orig_fn} -> ProblemGenerator::UserProblem in ${src_file}"
    sed -i "s/ProblemGenerator::${orig_fn}(ParameterInput/ProblemGenerator::UserProblem(ParameterInput/" "$src_file"
    renamed=1
  fi

  echo "==> Building ${problem_id} in ${build_dir}/"
  mkdir -p "$build_dir"

  local rc=0
  (
    cd "$build_dir" &&
    cmake -D Athena_ENABLE_MPI=ON -D PROBLEM="${problem_id}" ../ &&
    make -j10
  )
  rc=$?

  # Always revert the rename, regardless of build success/failure.
  if [[ "$renamed" -eq 1 ]]; then
    echo ">> Reverting ProblemGenerator::UserProblem -> ProblemGenerator::${orig_fn} in ${src_file}"
    sed -i "s/ProblemGenerator::UserProblem(ParameterInput/ProblemGenerator::${orig_fn}(ParameterInput/" "$src_file"
  fi

  if [[ "$rc" -ne 0 ]]; then
    echo "!! Build failed for ${problem_id} (exit ${rc})" >&2
    return "$rc"
  fi

  echo "==> Done: ${problem_id}"
  return 0
}

for entry in "${PROBLEMS[@]}"; do
  IFS=':' read -r problem_id build_name <<< "$entry"
  if ! build_one "$problem_id" "$build_name"; then
    FAILED+=("$problem_id")
  fi
done

echo
if [[ "${#FAILED[@]}" -eq 0 ]]; then
  echo "All requested problems built successfully."
else
  echo "Failed problems: ${FAILED[*]}" >&2
  exit 1
fi
