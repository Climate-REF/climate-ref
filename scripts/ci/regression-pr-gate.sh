#!/usr/bin/env bash
# PR-tier regression baseline gate.
#
# Runs the coupling gate (`ref test-cases ci-gate`) against the pull request's base branch and,
# for every case it routes to `replay`, re-checks the committed native baseline.
#
# Neither stage needs input data.
# `replay` rebuilds the committed bundle from native output blobs pulled from the public
# store (plus the committed catalog/manifest).
#
# Usage:
#   make regression-gate                          # the convenient local entry point
#   bash scripts/ci/regression-pr-gate.sh [base]  # or call directly, optionally overriding the base ref
#
#   base             ref to diff against. Defaults to origin/${GITHUB_BASE_REF:-main}.
#   GITHUB_BASE_REF  the PR's base branch (set automatically on pull_request events).
#
# Running under GitHub Actions emits log groups and ::error::/::warning:: annotations;
# locally it prints plain, readable output instead.
# Run it before pushing to catch a coupling violation or replay drift without waiting for CI.
set -euo pipefail

# Rich colour codes would corrupt the JSON we parse with jq
export NO_COLOR=1

# GitHub Actions log-grouping and annotation commands are just noise on a local terminal,
# so emit them only under Actions and print readable equivalents otherwise.
if [ -n "${GITHUB_ACTIONS:-}" ]; then
  begin_group() { echo "::group::$*"; }
  end_group() { echo "::endgroup::"; }
  error() { echo "::error::$*"; }
  warning() { echo "::warning::$*"; }
else
  begin_group() { printf '\n==> %s\n' "$*"; }
  end_group() { :; }
  error() { printf 'ERROR: %s\n' "$*" >&2; }
  warning() { printf 'WARNING: %s\n' "$*" >&2; }
fi

# CI passes the base via GITHUB_BASE_REF (and fetches origin/<ref> beforehand); locally,
# pass one as the first argument or rely on the origin/main default.
base="${1:-origin/${GITHUB_BASE_REF:-main}}"

begin_group "coupling gate (base: ${base})"
# `ci-gate` exits non-zero when any case is gated `fail`; capture the decisions
# regardless so the failing cases can be reported before the script exits.
set +e
decisions="$(uv run ref test-cases ci-gate --base "${base}" --json)"
gate_rc=$?
set -e
printf '%s\n' "${decisions}" | jq .
end_group

if [ "${gate_rc}" -ne 0 ]; then
  # `ci-gate` exits non-zero for two distinct reasons: at least one case was gated `fail`,
  # or a hard error (not a git repo, an unfetched base ref, a failed diff)
  # that printed no decisions. Only the former is a baseline-coupling violation, so
  # report each kind accurately rather than blaming every failure on the author.
  fail_cases="$(printf '%s\n' "${decisions}" | jq -r '.[]? | select(.action == "fail") | "  FAIL " + .case + " -- " + .reason')"
  if [ -n "${fail_cases}" ]; then
    error "Coupling gate failed -- a baseline changed without an authorised test_case_version bump."
    echo "${fail_cases}"
  else
    error "ci-gate exited ${gate_rc} without a fail decision -- see the gate output above (e.g. an unfetched base ref or a non-git checkout)."
  fi
  exit 1
fi


while IFS= read -r execute_case; do
  [ -n "${execute_case}" ] || continue
  warning "${execute_case} bumped test_case_version with no native baseline -- review the committed bundle, then mint on the trusted tier to enable replay verification."
done < <(printf '%s\n' "${decisions}" | jq -r '.[] | select(.action == "execute") | .case')

replay_cases="$(printf '%s\n' "${decisions}" | jq -r '.[] | select(.action == "replay") | .case')"
if [ -z "${replay_cases}" ]; then
  echo "No cases require replay; baseline gate passed."
  exit 0
fi


# Replay every case the gate routed to REPLAY.
failures=0
replayed=0
while IFS='/' read -r provider diagnostic test_case; do
  [ -n "${provider}" ] || continue
  begin_group "replay ${provider}/${diagnostic}/${test_case}"
  if uv run ref test-cases replay \
      --provider "${provider}" \
      --diagnostic "${diagnostic}" \
      --test-case "${test_case}"; then
    replayed=$((replayed + 1))
  else
    error "replay drift for ${provider}/${diagnostic}/${test_case}"
    failures=$((failures + 1))
  fi
  end_group
done < <(printf '%s\n' "${replay_cases}")

echo "Replayed ${replayed} case(s); ${failures} drifted."
if [ "${failures}" -ne 0 ]; then
  exit 1
fi
