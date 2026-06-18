#!/usr/bin/env bash
# PR-tier regression baseline gate.
#
# Runs the coupling gate (`ref test-cases ci-gate`) against the pull request's base branch and,
# for every case it routes to `replay`, re-checks the cached native baseline.
#
# All baseline data can be fetched with public read access so this does not require credentials.
#
# `ci-gate` reads only manifests and the git diff, so it needs no input data.
# The (slower) sample-data and catalog fetch is therefore deferred until the gate has found at least one case that actually needs replaying.
# The common pull request touches no baselines and skips the download entirely.
#
# Environment:
#   GITHUB_BASE_REF  the PR's base branch (set automatically on pull_request events).
set -euo pipefail

# Rich colour codes would corrupt the JSON we parse with jq. Force them off here so the
# script is self-contained and parseable even when run outside the workflow (which also
# sets NO_COLOR for the same reason).
export NO_COLOR=1

base="origin/${GITHUB_BASE_REF:-main}"

echo "::group::coupling gate (base: ${base})"
# `ci-gate` exits non-zero when any case is gated `fail`; capture the decisions
# regardless so the failing cases can be reported before the script exits.
set +e
decisions="$(uv run ref test-cases ci-gate --base "${base}" --json)"
gate_rc=$?
set -e
printf '%s\n' "${decisions}" | jq .
echo "::endgroup::"

if [ "${gate_rc}" -ne 0 ]; then
  # `ci-gate` exits non-zero for two distinct reasons: at least one case was gated
  # `fail`, or a hard error (not a git repo, an unfetched base ref, a failed diff)
  # that printed no decisions. Only the former is a baseline-coupling violation, so
  # report each kind accurately rather than blaming every failure on the author.
  fail_cases="$(printf '%s\n' "${decisions}" | jq -r '.[]? | select(.action == "fail") | "  FAIL " + .case + " -- " + .reason')"
  if [ -n "${fail_cases}" ]; then
    echo "::error::Coupling gate failed -- a baseline changed without an authorised test_case_version bump."
    echo "${fail_cases}"
  else
    echo "::error::ci-gate exited ${gate_rc} without a fail decision -- see the gate output above (e.g. an unfetched base ref or a non-git checkout)."
  fi
  exit 1
fi


while IFS= read -r execute_case; do
  [ -n "${execute_case}" ] || continue
  echo "::warning::${execute_case} bumped test_case_version with no native baseline -- review the committed bundle, then mint on the trusted tier to enable replay verification."
done < <(printf '%s\n' "${decisions}" | jq -r '.[] | select(.action == "execute") | .case')

replay_cases="$(printf '%s\n' "${decisions}" | jq -r '.[] | select(.action == "replay") | .case')"
if [ -z "${replay_cases}" ]; then
  echo "No cases require replay; baseline gate passed."
  exit 0
fi

# Only now -- with replay work to do -- pay for the input data the replay needs.
echo "::group::fetch inputs for replay"
uv run ref datasets fetch-sample-data
while IFS= read -r provider; do
  [ -n "${provider}" ] || continue
  uv run ref test-cases fetch --provider "${provider}" \
    || echo "::warning::test-case fetch incomplete for ${provider}"
done < <(printf '%s\n' "${replay_cases}" | cut -d/ -f1 | sort -u)
echo "::endgroup::"

# Replay every case the gate routed to REPLAY. Process substitution (rather than a
# pipe) keeps the loop in the current shell so the failure counter survives.
failures=0
replayed=0
while IFS='/' read -r provider diagnostic test_case; do
  [ -n "${provider}" ] || continue
  echo "::group::replay ${provider}/${diagnostic}/${test_case}"
  if uv run ref test-cases replay \
      --provider "${provider}" \
      --diagnostic "${diagnostic}" \
      --test-case "${test_case}"; then
    replayed=$((replayed + 1))
  else
    echo "::error::replay drift for ${provider}/${diagnostic}/${test_case}"
    failures=$((failures + 1))
  fi
  echo "::endgroup::"
done < <(printf '%s\n' "${replay_cases}")

echo "Replayed ${replayed} case(s); ${failures} drifted."
if [ "${failures}" -ne 0 ]; then
  exit 1
fi
