#!/usr/bin/env bash
# PR-tier regression baseline gate.
#
# Runs the coupling gate (`ref test-cases ci-gate`) against the pull request's base branch and,
# for every case it routes to `replay`, re-checks the cached native baseline.
#
# All baseline data can be fetched with public read access so this does not require credentials.
#
# `ci-gate` reads only manifests and the git diff, so it needs no input data.
# The (slower) sample-data and catalog fetch is therefore deferred until the gate hasfound at least one case that actually needs replaying.
# The common pull request touches no baselines and skips the download entirely.
#
# Environment:
#   GITHUB_BASE_REF  the PR's base branch (set automatically on pull_request events).
set -euo pipefail

base="origin/${GITHUB_BASE_REF:-main}"

echo "::group::coupling gate (base: ${base})"
# `ci-gate` exits non-zero when any case is gated `fail`; capture the decisions
# regardless so the failing cases can be reported before the script exits.
set +e
decisions="$(uv run ref test-cases ci-gate --base "${base}" --json)"
gate_rc=$?
set -e
echo "${decisions}" | jq .
echo "::endgroup::"

if [ "${gate_rc}" -ne 0 ]; then
  echo "::error::Coupling gate failed -- a baseline changed without an authorised test_case_version bump."
  echo "${decisions}" | jq -r '.[] | select(.action == "fail") | "  FAIL " + .case + " -- " + .reason'
  exit 1
fi


while IFS= read -r execute_case; do
  [ -n "${execute_case}" ] || continue
  echo "::warning::${execute_case} bumped test_case_version with no native baseline -- review the committed bundle, then mint on the trusted tier to enable replay verification."
done < <(echo "${decisions}" | jq -r '.[] | select(.action == "execute") | .case')

replay_cases="$(echo "${decisions}" | jq -r '.[] | select(.action == "replay") | .case')"
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
done < <(echo "${replay_cases}" | cut -d/ -f1 | sort -u)
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
