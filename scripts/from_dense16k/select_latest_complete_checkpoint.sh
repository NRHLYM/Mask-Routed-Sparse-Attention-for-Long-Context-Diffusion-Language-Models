#!/usr/bin/env bash
# Pick the newest complete step-* checkpoint. Ignore *.incomplete.
# Exit 1 if step dirs exist but none are complete, so callers cannot silently
# reinitialize from dense-500.

checkpoint_is_complete() {
  [[ -f "$1/checkpoint_manifest.json" && -f "$1/trainable_state.pt" \
     && -f "$1/optimizer.pt" && -f "$1/scheduler.pt" && -f "$1/trainer_state.pt" ]]
}

select_latest_complete_checkpoint() {
  local output="$1"
  local resume_from=""
  local cand
  while IFS= read -r cand; do
    [[ -n "$cand" ]] || continue
    if checkpoint_is_complete "$cand"; then
      resume_from="$cand"
      break
    fi
  done < <(find "$output" -maxdepth 1 -type d -name 'step-*' ! -name '*.incomplete' -print 2>/dev/null | sort -V -r)
  if [[ -n "$resume_from" ]]; then
    printf '%s\n' "$resume_from"
    return 0
  fi
  if find "$output" -maxdepth 1 -type d \( -name 'step-*' -o -name 'step-*.incomplete' \) -print 2>/dev/null | grep -q .; then
    echo "found step directories under $output but none are complete; refusing to reinitialize from dense-500" >&2
    return 1
  fi
  return 0
}
