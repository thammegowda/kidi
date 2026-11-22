#!/bin/sh
# CTest GPU scheduling helper, sized to the GPUs this host exposes.
#   gpu-launch.sh --spec FILE   writes the CTest resource spec: one `gpus` id per visible GPU,
#                               capped at 8. A single-GPU host gets 2 slots on that GPU so
#                               two-GPU tests stay schedulable; a GPU-less host gets an empty
#                               pool, so CTest reports GPU tests as Not Run.
#   gpu-launch.sh CMD...        runs CMD pinned via CUDA_VISIBLE_DEVICES to the GPUs CTest
#                               allocated; without an allocation it runs CMD unchanged.
# Visible GPUs are the caller's CUDA_VISIBLE_DEVICES list, or all nvidia-smi GPUs when unset.
set -e
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    visible=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ')
else
    visible=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)
fi
total=$(echo $visible | wc -w)

if [ "$1" = "--spec" ]; then
    pool=$((total < 8 ? total : 8))
    slots=1
    if [ "$pool" -eq 1 ]; then slots=2; fi
    ids=
    i=0
    while [ "$i" -lt "$pool" ]; do
        ids="${ids:+$ids, }{ \"id\": \"$i\", \"slots\": $slots }"
        i=$((i + 1))
    done
    printf '{ "version": { "major": 1, "minor": 0 }, "local": [ { "gpus": [ %s ] } ] }\n' "$ids" > "$2"
    exit 0
fi

count=${CTEST_RESOURCE_GROUP_COUNT:-0}
if [ "$count" -gt 0 ] && [ "$total" -gt 0 ]; then
    selected=
    group=0
    while [ "$group" -lt "$count" ]; do
        # CTEST_RESOURCE_GROUP_<n>_GPUS looks like `id:3,slots:1`.
        spec=$(printenv "CTEST_RESOURCE_GROUP_${group}_GPUS" || true)
        id=$(echo "$spec" | sed -n 's/^id:\([0-9]*\),.*/\1/p')
        if [ -n "$id" ]; then
            gpu=$(echo $visible | cut -d' ' -f$((id % total + 1)))
            case ",$selected," in *",$gpu,"*) ;; *) selected=${selected:+$selected,}$gpu ;; esac
        fi
        group=$((group + 1))
    done
    [ -n "$selected" ] && export CUDA_VISIBLE_DEVICES="$selected"
fi
exec "$@"
