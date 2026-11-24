#!/usr/bin/env bash
set -Eeuo pipefail

trap 'printf "managed_code_test.sh:%s: command failed (%s): %s\n" "$LINENO" "$?" "$BASH_COMMAND" >&2' ERR

kidi=$1
fixture=$2
curl=$3
standalone=${4:-}
root=$(mktemp -d)
export XDG_RUNTIME_DIR="$root"
export XDG_CACHE_HOME="$root/cache"
export KIDI_MODEL="$root/model"
export KIDI_LOG_LEVEL=info
unset KIDI_AGENT_BASE_URL

read_state() {
    state=$(find "$root" -name server.json -type f)
    test -f "$state" || return 1
    read -r pid port < <(sed -nE 's/^\{"port":([0-9]+),"pid":([0-9]+)\}$/\2 \1/p' "$state")
}

cleanup() {
    while IFS= read -r state; do
        read -r pid port < <(sed -nE 's/^\{"port":([0-9]+),"pid":([0-9]+)\}$/\2 \1/p' "$state")
        if [[ -n ${pid:-} && -n ${port:-} ]] && [[ $("$curl" -fsS --noproxy '*' --max-time 2 "http://127.0.0.1:$port/kidi-code/ready" 2>/dev/null) == "$pid" ]]; then
            kill "$pid" || true
        fi
    done < <(find "$root" -name server.json -type f)
    find "$root" -depth -delete
}
trap cleanup EXIT
trap 'exit 1' INT TERM

mkdir "$KIDI_MODEL"
cp "$fixture/"{config.json,model.safetensors} "$KIDI_MODEL/"
cat > "$KIDI_MODEL/tokenizer.json" <<'JSON'
{
  "version":"1.0", "pre_tokenizer":{"type":"WhitespaceSplit"},
  "decoder":{"type":"WordPiece","prefix":"##","cleanup":false},
  "model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
    "<pad>":0,"<eos>":1,"<bos>":2,"<turn|>":3,"<|turn>":4,"<unk>":5,
    "alpha":6,"beta":7,"gamma":8,"delta":9,"theta":10,"zeta":11,"eta":12,"iota":13,"kappa":14,"lambda":15}}
}
JSON
printf '{}\n' > "$KIDI_MODEL/tokenizer_config.json"
printf '{"eos_token_id":4}\n' > "$KIDI_MODEL/generation_config.json"
cat > "$KIDI_MODEL/chat_template.jinja" <<'JINJA'
{%- if 'You are Kidi Code' in messages[0].content and tools and tools[0].function.name == 'read_file' -%}
theta
{%- elif messages[0].content == 'custom prompt' and not tools -%}
theta
{%- else -%}
{{ raise_exception('wrong embedded or overridden agent configuration') }}
{%- endif -%}
JINJA
printf 'version: 1\nprompts:\n  system: custom prompt\ntools: []\n' > "$root/agent.yaml"
# The random-weight fixture needs greedy decoding for deterministic text and stopping.
run_code() {
    "$kidi" code --backend ynnpack --workspace "$root" --timeout-ms 5000 --max-tokens 8 --temperature 0 -p "$1" "${@:2}"
}

# Concurrent clients have independent configurations and share one cold start.
run_code first > "$root/first" 2> "$root/first.log" &
first=$!
run_code second --config "$root/agent.yaml" > "$root/second" 2> "$root/second.log" &
second=$!
status=0
wait "$first" || status=1
wait "$second" || status=1
if (( status )); then
    cat "$root/first.log" "$root/second.log"
    find "$root" -name serve.log -exec cat {} \;
    exit 1
fi

test "$(cat "$root/first")" = lambda
test "$(cat "$root/second")" = lambda
test "$(grep -h -c 'starting background inference' "$root/"{first,second}.log | sort | tr '\n' ' ')" = '0 1 '
read_state
test "$("$curl" -fsS --noproxy '*' --max-time 2 "http://127.0.0.1:$port/health")" = '{}'
test "$("$curl" -fsS --noproxy '*' --max-time 2 "http://127.0.0.1:$port/v1/chat/context" \
    -H 'Content-Type: application/json' -d '{"messages":[{"role":"system","content":"You are Kidi Code"}],
    "tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}]}')" \
    = '{"prompt_tokens":1,"context_tokens":16,"prompt_limit_tokens":15}'
grep -q 'inference device: cpu:0 (backend=ynnpack)' "$(dirname "$state")/serve.log"
previous_pid=$pid
test "$(run_code third)" = lambda
read_state
test "$pid" = "$previous_pid"
if [[ -n $standalone ]]; then
    test "$("$standalone" --base-url "http://127.0.0.1:$port/v1" --workspace "$root" \
        --max-tokens 8 --temperature 0 --timeout-ms 5000 -p standalone)" = lambda
fi

# A crashed daemon leaves stale metadata. The next client replaces it.
kill -KILL "$pid"
for ((i = 0; i < 50; ++i)); do
    if ! "$curl" -fsS --noproxy '*' --max-time 1 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then break; fi
    sleep .1
done
test "$(run_code fourth)" = lambda
read_state
test "$pid" != "$previous_pid"

# --restart replaces a healthy daemon instead of reusing it.
previous_pid=$pid
test "$(run_code fifth --restart 2> "$root/restart.log")" = lambda
grep -q "stopping background inference (PID $previous_pid)" "$root/restart.log"
read_state
test "$pid" != "$previous_pid"
for ((i = 0; i < 50; ++i)); do kill -0 "$previous_pid" 2>/dev/null || break; sleep .1; done
! kill -0 "$previous_pid" 2>/dev/null
# Optional wall-clock verification of the full production idle timeout.
if [[ ${KIDI_TEST_REAL_IDLE:-0} == 1 ]]; then
    started=$SECONDS
    while [[ -f $state ]] && (( SECONDS - started < 610 )); do sleep 1; done
    test ! -f "$state"
    (( SECONDS - started >= 599 ))
fi

# Different CUDA visibility settings must not reuse the same auto-backend daemon.
# With CUDA hidden, auto selects CPU on Linux and may select Metal on macOS.
auto_root="$root/auto-runtime"
mkdir -m 700 "$auto_root"
for visible in '' '-1'; do
    test "$(XDG_RUNTIME_DIR="$auto_root" CUDA_VISIBLE_DEVICES="$visible" "$kidi" code \
        --workspace "$root" --timeout-ms 5000 --max-tokens 8 --temperature 0 -p auto)" = lambda
done
test "$(find "$auto_root" -name server.json -type f | wc -l)" -eq 2
auto_device='cpu:0'
if [[ $(uname -s) == Darwin ]]; then auto_device='(cpu:0|a_gpu:0)'; fi
test "$(grep -lE "inference device: $auto_device \(backend=auto\)" "$auto_root"/kidi-code-*/*/serve.log | wc -l)" -eq 2

# Opt-in tiny-model CUDA smoke test; the caller selects an idle physical GPU.
if [[ -n ${KIDI_TEST_AUTO_CUDA:-} ]]; then
    test "$(XDG_RUNTIME_DIR="$auto_root" CUDA_VISIBLE_DEVICES="$KIDI_TEST_AUTO_CUDA" "$kidi" code \
        --workspace "$root" --timeout-ms 30000 --max-tokens 8 --temperature 0 -p auto-cuda)" = lambda
    test "$(grep -l 'inference device: cuda:0 (backend=auto)' "$auto_root"/kidi-code-*/*/serve.log | wc -l)" -eq 1
fi

# Real inference with a tiny context: accumulate tasks, compact and continue.
# The template measures long text without requiring the tiny fixture to
# understand summaries; summarization still runs through actual inference.
compact_runtime="$root/compact-runtime"
compact_model="$root/compact-model"
mkdir -m 700 "$compact_runtime" "$compact_model"
cp "$KIDI_MODEL/"* "$compact_model/"
cat > "$compact_model/chat_template.jinja" <<'JINJA'
{%- if messages[0].content == 'Summarize factual progress' -%}
theta
{%- else -%}
{%- for m in messages -%}
{%- if m.role != 'system' -%}
{%- for unused in range(1 + (m.content | length) // 1024) -%}theta {% endfor -%}
{%- endif -%}
{%- endfor -%}theta
{%- endif -%}
JINJA
printf 'version: 1\nprompts: {system: compacttest, compact: Summarize factual progress}\ntools: []\n' \
    > "$root/compact.yaml"
compact_backend=ynnpack
if [[ -n ${KIDI_TEST_AUTO_CUDA:-} ]]; then compact_backend=auto; fi
{
    for ((i = 0; i < 9; ++i)); do
        printf 'Task %s ' "$i"
        printf 'alpha %.0s' {1..650}
        printf '\n'
    done
    printf '/context\n/compact\n/context\n/exit\n'
} | XDG_RUNTIME_DIR="$compact_runtime" CUDA_VISIBLE_DEVICES="${KIDI_TEST_AUTO_CUDA:-}" \
    "$kidi" code --model "$compact_model" --backend "$compact_backend" \
    --workspace "$root" --config "$root/compact.yaml" --context-size 16 --max-tokens 8 --temperature 0 --timeout-ms 5000 \
    > "$root/compact.out" 2> "$root/compact.log"
if [[ $(wc -l < "$root/compact.out") -ne 9 ]] || grep -q 'kidi-code:' "$root/compact.log"; then
    cat "$root/compact.log"
    exit 1
fi
test "$(grep -c 'Compacting conversation' "$root/compact.log")" -ge 2
grep -q 'prompt tokens' "$root/compact.log"
grep -q '/ 16 context tokens' "$root/compact.log"
test "$(grep -c 'warning: 16-token context' "$root/compact.log")" -eq 1
compact_session=$(grep -l '^{"type":"compact"' "$XDG_CACHE_HOME"/kidi/code/chats/*/*.jsonl)
test -f "$compact_session"
test "$(grep -c '^{"type":"compact"' "$compact_session")" -ge 2
grep -q '"is_summary":true' "$compact_session"
grep -q '^{"type":"message","role":"user","content":"Task 8 ' "$compact_session"

# Context settings belong to server identity, even if the effective default
# happens to equal the explicitly requested setting.
XDG_RUNTIME_DIR="$compact_runtime" CUDA_VISIBLE_DEVICES="${KIDI_TEST_AUTO_CUDA:-}" \
    "$kidi" code --model "$compact_model" --backend "$compact_backend" \
    --workspace "$root" --config "$root/compact.yaml" --max-tokens 8 --temperature 0 --timeout-ms 5000 -p short \
    > "$root/compact-default.out"
test "$(find "$compact_runtime" -name server.json -type f | wc -l)" -eq 2
for state in "$compact_runtime"/kidi-code-*/*/server.json; do
    read -r pid port < <(sed -nE 's/^\{"port":([0-9]+),"pid":([0-9]+)\}$/\2 \1/p' "$state")
    test "$("$curl" -fsS --noproxy '*' --max-time 2 "http://127.0.0.1:$port/v1/chat/context" \
        -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"short"}]}')" \
        = '{"prompt_tokens":2,"context_tokens":16,"prompt_limit_tokens":15}'
done
# Without --context-size the daemon sizes the context from free memory when the backend can report it.
# TODO: The CPU backend does not report available memory on macOS, so it uses the model's default.
if [[ $(uname -s) != Darwin ]]; then
    test "$(grep -l 'context: [0-9]* tokens (' "$compact_runtime"/kidi-code-*/*/serve.log | wc -l)" -eq 1
fi
