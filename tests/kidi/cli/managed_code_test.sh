#!/usr/bin/env bash
set -euo pipefail

kidi=$1
fixture=$2
curl=$3
standalone=${4:-}
root=$(mktemp -d)
export XDG_RUNTIME_DIR="$root"
export KIDI_MODEL="$root/model"
export KIDI_LOG_LEVEL=info
unset KIDI_AGENT_BASE_URL

read_state() {
    state=$(find "$root" -name server.json -type f)
    test -f "$state" || return 1
    read -r pid port < <(sed -nE 's/^\{"port":([0-9]+),"pid":([0-9]+)\}$/\2 \1/p' "$state")
}

cleanup() {
    if read_state && [[ $("$curl" -fsS --noproxy '*' --max-time 2 "http://127.0.0.1:$port/kidi-code/ready" 2>/dev/null) == "$pid" ]]; then
        kill "$pid" || true
    fi
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
run_code() {
    "$kidi" code --backend ynnpack --workspace "$root" --timeout-ms 5000 --max-tokens 8 -p "$1" "${@:2}"
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
previous_pid=$pid
test "$(run_code third)" = lambda
read_state
test "$pid" = "$previous_pid"
if [[ -n $standalone ]]; then
    test "$("$standalone" --base-url "http://127.0.0.1:$port/v1" --workspace "$root" \
        --max-tokens 8 --timeout-ms 5000 -p standalone)" = lambda
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

# Optional wall-clock verification of the full production idle timeout.
if [[ ${KIDI_TEST_REAL_IDLE:-0} == 1 ]]; then
    started=$SECONDS
    while [[ -f $state ]] && (( SECONDS - started < 610 )); do sleep 1; done
    test ! -f "$state"
    (( SECONDS - started >= 599 ))
fi
