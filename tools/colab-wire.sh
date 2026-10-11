#!/usr/bin/env bash
# colab-wire.sh — привязать Colab-сервер к Hermes (обновляет провайдера "colab" в конфиге).
#
# Использование:
#   bash colab-wire.sh                 # вставить блок из Colab (URL/KEY/MODEL), Ctrl-D
#   bash colab-wire.sh '<блок>'        # то же, блок аргументом
#   bash colab-wire.sh --comfy '<блок>'  # сохранить URL ComfyUI
set -euo pipefail

HERMES_BIN="${HERMES_BIN:-}"
if [ -z "$HERMES_BIN" ]; then
  if command -v hermes >/dev/null 2>&1; then HERMES_BIN="$(command -v hermes)"
  else HERMES_BIN="$HOME/.local/bin/hermes"; fi
fi

MODE="llm"
if [ "${1:-}" = "--comfy" ]; then MODE="comfy"; shift; fi

INPUT="${1:-}"
if [ -z "$INPUT" ]; then
  echo "Вставь блок из Colab (строки URL/KEY/MODEL) и нажми Ctrl-D:"
  INPUT="$(cat)"
fi

URL="$(printf '%s\n' "$INPUT" | sed -n 's/^URL:[[:space:]]*//p' | head -1 | tr -d '[:space:]')"
KEY="$(printf '%s\n' "$INPUT" | sed -n 's/^KEY:[[:space:]]*//p' | head -1 | tr -d '[:space:]')"
MODEL="$(printf '%s\n' "$INPUT" | sed -n 's/^MODEL:[[:space:]]*//p' | head -1 | tr -d '[:space:]')"

if [ -z "$URL" ]; then
  echo "❌ Не нашёл строку 'URL:' в блоке. Пришли ровно тот блок, что печатает Colab." >&2
  exit 1
fi

mkdir -p "$HOME/.hermes/colab"

if [ "$MODE" = "comfy" ]; then
  printf '%s\n' "$URL" > "$HOME/.hermes/colab/COMFY_URL.txt"
  echo "→ ComfyUI URL сохранён: ~/.hermes/colab/COMFY_URL.txt"
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 30 "$URL/system_stats" || true)"
  if [ "$code" = "200" ]; then echo "✅ ComfyUI отвечает через туннель."; else echo "⚠️ ComfyUI не ответил (HTTP $code). Проверь, что ноутбук в Colab ещё запущен."; fi
  exit 0
fi

[ -n "$MODEL" ] || MODEL="colab-llm"
URL="${URL%/}"
echo "→ URL:   $URL"
echo "→ MODEL: $MODEL"

"$HERMES_BIN" config set providers.colab.api "$URL/v1"
"$HERMES_BIN" config set providers.colab.default_model "$MODEL"
"$HERMES_BIN" config set providers.colab.key_env COLAB_API_KEY
"$HERMES_BIN" config set providers.colab.context_length 65536
if [ -n "$KEY" ]; then "$HERMES_BIN" config set COLAB_API_KEY "$KEY"; fi

echo "→ Проверяю связь..."
HTTP_CODE="$(curl -s -o "$HOME/.hermes/colab/.models.json" -w '%{http_code}' --max-time 30 \
  -H "Authorization: Bearer ${KEY:-no-key}" "$URL/v1/models" || true)"
if [ "$HTTP_CODE" = "200" ]; then
  echo "✅ Сервер отвечает. Модели:"
  python3 - "$HOME/.hermes/colab/.models.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("   " + ", ".join(m.get("id", "?") for m in d.get("data", [])))
PY
else
  echo "⚠️ HTTP $HTTP_CODE — сервер не отвечает. Проверь, что ноутбук в Colab ещё запущен."
fi

echo
echo "Готово. В Hermes: /model custom:colab:$MODEL"
echo "Или проверка из терминала: hermes chat -q 'привет' --provider colab -m $MODEL"
