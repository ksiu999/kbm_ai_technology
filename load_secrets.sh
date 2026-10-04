#!/bin/bash
# Скрипт для безопасной загрузки secrets из .env.sops
# Расшифровывает файл и загружает переменные окружения

export SOPS_AGE_KEY_FILE="${HOME}/.config/sops/age/keys.txt"

# Расшифровываем .env.sops во временный файл
sops -d .env.sops > .env.tmp || { echo "ERROR: Не удалось расшифровать .env.sops"; exit 1; }

# Загружаем переменные окружения
set -a
source .env.tmp
set +a

# Удаляем временный файл
rm .env.tmp

# Выполняем переданную команду с загруженными переменными
exec "$@"