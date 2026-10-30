#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# iCloud preflight — DO NOT REMOVE. This folder is iCloud-synced; "Optimize Mac
# Storage" evicts file contents ("dataless" files), and any long process then
# blocks for MINUTES on first-touch reads while it looks like a hang.
# Materialize everything up front. Shared by every launcher — source or run
# from the project root.
# Durable fix (do this too, not instead): Finder → right-click the project
# folder → "Keep Downloaded".
N=$(find . -flags +dataless 2>/dev/null | wc -l | tr -d ' ')
if [ "${N:-0}" -gt 0 ]; then
  echo "iCloud has offloaded $N project files — downloading them before launch…"
  brctl download . 2>/dev/null || true
  # brctl alone sometimes does nothing — force-reading the stubs always works
  find . -flags +dataless -print0 2>/dev/null | xargs -0 -n 25 -P 8 cat > /dev/null 2>&1 || true
  for _ in $(seq 1 120); do
    N=$(find . -flags +dataless 2>/dev/null | wc -l | tr -d ' ')
    [ "${N:-0}" = "0" ] && break
    printf '\r  %s files remaining…   ' "$N"
    sleep 5
  done
  echo ""
  echo "All files local — continuing."
fi
