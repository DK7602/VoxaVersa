#!/bin/bash
# Text to Sing takes for the vowel recognizer, sung by AI singers with the voice's effects off (the plug-in's
# recognizer hears a dry vocal; takes through reverb and compression taught it the wrong thing). Each voice
# sings 6-line pieces of lyrics.txt and one of lyrics_open.txt; the melody plan (--segments) gives the labels.
# Use only singers whose recordings allow it (public-domain band singers, VocalSet: CC BY 4.0), never
# sample-pack singers: what they sing ends up in the shipped weights.
# usage: render_takes.sh VoxaVersaRender out_dir worker workers voice...   (HOME: a config with those voices)
set -u
R=$(realpath "$1"); OUT=$2; WORKER=$3; WORKERS=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/pieces" && cd "$OUT"
split -l 6 -d -a 1 "$HERE/lyrics.txt" pieces/lyrics; split -l 6 -d -a 1 "$HERE/lyrics_open.txt" pieces/open
k=0; i=0
for v in "$@"; do
  i=$((i + 1))
  for piece in lyrics$((i % 5)) lyrics$(((i + 2) % 5)) lyrics$(((i + 4) % 5)) open$((i % 5)); do
    k=$((k + 1)); [ $((k % WORKERS)) -eq "$WORKER" ] || continue
    name=$v-$piece; [ -f "$name.wav" ] && continue
    cp "pieces/$piece" "lyr-$name.txt"
    timeout 900 "$R" --sing "lyr-$name.txt" "guide-$name.wav" --voice "$v" --ai --segments "plan-$name.tsv" --chain "$name.wav" \
      --bpm $((80 + (k * 13) % 40)) --seed $((1 + k % 4)) --set vibe=0 --set reverbMix=0 --set delayMix=0 \
      --set compressor=0 --set saturation=0 --set deEsser=0 --set doublerOn=0 --set harmonyOn=0 > "log-$name.txt" 2>&1
    echo "$(date +%T) $name"
  done
done
