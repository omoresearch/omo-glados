#!/usr/bin/env bash
# Build the Omo GLaDOS client mod with plain javac (MC 26.2 ships unobfuscated). No Gradle.
#   ./build.sh                 build only (build/glados-1.0.0+mc26.2.jar)
#   ./build.sh test [URL...]   build + standalone harness (test/) against each sidecar URL (default http://127.0.0.1:8794).
#                              Without a sidecar: `python3 test/fake_sidecar.py 8789` (v3) + `... 8788 --v1` (old format),
#                              then ./build.sh test http://127.0.0.1:8789 http://127.0.0.1:8788
#   ./build.sh voicetest [ws://127.0.0.1:8793/svc a.pcm ...]   PCM/Opus/config/focus tests (+ live run against your backend)
#   ./build.sh install         build + install into $MC_MODS (takes effect next launch)
#
# Paths (override with env vars; defaults match an Omo Space install):
#   JDK_BIN        dir with javac/jar/java (JDK 21+)   default: $JAVA_HOME/bin, else ~/.local/share/omo/toolchain/jdk25/bin, else PATH
#   MC_LIBRARIES   launcher libraries dir holding minecraft-26.2-client.jar etc. (Prism/MultiMC "libraries")
#                  default: $PRISM/libraries, else ~/.local/share/omo/runtime/prism/libraries
#   MC_MODS        the instance's mods/ dir (fabric-api-*.jar is read from here; install target)
#   FABRIC_API_JAR explicit fabric-api jar (default: first $MC_MODS/fabric-api-*.jar)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
J="${JDK_BIN:-${JAVA_HOME:+$JAVA_HOME/bin}}"
J="${J:-$HOME/.local/share/omo/toolchain/jdk25/bin}"
[ -x "$J/javac" ] || J="$(dirname "$(command -v javac)")"
L="${MC_LIBRARIES:-${PRISM:+$PRISM/libraries}}"
L="${L:-$HOME/.local/share/omo/runtime/prism/libraries}"
MODS="${MC_MODS:-$HOME/.local/share/omo/runtime/game/mods}"
FAPI="${FABRIC_API_JAR:-$(ls "$MODS"/fabric-api-*.jar | head -1)}"
BACKUPS="${MC_BACKUPS:-$(dirname "$MODS")/../backups}"
NAME=glados-1.0.0+mc26.2.jar

# clean only our own build dir
[ -d "$HERE/build" ] && find "$HERE/build" -mindepth 1 -delete
mkdir -p build/fapi build/classes
"$J/jar" --extract --file "$FAPI" --dir build/fapi 'META-INF/jars/' 2>/dev/null || true
CP="$L/com/mojang/minecraft/26.2/minecraft-26.2-client.jar"
for f in build/fapi/META-INF/jars/*.jar \
         "$L"/net/fabricmc/fabric-loader/*/*.jar \
         "$L"/com/mojang/datafixerupper/*/*.jar \
         "$L"/com/mojang/brigadier/*/*.jar "$L"/com/mojang/logging/*/*.jar \
         "$L"/com/google/guava/guava/*/*.jar "$L"/com/google/code/gson/gson/*/*.jar \
         "$L"/org/joml/joml/*/*.jar "$L"/it/unimi/dsi/fastutil/*/*.jar \
         "$L"/org/slf4j/slf4j-api/*/*.jar "$L"/org/jspecify/jspecify/*/*.jar; do
  [ -f "$f" ] && CP="$CP:$f"
done
# Simple Voice Chat API: compile-only (SVC is optional at runtime; deps/fetch.py downloads + verifies it)
SVC_API="$(ls "$HERE"/deps/voicechat-api-*.jar 2>/dev/null | head -1)"
SVC_JAR="$(ls "$HERE"/deps/voicechat-fabric-*.jar 2>/dev/null | head -1)"
[ -n "$SVC_API" ] && CP="$CP:$SVC_API"
SRCS="$(find src/main/java -name '*.java')"
[ -n "$SVC_API" ] || { echo "deps/voicechat-api-*.jar missing (run deps/fetch.py): building without the voice chat plugin"; SRCS="$(echo "$SRCS" | grep -v GladosVoicePlugin.java)"; }
"$J/javac" --release 21 -proc:none -Xlint:-options -cp "$CP" -d build/classes $SRCS
cp -r src/main/resources/. build/classes/
OUT="build/$NAME"
"$J/jar" --create --file "$OUT" -C build/classes .
echo "built $OUT ($(stat -c %s "$OUT") bytes)"

GSON="$(ls "$L"/com/google/code/gson/gson/*/*.jar | head -1)"
if [ "${1:-}" = test ]; then
  mkdir -p build/test
  # the harness sees only the pure-Java classes (no Minecraft/Fabric on its classpath)
  "$J/javac" --release 21 -proc:none -cp "build/classes:$GSON" -d build/test $(find test -name '*.java')
  URLS=("${@:2}"); [ ${#URLS[@]} -gt 0 ] || URLS=(http://127.0.0.1:8794)
  "$J/java" -cp "build/test:build/classes:$GSON" dev.yifan.glados.Harness "${URLS[@]}"
fi

if [ "${1:-}" = voicetest ]; then
  [ -n "$SVC_JAR" ] || { echo "voicetest needs deps/voicechat-fabric-*.jar: run python3 deps/fetch.py"; exit 1; }
  LOG4J="$(ls "$L"/org/apache/logging/log4j/log4j-api/*/*.jar | head -1):$(ls "$L"/org/apache/logging/log4j/log4j-core/*/*.jar | head -1)"
  mkdir -p build/test-voice
  "$J/javac" --release 21 -proc:none -cp "build/classes:$GSON:$SVC_JAR" -d build/test-voice $(find test-voice -name '*.java')
  "$J/java" -cp "build/test-voice:build/classes:$GSON:$SVC_JAR:$LOG4J" dev.yifan.glados.VoiceTests "${@:2}"
fi

if [ "${1:-}" = install ]; then
  for old in "$MODS"/glados-*.jar; do
    if [ -e "$old" ] && [ "$(basename "$old")" != "$NAME" ]; then
      mkdir -p "$BACKUPS"   # backups live outside mods/ (no duplicate loading)
      mv "$old" "$BACKUPS/$(basename "$old").replaced"
    fi
  done
  cp "$OUT" "$MODS/.glados.tmp" && mv "$MODS/.glados.tmp" "$MODS/$NAME"
  sha256sum "$MODS/$NAME"
fi
