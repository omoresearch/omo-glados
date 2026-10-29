#!/usr/bin/env python3
"""javap check: every Minecraft / Fabric / JOML / slf4j / Gson / Simple Voice Chat API member referenced by the mod's
classes exists (name + descriptor, including inherited members) in the real target jars: the MC 26.2 client jar, the
fabric-api nested jars, the loader, joml, slf4j, gson and deps/voicechat-api-*.jar. Run after ./build.sh."""
import glob
import os
import re
import subprocess
import sys

H = os.path.expanduser("~")
J = os.path.join(os.environ.get("JDK_BIN", H + "/.local/share/omo/toolchain/jdk25/bin"), "javap")
L = os.environ.get("MC_LIBRARIES", H + "/.local/share/omo/runtime/prism/libraries")
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "..", "build")
cp = [L + "/com/mojang/minecraft/26.2/minecraft-26.2-client.jar"]
cp += glob.glob(B + "/fapi/META-INF/jars/*.jar")
for pat in ["net/fabricmc/fabric-loader/*/*.jar", "org/joml/joml/*/*.jar", "org/slf4j/slf4j-api/*/*.jar",
            "com/google/code/gson/gson/*/*.jar"]:
    cp += glob.glob(L + "/" + pat)
cp += glob.glob(os.path.join(HERE, "..", "deps", "voicechat-api-*.jar"))
CP = ":".join(cp)
PREFIXES = ("net/minecraft", "com/mojang", "net/fabricmc", "org/joml", "org/slf4j", "com/google/gson", "de/maxhenkel")

classes = sorted(glob.glob(B + "/classes/dev/yifan/glados/*.class"))
if not classes:
    sys.exit("no classes: run ./build.sh first")
refs = set()
for cls in classes:
    out = subprocess.run([J, "-v", "-p", cls], capture_output=True, text=True).stdout
    for m in re.finditer(r"= (Methodref|InterfaceMethodref|Fieldref)\s+#\d+\.#\d+\s+// (\S+)", out):
        kind, ref = m.groups()
        owner, rest = ref.split(".", 1)
        name, desc = rest.split(":", 1)
        name = name.strip('"')
        if owner.startswith(PREFIXES):
            refs.add((owner, name, desc, kind))


def javap(c):
    return subprocess.run([J, "-s", "-p", "-cp", CP, c], capture_output=True, text=True).stdout


def supers(text):
    head = text.split("{")[0]
    m = re.search(r"(?:extends|implements) (.+)$", head.strip().splitlines()[-1] if head.strip() else "")
    if not m:
        return []
    return [x.strip().split("<")[0] for x in re.split(r",|\bextends\b|\bimplements\b", m.group(1)) if x.strip()]


cache = {}
bad = 0
for owner, name, desc, kind in sorted(refs):
    if owner not in cache:
        text = javap(owner.replace("/", "."))
        seen = {owner.replace("/", ".")}
        queue = supers(text)
        while queue:
            s = queue.pop()
            if s in seen or s.startswith("java."):
                continue
            seen.add(s)
            t2 = javap(s)
            text += t2
            queue += supers(t2)
        cache[owner] = text
    lines = cache[owner].splitlines()
    found = False
    for i, line in enumerate(lines):
        if i + 1 < len(lines) and ("descriptor: " + desc) == lines[i + 1].strip():
            simple = owner.rsplit("/", 1)[-1].split("$")[-1]
            if re.search(r"[ .]" + re.escape(name) + r"(\(|;| =)", line) or (
                    name == "<init>" and re.search(r"[ .$]" + re.escape(simple) + r"\(", line)):
                found = True
                break
    if not found:
        bad += 1
    print(f"{'ok ' if found else 'MISSING'} {kind:18} {owner.replace('/', '.')}.{name} {desc}")
# CONTRACT v3 coverage: the texture / render-type / matrix members the atlas + skinning code depends on must be among
# the checked references (so a refactor can't silently drop them from the check)
REQUIRED = [
    ("com/mojang/blaze3d/platform/NativeImage", "read"),
    ("com/mojang/blaze3d/platform/NativeImage", "close"),
    ("com/mojang/blaze3d/platform/NativeImage$Format", "RGBA"),
    ("net/minecraft/client/renderer/texture/DynamicTexture", "<init>"),
    ("net/minecraft/client/Minecraft", "getTextureManager"),
    ("net/minecraft/client/renderer/texture/TextureManager", "register"),
    ("net/minecraft/client/renderer/texture/TextureManager", "release"),
    ("net/minecraft/client/renderer/rendertype/RenderTypes", "entityCutout"),
    ("net/minecraft/client/renderer/rendertype/RenderTypes", "entityTranslucent"),
    ("net/minecraft/resources/Identifier", "fromNamespaceAndPath"),
    ("com/mojang/blaze3d/vertex/PoseStack$Pose", "copy"),
    ("com/mojang/blaze3d/vertex/PoseStack$Pose", "set"),
    ("com/mojang/blaze3d/vertex/VertexConsumer", "addVertex"),
    ("org/joml/Matrix4f", "mul"),
    ("org/joml/Matrix3f", "mul"),
]
have = {(o, n) for o, n, _, _ in refs}
uncovered = [f"{o.replace('/', '.')}.{n}" for o, n in REQUIRED if (o, n) not in have]
for u in uncovered:
    print("NOT REFERENCED (v3 coverage) " + u)
bad += len(uncovered)
print(f"\n{len(refs)} references checked in {len(classes)} classes, {bad} missing"
      f" (v3 coverage {len(REQUIRED) - len(uncovered)}/{len(REQUIRED)})")
sys.exit(1 if bad else 0)
