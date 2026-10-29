# Third-party notices

Omo GLaDOS's own code (`mod/`, `sidecar/`, `docs/media/*.py`) and its procedural model (`sidecar/glados_model.py`,
`sidecar/assets/procedural/`) are MIT licensed (see [LICENSE](LICENSE)). It builds on the following work. Thank you.

| Component | By | Licence | How it is used |
|---|---|---|---|
| [MuJoCo](https://github.com/google-deepmind/mujoco) | Google DeepMind | Apache-2.0 | Physics (pip dependency, installed by `sidecar/setup.sh`). |
| [NumPy](https://numpy.org) | NumPy developers | BSD-3-Clause | Procedural geometry, maths, software renderer (pip dependency). |
| [Pillow](https://python-pillow.org) | Jeffrey A. Clark and contributors | MIT-CMU (HPND) | Atlas PNGs and preview renders (pip dependency). |
| [Fabric Loader / Fabric API](https://fabricmc.net) | FabricMC | Apache-2.0 | Compile and runtime dependency; not bundled. |
| [Simple Voice Chat API](https://modrinth.com/plugin/simple-voice-chat) | Max Henkel | see its project page | Optional, compile-only; fetched and hash-checked by `mod/deps/fetch.py`; not bundled. |
| [Crimson Text](https://github.com/googlefonts/Crimson), [IBM Plex Mono](https://github.com/IBM/plex) | Sebastian Kosch; IBM | SIL OFL 1.1 | Typefaces in the README images (`docs/media/make_*.py`); the fonts themselves are not included. |

## GLaDOS and Valve

GLaDOS and Portal are trademarks and characters of **Valve Corporation**. Omo GLaDOS is an independent fan project
of a GLaDOS-style ceiling AI. It is **not affiliated with, endorsed by or sponsored by Valve**. No Valve assets,
code, models, textures or audio are included: the model is original procedural geometry written in Python, and no
voice audio or reference clips ship with this repository.

If you load another model into the sidecar, its licence is your responsibility. Most GLaDOS models online are fan
art of Valve's character or rips of Valve's own assets; do not redistribute them.

NOT AN OFFICIAL MINECRAFT PRODUCT. NOT APPROVED BY OR ASSOCIATED WITH MOJANG OR
MICROSOFT. Minecraft is a trademark of Mojang/Microsoft.
