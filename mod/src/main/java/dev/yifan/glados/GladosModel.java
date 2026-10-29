package dev.yifan.glados;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Geometry from GET /model.json, baked ONCE into indexed flat arrays (MC axes C(x,y,z) = (x,z,-y)).
 *
 * <p>Formats (contract v1/v2 and v3, both accepted):
 * <ul>
 * <li>v1/v2: {@code bodies[].meshes[]} with body-local {@code verts}, {@code tris}, optional {@code normals},
 *     {@code rgba}, {@code tag:"eye"}; no uv -> drawn with the white texture, tinted by rgba.</li>
 * <li>v3: {@code bodies[]} = {name, parent, joint_pos, rest} and top-level {@code meshes[]} with {@code body}, per-vertex
 *     {@code uv} into the single atlas ({@code atlas} path), {@code tag} eye|emissive|glass, and optional linear-blend
 *     {@code skin} {bodies:[a,b], j:[2 per vertex], w:[1 per vertex]} whose verts are in the SIM WORLD rest frame:
 *     v = w*T_a*R_a^-1*v + (1-w)*T_b*R_b^-1*v with R = the bodies' {@code rest} poses.</li>
 * </ul>
 * Meshes with {@code normals} keep shared (smooth) vertices; without, every triangle gets its own three vertices with
 * the flat face normal. Parts are split per body by tag (MAIN / EYE / EMISSIVE / GLASS) and by textured-or-not.
 * Pure Java (Gson only); {@link #emit} and {@link #emitSkinned} use static scratch buffers: call them from ONE thread
 * (the render thread in the game).
 */
public final class GladosModel {

    /** Receives transformed vertices; the mod adapts this to a Minecraft VertexConsumer. */
    public interface Sink {
        void vertex(float x, float y, float z, int argb, float u, float v, float nx, float ny, float nz);
    }

    public enum Tag {
        /** Lit, cutout. */
        MAIN,
        /** Fullbright, tinted by pose.eye (+ talk pulse). */
        EYE,
        /** Fullbright, untinted. */
        EMISSIVE,
        /** Lit, translucent pass (drawn after the cutout parts). */
        GLASS;

        static Tag of(JsonObject m) {
            JsonElement t = m.get("tag");
            if (t == null || !t.isJsonPrimitive()) {
                return MAIN;
            }
            return switch (t.getAsString()) {
                case "eye" -> EYE;
                case "emissive" -> EMISSIVE;
                case "glass" -> GLASS;
                default -> MAIN;
            };
        }

        public boolean fullbright() {
            return this == EYE || this == EMISSIVE;
        }
    }

    /** A batch of indexed triangles sharing tag + texture (+ for skinned parts: blended between body pairs). */
    public static final class Part {
        public final Tag tag;
        /** true = uv into the atlas; false = white texture (uv unused). */
        public final boolean textured;
        /** 3 floats per vertex, MC axes; rigid: body-local; skinned: local to body {@link #ja}. */
        public final float[] pos;
        /** 3 floats per vertex unit normal, same frame as pos. */
        public final float[] nrm;
        /** 2 floats per vertex (null if untextured). */
        public final float[] uv;
        /** 1 ARGB per vertex (mesh rgba). */
        public final int[] col;
        /** 3 vertex indices per triangle. */
        public final int[] tris;
        /** Skinned only (else null): model body indices a/b per vertex, weight of a, pos/nrm local to b. */
        public final int[] ja, jb;
        public final float[] w, posB, nrmB;
        /** Skinned only: the distinct bodies this part depends on (light). */
        public final int[] skinBodies;

        Part(Tag tag, boolean textured, float[] pos, float[] nrm, float[] uv, int[] col, int[] tris, int[] ja, int[] jb,
             float[] w, float[] posB, float[] nrmB, int[] skinBodies) {
            this.tag = tag;
            this.textured = textured;
            this.pos = pos;
            this.nrm = nrm;
            this.uv = uv;
            this.col = col;
            this.tris = tris;
            this.ja = ja;
            this.jb = jb;
            this.w = w;
            this.posB = posB;
            this.nrmB = nrmB;
            this.skinBodies = skinBodies;
        }

        public int triangles() {
            return tris.length / 3;
        }

        public int vertices() {
            return col.length;
        }

        public boolean skinned() {
            return ja != null;
        }
    }

    public static final class Body {
        public final String name;
        /** Rigid parts, cutout ones first, glass last. */
        public final Part[] parts;
        /** Voice/focus point: v3 {@code voice.pos} if it names this body, else the mean of its rigid mesh verts; SIM axes, body-local. */
        public final float[] centroid;
        /** v3 rest pose [px,py,pz,qw,qx,qy,qz] (sim world), or null. */
        public final float[] rest;

        Body(String name, Part[] parts, float[] centroid, float[] rest) {
            this.name = name;
            this.parts = parts;
            this.centroid = centroid;
            this.rest = rest;
        }

        public int triangles() {
            int n = 0;
            for (Part p : parts) {
                n += p.triangles();
            }
            return n;
        }

        public int triangles(Tag t) {
            int n = 0;
            for (Part p : parts) {
                n += p.tag == t ? p.triangles() : 0;
            }
            return n;
        }
    }

    public final Body[] bodies;
    /** Skinned parts (cables/wires spanning two bodies), glass last. */
    public final Part[] skinned;
    public final Map<String, Integer> index = new HashMap<>();
    public final int version;
    /** /model.json "rev" (v3), -1 if absent. */
    public final long rev;
    /** Path of the atlas on the sidecar ("/atlas.png?rev=N"), or null (v1/v2: no texture). */
    public final String atlasPath;
    public final String name, title, credit;
    public final int triangles, eyeTriangles, glassTriangles, emissiveTriangles, skinnedTriangles, texturedTriangles;
    /** Largest distance (metres) of any vertex from its (first) body origin; pads the culling box. */
    public final float radius;
    /** PNG bytes of the atlas, set by the network thread BEFORE the model is published (null = none/failed). */
    byte[] atlasPng;

    private GladosModel(int version, long rev, String atlasPath, String name, String title, String credit, Body[] bodies,
                        Part[] skinned, float radius) {
        this.version = version;
        this.rev = rev;
        this.atlasPath = atlasPath;
        this.name = name;
        this.title = title;
        this.credit = credit;
        this.bodies = bodies;
        this.skinned = skinned;
        this.radius = radius;
        for (int i = 0; i < bodies.length; i++) {
            index.putIfAbsent(bodies[i].name, i);
        }
        int[] c = new int[6];
        List<Part> all = new ArrayList<>(Arrays.asList(skinned));
        for (Body b : bodies) {
            all.addAll(Arrays.asList(b.parts));
        }
        for (Part p : all) {
            int n = p.triangles();
            c[0] += n;
            c[1] += p.tag == Tag.EYE ? n : 0;
            c[2] += p.tag == Tag.GLASS ? n : 0;
            c[3] += p.tag == Tag.EMISSIVE ? n : 0;
            c[4] += p.skinned() ? n : 0;
            c[5] += p.textured ? n : 0;
        }
        triangles = c[0];
        eyeTriangles = c[1];
        glassTriangles = c[2];
        emissiveTriangles = c[3];
        skinnedTriangles = c[4];
        texturedTriangles = c[5];
    }

    /** The atlas PNG (null if the model has none or it failed to download/validate). */
    public byte[] atlasPng() {
        return atlasPng;
    }

    /** True if some part samples the atlas (so the renderer must wait for / use the atlas texture). */
    public boolean needsAtlas() {
        return texturedTriangles > 0;
    }

    /** The root body (ceiling mount): the first body in /model.json. */
    public int root() {
        return 0;
    }

    /** Index of the body named {@code name}, else of the first body that has eye meshes, else the last body; -1 if empty. */
    public int find(String name) {
        Integer i = name == null ? null : index.get(name);
        if (i != null) {
            return i;
        }
        for (int k = 0; k < bodies.length; k++) {
            if (bodies[k].triangles(Tag.EYE) > 0) {
                return k;
            }
        }
        return bodies.length - 1;
    }

    /** One-line summary for the log / harness. */
    public String summary() {
        return bodies.length + " bodies, " + triangles + " triangles (" + eyeTriangles + " eye, " + emissiveTriangles
                + " emissive, " + glassTriangles + " glass, " + skinnedTriangles + " skinned, " + texturedTriangles
                + " textured)" + (name != null ? ", model '" + name + "'" : "") + (rev >= 0 ? " rev " + rev : "");
    }

    // ------------------------------------------------------------------ parsing

    /** Growable vertex/triangle accumulator for one part. */
    private static final class Acc {
        final Tag tag;
        final boolean textured, skinned;
        float[] pos = new float[96], nrm = new float[96], uv, posB, nrmB, w;
        int[] col = new int[32], tris = new int[96], ja, jb;
        int nv, nt;
        final java.util.TreeSet<Integer> skinBodies = new java.util.TreeSet<>();

        Acc(Tag tag, boolean textured, boolean skinned) {
            this.tag = tag;
            this.textured = textured;
            this.skinned = skinned;
            if (textured) {
                uv = new float[64];
            }
            if (skinned) {
                posB = new float[96];
                nrmB = new float[96];
                w = new float[32];
                ja = new int[32];
                jb = new int[32];
            }
        }

        int addVertex() {
            if (nv == col.length) {
                int cap = col.length * 2;
                pos = Arrays.copyOf(pos, cap * 3);
                nrm = Arrays.copyOf(nrm, cap * 3);
                col = Arrays.copyOf(col, cap);
                if (uv != null) {
                    uv = Arrays.copyOf(uv, cap * 2);
                }
                if (skinned) {
                    posB = Arrays.copyOf(posB, cap * 3);
                    nrmB = Arrays.copyOf(nrmB, cap * 3);
                    w = Arrays.copyOf(w, cap);
                    ja = Arrays.copyOf(ja, cap);
                    jb = Arrays.copyOf(jb, cap);
                }
            }
            return nv++;
        }

        void addTri(int a, int b, int c) {
            if (3 * nt + 3 > tris.length) {
                tris = Arrays.copyOf(tris, tris.length * 2);
            }
            tris[3 * nt] = a;
            tris[3 * nt + 1] = b;
            tris[3 * nt + 2] = c;
            nt++;
        }

        Part done() {
            int[] sb = skinned ? skinBodies.stream().mapToInt(Integer::intValue).toArray() : null;
            return new Part(tag, textured, Arrays.copyOf(pos, nv * 3), Arrays.copyOf(nrm, nv * 3),
                    uv == null ? null : Arrays.copyOf(uv, nv * 2), Arrays.copyOf(col, nv), Arrays.copyOf(tris, nt * 3),
                    skinned ? Arrays.copyOf(ja, nv) : null, skinned ? Arrays.copyOf(jb, nv) : null,
                    skinned ? Arrays.copyOf(w, nv) : null, skinned ? Arrays.copyOf(posB, nv * 3) : null,
                    skinned ? Arrays.copyOf(nrmB, nv * 3) : null, sb);
        }
    }

    /** Per-container (one body, or the skinned set) accumulators keyed by tag + textured. */
    private static final class Group {
        final boolean skinned;
        final Map<String, Acc> accs = new LinkedHashMap<>();
        double cx, cy, cz;
        long cn;

        Group(boolean skinned) {
            this.skinned = skinned;
        }

        Acc acc(Tag t, boolean textured) {
            return accs.computeIfAbsent(t + "/" + textured, k -> new Acc(t, textured, skinned));
        }

        /** Non-empty parts: MAIN, EMISSIVE, EYE first, GLASS last. */
        Part[] parts() {
            List<Part> out = new ArrayList<>();
            for (Tag t : new Tag[] {Tag.MAIN, Tag.EMISSIVE, Tag.EYE, Tag.GLASS}) {
                for (Acc a : accs.values()) {
                    if (a.tag == t && a.nt > 0) {
                        out.add(a.done());
                    }
                }
            }
            return out.toArray(new Part[0]);
        }
    }

    /** Running max vertex distance from body origins. */
    private static final class Radius {
        float r;

        void add(float x, float y, float z) {
            float d = (float) Math.sqrt(x * x + y * y + z * z);
            if (Float.isFinite(d)) {
                r = Math.max(r, d);
            }
        }
    }

    public static GladosModel parse(String json) {
        JsonObject root = JsonParser.parseString(json).getAsJsonObject();
        int version = root.has("version") && root.get("version").isJsonPrimitive() ? root.get("version").getAsInt() : 1;
        long rev = root.has("rev") && root.get("rev").isJsonPrimitive() ? root.get("rev").getAsLong() : -1;
        String atlas = atlasPath(root.get("atlas"));
        JsonArray bs = root.has("bodies") && root.get("bodies").isJsonArray() ? root.getAsJsonArray("bodies") : new JsonArray();
        List<String> names = new ArrayList<>();
        List<float[]> rests = new ArrayList<>();
        Map<String, Integer> idx = new HashMap<>();
        for (JsonElement be : bs) {
            JsonObject b = be.getAsJsonObject();
            String n = b.get("name").getAsString();
            idx.putIfAbsent(n, names.size());
            names.add(n);
            float[] r = b.has("rest") && b.get("rest").isJsonArray() ? floats(b.getAsJsonArray("rest")) : null;
            rests.add(r != null && r.length >= 7 ? normRest(r) : null);
        }
        Group[] groups = new Group[names.size()];
        for (int i = 0; i < groups.length; i++) {
            groups[i] = new Group(false);
        }
        Group skin = new Group(true);
        Radius rad = new Radius();
        // v1/v2: meshes nested in their body
        for (int i = 0; i < bs.size(); i++) {
            JsonObject b = bs.get(i).getAsJsonObject();
            if (b.has("meshes") && b.get("meshes").isJsonArray()) {
                for (JsonElement me : b.getAsJsonArray("meshes")) {
                    addMesh(me.getAsJsonObject(), i, groups, skin, rests, idx, atlas != null, rad);
                }
            }
        }
        // v3: top-level meshes with "body"
        if (root.has("meshes") && root.get("meshes").isJsonArray()) {
            for (JsonElement me : root.getAsJsonArray("meshes")) {
                JsonObject m = me.getAsJsonObject();
                Integer bi = m.has("body") && m.get("body").isJsonPrimitive() ? idx.get(m.get("body").getAsString()) : null;
                addMesh(m, bi == null ? -1 : bi, groups, skin, rests, idx, atlas != null, rad);
            }
        }
        // voice point override (v3)
        String voiceBody = null;
        float[] voicePos = null;
        if (root.has("voice") && root.get("voice").isJsonObject()) {
            JsonObject v = root.getAsJsonObject("voice");
            if (v.has("body") && v.has("pos") && v.get("pos").isJsonArray()) {
                float[] p = floats(v.getAsJsonArray("pos"));
                if (p.length >= 3) {
                    voiceBody = v.get("body").getAsString();
                    voicePos = new float[] {p[0], p[1], p[2]};
                }
            }
        }
        Body[] out = new Body[names.size()];
        for (int i = 0; i < out.length; i++) {
            Group g = groups[i];
            float[] cen = names.get(i).equals(voiceBody) ? voicePos
                    : g.cn == 0 ? new float[3] : new float[] {(float) (g.cx / g.cn), (float) (g.cy / g.cn), (float) (g.cz / g.cn)};
            out[i] = new Body(names.get(i), g.parts(), cen, rests.get(i));
        }
        return new GladosModel(version, rev, atlas, str(root, "name"), str(root, "title"), str(root, "credit"), out,
                skin.parts(), rad.r);
    }

    /** "/atlas.png?rev=3" stays; "atlas.png" -> "/atlas.png"; full URLs keep only their path (always our sidecar). */
    static String atlasPath(JsonElement e) {
        if (e == null || !e.isJsonPrimitive()) {
            return null;
        }
        String s = e.getAsString().trim();
        if (s.isEmpty()) {
            return null;
        }
        int scheme = s.indexOf("://");
        if (scheme >= 0) {
            int slash = s.indexOf('/', scheme + 3);
            s = slash < 0 ? "/atlas.png" : s.substring(slash);
        }
        return s.startsWith("/") ? s : "/" + s;
    }

    private static void addMesh(JsonObject m, int body, Group[] groups, Group skin, List<float[]> rests,
                                Map<String, Integer> idx, boolean hasAtlas, Radius rad) {
        int argb = argb(m.has("rgba") && m.get("rgba").isJsonArray() ? m.getAsJsonArray("rgba") : null);
        if ((argb >>> 24) < 26) {
            return;                                    // ~invisible geoms (alpha < 0.1) would be cut out anyway
        }
        Tag tag = Tag.of(m);
        if (tag != Tag.GLASS) {
            argb |= 0xFF000000;                        // only glass is translucent
        }
        float[] v = floats(m.has("verts") && m.get("verts").isJsonArray() ? m.getAsJsonArray("verts") : null);
        int nv = v.length / 3;
        int[] idxs = ints(m.has("tris") && m.get("tris").isJsonArray() ? m.getAsJsonArray("tris") : null);
        float[] vn = m.has("normals") && m.get("normals").isJsonArray() ? floats(m.getAsJsonArray("normals")) : null;
        if (vn != null && vn.length != v.length) {
            vn = null;                                 // malformed: fall back to flat normals
        }
        float[] uv = hasAtlas && m.has("uv") && m.get("uv").isJsonArray() ? floats(m.getAsJsonArray("uv")) : null;
        if (uv != null && uv.length != 2 * nv) {
            uv = null;                                 // malformed / missing: white texture with the rgba tint
        }
        // skin -> per vertex (a, b, w) in model body indices, verts in the sim world rest frame
        int[] sa = null, sb = null;
        float[] sw = null;
        if (m.has("skin") && m.get("skin").isJsonObject()) {
            JsonObject s = m.getAsJsonObject("skin");
            JsonArray sbs = s.has("bodies") && s.get("bodies").isJsonArray() ? s.getAsJsonArray("bodies") : new JsonArray();
            int[] map = new int[sbs.size()];
            boolean ok = sbs.size() > 0;
            for (int k = 0; k < map.length && ok; k++) {
                Integer bi = idx.get(sbs.get(k).getAsString());
                ok = bi != null && rests.get(bi) != null;
                map[k] = ok ? bi : -1;
            }
            int[] j = ok && s.has("j") && s.get("j").isJsonArray() ? ints(s.getAsJsonArray("j")) : null;
            float[] w = ok && s.has("w") && s.get("w").isJsonArray() ? floats(s.getAsJsonArray("w")) : null;
            if (j != null && w != null && j.length == 2 * nv && w.length == nv) {
                sa = new int[nv];
                sb = new int[nv];
                sw = new float[nv];
                for (int i = 0; i < nv && sa != null; i++) {
                    int a = j[2 * i], b = j[2 * i + 1];
                    if (a < 0 || a >= map.length || b < 0 || b >= map.length || !Float.isFinite(w[i])) {
                        sa = null;
                        break;
                    }
                    sa[i] = map[a];
                    sb[i] = map[b];
                    sw[i] = Math.max(0f, Math.min(1f, w[i]));
                }
            }
            if (sa == null) {
                // unusable skin: rigid in its "body" if that body has a rest pose (verts are world rest)
                if (body < 0 || rests.get(body) == null) {
                    return;
                }
                float[] r = rests.get(body);
                float[] lv = new float[v.length], ln = vn == null ? null : new float[vn.length];
                for (int i = 0; i < nv; i++) {
                    worldToLocal(r, v, i, lv, true);
                    if (ln != null) {
                        worldToLocal(r, vn, i, ln, false);
                    }
                }
                v = lv;
                vn = ln;
            }
        } else if (body < 0) {
            return;                                    // v3 mesh for an unknown body
        }
        boolean skinned = sa != null;
        Group g = skinned ? skin : groups[body];
        Acc acc = g.acc(tag, uv != null);
        if (skinned) {
            for (int i = 0; i < nv; i++) {
                acc.skinBodies.add(sa[i]);
                acc.skinBodies.add(sb[i]);
            }
        }
        int nt = idxs.length / 3;
        int added = 0;
        float[] tmpA = new float[3], tmpB = new float[3];
        if (vn != null) {
            // smooth: shared vertices, index remap; a zero/NaN normal becomes the area-weighted mean of its faces
            vn = repairNormals(v, vn, idxs);
            int base = acc.nv;
            for (int i = 0; i < nv; i++) {
                int o = acc.addVertex();
                writeVertex(acc, o, v, vn, i, uv, argb, sa, sb, sw, rests, tmpA, tmpB);
            }
            for (int t = 0; t < nt; t++) {
                int k0 = idxs[3 * t], k1 = idxs[3 * t + 1], k2 = idxs[3 * t + 2];
                if (k0 < 0 || k1 < 0 || k2 < 0 || k0 >= nv || k1 >= nv || k2 >= nv) {
                    continue;
                }
                acc.addTri(base + k0, base + k1, base + k2);
                added++;
            }
        } else {
            // flat: three own vertices per triangle with the face normal
            float[] fn = new float[9];
            for (int t = 0; t < nt; t++) {
                int k0 = idxs[3 * t], k1 = idxs[3 * t + 1], k2 = idxs[3 * t + 2];
                if (k0 < 0 || k1 < 0 || k2 < 0 || k0 >= nv || k1 >= nv || k2 >= nv) {
                    continue;
                }
                float e1x = v[3 * k1] - v[3 * k0], e1y = v[3 * k1 + 1] - v[3 * k0 + 1], e1z = v[3 * k1 + 2] - v[3 * k0 + 2];
                float e2x = v[3 * k2] - v[3 * k0], e2y = v[3 * k2 + 1] - v[3 * k0 + 1], e2z = v[3 * k2 + 2] - v[3 * k0 + 2];
                float nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
                float len = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
                if (!(len > 1e-12f)) {
                    nx = 0;
                    ny = 0;
                    nz = 1;                            // sim +z = MC up
                } else {
                    nx /= len;
                    ny /= len;
                    nz /= len;
                }
                int[] ks = {k0, k1, k2};
                int[] o = new int[3];
                for (int j = 0; j < 3; j++) {
                    fn[0] = nx;
                    fn[1] = ny;
                    fn[2] = nz;
                    o[j] = acc.addVertex();
                    writeVertexFlat(acc, o[j], v, fn, ks[j], uv, argb, sa, sb, sw, rests, tmpA, tmpB);
                }
                acc.addTri(o[0], o[1], o[2]);
                added++;
            }
        }
        if (added > 0) {
            for (int i = 0; i < nv; i++) {
                if (skinned) {
                    worldToLocal(rests.get(sa[i]), v, i, tmpA, true);
                    rad.add(tmpA[0], tmpA[1], tmpA[2]);
                    worldToLocal(rests.get(sb[i]), v, i, tmpB, true);
                    rad.add(tmpB[0], tmpB[1], tmpB[2]);
                } else {
                    rad.add(v[3 * i], v[3 * i + 1], v[3 * i + 2]);
                    g.cx += v[3 * i];
                    g.cy += v[3 * i + 1];
                    g.cz += v[3 * i + 2];
                    g.cn++;
                }
            }
        }
    }

    /** Copy of vn where unusable normals are replaced by the (area-weighted) sum of the adjacent face normals. */
    private static float[] repairNormals(float[] v, float[] vn, int[] idxs) {
        int nv = v.length / 3;
        boolean[] bad = new boolean[nv];
        boolean any = false;
        for (int i = 0; i < nv; i++) {
            float l = vn[3 * i] * vn[3 * i] + vn[3 * i + 1] * vn[3 * i + 1] + vn[3 * i + 2] * vn[3 * i + 2];
            bad[i] = !(l > 1e-12f) || !Float.isFinite(l);
            any |= bad[i];
        }
        if (!any) {
            return vn;
        }
        float[] out = vn.clone();
        for (int i = 0; i < nv; i++) {
            if (bad[i]) {
                out[3 * i] = out[3 * i + 1] = out[3 * i + 2] = 0;
            }
        }
        for (int t = 0; t + 2 < idxs.length; t += 3) {
            int k0 = idxs[t], k1 = idxs[t + 1], k2 = idxs[t + 2];
            if (k0 < 0 || k1 < 0 || k2 < 0 || k0 >= nv || k1 >= nv || k2 >= nv) {
                continue;
            }
            float e1x = v[3 * k1] - v[3 * k0], e1y = v[3 * k1 + 1] - v[3 * k0 + 1], e1z = v[3 * k1 + 2] - v[3 * k0 + 2];
            float e2x = v[3 * k2] - v[3 * k0], e2y = v[3 * k2 + 1] - v[3 * k0 + 1], e2z = v[3 * k2 + 2] - v[3 * k0 + 2];
            float nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
            for (int k : new int[] {k0, k1, k2}) {
                if (bad[k]) {
                    out[3 * k] += nx;
                    out[3 * k + 1] += ny;
                    out[3 * k + 2] += nz;
                }
            }
        }
        return out;
    }

    /** Vertex i with its own (smooth) normal vn[i]. */
    private static void writeVertex(Acc acc, int o, float[] v, float[] vn, int i, float[] uv, int argb, int[] sa, int[] sb,
                                    float[] sw, List<float[]> rests, float[] tmpA, float[] tmpB) {
        float nx = vn[3 * i], ny = vn[3 * i + 1], nz = vn[3 * i + 2];
        float l = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (!(l > 1e-6f) || !Float.isFinite(l)) {
            nx = 0;
            ny = 0;
            nz = 1;
            l = 1;
        }
        float[] n = {nx / l, ny / l, nz / l};
        writeVertexFlat(acc, o, v, n, i, uv, argb, sa, sb, sw, rests, tmpA, tmpB);
    }

    /** Vertex i of v with the given unit sim-axis normal n[0..2]. */
    private static void writeVertexFlat(Acc acc, int o, float[] v, float[] n, int i, float[] uv, int argb, int[] sa,
                                        int[] sb, float[] sw, List<float[]> rests, float[] tmpA, float[] tmpB) {
        if (sa != null) {
            float[] nn = {n[0], n[1], n[2]};
            float[] ra = rests.get(sa[i]), rb = rests.get(sb[i]);
            worldToLocal(ra, v, i, tmpA, true);
            mc(acc.pos, o, tmpA);
            worldToLocal(rb, v, i, tmpB, true);
            mc(acc.posB, o, tmpB);
            worldToLocal(ra, nn, 0, tmpA, false);
            mc(acc.nrm, o, tmpA);
            worldToLocal(rb, nn, 0, tmpB, false);
            mc(acc.nrmB, o, tmpB);
            acc.ja[o] = sa[i];
            acc.jb[o] = sb[i];
            acc.w[o] = sw[i];
        } else {
            acc.pos[3 * o] = v[3 * i];
            acc.pos[3 * o + 1] = v[3 * i + 2];
            acc.pos[3 * o + 2] = -v[3 * i + 1];
            acc.nrm[3 * o] = n[0];
            acc.nrm[3 * o + 1] = n[2];
            acc.nrm[3 * o + 2] = -n[1];
        }
        if (acc.uv != null) {
            acc.uv[2 * o] = uv[2 * i];
            acc.uv[2 * o + 1] = uv[2 * i + 1];
        }
        acc.col[o] = argb;
    }

    /** dst[3o..] = C(s) (sim -> MC axes). */
    private static void mc(float[] dst, int o, float[] s) {
        dst[3 * o] = s[0];
        dst[3 * o + 1] = s[2];
        dst[3 * o + 2] = -s[1];
    }

    /** out = R^T (v_i - p) (point) or R^T v_i (direction) for rest = [p, q]; sim axes. */
    static void worldToLocal(float[] rest, float[] v, int i, float[] out, boolean point) {
        float x = v[3 * i], y = v[3 * i + 1], z = v[3 * i + 2];
        if (point) {
            x -= rest[0];
            y -= rest[1];
            z -= rest[2];
        }
        // rotate by the conjugate quaternion
        float w = rest[3], qx = -rest[4], qy = -rest[5], qz = -rest[6];
        float tx = 2 * (qy * z - qz * y), ty = 2 * (qz * x - qx * z), tz = 2 * (qx * y - qy * x);
        out[0] = x + w * tx + (qy * tz - qz * ty);
        out[1] = y + w * ty + (qz * tx - qx * tz);
        out[2] = z + w * tz + (qx * ty - qy * tx);
    }

    private static float[] normRest(float[] r) {
        float n = (float) Math.sqrt(r[3] * r[3] + r[4] * r[4] + r[5] * r[5] + r[6] * r[6]);
        if (!(n > 1e-6f)) {
            return null;
        }
        return new float[] {r[0], r[1], r[2], r[3] / n, r[4] / n, r[5] / n, r[6] / n};
    }

    // ------------------------------------------------------------------ emission (hot path)

    private static float[] sx = new float[0], sn = new float[0];
    private static int[] sc = new int[0];

    private static void ensureScratch(int nv) {
        if (sc.length < nv) {
            int cap = Math.max(nv, sc.length * 2);
            sx = new float[cap * 3];
            sn = new float[cap * 3];
            sc = new int[cap];
        }
    }

    /**
     * Emit a rigid part as degenerate quads (entity render types draw QUADS; the 3rd vertex is repeated). {@code m} is
     * the column-major 4x4 pose matrix and {@code n} the column-major 3x3 normal matrix (JOML {@code get(float[])}
     * layout). {@code tint} != null multiplies the per-vertex rgb (the eye); vertices are transformed once each.
     */
    public static void emit(Part b, float[] m, float[] n, float[] tint, Sink sink) {
        if (b.skinned()) {
            throw new IllegalArgumentException("skinned part: use emitSkinned");
        }
        final int nv = b.vertices();
        ensureScratch(nv);
        final float m00 = m[0], m01 = m[1], m02 = m[2];
        final float m10 = m[4], m11 = m[5], m12 = m[6];
        final float m20 = m[8], m21 = m[9], m22 = m[10];
        final float m30 = m[12], m31 = m[13], m32 = m[14];
        final float n00 = n[0], n01 = n[1], n02 = n[2];
        final float n10 = n[3], n11 = n[4], n12 = n[5];
        final float n20 = n[6], n21 = n[7], n22 = n[8];
        final float[] p = b.pos, nr = b.nrm, x = sx, xn = sn;
        for (int i = 0, q = 0; i < nv; i++, q += 3) {
            float lx = p[q], ly = p[q + 1], lz = p[q + 2];
            x[q] = m00 * lx + m10 * ly + m20 * lz + m30;
            x[q + 1] = m01 * lx + m11 * ly + m21 * lz + m31;
            x[q + 2] = m02 * lx + m12 * ly + m22 * lz + m32;
            lx = nr[q];
            ly = nr[q + 1];
            lz = nr[q + 2];
            float nx = n00 * lx + n10 * ly + n20 * lz;
            float ny = n01 * lx + n11 * ly + n21 * lz;
            float nz = n02 * lx + n12 * ly + n22 * lz;
            float inv = nx * nx + ny * ny + nz * nz;
            if (inv > 0 && (inv < 0.999f || inv > 1.001f)) {   // the pose normal matrix is a rotation: usually unit
                inv = (float) (1.0 / Math.sqrt(inv));
                nx *= inv;
                ny *= inv;
                nz *= inv;
            }
            xn[q] = nx;
            xn[q + 1] = ny;
            xn[q + 2] = nz;
        }
        flush(b, tint, sink);
    }

    /**
     * Emit a skinned part: {@code mats[i]} / {@code nmats[i]} = the full pose matrix / normal matrix of body i (null
     * if body i has no pose: its vertices then follow the other body). v = w*M_a*la + (1-w)*M_b*lb.
     */
    public static void emitSkinned(Part b, float[][] mats, float[][] nmats, float[] tint, Sink sink) {
        final int nv = b.vertices();
        ensureScratch(nv);
        final float[] pa = b.pos, pb = b.posB, na = b.nrm, nb = b.nrmB, ws = b.w, x = sx, xn = sn;
        final int[] ja = b.ja, jb = b.jb;
        for (int i = 0, q = 0; i < nv; i++, q += 3) {
            float[] ma = mats[ja[i]], mb = mats[jb[i]], ra = nmats[ja[i]], rb = nmats[jb[i]];
            float w = ws[i];
            if (ma == null) {
                if (mb == null) {
                    return;                            // neither body posed: skip the whole part this frame
                }
                ma = mb;
                ra = rb;
                w = 0;
            } else if (mb == null) {
                mb = ma;
                rb = ra;
                w = 1;
            }
            float u = 1 - w;
            float ax = pa[q], ay = pa[q + 1], az = pa[q + 2], bx = pb[q], by = pb[q + 1], bz = pb[q + 2];
            x[q] = w * (ma[0] * ax + ma[4] * ay + ma[8] * az + ma[12]) + u * (mb[0] * bx + mb[4] * by + mb[8] * bz + mb[12]);
            x[q + 1] = w * (ma[1] * ax + ma[5] * ay + ma[9] * az + ma[13]) + u * (mb[1] * bx + mb[5] * by + mb[9] * bz + mb[13]);
            x[q + 2] = w * (ma[2] * ax + ma[6] * ay + ma[10] * az + ma[14]) + u * (mb[2] * bx + mb[6] * by + mb[10] * bz + mb[14]);
            ax = na[q];
            ay = na[q + 1];
            az = na[q + 2];
            bx = nb[q];
            by = nb[q + 1];
            bz = nb[q + 2];
            float nx = w * (ra[0] * ax + ra[3] * ay + ra[6] * az) + u * (rb[0] * bx + rb[3] * by + rb[6] * bz);
            float ny = w * (ra[1] * ax + ra[4] * ay + ra[7] * az) + u * (rb[1] * bx + rb[4] * by + rb[7] * bz);
            float nz = w * (ra[2] * ax + ra[5] * ay + ra[8] * az) + u * (rb[2] * bx + rb[5] * by + rb[8] * bz);
            float l = nx * nx + ny * ny + nz * nz;
            if (l > 1e-12f) {
                l = (float) (1.0 / Math.sqrt(l));
                nx *= l;
                ny *= l;
                nz *= l;
            } else {
                nx = 0;
                ny = 1;
                nz = 0;
            }
            xn[q] = nx;
            xn[q + 1] = ny;
            xn[q + 2] = nz;
        }
        flush(b, tint, sink);
    }

    /** Colours (tinted once per vertex), then the triangles as degenerate quads from the transformed scratch. */
    private static void flush(Part b, float[] tint, Sink sink) {
        final int nv = b.vertices();
        final int[] col = b.col, c = sc;
        if (tint == null) {
            System.arraycopy(col, 0, c, 0, nv);
        } else {
            for (int i = 0; i < nv; i++) {
                c[i] = mulTint(col[i], tint);
            }
        }
        final float[] x = sx, xn = sn, uv = b.uv;
        final int[] tr = b.tris;
        final int nt = tr.length / 3;
        for (int t = 0, o = 0; t < nt; t++, o += 3) {
            for (int j = 0; j < 4; j++) {
                int k = tr[o + (j < 3 ? j : 2)];
                int q = 3 * k;
                float u = 0.5f, v = 0.5f;
                if (uv != null) {
                    u = uv[2 * k];
                    v = uv[2 * k + 1];
                }
                sink.vertex(x[q], x[q + 1], x[q + 2], c[k], u, v, xn[q], xn[q + 1], xn[q + 2]);
            }
        }
    }

    /** ARGB with rgb multiplied by tint[0..2] (clamped), alpha kept. */
    public static int mulTint(int argb, float[] tint) {
        int r = Math.round(((argb >> 16) & 0xFF) * tint[0]);
        int g = Math.round(((argb >> 8) & 0xFF) * tint[1]);
        int b = Math.round((argb & 0xFF) * tint[2]);
        return (argb & 0xFF000000) | (Math.max(0, Math.min(255, r)) << 16) | (Math.max(0, Math.min(255, g)) << 8)
                | Math.max(0, Math.min(255, b));
    }

    // ------------------------------------------------------------------ PNG sanity (atlas)

    /** Max atlas edge accepted (contract: <= 4096 px). */
    public static final int MAX_ATLAS = 4096;

    /** {width, height} from a PNG's IHDR, or null if the bytes are not a PNG. */
    public static int[] pngSize(byte[] b) {
        if (b == null || b.length < 24) {
            return null;
        }
        int[] sig = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
        for (int i = 0; i < 8; i++) {
            if ((b[i] & 0xFF) != sig[i]) {
                return null;
            }
        }
        if (b[12] != 'I' || b[13] != 'H' || b[14] != 'D' || b[15] != 'R') {
            return null;
        }
        return new int[] {be32(b, 16), be32(b, 20)};
    }

    private static int be32(byte[] b, int o) {
        return ((b[o] & 0xFF) << 24) | ((b[o + 1] & 0xFF) << 16) | ((b[o + 2] & 0xFF) << 8) | (b[o + 3] & 0xFF);
    }

    // ------------------------------------------------------------------ json helpers

    static int argb(JsonArray rgba) {
        float r = 0.8f, g = 0.8f, b = 0.8f, a = 1f;
        if (rgba != null && rgba.size() >= 3) {
            r = rgba.get(0).getAsFloat();
            g = rgba.get(1).getAsFloat();
            b = rgba.get(2).getAsFloat();
            a = rgba.size() >= 4 ? rgba.get(3).getAsFloat() : 1f;
        }
        return (c(a) << 24) | (c(r) << 16) | (c(g) << 8) | c(b);
    }

    private static int c(float f) {
        return Float.isFinite(f) ? Math.max(0, Math.min(255, Math.round(f * 255f))) : 255;
    }

    private static String str(JsonObject o, String k) {
        JsonElement e = o.get(k);
        return e != null && e.isJsonPrimitive() ? e.getAsString() : null;
    }

    private static float[] floats(JsonArray a) {
        float[] f = new float[a == null ? 0 : a.size()];
        for (int i = 0; i < f.length; i++) {
            JsonElement e = a.get(i);
            f[i] = e.isJsonPrimitive() ? e.getAsFloat() : Float.NaN;
        }
        return f;
    }

    private static int[] ints(JsonArray a) {
        int[] f = new int[a == null ? 0 : a.size()];
        for (int i = 0; i < f.length; i++) {
            JsonElement e = a.get(i);
            f[i] = e.isJsonPrimitive() ? e.getAsInt() : -1;
        }
        return f;
    }
}
