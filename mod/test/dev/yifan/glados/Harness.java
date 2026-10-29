package dev.yifan.glados;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

/**
 * Standalone test harness for the mod's pure-Java parts (not loaded by Fabric; not in the jar).
 * Run: ./build.sh test [http://127.0.0.1:PORT]   (fake: python3 test/fake_sidecar.py 8789)
 */
public final class Harness {
    static int pass, fail;

    static void check(String name, boolean ok, String detail) {
        if (ok) {
            pass++;
        } else {
            fail++;
        }
        System.out.println((ok ? "PASS " : "FAIL ") + name + (detail == null ? "" : "  " + detail));
    }

    static boolean near(double[] a, double[] b, double eps) {
        for (int i = 0; i < a.length; i++) {
            if (!(Math.abs(a[i] - b[i]) <= eps)) {
                return false;
            }
        }
        return true;
    }

    static String v(double... d) {
        StringBuilder sb = new StringBuilder("(");
        for (int i = 0; i < d.length; i++) {
            sb.append(i > 0 ? ", " : "").append(String.format(Locale.ROOT, "%.4f", d[i]));
        }
        return sb.append(")").toString();
    }

    public static void main(String[] args) throws Exception {
        String[] urls = args.length > 0 ? args : new String[] {"http://127.0.0.1:8794"};
        math();
        commands();
        config();
        model();
        modelV3();
        skinning();
        emitter();
        focus();
        perf();
        for (String url : urls) {
            http(url);
        }
        down();
        System.out.println();
        System.out.println("RESULT: " + pass + " passed, " + fail + " failed");
        System.exit(fail == 0 ? 0 : 1);
    }

    // ------------------------------------------------------------------ frame mapping (+ yaw)

    static void math() {
        System.out.println("== frame mapping (default anchor A=(26.5,33,271.5)) ==");
        double ax = 26.5, ay = 33, az = 271.5;
        double[] o = GladosMath.simToMc(ax, ay, az, 1, 0, 0, 0, 0);
        check("sim origin -> anchor (ceiling underside)", near(o, new double[] {26.5, 33, 271.5}, 1e-9), v(o));
        double[] down = GladosMath.simToMc(ax, ay, az, 1, 0, 0, 0, -7.5);
        check("sim -z 7.5 m (hanging) -> MC y 25.5", near(down, new double[] {26.5, 25.5, 271.5}, 1e-9), v(down));
        double[] p = GladosMath.simToMc(ax, ay, az, 2, 0, 0.4, 0.8, -1);
        check("s=2: sim (0.4,0.8,-1) -> MC (27.3,31,269.9)", near(p, new double[] {27.3, 31, 269.9}, 1e-9), v(p));
        double[] fy = GladosMath.simToMc(ax, ay, az, 1, 90, 1, 0, 0);
        check("yaw 90: sim +x (her facing) -> MC -z (north)", near(fy, new double[] {26.5, 33, 270.5}, 1e-9), v(fy));
        double[] fy2 = GladosMath.simToMc(ax, ay, az, 1, 90, 0, 1, 0);
        check("yaw 90: sim +y -> MC -x", near(fy2, new double[] {25.5, 33, 271.5}, 1e-9), v(fy2));
        // yaw Y == a sim z-rotation by Y applied before the contract mapping
        double[] r = GladosMath.simToMc(ax, ay, az, 1.5, 37, 0.3, -0.7, -2);
        double c = Math.cos(Math.toRadians(37)), sn = Math.sin(Math.toRadians(37));
        double[] r2 = GladosMath.simToMc(ax, ay, az, 1.5, 0, c * 0.3 - sn * -0.7, sn * 0.3 + c * -0.7, -2);
        check("yaw == sim z-yaw before mapping (37 deg)", near(r, r2, 1e-9), v(r) + " vs " + v(r2));
        // player eye round trips through the inverse, with yaw and scale
        double[] eye = {30.2, 21.62, 268.9};
        for (double[] sy : new double[][] {{1, 0}, {2.5, 0}, {1, 90}, {0.5, -133}}) {
            double[] sim = GladosMath.mcToSim(ax, ay, az, sy[0], sy[1], eye[0], eye[1], eye[2]);
            double[] back = GladosMath.simToMc(ax, ay, az, sy[0], sy[1], sim[0], sim[1], sim[2]);
            check("player eye MC -> sim -> MC (s=" + sy[0] + ", yaw=" + sy[1] + ")", near(back, eye, 1e-9), "sim " + v(sim));
        }
        double[] s0 = GladosMath.mcToSim(ax, ay, az, 1, 0, 30.2, 21.62, 268.9);
        check("player eye, no yaw: sim ((px-ax), -(pz-az), (py-ay))", near(s0, new double[] {3.7, 2.6, -11.38}, 1e-9), v(s0));

        double cq = Math.cos(Math.PI / 4), sq = Math.sin(Math.PI / 4);
        double[] qmc = GladosMath.quatSimToMc(cq, 0, 0, sq);
        check("yaw90 quat sim [c,0,0,s] -> MC [c,0,s,0]", near(qmc, new double[] {cq, 0, sq, 0}, 1e-12), v(qmc));
        double[] yq = GladosMath.yawQuat(90);
        double[] fwd = GladosMath.rotate(yq, 1, 0, 0);
        check("yawQuat(90) turns MC +x to -z (same as yawVec)", near(fwd, GladosMath.yawVec(90, 1, 0, 0), 1e-12)
                && near(fwd, new double[] {0, 0, -1}, 1e-12), v(fwd));
        double[][] C = {{1, 0, 0}, {0, 0, 1}, {0, -1, 0}};
        double[] q = norm(new double[] {0.7, 0.2, -0.4, 0.55});
        double[][] expect = mul(mul(C, GladosMath.matrix(q)), transpose(C));
        double[][] rmc = GladosMath.matrix(GladosMath.quatSimToMc(q[0], q[1], q[2], q[3]));
        double err = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                err = Math.max(err, Math.abs(expect[i][j] - rmc[i][j]));
            }
        }
        check("R_mc == C R_sim C^T (max err " + String.format(Locale.ROOT, "%.1e", err) + ")", err < 1e-12, null);
        float[] a = {0, 0, 0, 1, 0, 0, 0};
        float[] b = {1, 2, 3, (float) -cq, 0, 0, (float) -sq};
        float[] m = GladosMath.lerpPose(a, b, 0.5f, null);
        double yaw = Math.toDegrees(2 * Math.atan2(m[6] * Math.signum(m[3]), Math.abs(m[3])));
        check("lerpPose shortest arc (q vs -q), yaw 45", Math.abs(yaw - 45) < 1e-3, String.format(Locale.ROOT, "%.4f", yaw));

        // !glados here: first solid block above the head, anchor at its underside
        double[] col = {Double.NaN, Double.NaN, Double.NaN, 0.5, 0};          // head block, 2 air, top-slab above
        double y = GladosMath.ceilingAbove(21, 10, dy -> dy < col.length ? col[dy] : 0);
        check("here: top slab 3 above head block 21 -> anchor y 24.5", y == 24.5, "y=" + y);
        double y2 = GladosMath.ceilingAbove(21, 20, dy -> dy == 11 ? 0 : Double.NaN);
        check("here: full block at y 32 -> anchor y 32.0 (lab ceiling block 33 sits at 33.0)", y2 == 32.0, "y=" + y2);
        double y3 = GladosMath.ceilingAbove(21, 5, dy -> Double.NaN);
        check("here: open sky -> NaN", Double.isNaN(y3), "y=" + y3);

        double[] box = {0, 25.5, 0, 2, 33, 2};
        check("distToBox: inside 0, beside 3, diagonal 5", GladosMath.distToBox(1, 30, 1, box) == 0
                && Math.abs(GladosMath.distToBox(5, 30, 1, box) - 3) < 1e-12
                && Math.abs(GladosMath.distToBox(5, 21.5, 1, box) - 5) < 1e-12, null);
        // the 96-block cull uses the box, not the mount: standing 95 blocks below-sideways of the head still renders
        double dMount = Math.sqrt(95 * 95 + 7.5 * 7.5);
        check("tall bbox: camera 95 blocks out at floor level is within 96 of the box though " + String.format(
                Locale.ROOT, "%.1f", dMount) + " from the mount", GladosMath.distToBox(97, 25.5, 1, box) <= 96 && dMount > 95, null);
        check("angleTo: straight ahead 0, perpendicular 90", Math.abs(GladosMath.angleTo(new double[] {0, 0, -1},
                new double[] {0, 0, 0}, new double[] {0, 0, -5})) < 1e-9 && Math.abs(GladosMath.angleTo(new double[] {1, 0, 0},
                new double[] {0, 0, 0}, new double[] {0, 3, 0}) - 90) < 1e-9, null);

        int e1 = GladosMath.eyeArgb(new float[] {1f, 0.82f, 0.29f, 1f}, 0);
        int e2 = GladosMath.eyeArgb(new float[] {1f, 0.82f, 0.29f, 1f}, 1);
        int e3 = GladosMath.eyeArgb(new float[] {1f, 0.82f, 0.29f, 0.1f}, 0);
        int e4 = GladosMath.eyeArgb(null, Float.NaN);
        check("eye colour: yellow at glow 1", e1 == 0xFFFFD14A, String.format("0x%08X", e1));
        check("eye pulses brighter with talk (g/b up, clamped)", ((e2 >> 8) & 0xFF) > ((e1 >> 8) & 0xFF) && (e2 & 0xFF) > (e1 & 0xFF)
                && ((e2 >> 16) & 0xFF) == 255, String.format("0x%08X", e2));
        check("eye dims at glow 0.1 (sleep)", ((e3 >> 16) & 0xFF) < 120, String.format("0x%08X", e3));
        check("eye default when absent / NaN talk", e4 == e1, String.format("0x%08X", e4));
    }

    static double[] norm(double[] q) {
        double n = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        return new double[] {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
    }

    static double[][] mul(double[][] a, double[][] b) {
        double[][] r = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    r[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return r;
    }

    static double[][] transpose(double[][] a) {
        double[][] r = new double[3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                r[i][j] = a[j][i];
            }
        }
        return r;
    }

    // ------------------------------------------------------------------ command parser

    static void commands() {
        System.out.println("== command parser ==");
        String[][] cases = {
            {"hello world", "NOT_GLADOS", null},
            {"!duck walk", "NOT_GLADOS", null},
            {"!rp1 wave", "NOT_GLADOS", null},
            {"!gladosx", "ERROR", null},
            {"!glados dance", "SIDECAR", "{\"cmd\":\"dance\",\"from\":\"mc\"}"},
            {"!glados spooky", "SIDECAR", "{\"cmd\":\"spooky\",\"from\":\"mc\"}"},
            {"!glados spookydance", "SIDECAR", "{\"cmd\":\"spooky\",\"from\":\"mc\"}"},
            {"!glados Creepy", "SIDECAR", "{\"cmd\":\"spooky\",\"from\":\"mc\"}"},
            {"!gladosspooky", "SIDECAR", "{\"cmd\":\"spooky\",\"from\":\"mc\"}"},
            {"!gladosspookydance", "SIDECAR", "{\"cmd\":\"spooky\",\"from\":\"mc\"}"},
            {"!glados models", "MODELS", null},
            {"!glados model", "MODELS", null},
            {"!glados model mymodel", "SIDECAR", "{\"cmd\":\"model\",\"name\":\"mymodel\",\"from\":\"mc\"}"},
            {"!glados model MyOther", "SIDECAR", "{\"cmd\":\"model\",\"name\":\"myother\",\"from\":\"mc\"}"},
            {"!glados model ../x\"", "ERROR", null},
            {"!gladoshere", "HERE", null},
            {"!GLADOSHERE", "HERE", null},
            {"!gladosdance", "SIDECAR", "{\"cmd\":\"dance\",\"from\":\"mc\"}"},
            {"!gladoslook me", "SIDECAR", "{\"cmd\":\"look\",\"dir\":\"player\",\"from\":\"mc\"}"},
            {"!gladoslookleft", "SIDECAR", "{\"cmd\":\"look\",\"dir\":\"left\",\"from\":\"mc\"}"},
            {"!gladosmodels", "MODELS", null},
            {"!gladosmodel procedural", "SIDECAR", "{\"cmd\":\"model\",\"name\":\"procedural\",\"from\":\"mc\"}"},
            {"!gladosscale2", "SCALE", "2.0"},
            {"!gladoshelp", "HELP", null},
            {"!gladosidle", "SIDECAR", "{\"cmd\":\"idle\",\"from\":\"mc\"}"},
            {"!glados", "HELP", null},
            {"  !GLaDOS  ", "HELP", null},
            {"!glados help", "HELP", null},
            {"!glados idle", "SIDECAR", "{\"cmd\":\"idle\",\"from\":\"mc\"}"},
            {"!glados TRACK", "SIDECAR", "{\"cmd\":\"track\",\"from\":\"mc\"}"},
            {"!glados angry", "SIDECAR", "{\"cmd\":\"angry\",\"from\":\"mc\"}"},
            {"!glados scan", "SIDECAR", "{\"cmd\":\"scan\",\"from\":\"mc\"}"},
            {"!glados nod", "SIDECAR", "{\"cmd\":\"nod\",\"from\":\"mc\"}"},
            {"!glados shake", "SIDECAR", "{\"cmd\":\"shake\",\"from\":\"mc\"}"},
            {"!glados bow", "SIDECAR", "{\"cmd\":\"bow\",\"from\":\"mc\"}"},
            {"!glados sleep", "SIDECAR", "{\"cmd\":\"sleep\",\"from\":\"mc\"}"},
            {"!glados wake", "SIDECAR", "{\"cmd\":\"wake\",\"from\":\"mc\"}"},
            {"!glados reset", "SIDECAR", "{\"cmd\":\"reset\",\"from\":\"mc\"}"},
            {"!glados look left", "SIDECAR", "{\"cmd\":\"look\",\"dir\":\"left\",\"from\":\"mc\"}"},
            {"!glados look Right", "SIDECAR", "{\"cmd\":\"look\",\"dir\":\"right\",\"from\":\"mc\"}"},
            {"!glados look up", "SIDECAR", "{\"cmd\":\"look\",\"dir\":\"up\",\"from\":\"mc\"}"},
            {"!glados look down", "SIDECAR", "{\"cmd\":\"look\",\"dir\":\"down\",\"from\":\"mc\"}"},
            {"!glados look me", "SIDECAR", "{\"cmd\":\"look\",\"dir\":\"player\",\"from\":\"mc\"}"},
            {"!glados look", "ERROR", null},
            {"!glados look sideways", "ERROR", null},
            {"!glados here", "HERE", null},
            {"!glados scale 2.5", "SCALE", "2.5"},
            {"!glados scale 0.25", "SCALE", "0.25"},
            {"!glados scale 4", "SCALE", "4.0"},
            {"!glados scale 5", "ERROR", null},
            {"!glados scale 0.2", "ERROR", null},
            {"!glados scale abc", "ERROR", null},
            {"!glados scale NaN", "ERROR", null},
            {"!glados yaw 90", "YAW", "90.0"},
            {"!glados yaw -45.5", "YAW", "-45.5"},
            {"!glados yaw 270", "YAW", "-90.0"},
            {"!glados yaw", "ERROR", null},
            {"!glados yaw Infinity", "ERROR", null},
            {"!glados talk", "ERROR", null},
            {"!glados cake", "ERROR", null},
        };
        for (String[] t : cases) {
            GladosCommand c = GladosCommand.parse(t[0]);
            boolean ok = c.kind.name().equals(t[1]);
            boolean num = c.kind == GladosCommand.Kind.SCALE || c.kind == GladosCommand.Kind.YAW;
            String det = c.kind + (c.json != null ? " " + c.json : "") + (num ? " " + c.value : "")
                    + (c.message != null && c.kind == GladosCommand.Kind.ERROR ? " (" + c.message + ")" : "");
            if (t[2] != null) {
                ok &= num ? Double.toString(c.value).equals(t[2]) : t[2].equals(c.json);
            }
            check(String.format("%-24s", "'" + t[0] + "'"), ok, det);
        }
        check("isGlados: '!gladoshere' must never reach the server; '!duck'/'glados' pass", GladosCommand.isGlados("!gladoshere")
                && GladosCommand.isGlados(" !GLaDOS") && !GladosCommand.isGlados("!duck here") && !GladosCommand.isGlados("glados here")
                && !GladosCommand.isGlados(null), null);
        check("help lists dance, models, model <name>", GladosCommand.HELP.contains("dance") && GladosCommand.HELP.contains("!glados models")
                && GladosCommand.HELP.contains("model <name>"), GladosCommand.HELP);
        java.util.List<String> fm = SidecarClient.formatModels("{\"active\":\"b\",\"models\":[{\"name\":\"a\",\"title\":\"A\","
                + "\"credit\":\"x, CC BY\",\"triangles\":10},{\"name\":\"b\",\"title\":null,\"triangles\":5}]}");
        check("formatModels: header, '- a - A (10 tris): x, CC BY', '* b (5 tris)'", fm.size() == 3 && fm.get(1).equals("- a - A (10 tris): x, CC BY")
                && fm.get(2).equals("* b (5 tris)"), fm.toString());
    }

    // ------------------------------------------------------------------ properties round trip

    static void config() throws Exception {
        System.out.println("== config round trip ==");
        Path dir = Files.createTempDirectory("glados-test");
        Path f = dir.resolve("config/glados.properties");
        GladosConfig d = GladosConfig.load(f);
        check("missing file -> defaults (port 8794, scale 1, yaw 0, lab anchor 26.5/33/271.5 overworld)",
                d.enabled && d.port == 8794 && d.scale == 1.0 && d.yaw == 0 && d.hasAnchor() && d.ax == 26.5 && d.ay == 33.0
                        && d.az == 271.5 && "minecraft:overworld".equals(d.dim), null);
        d.save(f);
        String def = Files.readString(f);
        check("defaults file written with the anchor", def.contains("anchor.x=26.5") && def.contains("anchor.y=33.0")
                && def.contains("port=8794") && def.contains("yaw=0.0"), def.replace("\n", " | "));
        d.setAnchor(-12, 40.5, 40, "minecraft:the_nether");
        d.scale = 2.5;
        d.yaw = -90;
        d.port = 8789;
        d.enabled = false;
        d.save(f);
        GladosConfig r = GladosConfig.load(f);
        check("round trip", !r.enabled && r.port == 8789 && r.scale == 2.5 && r.yaw == -90 && r.ax == -11.5 && r.ay == 40.5
                && r.az == 40.5 && "minecraft:the_nether".equals(r.dim),
                "anchor=" + v(r.ax, r.ay, r.az) + " dim=" + r.dim + " scale=" + r.scale + " yaw=" + r.yaw);
        Files.writeString(f, "scale=99\nport=oops\nyaw=450\nanchor.x=1\n");
        GladosConfig bad = GladosConfig.load(f);
        check("bad values clamp/default; partial anchor -> default anchor; yaw normalised", bad.scale == 4.0
                && bad.port == 8794 && bad.yaw == 90 && bad.ax == 26.5, "scale=" + bad.scale + " port=" + bad.port + " yaw=" + bad.yaw);
        check("no .tmp left behind", !Files.exists(f.resolveSibling("glados.properties.tmp")), null);

        Path vf = dir.resolve("config/glados-voice.properties");
        GladosVoiceConfig vc = GladosVoiceConfig.load(vf);
        check("voice defaults: 8793, hear_range 16, head_body head, look 30", vc.url.equals("ws://127.0.0.1:8793/svc")
                && vc.hearRange == 16 && vc.headBody.equals("head") && vc.lookAngle == 30 && vc.enabled, vc.url);
        Files.writeString(vf, "url=ws://evil.example/svc\nhear_range=always\nhead_body=skull\nlook_angle=500\n");
        GladosVoiceConfig v2 = GladosVoiceConfig.load(vf);
        check("voice config: non-loopback url refused, head_body configurable, clamps", v2.url.startsWith("ws://127.0.0.1:")
                && v2.hearRange == 0 && v2.headBody.equals("skull") && v2.lookAngle == 180, v2.headBody + " " + v2.lookAngle);
        v2.save(vf);
        GladosVoiceConfig v3 = GladosVoiceConfig.load(vf);
        check("voice config round trip", v3.hearRange == 0 && v3.headBody.equals("skull"), Files.readString(vf).lines()
                .filter(s -> !s.startsWith("#")).reduce((x, y) -> x + " | " + y).orElse(""));
    }

    // ------------------------------------------------------------------ model parse v1/v2 (eye tag, normals, centroid)

    static GladosModel.Part part(GladosModel.Body b, GladosModel.Tag t) {
        for (GladosModel.Part p : b.parts) {
            if (p.tag == t) {
                return p;
            }
        }
        return null;
    }

    static int triangles(GladosModel.Body b, GladosModel.Tag t) {
        return b.triangles(t);                         // summed over textured + untextured parts of that tag
    }

    static void model() {
        System.out.println("== model parse (v1/v2 format: backwards compatibility) ==");
        String json = "{\"version\":1,\"bodies\":[{\"name\":\"mount\",\"meshes\":[{\"rgba\":[1,0.5,0,1],"
                + "\"verts\":[0,0,0, 1,0,0, 0,1,0],\"tris\":[0,1,2]},{\"rgba\":[1,1,1,0],\"verts\":[0,0,0,1,0,0,0,1,0],\"tris\":[0,1,2]}]},"
                + "{\"name\":\"head\",\"meshes\":[{\"rgba\":[0.9,0.9,0.9,1],\"verts\":[0,0,0, 2,0,0, 0,0,-1],\"tris\":[0,1,2],"
                + "\"normals\":[0,-2,0, 0,-1,0, 0,0,0]},"
                + "{\"rgba\":[1,0.82,0.29,1],\"tag\":\"eye\",\"verts\":[1,0,0, 1,1,0, 1,0,1],\"tris\":[0,1,2,0,0,9]}]},"
                + "{\"name\":\"cable\"}]}";
        GladosModel m = GladosModel.parse(json);
        check("bodies/order from JSON, root = first (mount)", m.bodies.length == 3 && m.bodies[m.root()].name.equals("mount")
                && m.index.get("head") == 1, null);
        GladosModel.Body h = m.bodies[1];
        GladosModel.Part main = part(h, GladosModel.Tag.MAIN);
        check("eye-tagged mesh in its own part; bad indices dropped; alpha-0 mesh dropped", main.triangles() == 1
                && triangles(h, GladosModel.Tag.EYE) == 1 && m.eyeTriangles == 1 && m.triangles == 3
                && triangles(m.bodies[0], GladosModel.Tag.EYE) == 0, "tris=" + m.triangles);
        check("v1: no rev / atlas / uv / skin -> untextured (white texture + rgba tint)", m.rev == -1 && m.atlasPath == null
                && !m.needsAtlas() && !main.textured && main.uv == null && m.skinned.length == 0 && m.version == 1, m.summary());
        check("per-vertex normals used (normalised, sim -y -> MC +z), zero normal repaired from its face",
                near(new double[] {main.nrm[0], main.nrm[1], main.nrm[2]}, new double[] {0, 0, 1}, 1e-6)
                        && near(new double[] {main.nrm[6], main.nrm[7], main.nrm[8]}, new double[] {0, 0, -1}, 1e-6),
                v(main.nrm[0], main.nrm[1], main.nrm[2]) + " / repaired " + v(main.nrm[6], main.nrm[7], main.nrm[8]));
        check("smooth mesh keeps shared vertices (3), flat eye mesh gets its own (3 per tri)", main.vertices() == 3
                && part(h, GladosModel.Tag.EYE).vertices() == 3, null);
        check("centroid (sim axes) of head mesh verts", near(new double[] {h.centroid[0], h.centroid[1], h.centroid[2]},
                new double[] {5.0 / 6, 1.0 / 6, 0}, 1e-6), v(h.centroid[0], h.centroid[1], h.centroid[2]));
        check("find('head') = 1; unknown name -> body with the eye; ", m.find("head") == 1 && m.find("skull") == 1
                && m.find(null) == 1, null);
        check("radius = max local vertex distance", Math.abs(m.radius - 2) < 1e-6, "r=" + m.radius);
    }

    // ------------------------------------------------------------------ model parse v3 (atlas, uv, tags, skin)

    static final String V3 = "{\"version\":3,\"name\":\"t\",\"title\":\"Test\",\"credit\":\"someone, CC BY\",\"rev\":7,"
            + "\"atlas\":\"/atlas.png?rev=7\",\"bodies\":["
            + "{\"name\":\"a\",\"parent\":null,\"joint_pos\":[0,0,0],\"rest\":[0,0,0,1,0,0,0]},"
            + "{\"name\":\"b\",\"parent\":\"a\",\"joint_pos\":[0,0,-2],\"rest\":[0,0,-2,0.70710678,0,0,0.70710678]}],"
            + "\"meshes\":["
            + "{\"body\":\"a\",\"verts\":[0,0,0, 1,0,0, 0,1,0],\"uv\":[0,0, 1,0, 0,1],\"tris\":[0,1,2],\"rgba\":[1,1,1,1]},"
            + "{\"body\":\"b\",\"tag\":\"glass\",\"verts\":[0,0,0, 1,0,0, 0,1,0],\"uv\":[0,0, 1,0, 0,1],\"tris\":[0,1,2],\"rgba\":[1,1,1,0.5]},"
            + "{\"body\":\"b\",\"tag\":\"emissive\",\"verts\":[0,0,0, 1,0,0, 0,1,0],\"tris\":[0,1,2],\"rgba\":[0.2,0.9,1,1]},"
            + "{\"body\":\"b\",\"tag\":\"eye\",\"verts\":[0,0,0, 1,0,0, 0,1,0],\"uv\":[0.1,0.2, 0.3,0.4, 0.5,0.6],\"tris\":[0,1,2],\"rgba\":[1,1,1,1]},"
            + "{\"body\":\"a\",\"name\":\"cable\",\"verts\":[0.3,0,-1, 0.2,0.1,-1.5, 0.1,0,-2.4],\"normals\":[1,0,0, 1,0,0, 1,0,0],"
            + "\"uv\":[0,0, 1,0, 0,1],\"tris\":[0,1,2],\"rgba\":[1,1,1,1],\"skin\":{\"bodies\":[\"a\",\"b\"],\"j\":[0,1, 0,1, 0,1],\"w\":[1,0.5,0]}},"
            + "{\"body\":\"nope\",\"verts\":[0,0,0, 1,0,0, 0,1,0],\"tris\":[0,1,2]},"
            + "{\"body\":\"a\",\"tag\":\"bogus\",\"verts\":[0,0,0, 1,0,0, 0,1,0],\"uv\":[0,0],\"tris\":[0,1,2],\"rgba\":[1,0,0,1]}],"
            + "\"eye\":{\"body\":\"b\",\"pos\":[1,0,0],\"axis\":[1,0,0]},\"voice\":{\"body\":\"b\",\"pos\":[0.5,0.25,0]},\"triangles\":6}";

    static void modelV3() {
        System.out.println("== model parse v3 (atlas, uv + rgba, tags, skinned) ==");
        GladosModel m = GladosModel.parse(V3);
        check("v3 header: version 3, rev 7, atlas path, name/title/credit", m.version == 3 && m.rev == 7
                && "/atlas.png?rev=7".equals(m.atlasPath) && "t".equals(m.name) && "Test".equals(m.title)
                && "someone, CC BY".equals(m.credit), m.summary());
        check("top-level meshes go to their body; unknown body dropped", m.bodies.length == 2 && m.triangles == 6,
                m.summary());
        GladosModel.Body a = m.bodies[0], b = m.bodies[1];
        GladosModel.Part gl = part(b, GladosModel.Tag.GLASS), em = part(b, GladosModel.Tag.EMISSIVE),
                ey = part(b, GladosModel.Tag.EYE);
        check("tags: glass / emissive / eye parts; unknown tag -> main", gl != null && em != null && ey != null
                && m.glassTriangles == 1 && m.emissiveTriangles == 1 && m.eyeTriangles == 1
                && triangles(a, GladosModel.Tag.MAIN) == 2, null);
        check("glass is the LAST part of its body (translucent after cutout)", b.parts[b.parts.length - 1] == gl, null);
        check("fullbright: eye + emissive, not main/glass", GladosModel.Tag.EYE.fullbright() && GladosModel.Tag.EMISSIVE.fullbright()
                && !GladosModel.Tag.MAIN.fullbright() && !GladosModel.Tag.GLASS.fullbright(), null);
        check("uv kept (textured); no uv / wrong-length uv -> untextured", gl.textured && ey.textured && !em.textured
                && em.uv == null && near(new double[] {ey.uv[0], ey.uv[1], ey.uv[4], ey.uv[5]}, new double[] {0.1, 0.2, 0.5, 0.6}, 1e-6)
                && a.parts.length == 2 && (a.parts[0].textured != a.parts[1].textured), null);
        check("glass keeps its alpha (0.5 -> 0x80), others opaque", (gl.col[0] >>> 24) == 0x80 && (em.col[0] >>> 24) == 0xFF,
                String.format("0x%08X", gl.col[0]));
        check("needsAtlas, textured triangle count", m.needsAtlas() && m.texturedTriangles == 4, "textured " + m.texturedTriangles);
        check("skinned mesh -> model.skinned (not a body part)", m.skinned.length == 1 && m.skinnedTriangles == 1
                && m.skinned[0].skinned() && m.skinned[0].skinBodies.length == 2, null);
        check("voice.pos overrides the head-point of its body", near(new double[] {b.centroid[0], b.centroid[1], b.centroid[2]},
                new double[] {0.5, 0.25, 0}, 1e-6) && m.find("b") == 1, v(b.centroid[0], b.centroid[1], b.centroid[2]));
        check("atlasPath: relative -> '/atlas.png', full URL -> path only", "/atlas.png".equals(GladosModel.atlasPath(
                new com.google.gson.JsonPrimitive("atlas.png"))) && "/atlas.png?rev=2".equals(GladosModel.atlasPath(
                new com.google.gson.JsonPrimitive("http://127.0.0.1:8794/atlas.png?rev=2"))), null);
        byte[] png = {(byte) 0x89, 'P', 'N', 'G', 13, 10, 26, 10, 0, 0, 0, 13, 'I', 'H', 'D', 'R', 0, 0, 16, 0, 0, 0, 8, 0};
        int[] wh = GladosModel.pngSize(png);
        check("pngSize reads IHDR (4096x2048); junk -> null", wh != null && wh[0] == 4096 && wh[1] == 2048
                && GladosModel.pngSize(new byte[30]) == null, wh == null ? "null" : wh[0] + "x" + wh[1]);
        // a v3 skinned mesh whose skin is unusable becomes rigid in its body (world rest -> body-local)
        String bad = V3.replace("\"j\":[0,1, 0,1, 0,1]", "\"j\":[0,1, 0,5, 0,1]");
        GladosModel mb = GladosModel.parse(bad);
        check("bad skin indices -> rigid in its \"body\" (verts world rest -> local)", mb.skinned.length == 0
                && triangles(mb.bodies[0], GladosModel.Tag.MAIN) == 3, mb.summary());
    }

    /** Full render matrix of a body as Glados.renderGlados builds it: translate(A) rotY(yaw) translate(s C p) rot(q_mc) scale(s). */
    static float[] fullMat(double ax, double ay, double az, double s, double yaw, float[] pose) {
        double[] o = GladosMath.yawVec(yaw, s * pose[0], s * pose[2], -s * pose[1]);
        double[] qmc = GladosMath.qmul(GladosMath.yawQuat(yaw), GladosMath.quatSimToMc(pose[3], pose[4], pose[5], pose[6]));
        return GladosMath.bodyMatrix(ax + o[0], ay + o[1], az + o[2], qmc, s);
    }

    static float[] fullNrm(double yaw, float[] pose) {
        return GladosMath.normalMatrix(GladosMath.qmul(GladosMath.yawQuat(yaw), GladosMath.quatSimToMc(pose[3], pose[4], pose[5], pose[6])));
    }

    static void skinning() {
        System.out.println("== linear-blend skinning (v = w*T_a*R_a^-1*v + (1-w)*T_b*R_b^-1*v) ==");
        GladosModel m = GladosModel.parse(V3);
        GladosModel.Part c = m.skinned[0];
        double ax = 26.5, ay = 33, az = 271.5;
        double[][] world = {{0.3, 0, -1}, {0.2, 0.1, -1.5}, {0.1, 0, -2.4}};
        double[] wts = {1, 0.5, 0};
        // 1) pose == rest: every vertex lands exactly on its world rest position
        float[][] rest = {m.bodies[0].rest, m.bodies[1].rest};
        float[][] mats = {fullMat(ax, ay, az, 1, 0, rest[0]), fullMat(ax, ay, az, 1, 0, rest[1])};
        float[][] nm = {fullNrm(0, rest[0]), fullNrm(0, rest[1])};
        List<float[]> out = new ArrayList<>();
        GladosModel.emitSkinned(c, mats, nm, null, (x, y, z, col, u, vv, nx, ny, nz) -> out.add(new float[] {x, y, z, nx, ny, nz, u, vv}));
        double err = 0;
        for (int i = 0; i < 3; i++) {
            double[] w = GladosMath.simToMc(ax, ay, az, 1, 0, world[i][0], world[i][1], world[i][2]);
            for (int k = 0; k < 3; k++) {
                err = Math.max(err, Math.abs(out.get(i)[k] - w[k]));
            }
        }
        check("pose == rest -> skinned verts at their world rest positions (max err " + String.format(Locale.ROOT, "%.1e", err) + ")",
                out.size() == 4 && err < 1e-4, null);
        check("skinned normal at rest: world +x -> MC +x", near(new double[] {out.get(0)[3], out.get(0)[4], out.get(0)[5]},
                new double[] {1, 0, 0}, 1e-5), v(out.get(0)[3], out.get(0)[4], out.get(0)[5]));
        check("skinned uv passthrough", out.get(1)[6] == 1f && out.get(1)[7] == 0f && out.get(2)[7] == 1f, null);
        // 2) arbitrary different poses of a and b, scale 1.4, yaw 30: compare against the formula in doubles
        double s = 1.4, yaw = 30;
        double[] qa = norm(new double[] {0.9, 0.2, -0.1, 0.3}), qb = norm(new double[] {0.4, -0.5, 0.6, 0.2});
        float[] pa = {0.3f, -0.2f, -0.5f, (float) qa[0], (float) qa[1], (float) qa[2], (float) qa[3]};
        float[] pb = {0.9f, 0.4f, -2.7f, (float) qb[0], (float) qb[1], (float) qb[2], (float) qb[3]};
        float[][] mats2 = {fullMat(ax, ay, az, s, yaw, pa), fullMat(ax, ay, az, s, yaw, pb)};
        float[][] nm2 = {fullNrm(yaw, pa), fullNrm(yaw, pb)};
        List<float[]> out2 = new ArrayList<>();
        GladosModel.emitSkinned(c, mats2, nm2, null, (x, y, z, col, u, vv, nx, ny, nz) -> out2.add(new float[] {x, y, z, nx, ny, nz}));
        double err2 = 0;
        float[][] ps = {pa, pb};
        for (int i = 0; i < 3; i++) {
            double[] want = new double[3];
            for (int bi = 0; bi < 2; bi++) {
                float[] r = rest[bi];
                double[] rq = {r[3], -r[4], -r[5], -r[6]};
                double[] loc = GladosMath.rotate(rq, world[i][0] - r[0], world[i][1] - r[1], world[i][2] - r[2]);
                double[] w = GladosMath.bodyVertexToMc(ax, ay, az, s, yaw, ps[bi], loc[0], loc[1], loc[2]);
                double f = bi == 0 ? wts[i] : 1 - wts[i];
                for (int k = 0; k < 3; k++) {
                    want[k] += f * w[k];
                }
            }
            for (int k = 0; k < 3; k++) {
                err2 = Math.max(err2, Math.abs(out2.get(i)[k] - want[k]));
            }
        }
        check("posed a/b, s 1.4, yaw 30: emitted == w*M_a*la + (1-w)*M_b*lb (max err " + String.format(Locale.ROOT, "%.1e", err2) + ")",
                err2 < 1e-3, null);
        // w=1 vertex normal follows body a only: Ry C R_a R_a,rest^-1 n
        double[] na = GladosMath.rotate(qa, 1, 0, 0);   // rest a = identity
        double[] cn = GladosMath.axis(na[0], na[1], na[2]);
        double[] wantN = GladosMath.yawVec(yaw, cn[0], cn[1], cn[2]);
        check("w=1 normal == Ry C R_a n", near(new double[] {out2.get(0)[3], out2.get(0)[4], out2.get(0)[5]}, wantN, 1e-5),
                v(out2.get(0)[3], out2.get(0)[4], out2.get(0)[5]));
        // 3) a body without pose: its verts follow the other body
        List<float[]> out3 = new ArrayList<>();
        GladosModel.emitSkinned(c, new float[][] {mats2[0], null}, new float[][] {nm2[0], null}, null,
                (x, y, z, col, u, vv, nx, ny, nz) -> out3.add(new float[] {x, y, z}));
        check("one body unposed -> part still drawn, rigid with the other", out3.size() == 4 && Float.isFinite(out3.get(2)[0]), null);
    }

    // ------------------------------------------------------------------ baked emitter == contract mapping (with yaw)

    static void emitter() {
        System.out.println("== baked emitter (renderer composition incl. yaw, uv, tint) ==");
        String json = "{\"bodies\":[{\"name\":\"head\",\"meshes\":[{\"rgba\":[0.2,0.4,0.6,1],"
                + "\"verts\":[0.1,0,0, 0,0.2,0, 0,0,0.3],\"tris\":[0,1,2]}]}]}";
        GladosModel m = GladosModel.parse(json);
        double ax = 26.5, ay = 33, az = 271.5, s = 1.3, yaw = 63;
        double[] q = norm(new double[] {0.6, -0.3, 0.5, 0.2});
        float[] pose = {0.2f, -0.4f, -5.9f, (float) q[0], (float) q[1], (float) q[2], (float) q[3]};
        float[] mat = fullMat(ax, ay, az, s, yaw, pose);
        float[] nm = fullNrm(yaw, pose);
        GladosModel.Part p0 = m.bodies[0].parts[0];
        List<float[]> out = new ArrayList<>();
        GladosModel.emit(p0, mat, nm, null, (x, y, z, c, u, vv, nx, ny, nz) -> out.add(new float[] {x, y, z, c, nx, ny, nz, u, vv}));
        check("one triangle -> 4 vertices (degenerate quad)", out.size() == 4, "n=" + out.size());
        double[][] sv = {{0.1, 0, 0}, {0, 0.2, 0}, {0, 0, 0.3}};
        double maxErr = 0;
        for (int i = 0; i < 4; i++) {
            double[] w = GladosMath.bodyVertexToMc(ax, ay, az, s, yaw, pose, sv[Math.min(i, 2)][0], sv[Math.min(i, 2)][1],
                    sv[Math.min(i, 2)][2]);
            float[] g = out.get(i);
            for (int k = 0; k < 3; k++) {
                maxErr = Math.max(maxErr, Math.abs(g[k] - w[k]));
            }
        }
        check("emitted vertices == A + Ry(yaw) s C (p + R v) (max err " + String.format(Locale.ROOT, "%.1e", maxErr) + ")",
                maxErr < 1e-3, null);
        double[] e1 = {-0.1, 0.2, 0}, e2 = {-0.1, 0, 0.3};
        double[] cr = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
        double ln = Math.sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
        double[] wn = GladosMath.rotate(q, cr[0] / ln, cr[1] / ln, cr[2] / ln);
        double[] cn = GladosMath.axis(wn[0], wn[1], wn[2]);
        double[] wantN = GladosMath.yawVec(yaw, cn[0], cn[1], cn[2]);
        float[] g = out.get(0);
        check("emitted normal == Ry C R n", near(new double[] {g[4], g[5], g[6]}, wantN, 1e-5), v(g[4], g[5], g[6]));
        int[] col = new int[1];
        float[] uvs = new float[2];
        GladosModel.emit(p0, mat, nm, null, (x, y, z, c, u, vv, nx, ny, nz) -> {
            col[0] = c;
            uvs[0] = u;
            uvs[1] = vv;
        });
        check("untextured part emits uv (0.5, 0.5) and its rgba", col[0] == 0xFF336699 && uvs[0] == 0.5f && uvs[1] == 0.5f,
                String.format("0x%08X uv %.2f,%.2f", col[0], uvs[0], uvs[1]));
        List<Integer> cols = new ArrayList<>();
        GladosModel.emit(p0, mat, nm, new float[] {0, 1, 2}, (x, y, z, c, u, vv, nx, ny, nz) -> cols.add(c));
        check("tint multiplies rgb per channel (clamped), alpha kept", cols.stream().allMatch(c -> c == 0xFF0066FF),
                String.format("0x%08X", cols.get(0)));
        // eye tint: v1/v2 yellow mesh * eyeTint(base 1) == the old eyeArgb colour; textured eye pulses with talk
        float[] angry = {1f, 0.35f, 0.1f, 1f};
        int yellow = 0xFFFFD14A;                       // rgba [1, 0.82, 0.29, 1] as baked by the parser
        int diff = 0;
        for (float[] e : new float[][] {{1f, 0.82f, 0.29f, 1f}, angry, {1f, 0.82f, 0.29f, 0.15f}}) {
            for (float talk : new float[] {0f, 0.5f}) {
                int a = GladosModel.mulTint(yellow, GladosMath.eyeTint(e, talk, 1f, null));
                int b = GladosMath.eyeArgb(e, talk);
                for (int sh = 0; sh <= 16; sh += 8) {
                    diff = Math.max(diff, Math.abs(((a >> sh) & 0xFF) - ((b >> sh) & 0xFF)));
                }
            }
        }
        check("eye tint on a v1/v2 yellow mesh == eyeArgb (angry, sleep, talk; max diff " + diff + "/255)", diff <= 3, null);
        float[] t0 = GladosMath.eyeTint(new float[] {1f, 0.82f, 0.29f, 1f}, 0, GladosMath.EYE_TEXTURED_BASE, null);
        float[] t1 = GladosMath.eyeTint(new float[] {1f, 0.82f, 0.29f, 1f}, 1, GladosMath.EYE_TEXTURED_BASE, null);
        float[] ta = GladosMath.eyeTint(angry, 0, GladosMath.EYE_TEXTURED_BASE, null);
        check("textured eye: neutral at rest (<=1, grey tint), brighter with talk, angry shifts to red", t0[0] <= 1 && t1[0] > t0[0]
                && Math.abs(t0[0] - t0[1]) < 0.02 && ta[1] < 0.6f * ta[0], v(t0[0], t0[1], t0[2]) + " talk " + v(t1[0], t1[1], t1[2])
                + " angry " + v(ta[0], ta[1], ta[2]));
        // textured parts pass uv through
        GladosModel mv = GladosModel.parse(V3);
        GladosModel.Part eyeP = part(mv.bodies[1], GladosModel.Tag.EYE);
        List<float[]> uo = new ArrayList<>();
        GladosModel.emit(eyeP, mat, nm, null, (x, y, z, c, u, vv, nx, ny, nz) -> uo.add(new float[] {u, vv}));
        check("textured part: uv passed through per vertex (3rd repeated)", near(new double[] {uo.get(0)[0], uo.get(0)[1], uo.get(2)[0],
                uo.get(3)[1]}, new double[] {0.1, 0.2, 0.5, 0.6}, 1e-6), null);
        boolean threw = false;
        try {
            GladosModel.emit(mv.skinned[0], mat, nm, null, (x, y, z, c, u, vv, nx, ny, nz) -> { });
        } catch (IllegalArgumentException e) {
            threw = true;
        }
        check("emit() refuses a skinned part (must use emitSkinned)", threw, null);
    }

    // ------------------------------------------------------------------ voice focus arbitration

    static void focus() {
        System.out.println("== voice focus (omo.voice.focus) ==");
        check("within range + looking at her -> glados", GladosFocus.wants(8, 16, 12, 30, 3), null);
        check("within range, not looking, closer to her than the duck -> glados", GladosFocus.wants(8, 16, 90, 30, 12), null);
        check("within range, not looking, closer to the duck -> duck keeps the mic", !GladosFocus.wants(8, 16, 90, 30, 3), null);
        check("out of hear_range -> never (even looking)", !GladosFocus.wants(17, 16, 0, 30, Double.NaN), null);
        check("no duck in this world counts as closer to her", GladosFocus.wants(10, 16, 170, 30, Double.NaN), null);
        check("other dimension (NaN distance) -> no", !GladosFocus.wants(Double.NaN, 16, 0, 30, Double.NaN), null);
        long now = 1_000_000L;
        check("heldBy: fresh 'glados:<ms>'", GladosFocus.heldBy("glados", "glados:" + (now - 999), now), null);
        check("heldBy: stale after 1000 ms", !GladosFocus.heldBy("glados", "glados:" + (now - 1000), now), null);
        check("heldBy: other owner / garbage / null", !GladosFocus.heldBy("glados", "duck:" + now, now)
                && !GladosFocus.heldBy("glados", "glados:xyz", now) && !GladosFocus.heldBy("glados", null, now), null);
        System.clearProperty(GladosFocus.PROP);
        GladosFocus.publish(System.currentTimeMillis());
        String val = System.getProperty("omo.voice.focus");
        boolean owned = GladosFocus.ownedNow();
        // the exact test the microduck plugin applies (copied from its patch)
        String f = System.getProperty("omo.voice.focus", "");
        boolean duckYields = f.startsWith("glados:") && System.currentTimeMillis() - Long.parseLong(f.substring(7)) < 1000;
        check("publish sets 'glados:<now>' and the duck-side check yields", owned && duckYields && val.startsWith("glados:"), val);
        System.setProperty(GladosFocus.PROP, "someone:1");
        GladosFocus.release();
        check("release never clears another owner's claim", "someone:1".equals(System.getProperty(GladosFocus.PROP)), null);
        GladosFocus.publish(System.currentTimeMillis());
        GladosFocus.release();
        check("release clears our claim", System.getProperty(GladosFocus.PROP) == null, null);
        check("duck lookup without the microduck mod -> null", GladosFocus.duckPos("minecraft:overworld") == null, null);
    }

    // ------------------------------------------------------------------ cost of a 40k-triangle v3 model

    static double sink;

    /** v3 JSON: 6 bodies x `rigidTris` textured tris with per-vertex normals + one skinned mesh of `skinTris`. */
    static String bigV3(int rigidTris, int skinTris, java.util.Random rnd) {
        StringBuilder sb = new StringBuilder("{\"version\":3,\"rev\":1,\"atlas\":\"/atlas.png?rev=1\",\"bodies\":[");
        String[] names = {"mount", "arm1", "arm2", "torso", "neck", "head"};
        for (int b = 0; b < 6; b++) {
            sb.append(b > 0 ? "," : "").append("{\"name\":\"").append(names[b]).append("\",\"rest\":[0,0,").append(-1.2 * b)
                    .append(",1,0,0,0]}");
        }
        sb.append("],\"meshes\":[");
        for (int b = 0; b <= 6; b++) {
            boolean skin = b == 6;
            int nt = skin ? skinTris : rigidTris;
            int nv = nt / 2 + 2;                       // closed meshes have ~half as many vertices as triangles
            sb.append(b > 0 ? "," : "").append("{\"body\":\"").append(skin ? "arm2" : names[b]).append("\",\"rgba\":[1,1,1,1],\"verts\":[");
            StringBuilder nrm = new StringBuilder(), uv = new StringBuilder(), j = new StringBuilder(), w = new StringBuilder();
            for (int i = 0; i < nv * 3; i++) {
                sb.append(i > 0 ? "," : "").append(String.format(Locale.ROOT, "%.4f", (rnd.nextDouble() - 0.5) * 0.8 - (skin && i % 3 == 2 ? 3 : 0)));
                nrm.append(i > 0 ? "," : "").append(String.format(Locale.ROOT, "%.3f", rnd.nextDouble() - 0.5));
            }
            for (int i = 0; i < nv; i++) {
                uv.append(i > 0 ? "," : "").append(String.format(Locale.ROOT, "%.4f,%.4f", rnd.nextDouble(), rnd.nextDouble()));
                j.append(i > 0 ? "," : "").append("0,1");
                w.append(i > 0 ? "," : "").append(String.format(Locale.ROOT, "%.3f", rnd.nextDouble()));
            }
            sb.append("],\"normals\":[").append(nrm).append("],\"uv\":[").append(uv).append("],\"tris\":[");
            for (int i = 0; i < nt * 3; i++) {
                sb.append(i > 0 ? "," : "").append(rnd.nextInt(nv));
            }
            sb.append("]");
            if (skin) {
                sb.append(",\"skin\":{\"bodies\":[\"arm2\",\"torso\"],\"j\":[").append(j).append("],\"w\":[").append(w).append("]}");
            }
            sb.append("}");
        }
        return sb.append("]}").toString();
    }

    static void perf() {
        System.out.println("== perf: 40k-triangle v3 model (6 x 6000 textured rigid + 4000 skinned; contract budget) ==");
        String json = bigV3(6000, 4000, new java.util.Random(1));
        long t0 = System.nanoTime();
        GladosModel m = GladosModel.parse(json);
        double parseMs = (System.nanoTime() - t0) / 1e6;
        check("parse " + (json.length() / 1024) + " KiB model.json", m.triangles == 40000 && m.skinnedTriangles == 4000
                && m.texturedTriangles == 40000, String.format(Locale.ROOT, "%d tris in %.0f ms (once per model load, network thread)",
                m.triangles, parseMs));
        float[] pose = {0.1f, 0.2f, -3f, 0.9f, 0.1f, 0.2f, 0.3f};
        double[] q = norm(new double[] {0.9, 0.1, 0.2, 0.3});
        pose[3] = (float) q[0];
        pose[4] = (float) q[1];
        pose[5] = (float) q[2];
        pose[6] = (float) q[3];
        float[][] mats = new float[6][], nms = new float[6][];
        for (int i = 0; i < 6; i++) {
            mats[i] = fullMat(26.5, 33, 271.5, 1, 20, pose);
            nms[i] = fullNrm(20, pose);
        }
        float[] tint = {1f, 0.9f, 0.8f};
        GladosModel.Sink s = (x, y, z, c, u, v, nx, ny, nz) -> sink += x + y + z + nx + u + c;
        Runnable frame = () -> {
            for (int i = 0; i < m.bodies.length; i++) {
                for (GladosModel.Part p : m.bodies[i].parts) {
                    GladosModel.emit(p, mats[i], nms[i], p.tag == GladosModel.Tag.EYE ? tint : null, s);
                }
            }
            for (GladosModel.Part p : m.skinned) {
                GladosModel.emitSkinned(p, mats, nms, null, s);
            }
        };
        for (int w = 0; w < 400; w++) {
            frame.run();                               // JIT warm-up
        }
        // best of 10 batches of 50 frames: robust against the (busy, gaming) host stealing the CPU now and then
        double best = Double.MAX_VALUE, sum = 0;
        int batches = 10, per = 50;
        for (int b = 0; b < batches; b++) {
            t0 = System.nanoTime();
            for (int f = 0; f < per; f++) {
                frame.run();
            }
            double ms = (System.nanoTime() - t0) / 1e6 / per;
            best = Math.min(best, ms);
            sum += ms;
        }
        check("emit 40k tris (160k vertices: uv, tint, per-vertex normals, 4k skinned) per frame < 1 ms target",
                best < 1.0, String.format(Locale.ROOT, "best %.3f ms/frame, mean %.3f ms (vertex math only; zero allocation)",
                        best, sum / batches));
    }

    // ------------------------------------------------------------------ v3 over HTTP: atlas, /models, rev-driven refetch

    static List<String> models(SidecarClient cl) throws Exception {
        CompletableFuture<List<String>> f = new CompletableFuture<>();
        cl.fetchModels(f::complete);
        return f.get(5, TimeUnit.SECONDS);
    }

    static GladosModel waitRev(SidecarClient cl, long oldRev, long ms) throws Exception {
        long until = System.currentTimeMillis() + ms;
        while (System.currentTimeMillis() < until) {
            SidecarClient.Frames f = cl.frames();
            if (f != null && f.model.rev != oldRev) {
                return f.model;
            }
            Thread.sleep(50);
        }
        return cl.frames().model;
    }

    static void v3(String url, SidecarClient cl, boolean fake) throws Exception {
        GladosModel m = cl.frames().model;
        if (m.version < 3 && m.rev < 0) {
            System.out.println("== v1/v2 sidecar at " + url + ": backwards compatibility ==");
            check("v1/v2: no rev, no atlas, untextured (white + rgba)", m.atlasPath == null && m.atlasPng() == null
                    && !m.needsAtlas(), m.summary());
            check("v1/v2: /pose without rev -> -1", cl.frames().cur.rev == -1, "rev " + cl.frames().cur.rev);
            int l0 = cl.loads();
            Thread.sleep(1500);
            check("v1/v2: exactly one model load, no refetch loop", cl.loads() == l0 && l0 == 1, "loads " + cl.loads());
            List<String> lines = models(cl);
            check("!glados models on an old sidecar -> one friendly line", lines.size() == 1
                    && lines.get(0).contains("no model list"), lines.toString());
            return;
        }
        System.out.println("== v3 at " + url + ": atlas, /models, live model switch ==");
        check("v3: rev + atlas path + atlas PNG downloaded and validated", m.rev >= 0 && m.atlasPath != null
                && m.atlasPng() != null, m.summary() + ", atlas " + m.atlasPath);
        int[] wh = m.atlasPng() == null ? null : GladosModel.pngSize(m.atlasPng());
        check("atlas is a PNG <= 4096 px", wh != null && wh[0] <= 4096 && wh[1] <= 4096,
                wh == null ? "none" : wh[0] + "x" + wh[1] + ", " + m.atlasPng().length / 1024 + " KiB");
        check("pose.rev == model.rev", cl.frames().cur.rev == m.rev, "pose rev " + cl.frames().cur.rev + ", model rev " + m.rev);
        check("model has textured (uv) triangles", m.texturedTriangles > 0, m.texturedTriangles + " textured");
        check("model has a credit line", m.credit != null && !m.credit.isBlank(), m.title + " / " + m.credit);
        if (fake) {
            check("(fake) skinned + glass + emissive + eye parts parsed", m.skinnedTriangles > 0 && m.glassTriangles > 0
                    && m.emissiveTriangles > 0 && m.eyeTriangles > 0, m.summary());
        }
        List<String> lines = models(cl);
        check("!glados models: header + one line per model, active marked '*', with credits", lines.size() >= 2
                && lines.stream().anyMatch(s -> s.startsWith("* ")) && lines.stream().skip(1).allMatch(s -> s.contains(": ")),
                String.join(" | ", lines));
        String rd = cmd(cl, GladosCommand.parse("!gladosdance").json);
        check("!gladosdance (no space) -> POST dance -> ok", rd.startsWith("ok"), rd);
        if (!fake) {
            System.out.println("SKIP live model switching on the real sidecar (she may be on stream); covered by the fake");
            return;
        }
        String other = "boxes".equals(m.name) ? "procedural" : "boxes";
        int l0 = cl.loads();
        long rev0 = m.rev;
        byte[] png0 = m.atlasPng();
        long t0 = System.nanoTime();
        String rs = cmd(cl, GladosCommand.parse("!glados model " + other).json);
        GladosModel m2 = waitRev(cl, rev0, 8000);
        double sw = (System.nanoTime() - t0) / 1e6;
        check("!glados model " + other + " -> /pose.rev changes -> model refetched", rs.startsWith("ok") && m2.rev != rev0
                && other.equals(m2.name) && cl.loads() == l0 + 1, String.format(Locale.ROOT, "%s; rev %d -> %d in %.0f ms, loads %d",
                        rs, rev0, m2.rev, sw, cl.loads()));
        check("procedural (v1-style meshes, no atlas): untextured, nothing to upload", m2.atlasPath == null
                && m2.atlasPng() == null && !m2.needsAtlas() && m2.triangles > 0, m2.summary());
        check("frames switch to the new model with a pose of the new rev", cl.frames().model == m2 && cl.frames().cur.rev == m2.rev, null);
        String rb = cmd(cl, GladosCommand.parse("!gladosmodel " + m.name).json);
        GladosModel m3 = waitRev(cl, m2.rev, 8000);
        check("switch back -> textured again with a NEW atlas (rev-specific URL refetched)", rb.startsWith("ok")
                && m.name.equals(m3.name) && m3.atlasPng() != null && !java.util.Arrays.equals(png0, m3.atlasPng())
                && cl.loads() == l0 + 2, m3.summary() + ", loads " + cl.loads());
        String ru = cmd(cl, GladosCommand.parse("!glados model nosuchmodel").json);
        check("unknown model -> readable error", ru.startsWith("error"), ru);
        String rt = cmd(cl, "{\"cmd\":\"model\",\"name\":\"procedural\",\"from\":\"twitch:someone\"}");
        check("(fake) twitch may not switch models -> error", rt.startsWith("error"), rt);
    }

    // ------------------------------------------------------------------ HTTP against the live (or fake) sidecar

    static void http(String url) throws Exception {
        System.out.println("== HTTP client against " + url + " ==");
        String health = rawGet(url + "/health");
        boolean fake = health != null && health.contains("\"fake\"");
        System.out.println("  /health: " + health);
        int port = URI.create(url).getPort();
        List<String> logs = new ArrayList<>();
        SidecarClient cl = new SidecarClient(() -> port, () -> true, s -> {
            logs.add(s);
            System.out.println("  [log] " + s);
        });
        cl.start();
        long deadline = System.currentTimeMillis() + 20000;
        while (cl.frames() == null && System.currentTimeMillis() < deadline) {
            Thread.sleep(50);
        }
        SidecarClient.Frames f = cl.frames();
        check("model + first pose fetched", f != null, null);
        if (f == null) {
            cl.stop();
            return;
        }
        GladosModel m = f.model;
        StringBuilder names = new StringBuilder();
        for (GladosModel.Body b : m.bodies) {
            names.append(b.name).append(triangles(b, GladosModel.Tag.EYE) > 0 ? "(eye)" : "").append(' ');
        }
        System.out.println("  model: " + m.summary() + ", max local extent " + String.format(Locale.ROOT, "%.3f m", m.radius) + "; bodies: " + names.toString().trim());
        check("root (first) body is the ceiling mount", m.bodies.length > 0 && m.bodies[0].name.equals("mount"),
                m.bodies.length > 0 ? m.bodies[0].name : "none");
        check("triangle budget <= ~40k (v3)", m.triangles <= 44000, m.triangles + " tris");
        check("an eye-tagged mesh exists", m.eyeTriangles > 0, m.eyeTriangles + " eye tris");
        int hi = m.find("head");
        check("voice/focus body 'head' resolves", hi >= 0 && m.bodies[hi].name.equals("head"), hi >= 0 ? m.bodies[hi].name : "none");
        long t0 = System.nanoTime();
        int changes = 0;
        long last = f.cur.seq;
        while (System.nanoTime() - t0 < 2_000_000_000L) {
            SidecarClient.Frames g = cl.frames();
            if (g != null && g.cur.seq != last) {
                changes++;
                last = g.cur.seq;
            }
            Thread.sleep(5);
        }
        SidecarClient.Frames g = cl.frames();
        check("pose stream advances (" + changes + " new poses in 2 s ~ " + changes / 2 + " Hz)", changes >= 20, null);
        int present = 0;
        double minZ = Double.MAX_VALUE, maxZ = -Double.MAX_VALUE;
        for (float[] b : g.cur.bodies) {
            if (b != null) {
                present++;
                minZ = Math.min(minZ, b[2]);
                maxZ = Math.max(maxZ, b[2]);
            }
        }
        check("pose covers every model body", present == m.bodies.length, present + "/" + m.bodies.length);
        check("she hangs from the ceiling: body origins z in [-7.6, 0.05], lowest below -3 m", maxZ <= 0.05 && minZ >= -7.6
                && minZ < -3, String.format(Locale.ROOT, "z %.3f..%.3f m", minZ, maxZ));
        check("pose.eye present (r,g,b,glow)", g.cur.eye != null && g.cur.eye.length == 4, g.cur.eye == null ? "null"
                : v(g.cur.eye[0], g.cur.eye[1], g.cur.eye[2], g.cur.eye[3]));
        float[] hp = g.cur.bodies[hi];
        float[] cen = m.bodies[hi].centroid;
        double[] head = GladosMath.bodyVertexToMc(26.5, 33, 271.5, 1, 0, hp, cen[0], cen[1], cen[2]);
        System.out.println("  latest: seq " + g.cur.seq + " mode " + g.cur.mode + " talk " + g.cur.talk + "; head (mesh centre) at MC "
                + v(head) + " with the default anchor");
        check("head is below the lab ceiling and above the floor (y 20..33)", head[1] > 20 && head[1] < 33, v(head));
        float a = SidecarClient.alpha(g, System.nanoTime());
        check("interp alpha in [0,1] and prev != cur", a >= 0 && a <= 1 && g.prev.seq != g.cur.seq, "alpha=" + a);

        // player eye position is POSTed as {x,y,z} at ~5 Hz
        double[] sim = GladosMath.mcToSim(26.5, 33, 271.5, 1, 0, 30.2, 21.62, 268.9);
        cl.setPlayer(sim);
        Thread.sleep(300);
        cl.setPlayer(new double[] {sim[0], sim[1], sim[2] + 0.25});
        Thread.sleep(500);
        String lastJson = rawGet(url + "/_last");
        if (lastJson != null) {
            com.google.gson.JsonObject pl = com.google.gson.JsonParser.parseString(lastJson).getAsJsonObject().getAsJsonObject("player");
            double[] got = {pl.get("x").getAsDouble(), pl.get("y").getAsDouble(), pl.get("z").getAsDouble()};
            check("POST /player {x,y,z} arrived (latest eye, sim metres)", near(got, new double[] {3.7, 2.6, -11.13}, 1e-6), lastJson);
        } else {
            String h = rawGet(url + "/health");
            System.out.println("SKIP POST /player echo (no /_last on the real sidecar); /health: " + h);
        }

        // harmless commands only (the real sidecar may be in view on stream): look, talk for 1.5 s, an unknown cmd
        String r = cmd(cl, GladosCommand.parse("!glados look me").json);
        check("POST /cmd look me (dir player) -> ok", r.startsWith("ok"), r);
        String rt = cmd(cl, "{\"cmd\":\"talk\",\"duration\":1.5,\"level\":0.8,\"from\":\"mc\"}");
        check("POST /cmd talk -> ok", rt.startsWith("ok"), rt);
        float maxTalk = 0;
        long until = System.currentTimeMillis() + 1500;
        while (System.currentTimeMillis() < until) {
            SidecarClient.Frames h = cl.frames();
            if (h != null) {
                maxTalk = Math.max(maxTalk, h.cur.talk);
            }
            Thread.sleep(40);
        }
        check("pose.talk rises while talking (drives the eye pulse)", maxTalk > 0.2, "max talk " + maxTalk);
        String r3 = cmd(cl, "{\"cmd\":\"nonsense\",\"from\":\"mc\"}");
        check("unknown cmd -> readable error", r3.startsWith("error"), r3);
        if (fake) {
            String ra = cmd(cl, GladosCommand.parse("!glados angry").json);
            Thread.sleep(300);
            float[] eye = cl.frames().cur.eye;
            check("(fake) angry -> eye red-orange", ra.startsWith("ok") && eye[1] < 0.5f, v(eye[0], eye[1], eye[2]));
            String rr = cmd(cl, "{\"cmd\":\"reset\",\"from\":\"twitch:someone\"}");
            check("(fake) twitch may not reset -> error", rr.startsWith("error"), rr);
        }
        v3(url, cl, fake);
        cl.stop();
    }

    static void down() throws Exception {
        System.out.println("== sidecar down (port 1) ==");
        List<String> dl = new ArrayList<>();
        SidecarClient dead = new SidecarClient(() -> 1, () -> true, dl::add);
        dead.start();
        Thread.sleep(4000);
        String r2 = cmd(dead, GladosCommand.parse("!glados nod").json);
        check("down: frames null", dead.frames() == null, null);
        check("down: exactly one log line in 4 s of retries", dl.size() == 1, dl.toString());
        check("down: command reply is friendly", r2.contains("not reachable"), r2);
        dead.stop();
    }

    static String cmd(SidecarClient cl, String json) throws Exception {
        CompletableFuture<String> rep = new CompletableFuture<>();
        cl.sendCommand(json, rep::complete);
        return rep.get(5, TimeUnit.SECONDS);
    }

    static String rawGet(String url) {
        try {
            HttpResponse<String> r = HttpClient.newHttpClient().send(HttpRequest.newBuilder(URI.create(url)).build(),
                    HttpResponse.BodyHandlers.ofString());
            return r.statusCode() == 200 ? r.body() : null;
        } catch (Exception e) {
            return null;
        }
    }
}
