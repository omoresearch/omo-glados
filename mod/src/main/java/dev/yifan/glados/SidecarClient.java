package dev.yifan.glados;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BooleanSupplier;
import java.util.function.Consumer;
import java.util.function.IntSupplier;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Talks to the glados-sim sidecar on 127.0.0.1 from one daemon thread: fetches /model.json + its atlas PNG (retrying
 * with backoff; again whenever /pose.rev changes = a v3 model switch), polls /pose at ~25 Hz, POSTs the player's EYE position (/player {x,y,z}, sim metres) at ~5 Hz. Render/tick threads
 * only read the volatile {@link #frames()} snapshot and never block on HTTP. Pure Java (java.net.http + Gson).
 */
public final class SidecarClient implements Runnable {

    /** One received /pose, with body transforms aligned to the model's body order. */
    public static final class Pose {
        public final long seq;
        public final double t;
        public final String mode;
        /** 0..1 speech level. */
        public final float talk;
        /** [r,g,b,glow] 0..1, or null if absent. */
        public final float[] eye;
        /** [bodyIndex] -> {px,py,pz,qw,qx,qy,qz} in the sim frame, or null if absent. */
        public final float[][] bodies;
        public final long recvNanos;
        /** v3 model revision (changes on every model switch), -1 if the sidecar does not send it. */
        public final long rev;

        Pose(long seq, double t, String mode, float talk, float[] eye, float[][] bodies, long recvNanos) {
            this(seq, t, mode, talk, eye, bodies, recvNanos, -1);
        }

        Pose(long seq, double t, String mode, float talk, float[] eye, float[][] bodies, long recvNanos, long rev) {
            this.seq = seq;
            this.t = t;
            this.mode = mode;
            this.talk = talk;
            this.eye = eye;
            this.bodies = bodies;
            this.recvNanos = recvNanos;
            this.rev = rev;
        }
    }

    /** Last two poses (prev may equal cur), for interpolation. */
    public static final class Frames {
        public final GladosModel model;
        public final Pose prev;
        public final Pose cur;

        Frames(GladosModel model, Pose prev, Pose cur) {
            this.model = model;
            this.prev = prev;
            this.cur = cur;
        }
    }

    public static final long POSE_PERIOD_MS = 40;      // 25 Hz
    public static final long PLAYER_PERIOD_MS = 200;   // 5 Hz
    /** Min gap between model refetches triggered by a /pose.rev mismatch (a sidecar that never matches can't loop us). */
    public static final long REFETCH_GAP_MS = 3000;
    /** Atlas download cap (a 4096^2 RGBA PNG is at most ~64 MB; real ones are a few MB). */
    public static final int MAX_ATLAS_BYTES = 80 << 20;

    private final IntSupplier port;
    private final BooleanSupplier enabled;
    private final Consumer<String> log;
    private final HttpClient http;
    private volatile boolean running = true;
    private volatile GladosModel model;
    private volatile Frames frames;
    private volatile double[] playerSim;             // latest eye {x,y,z} in sim metres, or null
    private boolean down;
    private boolean everUp;
    private long nextRefetch;
    /** Number of successful model (re)loads, for tests. */
    private volatile int loads;

    public SidecarClient(IntSupplier port, BooleanSupplier enabled, Consumer<String> log) {
        this.port = port;
        this.enabled = enabled;
        this.log = log;
        this.http = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofMillis(500))
                .build();
    }

    public Thread start() {
        Thread t = new Thread(this, "glados-sidecar");
        t.setDaemon(true);
        t.setPriority(Thread.MIN_PRIORITY);
        t.start();
        return t;
    }

    public void stop() {
        running = false;
    }

    /** Null while the sidecar is down or before the first pose. */
    public Frames frames() {
        return frames;
    }

    public GladosModel model() {
        return model;
    }

    /** Player eye position in sim metres {x,y,z} (null = not in the anchor's world). Sent at ~5 Hz. */
    public void setPlayer(double[] simXYZ) {
        playerSim = simXYZ;
    }

    public String base() {
        return "http://127.0.0.1:" + port.getAsInt();
    }

    @Override
    public void run() {
        long backoff = 1000;
        long nextPlayer = 0;
        double[] lastSentPlayer = null;
        while (running) {
            try {
                if (!enabled.getAsBoolean()) {
                    frames = null;
                    Thread.sleep(500);
                    continue;
                }
                if (model == null) {
                    model = loadModel();
                    markUp("model: " + model.summary());
                    backoff = 1000;
                }
                long t0 = System.nanoTime();
                Pose p = parsePose(get("/pose", 1000), model, System.nanoTime());
                if (p.rev >= 0 && p.rev != model.rev && System.currentTimeMillis() >= nextRefetch) {
                    // the sidecar switched models (v3): refetch geometry + atlas; keep showing the old one meanwhile
                    nextRefetch = System.currentTimeMillis() + REFETCH_GAP_MS;
                    long oldRev = model.rev;
                    try {
                        GladosModel m = loadModel();
                        model = m;
                        log.accept("model changed (rev " + oldRev + " -> " + m.rev + "): " + m.summary());
                        p = parsePose(get("/pose", 1000), model, System.nanoTime());
                    } catch (Exception e) {
                        log.accept("model refetch failed (" + e.getClass().getSimpleName() + "); retrying in "
                                + REFETCH_GAP_MS / 1000 + " s");
                    }
                }
                Frames f = frames;
                if (f == null || f.model != model) {
                    frames = new Frames(model, p, p);
                } else if (p.seq != f.cur.seq) {
                    frames = new Frames(model, f.cur, p);
                }
                markUp(null);
                backoff = 1000;
                long now = System.currentTimeMillis();
                double[] pl = playerSim;
                if (pl != null && now >= nextPlayer && pl != lastSentPlayer) {
                    nextPlayer = now + PLAYER_PERIOD_MS;
                    lastSentPlayer = pl;
                    post("/player", playerJson(pl), 1000);
                }
                long sleep = POSE_PERIOD_MS - (System.nanoTime() - t0) / 1_000_000L;
                if (sleep > 0) {
                    Thread.sleep(sleep);
                }
            } catch (InterruptedException ie) {
                return;
            } catch (Exception e) {
                frames = null;                        // nothing renders while the sidecar is down
                if (!down) {
                    down = true;
                    log.accept("sidecar " + base() + " unavailable (" + e.getClass().getSimpleName()
                            + (e.getMessage() != null ? ": " + e.getMessage() : "") + "); retrying quietly");
                    model = null;                     // it may come back with different geometry
                }
                try {
                    Thread.sleep(backoff);
                } catch (InterruptedException ie) {
                    return;
                }
                backoff = Math.min(backoff * 2, 15000);
            }
        }
    }

    /** GET /model.json (+ its atlas, validated); the model is complete before it is returned/published. */
    GladosModel loadModel() throws Exception {
        GladosModel m = GladosModel.parse(get("/model.json", 30000));
        if (m.atlasPath != null) {
            try {
                byte[] png = getBytes(m.atlasPath, 30000);
                int[] wh = GladosModel.pngSize(png);
                if (wh == null || wh[0] <= 0 || wh[1] <= 0 || wh[0] > GladosModel.MAX_ATLAS || wh[1] > GladosModel.MAX_ATLAS) {
                    log.accept("atlas " + m.atlasPath + " rejected (" + (wh == null ? "not a PNG" : wh[0] + "x" + wh[1])
                            + "); textured parts drawn white");
                } else {
                    m.atlasPng = png;
                }
            } catch (java.io.IOException e) {
                log.accept("atlas " + m.atlasPath + " unavailable (" + e.getMessage() + "); textured parts drawn white");
            }
        }
        loads++;
        return m;
    }

    /** Successful model loads so far (initial + every rev change). */
    public int loads() {
        return loads;
    }

    /** Async GET /models -> human lines ("* name - title (N tris): credit"); errors become one line. */
    public void fetchModels(Consumer<List<String>> reply) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + "/models")).timeout(Duration.ofSeconds(3)).GET().build();
        http.sendAsync(req, HttpResponse.BodyHandlers.ofString()).whenComplete((r, err) -> {
            if (err != null) {
                reply.accept(List.of("sidecar not reachable at " + base()));
            } else if (r.statusCode() != 200) {
                reply.accept(List.of("this sidecar has no model list (HTTP " + r.statusCode() + ")"));
            } else {
                try {
                    reply.accept(formatModels(r.body()));
                } catch (RuntimeException e) {
                    reply.accept(List.of("bad /models reply"));
                }
            }
        });
    }

    /** GET /models JSON -> chat lines; the active one is marked with '*'. */
    public static List<String> formatModels(String json) {
        JsonObject o = JsonParser.parseString(json).getAsJsonObject();
        String active = o.has("active") && o.get("active").isJsonPrimitive() ? o.get("active").getAsString() : null;
        List<String> out = new ArrayList<>();
        JsonArray ms = o.has("models") && o.get("models").isJsonArray() ? o.getAsJsonArray("models") : new JsonArray();
        for (JsonElement e : ms) {
            if (!e.isJsonObject()) {
                continue;
            }
            JsonObject m = e.getAsJsonObject();
            String n = m.has("name") ? str(m.get("name")) : "?";
            StringBuilder sb = new StringBuilder(n.equals(active) ? "* " : "- ").append(n);
            if (m.has("title") && !m.get("title").isJsonNull()) {
                sb.append(" - ").append(str(m.get("title")));
            }
            if (m.has("triangles") && m.get("triangles").isJsonPrimitive()) {
                sb.append(" (").append(m.get("triangles").getAsLong()).append(" tris)");
            }
            if (m.has("credit") && !m.get("credit").isJsonNull()) {
                sb.append(": ").append(str(m.get("credit")));
            }
            out.add(sb.toString());
        }
        if (out.isEmpty()) {
            out.add("no models listed");
        } else {
            out.add(0, "models (" + out.size() + "), switch with !glados model <name>:");
        }
        return out;
    }

    static String playerJson(double[] p) {
        return "{\"x\":" + num(p[0]) + ",\"y\":" + num(p[1]) + ",\"z\":" + num(p[2]) + "}";
    }

    private void markUp(String detail) {
        if (down || !everUp) {
            log.accept("sidecar " + base() + " connected" + (detail != null ? " (" + detail + ")" : ""));
            down = false;
            everUp = true;
        } else if (detail != null) {
            log.accept(detail);
        }
    }

    /** Fire-and-forget POST /cmd; reply gets a one-line human result (on an HttpClient thread). */
    public void sendCommand(String json, Consumer<String> reply) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + "/cmd"))
                .timeout(Duration.ofSeconds(2))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json))
                .build();
        http.sendAsync(req, HttpResponse.BodyHandlers.ofString()).whenComplete((r, err) -> {
            if (err != null) {
                reply.accept("sidecar not reachable at " + base());
                return;
            }
            try {
                JsonObject o = JsonParser.parseString(r.body()).getAsJsonObject();
                boolean ok = o.has("ok") && o.get("ok").getAsBoolean();
                if (ok) {
                    reply.accept("ok" + (o.has("mode") ? " (mode: " + str(o.get("mode")) + ")" : ""));
                } else {
                    reply.accept("error: " + (o.has("error") ? str(o.get("error")) : "HTTP " + r.statusCode()));
                }
            } catch (Exception e) {
                reply.accept("bad reply (HTTP " + r.statusCode() + ")");
            }
        });
    }

    /** Fire-and-forget POST (e.g. /config scale_hint); errors ignored. */
    public void postAsync(String path, String json) {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + path))
                .timeout(Duration.ofSeconds(2))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json)).build();
        http.sendAsync(req, HttpResponse.BodyHandlers.discarding());
    }

    private static String str(JsonElement e) {
        return e == null || e.isJsonNull() ? "?" : e.isJsonPrimitive() ? e.getAsString() : e.toString();
    }

    String get(String path, long timeoutMs) throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + path))
                .timeout(Duration.ofMillis(timeoutMs)).GET().build();
        HttpResponse<String> r = http.send(req, HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() != 200) {
            throw new java.io.IOException("HTTP " + r.statusCode() + " for " + path);
        }
        return r.body();
    }

    byte[] getBytes(String path, long timeoutMs) throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + path))
                .timeout(Duration.ofMillis(timeoutMs)).GET().build();
        HttpResponse<java.io.InputStream> r = http.send(req, HttpResponse.BodyHandlers.ofInputStream());
        try (java.io.InputStream in = r.body()) {
            if (r.statusCode() != 200) {
                throw new java.io.IOException("HTTP " + r.statusCode() + " for " + path);
            }
            byte[] b = in.readNBytes(MAX_ATLAS_BYTES + 1);
            if (b.length > MAX_ATLAS_BYTES) {
                throw new java.io.IOException("larger than " + (MAX_ATLAS_BYTES >> 20) + " MB");
            }
            return b;
        }
    }

    void post(String path, String json, long timeoutMs) throws Exception {
        HttpRequest req = HttpRequest.newBuilder(URI.create(base() + path))
                .timeout(Duration.ofMillis(timeoutMs))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(json)).build();
        http.send(req, HttpResponse.BodyHandlers.discarding());
    }

    static String num(double d) {
        return Double.isFinite(d) ? Double.toString(d) : "0";
    }

    public static Pose parsePose(String json, GladosModel model, long recvNanos) {
        JsonObject o = JsonParser.parseString(json).getAsJsonObject();
        float[][] bodies = new float[model.bodies.length][];
        JsonObject b = o.has("bodies") && o.get("bodies").isJsonObject() ? o.getAsJsonObject("bodies") : new JsonObject();
        for (var e : b.entrySet()) {
            Integer i = model.index.get(e.getKey());
            if (i == null || !e.getValue().isJsonArray() || e.getValue().getAsJsonArray().size() < 7) {
                continue;
            }
            float[] v = new float[7];
            for (int k = 0; k < 7; k++) {
                v[k] = e.getValue().getAsJsonArray().get(k).getAsFloat();
            }
            float n = (float) Math.sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5] + v[6] * v[6]);
            if (!(n > 1e-6f) || !Float.isFinite(v[0] + v[1] + v[2])) {
                continue;
            }
            for (int k = 3; k < 7; k++) {
                v[k] /= n;
            }
            bodies[i] = v;
        }
        float[] eye = null;
        if (o.has("eye") && o.get("eye").isJsonArray()) {
            JsonArray ea = o.getAsJsonArray("eye");
            if (ea.size() >= 3) {
                eye = new float[] {ea.get(0).getAsFloat(), ea.get(1).getAsFloat(), ea.get(2).getAsFloat(),
                    ea.size() >= 4 ? ea.get(3).getAsFloat() : 1f};
            }
        }
        float talk = 0;
        if (o.has("talk") && o.get("talk").isJsonPrimitive()) {
            talk = o.get("talk").getAsFloat();
            talk = Float.isFinite(talk) ? Math.max(0, Math.min(1, talk)) : 0;
        }
        long rev = o.has("rev") && o.get("rev").isJsonPrimitive() ? o.get("rev").getAsLong() : -1;
        return new Pose(
                o.has("seq") ? o.get("seq").getAsLong() : 0,
                o.has("t") ? o.get("t").getAsDouble() : 0,
                o.has("mode") ? str(o.get("mode")) : "?",
                talk, eye, bodies, recvNanos, rev);
    }

    /** Interpolated sim pose of body i, or null if the newest pose lacks it. */
    public static float[] interp(Frames f, int i, float alpha) {
        if (i < 0 || i >= f.cur.bodies.length) {
            return null;
        }
        float[] a = f.prev.bodies[i], b = f.cur.bodies[i];
        if (b == null) {
            return null;
        }
        return a == null ? b : GladosMath.lerpPose(a, b, alpha, null);
    }

    /**
     * Interpolation factor between prev and cur: we display one poll interval behind the newest pose, so
     * alpha = (now - cur.recv) / (cur.recv - prev.recv), clamped to [0, 1].
     */
    public static float alpha(Frames f, long nowNanos) {
        long span = f.cur.recvNanos - f.prev.recvNanos;
        if (span <= 0) {
            return 1f;
        }
        float a = (float) (nowNanos - f.cur.recvNanos) / (float) span;
        return a < 0 ? 0 : a > 1 ? 1 : a;
    }
}
