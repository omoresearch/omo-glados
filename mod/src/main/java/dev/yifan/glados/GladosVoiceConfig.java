package dev.yifan.glados;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

/** config/glados-voice.properties: the Simple Voice Chat bridge to a local GLaDOS voice backend (not included). Pure Java. */
public final class GladosVoiceConfig {
    public volatile boolean enabled = true;
    /** Voice backend (your own, see the README \"Voice\"). Loopback only. */
    public volatile String url = "ws://127.0.0.1:8793/svc";
    /** Only the local player's mic within this many blocks of her head is sent; 0 = always (any distance/dimension). */
    public volatile double hearRange = 16.0;
    /** Optional player name to listen to instead of the local (host) player. */
    public volatile String player = "";
    /** How far her voice carries (SVC locational channel distance, blocks). She fills the room: a bit more than the duck. */
    public volatile float voiceDistance = 32f;
    /** Show what she says (and what she heard) in chat. */
    public volatile boolean captions = true;
    /** Body from /model.json whose mesh centre is her voice position and focus target (fallback: the eye's body). */
    public volatile String headBody = "head";
    /** Looking within this many degrees of her head grabs mic focus (when also within hear_range). */
    public volatile double lookAngle = 30.0;

    public static GladosVoiceConfig load(Path file) {
        GladosVoiceConfig c = new GladosVoiceConfig();
        if (!Files.exists(file)) {
            return c;
        }
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            p.load(r);
        } catch (IOException e) {
            return c;
        }
        c.enabled = !"false".equalsIgnoreCase(p.getProperty("enabled", "true").trim());
        String u = p.getProperty("url", c.url).trim();
        if (u.startsWith("ws://127.0.0.1:") || u.startsWith("ws://localhost:")) {
            c.url = u; // loopback only: this carries the player's microphone
        }
        String range = p.getProperty("hear_range", "16").trim();
        c.hearRange = "always".equalsIgnoreCase(range) ? 0 : clamp(parse(range, 16), 0, 256);
        c.player = p.getProperty("player", "").trim();
        c.voiceDistance = (float) clamp(parse(p.getProperty("voice_distance", "32"), 32), 4, 128);
        c.captions = !"false".equalsIgnoreCase(p.getProperty("captions", "true").trim());
        String hb = p.getProperty("head_body", "head").trim();
        c.headBody = hb.isEmpty() ? "head" : hb;
        c.lookAngle = clamp(parse(p.getProperty("look_angle", "30"), 30), 1, 180);
        return c;
    }

    public void save(Path file) throws IOException {
        Properties p = new Properties();
        p.setProperty("enabled", Boolean.toString(enabled));
        p.setProperty("url", url);
        p.setProperty("hear_range", hearRange <= 0 ? "always" : Double.toString(hearRange));
        p.setProperty("player", player);
        p.setProperty("voice_distance", Float.toString(voiceDistance));
        p.setProperty("captions", Boolean.toString(captions));
        p.setProperty("head_body", headBody);
        p.setProperty("look_angle", Double.toString(lookAngle));
        Files.createDirectories(file.toAbsolutePath().getParent());
        try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            p.store(w, "GLaDOS voice (Simple Voice Chat push-to-talk near her head -> your local voice backend; reply plays from"
                    + " her head). hear_range: blocks or 'always'. player: blank = you. head_body: /model.json body name.");
        }
    }

    /** Whether a mic packet from a player at distance {@code dist} (blocks, NaN = other dimension) is in range. */
    public boolean inRange(double dist) {
        if (hearRange <= 0) {
            return true;
        }
        return !Double.isNaN(dist) && dist <= hearRange;
    }

    private static double parse(String s, double def) {
        try {
            double v = Double.parseDouble(s.trim());
            return Double.isFinite(v) ? v : def;
        } catch (RuntimeException e) {
            return def;
        }
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
