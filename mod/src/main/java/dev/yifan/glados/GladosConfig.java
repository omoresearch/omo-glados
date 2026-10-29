package dev.yifan.glados;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Locale;
import java.util.Properties;

/**
 * config/glados.properties: anchor (MC coords of the sim origin = ceiling underside), dimension, scale, yaw, sidecar
 * port, enabled. Defaults to the lab ceiling from the contract (26.5, 33.0, 271.5, overworld). Pure Java.
 */
public final class GladosConfig {
    public static final int DEFAULT_PORT = 8794;
    public static final double DEFAULT_SCALE = 1.0;
    public static final double DEF_AX = 26.5, DEF_AY = 33.0, DEF_AZ = 271.5;
    public static final String DEF_DIM = "minecraft:overworld";

    public volatile boolean enabled = true;
    public volatile int port = DEFAULT_PORT;
    public volatile double scale = DEFAULT_SCALE;
    /** Degrees about the anchor's vertical axis (-180..180]. */
    public volatile double yaw = 0;
    public volatile double ax = DEF_AX, ay = DEF_AY, az = DEF_AZ;
    public volatile String dim = DEF_DIM;

    public boolean hasAnchor() {
        return dim != null && !dim.isEmpty();
    }

    /** Anchor at the centre (x/z) of block column bx/bz at height y (the ceiling underside). */
    public void setAnchor(int bx, double y, int bz, String dimension) {
        ax = bx + 0.5;
        ay = y;
        az = bz + 0.5;
        dim = dimension;
    }

    public static double normYaw(double d) {
        if (!Double.isFinite(d)) {
            return 0;
        }
        double r = d % 360.0;
        if (r > 180) {
            r -= 360;
        } else if (r <= -180) {
            r += 360;
        }
        return r;
    }

    public static GladosConfig load(Path file) {
        GladosConfig c = new GladosConfig();
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
        c.port = parseInt(p.getProperty("port"), DEFAULT_PORT);
        double s = parseDouble(p.getProperty("scale"), DEFAULT_SCALE);
        c.scale = Math.max(GladosCommand.MIN_SCALE, Math.min(GladosCommand.MAX_SCALE, s));
        c.yaw = normYaw(parseDouble(p.getProperty("yaw"), 0));
        String d = p.getProperty("dim", DEF_DIM).trim();
        if (!d.isEmpty() && p.getProperty("anchor.x") != null && p.getProperty("anchor.y") != null
                && p.getProperty("anchor.z") != null) {
            c.ax = parseDouble(p.getProperty("anchor.x"), DEF_AX);
            c.ay = parseDouble(p.getProperty("anchor.y"), DEF_AY);
            c.az = parseDouble(p.getProperty("anchor.z"), DEF_AZ);
            c.dim = d;
        }
        return c;
    }

    public void save(Path file) throws IOException {
        Properties p = new Properties();
        p.setProperty("enabled", Boolean.toString(enabled));
        p.setProperty("port", Integer.toString(port));
        p.setProperty("scale", fmt(scale));
        p.setProperty("yaw", fmt(yaw));
        p.setProperty("anchor.x", fmt(ax));
        p.setProperty("anchor.y", fmt(ay));
        p.setProperty("anchor.z", fmt(az));
        p.setProperty("dim", dim == null ? DEF_DIM : dim);
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        try (Writer w = Files.newBufferedWriter(tmp, StandardCharsets.UTF_8)) {
            p.store(w, "GLaDOS renderer (anchor = ceiling mount = sim origin in MC coords; !glados here / scale / yaw)");
        }
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    static String fmt(double d) {
        return String.format(Locale.ROOT, "%.4f", d).replaceAll("0+$", "").replaceAll("\\.$", ".0");
    }

    private static int parseInt(String s, int def) {
        try {
            return s == null ? def : Integer.parseInt(s.trim());
        } catch (NumberFormatException e) {
            return def;
        }
    }

    private static double parseDouble(String s, double def) {
        try {
            double v = s == null ? def : Double.parseDouble(s.trim());
            return Double.isFinite(v) ? v : def;
        } catch (NumberFormatException e) {
            return def;
        }
    }
}
