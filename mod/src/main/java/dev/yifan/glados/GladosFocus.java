package dev.yifan.glados;

import java.lang.reflect.Method;

/**
 * Microphone focus between the companions' Simple Voice Chat plugins (contract "Voice"): the plugins live in one JVM
 * and share the System property {@code omo.voice.focus} = {@code "<owner>:<epoch ms>"}. GLaDOS wants the mic when the
 * player is within hear_range of her head AND (looking within ~30 deg of her head OR closer to her head than to the
 * duck). Several companions can want it at once (the RP1 mod does the same), so each publishes a scored claim
 * {@code omo.voice.claim.<owner>} = {@code "<epoch ms>:<score>"} every tick (looking: 2 + (1 - angle/180); otherwise
 * 1 / (1 + dist)) and only the best fresh claim publishes the focus ({@link #arbitrate}). The microduck plugin ignores
 * mic packets while any non-duck focus is < 1000 ms old. Pure Java (the duck lookup is reflection).
 */
public final class GladosFocus {
    public static final String PROP = "omo.voice.focus";
    public static final String CLAIM = "omo.voice.claim.";
    public static final String OWNER = "glados";
    public static final long FRESH_MS = 1000;

    private GladosFocus() {
    }

    /**
     * The contract rule. distHead: blocks from the player's eye to her head (NaN = other dimension / unknown);
     * hearRange <= 0 means "always"; angleDeg: between the view direction and the direction to her head; distDuck:
     * blocks to the duck's head (NaN/infinite = no duck in this world).
     */
    public static boolean wants(double distHead, double hearRange, double angleDeg, double maxAngleDeg, double distDuck) {
        if (Double.isNaN(distHead)) {
            return hearRange <= 0 && Double.isNaN(distDuck);
        }
        if (hearRange > 0 && distHead > hearRange) {
            return false;
        }
        boolean looking = angleDeg <= maxAngleDeg;
        boolean closer = !(distDuck <= distHead);       // NaN (no duck) counts as "closer to her"
        return looking || closer;
    }

    /** {@link #wants} as a claim score: 0 = no claim; looking = 2 + (1 - angle/180); else 1 / (1 + dist) (nearest wins). */
    public static double score(double distHead, double hearRange, double angleDeg, double maxAngleDeg, double distDuck) {
        if (!wants(distHead, hearRange, angleDeg, maxAngleDeg, distDuck)) {
            return 0;
        }
        if (Double.isNaN(distHead)) {
            return 0.01;
        }
        return angleDeg <= maxAngleDeg ? 2 + (1 - angleDeg / 180.0) : 1 / (1 + distHead);
    }

    /** The owner of the best fresh claim (any companion), or null. Ties: owner name order. */
    public static String winner(long nowMs) {
        String best = null;
        double bestScore = 0;
        for (String k : System.getProperties().stringPropertyNames()) {
            if (!k.startsWith(CLAIM) || k.length() == CLAIM.length()) {
                continue;
            }
            String v = System.getProperty(k);
            int c = v == null ? -1 : v.indexOf(':');
            if (c <= 0) {
                continue;
            }
            try {
                long age = nowMs - Long.parseLong(v.substring(0, c).trim());
                double sc = Double.parseDouble(v.substring(c + 1).trim());
                String owner = k.substring(CLAIM.length());
                if (age >= -FRESH_MS && age < FRESH_MS && sc > 0
                        && (best == null || sc > bestScore || (sc == bestScore && owner.compareTo(best) < 0))) {
                    best = owner;
                    bestScore = sc;
                }
            } catch (NumberFormatException e) {
                // malformed claim: ignore
            }
        }
        return best;
    }

    /** One client tick: publish our claim (score 0 = withdraw it); hold the focus only while our claim is the best. */
    public static boolean arbitrate(double score, long nowMs) {
        if (score > 0) {
            System.setProperty(CLAIM + OWNER, nowMs + ":" + score);
        } else {
            System.clearProperty(CLAIM + OWNER);
        }
        boolean win = score > 0 && OWNER.equals(winner(nowMs));
        if (win) {
            publish(nowMs);
        } else {
            release();
        }
        return win;
    }

    /** Withdraw both our claim and our focus (voice chat stopped). */
    public static void withdraw() {
        System.clearProperty(CLAIM + OWNER);
        release();
    }

    /** Publish ownership for this tick. */
    public static void publish(long nowMs) {
        System.setProperty(PROP, OWNER + ":" + nowMs);
    }

    /** Drop our claim (only if the property is ours), so the duck gets the mic back immediately. */
    public static void release() {
        String v = System.getProperty(PROP);
        if (v != null && v.startsWith(OWNER + ":")) {
            System.clearProperty(PROP);
        }
    }

    /** True if {@code value} is "owner:ms" with ms no older than FRESH_MS at nowMs. */
    public static boolean heldBy(String owner, String value, long nowMs) {
        if (value == null || !value.startsWith(owner + ":")) {
            return false;
        }
        try {
            long t = Long.parseLong(value.substring(owner.length() + 1).trim());
            long age = nowMs - t;
            return age >= -FRESH_MS && age < FRESH_MS;   // small negative = clock skew between threads: still fresh
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /** Does GLaDOS currently own the mic (fresh property)? */
    public static boolean ownedNow() {
        return heldBy(OWNER, System.getProperty(PROP), System.currentTimeMillis());
    }

    // ---------------------------------------------------------------- the duck's head (microduck mod, if present)

    private static volatile Method duckPos, duckDim;
    private static volatile boolean duckLookedUp;

    /** Microduck head position in MC coords via its DuckVoicePlugin.duckPos(), or null (mod absent / no anchor). */
    public static double[] duckPos(String wantDim) {
        try {
            if (!duckLookedUp) {
                duckLookedUp = true;
                Class<?> c = Class.forName("dev.yifan.microduck.DuckVoicePlugin", false, GladosFocus.class.getClassLoader());
                Method p = c.getDeclaredMethod("duckPos");
                Method d = c.getDeclaredMethod("duckDim");
                p.setAccessible(true);
                d.setAccessible(true);
                duckPos = p;
                duckDim = d;
            }
            Method p = duckPos, d = duckDim;
            if (p == null) {
                return null;
            }
            Object dim = d.invoke(null);
            if (wantDim != null && !wantDim.equals(dim)) {
                return null;
            }
            Object r = p.invoke(null);
            return r instanceof double[] a && a.length >= 3 ? a : null;
        } catch (Throwable t) {                          // class absent (no microduck / no SVC) or API changed
            duckPos = null;
            return null;
        }
    }
}
