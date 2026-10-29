package dev.yifan.glados;

import java.util.Locale;

/** Parser for `!glados ...` chat lines (contract "Chat commands"). Pure Java. */
public final class GladosCommand {

    public enum Kind {
        /** Not a !glados message: let it through (to other mods / the server). */
        NOT_GLADOS,
        HELP,
        /** Forward to the sidecar: {@link #json} is the POST /cmd body. */
        SIDECAR,
        HERE,
        SCALE,
        YAW,
        /** List the sidecar's models (GET /models) with credits. */
        MODELS,
        /** Bad usage; {@link #message} explains. */
        ERROR
    }

    public static final String PREFIX = "!glados";
    public static final double MIN_SCALE = 0.25;
    public static final double MAX_SCALE = 4.0;

    public static final String HELP = "!glados idle|track|angry|scan|nod|shake|bow|dance|spooky|sleep|wake|reset, "
            + "!glados look left|right|up|down|me, !glados here (hang from the ceiling above you), "
            + "!glados scale <0.25-4>, !glados yaw <deg>, !glados models, !glados model <name>";

    private static final java.util.Set<String> SIMPLE = java.util.Set.of(
            "idle", "track", "angry", "scan", "nod", "shake", "bow", "dance", "spooky", "sleep", "wake", "reset");
    /** Alternative words for a sidecar command (sent as the canonical command). */
    private static final java.util.Map<String, String> ALIASES = java.util.Map.of(
            "spookydance", "spooky", "creepy", "spooky");
    /** Every first word, longest first: `!glados<cmd>` without the space is split on the longest match. */
    private static final String[] WORDS;

    static {
        java.util.List<String> w = new java.util.ArrayList<>(SIMPLE);
        w.addAll(ALIASES.keySet());
        w.addAll(java.util.List.of("look", "here", "scale", "yaw", "models", "model", "help"));
        w.sort((x, y) -> y.length() - x.length());
        WORDS = w.toArray(new String[0]);
    }

    /** Model names the sidecar can have (asset dir names). */
    private static final java.util.regex.Pattern MODEL_NAME = java.util.regex.Pattern.compile("[a-z0-9][a-z0-9_.-]{0,39}");
    private static final java.util.Set<String> DIRS = java.util.Set.of("left", "right", "up", "down", "me", "player");

    public final Kind kind;
    public final String cmd;
    public final String json;
    public final double value;
    public final String message;

    private GladosCommand(Kind kind, String cmd, String json, double value, String message) {
        this.kind = kind;
        this.cmd = cmd;
        this.json = json;
        this.value = value;
        this.message = message;
    }

    /**
     * True if the chat line is addressed to GLaDOS (so it must not be sent to the server): it starts with `!glados`,
     * with or without a space before the command (`!gladoshere` once leaked to the server chat).
     */
    public static boolean isGlados(String line) {
        if (line == null) {
            return false;
        }
        String t = line.trim();
        return t.regionMatches(true, 0, PREFIX, 0, PREFIX.length());
    }

    /** "here" / "lookme" / "scale2" -> "here" / "look me" / "scale 2" (longest known word first); else unchanged. */
    static String splitGlued(String rest) {
        String low = rest.toLowerCase(Locale.ROOT);
        for (String w : WORDS) {
            if (low.startsWith(w)) {
                return rest.substring(0, w.length()) + " " + rest.substring(w.length());
            }
        }
        return rest;
    }

    public static GladosCommand parse(String line) {
        if (!isGlados(line)) {
            return new GladosCommand(Kind.NOT_GLADOS, null, null, 0, null);
        }
        String rest = line.trim().substring(PREFIX.length());
        if (!rest.isEmpty() && !Character.isWhitespace(rest.charAt(0))) {
            rest = splitGlued(rest);                   // !gladoshere, !gladoslook me, !gladosmodels
        }
        String[] a = rest.trim().toLowerCase(Locale.ROOT).split("\\s+");
        if (a.length == 0 || a[0].isEmpty() || a[0].equals("help") || a[0].equals("?")) {
            return new GladosCommand(Kind.HELP, "help", null, 0, HELP);
        }
        String c = a[0];
        c = ALIASES.getOrDefault(c, c);
        if (SIMPLE.contains(c)) {
            return sidecar(c, "{\"cmd\":\"" + c + "\",\"from\":\"mc\"}");
        }
        switch (c) {
            case "look": {
                if (a.length < 2 || !DIRS.contains(a[1])) {
                    return error("usage: !glados look left|right|up|down|me");
                }
                String dir = a[1].equals("me") ? "player" : a[1];
                return sidecar("look " + a[1], "{\"cmd\":\"look\",\"dir\":\"" + dir + "\",\"from\":\"mc\"}");
            }
            case "here":
                return new GladosCommand(Kind.HERE, "here", null, 0, null);
            case "models":
                return new GladosCommand(Kind.MODELS, "models", null, 0, null);
            case "model": {
                if (a.length < 2) {
                    return new GladosCommand(Kind.MODELS, "models", null, 0, null);
                }
                if (!MODEL_NAME.matcher(a[1]).matches()) {
                    return error("usage: !glados model <name> (see !glados models)");
                }
                return sidecar("model " + a[1], "{\"cmd\":\"model\",\"name\":\"" + a[1] + "\",\"from\":\"mc\"}");
            }
            case "scale": {
                Double s = number(a);
                if (s == null) {
                    return error("usage: !glados scale <" + MIN_SCALE + "-" + MAX_SCALE + ">");
                }
                if (!(s >= MIN_SCALE && s <= MAX_SCALE)) {
                    return error("scale must be between " + MIN_SCALE + " and " + MAX_SCALE);
                }
                return new GladosCommand(Kind.SCALE, "scale", null, s, null);
            }
            case "yaw": {
                Double y = number(a);
                if (y == null || !Double.isFinite(y)) {
                    return error("usage: !glados yaw <degrees>");
                }
                return new GladosCommand(Kind.YAW, "yaw", null, GladosConfig.normYaw(y), null);
            }
            default:
                return error("unknown command '" + c + "'. " + HELP);
        }
    }

    private static Double number(String[] a) {
        if (a.length < 2) {
            return null;
        }
        try {
            return Double.parseDouble(a[1]);
        } catch (NumberFormatException e) {
            return null;
        }
    }

    private static GladosCommand sidecar(String cmd, String json) {
        return new GladosCommand(Kind.SIDECAR, cmd, json, 0, null);
    }

    private static GladosCommand error(String msg) {
        return new GladosCommand(Kind.ERROR, null, null, 0, msg);
    }
}
