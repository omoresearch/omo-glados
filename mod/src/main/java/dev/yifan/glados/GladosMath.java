package dev.yifan.glados;

/**
 * Sim (MuJoCo, z-up, metres, origin = ceiling mount) <-> Minecraft mapping from sidecar/CONTRACT.md, plus the
 * optional {@code yaw} config that turns the whole model about the anchor's vertical axis. Pure Java, no MC classes.
 * <p>Axis change C: (x, y, z) -> (x, z, -y). Without yaw: MC = A + s*C*p, rotations R_mc = C R C^T, i.e. quaternion
 * [w, x, z, -y]. With yaw Y (degrees, MC rotation about +y, JOML/right-handed convention: +90 turns MC +x toward -z,
 * which is the same as a sim +z yaw because sim +y maps to MC -z): MC = A + Ry(Y) * s*C*p and R_mc = Ry(Y) C R C^T.
 * The player conversion inverts that: sim = C^T Ry(-Y) (P - A) / s.
 */
public final class GladosMath {
    private GladosMath() {
    }

    /** C applied to a vector. */
    public static double[] axis(double x, double y, double z) {
        return new double[] {x, z, -y};
    }

    /** Rotate an MC vector about +y by yaw degrees (x' = x cos + z sin, z' = -x sin + z cos). */
    public static double[] yawVec(double yawDeg, double x, double y, double z) {
        double r = Math.toRadians(yawDeg), c = Math.cos(r), s = Math.sin(r);
        return new double[] {c * x + s * z, y, -s * x + c * z};
    }

    /** MC position of sim point p: A + Ry(yaw) * s * C p. */
    public static double[] simToMc(double ax, double ay, double az, double s, double yawDeg, double x, double y, double z) {
        double[] v = yawVec(yawDeg, s * x, s * z, -s * y);
        return new double[] {ax + v[0], ay + v[1], az + v[2]};
    }

    /** Player MC eye (px,py,pz) -> sim (x,y,z) = C^T Ry(-yaw) (P - A) / s. */
    public static double[] mcToSim(double ax, double ay, double az, double s, double yawDeg, double px, double py, double pz) {
        double[] d = yawVec(-yawDeg, px - ax, py - ay, pz - az);
        return new double[] {d[0] / s, -d[2] / s, d[1] / s};
    }

    /** Sim quaternion [w,x,y,z] -> MC quaternion [w,x,y,z] (= C R C^T), before the yaw. */
    public static double[] quatSimToMc(double w, double x, double y, double z) {
        return new double[] {w, x, z, -y};
    }

    /** Quaternion [w,x,y,z] for a rotation of yaw degrees about MC +y. */
    public static double[] yawQuat(double yawDeg) {
        double h = Math.toRadians(yawDeg) / 2;
        return new double[] {Math.cos(h), 0, Math.sin(h), 0};
    }

    /** Hamilton product a*b of [w,x,y,z] quaternions. */
    public static double[] qmul(double[] a, double[] b) {
        return new double[] {
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
    }

    /** Rotate v by unit quaternion q = [w,x,y,z]. */
    public static double[] rotate(double[] q, double vx, double vy, double vz) {
        double w = q[0], qx = q[1], qy = q[2], qz = q[3];
        double tx = 2 * (qy * vz - qz * vy);
        double ty = 2 * (qz * vx - qx * vz);
        double tz = 2 * (qx * vy - qy * vx);
        return new double[] {
            vx + w * tx + (qy * tz - qz * ty),
            vy + w * ty + (qz * tx - qx * tz),
            vz + w * tz + (qx * ty - qy * tx)};
    }

    /** 3x3 rotation matrix (row-major) of unit quaternion [w,x,y,z]. */
    public static double[][] matrix(double[] q) {
        double w = q[0], x = q[1], y = q[2], z = q[3];
        return new double[][] {
            {1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
            {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
            {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
    }

    /**
     * MC position of a body-local sim vertex v for a body at sim pose (p, q), composed exactly like the renderer:
     * translate(A) * rotY(yaw) * translate(s*C p) * rotate(q_mc) * scale(s) applied to C v.
     */
    public static double[] bodyVertexToMc(double ax, double ay, double az, double s, double yawDeg, float[] pose,
                                          double vx, double vy, double vz) {
        double[] qmc = quatSimToMc(pose[3], pose[4], pose[5], pose[6]);
        double[] cv = axis(vx, vy, vz);
        double[] r = rotate(qmc, s * cv[0], s * cv[1], s * cv[2]);
        double[] local = {s * pose[0] + r[0], s * pose[2] + r[1], -s * pose[1] + r[2]};
        double[] w = yawVec(yawDeg, local[0], local[1], local[2]);
        return new double[] {ax + w[0], ay + w[1], az + w[2]};
    }

    /** Column-major (JOML layout) 4x4 of translate(t) * rotate(q) * scale(s). */
    public static float[] bodyMatrix(double tx, double ty, double tz, double[] q, double s) {
        double[][] r = matrix(q);
        float[] m = new float[16];
        for (int c = 0; c < 3; c++) {
            for (int row = 0; row < 3; row++) {
                m[4 * c + row] = (float) (r[row][c] * s);
            }
        }
        m[12] = (float) tx;
        m[13] = (float) ty;
        m[14] = (float) tz;
        m[15] = 1;
        return m;
    }

    /** Column-major 3x3 normal matrix (pure rotation) of q. */
    public static float[] normalMatrix(double[] q) {
        double[][] r = matrix(q);
        float[] n = new float[9];
        for (int c = 0; c < 3; c++) {
            for (int row = 0; row < 3; row++) {
                n[3 * c + row] = (float) r[row][c];
            }
        }
        return n;
    }

    /** Interpolate two [px,py,pz,qw,qx,qy,qz] poses: lerp position, shortest-arc nlerp rotation. */
    public static float[] lerpPose(float[] a, float[] b, float t, float[] out) {
        if (out == null) {
            out = new float[7];
        }
        for (int i = 0; i < 3; i++) {
            out[i] = a[i] + (b[i] - a[i]) * t;
        }
        float dot = a[3] * b[3] + a[4] * b[4] + a[5] * b[5] + a[6] * b[6];
        float sign = dot < 0 ? -1f : 1f;
        float n = 0;
        for (int i = 3; i < 7; i++) {
            out[i] = a[i] + (sign * b[i] - a[i]) * t;
            n += out[i] * out[i];
        }
        n = (float) Math.sqrt(n);
        if (n < 1e-6f) {
            System.arraycopy(b, 3, out, 3, 4);
        } else {
            for (int i = 3; i < 7; i++) {
                out[i] /= n;
            }
        }
        return out;
    }

    /**
     * Eye colour (0xFFRRGGBB) from pose.eye = [r,g,b,glow] and pose.talk: brightness 0.35 + 0.65*glow, pulsed up to
     * +40% by talk (the mod is mouthless, so the eye flickers with her speech level). Null eye -> default yellow.
     */
    public static int eyeArgb(float[] eye, float talk) {
        float r = 1f, g = 0.82f, b = 0.29f, glow = 1f;
        if (eye != null && eye.length >= 3) {
            r = eye[0];
            g = eye[1];
            b = eye[2];
            glow = eye.length >= 4 ? eye[3] : 1f;
        }
        float t = Float.isFinite(talk) ? Math.max(0, Math.min(1, talk)) : 0;
        float k = (0.35f + 0.65f * clamp01(glow)) * (1f + 0.4f * t);
        return 0xFF000000 | (c255(r * k) << 16) | (c255(g * k) << 8) | c255(b * k);
    }

    /** The design eye colour (#FFD24A-ish); eye meshes are authored in it, so pose.eye is applied relative to it. */
    public static final float[] EYE_DEFAULT = {1f, 0.82f, 0.29f};
    /** Brightness of a textured eye at rest: leaves headroom for the talk pulse (a texture cannot exceed 1). */
    public static final float EYE_TEXTURED_BASE = 0.75f;

    /**
     * Per-channel multiplier for eye-tagged vertices: (pose.eye rgb / {@link #EYE_DEFAULT}) * brightness * base, where
     * brightness = (0.35 + 0.65*glow)(1 + 0.4*talk) like {@link #eyeArgb}. A mesh authored in the default yellow
     * (v1/v2, white texture, base 1) thus gets exactly eyeArgb; a textured eye keeps its texture at rest (x base) and
     * turns red-orange / dims / pulses with pose.eye and talk.
     */
    public static float[] eyeTint(float[] eye, float talk, float base, float[] out) {
        if (out == null) {
            out = new float[3];
        }
        float r = EYE_DEFAULT[0], g = EYE_DEFAULT[1], b = EYE_DEFAULT[2], glow = 1f;
        if (eye != null && eye.length >= 3 && Float.isFinite(eye[0] + eye[1] + eye[2])) {
            r = eye[0];
            g = eye[1];
            b = eye[2];
            glow = eye.length >= 4 ? eye[3] : 1f;
        }
        float t = Float.isFinite(talk) ? Math.max(0, Math.min(1, talk)) : 0;
        float k = (0.35f + 0.65f * clamp01(glow)) * (1f + 0.4f * t) * base;
        out[0] = Math.max(0, r) / EYE_DEFAULT[0] * k;
        out[1] = Math.max(0, g) / EYE_DEFAULT[1] * k;
        out[2] = Math.max(0, b) / EYE_DEFAULT[2] * k;
        return out;
    }

    private static float clamp01(float f) {
        return Float.isFinite(f) ? Math.max(0, Math.min(1, f)) : 1f;
    }

    private static int c255(float f) {
        return Float.isFinite(f) ? Math.max(0, Math.min(255, Math.round(f * 255f))) : 255;
    }

    /**
     * `!glados here`: scan straight up from the block containing the player's head. {@code columnBottom.apply(dy)}
     * returns the underside offset (0..1, from the collision shape's min Y) of the block {@code dy} blocks above the
     * head block, or NaN if that block has no collision (air, torches, plants...). Returns the anchor y = headBlockY +
     * dy + underside of the first solid block, or NaN if none within maxUp blocks.
     */
    public static double ceilingAbove(int headBlockY, int maxUp, java.util.function.IntToDoubleFunction columnBottom) {
        for (int dy = 0; dy <= maxUp; dy++) {
            double b = columnBottom.applyAsDouble(dy);
            if (!Double.isNaN(b)) {
                return headBlockY + dy + Math.max(0, Math.min(1, b));
            }
        }
        return Double.NaN;
    }

    /** Distance from point p to an axis-aligned box (0 inside). */
    public static double distToBox(double px, double py, double pz, double[] box) {
        double dx = Math.max(Math.max(box[0] - px, 0), px - box[3]);
        double dy = Math.max(Math.max(box[1] - py, 0), py - box[4]);
        double dz = Math.max(Math.max(box[2] - pz, 0), pz - box[5]);
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    /** Angle in degrees between view direction v and the direction from eye e to target t (180 if degenerate). */
    public static double angleTo(double[] v, double[] e, double[] t) {
        double dx = t[0] - e[0], dy = t[1] - e[1], dz = t[2] - e[2];
        double ld = Math.sqrt(dx * dx + dy * dy + dz * dz), lv = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (ld < 1e-9 || lv < 1e-9) {
            return ld < 1e-9 ? 0 : 180;
        }
        double c = (dx * v[0] + dy * v[1] + dz * v[2]) / (ld * lv);
        return Math.toDegrees(Math.acos(Math.max(-1, Math.min(1, c))));
    }
}
