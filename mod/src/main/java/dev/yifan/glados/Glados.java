package dev.yifan.glados;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.joml.Matrix3f;
import org.joml.Matrix4f;
import org.joml.Quaternionf;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.mojang.blaze3d.platform.NativeImage;
import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.VertexConsumer;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.message.v1.ClientSendMessageEvents;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderContext;
import net.fabricmc.fabric.api.client.rendering.v1.level.LevelRenderEvents;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.client.renderer.culling.Frustum;
import net.minecraft.client.renderer.rendertype.RenderType;
import net.minecraft.client.renderer.rendertype.RenderTypes;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.texture.DynamicTexture;
import net.minecraft.client.renderer.texture.OverlayTexture;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.network.chat.Component;
import net.minecraft.network.chat.MutableComponent;
import net.minecraft.resources.Identifier;
import net.minecraft.util.LightCoordsUtil;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.phys.AABB;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.shapes.VoxelShape;

/**
 * Renders the GLaDOS-style AI from the local glados-sim sidecar (127.0.0.1:8794), articulated, hanging from a ceiling
 * anchor, posts the player's eye position, and turns `!glados ...` chat lines into sidecar commands (client-side only;
 * never sent to the server). Only the `!glados` prefix is intercepted (`!duck`, `!rp1` pass through untouched).
 */
public final class Glados implements ClientModInitializer {
    public static final String ID = "glados";
    static final Logger LOG = LoggerFactory.getLogger(ID);

    static final Identifier WHITE = Identifier.fromNamespaceAndPath(ID, "textures/misc/white.png");
    /** Cull when the nearest point of her bounding box is farther than this from the camera. */
    static final double MAX_DIST = 96.0;
    /** Nominal ceiling-to-lowest-point height in sim metres at scale 1 (contract). */
    static final double HEIGHT_M = 7.5;
    static final int FULLBRIGHT = 0xF000F0;

    static GladosConfig config;
    static Path configFile;
    static GladosVoiceConfig voiceConfig = new GladosVoiceConfig();
    static SidecarClient client;
    /** Set by the SVC plugin while its voice chat server runs (no SVC = never claim mic focus). */
    static volatile boolean voiceActive;
    static volatile boolean focusOwned;
    static final ConcurrentLinkedQueue<Component> REPLIES = new ConcurrentLinkedQueue<>();
    /** Scratch matrices for the geometry callbacks (they run one after another on the render thread). */
    static final float[] MAT = new float[16];
    static final float[] NRM = new float[9];
    static final float[] EYE_TINT_WHITE = new float[3], EYE_TINT_TEX = new float[3];

    @Override
    public void onInitializeClient() {
        Path dir = FabricLoader.getInstance().getConfigDir();
        configFile = dir.resolve("glados.properties");
        config = GladosConfig.load(configFile);
        saveConfig();                                  // write defaults on first run
        Path vf = dir.resolve("glados-voice.properties");
        voiceConfig = GladosVoiceConfig.load(vf);
        try {
            voiceConfig.save(vf);
        } catch (IOException e) {
            LOG.warn("glados voice: cannot write {}: {}", vf, e.toString());
        }
        client = new SidecarClient(() -> config.port, () -> config.enabled, s -> LOG.info("glados: {}", s));
        client.start();
        ClientSendMessageEvents.ALLOW_CHAT.register(Glados::allowChat);
        ClientTickEvents.END_CLIENT_TICK.register(Glados::tick);
        LevelRenderEvents.COLLECT_SUBMITS.register(Glados::render);
        LOG.info("glados: sidecar {}, scale {}, yaw {}, anchor {},{},{} in {}", client.base(), config.scale, config.yaw,
                config.ax, config.ay, config.az, config.dim);
    }

    // ---------------------------------------------------------------- chat

    private static boolean allowChat(String message) {
        if (!GladosCommand.isGlados(message)) {
            return true;                               // not ours: leave it to other mods / the server
        }
        try {
            handle(GladosCommand.parse(message));
        } catch (Exception e) {
            LOG.warn("glados: command failed", e);
        }
        return false;                                  // swallow: never reaches the server
    }

    private static void handle(GladosCommand c) {
        Minecraft mc = Minecraft.getInstance();
        switch (c.kind) {
            case HELP -> {
                say(c.message);
                say(creditLine());
            }
            case ERROR -> say(c.message);
            case MODELS -> client.fetchModels(lines -> lines.forEach(Glados::say));
            case SIDECAR -> {
                String what = c.cmd;
                client.sendCommand(c.json, r -> REPLIES.add(line(what + ": " + r)));
            }
            case HERE -> here(mc);
            case SCALE -> {
                config.scale = c.value;
                saveConfig();
                client.postAsync("/config", "{\"scale_hint\":" + c.value + "}");
                say("scale " + c.value + " (~" + Math.round(c.value * HEIGHT_M * 10) / 10.0 + " blocks from the ceiling down)");
            }
            case YAW -> {
                config.yaw = c.value;
                saveConfig();
                say("yaw " + GladosConfig.fmt(c.value) + " degrees");
            }
            default -> {
            }
        }
    }

    /** Anchor = underside of the first solid block straight above the player's head; x/z = that column's centre. */
    private static void here(Minecraft mc) {
        if (mc.player == null || mc.level == null) {
            return;
        }
        ClientLevel level = mc.level;
        Vec3 eye = mc.player.getEyePosition();
        BlockPos head = BlockPos.containing(eye.x, eye.y, eye.z);
        int x = head.getX(), z = head.getZ(), hy = head.getY();
        int maxUp = Math.max(0, Math.min(64, level.getMaxY() - hy));
        double y = GladosMath.ceilingAbove(hy, maxUp, dy -> {
            BlockPos p = new BlockPos(x, hy + dy, z);
            BlockState st = level.getBlockState(p);
            VoxelShape sh = st.getCollisionShape(level, p);
            return sh.isEmpty() ? Double.NaN : sh.min(Direction.Axis.Y);
        });
        if (Double.isNaN(y)) {
            say("no solid block above you within " + maxUp + " blocks; stand under a ceiling");
            return;
        }
        config.setAnchor(x, y, z, dimId(level));
        saveConfig();
        double room = y - mc.player.getY();
        String fit = room < HEIGHT_M * config.scale
                ? String.format(java.util.Locale.ROOT, "; only %.1f blocks to your feet, she is ~%.1f tall (try !glados scale %.2f)",
                        room, HEIGHT_M * config.scale, Math.max(GladosCommand.MIN_SCALE, Math.floor(room / HEIGHT_M * 100) / 100))
                : "";
        say("anchor set to " + GladosConfig.fmt(config.ax) + " " + GladosConfig.fmt(y) + " " + GladosConfig.fmt(config.az)
                + " (" + config.dim + ")" + fit);
    }

    /** "model: <title> - <credit>" of the model on screen (the procedural fallback is our own). */
    static String creditLine() {
        GladosModel m = client == null ? null : client.model();
        if (m == null) {
            return "model: (sidecar not connected)";
        }
        String t = m.title != null ? m.title : m.name != null ? m.name : "procedural GLaDOS-style model (original)";
        return "model: " + t + (m.credit != null && !m.credit.isBlank() ? " - credit: " + m.credit : "");
    }

    static void say(String s) {
        REPLIES.add(line(s));
    }

    static MutableComponent line(String s) {
        return Component.literal("[GLaDOS] ").withColor(0xFFD24A).append(Component.literal(s).withColor(0xE0E0E0));
    }

    static void saveConfig() {
        try {
            config.save(configFile);
        } catch (IOException e) {
            LOG.warn("glados: cannot save {}: {}", configFile, e.toString());
        }
    }

    static String dimId(ClientLevel level) {
        return level.dimension().identifier().toString();
    }

    // ---------------------------------------------------------------- tick: chat replies, player eye, mic focus

    private static void tick(Minecraft mc) {
        try {
            syncAtlas(mc);
        } catch (RuntimeException e) {
            LOG.warn("glados: atlas upload failed", e);
            atlasFailed = client.model();
        }
        Component r;
        int n = 0;
        while (n++ < 10 && (r = REPLIES.poll()) != null) {
            if (mc.level != null) {
                mc.gui.hud.getChat().addClientSystemMessage(r);
            }
        }
        boolean here = mc.level != null && mc.player != null && config.enabled && config.hasAnchor()
                && dimId(mc.level).equals(config.dim);
        if (!here) {
            client.setPlayer(null);
        } else if (mc.level.getGameTime() % 4 == 0) {
            Vec3 e = mc.player.getEyePosition();
            client.setPlayer(GladosMath.mcToSim(config.ax, config.ay, config.az, config.scale, config.yaw, e.x, e.y, e.z));
        }
        double want = 0;                               // claim score (GladosFocus.score), 0 = no claim
        if (here && voiceActive && voiceConfig.enabled) {
            try {
                double[] h = headPos();
                Vec3 e = mc.player.getEyePosition();
                Vec3 v = mc.player.getViewVector(1f);
                double[] eye = {e.x, e.y, e.z};
                double dist = Math.sqrt(sq(h[0] - e.x) + sq(h[1] - e.y) + sq(h[2] - e.z));
                double ang = GladosMath.angleTo(new double[] {v.x, v.y, v.z}, eye, h);
                double[] duck = GladosFocus.duckPos(config.dim);
                double dd = duck == null ? Double.NaN : Math.sqrt(sq(duck[0] - e.x) + sq(duck[1] - e.y) + sq(duck[2] - e.z));
                want = GladosFocus.score(dist, voiceConfig.hearRange, ang, voiceConfig.lookAngle, dd);
            } catch (RuntimeException ex) {
                want = 0;
            }
        }
        // the best claim among the companions (GLaDOS, the RP1 mod) gets the mic; the duck yields to any focus owner
        if (want > 0 || focusOwned) {
            focusOwned = GladosFocus.arbitrate(want, System.currentTimeMillis());
        } else {
            System.clearProperty(GladosFocus.CLAIM + GladosFocus.OWNER);
        }
    }

    private static double sq(double v) {
        return v * v;
    }

    /**
     * Her head (voice + focus target) in MC coords: the mesh centre of the {@code head_body} body (glados-voice.properties,
     * default "head"; if absent, the body carrying the eye) at the latest pose. Without a pose: ~5.5 blocks below the anchor.
     */
    static double[] headPos() {
        GladosConfig c = config;
        if (c == null) {
            return null;
        }
        SidecarClient sc = client;
        SidecarClient.Frames f = sc == null ? null : sc.frames();
        if (f != null) {
            int i = f.model.find(voiceConfig.headBody);
            float[] p = i >= 0 && i < f.cur.bodies.length ? f.cur.bodies[i] : null;
            if (p != null) {
                float[] cen = f.model.bodies[i].centroid;
                return GladosMath.bodyVertexToMc(c.ax, c.ay, c.az, c.scale, c.yaw, p, cen[0], cen[1], cen[2]);
            }
        }
        return new double[] {c.ax, c.ay - 5.5 * c.scale, c.az};
    }

    // ---------------------------------------------------------------- rendering

    // ---------------------------------------------------------------- atlas texture (v3)

    /** The model whose atlas is uploaded as {@link #atlasId} (null = none). Client thread only. */
    static GladosModel atlasModel;
    static Identifier atlasId;
    /** A model whose atlas failed to decode/upload: its textured parts are drawn with the white texture. */
    static GladosModel atlasFailed;
    /** The model whose atlas PNG is being decoded off-thread (null = idle). */
    static GladosModel decoding;
    static RenderType rtCutoutWhite, rtTranslucentWhite, rtCutoutAtlas, rtTranslucentAtlas;

    /**
     * Client thread, every tick: when the sidecar client holds a model whose atlas is not on the GPU yet, decode the
     * PNG on a background thread (NativeImage.read, RGBA), then on the client thread wrap it in a DynamicTexture,
     * register it as glados:dynamic/atlas_&lt;rev&gt; and release the previous atlas.
     */
    static void syncAtlas(Minecraft mc) {
        GladosModel m = client.model();
        if (m == null || m == atlasModel || m == atlasFailed || m == decoding) {
            return;
        }
        byte[] png = m.atlasPng();
        if (png == null) {
            if (m.needsAtlas() && m.atlasPath != null) {
                LOG.warn("glados: no atlas for '{}' (rev {}): textured parts drawn white", m.name, m.rev);
            }
            atlasFailed = m;                           // nothing to upload: textured parts (if any) use white
            if (atlasId != null) {                     // switched to an untextured model: free the old atlas
                mc.getTextureManager().release(atlasId);
                atlasId = null;
                atlasModel = null;
                rtCutoutAtlas = rtTranslucentAtlas = null;
            }
            return;
        }
        decoding = m;
        CompletableFuture.supplyAsync(() -> {
            try {
                return NativeImage.read(NativeImage.Format.RGBA, new ByteArrayInputStream(png));
            } catch (IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }).whenComplete((img, err) -> mc.execute(() -> {
            decoding = null;
            if (err != null || img == null) {
                LOG.warn("glados: atlas for '{}' did not decode: {}", m.name, String.valueOf(err));
                atlasFailed = m;
                return;
            }
            if (client.model() != m) {                // switched again meanwhile: drop it
                img.close();
                return;
            }
            Identifier id = Identifier.fromNamespaceAndPath(ID, "dynamic/atlas_" + Math.max(0, m.rev));
            DynamicTexture tex = new DynamicTexture(() -> "glados atlas rev " + m.rev, img);
            mc.getTextureManager().register(id, tex); // same id: replaces + closes the old texture
            if (atlasId != null && !atlasId.equals(id)) {
                mc.getTextureManager().release(atlasId);
            }
            atlasId = id;
            atlasModel = m;
            rtCutoutAtlas = RenderTypes.entityCutout(id);
            rtTranslucentAtlas = RenderTypes.entityTranslucent(id);
            LOG.info("glados: atlas {}x{} registered as {} for model '{}'", img.getWidth(), img.getHeight(), id, m.name);
        }));
    }

    // ---------------------------------------------------------------- rendering

    static boolean renderBroken;

    private static void render(LevelRenderContext ctx) {
        if (renderBroken) {
            return;
        }
        try {
            renderGlados(ctx);
        } catch (RuntimeException e) {
            renderBroken = true;                       // one log line, never spam or crash the frame loop
            LOG.warn("glados: render failed, GLaDOS rendering disabled until restart", e);
        }
    }

    private static void renderGlados(LevelRenderContext ctx) {
        Minecraft mc = Minecraft.getInstance();
        if (!config.enabled || !config.hasAnchor() || mc.level == null || !dimId(mc.level).equals(config.dim)) {
            return;
        }
        SidecarClient.Frames f = client.frames();
        if (f == null) {
            return;
        }
        GladosModel model = f.model;
        int nb = model.bodies.length;
        if (nb == 0) {
            return;
        }
        boolean atlasReady = model == atlasModel;
        if (model.needsAtlas() && !atlasReady && model != atlasFailed) {
            return;                                    // a just-switched model shows up once its atlas is on the GPU
        }
        CameraRenderState cam = ctx.levelState().cameraRenderState;
        Vec3 cp = cam.pos;
        double s = config.scale, ax = config.ax, ay = config.ay, az = config.az, yaw = config.yaw;
        float alpha = SidecarClient.alpha(f, System.nanoTime());

        // bounding box of every body origin (MC), padded by the largest body-local extent: she is ~7.5 blocks tall,
        // so culling on the mount point alone would pop her in/out at the edge of the view
        float[][] poses = new float[nb][];
        double[][] mcPos = new double[nb][];
        double[] box = {Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE, -Double.MAX_VALUE};
        for (int i = 0; i < nb; i++) {
            float[] p = SidecarClient.interp(f, i, alpha);
            if (p == null) {
                continue;
            }
            poses[i] = p;
            double[] m = GladosMath.simToMc(ax, ay, az, s, yaw, p[0], p[1], p[2]);
            mcPos[i] = m;
            for (int k = 0; k < 3; k++) {
                box[k] = Math.min(box[k], m[k]);
                box[k + 3] = Math.max(box[k + 3], m[k]);
            }
        }
        if (box[0] == Double.MAX_VALUE) {
            return;
        }
        double pad = model.radius * s + 0.25;
        for (int k = 0; k < 3; k++) {
            box[k] -= pad;
            box[k + 3] += pad;
        }
        if (GladosMath.distToBox(cp.x, cp.y, cp.z, box) > MAX_DIST) {
            return;
        }
        Frustum fr = cam.cullFrustum;
        if (fr != null && !fr.isVisible(new AABB(box[0], box[1], box[2], box[3], box[4], box[5]))) {
            return;
        }
        if (rtCutoutWhite == null) {
            rtCutoutWhite = RenderTypes.entityCutout(WHITE);
            rtTranslucentWhite = RenderTypes.entityTranslucent(WHITE);
        }
        float talk = f.prev.talk + (f.cur.talk - f.prev.talk) * alpha;
        GladosMath.eyeTint(f.cur.eye, talk, 1f, EYE_TINT_WHITE);
        GladosMath.eyeTint(f.cur.eye, talk, GladosMath.EYE_TEXTURED_BASE, EYE_TINT_TEX);
        final float[] tintWhite = EYE_TINT_WHITE.clone(), tintTex = EYE_TINT_TEX.clone();
        double[] yq = GladosMath.yawQuat(yaw);
        Quaternionf yawQ = new Quaternionf((float) yq[1], (float) yq[2], (float) yq[3], (float) yq[0]);
        PoseStack ps = ctx.poseStack();
        float fs = (float) s;
        int[] lights = new int[nb];
        float[][] local = new float[nb][];             // per body: translate(s C p) rot(q_mc) scale(s), yaw space
        float[][] localN = new float[nb][];            // per body: rotation only (normals)
        java.util.List<Runnable> glass = new java.util.ArrayList<>();
        ps.pushPose();
        ps.translate(ax - cp.x, ay - cp.y, az - cp.z);
        ps.mulPose(yawQ);
        for (int i = 0; i < nb; i++) {
            float[] p = poses[i];
            if (p == null) {
                continue;
            }
            // per-body light: the body origin's block and the one below it (the mount sits at the ceiling underside,
            // i.e. exactly on the solid ceiling block, which has no light of its own)
            double[] m = mcPos[i];
            lights[i] = LightCoordsUtil.max(
                    LightCoordsUtil.getLightCoords(mc.level, BlockPos.containing(m[0], m[1], m[2])),
                    LightCoordsUtil.getLightCoords(mc.level, BlockPos.containing(m[0], m[1] - 0.5, m[2])));
            double[] q = GladosMath.quatSimToMc(p[3], p[4], p[5], p[6]);
            local[i] = GladosMath.bodyMatrix(s * p[0], s * p[2], -s * p[1], q, s);
            localN[i] = GladosMath.normalMatrix(q);
            GladosModel.Body body = model.bodies[i];
            if (body.parts.length == 0) {
                continue;
            }
            ps.pushPose();
            ps.translate(s * p[0], s * p[2], -s * p[1]);
            ps.mulPose(new Quaternionf((float) q[1], (float) q[2], (float) q[3], (float) q[0]));
            ps.scale(fs, fs, fs);
            for (GladosModel.Part part : body.parts) {
                boolean tex = part.textured && atlasReady;
                RenderType rt = part.tag == GladosModel.Tag.GLASS ? (tex ? rtTranslucentAtlas : rtTranslucentWhite)
                        : (tex ? rtCutoutAtlas : rtCutoutWhite);
                int light = part.tag.fullbright() ? FULLBRIGHT : lights[i];
                float[] tint = part.tag == GladosModel.Tag.EYE ? (tex ? tintTex : tintWhite) : null;
                if (part.tag == GladosModel.Tag.GLASS) {
                    PoseStack.Pose snap = ps.last().copy();
                    glass.add(() -> ctx.submitNodeCollector().submitCustomGeometry(snapStack(snap), rt,
                            (pose, vc) -> emit(part, pose, vc, light, tint, tex)));
                } else {
                    ctx.submitNodeCollector().submitCustomGeometry(ps, rt, (pose, vc) -> emit(part, pose, vc, light, tint, tex));
                }
            }
            ps.popPose();
        }
        // skinned parts (cables/wires spanning two bodies): blended in the yaw space of the pose stack
        for (GladosModel.Part part : model.skinned) {
            int light = 0;
            for (int bi : part.skinBodies) {
                light = LightCoordsUtil.max(light, lights[bi]);
            }
            if (part.tag.fullbright()) {
                light = FULLBRIGHT;
            }
            boolean tex = part.textured && atlasReady;
            RenderType rt = part.tag == GladosModel.Tag.GLASS ? (tex ? rtTranslucentAtlas : rtTranslucentWhite)
                    : (tex ? rtCutoutAtlas : rtCutoutWhite);
            float[] tint = part.tag == GladosModel.Tag.EYE ? (tex ? tintTex : tintWhite) : null;
            final int l = light;
            if (part.tag == GladosModel.Tag.GLASS) {
                PoseStack.Pose snap = ps.last().copy();
                glass.add(() -> ctx.submitNodeCollector().submitCustomGeometry(snapStack(snap), rt,
                        (pose, vc) -> emitSkinned(part, local, localN, pose, vc, l, tint, tex)));
            } else {
                ctx.submitNodeCollector().submitCustomGeometry(ps, rt, (pose, vc) -> emitSkinned(part, local, localN, pose, vc, l, tint, tex));
            }
        }
        ps.popPose();
        glass.forEach(Runnable::run);                  // translucent parts after every cutout part
    }

    /** A one-entry PoseStack holding a snapshot pose (glass is submitted after the loop). */
    static PoseStack snapStack(PoseStack.Pose snap) {
        PoseStack p = new PoseStack();
        p.last().set(snap);
        return p;
    }

    /** Feeds the pre-baked flat arrays straight into the vertex buffer (one bulk addVertex per vertex). */
    static void emit(GladosModel.Part part, PoseStack.Pose pose, VertexConsumer vc, int light, float[] tint, boolean tex) {
        pose.pose().get(MAT);
        pose.normal().get(NRM);
        final int overlay = OverlayTexture.NO_OVERLAY;
        if (tex) {
            GladosModel.emit(part, MAT, NRM, tint,
                    (x, y, z, argb, u, v, nx, ny, nz) -> vc.addVertex(x, y, z, argb, u, v, overlay, light, nx, ny, nz));
        } else {
            GladosModel.emit(part, MAT, NRM, tint,
                    (x, y, z, argb, u, v, nx, ny, nz) -> vc.addVertex(x, y, z, argb, 0.5f, 0.5f, overlay, light, nx, ny, nz));
        }
    }

    /** Skinned part: full matrix of body i = pose (anchor, yaw) * local[i]; normal = pose normal * R_i. */
    static void emitSkinned(GladosModel.Part part, float[][] local, float[][] localN, PoseStack.Pose pose, VertexConsumer vc, int light,
                            float[] tint, boolean tex) {
        int nb = local.length;
        float[][] mats = new float[nb][], nmats = new float[nb][];
        Matrix4f m = new Matrix4f();
        Matrix3f n = new Matrix3f();
        for (int i = 0; i < nb; i++) {
            if (local[i] == null) {
                continue;
            }
            m.set(local[i]);
            mats[i] = new Matrix4f(pose.pose()).mul(m).get(new float[16]);
            n.set(localN[i]);
            nmats[i] = new Matrix3f(pose.normal()).mul(n).get(new float[9]);
        }
        final int overlay = OverlayTexture.NO_OVERLAY;
        GladosModel.emitSkinned(part, mats, nmats, tint, (x, y, z, argb, u, v, nx, ny, nz) ->
                vc.addVertex(x, y, z, argb, tex ? u : 0.5f, tex ? v : 0.5f, overlay, light, nx, ny, nz));
    }
}
