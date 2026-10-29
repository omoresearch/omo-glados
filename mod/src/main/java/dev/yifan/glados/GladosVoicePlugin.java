package dev.yifan.glados;

import java.util.UUID;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import de.maxhenkel.voicechat.api.Position;
import de.maxhenkel.voicechat.api.ServerLevel;
import de.maxhenkel.voicechat.api.ServerPlayer;
import de.maxhenkel.voicechat.api.VoicechatApi;
import de.maxhenkel.voicechat.api.VoicechatConnection;
import de.maxhenkel.voicechat.api.VoicechatPlugin;
import de.maxhenkel.voicechat.api.VoicechatServerApi;
import de.maxhenkel.voicechat.api.VolumeCategory;
import de.maxhenkel.voicechat.api.audiochannel.LocationalAudioChannel;
import de.maxhenkel.voicechat.api.events.EventRegistration;
import de.maxhenkel.voicechat.api.events.MicrophonePacketEvent;
import de.maxhenkel.voicechat.api.events.VoicechatServerStartedEvent;
import de.maxhenkel.voicechat.api.events.VoicechatServerStoppedEvent;
import de.maxhenkel.voicechat.api.opus.OpusDecoder;
import de.maxhenkel.voicechat.api.opus.OpusEncoder;
import net.minecraft.client.Minecraft;

/**
 * Simple Voice Chat plugin (entrypoint "voicechat"; only loaded when SVC is installed - the mod lists it under
 * "suggests" and nothing else references this class). Forked from microduck's DuckVoicePlugin.
 * <p>Runs on SVC's server side (the integrated server in singleplayer/LAN):
 * <ol>
 *   <li>MicrophonePacketEvent: keep only the local player's push-to-talk (or {@code player=} from the config) while
 *       GLaDOS owns the mic focus ({@link GladosFocus}: within {@code hear_range} of her head and looking at her or
 *       closer to her than to the duck), decode Opus and stream PCM to the voice backend (bring your own; default ws://127.0.0.1:8793/svc).</li>
 *   <li>The backend streams the reply PCM back; a 20 ms sender encodes it and sends it on a LocationalAudioChannel at
 *       her HEAD (it follows the head as she sways), so she talks in-game as game audio (OBS captures it).</li>
 * </ol>
 */
public final class GladosVoicePlugin implements VoicechatPlugin {
    static final String CATEGORY = "glados";
    static final UUID CHANNEL_ID = UUID.nameUUIDFromBytes("glados-voice".getBytes());

    private volatile VoicechatServerApi api;
    private volatile GladosVoiceLink link;
    private final VoicePcm.Frames out = new VoicePcm.Frames(3, 50 * 30);
    private OpusDecoder decoder;
    private OpusEncoder encoder;
    private volatile LocationalAudioChannel channel;
    private volatile ServerLevel headLevel;
    private ScheduledExecutorService sender;
    private boolean sending;
    private int quietTicks;
    private volatile boolean talking; // player's PTT currently streaming to her
    private long lastMicMs;

    private static GladosVoiceConfig config() {
        return Glados.voiceConfig;    // loaded by the client entrypoint (glados-voice.properties)
    }

    @Override
    public String getPluginId() {
        return Glados.ID;
    }

    @Override
    public void initialize(VoicechatApi voicechat) {
        GladosVoiceConfig c = config();
        Glados.LOG.info("glados voice: Simple Voice Chat plugin loaded (enabled={}, backend {}, hear_range {}, head body '{}')",
                c.enabled, c.url, c.hearRange <= 0 ? "always" : c.hearRange, c.headBody);
    }

    @Override
    public void registerEvents(EventRegistration events) {
        events.registerEvent(VoicechatServerStartedEvent.class, this::onStarted);
        events.registerEvent(VoicechatServerStoppedEvent.class, e -> stop());
        events.registerEvent(MicrophonePacketEvent.class, this::onMic);
    }

    private synchronized void onStarted(VoicechatServerStartedEvent e) {
        GladosVoiceConfig c = config();
        if (!c.enabled) {
            return;
        }
        api = e.getVoicechat();
        try {
            VolumeCategory cat = api.volumeCategoryBuilder().setId(CATEGORY).setName("GLaDOS")
                    .setDescription("GLaDOS's voice").build();
            api.registerVolumeCategory(cat);
        } catch (RuntimeException ex) {
            Glados.LOG.warn("glados voice: volume category: {}", ex.toString());
        }
        decoder = api.createDecoder();
        encoder = api.createEncoder();
        link = new GladosVoiceLink(c.url, this::onVoice, this::onControl, s -> Glados.LOG.info("glados voice: {}", s));
        link.ensureConnected();
        sender = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "glados-voice");
            t.setDaemon(true);
            return t;
        });
        sender.scheduleAtFixedRate(this::tick, 20, 20, TimeUnit.MILLISECONDS);
        Glados.voiceActive = true;
        Glados.LOG.info("glados voice: voice chat server started; listening for push-to-talk near her head");
    }

    private synchronized void stop() {
        Glados.voiceActive = false;
        GladosFocus.withdraw();
        if (sender != null) {
            sender.shutdownNow();
            sender = null;
        }
        if (link != null) {
            link.close();
            link = null;
        }
        if (decoder != null) {
            decoder.close();
            decoder = null;
        }
        if (encoder != null) {
            encoder.close();
            encoder = null;
        }
        channel = null;
        headLevel = null;
        api = null;
        out.clear();
    }

    // ---------------------------------------------------------------- mic in (SVC network thread)

    private void onMic(MicrophonePacketEvent e) {
        GladosVoiceLink l = link;
        VoicechatConnection from = e.getSenderConnection();
        if (l == null || from == null || decoder == null) {
            return;
        }
        ServerPlayer sp = from.getPlayer();
        if (sp == null || !isOurPlayer(sp)) {
            return;
        }
        GladosVoiceConfig c = config();
        double[] head = Glados.headPos();
        String dim = levelDim(sp.getServerLevel());
        double dist = Double.NaN;
        if (head != null && dim != null && Glados.config != null && dim.equals(Glados.config.dim)) {
            Position p = sp.getPosition();
            dist = Math.sqrt(sq(p.getX() - head[0]) + sq(p.getY() + 1.62 - head[1]) + sq(p.getZ() - head[2]));
        }
        // the local player: the client tick decides focus (range + look/closer-than-duck) and publishes it;
        // a configured other player: plain range check (no view direction available here)
        boolean ours = c.player.isEmpty() ? GladosFocus.ownedNow() : c.inRange(dist);
        if (!ours) {
            endUtterance(l);                              // focus moved away mid-sentence: close the turn
            return;
        }
        headLevel = sp.getServerLevel();
        byte[] opus = e.getPacket().getOpusEncodedData();
        if (opus == null || opus.length == 0) {          // SVC sends an empty packet when push-to-talk ends
            endUtterance(l);
            return;
        }
        short[] pcm;
        synchronized (this) {
            if (decoder == null) {
                return;
            }
            pcm = decoder.decode(opus);
        }
        if (!talking) {
            talking = true;
            l.sendText("{\"type\":\"ptt\",\"on\":true}");
        }
        lastMicMs = System.currentTimeMillis();
        l.sendPcm(pcm);
    }

    private void endUtterance(GladosVoiceLink l) {
        if (talking) {
            talking = false;
            l.sendText("{\"type\":\"ptt\",\"on\":false}");
            synchronized (this) {
                if (decoder != null) {
                    decoder.resetState();
                }
            }
        }
    }

    private boolean isOurPlayer(ServerPlayer sp) {
        String want = config().player;
        Object mc = sp.getPlayer();
        if (want != null && !want.isEmpty()) {
            return mc instanceof net.minecraft.world.entity.Entity ent && want.equalsIgnoreCase(ent.getScoreboardName());
        }
        Minecraft client = Minecraft.getInstance();
        var local = client == null ? null : client.player;
        return local != null && local.getUUID().equals(sp.getUuid());
    }

    // ---------------------------------------------------------------- voice out (backend -> her head)

    private void onVoice(short[] pcm) {
        out.push(pcm);
    }

    private void onControl(String json) {
        try {
            JsonObject m = JsonParser.parseString(json).getAsJsonObject();
            String type = m.has("type") ? m.get("type").getAsString() : "";
            switch (type) {
                case "clear" -> out.clear();
                case "transcript" -> {
                    if (config().captions && m.has("text")) {
                        String who = "assistant".equals(m.has("role") ? m.get("role").getAsString() : "") ? "" : "(heard) ";
                        Glados.say(who + m.get("text").getAsString().trim());
                    }
                }
                case "status" -> {
                    if (m.has("level") && "error".equals(m.get("level").getAsString())) {
                        Glados.say("voice: " + m.get("text").getAsString());
                    }
                }
                default -> {
                }
            }
        } catch (RuntimeException ignored) {
            // not JSON / unexpected shape: ignore
        }
    }

    /** 20 ms sender: one Opus frame per tick from the reply queue onto her head's locational channel. */
    private void tick() {
        try {
            GladosVoiceLink l = link;
            if (l != null && !l.isOpen()) {
                l.ensureConnected();
            }
            if (talking && System.currentTimeMillis() - lastMicMs > 400 && l != null) {
                endUtterance(l);                          // no end packet arrived (e.g. walked out of range)
            }
            short[] frame = out.poll();
            if (frame == null) {
                if (sending && ++quietTicks > 5) {
                    LocationalAudioChannel ch = channel;
                    if (ch != null) {
                        ch.flush();
                    }
                    synchronized (this) {
                        if (encoder != null) {
                            encoder.resetState();
                        }
                    }
                    sending = false;
                }
                return;
            }
            quietTicks = 0;
            LocationalAudioChannel ch = ensureChannel();
            if (ch == null) {
                return;
            }
            byte[] opus;
            synchronized (this) {
                if (encoder == null) {
                    return;
                }
                opus = encoder.encode(frame);
            }
            ch.send(opus);
            sending = true;
        } catch (RuntimeException e) {
            Glados.LOG.warn("glados voice: sender: {}", e.toString());
        }
    }

    private LocationalAudioChannel ensureChannel() {
        VoicechatServerApi a = api;
        double[] head = Glados.headPos();
        if (a == null || head == null) {
            return null;
        }
        ServerLevel level = headLevel;
        if (level == null) {
            level = findLevel(a);                         // she may answer before the player ever spoke (e.g. Twitch)
            headLevel = level;
            if (level == null) {
                return null;
            }
        }
        Position pos = a.createPosition(head[0], head[1], head[2]);
        LocationalAudioChannel ch = channel;
        if (ch == null || ch.isClosed()) {
            ch = a.createLocationalAudioChannel(CHANNEL_ID, level, pos);
            if (ch == null) {
                return null;
            }
            ch.setCategory(CATEGORY);
            ch.setDistance(config().voiceDistance);
            channel = ch;
        } else {
            ch.updateLocation(pos);                       // she sways: the voice follows her head
        }
        return ch;
    }

    /** The integrated server's level for her anchor dimension, or null (dedicated server / not loaded). */
    private static ServerLevel findLevel(VoicechatServerApi a) {
        try {
            Minecraft mc = Minecraft.getInstance();
            GladosConfig c = Glados.config;
            if (mc == null || c == null || !mc.hasSingleplayerServer()) {
                return null;
            }
            for (net.minecraft.server.level.ServerLevel l : mc.getSingleplayerServer().getAllLevels()) {
                if (l.dimension().identifier().toString().equals(c.dim)) {
                    return a.fromServerLevel(l);
                }
            }
        } catch (RuntimeException e) {
            return null;
        }
        return null;
    }

    static String levelDim(ServerLevel level) {
        Object o = level == null ? null : level.getServerLevel();
        if (o instanceof net.minecraft.world.level.Level l) {
            return l.dimension().identifier().toString();
        }
        return null;
    }

    private static double sq(double v) {
        return v * v;
    }
}
