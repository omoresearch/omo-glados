package dev.yifan.glados;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import de.maxhenkel.voicechat.api.opus.OpusDecoder;
import de.maxhenkel.voicechat.api.opus.OpusEncoder;
import de.maxhenkel.voicechat.concentus.OpusApplication;

/**
 * Standalone tests for GLaDOS's Simple Voice Chat audio path (no game needed):
 * PCM framing, LE conversion, resampling, a real Opus round trip through the SVC jar's own codecs
 * (pure-Java Concentus impl of the API interfaces, and the native opus4j SVC uses on Linux), and optionally a live
 * run against a GLaDOS voice backend (yours; not included) with the exact plugin pipeline: mic frames Opus-encoded (as the SVC client
 * does) -> API decoder (as MicrophonePacketEvent) -> GladosVoiceLink -> backend -> reply -> Frames -> API
 * encoder (as the LocationalAudioChannel sender).
 *
 *   ./build.sh voicetest                                   unit tests only
 *   ./build.sh voicetest ws://127.0.0.1:8793/svc a.pcm b.pcm   + live (PCM16 LE mono 48 kHz files)
 */
public final class VoiceTests {
    static int pass, fail;

    static void check(String name, boolean ok, String detail) {
        if (ok) {
            pass++;
        } else {
            fail++;
        }
        System.out.println((ok ? "PASS  " : "FAIL  ") + name + ": " + detail);
    }

    /** Byte-for-byte the condition the microduck plugin uses in DuckVoicePlugin.onMic (companionHasFocus). */
    static boolean duckYields() {
        String f = System.getProperty("omo.voice.focus", "");
        int c = f.lastIndexOf(':');
        if (c <= 0 || f.startsWith("duck")) {
            return false;
        }
        try {
            return System.currentTimeMillis() - Long.parseLong(f.substring(c + 1)) < 1000;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    static short[] tone(int n, double hz, double amp, int offset) {
        short[] s = new short[n];
        for (int i = 0; i < n; i++) {
            double t = (i + offset) / 48000.0;
            // a vowel-ish signal: fundamental + 2 harmonics, slow amplitude wobble
            double v = Math.sin(2 * Math.PI * hz * t) + 0.5 * Math.sin(2 * Math.PI * 2 * hz * t) + 0.25 * Math.sin(2 * Math.PI * 3 * hz * t);
            s[i] = (short) Math.round(amp * v / 1.75 * (0.75 + 0.25 * Math.sin(2 * Math.PI * 3 * t)));
        }
        return s;
    }

    /** Best normalised correlation of b against a over lags 0..maxLag (codec delay). */
    static double[] bestCorr(short[] a, short[] b, int maxLag) {
        double best = -2;
        int bestLag = 0;
        for (int lag = 0; lag <= maxLag; lag++) {
            double sab = 0, saa = 0, sbb = 0;
            for (int i = 2000; i + lag < b.length && i < a.length - 2000; i++) {
                double x = a[i], y = b[i + lag];
                sab += x * y;
                saa += x * x;
                sbb += y * y;
            }
            double c = sab / Math.sqrt(saa * sbb + 1e-9);
            if (c > best) {
                best = c;
                bestLag = lag;
            }
        }
        return new double[]{best, bestLag};
    }

    /** The API OpusEncoder over SVC's bundled Concentus, configured like SVC's JavaOpusEncoderImpl (VOIP, FEC). */
    static OpusEncoder javaEnc() {
        try {
            de.maxhenkel.voicechat.concentus.OpusEncoder e = new de.maxhenkel.voicechat.concentus.OpusEncoder(48000, 1, OpusApplication.OPUS_APPLICATION_VOIP);
            e.setUseInbandFEC(true);
            e.setPacketLossPercent(5);
            byte[] buf = new byte[1024];
            boolean[] closed = {false};
            return new OpusEncoder() {
                public byte[] encode(short[] p) {
                    try {
                        int n = e.encode(p, 0, p.length, buf, 0, buf.length);
                        return java.util.Arrays.copyOf(buf, n);
                    } catch (Exception x) {
                        throw new RuntimeException(x);
                    }
                }
                public void resetState() { e.resetState(); }
                public boolean isClosed() { return closed[0]; }
                public void close() { closed[0] = true; }
            };
        } catch (Exception x) {
            throw new RuntimeException(x);
        }
    }

    static OpusDecoder javaDec() {
        try {
            de.maxhenkel.voicechat.concentus.OpusDecoder d = new de.maxhenkel.voicechat.concentus.OpusDecoder(48000, 1);
            short[] buf = new short[960 * 6];
            boolean[] closed = {false};
            return new OpusDecoder() {
                public short[] decode(byte[] p) {
                    try {
                        int n = p == null || p.length == 0 ? d.decode(null, 0, 0, buf, 0, 960, false) : d.decode(p, 0, p.length, buf, 0, buf.length, false);
                        return java.util.Arrays.copyOf(buf, n);
                    } catch (Exception x) {
                        throw new RuntimeException(x);
                    }
                }
                public short[][] decode(byte[] p, int frames) { return new short[][]{decode(p)}; }
                public void resetState() { d.resetState(); }
                public boolean isClosed() { return closed[0]; }
                public void close() { closed[0] = true; }
            };
        } catch (Exception x) {
            throw new RuntimeException(x);
        }
    }

    static void roundTrip(String label, OpusEncoder enc, OpusDecoder dec) {
        int frames = 100; // 2 s
        short[] src = tone(frames * VoicePcm.FRAME, 220, 9000, 0);
        List<short[]> outFrames = new ArrayList<>();
        int bytes = 0;
        boolean sizesOk = true;
        for (int f = 0; f < frames; f++) {
            short[] frame = new short[VoicePcm.FRAME];
            System.arraycopy(src, f * VoicePcm.FRAME, frame, 0, VoicePcm.FRAME);
            byte[] pkt = enc.encode(frame);
            bytes += pkt.length;
            short[] d = dec.decode(pkt);
            sizesOk &= d.length == VoicePcm.FRAME;
            outFrames.add(d);
        }
        short[] out = new short[frames * VoicePcm.FRAME];
        for (int f = 0; f < frames; f++) {
            System.arraycopy(outFrames.get(f), 0, out, f * VoicePcm.FRAME, VoicePcm.FRAME);
        }
        double[] c = bestCorr(src, out, 1200);
        double kbps = bytes * 8.0 / (frames * 0.02) / 1000;
        check(label + " Opus 48k mono 20 ms round trip", sizesOk && c[0] > 0.9,
                String.format("corr=%.3f at codec delay %d samples (%.1f ms), %.1f kbit/s, frames decode to %d samples",
                        c[0], (int) c[1], c[1] / 48.0, kbps, outFrames.get(0).length));
        enc.close();
        dec.close();
    }

    public static void main(String[] args) throws Exception {
        // -------------------------------------------------------------- PCM helpers
        short[] s = {0, 1, -1, 32767, -32768, 1234};
        byte[] le = VoicePcm.toBytes(s);
        short[] back = VoicePcm.toShorts(le, 0, le.length);
        check("PCM16 LE round trip", java.util.Arrays.equals(s, back) && le[2] == 1 && le[3] == 0 && le[6] == (byte) 0xff && le[7] == 0x7f,
                le.length + " bytes");

        VoicePcm.Frames q = new VoicePcm.Frames(3, 10);
        q.push(new short[700]);
        q.push(new short[1500]);                                   // 2200 = 2 frames + 280
        boolean prebuf = q.poll() == null && q.size() == 2;
        q.push(new short[700]);                                    // 2900 = 3 frames + 20
        short[] f1 = q.poll();
        check("Frames: re-framing to 960 with a 3-frame prebuffer", prebuf && f1 != null && f1.length == 960 && q.size() == 2,
                "after 2200 samples poll=null, after 2900 got a 960 frame, " + q.size() + " left");
        q.poll();
        q.poll();
        boolean under = q.poll() == null && !q.isPlaying();
        q.flushTail();
        short[] tail = q.poll();
        check("Frames: underrun re-arms prebuffer; tail padded and played", under && tail != null && tail.length == 960,
                "tail frame " + (tail == null ? "missing" : tail.length));
        for (int i = 0; i < 15; i++) {
            q.push(new short[960]);
        }
        check("Frames: capped (oldest dropped, no runaway lag)", q.size() == 10 && q.dropped() == 5, "size " + q.size() + " dropped " + q.dropped());
        q.clear();
        check("Frames: clear (barge-in)", q.size() == 0 && q.poll() == null, "size 0");

        short[] r = VoicePcm.resample(tone(2400, 300, 8000, 0), 24000, 48000);
        check("resample 24k -> 48k doubles length", r.length == 4800, r.length + " samples");
        check("rms of a full-scale-ish tone", VoicePcm.rms(tone(960, 440, 10000, 0)) > 3000, String.format("%.0f", VoicePcm.rms(tone(960, 440, 10000, 0))));

        GladosVoiceConfig dc = new GladosVoiceConfig();
        check("config: 16-block hear range, NaN (other dim) rejected", dc.inRange(10) && !dc.inRange(20) && !dc.inRange(Double.NaN), "range " + dc.hearRange);
        dc.hearRange = 0;
        check("config: hear_range=always accepts anything", dc.inRange(Double.NaN) && dc.inRange(1e6), "ok");
        Path tmp = Files.createTempFile("gladosvoice", ".properties");
        Files.writeString(tmp, "url=ws://evil.example:80/svc\nhear_range=always\nvoice_distance=999\n");
        GladosVoiceConfig lc = GladosVoiceConfig.load(tmp);
        check("config: non-loopback url refused, clamps", lc.url.startsWith("ws://127.0.0.1:") && lc.hearRange == 0 && lc.voiceDistance == 128f,
                lc.url + " range=" + lc.hearRange + " dist=" + lc.voiceDistance);
        Files.deleteIfExists(tmp);
        check("config: GLaDOS defaults (8793 backend, voice from 'head', look 30 deg, 32-block voice)",
                dc.url.equals("ws://127.0.0.1:8793/svc") && dc.headBody.equals("head") && dc.lookAngle == 30 && dc.voiceDistance == 32f,
                dc.url + " head_body=" + dc.headBody);

        // mic focus: what GladosVoicePlugin.onMic (ownedNow) and the patched microduck DuckVoicePlugin.onMic see
        System.clearProperty(GladosFocus.PROP);
        check("focus: nobody owns it -> glados plugin ignores mic, duck hears", !GladosFocus.ownedNow() && !duckYields(), "unset");
        GladosFocus.publish(System.currentTimeMillis());
        check("focus: glados published this tick -> glados hears, duck yields", GladosFocus.ownedNow() && duckYields(),
                System.getProperty(GladosFocus.PROP));
        System.setProperty(GladosFocus.PROP, "glados:" + (System.currentTimeMillis() - 1500));
        check("focus: stale (>1 s, e.g. game paused / walked off) -> back to the duck", !GladosFocus.ownedNow() && !duckYields(),
                System.getProperty(GladosFocus.PROP));
        // several companions want the mic: scored claims, the best one owns it (the RP1 mod runs the same protocol)
        long now = System.currentTimeMillis();
        double near = GladosFocus.score(4, 16, 120, 30, Double.NaN);
        double look = GladosFocus.score(10, 16, 10, 30, Double.NaN);
        check("score: wants() as a claim score (looking 2.94, nearest 1/(1+d), out of range / nearer the duck 0)",
                Math.abs(look - (3 - 10 / 180.0)) < 1e-9 && Math.abs(near - 0.2) < 1e-9 && GladosFocus.score(20, 16, 0, 30, Double.NaN) == 0
                        && GladosFocus.score(6, 16, 90, 30, 3) == 0, String.format("look %.3f near %.3f", look, near));
        System.setProperty(GladosFocus.CLAIM + "rp1", now + ":" + (3 - 5 / 180.0));          // player looks at the RP1
        boolean won = GladosFocus.arbitrate(near, now);
        check("focus: RP1 looked at beats GLaDOS being nearer -> GLaDOS yields (no focus published)",
                !won && !GladosFocus.ownedNow() && "rp1".equals(GladosFocus.winner(now)), String.valueOf(System.getProperty(GladosFocus.PROP)));
        System.setProperty(GladosFocus.CLAIM + "rp1", now + ":" + (1 / (1 + 3.0)));          // RP1 only nearest
        won = GladosFocus.arbitrate(look, now);
        check("focus: looking at GLaDOS beats the RP1 being nearer -> GLaDOS owns, duck yields", won && GladosFocus.ownedNow() && duckYields(),
                System.getProperty(GladosFocus.PROP));
        System.setProperty(GladosFocus.CLAIM + "rp1", (now - 1500) + ":2.99");
        won = GladosFocus.arbitrate(near, now);
        check("focus: stale claims of others are ignored", won && GladosFocus.ownedNow(), System.getProperty(GladosFocus.PROP));
        won = GladosFocus.arbitrate(0, now);
        check("focus: no claim -> claim and focus withdrawn", !won && !GladosFocus.ownedNow()
                && System.getProperty(GladosFocus.CLAIM + GladosFocus.OWNER) == null, String.valueOf(System.getProperty(GladosFocus.PROP)));
        System.clearProperty(GladosFocus.CLAIM + "rp1");
        System.clearProperty(GladosFocus.PROP);

        // -------------------------------------------------------------- Opus through SVC's own codecs
        roundTrip("SVC Concentus (pure Java, what JavaOpus*Impl wraps)", javaEnc(),
                javaDec());
        try {
            de.maxhenkel.opus4j.OpusEncoder ne = new de.maxhenkel.opus4j.OpusEncoder(48000, 1, de.maxhenkel.opus4j.OpusEncoder.Application.VOIP);
            ne.setMaxPayloadSize(1024);
            de.maxhenkel.opus4j.OpusDecoder nd = new de.maxhenkel.opus4j.OpusDecoder(48000, 1);
            nd.setFrameSize(960);
            System.out.println("      native " + ne.getOpusVersion());
            roundTrip("SVC native opus4j", new OpusEncoder() {
                public byte[] encode(short[] p) { return ne.encode(p); }
                public void resetState() { ne.resetState(); }
                public boolean isClosed() { return ne.isClosed(); }
                public void close() { ne.close(); }
            }, new OpusDecoder() {
                public short[] decode(byte[] p) { return nd.decode(p); }
                public short[][] decode(byte[] p, int n) { return nd.decode(p, n); }
                public void resetState() { nd.resetState(); }
                public boolean isClosed() { return nd.isClosed(); }
                public void close() { nd.close(); }
            });
        } catch (Throwable t) {
            System.out.println("SKIP  native opus4j: " + t);
        }

        // -------------------------------------------------------------- live against a GLaDOS voice backend
        if (args.length >= 2) {
            live(args[0], java.util.Arrays.copyOfRange(args, 1, args.length));
        }
        System.out.println("SUMMARY: " + pass + "/" + (pass + fail) + " passed -> " + (fail == 0 ? "PASS" : "FAIL"));
        System.exit(fail == 0 ? 0 : 1);
    }

    static void live(String url, String[] files) throws Exception {
        // the SVC client's encoder (player's mic) and the server-side API decoder (MicrophonePacketEvent path)
        OpusEncoder micEnc = javaEnc();
        OpusDecoder micDec = javaDec();
        // the plugin's reply path: Frames -> API encoder -> (channel.send)
        VoicePcm.Frames out = new VoicePcm.Frames(3, 1500);
        OpusEncoder voiceEnc = javaEnc();
        OpusDecoder listener = javaDec();   // what a nearby SVC client would decode
        AtomicInteger sentPackets = new AtomicInteger();
        double[] voicedSec = {0};
        List<String> transcripts = new ArrayList<>();
        boolean[] live = {false};
        GladosVoiceLink link = new GladosVoiceLink(url, out::push, json -> {
            System.out.println("      <- " + json);
            if (json.contains("\"transcript\"")) {
                transcripts.add(json);
            }
            if (json.contains("\"live\"")) {
                live[0] = true;
            }
        }, m -> System.out.println("      link: " + m));
        // 20 ms sender thread, like GladosVoicePlugin.tick()
        Thread sender = new Thread(() -> {
            long next = System.nanoTime();
            while (!Thread.currentThread().isInterrupted()) {
                short[] f = out.poll();
                if (f != null) {
                    byte[] pkt = voiceEnc.encode(f);
                    sentPackets.incrementAndGet();
                    short[] heard = listener.decode(pkt);
                    if (VoicePcm.rms(heard) > 250) {
                        voicedSec[0] += 0.02;
                    }
                }
                next += 20_000_000L;
                long sleep = next - System.nanoTime();
                if (sleep > 0) {
                    try {
                        Thread.sleep(sleep / 1_000_000L, (int) (sleep % 1_000_000L));
                    } catch (InterruptedException e) {
                        return;
                    }
                }
            }
        }, "voice-sender");
        sender.setDaemon(true);
        sender.start();
        boolean open = link.awaitOpen(5000);
        check("live: GladosVoiceLink connects to " + url, open, open ? "open" : "not reachable");
        if (!open) {
            return;
        }
        link.sendText("{\"type\":\"start\"}");
        for (int i = 0; i < 300 && !live[0]; i++) {
            Thread.sleep(100);
        }
        check("live: backend reports the voice call is up", live[0], live[0] ? "live" : "no 'live' status in 30 s");
        if (!live[0]) {
            link.close();
            return;
        }
        for (String file : files) {
            byte[] raw = Files.readAllBytes(Path.of(file));
            short[] pcm = VoicePcm.toShorts(raw, 0, raw.length);
            System.out.printf("      push-to-talk %s (%.1f s)%n", file, pcm.length / 48000.0);
            long t0 = System.nanoTime();
            for (int off = 0, n = 0; off < pcm.length; off += 960, n++) {
                short[] frame = new short[960];
                System.arraycopy(pcm, off, frame, 0, Math.min(960, pcm.length - off));
                byte[] pkt = micEnc.encode(frame);             // SVC client
                link.sendPcm(micDec.decode(pkt));              // plugin: MicrophonePacketEvent -> decode -> backend
                long wait = t0 + (n + 1) * 20_000_000L - System.nanoTime();
                if (wait > 0) {
                    Thread.sleep(wait / 1_000_000L);
                }
            }
            link.sendText("{\"type\":\"ptt\",\"on\":false}");  // SVC's empty end-of-transmission packet
            int before = sentPackets.get();
            long until = System.currentTimeMillis() + 25000;
            long lastChange = System.currentTimeMillis();
            int last = before;
            while (System.currentTimeMillis() < until) {
                Thread.sleep(200);
                int now = sentPackets.get();
                if (now != last) {
                    last = now;
                    lastChange = System.currentTimeMillis();
                }
                if (now > before && System.currentTimeMillis() - lastChange > 2500) {
                    break;
                }
            }
            System.out.printf("      reply: %d Opus packets (%.1f s) sent to her head's channel%n", sentPackets.get() - before, (sentPackets.get() - before) * 0.02);
        }
        check("live: GLaDOS reply audio came back and was re-encoded for SVC", sentPackets.get() > 25 && voicedSec[0] > 0.5,
                String.format("%d packets, %.1f s voiced after a decode by a listener, link sent %d B / got %d B",
                        sentPackets.get(), voicedSec[0], link.sentBytes(), link.recvBytes()));
        check("live: transcripts arrived (captions)", !transcripts.isEmpty(), transcripts.size() + " transcript messages");
        link.sendText("{\"type\":\"hangup\"}");
        Thread.sleep(300);
        link.close();
        sender.interrupt();
    }
}
