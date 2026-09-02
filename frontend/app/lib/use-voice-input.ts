'use client';

/**
 * lib/use-voice-input.ts
 * ──────────────────────
 * Microphone capture → text, for the goal box and the follow-up chat bar.
 *
 * The one non-obvious step is the re-encode. Gemini's inline-audio support
 * covers wav/mp3/ogg/aac/flac, but Chrome's MediaRecorder produces webm and
 * Safari's produces mp4 — neither is on that list. So rather than shipping
 * ffmpeg server-side or an encoder library client-side, the recording is
 * decoded with the Web Audio API (which reads both containers natively,
 * everywhere) and re-encoded to 16 kHz mono 16-bit WAV in about forty lines
 * of DataView writes. That also shrinks the upload roughly tenfold versus
 * the browser's default stereo 48 kHz, which matters on a slow connection.
 */

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';

import { transcribeAudio } from './api';

/** Speech recognition gains nothing above 16 kHz, and mono halves the bytes. */
const TARGET_SAMPLE_RATE = 16000;

/**
 * Hard stop for a recording. Guards against a mic left open — by a user who
 * walked away, or a page that lost focus mid-recording — quietly uploading
 * minutes of room noise and spending Gemini quota on it.
 */
export const MAX_RECORDING_MS = 90_000;

/** Container preferences, best first; the first supported one wins. */
const PREFERRED_MIME_TYPES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
  'audio/ogg;codecs=opus',
];

export type VoiceInputState = 'idle' | 'recording' | 'transcribing' | 'error';

export interface UseVoiceInputOptions {
  /** Called with the transcript once transcription succeeds. Never called with an empty string. */
  onTranscript: (text: string) => void;
}

export interface UseVoiceInput {
  state: VoiceInputState;
  /** Seconds elapsed in the current recording; 0 when not recording. */
  elapsedSeconds: number;
  /** Human-readable failure, or null. Cleared when a new recording starts. */
  error: string | null;
  /** False when the browser cannot record at all — the UI should hide the control entirely. */
  isSupported: boolean;
  start: () => Promise<void>;
  stop: () => void;
  /** Start if idle, stop if recording. What the mic button calls. */
  toggle: () => void;
  dismissError: () => void;
}

// ── WAV encoding ─────────────────────────────────────────────────────────────

/** Minimal 16-bit PCM WAV wrapper around already-resampled mono samples. */
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM header size
  view.setUint16(20, 1, true); // format: uncompressed PCM
  view.setUint16(22, 1, true); // channels: mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate (mono, 2 bytes/sample)
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  // Float [-1, 1] → signed 16-bit. Clamping first matters: a sample slightly
  // over 1.0 (which decoding can produce) would otherwise wrap around to a
  // large negative value and land in the audio as a click.
  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

/**
 * Decode whatever the browser recorded and render it to 16 kHz mono WAV.
 *
 * OfflineAudioContext does the resampling and the channel downmix in one
 * render pass, so no hand-written interpolation is needed — and its output is
 * properly filtered rather than naively decimated.
 */
async function toWav(recorded: Blob): Promise<Blob> {
  const AudioContextCtor =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  const OfflineAudioContextCtor =
    window.OfflineAudioContext ??
    (window as unknown as { webkitOfflineAudioContext?: typeof OfflineAudioContext })
      .webkitOfflineAudioContext;

  if (!AudioContextCtor || !OfflineAudioContextCtor) {
    throw new Error('This browser cannot process the recording.');
  }

  const decodeContext = new AudioContextCtor();
  let decoded: AudioBuffer;
  try {
    decoded = await decodeContext.decodeAudioData(await recorded.arrayBuffer());
  } finally {
    // Every AudioContext holds an audio device handle; browsers cap how many
    // can exist at once, so a leak here breaks recording after a few tries.
    void decodeContext.close();
  }

  const frames = Math.ceil(decoded.duration * TARGET_SAMPLE_RATE);
  if (frames <= 0) {
    throw new Error('The recording was empty.');
  }

  const offline = new OfflineAudioContextCtor(1, frames, TARGET_SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();

  const rendered = await offline.startRendering();
  return encodeWav(rendered.getChannelData(0), TARGET_SAMPLE_RATE);
}

// ── Error mapping ────────────────────────────────────────────────────────────

/** Turn a getUserMedia rejection into something a user can act on. */
function describeMicError(err: unknown): string {
  const name = (err as { name?: string } | null)?.name ?? '';
  if (name === 'NotAllowedError' || name === 'SecurityError' || name === 'PermissionDeniedError') {
    return 'Microphone access was blocked. Allow it from your browser’s address bar, then try again.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
    return 'No microphone was found. Connect one and try again.';
  }
  if (name === 'NotReadableError' || name === 'TrackStartError') {
    return 'Your microphone is in use by another application.';
  }
  return 'Could not start recording. Type your goal instead.';
}

// ── Capability detection ─────────────────────────────────────────────────────
//
// Read through useSyncExternalStore rather than an effect: the server has no
// `navigator`, so the server snapshot is a flat `false` and the client's real
// answer is applied during hydration — no mismatch, and no cascading
// setState-in-effect render.

/** Support cannot change during a page's life, so nothing ever notifies. */
function subscribeToNothing(): () => void {
  return () => {};
}

function detectSupport(): boolean {
  return (
    typeof navigator !== 'undefined' &&
    // Undefined on plain HTTP served from anything but localhost — much the
    // most common reason voice silently fails to appear on a LAN address.
    Boolean(navigator.mediaDevices?.getUserMedia) &&
    typeof window !== 'undefined' &&
    typeof window.MediaRecorder !== 'undefined'
  );
}

// ── The hook ─────────────────────────────────────────────────────────────────

export function useVoiceInput({ onTranscript }: UseVoiceInputOptions): UseVoiceInput {
  const [state, setState] = useState<VoiceInputState>('idle');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const isSupported = useSyncExternalStore(subscribeToNothing, detectSupport, () => false);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const autoStopRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Guards the async tail: if the component unmounts (or the user navigates)
  // while transcription is in flight, the resolved transcript must not be
  // written into a textarea that no longer exists.
  const liveRef = useRef(true);

  /** Release the mic and every timer. Safe to call more than once. */
  const teardown = useCallback(() => {
    if (autoStopRef.current) {
      clearTimeout(autoStopRef.current);
      autoStopRef.current = null;
    }
    if (tickerRef.current) {
      clearInterval(tickerRef.current);
      tickerRef.current = null;
    }
    // Without an explicit stop() per track the browser keeps showing its
    // "recording" indicator long after the UI has moved on.
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    recorderRef.current = null;
    setElapsedSeconds(0);
  }, []);

  useEffect(
    () => () => {
      liveRef.current = false;
      teardown();
    },
    [teardown],
  );

  const start = useCallback(async () => {
    if (recorderRef.current) return; // already recording
    setError(null);

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        // Browser-side cleanup is free and meaningfully improves transcription
        // of a laptop mic in a room with other people in it.
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (err) {
      setError(describeMicError(err));
      setState('error');
      return;
    }

    const mimeType = PREFERRED_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
    // An unsupported/absent mimeType option makes the browser pick its own
    // default, which is fine — the re-encode normalises whatever comes out.
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);

    chunksRef.current = [];
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };

    recorder.onstop = async () => {
      const recorded = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' });
      chunksRef.current = [];
      teardown();

      if (recorded.size === 0) {
        if (liveRef.current) setState('idle');
        return;
      }

      if (liveRef.current) setState('transcribing');
      try {
        const wav = await toWav(recorded);
        const { transcript } = await transcribeAudio(wav);
        if (!liveRef.current) return;
        if (transcript.trim()) {
          onTranscript(transcript.trim());
          setState('idle');
        } else {
          // A successful call that heard nothing. Reported as a hint rather
          // than a failure, since nothing actually went wrong.
          setError('No speech was detected in that recording.');
          setState('error');
        }
      } catch (err) {
        if (!liveRef.current) return;
        setError(err instanceof Error ? err.message : 'Could not transcribe the recording.');
        setState('error');
      }
    };

    streamRef.current = stream;
    recorderRef.current = recorder;
    recorder.start();
    setState('recording');
    setElapsedSeconds(0);

    tickerRef.current = setInterval(() => setElapsedSeconds((s) => s + 1), 1000);
    autoStopRef.current = setTimeout(() => {
      // `stop()` fires onstop, which transcribes what was captured — the
      // ceiling truncates the recording rather than discarding it.
      recorderRef.current?.stop();
    }, MAX_RECORDING_MS);
  }, [onTranscript, teardown]);

  const stop = useCallback(() => {
    if (recorderRef.current?.state === 'recording') {
      recorderRef.current.stop();
    }
  }, []);

  const toggle = useCallback(() => {
    if (state === 'recording') stop();
    else if (state !== 'transcribing') void start();
  }, [state, start, stop]);

  const dismissError = useCallback(() => {
    setError(null);
    setState('idle');
  }, []);

  return { state, elapsedSeconds, error, isSupported, start, stop, toggle, dismissError };
}
