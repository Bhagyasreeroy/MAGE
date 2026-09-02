'use client';

/**
 * components/mic-button.tsx
 * ─────────────────────────
 * The voice-input control shared by the goal box and the follow-up chat bar.
 *
 * Purely presentational — every piece of behaviour lives in
 * `lib/use-voice-input.ts`, so the two call sites differ only in placement.
 * Renders nothing at all when the browser cannot record, which is deliberate:
 * a visible button that can never work is worse than no button.
 */

import type { UseVoiceInput } from '../lib/use-voice-input';
import { MAX_RECORDING_MS } from '../lib/use-voice-input';

const MicIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" />
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8" />
  </svg>
);

const StopIcon = () => (
  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);

const SpinnerIcon = () => (
  <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth={3} />
    <path className="opacity-90" fill="currentColor" d="M4 12a8 8 0 018-8v3a5 5 0 00-5 5H4z" />
  </svg>
);

function formatElapsed(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
}

interface MicButtonProps {
  voice: UseVoiceInput;
  /** Positioning classes from the call site — the two inputs differ only in this. */
  className?: string;
  /** Disables the control while the surrounding form is busy. */
  disabled?: boolean;
  /**
   * Drop the timer and status text, leaving only the button. The chat bar
   * needs this: the control sits *inside* the input's left padding, where a
   * timer beside it would overlap whatever the user is typing.
   */
  compact?: boolean;
}

export function MicButton({
  voice,
  className = '',
  disabled = false,
  compact = false,
}: MicButtonProps) {
  const { state, elapsedSeconds, isSupported, toggle } = voice;

  if (!isSupported) return null;

  const isRecording = state === 'recording';
  const isTranscribing = state === 'transcribing';
  const remaining = Math.max(0, Math.round(MAX_RECORDING_MS / 1000) - elapsedSeconds);

  const label = isRecording
    ? `Stop recording (${remaining}s left)`
    : isTranscribing
      ? 'Transcribing…'
      : 'Record your goal by voice';

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <button
        type="button"
        onClick={toggle}
        // Recording stays clickable while `disabled` — that click is how you
        // stop it, and stranding an open microphone would be worse than
        // honouring the surrounding form's busy state.
        disabled={(disabled && !isRecording) || isTranscribing}
        title={label}
        aria-label={label}
        aria-pressed={isRecording}
        className={`w-9 h-9 rounded-full flex items-center justify-center transition-all disabled:opacity-40 ${
          isRecording
            ? `bg-red-500 text-white shadow-lg shadow-red-500/30 hover:bg-red-600 ${compact ? 'animate-pulse' : ''}`
            : 'bg-cream text-navy/50 border border-dusty-rose/30 hover:text-navy hover:border-lavender'
        }`}
      >
        {isTranscribing ? <SpinnerIcon /> : isRecording ? <StopIcon /> : <MicIcon />}
      </button>

      {isRecording && !compact && (
        <span className="flex items-center gap-1.5 text-xs font-semibold text-red-500 tabular-nums">
          <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
          {formatElapsed(elapsedSeconds)}
        </span>
      )}
      {isTranscribing && !compact && (
        <span className="text-xs font-medium text-navy/40">Transcribing…</span>
      )}
    </div>
  );
}
