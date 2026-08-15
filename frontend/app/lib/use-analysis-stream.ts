'use client';

/**
 * lib/use-analysis-stream.ts
 * ──────────────────────────
 * React hook driving the live analysis WebSocket (WS /analysis/stream).
 *
 * The backend emits each Reason/Act/Observe step the moment its agent
 * finishes, rather than withholding everything until the pipeline ends. This
 * hook accumulates those frames into render-ready state so a view can narrate
 * the run as it happens — which is what makes goal-conditioning observable:
 * two different goals visibly recruit different agents and computations.
 *
 * Protocol (see backend/routers/analysis.py):
 *   → { goal, expertise_level, dataset_id }
 *   ← { type: 'accepted' }
 *   ← { type: 'step', index, agent_name, reasoning, observation, status, latency_ms }
 *   ← { type: 'complete', run_id, result }  |  { type: 'error', detail }
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { buildStreamUrl } from './api';

/** Application-defined close codes mirrored from the backend. */
const WS_UNAUTHORIZED = 4401;
const WS_BAD_REQUEST = 4400;

export type StreamPhase = 'idle' | 'connecting' | 'running' | 'complete' | 'error';

export interface AgentStep {
  index: number;
  agent_name: string;
  action: string;
  reasoning: string;
  observation: string;
  status: 'success' | 'error' | 'skipped';
  latency_ms: number;
}

export interface StartStreamOptions {
  goal: string;
  expertiseLevel: string;
  datasetId?: string | null;
}

export interface UseAnalysisStream {
  phase: StreamPhase;
  steps: AgentStep[];
  runId: string | null;
  error: string | null;
  /** Open the socket and begin a run. Safe to call again after it settles. */
  start: (options: StartStreamOptions) => void;
  /** Abandon an in-flight run and reset back to idle. */
  cancel: () => void;
}

function describeClose(event: CloseEvent): string | null {
  if (event.code === WS_UNAUTHORIZED) {
    return 'Your session has expired. Please sign in again.';
  }
  if (event.code === WS_BAD_REQUEST) {
    return event.reason || 'The analysis request was rejected.';
  }
  // 1000 (normal) and 1005 (no status — what a clean server close looks like
  // to the browser) are how a successful run ends, so neither is an error.
  if (event.code === 1000 || event.code === 1005) {
    return null;
  }
  return `Connection closed unexpectedly (${event.code}).`;
}

export function useAnalysisStream(): UseAnalysisStream {
  const [phase, setPhase] = useState<StreamPhase>('idle');
  const [steps, setSteps] = useState<AgentStep[]>([]);
  const [runId, setRunId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  // Tracks whether the run reached a terminal frame, so `onclose` can tell a
  // successful finish from a socket that dropped mid-analysis.
  const settledRef = useRef(false);

  const closeSocket = useCallback(() => {
    const socket = socketRef.current;
    socketRef.current = null;
    if (!socket) return;
    // Detach handlers first: closing otherwise fires `onclose`, which would
    // report a deliberate teardown as a connection failure.
    socket.onopen = null;
    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;
    if (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING) {
      socket.close();
    }
  }, []);

  // Close the socket if the component unmounts mid-run, so navigating away
  // does not leave it open. The analysis itself still completes and persists
  // server-side, so the run is recoverable from history.
  useEffect(() => closeSocket, [closeSocket]);

  const cancel = useCallback(() => {
    closeSocket();
    settledRef.current = false;
    setPhase('idle');
    setSteps([]);
    setRunId(null);
    setError(null);
  }, [closeSocket]);

  const start = useCallback(
    ({ goal, expertiseLevel, datasetId }: StartStreamOptions) => {
      closeSocket();
      settledRef.current = false;
      setSteps([]);
      setRunId(null);
      setError(null);
      setPhase('connecting');

      let socket: WebSocket;
      try {
        socket = new WebSocket(buildStreamUrl());
      } catch {
        setPhase('error');
        setError('Could not open a connection to the analysis service.');
        return;
      }
      socketRef.current = socket;

      socket.onopen = () => {
        setPhase('running');
        socket.send(
          JSON.stringify({
            goal: goal.trim(),
            expertise_level: expertiseLevel,
            dataset_id: datasetId ?? null,
          }),
        );
      };

      socket.onmessage = (event: MessageEvent<string>) => {
        let message: Record<string, unknown>;
        try {
          message = JSON.parse(event.data);
        } catch {
          return; // Ignore anything that isn't a valid frame.
        }

        switch (message.type) {
          case 'accepted':
            setPhase('running');
            break;
          case 'step':
            setSteps((current) => [...current, message as unknown as AgentStep]);
            break;
          case 'complete':
            settledRef.current = true;
            setRunId(String(message.run_id ?? ''));
            setPhase('complete');
            break;
          case 'error':
            settledRef.current = true;
            setError(String(message.detail ?? 'The analysis failed.'));
            setPhase('error');
            break;
          default:
            break;
        }
      };

      socket.onerror = () => {
        // `onerror` carries no detail by design (the browser withholds it to
        // avoid leaking cross-origin information), and always precedes
        // `onclose` — which does carry a code. Let that handler report.
      };

      socket.onclose = (event: CloseEvent) => {
        socketRef.current = null;
        if (settledRef.current) return; // Already finished; nothing to report.

        const reason = describeClose(event);
        setError(reason ?? 'The connection closed before the analysis finished.');
        setPhase('error');
      };
    },
    [closeSocket],
  );

  return { phase, steps, runId, error, start, cancel };
}
