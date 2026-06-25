"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export interface TextDeltaPacket {
  type: "text_delta";
  delta: string;
}

export interface TelemetryPacket {
  type: "telemetry";
  acceptance_rate: number;
  tokens_per_sec: number;
  proposal_count: number;
}

export type StreamPacket = TextDeltaPacket | TelemetryPacket;

export interface SpeculativeStreamState {
  tokens: string;
  metrics: TelemetryPacket | null;
  isStreaming: boolean;
  error: string | null;
  start: (prompt: string, signal?: AbortSignal) => void;
  stop: () => void;
  reset: () => void;
}

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1000;

export function useSpeculativeStream(): SpeculativeStreamState {
  const [tokens, setTokens] = useState("");
  const [metrics, setMetrics] = useState<TelemetryPacket | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tokensRef = useRef("");
  const controllerRef = useRef<AbortController | null>(null);
  const retryRef = useRef(0);
  const mountedRef = useRef(true);
  const activeRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, []);

  const stop = useCallback(() => {
    activeRef.current = false;
    controllerRef.current?.abort();
    controllerRef.current = null;
    setIsStreaming(false);
  }, []);

  const reset = useCallback(() => {
    stop();
    setTokens("");
    setMetrics(null);
    setError(null);
    tokensRef.current = "";
    retryRef.current = 0;
  }, [stop]);

  const stream = useCallback(async (prompt: string, outerSignal?: AbortSignal) => {
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "";
    const params = new URLSearchParams({ prompt, max_tokens: "500", temperature: "0.8", top_p: "0.95" });

    const controller = new AbortController();
    controllerRef.current = controller;
    activeRef.current = true;
    retryRef.current = 0;
    setError(null);
    setIsStreaming(true);

    const combinedSignal = outerSignal
      ? combineAbortSignals([outerSignal, controller.signal])
      : controller.signal;

    const attempt = async (): Promise<void> => {
      while (activeRef.current && mountedRef.current) {
        try {
          const res = await fetch(`${backendUrl}/api/generate/stream?${params}`, { signal: combinedSignal });
          if (!res.ok) throw new Error(`Server error: ${res.status}`);

          retryRef.current = 0;
          const reader = res.body?.getReader();
          if (!reader) throw new Error("No response body");

          const decoder = new TextDecoder();
          let buffer = "";

          while (activeRef.current && mountedRef.current) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (!line.startsWith("data: ")) continue;
              const raw = line.slice(6).trim();
              if (!raw) continue;

              try {
                const packet = JSON.parse(raw) as StreamPacket;
                if (packet.type === "text_delta") {
                  tokensRef.current += packet.delta;
                  setTokens(tokensRef.current);
                } else if (packet.type === "telemetry") {
                  setMetrics({
                    type: "telemetry",
                    acceptance_rate: packet.acceptance_rate,
                    tokens_per_sec: packet.tokens_per_sec,
                    proposal_count: packet.proposal_count,
                  });
                }
              } catch {
                /* skip malformed packets */
              }
            }
          }

          if (activeRef.current && mountedRef.current) {
            setIsStreaming(false);
          }
          return;
        } catch (err: any) {
          if (err.name === "AbortError") {
            setIsStreaming(false);
            return;
          }

          if (retryRef.current < MAX_RETRIES && activeRef.current && mountedRef.current) {
            retryRef.current++;
            const delay = BASE_DELAY_MS * Math.pow(2, retryRef.current - 1);
            setError(`Connection lost. Retry ${retryRef.current}/${MAX_RETRIES}...`);
            await new Promise((r) => setTimeout(r, delay));
            continue;
          }

          if (mountedRef.current) {
            setError(err.message ?? "Stream failed");
            setIsStreaming(false);
          }
          return;
        }
      }
    };

    attempt();
  }, []);

  const start = useCallback(
    (prompt: string, signal?: AbortSignal) => {
      stop();
      tokensRef.current = "";
      setTokens("");
      setMetrics(null);
      setError(null);
      stream(prompt, signal);
    },
    [stop, stream]
  );

  return { tokens, metrics, isStreaming, error, start, stop, reset };
}

function combineAbortSignals(signals: AbortSignal[]): AbortSignal {
  const controller = new AbortController();
  for (const signal of signals) {
    if (signal.aborted) {
      controller.abort(signal.reason);
      return controller.signal;
    }
    signal.addEventListener("abort", () => controller.abort(signal.reason), { once: true });
  }
  return controller.signal;
}
