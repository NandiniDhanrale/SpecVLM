"use client";

import { useMemo, useRef, useState, useEffect } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type Metrics = {
  accepted_draft_tokens: number;
  proposed_draft_tokens: number;
  acceptance_rate: number;
  tokens_per_second: number;
  t_ms: number;
};

type MetricPoint = Metrics & { i: number };

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export default function Page() {
  const [prompt, setPrompt] = useState("Describe what you see in the image and report speculative decoding metrics.");
  const [status, setStatus] = useState<"idle" | "streaming" | "done" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState("");
  const outputRef = useRef("");
  const rafRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [series, setSeries] = useState<MetricPoint[]>([]);
  const pointIndexRef = useRef(0);

  const latest = useMemo(() => series[series.length - 1], [series]);

  function flushOutput() {
    if (rafRef.current != null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      setOutput(outputRef.current);
    });
  }

  function stop() {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
  }

  async function start() {
    setError(null);
    setStatus("streaming");
    setSeries([]);
    pointIndexRef.current = 0;
    outputRef.current = "";
    setOutput("");

    const controller = new AbortController();
    abortRef.current = controller;

    const params = new URLSearchParams({
      prompt,
      max_tokens: "500",
      temperature: "0.8",
      top_p: "0.95",
    });

    try {
      const res = await fetch(`${API_BASE}/api/generate/stream?${params}`, {
        signal: controller.signal,
      });

      if (!res.ok) {
        throw new Error(`Server error: ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));

          if (data.type === "token") {
            outputRef.current += data.delta ?? data.text ?? "";
            flushOutput();
            const m = data.metrics as Metrics;
            const i = pointIndexRef.current++;
            setSeries((prev) => [...prev, { ...m, i }]);
          } else if (data.type === "done") {
            setStatus("done");
          }
        }
      }
    } catch (err: any) {
      if (err.name === "AbortError") return;
      setStatus("error");
      setError(err.message ?? "Connection failed");
    }
  }

  return (
    <main className="container">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <div>
          <div className="title" style={{ marginBottom: 2 }}>
            SpecVLM Streaming Dashboard
          </div>
          <div className="muted">SSE stream of tokens + speculative decoding telemetry (acceptance rate, TPS).</div>
        </div>
        <div className="row">
          <button className="btn secondary" onClick={stop} disabled={status !== "streaming"}>
            Stop
          </button>
          <button className="btn" onClick={start} disabled={status === "streaming"}>
            Generate
          </button>
        </div>
      </div>

      <div className="grid">
        <section className="panel">
          <div className="title">Prompt</div>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <div className="muted" style={{ marginTop: 10 }}>
            Status: {status}
            {error ? ` • Error: ${error}` : ""}
          </div>
        </section>

        <section className="panel">
          <div className="title">Generated Text</div>
          <div className="mono" style={{ minHeight: 140 }}>
            {output}
          </div>
        </section>

        <section className="panel" style={{ gridColumn: "1 / -1" }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div>
              <div className="title" style={{ marginBottom: 2 }}>
                Live Metrics
              </div>
              <div className="muted">
                Acceptance: {latest ? `${(latest.acceptance_rate * 100).toFixed(1)}%` : "—"} • TPS:{" "}
                {latest ? latest.tokens_per_second.toFixed(1) : "—"} • Draft accepted/proposed:{" "}
                {latest ? `${latest.accepted_draft_tokens}/${latest.proposed_draft_tokens}` : "—"}
              </div>
            </div>
            <div className="muted">Backend: SSE /api/generate/stream</div>
          </div>

          <div style={{ height: 320, marginTop: 10 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={series}>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="i" tick={{ fill: "rgba(231,236,255,0.7)" }} />
                <YAxis
                  yAxisId="left"
                  tick={{ fill: "rgba(231,236,255,0.7)" }}
                  label={{ value: "TPS", angle: -90, position: "insideLeft", fill: "rgba(231,236,255,0.7)" }}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  domain={[0, 1]}
                  tick={{ fill: "rgba(231,236,255,0.7)" }}
                  label={{
                    value: "Acceptance",
                    angle: 90,
                    position: "insideRight",
                    fill: "rgba(231,236,255,0.7)"
                  }}
                />
                <Tooltip
                  contentStyle={{ background: "#0b1020", border: "1px solid rgba(255,255,255,0.12)" }}
                  labelStyle={{ color: "rgba(231,236,255,0.8)" }}
                />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="tokens_per_second" stroke="#6ea8fe" dot={false} />
                <Line yAxisId="right" type="monotone" dataKey="acceptance_rate" stroke="#7ee3b5" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
    </main>
  );
}
