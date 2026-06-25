"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useSpeculativeStream } from "@/lib/useSpeculativeStream";

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
};

let msgId = 0;

export function ChatStream() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const stream = useSpeculativeStream();
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, stream.tokens]);

  const handleSend = useCallback(() => {
    if (!input.trim() || stream.isStreaming) return;
    const prompt = input;
    setInput("");

    const userMsg: Message = { id: String(++msgId), role: "user", content: prompt };
    const assistantMsg: Message = { id: String(++msgId), role: "assistant", content: "" };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    stream.start(prompt, undefined);
  }, [input, stream]);

  useEffect(() => {
    if (!stream.tokens) return;
    setMessages((prev) => {
      const next = [...prev];
      const last = next[next.length - 1];
      if (last?.role === "assistant") {
        next[next.length - 1] = { ...last, content: stream.tokens };
      }
      return next;
    });
  }, [stream.tokens]);

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h2 style={{ marginBottom: 8 }}>Speculative Chat</h2>

      <div
        style={{
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: 12,
          padding: 16,
          minHeight: 320,
          maxHeight: 480,
          overflowY: "auto",
          marginBottom: 16,
          background: "rgba(0,0,0,0.15)",
        }}
      >
        {messages.length === 0 && (
          <p style={{ opacity: 0.4, textAlign: "center", marginTop: 120 }}>Send a message to start streaming.</p>
        )}
        {messages.map((m) => (
          <div key={m.id} style={{ marginBottom: 12 }}>
            <strong style={{ color: m.role === "user" ? "#6ea8fe" : "#7ee3b5" }}>
              {m.role === "user" ? "You" : "SpecVLM"}
            </strong>
            <pre
              style={{
                margin: "4px 0 0",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                fontFamily: "ui-monospace, monospace",
                fontSize: 14,
                lineHeight: 1.5,
                opacity: 0.9,
              }}
            >
              {m.content || (stream.isStreaming && m.role === "assistant" ? "▊" : "")}
            </pre>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      {stream.metrics && (
        <div
          style={{
            display: "flex",
            gap: 16,
            fontSize: 13,
            marginBottom: 12,
            padding: "8px 12px",
            borderRadius: 8,
            background: "rgba(255,255,255,0.04)",
          }}
        >
          <span>Acceptance: <strong>{(stream.metrics.acceptance_rate * 100).toFixed(1)}%</strong></span>
          <span>TPS: <strong>{stream.metrics.tokens_per_sec.toFixed(1)}</strong></span>
          <span>Proposals: <strong>{stream.metrics.proposal_count}</strong></span>
        </div>
      )}

      {stream.error && (
        <p style={{ color: "#f87171", fontSize: 13, marginBottom: 8 }}>{stream.error}</p>
      )}

      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), handleSend())}
          placeholder="Enter a prompt..."
          disabled={stream.isStreaming}
          style={{
            flex: 1,
            padding: "10px 14px",
            borderRadius: 10,
            border: "1px solid rgba(255,255,255,0.1)",
            background: "rgba(0,0,0,0.25)",
            color: "inherit",
            outline: "none",
          }}
        />
        <button
          onClick={handleSend}
          disabled={stream.isStreaming || !input.trim()}
          style={{
            padding: "10px 18px",
            borderRadius: 10,
            border: "1px solid rgba(255,255,255,0.1)",
            background: stream.isStreaming
              ? "rgba(255,255,255,0.05)"
              : "linear-gradient(180deg, rgba(110,168,254,0.35), rgba(110,168,254,0.12))",
            color: "inherit",
            cursor: stream.isStreaming ? "not-allowed" : "pointer",
            fontWeight: 600,
          }}
        >
          {stream.isStreaming ? "Streaming..." : "Send"}
        </button>
        {stream.isStreaming && (
          <button
            onClick={stream.stop}
            style={{
              padding: "10px 18px",
              borderRadius: 10,
              border: "1px solid rgba(255,255,255,0.1)",
              background: "linear-gradient(180deg, rgba(248,113,113,0.35), rgba(248,113,113,0.12))",
              color: "inherit",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Stop
          </button>
        )}
      </div>
    </div>
  );
}
