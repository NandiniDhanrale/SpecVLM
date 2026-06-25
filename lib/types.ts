export type TokenSource = "draft" | "accepted" | "rejected" | "regenerated" | "target";

export interface TokenEvent {
  token_id: number;
  text: string;
  source: TokenSource;
  position: number;
}

export interface Metrics {
  accepted_draft_tokens: number;
  proposed_draft_tokens: number;
  acceptance_rate: number;
  tokens_per_second: number;
  t_ms: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  image?: string;
  metrics?: {
    latency: string;
    tokensPerSecond: string;
    acceptanceRate: string;
    model: string;
  };
}

export interface MetricPoint extends Metrics {
  i: number;
  ttft_ms: number;
  elapsed_ms: number;
}
