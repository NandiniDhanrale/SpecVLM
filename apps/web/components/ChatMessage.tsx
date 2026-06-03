"use client";

import { cn } from "@/lib/utils";
import { Bot, User } from "lucide-react";
import type { ChatMessage } from "@/lib/types";

interface ChatMessageProps {
  message: ChatMessage;
}

export function ChatMessageBubble({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <div className={cn("flex gap-3 w-full animate-fade-in", isUser ? "justify-end" : "justify-start")}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-primary/15 flex items-center justify-center flex-shrink-0 mt-0.5">
          <Bot className="w-4 h-4 text-primary" />
        </div>
      )}

      <div className={cn("max-w-[80%] md:max-w-[70%]", isUser && "order-1")}>
        {message.image && (
          <div className="mb-2 rounded-lg overflow-hidden border border-border/50 max-w-[260px]">
            <img src={message.image} alt="Uploaded" className="w-full h-auto object-cover" />
          </div>
        )}

        <div
          className={cn(
            "rounded-2xl px-4 py-3 text-sm leading-relaxed",
            isUser
              ? "bg-primary/15 text-primary-foreground border border-primary/20"
              : "bg-card border border-border/50"
          )}
        >
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>

        {/* Metrics footer for assistant responses */}
        {!isUser && message.metrics && (
          <div className="flex items-center gap-4 mt-2 px-1">
            <span className="text-[11px] text-muted-foreground">
              Latency: <span className="text-foreground font-medium">{message.metrics.latency}</span>
            </span>
            <span className="text-[11px] text-muted-foreground">
              Tokens/s: <span className="text-foreground font-medium">{message.metrics.tokensPerSecond}</span>
            </span>
            <span className="text-[11px] text-muted-foreground">
              Model: <span className="text-foreground font-medium">{message.metrics.model}</span>
            </span>
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-7 h-7 rounded-full bg-primary/10 flex items-center justify-center flex-shrink-0 mt-0.5">
          <User className="w-4 h-4 text-primary" />
        </div>
      )}
    </div>
  );
}
