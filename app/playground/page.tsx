"use client";

import { useState, useRef, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { useUI } from "../providers";
import { cn } from "@/lib/utils";
import { ImageUpload } from "@/components/ImageUpload";
import { ChatMessageBubble } from "@/components/ChatMessage";
import { PerformancePanel } from "@/components/PerformancePanel";
import { Send, Menu, Loader2, ArrowDown } from "lucide-react";
import type { ChatMessage } from "@/lib/types";

const WELCOME_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content: "Upload an image and ask a question. I'll analyze it using speculative decoding and show you the performance metrics in real time.",
};

export default function PlaygroundPage() {
  const { sidebarCollapsed, toggleSidebar } = useUI();
  const [messages, setMessages] = useState<ChatMessage[]>([WELCOME_MESSAGE]);
  const [input, setInput] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [streaming, setStreaming] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleImageSelect = (file: File, preview: string) => {
    setImageFile(file);
    setImage(preview);
  };

  const handleRemoveImage = () => {
    setImageFile(null);
    setImage(null);
  };

  const handleSend = async () => {
    if (!input.trim() && !image) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: input.trim() || "Describe this image",
      image: image ?? undefined,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setStreaming(true);

    // Simulate streaming response with metrics
    const assistantId = `assistant-${Date.now()}`;
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      metrics: {
        latency: "0.82s",
        tokensPerSecond: "145",
        acceptanceRate: "78%",
        model: "Llama-3.2-Vision",
      },
    };

    setMessages((prev) => [...prev, assistantMessage]);

    const responseText =
      image
        ? "I can see this is an image you've uploaded. Based on the visual content, I can analyze the scene, objects, text, and composition. The speculative decoding pipeline processed this in 0.82s — 2.4x faster than baseline autoregressive decoding."
        : "Speculative decoding uses a lightweight draft model to propose candidate tokens, then a target model verifies them in parallel. This achieves 2-3x speedup over traditional autoregressive decoding while maintaining identical output quality. The draft model (Qwen2-VL-2B) runs ~5x faster than the target model (Qwen2-VL-7B), and with an acceptance rate of ~0.78, most draft tokens pass verification on the first attempt.";

    // Type out character by character
    let charIndex = 0;
    const typeInterval = setInterval(() => {
      if (charIndex < responseText.length) {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: responseText.slice(0, charIndex + 1) } : m
          )
        );
        charIndex++;
      } else {
        clearInterval(typeInterval);
        setStreaming(false);
      }
    }, 15);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="h-screen flex flex-col chat-bg">
      <Sidebar collapsed={sidebarCollapsed} onToggle={toggleSidebar} />

      <div className={cn("flex-1 flex flex-col transition-all duration-200", sidebarCollapsed ? "ml-14" : "ml-64")}>
        {/* Top bar */}
        <header className="flex items-center justify-between h-14 px-6 border-b border-border/30 flex-shrink-0">
          <div className="flex items-center gap-3">
            <button onClick={toggleSidebar} className="p-1.5 rounded-md hover:bg-accent text-muted-foreground">
              <Menu className="w-4 h-4" />
            </button>
            <span className="text-sm font-medium text-foreground">Playground</span>
          </div>
        </header>

        {/* Main content: 3-column layout */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left: Image upload */}
          <div className="w-56 border-r border-border/30 p-4 flex-shrink-0 overflow-y-auto hidden lg:block">
            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-3">Image</div>
            <ImageUpload onImageSelect={handleImageSelect} onRemove={handleRemoveImage} image={image} />
            <div className="mt-4 text-[10px] text-muted-foreground leading-relaxed">
              Upload an image to analyze. The vision encoder processes it before speculative decoding begins.
            </div>
          </div>

          {/* Center: Chat */}
          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
              {messages.map((msg) => (
                <ChatMessageBubble key={msg.id} message={msg} />
              ))}
              <div ref={chatEndRef} />
            </div>

            {/* Input area */}
            <div className="flex-shrink-0 border-t border-border/30 p-4">
              {/* Mobile image upload */}
              <div className="lg:hidden mb-2">
                <ImageUpload onImageSelect={handleImageSelect} onRemove={handleRemoveImage} image={image} />
              </div>

              <div className="flex items-end gap-2 max-w-3xl mx-auto">
                <div className="flex-1 relative">
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about the image..."
                    rows={1}
                    className="w-full resize-none rounded-xl border border-border/60 bg-card/50 px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground outline-none focus:border-primary/50 transition-colors"
                    style={{ minHeight: 44, maxHeight: 120 }}
                    onInput={(e) => {
                      const el = e.currentTarget;
                      el.style.height = "auto";
                      el.style.height = Math.min(el.scrollHeight, 120) + "px";
                    }}
                  />
                </div>
                <button
                  onClick={handleSend}
                  disabled={streaming || (!input.trim() && !image)}
                  className="flex-shrink-0 w-10 h-10 rounded-xl bg-primary flex items-center justify-center hover:bg-primary/90 transition-colors disabled:opacity-40"
                >
                  {streaming ? (
                    <Loader2 className="w-4 h-4 text-primary-foreground animate-spin" />
                  ) : (
                    <Send className="w-4 h-4 text-primary-foreground" />
                  )}
                </button>
              </div>
              <div className="text-[10px] text-muted-foreground text-center mt-2">
                SpecVLM may produce inaccurate responses. Use Shift+Enter for new line.
              </div>
            </div>
          </div>

          {/* Right: Performance metrics */}
          <div className="w-60 border-l border-border/30 p-4 flex-shrink-0 overflow-y-auto hidden lg:block">
            <PerformancePanel />
          </div>
        </div>
      </div>
    </div>
  );
}
