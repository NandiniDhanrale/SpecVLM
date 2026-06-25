"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { MessageSquarePlus, History, BarChart3, Settings, Zap, ChevronLeft } from "lucide-react";
import { useState } from "react";

const navLinks = [
  { href: "/", label: "Home", icon: Zap },
  { href: "/playground", label: "New Chat", icon: MessageSquarePlus },
  { href: "/benchmark", label: "Benchmarks", icon: BarChart3 },
];

const bottomLinks = [
  { href: "#", label: "Model Settings", icon: Settings },
];

export function Sidebar({ collapsed }: { collapsed: boolean; onToggle: () => void }) {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "fixed left-0 top-0 bottom-0 z-50 flex flex-col border-r border-border/50 transition-all duration-200",
        collapsed ? "w-14" : "w-64"
      )}
      style={{ background: "hsl(var(--sidebar))" }}
    >
      {/* Header */}
      <div className={cn("flex items-center border-b border-border/30 px-4 h-14", collapsed ? "justify-center" : "gap-3")}>
        <div className="w-7 h-7 rounded-lg bg-primary/15 flex items-center justify-center flex-shrink-0">
          <Zap className="w-4 h-4 text-primary" />
        </div>
        {!collapsed && (
          <div className="flex flex-col min-w-0">
            <span className="text-sm font-semibold text-foreground truncate">SpecVLM</span>
            <span className="text-[10px] text-muted-foreground tracking-wider uppercase truncate">Inference Engine</span>
          </div>
        )}
      </div>

      {/* New Chat button */}
      <div className="p-3">
        <Link
          href="/playground"
          className={cn(
            "flex items-center gap-2 rounded-lg border border-border/60 hover:bg-accent transition-colors",
            collapsed ? "justify-center h-10 w-10 mx-auto" : "px-3 py-2.5"
          )}
        >
          <MessageSquarePlus className="w-4 h-4 text-muted-foreground flex-shrink-0" />
          {!collapsed && <span className="text-sm text-muted-foreground truncate">New Chat</span>}
        </Link>
      </div>

      {/* History */}
      {!collapsed && (
        <div className="px-3 mb-1">
          <div className="flex items-center gap-2 px-2 py-1.5">
            <History className="w-3.5 h-3.5 text-muted-foreground" />
            <span className="text-[11px] font-medium text-muted-foreground uppercase tracking-wider">History</span>
          </div>
          <div className="space-y-0.5">
            {["Intro to speculative decoding", "Benchmark: COCO captions", "Image analysis test"].map((item) => (
              <button
                key={item}
                className="w-full text-left px-2 py-1.5 rounded-md text-xs text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors truncate"
              >
                {item}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Nav */}
      <nav className="flex-1 px-3 space-y-0.5">
        {navLinks.map((link) => {
          const Icon = link.icon;
          const active = pathname === link.href;
          return (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "flex items-center gap-2.5 px-2.5 py-2 rounded-md text-sm transition-colors",
                collapsed && "justify-center px-0",
                active
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:text-foreground hover:bg-accent"
              )}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span className="truncate">{link.label}</span>}
            </Link>
          );
        })}
      </nav>

      {/* Bottom */}
      <div className="px-3 pb-3 space-y-0.5">
        {bottomLinks.map((link) => {
          const Icon = link.icon;
          return (
            <button
              key={link.label}
              className={cn(
                "flex items-center gap-2.5 w-full px-2.5 py-2 rounded-md text-sm text-muted-foreground hover:text-foreground hover:bg-accent transition-colors",
                collapsed && "justify-center px-0"
              )}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span className="truncate">{link.label}</span>}
            </button>
          );
        })}
      </div>
    </aside>
  );
}
