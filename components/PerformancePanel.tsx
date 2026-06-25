"use client";

import { cn } from "@/lib/utils";
import { Gauge, Zap, Cpu, Layers } from "lucide-react";
import { motion } from "framer-motion";

interface MetricRow {
  label: string;
  value: string;
  icon: typeof Gauge;
  color: string;
}

const metrics: MetricRow[] = [
  { label: "Latency", value: "0.82 s", icon: Gauge, color: "text-primary" },
  { label: "Tokens/sec", value: "145", icon: Zap, color: "text-secondary" },
  { label: "GPU Memory", value: "12.3 GB", icon: Cpu, color: "text-yellow-500" },
  { label: "Batch Size", value: "8", icon: Layers, color: "text-purple-400" },
];

export function PerformancePanel() {
  return (
    <div className="space-y-4">
      <div className="text-sm font-medium text-foreground">Performance</div>
      <div className="space-y-2">
        {metrics.map((m, i) => {
          const Icon = m.icon;
          return (
            <motion.div
              key={m.label}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className={cn(
                "flex items-center gap-3 rounded-lg border border-border/50 px-3 py-2.5 bg-card/50"
              )}
            >
              <Icon className={cn("w-4 h-4 flex-shrink-0", m.color)} />
              <div className="flex-1 min-w-0">
                <div className="text-[11px] text-muted-foreground">{m.label}</div>
              </div>
              <span className={cn("text-sm font-semibold tabular-nums", m.color)}>{m.value}</span>
            </motion.div>
          );
        })}
      </div>

      {/* Speculative decoding gain */}
      <div className="rounded-lg border border-primary/20 bg-primary/5 px-3 py-2.5 mt-4">
        <div className="text-[11px] text-muted-foreground mb-1">Speculative Decoding Gain</div>
        <div className="flex items-baseline gap-1">
          <span className="text-lg font-bold text-primary">2.4x</span>
          <span className="text-[11px] text-muted-foreground">vs baseline</span>
        </div>
        <div className="mt-2 w-full h-1.5 rounded-full bg-muted/50 overflow-hidden">
          <div className="h-full w-[70%] rounded-full bg-gradient-to-r from-primary to-secondary" />
        </div>
        <div className="flex justify-between text-[10px] text-muted-foreground mt-0.5">
          <span>Baseline</span>
          <span>SpecVLM</span>
        </div>
      </div>
    </div>
  );
}
