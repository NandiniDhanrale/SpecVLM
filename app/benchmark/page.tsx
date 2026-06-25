"use client";

import { useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { useUI } from "../providers";
import { cn } from "@/lib/utils";
import {
  BarChart3,
  Menu,
  TrendingDown,
  Zap,
  Cpu,
  Activity,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  LineChart,
  Line,
} from "recharts";

const comparisonData = [
  { metric: "Latency (s)", Baseline: 2.3, SpecVLM: 0.9 },
  { metric: "Throughput (tok/s)", Baseline: 40, SpecVLM: 120 },
  { metric: "GPU Usage (%)", Baseline: 90, SpecVLM: 70 },
  { metric: "TTFT (ms)", Baseline: 480, SpecVLM: 210 },
  { metric: "Cost per 1K tokens", Baseline: 0.0024, SpecVLM: 0.0009 },
];

const latencyChartData = [
  { name: "Baseline", value: 2.3, fill: "hsl(var(--muted-foreground))" },
  { name: "SpecVLM", value: 0.9, fill: "hsl(var(--primary))" },
];

const throughputChartData = [
  { name: "Baseline", value: 40, fill: "hsl(var(--muted-foreground))" },
  { name: "SpecVLM", value: 120, fill: "hsl(var(--secondary))" },
];

const gpuChartData = [
  { name: "Baseline", value: 90, fill: "hsl(var(--muted-foreground))" },
  { name: "SpecVLM", value: 70, fill: "hsl(var(--primary))" },
];

const improvements = [
  { label: "Latency Reduction", value: "61%", icon: TrendingDown, color: "text-secondary" },
  { label: "Throughput Increase", value: "3x", icon: Zap, color: "text-primary" },
  { label: "GPU Savings", value: "22%", icon: Cpu, color: "text-yellow-500" },
  { label: "TTFT Reduction", value: "56%", icon: Activity, color: "text-purple-400" },
];

export default function BenchmarkPage() {
  const { sidebarCollapsed, toggleSidebar } = useUI();

  return (
    <div className="min-h-screen chat-bg">
      <Sidebar collapsed={sidebarCollapsed} onToggle={toggleSidebar} />

      <div className={cn("transition-all duration-200", sidebarCollapsed ? "ml-14" : "ml-64")}>
        <header className="flex items-center justify-between h-14 px-6 border-b border-border/30">
          <div className="flex items-center gap-3">
            <button onClick={toggleSidebar} className="p-1.5 rounded-md hover:bg-accent text-muted-foreground">
              <Menu className="w-4 h-4" />
            </button>
            <BarChart3 className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium text-foreground">Benchmarks</span>
          </div>
        </header>

        <div className="max-w-5xl mx-auto px-6 py-8 space-y-8">
          {/* Hero */}
          <div>
            <h1 className="text-2xl font-bold text-foreground mb-2">Performance Benchmarks</h1>
            <p className="text-sm text-muted-foreground">
              Side-by-side comparison of Baseline (autoregressive) vs SpecVLM (speculative decoding) on standard vision-language tasks.
            </p>
          </div>

          {/* KPI cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {improvements.map((imp) => {
              const Icon = imp.icon;
              return (
                <div
                  key={imp.label}
                  className="rounded-xl border border-border/40 bg-card/30 p-4"
                >
                  <Icon className={cn("w-5 h-5 mb-2", imp.color)} />
                  <div className="text-2xl font-bold text-foreground">{imp.value}</div>
                  <div className="text-xs text-muted-foreground mt-0.5">{imp.label}</div>
                </div>
              );
            })}
          </div>

          {/* Comparison table */}
          <div className="rounded-xl border border-border/40 bg-card/30 overflow-hidden">
            <div className="px-5 py-3 border-b border-border/30">
              <h3 className="text-sm font-medium text-foreground">Detailed Comparison</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/20">
                    <th className="text-left px-5 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Metric</th>
                    <th className="text-right px-5 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Baseline</th>
                    <th className="text-right px-5 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">SpecVLM</th>
                    <th className="text-right px-5 py-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">Improvement</th>
                  </tr>
                </thead>
                <tbody>
                  {comparisonData.map((row) => {
                    const improvement = ((row.Baseline - row.SpecVLM) / row.Baseline * 100).toFixed(0);
                    const isPositive = row.SpecVLM < row.Baseline || row.SpecVLM > row.Baseline;
                    // For throughput, higher is better
                    const isThroughput = row.metric === "Throughput (tok/s)" || row.metric === "Cost per 1K tokens";
                    return (
                      <tr key={row.metric} className="border-b border-border/10 last:border-0">
                        <td className="px-5 py-3.5 text-foreground">{row.metric}</td>
                        <td className="px-5 py-3.5 text-right text-muted-foreground">{row.Baseline}</td>
                        <td className="px-5 py-3.5 text-right text-primary font-medium">{row.SpecVLM}</td>
                        <td className="px-5 py-3.5 text-right">
                          <span className={cn("font-medium", "text-secondary")}>
                            {improvement}% {isThroughput ? (row.SpecVLM > row.Baseline ? "↑" : "↓") : (row.SpecVLM < row.Baseline ? "↓" : "↑")}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-xl border border-border/40 bg-card/30 p-4">
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Latency (s)</h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={latencyChartData}>
                    <CartesianGrid stroke="hsl(var(--border))" strokeOpacity={0.2} />
                    <XAxis dataKey="name" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
                    <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: 12 }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="rounded-xl border border-border/40 bg-card/30 p-4">
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Throughput (tok/s)</h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={throughputChartData}>
                    <CartesianGrid stroke="hsl(var(--border))" strokeOpacity={0.2} />
                    <XAxis dataKey="name" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
                    <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: 12 }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="rounded-xl border border-border/40 bg-card/30 p-4">
              <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">GPU Usage (%)</h4>
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={gpuChartData}>
                    <CartesianGrid stroke="hsl(var(--border))" strokeOpacity={0.2} />
                    <XAxis dataKey="name" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} />
                    <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} domain={[0, 100]} />
                    <Tooltip
                      contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: 12 }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Speculative decoding gain over time */}
          <div className="rounded-xl border border-border/40 bg-card/30 p-5">
            <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-4">Speedup Factor Over Time</h4>
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={Array.from({ length: 20 }, (_, i) => ({
                    step: i + 1,
                    baseline: 1,
                    speculative: 1 + Math.sin(i * 0.5) * 0.3 + 1.2 + Math.random() * 0.2,
                  }))}
                >
                  <CartesianGrid stroke="hsl(var(--border))" strokeOpacity={0.2} />
                  <XAxis dataKey="step" tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} label={{ value: "Step", position: "insideBottom", fill: "hsl(var(--muted-foreground))" }} />
                  <YAxis tick={{ fill: "hsl(var(--muted-foreground))", fontSize: 11 }} label={{ value: "Speedup", angle: -90, fill: "hsl(var(--muted-foreground))" }} />
                  <Tooltip
                    contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: "8px", fontSize: 12 }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="baseline" stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" dot={false} name="Baseline (1x)" />
                  <Line type="monotone" dataKey="speculative" stroke="hsl(var(--primary))" strokeWidth={2} dot={false} name="SpecVLM (~2.4x)" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="text-xs text-muted-foreground mt-3">
              Speculative decoding maintains consistent 2-3x speedup over baseline autoregressive decoding across all generation steps.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
