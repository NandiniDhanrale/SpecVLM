"use client";

import { useCallback, useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { useUI } from "../providers";
import { cn } from "@/lib/utils";
import { Menu, GitBranch, Info, Cpu, ShieldCheck, Zap, Eye, FileText } from "lucide-react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeProps,
  Handle,
  Position,
  MarkerType,
} from "@xyflow/react";

// eslint-disable-next-line @typescript-eslint/no-unused-vars
const _MiniMap = MiniMap;

import "@xyflow/react/dist/style.css";

const nodeTypes = {
  pipelineNode: PipelineNode,
};

const initialNodes: Node[] = [
  {
    id: "input",
    type: "pipelineNode",
    position: { x: 0, y: 100 },
    data: { label: "Image", icon: "image", desc: "Input image (PNG/JPG) or image URL" },
  },
  {
    id: "encoder",
    type: "pipelineNode",
    position: { x: 180, y: 100 },
    data: { label: "Vision Encoder", icon: "eye", desc: "ViT extracts visual features from the input image" },
  },
  {
    id: "draft",
    type: "pipelineNode",
    position: { x: 380, y: 0 },
    data: { label: "Draft Model", icon: "zap", desc: "Qwen2-VL-2B proposes K candidate tokens. Fast but lower quality." },
  },
  {
    id: "verifier",
    type: "pipelineNode",
    position: { x: 380, y: 200 },
    data: { label: "Verifier Model", icon: "shield", desc: "Computes p_target/p_draft ratio. Accepts if ratio > U(0,1)." },
  },
  {
    id: "target",
    type: "pipelineNode",
    position: { x: 580, y: 100 },
    data: { label: "Target Model", icon: "cpu", desc: "Qwen2-VL-7B corrects rejected positions via rejection sampling." },
  },
  {
    id: "output",
    type: "pipelineNode",
    position: { x: 780, y: 100 },
    data: { label: "Generated Answer", icon: "file", desc: "Final output: accepted tokens decoded and emitted." },
  },
];

const initialEdges: Edge[] = [
  { id: "e-input-encoder", source: "input", target: "encoder", animated: true, style: { stroke: "hsl(var(--border))", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "hsl(var(--border))" } },
  { id: "e-encoder-draft", source: "encoder", target: "draft", animated: true, style: { stroke: "hsl(var(--border))", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "hsl(var(--border))" } },
  { id: "e-encoder-verifier", source: "encoder", target: "verifier", animated: true, style: { stroke: "hsl(var(--border))", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "hsl(var(--border))" } },
  { id: "e-draft-verifier", source: "draft", target: "verifier", animated: true, style: { stroke: "hsl(var(--border))", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "hsl(var(--border))" } },
  { id: "e-verifier-target", source: "verifier", target: "target", animated: true, style: { stroke: "hsl(var(--border))", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "hsl(var(--border))" } },
  { id: "e-target-output", source: "target", target: "output", animated: true, style: { stroke: "hsl(var(--border))", strokeWidth: 2 }, markerEnd: { type: MarkerType.ArrowClosed, color: "hsl(var(--border))" } },
];

function PipelineNode({ data }: NodeProps) {
  const d = data as { label: string; icon: string; desc: string };
  const icons: Record<string, typeof Eye> = { image: Eye, eye: Eye, zap: Zap, shield: ShieldCheck, cpu: Cpu, file: FileText };
  const Icon = icons[d.icon] ?? Eye;

  return (
    <div
      className="rounded-xl border border-border/50 bg-card/80 backdrop-blur-sm px-4 py-3 shadow-lg min-w-[140px] cursor-pointer hover:border-primary/40 hover:bg-card transition-all group"
    >
      <Handle type="target" position={Position.Top} className="!bg-border !w-2 !h-2" />
      <Handle type="source" position={Position.Bottom} className="!bg-border !w-2 !h-2" />

      <div className="flex items-center gap-2.5">
        <div className="w-7 h-7 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
          <Icon className="w-3.5 h-3.5 text-primary" />
        </div>
        <span className="text-xs font-semibold text-foreground whitespace-nowrap">{d.label}</span>
      </div>
    </div>
  );
}

const nodeDetails: Record<string, { title: string; details: string[] }> = {
  input: {
    title: "Image Input",
    details: [
      "Accepts PNG, JPG, WEBP up to 10MB",
      "Can be uploaded via drag & drop or file picker",
      "Also supports image URLs for remote images",
      "Resized to 224x224 for ViT processing",
    ],
  },
  encoder: {
    title: "Vision Encoder (ViT)",
    details: [
      "Vision Transformer extracts patch embeddings",
      "Outputs visual feature vectors (sequence length 256)",
      "Cross-attention between visual features and text tokens",
      "Processed once, cached for all generation steps",
    ],
  },
  draft: {
    title: "Draft Model — Qwen2-VL-2B",
    details: [
      "Lightweight 2B parameter model for fast token proposals",
      "Runs ~5x faster than target model per forward pass",
      "Proposes K=5 candidate tokens autoregressively",
      "Lower quality but sufficient for good acceptance rates",
    ],
  },
  verifier: {
    title: "Verifier (Rejection Sampling)",
    details: [
      "Computes acceptance probability: p_target / p_draft",
      "Accepts token if ratio > uniform random sample U(0,1)",
      "Guarantees output matches target model distribution exactly",
      "Typical acceptance rate: 0.6-0.9 depending on task",
    ],
  },
  target: {
    title: "Target Model — Qwen2-VL-7B",
    details: [
      "Full 7B parameter model for high-quality generations",
      "Only computes logits for K draft positions (not full autoregressive)",
      "When rejection occurs, resamples from target distribution",
      "Majority of tokens accepted on first draft attempt",
    ],
  },
  output: {
    title: "Generated Answer",
    details: [
      "Accepted tokens emitted immediately to output stream",
      "Process repeats until max_tokens or EOS token",
      "Total throughput: 2-3x faster than pure autoregressive",
      "Output quality identical to target model alone",
    ],
  },
};

export default function ArchitecturePage() {
  const { sidebarCollapsed, toggleSidebar } = useUI();
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      setSelectedNode(node.id);
    },
    []
  );

  const detail = selectedNode ? nodeDetails[selectedNode] : null;

  return (
    <div className="min-h-screen chat-bg">
      <Sidebar collapsed={sidebarCollapsed} onToggle={toggleSidebar} />

      <div className={cn("h-screen flex flex-col transition-all duration-200", sidebarCollapsed ? "ml-14" : "ml-64")}>
        <header className="flex items-center justify-between h-14 px-6 border-b border-border/30 flex-shrink-0">
          <div className="flex items-center gap-3">
            <button onClick={toggleSidebar} className="p-1.5 rounded-md hover:bg-accent text-muted-foreground">
              <Menu className="w-4 h-4" />
            </button>
            <GitBranch className="w-4 h-4 text-warning" />
            <span className="text-sm font-medium text-foreground">Architecture</span>
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden">
          {/* React Flow canvas */}
          <div className="flex-1">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onNodeClick={onNodeClick}
              nodeTypes={nodeTypes}
              fitView
              fitViewOptions={{ padding: 0.3 }}
              minZoom={0.5}
              maxZoom={2}
            >
              <Background color="hsl(var(--border))" gap={20} size={1} />
              <Controls className="!bg-card !border-border !rounded-lg" />
            </ReactFlow>
          </div>

          {/* Detail panel */}
          <div className="w-72 border-l border-border/30 p-4 overflow-y-auto flex-shrink-0">
            {detail ? (
              <div className="space-y-4 animate-fade-in">
                <div className="flex items-center gap-2">
                  <Info className="w-4 h-4 text-primary" />
                  <h3 className="text-sm font-semibold text-foreground">{detail.title}</h3>
                </div>
                <ul className="space-y-2">
                  {detail.details.map((d, i) => (
                    <li key={i} className="text-xs text-muted-foreground leading-relaxed flex gap-2">
                      <span className="text-primary mt-0.5 flex-shrink-0">•</span>
                      {d}
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center text-muted-foreground">
                <GitBranch className="w-8 h-8 mb-3 opacity-30" />
                <p className="text-xs">Click a node</p>
                <p className="text-[11px]">to see details</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
