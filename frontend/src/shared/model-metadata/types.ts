// Shared model metadata types (architecture-parsing output consumed by both views).
export interface ArchSpec {
  model_id: string; model_type: string; architecture: string;
  hidden_size: number; num_layers: number; vocab_size: number | null;
  context_length: number; intermediate_size: number; activation: string;
  attention: { kind: string; num_heads: number; num_kv_heads: number; head_dim: number };
  moe: { num_experts: number; experts_per_tok: number } | null;
  norm_type: string; positional: string; total_params: number; active_params: number; hf_url: string;
}
export const fmtParams = (n: any): string => typeof n !== 'number' || !n ? '—' : n >= 1e9 ? `${(n / 1e9).toFixed(2)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n.toLocaleString()
export const fmtNum = (n: any): string => typeof n !== 'number' || n == null ? '—' : n.toLocaleString()
