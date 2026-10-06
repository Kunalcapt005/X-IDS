export interface Alert {
  id: number;
  predicted_label: string;
  confidence: number;
  severity: string;
  created_at: string;
  explanation?: string;
}

export interface SessionSummary {
  id: number;
  status: string;
  replay_rate: number;
  started_at: string;
}
