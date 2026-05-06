export interface MotifOccurrence {
  occurrence_id: string;
  motif_key: string;
  start_timestamp: number;
  end_timestamp: number;
  [key: string]: unknown;
}
