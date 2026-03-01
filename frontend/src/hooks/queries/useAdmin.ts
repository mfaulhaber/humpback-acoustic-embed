import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchModels,
  createModel,
  deleteModel,
  setDefaultModel,
  scanModels,
  fetchTables,
  deleteAllRecords,
} from "@/api/client";
import type { ModelConfigCreate } from "@/api/types";

export function useModels() {
  return useQuery({
    queryKey: ["models"],
    queryFn: fetchModels,
  });
}

export function useScanModels() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: scanModels,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useCreateModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ModelConfigCreate) => createModel(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useDeleteModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (modelId: string) => deleteModel(modelId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useSetDefaultModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (modelId: string) => setDefaultModel(modelId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
}

export function useTables() {
  return useQuery({
    queryKey: ["tables"],
    queryFn: fetchTables,
  });
}

export function useDeleteAllRecords() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: deleteAllRecords,
    onSuccess: () => {
      qc.invalidateQueries();
    },
  });
}
