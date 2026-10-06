const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function getModels(): Promise<{ models: string[] }> {
  const response = await fetch(`${API_BASE_URL}/models`, { cache: "no-store" });
  if (!response.ok) throw new Error("Unable to load model list");
  return response.json();
}

export async function getHealth(): Promise<{ status: string; service: string }> {
  const response = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
  if (!response.ok) throw new Error("Backend unavailable");
  return response.json();
}
