export type ErrorCode =
  | "UNSUPPORTED_ENVIRONMENT"
  | "DOWNLOAD_FAILED"
  | "INVALID_MODEL"
  | "INVALID_REQUEST"
  | "INFERENCE_FAILED"
  | "DISPOSED";

export class DecisionError extends Error {
  readonly name = "DecisionError";
  constructor(
    public readonly code: ErrorCode,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
  }
}

export function checkAbort(signal?: AbortSignal): void {
  if (signal?.aborted)
    throw signal.reason ?? new DOMException("Operation aborted.", "AbortError");
}
