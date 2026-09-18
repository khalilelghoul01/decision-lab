import {
  compileJevRequest,
  type Description,
  type JevQuestion,
  type JevRequest,
  type JevResponse,
  type JevAnswer,
} from "./jev-protocol.js";
import { checkAbort, DecisionError } from "./errors.js";
import type { ModelManifest } from "./assets.js";

export type Questions = Record<string, JevQuestion>;
export type DecisionRequest<Q extends Questions = Questions> = Omit<
  JevRequest,
  "model" | "questions"
> & {
  model?: JevRequest["model"];
  questions: Q;
};
type AnswerFor<Q extends JevQuestion> = Q extends {
  type: "choice";
  criteria: infer C;
}
  ? Omit<Extract<JevAnswer, { type: "choice" }>, "choice" | "probabilities"> & {
      choice: keyof C & string;
      probabilities: Record<keyof C & string, number>;
    }
  : Extract<JevAnswer, { type: Q["type"] }>;
export type DecisionResponse<Q extends Questions = Questions> = Omit<
  JevResponse,
  "answers"
> & {
  answers: { [K in keyof Q]: AnswerFor<Q[K]> };
};
export interface InferenceOptions {
  /** accurate averages the original and reversed option orders; fast uses one. */
  mode?: "fast" | "accurate";
  /** Cancels queued work. An active GPU dispatch finishes before its result is discarded. */
  signal?: AbortSignal;
}

export function choice<const C extends Record<string, Description | null>>(
  instructions: Description,
  criteria: C,
) {
  return { type: "choice" as const, instructions, criteria };
}
export function noul(
  instructions: Description,
  criteria?: { true: string; false: string },
) {
  return criteria
    ? { type: "noul" as const, instructions, criteria }
    : { type: "noul" as const, instructions };
}
export function score(
  instructions: Description,
  levels: readonly Description[],
) {
  return { type: "score" as const, instructions, criteria: [...levels] };
}

/** Internal seam for the runtime and lifecycle tests; use loadModel() to create clients. */
export interface DecisionRuntime {
  readonly modelName: JevResponse["model"];
  readonly maxChoices: number;
  readonly metadata: Readonly<ModelManifest>;
  systemOne(request: JevRequest, fast: boolean): Promise<JevResponse>;
  dispose(): Promise<void>;
}

export class DecisionModel {
  private tail: Promise<unknown> = Promise.resolve();
  private release?: Promise<void>;
  private closing = false;
  private disposed = false;
  /** @internal */
  constructor(private readonly runtime: DecisionRuntime) {}

  get status(): "ready" | "disposing" | "disposed" {
    return this.disposed ? "disposed" : this.closing ? "disposing" : "ready";
  }
  get metadata(): Readonly<ModelManifest> {
    return structuredClone(this.runtime.metadata);
  }

  async systemOne<const Q extends Questions>(
    input: DecisionRequest<Q>,
    options: InferenceOptions = {},
  ): Promise<DecisionResponse<Q>> {
    const signal = options.signal;
    if (this.closing)
      throw new DecisionError(
        "DISPOSED",
        "This model is disposing or already disposed. Load a new model.",
      );
    checkAbort(signal);
    let request: JevRequest;
    try {
      if (
        options.mode !== undefined &&
        !["fast", "accurate"].includes(options.mode)
      )
        throw Error("mode must be fast or accurate.");
      // Snapshot before queuing so caller mutations cannot change pending decisions.
      const candidate = structuredClone({
        ...input,
        model: input.model ?? this.runtime.modelName,
      });
      if (
        candidate.model !== "jev-latest" &&
        candidate.model !== this.runtime.modelName
      )
        throw Error(
          `The loaded model is ${this.runtime.modelName}, not ${candidate.model}.`,
        );
      request = compileJevRequest(candidate, this.runtime.maxChoices).request;
    } catch (cause) {
      throw new DecisionError(
        "INVALID_REQUEST",
        cause instanceof Error ? cause.message : "Invalid request.",
        { cause },
      );
    }
    const fast = options.mode === "fast";
    const operation = this.tail.then(async () => {
      checkAbort(signal);
      try {
        const response = await this.runtime.systemOne(request, fast);
        checkAbort(signal);
        return response as DecisionResponse<Q>;
      } catch (cause) {
        checkAbort(signal);
        if (cause instanceof DecisionError) throw cause;
        throw new DecisionError(
          "INFERENCE_FAILED",
          cause instanceof Error ? cause.message : "Inference failed.",
          { cause },
        );
      }
    });
    // A failed request does not poison the next request's queue.
    this.tail = operation.catch(() => {});
    return operation;
  }

  /** Drains previously accepted requests, releases GPU resources, and rejects new work. */
  dispose(): Promise<void> {
    if (!this.release) {
      this.closing = true;
      this.release = this.tail
        .then(() => this.runtime.dispose())
        .finally(() => {
          this.disposed = true;
        });
    }
    return this.release;
  }
}
