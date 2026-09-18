import * as ort from "onnxruntime-web/webgpu";
import { PreTrainedTokenizer } from "@huggingface/transformers";
import { Ajv2020 } from "ajv/dist/2020.js";
import {
  ModelAssets,
  normalizeModelUrl,
  type AssetOptions,
  type ModelManifest,
} from "./assets.js";
import { checkAbort, DecisionError } from "./errors.js";
import { compileJevRequest, buildJevResponse } from "./jev-protocol.js";

export interface Row {
  context: string;
  question: string;
  options: string[];
}
export interface Prepared {
  sequences: number[][];
  sizes: number[];
  fields: number;
  orders: number;
  prefix_length: number;
}
export interface EngineLoadOptions extends AssetOptions {
  wasmPaths?: { mjs: string; wasm: string };
  fieldBatchSize?: number;
}

export type Value = string | number | boolean;
export interface Schema {
  type: "object";
  properties: Record<
    string,
    { type: string; enum?: Value[]; title?: string; description?: string }
  >;
  required: string[];
  additionalProperties: false;
  title?: string;
  description?: string;
  $schema?: string;
}

const validator = new Ajv2020({ strict: true });
export function compileSchema(schema: Schema) {
  if (
    schema.type !== "object" ||
    Object.keys(schema).some(
      (k) =>
        ![
          "type",
          "properties",
          "required",
          "additionalProperties",
          "title",
          "description",
          "$schema",
        ].includes(k),
    )
  )
    throw Error("Only a flat object schema is supported.");
  const entries = Object.entries(schema.properties ?? {});
  if (
    !entries.length ||
    entries.length > 32 ||
    schema.additionalProperties !== false ||
    new Set(schema.required).size !== entries.length ||
    entries.some(([k]) => !schema.required.includes(k))
  )
    throw Error("Use 1–32 required fields and additionalProperties: false.");
  const fields = entries.map(([name, spec]) => {
    if (
      Object.keys(spec).some(
        (k) => !["type", "enum", "title", "description"].includes(k),
      )
    )
      throw Error(`${name}: unsupported field constraint.`);
    let values: Value[], options: string[];
    if (spec.type === "boolean" && !spec.enum) {
      values = [false, true];
      options = ["No", "Yes"];
    } else if (
      ["string", "number", "integer"].includes(spec.type) &&
      spec.enum &&
      spec.enum.length >= 2 &&
      spec.enum.length <= 4
    ) {
      values = spec.enum;
      options = values.map(String);
      if (
        new Set(options).size !== options.length ||
        values.some((v) =>
          spec.type === "string"
            ? typeof v !== "string"
            : typeof v !== "number" ||
              !Number.isFinite(v) ||
              (spec.type === "integer" && !Number.isInteger(v)),
        )
      )
        throw Error(`${name}: invalid enum values.`);
    } else
      throw Error(`${name}: use boolean or a typed enum with 2–4 choices.`);
    return {
      name,
      values,
      options,
      question:
        spec.description || spec.title || name.replaceAll("_", " ") + "?",
    };
  });
  return { schema, fields, validate: validator.compile(schema) };
}

export class BrowserDecisionEngine {
  private session!: ort.InferenceSession;
  private manifest!: ModelManifest;
  private disposed = false;
  private tokenizer!: PreTrainedTokenizer;
  private busy = false;
  backend: "webgpu" | "wasm" = "webgpu";
  fieldBatchSize = 8;
  get modelName() {
    return this.manifest.model_name ?? "decision-lab-v2";
  }
  get maxChoices() {
    return this.manifest.max_choices ?? 4;
  }
  get maxLength() {
    return this.manifest.max_length;
  }

  get metadata(): Readonly<ModelManifest> {
    return structuredClone(this.manifest);
  }

  static async load(
    base = "/model-v3-q4",
    progress: (message: string) => void = () => {},
    backend: "webgpu" | "wasm" = "webgpu",
    options: EngineLoadOptions = {},
  ) {
    const engine = new BrowserDecisionEngine();
    engine.backend = backend;
    if (
      options.fieldBatchSize !== undefined &&
      (!Number.isInteger(options.fieldBatchSize) ||
        options.fieldBatchSize < 1 ||
        options.fieldBatchSize > 32)
    )
      throw new DecisionError(
        "INVALID_MODEL",
        "fieldBatchSize must be an integer between 1 and 32.",
      );
    engine.fieldBatchSize = options.fieldBatchSize ?? 8;
    const assets = new ModelAssets(normalizeModelUrl(base), {
      ...options,
      onProgress: (event) => {
        progress(event.message);
        options.onProgress?.(event);
      },
    });
    engine.manifest = await assets.manifest();
    if (backend === "webgpu") {
      const gpu = (
        globalThis.navigator as Navigator & {
          gpu?: {
            requestAdapter(): Promise<{
              features: { has(name: string): boolean };
            } | null>;
          };
        }
      )?.gpu;
      if (!gpu)
        throw new DecisionError(
          "UNSUPPORTED_ENVIRONMENT",
          "WebGPU is unavailable. Use a WebGPU-capable browser on localhost or HTTPS.",
        );
      const adapter = await gpu.requestAdapter();
      checkAbort(options.signal);
      if (
        !adapter ||
        (engine.manifest.precision === "fp16" &&
          !adapter.features.has("shader-f16"))
      )
        throw new DecisionError(
          "UNSUPPORTED_ENVIRONMENT",
          "This model needs a WebGPU adapter with shader-f16 support.",
        );
    }
    ort.env.wasm.numThreads = 1;
    if (options.wasmPaths) ort.env.wasm.wasmPaths = options.wasmPaths;
    const decoder = new TextDecoder();
    const tokenizerJSON = JSON.parse(
      decoder.decode(
        await assets.read("tokenizer/tokenizer.json", "tokenizer"),
      ),
    );
    const tokenizerConfig = JSON.parse(
      decoder.decode(
        await assets.read("tokenizer/tokenizer_config.json", "tokenizer"),
      ),
    );
    engine.tokenizer = new PreTrainedTokenizer(tokenizerJSON, tokenizerConfig);
    engine.tokenizer.chat_template = decoder.decode(
      await assets.read("tokenizer/chat_template.jinja", "tokenizer"),
    );
    const graph = await assets.read(
      "decision.onnx",
      "model",
      engine.manifest.graph_bytes,
    );
    const weights = await assets.read(
      "decision.weights",
      "model",
      engine.manifest.weights_bytes,
    );
    checkAbort(options.signal);
    assets.emit({
      stage: "initializing",
      message: `Initializing ${backend.toUpperCase()} session…`,
    });
    const preferred: Record<string, "cpu" | "gpu-buffer"> = { logits: "cpu" };
    for (const name of engine.manifest.state_names)
      preferred[`present_${name}`] = "gpu-buffer";
    try {
      engine.session = await ort.InferenceSession.create(graph, {
        executionProviders: [backend],
        externalData: [{ path: "decision.weights", data: weights }],
        ...(backend === "webgpu" ? { preferredOutputLocation: preferred } : {}),
      });
      checkAbort(options.signal);
      assets.emit({
        stage: "ready",
        message: `Ready · ${backend.toUpperCase()} · shared prefix cache`,
      });
      return engine;
    } catch (error) {
      if (engine.session) await engine.session.release();
      throw error;
    }
  }

  private chat(context: string, question: string, options: string[]): number[] {
    const content =
      "Context:\n" +
      context +
      "\nQuestion: " +
      question +
      "\n" +
      options.map((s, i) => `${String.fromCharCode(65 + i)}. ${s}`).join("\n") +
      "\nReply with only the letter of the correct choice.";
    return this.tokenizer.apply_chat_template(
      [
        { role: "system", content: "You answer multiple-choice questions." },
        { role: "user", content },
      ],
      {
        add_generation_prompt: true,
        tokenize: true,
        return_tensor: false,
        return_dict: false,
      },
    ) as number[];
  }

  prepare(rows: Row[], orders = 2): Prepared {
    if (!rows.length || ![1, 2].includes(orders))
      throw Error("Provide questions and one or two orders.");
    const sequences: number[][] = [],
      sizes: number[] = [];
    for (let reverse = 0; reverse < orders; reverse++)
      for (const row of rows) {
        if (row.options.length < 2 || row.options.length > this.maxChoices)
          throw Error(`Use 2–${this.maxChoices} choices.`);
        const options = reverse ? [...row.options].reverse() : row.options;
        if (this.modelName === "decision-lab-v3") {
          const direct = this.chat(row.context, row.question, options);
          if (direct.length <= this.manifest.max_length) {
            sequences.push(direct);
            sizes.push(options.length);
            continue;
          }
        }
        const budget =
          this.manifest.max_length -
          this.chat("", row.question, options).length -
          4;
        if (budget < 8) throw Error("Question/options exceed context budget.");
        let context = this.tokenizer
          .encode(row.context, { add_special_tokens: false })
          .slice(0, budget);
        let ids = this.chat(
          this.tokenizer.decode(context, { skip_special_tokens: true }),
          row.question,
          options,
        );
        while (ids.length > this.manifest.max_length) {
          context = context.slice(
            0,
            -(ids.length - this.manifest.max_length + 2),
          );
          ids = this.chat(
            this.tokenizer.decode(context, { skip_special_tokens: true }),
            row.question,
            options,
          );
        }
        sequences.push(ids);
        sizes.push(options.length);
      }
    let prefix = 0,
      limit = Math.min(...sequences.map((s) => s.length)) - 1;
    while (
      prefix < limit &&
      sequences.every((s) => s[prefix] === sequences[0][prefix])
    )
      prefix++;
    return {
      sequences,
      sizes,
      fields: rows.length,
      orders,
      prefix_length: prefix,
    };
  }

  private tensors(sequences: number[][], sizes: number[]) {
    const length = Math.max(...sequences.map((s) => s.length)),
      batch = sequences.length;
    const ids = new BigInt64Array(batch * length).fill(
      BigInt(this.manifest.pad_token_id),
    );
    const mask = new BigInt64Array(batch * length);
    sequences.forEach((seq, i) =>
      seq.forEach((id, j) => {
        ids[i * length + j] = BigInt(id);
        mask[i * length + j] = 1n;
      }),
    );
    return {
      input_ids: new ort.Tensor("int64", ids, [batch, length]),
      attention_mask: new ort.Tensor("int64", mask, [batch, length]),
      sizes: new ort.Tensor("int64", BigInt64Array.from(sizes.map(BigInt)), [
        batch,
      ]),
    };
  }

  private emptyCache() {
    const states: Record<string, ort.Tensor> = {};
    this.manifest.state_names.forEach((name, i) => {
      const dims = this.manifest.state_shapes[i],
        size = dims.reduce((a, b) => a * b, 1);
      states[name] =
        this.manifest.precision === "fp16"
          ? new ort.Tensor("float16", new Uint16Array(size), dims)
          : new ort.Tensor("float32", new Float32Array(size), dims);
    });
    return states;
  }

  async scorePrepared(prepared: Prepared, shared = true) {
    if (this.disposed)
      throw new DecisionError(
        "DISPOSED",
        "The model has been disposed. Load a new model.",
      );
    if (this.busy) throw Error("Inference is already running on this engine.");
    this.busy = true;
    let cache = this.emptyCache();
    const logits: number[][] = [];
    try {
      const prefix =
        shared && prepared.sequences.length > 1 ? prepared.prefix_length : 0;
      if (prefix) {
        const inputs = this.tensors(
          [prepared.sequences[0].slice(0, prefix)],
          [this.maxChoices],
        );
        let output: ort.InferenceSession.ReturnType;
        try {
          output = await this.session.run({ ...inputs, ...cache });
        } finally {
          Object.values(inputs).forEach((t) => t.dispose());
        }
        Object.values(cache).forEach((t) => t.dispose());
        cache = {};
        output.logits.dispose();
        for (const name of this.manifest.state_names)
          cache[name] = output[`present_${name}`];
      }
      for (
        let start = 0;
        start < prepared.sequences.length;
        start += this.fieldBatchSize
      ) {
        const seq = prepared.sequences
          .slice(start, start + this.fieldBatchSize)
          .map((s) => s.slice(prefix));
        const inputs = this.tensors(
          seq,
          prepared.sizes.slice(start, start + seq.length),
        );
        let output: ort.InferenceSession.ReturnType;
        try {
          output = await this.session.run({ ...inputs, ...cache }, ["logits"]);
        } finally {
          Object.values(inputs).forEach((t) => t.dispose());
        }
        const data = await output.logits.getData();
        for (let i = 0; i < seq.length; i++)
          logits.push(
            Array.from(
              data.slice(
                i * this.maxChoices,
                (i + 1) * this.maxChoices,
              ) as Float32Array,
            ),
          );
        output.logits.dispose();
      }
      const probabilities = logits.map((row) => {
        const values = row.map((x) => x / this.manifest.temperature),
          max = Math.max(...values);
        const exp = values.map((x) => Math.exp(x - max)),
          sum = exp.reduce((a, b) => a + b, 0);
        return exp.map((x) => x / sum);
      });
      if (prepared.orders === 2)
        for (let i = 0; i < prepared.fields; i++) {
          const size = prepared.sizes[i],
            reverse = probabilities[i + prepared.fields]
              .slice(0, size)
              .reverse();
          probabilities[i] = probabilities[i].map((p, j) =>
            j < size ? (p + reverse[j]) / 2 : 0,
          );
        }
      return { logits, probabilities: probabilities.slice(0, prepared.fields) };
    } finally {
      Object.values(cache).forEach((t) => t.dispose());
      this.busy = false;
    }
  }

  async decide(
    document: string,
    schema: Schema,
    fast = false,
    includeSchema?: boolean,
  ) {
    includeSchema ??= this.modelName === "decision-lab-v2";
    const compiled = compileSchema(schema);
    const schemaText = includeSchema
      ? "\n\nOutput schema:\n" + JSON.stringify(schema)
      : "";
    let ids = this.tokenizer.encode(document, { add_special_tokens: false });
    const originalLength = ids.length;
    const overhead = Math.max(
      ...compiled.fields.flatMap((f) =>
        [f.options, [...f.options].reverse()].map(
          (options) => this.chat("", f.question, options).length,
        ),
      ),
    );
    const budget = this.manifest.max_length - overhead - 4;
    let context = "";
    while (true) {
      context =
        this.tokenizer.decode(ids, { skip_special_tokens: true }) + schemaText;
      const length = this.tokenizer.encode(context, {
        add_special_tokens: false,
      }).length;
      if (length <= budget) break;
      if (!ids.length)
        throw Error(
          `Schema/questions exceed the ${this.manifest.max_length}-token budget.`,
        );
      ids = ids.slice(
        0,
        Math.max(0, ids.length - Math.max(1, length - budget)),
      );
    }
    const rows = compiled.fields.map((f) => ({
      context,
      question: f.question,
      options: f.options,
    }));
    const prepared = this.prepare(rows, fast ? 1 : 2),
      result = await this.scorePrepared(prepared);
    const value: Record<string, Value> = {},
      fields: Record<string, unknown> = {};
    compiled.fields.forEach((f, i) => {
      const p = result.probabilities[i].slice(0, f.values.length),
        index = p.indexOf(Math.max(...p));
      value[f.name] = f.values[index];
      fields[f.name] = {
        confidence: p[index],
        probabilities: f.values.map((v, j) => ({
          value: v,
          probability: p[j],
        })),
      };
    });
    if (!compiled.validate(value))
      throw Error("Output schema validation failed.");
    return {
      value,
      fields,
      document_truncated: ids.length < originalLength,
      prefix_tokens: prepared.prefix_length,
    };
  }

  async dispose() {
    if (this.disposed) return;
    if (this.busy)
      throw Error("Wait for inference to finish before disposing the engine.");
    this.disposed = true;
    await this.session.release();
  }

  async systemOne(input: unknown, fast = false) {
    const { request, rows } = compileJevRequest(input, this.maxChoices);
    const sequences: number[][] = [],
      sizes: number[] = [];
    const orders = fast ? 1 : 2;
    for (let reverse = 0; reverse < orders; reverse++)
      for (const row of rows) {
        const ids = this.chat(
          row.context,
          row.question,
          reverse ? [...row.options].reverse() : row.options,
        );
        if (ids.length > this.manifest.max_length)
          throw new DecisionError(
            "INVALID_REQUEST",
            `Request exceeds local ${this.manifest.max_length}-token limit. Shorten state/questions; no silent truncation.`,
          );
        sequences.push(ids);
        sizes.push(row.options.length);
      }
    let prefix = 0,
      limit = Math.min(...sequences.map((s) => s.length)) - 1;
    while (
      prefix < limit &&
      sequences.every((s) => s[prefix] === sequences[0][prefix])
    )
      prefix++;
    const prepared: Prepared = {
      sequences,
      sizes,
      fields: rows.length,
      orders,
      prefix_length: prefix,
    };
    const result = await this.scorePrepared(prepared);
    return buildJevResponse(
      request,
      result.probabilities,
      prefix + sequences.reduce((total, s) => total + s.length - prefix, 0),
      this.modelName,
    );
  }
}
