import {
  BrowserDecisionEngine,
  compileSchema,
  type Prepared,
  type Schema,
  type Value,
} from "./engine";
import type { JevRequest } from "./jev-protocol";
import "./style.css";
import jevExample from "../../examples/jev-review-request.json";

const modelProfiles: Record<
  string,
  { path: string; label: string; id: "decision-lab-v2" | "decision-lab-v3" }
> = {
  v2: { path: "/model", label: "V2 · 230M · FP16", id: "decision-lab-v2" },
  v3: { path: "/model-v3", label: "V3 · 350M · FP16", id: "decision-lab-v3" },
  "v2-q4": {
    path: "/model-v2-q4",
    label: "V2 · 230M · INT4 candidate",
    id: "decision-lab-v2",
  },
  "v3-q4": {
    path: "/model-v3-q4",
    label: "V3 · 350M · INT4",
    id: "decision-lab-v3",
  },
};
const profile =
  modelProfiles[new URLSearchParams(location.search).get("model") ?? "v3-q4"] ??
  modelProfiles["v3-q4"];

const schema: Schema = {
  type: "object",
  properties: {
    sentiment: {
      type: "string",
      enum: ["Negative", "Positive"],
      description: "What is the sentiment of this movie review?",
    },
    good_acting: { type: "boolean", description: "Was the acting good?" },
    topic: {
      type: "string",
      enum: ["Politics", "Sports", "Movies", "Business"],
      description: "What is this text about?",
    },
  },
  required: ["sentiment", "good_acting", "topic"],
  additionalProperties: false,
};
document.querySelector("#app")!.innerHTML = `
<header><span class="mark">D/</span><span>DECISION LAB</span><span class="local">LOCAL INFERENCE</span></header>
<div class="intro"><div class="eyebrow">${profile.label} · TYPED DECISIONS</div><h1>Read once.<br>Decide together.</h1><p>One shared context. Independent field scores.<br>JSON assembled from allowed values, directly in your browser.</p></div>
<section class="load"><div><strong id="status">Model is not loaded</strong><p>The selected model loads once. No prompt is sent to an AI server.</p></div><button id="load">Load model</button></section>
<section class="workspace"><div><label for="jev-request">JEV-COMPATIBLE REQUEST</label><textarea id="jev-request" rows="26" spellcheck="false"></textarea><div class="actions"><select id="api-mode" aria-label="Jev scoring mode"><option value="accurate">Two option orders</option><option value="fast">One option order · experimental</option></select><button id="system-one" disabled>Evaluate request →</button></div></div><div class="result"><div class="result-head"><label>API RESPONSE</label><span id="api-latency">—</span></div><pre id="api-response">Load the model to begin.</pre><div id="api-summary"></div><p class="note">Same JSON shapes for Choice, Score and Noul. This is the local model, not Jev. Noul is P(yes); it is not a confidence score or a guaranteed fact.</p></div></section>
<details class="legacy"><summary>Legacy JSON Schema interface</summary>
<section class="workspace"><div><label for="document">DOCUMENT</label><textarea id="document" rows="7">The film was charming and beautifully acted. The story was engaging and the direction excellent. I would happily recommend it.</textarea><label for="schema">OUTPUT SCHEMA</label><textarea id="schema" rows="16" spellcheck="false"></textarea><div class="actions"><select id="mode" aria-label="Scoring mode"><option value="accurate">Two option orders</option><option value="fast">One option order · faster</option></select><button id="decide" disabled>Score fields →</button></div></div><div class="result"><div class="result-head"><label>RESULT</label><span id="latency">—</span></div><pre id="value">Load the model to begin.</pre><div id="confidence"></div><p class="note">Valid JSON does not guarantee correct decisions. Confidence is a model probability; new schemas have not been calibrated separately.</p></div></section>
</details>
<section class="validation"><div><h2>Runtime checks</h2><p>Compare browser tokenization and scores with the exported Python fixture, then time cached and full-context execution.</p></div><button id="validate" disabled>Run checks</button><pre id="checks">Not run yet.</pre></section>
<section class="validation"><div><h2>Jev use-case evaluation</h2><p>84 original labeled scenarios plus 12 stress cases. This tests the local model on Jev-style workflows; it does not call Jev. Cases include inert tool-command text.</p></div><button id="jev-evaluate" disabled>Test Jev use cases</button><pre id="jev-results">Not run yet.</pre><details><summary>Case results and mistakes</summary><pre id="jev-details"></pre></details></section>
<footer><span id="model-details">${profile.label} · Jev-shaped typed decisions · Local WebGPU inference</span> · <a href="/sdk.html" style="color:inherit">Model loader & SDK →</a></footer>`;
const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id)! as T;
$<HTMLTextAreaElement>("schema").value = JSON.stringify(schema, null, 2);
$<HTMLTextAreaElement>("jev-request").value = JSON.stringify(
  { ...jevExample, model: profile.id },
  null,
  2,
);
let engine: BrowserDecisionEngine;
const status = (message: string) => {
  $("status").textContent = message;
};
function busy(value: boolean) {
  for (const id of ["decide", "validate", "jev-evaluate", "system-one"])
    $<HTMLButtonElement>(id).disabled = value;
}
$("load").onclick = async () => {
  $<HTMLButtonElement>("load").disabled = true;
  try {
    engine = await BrowserDecisionEngine.load(profile.path, status);
    busy(false);
    $("load").textContent = "Loaded";
    $("model-details").textContent =
      `${profile.label} · Up to ${engine.maxChoices} choices · ${engine.maxLength}-token input budget`;
  } catch (error) {
    status("Load failed: " + String(error));
    $<HTMLButtonElement>("load").disabled = false;
  }
};
$("system-one").onclick = async () => {
  busy(true);
  $("api-response").textContent = "Scoring…";
  $("api-summary").replaceChildren();
  try {
    const request = JSON.parse($<HTMLTextAreaElement>("jev-request").value);
    const start = performance.now();
    const response = await engine.systemOne(
      request,
      $<HTMLSelectElement>("api-mode").value === "fast",
    );
    $("api-latency").textContent =
      `${(performance.now() - start).toFixed(1)} ms`;
    $("api-response").textContent = JSON.stringify(response, null, 2);
    for (const [key, answer] of Object.entries(response.answers)) {
      const item = document.createElement("div");
      item.className = "confidence-row";
      const name = document.createElement("span");
      name.textContent = key;
      const value = document.createElement("strong");
      value.textContent =
        answer.type === "noul"
          ? `P(yes) ${(answer.noul * 100).toFixed(1)}%`
          : answer.type === "choice"
            ? answer.choice
            : answer.score.toFixed(2);
      item.append(name, value);
      $("api-summary").append(item);
    }
  } catch (error) {
    $("api-response").textContent = String(error);
  } finally {
    busy(false);
  }
};
$("decide").onclick = async () => {
  busy(true);
  $("value").textContent = "Scoring…";
  try {
    const start = performance.now();
    const result = await engine.decide(
      $<HTMLTextAreaElement>("document").value,
      JSON.parse($<HTMLTextAreaElement>("schema").value),
      $<HTMLSelectElement>("mode").value === "fast",
    );
    $("latency").textContent = `${(performance.now() - start).toFixed(1)} ms`;
    $("value").textContent = JSON.stringify(result.value, null, 2);
    $("confidence").replaceChildren();
    for (const [name, details] of Object.entries(result.fields)) {
      const item = document.createElement("div");
      item.className = "confidence-row";
      const label = document.createElement("span");
      label.textContent = name;
      const score = document.createElement("strong");
      score.textContent = `${((details as { confidence: number }).confidence * 100).toFixed(1)}%`;
      item.append(label, score);
      $("confidence").append(item);
    }
    const foot = document.createElement("p");
    foot.className = "note";
    foot.textContent = `${result.prefix_tokens} shared prefix tokens${result.document_truncated ? " · Document truncated to fit the input budget" : ""}`;
    $("confidence").append(foot);
  } catch (error) {
    $("value").textContent = String(error);
  } finally {
    busy(false);
  }
};
const maxDelta = (a: number[][], b: number[][]) =>
  Math.max(...a.flatMap((row, i) => row.map((v, j) => Math.abs(v - b[i][j]))));
const top = (p: number[]) => p.indexOf(Math.max(...p));
const median = (samples: number[]) =>
  [...samples].sort((a, b) => a - b)[Math.floor(samples.length / 2)];
$("validate").onclick = async () => {
  busy(true);
  const results: Record<string, unknown> = { backend: engine.backend };
  const show = (stage: string) => {
    $("checks").textContent = stage + "\n" + JSON.stringify(results, null, 2);
  };
  try {
    show("Loading reference fixture…");
    const fixture = await (await fetch(`${profile.path}/fixture.json`)).json();
    const prepared = engine.prepare(fixture.rows, fixture.orders);
    results.tokenizer_exact =
      JSON.stringify(prepared.sequences) === JSON.stringify(fixture.sequences);
    if (!results.tokenizer_exact)
      throw Error("Python/JavaScript tokenization differs.");
    show("Running the first full-context pass (shader compilation)…");
    const full = await engine.scorePrepared(prepared, false);
    show("Running shared-cache inference…");
    const cached = await engine.scorePrepared(prepared, true);
    results.max_logit_delta_vs_python = maxDelta(full.logits, fixture.logits);
    results.max_probability_delta_vs_python = maxDelta(
      cached.probabilities,
      fixture.probabilities,
    );
    results.cache_probability_delta = maxDelta(
      full.probabilities,
      cached.probabilities,
    );
    results.answers_match_python = cached.probabilities.every(
      (p, i) => top(p) === top(fixture.probabilities[i]),
    );
    if (
      !results.answers_match_python ||
      (results.max_probability_delta_vs_python as number) > 0.03
    )
      throw Error("Browser reference parity failed.");
    results.benchmarks = [];
    for (const long of [false, true]) {
      const rows = fixture.rows.map((r: { context: string }) => ({
        ...r,
        context: long ? (r.context + " ").repeat(35) : r.context,
      }));
      const workload = engine.prepare(rows, 2);
      const timing: Record<string, unknown> = {
        context: long ? "long" : "short",
        fields: fixture.rows.length,
        orders: 2,
        prefix_tokens: workload.prefix_length,
        max_input_tokens: Math.max(...workload.sequences.map((s) => s.length)),
      };
      for (const shared of [false, true]) {
        show(
          `Warming ${long ? "long" : "short"} ${shared ? "cached" : "full"} inference…`,
        );
        await engine.scorePrepared(workload, shared);
        const samples = [];
        for (let i = 0; i < 5; i++) {
          const start = performance.now();
          await engine.scorePrepared(workload, shared);
          samples.push(performance.now() - start);
          show(
            `Benchmark ${long ? "long" : "short"} · ${shared ? "cached" : "full"} · ${i + 1}/5`,
          );
        }
        timing[shared ? "shared_median_ms" : "full_median_ms"] =
          median(samples);
      }
      timing.speedup =
        (timing.full_median_ms as number) / (timing.shared_median_ms as number);
      (results.benchmarks as unknown[]).push(timing);
    }
    results.passed = true;
    show("PASS");
  } catch (error) {
    results.error = String(error);
    results.passed = false;
    show("FAILED");
  } finally {
    busy(false);
  }
};

interface EvalCase {
  id: string;
  group: string;
  split: string;
  context: string;
  schema: Schema;
  request?: JevRequest;
  expected: Record<string, unknown>;
  native_value: Record<string, unknown>;
  native_fields: Record<
    string,
    { probabilities: { value: unknown; probability: number }[] }
  >;
}
interface EvalResult {
  id: string;
  group: string;
  split: string;
  exact: boolean;
  correct_fields: number;
  fields: number;
  valid: boolean;
  truncated: boolean;
  latency_ms: number;
  native_agreement: number;
  high_confidence_errors: number;
  max_probability_delta: number;
  mistakes: unknown[];
  value?: Record<string, unknown>;
  error?: string;
}
function summarizeEval(rows: EvalResult[]) {
  const sum = (key: keyof EvalResult) =>
    rows.reduce((total, row) => total + Number(row[key]), 0);
  const fields = sum("fields"),
    correct = sum("correct_fields");
  return {
    cases: rows.length,
    fields,
    correct_fields: correct,
    field_accuracy: fields ? correct / fields : 0,
    exact_cases: sum("exact"),
    valid_json_cases: sum("valid"),
    rejected_cases: rows.filter((r) => r.error).length,
    truncated: sum("truncated"),
    native_agreement_fields: sum("native_agreement"),
    high_confidence_errors: sum("high_confidence_errors"),
    median_ms: rows.length ? median(rows.map((r) => r.latency_ms)) : null,
    max_probability_delta: rows.length
      ? Math.max(...rows.map((r) => r.max_probability_delta))
      : null,
  };
}
async function evaluateCase(c: EvalCase) {
  if (!c.request) return engine.decide(c.context, c.schema);
  const response = await engine.systemOne(c.request),
    compiled = compileSchema(c.schema);
  const value: Record<string, Value> = {},
    fields: Record<
      string,
      {
        confidence: number;
        probabilities: { value: Value; probability: number }[];
      }
    > = {};
  for (const f of compiled.fields) {
    const answer = response.answers[f.name];
    const p =
      answer.type === "noul"
        ? [1 - answer.noul, answer.noul]
        : f.values.map((v) => answer.probabilities[String(v)]);
    const index = p.indexOf(Math.max(...p));
    value[f.name] = f.values[index];
    fields[f.name] = {
      confidence: p[index],
      probabilities: f.values.map((v, j) => ({ value: v, probability: p[j] })),
    };
  }
  if (!compiled.validate(value))
    throw Error("Assembled result failed schema validation.");
  return { value, fields, document_truncated: false };
}
$("jev-evaluate").onclick = async () => {
  busy(true);
  $("jev-details").textContent = "";
  const results: EvalResult[] = [];
  try {
    const response = await fetch(
      profile.id === "decision-lab-v3"
        ? "/jev-suite-v3.json"
        : "/jev-suite.json",
    );
    if (!response.ok)
      throw Error(
        "Generate the frozen suite and run the Python evaluator first.",
      );
    const suite = (await response.json()) as {
      cases: EvalCase[];
      suite_sha256: string;
      configuration?: string;
      reference?: string;
    };
    const first = suite.cases[0];
    $("jev-results").textContent = "Warming the model for the evaluation…";
    await evaluateCase(first);
    for (const [index, c] of suite.cases.entries()) {
      $("jev-results").textContent =
        `Running ${index + 1}/${suite.cases.length} · ${c.id}`;
      const start = performance.now();
      const row: EvalResult = {
        id: c.id,
        group: c.group,
        split: c.split,
        fields: Object.keys(c.expected).length,
        correct_fields: 0,
        exact: false,
        valid: false,
        truncated: false,
        latency_ms: 0,
        native_agreement: 0,
        high_confidence_errors: 0,
        max_probability_delta: 0,
        mistakes: [],
      };
      try {
        const result = await evaluateCase(c);
        row.latency_ms = performance.now() - start;
        row.value = result.value;
        row.valid = true;
        row.truncated = result.document_truncated;
        for (const [name, expected] of Object.entries(c.expected)) {
          const actual = result.value[name],
            details = result.fields[name] as {
              confidence: number;
              probabilities: { value: unknown; probability: number }[];
            };
          const correct = actual === expected;
          if (correct) row.correct_fields++;
          else {
            row.mistakes.push({
              field: name,
              expected,
              predicted: actual,
              confidence: details.confidence,
            });
            if (details.confidence >= 0.9) row.high_confidence_errors++;
          }
          if (actual === c.native_value?.[name]) row.native_agreement++;
          for (const probability of details.probabilities) {
            const native = c.native_fields?.[name]?.probabilities.find(
              (p) => p.value === probability.value,
            );
            if (native)
              row.max_probability_delta = Math.max(
                row.max_probability_delta,
                Math.abs(native.probability - probability.probability),
              );
          }
        }
        row.exact = row.correct_fields === row.fields;
      } catch (error) {
        row.latency_ms = performance.now() - start;
        row.error = String(error);
      }
      results.push(row);
    }
    const core = results.filter((r) => r.split === "core");
    const summary = {
      backend: engine.backend,
      model: engine.modelName,
      reference: suite.reference ?? "V2 native",
      suite_sha256: suite.suite_sha256,
      configuration: suite.configuration ?? "schema_accurate",
      core: summarizeEval(core),
      groups: Object.fromEntries(
        [...new Set(core.map((r) => r.group))].map((group) => [
          group,
          summarizeEval(core.filter((r) => r.group === group)),
        ]),
      ),
      stress: Object.fromEntries(
        ["injection", "long_tail"].map((split) => [
          split,
          summarizeEval(results.filter((r) => r.split === split)),
        ]),
      ),
    };
    $("jev-results").textContent =
      "EVALUATION COMPLETE\n" + JSON.stringify(summary, null, 2);
    $("jev-details").textContent = JSON.stringify(results, null, 2);
  } catch (error) {
    $("jev-results").textContent = "Evaluation failed: " + String(error);
  } finally {
    busy(false);
  }
};
