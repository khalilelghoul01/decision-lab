import {
  loadModel,
  clearModelCache,
  choice,
  noul,
  score,
  type DecisionModel,
  type LoadProgress,
  type JevRequest,
} from "decision-lab-sdk";
import wasmModuleUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import wasmBinaryUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import rawExample from "../../examples/jev-review-request.json";
import "./style.css";
import "./sdk.css";
const example = { ...rawExample, model: "decision-lab-v3" } as JevRequest;

document.querySelector("#app")!.innerHTML = `
<header><a class="mark" href="/">D/</a><span>DECISION LAB SDK</span><span class="local">LOCAL INFERENCE</span></header>
<div class="intro"><div class="eyebrow">MODEL LOADER · TYPESCRIPT</div><h1>Load once.<br>Ask many questions.</h1><p>A small SDK for typed decisions in your browser.<br>Your text stays on this device.</p></div>
<section class="load sdk-loader"><div class="loader-body"><label for="model-url">MODEL DIRECTORY</label><input id="model-url" value="/model-v3-q4" spellcheck="false"><strong id="sdk-status" role="status" aria-live="polite">Ready to load · 296 MB of weights</strong><progress id="load-progress" max="1" value="0" aria-label="Current file download"></progress><p id="load-detail">Files are cached when browser storage is available. First inference includes shader compilation.</p></div><div class="loader-actions"><button id="sdk-load">Load model</button><button id="sdk-cancel" disabled>Cancel loading</button><button id="sdk-unload" disabled>Unload model</button><button id="sdk-clear">Clear cached files</button></div></section>
<section class="workspace"><div><label for="sdk-request">REQUEST</label><textarea id="sdk-request" rows="24" spellcheck="false"></textarea><div class="actions"><select id="sdk-mode" aria-label="Scoring mode"><option value="accurate">Two option orders</option><option value="fast">One option order · faster</option></select><button id="sdk-run" disabled>Run decision →</button></div></div><div class="result"><div class="result-head"><label>RESPONSE</label><span id="sdk-latency">—</span></div><pre id="sdk-response">Load the model to begin.</pre><p class="note">Noul is P(yes), from 0 to 1. This research model can be confidently wrong. The current limit is 1,024 tokens per question, including context and choices.</p></div></section>
<section class="validation"><div><h2>SDK checks</h2><p>Verify real predictions, concurrent requests, cancellation, limits, disposal and cached reload. This unloads and reloads the current model.</p></div><button id="sdk-check" disabled>Run SDK checks</button><pre id="sdk-checks">Not run yet.</pre></section>
<footer><a href="/">Full model playground →</a> · Browser WebGPU · 2–8 choices · Up to 32 questions</footer>`;
const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id)! as T;
$<HTMLTextAreaElement>("sdk-request").value = JSON.stringify(example, null, 2);
let model: DecisionModel | undefined, controller: AbortController | undefined;
let cachedFiles = new Set<string>();
const wasmPaths = {
  mjs: new URL(wasmModuleUrl, location.href).href,
  wasm: new URL(wasmBinaryUrl, location.href).href,
};
const modelUrl = () => $<HTMLInputElement>("model-url").value;
function controls(loading = false, testing = false) {
  $<HTMLButtonElement>("sdk-load").disabled = loading || testing || !!model;
  $<HTMLButtonElement>("sdk-cancel").disabled = !loading || testing;
  $<HTMLButtonElement>("sdk-unload").disabled = !model || loading || testing;
  for (const id of ["sdk-run", "sdk-check"])
    $<HTMLButtonElement>(id).disabled = !model || loading || testing;
  $<HTMLButtonElement>("sdk-clear").disabled = loading || testing;
  $<HTMLInputElement>("model-url").disabled = loading || testing || !!model;
}
function progress(event: LoadProgress) {
  $("sdk-status").textContent = event.message;
  const bar = $<HTMLProgressElement>("load-progress");
  if (event.progress !== undefined) bar.value = event.progress;
  else if (event.stage === "initializing") bar.removeAttribute("value");
  if (event.stage === "ready") bar.value = 1;
  if (event.cached && event.file) cachedFiles.add(event.file);
  if (event.file)
    $("load-detail").textContent =
      `${event.cached ? "Cached · " : ""}${((event.loaded ?? 0) / 1e6).toFixed(1)}${event.total ? ` / ${(event.total / 1e6).toFixed(1)}` : ""} MB · ${event.file}`;
}
async function load() {
  controller = new AbortController();
  cachedFiles = new Set();
  model = await loadModel({
    modelUrl: modelUrl(),
    wasmPaths,
    signal: controller.signal,
    onProgress: progress,
  });
  $("load-detail").textContent =
    `${model.metadata.model_name} · ${model.metadata.max_length} tokens · ${cachedFiles.size} cached files reused`;
}
$("sdk-load").onclick = async () => {
  controls(true);
  try {
    await load();
  } catch (e) {
    $("sdk-status").textContent =
      e instanceof Error && e.name === "AbortError"
        ? "Loading cancelled"
        : String(e);
  } finally {
    controller = undefined;
    controls();
  }
};
$("sdk-cancel").onclick = () => controller?.abort();
$("sdk-unload").onclick = async () => {
  controls(false, true);
  try {
    await model?.dispose();
    model = undefined;
    $("sdk-status").textContent = "Model unloaded · cached files retained";
  } catch (e) {
    $("sdk-status").textContent = String(e);
  } finally {
    controls();
  }
};
$("sdk-clear").onclick = async () => {
  controls(false, true);
  try {
    const n = await clearModelCache(modelUrl());
    $("load-detail").textContent =
      `Cleared ${n} cached model version(s). A loaded model stays usable.`;
  } catch (e) {
    $("load-detail").textContent = String(e);
  } finally {
    controls();
  }
};
$("sdk-run").onclick = async () => {
  controls(false, true);
  try {
    const start = performance.now();
    const result = await model!.systemOne(
      JSON.parse($<HTMLTextAreaElement>("sdk-request").value),
      { mode: $<HTMLSelectElement>("sdk-mode").value as "fast" | "accurate" },
    );
    $("sdk-latency").textContent =
      `${(performance.now() - start).toFixed(1)} ms`;
    $("sdk-response").textContent = JSON.stringify(result, null, 2);
  } catch (e) {
    $("sdk-response").textContent = String(e);
  } finally {
    controls();
  }
};
$("sdk-check").onclick = async () => {
  controls(false, true);
  const result: Record<string, unknown> = {};
  const show = (stage: string) => {
    $("sdk-checks").textContent =
      stage + "\n" + JSON.stringify(result, null, 2);
  };
  const assert = (ok: boolean, message: string) => {
    if (!ok) throw Error(message);
  };
  try {
    show("Checking negative and positive acting…");
    const negative = await model!.systemOne(example);
    const answer = negative.answers.good_acting;
    assert(
      answer.type === "noul" && answer.noul < 0.1,
      "Negative acting regression",
    );
    result.negative_acting_p_yes = answer.type === "noul" ? answer.noul : null;
    const positive = await model!.systemOne({
      state: "The plot was awful, but the acting was excellent.",
      questions: {
        acting: noul("Was the acting good?"),
        topic: choice("What is this text about?", {
          Politics: null,
          Sports: null,
          Movies: null,
          Business: null,
          Technology: null,
          Health: null,
          Travel: null,
          Food: null,
        }),
        quality: score("How good was the acting?", ["Bad", "Mixed", "Good"]),
      },
    });
    assert(
      positive.answers.acting.noul > 0.9 &&
        positive.answers.topic.choice === "Movies",
      "Positive/eight-choice regression",
    );
    result.positive_acting_p_yes = positive.answers.acting.noul;
    result.eight_choice_topic = positive.answers.topic.choice;
    result.score = positive.answers.quality.score;
    show("Checking request limits and queued cancellation…");
    try {
      await model!.systemOne({
        ...example,
        state: "long context ".repeat(1600),
      });
      throw Error("Oversized request was accepted");
    } catch (e) {
      assert(
        e instanceof Error && e.message.includes("1024-token"),
        "Expected input budget rejection",
      );
    }
    result.oversized_rejected = true;
    const abort = new AbortController();
    const first = model!.systemOne(example, { mode: "fast" });
    const cancelled = model!.systemOne(example, { signal: abort.signal });
    const caught = cancelled.then(
      () => false,
      (e) => e instanceof Error && e.name === "AbortError",
    );
    abort.abort();
    const last = model!.systemOne(example);
    await first;
    assert(await caught, "Queued cancellation failed");
    await last;
    result.queue_and_cancellation = true;
    show("Checking GPU cleanup and cached reload…");
    const old = model!;
    const pending = old.systemOne(example);
    const released = old.dispose();
    await pending;
    await released;
    assert(old.status === "disposed", "Dispose did not finish");
    try {
      await old.systemOne(example);
      throw Error("Disposed model accepted work");
    } catch (e) {
      assert(
        e instanceof Error && e.message.includes("disposed"),
        "Expected disposed rejection",
      );
    }
    result.dispose_drains_queue = true;
    model = undefined;
    await load();
    result.reload_cached_files = [...cachedFiles];
    assert(
      cachedFiles.has("decision.weights"),
      "Weights were not reused from cache",
    );
    const reloaded = await model!.systemOne(example);
    assert(
      JSON.stringify(reloaded.answers) === JSON.stringify(negative.answers),
      "Reload changed answers",
    );
    result.reload_same_answers = true;
    show("Checking cancellation while reading model files…");
    const cancelLoad = new AbortController();
    let cancelledLoad = false;
    try {
      await loadModel({
        modelUrl: modelUrl(),
        wasmPaths,
        signal: cancelLoad.signal,
        onProgress: (e) => {
          if (e.file === "decision.weights") cancelLoad.abort();
        },
      });
    } catch (e) {
      cancelledLoad = e instanceof Error && e.name === "AbortError";
    }
    assert(cancelledLoad, "Loading cancellation failed");
    result.loading_cancellation = true;
    const samples: number[] = [];
    for (let i = 0; i < 5; i++) {
      const start = performance.now();
      await model!.systemOne(example);
      samples.push(performance.now() - start);
    }
    result.warm_four_field_median_ms = [...samples].sort((a, b) => a - b)[2];
    result.warm_samples_ms = samples;
    result.passed = true;
    show("PASS");
  } catch (e) {
    result.passed = false;
    result.error = String(e);
    show("FAILED");
  } finally {
    controller = undefined;
    controls();
  }
};
