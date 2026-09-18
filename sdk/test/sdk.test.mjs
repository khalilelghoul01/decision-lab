import { test } from "node:test";
import assert from "node:assert/strict";
import {
  loadModel,
  choice,
  noul,
  score,
  DecisionError,
} from "../dist/index.js";
import { DecisionModel } from "../dist/client.js";

const request = () => ({
  state: "The acting was terrible.",
  questions: { acting: noul("Was the acting good?") },
});
function fakeRuntime(
  run = async () => ({
    model: "decision-lab-v3",
    answers: { acting: { type: "noul", noul: 0.02 } },
    usage: { input_tokens: 25, output_tokens: 0 },
  }),
) {
  let released = 0;
  return {
    modelName: "decision-lab-v3",
    maxChoices: 8,
    metadata: { model_name: "decision-lab-v3" },
    systemOne: run,
    async dispose() {
      released++;
    },
    get released() {
      return released;
    },
  };
}

test("SDK import is safe without WebGPU; loading fails clearly", async () => {
  await assert.rejects(
    loadModel(),
    (error) =>
      error instanceof DecisionError &&
      error.code === "UNSUPPORTED_ENVIRONMENT",
  );
});
test("question helpers retain the Jev contract and eight choices", async () => {
  const r = fakeRuntime(async (request) => {
    assert.equal(Object.keys(request.questions.topic.criteria).length, 8);
    return {
      model: "decision-lab-v3",
      answers: {},
      usage: { input_tokens: 0, output_tokens: 0 },
    };
  });
  const model = new DecisionModel(r);
  await model.systemOne({
    state: "x",
    questions: {
      topic: choice(
        "Topic?",
        Object.fromEntries("ABCDEFGH".split("").map((k) => [k, null])),
      ),
      rating: score("Rating?", ["bad", "good"]),
      yes: noul("Yes?", { true: "supported", false: "unsupported" }),
    },
  });
  await model.dispose();
  assert.equal(r.released, 1);
});
test("requests run serially, retain their mode, and snapshot caller data", async () => {
  let finish;
  const blocked = new Promise((resolve) => {
    finish = resolve;
  });
  const seen = [];
  let active = 0;
  const runtime = fakeRuntime(async (req, fast) => {
    assert.equal(++active, 1);
    seen.push([req.state, fast]);
    if (seen.length === 1) await blocked;
    active--;
    return {
      model: "decision-lab-v3",
      answers: {},
      usage: { input_tokens: 0, output_tokens: 0 },
    };
  });
  const model = new DecisionModel(runtime);
  const input = request();
  const first = model.systemOne(input, { mode: "fast" });
  const second = model.systemOne(input);
  input.state = "mutated";
  await Promise.resolve();
  assert.equal(seen.length, 1);
  finish();
  await Promise.all([first, second]);
  assert.deepEqual(seen, [
    ["The acting was terrible.", true],
    ["The acting was terrible.", false],
  ]);
  await model.dispose();
});
test("invalid requests never reach inference", async () => {
  let calls = 0;
  const model = new DecisionModel(
    fakeRuntime(async () => {
      calls++;
      throw Error("must not run");
    }),
  );
  for (const input of [
    { ...request(), model: "decision-lab-v2" },
    { state: "x", questions: {} },
    { state: "x", questions: { x: choice("x", { only: null }) } },
  ])
    await assert.rejects(
      model.systemOne(input),
      (e) => e.code === "INVALID_REQUEST",
    );
  await assert.rejects(
    model.systemOne(request(), { mode: "typo" }),
    (e) => e.code === "INVALID_REQUEST",
  );
  assert.equal(calls, 0);
  await model.dispose();
});
test("failed inference does not poison subsequent requests", async () => {
  let n = 0;
  const model = new DecisionModel(
    fakeRuntime(async () => {
      if (++n === 1) throw Error("GPU failure");
      return {
        model: "decision-lab-v3",
        answers: {},
        usage: { input_tokens: 0, output_tokens: 0 },
      };
    }),
  );
  await assert.rejects(
    model.systemOne(request()),
    (e) => e.code === "INFERENCE_FAILED",
  );
  await model.systemOne(request());
  await model.dispose();
});
test("cancellation skips queued GPU work; disposal drains accepted work once", async () => {
  let finish;
  const blocked = new Promise((resolve) => {
    finish = resolve;
  });
  let calls = 0;
  const runtime = fakeRuntime(async () => {
    calls++;
    await blocked;
    return {
      model: "decision-lab-v3",
      answers: {},
      usage: { input_tokens: 0, output_tokens: 0 },
    };
  });
  const model = new DecisionModel(runtime);
  const first = model.systemOne(request());
  const abort = new AbortController();
  const queued = model.systemOne(request(), { signal: abort.signal });
  const rejected = assert.rejects(queued, (e) => e.name === "AbortError");
  abort.abort();
  const release = model.dispose();
  assert.equal(model.status, "disposing");
  assert.strictEqual(release, model.dispose());
  await assert.rejects(
    model.systemOne(request()),
    (e) => e.code === "DISPOSED",
  );
  assert.equal(runtime.released, 0);
  finish();
  await first;
  await rejected;
  await release;
  assert.equal(calls, 1);
  assert.equal(runtime.released, 1);
  assert.equal(model.status, "disposed");
});
test("active cancellation discards the result without overlapping GPU calls", async () => {
  const abort = new AbortController();
  const model = new DecisionModel(
    fakeRuntime(async () => {
      abort.abort();
      return {
        model: "decision-lab-v3",
        answers: {},
        usage: { input_tokens: 0, output_tokens: 0 },
      };
    }),
  );
  await assert.rejects(
    model.systemOne(request(), { signal: abort.signal }),
    (e) => e.name === "AbortError",
  );
  await model.dispose();
});
