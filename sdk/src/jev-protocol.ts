import { Ajv2020 } from "ajv/dist/2020.js";
import { requestSchema, responseSchema } from "./schemas.js";

export type Description = string | Record<string, unknown> | unknown[];
export type JevQuestion =
  | {
      type: "choice";
      instructions: Description;
      criteria: Record<string, Description | null>;
    }
  | { type: "score"; instructions: Description; criteria: Description[] }
  | {
      type: "noul";
      instructions: Description;
      criteria?: { true: string; false: string };
    };
export interface JevRequest {
  model: "decision-lab-v2" | "decision-lab-v3" | "jev-latest";
  state: Description;
  questions: Record<string, JevQuestion>;
}
export type JevAnswer =
  | {
      type: "choice";
      choice: string;
      probabilities: Record<string, number>;
      confidence: number;
    }
  | {
      type: "score";
      score: number;
      legend: Record<string, Description>;
      probabilities: Record<string, number>;
      confidence: number;
    }
  | { type: "noul"; noul: number };
export interface JevResponse {
  model: "decision-lab-v2" | "decision-lab-v3";
  answers: Record<string, JevAnswer>;
  usage: { input_tokens: number; output_tokens: number };
}
const ajv = new Ajv2020({ strict: true });
const validateRequest = ajv.compile(requestSchema),
  validateResponse = ajv.compile(responseSchema);
const text = (value: Description) =>
  typeof value === "string" ? value : JSON.stringify(value);

export function compileJevRequest(input: unknown, maxChoices = 4) {
  if (!validateRequest(input))
    throw Error(
      "Invalid local Jev request (2–8 choices/levels): " +
        ajv.errorsText(validateRequest.errors),
    );
  const request = input as unknown as JevRequest,
    context = text(request.state);
  const rows = Object.values(request.questions).map((question) => {
    let options: string[];
    if (question.type === "choice")
      options = Object.entries(question.criteria).map(([key, value]) =>
        value === null ? key : `${key}: ${text(value)}`,
      );
    else if (question.type === "score") options = question.criteria.map(text);
    else
      options = question.criteria
        ? [`No: ${question.criteria.false}`, `Yes: ${question.criteria.true}`]
        : ["No", "Yes"];
    if (options.length > maxChoices)
      throw Error(`The loaded model supports at most ${maxChoices} choices.`);
    return { context, question: text(question.instructions), options };
  });
  return { request, rows };
}

export function buildJevResponse(
  request: JevRequest,
  probabilities: number[][],
  inputTokens: number,
  modelName: JevResponse["model"] = "decision-lab-v2",
): JevResponse {
  if (Object.keys(request.questions).length !== probabilities.length)
    throw Error("Answer count mismatch.");
  const answers = Object.fromEntries(
    Object.entries(request.questions).map(([key, question], i) => {
      const count =
        question.type === "noul" ? 2 : Object.keys(question.criteria).length;
      let p = probabilities[i].slice(0, count);
      const sum = p.reduce((a, b) => a + b, 0);
      if (
        p.length !== count ||
        !sum ||
        p.some((x) => !Number.isFinite(x) || x < 0)
      )
        throw Error("Invalid model probability distribution.");
      p = p.map((x) => x / sum);
      if (question.type === "noul")
        return [key, { type: "noul", noul: p[1] } as JevAnswer];
      const confidence = Math.max(
        0,
        Math.min(
          1,
          1 +
            p.reduce((sum, x) => sum + (x > 0 ? x * Math.log(x) : 0), 0) /
              Math.log(count),
        ),
      );
      const keys =
        question.type === "choice"
          ? Object.keys(question.criteria)
          : p.map((_, i) => String(i));
      const distribution = Object.fromEntries(keys.map((k, i) => [k, p[i]]));
      if (question.type === "choice")
        return [
          key,
          {
            type: "choice",
            choice: keys[p.indexOf(Math.max(...p))],
            probabilities: distribution,
            confidence,
          } as JevAnswer,
        ];
      return [
        key,
        {
          type: "score",
          score: p.reduce((sum, x, i) => sum + i * x, 0),
          legend: Object.fromEntries(
            question.criteria.map((x, i) => [String(i), x]),
          ),
          probabilities: distribution,
          confidence,
        } as JevAnswer,
      ];
    }),
  );
  const response: JevResponse = {
    model: modelName,
    answers,
    usage: { input_tokens: inputTokens, output_tokens: 0 },
  };
  if (!validateResponse(response))
    throw Error(
      "Response contract failure: " + ajv.errorsText(validateResponse.errors),
    );
  return response;
}
