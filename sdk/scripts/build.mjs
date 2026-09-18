import { readFile, writeFile, mkdir, access, rm } from "node:fs/promises";
import { execFileSync } from "node:child_process";
import { build } from "esbuild";

// The Python API and SDK share the canonical JSON contracts in the project.
try {
  await access("../python/decision_lab/schemas/jev-request.schema.json");
  const request = await readFile(
    "../python/decision_lab/schemas/jev-request.schema.json",
    "utf8",
  );
  const response = await readFile(
    "../python/decision_lab/schemas/jev-response.schema.json",
    "utf8",
  );
  await writeFile(
    "src/schemas.ts",
    `// Generated from ../python/decision_lab/schemas by scripts/build.mjs.\nexport const requestSchema = ${request.trim()};\nexport const responseSchema = ${response.trim()};\n`,
  );
} catch (error) {
  if (error.code !== "ENOENT") throw error;
  await access("src/schemas.ts");
}
await rm("dist", { recursive: true, force: true });
await mkdir("dist", { recursive: true });
execFileSync(process.execPath, ["node_modules/typescript/bin/tsc"], {
  stdio: "inherit",
});
await build({
  entryPoints: [
    "src/index.ts",
    "src/engine.ts",
    "src/assets.ts",
    "src/client.ts",
  ],
  outdir: "dist",
  bundle: true,
  splitting: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  packages: "external",
  sourcemap: true,
});
