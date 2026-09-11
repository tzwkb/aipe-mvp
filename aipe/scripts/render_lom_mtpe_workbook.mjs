/** Render fixed LOM workbook ranges for visual delivery QA. */

import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";


const RENDERS = [
  ["instructions", "试译说明", "C1:E54"],
  ["functional", "试译", "A10:F19"],
  ["item_skill", "试译", "A20:F27"],
  ["rules", "试译", "A28:F30"],
  ["dialogue_1", "试译", "A31:F49"],
  ["dialogue_2", "试译", "A50:F66"],
];


async function main() {
  const [inputArg, outputArg] = process.argv.slice(2);
  if (!inputArg || !outputArg) {
    throw new Error("Usage: node render_lom_mtpe_workbook.mjs INPUT.xlsx OUTPUT_DIR");
  }
  const modulePath = process.env.CODEX_ARTIFACT_TOOL_MODULE;
  if (!modulePath) throw new Error("CODEX_ARTIFACT_TOOL_MODULE is required");
  const { FileBlob, SpreadsheetFile } = await import(pathToFileURL(modulePath).href);
  const inputPath = path.resolve(inputArg);
  const outputDir = path.resolve(outputArg);
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  await fs.mkdir(outputDir, { recursive: true });

  const outputs = [];
  for (const [name, sheetName, range] of RENDERS) {
    const rendered = await workbook.render({
      sheetName,
      range,
      scale: 1.5,
      format: "png",
    });
    const outputPath = path.join(outputDir, `${name}.png`);
    await fs.writeFile(outputPath, new Uint8Array(await rendered.arrayBuffer()));
    outputs.push({ name, sheetName, range, output: outputPath });
  }

  const formulaErrors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 300 },
    summary: "LOM MTPE formula error scan",
  });
  console.log(JSON.stringify({ input: inputPath, outputs, formulaErrors: formulaErrors.ndjson }, null, 2));
}


await main();
