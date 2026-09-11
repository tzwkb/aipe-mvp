/** Fill the audited LOM MTPE result into D:E:F without changing workbook layout. */

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";


const EXPECTED_SOURCE_SHA256 =
  "f6dd829b6dc2a3b736004d9a4cbf68cd0b4326f7ef3fdef54766a635b47e6489";
const EXPECTED_ROWS = [
  ...Array.from({ length: 6 }, (_, index) => index + 14),
  ...Array.from({ length: 6 }, (_, index) => index + 22),
  30,
  ...Array.from({ length: 34 }, (_, index) => index + 33),
];
const WRITE_BLOCKS = [
  [14, 19],
  [22, 27],
  [30, 30],
  [33, 66],
];


function parseArgs(argv) {
  const values = {};
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index];
    const value = argv[index + 1];
    if (!key?.startsWith("--") || value === undefined) {
      throw new Error(`Invalid argument pair at ${key ?? "<end>"}`);
    }
    values[key.slice(2)] = value;
  }
  for (const key of [
    "source",
    "units",
    "non-dialogue",
    "dialogue",
    "output",
  ]) {
    if (!values[key]) throw new Error(`Missing --${key}`);
  }
  return values;
}


async function sha256(filePath) {
  const bytes = await fs.readFile(filePath);
  return crypto.createHash("sha256").update(bytes).digest("hex");
}


async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}


function countToken(value, token) {
  if (!token) return 0;
  return value.split(token).length - 1;
}


function validatePreservedTokens(sourceUnit, translation) {
  for (const tag of sourceUnit.source_cell?.tags ?? []) {
    if (countToken(translation, tag) !== countToken(sourceUnit.source, tag)) {
      throw new Error(`Tag mismatch at row ${sourceUnit.row}: ${tag}`);
    }
  }
  for (const variable of sourceUnit.source_cell?.variables ?? []) {
    if (
      countToken(translation, variable) !== countToken(sourceUnit.source, variable)
    ) {
      throw new Error(`Variable mismatch at row ${sourceUnit.row}: ${variable}`);
    }
  }
  const sourceBreaks = sourceUnit.source_cell?.line_breaks ?? {};
  const expectedActual = sourceBreaks.actual_count ?? 0;
  const expectedLiteral = sourceBreaks.literal_escape_count ?? 0;
  if (countToken(translation, "\n") !== expectedActual) {
    throw new Error(`Actual line-break mismatch at row ${sourceUnit.row}`);
  }
  if (countToken(translation, "\\n") !== expectedLiteral) {
    throw new Error(`Literal \\n mismatch at row ${sourceUnit.row}`);
  }
}


function normalizeTranslationUnit(unit) {
  const translation = unit.translation;
  const note = unit.translation_note ?? "";
  const query = unit.query ?? "";
  if (!Number.isInteger(unit.row)) throw new Error("Translation row must be integer");
  if (typeof translation !== "string" || !translation.trim()) {
    throw new Error(`Missing translation at row ${unit.row}`);
  }
  if (typeof note !== "string" || typeof query !== "string") {
    throw new Error(`Note/query must be strings at row ${unit.row}`);
  }
  return { ...unit, translation, translation_note: note, query };
}


async function main() {
  const args = parseArgs(process.argv.slice(2));
  const sourcePath = path.resolve(args.source);
  const unitsPath = path.resolve(args.units);
  const outputPath = path.resolve(args.output);
  const modulePath = process.env.CODEX_ARTIFACT_TOOL_MODULE;
  if (!modulePath) {
    throw new Error("CODEX_ARTIFACT_TOOL_MODULE is required");
  }
  const artifactTool = await import(pathToFileURL(modulePath).href);
  const { FileBlob, SpreadsheetFile } = artifactTool;

  const sourceShaBefore = await sha256(sourcePath);
  if (sourceShaBefore !== EXPECTED_SOURCE_SHA256) {
    throw new Error(`Source workbook SHA mismatch: ${sourceShaBefore}`);
  }
  const unitsSha = await sha256(unitsPath);
  const sourceUnits = await readJson(unitsPath);
  if (sourceUnits.status !== "ready_for_local_mtpe") {
    throw new Error(`Source units are not ready: ${sourceUnits.status}`);
  }

  const translationParts = await Promise.all([
    readJson(path.resolve(args["non-dialogue"])),
    readJson(path.resolve(args.dialogue)),
  ]);
  const translations = translationParts.flatMap((part) => {
    if (part.status !== "final") {
      throw new Error(`Translation part is not final: ${part.status}`);
    }
    if (part.source_units_sha256 !== unitsSha) {
      throw new Error("Translation part source_units_sha256 mismatch");
    }
    if (!Array.isArray(part.units)) throw new Error("Translation units must be an array");
    return part.units.map(normalizeTranslationUnit);
  });
  const byRow = new Map();
  for (const unit of translations) {
    if (byRow.has(unit.row)) throw new Error(`Duplicate translation row ${unit.row}`);
    byRow.set(unit.row, unit);
  }
  if (
    byRow.size !== EXPECTED_ROWS.length ||
    EXPECTED_ROWS.some((row) => !byRow.has(row))
  ) {
    throw new Error("Translation rows do not match the 47-unit contract");
  }

  const sourceByRow = new Map(sourceUnits.units.map((unit) => [unit.row, unit]));
  for (const row of EXPECTED_ROWS) {
    const sourceUnit = sourceByRow.get(row);
    const translationUnit = byRow.get(row);
    if (!sourceUnit) throw new Error(`Missing source unit at row ${row}`);
    if (
      translationUnit.source !== undefined &&
      translationUnit.source !== sourceUnit.source
    ) {
      throw new Error(`Translation source snapshot mismatch at row ${row}`);
    }
    validatePreservedTokens(sourceUnit, translationUnit.translation);
  }

  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(sourcePath));
  const sheet = workbook.worksheets.getItem("试译");
  for (const row of EXPECTED_ROWS) {
    const sourceUnit = sourceByRow.get(row);
    const currentSource = sheet.getRange(sourceUnit.source_cell.address).values[0][0];
    if (currentSource !== sourceUnit.source) {
      throw new Error(`Workbook source changed at ${sourceUnit.source_cell.address}`);
    }
    const currentOutputs = sheet.getRange(`D${row}:F${row}`).values[0];
    if (currentOutputs.some((value) => value !== null && value !== "")) {
      throw new Error(`Output cells are not blank at row ${row}`);
    }
  }

  for (const [startRow, endRow] of WRITE_BLOCKS) {
    const matrix = [];
    for (let row = startRow; row <= endRow; row += 1) {
      const unit = byRow.get(row);
      matrix.push([unit.translation, unit.translation_note, unit.query]);
    }
    sheet.getRange(`D${startRow}:F${endRow}`).values = matrix;
  }

  const readableQueryCells = [];
  for (const row of EXPECTED_ROWS) {
    const unit = byRow.get(row);
    if (!unit.query.trim()) continue;
    const coordinate = `F${row}`;
    const queryCell = sheet.getRange(coordinate);
    queryCell.format.font.color = "#000000";
    queryCell.format.wrapText = true;
    readableQueryCells.push(coordinate);
  }

  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  const sourceShaAfter = await sha256(sourcePath);
  if (sourceShaAfter !== sourceShaBefore) {
    throw new Error("Source workbook changed during export");
  }
  const outputSha = await sha256(outputPath);
  console.log(
    JSON.stringify(
      {
        source: sourcePath,
        source_sha256: sourceShaBefore,
        source_units_sha256: unitsSha,
        output: outputPath,
        output_sha256: outputSha,
        units_written: byRow.size,
        cells_written: byRow.size * 3,
        write_ranges: WRITE_BLOCKS.map(
          ([startRow, endRow]) => `D${startRow}:F${endRow}`,
        ),
        intentional_query_readability_styles: readableQueryCells,
      },
      null,
      2,
    ),
  );
}


await main();
