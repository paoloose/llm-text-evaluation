import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

type FinalOutput = {
  id: number;
  task: string;
  question: string;
  options: string[];
  answer: number;
  rationale: string | null;
};

const N = 500;

const run = async () => {
  const inputPath = join(__dirname, 'dataset.json');
  const outputPath = join(__dirname, 'processed_dataset.json');

  const content = await readFile(inputPath, 'utf8');
  const dataset: FinalOutput[] = JSON.parse(content);

  const byTask: Record<string, FinalOutput[]> = {};
  for (const item of dataset) {
    if (!byTask[item.task]) byTask[item.task] = [];
    byTask[item.task].push(item);
  }

  const picked: FinalOutput[] = [];
  const counts: Record<string, { requested: number; actual: number }> = {};

  for (const [task, items] of Object.entries(byTask)) {
    const selected = items.slice(0, N);
    picked.push(...selected);
    counts[task] = { requested: N, actual: selected.length };
  }

  await writeFile(outputPath, JSON.stringify(picked, null, 2));

  console.log(`Picked first N=${N} items per task type`);
  console.log(`Total written: ${picked.length}\n`);
  for (const [task, info] of Object.entries(counts)) {
    console.log(`- ${task}: ${info.actual}/${info.requested}`);
  }
};

run().catch(console.error);
