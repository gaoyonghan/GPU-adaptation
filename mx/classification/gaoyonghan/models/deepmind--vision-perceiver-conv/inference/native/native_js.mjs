// Publisher's Transformers.js pipeline, with an outer sequential image loop.
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
import { pipeline, env } from '@huggingface/transformers';

env.allowRemoteModels = false;
env.allowLocalModels = true;
const request = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const start = performance.now();
const classifier = await pipeline('image-classification', request.model_path, {
  device: 'cpu', dtype: 'q8', local_files_only: true,
});
const loaded = (performance.now() - start) / 1000;
const labels = Object.fromEntries(Object.entries(request.labels).map(([k,v]) => [v,Number(k)]));
const classes = Object.keys(labels).length;
const batches = request.completed_batches;
const completed = batches.reduce((sum,b) => sum+b.images,0);
const pending = request.rows.slice(completed);
const warmStart = performance.now();
let baseline;
for (let i=0; pending.length && i<request.warmup; i++) baseline = await classifier(pending[0].path, {top_k: classes});
const warm = (performance.now() - warmStart) / 1000;
let forward = 0, calls = 0;
const original = classifier.model.forward;
if (typeof original === 'function') {
  classifier.model.forward = async function(...args) {
    const t = performance.now();
    const output = await original.apply(this, args);
    forward += (performance.now() - t) / 1000;
    calls++;
    return output;
  };
}
const output = fs.openSync(`${request.output}/batches.jsonl`, 'a');
try {
  for (const [i,row] of pending.entries()) {
    const previousForward = forward, previousCalls = calls;
    const t = performance.now();
    const result = await classifier(row.path, {top_k: classes});
    const elapsed = (performance.now() - t) / 1000;
    if (i===0 && (result[0].label!==baseline[0].label || result.some((r,j) => Math.abs(r.score-baseline[j].score)>1e-6))) {
      throw new Error('Timing instrumentation changed publisher pipeline predictions');
    }
    const probs = Object.fromEntries(result.map(r => [labels[r.label],r.score]));
    const prediction = {id:row.id,target:row.target,prediction:labels[result[0].label],
      top5:result.slice(0,5).map(r=>labels[r.label]),confidence:result[0].score,
      true_probability:probs[row.target],positive_probability:classes===2 ? probs[1] : null};
    if (result.length!==classes || result.some(r => labels[r.label]===undefined || !Number.isFinite(r.score))) throw new Error('Invalid native output');
    const batch = {records:[prediction],images:1,segment:request.segment,
      forward_seconds:calls-previousCalls===1 ? forward-previousForward : null,inference_e2e_seconds:elapsed};
    fs.writeSync(output,JSON.stringify(batch)+'\n');
    batches.push(batch);
    if ((completed+i+1)%100===0 || i+1===pending.length) console.log(`173: ${completed+i+1}/${request.rows.length}`);
  }
} finally {
  fs.closeSync(output);
  if (typeof original==='function') classifier.model.forward=original;
}
const measured = batches.every(b => b.forward_seconds!==null);
forward = batches.reduce((sum,b) => sum+(b.forward_seconds ?? 0),0);
const total = batches.reduce((sum,b) => sum+b.inference_e2e_seconds,0);
fs.writeFileSync(`${request.output}/js_timing.json`,JSON.stringify({
  device:'cpu',dtype:'published ONNX q8',model_load_seconds:loaded,warmup_seconds:warm,warmup_batches:request.warmup,
  timed_images:request.rows.length,forward_seconds:measured ? forward : null,
  actual_batch_sizes:{1:request.rows.length},resume_segments:new Set(batches.map(b=>b.segment)).size,
  forward_amortized_ms_per_image:measured ? forward*1000/request.rows.length : null,
  forward_unavailable_reason:measured ? null : 'Runtime did not expose exactly one forward call per image',
  inference_e2e_seconds:total,e2e_images_per_second:request.rows.length/total,
  forward_method:'CPU monotonic clock around the unchanged async model.forward',
  instrumentation_predictions_unchanged_this_segment:pending.length ? true : null,
  comparison_note:'CPU ONNX q8 results must not be mixed with GPU FP32 latency rankings',
},null,2)+'\n');
await classifier.dispose();
