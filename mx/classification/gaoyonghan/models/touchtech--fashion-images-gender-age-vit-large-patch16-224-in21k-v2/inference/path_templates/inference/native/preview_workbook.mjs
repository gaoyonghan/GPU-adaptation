import fs from 'node:fs/promises';
import {FileBlob,SpreadsheetFile} from '@oai/artifact-tool';
const source='__MODEL_ROOT__/resources/runtime/assets/workspace/9.11 100个模型结果汇总_五维度.xlsx';
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(source));
console.log((await wb.inspect({kind:'region',sheetId:'100模型结果',range:'A82:F100',maxChars:5000})).ndjson);
const preview=await wb.render({sheetName:'100模型结果',autoCrop:'all',scale:0.5,format:'png'});
await fs.writeFile('/home/ubuntu/Documents/Codex/2026-09-27/codex-threads-01a0de0e-9d72-7fc0-b485/work/source_preview.png',new Uint8Array(await preview.arrayBuffer()));
console.log('preview saved');
