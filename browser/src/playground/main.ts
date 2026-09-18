import { Ajv2020 } from 'ajv/dist/2020.js';
import { clearModelCache, type JevRequest, type JevQuestion, type JevResponse, type LoadProgress } from 'decision-lab-sdk';
import { BrowserDecisionEngine, compileSchema, type Schema, type Row } from '../engine';
import requestSchema from '../../../python/decision_lab/schemas/jev-request.schema.json';
import { examples, defaultSchema, groupNames } from './examples';
import { escape, pct, median, icon, downloadJson, highlightTypeScript } from './ui';
import { template } from './template';
import { loadClarity } from './analytics';
import './playground.css';

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const value = (id: string) => $<HTMLInputElement>(id).value;
const pretty = (x: unknown) => JSON.stringify(x, null, 2);
const asText = (x: unknown) => typeof x === 'string' ? x : JSON.stringify(x);
const clone = <T>(x: T): T => structuredClone(x);
const base = '/model-v3-q4';
const ajv = new Ajv2020({ strict: true });
const validateRequest = ajv.compile(requestSchema);
type Field = { name: string; question: JevQuestion };
type Run = { id: number; at: string; request: unknown; response: unknown; decisions: Decision[]; ms: number; orders: number; tokens: number; prefix: number; revision: number };
type Decision = { name: string; type: string; value: string; distribution: [string, number][]; note: string };
let fields: Field[] = [], editor = 'fields', output = 'visual', revision = 0;
let engine: Awaited<ReturnType<typeof BrowserDecisionEngine.load>> | undefined;
let busy = false, loading = false, supported = false, stopSuite = false;
let controller: AbortController | undefined, current: Run | undefined, runs: Run[] = [], nextRun = 1;
let toastTimer: ReturnType<typeof setTimeout>;
document.querySelector('#app')!.innerHTML = template;
loadClarity();

function toast(message: string) { $('toast').textContent = message; $('toast').hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => $('toast').hidden = true, 4000); }
function error(e?: unknown) { $('global-error').hidden = !e; $('global-error').textContent = e instanceof Error ? e.message : String(e ?? ''); }
function syncControls() {
  for (const id of ['run-decision', 'benchmark', 'parity', 'run-suite']) $<HTMLButtonElement>(id).disabled = !engine || busy || loading;
  $<HTMLButtonElement>('load-model').disabled = !supported || busy || loading;
  $('load-model').hidden = !!engine || loading;
  $('cancel-load').hidden = !loading;
  $('unload-model').hidden = !engine;
  $<HTMLButtonElement>('unload-model').disabled = busy;
  $<HTMLButtonElement>('clear-cache').disabled = busy || loading;
  document.querySelectorAll<HTMLInputElement>('.input-panel input,.input-panel textarea,.input-panel select,.input-panel button,#scenarios button,#scoring-mode,#suite-group').forEach(el => el.disabled = busy);
  $<HTMLButtonElement>('run-decision').disabled = !engine || busy || loading;
  $<HTMLButtonElement>('add-field').disabled = busy || fields.length >= 32;
  $('model-state').textContent = loading ? 'LOADING' : busy ? 'RUNNING' : engine ? 'READY' : 'NOT LOADED';
  $('model-state').classList.toggle('ready', !!engine);
}
async function work(fn: () => Promise<void>) {
  if (busy || !engine) return;
  busy = true; error(); syncControls();
  try { await fn(); } catch (e) { error(e); } finally { busy = false; syncControls(); }
}
function changed() { revision++; if (current) $('result-stale').hidden = false; $('context-size').textContent = `${value('context').length.toLocaleString()} characters`; }
function typedRequest(): JevRequest {
  const input: unknown = editor === 'json' ? JSON.parse(value('request-json')) : { model: 'decision-lab-v3', state: value('context'), questions: Object.fromEntries(fields.map(f => [f.name, f.question])) };
  if (!validateRequest(input)) throw Error('Invalid request: ' + ajv.errorsText(validateRequest.errors));
  if (editor !== 'json' && new Set(fields.map(f => f.name)).size !== fields.length) throw Error('Field names must be unique.');
  if (editor !== 'json' && fields.some(f => !f.name.trim())) throw Error('Every field needs a name.');
  const request = input as unknown as JevRequest;
  if (request.model !== 'decision-lab-v3' && request.model !== 'jev-latest') throw Error('This playground loads decision-lab-v3. Use that model name or the jev-latest local alias.');
  return request;
}
function rowsFor(request: JevRequest): Row[] {
  return Object.values(request.questions).map(q => ({context:asText(request.state),question:asText(q.instructions),options: q.type === 'noul' ? (q.criteria ? ['No: '+q.criteria.false,'Yes: '+q.criteria.true] : ['No','Yes']) : q.type === 'score' ? q.criteria.map(asText) : Object.entries(q.criteria).map(([k,v])=>v === null?k:`${k}: ${asText(v)}`)}));
}
function loadRequest(request: JevRequest) {
  fields = Object.entries(clone(request.questions)).map(([name,question]) => ({name,question}));
  $<HTMLTextAreaElement>('context').value = asText(request.state);
  $<HTMLTextAreaElement>('request-json').value = pretty(request);
  renderFields(); changed();
}
function renderFields() {
  $('field-count').textContent = `${fields.length} / 32`;
  $('fields').innerHTML = fields.map((f,i) => {
    const q=f.question, options=q.type === 'noul' ? '' : q.type === 'score' ? q.criteria.map(asText).join('\n') : Object.keys(q.criteria).join('\n');
    return `<article class="field-card" data-field="${i}"><div class="field-heading"><span class="field-index">${String(i+1).padStart(2,'0')}</span><input class="field-name" data-key="name" aria-label="Field ${i+1} name" value="${escape(f.name)}"><select data-key="type" aria-label="${escape(f.name)} type">${['choice','noul','score'].map(type=>`<option value="${type}" ${q.type===type?'selected':''}>${type==='noul'?'Noul · yes/no':type==='score'?'Score · ordered':'Choice · category'}</option>`).join('')}</select><button class="icon-button remove-field" aria-label="Remove ${escape(f.name)}" ${fields.length===1?'disabled':''}>${icon('close')}</button></div><label class="field-label">Question<textarea data-key="instructions" rows="2" spellcheck="false">${escape(asText(q.instructions))}</textarea></label>${q.type==='noul'?`<p class="field-hint">Returns P(yes) from 0 to 1.${q.criteria?' Custom criteria are preserved; edit them in Request JSON.':''}</p>`:`<label class="field-label">${q.type==='score'?'Levels, lowest to highest':'Allowed labels'} <span>one per line · 2–8</span><textarea data-key="options" rows="${Math.min(q.type==='score'?q.criteria.length:Object.keys(q.criteria).length,4)}" class="options-input" spellcheck="false">${escape(options)}</textarea></label>${q.type==='choice'&&Object.values(q.criteria).some(x=>x!==null)?'<p class="field-hint">Label descriptions are preserved; edit them in Request JSON.</p>':''}`}</article>`;
  }).join('');
  syncControls();
}
$('fields').addEventListener('input', event => {
  const el=event.target as HTMLInputElement, card=el.closest<HTMLElement>('[data-field]'); if(!card) return;
  const f=fields[Number(card.dataset.field)], key=el.dataset.key;
  if(key==='name') f.name=el.value;
  if(key==='instructions') f.question.instructions=el.value;
  if(key==='options') {
    const options=el.value.split('\n').map(s=>s.trim()).filter(Boolean);
    if(f.question.type==='choice') { const previous=f.question.criteria; f.question.criteria=Object.fromEntries(options.map(x=>[x,previous[x]??null])); }
    if(f.question.type==='score') f.question.criteria=options;
  }
  changed();
});
$('fields').addEventListener('change', event => {
  const el=event.target as HTMLSelectElement, card=el.closest<HTMLElement>('[data-field]');
  if(!card || el.dataset.key!=='type') return;
  const f=fields[Number(card.dataset.field)];
  f.question = el.value==='noul' ? {type:'noul',instructions:f.question.instructions} : el.value==='score' ? {type:'score',instructions:f.question.instructions,criteria:['Low','Medium','High']} : {type:'choice',instructions:f.question.instructions,criteria:{Option_A:null,Option_B:null}};
  changed(); renderFields();
});
$('fields').addEventListener('click', event => {
  const button=(event.target as HTMLElement).closest('.remove-field'), card=button?.closest<HTMLElement>('[data-field]');
  if(card && fields.length>1) { fields.splice(Number(card.dataset.field),1); changed(); renderFields(); }
});
$('add-field').onclick = () => {
  if(fields.length>=32) return;
  let i=fields.length+1; while(fields.some(f=>f.name===`field_${i}`)) i++;
  fields.push({name:`field_${i}`,question:{type:'choice',instructions:'Which option best describes this context?',criteria:{Option_A:null,Option_B:null}}});
  changed(); renderFields();
  $('fields').lastElementChild?.querySelector('input')?.focus();
};
function setEditor(mode: string) {
  error();
  try {
    if(editor==='json' && mode==='fields') loadRequest(typedRequest());
    if(editor==='fields' && mode==='json') $<HTMLTextAreaElement>('request-json').value=pretty(typedRequest());
    if(editor==='json' && mode==='schema') $<HTMLTextAreaElement>('context').value=asText(typedRequest().state);
    editor=mode;
    document.querySelectorAll<HTMLElement>('[data-editor]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.editor===mode)));
    for(const name of ['builder','json','schema']) $(`${name}-editor`).hidden = name !== (mode==='fields'?'builder':mode);
    $('context-block').hidden=mode==='json';
    changed();
  } catch(e) { error(e); }
}
document.querySelectorAll<HTMLElement>('[data-editor]').forEach(el=>el.onclick=()=>setEditor(el.dataset.editor!));
for(const id of ['context','request-json','schema-json','scoring-mode']) $(id).addEventListener('input',changed);
$<HTMLTextAreaElement>('schema-json').value=pretty(defaultSchema);
$('scenarios').innerHTML=examples.map(e=>`<button class="scenario" data-example="${e.id}"><span>${escape(e.tag)}</span><strong>${escape(e.title)}</strong></button>`).join('');
function selectExample(id: string) {
  const example=examples.find(e=>e.id===id); if(!example) return;
  loadRequest(example.request);
  if(editor==='schema') setEditor('fields');
  document.querySelectorAll<HTMLElement>('[data-example]').forEach(el=>el.classList.toggle('active',el.dataset.example===id));
  $('scenario-description').textContent=example.description;
}
$('scenarios').onclick=event=>{const el=(event.target as HTMLElement).closest<HTMLElement>('[data-example]');if(el)selectExample(el.dataset.example!);};

const pages:Record<string,[string,string]>={playground:['The decision playground','Give it context. Define the possible answers. See what the model thinks.'],testbench:['Trust the measurements','Inspect the speed, numerical agreement, and mistakes on your own device.'],architecture:['Inside the experiment','A small language model, a shared context cache, and a constrained answer space.'],developers:['Take it apart. Make it better.','Use the SDK, reproduce the training, or bring a case that breaks the model.']};
function route() {
  const name=location.hash.slice(1).split('/')[0], page=pages[name]?name:'playground';
  document.querySelectorAll<HTMLElement>('[data-view]').forEach(el=>el.hidden=el.dataset.view!==page);
  document.querySelectorAll<HTMLElement>('[data-page]').forEach(el=>{el.classList.toggle('active',el.dataset.page===page);if(el.dataset.page===page)el.setAttribute('aria-current','page');else el.removeAttribute('aria-current');});
  $('page-title').textContent=pages[page][0]; $('page-description').textContent=pages[page][1]; $('breadcrumb').textContent=page==='testbench'?'TEST BENCH':page.toUpperCase();
  document.title=`${page==='playground'?'Playground':pages[page][0]} · Decision Lab`;
}
window.addEventListener('hashchange',route);

function progress(event: LoadProgress) {
  $('load-status').textContent=event.message;
  const p=$<HTMLProgressElement>('load-progress');
  if(event.progress!==undefined) p.value=event.progress; else p.removeAttribute('value');
  $('load-detail').textContent=[event.file?.split('/').pop(),event.cached?'cached':event.loaded!==undefined?`${(event.loaded/1e6).toFixed(1)}${event.total?' / '+(event.total/1e6).toFixed(1):''} MB`:''].filter(Boolean).join(' · ');
}
$('load-model').onclick=async()=>{
  if(loading||engine) return;
  error(); loading=true; controller=new AbortController(); syncControls(); $('progress-wrap').hidden=false;
  const start=performance.now();
  try {
    engine=await BrowserDecisionEngine.load(base,message=>$('load-status').textContent=message,'webgpu',{signal:controller.signal,cache:'auto',onProgress:progress});
    $('load-status').textContent=`Ready on WebGPU · loaded in ${((performance.now()-start)/1000).toFixed(1)}s · first inference includes shader warm-up.`;
    toast('Model ready. Your next decision runs on this device.');
  } catch(e) {
    if(controller.signal.aborted) $('load-status').textContent='Load cancelled. Completed downloads may remain cached.';
    else { error(e); $('load-status').textContent='Could not load the model. Check WebGPU support and try again.'; }
  } finally { loading=false; controller=undefined; $('progress-wrap').hidden=true;syncControls(); }
};
$('cancel-load').onclick=()=>{controller?.abort();$('load-status').textContent='Cancelling…';};
$('unload-model').onclick=async()=>{
  if(busy||!engine) return;
  busy=true;syncControls();
  try {await engine.dispose();engine=undefined;$('load-status').textContent='Model unloaded. Downloaded files remain in the browser cache.';}catch(e){error(e);}finally{busy=false;syncControls();}
};
$('clear-cache').onclick=async()=>{
  if(busy||loading)return;busy=true;syncControls();
  try {const count=await clearModelCache(base);toast(`Cleared ${count} model cache entr${count===1?'y':'ies'}. ${engine?'The loaded model is still ready.':''}`);}catch(e){error(e);}finally{busy=false;syncControls();}
};

function decisionsFor(response: JevResponse): Decision[] {
  return Object.entries(response.answers).map(([name,a])=>{
    if(a.type==='noul')return{name,type:'Noul',value:String(a.noul>=0.5),distribution:[['No',1-a.noul],['Yes',a.noul]],note:`P(yes) = ${a.noul.toFixed(4)} · boolean threshold 0.5`};
    if(a.type==='choice')return{name,type:'Choice',value:a.choice,distribution:Object.entries(a.probabilities),note:`Distribution concentration ${pct(a.confidence)} · not a correctness estimate`};
    return{name,type:'Score',value:a.score.toFixed(3),distribution:Object.entries(a.probabilities).map(([key,p])=>[`${key} · ${asText(a.legend[key])}`,p]),note:`Expected level index · range 0–${Object.keys(a.legend).length-1} · concentration ${pct(a.confidence)}`};
  });
}
function renderResult(run: Run) {
  current=run; $('empty-output').hidden=true;$('result-stale').hidden=run.revision===revision;
  $('latency-badge').innerHTML=`${icon('clock')} ${run.ms.toFixed(1)} ms`;
  $('result-visual').innerHTML=run.decisions.map(d=>{
    const max=Math.max(...d.distribution.map(x=>x[1]));
    return `<article class="decision-result"><div class="decision-heading"><div><span class="type-tag">${d.type}</span><h3>${escape(d.name)}</h3></div><strong class="decision-value">${escape(d.value)}</strong></div><div class="probabilities">${d.distribution.map(([label,p])=>`<div class="prob-row ${p===max?'winner':''}"><div><span>${escape(label)}</span><strong>${pct(p)}</strong></div><div class="prob-track"><span style="width:${p*100}%"></span></div></div>`).join('')}</div><p class="distribution-note">${escape(d.note)}</p></article>`;
  }).join('');
  $('result-raw').textContent=pretty(run.response);
  $('run-metrics').hidden=false;$('run-metrics').innerHTML=`<div><strong>${run.decisions.length}</strong><span>fields</span></div><div><strong>${run.prefix}</strong><span>shared tokens</span></div><div><strong>${run.tokens}</strong><span>processed tokens</span></div><div><strong>${run.orders}</strong><span>option order${run.orders===1?'':'s'}</span></div><p>Request time includes tokenization and inference. First runs may compile shaders. No JSON tokens are generated.</p>`;
  $<HTMLButtonElement>('copy-result').disabled=false;$<HTMLButtonElement>('export-result').disabled=false;
  setOutput(output);renderHistory();
}
function setOutput(mode:string){output=mode;document.querySelectorAll<HTMLElement>('[data-output]').forEach(el=>el.setAttribute('aria-selected',String(el.dataset.output===mode)));$('result-visual').hidden=!current||mode!=='visual';$('result-raw').hidden=!current||mode!=='raw';}
document.querySelectorAll<HTMLElement>('[data-output]').forEach(el=>el.onclick=()=>setOutput(el.dataset.output!));
function renderHistory(){
  $('history-count').textContent=`${runs.length} run${runs.length===1?'':'s'}`;
  $('history').innerHTML=runs.length?runs.map(r=>`<button class="history-row ${current?.id===r.id?'selected':''}" data-run="${r.id}"><span class="history-id">#${String(r.id).padStart(2,'0')}</span><span class="history-context">${escape(asText((r.request as any).state ?? (r.request as any).context))}</span><span>${r.decisions.length} fields</span><span>${r.orders} order${r.orders===1?'':'s'}</span><strong>${r.ms.toFixed(1)} ms</strong><span>${escape(r.at)}</span>${icon('arrow')}</button>`).join(''):'<p class="muted">Your recent runs appear here. History stays in memory and clears when you reload.</p>';
}
$('history').onclick=event=>{const el=(event.target as HTMLElement).closest<HTMLElement>('[data-run]');const run=runs.find(r=>r.id===Number(el?.dataset.run));if(run)renderResult(run);};
$('clear-history').onclick=()=>{runs=[];renderHistory();toast('Session history cleared.');};
$('run-decision').onclick=()=>work(async()=>{
  const fast=value('scoring-mode')==='fast';
  let request: unknown,response: JevResponse, schema:Schema|undefined, compiled:ReturnType<typeof compileSchema>|undefined;
  if(editor==='schema') {
    schema=JSON.parse(value('schema-json'));compiled=compileSchema(schema!);
    request={context:value('context'),schema};
    const questions=Object.fromEntries(compiled.fields.map(f=>[f.name,f.values[0]===false&&f.values[1]===true?{type:'noul' as const,instructions:f.question}:{type:'choice' as const,instructions:f.question,criteria:Object.fromEntries(f.options.map(x=>[x,null]))}]));
    response=undefined as unknown as JevResponse;
    const typed:JevRequest={model:'decision-lab-v3',state:value('context'),questions};
    const start=performance.now();response=await engine!.systemOne(typed,fast); const ms=performance.now()-start;
    const data=Object.fromEntries(compiled.fields.map(f=>{const a=response.answers[f.name];return[f.name,a.type==='noul'?a.noul>=.5:a.type==='choice'?f.values[f.options.indexOf(a.choice)]:a.score];}));
    if(!compiled.validate(data))throw Error('Schema validation failed.');
    saveRun(request,data,decisionsFor(response),ms,fast,response.usage.input_tokens,engine!.prepare(rowsFor(typed),fast?1:2).prefix_length);
  } else {
    const typed=typedRequest();request=typed;const start=performance.now();response=await engine!.systemOne(typed,fast);const ms=performance.now()-start;
    saveRun(request,response,decisionsFor(response),ms,fast,response.usage.input_tokens,engine!.prepare(rowsFor(typed),fast?1:2).prefix_length);
  }
});
function saveRun(request:unknown,response:unknown,decisions:Decision[],ms:number,fast:boolean,tokens:number,prefix:number){const run:Run={id:nextRun++,at:new Date().toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'}),request:clone(request),response:clone(response),decisions,ms,orders:fast?1:2,tokens,prefix,revision};runs=[run,...runs].slice(0,10);renderResult(run);}
$('copy-result').onclick=async()=>{try{if(current){await navigator.clipboard.writeText(pretty(current.response));toast('Response JSON copied.');}}catch(e){error(e);}};
$('export-result').onclick=()=>{if(current)downloadJson(`decision-lab-run-${current.id}.json`,current);};
$('export-request').onclick=()=>{try{downloadJson('decision-lab-request.json',editor==='schema'?{context:value('context'),schema:JSON.parse(value('schema-json'))}:typedRequest());}catch(e){error(e);}};

$('benchmark').onclick=()=>work(async()=>{
  if(editor==='schema')throw Error('Choose Builder or Request JSON before benchmarking. JSON Schema mode has a different output adapter.');
  const request=clone(typedRequest()), result:{mode:string;median_ms:number;min_ms:number;max_ms:number;samples_ms:number[]}[]=[];
  for(const fast of [true,false]){
    const mode=fast?'One order':'Two orders';$('benchmark-result').textContent=`${mode}: warming shaders…`;await engine!.systemOne(request,fast);
    const times:number[]=[];
    for(let i=0;i<5;i++){ $('benchmark-result').textContent=`${mode}: measuring ${i+1}/5…`;const start=performance.now();await engine!.systemOne(request,fast);times.push(performance.now()-start); }
    result.push({mode,median_ms:median(times),min_ms:Math.min(...times),max_ms:Math.max(...times),samples_ms:times});
  }
  $('benchmark-result').innerHTML=result.map(r=>`<div class="bench-row"><span>${r.mode}</span><strong>${r.median_ms.toFixed(1)}<small> ms median</small></strong><span>${r.min_ms.toFixed(1)}–${r.max_ms.toFixed(1)} ms</span></div>`).join('')+`<p class="muted">${Object.keys(request.questions).length} fields · WebGPU · 5 samples per mode · loading excluded</p>`;
});
$('parity').onclick=()=>work(async()=>{
  $('parity-result').textContent='Loading reference and compiling full-context shaders…';
  const res=await fetch(`${base}/fixture.json`);if(!res.ok)throw Error('Reference fixture is unavailable.');const fixture=await res.json();
  const prepared=engine!.prepare(fixture.rows,fixture.orders), exact=pretty(prepared.sequences)===pretty(fixture.sequences);
  if(!exact)throw Error('Reference check failed: tokenizer mismatch.');
  const full=await engine!.scorePrepared(prepared,false);$('parity-result').textContent='Comparing shared-cache inference…';const cached=await engine!.scorePrepared(prepared,true);
  const delta=(a:number[][],b:number[][])=>Math.max(...a.flatMap((row,i)=>row.map((v,j)=>Math.abs(v-b[i][j]))));
  const top=(p:number[])=>p.indexOf(Math.max(...p));
  const answers=cached.probabilities.every((p,i)=>top(p)===top(fixture.probabilities[i]));
  const pythonDelta=delta(cached.probabilities,fixture.probabilities),cacheDelta=delta(full.probabilities,cached.probabilities), passed=exact&&answers&&pythonDelta<=.03&&cacheDelta<=.01;
  $('parity-result').innerHTML=`<div class="check-status ${passed?'pass':'fail'}">${icon(passed?'check':'close')} ${passed?'Reference checks passed':'Reference check mismatch'}</div><dl class="check-list"><dt>Token IDs</dt><dd>Exact match</dd><dt>Top answers vs Python</dt><dd>${answers?'4 / 4 match':'Mismatch'}</dd><dt>Max probability difference vs Python</dt><dd>${pythonDelta.toFixed(6)} (limit 0.03)</dd><dt>Shared-cache vs full-context difference</dt><dd>${cacheDelta.toFixed(6)} (limit 0.01)</dd></dl>`;
});

interface EvalCase {id:string;group:string;split:string;request:JevRequest;expected:Record<string,unknown>}
interface EvalResult {id:string;group:string;split:string;expected:Record<string,unknown>;actual:Record<string,unknown>;correct:number;total:number;exact:boolean;ms:number;error?:string;expected_limit:boolean;response?:JevResponse}
let suite:{cases:EvalCase[];suite_sha256:string}|undefined, evalRows:EvalResult[]=[];
$('suite-group').insertAdjacentHTML('beforeend',Object.entries(groupNames).map(([id,label])=>`<option value="${id}">${label} · 12 cases</option>`).join(''));
function renderEvaluation(){
  const scored=evalRows.filter(r=>!r.expected_limit), correct=scored.reduce((s,r)=>s+r.correct,0), total=scored.reduce((s,r)=>s+r.total,0), exact=scored.filter(r=>r.exact).length,rejected=evalRows.filter(r=>r.error).length;
  $('suite-summary').innerHTML=`<div class="eval-stat"><strong>${total?pct(correct/total):'—'}</strong><span>field accuracy · ${correct}/${total}</span></div><div class="eval-stat"><strong>${scored.length?pct(exact/scored.length):'—'}</strong><span>exact records · ${exact}/${scored.length}</span></div><div class="eval-stat"><strong>${rejected}</strong><span>rejected · ${evalRows.filter(r=>r.expected_limit&&r.error).length} expected length limits</span></div><div class="eval-stat"><strong>${evalRows.length?median(evalRows.map(r=>r.ms)).toFixed(1):'—'}<small> ms</small></strong><span>median case time</span></div>`;
  $('suite-rows').innerHTML=evalRows.map((r,i)=>`<tr><td><strong>${escape(r.id)}</strong><small>${escape(groupNames[r.group]??r.group)}</small></td><td>${r.expected_limit?'Excluded · length stress':`${r.correct} / ${r.total}`}</td><td><span class="table-pill ${r.expected_limit&&r.error?'neutral':r.exact?'pass':'fail'}">${r.error?r.expected_limit?'Length limit':'Error':r.exact?'Exact':'Mismatch'}</span></td><td>${r.ms.toFixed(1)} ms</td><td><button class="text-button" data-inspect="${i}">Details ${icon('arrow')}</button></td></tr>`).join('');
  $<HTMLButtonElement>('export-suite').disabled=!evalRows.length;
}
$('run-suite').onclick=()=>work(async()=>{
  stopSuite=false;evalRows=[];$('suite-details').hidden=true;$('stop-suite').hidden=false;
  try {
    if(!suite){const res=await fetch('/jev-suite-v3.json');if(!res.ok)throw Error('Evaluation suite unavailable.');suite=await res.json();}
    const group=value('suite-group'), cases=suite!.cases.filter(c=>group==='core'?c.split==='core':group==='stress'?c.split!=='core':c.group===group&&c.split==='core');
    $('suite-progress').textContent='Warming the runtime…';await engine!.systemOne(examples[0].request);
    for(const c of cases){
      if(stopSuite)break;
      $('suite-progress').textContent=`Running ${evalRows.length+1} / ${cases.length} · ${c.id}`;
      const start=performance.now(), row:EvalResult={id:c.id,group:c.group,split:c.split,expected:c.expected,actual:{},correct:0,total:Object.keys(c.expected).length,exact:false,ms:0,expected_limit:false};
      try{
        const response=await engine!.systemOne(c.request);row.response=response;
        row.actual=Object.fromEntries(Object.entries(response.answers).map(([k,a])=>[k,a.type==='noul'?a.noul>=.5:a.type==='choice'?a.choice:a.score]));
        row.correct=Object.entries(c.expected).filter(([key,v])=>row.actual[key]===v).length;row.exact=row.correct===row.total;
      }catch(e){row.error=e instanceof Error?e.message:String(e);row.expected_limit=c.split==='long_tail'&&row.error.includes('1024-token limit');}
      row.ms=performance.now()-start;evalRows.push(row);renderEvaluation();
      await new Promise(resolve=>setTimeout(resolve,0));
    }
    $('suite-progress').textContent=`${stopSuite?'Stopped':'Completed'} · ${evalRows.length} / ${cases.length} cases · two option orders · known regression suite. Length-limit stress rejections are excluded from accuracy.`;
  }finally{$('stop-suite').hidden=true;}
});
$('stop-suite').onclick=()=>{stopSuite=true;$('suite-progress').textContent='Stopping after the current GPU request…';};
$('suite-rows').onclick=event=>{const el=(event.target as HTMLElement).closest<HTMLElement>('[data-inspect]');if(!el)return;const row=evalRows[Number(el.dataset.inspect)];$('suite-details').hidden=false;($('suite-details') as HTMLDetailsElement).open=true;$('suite-detail-json').textContent=pretty({...row,request:suite?.cases.find(c=>c.id===row.id)?.request});$('suite-details').scrollIntoView({behavior:'smooth',block:'nearest'});};
$('export-suite').onclick=()=>downloadJson('decision-lab-browser-evaluation.json',{created_at:new Date().toISOString(),backend:'webgpu',model:'decision-lab-v3-int4',configuration:'system_one_two_orders',suite_sha256:suite?.suite_sha256,rows:evalRows});
$('sdk-snippet').innerHTML=highlightTypeScript(`import { loadModel, choice, noul } from 'decision-lab-sdk';\n\nconst model = await loadModel({\n  modelUrl: '/model-v3-q4',\n  onProgress: e => console.log(e.message),\n});\n\nconst result = await model.systemOne({\n  state: 'Bad plot, excellent acting.',\n  questions: {\n    good_acting: noul('Was the acting good?'),\n    topic: choice('What is this about?', {\n      Movies: null, Sports: null,\n    }),\n  },\n}, { mode: 'accurate' });\n\nconsole.log(result.answers);\nawait model.dispose();`);
selectExample('review');route();syncControls();
(async()=>{
  try{const gpu=(navigator as any).gpu,adapter=gpu?await gpu.requestAdapter():null;supported=!!adapter&&adapter.features.has('shader-f16');$('device-status').textContent=supported?'WebGPU available':'WebGPU / FP16 unavailable';$('gpu-dot').classList.toggle('available',supported);if(!supported)$('load-status').textContent='Use a WebGPU browser with FP16 support (for example recent Chrome or Edge). Examples and documentation remain available.';}catch{$('device-status').textContent='GPU check failed';}finally{syncControls();}
})();
