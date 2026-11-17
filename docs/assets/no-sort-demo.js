import {summarizeImpact, queryImpact} from './no-sort-impact.mjs';
const $=id=>document.getElementById(id);
const labels={canonical:'Lens · permutation-invariant',organic:'Positional Lens · shuffled training',unaugmented:'Positional Lens · original-order training',slot_mean:'Global training average',observed_mean:'Average of visible neurons',zero:'Leave it blank'};
const connection=i=>i<10?`Incoming · left digit ${i}`:i<20?`Incoming · right digit ${i-10}`:i===20?'Hidden neuron bias':`Outgoing · answer ${i-21}`;
let data,slot=0,method='canonical',coordinates='normalized',query=0;
const fmt=x=>Number(x).toFixed(5);
function element(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;}
function render(){
 const s=data.slots[slot],m=s.methods[method],actual=coordinates==='normalized'?data.normalized_neurons[slot]:data.original_neurons[s.organic_neuron],pred=coordinates==='normalized'?m.normalized_prediction:m.organic_prediction;
 $('mapping').textContent=`Physical neuron ${slot+1} · original-order reinsertion · scale ${fmt(data.scales[slot])}`;
 [...$('slots').children].forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.slot)===slot)));
 $('shuffle-delta').textContent=m.saved_shuffle_max_weight_delta===undefined?'This baseline has no learned-reader permutation measurement.':`Saved shuffled-input check, this neuron: maximum normalized prediction difference ${m.saved_shuffle_max_weight_delta.toExponential(2)}.`;
 $('mse').textContent=m.normalized_neuron_mse.toFixed(4);
 const correct=answers=>answers.filter((x,i)=>x===data.query_pairs[i][0]+data.query_pairs[i][1]).length;
 $('score').textContent=`${correct(m.answers)} / 100`;$('blank-score').textContent=`${correct(s.methods.zero.answers)} / 100`;
 const impact=summarizeImpact(data.query_pairs,s.methods.zero.answers,m.answers);
 $('helped').textContent=impact.helped; $('hurt').textContent=impact.hurt; $('net').textContent=`${impact.net>0?'+':''}${impact.net} / 100`;
 $('weights').replaceChildren();$('weight-table').replaceChildren();
 const max=Math.max(...actual.map(Math.abs),...pred.map(Math.abs),.001);
 for(let i=0;i<40;i++){
  const pair=element('div',undefined,'weight-pair');pair.title=`${connection(i)}: actual ${fmt(actual[i])}; replacement ${fmt(pred[i])}`;
  const short=i<10?`Left ${i}`:i<20?`Right ${i-10}`:i===20?'Bias':`Answer ${i-21}`;
  pair.append(element('span',short,'pair-label')); const plot=element('div',undefined,'pair-plot'); pair.append(plot);
  for(const [kind,value] of [['actual',actual[i]],['predicted',pred[i]]]){const bar=element('span',undefined,kind);bar.style.height=`${Math.abs(value)/max*48}%`;bar.style[value>=0?'bottom':'top']='50%';plot.append(bar);}
  $('weights').append(pair);const row=element('tr');[connection(i),fmt(actual[i]),fmt(pred[i]),fmt(pred[i]-actual[i])].forEach(v=>row.append(element('td',v)));$('weight-table').append(row);
 }
 $('answer-grid').replaceChildren();
 data.query_pairs.forEach(([a,b],i)=>{const ok=m.answers[i]===a+b,button=element('button',String(m.answers[i]),ok?'':'wrong');button.type='button';button.append(element('small',ok?'✓':'×'));button.setAttribute('aria-label',`${a} plus ${b}: answer ${m.answers[i]}, ${ok?'correct':'incorrect'}`);button.setAttribute('aria-pressed',String(i===query));button.addEventListener('click',()=>{query=i;renderQuery();[...$('answer-grid').children].forEach((e,j)=>e.setAttribute('aria-pressed',String(j===query)));});$('answer-grid').append(button);});
 const blank=correct(s.methods.zero.answers),restored=correct(m.answers);
 $('variation').textContent=blank===100?`Removing this neuron leaves all 100 answers correct. This replacement answers ${restored} correctly; the weights above still show how close the prediction is.`:`With this neuron blank, ${100-blank} answers change. This replacement gives ${restored} correct answers: ${restored>blank?`${restored-blank} more`:restored<blank?`${blank-restored} fewer`:'the same total'} than leaving it blank.`;
 renderQuery();
}
function renderQuery(){const [a,b]=data.query_pairs[query];$('query-title').textContent=`${a} + ${b}`;$('query-truth').textContent=`Correct answer: ${a+b} · Original: ${data.original_answers[query]}`;const chosen=data.slots[slot].methods[method].answers[query],blank=data.slots[slot].methods.zero.answers[query]; $('query-impact').textContent=`Selected replacement: ${blank} blank → ${chosen} restored. ${queryImpact(a+b,blank,chosen)}.`; $('query-methods').replaceChildren();for(const key of Object.keys(labels)){const r=element('div',undefined,'query-row'),answer=data.slots[slot].methods[key].answers[query];r.append(element('span',labels[key]),element('strong',`${answer} ${answer===a+b?'✓':'×'}`));$('query-methods').append(r);}}
async function start(){
 const response=await fetch(new URL('./no-sort-example.json',import.meta.url)); if(!response.ok)throw Error('Example data could not be loaded.'); data=await response.json();
 if(data.schema!=='synaptic-lens-no-sort-example-v1'||data.slots.length!==32||data.query_pairs.length!==100)throw Error('Unexpected example data format.');
 for(let i=0;i<32;i++){const b=element('button',String(i+1));b.type='button';b.dataset.slot=String(i);b.setAttribute('aria-label',`Hide physical neuron ${i+1}`);b.addEventListener('click',()=>{slot=i;render();});$('slots').append(b);}
 $('method').addEventListener('change',e=>{method=e.target.value;render();});$('coordinates').addEventListener('change',e=>{coordinates=e.target.value;render();});
 let shuffled=false; $('shuffle').addEventListener('click',()=>{shuffled=!shuffled;const buttons=[...$('slots').children];const byId=new Map(buttons.map(b=>[Number(b.dataset.slot),b]));const order=shuffled?data.saved_permutation:Array.from({length:32},(_,i)=>i);$('slots').replaceChildren(...order.map(i=>byId.get(i)));$('shuffle').textContent=shuffled?'Restore original display order':'Shuffle the display';});
 const url=URL.createObjectURL(new Blob([JSON.stringify(data)],{type:'application/json'}));$('download').href=url;$('download').download='no-sort-example.json';
 render();$('loading').hidden=true;$('experiment').hidden=false;
}
start().catch(error=>{$('loading').textContent=`${error.message} Serve the site over HTTP, then reload. The JSON download remains available below.`;$('loading').setAttribute('role','alert');const a=element('a',' Download example data');a.href='../assets/no-sort-example.json';$('loading').append(a);});
