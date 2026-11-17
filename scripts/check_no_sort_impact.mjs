import assert from 'node:assert/strict';
import fs from 'node:fs';
import {summarizeImpact,queryImpact} from '../docs/assets/no-sort-impact.mjs';
const d=JSON.parse(fs.readFileSync(new URL('../docs/assets/no-sort-example.json',import.meta.url)));
let cases=0,questions=0;
for(const slot of d.slots)for(const m of Object.values(slot.methods)){
 const blank=slot.methods.zero.answers;
 const before=blank.map((x,i)=>x===d.query_pairs[i][0]+d.query_pairs[i][1]);
 const after=m.answers.map((x,i)=>x===d.query_pairs[i][0]+d.query_pairs[i][1]);
 const fixed=after.filter((ok,i)=>ok&&!before[i]).length,broken=before.filter((ok,i)=>ok&&!after[i]).length;
 assert.deepEqual(summarizeImpact(d.query_pairs,blank,m.answers),{helped:fixed,hurt:broken,net:after.filter(Boolean).length-before.filter(Boolean).length});
 d.query_pairs.forEach(([a,b],i)=>{const text=queryImpact(a+b,blank[i],m.answers[i]);assert.equal(text.startsWith('Helped'),!before[i]&&after[i]);assert.equal(text.startsWith('Hurt'),before[i]&&!after[i]);questions++;});cases++;
}
assert.deepEqual(summarizeImpact([[0,0],[1,1]],[1,2],[0,3]),{helped:1,hurt:1,net:0});
assert.equal(queryImpact(2,1,3),'Still incorrect: changed to another wrong answer');
console.log(JSON.stringify({status:'passed',cases,questions}));
