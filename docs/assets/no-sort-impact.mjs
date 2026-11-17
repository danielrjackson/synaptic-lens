export function summarizeImpact(pairs, blank, replacement) {
 let helped=0,hurt=0;
 pairs.forEach(([a,b],i)=>{const before=blank[i]===a+b,after=replacement[i]===a+b;helped+=Number(!before&&after);hurt+=Number(before&&!after);});
 return {helped,hurt,net:helped-hurt};
}
export function queryImpact(truth,blank,replacement) {
 const before=blank===truth,after=replacement===truth;
 return !before&&after?'Helped: fixed this answer':before&&!after?'Hurt: broke this answer':after?'Unchanged: still correct':blank===replacement?'Unchanged: still incorrect':'Still incorrect: changed to another wrong answer';
}
