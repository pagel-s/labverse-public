import assert from 'node:assert/strict';
import '../instruments/incubator-state.js';
const State=globalThis.LabIncubatorState;
let checks=0;function check(actual,expected){assert.deepEqual(actual,expected);checks++;}
const s=new State();check(s.phase,'stable');check(s.load(0),false);check(s.slideShelf(0),false);
check(s.toggleOuter(),true);check(s.toggleInner(),false);check(s.outerOpen,false);
check(s.toggleOuter(),true);check(s.toggleInner(),true);check(s.phase,'door open');
check(s.toggleOuter(),false);check(s.outerOpen,true);check(s.slideShelf(1),true);
check(s.toggleInner(),false);check(s.innerOpen,true);check(s.load(1),true);check(s.load(1),false);
check(s.unload(1),true);check(s.vessels.filter(v=>v.shelf===null).length,3);
check(s.slideShelf(1),true);s.tick(.1);check(s.recovery<1,true);check(s.toggleInner(),true);check(s.toggleOuter(),true);
for(let i=0;i<160;i++)s.tick(.1);check(s.phase,'stable');check(s.recovery,1);
s.togglePower();s.tick(.1);check(s.phase,'off');check(s.recovery<1,true);
s.togglePower();for(let i=0;i<260;i++)s.tick(.1);check(s.phase,'stable');
const before=s.recovery;for(const dt of [NaN,Infinity,-1,0])s.tick(dt);check(s.recovery,before);
for(const i of [-1,3,NaN,1.5]){check(s.load(i),false);check(s.unload(i),false);check(s.slideShelf(i),false);}
const copy=s.snapshot();copy.shelf_extended[0]=true;copy.sample_locations[0].shelf=2;check(s.shelves[0],false);check(s.vessels[0].shelf,0);
check(new Set(s.vessels.map(v=>v.id)).size,6);check(s.snapshot().equipment_type,'incubator');
console.log('Incubator: '+checks+' assertions passed');export {checks};
