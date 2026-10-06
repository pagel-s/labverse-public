import assert from 'node:assert/strict';
import '../asset-tools.js';import '../equipment-state.js';import '../collection-tools.js';
const ids=['qpcr','gelimager','electrophoresis','transport','fumehood','bench','storage','labware','consumables','waste','emergency','ppe'];
for(const id of ids)await import('../instruments/'+id+'.js');
let checks=0;const check=(a,b)=>{assert.deepEqual(a,b);checks++;};
for(const id of ['qpcr','gelimager','electrophoresis']){const d=LabEquipmentDefinitions[id],s=d.state(),a=i=>d.actions[i].run(s);check(d.start(s),false);if(!s.open)check(a(0),true);check(a(1),true);check(d.start(s),false);check(a(0),true);check(d.start(s),true);check(a(0),false);check(a(1),false);s.pause();check(a(0),false);s.pause();for(let i=0;i<130;i++)s.tick(.1);check(s.phase,'complete');check(a(0),true);check(a(1),true);check(s.loaded,false);}
{const d=LabEquipmentDefinitions.transport,s=d.state(),a=i=>d.actions[i].run(s);check(a(0),true);check(a(1),false);check(s.plateAt,0);check(a(0),true);check(a(1),true);check(s.plateAt,-1);check(s.holding,true);check(a(0),true);check(a(1),true);check(s.plateAt,1);check(s.holding,false);}
for(const id of ids.slice(4)){const d=LabEquipmentDefinitions[id],s=d.state();for(const a of d.actions){check(a.run(s),true);check(s[a.id],true);check(a.run(s),true);check(s[a.id],false);}}
console.log('Remaining library: '+checks+' state assertions passed');
