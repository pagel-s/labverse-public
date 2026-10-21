import assert from 'node:assert/strict';
import '../instruments/pipetting-robot-state.js';
const s=new globalThis.LabPipettingRobotState();
assert.throws(()=>s.commit('aspirate','source',0,50));
s.setDoor(true);assert.throws(()=>s.commit('pick'));s.setDoor(false);
s.commit('pick');s.commit('aspirate','source',0,100);s.commit('dispense','plate',0,100);
assert.equal(s.data.wells[0],100);assert.equal(s.data.source,9900);s.commit('eject');
const saved=s.snapshot();s.restore(saved);saved.wells[0]=0;assert.equal(s.data.wells[0],100);
assert.throws(()=>s.restore({...s.snapshot(),source:10000}));
assert.throws(()=>s.restore({...s.snapshot(),used:2}));
s.busy=true;assert.equal(s.setDoor(true),false);assert.throws(()=>s.restore(s.snapshot()));s.busy=false;
s.commit('pick');s.commit('aspirate','plate',0,100);s.commit('dispense','source',0,100);s.commit('eject');
assert.equal(s.data.source,10000);assert.equal(s.data.discarded,2);
assert.throws(()=>s.commit('move','plate',96));assert.throws(()=>s.commit('unknown'));
console.log('Pipetting robot state: custody, conservation, restore isolation and door interlocks passed');

// Restoring a parked machine preserves its safety interlock; active motion is not serialized.
s.setDoor(true);const parked=s.snapshot();const restored=new globalThis.LabPipettingRobotState();
restored.restore(parked);assert.equal(restored.doorOpen,true);assert.throws(()=>restored.commit('pick'));
for(const invalid of [{schemaVersion:2},{equipmentType:'centrifuge'},{doorOpen:'false'},{busy:true}]){
 const before=restored.snapshot();assert.throws(()=>restored.restore({...parked,...invalid}));assert.deepEqual(restored.snapshot(),before);
}
restored.restore(s.data);assert.equal(restored.doorOpen,false); // legacy idle quantities
