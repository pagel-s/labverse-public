import assert from 'node:assert/strict';import fs from 'node:fs';import vm from 'node:vm';
const base=new URL('../',import.meta.url),paint=new Proxy({},{get:()=>()=>{}});
const c=vm.createContext({console,document:{createElement:()=>({width:1,height:1,getContext:()=>paint})}});
for(const file of ['vendor/three.min.js','instruments/pipetting-robot-state.js','instruments/pipetting-robot-model.js'])vm.runInContext(fs.readFileSync(new URL(file,base),'utf8'),c);
const a=c.createLabPipettingRobot(c.THREE),s=new c.LabPipettingRobotState();
a.update(s,0,{position:{x:.01,y:.17,z:.07}});assert.equal(a.model.scale.x,.001);assert.equal(a.model.userData.units,'metres');
const b=new c.THREE.Box3().setFromObject(a.model);for(const k of ['x','y','z']){assert.ok(Number.isFinite(b.min[k]));assert.ok(b.max[k]-b.min[k]<.6);}
let count=0;a.model.traverse(o=>{if(o.isMesh){count++;for(const n of o.geometry.attributes.position.array)assert.ok(Number.isFinite(n));}});assert.ok(count>300);
s.commit('pick');s.commit('aspirate','source',0,100);a.update(s);assert.equal(a.parts.attachedTip.visible,true);assert.equal(a.parts.tipLiquid.visible,true);
s.commit('dispense','plate',0,100);a.update(s);assert.equal(a.parts.wellLiquids[0].visible,true);assert.equal(a.parts.tipLiquid.visible,false);
s.setDoor(true);a.update(s);assert.equal(a.parts.doorPivot.rotation.y,-1.5);
const other=c.createLabPipettingRobot(c.THREE);assert.equal(other.parts.doorPivot.rotation.y,0);
assert.equal(a.anchors.source[0],0);assert.ok(a.targets.length>96);
console.log('Robot geometry: finite meshes, metre envelope, independent instances and state-driven poses passed');
