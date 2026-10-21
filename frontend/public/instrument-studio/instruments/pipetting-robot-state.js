(function(root){
 'use strict';
 const initial=()=>({tip:false,tipVolume:0,source:10000,wells:Array(96).fill(0),used:0,discarded:0});
 class PipettingRobotState{
  constructor(){this.reset();}
  reset(){this.data=initial();this.doorOpen=false;this.busy=false;}
  snapshot(){return {schemaVersion:1,equipmentType:'liquid-handler',...this.data,wells:this.data.wells.slice(),doorOpen:this.doorOpen,busy:this.busy};}
  restore(value){
   if(!value||typeof value.tip!=='boolean'||!Array.isArray(value.wells)||value.wells.length!==96)throw Error('Invalid liquid-handler snapshot');
   if(value.schemaVersion!==undefined&&value.schemaVersion!==1)throw Error('Unsupported snapshot version');
   if(value.equipmentType!==undefined&&value.equipmentType!=='liquid-handler')throw Error('Wrong equipment type');
   if(value.doorOpen!==undefined&&typeof value.doorOpen!=='boolean')throw Error('Invalid door state');
   if(value.busy!==undefined&&typeof value.busy!=='boolean')throw Error('Invalid motion state');
   if(value.busy)throw Error('Active motion snapshots cannot be resumed');
   const bounded=(n,max)=>Number.isInteger(n)&&n>=0&&n<=max;
   if(!value.wells.every(v=>bounded(v,300))||!bounded(value.source,10000)||!bounded(value.tipVolume,300)||!bounded(value.used,96)||!bounded(value.discarded,96))throw Error('Invalid volumes or tip counts');
   if(value.used!==value.discarded+Number(value.tip)||(!value.tip&&value.tipVolume!==0))throw Error('Tip custody mismatch');
   if(value.source+value.tipVolume+value.wells.reduce((a,b)=>a+b,0)!==10000)throw Error('Liquid conservation mismatch');
   if(this.busy)throw Error('Cannot restore during motion');
   this.doorOpen=value.doorOpen??false;
   this.data={tip:value.tip,tipVolume:value.tipVolume,source:value.source,wells:value.wells.slice(),used:value.used,discarded:value.discarded};
  }
  setDoor(open){if(typeof open!=='boolean')throw Error('Boolean door state required');if(this.busy)return false;this.doorOpen=open;return true;}
  validate(action,target='source',well=0,volume=0){
   const s=this.data;
   if(this.doorOpen)throw Error('Close the door before robot motion.');
   if(!['move','pick','eject','aspirate','dispense'].includes(action))throw Error('Unknown action');
   if(!['tips','source','plate','waste'].includes(target)||!Number.isInteger(well)||well<0||well>=96)throw Error('Invalid destination');
   if(action==='move')return;
   if(action==='pick'){if(s.tip||s.used>=96)throw Error(s.tip?'Eject the attached tip first.':'The tip rack is empty.');return;}
   if(!s.tip)throw Error('Pick up a tip first.');
   if(action==='eject'){if(s.tipVolume)throw Error('Dispense remaining liquid before ejecting.');return;}
   if(!Number.isInteger(volume)||volume<1||volume>300||!['source','plate'].includes(target))throw Error('Invalid liquid transfer');
   const amount=target==='source'?s.source:s.wells[well],capacity=target==='source'?10000:300;
   if(action==='aspirate'&&(s.tipVolume+volume>300||amount<volume))throw Error('Insufficient liquid or tip capacity');
   if(action==='dispense'&&(s.tipVolume<volume||amount+volume>capacity))throw Error('Insufficient tip liquid or destination capacity');
  }
  commit(action,target='source',well=0,volume=0){
   this.validate(action,target,well,volume);const s=this.data;
   if(action==='pick'){s.tip=true;s.used++;}
   if(action==='eject'){s.tip=false;s.discarded++;}
   if(action==='aspirate'||action==='dispense'){
    const delta=action==='aspirate'?volume:-volume;s.tipVolume+=delta;
    if(target==='source')s.source-=delta;else s.wells[well]-=delta;
   }
   return this.snapshot();
  }
 }
 root.LabPipettingRobotState=PipettingRobotState;
})(globalThis);
