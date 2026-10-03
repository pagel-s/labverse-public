(function(root){
  'use strict';
  class CentrifugeState {
    constructor(){this.lidOpen=true;this.rotorInstalled=true;this.coverInstalled=false;this.slots=Array(24).fill(null);this.phase='idle';this.elapsed=0;this.speed=0;this.duration=12;this.message='Open chamber · ready to load';this.temperature=20;this.events=[];}
    get running(){return ['accelerating','running','decelerating'].includes(this.phase);}
    get balanced(){return this.slots.every((v,i)=>v===this.slots[(i+12)%24]);}
    get loaded(){return this.slots.filter(v=>v!==null).length;}
    record(type){this.events.push({type,phase:this.phase,loaded:this.loaded});if(this.events.length>100)this.events.shift();}
    fail(message){this.message=message;return false;}
    toggleLid(){if(this.running)return this.fail('Lid locked until the rotor has stopped.');this.lidOpen=!this.lidOpen;this.message=this.lidOpen?'Lid open':'Lid closed';this.record('lid');return true;}
    toggleCover(){if(this.running||!this.lidOpen)return this.fail('Open the stopped instrument before changing the rotor cover.');if(!this.rotorInstalled)return this.fail('Install the rotor first.');this.coverInstalled=!this.coverInstalled;this.message=this.coverInstalled?'Rotor cover secured':'Rotor cover removed';this.record('cover');return true;}
    toggleRotor(){if(this.running||!this.lidOpen||this.coverInstalled||this.loaded)return this.fail('Open the lid, remove the cover, and unload all tubes first.');this.rotorInstalled=!this.rotorInstalled;this.message=this.rotorInstalled?'Rotor installed':'Rotor removed to the workbench';this.record('rotor');return true;}
    toggleSlot(i){if(!Number.isInteger(i)||i<0||i>23)return this.fail('Invalid slot.');if(this.running||!this.lidOpen||this.coverInstalled||!this.rotorInstalled)return this.fail('Open the lid and uncover the installed rotor to load tubes.');this.slots[i]=this.slots[i]===null?'dummy':null;this.message='Slot '+(i+1)+(this.slots[i]?' loaded':' unloaded');this.record('sample');return true;}
    loadPair(){if(this.running||!this.lidOpen||this.coverInstalled||!this.rotorInstalled)return this.fail('Open the lid and uncover the installed rotor first.');for(let i=0;i<12;i++){if(!this.slots[i]&&!this.slots[i+12]){this.slots[i]=this.slots[i+12]='dummy';this.message='Matched demonstration tubes loaded';this.record('pair');return true;}}return this.fail('No empty opposite pair remains.');}
    unload(){if(this.running||!this.lidOpen||this.coverInstalled||!this.rotorInstalled)return this.fail('Open the lid and uncover the installed rotor first.');this.slots.fill(null);this.message='Rotor unloaded';this.record('unload');return true;}
    start(){if(this.running)return this.fail('A cycle is already active.');if(this.lidOpen)return this.fail('Close the outer lid first.');if(!this.rotorInstalled)return this.fail('No rotor installed.');if(!this.coverInstalled)return this.fail('Secure the rotor cover first.');if(!this.loaded)return this.fail('Load demonstration tubes first.');if(!this.balanced)return this.fail('Simulation interlock: pair each tube with an opposite matched tube.');this.phase='accelerating';this.elapsed=0;this.message='Accelerating · lid locked';this.record('start');return true;}
    stop(){if(!this.running)return this.fail('The instrument is stopped.');if(this.phase==='decelerating')return true;this.phase='decelerating';this.elapsed=0;this.message='Decelerating · lid locked';this.record('stop');return true;}
    tick(dt){if(!Number.isFinite(dt)||dt<=0)return;dt=Math.min(dt,.1);this.elapsed+=dt;if(this.phase==='accelerating'){this.speed=Math.min(1,this.speed+dt/2.5);if(this.speed>=1){this.phase='running';this.elapsed=0;this.message='Demonstration cycle running';}}else if(this.phase==='running'){if(this.elapsed>=this.duration){this.phase='decelerating';this.elapsed=0;this.message='Decelerating · lid locked';}}else if(this.phase==='decelerating'){this.speed=Math.max(0,this.speed-dt/2.5);if(this.speed<=0){this.phase='complete';this.elapsed=0;this.message='Cycle complete · lid unlocked';this.record('complete');}}}
    snapshot(){return {equipment_type:'centrifuge',status:this.phase,lid_open:this.lidOpen,rotor_installed:this.rotorInstalled,rotor_cover_installed:this.coverInstalled,slots:[...this.slots],balanced:this.balanced,rotor_fraction:this.speed};}
  }
  root.LabCentrifugeState=CentrifugeState;
  if(typeof module!=='undefined')module.exports=CentrifugeState;
})(typeof globalThis!=='undefined'?globalThis:window);
