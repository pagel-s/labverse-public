(function(root){
'use strict';
class EquipmentState{
 constructor(type,initial={}){this.type=type;this.phase='idle';this.elapsed=0;this.duration=12;this.paused=false;this.cooling=0;this.message='Ready';Object.assign(this,initial);}
 get running(){return this.phase==='running';}get locked(){return this.running||this.cooling>0;}
 fail(message){this.message=message;return false;}
 toggle(field,guard=null){if(guard){const error=guard(this);if(error)return this.fail(error);}this[field]=!this[field];this.message=field+' '+(this[field]?'on / open':'off / closed');return true;}
 start(guard=()=>null){const error=guard(this);if(error)return this.fail(error);if(this.locked)return this.fail('A demonstration is already active.');this.phase='running';this.elapsed=0;this.paused=false;this.message='Demonstration running';return true;}
 pause(){if(!this.running)return this.fail('No demonstration is running.');this.paused=!this.paused;this.message=this.paused?'Paused':'Resumed';return true;}
 stop(cooldown=0){if(!this.running)return this.fail('No demonstration is running.');this.phase='stopped';this.paused=false;this.cooling=cooldown;this.message=cooldown?'Illustrative release delay':'Demonstration stopped';return true;}
 tick(dt){if(!Number.isFinite(dt)||dt<=0)return;dt=Math.min(dt,.1);if(this.cooling>0){this.cooling=Math.max(0,this.cooling-dt);if(this.cooling<1e-8){this.cooling=0;this.message='Release complete';}}if(!this.running||this.paused)return;this.elapsed=Math.min(this.duration,this.elapsed+dt);if(this.elapsed>=this.duration-1e-8){this.elapsed=this.duration;this.phase='complete';this.cooling=this.releaseDelay||0;this.message=this.cooling?'Illustrative release delay':'Demonstration complete';}}
 snapshot(){const data={};for(const [k,v] of Object.entries(this))if(typeof v!=='function')data[k]=v;return JSON.parse(JSON.stringify({...data,equipment_type:this.type,status:this.cooling>0?'release delay':this.paused?'paused':this.phase,progress:this.elapsed/this.duration,illustrative:true}));}
}
root.LabEquipmentState=EquipmentState;
})(globalThis);
