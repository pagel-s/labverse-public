(function(root){
 'use strict';
 class IncubatorState {
  constructor(){this.outerOpen=true;this.innerOpen=false;this.power=true;this.shelves=[false,false,false];this.vessels=Array.from({length:6},(_,i)=>({id:'vessel-'+(i+1),shelf:i<3?i:null}));this.recovery=1;this.openSeconds=0;this.message='Outer door open · inner glass sealed';}
  get phase(){return !this.power?'off':this.innerOpen?'door open':this.recovery<.999?'recovering':'stable';}
  fail(m){this.message=m;return false;}
  toggleOuter(){if(this.outerOpen&&(this.innerOpen||this.shelves.some(Boolean)))return this.fail('Close the glass door and retract shelves first.');this.outerOpen=!this.outerOpen;this.message=this.outerOpen?'Outer door open':'Outer door closed';return true;}
  toggleInner(){if(!this.outerOpen)return this.fail('Open the insulated outer door first.');if(this.innerOpen&&this.shelves.some(Boolean))return this.fail('Retract all shelves before closing the glass door.');this.innerOpen=!this.innerOpen;this.message=this.innerOpen?'Glass door open · chamber exposed':'Glass door sealed · recovering';return true;}
  slideShelf(i){if(!Number.isInteger(i)||i<0||i>2)return this.fail('Invalid shelf.');if(!this.outerOpen||!this.innerOpen)return this.fail('Open both doors before sliding a shelf.');this.shelves[i]=!this.shelves[i];this.message='Shelf '+(i+1)+(this.shelves[i]?' extended':' retracted');return true;}
  load(i){if(!Number.isInteger(i)||i<0||i>2)return this.fail('Invalid shelf.');if(!this.outerOpen||!this.innerOpen)return this.fail('Open both doors before loading.');if(this.vessels.filter(v=>v.shelf===i).length>=2)return this.fail('Both demonstration positions on this shelf are occupied.');const v=this.vessels.find(v=>v.shelf===null);if(!v)return this.fail('All demonstration vessels are stored.');v.shelf=i;this.message=v.id+' stored on shelf '+(i+1);return true;}
  unload(i){if(!Number.isInteger(i)||i<0||i>2)return this.fail('Invalid shelf.');if(!this.outerOpen||!this.innerOpen)return this.fail('Open both doors before unloading.');const v=this.vessels.find(v=>v.shelf===i);if(!v)return this.fail('This shelf is empty.');v.shelf=null;this.message=v.id+' returned to bench';return true;}
  togglePower(){this.power=!this.power;this.message=this.power?'Power on · restoring chamber':'Power off';return true;}
  tick(dt){if(!Number.isFinite(dt)||dt<=0)return;dt=Math.min(dt,.1);if(this.innerOpen){this.openSeconds+=dt;this.recovery=Math.max(0,this.recovery-dt/12);}else{this.openSeconds=0;if(this.power)this.recovery=Math.min(1,this.recovery+dt/15);}if(!this.power)this.recovery=Math.max(0,this.recovery-dt/25);}
  snapshot(){return{equipment_type:'incubator',status:this.phase,power:this.power,outer_door_open:this.outerOpen,inner_door_open:this.innerOpen,shelf_extended:[...this.shelves],sample_locations:this.vessels.map(v=>({...v})),visual_recovery:this.recovery,illustrative:true};}
 }
 root.LabIncubatorState=IncubatorState;
})(globalThis);
