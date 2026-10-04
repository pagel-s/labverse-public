(function(root){
'use strict';
class ThermocyclerState{
 constructor(){this.lidOpen=true;this.plateLoaded=false;this.sealed=true;this.power=true;this.phase='idle';this.elapsed=0;this.duration=12;this.paused=false;this.message='Lid open · block empty';}
 get running(){return this.phase==='running';}
 fail(message){this.message=message;return false;}
 toggleLid(){if(this.running)return this.fail('Stop the demonstration before opening the lid.');this.lidOpen=!this.lidOpen;this.message=this.lidOpen?'Lid open':'Lid closed';return true;}
 togglePlate(){if(this.running||!this.lidOpen)return this.fail('Open the lid before moving the plate.');this.plateLoaded=!this.plateLoaded;this.phase='idle';this.elapsed=0;this.message=this.plateLoaded?'Demonstration plate loaded':'Plate returned to staging pad';return true;}
 toggleSeal(){if(this.running||!this.lidOpen)return this.fail('Open the lid before changing the plate cover.');this.sealed=!this.sealed;this.message=this.sealed?'Plate cover fitted':'Plate cover removed';return true;}
 togglePower(){if(this.running)return this.fail('Stop the demonstration before switching power.');this.power=!this.power;this.message=this.power?'Power on':'Power off';return true;}
 start(){if(!this.power)return this.fail('Turn on the instrument first.');if(this.running)return this.fail('A demonstration is already running.');if(this.lidOpen)return this.fail('Close the lid first.');if(!this.plateLoaded)return this.fail('Load the demonstration plate first.');if(!this.sealed)return this.fail('Fit the demonstration plate cover first.');this.phase='running';this.elapsed=0;this.paused=false;this.message='Illustrative cycle running';return true;}
 pause(){if(!this.running)return this.fail('No demonstration is running.');this.paused=!this.paused;this.message=this.paused?'Demonstration paused':'Demonstration resumed';return true;}
 stop(){if(!this.running)return this.fail('No demonstration is running.');this.phase='stopped';this.paused=false;this.message='Demonstration stopped · lid released';return true;}
 tick(dt){if(!this.running||this.paused||!Number.isFinite(dt)||dt<=0)return;this.elapsed=Math.min(this.duration,this.elapsed+Math.min(dt,.1));if(this.elapsed>=this.duration-1e-8){this.elapsed=this.duration;this.phase='complete';this.message='Demonstration complete · lid released';}}
 snapshot(){return{equipment_type:'pcr_thermocycler',power:this.power,lid_open:this.lidOpen,plate_loaded:this.plateLoaded,plate_id:'demo-plate-1',plate_covered:this.sealed,status:!this.power?'off':this.paused?'paused':this.phase,progress:this.elapsed/this.duration,illustrative:true};}
}
root.LabThermocyclerState=ThermocyclerState;
})(globalThis);
