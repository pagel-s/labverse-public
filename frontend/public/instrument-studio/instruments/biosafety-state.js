(function(root){
'use strict';
class BiosafetyState{
 constructor(){this.sash='work';this.power=true;this.blower=false;this.light=true;this.outlets=false;this.flowOverlay=false;this.message='Work opening · blower off';}
 get status(){return !this.power?'off':this.sash==='service'?'sash warning':!this.blower?'blower off':this.sash==='closed'?'standby':'blower on';}
 fail(message){this.message=message;return false;}
 setSash(value){if(!['closed','work','service'].includes(value))return this.fail('Unknown sash position.');this.sash=value;this.message=value==='service'?'Raised sash · visual warning':value==='closed'?'Sash closed': 'Sash at demonstration work opening';return true;}
 togglePower(){this.power=!this.power;if(!this.power){this.blower=false;this.light=false;this.outlets=false;}this.message=this.power?'Power on · controls available':'Power off';return true;}
 toggleBlower(){if(!this.power)return this.fail('Turn on cabinet power first.');this.blower=!this.blower;this.message=this.blower?'Blower indicator on':'Blower off';return true;}
 toggleLight(){if(!this.power)return this.fail('Turn on cabinet power first.');this.light=!this.light;this.message=this.light?'Work light on':'Work light off';return true;}
 toggleOutlets(){if(!this.power)return this.fail('Turn on cabinet power first.');this.outlets=!this.outlets;this.message=this.outlets?'Utility indicator on':'Utility indicator off';return true;}
 snapshot(){return{equipment_type:'biosafety_cabinet',power:this.power,sash_position:this.sash,blower_on:this.blower,light_on:this.light,outlets_on:this.outlets,status:this.status,illustrative:true};}
}
root.LabBiosafetyState=BiosafetyState;
})(globalThis);
