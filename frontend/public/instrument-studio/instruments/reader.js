(function(root){'use strict';
root.LabEquipmentDefinitions.reader={
 title:'Microplate reader',category:'OPTICAL / MULTIMODE READER',
 description:'Sliding plate carrier, removable 96-well demonstration plate and enclosed reading cycle.',
 reference:'reader-references.md',notes:'Unbranded CLARIOstar Plus-inspired exterior, 450 × 510 × 400 mm. Carrier and hidden mechanisms are approximate. The cycle displays synthetic progress only; no optical measurement or assay result is calculated.',
 view:{target:[0,.21,.05],distance:1.55,portraitFov:44},
 state:()=>new LabEquipmentState('microplate_reader',{open:false,loaded:false}),
 actions:[{id:'tray',moves:true,label:s=>s.open?'Retract carrier':'Extend carrier',run:s=>{if(s.running)return s.fail('Stop reading before extending the carrier.');s.open=!s.open;s.message=s.open?'Carrier extended':'Carrier retracted';return true;}},{id:'plate',label:s=>s.loaded?'Remove demonstration plate':'Load demonstration plate',run:s=>{if(s.running||!s.open)return s.fail('Extend the carrier before changing the plate.');s.loaded=!s.loaded;s.phase='idle';s.elapsed=0;s.message=s.loaded?'96-well demonstration plate seated':'Carrier empty';return true;}}],
 start:s=>s.start(s=>s.open?'Retract the carrier first.':!s.loaded?'Load a demonstration plate first.':null),stop:s=>s.stop(),
 build(T){
 const a=LabAssetTools(T),{model,M,box,cyl,group,label}=a;model.name='unbranded-microplate-reader';
 const purple=a.material(0x4b315e,.4,.15),skin=a.material(0xe3e5e5,.31,.22);
 a.feet([-181,181],[-202,202],9,18);box(447,14,506,0,24,0,M.dark,model,9);
 // Two separate skins and a narrow perimeter seam. Front carrier aperture remains genuinely open.
 box(450,174,510,0,311,0,skin,model,12);box(446,5,503,0,221,0,purple,model,2);
 const front=new T.Shape();front.moveTo(-215,31);front.lineTo(215,31);front.quadraticCurveTo(225,31,225,41);front.lineTo(225,208);front.quadraticCurveTo(225,218,215,218);front.lineTo(-215,218);front.quadraticCurveTo(-225,218,-225,208);front.lineTo(-225,41);front.quadraticCurveTo(-225,31,-215,31);
 const opening=new T.Path();opening.moveTo(-200,77);opening.lineTo(-200,153);opening.lineTo(2,153);opening.lineTo(2,77);opening.closePath();front.holes.push(opening);
 a.mesh(new T.ExtrudeGeometry(front,{depth:9,bevelEnabled:false}),skin,0,0,246);
 for(const x of [-220,220])box(10,183,497,x,124,-4,skin,model,4);box(438,183,10,0,124,-250,skin,model,3);box(435,7,490,0,35,0,skin,model,2);box(436,7,490,0,214,0,skin,model,2);
 box(202,3,17,-99,153,251,M.dark);box(202,3,17,-99,77,251,M.dark);for(const x of [-198,0])box(4,76,17,x,115,251,M.dark);
 box(196,66,4,-99,115,150,M.dark);
 // Recessed upper-right grip and seam details, without reference brand lettering.
 box(75,8,21,158,382,249,M.dark,model,5);box(82,5,24,158,389,247,skin,model,4);
 for(let i=0;i<15;i++)box(2,47,4,224,121,-165+i*14,M.dark,model,.6);
 for(const y of [197,181]){const l=cyl(2.4,2,-185,y,256,y===197?M.accent:M.dark);l.rotation.x=Math.PI/2;}
 label('PLATE READER',100,8,143,253,257);label('DEMO',30,5,-176,161,257);
 const tray=group(-99,93,180);box(190,7,145,0,0,0,M.dark,tray,3);box(190,17,7,0,7,74,M.dark,tray,2);
 for(const x of [-88,88])box(8,8,143,x,7,0,M.steel,tray,1);
 const plate=group(7,6,6,tray);box(128,3,86,0,0,0,M.white,plate,1);
 // Actual open wells in a pierced top sheet, rather than dots printed on a solid box.
 const top=new T.Shape();top.moveTo(-64,-43);top.lineTo(64,-43);top.lineTo(64,43);top.lineTo(-64,43);top.closePath();
 for(let r=0;r<8;r++)for(let c=0;c<12;c++){const x=-49.5+c*9,z=-31.5+r*9,h=new T.Path();h.absarc(x,z,3.25,0,Math.PI*2,true);top.holes.push(h);cyl(3.8,10,x,6,z,M.white,plate,3.8,true);cyl(3.2,1,x,2,z,M.glass,plate);}
 const sheet=a.mesh(new T.ShapeGeometry(top),M.white,0,12,0,plate);sheet.rotation.x=Math.PI/2;sheet.material=sheet.material.clone();sheet.material.side=T.DoubleSide;
 for(const x of [-63,63])box(2,12,86,x,6,0,M.white,plate,0);for(const z of [-42,42])box(126,12,2,0,6,z,M.white,plate,0);
 a.batch();model.userData={equipment_type:'microplate_reader',units:'metres',referenceEnvelope:{width:.45,depth:.51,height:.4},anchors:{carrierRetracted:[-.092,.111,.186],carrierExtended:[-.092,.111,.341]},simulationOnly:true};
 return{model,update(s,dt,{reducedMotion=false}={}){const target=s.open?335:180;tray.position.z=reducedMotion?target:tray.position.z+(target-tray.position.z)*Math.min(1,dt*10);plate.visible=s.loaded;}};
 }
};})(globalThis);
