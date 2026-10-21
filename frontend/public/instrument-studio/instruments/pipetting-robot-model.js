(function(root){
 'use strict';
 root.createLabPipettingRobot=function(T){
 const model=new T.Group();model.name='unbranded-liquid-handler';model.scale.setScalar(.001);const scene=model;
 const colors={body:new T.Color('#d8d8d6'),ink:new T.Color('#151617'),line:new T.Color('#bfc0c0'),liquid:new T.Color('#27a9cf'),mark:new T.Color('#ded8b8'),paper:new T.Color('#edeeec')};
 const mats=[];function material(key,extra={}){const m=new T.MeshStandardMaterial({color:colors[key],roughness:.55,metalness:.18,...extra});m.userData.key=key;mats.push(m);return m;}
 const body=material('body',{roughness:.32,metalness:.32}),dark=material('ink',{roughness:.58,metalness:.04,envMapIntensity:.35}),metal=material('line',{roughness:.25,metalness:.86}),white=material('paper',{roughness:.28,metalness:.02}),fluid=new T.MeshPhysicalMaterial({color:0x29aaca,roughness:.09,metalness:0,transparent:true,opacity:.78,clearcoat:1}),glass=new T.MeshPhysicalMaterial({color:0xf0f3f2,roughness:.035,metalness:0,transparent:true,opacity:.12,clearcoat:1,clearcoatRoughness:.03,envMapIntensity:1.5,side:T.DoubleSide,depthWrite:false}),accent=material('mark',{roughness:.24,metalness:0,transparent:true,opacity:.76});
 const rubber=new T.MeshStandardMaterial({color:0x17181a,roughness:.86}),blueLED=new T.MeshStandardMaterial({color:0x49c4ff,emissive:0x2387c7,emissiveIntensity:1,roughness:.22}),steel=new T.MeshStandardMaterial({color:0xd6dce0,metalness:1,roughness:.17});
 const grainCanvas=document.createElement('canvas');grainCanvas.width=grainCanvas.height=256;const gc=grainCanvas.getContext('2d');let seed=41;function random(){seed=(seed*1664525+1013904223)>>>0;return seed/4294967296;}for(let y=0;y<256;y++){const g=Math.floor(125+random()*14);gc.fillStyle='rgb('+g+','+g+','+g+')';gc.fillRect(0,y,256,1);}const grain=new T.CanvasTexture(grainCanvas);grain.wrapS=grain.wrapT=T.RepeatWrapping;grain.repeat.set(2,2);metal.bumpMap=grain;metal.bumpScale=.055;
 const geoCache=new Map();
 function bevelBox(w,h,d){const key=w+','+h+','+d;if(geoCache.has(key))return geoCache.get(key);const r=Math.min(3,w*.12,h*.12,d*.12),sw=w/2-r,sh=h/2-r;const s=new T.Shape();s.moveTo(-sw,-sh);s.lineTo(sw,-sh);s.lineTo(sw,sh);s.lineTo(-sw,sh);s.closePath();const g=new T.ExtrudeGeometry(s,{depth:d-2*r,steps:1,bevelEnabled:true,bevelSegments:3,bevelSize:r,bevelThickness:r,curveSegments:4});g.translate(0,0,-d/2+r);g.computeVertexNormals();geoCache.set(key,g);return g;}
 function box(w,h,d,x,y,z,mat,parent=scene){const o=new T.Mesh(mat===glass?new T.BoxGeometry(w,h,d):bevelBox(w,h,d),mat);o.position.set(x,y,z);o.castShadow=mat!==glass;o.receiveShadow=mat!==glass;parent.add(o);return o;}
 function cylinder(rt,rb,h,x,y,z,mat,parent=scene){const o=new T.Mesh(new T.CylinderGeometry(rt,rb,h,32),mat);o.position.set(x,y,z);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}
 function screw(x,y,z,parent=scene,face='front'){const o=cylinder(2.1,2.1,1.4,x,y,z,steel,parent);if(face==='front')o.rotation.x=Math.PI/2;const slot=box(2.7,.4,.2,x,y,z+.8,dark,parent);if(face!=='front'){slot.rotation.x=Math.PI/2;slot.position.set(x,y+.8,z);}return o;}
 function decal(text,w,h,x,y,z,parent=scene,ink='#35383b',face='front'){const cv=document.createElement('canvas');cv.width=1024;cv.height=128;const c=cv.getContext('2d');c.clearRect(0,0,1024,128);c.fillStyle=ink;c.font='500 75px Arial';c.textAlign='center';c.fillText(text,512,92);const tex=new T.CanvasTexture(cv);tex.encoding=T.sRGBEncoding;const mesh=new T.Mesh(new T.PlaneGeometry(w,h),new T.MeshBasicMaterial({map:tex,transparent:true,depthWrite:false}));mesh.position.set(x,y,z);if(face==='up')mesh.rotation.x=-Math.PI/2;parent.add(mesh);return mesh;}
 const shell=new T.Group();scene.add(shell);
 box(390,24,410,0,0,0,body);box(362,5,382,0,15,0,metal);box(386,5,406,0,-12,0,dark);
 for(const x of [-183,183])for(const z of [-194,194]){box(23,371,24,x,195,z,dark,shell);cylinder(12,13,10,x,-18,z,rubber);}
 for(const x of [-183,183]){box(23,23,410,x,380,0,dark,shell);box(2,338,360,x,195,0,glass,shell);box(20,20,400,x,28,0,dark,shell);box(3,3,341,x-5,340,0,white,shell);}
 for(const z of [-193,193]){box(364,26,20,0,378,z,body,shell);box(350,8,12,0,36,z,dark,shell);}
 box(356,337,2,0,201,-191,glass,shell);box(346,65,8,0,336,-181,white,shell);
 for(let i=0;i<21;i++)box(1.8,30,1,51+i*4,337,-176,dark,shell);
 for(const x of [-162,162])for(const y of [365,388])screw(x,y,204,shell);
 decal('LIQUID HANDLER',87,10,-112,379,204,shell);decal('LIQUID HANDLING SYSTEM',124,6,80,380,204,shell);
 const power=cylinder(5.5,5.5,1.7,-183,344,207,steel,shell);power.rotation.x=Math.PI/2;const led=cylinder(3.6,3.6,2,-183,344,208,blueLED,shell);led.rotation.x=Math.PI/2;
 const doorPivot=new T.Group();doorPivot.position.set(-172,34,202);shell.add(doorPivot);box(340,322,2,170,161,0,glass,doorPivot);
 // Compact pull handle with mounts fixed to the transparent door.
 box(22,29,2,323,161,2,dark,doorPivot);
 for(const y of [152,170]){box(8,6,10,323,y,8,dark,doorPivot);screw(323,y,3.8,doorPivot);}
 box(8,24,6,323,161,14,dark,doorPivot);

 for(const y of [88,250]){box(14,19,10,0,y,0,dark,doorPivot);screw(0,y,6,doorPivot);}
 doorPivot.rotation.y=0;
 for(let i=0;i<6;i++){box(34,1,62,-120+i*48,20,-147,metal);}
 const labelObjects=[];
 function label(text,x,y,z,size=18){const cv=document.createElement('canvas');cv.width=512;cv.height=64;const ctx=cv.getContext('2d');function draw(){ctx.clearRect(0,0,512,64);ctx.fillStyle='#'+colors.ink.getHexString();ctx.font='500 32px sans-serif';ctx.textAlign='center';ctx.fillText(text,256,43);}draw();const tex=new T.CanvasTexture(cv);const sp=new T.Sprite(new T.SpriteMaterial({map:tex,depthTest:false,transparent:true}));sp.position.set(x,y,z);sp.scale.set(size*8,size,1);scene.add(sp);sp.visible=false;labelObjects.push({draw,tex,sp});return sp;}

 const slots=[];for(let row=0;row<4;row++)for(let col=0;col<3;col++){const p={x:(col-1)*116,z:145-row*94};slots.push(p);box(106,1,84,p.x,18,p.z,metal);decal(String(slots.length),8,5,p.x-44,19,p.z+33,scene,'#62696d','up');for(const dx of [-47,47])for(const dz of [-34,34])screw(p.x+dx,19,p.z+dz,scene,'up');}
 const pos={tips:slots[0],source:slots[1],plate:slots[4],waste:slots[11]};
 const hitObjects=[];function hit(mesh,target,well){mesh.userData={target,well};hitObjects.push(mesh);}
 const tipMeshes=[],wellLiquids=[];
 function wellPos(i,p){return{x:p.x-45+(i%12)*8.2,z:p.z-28+Math.floor(i/12)*8};}
 box(108,7,80,pos.tips.x,23,pos.tips.z,body);const rack=box(105,9,77,pos.tips.x,30,pos.tips.z,white);hit(rack,'tips');
 for(let i=0;i<96;i++){const p=wellPos(i,pos.tips);const o=cylinder(2.7,.8,17,p.x,38,p.z,accent);tipMeshes.push(o);hit(o,'tips');}
 const plate=box(106,9,77,pos.plate.x,23,pos.plate.z,white);hit(plate,'plate');
 for(let i=0;i<96;i++){const p=wellPos(i,pos.plate);const ring=new T.Mesh(new T.TorusGeometry(3.1,.6,8,24),white);ring.rotation.x=Math.PI/2;ring.position.set(p.x,29.5,p.z);ring.castShadow=true;scene.add(ring);const o=cylinder(2.9,2.9,2,p.x,28.5,p.z,dark);hit(o,'plate',i);const l=cylinder(2.75,2.75,.8,p.x,31,p.z,fluid);l.visible=false;wellLiquids.push(l);hit(l,'plate',i);}
 function tub(p,w,d,h){const group=new T.Group();box(w,3,d,p.x,20,p.z,white,group);for(const x of [-1,1])box(3,h,d,p.x+x*w/2,20+h/2,p.z,white,group);for(const z of [-1,1])box(w,h,3,p.x,20+h/2,p.z+z*d/2,white,group);scene.add(group);return group;}
 const source=tub(pos.source,94,62,33);source.children.forEach(o=>hit(o,'source'));const sourceFluid=box(90,23,58,pos.source.x,33,pos.source.z,fluid);hit(sourceFluid,'source');
 const waste=tub(pos.waste,96,68,47);waste.children.forEach(o=>hit(o,'waste'));const wasteInside=box(88,2,60,pos.waste.x,23,pos.waste.z,dark);hit(wasteInside,'waste');
 label('TIPS',pos.tips.x,65,pos.tips.z+27,9);label('RESERVOIR',pos.source.x,66,pos.source.z+25,9);label('96-WELL PLATE',pos.plate.x,47,pos.plate.z+38,9);label('WASTE',pos.waste.x,82,pos.waste.z,9);
 for(const x of [-162,162]){box(17,25,350,x,328,0,metal);for(const dx of [-5,5])box(1,2,340,x+dx,341,0,dark);const rod=cylinder(3.8,3.8,348,x,345,0,steel);rod.rotation.x=Math.PI/2;for(const z of [-174,174]){box(25,24,15,x,330,z,dark);screw(x,342,z,scene,'up');}}
 const gantry=new T.Group();scene.add(gantry);box(335,18,20,0,330,0,metal,gantry);box(335,6,8,0,341,0,dark,gantry);for(const y of [320,339]){const rod=cylinder(3,3,328,0,y,13,steel,gantry);rod.rotation.z=Math.PI/2;}for(let i=0;i<65;i++)box(1.4,4,2,-159+i*5,340,6,rubber,gantry);
 const carriage=new T.Group();gantry.add(carriage);box(55,64,42,0,309,0,body,carriage);box(43,228,8,0,216,-23,metal,carriage);for(const x of [-14,14])cylinder(2.7,2.7,216,x,218,-17,steel,carriage);for(const x of [-22,22])for(const y of [287,334])screw(x,y,22,carriage);const motor=cylinder(20,20,37,0,334,-5,dark,carriage);motor.rotation.x=Math.PI/2;
 const head=new T.Group();carriage.add(head);box(48,119,38,0,82,1,white,head);box(45,3,35,0,139,1,body,head);box(42,14,34,0,28,1,body,head);for(const x of [-17,17])for(const y of [39,125])screw(x,y,21,head);decal('300 µL',20,4,0,114,20.2,head);decal('SINGLE CHANNEL',29,3,0,106,20.2,head);box(23,13,25,0,23,0,dark,head);cylinder(5,3,14,0,10,0,metal,head);
 const attachedTip=cylinder(3.6,.65,26,0,-9,0,accent,head),tipLiquid=cylinder(2,.5,15,0,-10,0,fluid,head);
 const halo=new T.Mesh(new T.TorusGeometry(7,.8,8,32),accent);halo.rotation.x=Math.PI/2;halo.visible=false;scene.add(halo);

 function update(state,dt=0,options={}){
  const s=state.data||state;
  attachedTip.visible=s.tip;tipLiquid.visible=s.tip&&s.tipVolume>0;tipLiquid.scale.y=Math.max(.05,s.tipVolume/300);
  tipMeshes.forEach((o,i)=>o.visible=i>=s.used);
  wellLiquids.forEach((o,i)=>{o.visible=s.wells[i]>0;o.scale.y=1+s.wells[i]/50;});
  sourceFluid.scale.y=Math.max(.001,s.source/10000);sourceFluid.position.y=21+11.5*s.source/10000;
  doorPivot.rotation.y=state.doorOpen?-1.5:0;
  if(options.position){const p=options.position;if(!['x','y','z'].every(k=>Number.isFinite(p[k])))throw Error('Finite metre position required');gantry.position.z=p.z*1000;carriage.position.x=p.x*1000;head.position.y=p.y*1000;}
  if(options.shellVisible!==undefined)shell.visible=!!options.shellVisible;
  if(options.labelsVisible!==undefined)labelObjects.forEach(o=>o.sp.visible=!!options.labelsVisible);
 }
 const anchors=Object.fromEntries(Object.entries(pos).map(([key,p])=>[key,[p.x*.001,.03,p.z*.001]]));
 model.userData={equipment_type:'liquid-handler',units:'metres',constructionUnits:'millimetres',anchors,simulationOnly:true,dimensionsApproximate:true};
 return {model,update,materials:mats,targets:hitObjects,anchors,
  parts:{shell,doorPivot,gantry,carriage,head,attachedTip,tipLiquid,tipMeshes,wellLiquids,sourceFluid,halo},
  layout:{pos,wellPos},labelObjects};
 };
})(globalThis);
