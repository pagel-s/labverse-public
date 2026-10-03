(function(root){
'use strict';
root.createLabIncubator=function(T){
 const model=new T.Group();model.name='unbranded-co2-incubator';model.scale.setScalar(.001);
 const M={shell:new T.MeshStandardMaterial({color:0xcdd2cf,roughness:.4,metalness:.18}),steel:new T.MeshStandardMaterial({color:0x9eabae,roughness:.29,metalness:.85}),trim:new T.MeshStandardMaterial({color:0x222a2c,roughness:.5,metalness:.12}),rubber:new T.MeshStandardMaterial({color:0x454c49,roughness:.86}),light:new T.MeshStandardMaterial({color:0xeaf3ef,emissive:0xc8ddd7,emissiveIntensity:.35}),glass:new T.MeshPhysicalMaterial({color:0xe4f1eb,roughness:.06,transparent:true,opacity:.12,clearcoat:1,depthWrite:false,side:T.DoubleSide}),vessel:new T.MeshPhysicalMaterial({color:0xd1e5e0,roughness:.14,transparent:true,opacity:.35,depthWrite:false}),blue:new T.MeshStandardMaterial({color:0x559eae,roughness:.3}),water:new T.MeshPhysicalMaterial({color:0x8ea8a8,roughness:.1,transparent:true,opacity:.5})};for(const m of Object.values(M))m.color.convertSRGBToLinear();
 // Subtle procedural surface finish; no reference photography is embedded.
 const finishCanvas=document.createElement('canvas');finishCanvas.width=256;finishCanvas.height=256;const finishContext=finishCanvas.getContext('2d'),finishImage=finishContext.createImageData(256,256);let noiseSeed=913;for(let i=0;i<finishImage.data.length;i+=4){noiseSeed=(1664525*noiseSeed+1013904223)>>>0;const value=110+(noiseSeed>>>24)/8;finishImage.data[i]=finishImage.data[i+1]=finishImage.data[i+2]=value;finishImage.data[i+3]=255;}finishContext.putImageData(finishImage,0,0);const finish=new T.CanvasTexture(finishCanvas);finish.wrapS=finish.wrapT=T.RepeatWrapping;finish.repeat.set(3,3);M.shell.bumpMap=finish;M.shell.bumpScale=.00008;
 const cache=new Map();function geometry(w,h,d,r=5){r=Math.min(r,w/3,h/3,d/3);const k=[w,h,d,r].join();if(cache.has(k))return cache.get(k);const s=new T.Shape(),x=w/2-r,y=h/2-r;s.moveTo(-x,-y);s.lineTo(x,-y);s.lineTo(x,y);s.lineTo(-x,y);s.closePath();const g=new T.ExtrudeGeometry(s,{depth:d-2*r,steps:1,bevelEnabled:true,bevelSize:r,bevelThickness:r,bevelSegments:4});g.translate(0,0,-d/2+r);g.computeVertexNormals();cache.set(k,g);return g;}
 function mesh(g,m,x,y,z,p=model){const o=new T.Mesh(g,m);o.position.set(x,y,z);o.castShadow=!m.transparent;o.receiveShadow=!m.transparent;p.add(o);return o;}
 function box(w,h,d,x,y,z,m,p=model,r=5){return mesh(geometry(w,h,d,r),m,x,y,z,p);}
 function cyl(r,h,x,y,z,m,p=model,open=false){return mesh(new T.CylinderGeometry(r,r,h,48,1,open),m,x,y,z,p);}
 function label(text,w,h,x,y,z,p=model,color='#34433f'){const c=document.createElement('canvas');c.width=1024;c.height=128;const q=c.getContext('2d');q.fillStyle=color;q.font='500 70px Arial';q.textAlign='center';q.fillText(text,512,88);const t=new T.CanvasTexture(c);t.encoding=T.sRGBEncoding;const o=mesh(new T.PlaneGeometry(w,h),new T.MeshBasicMaterial({map:t,transparent:true,depthWrite:false}),x,y,z,p);o.castShadow=false;return o;}
 // Manufacturer reference envelope: 637 x 900 x 880 mm including fittings.
 box(627,30,775,0,38,-10,M.trim);box(637,72,790,0,863,-10,M.shell);box(637,65,790,0,90,-10,M.shell);
 box(76,710,790,-280,477,-10,M.shell);box(76,710,790,280,477,-10,M.shell);box(490,710,170,0,477,-320,M.shell);
 for(const x of [-265,265])for(const z of [-310,310])cyl(23,30,x,15,z,M.rubber);
 // Open stainless chamber, with no hidden solid box across the door opening.
 box(6,619,570,-237,452,69,M.steel);box(6,619,570,237,452,69,M.steel);box(470,6,570,0,144,69,M.steel);box(470,6,570,0,757,69,M.steel);box(470,619,6,0,452,-214,M.steel);
 for(const x of [-246,246])box(10,635,10,x,452,361,M.rubber);for(const y of [137,767])box(500,10,10,0,y,361,M.rubber);
 for(const x of [-228,228])for(let i=0;i<17;i++)box(8,4,8,x,180+i*31,250,M.trim,model,1);
 // Perforated rear airflow panel, sensor head and service opening.
 box(330,540,7,0,455,-201,M.steel);for(let j=0;j<12;j++)for(let i=0;i<9;i++)box(15,3,1,-128+i*32,220+j*40,-196,M.trim,model,.6);
 const sensor=cyl(17,18,-184,700,-185,M.trim);sensor.rotation.x=Math.PI/2;const port=cyl(22,8,170,695,-190,M.rubber);port.rotation.x=Math.PI/2;
 // Three real perforated shelves on side supports.
 const shelves=[];const shelfYs=[255,425,595];for(let i=0;i<3;i++){const group=new T.Group();group.position.y=shelfYs[i];model.add(group);const s=new T.Shape();s.moveTo(-211,-195);s.lineTo(211,-195);s.lineTo(211,270);s.lineTo(-211,270);s.closePath();for(let row=0;row<15;row++)for(let col=0;col<13;col++){const h=new T.Path();h.absarc(-187+col*31,-174+row*30,5.5,0,Math.PI*2,true);s.holes.push(h);}const g=new T.ExtrudeGeometry(s,{depth:2.5,bevelEnabled:false,curveSegments:10});const m=mesh(g,M.steel,0,0,0,group);m.rotation.x=Math.PI/2;box(422,15,3,0,-7,271,M.steel,group,1);for(const x of [-213,213])box(4,16,465,x,-8,37,M.steel,group,1);label('SHELF '+(i+1),55,7,-161,-7,273,group);shelves.push(group);for(const x of [-229,229])box(16,8,485,x,shelfYs[i]-10,36,M.steel);}
 // Water pan: metal walls and a shallow, purely visual surface.
 box(365,3,330,0,158,35,M.steel);for(const x of [-181,181])box(3,22,330,x,170,35,M.steel);for(const z of [-129,199])box(365,22,3,0,170,z,M.steel);box(358,1,321,0,170,35,M.water);
 // Inner glass door is independently hinged and latched.
 const inner=new T.Group();inner.position.set(-246,140,374);model.add(inner);box(487,617,5,243.5,308.5,0,M.glass,inner,1);for(const x of [0,487])box(7,617,6,x,308.5,0,M.steel,inner,2);for(const y of [0,617])box(487,7,6,243.5,y,0,M.steel,inner,2);box(13,62,16,468,310,12,M.trim,inner,3);for(const y of [70,540])box(14,28,16,0,y,0,M.steel,inner,3);
 // Insulated external door, gasket, and a neutral door-mounted control panel.
 const door=new T.Group();door.position.set(-304,90,404);model.add(door);box(606,796,57,303,398,0,M.shell,door,9);box(563,750,8,303,398,-33,M.rubber,door,5);box(525,705,16,303,398,-41,M.shell,door,6);
 for(const y of [135,667])box(25,50,39,0,y,-7,M.trim,door,4);
 for(const y of [444,620])box(19,19,45,576,y,46,M.trim,door,4);box(23,194,22,576,532,75,M.trim,door,7);
 box(426,108,21,291,683,39,M.trim,door,6);label('CO2 INCUBATOR',190,12,182,696,51,door,'#dce4e0');label('ENVIRONMENT CONTROL',170,7,182,675,51,door,'#a2b6ac');
 const canvas=document.createElement('canvas');canvas.width=480;canvas.height=300;const texture=new T.CanvasTexture(canvas);texture.encoding=T.sRGBEncoding;const screen=mesh(new T.PlaneGeometry(137,86),new T.MeshBasicMaterial({map:texture}),422,683,51,door);screen.castShadow=false;
 label('CONTROLLED CHAMBER',205,9,308,97,30,door);box(27,5,2,370,628,52,M.trim,door,1);
 for(let i=0;i<13;i++)box(34,3,1,268,203+i*13,-408,M.trim);box(55,25,5,200,135,-410,M.trim);cyl(7,13,-180,120,-410,M.steel).rotation.x=Math.PI/2;
 // Six neutral sealed demonstration vessels, with stable IDs across placements.
 const vessels=[];for(let i=0;i<6;i++){const v=new T.Group();v.name='vessel-'+(i+1);model.add(v);cyl(43,15,0,8,0,M.vessel,v,true);cyl(43,2,0,1,0,M.vessel,v);cyl(45,2,0,17,0,M.vessel,v);cyl(44,5,0,15,0,M.vessel,v,true);box(24,1,10,0,18,0,M.blue,v,1);vessels.push(v);}
 const bench=box(225,12,350,470,125,165,M.shell);for(const x of [385,555])for(const z of [25,305])cyl(7,119,x,61,z,M.steel);
 let outerAngle=-1.95,innerAngle=0,lastDisplay='';const positions=shelves.map(()=>0);
 function update(state,dt,{reducedMotion=false}={}){const f=reducedMotion?1:1-Math.exp(-dt*7);outerAngle+=((state.outerOpen?-1.95:0)-outerAngle)*f;innerAngle+=((state.innerOpen?-1.72:0)-innerAngle)*f;door.rotation.y=outerAngle;inner.rotation.y=innerAngle;shelves.forEach((s,i)=>{positions[i]+=((state.shelves[i]?230:0)-positions[i])*f;s.position.z=positions[i];});
 const counts=[0,0,0];let benchIndex=0;state.vessels.forEach((v,i)=>{if(v.shelf===null){const b=benchIndex++;vessels[i].position.set(425+(b%2)*92,132,70+Math.floor(b/2)*98);}else{const n=counts[v.shelf]++;vessels[i].position.set(n?95:-95,shelfYs[v.shelf]+2,55+positions[v.shelf]);}});
 const key=[state.phase,Math.round(state.recovery*100)].join(':');if(key!==lastDisplay){lastDisplay=key;const q=canvas.getContext('2d');q.fillStyle='#123f4a';q.fillRect(0,0,480,300);q.fillStyle='#d3f1ec';q.font='24px Arial';q.fillText('CHAMBER STATUS',24,45);q.font='38px Arial';q.fillText(state.phase.toUpperCase(),24,111);q.font='23px Arial';q.fillText(state.power?'CONTROL ACTIVE':'POWER OFF',24,169);q.fillStyle='#34717a';q.fillRect(24,203,425,16);q.fillStyle='#b2e2d8';q.fillRect(24,203,425*state.recovery,16);q.font='18px Arial';q.fillText('ILLUSTRATIVE RECOVERY',24,267);texture.needsUpdate=true;}}
 // Batch rigid parts by material; moving assemblies keep independent transforms.
 function batchRigid(group){
  for(const child of [...group.children])if(child.isGroup)batchRigid(child);
  const groups=new Map();for(const child of group.children){if(!child.isMesh||child.material.transparent||child.material.isMeshBasicMaterial)continue;const list=groups.get(child.material)||[];list.push(child);groups.set(child.material,list);}
  for(const [material,parts] of groups){if(parts.length<2)continue;const geoms=parts.map(part=>{part.updateMatrix();const g=part.geometry.index?part.geometry.toNonIndexed():part.geometry.clone();return g.applyMatrix4(part.matrix);});const merged=new T.BufferGeometry();for(const name of ['position','normal','uv']){if(!geoms.every(g=>g.getAttribute(name)))continue;const attributes=geoms.map(g=>g.getAttribute(name)),array=new Float32Array(attributes.reduce((n,a)=>n+a.array.length,0));let offset=0;for(const a of attributes){array.set(a.array,offset);offset+=a.array.length;}merged.setAttribute(name,new T.BufferAttribute(array,attributes[0].itemSize));}const combined=new T.Mesh(merged,material);combined.castShadow=true;combined.receiveShadow=true;group.add(combined);parts.forEach(part=>group.remove(part));geoms.forEach(g=>g.dispose());}
 }
 batchRigid(model);
 model.userData={equipment_type:'incubator',units:'metres',referenceEnvelope:{width:.637,height:.900,depth:.880},anchors:{sampleAccess:[0,.45,.4],shelves:shelfYs.map(y=>[0,y/1000,.055])},simulationOnly:true};
 return {model,update,materials:M};
};
})(globalThis);
