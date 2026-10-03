(function(root){
 'use strict';
 root.createLabCentrifuge=function(T){
 const model=new T.Group();model.name='unbranded-refrigerated-centrifuge';model.scale.setScalar(.001);
 const M={shell:new T.MeshStandardMaterial({color:0xe6e7e6,roughness:.34,metalness:.1}),base:new T.MeshStandardMaterial({color:0x969b9e,roughness:.47,metalness:.32}),black:new T.MeshStandardMaterial({color:0x10161b,roughness:.35,metalness:.2}),rubber:new T.MeshStandardMaterial({color:0x181c1e,roughness:.86}),steel:new T.MeshStandardMaterial({color:0xc5ccd0,roughness:.25,metalness:.9}),rotor:new T.MeshStandardMaterial({color:0x1c2329,roughness:.29,metalness:.7}),blue:new T.MeshStandardMaterial({color:0x427a9d,roughness:.35}),cap:new T.MeshStandardMaterial({color:0xb1d9e1,roughness:.26}),tube:new T.MeshPhysicalMaterial({color:0xe5f3ee,roughness:.12,transparent:true,opacity:.65,depthWrite:false}),liquid:new T.MeshStandardMaterial({color:0x55aebf,roughness:.23}),red:new T.MeshStandardMaterial({color:0xad4439,roughness:.32,metalness:.45})};
 for(const material of Object.values(M))material.color.convertSRGBToLinear();
 const cache=new Map();
 function rounded(w,h,d,r=3){r=Math.min(r,w/3,h/3,d/3);const key=[w,h,d,r].join();if(cache.has(key))return cache.get(key);const s=new T.Shape(),x=w/2-r,y=h/2-r;s.moveTo(-x,-y);s.lineTo(x,-y);s.lineTo(x,y);s.lineTo(-x,y);s.closePath();const g=new T.ExtrudeGeometry(s,{depth:d-r*2,bevelEnabled:true,bevelSize:r,bevelThickness:r,bevelSegments:4,steps:1});g.translate(0,0,-d/2+r);g.computeVertexNormals();cache.set(key,g);return g;}
 function mesh(g,m,p,x=0,y=0,z=0,name=''){const o=new T.Mesh(g,m);o.position.set(x,y,z);o.castShadow=m!==M.tube;o.receiveShadow=true;o.name=name;p.add(o);return o;}
 function box(w,h,d,x,y,z,m,p=model,r=3){return mesh(rounded(w,h,d,r),m,p,x,y,z);}
 function cyl(rt,rb,h,x,y,z,m,p=model,n=64,open=false){return mesh(new T.CylinderGeometry(rt,rb,h,n,1,open),m,p,x,y,z);}
 function ring(r,t,x,y,z,m,p=model){const o=mesh(new T.TorusGeometry(r,t,12,96),m,p,x,y,z);o.rotation.x=Math.PI/2;return o;}
 function screw(x,y,z,p=model){const o=cyl(2,2,1.2,x,y,z,M.steel,p,12);o.rotation.x=Math.PI/2;box(2,.4,.2,x,y,z+.7,M.black,p,.1);}
 function rectShape(w,d,r){const s=new T.Shape(),x=-w/2,z=-d/2;s.moveTo(x+r,z);s.lineTo(x+w-r,z);s.quadraticCurveTo(x+w,z,x+w,z+r);s.lineTo(x+w,z+d-r);s.quadraticCurveTo(x+w,z+d,x+w-r,z+d);s.lineTo(x+r,z+d);s.quadraticCurveTo(x,z+d,x,z+d-r);s.lineTo(x,z+r);s.quadraticCurveTo(x,z,x+r,z);return s;}
 function horizontal(s,thickness,y,z,mat,p=model){const g=new T.ExtrudeGeometry(s,{depth:thickness,bevelEnabled:false,curveSegments:64});const o=mesh(g,mat,p,0,y,z);o.rotation.x=Math.PI/2;return o;}
 function decal(text,w,h,x,y,z,p=model,color='#30383e',rotation=0){const c=document.createElement('canvas');c.width=1024;c.height=128;const q=c.getContext('2d');q.fillStyle=color;q.font='500 72px Arial';q.textAlign='center';q.fillText(text,512,90);const tex=new T.CanvasTexture(c);tex.encoding=T.sRGBEncoding;const o=mesh(new T.PlaneGeometry(w,h),new T.MeshBasicMaterial({map:tex,transparent:true,depthWrite:false}),p,x,y,z);o.rotation.x=rotation;o.castShadow=false;return o;}
 // Reference envelope: 290 mm wide, 480 mm deep, 260 mm closed height.
 box(286,25,476,0,25,0,M.base,model,8);box(278,120,12,0,96,231,M.shell,model,4);
 box(11,192,455,-139,133,-2,M.shell,model,4);box(11,192,455,139,133,-2,M.shell,model,4);
 box(268,190,77,0,133,-193,M.shell,model,5);box(263,95,38,0,80,206,M.shell,model,4);
 box(278,7,473,0,39,0,M.rubber,model,2);
 for(const x of [-113,113])for(const z of [-202,202])cyl(12,14,15,x,8,z,M.rubber);
 // Real chamber opening: the top is cut through, not a dark disc on a solid box.
 const top=rectShape(278,390,13),hole=new T.Path();hole.absarc(0,22,113,0,Math.PI*2,true);top.holes.push(hole);horizontal(top,9,235,-49,M.shell);
 cyl(111,107,82,0,191,-27,M.steel,model,96,true);cyl(105,105,5,0,150,-27,M.steel);ring(114,4,0,235,-27,M.rubber);ring(108,2,0,225,-27,M.black);cyl(15,19,21,0,164,-27,M.steel);cyl(5,5,28,0,180,-27,M.steel);
 // Sloped bridge closes the casing between the chamber deck and the display fascia.
 const fasciaBridge=box(277,15,63,0,223,173,M.shell,model,3);fasciaBridge.rotation.x=.38;
 // Control fascia tilts upward toward the user, matching the reference silhouette.
 const panel=new T.Group();panel.position.set(0,184,219);panel.rotation.x=-.66;model.add(panel);
 box(269,71,8,0,0,0,M.shell,panel,4);box(180,26,2,0,14,5,M.black,panel,2);
 const displayCanvas=document.createElement('canvas');displayCanvas.width=1024;displayCanvas.height=160;const displayTexture=new T.CanvasTexture(displayCanvas);displayTexture.encoding=T.sRGBEncoding;
 const display=mesh(new T.PlaneGeometry(174,23),new T.MeshBasicMaterial({map:displayTexture}),panel,0,14,6.1);display.castShadow=false;
 for(const x of [-66,0,66]){const b=cyl(8,8,2,x,-14,5,M.blue,panel,32);b.rotation.x=Math.PI/2;decal('−  +',29,5,x,-30,5.5,panel);}
 const targets=[];function button(text,x,y,r,action){const b=cyl(r,r,3,x,y,6,M.blue,panel,40);b.rotation.x=Math.PI/2;b.userData.action=action;targets.push(b);decal(text,r*1.65,4.2,x,y,8,panel,'#ffffff');}
 button('LID',112,19,9,'lid');button('RUN',112,-13,11,'run');button('STOP',-112,-13,9,'stop');decal('CYCLE',25,4,-67,-2,5.7,panel);decal('TEMP',25,4,0,-2,5.7,panel);decal('SPEED',25,4,66,-2,5.7,panel);
 decal('REFRIGERATED CENTRIFUGE',145,6,0,91,238);decal('24 POSITION ROTOR',100,4,0,77,238);
 for(const x of [-123,123])screw(x,58,239);
 // Recessed side ventilation, rear socket, seam lines, and fasteners.
 for(const side of [-1,1])for(let i=0;i<22;i++){const vent=box(1,3,100,side*145,57+i*6,-152,M.black,model,.3);vent.castShadow=false;}
 for(let i=0;i<31;i++)box(4,4,2,-105+i*7,23,239,M.black,model,.6);
 box(53,31,2,62,77,-234,M.black);box(29,17,6,62,77,-238,M.rubber);box(14,22,4,108,78,-237,M.black);
 const lead=new T.CatmullRomCurve3([new T.Vector3(62,77,-241),new T.Vector3(65,59,-261),new T.Vector3(89,8,-282),new T.Vector3(150,6,-305)]);mesh(new T.TubeGeometry(lead,24,2.7,10,false),M.rubber,model);
 // Lid articulates about its rear hinge. Closed top aligns with the casing.
 const lid=new T.Group();lid.position.set(0,243,-190);model.add(lid);
 box(279,17,340,0,4,165,M.shell,lid,7);box(256,5,318,0,-7,165,M.base,lid,6);cyl(92,94,6,0,-12,163,M.base,lid);ring(91,2,0,-16,163,M.rubber,lid);
 const viewing=cyl(12,12,2,0,14,163,M.black,lid);cyl(6,6,2,0,15.3,163,M.steel,lid);
 for(const x of [-113,113]){box(9,3,280,x,-11,163,M.steel,lid,1);box(10,16,8,x,-16,309,M.steel,lid,1);cyl(4,4,17,x,235,-185,M.steel);}
 for(const x of [-104,104]){const h=cyl(7,7,24,x,243,-183,M.steel);h.rotation.z=Math.PI/2;}
 box(68,4,6,0,0,336,M.rubber,lid,1);
 // Removable fixed-angle rotor, pierced deck, central hub and individual wells.
 const rotorDock=new T.Group();rotorDock.position.set(0,0,-27);model.add(rotorDock);const rotor=new T.Group();rotorDock.add(rotor);
 cyl(93,69,39,0,188,0,M.rotor,rotor,96,true);cyl(68,68,5,0,167,0,M.rotor,rotor);ring(92,1.6,0,209,0,M.steel,rotor);
 const rotorShape=new T.Shape();rotorShape.absarc(0,0,92,0,Math.PI*2,false);for(let i=0;i<24;i++){const a=i*Math.PI/12;const h=new T.Path();h.absarc(Math.sin(a)*75,Math.cos(a)*75,5.7,0,Math.PI*2,true);rotorShape.holes.push(h);}horizontal(rotorShape,5,209,0,M.rotor,rotor);
 cyl(31,48,28,0,197,0,M.rotor,rotor);cyl(14,17,17,0,219,0,M.steel,rotor);cyl(5,5,5,0,230,0,M.black,rotor,6);
 const tubes=[],slotTargets=[];for(let i=0;i<24;i++){const a=i*Math.PI/12,x=Math.sin(a)*75,z=Math.cos(a)*75;
 const sleeve=cyl(5.7,4.7,30,x,193,z,M.rubber,rotor,24,true);sleeve.userData.slot=i;slotTargets.push(sleeve);
 const g=new T.Group();g.position.set(x,208,z);g.name='tube-slot-'+(i+1);rotor.add(g);cyl(4.5,3.7,24,0,-8,0,M.tube,g,24);cyl(3.7,.8,11,0,-25,0,M.tube,g,24);cyl(3.3,1,13,0,-21,0,M.liquid,g,24);const cap=cyl(5.7,5.7,4,0,7,0,M.cap,g,24);cap.userData.slot=i;slotTargets.push(cap);box(3,2,7,0,7,-5,M.cap,g,1);tubes.push(g);
 const number=decal(String(i+1),6,3.3,Math.sin(a)*86,210,Math.cos(a)*86,rotor,'#d9dee0',-Math.PI/2);number.rotation.z=-a;
 }
 const cover=new T.Group();rotor.add(cover);cyl(91,91,6,0,223,0,M.rotor,cover);cyl(35,81,9,0,230,0,M.rotor,cover);cyl(18,18,15,0,242,0,M.red,cover,12);ring(88,1.1,0,227,0,M.steel,cover);
 const coverTray=new T.Group();coverTray.position.set(-235,-221,25);model.add(coverTray);
 // A compact bench pad visually supports removed hardware.
 const pad=box(210,3,215,237,2,35,M.base);pad.visible=false;const coverPad=box(205,3,205,-235,2,25,M.base);
 let lidAngle=-1.52,rotorLift=0,rotorX=0,coverLift=0,coverX=0,spin=0,lastDisplay='';
 function update(state,dt,{cutaway=false,reducedMotion=false}={}){const f=reducedMotion?1:1-Math.exp(-dt*8);lidAngle+=((state.lidOpen?-1.52:0)-lidAngle)*f;lid.rotation.x=lidAngle;
 rotorX+=((state.rotorInstalled?0:237)-rotorX)*f;rotorDock.position.x=rotorX;rotorDock.position.y=state.rotorInstalled?0:-164;rotorDock.position.z=state.rotorInstalled?-27:35;pad.visible=!state.rotorInstalled;coverPad.visible=!state.coverInstalled;
 if(state.rotorInstalled&&state.speed>0)spin+=dt*state.speed*7;rotor.rotation.y=spin;
 if(cover.parent!== (state.coverInstalled?rotor:coverTray)){(state.coverInstalled?rotor:coverTray).add(cover);}coverTray.visible=!state.coverInstalled;
 tubes.forEach((g,i)=>g.visible=state.slots[i]!==null);cover.visible=true;lid.visible=!(cutaway&&state.running);
 const txt=[state.phase,Math.ceil(Math.max(0,state.duration-state.elapsed)),state.loaded].join(':');if(txt!==lastDisplay){lastDisplay=txt;const q=displayCanvas.getContext('2d');q.fillStyle='#bad2d4';q.fillRect(0,0,1024,160);q.fillStyle='#213f43';q.font='54px monospace';q.fillText(state.running?'RUN':state.phase==='complete'?'DONE':'IDLE',30,75);q.fillText('20°',455,75);q.fillText(state.running?'SIM':'STOP',740,75);q.font='23px monospace';q.fillText(state.running?'DEMO CYCLE':'STANDBY',30,132);q.fillText('ROOM',450,132);q.fillText(state.loaded+' TUBES',740,132);displayTexture.needsUpdate=true;}
 }
 update({lidOpen:true,rotorInstalled:true,coverInstalled:false,slots:Array(24).fill(null),running:false,speed:0,phase:'idle',duration:12,elapsed:0,loaded:0},1);
 model.userData={equipment_type:'centrifuge',units:'metres',referenceEnvelope:{width:.290,depth:.480,height:.260},anchors:{chamber:[0,.235,-.027],sampleAccess:[0,.28,-.027],front:[0,.18,.24]},simulationOnly:true};
 return {model,update,targets,slotTargets,materials:M};
 };
})(globalThis);
