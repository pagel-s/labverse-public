(function(root){
'use strict';
root.createLabBiosafetyCabinet=function(T){
 const model=new T.Group();model.name='unbranded-class-ii-cabinet';model.scale.setScalar(.001);
 const M={shell:new T.MeshStandardMaterial({color:0xd4d8d4,roughness:.43,metalness:.2}),steel:new T.MeshStandardMaterial({color:0xa6b2b2,roughness:.28,metalness:.85}),trim:new T.MeshStandardMaterial({color:0x303b3c,roughness:.55}),accent:new T.MeshStandardMaterial({color:0x6d8d97,roughness:.45,metalness:.25}),glass:new T.MeshPhysicalMaterial({color:0xbcd5d1,roughness:.07,transparent:true,opacity:.10,clearcoat:1,side:T.DoubleSide,depthWrite:false}),light:new T.MeshStandardMaterial({color:0xdce2d8,emissive:0xe9fff1,emissiveIntensity:1}),green:new T.MeshStandardMaterial({color:0x799e8b,emissive:0x387754,emissiveIntensity:.5})};for(const material of Object.values(M))material.color.convertSRGBToLinear();
 const cache=new Map();function geom(w,h,d,r=3){r=Math.min(r,w/3,h/3,d/3);const key=[w,h,d,r].join();if(cache.has(key))return cache.get(key);const s=new T.Shape(),x=w/2-r,y=h/2-r;s.moveTo(-x,-y);s.lineTo(x,-y);s.lineTo(x,y);s.lineTo(-x,y);s.closePath();const g=new T.ExtrudeGeometry(s,{depth:d-2*r,bevelEnabled:true,bevelSize:r,bevelThickness:r,bevelSegments:3,steps:1});g.translate(0,0,-d/2+r);cache.set(key,g);return g;}
 function mesh(g,m,x,y,z,p=model){const o=new T.Mesh(g,m);o.position.set(x,y,z);o.castShadow=o.receiveShadow=!m.transparent;p.add(o);return o;}
 function box(w,h,d,x,y,z,m,p=model,r=3){return mesh(geom(w,h,d,r),m,x,y,z,p);}
 function cylinder(radius,h,x,y,z,m,p=model){return mesh(new T.CylinderGeometry(radius,radius,h,32),m,x,y,z,p);}
 function label(text,w,h,x,y,z,p=model,color='#3e5352'){const c=document.createElement('canvas');c.width=1024;c.height=128;const q=c.getContext('2d');q.fillStyle=color;q.font='500 66px Arial';q.textAlign='center';q.fillText(text,512,85);const map=new T.CanvasTexture(c);map.encoding=T.sRGBEncoding;return mesh(new T.PlaneGeometry(w,h),new T.MeshBasicMaterial({map,transparent:true,depthWrite:false}),x,y,z,p);}
 // Adjustable stand: telescoping uprights, rails, screws and levelling feet.
 for(const x of [-581,581]){for(const z of [-290,280]){box(43,535,43,x,351,z,M.shell);box(31,230,31,x,135,z,M.steel);cylinder(23,18,x,9,z,M.trim);for(let i=0;i<5;i++){const bolt=cylinder(3,2,x,255+i*26,z+23,M.trim);bolt.rotation.x=Math.PI/2;}}box(49,38,660,x,76,-5,M.shell);box(40,43,622,x,590,-5,M.shell);}box(1120,45,35,0,190,-290,M.shell);box(1140,42,35,0,590,-290,M.shell);
 // 1300 x 1568 x 798 mm cabinet envelope, above a generic 650 mm stand.
 const sideShape=new T.Shape();sideShape.moveTo(-399,650);sideShape.lineTo(399,650);sideShape.lineTo(399,850);sideShape.lineTo(261,1630);sideShape.lineTo(261,2218);sideShape.lineTo(-399,2218);sideShape.closePath();const sg=new T.ExtrudeGeometry(sideShape,{depth:40,bevelEnabled:false});sg.rotateY(-Math.PI/2);for(const x of [-610,650])mesh(sg,M.shell,x,0,0);
 box(1220,1560,18,0,1434,-390,M.shell);box(1298,18,656,0,2208,-70,M.shell);box(1215,555,650,0,1930,-69,M.shell);
 // Fine upper-panel seams, screws and top exhaust grille.
 box(1180,2,2,0,1646,258,M.trim,model,.4);box(1230,7,5,0,1664,263,M.accent,model,1);
 for(const x of [-591,591])for(const y of [1700,2160]){const screw=cylinder(3.5,2,x,y,258,M.steel);screw.rotation.x=Math.PI/2;}
 box(490,12,270,0,2223,-75,M.steel);for(let i=0;i<25;i++)box(5,2,230,-228+i*19,2231,-75,M.trim,model,.5);
 // Stainless chamber and solid work surface, framed by intake grilles.
 box(1200,7,618,0,842,38,M.steel);box(1200,760,6,0,1230,-270,M.steel);
 for(const x of [-602,602])box(5,779,627,x,1240,43,M.steel);
 box(1238,95,790,0,700,0,M.shell);box(1240,22,95,0,813,347,M.steel);
 for(const z of [-253,316]){box(1186,3,49,0,851,z,M.trim,model,1);for(let i=0;i<120;i++)box(5,5,48,-592+i*9.95,854,z,M.steel,model,.6);}
 box(1160,4,505,0,854,28,M.steel,model,2);for(const x of [-595,595])box(11,15,612,x,857,35,M.steel);
 for(const x of [-360,360])box(212,19,65,x,858,390,M.trim,model,5);
 // Sloped front rails; a retracting glass sash disappears behind the upper fascia.
 for(const x of [-625,625]){const rail=box(36,795,33,x,1240,330,M.accent);rail.rotation.x=-Math.atan(138/780);const guide=box(6,790,7,x+(x<0?20:-20),1240,347,M.trim);guide.rotation.x=rail.rotation.x;}
 const fascia=box(1210,152,42,0,1580,295,M.shell,model,6);
 const sash=new T.Group();sash.position.set(0,850,393);sash.rotation.x=-Math.atan(138/780);model.add(sash);box(1190,790,5,0,395,0,M.glass,sash,1);box(1197,19,20,0,7,8,M.steel,sash,3);for(const x of [-230,230]){box(90,14,28,x,13,26,M.trim,sash,4);}for(const x of [-594,594])box(5,786,5,x,395,0,M.steel,sash,1);
 // Work lamp, service ports and generic switched receptacles.
 box(1000,16,55,0,1603,170,M.trim);box(970,5,42,0,1592,170,M.light);const workLight=new T.PointLight(0xe5f7ef,.65,1.3,2);workLight.position.set(0,1510,40);model.add(workLight); // intensity adjusted below; scene uses metre root.
 for(const x of [-430,430]){box(76,49,9,x,1090,-263,M.steel);for(const dx of [-16,16]){box(2.5,9,1,x+dx-4,1095,-257,M.trim,model,.2);box(2.5,9,1,x+dx+4,1095,-257,M.trim,model,.2);box(4,3,1,x+dx,1081,-257,M.trim,model,.2);}}
 for(const x of [-609,609]){const port=cylinder(32,8,x,1150,165,M.trim);port.rotation.z=Math.PI/2;const cap=cylinder(25,10,x+(x<0?4:-4),1150,165,M.steel);cap.rotation.z=Math.PI/2;}
 const utilityLed=box(7,4,1,482,1090,-257,M.green,model,1);
 // Display stays readable and uses generic status rather than real operating values.
 box(306,62,5,355,1586,319,M.accent,model,3);const screenCanvas=document.createElement('canvas');screenCanvas.width=640;screenCanvas.height=180;const displayMap=new T.CanvasTexture(screenCanvas);displayMap.encoding=T.sRGBEncoding;mesh(new T.PlaneGeometry(175,49),new T.MeshBasicMaterial({map:displayMap}),297,1586,323);
 for(let i=0;i<5;i++)box(16,16,3,415+i*19,1591,323,M.shell,model,2);label('AIR   LIGHT   AUX',110,7,452,1569,324);label('BIOLOGICAL SAFETY CABINET',380,20,-370,1720,259);label('CLASS II · REFERENCE MODEL',272,12,-423,1688,259);
 // Symbolic arrows are deliberately separate from physical geometry.
 const flow=new T.Group();flow.name='illustrative-airflow-overlay';model.add(flow);const arrowMaterial=new T.MeshBasicMaterial({color:0x458c9a,transparent:true,opacity:.65,depthWrite:false});for(const x of [-430,-215,0,215,430]){mesh(new T.CylinderGeometry(2,2,220,8),arrowMaterial,x,1300,25,flow);mesh(new T.ConeGeometry(14,35,12),arrowMaterial,x,1175,25,flow).rotation.z=Math.PI;}flow.visible=false;
 // Merge rigid opaque meshes per assembly to keep rendering inexpensive.
 function batch(group){for(const c of [...group.children])if(c.isGroup)batch(c);const buckets=new Map();for(const c of group.children){if(!c.isMesh||c.material.transparent||c.material.isMeshBasicMaterial||c===utilityLed)continue;const a=buckets.get(c.material)||[];a.push(c);buckets.set(c.material,a);}for(const [material,parts] of buckets){if(parts.length<2)continue;const gs=parts.map(p=>{p.updateMatrix();return(p.geometry.index?p.geometry.toNonIndexed():p.geometry.clone()).applyMatrix4(p.matrix);});const g=new T.BufferGeometry();for(const key of ['position','normal','uv']){const attrs=gs.map(g=>g.getAttribute(key));if(attrs.some(a=>!a))continue;const array=new Float32Array(attrs.reduce((n,a)=>n+a.array.length,0));let at=0;for(const a of attrs){array.set(a.array,at);at+=a.array.length;}g.setAttribute(key,new T.BufferAttribute(array,attrs[0].itemSize));}const merged=new T.Mesh(g,material);merged.castShadow=merged.receiveShadow=true;group.add(merged);parts.forEach(p=>group.remove(p));gs.forEach(g=>g.dispose());}}
 batch(model);let travel=200,lastDisplay='';
 function update(state,dt,{reducedMotion=false}={}){const target={closed:0,work:200,service:530}[state.sash],f=reducedMotion?1:1-Math.exp(-dt*7);travel+=(target-travel)*f;sash.position.y=850+travel;sash.position.z=393-travel*138/780;M.light.emissiveIntensity=state.light?1.3:0;workLight.intensity=state.light?1.1:0;utilityLed.visible=state.outlets;flow.visible=state.flowOverlay&&state.blower;const key=[state.status,state.sash,state.light,state.outlets].join(':');if(key!==lastDisplay){lastDisplay=key;const q=screenCanvas.getContext('2d');q.fillStyle='#162c2c';q.fillRect(0,0,640,180);q.fillStyle=state.status==='sash warning'?'#ebbd78':'#bfe3bb';q.font='36px Arial';q.fillText(state.status.toUpperCase(),22,66);q.font='23px Arial';q.fillText('SASH '+state.sash.toUpperCase(),22,116);q.font='17px Arial';q.fillText('ILLUSTRATIVE STATUS',22,151);displayMap.needsUpdate=true;}}
 model.userData={equipment_type:'biosafety_cabinet',units:'metres',referenceEnvelope:{width:1.3,height:1.568,depth:.798},standHeight:.65,anchors:{workSurface:[0,.858,.028],frontAccess:[0,1.05,.36]},simulationOnly:true};return{model,update,materials:M};
};
})(globalThis);
