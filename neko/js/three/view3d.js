import * as THREE from '../../assets/vendor/three/three.module.min.js';
import { createCharacter3D } from './characters3d.js';
import { createTextureKit } from './textures3d.js';

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

// The same scene controller owns the camera, room props, and every invited model.
// Character actions remain in the application; this view reads their current state.
export class Room3D {
  constructor(container, app) {
    this.container = container;
    this.app = app;
    this.models = new Map();
    this.props = new Map();
    this.materials = new Map();
    this.surfaceMaterials = new Map();
    this.geometries = new Map();
    this.textures = new Set();
    this.active = false;
    this.frame = null;
    this.yaw = 0;
    this.pitch = .28;
    this.zoom = 1;
    this.pointers = new Map();
    this.motionPreference = matchMedia('(prefers-reduced-motion: reduce)');
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#e9ddcb');
    this.scene.fog = new THREE.Fog('#e9ddcb', 24, 48);
    this.camera = new THREE.OrthographicCamera(-7, 7, 4, -4, .1, 90);
    const surface=document.createElement('canvas');
    const context=surface.getContext('webgl2',{antialias:true,alpha:false,powerPreference:'low-power'});
    if(!context)throw new Error('WebGL 2 is unavailable');
    this.renderer = new THREE.WebGLRenderer({ canvas:surface,context,antialias:true,alpha:false,powerPreference:'low-power' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1;
    const canvas = this.renderer.domElement;
    canvas.className = 'room-three-canvas';
    canvas.setAttribute('aria-label', '立體公寓，點喵喵或朋友互動，拖曳轉動視角');
    canvas.setAttribute('role', 'img');
    canvas.style.touchAction = 'none';
    canvas.style.display = 'block';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    container.append(canvas);
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.vector = new THREE.Vector3();
    try {
      this.textureKit = createTextureKit({}, 'room');
      this.buildRoom();
      this.bindInput();
      this.observer = new ResizeObserver(() => this.resize());
      this.observer.observe(container);
      this.resize();
    } catch(error) {
      this.dispose();
      throw error;
    }
  }

  material(color, options = {}) {
    const key = JSON.stringify([color, options]);
    const {surface, ...properties} = options;
    const surfaceOptions = surface === 'wood' ? {bumpScale:.003,...properties} : properties;
    const materials = surface ? this.surfaceMaterials : this.materials;
    if (!materials.has(key)) materials.set(key, surface
      ? this.textureKit.material(surface, color, surfaceOptions)
      : new THREE.MeshStandardMaterial({color, roughness:.84, ...properties}));
    return materials.get(key);
  }

  geometry(kind, values) {
    const key = `${kind}:${values.join(',')}`;
    if (!this.geometries.has(key)) {
      let geometry;
      if (kind === 'ball') geometry = new THREE.SphereGeometry(1, 24, 16);
      if (kind === 'cylinder') geometry = new THREE.CylinderGeometry(...values, 64);
      if (kind === 'torus') geometry = new THREE.TorusGeometry(...values, 10, 48);
      if (kind === 'box') {
        const [w,h,d,r] = values;
        geometry = new THREE.BoxGeometry(w,h,d,8,8,8);
        const pos = geometry.attributes.position;
        const normal = geometry.attributes.normal;
        const p = new THREE.Vector3();
        const core = new THREE.Vector3();
        const inner = new THREE.Vector3(w/2-r,h/2-r,d/2-r);
        const minimum = inner.clone().negate();
        for (let i=0; i<pos.count; i++) {
          p.fromBufferAttribute(pos,i);
          core.copy(p).clamp(minimum,inner);
          p.sub(core).normalize();
          // Analytic normals agree across duplicated face vertices. Recomputing
          // face normals here makes every rounded corner look cracked/faceted.
          normal.setXYZ(i,p.x,p.y,p.z);
          p.multiplyScalar(r).add(core);
          pos.setXYZ(i,p.x,p.y,p.z);
        }
      }
      this.geometries.set(key,geometry);
    }
    return this.geometries.get(key);
  }

  mesh(geometry, color, parent, position = [0,0,0], options = {}) {
    const mesh = new THREE.Mesh(geometry,this.material(color,options));
    mesh.position.set(...position);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    parent.add(mesh);
    return mesh;
  }

  box(parent, color, position, size, radius=.08, options={}) {
    return this.mesh(this.geometry('box',[...size,Math.min(radius,...size.map(v=>v/2-.001))]),color,parent,position,options);
  }

  ball(parent, color, position, size, options={}) {
    const mesh = this.mesh(this.geometry('ball',[]),color,parent,position,options);
    mesh.scale.set(...size);
    return mesh;
  }

  cylinder(parent,color,position,radius,height,options={}) {
    return this.mesh(this.geometry('cylinder',[radius,radius,height]),color,parent,position,options);
  }

  group(position=[0,0,0], parent=this.scene) {
    const group = new THREE.Group();
    group.position.set(...position);
    parent.add(group);
    return group;
  }

  prop(id, group, anchor) {
    group.traverse(node => { if(node.isMesh) node.userData.propId = id; });
    group.userData.anchor = new THREE.Vector3(...anchor);
    this.props.set(id,group);
  }

  canvasTexture(width,height,draw) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    draw(canvas.getContext('2d'));
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    this.textures.add(texture);
    return texture;
  }

  buildRoom() {
    this.hemisphere = new THREE.HemisphereLight('#eaf0f8','#b5aca1',2);
    this.scene.add(this.hemisphere);
    this.sun = new THREE.DirectionalLight('#fff7ed',1.85);
    this.sun.position.set(-3.5,8,6);
    this.sun.castShadow = true;
    const shadowSize = innerWidth > 900 ? 2048 : 1024;
    this.sun.shadow.mapSize.set(shadowSize,shadowSize);
    Object.assign(this.sun.shadow.camera,{left:-7.5,right:7.5,top:7,bottom:-6,near:.5,far:24});
    this.sun.shadow.normalBias = .035;
    this.sun.shadow.bias = -.0003;
    this.sun.shadow.radius = 3;
    this.scene.add(this.sun);
    this.fill = new THREE.DirectionalLight('#f0f3f7',.85);
    this.fill.position.set(3,3,8);
    this.scene.add(this.fill);
    this.lampLight = new THREE.PointLight('#ffbf78',1.2,12,1.6);
    this.lampLight.position.set(3.9,3.8,-2.4);
    this.scene.add(this.lampLight);

    const wood=this.canvasTexture(512,512,ctx=>{
      ctx.fillStyle='#c4a27d';ctx.fillRect(0,0,512,512);
      for(let i=0;i<4;i++){
        ctx.fillStyle=i%2?'#c6a680':'#c1a17d';ctx.fillRect(0,i*128,512,128);
        ctx.fillStyle='#ac8d6c';ctx.fillRect(0,i*128,512,2);ctx.fillRect(i%2?256:0,i*128,2,128);
        ctx.strokeStyle='rgba(135,98,57,.08)';ctx.lineWidth=1;
        for(let j=0;j<9;j++){ctx.beginPath();ctx.moveTo(0,i*128+j*13+7);ctx.bezierCurveTo(120,i*128+j*13+4,350,i*128+j*13+14,512,i*128+j*13+8);ctx.stroke();}
      }
    });
    wood.wrapS=wood.wrapT=THREE.RepeatWrapping;wood.repeat.set(9,12);
    const floor=this.box(this.scene,'#ffffff',[0,-.18,0],[40,.35,36],.05);
    floor.material=new THREE.MeshStandardMaterial({map:wood,roughness:.86});
    this.materials.set('wood-floor',floor.material);
    this.box(this.scene,'#e9e4d9',[0,4,-5],[30,8,.25],.02,{surface:'wall',roughness:1,bumpScale:.008});
    this.box(this.scene,'#bfa887',[0,.17,-4.79],[30,.34,.14],.02,{surface:'wood'});
    this.box(this.scene,'#dedbd1',[-7.4,4,0],[.25,8,10],.02,{surface:'wall',roughness:1,bumpScale:.008});
    // A single procedural sky is authored locally, never fetched as an image.
    const sky = this.canvasTexture(256,256,ctx=>{
      const gradient=ctx.createLinearGradient(0,0,0,256);
      gradient.addColorStop(0,'#9cbec5');gradient.addColorStop(.65,'#e7ccb2');gradient.addColorStop(1,'#f6d5a0');
      ctx.fillStyle=gradient;ctx.fillRect(0,0,256,256);
      ctx.fillStyle='#fff0b5';ctx.beginPath();ctx.arc(175,104,28,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#89999a';
      [58,85,48,72,109,65,50,90,54,64,81].forEach((h,i)=>ctx.fillRect(i*25,256-h,23,h));
      ctx.fillStyle='#f3d3a4';for(let x=8;x<256;x+=25)for(let y=202;y<249;y+=16)ctx.fillRect(x,y,5,7);
    });
    const skyMaterial = new THREE.MeshBasicMaterial({map:sky});
    this.materials.set('window-sky',skyMaterial);
    const windowGlass = new THREE.Mesh(new THREE.PlaneGeometry(3.25,2.8),skyMaterial);
    this.geometries.set('window-plane',windowGlass.geometry);
    windowGlass.position.set(-3.15,3.5,-4.81);
    this.scene.add(windowGlass);
    const frame='#fcf2dc';
    [-4.87,-1.43].forEach(x=>this.box(this.scene,frame,[x,3.5,-4.6],[.18,3.04,.27],.04));
    [2.03,4.98].forEach(y=>this.box(this.scene,frame,[-3.15,y,-4.6],[3.63,.17,.28],.035));
    this.box(this.scene,frame,[-3.15,3.5,-4.59],[.1,2.84,.2],.02);
    this.box(this.scene,frame,[-3.15,3.35,-4.59],[3.3,.1,.2],.02);
    this.box(this.scene,'#c2a37c',[-3.15,1.94,-4.44],[3.85,.15,.65],.05,{surface:'wood'});
    // Pleated curtains are low polygon cloth volumes, with a visible wooden rail.
    this.box(this.scene,'#887354',[-3.15,5.23,-4.33],[4.1,.08,.08],.03,{surface:'wood'});
    [-5.03,-1.26].forEach(x=>{
      for(let i=0;i<4;i++) this.ball(this.scene,i%2?'#dfd7c5':'#eee5d4',[x+i*.09,3.6,-4.35],[.13,1.5,.13],{surface:'cloth',bumpScale:.012});
    });
    // The softly rounded furniture shares geometry/material factories.
    const sofa=this.group([.4,0,-2.75]);
    [-2.2,2.2].forEach(x=>[-.6,.58].forEach(z=>this.box(sofa,'#735741',[x,.26,z],[.17,.52,.17],.04,{surface:'wood'})));
    this.box(sofa,'#687d73',[0,.66,0],[5.6,.7,1.8],.23,{surface:'cloth',bumpScale:.018});
    this.box(sofa,'#84968a',[0,1.52,-.63],[5.4,1.54,.62],.26,{surface:'cloth',bumpScale:.018});
    [-1.32,1.32].forEach(x=>this.box(sofa,'#94a497',[x,1,.1],[2.58,.3,1.44],.14,{surface:'cloth',bumpScale:.018}));
    [-2.74,2.74].forEach(x=>this.box(sofa,'#788e80',[x,1.14,0],[.54,.97,1.9],.25,{surface:'cloth',bumpScale:.018}));
    const pillow=this.box(sofa,'#d4b58e',[-1.95,1.55,-.3],[.85,.8,.3],.14,{surface:'knit',bumpScale:.022});pillow.rotation.z=.22;
    const pillow2=this.box(sofa,'#ddd9c8',[1.75,1.55,-.3],[.85,.8,.3],.14,{surface:'cloth',bumpScale:.018});pillow2.rotation.z=-.17;

    // Elliptical rug and a small foreground tea table leave the cast unobstructed.
    const rug=this.cylinder(this.scene,'#cbb89c',[0,.025,.75],1,.035,{surface:'knit',bumpScale:.006});rug.scale.set(4.8,1,2.9);
    const rugInner=this.cylinder(this.scene,'#ded3bc',[0,.046,.75],1,.014,{surface:'knit',bumpScale:.006});rugInner.scale.set(4.57,1,2.7);
    const table=this.group([2.6,0,3.45]);
    [[-.8,-.32],[.8,-.32],[0,.5]].forEach(([x,z])=>this.box(table,'#7f6244',[x,.41,z],[.17,.82,.17],.03,{surface:'wood'}));
    const tabletop=this.cylinder(table,'#ac865d',[0,.84,0],1,.15,{surface:'wood',roughness:.65,bumpScale:.003});tabletop.scale.set(1.58,1,.82);
    const cup=this.cylinder(table,'#e9e2ce',[-.6,1.05,0],.2,.3);
    this.cylinder(table,'#786046',[-.6,1.205,0],.155,.012);
    const cuphandle=this.mesh(this.geometry('torus',[.13,.035]),'#e9e2ce',table,[-.81,1.04,0]);
    this.box(table,'#94a89e',[.46,.97,0],[.65,.09,.43],.02).rotation.y=-.18;
    this.box(table,'#ede3ce',[.46,1.02,0],[.59,.02,.38],.01).rotation.y=-.18;

    this.buildPlant([-5.5,0,-2.4]);
    this.buildRadio([-4,0,.6]);
    this.buildFan([3.95,0,.15]);
    this.buildLamp([3.8,0,-3.2]);
    // A compact framed drawing and wall clock give the room a lived-in scale.
    this.box(this.scene,'#b49269',[1.5,3.55,-4.67],[1.35,1.56,.16],.04,{surface:'wood'});
    this.box(this.scene,'#f1e5cc',[1.5,3.55,-4.56],[1.15,1.34,.04],.01);
    this.ball(this.scene,'#c69d68',[1.44,3.62,-4.51],[.3,.3,.015]);
    this.box(this.scene,'#9daa90',[1.58,3.28,-4.49],[.55,.12,.02],.02);
    const clock=this.cylinder(this.scene,'#aa8760',[4.35,4.48,-4.69],.46,.13);clock.rotation.x=Math.PI/2;
    const clockFace=this.cylinder(this.scene,'#f8efd8',[4.35,4.48,-4.61],.395,.025);clockFace.rotation.x=Math.PI/2;
    this.box(this.scene,'#71614f',[4.34,4.59,-4.57],[.025,.24,.015],.008);
    const clockHand=this.box(this.scene,'#71614f',[4.48,4.46,-4.56],[.29,.025,.015],.008);clockHand.rotation.z=-.25;

    const contact=this.canvasTexture(128,128,ctx=>{
      const gradient=ctx.createRadialGradient(64,64,2,64,64,64);
      gradient.addColorStop(0,'rgba(48,34,18,.36)');gradient.addColorStop(.4,'rgba(48,34,18,.22)');gradient.addColorStop(1,'rgba(48,34,18,0)');
      ctx.fillStyle=gradient;ctx.fillRect(0,0,128,128);
    });
    this.contactMaterial = new THREE.MeshBasicMaterial({map:contact,transparent:true,depthWrite:false});
    this.materials.set('contact-shadow',this.contactMaterial);
    this.contactGeometry=new THREE.PlaneGeometry(2.7,2.25);
    this.geometries.set('contact-plane',this.contactGeometry);
    this.contacts = new Map();
  }

  buildPlant(position) {
    const plant=this.group(position);
    this.mesh(this.geometry('cylinder',[.48,.35,.73]),'#b78560',plant,[0,.38,0]);
    this.cylinder(plant,'#765542',[0,.75,0],.42,.025);
    for(let i=0;i<7;i++) {
      const angle=i*2.4;
      const x=Math.sin(angle)*.55,z=Math.cos(angle)*.38,y=1.55+(i%3)*.38;
      const stem=this.cylinder(plant,'#778563',[x*.48,(y+.6)/2,z*.48],.026,y-.6);
      stem.rotation.z=-x*.4;
      const leaf=this.ball(plant,i%2?'#839574':'#647e64',[x,y,z],[.26,.63,.075]);
      leaf.rotation.set(.15,angle,x>0?-.55:.55);
    }
  }

  buildRadio(position) {
    const group=this.group(position);
    this.box(group,'#8c6d49',[0,.31,0],[1.5,.62,.88],.11,{surface:'wood'});
    this.box(group,'#b6946f',[0,.87,0],[1.17,.62,.57],.1,{surface:'wood'});
    this.box(group,'#6e6957',[-.25,.87,.29],[.51,.43,.035],.07);
    for(let i=0;i<5;i++) this.box(group,'#aca385',[-.25,.72+i*.073,.319],[.43,.015,.014],.005);
    this.box(group,'#dfdbc2',[.3,1.02,.3],[.32,.1,.026],.015);
    const dial=this.cylinder(group,'#eee3c7',[.32,.8,.33],.09,.06);dial.rotation.x=Math.PI/2;
    this.box(group,'#81735d',[0,1.32,0],[.48,.06,.06],.02);
    [-.25,.25].forEach(x=>this.box(group,'#81735d',[x,1.25,0],[.05,.19,.06],.02));
    this.prop('radio',group,[0,.9,.32]);
  }

  buildFan(position) {
    const group=this.group(position);
    this.ball(group,'#6f9593',[0,.13,0],[.58,.13,.38]);
    this.cylinder(group,'#719593',[0,.7,0],.065,1.2);
    const guard=this.mesh(this.geometry('torus',[.52,.034]),'#88aba5',group,[0,1.64,.09]);
    this.ball(group,'#8badab',[0,1.64,-.09],[.26,.26,.25]);
    const blades=this.group([0,1.64,.12],group);
    for(let i=0;i<3;i++) {
      const angle=i*Math.PI*2/3;
      const blade=this.ball(blades,'#bacac0',[Math.sin(angle)*.25,Math.cos(angle)*.25,0],[.15,.29,.028]);
      blade.rotation.z=-angle;
    }
    this.fanBlades=blades;
    for(let i=0;i<4;i++) {
      const spoke=this.box(group,'#95b2aa',[0,1.64,.18],[.019,1.02,.019],.008);
      spoke.rotation.z=i*Math.PI/4;
    }
    this.ball(group,'#769c97',[0,1.64,.21],[.1,.1,.035]);
    this.prop('fan',group,[0,1.64,.23]);
  }

  buildLamp(position) {
    const group=this.group(position);
    this.cylinder(group,'#827453',[0,.1,0],.48,.16);
    this.cylinder(group,'#96825b',[0,1.85,0],.038,3.7);
    this.lampShade=this.mesh(this.geometry('cylinder',[.48,.8,.9]),'#e8d6b3',group,[0,3.72,0],{surface:'cloth',bumpScale:.008,emissive:'#f4b34d',emissiveIntensity:.12,side:THREE.DoubleSide});
    this.ball(group,'#fbdf98',[0,3.29,0],[.18,.15,.18],{emissive:'#ffc96b',emissiveIntensity:.25});
    this.prop('lamp',group,[0,3.7,0]);
  }

  syncActors(actors) {
    this.actors = actors;
    for(const [id,model] of this.models) if(!actors.has(id)) {
      this.scene.remove(model.group);model.dispose();this.models.delete(id);
      this.scene.remove(this.contacts.get(id));this.contacts.delete(id);
    }
    for(const [id] of actors) if(!this.models.has(id)) {
      const model=createCharacter3D(id);
      model.group.traverse(node=>{if(node.isMesh)node.userData.actorId=id;});
      this.models.set(id,model);this.scene.add(model.group);
      const contact=new THREE.Mesh(this.contactGeometry,this.contactMaterial);
      contact.rotation.x=-Math.PI/2;contact.position.y=.063;
      this.contacts.set(id,contact);this.scene.add(contact);
    }
    this.placeModels();
    this.render();
  }

  placeModels() {
    const narrow=this.width/this.height<.8;
    for(const [id,model] of this.models) {
      const x=id==='yani' ? (narrow?-.9:-1.2) : (narrow?.94:1.35);
      model.group.position.set(x,.065,id==='yani'?.7:.48);
      model.group.rotation.y=id==='yani'?.09:-.1;
      const scale=narrow?.9:1;
      model.group.scale.setScalar(scale);
      this.contacts.get(id)?.position.set(x,.064,id==='yani'?.7:.48);
    }
  }

  resize() {
    this.width=Math.max(1,this.container.clientWidth);
    this.height=Math.max(1,this.container.clientHeight);
    this.renderer.setSize(this.width,this.height,false);
    this.updateCamera();
    this.placeModels();
    this.render();
  }

  updateCamera() {
    const aspect=this.width/this.height;
    const height=Math.max(7.15,4.8/aspect)/this.zoom;
    const halfWidth=height*aspect/2;
    this.camera.left=-halfWidth;this.camera.right=halfWidth;
    this.camera.top=height/2;this.camera.bottom=-height/2;
    // Shift the world away from the portrait rail, retaining both faces on phones.
    const rail=Math.min(.48,halfWidth*.14);
    const target=new THREE.Vector3(-rail,1.9,0);
    const distance=16;
    this.camera.position.set(target.x+Math.sin(this.yaw)*Math.cos(this.pitch)*distance,target.y+Math.sin(this.pitch)*distance,target.z+Math.cos(this.yaw)*Math.cos(this.pitch)*distance);
    this.camera.lookAt(target);
    this.camera.updateProjectionMatrix();
    this.camera.updateMatrixWorld();
  }

  point(id,anchor='mouth') {
    const model=this.models.get(id);
    if(!model)return{x:50,y:50};
    const object=model.anchors[anchor] || model.anchors.center;
    this.scene.updateMatrixWorld(true);
    object.getWorldPosition(this.vector);
    this.vector.project(this.camera);
    return{x:(this.vector.x+1)*50,y:(1-this.vector.y)*50};
  }

  pointProp(id) {
    const prop=this.props.get(id);
    if(!prop)return{x:50,y:50};
    this.scene.updateMatrixWorld(true);
    this.vector.copy(prop.userData.anchor);prop.localToWorld(this.vector);this.vector.project(this.camera);
    return{x:(this.vector.x+1)*50,y:(1-this.vector.y)*50};
  }

  bindInput() {
    const canvas=this.renderer.domElement;
    this.onDown=event=>{
      if(event.button && event.pointerType==='mouse')return;
      this.pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
      canvas.setPointerCapture(event.pointerId);
      if(this.pointers.size===1)this.gesture={x:event.clientX,y:event.clientY,moved:false};
      else if(this.gesture)this.gesture.moved=true;
      this.pinchDistance=this.pointerDistance();
    };
    this.onMove=event=>{
      const previous=this.pointers.get(event.pointerId);
      if(!previous)return;
      const dx=event.clientX-previous.x,dy=event.clientY-previous.y;
      this.pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
      if(this.pointers.size>1){
        const distance=this.pointerDistance();
        if(this.pinchDistance>0)this.zoom=clamp(this.zoom*distance/this.pinchDistance,.78,1.32);
        this.pinchDistance=distance;
      }else if(this.gesture){
        if(Math.hypot(event.clientX-this.gesture.x,event.clientY-this.gesture.y)>6)this.gesture.moved=true;
        if(this.gesture.moved){this.yaw=clamp(this.yaw-dx*.005,-Math.PI/4,Math.PI/4);this.pitch=clamp(this.pitch+dy*.003,.14,.58);}
      }
      this.updateCamera();
    };
    this.onUp=event=>{
      const click=this.pointers.size===1 && this.gesture && !this.gesture.moved;
      this.pointers.delete(event.pointerId);
      if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);
      if(!this.pointers.size)this.gesture=null;
      if(click)this.hit(event.clientX,event.clientY);
    };
    this.onCancel=event=>{this.pointers.delete(event.pointerId);this.gesture=null;};
    this.onWheel=event=>{event.preventDefault();this.zoom=clamp(this.zoom*Math.exp(-event.deltaY*.001),.78,1.32);this.updateCamera();};
    this.onDouble=()=>this.resetCamera();
    this.onContextLost=event=>{event.preventDefault();this.setActive(false);this.app.handle3DFailure?.();};
    canvas.addEventListener('pointerdown',this.onDown);
    canvas.addEventListener('pointermove',this.onMove);
    canvas.addEventListener('pointerup',this.onUp);
    canvas.addEventListener('pointercancel',this.onCancel);
    canvas.addEventListener('wheel',this.onWheel,{passive:false});
    canvas.addEventListener('dblclick',this.onDouble);
    canvas.addEventListener('webglcontextlost',this.onContextLost);
    this.onVisibility=()=>{
      if(document.hidden && this.frame!==null){cancelAnimationFrame(this.frame);this.frame=null;}
      else if(this.active && this.frame===null)this.tick();
    };
    document.addEventListener('visibilitychange',this.onVisibility);
  }

  pointerDistance() {
    if(this.pointers.size<2)return 0;
    const [a,b]=this.pointers.values();return Math.hypot(a.x-b.x,a.y-b.y);
  }

  hit(x,y) {
    if(!this.active)return;
    const rect=this.renderer.domElement.getBoundingClientRect();
    this.pointer.set((x-rect.left)/rect.width*2-1,1-(y-rect.top)/rect.height*2);
    this.raycaster.setFromCamera(this.pointer,this.camera);
    // Test the full scene so foreground furniture correctly occludes a target.
    const hits=this.raycaster.intersectObjects(this.scene.children,true);
    for(const {object} of hits) {
      if(object.material?.transparent && object.material.depthWrite===false)continue;
      if(object.userData.actorId){this.app.activateActor(object.userData.actorId);return;}
      if(object.userData.propId){this.app.activateProp(object.userData.propId);return;}
      if(object.isMesh)return;
    }
  }

  updateLight() {
    const night=this.app.stage.dataset.light==='night';
    if(night===this.night)return;
    this.night=night;
    this.scene.background.set(night?'#576269':'#e9ddcb');
    this.scene.fog.color.copy(this.scene.background);
    this.hemisphere.intensity=night?.9:2;
    this.sun.intensity=night?.5:1.85;
    this.sun.color.set(night?'#9cb9de':'#fff7ed');
    this.fill.intensity=night?.3:.85;
    this.lampLight.intensity=night?8:1.2;
    this.lampShade.material.emissiveIntensity=night?.9:.12;
    this.renderer.toneMappingExposure=night?1.1:1;
  }

  render(time=performance.now()/1000) {
    if(this.disposed)return;
    this.updateLight();
    for(const [id,model]of this.models){
      const dataset=this.actors?.get(id)?.dataset || {};
      model.update(dataset,time,this.motionPreference.matches);
    }
    if(!this.motionPreference.matches && this.app.stage.classList.contains('windy'))this.fanBlades.rotation.z=time*35;
    this.renderer.render(this.scene,this.camera);
  }

  tick() {
    if(!this.active || this.disposed || document.hidden){this.frame=null;return;}
    this.render();
    this.frame=requestAnimationFrame(()=>this.tick());
  }

  setActive(active) {
    this.active=Boolean(active);
    if(!active){
      if(this.frame!==null)cancelAnimationFrame(this.frame);
      this.frame=null;this.pointers.clear();this.gesture=null;
      return;
    }
    this.resize();
    if(this.frame===null)this.tick();
  }

  resetCamera() {
    this.yaw=0;this.pitch=.28;this.zoom=1;
    this.updateCamera();this.render();
  }

  dispose() {
    this.setActive(false);this.disposed=true;
    this.observer?.disconnect();
    document.removeEventListener('visibilitychange',this.onVisibility);
    const canvas=this.renderer.domElement;
    [['pointerdown',this.onDown],['pointermove',this.onMove],['pointerup',this.onUp],['pointercancel',this.onCancel],['wheel',this.onWheel],['dblclick',this.onDouble],['webglcontextlost',this.onContextLost]].forEach(([name,handler])=>canvas.removeEventListener(name,handler));
    this.models.forEach(model=>model.dispose());this.models.clear();
    this.geometries.forEach(geometry=>geometry.dispose());
    this.materials.forEach(material=>material.dispose());
    this.textureKit?.dispose();
    this.surfaceMaterials.clear();
    this.textures.forEach(texture=>texture.dispose());
    this.renderer.dispose();this.renderer.forceContextLoss();canvas.remove();
  }
}
