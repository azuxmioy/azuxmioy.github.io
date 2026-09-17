import * as THREE from '../../assets/vendor/three/three.module.min.js';

const clamp = (value, low=0, high=255) => Math.max(low, Math.min(high, value));
const noise = (x, y, seed=0) => {
  const value = Math.sin(x*127.1+y*311.7+seed*74.7)*43758.5453;
  return value-Math.floor(value);
};

// Authored surface maps, sampled through mesh UVs. No character artwork or
// texture files are downloaded. Each kit owns and disposes its GPU resources.
export function createTextureKit(appearance={}, id='yani') {
  const materials = new Map();
  const surfaces = new Map();
  const textures = new Set();
  let disposed = false;
  const makeTexture = (canvas, color=false, repeat=true) => {
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = color ? THREE.SRGBColorSpace : THREE.NoColorSpace;
    texture.wrapS = texture.wrapT = repeat ? THREE.RepeatWrapping : THREE.ClampToEdgeWrapping;
    texture.anisotropy = 4;
    texture.userData.authored = true;
    textures.add(texture);
    return texture;
  };
  const makeCanvas = (size=256) => {
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    return canvas;
  };

  function surface(kind) {
    if(surfaces.has(kind)) return surfaces.get(kind);
    const size=256;
    const albedo=makeCanvas(size), relief=makeCanvas(size), roughness=makeCanvas(size);
    const ca=albedo.getContext('2d'), cb=relief.getContext('2d'), cr=roughness.getContext('2d');
    const color=ca.createImageData(size,size), height=cb.createImageData(size,size), rough=cr.createImageData(size,size);
    for(let y=0;y<size;y++) for(let x=0;x<size;x++) {
      const i=(y*size+x)*4;
      const n=noise(x,y,3)-.5;
      let tone=250, bump=128, matte=220;
      if(['cloth','denim','knit'].includes(kind)) {
        const weave=Math.sin(x*Math.PI/2)*Math.cos(y*Math.PI/2);
        const crossing=(Math.floor(x/3)+Math.floor(y/3))%2 ? 1 : -1;
        const twill=Math.sin((x+y)*Math.PI/4);
        const stitch=Math.cos(x*Math.PI/8+Math.sin(y*Math.PI/8)*.85);
        const pattern=kind==='denim'?twill:kind==='knit'?stitch:weave;
        tone=244+pattern*5+crossing*2+n*4;
        bump=128+pattern*(kind==='knit'?30:18)+n*9;
        matte=220+pattern*10+n*9;
      } else if(kind==='hair') {
        const strands=Math.sin(x*.62+Math.sin(y*.027)*.7)+Math.sin(x*2.3+y*.019)*.35;
        const sheen=Math.pow(Math.max(0,Math.cos(y/size*Math.PI*2+.2)),8);
        tone=232+strands*7+sheen*14+n*2;
        bump=128+strands*10+n*3;
        matte=150+strands*14-sheen*17;
      } else if(kind==='skin') {
        tone=253+n*2;
        bump=128+n*12;
        matte=194+n*12;
      } else if(kind==='wood') {
        const grain=Math.sin(x*.18+Math.sin(y*.033)*2+Math.sin(y*.09)*.4);
        const fine=Math.sin(x*.71+Math.sin(y*.017)*3);
        tone=248+grain*3+fine*1.5+n*2;
        bump=128+grain*6+fine*2;
        matte=194+grain*5+n*4;
      } else if(kind==='leather') {
        const pore=noise(Math.floor(x/2),Math.floor(y/2),8)-.5;
        tone=247+pore*8+n*3;
        bump=128+pore*24+n*5;
        matte=160+pore*25;
      } else {
        tone=251+n*4;
        bump=128+n*15;
        matte=235+n*8;
      }
      for(let channel=0;channel<3;channel++) {
        color.data[i+channel]=clamp(tone);
        height.data[i+channel]=clamp(bump);
        rough.data[i+channel]=clamp(matte);
      }
      color.data[i+3]=height.data[i+3]=rough.data[i+3]=255;
    }
    ca.putImageData(color,0,0);cb.putImageData(height,0,0);cr.putImageData(rough,0,0);
    const result={map:makeTexture(albedo,true),bumpMap:makeTexture(relief),roughnessMap:makeTexture(roughness)};
    const repeat = kind==='hair' ? [2,1] : kind==='knit' ? [3,3] : ['cloth','denim'].includes(kind) ? [4,4] : kind==='wood' ? [2,1] : [2,2];
    Object.values(result).forEach(texture=>texture.repeat.set(...repeat));
    surfaces.set(kind,result);
    return result;
  }

  function material(kind, color='#ffffff', options={}) {
    if(disposed) throw new Error('Texture kit is disposed');
    const key=JSON.stringify([kind,color,options]);
    if(materials.has(key)) return materials.get(key);
    const bumpScale={skin:.002,hair:.004,cloth:.012,denim:.014,knit:.018,leather:.008,wood:.003,wall:.005}[kind] ?? .006;
    const value=new THREE.MeshStandardMaterial({
      color,metalness:0,roughness:1,...surface(kind),bumpScale,...options,
    });
    if(kind==='cloth' && id==='yani' && color===appearance.shirt) {
      const canvas=makeCanvas(1024),ctx=canvas.getContext('2d'),tile=surface(kind).map.image;
      for(let y=0;y<4;y++)for(let x=0;x<4;x++)ctx.drawImage(tile,x*256,y*256);
      // A tiny worn cat emblem is painted into the shirt's UVs, alongside its weave.
      ctx.save();ctx.translate(558,436);ctx.fillStyle='#d6d0bf';
      ctx.beginPath();ctx.moveTo(-19,-5);ctx.lineTo(-20,-19);ctx.lineTo(-8,-11);
      ctx.quadraticCurveTo(0,-15,9,-10);ctx.lineTo(19,-18);ctx.lineTo(19,-3);
      ctx.bezierCurveTo(24,14,-22,16,-19,-5);ctx.fill();ctx.restore();
      value.map=makeTexture(canvas,true,false);
    }
    value.name=`${id}-${kind}-textured`;
    value.userData.surface=kind;
    materials.set(key,value);
    return value;
  }

  function paintFace(expression, blink) {
    const canvas=makeCanvas(1024),ctx=canvas.getContext('2d'),size=canvas.width;
    ctx.fillStyle=appearance.skin || '#ffe8ce';ctx.fillRect(0,0,size,size);
    const px=x=>(.5+Math.asin(clamp(x/.675,-.995,.995))/(Math.PI*2))*size;
    const py=y=>(.65-y)/1.27*size;
    const path=(commands,fill,stroke,width=.009)=>{
      ctx.beginPath();
      for(const [type,...points] of commands) {
        if(type==='M')ctx.moveTo(px(points[0]),py(points[1]));
        if(type==='L')ctx.lineTo(px(points[0]),py(points[1]));
        if(type==='Q')ctx.quadraticCurveTo(px(points[0]),py(points[1]),px(points[2]),py(points[3]));
        if(type==='C')ctx.bezierCurveTo(px(points[0]),py(points[1]),px(points[2]),py(points[3]),px(points[4]),py(points[5]));
        if(type==='Z')ctx.closePath();
      }
      if(fill){ctx.fillStyle=fill;ctx.fill();}
      if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=width*size/2.2;ctx.lineCap='round';ctx.lineJoin='round';ctx.stroke();}
    };
    const ellipse=(x,y,rx,ry,fill)=>{
      ctx.beginPath();ctx.ellipse(px(x),py(y),(px(x+rx)-px(x-rx))/2,ry*size/1.27,0,0,Math.PI*2);
      ctx.fillStyle=fill;ctx.fill();
    };
    // Soft, painted cheek warmth is part of the skin albedo, never a stuck-on oval.
    for(const side of [-1,1]) {
      const x=px(side*.40),y=py(-.20),radius=48;
      const blush=ctx.createRadialGradient(x,y,0,x,y,radius);
      blush.addColorStop(0,id==='aru'?'rgba(202,98,91,.38)':'rgba(221,131,119,.23)');
      blush.addColorStop(1,'rgba(224,147,132,0)');
      ctx.fillStyle=blush;ctx.fillRect(x-radius,y-radius,radius*2,radius*2);
    }
    const ink='#4a3834';
    const tired=['yani','yaku','kansai'].includes(id) || expression==='annoyed';
    for(const side of [-1,1]) {
      const x=side*.265, cy=-.04;
      if(blink || (id==='aru' && expression==='normal')) {
        path([['M',x-.155,cy],['Q',x,cy+(blink?-.038:.055),x+.155,cy]],null,ink,.017);
      } else if(expression==='goofy') {
        path([['M',x-side*.14,cy+.078],['L',x+side*.06,cy],['L',x-side*.14,cy-.067]],null,ink,.017);
      } else {
        const shock=expression==='shock';
        const top=shock?.165:tired?.026:.103;
        const bottom=shock?.16:tired?.09:.115;
        const eye=[['M',x-.163,cy+.009],['Q',x-.035,cy+top,x+.16,cy+.007],['Q',x+.14,cy-bottom,x,cy-bottom],['Q',x-.14,cy-bottom,x-.163,cy+.009],['Z']];
        ctx.save();path(eye,'#fff9ef');ctx.clip();
        const iris=ctx.createLinearGradient(0,py(cy+.08),0,py(cy-.13));
        iris.addColorStop(0,'#49352b');iris.addColorStop(.42,appearance.eye || '#b88835');iris.addColorStop(1,id==='imouto'?'#db9972':'#debd76');
        ellipse(x+.007,cy-.022,shock?.038:.072,shock?.065:.103,iris);
        ellipse(x+.007,cy-.016,shock?.013:.020,shock?.041:.068,'#302b27');
        ellipse(x+.030,cy+.035,.018,.022,'#fffdf4');
        ellipse(x-.018,cy-.069,.009,.010,'#f5d995');
        ctx.restore();
        path([['M',x-.17,cy+.009],['Q',x-.035,cy+top,x+.17,cy+.007]],null,ink,.019);
        path([['M',x+.16,cy+.007],['Q',x+.14,cy-bottom,x,cy-bottom],['Q',x-.14,cy-bottom,x-.163,cy+.009]],null,'#af8170',.0045);
        path([['M',x+side*.15,cy+.010],['L',x+side*.19,cy+.032]],null,ink,.014);
        if(id==='yaku')path([['M',x-.13,cy-.13],['Q',x,cy-.163,x+.13,cy-.12]],null,'#b99287',.005);
      }
      const browY=expression==='shock'?.21:.155;
      const tilt=expression==='annoyed'?side*.055:side*-.012;
      path([['M',x-.135,browY+tilt],['Q',x,browY+.018,x+.132,browY-tilt]],null,appearance.hairShade || '#818d85',id==='imouto'?.025:.009);
    }
    path([['M',-.01,-.105],['Q',-.030,-.164,.005,-.168]],null,'#c9a28b',.0035);
    if(expression==='shock') {
      path([['M',-.053,-.27],['C',-.035,-.23,.055,-.23,.061,-.28],['L',.055,-.365],['Q',0,-.401,-.052,-.367],['Z']],'#925249','#765144',.005);
    } else if(expression==='goofy' || (id==='aru' && expression==='normal')) {
      path([['M',-.105,-.268],['Q',0,-.30,.106,-.265],['Q',.09,-.39,0,-.384],['Q',-.078,-.38,-.105,-.268],['Z']],'#864b43',ink,.004);
      path([['M',-.055,-.352],['Q',0,-.326,.059,-.347],['Q',.04,-.389,-.027,-.38],['Z']],'#da9088');
    } else if(expression==='annoyed') {
      path([['M',-.088,-.318],['Q',0,-.29,.095,-.313]],null,ink,.007);
    } else {
      path(id==='yani'?[['M',-.085,-.308],['Q',-.025,-.293,.024,-.31],['L',.092,-.302]]:[['M',-.087,-.30],['Q',.01,-.342,.09,-.298]],null,ink,.007);
    }
    return makeTexture(canvas,true,false);
  }

  function face(expression='normal', blink=false) {
    if(disposed) throw new Error('Texture kit is disposed');
    const key=`face:${expression}:${Boolean(blink)}`;
    if(materials.has(key)) return materials.get(key);
    const skin=surface('skin');
    const value=new THREE.MeshStandardMaterial({
      color:'#ffffff',map:paintFace(expression,blink),bumpMap:skin.bumpMap,
      roughnessMap:skin.roughnessMap,bumpScale:.0015,roughness:.92,metalness:0,
    });
    value.name=`${id}-${key}`;value.userData.surface='face';value.userData.expression=expression;value.userData.blink=Boolean(blink);
    materials.set(key,value);
    return value;
  }

  return { material,face,dispose() {
    if(disposed)return;disposed=true;
    materials.forEach(value=>value.dispose());textures.forEach(value=>value.dispose());
    materials.clear();textures.clear();surfaces.clear();
  } };
}
