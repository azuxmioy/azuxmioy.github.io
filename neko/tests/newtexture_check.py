#!/usr/bin/env python3
"""Inspect textured character meshes, expression maps, UVs, and GPU cleanup.

Requires localhost:4173 and the SwiftShader Chrome endpoint on port 9223.
No browser run should start while character/texture modules are being rewritten.
"""
import argparse
import json
import sys
import time
import urllib.parse
from browser_check import CDP


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:4173')
    parser.add_argument('--endpoint',default='http://127.0.0.1:9223')
    parser.add_argument('--models-only',action='store_true')
    args=parser.parse_args()
    browser=CDP(args.endpoint)
    checks=[]
    model_reports=[]
    memory=[]

    def expect(label,expression):
        result=browser.evaluate(expression)
        passed=result is True
        checks.append({'check':label,'passed':passed,'result':result})
        print(('PASS ' if passed else 'FAIL ')+label,flush=True)

    def click(selector):
        point=browser.evaluate(f"(() => {{const r=document.querySelector({json.dumps(selector)}).getBoundingClientRect();return {{x:r.x+r.width/2,y:r.y+r.height/2}};}})()")
        for kind in ['mousePressed','mouseReleased']:
            browser.call('Input.dispatchMouseEvent',{'type':kind,'button':'left','clickCount':1,**point})

    def invite(identifier):
        click(f'[data-invite-character={identifier}]')
        target='null' if identifier=='none' else json.dumps(identifier)
        browser.wait_for(f'app.guestId==={target} && app.view3D.models.size==={1 if identifier=="none" else 2}')
        click('.reset-button')
        browser.evaluate('app.view3D.render();true')

    try:
        browser.navigate(args.url,1440,900)
        browser.wait_for("!!document.querySelector('[data-view-style=\"3d\"]')")
        browser.evaluate("window.app=document.querySelector('neko-app');true")
        browser.evaluate("app.setStyle('3d').then(()=>true)")
        browser.wait_for("app.styleMode==='3d' && app.view3D?.models.has('yani')",timeout=30)
        invite('none')
        browser.evaluate("window.hostIdentity=app.actor('yani');window.faceOf=id=>{let found;app.view3D.models.get(id).group.traverse(n=>{if(n.isMesh&&(n.userData.part==='head'||n.name===id+'-head'))found=n});return found};true")
        browser.evaluate("""
window.inspectTexturedModel=id=>{
 const model=app.view3D.models.get(id),meshes=[];model.group.traverse(n=>{if(n.isMesh)meshes.push(n)});
 const materials=[...new Set(meshes.flatMap(n=>Array.isArray(n.material)?n.material:[n.material]))];
 const surfaces=meshes.filter(n=>/head|hair|shirt|pants/i.test(n.name));
 const uvIssues=[];
 for(const mesh of meshes){
   const mats=Array.isArray(mesh.material)?mesh.material:[mesh.material];
   if(!mats.some(m=>m.map))continue;
   const uv=mesh.geometry.getAttribute('uv');
   if(!uv||![...uv.array].every(Number.isFinite)){uvIssues.push(mesh.name+': missing/invalid UV');continue}
   const values=uv.array,us=[],vs=[];for(let i=0;i<values.length;i+=2){us.push(values[i]);vs.push(values[i+1])}
   if(Math.max(...us)-Math.min(...us)<.01||Math.max(...vs)-Math.min(...vs)<.01)uvIssues.push(mesh.name+': collapsed UV range');
 }
 const head=faceOf(id),g=head?.geometry,p=g?.getAttribute('position'),uv=g?.getAttribute('uv');
 let jawWidth=0,cheekWidth=0,nose=-Infinity,side=-Infinity,uvAreaRatio=0;
 if(p&&uv){
   g.computeBoundingBox();const b=g.boundingBox,h=b.max.y-b.min.y,w=b.max.x-b.min.x;
   for(let i=0;i<p.count;i++){
     const x=p.getX(i),y=(p.getY(i)-b.min.y)/h,z=p.getZ(i);
     if(y>.13&&y<.3)jawWidth=Math.max(jawWidth,Math.abs(x)*2);
     if(y>.4&&y<.62)cheekWidth=Math.max(cheekWidth,Math.abs(x)*2);
     if(y>.3&&y<.75&&Math.abs(x)<w*.06)nose=Math.max(nose,z);
     if(y>.3&&y<.75&&Math.abs(x)>w*.14&&Math.abs(x)<w*.26)side=Math.max(side,z);
   }
   const ids=g.index?.array||Array.from({length:p.count},(_,i)=>i);let positive=0,total=0;
   for(let i=0;i+2<ids.length;i+=3){const a=ids[i],b=ids[i+1],c=ids[i+2],area=Math.abs((uv.getX(b)-uv.getX(a))*(uv.getY(c)-uv.getY(a))-(uv.getY(b)-uv.getY(a))*(uv.getX(c)-uv.getX(a)));total++;if(area>1e-10)positive++}
   uvAreaRatio=positive/total;
 }
 return {id,meshes:meshes.length,mappedMaterials:materials.filter(m=>m.map).length,facialPrimitives:meshes.filter(n=>/eye|blush|brow|mouth/i.test(n.name)).map(n=>n.name),
   surfaces:surfaces.map(n=>({name:n.name,type:n.material.type,map:!!n.material.map,bump:!!(n.material.bumpMap||n.material.normalMap),roughness:!!n.material.roughnessMap})),
   uvIssues,head:head?{name:head.name,type:g.type,vertices:p.count,uvAreaRatio,jawWidth,cheekWidth,noseProjection:nose-side,map:head.material.map?.uuid}:null};
};true
""")

        for identifier in ['yani','imouto','yaku','kansai','aru']:
            if identifier!='yani':
                invite(identifier)
            report=browser.evaluate(f'inspectTexturedModel({json.dumps(identifier)})')
            model_reports.append(report)
            browser.evaluate(f'window.currentTextureReport={json.dumps(report)};true')
            expect(identifier+' major skin/hair/clothing surfaces use full texture maps',"currentTextureReport.surfaces.length>=3 && currentTextureReport.surfaces.every(s=>s.type==='MeshStandardMaterial'&&s.map&&s.bump&&s.roughness)")
            expect(identifier+' texture coordinates are finite and cover real surface area',"currentTextureReport.uvIssues.length===0 && currentTextureReport.head.uvAreaRatio>.9")
            expect(identifier+' head has integrated jaw/cheek/nose geometry',"currentTextureReport.head.type==='BufferGeometry' && currentTextureReport.head.vertices>500 && currentTextureReport.head.cheekWidth>currentTextureReport.head.jawWidth*1.02 && currentTextureReport.head.noseProjection>.01")
            expect(identifier+' uses fewer than 80 meshes without separate facial primitives',"currentTextureReport.meshes<80 && currentTextureReport.mappedMaterials>=3 && currentTextureReport.facialPrimitives.length===0")
            expression_data=browser.evaluate(f"(() => {{const model=app.view3D.models.get('{identifier}'),head=faceOf('{identifier}'),maps=[],fingerprints=[];for(const reaction of ['normal','shock','annoyed','goofy']){{model.update({{reaction}},0,true);maps.push(head.material.map?.uuid);const canvas=head.material.map?.image;if(canvas?.getContext){{const pixels=canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;let hash=2166136261;for(let i=0;i<pixels.length;i+=4*(Math.max(1,Math.floor(pixels.length/4096/4))|1))hash=Math.imul(hash^pixels[i]^pixels[i+1]<<8^pixels[i+2]<<16,16777619);fingerprints.push(hash>>>0)}}}}const blinks=new Map();for(let t=0;t<12;t+=.04){{model.update({{reaction:'normal'}},t,false);blinks.set(head.material.map?.uuid,t)}}model.update({{}},0,true);return {{maps,fingerprints,blinkMaps:[...blinks.keys()],blinkTimes:[...blinks.values()]}}}})()")
            browser.evaluate(f'window.expressionReport={json.dumps(expression_data)};true')
            expect(identifier+' expression and blink states change painted face textures',"new Set(expressionReport.maps).size===4 && new Set(expressionReport.fingerprints).size===4 && expressionReport.blinkMaps.length>=2 && expressionReport.maps.every(Boolean)")
            click('.reset-button')
            browser.evaluate('app.view3D.render();true')

        if not args.models_only:
            invite('none')
            # Upload host expression/blink variants before comparing disposal baselines.
            browser.evaluate("(() => {const v=app.view3D,m=v.models.get('yani');for(const reaction of ['normal','shock','annoyed','goofy'])for(const t of [0,2,3,4,5,6,7,8]){m.update({reaction},t,false);v.renderer.render(v.scene,v.camera)}v.render();window.textureBaseline=v.renderer.info.memory.textures;return true})()")
            for cycle in range(2):
                for identifier in ['imouto','yaku','kansai','aru']:
                    invite(identifier)
                    browser.evaluate(f"(() => {{const textures=new Set(),geometries=new Set();app.view3D.models.get('{identifier}').group.traverse(n=>{{if(!n.isMesh)return;geometries.add(n.geometry);for(const m of (Array.isArray(n.material)?n.material:[n.material]))for(const key of ['map','normalMap','bumpMap','roughnessMap'])if(m[key])textures.add(m[key])}});window.disposal={{textureTotal:textures.size,geometryTotal:geometries.size,textures:0,geometries:0}};textures.forEach(t=>t.addEventListener('dispose',()=>disposal.textures++));geometries.forEach(g=>g.addEventListener('dispose',()=>disposal.geometries++));return true}})()")
                    invite('none')
                    time.sleep(.08)
                    entry=browser.evaluate(f"({{cycle:{cycle},id:'{identifier}',baseline:textureBaseline,current:app.view3D.renderer.info.memory.textures,...disposal}})")
                    memory.append(entry)
                    expect(f'{identifier} cycle {cycle+1} disposes guest geometry and textures',"disposal.textures===disposal.textureTotal && disposal.geometries===disposal.geometryTotal && disposal.textureTotal>0")
                    expect(f'{identifier} cycle {cycle+1} returns GPU textures to baseline',"app.view3D.renderer.info.memory.textures<=textureBaseline+2 && app.actor('yani')===hostIdentity")

            invite('imouto')
            browser.evaluate("app.view3D.resetCamera();true")
            browser.screenshot('/tmp/neko2-textured-after-desktop.png')
            for width,height in [(320,740),(390,844),(768,1024),(1440,900)]:
                browser.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                time.sleep(.15)
                expect(f'{width}px textured models render inside the 3D viewport',"app.view3D.renderer.info.render.calls>0 && [...app.view3D.models.keys()].every(id=>{const p=app.view3D.point(id,'center');return p.x>0&&p.x<100&&p.y>0&&p.y<100})")
                browser.screenshot(f'/tmp/neko2-textured-{width}.png')

        browser.evaluate('true')
        origin=urllib.parse.urlparse(args.url).netloc
        remote=sorted({e['params']['request']['url'] for e in browser.events if e.get('method')=='Network.requestWillBeSent' and urllib.parse.urlparse(e['params']['request']['url']).scheme in ('http','https') and urllib.parse.urlparse(e['params']['request']['url']).netloc!=origin})
        result={'passed':sum(c['passed'] for c in checks),'failed':[c for c in checks if not c['passed']],'models':model_reports,'memory':memory,'errors':browser.errors,'remote_requests':remote}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 1 if result['failed'] or browser.errors or remote else 0
    finally:
        browser.close()


if __name__=='__main__':
    sys.exit(main())
