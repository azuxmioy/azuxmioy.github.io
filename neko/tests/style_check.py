#!/usr/bin/env python3
"""Browser checks for preserved 2D and genuine, optional WebGL 3D styles.

Requires the local site and a Chrome instance supporting WebGL. The prepared
SwiftShader browser listens on port 9223. No Python dependencies are required.
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
    parser.add_argument('--fallback-only',action='store_true')
    args=parser.parse_args()
    checks=[]
    errors=[]
    remote_requests=set()

    def expect(browser,label,expression):
        result=browser.evaluate(expression)
        passed=result is True
        checks.append({'check':label,'passed':passed,'result':result})
        print(('PASS ' if passed else 'FAIL ')+label,flush=True)

    def click(browser,selector):
        point=browser.evaluate(f"(() => {{const r=document.querySelector({json.dumps(selector)}).getBoundingClientRect();return {{x:r.x+r.width/2,y:r.y+r.height/2}};}})()")
        click_point(browser,point)

    def click_point(browser,point):
        for kind in ['mousePressed','mouseReleased']:
            browser.call('Input.dispatchMouseEvent',{'type':kind,'button':'left','clickCount':1,**point})

    def project(browser,identifier,prop=False):
        method=f"pointProp({json.dumps(identifier)})" if prop else f"point({json.dumps(identifier)},'center')"
        return browser.evaluate(f"(() => {{const p=app.view3D.{method},r=app.stage.getBoundingClientRect();return {{x:r.x+p.x*r.width/100,y:r.y+p.y*r.height/100}};}})()")

    def fresh(browser):
        browser.navigate(args.url,1280,800)
        browser.evaluate("localStorage.removeItem('neko-style')")
        browser.call('Page.reload',{'ignoreCache':True})
        browser.wait_for("document.readyState==='complete' && !!document.querySelector('[data-view-style=\"2d\"]')")
        browser.evaluate("window.app=document.querySelector('neko-app');window.originalYani=app.actor('yani');true")

    def reset(browser):
        click(browser,'.reset-button')

    def collect(browser,expected_webgl=False):
        browser.evaluate('true')
        origin=urllib.parse.urlparse(args.url).netloc
        for event in browser.events:
            if event.get('method')=='Network.requestWillBeSent':
                url=event['params']['request']['url']
                parsed=urllib.parse.urlparse(url)
                if parsed.scheme in ('http','https') and parsed.netloc!=origin:
                    remote_requests.add(url)
        for error in browser.errors:
            encoded=json.dumps(error)
            if expected_webgl and 'WebGLRenderer' in encoded and 'Error creating WebGL context' in encoded:
                continue
            errors.append(error)

    if not args.fallback_only:
        browser=CDP(args.endpoint)
        try:
            fresh(browser)
            expect(browser,'2D remains the default and keeps original Yani',"app.styleMode==='2d' && app.stage.dataset.style==='2d' && app.actor('yani')===originalYani && app.guestId===null")
            expect(browser,'Both style controls expose their selected state',"document.querySelector('[data-view-style=\"2d\"]').getAttribute('aria-pressed')==='true' && document.querySelector('[data-view-style=\"3d\"]').getAttribute('aria-pressed')==='false'")

            click(browser,'[data-invite-character=imouto]')
            browser.wait_for("app.guestId==='imouto'")
            reset(browser)
            browser.evaluate("window.originalGuest=app.actor('imouto')")
            click(browser,'room-prop[data-id=lamp] .prop-hit')
            browser.wait_for("app.stage.dataset.light==='night'")
            click(browser,'.sound-toggle')
            browser.wait_for('app.audio.muted')
            click(browser,'[data-view-style=\"3d\"]')
            browser.wait_for("app.styleMode==='3d' && !!app.view3D?.renderer && app.view3D.models.size===2",timeout=30)
            browser.wait_for('app.view3D.renderer.info.render.calls>0',timeout=30)
            expect(browser,'3D creates a WebGL canvas and renders actual triangles',"!!document.querySelector('.scene-three canvas') && app.view3D.active && app.view3D.renderer.info.render.calls>0 && app.view3D.renderer.info.render.triangles>100")
            expect(browser,'Style switching preserves host, guest, night, and mute',"app.actor('yani')===originalYani && app.actor('imouto')===originalGuest && app.guestId==='imouto' && app.stage.dataset.light==='night' && app.audio.muted")
            expect(browser,'Both models contain volumetric geometry rather than flat portraits',"[...app.view3D.models.values()].every(model=>{let meshes=0,volume=0;model.group.traverse(n=>{if(!n.isMesh)return;meshes++;n.geometry.computeBoundingBox();const b=n.geometry.boundingBox;if(b.max.x-b.min.x>.001&&b.max.y-b.min.y>.001&&b.max.z-b.min.z>.001)volume++});return meshes>=8&&volume>=6})")
            expect(browser,'The rendered canvas contains varied nonblank pixel colors',"(() => {const v=app.view3D,r=v.renderer,g=r.getContext();r.render(v.scene,v.camera);const w=g.drawingBufferWidth,h=g.drawingBufferHeight,p=new Uint8Array(w*h*4);g.readPixels(0,0,w,h,g.RGBA,g.UNSIGNED_BYTE,p);const colors=new Set();for(let i=0;i<p.length;i+=Math.max(4,Math.floor(p.length/2000/4)*4))colors.add(p[i]+','+p[i+1]+','+p[i+2]);return colors.size>20})()")
            checks[-1]['render']=browser.evaluate("({calls:app.view3D.renderer.info.render.calls,triangles:app.view3D.renderer.info.render.triangles,models:[...app.view3D.models.keys()]})")

            reset(browser)
            click_point(browser,project(browser,'yani'))
            browser.wait_for("originalYani.dataset.smoking==='true'")
            expect(browser,'Raycasting a 3D Yani click starts smoking',"app.puffNumber===1 && originalYani.dataset.motion==='smoke'")
            reset(browser)
            click_point(browser,project(browser,'imouto'))
            browser.wait_for("originalYani.dataset.reaction==='shock'")
            expect(browser,'Raycasting a 3D guest click performs the paired gag',"!!app.actor('imouto').dataset.motion && app.gags.records.size>0")
            reset(browser)

            browser.evaluate("window.cameraBefore={yaw:app.view3D.yaw,pitch:app.view3D.pitch,zoom:app.view3D.zoom}")
            start={'x':780,'y':280}
            browser.call('Input.dispatchMouseEvent',{'type':'mousePressed','button':'left','clickCount':1,**start})
            for offset in [25,55,95,130]:
                browser.call('Input.dispatchMouseEvent',{'type':'mouseMoved','button':'left','buttons':1,'x':start['x']+offset,'y':start['y']+offset/3})
            browser.call('Input.dispatchMouseEvent',{'type':'mouseReleased','button':'left','clickCount':1,'x':910,'y':323})
            time.sleep(.2)
            expect(browser,'Dragging orbits the camera without triggering a character',"(Math.abs(app.view3D.yaw-cameraBefore.yaw)>.01 || Math.abs(app.view3D.pitch-cameraBefore.pitch)>.01) && app.puffNumber===0 && [...app.actors.values()].every(a=>!a.dataset.motion)")
            browser.call('Input.dispatchMouseEvent',{'type':'mouseWheel','x':780,'y':280,'deltaX':0,'deltaY':130})
            time.sleep(.2)
            expect(browser,'Mouse wheel changes camera zoom',"Math.abs(app.view3D.zoom-cameraBefore.zoom)>.001")
            click(browser,'[data-three-action=camera]')
            browser.wait_for('Math.abs(app.view3D.yaw-cameraBefore.yaw)<.001 && Math.abs(app.view3D.pitch-cameraBefore.pitch)<.001 && Math.abs(app.view3D.zoom-cameraBefore.zoom)<.001')
            expect(browser,'Camera control restores the default view',"Math.abs(app.view3D.yaw-cameraBefore.yaw)<.001 && Math.abs(app.view3D.pitch-cameraBefore.pitch)<.001")

            click_point(browser,project(browser,'lamp',prop=True))
            browser.wait_for("app.stage.dataset.light==='night'")
            expect(browser,'Raycasting the 3D lamp switches lighting',"app.stage.dataset.light==='night'")
            reset(browser)
            click(browser,'[data-three-action=radio]')
            browser.wait_for("app.stage.classList.contains('dancing')")
            expect(browser,'Accessible 3D radio control animates both characters',"[...app.actors.values()].every(a=>a.dataset.motion==='dance')")
            reset(browser)

            click(browser,'[data-invite-character=aru]')
            browser.wait_for("app.guestId==='aru' && app.view3D.models.has('aru') && !app.view3D.models.has('imouto')")
            expect(browser,'3D invitations replace only the guest model',"app.view3D.models.size===2 && app.view3D.models.has('yani') && app.actor('yani')===originalYani")
            reset(browser)
            click(browser,'[data-invite-character=none]')
            browser.wait_for('app.view3D.models.size===1')
            expect(browser,'Guest dismissal leaves one model and hides the guest action',"app.view3D.models.has('yani') && app.guestId===null && (document.querySelector('[data-three-action=guest]').hidden || getComputedStyle(document.querySelector('[data-three-action=guest]')).display==='none')")
            click(browser,'[data-invite-character=imouto]')
            browser.wait_for('app.view3D.models.size===2')
            reset(browser)
            click_point(browser,project(browser,'imouto'))
            browser.wait_for("!!document.querySelector('.gag-effect--slipper')")
            click(browser,'[data-view-style=\"2d\"]')
            browser.wait_for("app.styleMode==='2d' && !app.view3D.active")
            time.sleep(.9)
            expect(browser,'Switching to 2D stops RAF and cancels an in-flight 3D gag',"app.view3D.frame===null && app.gags.records.size===0 && app.timers.size===0 && [...app.actors.values()].every(a=>!a.dataset.motion&&!a.dataset.reaction) && app.actor('yani')===originalYani && app.guestId==='imouto'")
            click(browser,'[data-view-style=\"3d\"]')
            browser.wait_for("app.styleMode==='3d' && app.view3D.active")

            for width,height in [(320,740),(390,844),(768,1024),(1440,900)]:
                browser.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                time.sleep(.2)
                expect(browser,f'{width}px 3D canvas and style controls fit the viewport',"document.documentElement.scrollWidth===innerWidth && document.documentElement.scrollHeight===innerHeight && [...document.querySelectorAll('[data-view-style]')].every(b=>{const r=b.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight&&r.left>=0&&r.right<=innerWidth&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)?.closest('[data-view-style]')===b})")
                expect(browser,f'{width}px projected 3D actors remain onscreen',"[...app.view3D.models.keys()].every(id=>{const p=app.view3D.point(id,'center');return p.x>0&&p.x<100&&p.y>0&&p.y<100})")
                browser.screenshot(f'/tmp/neko2-3d-{width}.png')

            expect(browser,'3D preference is persisted locally',"localStorage.getItem('neko-style')==='3d'")
            browser.navigate(args.url,1280,800)
            browser.wait_for("document.querySelector('neko-app').styleMode==='3d' && !!document.querySelector('neko-app').view3D?.active",timeout=30)
            browser.evaluate("window.app=document.querySelector('neko-app');true")
            expect(browser,'Reload restores the persisted 3D style',"app.styleMode==='3d' && app.view3D.active")
            click(browser,'[data-view-style=\"2d\"]')
            browser.wait_for("app.styleMode==='2d'")
            collect(browser)
        finally:
            browser.close()

    # Isolate unavailable-WebGL behavior in a fresh tab; 2D canvas remains usable.
    browser=CDP(args.endpoint)
    try:
        browser.call('Page.addScriptToEvaluateOnNewDocument',{'source':"const originalGetContext=HTMLCanvasElement.prototype.getContext;HTMLCanvasElement.prototype.getContext=function(type,...args){return /webgl/i.test(type)?null:originalGetContext.call(this,type,...args)};"})
        fresh(browser)
        browser.evaluate("app.setStyle('3d')")
        expect(browser,'Unavailable WebGL gracefully preserves the 2D scene',"app.styleMode==='2d' && app.stage.dataset.style==='2d' && app.actor('yani')===originalYani && document.querySelector('[data-view-style=\"2d\"]').getAttribute('aria-pressed')==='true'")
        expect(browser,'WebGL fallback explains the retained 2D scene',"document.querySelector('.style-notice').textContent.includes('已保留 2D') && app.view3D===null && localStorage.getItem('neko-style')==='2d'")
        click(browser,'.cast neko-actor[data-id=yani] .actor-hit')
        browser.wait_for("originalYani.dataset.smoking==='true'")
        expect(browser,'2D interactions still work after the WebGL fallback',"app.puffNumber===1 && originalYani.dataset.motion==='smoke'")
        collect(browser,expected_webgl=True)
    finally:
        browser.close()

    report={'passed':sum(c['passed'] for c in checks),'failed':[c for c in checks if not c['passed']],'errors':errors,'remote_requests':sorted(remote_requests)}
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 1 if report['failed'] or errors or remote_requests else 0


if __name__=='__main__':
    sys.exit(main())
