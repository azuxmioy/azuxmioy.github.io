#!/usr/bin/env python3
"""Browser checks for persistent Yani, one invited friend, and paired reactions.

Requires localhost:4173 and Chrome debugging on port 9222. No dependencies.
Screenshots are saved to /tmp/neko2-pair-WIDTH.png and /tmp/neko2-solo.png.
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
    parser.add_argument('--layout-only',action='store_true')
    parser.add_argument('--functional-only',action='store_true')
    args=parser.parse_args()
    browser=CDP()
    checks=[]

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
        browser.wait_for(f'app.guestId==={target}')

    def actor(identifier):
        click(f'.cast neko-actor[data-id={identifier}] .actor-hit')

    def prop(identifier):
        click(f'room-prop[data-id={identifier}] .prop-hit')

    def reset():
        click('.reset-button')

    def key(code,name,number,blur=True):
        if blur:
            browser.evaluate('document.activeElement.blur()')
        for kind in ['keyDown','keyUp']:
            params={'type':kind,'code':code,'key':name,'windowsVirtualKeyCode':number}
            if kind=='keyDown' and code=='Enter':
                params.update({'text':'\r','unmodifiedText':'\r'})
            browser.call('Input.dispatchKeyEvent',params)

    idle="app.timers.size===0 && app.gags.records.size===0 && app.smokeEffect.particles.size===0 && [...app.actors.values()].every(a=>!a.dataset.motion&&!a.dataset.reaction&&!a.dataset.smoking)"
    persistent="app.actor('yani')===originalYani && originalYani.isConnected && document.querySelectorAll('.cast neko-actor').length===app.actors.size && app.actors.size>=1 && app.actors.size<=2"
    try:
        browser.navigate(args.url,1440,900)
        browser.wait_for("document.querySelectorAll('[data-invite-character]').length===5")
        browser.evaluate("window.app=document.querySelector('neko-app');window.originalYani=app.actor('yani')")
        expect('Yani is always present with no guest initially', persistent+" && app.guestId===null && app.actors.size===1 && !!originalYani.querySelector('.seat-cushion')")
        expect('Sidebar has four friends and no-guest option, without a Yani option', "document.querySelectorAll('[data-invite-character]').length===5 && !document.querySelector('[data-invite-character=yani]') && document.querySelector('[data-invite-character=none]').getAttribute('aria-pressed')==='true'")
        expect('Controls have accessible names and all artwork is SVG', "[...document.querySelectorAll('button')].every(b=>!!b.getAttribute('aria-label')) && !document.querySelector('img,image')")

        if not args.layout_only:
            expect('Audio is not initialized before a gesture','app.audio.context===null')
            actor('yani')
            browser.wait_for("originalYani.dataset.smoking==='true'")
            browser.wait_for("document.querySelectorAll('.smoke-effect>span').length>0")
            expect('Solo Yani smokes with particles and running audio', "originalYani.dataset.motion==='smoke' && app.audio.context.state==='running' && document.querySelectorAll('.smoke-effect>span').length>=10")
            browser.wait_for("!!document.querySelector('.gag-effect--ring')")
            expect('Solo Yani creates smoke rings',"!!document.querySelector('.gag-effect--ring')")
            reset()

            for identifier in ['imouto','yaku','kansai','aru']:
                invite(identifier)
                browser.wait_for(f"!!app.actor('{identifier}') && !!originalYani.dataset.motion && !!app.actor('{identifier}').dataset.motion")
                expect(identifier+' invitation greets both characters without replacing Yani',persistent+f" && app.actors.size===2 && document.querySelector('[data-invite-character={identifier}]').getAttribute('aria-pressed')==='true' && document.querySelectorAll('[data-invite-character][aria-pressed=true]').length===1 && !!originalYani.dataset.motion && !!app.actor('{identifier}').dataset.motion")
                reset()
                actor(identifier)
                if identifier=='aru':
                    # The throw ends briefly before the shared hiccups begin.
                    browser.wait_for("originalYani.dataset.motion==='hiccup' && app.actor('aru').dataset.motion==='laugh'")
                else:
                    browser.wait_for(f"!!originalYani.dataset.reaction && originalYani.dataset.reaction!=='normal' && !!app.actor('{identifier}').dataset.motion")
                expect(identifier+' click produces a paired reaction on Yani',persistent+f" && !!originalYani.dataset.motion && !!app.actor('{identifier}').dataset.motion && app.gags.records.size>0")
                checks[-1]['states']=browser.evaluate(f"({{host:{{...originalYani.dataset}},guest:{{...app.actor('{identifier}').dataset}},effects:[...document.querySelectorAll('.gag-effect')].map(e=>e.className)}})")
                reset()
                actor('yani')
                browser.wait_for(f"!!app.actor('{identifier}').dataset.motion")
                expect(identifier+' reacts when Yani smokes',persistent+f" && !!app.actor('{identifier}').dataset.reaction && app.puffNumber===1")
                checks[-1]['states']=browser.evaluate(f"({{host:{{...originalYani.dataset}},guest:{{...app.actor('{identifier}').dataset}},effects:[...document.querySelectorAll('.gag-effect')].map(e=>e.className)}})")
                reset()

            invite('imouto')
            reset()
            actor('imouto')
            browser.wait_for("!!document.querySelector('.gag-effect--slipper')")
            invite('none')
            time.sleep(1.1)
            expect('Dismissing a friend mid-projectile cancels impact and keeps Yani',persistent+" && app.actors.size===1 && app.guestId===null && app.audio.voices.size===0 && "+idle)
            invite('yaku')
            reset()
            actor('yani')
            invite('none')
            time.sleep(2)
            expect('Dismissal cancels pending smoke and delayed friend reactions',persistent+" && app.guestId===null && "+idle)

            invite('aru')
            reset()
            actor('aru')
            invite('kansai')
            reset()
            time.sleep(1.2)
            expect('Replacement cancels old choreography without replacing the host',persistent+" && app.guestId==='kansai' && !app.actor('aru') && "+idle)

            prop('lamp')
            browser.wait_for("app.stage.dataset.light==='night'")
            click('.sound-toggle')
            browser.wait_for('app.audio.muted')
            invite('yaku')
            expect('Invitations preserve night and mute settings',"app.stage.dataset.light==='night' && app.audio.muted && document.querySelector('.sound-toggle').getAttribute('aria-pressed')==='true'")
            reset()
            expect('Reset keeps the guest, host identity, and mute while restoring day',persistent+" && app.guestId==='yaku' && app.audio.muted && app.stage.dataset.light==='day' && "+idle)
            key('Space',' ',32)
            browser.wait_for("originalYani.dataset.smoking==='true'")
            expect('Space always activates Yani while a guest is present',"app.puffNumber===1 && originalYani.dataset.motion==='smoke' && app.guestId==='yaku'")
            reset()
            prop('radio')
            browser.wait_for("[...app.actors.values()].every(a=>a.dataset.motion==='dance')")
            expect('Radio makes both present characters dance',"app.actors.size===2 && app.stage.classList.contains('dancing') && document.querySelectorAll('.gag-effect--notes').length===6")
            reset()
            prop('fan')
            browser.wait_for("app.stage.classList.contains('windy')")
            expect('Fan affects both present characters',"[...app.actors.values()].every(a=>a.dataset.motion==='wind')")
            reset()
            invite('none')
            expect('No-guest option leaves only the same Yani',persistent+" && app.actors.size===1 && app.guestId===null")
            key('Space',' ',32)
            browser.wait_for("originalYani.dataset.smoking==='true'")
            expect('Space still smokes after the guest leaves',"app.puffNumber===1 && app.actors.size===1")
            reset()
            browser.screenshot('/tmp/neko2-solo.png')

            browser.call('Emulation.setEmulatedMedia',{'features':[{'name':'prefers-reduced-motion','value':'reduce'}]})
            invite('imouto')
            reset()
            actor('imouto')
            browser.wait_for("originalYani.dataset.reaction==='shock'")
            expect('Reduced motion preserves paired interaction outcomes',persistent+" && app.actors.size===2 && getComputedStyle(originalYani.querySelector('svg')).animationName==='none'")
            reset()
            browser.call('Emulation.setEmulatedMedia',{'features':[]})

        if not args.functional_only:
            browser.evaluate("window.measureCast=()=>[...app.actors].map(([id,a])=>{const s=a.querySelector('svg'),p=s.createSVGPoint();p.x=120;p.y=a.dataset.pose==='sit'?370:395;const actual=p.matrixTransform(s.getScreenCTM()),stage=app.stage.getBoundingClientRect(),g=app.grounds.get(id),shadow=a.querySelector('.ground-shadow').getBoundingClientRect(),cushion=a.querySelector('.seat-cushion')?.getBoundingClientRect(),r=a.querySelector('button').getBoundingClientRect();return {id,error:Math.abs(actual.y-stage.top-g.groundY),touches:cushion?actual.y>=cushion.top&&actual.y<=cushion.bottom:Math.abs(actual.y-(shadow.top+shadow.height/2))<1,inside:actual.y>0&&actual.y<innerHeight,hit:document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)?.closest('neko-actor')===a}})")
            for width,height in [(320,740),(390,844),(768,1024),(1024,1366),(1440,900),(844,390)]:
                browser.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':height,'deviceScaleFactor':1,'mobile':False})
                time.sleep(.15)
                expect(f'{width}x{height} scene fills viewport without scrolling',"document.documentElement.scrollWidth===innerWidth && document.documentElement.scrollHeight===innerHeight && app.stage.clientWidth===innerWidth && app.stage.clientHeight===innerHeight")
                expect(f'{width}x{height} invitation options stay visible and clickable',"[...document.querySelectorAll('[data-invite-character]')].every(b=>{const r=b.getBoundingClientRect();return r.top>=0&&r.bottom<=innerHeight&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)?.closest('[data-invite-character]')===b})")
                readings=[]
                for identifier in ['none','imouto','yaku','kansai','aru']:
                    invite(identifier)
                    reset()
                    readings.append({'guest':identifier,'actors':browser.evaluate('measureCast()'),'hostPreserved':browser.evaluate(persistent)})
                passed=all(c['hostPreserved'] and all(a['error']<.5 and a['touches'] and a['inside'] and a['hit'] for a in c['actors']) for c in readings)
                checks.append({'check':f'{width}x{height} solo and all guest pairs stay grounded and clickable','passed':passed,'result':readings})
                print(('PASS ' if passed else 'FAIL ')+checks[-1]['check'],flush=True)
                if width in (768,1024):
                    for _ in range(15):
                        key('Tab','Tab',9,blur=False)
                        if browser.evaluate("document.activeElement.dataset.inviteCharacter==='yaku'"):
                            break
                    key('Enter','Enter',13,blur=False)
                    browser.wait_for("app.guestId==='yaku'")
                    reset()
                    expect(f'{width}x{height} keyboard invitation does not scroll the scene',"app.stage.scrollTop===0 && app.stage.scrollLeft===0 && scrollY===0 && measureCast().every(a=>a.error<.5)")
                invite('imouto')
                reset()
                browser.screenshot(f'/tmp/neko2-pair-{width}.png')

        browser.evaluate('true')
        origin=urllib.parse.urlparse(args.url).netloc
        remote=sorted({e['params']['request']['url'] for e in browser.events if e.get('method')=='Network.requestWillBeSent' and urllib.parse.urlparse(e['params']['request']['url']).scheme in ('http','https') and urllib.parse.urlparse(e['params']['request']['url']).netloc!=origin})
        report={'passed':sum(c['passed'] for c in checks),'failed':[c for c in checks if not c['passed']],'errors':browser.errors,'remote_requests':remote}
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 1 if report['failed'] or browser.errors or remote else 0
    finally:
        browser.close()


if __name__=='__main__':
    sys.exit(main())
