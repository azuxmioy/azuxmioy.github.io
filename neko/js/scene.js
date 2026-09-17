// Original SVG scenery. A registry + instances reuse every repeated room object.
const objects = {
  books:'<path d="M-36 0h77v12h-77Z" fill="#a9bbae"/><path d="M-29-12h66V0h-66Z" fill="#d69877"/><path d="M-40-23h68v11h-68Z" fill="#e6c86c"/><path d="M-29-19h48m-41 11h46m-49 15h53" fill="none" stroke="#f7e8d0" stroke-width="3"/>',
  plant:'<path d="M-24 0h48l-7 45h-34Z" fill="#bd8063"/><path d="M-28-5h56v10h-56Z" fill="#d9956d"/><path d="M0 0v-72m0 31q-39-1-31-37Q-2-72 0-41Zm0-13q37-6 31-38Q0-88 0-54Zm0 29q-35-1-37-28Q-6-53 0-25Zm0 8q39-3 40-30Q9-49 0-17Z" fill="#809a70" stroke="#647e57" stroke-width="2"/><path d="m0-42-22-25m23 16 20-27m-22 54-24-18m26 23 27-16" fill="none" stroke="#aebc85" stroke-width="1.6"/>',
  cup:'<path d="M17-27q28-1 20 18-6 12-20 6" fill="none" stroke="#798e8b" stroke-width="7"/><path d="M-23-31h43v28q0 12-21 12T-23-3Z" fill="#bed0c2"/><ellipse cy="-31" rx="22" ry="5" fill="#766a4c"/><path d="M-14-20v16" stroke="#e1e5cc" stroke-width="3" stroke-linecap="round"/>',
  can:'<path d="M-12-37q12-6 24 0v39q-12 6-24 0Z" fill="#db967d"/><path d="M-12-28h24v16h-24Z" fill="#f4deb5"/><ellipse cy="-37" rx="12" ry="4" fill="#dcd4bb"/><path d="M-5-37h8" stroke="#8d9279" stroke-width="2"/>',
  pillow:'<path d="M-67-26Q-7-42 57-23l10 39Q4 42-73 16Z" fill="#d8b66c" stroke="#b99a60" stroke-width="2"/><path d="M-57 14q50 18 111 0" fill="none" stroke="#e9cb84" stroke-width="2"/>',
};
const furnishings = [
  ['plant',1450,528,1.6],['plant',458,359,.77],['books',353,365,.95],['books',1385,575,1.1],
  ['pillow',617,625,1,-16],['pillow',973,624,.88,13],['cup',639,862,.83],['can',1006,901,.85,76],['can',1320,868,.8,-12],['books',265,871,.9,-5],
];
let sequence = 0;
export function roomIllustration() {
  const prefix = `room-${++sequence}-`;
  const svg = `<svg class="room-art" viewBox="0 0 1600 1000" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
  <defs>
    <linearGradient id="wall" x2="1" y2="1"><stop stop-color="#f0dfbe"/><stop offset="1" stop-color="#eadbb7"/></linearGradient>
    <linearGradient id="sky" x2="0" y2="1"><stop stop-color="#aecbcc"/><stop offset="1" stop-color="#f2dfb0"/></linearGradient>
    <linearGradient id="floor" x2="0" y2="1"><stop stop-color="#c5aa79"/><stop offset="1" stop-color="#e2c699"/></linearGradient>
    <pattern id="grain" width="13" height="13" patternUnits="userSpaceOnUse"><circle cx="1" cy="4" r=".65" fill="#857451" opacity=".11"/><circle cx="9" cy="11" r=".45" fill="#fff8db" opacity=".45"/></pattern>
    <pattern id="wallpaper" width="34" height="34" patternUnits="userSpaceOnUse"><path d="M17 3v5m-2-2h4" stroke="#a59b72" stroke-width=".7" opacity=".16"/></pattern>
    ${Object.entries(objects).map(([id,body])=>`<g id="${id}">${body}</g>`).join('')}
  </defs>
  <path d="M0 0h1600v1000H0Z" fill="url(#wall)"/><path d="M0 0h1600v670H0Z" fill="url(#wallpaper)"/>
  <path d="M0 0h1600v34H0Z" fill="#c5b18c"/><path d="M0 35h1600" stroke="#fbebcf" stroke-width="5"/>
  <path d="M0 664h1600v336H0Z" fill="url(#floor)"/>
  <g stroke="#a18762" fill="none" opacity=".3" stroke-width="1.7">${Array.from({length:13},(_,i)=>`<path d="M${i*160-160} 665 ${i*225-400} 1000"/>`).join('')}${[701,749,811,891,984].map(y=>`<path d="M0 ${y}h1600"/>`).join('')}</g>
  <path d="M0 645h1600v20H0Z" fill="#bfa079"/><path d="M0 645h1600" stroke="#f4dfba" stroke-width="5"/>
  <g stroke="#927c5e" stroke-width="3"><path d="M53 167h197v479H53Z" fill="#baa483"/><path d="M65 178h172v467H65Z" fill="#d5bc93"/><path d="M79 194h142v263H79Z" fill="#ebd3ad"/><path d="M79 474h142v155H79Z" fill="#dec29a"/><path d="M208 416v20" stroke="#736e58" stroke-width="7" stroke-linecap="round"/><path d="M109 192h82v113h-82Z" fill="#c6ceaf" stroke="#b9ad85"/><path d="m116 298 31-58 39 58Z" fill="#a1b295" stroke="none"/></g>
  <g class="room-window">
    <rect x="598" y="110" width="694" height="386" rx="3" fill="#aa9570"/><path d="M611 121h668v359H611Z" fill="url(#sky)"/>
    <circle class="sky-disc" cx="1115" cy="205" r="45" fill="#fff1b9"/>
    <g class="sky-clouds" fill="#f8efcd" opacity=".65"><path d="M674 189q7-23 28-13 12-33 42-14 22-9 28 17h20q20 2 18 15H674Z"/><path d="M998 271q12-20 29-11 10-26 31-10 24-7 32 17 25-5 30 15H998Z"/></g>
    <g fill="#a4b8a5">${[{x:615,y:370,w:80},{x:710,y:313,w:93},{x:818,y:345,w:56},{x:886,y:359,w:89},{x:991,y:337,w:94},{x:1107,y:319,w:72},{x:1195,y:356,w:84}].map(b=>`<path d="M${b.x} ${b.y}h${b.w}v${480-b.y}h-${b.w}Z"/><g class="city-lights" fill="#d5d4b0">${[0,1,2].map(r=>`<path d="M${b.x+10} ${b.y+11+r*25}h9v12h-9m22-12h9v12h-9"/>`).join('')}</g>`).join('')}</g>
    <path d="M610 455q53-70 97 0 44-98 109 0 53-68 108 0 43-72 105 0 48-81 104 0 81-70 149-2v28H610Z" fill="#bec79d"/>
    <g fill="#cbb592" stroke="#a28d6b" stroke-width="2"><path d="M833 120h10v360h-10m215-360h10v360h-10"/><path d="M611 303h668v8H611Z"/><path d="M584 479h723v17H584Z"/></g>
    <path d="M555 89h779" stroke="#8d7c5e" stroke-width="7" stroke-linecap="round"/>
    <g class="curtain"><path d="M566 92h108q-17 133-20 202t-38 214l-83-12q54-142 33-404" fill="#dbcbac"/><path d="M589 95q28 211-32 398m55-398q-1 214-40 399m65-398q-18 231-37 404" fill="none" stroke="#c7b895" stroke-width="3"/></g>
    <g class="curtain"><path d="M1260 92h64q-11 114 11 240t23 165l-78 9q-18-139-20-414" fill="#dfcfb2"/><path d="M1280 96q-8 213 16 401m10-401q-8 216 28 398" fill="none" stroke="#cbbb9d" stroke-width="3"/></g>
  </g>
  <path class="sunbeam" d="m639 497 627-1 334 436-701 68-548-76Z" fill="#fff0bd" opacity=".19"/>
  <g transform="translate(364 140) rotate(-4)"><path d="M-53-53H53V73H-53Z" fill="#b0956e"/><path d="M-47-47h94V67h-94Z" fill="#f8e7c9"/><path d="m-30 38 29-52 20 31 10-12 10 33Z" fill="#a5b79b"/><circle cx="23" cy="-21" r="12" fill="#d89964"/><path d="M-26 50h55" stroke="#d8c4a0" stroke-width="2"/><path d="M-16-60h33v17h-33Z" fill="#e3d295" opacity=".85"/></g>
  <g transform="translate(477 206) rotate(6)"><path d="M-29-36h58v77h-58Z" fill="#aa926e"/><path d="M-24-31h48v67h-48Z" fill="#e5d6b8"/><path d="m-15 6 5-22 9 10h9l9-10 3 26q-17 14-35-4" fill="#acb494"/><path d="M-7 4h2m12 0h2" stroke="#777d62" stroke-width="2"/></g>
  <path d="M288 363h230v14H288Z" fill="#b49a73"/><path d="M310 377v24m181-24v24" stroke="#9e865f" stroke-width="6"/>
  <g class="sofa" stroke="#75866c" stroke-width="2.5"><path d="M507 520q0-48 50-51h409q63 0 64 58v128H507Z" fill="#98a78a"/><path d="M532 502q0-14 20-15h196v145H532Zm226-15h208q37 0 37 32v111H758Z" fill="#b0b899"/><path d="M517 619q256-18 505 0v60H517Z" fill="#91a080"/><path d="M532 621q105-13 216-1v45H532Zm226-1q132-10 248 2v43H758Z" fill="#bcc09d"/><rect x="485" y="561" width="53" height="130" rx="18" fill="#a5b091"/><rect x="1005" y="561" width="53" height="130" rx="18" fill="#a5b091"/><path d="M512 690v21m516-21v21" stroke="#766848" stroke-width="13"/><path d="M548 514h180m50 0h205M545 643h189m38 0h220" stroke="#c4c8a9" stroke-width="2" fill="none"/></g>
  <path d="M363 786q402-93 810-6l180 197q-583 64-1098 0Z" fill="#a9ac85" opacity=".7"/><path d="M384 797q398-78 768-3m-831 151q510 44 974 0" fill="none" stroke="#c3c39a" stroke-width="5" opacity=".7"/>
  <g class="coffee-table"><path d="m530 869-22 84m337-84 24 84" stroke="#917551" stroke-width="15"/><ellipse cx="687" cy="865" rx="202" ry="53" fill="#b59060"/><ellipse cx="687" cy="855" rx="202" ry="53" fill="#d9b885" stroke="#b39363" stroke-width="3"/><path d="M515 853q164 56 350 2" fill="none" stroke="#e5cc9f" stroke-width="2"/></g>
  <g><path d="M1346 561h160v35h-160Z" fill="#b89970"/><path d="M1358 596v80m134-80v80" stroke="#9e7f59" stroke-width="9"/></g>
  ${furnishings.map(([kind,x,y,scale,rotate=0])=>`<use href="#${kind}" transform="translate(${x} ${y}) scale(${scale}) rotate(${rotate})"/>`).join('')}
  <path d="M0 0h1600v1000H0Z" fill="url(#grain)" pointer-events="none"/>
  </svg>`;
  return svg.replace(/\bid="([^"]+)"/g,(_,id)=>`id="${prefix}${id}"`).replace(/url\(#([^)]+)\)/g,(_,id)=>`url(#${prefix}${id})`).replace(/href="#([^"]+)"/g,(_,id)=>`href="#${prefix}${id}"`);
}

export function renderProp(id) {
  const art = {
    radio:`<rect x="8" y="34" width="164" height="100" rx="13" fill="#c9835f" stroke="#775c49" stroke-width="4"/><path d="m112 33 28-23" stroke="#776e57" stroke-width="4" stroke-linecap="round"/><rect x="18" y="44" width="90" height="79" rx="8" fill="#5b645b"/><g stroke="#929681" stroke-width="2">${Array.from({length:9},(_,i)=>`<path d="M28 ${53+i*7}h70"/>`).join('')}</g><rect x="118" y="48" width="43" height="22" rx="3" fill="#e7d5a6"/><path d="M123 63h31m-21-8v10" stroke="#aa7d4e" stroke-width="2"/><circle cx="139" cy="99" r="16" fill="#ece0bd" stroke="#896c4f" stroke-width="3"/><path d="m139 99 9-6" stroke="#896c4f" stroke-width="3"/><path d="M22 135v6m133-6v6" stroke="#7e694e" stroke-width="6"/>`,
    fan:`<path d="M83 90h16l5 61H77Z" fill="#92aa9d" stroke="#688178" stroke-width="3"/><ellipse cx="91" cy="155" rx="45" ry="9" fill="#a7b8a6" stroke="#748c7d" stroke-width="3"/><circle cx="91" cy="64" r="52" fill="#d8dfc0" stroke="#718c82" stroke-width="4"/><g class="fan-blades" fill="#8fafa3">${[0,120,240].map(r=>`<path d="M91 64q-45-4-31-35 13-20 30 5Z" transform="rotate(${r} 91 64)"/>`).join('')}</g><g fill="none" stroke="#698278" opacity=".7" stroke-width="1.8"><circle cx="91" cy="64" r="42"/><circle cx="91" cy="64" r="31"/><path d="M91 13v103M39 64h104m-89-36 74 73m-74 0 74-73"/></g><circle cx="91" cy="64" r="9" fill="#dce2c8" stroke="#748b7c" stroke-width="2"/>`,
    lamp:'<path d="M91-100v134" stroke="#776e51" stroke-width="3"/><path d="M57 32h67l27 62H31Z" fill="#d8af62" stroke="#9a7c48" stroke-width="3"/><ellipse cx="91" cy="94" rx="60" ry="10" fill="#f4dda6" stroke="#9a7c48" stroke-width="3"/><path d="M136 96v34" stroke="#a29672" stroke-width="2"/><circle cx="136" cy="133" r="5" fill="#7c8668"/><path d="M70 40 55 84" stroke="#f2d88d" stroke-width="4"/><ellipse class="lamp-glow" cx="91" cy="96" rx="23" ry="5" fill="#fff8cc"/>',
  };
  return `<svg viewBox="0 0 180 180" aria-hidden="true">${art[id]||''}</svg>`;
}
