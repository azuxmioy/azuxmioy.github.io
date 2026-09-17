/**
 * Original, hand-drawn SVG character kit. All geometry is authored here.
 * No artwork, fonts, paths, or image files are taken from the animation website.
 * The cast shares a face, hands, anatomy and garment renderers; only appearance
 * data and hair silhouettes differ. Mouth origin: (120, 119); cigarette tip: approximately (168, 130).
 */
export const CHARACTERS = Object.freeze([
  { id: 'yani', name: '尼古喵喵', appearance: { hair: '#a7afa5', hairShade: '#818d85', hairLight: '#c4cbbd', eye: '#b88835', skin: '#ffe8ce', outfit: 'tee', shirt: '#f2ecce', shirtShade: '#d7d4b7', pants: '#627b87', accent: '#dcab67', hairstyle: 'shag' } },
  { id: 'imouto', name: '妹妹喵', appearance: { hair: '#b9b5c9', hairShade: '#9291ad', hairLight: '#d8d0dd', eye: '#a54e38', skin: '#ffe7d2', outfit: 'uniform', shirt: '#424658', shirtShade: '#313444', pants: '#424658', accent: '#a94542', hairstyle: 'straight' } },
  { id: 'yaku', name: '藥喵', appearance: { hair: '#dac28a', hairShade: '#b9a073', hairLight: '#f4dca4', eye: '#8c7557', skin: '#ffead2', outfit: 'sweater', shirt: '#559b99', shirtShade: '#397e80', pants: '#454d51', accent: '#f2b16a', hairstyle: 'bob' } },
  { id: 'kansai', name: '關西喵', appearance: { hair: '#816553', hairShade: '#604f47', hairLight: '#a78665', eye: '#8c955e', skin: '#fce3c6', outfit: 'ribbed', shirt: '#ddd1ae', shirtShade: '#beb694', pants: '#494647', accent: '#a38059', hairstyle: 'long' } },
  { id: 'aru', name: '酒喵', appearance: { hair: '#ab6d4d', hairShade: '#865137', hairLight: '#c68a5d', eye: '#aa753d', skin: '#ffe3c4', outfit: 'hoodie', shirt: '#e6dac0', shirtShade: '#c8b695', pants: '#74775b', accent: '#845f46', hairstyle: 'curly' } },
]);

const INK = '#3f3935';
const LINE = `stroke="${INK}" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"`;
const path = (d, fill, extra = '') => {
  const attributes = { stroke: INK, 'stroke-width': '2.6', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' };
  for (const match of extra.matchAll(/([\w-]+)="([^"]*)"/g)) attributes[match[1]] = match[2];
  return `<path d="${d}" fill="${fill}" ${Object.entries(attributes).map(([key, value]) => `${key}="${value}"`).join(' ')}/>`;
};
const line = (d, extra = '') => path(d, 'none', extra);
const skinShade = '#eec5ac';

function ears(a) {
  return `<g class="cat-ears">
    ${path('M67 63 Q56 42 60 16 Q81 22 94 45Z', a.hair)}
    ${path('M148 42 Q162 21 180 14 Q180 42 169 64Z', a.hair)}
    ${path('M68 46 L67 28 Q79 35 82 47Z', '#d7a6a0', 'stroke-width="1.6"')}
    ${path('M158 45 Q164 33 173 26 L169 49Z', '#d7a6a0', 'stroke-width="1.6"')}
    ${line('M62 24 L65 38 M177 22 L174 38', 'stroke-width="1.4"')}
  </g>`;
}

function backHair(a) {
  const variants = {
    shag: 'M63 68 Q53 31 98 28 Q145 14 174 53 Q185 85 174 111 L185 119 L168 123 L175 137 L158 132 L149 147 L139 133 L87 141 L77 130 L62 134 L66 120 L53 119 L61 106 Q52 90 63 68Z',
    straight: 'M61 71 Q58 30 104 26 Q159 21 175 64 L180 137 Q169 146 153 143 L82 144 L60 136Z',
    bob: 'M58 78 Q53 35 96 28 Q140 15 169 47 Q183 62 176 112 L183 126 L166 124 L162 145 L151 131 L83 145 L75 131 L55 130 L63 115Z',
    long: 'M61 76 Q55 30 100 26 Q145 13 169 49 Q181 70 178 109 Q171 163 188 213 L181 228 L172 219 L178 250 Q154 239 146 218 L85 229 Q72 247 58 247 L66 222 L54 224 Q69 176 61 142Z',
    curly: 'M65 65 Q49 33 92 30 Q99 13 132 25 Q165 24 176 58 Q184 71 177 93 Q192 110 181 130 Q199 150 183 171 Q198 190 181 205 Q188 228 168 239 L154 215 L85 233 Q57 247 59 222 Q41 209 56 190 Q43 175 57 156 Q46 137 57 121 Q44 102 62 89Z',
  };
  return `<g class="cat-hair-back">${path(variants[a.hairstyle], a.hairShade)}
    ${a.hairstyle === 'curly' ? `${path('M116 31 Q95 14 112 6 Q122 1 132 9 Q154 5 151 22 L135 39Z', a.hair)}${line('M117 14 Q130 12 136 24 M141 13 L134 27', 'stroke-width="1.5"')}` : ''}
    ${['long', 'curly'].includes(a.hairstyle) ? `${line('M69 111 Q80 161 67 199 M170 103 Q158 149 174 187 M72 208 L69 226 M171 206 L175 226', 'stroke-width="1.5" opacity=".55"')}` : ''}
  </g>`;
}

function bangs(a) {
  const variants = {
    shag: 'M60 76 Q57 39 86 33 Q94 26 107 30 Q149 15 171 54 L179 91 L165 85 L165 105 L151 91 L149 70 L139 84 L133 65 L121 85 L111 63 L101 83 L93 68 L83 93 L77 85 L70 109 L64 95 L55 100Z',
    straight: 'M61 75 Q57 34 99 28 Q151 20 170 54 L177 119 L165 120 L157 72 L147 77 L146 60 L138 77 L124 77 L119 64 L117 77 L103 78 L102 64 L96 79 L81 79 L73 122 L62 121Z',
    bob: 'M57 80 Q54 37 97 28 Q138 20 164 43 Q178 55 177 88 L170 123 L159 117 L157 71 L146 85 L138 73 L128 83 L123 67 L113 83 L104 68 L96 84 L84 75 L76 119 L63 122 L67 92Z',
    long: 'M61 76 Q60 35 97 29 Q146 14 169 49 Q177 64 173 86 L164 123 L157 112 L157 72 L149 63 Q141 83 120 85 L127 72 Q108 83 88 78 L80 98 L75 122 L64 120Z',
    curly: 'M62 75 Q52 39 94 30 Q128 19 155 36 Q178 43 176 75 L169 98 L165 123 L153 112 L159 83 L149 63 Q140 81 123 83 L129 66 Q106 90 91 78 L89 63 L79 84 L77 115 L64 124 L69 101 L58 94Z',
  };
  return `<g class="cat-fringe">${path(variants[a.hairstyle], a.hair)}
    ${a.hairstyle === 'shag' ? `${path('M69 61 Q81 35 105 37 Q139 25 158 49 Q144 39 130 40 L128 55 L115 42 L112 56 L99 45 L89 58 L87 48Z', a.hairLight, 'stroke="none" opacity=".6"')}${line('M85 39 Q77 48 76 61 M148 39 Q158 47 163 62 M108 34 L108 49', 'stroke-width="1.4"')}` : `${path('M70 54 Q85 35 108 34 Q139 30 156 49 Q138 39 117 40 Q89 40 70 64Z', a.hairLight, 'stroke="none" opacity=".55"')}${line('M77 52 Q87 40 103 38 M147 42 Q160 53 163 68', 'stroke-width="1.3"')}`}
    ${a.hairstyle === 'straight' ? `${line('M67 84 L66 109 M169 86 L171 109 M89 51 L85 66 M111 44 L110 64 M137 46 L139 65', 'stroke-width="1.3"')}` : ''}
    ${a.hairstyle === 'long' ? `${line('M148 43 Q146 56 135 65 M93 43 Q80 61 81 76', 'stroke-width="1.4"')}` : ''}
  </g>`;
}

function eye(x, a, {tired = false, under = false} = {}) {
  return `<g transform="translate(${x} 91)">
    ${under ? path('M-12 3 Q0 15 13 4 Q10 19 -5 16 Q-14 13 -12 3Z', '#c4a5a0', 'stroke="none"') : ''}
    ${path(tired ? 'M-14 -1 L13 1 Q13 14 0 14 Q-11 14 -14 -1Z' : 'M-14 3 Q-1 -9 13 0 Q14 15 1 15 Q-12 15 -14 3Z', '#fff7e9', 'stroke-width="1.5"')}
    <ellipse cx="1" cy="${tired ? '6' : '5'}" rx="6.3" ry="8.1" fill="${a.eye}"/>
    <ellipse cx="1" cy="6" rx="2.1" ry="6.3" fill="${INK}"/>
    <circle cx="3" cy="2" r="1.8" fill="#fff7e3"/>
    ${line(tired ? 'M-16 -1 Q-2 0 15 1' : 'M-15 3 Q-2 -9 15 1', 'stroke-width="3.4"')}
    ${line('M-15 1 L-17 -2 M12 0 L15 -3', 'stroke-width="1.5"')}
    ${tired ? line('M-12 17 Q-1 20 9 17', 'stroke="#b58e7d" stroke-width="1.2"') : ''}
  </g>`;
}

function face(a, id) {
  const tired = ['yani', 'yaku', 'kansai'].includes(id);
  const closed = id === 'aru';
  return `<g class="cat-face">
    ${path('M75 73 Q79 56 119 56 Q158 57 166 77 L160 111 Q153 133 122 139 Q92 135 80 115Z', a.skin)}
    ${path('M78 81 L85 97 L87 116 Q100 134 123 139 Q95 136 81 119Z', skinShade, 'stroke="none" opacity=".6"')}
    ${path('M76 95 Q66 90 69 104 Q73 114 82 113 M165 95 Q175 90 171 103 Q169 112 160 113', a.skin, 'stroke-width="2"')}
    <g fill="#e8a09a" opacity="${id === 'aru' ? '.65' : '.3'}"><ellipse cx="90" cy="113" rx="10" ry="5"/><ellipse cx="149" cy="113" rx="10" ry="5"/></g>
    <g class="cat-eyes expression-normal">
      ${closed ? `${line('M85 99 Q96 86 106 97 M133 97 Q144 86 156 98', 'stroke-width="3.2"')}${line('M86 102 L84 99 M155 101 L159 98', 'stroke-width="1.5"')}` : eye(97, a, {tired, under: id === 'yaku'}) + eye(143, a, {tired, under: id === 'yaku'})}
      ${line(id === 'imouto' ? 'M83 82 L105 81 M133 81 L155 83' : id === 'aru' ? 'M85 85 Q94 81 102 85 M137 85 Q146 81 154 85' : 'M82 80 Q94 83 105 81 M134 81 Q147 84 157 80', `stroke-width="${id === 'imouto' ? 4 : 1.8}"`)}
    </g>
    <g class="cat-eyes expression-shock">
      <ellipse cx="96" cy="97" rx="11" ry="15" fill="#fff9ed" ${LINE}/><ellipse cx="144" cy="97" rx="11" ry="15" fill="#fff9ed" ${LINE}/>
      <ellipse cx="96" cy="99" rx="2.7" ry="5" fill="${INK}"/><ellipse cx="144" cy="99" rx="2.7" ry="5" fill="${INK}"/>
      ${line('M85 76 L104 73 M135 73 L155 76', 'stroke-width="2.4"')}
    </g>
    <g class="cat-eyes expression-annoyed">
      ${eye(97, a, {tired: true})}${eye(143, a, {tired: true})}
      ${line('M83 79 L106 86 M133 86 L157 78', 'stroke-width="3.2"')}
      ${line('M151 68 L154 73 L160 71 M160 62 L159 68 L165 70', 'stroke="#a34936" stroke-width="2.2"')}
    </g>
    <g class="cat-eyes expression-goofy">
      ${line('M85 92 L105 98 L85 104 M155 92 L135 98 L155 104', 'stroke-width="3"')}
      ${line('M81 113 L84 118 M88 113 L90 118 M151 112 L153 117 M158 110 L160 115', 'stroke="#cf7f75" stroke-width="1.3"')}
    </g>
    ${line('M121 102 L119 107 L123 108', 'stroke="#bf9984" stroke-width="1.2"')}
    <g class="cat-mouth expression-normal">
      ${id === 'aru' ? path('M110 115 Q121 122 132 115 Q130 134 120 133 Q112 130 110 115Z', '#9d5146', 'stroke-width="1.8"') : id === 'yani' ? line('M111 119 Q117 117 123 119 L129 117', 'stroke-width="1.8"') : line('M112 120 Q120 124 128 119', 'stroke-width="1.8"')}
      ${id === 'aru' ? path('M115 127 Q122 121 128 128 Q121 135 115 127', '#e79d8d', 'stroke="none"') : ''}
    </g>
    <g class="cat-mouth expression-shock">${path('M115 118 Q122 112 129 119 L127 130 Q120 136 115 129Z', '#925045', 'stroke-width="1.7"')}</g>
    <g class="cat-mouth expression-annoyed">${line('M111 123 Q119 117 130 121', 'stroke-width="2"')}</g>
    <g class="cat-mouth expression-goofy">${path('M109 115 Q120 122 133 113 Q131 131 121 132 Q113 130 109 115Z', '#955047', 'stroke-width="1.8"')}${path('M120 126 Q126 121 130 126 L127 138 Q120 142 118 135Z', '#e29689', 'stroke-width="1.6"')}</g>
  </g>`;
}

function tail(a, pose) {
  return `<g class="cat-tail">${path(pose === 'sit' ? 'M171 294 Q221 300 218 258 Q216 235 226 234 Q239 234 236 258 Q233 316 184 319 L167 312Z' : 'M160 289 Q209 300 214 266 Q218 244 227 246 Q237 249 230 274 Q220 313 179 309Z', a.hair)}${line(pose === 'sit' ? 'M219 253 Q224 264 226 267' : 'M216 263 L229 269', 'stroke-width="1.4" opacity=".45"')}</g>`;
}

function legs(a, pose) {
  if (pose === 'sit') {
    return `<g class="cat-legs">
      ${path('M78 256 Q67 268 48 296 Q31 319 45 344 Q62 359 101 360 L167 354 Q201 348 204 328 Q201 308 172 288 L159 259Z', a.pants)}
      ${path('M70 291 Q85 306 131 318 L169 340 Q148 359 88 356 L68 344 Q108 342 121 335 Q79 334 56 316Z', '#4d626c', 'stroke="none" opacity=".48"')}
      ${line('M67 286 Q82 307 126 320 M166 287 Q146 308 134 318 M72 344 Q103 344 130 332 M165 342 Q170 331 158 323', 'stroke-width="2"')}
      ${path('M91 338 Q72 336 60 347 L50 355 Q47 364 60 367 L103 365 Q115 365 112 355Z', a.skin)}
      ${path('M153 347 Q173 341 181 350 L192 359 Q195 368 183 370 L148 366 Q140 363 144 356Z', a.skin)}
      ${line('M55 357 L60 360 M62 352 L68 356 M184 359 L180 362 M178 354 L174 358', 'stroke-width="1.2"')}
    </g>`;
  }
  const skirt = ['uniform', 'ribbed'].includes(a.outfit);
  return `<g class="cat-legs">
    ${path('M87 248 L117 252 L115 371 L90 372 Q86 334 88 313Z', skirt ? a.skin : a.pants)}
    ${path('M119 252 L151 249 L156 371 L130 373 L120 310Z', skirt ? a.skin : a.pants)}
    ${!skirt ? `${path('M104 273 L115 271 L111 369 L104 367Z M140 270 L148 268 L152 368 L144 367Z', '#242e35', 'stroke="none" opacity=".15"')}${line('M96 310 L104 315 M143 337 L150 333 M121 263 L123 286', 'stroke-width="1.5"')}` : `${path('M89 333 L115 333 L113 376 L90 376Z M129 333 L154 333 L157 376 L132 376Z', a.outfit === 'uniform' ? '#373740' : '#ede3c5')}${line('M91 338 L112 338 M132 338 L151 338', 'stroke-width="1.2"')}`}
    ${path('M91 369 Q102 375 114 370 L117 385 Q117 393 104 395 L74 393 Q66 388 73 382Z', a.outfit === 'uniform' ? '#665648' : '#e7dcc1')}
    ${path('M132 370 Q143 375 155 369 L172 382 Q178 390 168 393 L139 395 Q129 394 129 386Z', a.outfit === 'uniform' ? '#665648' : '#e7dcc1')}
    ${line('M74 388 Q94 392 113 387 M134 389 Q152 391 170 387', 'stroke-width="1.5"')}
  </g>`;
}

function body(a) {
  const basic = 'M97 143 Q80 145 70 160 L79 213 L76 269 Q119 281 165 266 L158 211 L169 161 Q155 148 143 145Z';
  const uniform = a.outfit === 'uniform';
  const hoodie = a.outfit === 'hoodie';
  const ribbed = a.outfit === 'ribbed';
  return `<g class="cat-body">
    ${path('M107 129 L105 151 Q120 166 137 151 L133 129Z', a.skin)}
    ${path('M109 133 Q120 143 134 133 L135 142 Q120 149 108 142Z', skinShade, 'stroke="none"')}
    ${path(basic, a.shirt)}
    ${path('M79 210 Q87 241 82 263 Q119 273 158 263 L157 247 Q115 257 89 246 L89 203Z', a.shirtShade, 'stroke="none"')}
    ${path(uniform ? 'M101 146 L120 190 L140 146 L136 141 L120 153 L107 142Z' : ribbed ? 'M103 141 L103 156 Q120 166 140 155 L138 139Z' : 'M101 147 Q120 162 142 146 L139 155 Q120 170 104 157Z', uniform ? '#faf0d7' : a.shirtShade, 'stroke-width="1.8"')}
    ${uniform ? `
      ${path('M84 254 L153 254 L174 290 Q119 307 68 290Z', a.pants)}
      ${line('M86 259 L80 288 M102 262 L99 294 M120 264 L120 297 M140 261 L145 293 M154 260 L163 287', 'stroke-width="1.4"')}
      ${path('M100 145 L99 162 L112 181 L120 169Z M141 145 L141 162 L128 181 L120 169Z', '#676976', 'stroke-width="1.7"')}
      ${path('M118 164 L102 157 L103 176 L119 170 L133 178 L137 159 L123 165Z', a.accent, 'stroke-width="1.7"')}
      ${path('M117 165 L123 164 L124 172 L118 172Z', '#b96953', 'stroke-width="1.3"')}
      ${line('M120 180 L121 263 M86 226 L105 226 M136 226 L153 226', 'stroke-width="1.5"')}
      <circle cx="124" cy="205" r="2" fill="#c6af80"/><circle cx="124" cy="234" r="2" fill="#c6af80"/>
    ` : ''}
    ${ribbed ? `${path('M81 261 L160 261 L172 300 Q121 316 68 298Z', a.pants)}${line('M83 273 L78 298 M103 274 L101 304 M139 275 L143 304 M156 271 L163 298', 'stroke-width="1.4"')}${Array.from({length: 8}, (_, i) => line(`M${92 + i * 8} 170 Q${90 + i * 8} 211 ${87 + i * 10} 261`, 'stroke="#b6ab8d" stroke-width="1" opacity=".7"')).join('')}${line('M111 143 L111 155 M120 143 L120 159 M130 143 L130 157', 'stroke-width="1" stroke="#a99c81"')}` : ''}
    ${hoodie ? `${path('M95 145 Q83 150 91 175 Q101 189 121 184 Q137 187 151 173 Q161 155 144 144 L136 155 L121 163 L108 155Z', a.accent)}${line('M99 154 Q99 170 118 175 M144 153 Q143 170 126 176', 'stroke="#b59670" stroke-width="2"')}${path('M93 222 L107 211 L137 211 L150 221 L145 244 L98 244Z', a.shirtShade, 'stroke-width="1.8"')}${line('M104 176 L101 202 M136 175 L140 199', 'stroke-width="2"')}${path('M99 198 L104 198 L104 204 L99 204Z M137 196 L142 196 L143 202 L138 202Z', a.accent, 'stroke-width="1.2"')}` : ''}
    ${a.outfit === 'tee' ? `${line('M79 239 L93 235 M147 248 L156 253 M91 271 Q121 276 151 270', 'stroke-width="1.3"')}${path('M110 188 Q114 182 121 186 Q128 183 132 190 L128 193 Q120 198 112 193Z', '#d4ccb0', 'stroke="none"')}` : ''}
    ${a.outfit === 'sweater' ? `${line('M82 260 Q118 272 159 259 M89 265 L89 270 M97 267 L97 272 M106 270 L106 274 M115 271 L115 275 M125 271 L125 275 M136 270 L136 274 M147 267 L147 272 M155 264 L155 269', 'stroke-width="1.2"')}${line('M90 186 Q101 196 103 212 M150 185 L140 200', 'stroke-width="1.5"')}` : ''}
  </g>`;
}

function hand(a, {side, pose, outfit}) {
  // The right hand lifts toward the cigarette, the left relaxes at the knee.
  const smokeHand = side === 'right' && pose === 'sit';
  if (smokeHand) return `<g class="cat-arm-right">
    ${path('M154 154 Q168 154 177 173 L184 196 L164 207 L149 177Z', a.shirt)}
    ${path('M175 190 Q183 194 181 202 L163 223 Q155 230 148 222 L137 179 L143 162 L151 164 L154 183 L156 203Z', a.skin)}
    ${path('M138 176 L131 149 Q130 142 134 142 Q138 142 142 156 L140 138 Q141 133 145 137 L149 155 L148 144 Q150 139 153 144 L157 166 Q157 172 150 180Z', a.skin, 'stroke-width="2"')}
    ${line('M146 159 L150 171 M138 172 L144 171 M163 214 L176 198', 'stroke-width="1.3"')}
    ${line('M161 198 L177 190', 'stroke-width="1.7"')}
  </g>`;
  const left = side === 'left';
  const flip = left ? '' : 'transform="translate(240 0) scale(-1 1)"';
  const shortSleeve = outfit === 'tee';
  const bent = pose === 'sit';
  return `<g class="cat-arm-${side}"><g ${flip}>
    ${path(shortSleeve ? 'M83 153 Q67 151 61 172 L54 194 Q66 203 82 202 L93 175Z' : 'M82 153 Q66 150 60 174 L54 214 L62 248 Q74 253 85 243 L79 211 L95 175Z', a.shirt)}
    ${path(shortSleeve ? bent ? 'M57 198 L54 234 Q54 245 63 253 L89 275 L101 266 L73 237 L78 202Z' : 'M57 198 L54 235 L60 273 L75 274 L76 237 L80 202Z' : bent ? 'M64 246 L70 261 L91 278 L103 269 L83 247Z' : 'M63 246 L62 267 L76 273 L82 246Z', a.skin)}
    ${path(bent ? 'M90 265 Q97 261 100 265 L111 280 Q115 287 111 288 L103 280 L108 291 Q107 297 102 291 L94 281 L99 292 Q96 297 92 291 L84 278 Q82 274 86 269Z' : 'M61 266 Q57 269 57 281 L60 292 Q62 295 64 291 L64 281 L68 295 Q71 298 73 293 L72 282 L76 291 Q79 293 81 289 L79 274 L76 269 L71 278 L71 270Z', a.skin, 'stroke-width="2"')}
    ${line(shortSleeve ? 'M57 194 Q68 201 81 198' : 'M62 240 Q75 247 85 240 M60 207 L67 213', 'stroke-width="1.5"')}
  </g></g>`;
}

function cigarette() {
  return `<g class="cat-cigarette" transform="rotate(13 122 119)">
    ${path('M120 116 L168 116 L168 122 L120 122Z', '#f6eddc', 'stroke-width="1.5"')}
    ${path('M120 116 L134 116 L134 122 L120 122Z', '#c99865', 'stroke-width="1.1"')}
    ${path('M163 116 L168 116 L169 119 L167 122 L163 122Z', '#858075', 'stroke-width="1.1"')}
    <path class="cat-ember" d="M163 116 L164 122" stroke="#ea693f" stroke-width="3"/>
    <g class="cat-smoke" fill="none" stroke="#fff5e4" stroke-width="3" opacity=".65"><path d="M170 115 Q181 103 169 93 Q157 81 171 68"/><path d="M174 97 Q182 87 177 81" stroke-width="2"/></g>
  </g>`;
}

export function renderCharacter(characterOrId, {pose = 'stand', instanceId, portrait = false} = {}) {
  const character = typeof characterOrId === 'string' ? CHARACTERS.find(item => item.id === characterOrId) : characterOrId;
  if (!character || !character.appearance || !character.id) throw new TypeError('Unknown character');
  const {appearance: a, id} = character;
  const safeInstance = String(instanceId || `${id}-${pose}`).replace(/[^a-zA-Z0-9_-]/g, '');
  return `<svg class="character-vector character-${id}" data-character="${id}" data-pose="${pose}" data-instance="${safeInstance}" viewBox="${portrait ? '48 7 145 138' : '0 0 240 420'}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" focusable="false">
    ${tail(a, pose)}
    ${legs(a, pose)}
    <g class="cat-back-hair">${['long', 'curly'].includes(a.hairstyle) ? backHair(a) : ''}</g>
    ${body(a)}
    ${hand(a, {side: 'left', pose, outfit: a.outfit})}
    ${hand(a, {side: 'right', pose, outfit: a.outfit})}
    <g class="cat-head">
      ${!['long', 'curly'].includes(a.hairstyle) ? backHair(a) : ''}
      ${ears(a)}
      ${face(a, id)}
      ${bangs(a)}
      ${cigarette()}
    </g>
  </svg>`;
}
