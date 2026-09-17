// Character actions are independent from the shared scene-ground placement.
export const CHARACTER_ACTIONS = [
  { id:'yani', delay:-.8, action:'smoke' },
  { id:'imouto', delay:-1.7, action:'bonk' },
  { id:'yaku', delay:-3.1, action:'sneeze' },
  { id:'kansai', delay:-2.3, action:'fan' },
  { id:'aru', delay:-4.2, action:'hiccup' },
];
export const PROP_LAYOUT = [
  {id:'radio',name:'收音機，點擊讓大家跳舞',x:17,y:89,width:8,mobile:{x:46,y:92,width:17}},
  {id:'fan',name:'電風扇，點擊吹走煙霧',x:94,y:89,width:8.5,mobile:{x:89,y:94,width:17}},
  {id:'lamp',name:'拉燈，切換白天與夜晚',x:77,y:17,width:5,mobile:{x:81,y:13,width:13}},
];
