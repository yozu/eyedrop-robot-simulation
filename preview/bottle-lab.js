const form=document.getElementById('trial'),button=document.getElementById('run');
const result=document.getElementById('result'),plots=document.getElementById('plots');
const PRESETS={light:{target_mm:.2,hold_s:2,press_speed_mm_s:2,release_speed_mm_s:2},
 deep:{target_mm:2,hold_s:2,press_speed_mm_s:2,release_speed_mm_s:2},
 delay:{target_mm:1,hold_s:.45,press_speed_mm_s:2,release_speed_mm_s:.1}};
const SVG_NS='http://www.w3.org/2000/svg',W=480,H=220,LEFT=55,RIGHT=15,TOP=12,BOTTOM=30;
function node(tag,attributes={},text=''){const n=document.createElementNS(SVG_NS,tag);for(const [k,v] of Object.entries(attributes))n.setAttribute(k,v);n.textContent=text;return n;}
function plot(data,field,title){
 const figure=document.createElement('figure'),caption=document.createElement('figcaption');caption.textContent=title;figure.append(caption);
 const svg=node('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':title});figure.append(svg);
 const values=data.frames.map(f=>f[field]),low=Math.min(0,...values),high=Math.max(.01,...values),end=data.frames.at(-1).time_s;
 const x=t=>LEFT+t/end*(W-LEFT-RIGHT),y=v=>TOP+(high-v)/(high-low)*(H-TOP-BOTTOM);
 for(let i=0;i<=4;i++){const v=low+(high-low)*i/4;svg.append(node('line',{x1:LEFT,y1:y(v),x2:W-RIGHT,y2:y(v),stroke:'#e1e7e2'}));svg.append(node('text',{x:LEFT-7,y:y(v)+4,'text-anchor':'end','font-size':11,fill:'#5a716d'},v.toFixed(2)));}
 for(let t=0;t<=end;t++){svg.append(node('text',{x:x(t),y:H-7,'text-anchor':'middle','font-size':11,fill:'#5a716d'},`${t}s`));}
 svg.append(node('line',{x1:x(data.release_at_s),x2:x(data.release_at_s),y1:TOP,y2:H-BOTTOM,stroke:'#5a716d','stroke-dasharray':'4 4'}));
 for(const drop of data.drops)svg.append(node('line',{x1:x(drop.time_s),x2:x(drop.time_s),y1:TOP,y2:H-BOTTOM,stroke:'#aa5838',opacity:.6}));
 svg.append(node('polyline',{points:data.frames.map(f=>`${x(f.time_s)},${y(f[field])}`).join(' '),fill:'none',stroke:'#176d61','stroke-width':2}));plots.append(figure);
}
async function run(){button.disabled=true;result.textContent='計算中…';
 try{const body=Object.fromEntries(new FormData(form).entries());for(const key of Object.keys(body))body[key]=Number(body[key]);
  const response=await fetch('/api/bottle/trial',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!response.ok)throw new Error(`計算に失敗しました (${response.status})`);
  const data=await response.json();window.bottleTrial=data;plots.replaceChildren();
  result.textContent=`${data.metrics.count}滴 ／ 最初の滴：${data.metrics.first_drop_s===null?'なし':data.metrics.first_drop_s.toFixed(2)+'秒'} ／ 解除指令後：${data.metrics.after_release}滴`;
  plot(data,'displacement_mm','実際の押し込み量（mm）');plot(data,'pressure_kpa','容器内の相対圧力（kPa・モデル値）');
  plot(data,'tip_ml','先端にたまった液量（mL）');plot(data,'drop_count','離れた滴の累計（滴）');
 }catch(error){result.textContent=error.message;plots.replaceChildren();}finally{button.disabled=false;}}
form.onsubmit=event=>{event.preventDefault();run();};
for(const preset of document.querySelectorAll('[data-preset]'))preset.onclick=()=>{for(const [k,v] of Object.entries(PRESETS[preset.dataset.preset]))form.elements[k].value=v;run();};
