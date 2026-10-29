const fs=require('fs'),path=require('path'),http=require('http'),readline=require('readline');
const ROOT=path.resolve(__dirname,'..');
const {chromium}=require(path.join(ROOT,'preview/node_modules/playwright'));
const PORT=18811;
const {pathToFileURL}=require('url');
const MIME={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.json':'application/json','.vrm':'application/octet-stream','.glb':'model/gltf-binary','.png':'image/png','.jpg':'image/jpeg','.css':'text/css'};
(async()=>{
 const assets=process.env.PANDA_ASSETS;
 if(!assets)throw Error('PANDA_ASSETSが必要です');
 const {selectedPandaPerson}=await import(pathToFileURL(path.join(ROOT,'preview/panda-condition.mjs')));
 const people=JSON.parse(fs.readFileSync(path.join(ROOT,'preview/assets/people/manifest.json'),'utf8')).models;
 const person=selectedPandaPerson(process.env.PANDA_PERSON??'original',people);
 const server=http.createServer((request,response)=>{
  const url=new URL(request.url,'http://localhost');
  const isPanda=url.pathname.startsWith('/panda-assets/');
  const root=isPanda?assets:path.join(ROOT,'preview');
  const relative=isPanda?url.pathname.slice('/panda-assets/'.length):url.pathname==='/'?'index.html':url.pathname.slice(1);
  const file=path.resolve(root,decodeURIComponent(relative));
  if(!file.startsWith(path.resolve(root)+path.sep)){response.writeHead(403);response.end();return;}
  fs.readFile(file,(error,body)=>{if(error){response.writeHead(404);response.end();return;}response.setHeader('Content-Type',MIME[path.extname(file)]||'application/octet-stream');response.end(body);});
 });
 await new Promise((resolve,reject)=>{server.once('error',reject);server.listen(PORT,'127.0.0.1',resolve);});
 const browser=await chromium.launch({channel:'chrome',headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1100,height:800}});
  page.on('pageerror',e=>process.stderr.write(e.stack+'\n'));
  const source=fs.readFileSync(path.join(ROOT,'preview/scene.js'),'utf8')+'\n'+fs.readFileSync(path.join(ROOT,'preview/panda-runtime.js'),'utf8')+(process.env.PANDA_HYBRID_RGB==='1'?'\n'+fs.readFileSync(path.join(ROOT,'preview/panda-hybrid-rgb.js'),'utf8'):'');
  await page.route('**/scene.js',route=>route.fulfill({body:source,contentType:'text/javascript'}));
  await page.goto('http://127.0.0.1:'+PORT+'/?person='+encodeURIComponent(person));
  await page.waitForFunction(()=>window.pandaExperiment,{timeout:60000});
  process.stdout.write(JSON.stringify({ready:true,person})+'\n');
  const lines=readline.createInterface({input:process.stdin});
  for await(const line of lines){
   try{
    const request=JSON.parse(line);
    if(request.operation==='close')break;
    if(request.operation==='screenshot'){await page.screenshot({path:request.path});process.stdout.write('{}\n');continue;}
    const result=await page.evaluate(async r=>{
     const api=window.pandaExperiment;
     if(r.operation==='diagnosticParts')return api.diagnosticParts(r.anchors);
      if(r.operation==='visibility')return api.visibility(r.point);
      if(r.operation==='blinkGeometry')return api.blinkGeometry();
     if(r.operation==='teacherV2')return api.teacherV2();
     if(r.operation==='eyeAnatomy')return api.eyeAnatomy();
     if(r.operation==='geometry')return api.geometry();
     if(r.operation==='reset')return api.reset(r.depth,r.phase,r.height,r.horizontal);
     if(r.operation==='setup')return api.setup(r.shapes);
     if(r.operation==='render')return api.render(r.state,r.capture);
     if(r.operation==='score')return api.score(r.start,r.end,r.open);
     if(r.operation==='scoreSegments')return api.scoreSegments(r.segments);
     if(r.operation==='pressureSamples')return api.pressureSamples(r.samples);
     throw Error('未登録の操作');
    },request);
    process.stdout.write(JSON.stringify({result:result??null})+'\n');
   }catch(error){process.stdout.write(JSON.stringify({error:String(error.stack)})+'\n');}
  }
 }finally{await browser.close();server.closeAllConnections();server.close();process.stdin.destroy();}
})().catch(error=>{process.stderr.write(error.stack+'\n');process.exitCode=1;});
