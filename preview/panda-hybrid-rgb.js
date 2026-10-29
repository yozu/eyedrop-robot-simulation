// 同じ左カメラを認識器が学習した解像度でも描画して比較する。
const pandaRenderBeforeFaceCapture=window.pandaExperiment.render;
window.pandaExperiment.render=function(state,capture=true){
 const result=pandaRenderBeforeFaceCapture.call(this,state,capture);
 if(capture){
  learningRenderer.setSize(256,256);
  learningRenderer.render(scene,learningCamera);
  result.face=learningRenderer.domElement.toDataURL('image/png').split(',')[1];
  if(state.capture_right_face===true){
   learningRenderer.render(scene,pandaCameraRight);
   result.face_right=learningRenderer.domElement.toDataURL('image/png').split(',')[1];
  }
  learningRenderer.setSize(512,512);
  if(state.capture_robot_mask===true){
   // 既知の自己形状だけを描画し、人物・眼の領域情報を混ぜない。
   const maskScene=new THREE.Scene();
   maskScene.background=new THREE.Color(0x000000);
   const maskMaterial=new THREE.MeshBasicMaterial({color:0xffffff,toneMapped:false});
   maskScene.overrideMaterial=maskMaterial;
   maskScene.add(pandaRoot.clone(true));
   try{
    for(const [view,maskCamera] of [['left',learningCamera],['right',pandaCameraRight]]){
     learningRenderer.render(maskScene,maskCamera);
     result['robot_mask_'+view]=learningRenderer.domElement.toDataURL('image/png').split(',')[1];
    }
   }finally{maskMaterial.dispose();}
  }
 }
 return result;
};
