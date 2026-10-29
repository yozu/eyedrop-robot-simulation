// 人物設定は描画・教師・採点だけで扱い、推論の自己状態へ混ぜない。
export function pandaCondition({depth=0,phase=0,height=0,horizontal=0}={}){
 if(![depth,phase,height,horizontal].every(value=>typeof value==='number'&&Number.isFinite(value))||phase<0)
  throw new Error('人物の奥行き・高さ・まばたき位相が不正です');
 return {offset:[horizontal,height,depth],depth_m:depth,height_m:height,blink_phase_s:phase};
}

export function selectedPandaPerson(person,models){
 if(typeof person!=='string'||!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(person))
  throw new Error('人物IDが不正です');
 if(person==='original')return person;
 if(!Array.isArray(models)||!models.some(model=>model.id===person&&model.status==='downloaded_structurally_checked'))
  throw new Error('人物IDが検証済み素材に登録されていません');
 return person;
}
