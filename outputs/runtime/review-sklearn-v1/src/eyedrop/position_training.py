"""位置モデルの更新範囲を明示し、開眼出力の重みを保護する。"""
def position_trainable_parameters(model,*,adapt_features=False):
    for parameter in model.parameters():parameter.requires_grad_(False)
    modules=(model.readout,model.fine,model.context) if adapt_features else (model.readout,)
    for module in modules:
        for parameter in module.parameters():parameter.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]
