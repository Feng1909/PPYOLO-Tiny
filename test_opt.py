import paddlelite.lite as lite

a=lite.Opt()
# 非 combined 形式
# a.set_model_dir("F:\\yolo\\best_model")

# conmbined 形式，具体模型和参数名称，请根据实际修改
a.set_model_file("F:\\yolo\\inference_model\\inference_model\\model.pdmodel")
a.set_param_file("F:\\yolo\\inference_model\\inference_model\\model.pdiparams")

a.set_optimize_out("ppyolotiny")
a.set_valid_places("arm")

a.run()