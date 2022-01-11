import paddlex as pdx

# model = pdx.load_model('best_model')
model = pdx.load_model('output/ppyoloTiny/best_model')
image_name = 'train.jpg'
result = model.predict(image_name)
pdx.det.visualize(image_name, result, threshold=0.5)