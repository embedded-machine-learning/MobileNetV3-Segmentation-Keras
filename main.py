from model.deeplab_v3 import Deeplabv3

model = Deeplabv3(weights=None, OS=32)
print(model.summary())