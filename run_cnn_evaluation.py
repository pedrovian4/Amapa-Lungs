from amapa_ar.evaluation.evaluate_cnn import ModelEvaluator


evaluator = ModelEvaluator()
model_names = ["EfficientNetB0", "ResNet50", "InceptionV3", "VGG16"]
evaluator.run(model_names)