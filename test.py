from transformers import pipeline

corrector = pipeline("text2text-generation", model="bmd1905/vietnamese-correction-v2")

texts = "Ph廕《 Thanh T璽n"

predictions = corrector(texts)
print(predictions)