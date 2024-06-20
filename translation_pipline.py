from transformers import pipeline

pipe_en_es = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
pipe_es_en = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")
# pipe_es_ca = pipeline("translation", model="Helsinki-NLP/opus-mt-es-ca")
# pipe_ca_es = pipeline("translation", model="Helsinki-NLP/opus-mt-ca-es")

def apply_translation_pipeline(input):
  res1 = pipe_es_en(input)
  res2 = pipe_en_es([sentence['translation_text'] for sentence in res1])
  # res3 = pipe_es_ca([sentence['translation_text'] for sentence in res2])
  # res4 = pipe_ca_es ([sentence['translation_text'] for sentence in res3])
  return [sentence['translation_text'] for sentence in res2]

