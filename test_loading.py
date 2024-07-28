from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
out = generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5, temperature=0.0001, top_k = 10)

for x in out:
    print(x['generated_text'])