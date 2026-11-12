import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")

#one long sentence

sentence =(
     "The document retrieval system converts long documents into "
    "smaller chunks so that relevant information can be searched "
    "efficiently by users."
)

#Use a deliberately small chunk size to demonstrate splitting
tokens = tokenizer.encode(sentence)

chunk_size =8
chunks = []

for start in range(0, len(tokens), chunk_size):
    chunk_tokens =tokens[start:start + chunk_size]
    chunks.append(tokenizer.decode(chunk_tokens))

print("ORIGINAL SENTENCE:", sentence)


print("\nSPLIT INTO CHUNKS:")
for index, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {index}:")
    print(chunk)

print("\nTotal chunks:", len(chunks))
print("Sentence was split:", len(chunks) > 1)
