# generation_module.py

import torch
from transformers import pipeline


# === Initialize Text Generation Model ===
def load_generator(model_name="Qwen/Qwen2-0.5B-Instruct"):
    """
    Load the Qwen model for answer generation.
    Automatically uses GPU if available.
    """
    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading generation model '{model_name}' on {'GPU' if device == 0 else 'CPU'}...")
    return pipeline("text-generation", model=model_name, device=device)


# === Answer Generation ===
def generate_answer(query, retrieved_docs, max_new_tokens=100):
    """
    Generate a clear, concise answer using retrieved context and a given query.

    Args:
        query (str): The user question.
        retrieved_docs (list[str]): A list of retrieved document texts.
        max_new_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated answer text.
    """
    if not isinstance(retrieved_docs, list):
        retrieved_docs = [retrieved_docs]

    # Combine all retrieved text segments into a single context
    context_text = " ".join(retrieved_docs).replace("\n", " ").strip()

    # Build the generation prompt
    prompt = (
        f"Answer the following question concisely and accurately using the provided context.\n\n"
        f"Context: {context_text}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    print("\n--- Full Prompt Constructed for Generation ---\n")
    print(prompt)

    # Load the model (only once)
    global generator_pipeline
    if "generator_pipeline" not in globals():
        generator_pipeline = load_generator()

    # Generate response
    try:
        response = generator_pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            num_return_sequences=1,
            truncation=True,
            pad_token_id=generator_pipeline.tokenizer.eos_token_id,
            temperature=0.7,
        )
        output_text = response[0]["generated_text"].replace(prompt, "").strip()
        return output_text or "No clear answer generated."
    except Exception as e:
        return f"[Error during generation: {e}]"


# === Quick Test ===
if __name__ == "__main__":
    example_query = "What is the capital of France?"
    example_docs = [
        "The capital city of France is Paris, known for its rich culture and history.",
        "Paris is home to the Eiffel Tower, built in 1889."
    ]
    result = generate_answer(example_query, example_docs, max_new_tokens=80)
    print("\nQuestion:", example_query)
    print("Answer:", result)
