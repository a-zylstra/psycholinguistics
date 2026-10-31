import pandas as pd
import secrets
from ollama import chat

def generate_completions(fragments, model="qwen3:8b", n_runs=20):
    results = []

    for item_number, fragment in enumerate(fragments, start=1):
        for run in range(1, n_runs + 1):
            prompt = f"""
            Continue this sentence naturally. Make sure the result is a complete, grammatical sentence.
            Do not explain your answer, output only the continuation.
            Sentence: {fragment}
            """

            response = chat(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                think=False,
                options={
                    "temperature": 1.0,
                    "num_predict": 60,
                    "seed": secrets.randbelow(2**31 - 1) + 1 #put due to the number of repeat sentences
                }
            )

            completion = response.message.content.strip()

            results.append({
                "run_number": run,
                "item_number": item_number,
                "fragment": fragment,
                "completion": completion,
                "full_sentence": f"{fragment} {completion}"
            })

    return pd.DataFrame(results)