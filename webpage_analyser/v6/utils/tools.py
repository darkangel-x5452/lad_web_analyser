# ─────────────────────────────────────────────────────────────────────────────
# Step 3a – Groq backend  (Llama Vision, free cloud API)
# ─────────────────────────────────────────────────────────────────────────────


def query_groq(
    image_path: str,
    query: str,
    api_key: str,
    model: str = GROQ_DEFAULT_MODEL,
) -> str:
    """
    Send the screenshot to Groq's hosted Llama Vision endpoint.

    Free-tier Groq models with vision support (pick via --groq-model):
        meta-llama/llama-4-scout-17b-16e-instruct  (latest, default)
        llama-3.2-11b-vision-preview               (lighter, fastest)
        llama-3.2-90b-vision-preview               (most accurate)
    """
    Groq = _import_groq()
    client = Groq(api_key=api_key)

    print(f"{datetime.now()}, [2/3] Querying Groq ({model})...")
    b64 = _to_base64(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Query: {query}\n\n"
                            "Extract the relevant information from this image."
                        ),
                    },
                ],
            },
        ],
        temperature=0.1,  # Low = factual / exact output
        max_tokens=4096,  # Increase if you have a lot of content to extract (Groq supports up to 32k)
        # max_tokens=8192,  # Increase if you have a lot of content to extract (Groq supports up to 32k)
    )
    return response.choices[0].message.content


if backend == "groq":
    markdown = query_groq(ready_img, query, api_key, model=groq_model)