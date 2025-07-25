def call_llm(llm, system_prompt, prompt):
    """Makes the LLM call"""
    ans = llm(
        prompt,
        chat_history=[],
        system_prompt=system_prompt,
    )
    for x in ans:
        pass
    return x
