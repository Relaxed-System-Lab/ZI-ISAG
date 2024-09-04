inference_evaluation_tmplate = [
    {
        "role": "user",
        "content": "Reference Materials: {reference}. Please follow these instructions carefully: 1. The references are sourced from the Internet, so critically evaluate and use only those relevant to the Query. Ignore irrelevant content, but ensure no relevant information is omitted. Provide a comprehensive response, and make your answer as rich in detail as possible. 2. Answer the Query with thorough reasoning, adhering to facts and logic. 3. Unless specified otherwise, respond in Markdown formatting. 4. Unless specified otherwise, respond in the same language as the Query. Now, using the Reference Materials above (if any), answer the following Query: {query}"
    }
]