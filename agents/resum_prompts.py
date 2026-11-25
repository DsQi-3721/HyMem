"""ReSum prompts — vendored verbatim from Alibaba-NLP/DeepResearch.

Source: external/DeepResearch/WebAgent/WebResummer/src/prompt.py:97-169
Paper: https://arxiv.org/abs/2509.13313 (Wu et al. 2025)
License: vendored under upstream's repo license — see external/DeepResearch/

DO NOT modify these prompts — keeping them verbatim is what makes our
re-implementation a faithful "training-free ReSum" baseline. Any change here
breaks the comparability claim in the paper rebuttal.
"""

# First-pass summarizer (no prior summary exists).
# Vendored from external/DeepResearch/WebAgent/WebResummer/src/prompt.py:97
QUERY_SUMMARY_PROMPT = """You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history and current question to provide a comprehensive summary that will help answer the question.

Task Guidelines
1. Information Analysis:
   - Carefully analyze the conversation history to identify truly useful information.
   - Focus on information that directly contributes to answering the question.
   - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated in the conversation.
   - If information is missing or unclear, do NOT include it in your summary.

2. Summary Requirements:
   - Extract only the most relevant information that is explicitly present in the conversation.
   - Synthesize information from multiple exchanges when relevant.
   - Only include information that is certain and clearly stated in the conversation.
   - Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed from the conversation.

3. Output Format: Your response should be structured as follows:
<summary>
- Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question.]
</summary>

Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.

Question
{question}

Conversation History
{recent_history_messages}

Please generate a comprehensive and useful summary. Note that you are not permitted to invoke tools during this process.
"""


# Second-pass summarizer (a previous summary exists, keep continuity).
# Vendored from external/DeepResearch/WebAgent/WebResummer/src/prompt.py:129
QUERY_SUMMARY_PROMPT_LAST = """You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history and current question to provide a comprehensive summary that will help answer the question.

The last summary serves as your starting point, marking the information landscape previously collected. Your role is to:
- Analyze progress made since the last summary
- Identify remaining information gaps
- Generate a useful summary that combines previous and new information
- Maintain continuity, especially when recent conversation history is limited

Task Guidelines

1. Information Analysis:
   - Carefully analyze the conversation history to identify truly useful information.
   - Focus on information that directly contributes to answering the question.
   - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated.
   - If information is missing or unclear, do NOT include it in your summary.
   - Use the last summary as a baseline when recent history is sparse.

2. Summary Requirements:
   - Extract only the most relevant information that is explicitly present in the conversation.
   - Synthesize information from multiple exchanges when relevant.
   - Only include information that is certain and clearly stated.
   - Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed.

3. Output Format: Your response should be structured as follows:
<summary>
- Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question.]
</summary>

Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.

Question
{question}

Last Summary
{last_summary}

Conversation History
{recent_history_messages}

Please generate a comprehensive and useful summary. Note that you are not permitted to invoke tools during this process.
"""


# Restart message format after summarization fires.
# Adapted from external/DeepResearch/WebAgent/WebResummer/src/react_agent.py:148-150
RESUM_RESTART_MESSAGE = """Question: {question}
Below is a summary of the previous conversation. This summary condenses key information from earlier steps, so please consider it carefully. Assess whether the summary provides enough information to answer the question and use it as the basis for further reasoning and information gathering to answer the question.
Summary: {summary}
"""


import re


def extract_summary(response: str) -> str:
    """Pull <summary>...</summary> from summarizer output. Falls back to the
    full response if the tag isn't found (model didn't follow format)."""
    m = re.search(r"<summary>(.*?)</summary>", response, re.DOTALL)
    return m.group(1).strip() if m else response.strip()
