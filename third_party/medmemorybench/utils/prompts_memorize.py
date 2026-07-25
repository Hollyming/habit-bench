"""Memory build phase prompt templates."""

from typing import Dict

MEMORIZE_TEMPLATES: Dict[str, str] = {

    # MedMemoryBench
    "medmemorybench_long_context_memorize": """以下是一段医疗对话记录，请仔细阅读并记忆其中的关键信息：

{context}""",

    "medmemorybench_rag_memorize": """以下是一段医疗对话记录，请仔细阅读并记忆其中的关键信息：

{context}""",

    "medmemorybench_agentic_memorize": """以下是一段医疗对话记录，请仔细阅读并记忆其中的关键信息：

{context}""",

    # LoCoMo
    "locomo_long_context_memorize": """Dialogue between User and Assistant {timestamp}
<User> The following context is the conversation record. Pay attention to specific DATES and TIMES mentioned - convert any relative time references (like "yesterday", "last week") to absolute dates based on the conversation timestamp.
{context}
<Assistant> I have memorized the dialogue including all dates and time references. I will provide concise, direct answers.""",

    "locomo_rag_memorize": """Dialogue between User and Assistant {timestamp}
<User> The following context is the conversation record. Pay attention to specific DATES and TIMES mentioned - convert any relative time references (like "yesterday", "last week") to absolute dates based on the conversation timestamp.
{context}
<Assistant> I have memorized the dialogue including all dates and time references. I will provide concise, direct answers.""",

    "locomo_agentic_memorize": """Dialogue between User and Assistant {timestamp}
<User> The following context is the conversation record. Pay attention to specific DATES and TIMES mentioned - convert any relative time references (like "yesterday", "last week") to absolute dates based on the conversation timestamp.
{context}
<Assistant> I have memorized the dialogue including all dates and time references. I will provide concise, direct answers.""",

    # MedMemoryBench - English
    "medmemorybench_en_long_context_memorize": """The following is a medical dialogue record. Please read it carefully and memorize the key information:

{context}""",

    "medmemorybench_en_rag_memorize": """The following is a medical dialogue record. Please read it carefully and memorize the key information:

{context}""",

    "medmemorybench_en_agentic_memorize": """The following is a medical dialogue record. Please read it carefully and memorize the key information:

{context}""",

}
