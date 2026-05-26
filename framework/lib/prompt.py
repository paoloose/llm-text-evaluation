"""Prompt engineering for LLM verbal reasoning evaluation.

Handles:
- Building system and user messages per task type
- Constructing JSON schema for structured output (single and batch)
- Parsing model responses with fallback strategies
"""

from __future__ import annotations

import json
import re

from .types import CrossLingualLanguage, Sample

# -- System promp

SYSTEM_PROMPT = """\
You are an expert evaluator for verbal reasoning tasks in Spanish.
You will receive questions with numbered answer options (0-based index).
Analyze each question carefully and select the BEST answer.

TASK-SPECIFIC GUIDELINES:
- Reading comprehension: Focus on what the text explicitly states or strongly implies.
- Sentence ordering: Find the logical sequence that creates a coherent, well-structured text.
- Sentence elimination: Identify the sentence that does NOT belong thematically or logically.
- Verbal series: Identify the pattern connecting the words (synonyms, antonyms, categories, relationships).
- Analogies: Match the underlying relationship between the given pair of concepts.
- Synonyms and antonyms: Select the word with the closest or most opposite meaning in context.
- Incomplete sentences: Choose the option that best completes the sentence's meaning and grammar.

RULES:
- Consider the context, question, and ALL options before deciding.
- Your response must be valid JSON matching the required schema.
- Provide ONLY the answer index, no explanations."""


# -- JSON schemas

SINGLE_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "single_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "integer",
                    "description": "0-based index of the correct option",
                },
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}

BATCH_ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "batch_answers",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Sample ID from the question",
                            },
                            "answer": {
                                "type": "integer",
                                "description": "0-based index of the correct option",
                            },
                        },
                        "required": ["id", "answer"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["answers"],
            "additionalProperties": False,
        },
    },
}


# -- Prompt building


def _format_options(options: tuple[str, ...]) -> str:
    """Format options as numbered list."""
    return "\n".join(f"{i}) {opt}" for i, opt in enumerate(options))


def build_single_prompt(sample: Sample) -> str:
    """Build user message for a single sample.

    Args:
        sample: The sample to format as a prompt.

    Returns:
        Formatted user message string.
    """
    return (
        f"<question id={sample.id} type={sample.task.value}>\n"
        f"<content>"
        f"{sample.question}"
        f"</content>"
        f"<options n={len(sample.options)}>\n"
        f"{_format_options(sample.options)}"
        f"</options>\n"
        f"</question>\n"
    )


def build_batch_prompt(samples: list[Sample]) -> str:
    """Build user message for a batch of samples.

    Args:
        samples: List of samples to format as a single prompt.

    Returns:
        Formatted user message string with all questions.
    """
    parts: list[str] = ["<instructions>Answer each of the following questions</instructions>\n"]

    for i, sample in enumerate(samples, 1):
        parts.append(build_single_prompt(sample))

    return "\n".join(parts)


def build_messages(
    samples: list[Sample],
) -> tuple[list[dict[str, str]], dict]:
    """Build the full message list and response format for a batch of samples.

    Args:
        samples: One or more samples to include in this request.

    Returns:
        Tuple of (messages, response_format) ready for the provider.
    """
    if len(samples) == 1:
        user_msg = build_single_prompt(samples[0])
        response_format = SINGLE_ANSWER_SCHEMA
    else:
        user_msg = build_batch_prompt(samples)
        response_format = BATCH_ANSWER_SCHEMA

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    return messages, response_format


# -- Response parsin


def parse_single_response(raw: str) -> int | None:
    """Parse a single-answer JSON response.

    Tries JSON parsing first, then falls back to regex extraction.

    Args:
        raw: Raw model response text.

    Returns:
        The predicted answer index, or None if parsing fails.
    """
    # Try JSON parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict) and "answer" in data:
            answer = data["answer"]
            if isinstance(answer, int):
                return answer
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: extract first integer from response
    match = re.search(r"\b(\d+)\b", raw)
    if match:
        return int(match.group(1))

    return None


def parse_batch_response(
    raw: str,
    expected_ids: list[int],
) -> dict[int, int | None]:
    """Parse a batch-answer JSON response.

    Returns a mapping from sample ID to predicted answer index.

    Args:
        raw: Raw model response text.
        expected_ids: List of sample IDs we expect in the response.

    Returns:
        Dict mapping sample_id → predicted_answer (None if missing/failed).
    """
    results: dict[int, int | None] = {sid: None for sid in expected_ids}

    # Try JSON parse
    try:
        data = json.loads(raw.strip())
        if isinstance(data, dict) and "answers" in data:
            for item in data["answers"]:
                if isinstance(item, dict) and "id" in item and "answer" in item:
                    sid = item["id"]
                    answer = item["answer"]
                    if sid in results and isinstance(answer, int):
                        results[sid] = answer
    except (json.JSONDecodeError, TypeError):
        pass

    return results


# -- Cross-lingual translation -------------------------------------------------

TRANSLATION_SYSTEM_PROMPT = """\
Translate the user's question & options into {language_name}.\nTags:\n\
- The "question" tag contains the full question. Do not stop until you find the closing tag: </question>. It has a length property, and your translated question should roughly have the same length.\n\
- The "options" tag contains a list of human readable possible answers to the question. Translate them until closing tag: </options>.\n\
\n\
You MUST translate BOTH the question AND every option. Do NOT leave any option in the original language.\n\
\n\
OUTPUT FORMAT — return ONLY a JSON object with a single key:\n\
  "translations": an array of objects, each with:\n\
    "id":       the sample id (integer)\n\
    "question": translated question text (string)\n\
    "options":  translated answer options in original order (array of strings)\n\
\n\
Do NOT wrap the JSON in markdown code blocks. No extra text. Translate every word."""

_TRANSLATION_EXAMPLES: dict[CrossLingualLanguage, str] = {
    CrossLingualLanguage.FRENCH: """\
EXEMPLE 1 — Entrée :
<question length=40>
Indique el sinónimo de "valiente".
</question>
<options n=4>
0) cobarde
1) audaz
2) tímido
3) perezoso
</options>

EXEMPLE 1 — Sortie :
{"translations": [{"id": 0, "question": "Indiquez le synonyme de \\"courageux\\".", "options": ["lâche", "audacieux", "timide", "paresseux"]}]}

EXEMPLE 2 — Entrée (options en chiffres romains — ne PAS traduire les options) :
<question length=120>
Ordene las oraciones. I) La Revolución Francesa cambió Europa. II) Los filósofos alemanes quedaron fascinados. III) Hegel reflexionó sobre este evento.
</question>
<options n=3>
0) I - II - III
1) III - II - I
2) II - I - III
</options>

EXEMPLE 2 — Sortie :
{"translations": [{"id": 1, "question": "Ordonnez les phrases. I) La Révolution française a changé l'Europe. II) Les philosophes allemands ont été fascinés. III) Hegel a réfléchi à cet événement.", "options": ["I - II - III", "III - II - I", "II - I - III"]}]}""",
    CrossLingualLanguage.CHINESE: """\
示例1 — 输入：
<question length=40>
Indique el sinónimo de "valiente".
</question>
<options n=4>
0) cobarde
1) audaz
2) tímido
3) perezoso
</options>

示例1 — 输出：
{"translations": [{"id": 0, "question": "指出\\"勇敢\\"的近义词。", "options": ["胆小的", "大胆的", "害羞的", "懒惰的"]}]}

示例2 — 输入（选项为罗马数字——不要翻译选项）：
<question length=120>
Ordene las oraciones. I) La Revolución Francesa cambió Europa. II) Los filósofos alemanes quedaron fascinados. III) Hegel reflexionó sobre este evento.
</question>
<options n=3>
0) I - II - III
1) III - II - I
2) II - I - III
</options>

示例2 — 输出：
{"translations": [{"id": 1, "question": "排列句子顺序。I) 法国大革命改变了欧洲。II) 德国哲学家为之着迷。III) 黑格尔对这一事件进行了反思。", "options": ["I - II - III", "III - II - I", "II - I - III"]}]}""",
    CrossLingualLanguage.ARABIC: """\
مثال 1 — الإدخال:
<question length=40>
Indique el sinónimo de "valiente".
</question>
<options n=4>
0) cobarde
1) audaz
2) tímido
3) perezoso
</options>

مثال 1 — الإخراج:
{"translations": [{"id": 0, "question": "حدد مرادف \\"شجاع\\".", "options": ["جبان", "جريء", "خجول", "كسول"]}]}

مثال 2 — الإدخال (خيارات بأرقام رومانية — لا تترجم الخيارات):
<question length=120>
Ordene las oraciones. I) La Revolución Francesa cambió Europa. II) Los filósofos alemanes quedaron fascinados. III) Hegel reflexionó sobre este evento.
</question>
<options n=3>
0) I - II - III
1) III - II - I
2) II - I - III
</options>

مثال 2 — الإخراج:
{"translations": [{"id": 1, "question": "رتّب الجمل. I) غيرت الثورة الفرنسية أوروبا. II) انبهر الفلاسفة الألمان. III) تأمل هيغل في هذا الحدث.", "options": ["I - II - III", "III - II - I", "II - I - III"]}]}""",
    CrossLingualLanguage.JAPANESE: """\
例1 — 入力：
<question length=40>
Indique el sinónimo de "valiente".
</question>
<options n=4>
0) cobarde
1) audaz
2) tímido
3) perezoso
</options>

例1 — 出力：
{"translations": [{"id": 0, "question": "\\"勇敢\\"の同義語を示しなさい。", "options": ["臆病な", "大胆な", "内気な", "怠惰な"]}]}

例2 — 入力（ローマ数字のオプション — オプションは翻訳しないでください）：
<question length=120>
Ordene las oraciones. I) La Revolución Francesa cambió Europa. II) Los filósofos alemanes quedaron fascinados. III) Hegel reflexionó sobre este evento.
</question>
<options n=3>
0) I - II - III
1) III - II - I
2) II - I - III
</options>

例2 — 出力：
{"translations": [{"id": 1, "question": "文を並べ替えなさい。I) フランス革命はヨーロッパを変えた。II) ドイツの哲学者たちは魅了された。III) ヘーゲルはこの出来事について考察した。", "options": ["I - II - III", "III - II - I", "II - I - III"]}]}""",
    CrossLingualLanguage.SWAHILI: """\
Mfano 1 — Pembejeo:
<question length=40>
Indique el sinónimo de "valiente".
</question>
<options n=4>
0) cobarde
1) audaz
2) tímido
3) perezoso
</options>

Mfano 1 — Matokeo:
{"translations": [{"id": 0, "question": "Onyesha kihusishi cha \\"jasiri\\".", "options": ["mwoga", "shupavu", "aibu", "vivu"]}]}

Mfano 2 — Pembejeo (chaguzi za namba za Kiroma — USITAFSIRI chaguzi):
<question length=120>
Ordene las oraciones. I) La Revolución Francesa cambió Europa. II) Los filósofos alemanes quedaron fascinados. III) Hegel reflexionó sobre este evento.
</question>
<options n=3>
0) I - II - III
1) III - II - I
2) II - I - III
</options>

Mfano 2 — Matokeo:
{"translations": [{"id": 1, "question": "Panga sentensi. I) Mapinduzi ya Ufaransa yalibadilisha Ulaya. II) Wanafalsafa wa Kijerumani walishangaa. III) Hegel alifikiria kuhusu tukio hili.", "options": ["I - II - III", "III - II - I", "II - I - III"]}]}""",
    CrossLingualLanguage.RUSSIAN: """\
Пример 1 — Ввод:
<question length=40>
Indique el sinónimo de "valiente".
</question>
<options n=4>
0) cobarde
1) audaz
2) tímido
3) perezoso
</options>

Пример 1 — Вывод:
{"translations": [{"id": 0, "question": "Укажите синоним слова \\"храбрый\\".", "options": ["трусливый", "смелый", "застенчивый", "ленивый"]}]}

Пример 2 — Ввод (варианты с римскими цифрами — НЕ переводите варианты):
<question length=120>
Ordene las oraciones. I) La Revolución Francesa cambió Europa. II) Los filósofos alemanes quedaron fascinados. III) Hegel reflexionó sobre este evento.
</question>
<options n=3>
0) I - II - III
1) III - II - I
2) II - I - III
</options>

Пример 2 — Вывод:
{"translations": [{"id": 1, "question": "Расставьте предложения по порядку. I) Французская революция изменила Европу. II) Немецкие философы были очарованы. III) Гегель размышлял об этом событии.", "options": ["I - II - III", "III - II - I", "II - I - III"]}]}""",
}

_TRANSLATION_SYSTEM_PROMPTS: dict[CrossLingualLanguage, str] = {
    CrossLingualLanguage.ENGLISH: """\
Translate the user's question and options into English.\nTags:\n\
- The "question" tag contains the full question. Do not stop until you find the closing tag: </question>. It has a length property, and your translated question should roughly have the same length.\n\
- The "options" tag contains a list of human readable possible answers to the question. Translate them until closing tag: </options>.\n\
\n\
You MUST translate BOTH the question AND every option. Do NOT leave any option in the original language.\n\
\n\
OUTPUT FORMAT — return ONLY a JSON object with a single key:\n\
  "translations": an array of objects, each with:\n\
    "id":       the sample id (integer)\n\
    "question": translated question text (string)\n\
    "options":  translated answer options in original order (array of strings)\n\
\n\
Do NOT wrap the JSON in markdown code blocks. No extra text. Translate every word.""",
    CrossLingualLanguage.FRENCH: """\
Traduisez la question et les options de l'utilisateur en français.\nBalises :\n\
- La balise « question » contient la question complète. Ne vous arrêtez pas avant d'avoir trouvé la balise de fermeture : </question>. Elle possède une propriété de longueur, et votre question traduite devrait avoir approximativement la même longueur.\n\
- La balise « options » contient une liste de réponses possibles lisibles. Traduisez-les jusqu'à la balise de fermeture : </options>.\n\
\n\
Vous DEVEZ traduire ET la question ET chaque option. Ne laissez AUCUNE option dans la langue d'origine.\n\
\n\
FORMAT DE SORTIE — retournez UNIQUEMENT un objet JSON avec une seule clé :\n\
  "translations" : un tableau d'objets, chacun contenant :\n\
    "id" :       l'identifiant de l'échantillon (entier)\n\
    "question" : texte de la question traduite (chaîne)\n\
    "options" :  options de réponse traduites dans l'ordre d'origine (tableau de chaînes)\n\
\n\
Ne PAS envelopper le JSON dans des blocs de code markdown. Pas de texte supplémentaire. Traduisez chaque mot.""",
    CrossLingualLanguage.CHINESE: """\
将用户的问题和选项翻译成中文。\n标签：\n\
- "question"标签包含完整的问题。在找到闭合标签</question>之前不要停止。它有一个length属性，翻译后的问题长度应大致相同。\n\
- "options"标签包含一组可读的可能答案。翻译它们直到闭合标签</options>。\n\
\n\
您必须翻译问题和每个选项。不要将任何选项保留在原始语言中。\n\
\n\
输出格式 — 仅返回一个包含单个键的JSON对象：\n\
  "translations"：一个对象数组，每个对象包含：\n\
    "id"：      样本ID（整数）\n\
    "question"：翻译后的问题文本（字符串）\n\
    "options"： 按原始顺序翻译的答案选项（字符串数组）\n\
\n\
不要将JSON包裹在markdown代码块中。不要添加额外文本。翻译每个词。""",
    CrossLingualLanguage.ARABIC: """\
ترجم سؤال المستخدم وخياراته إلى العربية.\nالعلامات:\n\
- تحتوي علامة "question" على السؤال الكامل. لا تتوقف حتى تجد علامة الإغلاق: </question>. يحتوي على خاصية الطول، ويجب أن يكون طول سؤالك المترجم تقريبًا نفس الطول.\n\
- تحتوي علامة "options" على قائمة بالإجابات الممكنة المقروءة. ترجمها حتى علامة الإغلاق: </options>.\n\
\n\
يجب عليك ترجمة السؤال وكل خيار. لا تترك أي خيار باللغة الأصلية.\n\
\n\
تنسيق الإخراج — أعد فقط كائن JSON بمفتاح واحد:\n\
  "translations": مصفوفة من الكائنات، كل منها يحتوي على:\n\
    "id":       معرف العينة (عدد صحيح)\n\
    "question": نص السؤال المترجم (سلسلة)\n\
    "options":  خيارات الإجابة المترجمة بالترتيب الأصلي (مصفوفة سلاسل)\n\
\n\
لا تقم بلف JSON في كتل كود markdown. لا نص إضافي. ترجم كل كلمة.""",
    CrossLingualLanguage.JAPANESE: """\
ユーザーの質問とオプションを日本語に翻訳してください。\nタグ：\n\
- "question"タグには完全な質問が含まれています。終了タグ</question>が見つかるまで停止しないでください。lengthプロパティがあり、翻訳された質問の長さはほぼ同じである必要があります。\n\
- "options"タグには、読み取り可能な回答のリストが含まれています。終了タグ</options>まで翻訳してください。\n\
\n\
質問と各オプションの両方を翻訳する必要があります。元の言語のままにしないでください。\n\
\n\
出力形式 — 単一のキーを持つJSONオブジェクトのみを返してください：\n\
  "translations"：オブジェクトの配列、各オブジェクトには：\n\
    "id"：       サンプルID（整数）\n\
    "question"： 翻訳された質問テキスト（文字列）\n\
    "options"：  元の順序で翻訳された回答オプション（文字列の配列）\n\
\n\
JSONをマークダウンコードブロックで囲まないでください。余分なテキストは不要。すべての単語を翻訳してください。""",
    CrossLingualLanguage.SWAHILI: """\
Tafsiri swali na chaguzi za mtumiaji kwa Kiswahili.\nLebo:\n\
- Lebo ya "question" ina swali kamili. Usisimame hadi upate lebo ya kufunga: </question>. Ina sifa ya urefu, na swali lako lililotafsiriwa linafaa kuwa na urefu takriban sawa.\n\
- Lebo ya "options" ina orodha ya majibu yanayowezekana yanayosomeka. Tafsiri hadi lebo ya kufunga: </options>.\n\
\n\
Lazima utafsiri swali na kila chaguo. Usiache chaguo lolote kwa lugha ya asili.\n\
\n\
Muundo wa Matokeo — rudisha TU kitu cha JSON chenye ufunguo mmoja:\n\
  "translations": safu ya vitu, kila kimoja kikiwa na:\n\
    "id":       kitambulisho cha sampuli (nambari kamili)\n\
    "question": maandishi ya swali lililotafsiriwa (mfuatano)\n\
    "options":  chaguzi za majibu zilizotafsiriwa kwa mpangilio wa awali (safu ya mifuatano)\n\
\n\
Usizunguke JSON katika vitalu vya msimbo wa markdown. Hakuna maandishi ya ziada. Tafsiri kila neno.""",
    CrossLingualLanguage.RUSSIAN: """\
Переведите вопрос и параметры пользователя на русский.\nТеги:\n\
- Тег "question" содержит полный вопрос. Не останавливайтесь, пока не найдёте закрывающий тег: </question>. Он имеет свойство длины, и ваш переведённый вопрос должен быть примерно такой же длины.\n\
- Тег "options" содержит список читаемых возможных ответов. Переведите их до закрывающего тега: </options>.\n\
\n\
Вы ОБЯЗАНЫ перевести и вопрос, и каждый вариант ответа. Не оставляйте ни один вариант на исходном языке.\n\
\n\
ФОРМАТ ВЫВОДА — верните ТОЛЬКО объект JSON с одним ключом:\n\
  "translations": массив объектов, каждый из которых содержит:\n\
    "id":       идентификатор примера (целое число)\n\
    "question": текст переведённого вопроса (строка)\n\
    "options":  переведённые варианты ответов в исходном порядке (массив строк)\n\
\n\
НЕ оборачивайте JSON в блоки кода markdown. Без лишнего текста. Переведите каждое слово.""",
}

TRANSLATION_SINGLE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation_single",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Translated question text",
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Translated answer options in original order",
                },
            },
            "required": ["question", "options"],
            "additionalProperties": False,
        },
    },
}

TRANSLATION_BATCH_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "translation_batch",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "translations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "integer",
                                "description": "Sample ID from the question",
                            },
                            "question": {
                                "type": "string",
                                "description": "Translated question text",
                            },
                            "options": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Translated answer options in original order",
                            },
                        },
                        "required": ["id", "question", "options"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["translations"],
            "additionalProperties": False,
        },
    },
}


def build_translation_messages(
    samples: list[Sample],
    language: CrossLingualLanguage,
) -> tuple[list[dict[str, str]], dict]:
    if len(samples) == 1:
        response_format = TRANSLATION_SINGLE_SCHEMA
        user_msg = _format_translation_user(samples)
    else:
        response_format = TRANSLATION_BATCH_SCHEMA
        user_msg = _format_translation_batch_user(samples)
    system_msg = _TRANSLATION_SYSTEM_PROMPTS.get(
        language,
        TRANSLATION_SYSTEM_PROMPT.format(language_name=language.value.capitalize()),
    )
    examples = _TRANSLATION_EXAMPLES.get(language, "")
    if examples:
        system_msg = system_msg + "\n\n" + examples
    return (
        [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
        response_format,
    )

def _format_translation_user(samples: list[Sample]) -> str:
    q = samples[0].question
    return (
        f"<question length={len(q)}>\n"
        f"{q}\n"
        f"</question>\n"
        f"<options n={len(samples[0].options)}>\n"
        f"{_format_options(samples[0].options)}\n"
        f"</options>"
    )


def _format_translation_batch_user(samples: list[Sample]) -> str:
    parts: list[str] = []
    for i, s in enumerate(samples, 1):
        parts.append(
            f"<sample id={s.id}>\n"
            f"{_format_translation_user([s])}\n"
            f"</sample>"
        )
    return "\n".join(parts)


def _repair_translation_json(raw: str) -> str:
    """Repair LLM-generated JSON that contains unescaped typographic double
    quotes inside string values (e.g. French ``d"interprétation``).

    Heuristic: a ``"`` preceded and followed by an alphanumeric character
    is treated as an inner quote and escaped with a backslash.
    """
    chars = list(raw)
    escape_next = False
    i = 0
    while i < len(chars):
        c = chars[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if c == "\\":
            escape_next = True
            i += 1
            continue
        if c == '"':
            prev_char = chars[i - 1] if i > 0 else ""
            next_char = chars[i + 1] if i + 1 < len(chars) else ""
            if prev_char.isalnum() and next_char.isalnum():
                chars.insert(i, "\\")
                i += 2
                continue
        i += 1
    return "".join(chars)


def parse_translation_response(
    raw: str,
    expected_ids: list[int],
) -> dict[int, tuple[str, tuple[str, ...] | None]]:
    """Return ``{sample_id: (translated_question, translated_options_or_None)}``."""
    results: dict[int, tuple[str, tuple[str, ...] | None]] = {}
    expected_set = set(expected_ids)

    raw_fixed = (
        raw.replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        data = json.loads(raw_fixed)
    except (json.JSONDecodeError, TypeError):
        repaired = _repair_translation_json(raw_fixed)
        try:
            data = json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            return results

    if isinstance(data, dict):
        _parse_dict_response(data, expected_ids, expected_set, results)
    elif isinstance(data, list):
        _parse_list_response(data, expected_ids, expected_set, results)

    return results


def _parse_dict_response(
    data: dict,
    expected_ids: list[int],
    expected_set: set[int],
    results: dict[int, tuple[str, tuple[str, ...] | None]],
) -> None:
    if "translations" in data:
        for item in data["translations"]:
            _ingest(item, expected_set, results)
    elif "id" in data and "question" in data:
        _ingest(data, expected_set, results)
    elif "question" in data and len(expected_ids) == 1:
        _ingest(
            {"id": expected_ids[0], "question": data["question"], **({"options": data["options"]} if "options" in data else {})},
            expected_set,
            results,
        )


def _parse_list_response(
    data: list,
    expected_ids: list[int],
    expected_set: set[int],
    results: dict[int, tuple[str, tuple[str, ...] | None]],
) -> None:
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        question = item.get("question")
        if not isinstance(question, str):
            continue
        options = item.get("options")
        opts_tuple = tuple(options) if isinstance(options, list) and all(isinstance(o, str) for o in options) else None
        sid = item.get("id")
        entry = (question, opts_tuple)
        if isinstance(sid, int) and sid in expected_set:
            results[sid] = entry
        elif idx < len(expected_ids):
            sid = expected_ids[idx]
            if sid not in results:
                results[sid] = entry


def _ingest(
    item: object,
    expected_ids: set[int],
    results: dict[int, tuple[str, tuple[str, ...] | None]],
) -> None:
    if not isinstance(item, dict):
        return
    sid = item.get("id")
    question = item.get("question")
    if isinstance(sid, int) and sid in expected_ids and isinstance(question, str):
        options = item.get("options")
        opts_tuple = tuple(options) if isinstance(options, list) and all(isinstance(o, str) for o in options) else None
        results[sid] = (question, opts_tuple)
