# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass, field

from tasks.default_prompts import LETTER_INDICES
from utils import Language


# TODO(hynky1999): The typing still is not great, it should be able to infer that you can't access the
# attributes that are not defined in the class. Don't want to waste time on this though.
@dataclass
class TranslationLiterals:
    # This is just to create nice error messages
    language: Language

    question_word: str = None  # type: ignore
    answer: str = None  # type: ignore
    confirmation_word: str = None  # type: ignore
    yes: str = None  # type: ignore
    no: str = None  # type: ignore
    also: str = None  # type: ignore
    cause_word: str = None  # type: ignore
    effect_word: str = None  # type: ignore
    or_word: str = None  # type: ignore

    # NLI
    true: str = None  # type: ignore
    false: str = None  # type: ignore
    neither: str = None  # type: ignore

    # Punctuation
    full_stop: str = "."
    comma: str = ","
    question_mark: str = "?"
    exclamation_mark: str = "!"
    word_space: str = " "
    sentence_space: str = " "
    colon: str = ":"
    semicolon: str = ";"

    # Indices
    indices: list[str] = field(default_factory=lambda: LETTER_INDICES)

    def __getattribute__(self, name: str) -> str:
        value = super().__getattribute__(name)
        if value is None:
            raise AttributeError(
                f"""
Translation for '{name}' is needed for {self.language}. Please provide its implementation by editing
the 'src/lighteval/tasks/templates/utils/translation_literals.py'
"""
            )
        return value


TRANSLATION_LITERALS: dict[Language, TranslationLiterals] = {
    Language.AFRIKAANS: TranslationLiterals(language=Language.AFRIKAANS),
    Language.ALBANIAN: TranslationLiterals(language=Language.ALBANIAN),
    Language.AMHARIC: TranslationLiterals(language=Language.AMHARIC),
    Language.ARABIC: TranslationLiterals(
        language=Language.ARABIC,
        question_word="سؤال",
        answer="إجابة",
        confirmation_word="صحيح",
        yes="نعم",
        no="لا",
        also="كذلك",
        cause_word="لأن",
        effect_word="لذلك",
        true="صحيح",
        false="خاطئ",
        neither="لا هذا ولا ذاك",
        or_word="أو",
        full_stop=".",
        comma="،",
        question_mark="؟",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        indices=["أ", "ب", "ج", "د", "هـ", "و", "ز", "ح"],
    ),
    Language.ARMENIAN: TranslationLiterals(language=Language.ARMENIAN),
    Language.ASSAMESE: TranslationLiterals(language=Language.ASSAMESE),
    Language.AZERBAIJANI: TranslationLiterals(language=Language.AZERBAIJANI),
    Language.BASHKIR: TranslationLiterals(
        language=Language.BASHKIR,
        question_word="һорау",
        answer="яуап",
        confirmation_word="шулаймы",
        yes="эйе",
        no="яҡ",
        also="шулай уҡ",
        cause_word="сөнки",
        effect_word="шуға",
        or_word="йәки",
        true="дөрөҫ",
        false="ялған",
        neither="башҡа",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        indices=["А", "Б", "В", "Г", "Д", "Е"],
    ),
    Language.BASQUE: TranslationLiterals(
        language=Language.BASQUE,
        question_word="galdera",
        answer="erantzuna",
        confirmation_word="ezta",
        yes="bai",
        no="ez",
        also="halaber",
        cause_word="izan ere",
        effect_word="beraz",
        or_word="ala",
        true="egia",
        false="faltsua",
        neither="bat ere ez",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.BELARUSIAN: TranslationLiterals(
        language=Language.BELARUSIAN,
        question_word="пытанне",
        answer="адказ",
        confirmation_word="ці не так",
        yes="так",
        no="не",
        also="апроч таго",
        cause_word="бо",
        effect_word="таму",
        true="праўда",
        false="няпраўда",
        neither="ні тое, ні тое",
        or_word="ці",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        indices=["А", "Б", "В", "Г", "Д", "Е"],
    ),
    Language.BENGALI: TranslationLiterals(language=Language.BENGALI, question_word="প্রশ্ন"),
    Language.BIHARI: TranslationLiterals(language=Language.BIHARI),  # Deprecated
    Language.BOSNIAN: TranslationLiterals(language=Language.BOSNIAN),
    Language.BRETON: TranslationLiterals(language=Language.BRETON),
    Language.BULGARIAN: TranslationLiterals(language=Language.BULGARIAN),
    Language.BURMESE: TranslationLiterals(language=Language.BURMESE),
    Language.CATALAN: TranslationLiterals(language=Language.CATALAN),
    Language.CEBUANO: TranslationLiterals(language=Language.CEBUANO),
    Language.CHINESE: TranslationLiterals(
        language=Language.CHINESE,
        question_word="问题",
        answer="答案",
        confirmation_word="对吗",
        yes="是的",
        no="不是",
        also="而且",
        cause_word="因为",
        effect_word="所以",
        true="真",
        false="假",
        neither="都不是",
        or_word="或",
        full_stop="。",
        comma="，",
        question_mark="？",
        exclamation_mark="！",
        word_space="",
        sentence_space="",
        colon="：",
        indices=["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"],
    ),
    Language.CROATIAN: TranslationLiterals(
        language=Language.CROATIAN,
        question_word="pitanje",
        answer="odgovor",
        confirmation_word="zar ne",
        yes="da",
        no="ne",
        also="također",
        cause_word="jer",
        effect_word="dakle",
        or_word="ili",
        true="točno",
        false="netočno",
        neither="ništa od navedenog",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.CZECH: TranslationLiterals(
        language=Language.CZECH,
        question_word="otázka",
        answer="odpověď",
        confirmation_word="že ano",
        yes="ano",
        no="ne",
        also="navíc",
        cause_word="protože",
        effect_word="a tedy",
        or_word="nebo",
        true="pravda",
        false="nepravda",
        neither="ani jedno",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.DANISH: TranslationLiterals(language=Language.DANISH),
    Language.DIVEHI: TranslationLiterals(language=Language.DIVEHI),
    Language.DUTCH: TranslationLiterals(
        language=Language.DUTCH,
        question_word="vraag",
        answer="antwoord",
        confirmation_word="toch",
        yes="ja",
        no="nee",
        also="ook",
        cause_word="want",
        effect_word="dus",
        or_word="of",
        true="waar",
        false="onwaar",
        neither="geen van beide",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.ENGLISH: TranslationLiterals(
        language=Language.ENGLISH,
        question_word="question",
        answer="answer",
        confirmation_word="right",
        yes="yes",
        no="no",
        also="also",
        cause_word="because",
        effect_word="therefore",
        true="true",
        false="false",
        neither="neither",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        or_word="or",
    ),
    Language.ESPERANTO: TranslationLiterals(language=Language.ESPERANTO),
    Language.ESTONIAN: TranslationLiterals(
        # From https://github.com/EleutherAI/lm-evaluation-harness/blob/0845b588303f1f59af98dd1c5bdbd78a9e75a1e2/lm_eval/tasks/xcopa/utils.py
        language=Language.ESTONIAN,
        cause_word="sest",
        effect_word="seetõttu",
    ),
    Language.FINNISH: TranslationLiterals(
        language=Language.FINNISH,
        question_word="kysymys",
        answer="vastaus",
        confirmation_word="eikö niin",
        yes="kyllä",
        no="ei",
        also="myös",
        cause_word="koska",
        effect_word="siksi",
        or_word="tai",
        true="totta",
        false="tarua",
        neither="ei kumpikaan",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.FRENCH: TranslationLiterals(
        language=Language.FRENCH,
        question_word="question",
        answer="réponse",
        confirmation_word="n'est-ce pas",
        yes="oui",
        no="non",
        also="de plus",
        cause_word="parce que",
        effect_word="donc",
        or_word="ou",
        true="vrai",
        false="faux",
        neither="aucun des deux",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.GALICIAN: TranslationLiterals(language=Language.GALICIAN),
    Language.GEORGIAN: TranslationLiterals(language=Language.GEORGIAN),
    Language.GERMAN: TranslationLiterals(
        language=Language.GERMAN,
        question_word="frage",
        answer="antwort",
        confirmation_word="richtig",
        yes="ja",
        no="nein",
        also="auch",
        cause_word="weil",
        effect_word="deshalb",
        or_word="oder",
        true="wahr",
        false="falsch",
        neither="weder noch",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.GREEK: TranslationLiterals(
        language=Language.GREEK,
        question_word="ερώτηση",
        answer="απάντηση",
        confirmation_word="σωστά",
        yes="ναι",
        no="όχι",
        also="επίσης",
        cause_word="επειδή",
        effect_word="άρα",
        or_word="ή",
        true="σωστό",
        false="λάθος",
        neither="καμία απάντηση",
        full_stop=".",
        comma=",",
        question_mark=";",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon="·",
    ),
    Language.GUJARATI: TranslationLiterals(language=Language.GUJARATI),
    Language.HAITIAN: TranslationLiterals(
        # From https://github.com/EleutherAI/lm-evaluation-harness/blob/0845b588303f1f59af98dd1c5bdbd78a9e75a1e2/lm_eval/tasks/xcopa/utils.py
        language=Language.HAITIAN,
        cause_word="poukisa",
        effect_word="donk sa",
    ),
    Language.HEBREW: TranslationLiterals(language=Language.HEBREW),
    Language.HINDI: TranslationLiterals(
        language=Language.HINDI,
        question_word="सवाल",
        answer="उत्तर",
        confirmation_word="है ना",
        yes="हाँ",
        no="नहीं",
        also="साथ ही",
        cause_word="क्योंकि",
        effect_word="इसलिए",
        true="सत्य",
        false="असत्य",
        neither="न तो यह, न वह",
        or_word="या",
        full_stop="।",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        indices=["क", "ख", "ग", "घ", "ङ", "च"],
    ),
    Language.HUNGARIAN: TranslationLiterals(
        language=Language.HUNGARIAN,
        question_word="kérdés",
        answer="válasz",
        confirmation_word="ugye",
        yes="igen",
        no="nem",
        also="is",
        cause_word="mert",
        effect_word="ezért",
        or_word="vagy",
        true="igaz",
        false="hamis",
        neither="egyik sem",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.ICELANDIC: TranslationLiterals(language=Language.ICELANDIC),
    Language.INDONESIAN: TranslationLiterals(
        language=Language.INDONESIAN,
        question_word="pertanyaan",
        answer="jawaban",
        confirmation_word="kan",
        yes="ya",
        no="tidak",
        also="juga",
        cause_word="karena",
        effect_word="maka",
        or_word="atau",
        true="benar",
        false="salah",
        neither="tidak satu pun",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.IRISH: TranslationLiterals(language=Language.IRISH),
    Language.ITALIAN: TranslationLiterals(
        language=Language.ITALIAN,
        question_word="domanda",
        answer="risposta",
        confirmation_word="vero",
        yes="sì",
        no="no",
        also="inoltre",
        cause_word="perchè",
        effect_word="quindi",
        or_word="o",
        true="vero",
        false="falso",
        neither="nessuno dei due",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.JAPANESE: TranslationLiterals(
        language=Language.JAPANESE,
        question_word="質問",
        answer="回答",
        confirmation_word="でしょうか",
        yes="はい",
        no="いいえ",
        also="また",
        cause_word="なので",
        effect_word="なぜなら",
        or_word="または",
        true="正解",
        false="不正解",
        neither="どちらでもない",
        full_stop="。",
        comma="、",
        question_mark="？",
        exclamation_mark="！",
        word_space="",
        sentence_space="",
        colon="：",
        semicolon="；",
    ),
    Language.JAVANESE: TranslationLiterals(language=Language.JAVANESE),
    Language.KANNADA: TranslationLiterals(language=Language.KANNADA),
    Language.KAZAKH: TranslationLiterals(language=Language.KAZAKH),
    Language.KHMER: TranslationLiterals(language=Language.KHMER),
    Language.KIRGHIZ: TranslationLiterals(language=Language.KIRGHIZ),
    Language.KOREAN: TranslationLiterals(
        language=Language.KOREAN,
        confirmation_word="맞죠",
        yes="예",
        no="아니오",
    ),
    Language.KURDISH: TranslationLiterals(language=Language.KURDISH),
    Language.LAO: TranslationLiterals(language=Language.LAO),
    Language.LATIN: TranslationLiterals(language=Language.LATIN),
    Language.LATVIAN: TranslationLiterals(language=Language.LATVIAN),
    Language.LITHUANIAN: TranslationLiterals(language=Language.LITHUANIAN),
    Language.LUXEMBOURGISH: TranslationLiterals(language=Language.LUXEMBOURGISH),
    Language.MACEDONIAN: TranslationLiterals(language=Language.MACEDONIAN),
    Language.MALAGASY: TranslationLiterals(language=Language.MALAGASY),
    Language.MALAY: TranslationLiterals(language=Language.MALAY),
    Language.MALAYALAM: TranslationLiterals(language=Language.MALAYALAM),
    Language.MALTESE: TranslationLiterals(language=Language.MALTESE),
    Language.MARATHI: TranslationLiterals(language=Language.MARATHI),
    Language.NEPALI: TranslationLiterals(language=Language.NEPALI),
    Language.NORWEGIAN: TranslationLiterals(
        language=Language.NORWEGIAN,
        question_word="spørsmål",
        answer="svar",
        confirmation_word="ikke sant",
        yes="ja",
        no="nei",
        also="i tillegg",
        cause_word="fordi",
        effect_word="derfor",
        or_word="eller",
        true="sant",
        false="usant",
        neither="ingen av delene",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.NORWEGIAN_NYNORSK: TranslationLiterals(language=Language.NORWEGIAN_NYNORSK),
    Language.OCCITAN: TranslationLiterals(language=Language.OCCITAN),
    Language.ORIYA: TranslationLiterals(language=Language.ORIYA),
    Language.PASHTO: TranslationLiterals(language=Language.PASHTO),
    Language.PERSIAN: TranslationLiterals(language=Language.PERSIAN),
    Language.POLISH: TranslationLiterals(
        language=Language.POLISH,
        question_word="pytanie",
        answer="odpowiedź",
        confirmation_word="prawda",
        yes="tak",
        no="nie",
        also="ponadto",
        cause_word="ponieważ",
        effect_word="więc",
        or_word="lub",
        true="prawda",
        false="fałsz",
        neither="ani jedno ani drugie",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.PORTUGUESE: TranslationLiterals(
        language=Language.PORTUGUESE,
        question_word="pergunta",
        answer="resposta",
        confirmation_word="certo",
        yes="sim",
        no="não",
        also="adicionalmente",
        cause_word="porque",
        effect_word="logo",
        or_word="ou",
        true="verdadeiro",
        false="falso",
        neither="nenhum",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.PUNJABI: TranslationLiterals(language=Language.PUNJABI),
    Language.QUECHUA: TranslationLiterals(
        # From https://github.com/EleutherAI/lm-evaluation-harness/blob/0845b588303f1f59af98dd1c5bdbd78a9e75a1e2/lm_eval/tasks/xcopa/utils.py
        language=Language.QUECHUA,
        cause_word="imataq",
        effect_word="chaymi",
    ),
    Language.ROMANIAN: TranslationLiterals(language=Language.ROMANIAN),
    Language.RUSSIAN: TranslationLiterals(
        language=Language.RUSSIAN,
        question_word="вопрос",
        answer="ответ",
        confirmation_word="верно",
        yes="да",
        no="нет",
        also="к тому же",
        cause_word="потому что",
        effect_word="поэтому",
        true="истина",
        false="ложь",
        neither="ни то ни другое",
        or_word="или",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        indices=["А", "Б", "В", "Г", "Д", "Е"],
    ),
    Language.SANSKRIT: TranslationLiterals(language=Language.SANSKRIT),
    # Latin serbian script for future when separating scipts
    # Language.SERBIAN_LATIN: TranslationLiterals(language=Language.SERBIAN_LATIN,
    #     question_word="pitanje",
    #     answer="odgovor",
    #     confirmation_word="zar ne",
    #     yes="da",
    #     no="ne",
    #     also="takođe",
    #     cause_word="jer",
    #     effect_word="dakle",
    #     or_word="ili",
    #     true="tačno",
    #     false="netačno",
    #     neither="ništa od navedenog",
    # ),
    Language.SERBIAN: TranslationLiterals(
        language=Language.SERBIAN,
        question_word="питање",
        answer="одговор",
        confirmation_word="зар не",
        yes="да",
        no="не",
        also="такође",
        cause_word="јер",
        effect_word="дакле",
        or_word="или",
        true="тачно",
        false="нетачно",
        neither="ништа од наведеног",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.SERBOCROATIAN: TranslationLiterals(language=Language.SERBOCROATIAN),  # Deprecated
    Language.SHAN: TranslationLiterals(
        language=Language.SHAN,
        question_word="ၶေႃႈထၢမ်",
        answer="ၶေႃႈတွပ်ႇ",
        confirmation_word="ၸွင်ႇၸႂ်ႈ",
        yes="ၸႂ်ႈ",
        no="ဢမ်ႇ",
        also="လႄႈ",
        cause_word="ၵွပ်ႈပိူဝ်ႈ",
        effect_word="ၵွပ်ႈၼၼ်",
        true="တႄႉ",
        false="ဢမ်ႇတႄႉ",
        neither="ဢမ်ႇၸႂ်ႈတင်းသွင်ဢၼ်",
        or_word="ဢမ်ႇၼၼ်",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space="",
        sentence_space=" ",
        colon=":",
        indices=["ၵ", "ၶ", "င", "ၸ", "သ", "ၺ"],
    ),
    Language.SINDHI: TranslationLiterals(language=Language.SINDHI),
    Language.SINHALA: TranslationLiterals(language=Language.SINHALA),
    Language.SLOVAK: TranslationLiterals(
        language=Language.SLOVAK,
        question_word="otázka",
        answer="odpoveď",
        confirmation_word="že áno",
        yes="áno",
        no="nie",
        also="taktiež",
        cause_word="pretože",
        effect_word="takže",
        or_word="alebo",
        true="pravda",
        false="nepravda",
        neither="ani jeden",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.SOMALI: TranslationLiterals(language=Language.SOMALI),
    Language.SORANI: TranslationLiterals(language=Language.SORANI),
    Language.SOUTH_AZERBAIJANI: TranslationLiterals(language=Language.SOUTH_AZERBAIJANI),
    Language.SPANISH: TranslationLiterals(
        language=Language.SPANISH,
        question_word="pregunta",
        answer="respuesta",
        confirmation_word="cierto",
        yes="sí",
        no="no",
        also="también",
        cause_word="porque",
        effect_word="por lo tanto",
        or_word="o",
        true="verdadero",
        false="falso",
        neither="ninguno",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.SWAHILI: TranslationLiterals(
        language=Language.SWAHILI,
        question_word="swali",
        answer="jibu",
        confirmation_word="sahihi",
        yes="ndiyo",
        no="hapana",
        also="pia",
        cause_word="kwa sababu",
        effect_word="kwa hiyo",
        true="kweli",
        false="uongo",
        neither="hakuna kati ya hizo",
        or_word="au",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.SWEDISH: TranslationLiterals(
        language=Language.SWEDISH,
        question_word="fråga",
        answer="svar",
        confirmation_word="eller hur",
        yes="ja",
        no="nej",
        also="också",
        cause_word="eftersom",
        effect_word="därför att",
        or_word="eller",
        true="sant",
        false="falskt",
        neither="ingendera",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.TAGALOG: TranslationLiterals(language=Language.TAGALOG),
    Language.TAJIK: TranslationLiterals(language=Language.TAJIK),
    Language.TAMIL: TranslationLiterals(
        # From https://github.com/EleutherAI/lm-evaluation-harness/blob/0845b588303f1f59af98dd1c5bdbd78a9e75a1e2/lm_eval/tasks/xcopa/utils.py
        language=Language.TAMIL,
        cause_word="காரணமாக",
        effect_word="எனவே",
    ),
    Language.TATAR: TranslationLiterals(
        language=Language.TATAR,
        question_word="сорау",
        answer="җавап",
        confirmation_word="шулай түгелме",
        yes="әйе",
        no="юк",
        also="шулай ук",
        cause_word="чөнки",
        effect_word="шуңа күрә",
        or_word="яки",
        true="дөрес",
        false="ялган",
        neither="бер генә дә",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        indices=["А", "Б", "В", "Г", "Д", "Е"],
    ),
    Language.TELUGU: TranslationLiterals(
        language=Language.TELUGU,
        question_word="ప్రశ్న",
        answer="జవాబు",
        confirmation_word="కదా",
        yes="అవును",
        no="కాదు",
        also="అలాగే",
        cause_word="ఎందుకంటే",
        effect_word="అందువలన",
        or_word="లేదా",
        true="నిజం",
        false="తప్పు",
        neither="ఏదీ కాదు",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        indices=["ఎ", "బి", "సి", "డి", "ఇ"],
    ),
    Language.THAI: TranslationLiterals(
        language=Language.THAI,
        question_word="คำถาม",
        answer="คำตอบ",
        confirmation_word="ใช่ไหม",
        yes="ใช่",
        no="ไม่",
        also="และ",
        cause_word="เพราะ",
        effect_word="ดังนั้น",
        true="จริง",
        false="เท็จ",
        neither="ไม่ใช่ทั้งสองอย่าง",
        or_word="หรือ",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space="",
        sentence_space=" ",
        colon=":",
        indices=["๑", "๒", "๓", "๔", "๕", "๖", "๗", "๘", "๙", "๐"],
    ),
    Language.TURKISH: TranslationLiterals(
        language=Language.TURKISH,
        question_word="soru",
        answer="cevap",
        confirmation_word="değil mi",
        yes="evet",
        no="hayır",
        also="ayrıca",
        cause_word="çünkü",
        effect_word="bu yüzden",
        true="doğru",
        false="yanlış",
        neither="hiçbiri",
        or_word="veya",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
    ),
    Language.TURKMEN: TranslationLiterals(language=Language.TURKMEN),
    Language.UDMURT: TranslationLiterals(
        language=Language.UDMURT,
        question_word="юан",
        answer="валэктон",
        confirmation_word="озьы-а",
        yes="бен",
        no="ӧвӧл",
        also="озьы ик",
        cause_word="малы ке шуоно",
        effect_word="соин ик",
        true="шонерлык",
        false="пӧяськон",
        neither="мукет",
        or_word="яке",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        indices=["А", "Б", "В", "Г", "Д", "Е"],
    ),
    Language.UKRAINIAN: TranslationLiterals(
        language=Language.UKRAINIAN,
        question_word="питання",
        answer="відповідь",
        confirmation_word="вірно",
        yes="так",
        no="ні",
        also="також",
        cause_word="тому що",
        effect_word="отже",
        or_word="або",
        true="правда",
        false="неправда",
        neither="ні те, ні інше",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.URDU: TranslationLiterals(
        language=Language.URDU,
        question_word="سوال",
        answer="جواب",
        confirmation_word="نا",
        yes="ہاں",
        no="نہیں",
        also="اور",
        cause_word="کیونکہ",
        effect_word="اس لئے",
        or_word="یا",
        true="درست",
        false="غلط",
        neither="کوئی نہیں",
        full_stop="۔",
        comma="،",
        question_mark="؟",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon="؛",
    ),
    Language.UZBEK: TranslationLiterals(language=Language.UZBEK),
    Language.VIETNAMESE: TranslationLiterals(
        language=Language.VIETNAMESE,
        question_word="câu hỏi",
        answer="trả lời",
        confirmation_word="đúng",
        yes="có",
        no="không",
        also="cũng",
        cause_word="vì",
        effect_word="do đó",
        or_word="hoặc",
        true="đúng",
        false="sai",
        neither="không đúng cũng không sai",
        full_stop=".",
        comma=",",
        question_mark="?",
        exclamation_mark="!",
        word_space=" ",
        sentence_space=" ",
        colon=":",
        semicolon=";",
    ),
    Language.WAR: TranslationLiterals(language=Language.WAR),
    Language.WELSH: TranslationLiterals(language=Language.WELSH),
    Language.WESTERN_FRISIAN: TranslationLiterals(language=Language.WESTERN_FRISIAN),
    Language.YIDDISH: TranslationLiterals(language=Language.YIDDISH),
    Language.YORUBA: TranslationLiterals(language=Language.YORUBA),
    Language.ZULU: TranslationLiterals(language=Language.ZULU),
}
