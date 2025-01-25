# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import dataclass
from typing import TypeVar
import importlib
from enum import Enum


ListLikeTypeVar = TypeVar("ListLikeTypeVar")
ListLike = list[ListLikeTypeVar] | tuple[ListLikeTypeVar, ...]


ElementType = TypeVar("ElementType")


def as_list(item: ListLike[ElementType] | ElementType) -> list[ElementType]:
    """
    Convert the given item into a list.

    If the item is already a list, it is returned as is.
    If the item is a tuple, it is converted into a list.
    Otherwise, the item is wrapped in a list.

    Args:
        item (Union[list, tuple, Any]): The item to be converted.

    Returns:
        list: The converted list.

    """
    if isinstance(item, list):
        return item

    elif isinstance(item, tuple):
        return list(item)

    return [item]


@dataclass
class EnvConfig:
    """
    Configuration class for environment settings.

    Attributes:
        cache_dir (str): directory for caching data.
        token (str): authentication token used for accessing the HuggingFace Hub.
    """

    cache_dir: str = os.getenv("HF_HUB_CACHE", "/scratch")
    token: str = os.getenv("HF_TOKEN")



def is_vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None

NO_VLLM_ERROR_MSG = "You are trying to use an VLLM model, for which you need `vllm`, which is not available in your environment. Please install it using pip, `pip install vllm`."


class Language(Enum):
    ENGLISH = "eng"
    SPANISH = "spa"
    PORTUGUESE = "por"
    ITALIAN = "ita"
    FRENCH = "fra"
    ROMANIAN = "ron"
    GERMAN = "deu"
    LATIN = "lat"
    CZECH = "ces"
    DANISH = "dan"
    FINNISH = "fin"
    GREEK = "ell"
    NORWEGIAN = "nor"
    POLISH = "pol"
    RUSSIAN = "rus"
    SLOVENIAN = "slv"
    SWEDISH = "swe"
    TURKISH = "tur"
    DUTCH = "nld"
    CHINESE = "zho"
    JAPANESE = "jpn"
    VIETNAMESE = "vie"
    INDONESIAN = "ind"
    PERSIAN = "fas"
    KOREAN = "kor"
    ARABIC = "ara"
    THAI = "tha"
    HINDI = "hin"
    BENGALI = "ben"
    TAMIL = "tam"
    HUNGARIAN = "hun"
    UKRAINIAN = "ukr"
    SLOVAK = "slk"
    BULGARIAN = "bul"
    CATALAN = "cat"
    CROATIAN = "hrv"
    SERBIAN = "srp"
    LITHUANIAN = "lit"
    ESTONIAN = "est"
    HEBREW = "heb"
    LATVIAN = "lav"
    SERBOCROATIAN = "hbs"  # Deprecated
    ALBANIAN = "sqi"
    AZERBAIJANI = "aze"
    ICELANDIC = "isl"
    MACEDONIAN = "mkd"
    GEORGIAN = "kat"
    GALICIAN = "glg"
    ARMENIAN = "hye"
    BASQUE = "eus"
    SWAHILI = "swa"
    MALAY = "msa"
    TAGALOG = "tgl"
    JAVANESE = "jav"
    PUNJABI = "pan"
    BIHARI = "bih"  # Deprecated
    GUJARATI = "guj"
    YORUBA = "yor"
    MARATHI = "mar"
    URDU = "urd"
    AMHARIC = "amh"
    TELUGU = "tel"
    HAITIAN = "hti"
    MALAYALAM = "mal"
    KANNADA = "kan"
    NEPALI = "nep"
    KAZAKH = "kaz"
    BELARUSIAN = "bel"
    BURMESE = "mya"
    ESPERANTO = "epo"
    UZBEK = "uzb"
    KHMER = "khm"
    TAJIK = "tgk"
    WELSH = "cym"
    NORWEGIAN_NYNORSK = "nno"
    BOSNIAN = "bos"
    SINHALA = "sin"
    TATAR = "tat"
    AFRIKAANS = "afr"
    ORIYA = "ori"
    KIRGHIZ = "kir"
    IRISH = "gle"
    OCCITAN = "oci"
    KURDISH = "kur"
    LAO = "lao"
    LUXEMBOURGISH = "ltz"
    BASHKIR = "bak"
    WESTERN_FRISIAN = "fry"
    PASHTO = "pus"
    MALTESE = "mlt"
    BRETON = "bre"
    ASSAMESE = "asm"
    MALAGASY = "mlg"
    DIVEHI = "div"
    YIDDISH = "yid"
    SOMALI = "som"
    SANSKRIT = "san"
    SINDHI = "snd"
    QUECHUA = "que"
    TURKMEN = "tuk"
    SOUTH_AZERBAIJANI = "azb"
    SORANI = "ckb"
    CEBUANO = "ceb"
    WAR = "war"
    SHAN = "shn"
    UDMURT = "udm"
    ZULU = "zul"
