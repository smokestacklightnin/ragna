# import json
from typing import AsyncIterator  # , cast

from ragna.core import Source  # RagnaException

from ._api import ApiAssistant


class AsyncIteratorReader:
    def __init__(self, ait: AsyncIterator[bytes]) -> None:
        self._ait = ait

    async def read(self, n: int) -> bytes:
        # n is usually used to indicate how many bytes to read, but since we want to
        # return a chunk as soon as it is available, we ignore the value of n. The only
        # exception is n == 0, which is used by ijson to probe the return type and
        # set up decoding.
        if n == 0:
            return b""
        return await anext(self._ait, b"")  # type: ignore[call-arg]


class OllamaApiAssistant(ApiAssistant):
    _MODEL: str

    @classmethod
    def display_name(cls) -> str:
        return f"Ollama/{cls._MODEL}"

    def _make_system_content(self, sources: list[Source]) -> str:
        instruction = (
            "You are an helpful assistants that answers user questions given the context below. "
            "If you don't know the answer, just say so. Don't try to make up an answer. "
            "Only use the following sources to generate the answer."
        )
        return instruction + "\n\n".join(source.content for source in sources)
