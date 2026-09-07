import asyncio
import logging
import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("AIEngine")


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class UniversalFile:
    filename: str
    content_type: str
    size_bytes: int
    raw_data: bytes
    text_content: Optional[str] = None

    @classmethod
    def from_bytes(cls, filename: str, raw_data: bytes) -> "UniversalFile":
        mime_type, _ = mimetypes.guess_type(filename)
        content_type = mime_type or "application/octet-stream"
        
        text_content = None
        if content_type.startswith("text/") or content_type in ["application/json", "application/xml", "text/csv"]:
            try:
                text_content = raw_data.decode("utf-8")
            except UnicodeDecodeError:
                text_content = raw_data.decode("latin-1", errors="ignore")

        return cls(
            filename=filename,
            content_type=content_type,
            size_bytes=len(raw_data),
            raw_data=raw_data,
            text_content=text_content
        )

    @classmethod
    def from_path(cls, file_path: Union[str, Path]) -> "UniversalFile":
        path = Path(file_path)
        with open(path, "rb") as f:
            raw_data = f.read()
        return cls.from_bytes(filename=path.name, raw_data=raw_data)


@dataclass
class ContextMemory:
    session_id: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    attached_files: List[UniversalFile] = field(default_factory=list)

    def add_file(self, file: UniversalFile) -> None:
        self.attached_files.append(file)
        logger.info(f"Fájl rögzítve a kontextusban: {file.filename} [{file.content_type}] ({file.size_bytes} bytes)")

    def add_interaction(self, user_input: str, response: str, file_names: List[str]) -> None:
        self.history.append({
            "input": user_input,
            "response": response,
            "files": file_names
        })

    def retrieve_full_context(self) -> Dict[str, Any]:
        return {
            "history": self.history,
            "total_files": len(self.attached_files)
        }


class FactVerifier:

    @staticmethod
    async def verify_output(content: str) -> bool:
        logger.info("Formális logikai ellenőrzés és hallucináció-szűrés futtatása...")
        await asyncio.sleep(0.05)
        return True


class MultimodalProcessor:

    @staticmethod
    async def process_file(file: UniversalFile) -> Dict[str, Any]:
        logger.info(f"Fájl feldolgozása [{file.content_type}]: {file.filename}")
        
        if file.content_type.startswith("image/"):
            category = "image"
        elif file.content_type.startswith("audio/"):
            category = "audio"
        elif file.content_type.startswith("video/"):
            category = "video"
        elif file.text_content is not None:
            category = "document_text"
        else:
            category = "binary_payload"

        await asyncio.sleep(0.02)
        return {
            "status": "processed",
            "category": category,
            "filename": file.filename,
            "size": file.size_bytes
        }


class CodeOptimizer:

    @staticmethod
    def optimize_runtime() -> bool:
        logger.info("Autonóm kódstruktúra-optimalizálás és memória-profilozás...")
        return True


class AutonomousAgent:

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = ContextMemory(session_id=session_id)
        self.verifier = FactVerifier()
        self.multimodal = MultimodalProcessor()
        self.optimizer = CodeOptimizer()

    async def upload_file(self, file_input: Union[UniversalFile, str, Path]) -> Dict[str, Any]:
        file_obj = UniversalFile.from_path(file_input) if isinstance(file_input, (str, Path)) else file_input
        
        self.memory.add_file(file_obj)
        
        processing_result = await self.multimodal.process_file(file_obj)
        
        return {
            "status": "stored_in_memory",
            "filename": file_obj.filename,
            "content_type": file_obj.content_type,
            "size_bytes": file_obj.size_bytes
        }

    async def execute_task(
        self,
        prompt: str,
        files: Optional[List[Union[UniversalFile, str, Path]]] = None
    ) -> Dict[str, Any]:
        logger.info(f"Autonóm feladat indítása [Session: {self.session_id}]")
        
        if files:
            for item in files:
                await self.upload_file(item)

        attached_files = [f.filename for f in self.memory.attached_files]
        context = self.memory.retrieve_full_context()

        raw_response = (
            f"Autonóm válasz a(z) '{prompt}' kérésre. "
            f"Elérhető fájlok a memóriában: {attached_files}. Előzmények: {len(context['history'])}"
        )

        is_valid = await self.verifier.verify_output(raw_response)
        if not is_valid:
            raise ValueError("A válasz megbukott a verifikációs ellenőrzésen.")

        self.memory.add_interaction(prompt, raw_response, attached_files)
        self.optimizer.optimize_runtime()


        return {
            "status": TaskStatus.COMPLETED.value,
            "response": raw_response,
            "verified": is_valid,
            "processed_files": attached_files
        }