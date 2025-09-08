"""
Document processing service for handling file uploads and text extraction
"""

import os
import uuid
import asyncio
from typing import List
from datetime import datetime
import aiofiles
import magic
from loguru import logger

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument

from app.core.config import settings
from app.models.schemas import DocumentType, DocumentMetadata, TextChunk


class DocumentService:
    """Service for document processing and text extraction"""

    def __init__(self):
        self.upload_dir = settings.UPLOAD_DIR
        self.max_file_size = settings.MAX_FILE_SIZE
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS

    async def upload_document(self, file_content: bytes, filename: str) -> DocumentMetadata:
        """Upload and process a document"""

        # Validate file
        await self._validate_file(file_content, filename)

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Determine document type
        document_type = self._get_document_type(filename)

        # Save file
        await self._save_file(document_id, filename, file_content)

        # Create metadata
        metadata = DocumentMetadata(
            document_id=document_id,
            filename=filename,
            file_size=len(file_content),
            document_type=document_type,
            upload_timestamp=datetime.utcnow(),
        )

        logger.info(f"Document uploaded successfully: {document_id}")
        return metadata

    async def extract_text(
        self, document_id: str, file_path: str, document_type: DocumentType
    ) -> str:
        """Extract text from document based on type"""

        try:
            if document_type == DocumentType.PDF:
                return await self._extract_pdf_text(file_path)
            if document_type == DocumentType.DOCX:
                return await self._extract_docx_text(file_path)
            if document_type == DocumentType.TXT:
                return await self._extract_txt_text(file_path)
            raise ValueError(f"Unsupported document type: {document_type}")

        except Exception as e:
            logger.error(f"Error extracting text from {document_id}: {str(e)}")
            raise

    async def chunk_text(self, text: str, document_id: str) -> List[TextChunk]:
        """Split text into chunks for processing"""

        chunks: List[TextChunk] = []
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP

        # Simple chunking strategy
        words = text.split()

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            if not chunk_text.strip():
                continue

            chunk = TextChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document_id,
                content=chunk_text,
                chunk_index=len(chunks),
                start_position=i,
                end_position=min(i + chunk_size, len(words)),
            )

            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks

    async def _validate_file(self, file_content: bytes, filename: str):
        """Validate uploaded file"""

        # Check file size
        if len(file_content) > self.max_file_size:
            raise ValueError(
                f"File size exceeds maximum allowed size of " f"{self.max_file_size} bytes"
            )

        # Check file extension
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in list(self.allowed_extensions):
            raise ValueError(f"File extension {file_ext} not allowed")

        # Check file type using magic
        try:
            file_type = magic.from_buffer(file_content, mime=True)
            logger.debug(f"Detected file type: {file_type}")
        except (OSError, ValueError, TypeError) as e:
            logger.warning(f"Could not detect file type: {str(e)}")

    def _get_document_type(self, filename: str) -> DocumentType:
        """Determine document type from filename"""

        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext == ".pdf":
            return DocumentType.PDF
        if file_ext in [".docx", ".doc"]:
            return DocumentType.DOCX
        if file_ext == ".txt":
            return DocumentType.TXT
        raise ValueError(f"Unsupported file extension: {file_ext}")

    async def _save_file(self, document_id: str, filename: str, content: bytes) -> str:
        """Save uploaded file to disk"""

        file_ext = os.path.splitext(filename)[1]
        file_path = os.path.join(self.upload_dir, f"{document_id}{file_ext}")

        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

        return file_path

    async def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""

        def extract_sync():
            text = ""
            with open(file_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, extract_sync)

        return text.strip()

    async def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""

        def extract_sync():
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, extract_sync)

        return text.strip()

    async def _extract_txt_text(self, file_path: str) -> str:
        """Extract text from TXT file"""

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            text = await f.read()

        return text.strip()

    async def get_document_path(self, document_id: str, filename: str) -> str:
        """Get the file path for a document"""
        file_ext = os.path.splitext(filename)[1]
        return os.path.join(self.upload_dir, f"{document_id}{file_ext}")

    async def delete_document(self, document_id: str, filename: str):
        """Delete a document file"""
        file_path = await self.get_document_path(document_id, filename)

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted document file: {file_path}")
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {str(e)}")
            raise
