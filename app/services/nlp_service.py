"""
NLP service using Hugging Face transformers for various NLP tasks
"""

import asyncio
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import pipeline, Pipeline
from sentence_transformers import SentenceTransformer
from loguru import logger
from joblib import Parallel, delayed

from app.core.config import settings
from app.models.schemas import SummarizationResponse, QAResponse, NERResponse, Entity


class NLPService:
    """Service for NLP tasks using Hugging Face models"""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_workers = settings.MAX_WORKERS
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        # Initialize models lazily
        self._summarization_pipeline = None
        self._qa_pipeline = None
        self._ner_pipeline = None
        self._embedding_model = None

        logger.info(f"NLP Service initialized with device: {self.device}")

    @property
    def summarization_pipeline(self) -> Pipeline:
        """Lazy load summarization pipeline"""
        if self._summarization_pipeline is None:
            logger.info("Loading summarization model...")
            self._summarization_pipeline = pipeline(
                "summarization",
                model=settings.SUMMARIZATION_MODEL,
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("Summarization model loaded successfully")
        return self._summarization_pipeline

    @property
    def qa_pipeline(self) -> Pipeline:
        """Lazy load Q&A pipeline"""
        if self._qa_pipeline is None:
            logger.info("Loading Q&A model...")
            self._qa_pipeline = pipeline(
                "question-answering",
                model=settings.QA_MODEL,
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("Q&A model loaded successfully")
        return self._qa_pipeline

    @property
    def ner_pipeline(self) -> Pipeline:
        """Lazy load NER pipeline"""
        if self._ner_pipeline is None:
            logger.info("Loading NER model...")
            self._ner_pipeline = pipeline(
                "ner",
                model=settings.NER_MODEL,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("NER model loaded successfully")
        return self._ner_pipeline

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load embedding model"""
        if self._embedding_model is None:
            logger.info("Loading embedding model...")
            self._embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL, device=self.device
            )
            logger.info("Embedding model loaded successfully")
        return self._embedding_model

    async def summarize_text(
        self, text: str, max_length: int = 150, min_length: int = 30
    ) -> SummarizationResponse:
        """Generate text summary"""
        try:
            # Run summarization in thread pool
            loop = asyncio.get_event_loop()
            summary_result = await loop.run_in_executor(
                self.executor, self._summarize_sync, text, max_length, min_length
            )

            summary = summary_result[0]["summary_text"]

            # Calculate metrics
            original_length = len(text.split())
            summary_length = len(summary.split())
            compression_ratio = (
                summary_length / original_length if original_length > 0 else 0
            )

            response = SummarizationResponse(
                summary=summary,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=compression_ratio,
            )

            logger.info(f"Text summarized: {original_length} -> {summary_length} words")
            return response
        except Exception as e:
            logger.error(f"Error in text summarization: {str(e)}")
            raise

    async def answer_question(self, question: str, context: str) -> QAResponse:
        """Answer question based on context"""
        try:
            # Run Q&A in thread pool
            loop = asyncio.get_event_loop()
            qa_result = await loop.run_in_executor(
                self.executor, self._qa_sync, question, context
            )

            response = QAResponse(
                answer=qa_result["answer"],
                confidence=qa_result["score"],
                start_position=qa_result.get("start"),
                end_position=qa_result.get("end"),
            )

            logger.info(f"Question answered with confidence: {qa_result['score']:.3f}")
            return response
        except Exception as e:
            logger.error(f"Error in question answering: {str(e)}")
            raise

    async def extract_entities(self, text: str) -> NERResponse:
        """Extract named entities from text"""
        try:
            # Run NER in thread pool
            loop = asyncio.get_event_loop()
            ner_results = await loop.run_in_executor(
                self.executor, self._ner_sync, text
            )

            # Convert to Entity objects
            entities = []
            for result in ner_results:
                entity = Entity(
                    text=result["word"],
                    label=result["entity_group"],
                    confidence=result["score"],
                    start_position=result["start"],
                    end_position=result["end"],
                )
                entities.append(entity)

            response = NERResponse(entities=entities, entity_count=len(entities))

            logger.info(f"Extracted {len(entities)} entities")
            return response
        except Exception as e:
            logger.error(f"Error in named entity recognition: {str(e)}")
            raise

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        try:
            # Run embedding generation in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self.executor, self._generate_embeddings_sync, texts
            )

            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    async def batch_process_chunks(
        self, chunks: List[str], task: str = "embeddings"
    ) -> List[Any]:
        """Process multiple text chunks in parallel"""
        try:
            if task == "embeddings":
                # Process embeddings in batches
                batch_size = 32
                all_embeddings = []

                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    embeddings = await self.generate_embeddings(batch)
                    all_embeddings.extend(embeddings)

                return all_embeddings

            if task == "summarization":
                # Process summaries in parallel using joblib
                loop = asyncio.get_event_loop()
                summaries = await loop.run_in_executor(
                    None,
                    lambda: Parallel(n_jobs=self.max_workers)(
                        delayed(self._summarize_sync)(chunk, 100, 20)
                        for chunk in chunks
                    ),
                )
                return [s[0]["summary_text"] for s in summaries]

            if task == "ner":
                # Process NER in parallel using joblib
                loop = asyncio.get_event_loop()
                ner_results = await loop.run_in_executor(
                    None,
                    lambda: Parallel(n_jobs=self.max_workers)(
                        delayed(self._ner_sync)(chunk) for chunk in chunks
                    ),
                )
                return ner_results

            raise ValueError(f"Unsupported batch processing task: {task}")
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise

    def _summarize_sync(self, text: str, max_length: int, min_length: int):
        """Synchronous summarization"""
        return self.summarization_pipeline(
            text, max_length=max_length, min_length=min_length, do_sample=False
        )

    def _qa_sync(self, question: str, context: str):
        """Synchronous question answering"""
        return self.qa_pipeline(question=question, context=context)

    def _ner_sync(self, text: str):
        """Synchronous named entity recognition"""
        return self.ner_pipeline(text)

    def _generate_embeddings_sync(self, texts: List[str]):
        """Synchronous embedding generation"""
        return self.embedding_model.encode(texts, convert_to_tensor=False)

    async def detect_language(self, text: str) -> str:
        """Detect language of text (simple implementation)"""
        # This is a placeholder - you might want to use a proper language
        # detection library
        # like langdetect or polyglot
        try:
            # Simple heuristic based on common words
            english_words = [
                "the",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
            ]
            text_lower = text.lower()
            english_count = sum(1 for word in english_words if word in text_lower)

            if english_count >= 3:
                return "en"
            return "unknown"
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            return "unknown"

    async def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
        logger.info("NLP Service cleaned up")
