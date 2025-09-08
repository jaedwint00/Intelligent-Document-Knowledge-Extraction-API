"""
NLP processing endpoints for summarization, Q&A, and entity extraction
"""

from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from app.models.schemas import (
    SummarizationRequest,
    SummarizationResponse,
    QARequest,
    QAResponse,
    NERRequest,
    NERResponse,
)
from app.services.nlp_service import NLPService
from app.services.vector_service import VectorService


router = APIRouter()


def get_nlp_service() -> NLPService:
    """Dependency to get NLP service"""
    from app.main import app  # pylint: disable=import-outside-toplevel

    return app.state.nlp_service


def get_vector_service() -> VectorService:
    """Dependency to get vector service"""
    from app.main import app  # pylint: disable=import-outside-toplevel

    return app.state.vector_service


@router.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(
    request: SummarizationRequest,
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service),
):
    """Generate text summary"""

    try:
        text_to_summarize = None

        # Get text from request or document
        if request.text:
            text_to_summarize = request.text
        elif request.document_id:
            # Get document text from chunks
            await vector_service.initialize()
            chunks = await vector_service.get_document_chunks(request.document_id)

            if not chunks:
                raise HTTPException(status_code=404, detail="Document not found")

            # Combine all chunks
            text_to_summarize = " ".join([chunk["content"] for chunk in chunks])
        else:
            raise HTTPException(
                status_code=400, detail="Either text or document_id must be provided"
            )

        # Generate summary
        summary_response = await nlp_service.summarize_text(
            text_to_summarize, max_length=request.max_length, min_length=request.min_length
        )

        logger.info("Text summarized successfully")
        return summary_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in text summarization: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/qa", response_model=QAResponse)
async def answer_question(
    request: QARequest,
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service),
):
    """Answer question based on context"""

    try:
        context_text = None

        # Get context from request or document
        if request.context:
            context_text = request.context
        elif request.document_id:
            # Get document text from chunks
            await vector_service.initialize()
            chunks = await vector_service.get_document_chunks(request.document_id)

            if not chunks:
                raise HTTPException(status_code=404, detail="Document not found")

            # Combine all chunks as context
            context_text = " ".join([chunk["content"] for chunk in chunks])
        else:
            raise HTTPException(
                status_code=400, detail="Either context or document_id must be provided"
            )

        # Answer question
        qa_response = await nlp_service.answer_question(request.question, context_text)

        logger.info("Question answered successfully")
        return qa_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in question answering: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/ner", response_model=NERResponse)
async def extract_entities(
    request: NERRequest,
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service),
):
    """Extract named entities from text"""

    try:
        text_to_analyze = None

        # Get text from request or document
        if request.text:
            text_to_analyze = request.text
        elif request.document_id:
            # Get document text from chunks
            await vector_service.initialize()
            chunks = await vector_service.get_document_chunks(request.document_id)

            if not chunks:
                raise HTTPException(status_code=404, detail="Document not found")

            # Combine all chunks
            text_to_analyze = " ".join([chunk["content"] for chunk in chunks])
        else:
            raise HTTPException(
                status_code=400, detail="Either text or document_id must be provided"
            )

        # Extract entities
        ner_response = await nlp_service.extract_entities(text_to_analyze)

        logger.info("Named entities extracted successfully")
        return ner_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in named entity recognition: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/batch/summarize")
async def batch_summarize(
    document_ids: list[str],
    max_length: int = 150,
    min_length: int = 30,
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service),
):
    """Batch summarization of multiple documents"""

    try:
        await vector_service.initialize()

        results = []

        for document_id in document_ids:
            try:
                # Get document chunks
                chunks = await vector_service.get_document_chunks(document_id)

                if not chunks:
                    results.append({"document_id": document_id, "error": "Document not found"})
                    continue

                # Combine chunks
                text = " ".join([chunk["content"] for chunk in chunks])

                # Generate summary
                summary_response = await nlp_service.summarize_text(
                    text, max_length=max_length, min_length=min_length
                )

                results.append(
                    {
                        "document_id": document_id,
                        "summary": summary_response.summary,
                        "compression_ratio": str(summary_response.compression_ratio),
                    }
                )

            except (ValueError, FileNotFoundError, RuntimeError) as e:
                results.append({"document_id": document_id, "error": str(e)})

        return {
            "results": results,
            "total_processed": len(document_ids),
            "successful": len([r for r in results if "error" not in r]),
        }

    except Exception as e:
        logger.error(f"Error in batch summarization: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/analyze")
async def analyze_document(
    document_id: str,
    include_summary: bool = True,
    include_entities: bool = True,
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service),
):
    """Comprehensive document analysis including summary and entities"""

    try:
        await vector_service.initialize()

        # Get document chunks
        chunks = await vector_service.get_document_chunks(document_id)

        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")

        # Combine chunks
        full_text = " ".join([chunk["content"] for chunk in chunks])

        analysis_results = {
            "document_id": document_id,
            "text_length": len(full_text.split()),
            "chunk_count": len(chunks),
        }

        # Generate summary if requested
        if include_summary:
            summary_response = await nlp_service.summarize_text(full_text)
            analysis_results["summary"] = {
                "text": summary_response.summary,
                "compression_ratio": summary_response.compression_ratio,
            }

        # Extract entities if requested
        if include_entities:
            ner_response = await nlp_service.extract_entities(full_text)
            analysis_results["entities"] = {
                "total_count": ner_response.entity_count,
                "entities": [
                    {"text": entity.text, "label": entity.label, "confidence": entity.confidence}
                    for entity in ner_response.entities
                ],
            }

        logger.info(f"Document analysis completed for: {document_id}")
        return analysis_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
