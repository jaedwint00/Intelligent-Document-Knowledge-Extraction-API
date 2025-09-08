"""
Document ingestion and processing endpoints
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from loguru import logger

from app.models.schemas import DocumentUploadResponse, DocumentMetadata, ProcessingStatus
from app.services.document_service import DocumentService
from app.services.nlp_service import NLPService
from app.services.vector_service import VectorService


router = APIRouter()


def get_document_service() -> DocumentService:
    """Dependency to get document service"""
    return DocumentService()


def get_nlp_service() -> NLPService:
    """Dependency to get NLP service"""
    from app.main import app  # pylint: disable=import-outside-toplevel

    return app.state.nlp_service


def get_vector_service() -> VectorService:
    """Dependency to get vector service"""
    from app.main import app  # pylint: disable=import-outside-toplevel

    return app.state.vector_service


async def process_document_background(
    document_id: str,
    file_path: str,
    metadata: DocumentMetadata,
    document_service: DocumentService,
    nlp_service: NLPService,
    vector_service: VectorService,
):
    """Background task to process uploaded document"""
    try:
        logger.info(f"Starting background processing for document: {document_id}")

        # Extract text from document
        text = await document_service.extract_text(document_id, file_path, metadata.document_type)

        # Update metadata with text statistics
        metadata.word_count = len(text.split())
        # Use current time in production
        metadata.processing_timestamp = metadata.upload_timestamp

        # Detect language
        language = await nlp_service.detect_language(text)
        metadata.language = language

        # Store document metadata
        await vector_service.store_document_metadata(metadata)

        # Create text chunks
        chunks = await document_service.chunk_text(text, document_id)

        # Generate embeddings for chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await nlp_service.generate_embeddings(chunk_texts)

        # Store chunks with embeddings in vector database
        await vector_service.store_chunks_with_embeddings(chunks, embeddings)

        logger.info(f"Successfully processed document: {document_id}")

    except (ValueError, FileNotFoundError, RuntimeError, IOError) as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        # In production, you might want to update document status to "failed"


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service),
):
    """Upload and process a document"""

    try:
        # Read file content
        file_content = await file.read()

        # Upload document
        filename = file.filename or "unknown_file"
        metadata = await document_service.upload_document(file_content, filename)

        # Get file path for processing
        file_path = await document_service.get_document_path(
            metadata.document_id, metadata.filename
        )

        # Start background processing
        background_tasks.add_task(
            process_document_background,
            metadata.document_id,
            file_path,
            metadata,
            document_service,
            nlp_service,
            vector_service,
        )

        # Return immediate response
        response = DocumentUploadResponse(
            document_id=metadata.document_id,
            filename=metadata.filename,
            file_size=metadata.file_size,
            document_type=metadata.document_type,
            status=ProcessingStatus.PROCESSING,
            upload_timestamp=metadata.upload_timestamp,
        )

        logger.info(f"Document upload initiated: {metadata.document_id}")
        return response

    except ValueError as e:
        logger.error(f"Validation error in document upload: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(
    document_id: str, vector_service: VectorService = Depends(get_vector_service)
):
    """Get document metadata"""

    try:
        await vector_service.initialize()

        # Query document from DuckDB
        result = vector_service.db_conn.execute(
            """
            SELECT document_id, filename, file_size, document_type,
                   upload_timestamp, processing_timestamp, word_count, page_count, language
            FROM documents
            WHERE document_id = ?
        """,
            [document_id],
        ).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Document not found")

        metadata = DocumentMetadata(
            document_id=result[0],
            filename=result[1],
            file_size=result[2],
            document_type=result[3],
            upload_timestamp=result[4],
            processing_timestamp=result[5],
            word_count=result[6],
            page_count=result[7],
            language=result[8],
        )

        return metadata

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/{document_id}/chunks")
async def get_document_chunks(
    document_id: str, vector_service: VectorService = Depends(get_vector_service)
):
    """Get all chunks for a document"""

    try:
        chunks = await vector_service.get_document_chunks(document_id)

        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found or no chunks available")

        return {"document_id": document_id, "chunks": chunks, "total_chunks": len(chunks)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunks for document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
    vector_service: VectorService = Depends(get_vector_service),
):
    """Delete a document and its associated data"""

    try:
        await vector_service.initialize()

        # Check if document exists
        result = vector_service.db_conn.execute(
            """
            SELECT filename FROM documents WHERE document_id = ?
        """,
            [document_id],
        ).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="Document not found")

        filename = result[0]

        # Delete from vector database
        await vector_service.delete_document(document_id)

        # Delete file from disk
        await document_service.delete_document(document_id, filename)

        return {"message": f"Document {document_id} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.get("/")
async def list_documents(
    limit: int = 50, offset: int = 0, vector_service: VectorService = Depends(get_vector_service)
):
    """List all documents with pagination"""

    try:
        await vector_service.initialize()

        # Get documents with pagination
        results = vector_service.db_conn.execute(
            """
            SELECT document_id, filename, file_size, document_type,
                   upload_timestamp, processing_timestamp, word_count, page_count, language
            FROM documents
            ORDER BY upload_timestamp DESC
            LIMIT ? OFFSET ?
        """,
            [limit, offset],
        ).fetchall()

        # Get total count
        total_count = vector_service.db_conn.execute(
            """
            SELECT COUNT(*) FROM documents
        """
        ).fetchone()[0]

        documents = []
        for result in results:
            metadata = DocumentMetadata(
                document_id=result[0],
                filename=result[1],
                file_size=result[2],
                document_type=result[3],
                upload_timestamp=result[4],
                processing_timestamp=result[5],
                word_count=result[6],
                page_count=result[7],
                language=result[8],
            )
            documents.append(metadata)

        return {
            "documents": documents,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") from e
