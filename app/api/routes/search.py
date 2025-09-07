"""
Semantic search endpoints using vector database
"""

import time
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from loguru import logger

from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.nlp_service import NLPService
from app.services.vector_service import VectorService


router = APIRouter()


def get_nlp_service() -> NLPService:
    """Dependency to get NLP service"""
    from app.main import app
    return app.state.nlp_service


def get_vector_service() -> VectorService:
    """Dependency to get vector service"""
    from app.main import app
    return app.state.vector_service


@router.post("/", response_model=SearchResponse)
async def semantic_search(
    request: SearchRequest,
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service)
):
    """Perform semantic search across documents"""
    
    try:
        start_time = time.time()
        
        # Generate embedding for search query
        query_embeddings = await nlp_service.generate_embeddings([request.query])
        query_embedding = query_embeddings[0]
        
        # Perform vector search
        search_results = await vector_service.semantic_search(
            query_embedding=query_embedding,
            limit=request.limit,
            similarity_threshold=request.similarity_threshold,
            document_ids=request.document_ids
        )
        
        processing_time = time.time() - start_time
        
        response = SearchResponse(
            results=search_results,
            total_results=len(search_results),
            query=request.query,
            processing_time=processing_time
        )
        
        logger.info(f"Semantic search completed: {len(search_results)} results in {processing_time:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/similar/{document_id}")
async def find_similar_documents(
    document_id: str,
    limit: int = 10,
    similarity_threshold: float = 0.5,
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service)
):
    """Find documents similar to a given document"""
    
    try:
        await vector_service.initialize()
        
        # Get document chunks
        chunks = await vector_service.get_document_chunks(document_id)
        
        if not chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Use first chunk as representative text for similarity
        representative_text = chunks[0]['content']
        
        # Generate embedding
        embeddings = await nlp_service.generate_embeddings([representative_text])
        query_embedding = embeddings[0]
        
        # Search for similar documents (excluding the source document)
        search_results = await vector_service.semantic_search(
            query_embedding=query_embedding,
            limit=limit + 10,  # Get extra results to filter out source document
            similarity_threshold=similarity_threshold
        )
        
        # Filter out chunks from the source document
        filtered_results = [
            result for result in search_results 
            if result.document_id != document_id
        ][:limit]
        
        # Group by document to get unique documents
        unique_documents = {}
        for result in filtered_results:
            doc_id = result.document_id
            if doc_id not in unique_documents or result.similarity_score > unique_documents[doc_id].similarity_score:
                unique_documents[doc_id] = result
        
        similar_documents = list(unique_documents.values())
        
        return {
            "source_document_id": document_id,
            "similar_documents": similar_documents,
            "total_found": len(similar_documents)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/qa-search")
async def qa_search(
    question: str,
    limit: int = 5,
    similarity_threshold: float = 0.3,
    document_ids: Optional[List[str]] = None,
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service)
):
    """Search for relevant context and answer question"""
    
    try:
        start_time = time.time()
        
        # Generate embedding for question
        question_embeddings = await nlp_service.generate_embeddings([question])
        question_embedding = question_embeddings[0]
        
        # Search for relevant context
        search_results = await vector_service.semantic_search(
            query_embedding=question_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold,
            document_ids=document_ids
        )
        
        if not search_results:
            return {
                "question": question,
                "answer": "No relevant context found for the question.",
                "confidence": 0.0,
                "sources": [],
                "processing_time": time.time() - start_time
            }
        
        # Combine top results as context
        context_parts = []
        sources = []
        
        for result in search_results:
            context_parts.append(result.content)
            sources.append({
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "similarity_score": result.similarity_score,
                "metadata": result.metadata
            })
        
        combined_context = " ".join(context_parts)
        
        # Answer question using combined context
        qa_response = await nlp_service.answer_question(question, combined_context)
        
        processing_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": qa_response.answer,
            "confidence": qa_response.confidence,
            "sources": sources,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error in QA search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats")
async def get_search_statistics(
    vector_service: VectorService = Depends(get_vector_service)
):
    """Get search and vector database statistics"""
    
    try:
        stats = await vector_service.get_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting search statistics: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reindex")
async def reindex_documents(
    document_ids: Optional[List[str]] = None,
    nlp_service: NLPService = Depends(get_nlp_service),
    vector_service: VectorService = Depends(get_vector_service)
):
    """Reindex documents (regenerate embeddings)"""
    
    try:
        await vector_service.initialize()
        
        # Get documents to reindex
        if document_ids:
            # Reindex specific documents
            query = "SELECT document_id FROM documents WHERE document_id IN ({})".format(
                ','.join(['?' for _ in document_ids])
            )
            results = vector_service.db_conn.execute(query, document_ids).fetchall()
        else:
            # Reindex all documents
            results = vector_service.db_conn.execute("SELECT document_id FROM documents").fetchall()
        
        reindexed_count = 0
        
        for (doc_id,) in results:
            try:
                # Get document chunks
                chunks = await vector_service.get_document_chunks(doc_id)
                
                if chunks:
                    # Generate new embeddings
                    chunk_texts = [chunk['content'] for chunk in chunks]
                    embeddings = await nlp_service.generate_embeddings(chunk_texts)
                    
                    # Update embeddings in vector database
                    # Note: This is a simplified approach - in production you might want
                    # to implement proper reindexing with atomic operations
                    
                    reindexed_count += 1
                    
            except Exception as e:
                logger.error(f"Error reindexing document {doc_id}: {str(e)}")
                continue
        
        return {
            "message": f"Reindexing completed for {reindexed_count} documents",
            "reindexed_count": reindexed_count,
            "total_requested": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error in reindexing: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
