"""
Pydantic models for request/response schemas
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentType(str, Enum):
    """Document type enumeration"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    DOC = "doc"


class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    document_type: DocumentType = Field(..., description="Document type")
    status: ProcessingStatus = Field(..., description="Processing status")
    upload_timestamp: datetime = Field(..., description="Upload timestamp")


class DocumentMetadata(BaseModel):
    """Document metadata model"""
    document_id: str
    filename: str
    file_size: int
    document_type: DocumentType
    upload_timestamp: datetime
    processing_timestamp: Optional[datetime] = None
    word_count: Optional[int] = None
    page_count: Optional[int] = None
    language: Optional[str] = None


class TextChunk(BaseModel):
    """Text chunk model"""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    start_position: int
    end_position: int
    embedding: Optional[List[float]] = None


class SummarizationRequest(BaseModel):
    """Request model for text summarization"""
    text: Optional[str] = Field(None, description="Text to summarize")
    document_id: Optional[str] = Field(None, description="Document ID to summarize")
    max_length: int = Field(150, description="Maximum summary length")
    min_length: int = Field(30, description="Minimum summary length")


class SummarizationResponse(BaseModel):
    """Response model for text summarization"""
    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original text length")
    summary_length: int = Field(..., description="Summary length")
    compression_ratio: float = Field(..., description="Compression ratio")


class QARequest(BaseModel):
    """Request model for question answering"""
    question: str = Field(..., description="Question to answer")
    context: Optional[str] = Field(None, description="Context text")
    document_id: Optional[str] = Field(None, description="Document ID for context")


class QAResponse(BaseModel):
    """Response model for question answering"""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score")
    start_position: Optional[int] = Field(None, description="Start position in context")
    end_position: Optional[int] = Field(None, description="End position in context")


class Entity(BaseModel):
    """Named entity model"""
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity label")
    confidence: float = Field(..., description="Confidence score")
    start_position: int = Field(..., description="Start position")
    end_position: int = Field(..., description="End position")


class NERRequest(BaseModel):
    """Request model for named entity recognition"""
    text: Optional[str] = Field(None, description="Text to analyze")
    document_id: Optional[str] = Field(None, description="Document ID to analyze")


class NERResponse(BaseModel):
    """Response model for named entity recognition"""
    entities: List[Entity] = Field(..., description="Extracted entities")
    entity_count: int = Field(..., description="Total number of entities")


class SearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str = Field(..., description="Search query")
    limit: int = Field(10, description="Maximum number of results")
    similarity_threshold: float = Field(0.5, description="Minimum similarity threshold")
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")


class SearchResult(BaseModel):
    """Search result model"""
    document_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID")
    content: str = Field(..., description="Matching content")
    similarity_score: float = Field(..., description="Similarity score")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")


class SearchResponse(BaseModel):
    """Response model for semantic search"""
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original query")
    processing_time: float = Field(..., description="Processing time in seconds")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: datetime = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
