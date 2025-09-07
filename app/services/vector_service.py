"""
Vector database service using FAISS and DuckDB for semantic search
"""

import os
import asyncio
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
import duckdb
from loguru import logger

from app.core.config import settings
from app.models.schemas import TextChunk, SearchResult, DocumentMetadata


class VectorService:
    """Service for vector storage and semantic search"""
    
    def __init__(self):
        self.faiss_index_path = settings.FAISS_INDEX_PATH
        self.duckdb_path = settings.DUCKDB_PATH
        self.vector_dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Initialize FAISS index
        self.index = None
        self.document_chunks = {}  # Map index_id to chunk metadata
        
        # Initialize DuckDB connection
        self.db_conn = None
        
        # Initialize on first use
        self._initialized = False
    
    async def initialize(self):
        """Initialize vector database and FAISS index"""
        if self._initialized:
            return
            
        try:
            # Initialize DuckDB
            await self._init_duckdb()
            
            # Initialize FAISS index
            await self._init_faiss()
            
            self._initialized = True
            logger.info("Vector service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector service: {str(e)}")
            raise
    
    async def _init_duckdb(self):
        """Initialize DuckDB database"""
        try:
            # Create connection
            self.db_conn = duckdb.connect(self.duckdb_path)
            
            # Create tables
            await self._create_tables()
            
            logger.info("DuckDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing DuckDB: {str(e)}")
            raise
    
    async def _init_faiss(self):
        """Initialize FAISS index"""
        try:
            # Check if index exists
            if os.path.exists(f"{self.faiss_index_path}.index"):
                # Load existing index
                self.index = faiss.read_index(f"{self.faiss_index_path}.index")
                
                # Load chunk metadata
                if os.path.exists(f"{self.faiss_index_path}.metadata"):
                    with open(f"{self.faiss_index_path}.metadata", 'rb') as f:
                        self.document_chunks = pickle.load(f)
                
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.vector_dimension)  # Inner product for cosine similarity
                logger.info("Created new FAISS index")
                
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create DuckDB tables"""
        
        # Documents table
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                document_id VARCHAR PRIMARY KEY,
                filename VARCHAR NOT NULL,
                file_size INTEGER NOT NULL,
                document_type VARCHAR NOT NULL,
                upload_timestamp TIMESTAMP NOT NULL,
                processing_timestamp TIMESTAMP,
                word_count INTEGER,
                page_count INTEGER,
                language VARCHAR
            )
        """)
        
        # Chunks table
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR PRIMARY KEY,
                document_id VARCHAR NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_position INTEGER NOT NULL,
                end_position INTEGER NOT NULL,
                vector_index_id INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents(document_id)
            )
        """)
        
        # Create indexes
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)")
        self.db_conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_vector_index_id ON chunks(vector_index_id)")
        
        logger.info("DuckDB tables created successfully")
    
    async def store_document_metadata(self, metadata: DocumentMetadata):
        """Store document metadata in DuckDB"""
        await self.initialize()
        
        try:
            self.db_conn.execute("""
                INSERT OR REPLACE INTO documents 
                (document_id, filename, file_size, document_type, upload_timestamp, 
                 processing_timestamp, word_count, page_count, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                metadata.document_id,
                metadata.filename,
                metadata.file_size,
                metadata.document_type.value,
                metadata.upload_timestamp,
                metadata.processing_timestamp,
                metadata.word_count,
                metadata.page_count,
                metadata.language
            ])
            
            logger.info(f"Stored metadata for document: {metadata.document_id}")
            
        except Exception as e:
            logger.error(f"Error storing document metadata: {str(e)}")
            raise
    
    async def store_chunks_with_embeddings(
        self, 
        chunks: List[TextChunk], 
        embeddings: List[List[float]]
    ):
        """Store text chunks and their embeddings"""
        await self.initialize()
        
        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings_array)
            
            # Get current index size
            start_index = self.index.ntotal
            
            # Add embeddings to FAISS index
            self.index.add(embeddings_array)
            
            # Store chunks in DuckDB and update metadata
            for i, chunk in enumerate(chunks):
                vector_index_id = start_index + i
                
                # Store in DuckDB
                self.db_conn.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, document_id, content, chunk_index, start_position, end_position, vector_index_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.content,
                    chunk.chunk_index,
                    chunk.start_position,
                    chunk.end_position,
                    vector_index_id
                ])
                
                # Store in memory mapping
                self.document_chunks[vector_index_id] = {
                    'chunk_id': chunk.chunk_id,
                    'document_id': chunk.document_id,
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index
                }
            
            # Save FAISS index and metadata
            await self._save_index()
            
            logger.info(f"Stored {len(chunks)} chunks with embeddings")
            
        except Exception as e:
            logger.error(f"Error storing chunks with embeddings: {str(e)}")
            raise
    
    async def semantic_search(
        self, 
        query_embedding: List[float], 
        limit: int = 10,
        similarity_threshold: float = 0.5,
        document_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform semantic search using FAISS"""
        await self.initialize()
        
        try:
            # Convert query embedding to numpy array and normalize
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search FAISS index
            similarities, indices = self.index.search(query_vector, limit * 2)  # Get more results for filtering
            
            results = []
            for i, (similarity, index_id) in enumerate(zip(similarities[0], indices[0])):
                if similarity < similarity_threshold:
                    continue
                
                if index_id == -1:  # FAISS returns -1 for empty slots
                    continue
                
                # Get chunk metadata
                chunk_metadata = self.document_chunks.get(int(index_id))
                if not chunk_metadata:
                    continue
                
                # Filter by document IDs if specified
                if document_ids and chunk_metadata['document_id'] not in document_ids:
                    continue
                
                # Get additional metadata from DuckDB
                doc_metadata = self.db_conn.execute("""
                    SELECT d.filename, d.document_type, d.upload_timestamp
                    FROM documents d
                    WHERE d.document_id = ?
                """, [chunk_metadata['document_id']]).fetchone()
                
                metadata = {}
                if doc_metadata:
                    metadata = {
                        'filename': doc_metadata[0],
                        'document_type': doc_metadata[1],
                        'upload_timestamp': doc_metadata[2].isoformat() if doc_metadata[2] else None,
                        'chunk_index': chunk_metadata['chunk_index']
                    }
                
                result = SearchResult(
                    document_id=chunk_metadata['document_id'],
                    chunk_id=chunk_metadata['chunk_id'],
                    content=chunk_metadata['content'],
                    similarity_score=float(similarity),
                    metadata=metadata
                )
                
                results.append(result)
                
                if len(results) >= limit:
                    break
            
            logger.info(f"Found {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        await self.initialize()
        
        try:
            chunks = self.db_conn.execute("""
                SELECT chunk_id, content, chunk_index, start_position, end_position
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
            """, [document_id]).fetchall()
            
            return [
                {
                    'chunk_id': chunk[0],
                    'content': chunk[1],
                    'chunk_index': chunk[2],
                    'start_position': chunk[3],
                    'end_position': chunk[4]
                }
                for chunk in chunks
            ]
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            raise
    
    async def delete_document(self, document_id: str):
        """Delete document and its chunks from vector database"""
        await self.initialize()
        
        try:
            # Get vector indices for chunks to remove from FAISS
            chunk_indices = self.db_conn.execute("""
                SELECT vector_index_id FROM chunks WHERE document_id = ?
            """, [document_id]).fetchall()
            
            # Remove from DuckDB
            self.db_conn.execute("DELETE FROM chunks WHERE document_id = ?", [document_id])
            self.db_conn.execute("DELETE FROM documents WHERE document_id = ?", [document_id])
            
            # Remove from memory mapping
            for (index_id,) in chunk_indices:
                if index_id is not None and index_id in self.document_chunks:
                    del self.document_chunks[index_id]
            
            # Note: FAISS doesn't support efficient deletion, so we keep the vectors
            # In a production system, you might want to rebuild the index periodically
            
            logger.info(f"Deleted document: {document_id}")
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get vector database statistics"""
        await self.initialize()
        
        try:
            # Get document count
            doc_count = self.db_conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            
            # Get chunk count
            chunk_count = self.db_conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            
            # Get FAISS index size
            vector_count = self.index.ntotal if self.index else 0
            
            return {
                'document_count': doc_count,
                'chunk_count': chunk_count,
                'vector_count': vector_count,
                'index_dimension': self.vector_dimension
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            raise
    
    async def _save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{self.faiss_index_path}.index")
            
            # Save metadata
            with open(f"{self.faiss_index_path}.metadata", 'wb') as f:
                pickle.dump(self.document_chunks, f)
            
            logger.debug("FAISS index and metadata saved")
            
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.db_conn:
                self.db_conn.close()
            
            if self.index:
                await self._save_index()
            
            logger.info("Vector service cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise
