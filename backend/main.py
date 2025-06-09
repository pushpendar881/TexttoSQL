import os
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

import cx_Oracle
import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection details from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "service_name": os.getenv("DB_SERVICE_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "schema": os.getenv("DB_SCHEMA"),
}

# API keys from environment variables
MISTRAL_API_KEY = "yrtVdh8NwcaZVGRuMVhFTnGONVN9zDTP"

# File paths
SCHEMA_FILE = "table_details.json"
VECTOR_INDEX_FILE = "faiss_index.bin"
VECTOR_NAMES_FILE = "table_names.json"
VECTOR_DESC_FILE = "table_descriptions.json"
LAST_FETCH_FILE = "last_fetch.txt"

# Available models
MISTRAL_MODELS = ["mistral-large-latest", "mistral-medium", "mistral-small"]

# Use smallest sentence transformer model for local embedding
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 22MB model, fastest and smallest

# Pydantic models for request/response
class NLQueryRequest(BaseModel):
    query: str
    model: str = "mistral-large-latest"

class SQLResponse(BaseModel):
    nl_query: str
    sql_query: str
    relevant_tables: List[Dict[str, Any]]
    execution_time: float

class StatusResponse(BaseModel):
    db_connected: bool
    last_updated: Optional[str] = None
    schema_tables_count: Optional[int] = None

class ExecuteSQLRequest(BaseModel):
    sql_query: str

class LocalEmbedder:
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model locally"""
        try:
            logger.info(f"Loading local embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Successfully loaded model {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise ValueError(f"Failed to load embedding model: {str(e)}")
    
    def encode(self, texts):
        """Get embeddings using local sentence transformer model"""
        if self.model is None:
            raise ValueError("Embedding model not loaded")
        
        if not isinstance(texts, list):
            texts = [texts]
        
        try:
            # Use the local model to generate embeddings
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for better similarity search
            )
            
            # Ensure float32 for FAISS compatibility
            embeddings = embeddings.astype(np.float32)
            
            logger.info(f"Successfully generated embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise ValueError(f"Failed to generate embeddings: {str(e)}")

class SchemaVectorSearch:
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL):
        self.model = LocalEmbedder(model_name)
        self.index = None
        self.table_data = []
        self.table_names = []
    
    def fetch_schema(self):
        # Validate DB configuration
        if not all([DB_CONFIG["user"], DB_CONFIG["password"], DB_CONFIG["schema"]]):
            return None, "Database configuration incomplete. Please check environment variables."
        
        try:
            dsn_tns = cx_Oracle.makedsn(
                DB_CONFIG["host"], 
                DB_CONFIG["port"], 
                service_name=DB_CONFIG["service_name"]
            )
            
            connection = cx_Oracle.connect(
                user=DB_CONFIG["user"], 
                password=DB_CONFIG["password"], 
                dsn=dsn_tns
            )
            
            query = f"""
            SELECT 
                table_name, 
                column_name, 
                data_type, 
                data_length, 
                nullable 
            FROM 
                all_tab_columns 
            WHERE 
                owner = '{DB_CONFIG["schema"]}'
            ORDER BY 
                table_name, column_id
            """
            
            cursor = connection.cursor()
            cursor.execute(query)
            
            tables = {}
            for row in cursor:
                table_name, column_name, data_type, data_length, nullable = row
                if table_name not in tables:
                    tables[table_name] = []
                tables[table_name].append({
                    'column_name': column_name,
                    'data_type': data_type,
                    'data_length': data_length,
                    'nullable': nullable
                })
            
            cursor.close()
            connection.close()
            
            # Check if any tables were found
            if not tables:
                return None, f"No tables found in schema '{DB_CONFIG['schema']}'"
            
            with open(SCHEMA_FILE, 'w') as json_file:
                json.dump(tables, json_file, indent=4)
            
            with open(LAST_FETCH_FILE, 'w') as f:
                f.write(datetime.now().isoformat())
            
            logger.info(f"Successfully fetched schema with {len(tables)} tables")
            return tables, None
        
        except cx_Oracle.DatabaseError as e:
            error_msg = f"Database error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    def load_schema_data(self, schema_json):
        self.table_data = []
        self.table_names = []
        
        for table_name, columns in schema_json.items():
            self.table_names.append(table_name)
            
            column_descriptions = [
                f"{col['column_name']} ({col['data_type']}, {'NULL' if col['nullable'] == 'Y' else 'NOT NULL'})" 
                for col in columns
            ]
            
            table_desc = f"Table {table_name} with columns: {', '.join(column_descriptions)}"
            self.table_data.append(table_desc)
        
        # Check if FAISS index already exists and is valid
        if self._is_index_valid():
            logger.info("Using existing FAISS index file - skipping encoding")
            try:
                self.index = faiss.read_index(VECTOR_INDEX_FILE)
                logger.info(f"Loaded existing vector index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {str(e)}, rebuilding...")
                self.build_index()
        else:
            logger.info("FAISS index not found or invalid - building new index")
            self.build_index()
        
        # Save the current table data for future reference
        with open(VECTOR_NAMES_FILE, 'w') as f:
            json.dump(self.table_names, f)
        
        with open(VECTOR_DESC_FILE, 'w') as f:
            json.dump(self.table_data, f)
        
        logger.info(f"Schema data loaded with {len(self.table_names)} tables")
    
    def _is_index_valid(self):
        """Check if the existing FAISS index is valid and matches current schema"""
        if not os.path.exists(VECTOR_INDEX_FILE):
            return False
        
        if not os.path.exists(VECTOR_NAMES_FILE) or not os.path.exists(VECTOR_DESC_FILE):
            return False
        
        try:
            # Load existing table names and descriptions
            with open(VECTOR_NAMES_FILE, 'r') as f:
                existing_names = json.load(f)
            
            with open(VECTOR_DESC_FILE, 'r') as f:
                existing_descriptions = json.load(f)
            
            # Check if the schema has changed
            if (len(existing_names) != len(self.table_names) or 
                set(existing_names) != set(self.table_names) or
                existing_descriptions != self.table_data):
                logger.info("Schema has changed - index needs to be rebuilt")
                return False
            
            # Try to load the index to verify it's not corrupted
            test_index = faiss.read_index(VECTOR_INDEX_FILE)
            if test_index.ntotal != len(self.table_names):
                logger.info("Index size mismatch - index needs to be rebuilt")
                return False
            
            logger.info("Existing FAISS index is valid and matches current schema")
            return True
            
        except Exception as e:
            logger.warning(f"Error validating existing index: {str(e)}")
            return False
    
    def build_index(self):
        if not self.table_data:
            self.index = None
            return
        
        try:
            # Get embeddings using local model
            logger.info("Building vector index with local embeddings...")
            table_vectors = self.model.encode(self.table_data)
            
            # Create FAISS index (using cosine similarity since embeddings are normalized)
            vector_dimension = table_vectors.shape[1]
            self.index = faiss.IndexFlatIP(vector_dimension)  # Inner product for normalized vectors
            self.index.add(table_vectors)
            
            faiss.write_index(self.index, VECTOR_INDEX_FILE)
            logger.info(f"Vector index built with dimension {vector_dimension}")
        except Exception as e:
            error_msg = f"Error building index: {str(e)}"
            logger.error(error_msg)
            self.index = None
            # Fallback to keyword search if vector indexing fails
            logger.warning("Vector indexing failed, will use keyword search as fallback")
    
    def load_from_disk(self):
        """Load existing vector database from disk without rebuilding"""
        try:
            if os.path.exists(VECTOR_INDEX_FILE) and os.path.exists(VECTOR_NAMES_FILE) and os.path.exists(VECTOR_DESC_FILE):
                # Load the index and metadata
                self.index = faiss.read_index(VECTOR_INDEX_FILE)
                
                with open(VECTOR_NAMES_FILE, 'r') as f:
                    self.table_names = json.load(f)
                
                with open(VECTOR_DESC_FILE, 'r') as f:
                    self.table_data = json.load(f)
                
                # Verify the loaded data is consistent
                if self.index.ntotal == len(self.table_names) == len(self.table_data):
                    logger.info(f"Vector database loaded from disk with {len(self.table_names)} tables")
                    return True
                else:
                    logger.warning("Inconsistency in loaded vector database files")
                    return False
            
            logger.info("Vector database files not found on disk")
            return False
        except Exception as e:
            logger.error(f"Error loading vector database from disk: {str(e)}")
            return False
    
    def keyword_search(self, query, top_k=5):
        """Fallback keyword search when vector search fails"""
        query_lower = query.lower()
        results = []
        
        for i, table_name in enumerate(self.table_names):
            score = 0
            table_lower = table_name.lower()
            
            # Check if query contains table name
            if table_lower in query_lower or query_lower in table_lower:
                score += 10
            
            # Check description for keywords
            if i < len(self.table_data):
                desc_lower = self.table_data[i].lower()
                query_words = query_lower.split()
                for word in query_words:
                    if len(word) > 2 and word in desc_lower:
                        score += 3
            
            if score > 0:
                results.append({
                    "table_name": table_name,
                    "similarity_score": float(score),
                    "description": self.table_data[i] if i < len(self.table_data) else f"Table {table_name}"
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = results[:top_k]
        
        # If no results, return all tables (fallback)
        if not results:
            for i, table_name in enumerate(self.table_names[:top_k]):
                results.append({
                    "table_name": table_name,
                    "similarity_score": 1.0,
                    "description": self.table_data[i] if i < len(self.table_data) else f"Table {table_name}"
                })
        
        logger.info(f"Keyword search for '{query}' returned {len(results)} results")
        return results
    
    def search(self, query, top_k=5):
        # Try vector search first
        if self.index is not None and self.index.ntotal > 0:
            try:
                # Get query embedding using local model
                query_vector = self.model.encode([query])
                
                # Search in FAISS index
                scores, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx >= 0 and idx < len(self.table_names) and scores[0][i] > 0:
                        results.append({
                            "table_name": self.table_names[idx],
                            "similarity_score": float(scores[0][i]),
                            "description": self.table_data[idx]
                        })
                
                # Sort by similarity score in descending order
                results.sort(key=lambda x: x["similarity_score"], reverse=True)
                
                if results:
                    logger.info(f"Vector search for '{query}' returned {len(results)} results")
                    return results
                else:
                    logger.warning("Vector search returned no results, falling back to keyword search")
                    
            except Exception as e:
                logger.error(f"Error during vector search: {str(e)}, falling back to keyword search")
        
        # Fallback to keyword search
        logger.info("Using keyword search fallback")
        return self.keyword_search(query, top_k)

def should_fetch_schema():
    if not os.path.exists(LAST_FETCH_FILE):
        return True
    
    try:
        with open(LAST_FETCH_FILE, 'r') as f:
            last_fetch_str = f.read().strip()
            last_fetch = datetime.fromisoformat(last_fetch_str)
            hours_since_update = (datetime.now() - last_fetch).total_seconds() / 3600
            logger.info(f"Hours since last schema update: {hours_since_update:.2f}")
            return hours_since_update > 24
    except Exception as e:
        logger.warning(f"Error checking schema freshness: {str(e)}")
        return True

def generate_sql(schema_text, nl_query, model="mistral-large-latest"):
    # Validate Mistral API key
    if not MISTRAL_API_KEY:
        return None, "Mistral API key not found. Please set the MISTRAL_API_KEY environment variable."
    
    prompt = f"""
    You are an SQL expert. Convert the following natural language query into a detailed SQL query.
    
    Database Schema (including table details):
    {schema_text}
    
    User Query: {nl_query}
    
    Follow these guidelines:
    1. Use appropriate JOINs based on the table relationships
    2. Include relevant WHERE clauses for filtering
    3. Use aliases for table names to improve readability
    4. Add appropriate aggregate functions (COUNT, SUM, AVG, etc.) if needed
    5. Include ORDER BY, GROUP BY, and HAVING clauses as necessary
    6. Include comments for complex parts of the query
    
    Return ONLY the SQL query without any explanations or code block formatting.
    """
    
    try:
        # Call Mistral API directly
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        
        result = response.json()
        sql_query = result["choices"][0]["message"]["content"].strip()
        
        # Remove any existing formatting if present
        sql_query = sql_query.replace("sql", "").replace("", "").strip()
        
        logger.info(f"SQL query generated successfully using {model}")
        return sql_query, None
    except Exception as e:
        error_msg = f"Error generating SQL: {str(e)}"
        logger.error(error_msg)
        return None, error_msg



# Initialize FastAPI app
app = FastAPI(
    title="Natural Language to SQL API",
    description="Convert natural language queries to SQL using AI",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize schema search
schema_search = SchemaVectorSearch()
schema_loaded = schema_search.load_from_disk()

# Dependency to ensure schema is loaded
async def get_schema_search():
    global schema_loaded
    if not schema_loaded and not os.path.exists(SCHEMA_FILE):
        # Try to fetch schema if not loaded
        schema_json, error = schema_search.fetch_schema()
        if not error:
            schema_search.load_schema_data(schema_json)
            schema_loaded = True
        else:
            raise HTTPException(status_code=503, detail=f"Schema not available: {error}")
    elif not schema_loaded and os.path.exists(SCHEMA_FILE):
        # Load from file if exists but not loaded in memory
        try:
            with open(SCHEMA_FILE, 'r') as f:
                schema_json = json.load(f)
            schema_search.load_schema_data(schema_json)
            schema_loaded = True
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Failed to load schema from file: {str(e)}")
    return schema_search

# Background task to update schema
def update_schema_background():
    if should_fetch_schema():
        schema_json, error = schema_search.fetch_schema()
        if not error:
            schema_search.load_schema_data(schema_json)
            logger.info("Schema updated in background task")
        else:
            logger.error(f"Error updating schema in background: {error}")

@app.on_event("startup")
async def startup_event():
    # Check if we need to fetch schema on startup
    if should_fetch_schema():
        schema_json, error = schema_search.fetch_schema()
        if not error:
            schema_search.load_schema_data(schema_json)
            global schema_loaded
            schema_loaded = True
        else:
            logger.error(f"Error fetching schema on startup: {error}")

@app.get("/", tags=["Status"])
async def root():
    return {"message": "Natural Language to SQL API is running with local embeddings"}

@app.get("/status", response_model=StatusResponse, tags=["Status"])
async def get_status():
    is_connected = os.path.exists(SCHEMA_FILE)
    tables_count = None
    last_updated = None
    
    if is_connected:
        try:
            with open(SCHEMA_FILE, 'r') as f:
                schema_json = json.load(f)
                tables_count = len(schema_json)
            
            if os.path.exists(LAST_FETCH_FILE):
                with open(LAST_FETCH_FILE, 'r') as f:
                    last_fetch_str = f.read().strip()
                    last_fetch = datetime.fromisoformat(last_fetch_str)
                    last_updated = last_fetch.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
    
    return StatusResponse(
        db_connected=is_connected,
        last_updated=last_updated,
        schema_tables_count=tables_count
    )

@app.post("/update-schema", tags=["Schema"])
async def update_schema():
    schema_json, error = schema_search.fetch_schema()
    
    if error:
        raise HTTPException(status_code=500, detail=f"Error fetching schema: {error}")
    
    schema_search.load_schema_data(schema_json)
    global schema_loaded
    schema_loaded = True
    
    return {
        "status": "success",
        "message": "Schema updated successfully",
        "tables_count": len(schema_json)
    }

@app.get("/models", tags=["Models"])
async def get_models():
    return {
        "models": MISTRAL_MODELS
    }

@app.get("/schema-tables", tags=["Schema"])
async def get_schema_tables(schema_search: SchemaVectorSearch = Depends(get_schema_search)):
    try:
        with open(SCHEMA_FILE, 'r') as f:
            schema_json = json.load(f)
        
        return {
            "tables_count": len(schema_json),
            "tables": schema_json
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading schema: {str(e)}")

@app.post("/generate-sql", response_model=SQLResponse, tags=["SQL"])
async def generate_sql_endpoint(
    request: NLQueryRequest, 
    background_tasks: BackgroundTasks,
    schema_search: SchemaVectorSearch = Depends(get_schema_search)
):
    import time
    start_time = time.time()
    
    # Schedule background schema update if needed
    background_tasks.add_task(update_schema_background)
    
    nl_query = request.query
    model = request.model
    
    if model not in MISTRAL_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Choose from: {', '.join(MISTRAL_MODELS)}")
    
    # Search for relevant tables
    relevant_tables = schema_search.search(nl_query)
    
    if not relevant_tables:
        raise HTTPException(status_code=404, detail="No relevant tables found for your query")
    
    try:
        with open(SCHEMA_FILE, 'r') as f:
            schema_json = json.load(f)
        
        # Prepare schema text from relevant tables
        schema_text = ""
        for table in relevant_tables:
            table_name = table['table_name']
            if table_name in schema_json:
                schema_text += f"Table: {table_name}\nColumns:\n"
                for col in schema_json[table_name]:
                    nullable = "NULL" if col['nullable'] == 'Y' else "NOT NULL"
                    schema_text += f"- {col['column_name']} ({col['data_type']}, {nullable})\n"
            else:
                logger.warning(f"Table {table_name} found in vector search but not in schema file")
        
        # Generate SQL query
        sql_query, error = generate_sql(schema_text, nl_query, model)
        
        if error:
            raise HTTPException(status_code=500, detail=f"Error generating SQL: {error}")
        
        execution_time = time.time() - start_time
        
        return SQLResponse(
            nl_query=nl_query,
            sql_query=sql_query,
            relevant_tables=relevant_tables,
            execution_time=execution_time
        )
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)