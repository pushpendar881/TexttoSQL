import streamlit as st
import cx_Oracle
import json
import os
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_mistralai import ChatMistralAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
#import ollama

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection details from environment variables
# DB_CONFIG = {
#     "host": os.getenv("DB_HOST", "localhost"),
#     "port": os.getenv("DB_PORT", "1521"),
#     "service_name": os.getenv("DB_SERVICE_NAME", "ORCLPDB"),
#     "user": os.getenv("DB_USER", ""),
#     "password": os.getenv("DB_PASSWORD", ""),
#     "schema": os.getenv("DB_SCHEMA", "")
# }
DB_CONFIG = {
    "host": "localhost",
    "port": "1521",
    "service_name": "ORCLPDB",
    "user": "TECH_SCHEMA",
    "password": "5669",
    "schema": "TECH_SCHEMA",
}

# Mistral API key from environment variables
# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_API_KEY ="API_KEY"

# File paths
SCHEMA_FILE = "table_details.json"
VECTOR_INDEX_FILE = "faiss_index.bin"
VECTOR_NAMES_FILE = "table_names.json"
VECTOR_DESC_FILE = "table_descriptions.json"
LAST_FETCH_FILE = "last_fetch.txt"

# Available models
MISTRAL_MODELS = ["mistral-large-latest", "mistral-medium", "mistral-small"]

class SchemaVectorSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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
        
        self.build_index()
        
        with open(VECTOR_NAMES_FILE, 'w') as f:
            json.dump(self.table_names, f)
        
        with open(VECTOR_DESC_FILE, 'w') as f:
            json.dump(self.table_data, f)
        
        logger.info(f"Schema data loaded and indexed with {len(self.table_names)} tables")
    
    def build_index(self):
        if not self.table_data:
            self.index = None
            return
        
        table_vectors = self.model.encode(self.table_data)
        faiss.normalize_L2(table_vectors)
        
        vector_dimension = table_vectors.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)
        self.index.add(table_vectors)
        
        faiss.write_index(self.index, VECTOR_INDEX_FILE)
        logger.info(f"Vector index built with dimension {vector_dimension}")
    
    def load_from_disk(self):
        try:
            if os.path.exists(VECTOR_INDEX_FILE) and os.path.exists(VECTOR_NAMES_FILE) and os.path.exists(VECTOR_DESC_FILE):
                self.index = faiss.read_index(VECTOR_INDEX_FILE)
                
                with open(VECTOR_NAMES_FILE, 'r') as f:
                    self.table_names = json.load(f)
                
                with open(VECTOR_DESC_FILE, 'r') as f:
                    self.table_data = json.load(f)
                
                logger.info(f"Vector database loaded from disk with {len(self.table_names)} tables")
                return True
            
            logger.warning("Vector database files not found")
            return False
        except Exception as e:
            error_msg = f"Error loading vector database: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return False
    
    def search(self, query, top_k=5):  # Increased from 3 to 5 for better coverage
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_vector = self.model.encode([query])
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.table_names) and scores[0][i] > 0:
                results.append({
                    "table_name": self.table_names[idx],
                    "similarity_score": float(scores[0][i]),
                    "description": self.table_data[idx]
                })
        
        # Sort by similarity score in descending order
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        logger.info(f"Search for '{query}' returned {len(results)} results")
        return results

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
    
    prompt_template = PromptTemplate(
        input_variables=["schema", "query"],
        template="""
        You are an SQL expert. Convert the following natural language query into a detailed SQL query.
        
        Database Schema (including table details):
        {schema}
        
        User Query: {query}
        
        Follow these guidelines:
        1. Use appropriate JOINs based on the table relationships
        2. Include relevant WHERE clauses for filtering
        3. Use aliases for table names to improve readability
        4. Add appropriate aggregate functions (COUNT, SUM, AVG, etc.) if needed
        5. Include ORDER BY, GROUP BY, and HAVING clauses as necessary
        6. Include comments for complex parts of the query
        
        Return ONLY the SQL query without any explanations or code block formatting.
        """
    )
    
    try:
        # Use environment variable for API key
        llm = ChatMistralAI(model=model, api_key=MISTRAL_API_KEY)
        # Alternative: Use Ollama
        # llm = Ollama(model=model)
        
        chain = LLMChain(llm=llm, prompt=prompt_template)
        sql_query = chain.run({"schema": schema_text, "query": nl_query})
        
        # Remove any existing formatting if present
        sql_query = sql_query.replace("sql", "").replace("", "").strip()
        
        logger.info(f"SQL query generated successfully using {model}")
        return sql_query, None
    except Exception as e:
        error_msg = f"Error generating SQL: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

# def execute_sql_query(sql_query):
#     """Execute the SQL query and return results (for future implementation)"""
#     # This is a placeholder function for future functionality
#     # It would connect to the database, execute the query, and return results
#     pass

# Custom styling
st.set_page_config(
    page_title="Natural Language to SQL Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextArea {
        border-radius: 10px;
    }
    .stExpander {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.title("🔍 Natural Language to SQL Generator")
st.markdown("Convert your natural language questions into SQL queries using AI")

# Initialize session state
if "db_connected" not in st.session_state:
    st.session_state.db_connected = os.path.exists(SCHEMA_FILE)
    st.session_state.last_checked = datetime.now()

if "last_updated" not in st.session_state and os.path.exists(SCHEMA_FILE):
    try:
        st.session_state.last_updated = datetime.fromtimestamp(
            os.path.getmtime(SCHEMA_FILE)
        ).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        logger.warning(f"Error getting file modification time: {str(e)}")
        st.session_state.last_updated = "Unknown"

if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Initialize schema search
schema_search = SchemaVectorSearch()
vector_db_loaded = schema_search.load_from_disk()

# Auto-refresh logic
if should_fetch_schema():
    with st.spinner("Auto-updating schema..."):
        schema_json, error = schema_search.fetch_schema()
        
        if error:
            st.error(f"Error fetching schema: {error}")
        else:
            schema_search.load_schema_data(schema_json)
            st.session_state.db_connected = True
            st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.last_checked = datetime.now()
            st.rerun()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙ Configuration")
    
    # Connection status
    st.subheader("Database Connection")
    if st.session_state.db_connected:
        st.success("✅ Schema loaded")
        if "last_updated" in st.session_state:
            st.info(f"Last updated: {st.session_state.last_updated}")
    else:
        st.warning("⚠ No schema loaded")
    
    # Model selection
    st.subheader("AI Model Settings")
    model_selection = st.selectbox(
        "Select Mistral Model:",
        MISTRAL_MODELS,
        index=0,
        help="Select which Mistral AI model to use for SQL generation"
    )
    
    # Query history
    st.subheader("Query History")
    if not st.session_state.query_history:
        st.info("No queries yet")
    else:
        for i, query in enumerate(st.session_state.query_history[-5:]):  # Show last 5 queries
            if st.button(f"📝 {query[:30]}...", key=f"hist_{i}"):
                st.session_state.current_query = query

# Button to fetch and update schema
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔄 Update Schema", help="Fetch latest database schema"):
        with st.spinner("Connecting to database and fetching schema..."):
            schema_json, error = schema_search.fetch_schema()
            
            if error:
                st.error(f"Error fetching schema: {error}")
            else:
                schema_search.load_schema_data(schema_json)
                st.session_state.db_connected = True
                st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("✅ Schema updated successfully! Vector database refreshed.")

# Main Query UI
st.subheader("📝 Enter Your Query")
nl_query = st.text_area(
    "Enter your query in natural language:", 
    value=st.session_state.get("current_query", ""),
    placeholder="e.g., Show me all customers who made a purchase last month",
    height=100
)

if st.button("🚀 Generate SQL Query"):
    if not nl_query:
        st.warning("Please enter a query first.")
    else:
        # Add to history if not already there
        if nl_query not in st.session_state.query_history:
            st.session_state.query_history.append(nl_query)
            # Keep only the last 10 queries
            if len(st.session_state.query_history) > 10:
                st.session_state.query_history.pop(0)
        
        if not vector_db_loaded and not st.session_state.db_connected:
            st.error("⚠ No database schema available. Please click 'Update Schema' first.")
        else:
            with st.spinner("Searching for relevant tables..."):
                results = schema_search.search(nl_query)
            
            if not results:
                st.warning("No relevant tables found for your query.")
            else:
                st.subheader("📌 Relevant Tables")
                
                try:
                    with open(SCHEMA_FILE, 'r') as f:
                        schema_json = json.load(f)
                    
                    schema_text = ""
                    for table in results:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"Table: {table['table_name']}")
                        with col2:
                            # Color code based on relevance score
                            score = table['similarity_score']
                            if score > 0.8:
                                st.markdown(f"<span style='color:green'>Relevance: {score:.2f}</span>", unsafe_allow_html=True)
                            elif score > 0.5:
                                st.markdown(f"<span style='color:orange'>Relevance: {score:.2f}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color:red'>Relevance: {score:.2f}</span>", unsafe_allow_html=True)
                        
                        schema_text += f"Table: {table['table_name']}\nColumns:\n"
                        with st.expander(f"Show columns for {table['table_name']}"):
                            for col in schema_json[table["table_name"]]:
                                nullable = "NULL" if col['nullable'] == 'Y' else "NOT NULL"
                                color = "gray" if nullable == "NULL" else "black"
                                st.markdown(f"- *{col['column_name']}* ({col['data_type']}, <span style='color:{color}'>{nullable}</span>)", unsafe_allow_html=True)
                                schema_text += f"- {col['column_name']} ({col['data_type']}, {nullable})\n"
                    
                    with st.spinner("Generating SQL query..."):
                        sql_query, error = generate_sql(schema_text, nl_query, model_selection)
                    
                    if error:
                        st.error(f"Error generating SQL: {error}")
                    else:
                        st.subheader("🔍 Generated SQL Query:")
                        with st.expander("View Full SQL Query", expanded=True):
                            st.code(sql_query, language="sql")
                        
                        # Copy button
                        st.button("📋 Copy SQL to Clipboard", type="primary", help="Copy the generated SQL query to clipboard")
                        
                        # Add execution button for future feature
                        # st.button("▶ Execute Query", help="Execute this SQL query and see results")
                
                except Exception as e:
                    st.error(f"Error processing schema: {str(e)}")
                    logger.error(f"Error processing schema: {str(e)}")

# Footer with helpful tips
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    - Be specific with your query (e.g., "Show all customers who placed orders in March 2024" instead of "Show customer orders")
    - Include filtering criteria when needed (e.g., "Find products with price greater than $100")
    - Mention specific columns you want to see (e.g., "Show customer name, email, and total purchases")
    - For complex queries, break them down into simpler parts
    """)

st.markdown("---")
st.caption("Natural Language to SQL Generator powered by Mistral AI and Sentence Transformers")

# Auto-refresh timer component
if 'last_checked' in st.session_state:
    seconds_since_last_check = (datetime.now() - st.session_state.last_checked).seconds
    if seconds_since_last_check >= 60:  # Check every minute instead of every second
        st.session_state.last_checked = datetime.now()
        if should_fetch_schema():
            st.experimental_rerun()
