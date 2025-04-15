import streamlit as st
import cx_Oracle
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain_mistralai import ChatMistralAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import logging
# Import pyngrok for tunneling
from pyngrok import ngrok

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env for local development
load_dotenv()

# Create data directory if it doesn't exist
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Helper function to get configuration value from either Streamlit secrets or environment variables
def get_config(key, default=""):
    # First try to get from Streamlit secrets
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    # Then try environment variables
    return os.getenv(key, default)

# Database connection details
DB_CONFIG = {
    "host": get_config("DB_HOST", "localhost"),
    "port": get_config("DB_PORT", "1521"),
    "service_name": get_config("DB_SERVICE_NAME", "ORCLPDB"),
    "user": get_config("DB_USER", ""),
    "password": get_config("DB_PASSWORD", ""),
    "schema": get_config("DB_SCHEMA", "")
}

# Ngrok configuration
NGROK_AUTH_TOKEN = get_config("NGROK_AUTH_TOKEN", "")
USE_NGROK = get_config("USE_NGROK", "False").lower() == "true"
NGROK_TUNNEL_URL = None

# Mistral API key
MISTRAL_API_KEY = get_config("MISTRAL_API_KEY", "")

# File paths - using relative paths for deployment
SCHEMA_FILE = os.path.join(DATA_DIR, "table_details.json")
VECTOR_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
VECTOR_NAMES_FILE = os.path.join(DATA_DIR, "table_names.json")
VECTOR_DESC_FILE = os.path.join(DATA_DIR, "table_descriptions.json")
LAST_FETCH_FILE = os.path.join(DATA_DIR, "last_fetch.txt")

# Available models
MISTRAL_MODELS = ["mistral-large-latest", "mistral-medium", "mistral-small"]


def setup_ngrok():
    """
    Set up ngrok tunnel to make the local database accessible
    """
    global NGROK_TUNNEL_URL

    if not USE_NGROK:
        logger.info("Ngrok is disabled, skipping tunnel setup")
        return None

    try:
        # Check if tunnel is already established
        if NGROK_TUNNEL_URL:
            logger.info(f"Using existing ngrok tunnel: {NGROK_TUNNEL_URL}")
            return NGROK_TUNNEL_URL

        # Set auth token if provided
        if NGROK_AUTH_TOKEN:
            logger.info("Setting ngrok auth token")
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        else:
            logger.warning("No ngrok auth token provided. Tunnels may expire after 2 hours.")

        # Get the local database port
        db_port = int(DB_CONFIG["port"])

        # Create a TCP tunnel for the database port
        logger.info(f"Creating ngrok TCP tunnel for local port {db_port}")
        tunnel = ngrok.connect(addr=db_port, proto="tcp")
        public_url = tunnel.public_url
        
        logger.info(f"Ngrok tunnel established: {public_url}")
        
        # Parse the public URL to get host and port
        # Format is typically tcp://X.tcp.ngrok.io:XXXXX
        ngrok_host = public_url.split("//")[1].split(":")[0]
        ngrok_port = int(public_url.split(":")[-1])
        
        # Store original values before updating
        DB_CONFIG["original_host"] = DB_CONFIG["host"]
        DB_CONFIG["original_port"] = DB_CONFIG["port"]
        
        # Update DB_CONFIG to use ngrok endpoint
        DB_CONFIG["host"] = ngrok_host
        DB_CONFIG["port"] = str(ngrok_port)
        
        # Store the tunnel URL for reference
        NGROK_TUNNEL_URL = public_url
        
        return public_url
    except Exception as e:
        logger.error(f"Error setting up ngrok tunnel: {str(e)}")
        return None


class SchemaVectorSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.table_data = []
        self.table_names = []
    
    def fetch_schema(self):
        # Validate DB configuration
        if not all([DB_CONFIG["user"], DB_CONFIG["password"], DB_CONFIG["schema"]]):
            return None, "Database configuration incomplete. Please check your secrets or environment variables."
        
        try:
            # Set up ngrok tunnel if enabled
            if USE_NGROK:
                tunnel_url = setup_ngrok()
                if not tunnel_url:
                    return None, "Failed to establish ngrok tunnel. Please check logs for details."
            
            # Try to create dsn
            logger.info(f"Connecting to database at {DB_CONFIG['host']}:{DB_CONFIG['port']}")
            
            dsn_tns = cx_Oracle.makedsn(
                DB_CONFIG["host"], 
                DB_CONFIG["port"], 
                service_name=DB_CONFIG["service_name"]
            )
            
            # Try to establish connection
            logger.info(f"Establishing connection to database as user {DB_CONFIG['user']}")
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
            
            logger.info("Executing schema query")
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
            
            # Save schema to file
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
    
    def search(self, query, top_k=5):
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


def generate_sql(schema_text, nl_query, model="mistral-large-latest"):
    # Validate Mistral API key
    if not MISTRAL_API_KEY:
        return None, "Mistral API key not found. Please check your secrets or environment variables."
    
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
        logger.info(f"Generating SQL using {model} model")
        llm = ChatMistralAI(model=model, api_key=MISTRAL_API_KEY)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        sql_query = chain.run({"schema": schema_text, "query": nl_query})
        
        # Remove any existing formatting if present
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        logger.info(f"SQL query generated successfully using {model}")
        return sql_query, None
    except Exception as e:
        error_msg = f"Error generating SQL: {str(e)}"
        logger.error(error_msg)
        return None, error_msg


def try_cx_oracle_init():
    """
    Try to initialize cx_Oracle if we're in Streamlit Cloud environment
    with a library path configured
    """
    try:
        lib_dir = get_config("LD_LIBRARY_PATH", "")
        if lib_dir and os.path.exists(lib_dir):
            logger.info(f"Initializing cx_Oracle with lib_dir: {lib_dir}")
            cx_Oracle.init_oracle_client(lib_dir=lib_dir)
            logger.info("cx_Oracle initialized successfully")
        elif lib_dir:
            logger.warning(f"LD_LIBRARY_PATH ({lib_dir}) not found")
    except Exception as e:
        logger.warning(f"Failed to initialize cx_Oracle: {str(e)}")


def main():
    # Try to initialize cx_Oracle if needed
    try_cx_oracle_init()
    
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
        .small-text {
            font-size: 0.8em;
            color: #666;
        }
        .highlight {
            background-color: #f0f8ff;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #4361ee;
        }
        </style>
        """, unsafe_allow_html=True)

    # App header
    st.title("🔍 Natural Language to SQL Generator")
    st.markdown("Convert your natural language questions into SQL queries using AI")

    # Initialize session state
    if "db_connected" not in st.session_state:
        st.session_state.db_connected = os.path.exists(SCHEMA_FILE)
    
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

    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Connection status
        st.subheader("Database Connection")
        
        # Environment variable status
        db_config_ok = all([
            DB_CONFIG["host"] or USE_NGROK, 
            DB_CONFIG["port"] or USE_NGROK, 
            DB_CONFIG["service_name"],
            DB_CONFIG["user"], 
            DB_CONFIG["password"], 
            DB_CONFIG["schema"]
        ])
        
        # Ngrok configuration section
        st.subheader("Ngrok Configuration")
        use_ngrok = st.checkbox("Use Ngrok for Local DB", value=USE_NGROK, 
                              help="Enable ngrok to access your local database remotely")
        
        if use_ngrok != USE_NGROK:
            # Store the new value
            os.environ["USE_NGROK"] = str(use_ngrok)
            # Force refresh to apply change
            st.experimental_rerun()
        
        ngrok_token = st.text_input(
            "Ngrok Auth Token",
            value=NGROK_AUTH_TOKEN,
            type="password",
            help="Your ngrok authentication token"
        )
        
        if ngrok_token and ngrok_token != NGROK_AUTH_TOKEN:
            os.environ["NGROK_AUTH_TOKEN"] = ngrok_token
            st.experimental_rerun()
        
        if use_ngrok:
            if not ngrok_token:
                st.warning("⚠️ Ngrok Auth Token not set. Tunnels will expire after 2 hours.")
            
            # Show ngrok tunnel info if available
            if NGROK_TUNNEL_URL:
                st.success(f"✅ Ngrok tunnel active")
                with st.expander("Tunnel Details"):
                    st.code(f"Host: {DB_CONFIG['host']}\nPort: {DB_CONFIG['port']}")
            else:
                # Try to establish tunnel
                with st.spinner("Setting up ngrok tunnel..."):
                    tunnel_url = setup_ngrok()
                if tunnel_url:
                    st.success(f"✅ Ngrok tunnel established")
                else:
                    st.error("❌ Failed to establish ngrok tunnel")
        
        if not db_config_ok:
            st.warning("⚠️ Database configuration incomplete")
            
            with st.expander("Configuration Instructions"):
                st.markdown("""
                For deployment, add these settings to your environment:
                
                **Required settings:**
                - `DB_USER` - Database username
                - `DB_PASSWORD` - Database password
                - `DB_SCHEMA` - Schema name
                - `DB_SERVICE_NAME` - Oracle service name
                - `MISTRAL_API_KEY` - Mistral API key
                
                **For local database access:**
                - `USE_NGROK` - Set to "true"
                - `NGROK_AUTH_TOKEN` - Your ngrok auth token
                
                **For direct database access:**
                - `DB_HOST` - Database host
                - `DB_PORT` - Database port
                """)
        
        if st.session_state.db_connected:
            st.success("✅ Schema loaded")
            if "last_updated" in st.session_state:
                st.info(f"Last updated: {st.session_state.last_updated}")
        else:
            st.warning("⚠ No schema loaded")
        
        # API key status
        if not MISTRAL_API_KEY:
            st.warning("⚠️ Mistral API key not set")
        else:
            st.success("✅ Mistral API key configured")
        
        # Model selection
        st.subheader("AI Model Settings")
        model_selection = st.selectbox(
            "Select Mistral Model:",
            MISTRAL_MODELS,
            index=0,
        )
        
        # Query history
        st.subheader("Query History")
        if not st.session_state.query_history:
            st.info("No queries yet")
        else:
            for i, query in enumerate(st.session_state.query_history[-5:]):  # Show last 5 queries
                if st.button(f"📝 {query[:30]}...", key=f"hist_{i}"):
                    st.session_state.current_query = query
                    # Using experimental_rerun because we've updated session state
                    st.experimental_rerun()

    # Button to fetch and update schema
    col1, col2 = st.columns([1, 3])
    with col1:
        schema_button = st.button(
            "🔄 Update Schema", 
            help="Fetch latest database schema",
            disabled=not db_config_ok
        )
        
        if schema_button:
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
            # Save current query for history
            current_query = nl_query
            
            # Add to history if not already there
            if current_query not in st.session_state.query_history:
                st.session_state.query_history.append(current_query)
                # Keep only the last 10 queries
                if len(st.session_state.query_history) > 10:
                    st.session_state.query_history.pop(0)
            
            if not vector_db_loaded and not st.session_state.db_connected:
                st.error("⚠ No database schema available. Please click 'Update Schema' first.")
            else:
                with st.spinner("Searching for relevant tables..."):
                    results = schema_search.search(current_query)
                
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
                            sql_query, error = generate_sql(schema_text, current_query, model_selection)
                        
                        if error:
                            st.error(f"Error generating SQL: {error}")
                        else:
                            st.subheader("🔍 Generated SQL Query:")
                            with st.expander("View Full SQL Query", expanded=True):
                                st.code(sql_query, language="sql")
                                
                                # We can't use clipboard directly in Streamlit Cloud
                                # but we can provide a text area for easy copying
                                st.text_area(
                                    "Copy SQL Query:",
                                    value=sql_query,
                                    height=100,
                                    help="Copy this SQL query to clipboard"
                                )
                    
                    except Exception as e:
                        st.error(f"Error processing schema: {str(e)}")
                        logger.error(f"Error processing schema: {str(e)}")

    # Ngrok status notification
    if USE_NGROK and NGROK_TUNNEL_URL:
        st.markdown("---")
        st.markdown(
            f"<div class='highlight'>⚠️ <strong>Important:</strong> Your local database is being accessed through an ngrok tunnel. "
            f"If you're using the free tier, this tunnel will expire after 2 hours.</div>", 
            unsafe_allow_html=True
        )

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


if __name__ == "__main__":
    main()