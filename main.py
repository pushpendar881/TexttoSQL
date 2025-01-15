import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
import cx_Oracle
import pandas as pd

# Set up API token and model
MISTRAL_API_TOKEN = "xqMCVQSjhr4cnjhrZF9WmChu9qXLlT6N"
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-large-latest", api_key=MISTRAL_API_TOKEN)

def connect_to_db():
    """Create and return a database connection."""
    try:
        dsn = cx_Oracle.makedsn("localhost", 1521, service_name="orcl")
        connection = cx_Oracle.connect(
            user="sys",
            password="5669",
            dsn=dsn,
            mode=cx_Oracle.SYSDBA
        )
        return connection
    except cx_Oracle.Error as e:
        st.error(f"Database connection error: {str(e)}")
        return None

def get_schema_ddl(connection, schema_name):
    """Get DDL statements for all tables in the schema."""
    ddl_statements = {}
    try:
        cursor = connection.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM all_tables 
            WHERE owner = :schema_name
        """, [schema_name.upper()])
        
        for table_name, in cursor.fetchall():
            cursor.execute("""
                SELECT DBMS_METADATA.GET_DDL('TABLE', :table_name, :schema_name) 
                FROM dual
            """, [table_name, schema_name.upper()])
            ddl_result = cursor.fetchone()
            if ddl_result:
                ddl_statements[table_name] = ddl_result[0].read()
        
        cursor.close()
        return ddl_statements
    
    except cx_Oracle.Error as e:
        st.error(f"Error fetching schema DDL: {str(e)}")
        return {}

def create_sql_chain(schema, question):
    """Create a LangChain chain for SQL query generation."""
    template = """
    You are a helpful SQL expert. Based on the table schema below, write a SQL query 
    that answers the user's question.
    
    Schema:
    {schema}
    
    Write only the SQL query that answers the question. Do not include any explanations.
    
    Question: {question}
    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        RunnablePassthrough.assign(schema=lambda x: schema, question=lambda x: question)
        | prompt
        | model
        | StrOutputParser()
    )
    return chain

def generate_sql_query(schema_name, question):
    connection = connect_to_db()
    if connection:
        ddl_statements = get_schema_ddl(connection, schema_name)
        schema = "\n".join(ddl_statements.values())
        
        if schema:
            chain = create_sql_chain(schema, question)
            input_data = {"question": question, "schema": schema}
            sql_query = chain.invoke(input_data)
            
            if sql_query:
                sql_query_clean = sql_query.strip("```").strip("sql").strip().strip(";")
                try:
                    cursor = connection.cursor()
                    cursor.execute(sql_query_clean)
                    result = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    cursor.close()
                    connection.close()

                    # Convert result to DataFrame
                    if result:
                        df_result = pd.DataFrame(result, columns=columns)
                    else:
                        df_result = pd.DataFrame()  # In case of no result

                    return sql_query_clean, df_result
                except cx_Oracle.Error as e:
                    st.error(f"Error executing SQL query: {str(e)}")
                    connection.close()
                    return sql_query_clean, None
            else:
                st.error("Failed to generate SQL query.")
                connection.close()
                return None, None
        else:
            st.error("No tables found in the specified schema.")
            connection.close()
            return None, None
    else:
        st.error("Failed to connect to the database.")
        return None, None

# Streamlit interface
st.title("SQL Query Generator")

schema_name = st.text_input("Enter Schema Name", value="C##newschema")
question = st.text_input("Enter your question", value="How can I find the name of the teacher and course ID?")

if st.button("Generate SQL Query"):
    sql_query, result = generate_sql_query(schema_name, question)
    if sql_query:
        st.text("Generated SQL Query:")
        st.text("/////////////////////////////////////////////////////////////////////////////////")
        st.code(sql_query)
        st.text("/////////////////////////////////////////////////////////////////////////////////")
        if result is not None:
            st.text("Query Result:")
            st.dataframe(result)  # Display the DataFrame without needing columns argument
        else:
            st.error("No result returned from the query.")