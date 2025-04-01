import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
import os
import json
from io import BytesIO
import traceback
import uuid
from typing import List, Dict, Any, TypedDict, Literal
import dotenv

# For LangChain and LangGraph imports
try:
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langgraph.graph import StateGraph, START, END
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    LANGGRAPH_AVAILABLE = False

# Load environment variables from .env file if it exists
dotenv.load_dotenv()

# Set page config at the very beginning
st.set_page_config(
    page_title="Snowflake RAG Assistant",
    page_icon="❄️",
    layout="wide",
)

# Create a basic secrets management class for API keys
class ApiKeyManager:
    def __init__(self):
        self.api_key = None
        
    def get_api_key(self):
        # Try environment variables first
        api_key = os.environ.get("OPENAI_API_KEY", "")
        
        # If we have it, return it
        if api_key:
            self.api_key = api_key
            return api_key
        
        # Otherwise, return whatever we have stored (could be None)
        return self.api_key
    
    def set_api_key(self, key):
        self.api_key = key
        os.environ["OPENAI_API_KEY"] = key
        return key

# Create the API key manager
api_key_manager = ApiKeyManager()

# Create a container for notifications
if "notifications" not in st.session_state:
    st.session_state.notifications = []

# Function to add a notification
def add_notification(message, type="success"):
    st.session_state.notifications.append({"message": message, "type": type})

# Configure Snowflake connection
def configure_snowflake_connection():
    """Configure and return a Snowflake session."""
    try:
        # Try to get connection parameters from environment variables or UI inputs
        with st.sidebar:
            # If we already have a session in st.session_state, don't ask again
            if 'snowflake_session' in st.session_state:
                return st.session_state.snowflake_session
            
            st.subheader("Snowflake Connection")
            
            # Get connection parameters from environment variables first
            account = os.environ.get("SNOWFLAKE_ACCOUNT", "")
            user = os.environ.get("SNOWFLAKE_USER", "")
            password = os.environ.get("SNOWFLAKE_PASSWORD", "")
            warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", "")
            database = os.environ.get("SNOWFLAKE_DATABASE", "")
            schema = os.environ.get("SNOWFLAKE_SCHEMA", "")
            
            # Get missing values from user input if needed
            account = st.text_input("Snowflake Account", value=account)
            user = st.text_input("Username", value=user)
            password = st.text_input("Password", type="password", value=password)
            warehouse = st.text_input("Warehouse", value=warehouse)
            database = st.text_input("Database", value=database)
            schema = st.text_input("Schema", value=schema)
            
            # Connect button
            if st.button("Connect to Snowflake"):
                if not (account and user and password and warehouse and database and schema):
                    add_notification("All fields are required", "error")
                    return None
                
                try:
                    # Create the Snowflake session
                    session = Session.builder.configs({
                        "account": account,
                        "user": user,
                        "password": password,
                        "warehouse": warehouse,
                        "database": database,
                        "schema": schema
                    }).create()
                    
                    # Test the connection
                    test_query = session.sql("SELECT 1").collect()
                    
                    # Store in session state
                    st.session_state.snowflake_session = session
                    add_notification("Connected to Snowflake successfully! ✅")
                    return session
                except Exception as e:
                    add_notification(f"Failed to connect to Snowflake: {str(e)}", "error")
                    return None
            
            return None  # Return None if the user hasn't clicked connect yet
            
    except Exception as e:
        add_notification(f"Error configuring Snowflake: {str(e)}", "error")
        return None

# Function to get available tables
def get_available_tables(session):
    if session is None:
        return []
    try:
        tables_df = session.sql("SHOW TABLES").collect()
        return tables_df
    except Exception as e:
        st.error(f"Error loading tables: {str(e)}")
        return []

# Function to get table schema
def get_table_schema(session, table_name):
    if session is None or not table_name:
        return []
    try:
        schema_df = session.sql(f"DESCRIBE TABLE {table_name}").collect()
        return schema_df
    except Exception as e:
        st.error(f"Error getting table schema: {str(e)}")
        return []

# Setup LLM based on API key
def setup_llm():
    """Setup and return an LLM based on the API key."""
    # Check if LangChain and LangGraph are available
    if not LANGGRAPH_AVAILABLE:
        add_notification("Required libraries (LangChain/LangGraph) are not available. Please install them.", "error")
        return None
    
    # Get API key
    api_key = api_key_manager.get_api_key()
    
    if not api_key:
        # No API key found, we'll need to get one from the user
        with st.sidebar:
            st.subheader("OpenAI API Key Settings")
            api_key_input = st.text_input("OpenAI API Key:", type="password", key="api_key_input")
            if api_key_input:
                api_key = api_key_manager.set_api_key(api_key_input)
                add_notification("API key set successfully! ✅")
    
    if api_key:
        # Configure the LLM
        try:
            llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=api_key)
            add_notification("OpenAI LLM configured ✅")
            return llm
        except Exception as e:
            add_notification(f"Error configuring LLM: {str(e)}", "error")
            return None
    
    return None

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_results" not in st.session_state:
    st.session_state.current_results = None

# Get table info for the selected table and cache it
@st.cache_data(ttl=300, hash_funcs={"snowflake.snowpark.session.Session": lambda _: None})
def get_table_info_for_llm(_session, table_name):
    try:
        # Get schema
        schema = get_table_schema(_session, table_name)
        table_info = f"Table: {table_name}\nColumns:\n"
        for row in schema:
            table_info += f"- {row['name']}: {row['type']}\n"
            
        # Get sample data
        sample_data = _session.sql(f"SELECT * FROM {table_name} LIMIT 3").collect()
        if sample_data:
            table_info += "\nSample data:\n"
            for row in sample_data:
                table_info += f"{row}\n"
                
        return table_info
    except Exception as e:
        st.error(f"Error getting table info: {str(e)}")
        return f"Table: {table_name} (error loading schema)"

# Implement safe query execution
def execute_query(session, sql_query):
    try:
        # Add validation before execution
        if "DROP" in sql_query.upper() or "DELETE" in sql_query.upper() or "UPDATE" in sql_query.upper():
            return None, "For safety reasons, DROP, DELETE, and UPDATE operations are not allowed"
        
        # Execute the query
        result = session.sql(sql_query).collect()
        
        # Check result size
        if len(result) > 10000:
            return result[:10000], "Warning: Result set truncated to 10,000 rows due to size limitations"
        
        return result, None
    except Exception as e:
        return None, f"Error executing query: {str(e)}"

# Export functions
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# Helper function to ensure objects are serializable
def ensure_serializable(obj):
    """Convert any non-serializable objects to serializable form"""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    return obj

# ========== AGENT IMPLEMENTATION ==========

# Define router node for the LangGraph
def router(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Router node that decides whether to use SQL or help based on the query.
    """
    llm = state.get("llm")
    if not llm:
        return {"next": "help_node"}
    
    messages = state["messages"]
    user_message = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
    
    router_prompt = f"""
    You are a router for a data query system. Your task is to decide whether a user question 
    should be answered using:
    
    1. SQL (if the question is about specific data in tables, like "how many", "find all", etc.)
    2. HELP (if the question is about how to use the system or other general help)
    
    Here's the user question: "{user_message}"
    
    Return ONLY one of these exactly: "SQL" or "HELP"
    """
    
    try:
        messages = [SystemMessage(content="You are a helpful assistant."), 
                   HumanMessage(content=router_prompt)]
        response = llm.invoke(messages).content
            
        # Extract just the decision
        response = response.strip().upper()
        if "SQL" in response:
            return {"next": "sql_node"}
        else:
            return {"next": "help_node"}
            
    except Exception as e:
        st.error(f"Error in router: {str(e)}")
        return {"next": "help_node"}  # Default to help on error

# Define SQL node for the LangGraph
def sql_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that generates and executes SQL based on user query.
    """
    llm = state.get("llm")
    session = state.get("session")
    selected_table = state.get("selected_table")
    
    if not llm or not session:
        return {
            "messages": state["messages"] + [AIMessage(content="Error: LLM or database session not available.")]
        }
    
    messages = state["messages"]
    user_message = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
    
    if not selected_table:
        return {
            "messages": messages + [AIMessage(content="Please select a table first before asking a data question.")]
        }
    
    # Get table info
    table_info = get_table_info_for_llm(_session=session, table_name=selected_table)
    
    # Create prompt for SQL generation
    sql_prompt = f"""
    Based on the table information below, write a SQL query that answers the user's question.
    
    {table_info}
    
    The SQL should be written using Snowflake SQL syntax.
    
    User question: {user_message}
    
    SQL Query (just return the SQL query, no other text):
    """
    
    try:
        # Generate SQL using LLM
        prompt_messages = [SystemMessage(content="You are a helpful SQL assistant."), 
                          HumanMessage(content=sql_prompt)]
        sql_query = llm.invoke(prompt_messages).content
            
        # Clean up query - extract just the SQL
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].strip()
            
        # Execute SQL
        results, error = execute_query(session, sql_query)
        
        if error:
            response = f"Error executing query: {error}\n\nGenerated SQL: {sql_query}"
        else:
            # Convert results to DataFrame for easier handling
            results_df = pd.DataFrame(results)
            st.session_state.current_results = results_df
            
            # Format results for display
            result_str = "Query results:\n"
            if not results_df.empty:
                result_str += str(results_df.head(10))
                if len(results_df) > 10:
                    result_str += f"\n... and {len(results_df) - 10} more rows"
            else:
                result_str += "No results found"
                
            # Create summary of results
            summary_prompt = f"""
            I executed the following SQL query:
            
            ```sql
            {sql_query}
            ```
            
            And got these results:
            
            {result_str}
            
            Can you provide a clear, concise summary of what this data tells us in response to the original question: "{user_message}"?
            The summary should be 1-3 short paragraphs.
            """
            
            summary_messages = [SystemMessage(content="You are a data analyst."), 
                               HumanMessage(content=summary_prompt)]
            summary = llm.invoke(summary_messages).content
                
            response = f"Tool Used: SQL Query\n\nQuery:\n```sql\n{sql_query}\n```\n\n{summary}"
            
            # Store in session state
            state["sql_query"] = sql_query
            state["results"] = ensure_serializable(results)
            
    except Exception as e:
        response = f"Error generating or executing SQL: {str(e)}"
    
    return {
        "messages": messages + [AIMessage(content=response)]
    }

# Define help node for the LangGraph
def help_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that provides help information based on user query.
    """
    llm = state.get("llm")
    if not llm:
        return {
            "messages": state["messages"] + [AIMessage(content="Error: LLM not available.")]
        }
    
    messages = state["messages"]
    user_message = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
    
    help_prompt = f"""
    The user has asked: "{user_message}"
    
    You are a helpful assistant for a Snowflake natural language query system.
    If the user is asking about how to use the system, provide clear instructions.
    If they're asking about available tables, mention they can select tables from the sidebar.
    
    Keep your response concise, helpful, and focused on helping them use the system effectively.
    """
    
    try:
        help_messages = [SystemMessage(content="You are a helpful assistant."), 
                        HumanMessage(content=help_prompt)]
        response = llm.invoke(help_messages).content
            
        response = f"Tool Used: Help Assistant\n\n{response}"
    except Exception as e:
        response = f"I encountered an error while trying to help: {str(e)}. Please try rephrasing your question or asking something else."
    
    return {
        "messages": messages + [AIMessage(content=response)]
    }

# ======== LANGGRAPH IMPLEMENTATION =========

def initialize_langgraph(llm):
    """Initialize the LangGraph agent with the given LLM."""
    if not llm:
        add_notification("Cannot initialize LangGraph without an LLM", "error")
        return None
    
    try:
        # Define the graph state
        class GraphState(TypedDict):
            messages: List
            next: str
            session: Any
            selected_table: str
            llm: Any

        # Build the graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("router", router)
        workflow.add_node("sql_node", sql_node)
        workflow.add_node("help_node", help_node)
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "router",
            lambda x: x["next"]
        )
        
        # Connect the starting node
        workflow.add_edge(START, "router")
        
        # Connect ending nodes back to END
        workflow.add_edge("sql_node", END)
        workflow.add_edge("help_node", END)
        
        # Compile the graph
        agentic_app = workflow.compile()
        
        return agentic_app
    except Exception as e:
        add_notification(f"Failed to initialize LangGraph: {str(e)}", "error")
        return None

# Main application logic
def main():
    """Main application function."""
    # App title
    st.title("❄️ Snowflake RAG Assistant")
    st.write("Ask questions about your Snowflake data using natural language")
    
    # Add app info to sidebar
    with st.sidebar:
        st.title("❄️ Snowflake RAG Assistant")
        
        # About section
        with st.expander("About this app", expanded=True):
            st.markdown("""
            ### What is this app?
            
            This application allows you to query your Snowflake database using natural language. 
            It converts your questions into SQL, executes them, and returns the results in an 
            easy-to-understand format.
            
            ### Features:
            
            - **Natural Language to SQL**: Ask questions in plain English
            - **Interactive Chat**: Have a conversation about your data
            - **Export Results**: Download query results as CSV or Excel
            - **Powered by LangGraph**: Advanced AI agent technology for accurate results
            
            ### How to use:
            
            1. Connect to your Snowflake database
            2. Select a table to query
            3. Ask questions about your data in the chat
            4. Export results if needed
            """)
    
    # Display any notifications in a dedicated area at the top
    notification_container = st.container()
    
    # Check LangGraph status and add appropriate notification
    if LANGGRAPH_AVAILABLE:
        add_notification("LangGraph successfully loaded ✅")
    else:
        add_notification("LangGraph not available. Some features may be limited.", "warning")
    
    # Initialize session connections
    session = configure_snowflake_connection()
    llm = setup_llm()
    
    # Display notifications after connections are attempted
    with notification_container:
        for notification in st.session_state.notifications:
            if notification["type"] == "success":
                st.success(notification["message"])
            elif notification["type"] == "error":
                st.error(notification["message"])
            elif notification["type"] == "warning":
                st.warning(notification["message"])
            else:
                st.info(notification["message"])
        
        # Clear notifications after displaying them
        st.session_state.notifications = []
    
    # Check if we have everything we need
    if not session:
        st.warning("Please connect to Snowflake using the sidebar.")
        if 'snowflake_session' not in st.session_state:
            return  # Exit early
    else:
        session = st.session_state.snowflake_session  # Use the stored session
    
    if not llm:
        st.warning("Please provide an OpenAI API key in the sidebar.")
        return  # Exit early
    
    # Table selection in sidebar
    selected_table = None
    with st.sidebar:
        st.header("Table Selection")
        
        # Get available tables
        tables = get_available_tables(session)
        if tables:
            table_names = [row["name"] for row in tables]
            selected_table = st.selectbox("Select a table to query", table_names)
            
            if selected_table:
                st.write(f"Schema for {selected_table}:")
                schema = get_table_schema(session, selected_table)
                st.dataframe(schema)
        else:
            st.info("No tables available or not connected to Snowflake yet.")
    
    # Initialize or retrieve LangGraph agent
    if 'agentic_app' not in st.session_state and llm:
        st.session_state.agentic_app = initialize_langgraph(llm)
        if st.session_state.agentic_app:
            add_notification("Agentic workflow initialized successfully ✅")
    
    agentic_app = st.session_state.get('agentic_app')
    
    # Initialize chat history if needed
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add a welcome message
        welcome_msg = "I'm an AI assistant that can help you query your Snowflake data using natural language. What would you like to know about your data?"
        st.session_state.messages.append(AIMessage(content=welcome_msg))
    
    # Display chat history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process with LangGraph and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if agentic_app and selected_table:
                    try:
                        # Prepare the agent state
                        agent_state = {
                            "messages": st.session_state.messages,
                            "session": session,
                            "selected_table": selected_table,
                            "llm": llm
                        }
                        
                        # Execute the agent
                        result = agentic_app.invoke(agent_state)
                        
                        # Get the response
                        ai_message = result["messages"][-1]
                        response_text = ai_message.content
                        
                        # Display the response
                        st.markdown(response_text)
                        
                        # Add to chat history
                        st.session_state.messages.append(ai_message)
                        
                        # Display results if available
                        if "current_results" in st.session_state and st.session_state.current_results is not None:
                            results_df = st.session_state.current_results
                            
                            st.subheader("Data Results")
                            st.dataframe(results_df)
                            
                            # Add export buttons
                            col1, col2 = st.columns(2)
                            with col1:
                                st.download_button(
                                    label="Download CSV",
                                    data=convert_df_to_csv(results_df),
                                    file_name=f"query_results_{selected_table}.csv",
                                    mime="text/csv"
                                )
                            
                            with col2:
                                st.download_button(
                                    label="Download Excel",
                                    data=convert_df_to_excel(results_df),
                                    file_name=f"query_results_{selected_table}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        st.error(error_msg)
                        add_notification(f"Query error: {str(e)}", "error")
                        st.session_state.messages.append(AIMessage(content=f"I encountered an error: {str(e)}"))
                else:
                    if not selected_table:
                        msg = "Please select a table from the sidebar first."
                    else:
                        msg = "LangGraph agent not initialized. Please check your API key and try again."
                    
                    st.warning(msg)
                    st.session_state.messages.append(AIMessage(content=msg))

if __name__ == "__main__":
    main() 