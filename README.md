# Snowflake Chat (RAG Agent)

A local Streamlit application that allows you to query Snowflake data using natural language with the power of LangGraph agents.

[Click for streamlit live demo](https://snow-rag-chat.streamlit.app/) - Requires DB credentials and API key

## Features

- Natural language to SQL translation
- Interactive chat interface
- Query results in tabular format
- CSV and Excel export functionality
- LangGraph agent-based processing

## Setup Instructions

### Prerequisites

- Python 3.8+
- Snowflake account with appropriate credentials
- OpenAI API key

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd snowflake-rag
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory with your credentials:
   ```
   # OpenAI API Key
   OPENAI_API_KEY=your_openai_api_key

   # Snowflake Connection (optional, can also be entered in the UI)
   SNOWFLAKE_ACCOUNT=your_account
   SNOWFLAKE_USER=your_username
   SNOWFLAKE_PASSWORD=your_password
   SNOWFLAKE_WAREHOUSE=your_warehouse
   SNOWFLAKE_DATABASE=your_database
   SNOWFLAKE_SCHEMA=your_schema
   ```

### Running the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

2. Open your browser and navigate to http://localhost:8501

3. If you didn't set up Snowflake credentials in the `.env` file, you'll need to enter them in the sidebar.

4. Enter your OpenAI API key in the sidebar if not already provided in the `.env` file.

5. Select a table from the dropdown and start asking questions!

## Usage

1. Connect to your Snowflake database using the connection form in the sidebar
2. Select a table to query from the dropdown menu
3. Enter a natural language question in the chat input, such as:
   - "How many rows are in this table?"
   - "What are the top 5 entries sorted by date?"
   - "Show me records where the value is greater than 100"
4. View the results in the table view
5. Download results as CSV or Excel using the provided buttons

## Technology Stack

- Streamlit - UI framework
- LangChain - LLM framework
- LangGraph - Agent orchestration
- Snowflake Snowpark - Database connectivity
- OpenAI GPT - Natural language processing

## Examples of Natural Language Queries

- "How many records are in this table?"
- "What are the top 5 customers by order value?"
- "Show me sales data from last month"
- "What is the average transaction amount by product category?"
- "Find all records where the status is 'pending'"

## Security Considerations

This application includes several security features:
- Prevention of DROP, DELETE, and UPDATE operations
- Result set size limitations
- Input validation
- Secure credential handling

## Troubleshooting

- If you encounter connection issues, ensure your Snowflake credentials are correctly configured
- For SQL generation errors, try rephrasing your question to be more specific
- If export fails, check if your result set is within the size limitations

## Development

This application is built with:
- Streamlit
- LangChain
- Snowflake Snowpark Python
- LangGraph
- OpenAI's language models (GPT-4o-mini by default) 
