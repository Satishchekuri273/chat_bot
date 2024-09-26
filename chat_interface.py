import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
#from langchain_core.pydantic_v1 import BaseModel, Field, validator
from pydantic import BaseModel, Field, validator
from typing import Literal, List
import psycopg2
from urllib.parse import urlparse
from psycopg2 import sql
import os

# Initialize the ChatGroq model with specific parameters
chat_model = ChatGroq(temperature=0, groq_api_key="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", model_name="llama3-groq-70b-8192-tool-use-preview")

# Define chat history session state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []  # Initialize chat history

# Streamlit App Title
st.title('Chat with us')

#def connect():
    #url = os.getenv("DATABASE_URL")

def connect():
    url = os.getenv("DATABASE_URL")
    parsed_url = urlparse(url)  # Parse the URL
    conn = psycopg2.connect(  # Create a connection object
        database=parsed_url.path[1:],  # Extract the database name
        user=parsed_url.username,  # Extract username
        password=parsed_url.password,  # Extract password
        host=parsed_url.hostname,  # Extract host
        port=parsed_url.port  # Extract port
    )
    return conn  # This returns the connection object

    

def fetch_market_trends(market: str, data_type: str):
    """Fetch the specified data type (e.g., 'Market Trends', 'Market Drivers') for the given market from the PostgreSQL database."""
    try:
        # Establish the database connection
        conn = connect()
        cur = conn.cursor()
        
        # Capitalize the first letter of each word in data_type (title case)
        formatted_data_type = data_type.title()  # 'market trends' becomes 'Market Trends'
        
        # First, try to find an exact match for the market name
        query_exact = sql.SQL("SELECT DISTINCT {data_type} FROM market_data WHERE LOWER(segment) = LOWER(%s)").format(
            data_type=sql.Identifier(formatted_data_type)
        )
        
        cur.execute(query_exact, (market.lower(),))
        exact_results = cur.fetchall()
        
        if exact_results:
            # If exact match found, return the results
            cur.close()
            conn.close()
            return exact_results
        
        # If no exact match, try a partial match with LIMIT 1
        query_partial = sql.SQL("SELECT DISTINCT {data_type} FROM market_data WHERE LOWER(segment) LIKE %s LIMIT 1").format(
            data_type=sql.Identifier(formatted_data_type)
        )
        
        cur.execute(query_partial, (f'%{market.lower()}%',))
        partial_results = cur.fetchall()
        
        # Close the connection
        cur.close()
        conn.close()
        
        # Return partial results or a message if no data is found
        if partial_results:
            return partial_results
        else:
            return f"No data found for {data_type} in market {market}."
    
    except Exception as e:
        return f"Error fetching {data_type} for {market}: {str(e)}"

# Define the data structure for query moderation result
class QueryModerationResult(BaseModel):
    relevance: Literal[0, 1] = Field(description="Output '1' if the query is related to market research or market intelligence, '0' otherwise.")

# Define the data structure for parsing market name, geography, and data type
class MarketQuery(BaseModel):
    market_name: str = Field(description="The name of the market mentioned in the query.")
    geography: str = Field(description="The geography mentioned in the query, which could be a country or region. Defaults to 'global' if not specified.")
    data_type: str = Field(description="The type of market data requested. Options are: 'market size', 'market trends', 'market drivers', 'market restraints', 'competitive landscape'.")

# Define a list of MarketQuery objects to handle multiple results
class MarketQueryList(BaseModel):
    queries: List[MarketQuery] = Field(description="A list of MarketQuery objects representing each unique combination of market name, geography, and data type.")

# Define the data structure for the most relevant market or industry in case of non-relevant queries
class MarketOrIndustry(BaseModel):
    market: str = Field(description="Name of the most relevant market or industry")

# Validator to ensure that data_type is one of the allowed options
@validator("data_type", allow_reuse=True)  # Add allow_reuse=True
def validate_data_type(cls, value):
    allowed_types = {'market size', 'market trends', 'market drivers', 'market restraints', 'competitive landscape'}
    if value not in allowed_types:
        raise ValueError(f"data_type must be one of {allowed_types}")
    return value


def setup_query_moderation_chain() -> ChatPromptTemplate:
    """Sets up the query moderation chain."""
    query_moderation_template = """\
    You are a query moderator for The Business Research Company's flagship database, \
    the Global Market Model (GMM). The GMM provides comprehensive market intelligence, including market size,\
    trends, drivers, restraints, and competitor analysis. Your task is to moderate the user queries sent by GMM users.\
    For the user query {query} check if the query is related to market research or market intelligence.\
    Output the result as a JSON object with the key 'relevance' and the value being '1' if the query is related to market research or market intelligence, and '0' if it is not."""
    
    query_moderation_parser = PydanticOutputParser(pydantic_object=QueryModerationResult)
    
    query_moderation_prompt = PromptTemplate(
        template=query_moderation_template,
        input_variables=["query"],
        partial_variables={"format_instructions": query_moderation_parser.get_format_instructions()},
    )
    
    return query_moderation_prompt | chat_model | query_moderation_parser

def setup_market_query_extraction_chain() -> ChatPromptTemplate:
    """Sets up the market query extraction chain."""
    market_query_parser = PydanticOutputParser(pydantic_object=MarketQueryList)
    
    market_query_extraction_prompt = PromptTemplate(
        template="Extract the market name, geography, and data type from the user query.\n{format_instructions}\nQuery: {query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": market_query_parser.get_format_instructions()},
    )
    
    return market_query_extraction_prompt | chat_model | market_query_parser

def setup_non_relevant_query_chain() -> ChatPromptTemplate:
    """Sets up the chain for handling non-relevant queries."""
    non_relevant_query_template = """\
    You are a query moderator for The Business Research Company's flagship database, the Global Market Model (GMM).
    The GMM provides comprehensive market intelligence, including market size, trends, drivers, restraints, and competitor analysis
    across various industries. For the user query "{query}" - Understand the user's query and analyze its content, then identify the 
    most relevant market or industry that aligns with the context of the query for accurate information retrieval from the GMM database.
    Ensure you output ONLY the market or industry name in the following JSON format:
    {{"market": "name_of_market_or_industry"}}
    Do not provide any additional information or explanations."""
    
    non_relevant_query_parser = PydanticOutputParser(pydantic_object=MarketOrIndustry)
    
    non_relevant_query_prompt = PromptTemplate(
        template=non_relevant_query_template,
        input_variables=["query"],
        partial_variables={"format_instructions": non_relevant_query_parser.get_format_instructions()},
    )
    
    return non_relevant_query_prompt | chat_model | non_relevant_query_parser



def process_user_query(user_query: str):
    """Processes the user query and dynamically outputs relevant information."""
    
    # Setup moderation, extraction, and non-relevant query chains
    moderation_chain = setup_query_moderation_chain()
    extraction_chain = setup_market_query_extraction_chain()
    non_relevant_chain = setup_non_relevant_query_chain()
    
    # Invoke the moderation chain to determine query relevance
    try:
        moderation_result = moderation_chain.invoke(user_query)
    except OutputParserException as e:
        st.session_state['chat_history'].append({'role': 'system', 'message': f"Error during query moderation: {str(e)}"})
        return
    
    if moderation_result.relevance == 1:
        try:
            extracted_data = extraction_chain.invoke({"query": user_query})
            for market_query in extracted_data.queries:
                market_name = market_query.market_name
                geography = market_query.geography
                data_type = market_query.data_type
                
                # Fetch market trends based on the extracted market name
                market_trends = fetch_market_trends(market_name, data_type)
                
                if isinstance(market_trends, list):
                    trends = "\n".join(f"- {trend[0]}" for trend in market_trends)
                    st.session_state['chat_history'].append({
                        'role': 'assistant',
                        'message': f"Market: {market_name}, Geography: {geography}, Data Type: {data_type}\n{trends}"
                    })
                else:
                    st.session_state['chat_history'].append({
                        'role': 'assistant',
                        'message': market_trends
                    })
        except OutputParserException as e:
            st.session_state['chat_history'].append({'role': 'system', 'message': f"Error during market query extraction: {str(e)}"})
    
    else:
        try:
            non_relevant_result = non_relevant_chain.invoke({"query": user_query})
            non_relevant_market = non_relevant_result.market
            # Provide a conversational response to the user's query using ChatGroq
            conversational_template = """\
                You are a friendly assistant. Respond briefly and informatively to the user query: "{query}"."""
            conversational_prompt_template = ChatPromptTemplate.from_template(conversational_template)
            conversational_messages = conversational_prompt_template.format_messages(query=user_query)
    
            try:
                conversational_response = chat_model(conversational_messages)
                st.session_state['chat_history'].append({
                    'role': 'assistant',
                    'message': conversational_response.content
                })
            except Exception as e:
                st.session_state['chat_history'].append({
                    'role': 'system',
                    'message': f"Error generating conversational response: {str(e)}"
                })
            st.session_state['chat_history'].append({
                'role': 'assistant',
                'message': f"The most relevant market or industry: {non_relevant_market}"
            })
            market_trends = fetch_market_trends(non_relevant_market, "Market Trends")
            
            if isinstance(market_trends, list):
                trends = "\n".join(f"- {trend[0]}" for trend in market_trends)
                st.session_state['chat_history'].append({'role': 'assistant', 'message': trends})
            else:
                st.session_state['chat_history'].append({'role': 'assistant', 'message': market_trends})
        
        except OutputParserException as e:
            st.session_state['chat_history'].append({'role': 'system', 'message': f"Error during non-relevant query processing: {str(e)}"})
    
    

# Display chat history
def display_chat():
    """Display chat history in a user-friendly format."""
    for chat in st.session_state['chat_history']:
        if chat['role'] == 'user':
            st.markdown(f"**You:** {chat['message']}")
        elif chat['role'] == 'assistant':
            st.markdown(f"**Assistant:** {chat['message']}")
        else:
            st.markdown(f"**System:** {chat['message']}")


# Input field for user query
user_input = st.text_input("Ask me something:")

# Submit button
if st.button("Submit"):
    if user_input:
        st.session_state['chat_history'].append({'role': 'user', 'message': user_input})
        process_user_query(user_input)

# Display the chat history
display_chat()
