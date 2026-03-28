import os
import csv
import streamlit as st
from typing_extensions import TypedDict
from typing import Annotated
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq LLM - Try .env file first
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY not found in .env file. Please create a .env file with: GROQ_API_KEY=your_key_here")
    st.stop()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=groq_api_key)

# Defining Graph State 
class GraphState(TypedDict):
    ticket_text: str
    category: Annotated[str, "Predicted issue type"]
    summary: Annotated[str, "Ticket summary"]
    department: Annotated[str, "Department assigned"]

# Define Nodes
# Node 1: Ticket Classification
def ticket_classification(state: GraphState):
    ticket_text = state["ticket_text"]
    prompt = f"""
    Classify the following customer ticket into one of the categories:
    [Product Issue, Billing Issue, Appointment Scheduling, Login Issue, Reminder Problem, 
     Data Sync Error, Report Generation, General Inquiry].
    
    Ticket: {ticket_text}
    Only return the category name.
    """
    category = llm.invoke(prompt).content.strip()
    return {"ticket_text": ticket_text, "category": category, "summary": "", "department": ""}

# Node 2: Ticket Summarizer
def ticket_summarizer(state: GraphState):
    text = state["ticket_text"]
    prompt = f"Summarize this healthcare support ticket in less than 50 words:\n{text}"
    summary = llm.invoke(prompt).content.strip()
    return {"ticket_text": text, "category": state["category"], "summary": summary, "department": ""}

# Node 3: Department Routing 
def ticket_router(state: GraphState):
    category = state["category"]
    prompt = f""" Based on this ticket category: {category},
    route the tickets to most appropriate department who will solve this issue.
    Choose from departments : [Product Engineering, Finance, Customer Service, Technical Support, 
    Integrations Team, Analytics Team].
    
    Only return the department name"""
    department = llm.invoke(prompt).content.strip()
    return {
        "ticket_text": state["ticket_text"], 
        "category": category, 
        "summary": state["summary"],
        "department": department
    }

# Building Workflow
workflow = StateGraph(GraphState)
workflow.add_node("ticket_classification", ticket_classification)
workflow.add_node("ticket_summarizer", ticket_summarizer)
workflow.add_node("ticket_router", ticket_router)

workflow.add_edge(START, "ticket_classification")
workflow.add_edge("ticket_classification", "ticket_summarizer")
workflow.add_edge("ticket_summarizer", "ticket_router")
workflow.add_edge("ticket_router", END)

graph = workflow.compile()

# Setting up Streamlit UI
st.title("Healthcare Support Ticket Classifier")
st.write("""
         Upload a CSV file containing healthcare support tickets.
         This app will automatically:
         1. Classify each ticket into a single category.
         2. Generate a short summary about the ticket.
         3. Route to the appropriate department.
         4. Let you download the tagged results
         """)

uploaded_file = st.file_uploader("Upload CSV file (must have 'ticket_text' column)", type=["csv"])

if uploaded_file:
    # Read CSV without pandas
    try:
        file_content = uploaded_file.read().decode('utf-8')
        reader = csv.DictReader(file_content.splitlines())
        rows = list(reader)
        
        if not rows:
            st.error("CSV file is empty.")
        elif 'ticket_text' not in rows[0]:
            st.error("Uploaded file must contain a 'ticket_text' column.")
        else:
            st.info(f"There are {len(rows)} issues in the uploaded file.")

            if st.button("Run Classification"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, row in enumerate(rows):  
                    ticket_text = row.get("ticket_text", "").strip()
                    
                    if not ticket_text:
                        st.warning(f"Row {idx + 1} has empty ticket_text, skipping...")
                        continue
                    
                    input_state = {
                        "ticket_text": ticket_text,
                        "category": "",
                        "summary": "",
                        "department": ""
                    }
                    output = graph.invoke(input_state)
                    results.append(output)
                    
                    # Update progress
                    progress = (idx + 1) / len(rows)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing ticket {idx + 1} of {len(rows)}")

                st.success("Classification and summarization complete!")

                # Display results as table
                st.subheader("Results")
                for i, result in enumerate(results, 1):
                    st.write(f"**Ticket {i}:**")
                    st.write(f"- Category: {result['category']}")
                    st.write(f"- Summary: {result['summary']}")
                    st.write(f"- Department: {result['department']}")
                    st.divider()

                # Download as CSV
                if results:
                    output_csv = "ticket_text,category,summary,department\n"
                    for result in results:
                        ticket = result['ticket_text'].replace('"', '""')
                        category = result['category'].replace('"', '""')
                        summary = result['summary'].replace('"', '""')
                        department = result['department'].replace('"', '""')
                        output_csv += f'"{ticket}","{category}","{summary}","{department}"\n'
                    
                    st.download_button(
                        "Download Results", 
                        output_csv.encode('utf-8'), 
                        "classified_tickets.csv", 
                        "text/csv"
                    )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")