
# AI Ticket Triage System (LangGraph)

**Deployed on Streamlit:** [View Live App](https://generativeai-projects-qtxhjklkg8x3jikesckwz9.streamlit.app/)

**AI Ticket Triage System** is an intelligent automation application built using **LangGraph**, **Groq**, and **Streamlit**.  
It classifies healthcare support tickets, summarizes them, and automatically routes each ticket to the most relevant department.  

This project demonstrates how **Large Language Models (LLMs)** and **graph-based orchestration** can automate multi-step reasoning tasks — such as classification, summarization, and decision routing — in a single cohesive pipeline.

---

## Features

- Upload a CSV file containing healthcare support tickets.
- Automatically:
  - Classify each ticket into categories (e.g., Product Issue, Billing Issue, Login Issue, etc.).
  - Summarize ticket content in under 50 words.
  - Route each ticket to the most appropriate department.
- Download the processed results as a CSV file.
- Built with **LangGraph** for dynamic task orchestration and state management.
- Uses **Groq’s LLaMA 3.1 8B Instant model** for fast and accurate reasoning.

---

## Tech Stack

| Component | Description |
|------------|-------------|
| **Streamlit** | Interactive user interface for the application |
| **LangGraph** | Graph-based LLM workflow orchestration |
| **LangChain Groq** | LLM integration using Groq’s API |
| **Pandas** | Data handling and CSV processing |
| **Python** | Core programming language |
| **LLaMA 3.1 8B Instant** | Model used for text classification, summarization, and routing |

---

## How It Works

1. **Ticket Upload**  
   - User uploads a CSV file containing a column named `ticket_text`.

2. **Graph Pipeline (LangGraph)**  
   The workflow is composed of three nodes:
   - **Ticket Classification Node:** Predicts issue category.  
   - **Ticket Summarizer Node:** Summarizes the ticket in under 50 words.  
   - **Ticket Router Node:** Assigns a department based on the category.

3. **LLM Processing**  
   - Each node uses **Groq’s LLaMA 3.1 8B model** to execute tasks in sequence.  
   - The state transitions are handled automatically by **LangGraph**.

4. **Result Generation**  
   - The app displays classified and summarized tickets in a Streamlit table.  
   - Users can download the processed CSV for reporting or automation.

---

## Usage
1. Open the Streamlit app.
2. Upload a CSV file containing a ticket_text column.
3. Click on Run Classification
4. The app will:
  - Classify each ticket into a category.
  - Summarize the content.
  - Route it to the appropriate department.
5. Review the processed data and download the results  as  a CSV.

---




