import os
import json
import boto3
import pandas as pd
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI

# Part 1 (Setup)
# Load environment variables (AWS credentials)
load_dotenv()

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Configuring Boto3 for retries
retry_config = Config(
    region_name=os.environ.get("AWS_DEFAULT_REGION"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

# Create a boto3 session for accessing Bedrock and Textract
session = boto3.Session()
client = session.client(service_name='bedrock-runtime',
                        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.environ.get(
                            "AWS_SECRET_ACCESS_KEY"),
                        config=retry_config)
textract = session.client('textract', config=retry_config)

# Function to upload the document
def upload_document(uploaded_file):
    """Save the uploaded document to the 'uploaded_files' folder and return its file path."""
    if uploaded_file is not None:
        try:
            # Create the folder if it doesn't exist
            upload_folder = "./uploaded_files"
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # Get the original file name and path
            file_path = os.path.join(upload_folder, uploaded_file.name)

            # Write the file to the folder (this will overwrite if it already exists)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return file_path
        except Exception as e:
            st.error(f"Error uploading file: {e}")
            return None
    return None


# Function to process the document (Extract text using Textract and split it)
def process_document(file_path, file_type):
    """Extract text from the document (TIFF, PDF, JPG, JPEG) and split it into manageable chunks."""

    # Extract text from TIFF or image (JPG, JPEG)
    def extract_text_from_image(file_path):
        try:
            with open(file_path, 'rb') as document:
                response = textract.detect_document_text(
                    Document={'Bytes': document.read()})

            text = ""
            for item in response["Blocks"]:
                if item["BlockType"] == "LINE":
                    text += item["Text"] + "\n"
            return text
        except ClientError as e:
            st.error(
                f"Amazon Textract error: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            return None

    # Extract text from PDF using Textract's `analyze_document` API
    def extract_text_from_pdf(file_path):
        try:
            loader = AmazonTextractPDFLoader(file_path)
            response = loader.load()
            response = response[0].page_content
            return response
        except ClientError as e:
            st.error(
                f"Amazon Textract error: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return None

    # Process the file based on its type
    if file_path and file_type:
        if file_type in ["jpg", "jpeg", "tiff"]:
            extracted_text = extract_text_from_image(file_path)
        elif file_type == "pdf":
            extracted_text = extract_text_from_pdf(file_path)
        else:
            st.error("Unsupported file format.")
            return []

        # If text was successfully extracted, split it into chunks
        if extracted_text:
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                    chunk_overlap=0
                )
                texts = text_splitter.split_text(extracted_text)
                return texts
            except Exception as e:
                st.error(f"Error splitting text into chunks: {e}")
                return []
        else:
            st.error("Failed to extract text from the document.")
            return []
    else:
        st.error("Invalid file path or file type.")
        return []


# Function to extract data based on selected model
def extract_data(document_data, selected_model, input_prompt, json_data):
    def get_llm_response(model, pdf_data, json_template, processed_data=""):
        prompt = PromptTemplate(
            template=input_prompt,
            input_variables=["json_template", "pdf_data", "processed_data"],
        )

        # Chain execution
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({
            "json_template": json_template,
            "pdf_data": pdf_data,
            "processed_data": processed_data,
        })

        return response

    def extract_all_data(pdf_data, json_template):
        extracted_data = ""
        iteration_count = 0
        max_iterations = 3  # Set a limit for max attempts to avoid infinite loops

        model_name = selected_model.get("model")
        model = ""
        if selected_model.get("name") == "OpenAI":
            model = ChatOpenAI(model=model_name)
        elif selected_model.get("name") == "Claude":
            model = ChatAnthropic(
                model=model_name, temperature=0, max_tokens=1024, timeout=None, max_retries=2)
        elif selected_model.get("name") == "Self-Hosted LLM":
            model_url = "https://expert-eft-innocent.ngrok-free.app/v1"
            model = ChatOpenAI(model=model_name, base_url=model_url)
        elif selected_model.get("name") == "Mistral":
            model = ChatMistralAI(
                model=model_name, temperature=0, max_retries=2)

        while iteration_count < max_iterations:
            iteration_count += 1
            # print(f"LLM CALL COUNT: {iteration_count}")

            # Call LLM with the full document but with processed data to help it continue
            response = get_llm_response(
                model=model,
                pdf_data=pdf_data,
                json_template=json_template,
                processed_data=extracted_data  # Pass previously extracted data
            )

            # If no new data is found, stop
            if not response:
                print("No new data found. Stopping extraction.")
                break

            # Append new response to extracted data
            extracted_data += response

        return extracted_data

    # Usage example
    result = extract_all_data(pdf_data=document_data, json_template=json_data)
    return result


# Function to generate Excel file from extracted data 
def generate_output_file(result, json_data):
    # Split the string into individual records
    records = result.split('\n\n')

    # Convert each record into a dictionary
    json_list = []
    for record in records:
        entry = {}
        lines = record.split('\n')
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                entry[key.strip()] = value.strip()
        json_list.append(entry)

    # Convert the list of dictionaries to JSON format
    json_output = json.dumps(json_list, indent=4)

    entries = json.loads(json_output)

    # Remove duplicates and incomplete entries
    unique_entries = []
    seen_entries = set()

    required_keys = json_data  # Assuming json_data is a list of required keys

    for entry in entries:
        # Check if all required keys are present
        if all(key in entry for key in required_keys):
            entry_tuple = tuple(entry.items())
            if entry_tuple not in seen_entries:
                seen_entries.add(entry_tuple)
                unique_entries.append(entry)

    # Create a DataFrame from unique entries
    df = pd.DataFrame(unique_entries)

    # Display the DataFrame in Streamlit
    st.subheader("Extracted Billing Data")
    st.markdown(
        """
        <style>
        [data-testid=stElementToolbarButton]:first-of-type {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(df,)  # Display DataFrame in tabular format
    # st.table(df)

    # Generate Excel file and provide download link
    output_file = "billing_data.xlsx"
    df.to_excel(output_file, index=False)

    # Add a download button for the Excel file
    st.download_button(
        label="Download Excel File",
        data=open(output_file, 'rb').read(),
        file_name=output_file,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# Function to check login credentials
def check_login(username, password):
    # You can replace this with a more secure method of handling passwords
    # Example credentials
    return username == os.getenv("LOGIN_USERNAME") and password == os.getenv("LOGIN_PASSWORD")


# Check if the user is logged in
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login Screen
if not st.session_state.logged_in:
    st.subheader("Login")

    # Create a form for username and password
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Login successful!")
                # Rerun to show the app after successful login
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")
else:
    st.set_page_config(layout="wide")
    # If logged in, show the main app
    st.title("Bill Document Extractor")

    # Step 1: Enter your prompt
    prompt = st.text_area("Enter your custom prompt", value="")

    output_columns = st.text_area(
        "Enter your JSON Object (required for Columns)", value="")

    # Step 2: Upload the document
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, TIFF, JPG, JPEG)", type=["pdf", "tiff", "jpg", "jpeg"])

    # Model selection dropdown
    model_options = [
        {"name": "Self-Hosted LLM",
            "model": "qwen2.5-coder-7b-instruct", "status": "Free"},
        {"name": "Mistral", "model": "open-mistral-nemo", "status": "Free"},
        {"name": "Mistral", "model": "mistral-small-latest", "status": "Paid"},
        {"name": "OpenAI", "model": "gpt-4o-mini", "status": "Paid"},
        {"name": "Claude", "model": "claude-3-5-sonnet-20240620", "status": "Paid"},
        {"name": "Claude", "model": "claude-3-haiku-20240307", "status": "Paid"},
    ]

    model_dropdown_options = [
        f"{option['name']} - ({option['model']}) - {option['status']}" for option in model_options
    ]

    selected_model_text = st.selectbox(
        "Select a model to use", model_dropdown_options)

    # Find the selected model's dictionary from model_options
    selected_model = next(
        (model for model in model_options if f"{model['name']} - ({model['model']}) - {model['status']}" == selected_model_text), None)

    # Submit button to trigger processing
    submit_button = st.button("Submit", disabled=(
        uploaded_file is None or selected_model is None))

    if submit_button:
        if uploaded_file is not None and selected_model is not None:
            # Step 3: Process the document
            file_type = uploaded_file.name.split(".")[-1].lower()
            file_path = upload_document(uploaded_file)

            if file_path:
                st.write(f"Processing document: {uploaded_file.name}")
                texts = process_document(file_path, file_type)

                if texts:
                    json_data = json.loads(output_columns)

                    # Step 4: Generate summary
                    st.write(
                        f"Extracting data using {selected_model_text}...")
                    summary = extract_data(
                        texts, selected_model, prompt, json_data)

                    if summary:
                        generate_output_file(summary, json_data)
                    else:
                        st.error("Failed to extract data from document.")
                else:
                    st.error("No text extracted from the document.")
        else:
            st.warning(
                "Please upload a document and select a model before submitting.")
    else:
        st.info("Please upload a document to proceed.")
