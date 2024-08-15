from langchain.document_loaders import PyPDFLoader # type: ignore
from langchain.vectorstores import FAISS # type: ignore
from langchain import HuggingFaceHub # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings # type: ignore
# from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader # type: ignore
# from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate # type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI # type: ignore
from langchain.chains.question_answering import load_qa_chain # type: ignore
# from docx import Document
# from docx2pdf import convert
import streamlit as st # type: ignore
# import base64
# import PyPDF2
# from reportlab.pdfgen import canvas
import os
import re
import json


import os
os.environ['GOOGLE_API_KEY']='AIzaSyD7A8yvV81-4y6JGjyqjyMQnFO_CC6H_cY'

st.title("\nPERSONALISED LEARNING SYSTEM")


uploaded_main_document = st.file_uploader("Upload the Book", type=["pdf"],)
main_pdf_folder_name = "main_book_files"
os.makedirs(main_pdf_folder_name, exist_ok=True)
if uploaded_main_document:
    main_document_path = ""
    with open(os.path.join(main_pdf_folder_name, uploaded_main_document.name), "wb") as f:
        f.write(uploaded_main_document.read())
        main_document_path = os.path.join(main_pdf_folder_name,uploaded_main_document.name)
        
         
def pdf_embedding(book_path):
    text=""
    pdf_reader= PdfReader(book_path)
    for page in pdf_reader.pages:
        text+= page.extract_text()
    print("pdf loaded")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print("pdf splited")

    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_db = FAISS.from_texts(chunks, embedding=embeddings)
    print("vector db created")
    return vector_db


if uploaded_main_document and "book_db" not in st.session_state:
    st.session_state.book_db = pdf_embedding(main_document_path)
    print("book is embedded")



if "llm_model" not in st.session_state:
    st.session_state.llm_model = ChatGoogleGenerativeAI(model="gemini-pro",
                                                        temperature=0.1)
    print("model is loaded")


def chains(prompt_template):
   
        qa_chain = load_qa_chain(st.session_state.llm_model, 
                                                  chain_type="stuff",
                                                  prompt = prompt_template)
        return qa_chain
            


if uploaded_main_document:
    tab1, tab2, tab3, tab4 = st.tabs(["Question Answer","Syllabus summary ", "Quiz", "Answer Key Generation"])

    with tab1:
        st.header("Question Answer")
        prompt_template_qa =PromptTemplate(template = """ You are a content retrieval bot from the given context. You will be given with a context and your task is to retrive contents relevant to the Questions. 
                                                          context :\n{context}?\n
                                                          question:\n{question}\n
                                                          Answer:  """, 
                                                          input_variables=["context", "question"])

        if uploaded_main_document:
            question = st.text_input('Enter your question')
            if question:
                docs = st.session_state.book_db.similarity_search(question)
                if question and st.button('Generate'):
                        chain = chains(prompt_template_qa)
                        answer = chain({"input_documents":docs, "question": question})
                        st.write(answer["output_text"])


    with tab2:
        st.header("Syllabus summary ")
        uploaded_syllabus_document = st.file_uploader("Upload your Syllabus PDF", type="pdf")
        syllabus_pdf_folder_name = "Syllabus_book_files"
        os.makedirs(syllabus_pdf_folder_name, exist_ok=True)

        if uploaded_syllabus_document:
            syllabus_document_path = ""
            with open(os.path.join(syllabus_pdf_folder_name, uploaded_syllabus_document.name), "wb") as f:
                f.write(uploaded_syllabus_document.read())
                syllabus_document_path = os.path.join(syllabus_pdf_folder_name,uploaded_syllabus_document.name)

        def syllabus_subtopic(pdf_path):
            pdf = PyPDFLoader(pdf_path)
            syllabus_docs = pdf.load()
            syllabus_doc = syllabus_docs[0].page_content
            pattern = r'[-,:]'
            subtopics = re.split(pattern, syllabus_doc)
            subtopics = [topic.strip() for topic in subtopics if topic.strip()]
            subtopics_lst = []
            for i, subtopic in enumerate(subtopics, start=1):
                subtopics_lst.append(subtopic)
            return subtopics_lst
        
        if uploaded_syllabus_document:
            subtopics = syllabus_subtopic(syllabus_document_path)
            
            if subtopics:
                if st.button('Generate '):
                    prompt_template_qa =PromptTemplate(template = """ You are a content retrieval bot from the given context. You will be given with a context and your task is to retrive contents relevant to the question. And if you don't know the answer try to generate possible answer from the context. 
                                                                      context :\n{context}?\n
                                                                      question:\n{question}\n
                                                                      Answer: If you don't know the answer, try to answer on your own """, 
                                                                      input_variables=["context", "question"])

                    result = ""
                    for subtopic in subtopics:
                        if subtopic != "Healthcare fraud Detection":
                            answer = st.session_state.book_db.similarity_search(subtopic)
                            chain = chains(prompt_template_qa)
                            answer = chain({"input_documents":answer, "question": subtopic})
                            st.write("Subtopic : ", subtopic )
                            st.write("Answer : ",answer["output_text"])
                            st.write("--------------------------------------------------------------------------------------------------------- ")


    with tab3:
        st.header("Quiz")

       
        if "quiz_json" not in st.session_state:
           
            if st.button("Generate Quiz"):
                
                prompt_template = """ Your task is to generate 10 questions from the Context.You will be the given with context and you have to generate multiple choice questions with options and correct answers
                                    Question:{question}
                                    Context: {context}
                                    Answer:   Generate the answer in JSON format with question, options and correct answer in it. """
                
                prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

                if "quiz_chain" not in st.session_state:
                    st.session_state.quiz_chain = load_qa_chain(st.session_state.llm_model, chain_type="stuff", prompt=prompt)
                    print("quiz chain created")
                if "answer" not in st.session_state:
                    st.session_state.answer = st.session_state.book_db.similarity_search("generate any 10 questions")
                if "quiz_response" not in st.session_state:
                    st.session_state.quiz_response = st.session_state.quiz_chain({"input_documents": st.session_state.answer, "question": "generate any 10 questions"}, return_only_outputs=True)
                if "text" not in st.session_state:
                    st.session_state.text = st.session_state.quiz_response["output_text"]
                
                print("quiz response created")

                
                st.session_state.quiz_json = st.session_state.text

        def create_quiz_in_ui(quiz_json):
            if quiz_json:
                score = 0
                quiz_data = json.loads(quiz_json)
                num_questions = len(quiz_data)
                for question_id, question_data in quiz_data.items():
                    st.write(f"**Question {question_id}:** {question_data['question']}")
                    st.write("Select an option:", [qn for qn in question_data['options']])
                    user_answer = st.text_input(f"Enter your answer for question {question_id} from the above option")
                    st.write("")
                    
                    if user_answer:
                        correct_answer = question_data['correct_answer']
                        print("Correct answer:", correct_answer)
                        print("User's answer:", user_answer)
                        if user_answer.lower() == correct_answer.lower():
                            st.write("**Correct!**")
                            score += 1
                        else:
                            st.write(f"**Incorrect!** The correct answer is: {correct_answer}")
                    st.write("----------------------------------------------")

                
                st.session_state.quiz_score = score
                st.write(f"**Your Score:** {score}/{num_questions}")
            else:
                st.write("No quiz data available. Please generate the quiz.")

        
        if "quiz_json" in st.session_state:
            create_quiz_in_ui(st.session_state.quiz_json)

    
    if "quiz_score" in st.session_state:
        st.write(f"Your Final Score: {st.session_state.quiz_score}")


    with tab4:
        st.header("Answer Key Generator")
        uploaded_QP_document = st.file_uploader("Upload your Question paper PDF", type="pdf")
        QP_pdf_folder_name = "Question_paper_files"
        os.makedirs(QP_pdf_folder_name, exist_ok=True)

        if uploaded_QP_document:
            QP_document_path = ""
            with open(os.path.join(QP_pdf_folder_name, uploaded_QP_document.name), "wb") as f:
                f.write(uploaded_QP_document.read())
                QP_document_path = os.path.join(QP_pdf_folder_name,uploaded_QP_document.name)

            if "QP_db" not in st.session_state:
                st.session_state.QP_db = pdf_embedding(QP_document_path)
                print("question paper is embedded")

        prompt_template_QP = PromptTemplate(template = """You are a content retrieval bot from the given context. You will be given with a context and your task is to retrive contents relevant to the Questions.
                                                context:\n{context}?\n
                                                cuestion:\n{question}\n
                                                Answer: """,
                                                input_variables=["context", "question"])

        if uploaded_QP_document:
            question = "Extract all the questions with its options from the question paper and display each part in list wise"
            if question:
                qp_docs = st.session_state.QP_db.similarity_search(question)
                chain = chains(prompt_template_QP)
                answer = chain({"input_documents":qp_docs, "question": question})
                
                QP_lists = answer["output_text"].split("**PART - ")
                List_1m = QP_lists[1].split("\n\n")[1:-1]
                List_3m = QP_lists[2].split("\n")[1:-1]
                List_15m = QP_lists[3].split("\n")[1:-1]


                if st.button("Generate answer key"):
                    
                    book_docs = st.session_state.book_db.similarity_search(question)
                    st.subheader("1 Marks")
                    for one_mark in List_1m:
                        if len(one_mark) > 10:
                            prompt_template_1m = PromptTemplate(template = """You will be given with a question paper. Your task is to answer one mark questions within the given option. Provide accurate answers\n\n
                                                                            Context:\n{context}?\n
                                                                            Question:\n{question}\n
                                                                            Answer: """,
                                                                            input_variables=["context", "question"])

                            chain = chains(prompt_template_1m)
                            answer = chain({"input_documents":book_docs, "question": one_mark})
                            st.write("Question : ", one_mark)
                            st.write("Answer : ",answer["output_text"])
                            st.write("")
                    st.write("------------------------------------------------")
                    st.subheader("3 Marks")
                    for three_mark in List_3m:
                        if len(three_mark) > 10:
                            prompt_template_3m = PromptTemplate(template = """You will be given with a question paper. Your task is to answer three mark questions in range of 50 words. Provide accurate answers\n\n
                                                                            Context:\n{context}?\n
                                                                            Question:\n{question}\n
                                                                            Answer: """,
                                                                            input_variables=["context", "question"])

                            chain = chains(prompt_template_3m)
                            answer = chain({"input_documents":book_docs, "question": three_mark})
                            st.write("Question : ", three_mark)
                            st.write("Answer : ",answer["output_text"])
                            st.write("")

                    st.subheader("15 Marks")
                    for fifteen_mark in List_15m:
                        if len(fifteen_mark) >10:
                            prompt_template_15m = PromptTemplate(template = """You will be given with a question paper. Your task is to answer 15 mark questions in range of 200 words. Provide accurate answers\n\n
                                                                            Context:\n{context}?\n
                                                                            Question:\n{question}\n
                                                                            Answer: """,
                                                                            input_variables=["context", "question"])

                            chain = chains(prompt_template_15m)
                            answer = chain({"input_documents":book_docs, "question": fifteen_mark})
                            st.write("Question : ", fifteen_mark)
                            st.write("Answer : ",answer["output_text"])
                            st.write("")
