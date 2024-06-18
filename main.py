import uvicorn
from fastapi import FastAPI, Query
import csv
import json
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_transformers import LongContextReorder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
import jieba

app = FastAPI()

# Load and split documents
loader = CSVLoader(file_path='100-112年國科會計畫資料csv.csv', encoding='utf-8-sig')
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = loader.load_and_split(splitter)

# Initialize vector embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#vectordb = Chroma.from_documents(docs, embeddings, persist_directory = '110to112')
vectordb = Chroma(persist_directory='110to112', embedding_function=embeddings)

# jieba tokenizer
def cut_words(text):
    return jieba.lcut_for_search(text)

# Initialize retrievers
bm25_params = { 
    "k1": 1.2,
    "b": 0.75
}
bm25_retriever = BM25Retriever.from_documents(docs, bm25_params=bm25_params, preprocess_func = cut_words, k = 12)
base_retriever = vectordb.as_retriever(search_kwargs={"k": 12}, search_type="mmr") 
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, base_retriever], weights=[0.32, 0.68],)

def fetch_row_with_headers(file_path, row_number):
    """Fetch data for a specific row from a CSV file."""
    required_keys = {
        'id': 'ID', 'year': 'YEAR', 'projectName': 'PROJECTNAME', 'projectStatus': 'PROJECTSTATUS',
        'startDate': None, 'draftDescription': 'DRAFTDESCRIPTION', 'projectDuration': 'PROJECTDURATION',
        'instructor': 'INSTRUCTOR', 'endDate': None, 'chineseAbstract': 'CHINESEABSTRACT',
        'englishAbstract': 'ENGLISHABSTRACT', 'institute': 'INSTITUTE', 'chineseKeywords': 'CHINESEKEYWORDS',
        'englishKeywords': 'ENGLISHKEYWORDS', 'projectFunding': 'PROJECTFUNDING'
    }
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for idx, row in enumerate(reader, start=0):
            if idx == row_number:
                row_data = {key: row.get(val) if val else None for key, val in required_keys.items()}
                if row.get('PROJECTDURATION'):
                    duration_parts = row['PROJECTDURATION'].split('~')
                    if len(duration_parts) == 2:
                        row_data['startDate'], row_data['endDate'] = duration_parts
                if row.get('PROJECTFUNDING'):
                    # Remove commas and currency symbols, then convert to int
                    funding_str = row['PROJECTFUNDING'].replace(',', '').replace('元', '')
                    row_data['projectFunding'] = int(funding_str)
                return row_data
    return {key: None for key in required_keys}

def process_documents_to_json(relevant_docs):
    """Process documents and convert them to JSON format."""
    all_data = []
    for doc in relevant_docs:
        row_data = fetch_row_with_headers(doc.metadata['source'], doc.metadata['row'])
        if row_data:
            all_data.append(row_data)
    return json.dumps(all_data, ensure_ascii=False, indent=4)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/get_research_data")
async def get_research_data(keyword: str = Query(default=None, title="Search Keyword", description="Enter the keyword to search research projects")):
    if keyword:
        relevant_docs = ensemble_retriever.get_relevant_documents(keyword)  # Perform search based on user input
        #print(bm25_retriever.get_relevant_documents(keyword) )
        json_output = process_documents_to_json(relevant_docs)

        return json.loads(json_output) 

    return {"error": "Keyword is required for searching."}


@app.get("/analyze_research_data")
async def analyze_research_data(keyword: str = Query(default=None, title="Analyze Keyword", description="Enter the keyword to analyze research projects")):
    if keyword:
        relevant_docs = ensemble_retriever.get_relevant_documents(keyword)  # Perform search based on user input
        json_output = process_documents_to_json(relevant_docs)
    
        # Prepare the input for the retrieval chain
        inputs = {"context": json_output, "keyword": keyword}

        template = """你是一個推薦系統
我會給你10筆計畫和內容，根據這些內容推薦使用者合適的人選
並且簡單介紹這個人之前做過的計畫還有說明推薦原因。
請用json格式回傳，格式如下
"id": "string",
"name": "string",
"projectName": "string",
"institute": "string",
"reason": "string"
----------
{context}
----------
Question: 我想找可以做{keyword}的研究人員

只給我json即可，其他文字絕對不要有
"""

        prompt =  ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(model = 'gpt-4o')

        retrieval_chain = (
            {"context": RunnablePassthrough(), "keyword": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        # Generate the response using the model
        response = retrieval_chain.invoke(inputs)

        response = response.strip("```json\n")

        return json.loads(response)
    return {"error": "Keyword is required for analysis."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)
 