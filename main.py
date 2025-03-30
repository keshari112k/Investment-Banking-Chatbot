from fastapi import FastAPI
from pydantic import BaseModel
from src.process import qa_pipeline

# Assuming qa_pipeline is already defined in your script
# Example: qa_pipeline = SomePipelineModel()

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    response = qa_pipeline.invoke(request.query)
    return {"response": response["result"]}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




