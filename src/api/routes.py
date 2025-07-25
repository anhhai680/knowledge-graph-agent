from fastapi import FastAPI


app = FastAPI(
    title="Knowledge Graph Agent API",
    description="API for the Knowledge Graph Agent, providing endpoints to interact with the knowledge graph.",
    version="1.0.0",
)

@app.get("/")
def index():
    return {"message": "Welcome to the Knowledge Graph Agent API!"}