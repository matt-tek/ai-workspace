from dotenv import load_dotenv
from fastapi import FastAPI
from contextlib import asynccontextmanager
from database.init_db import create_db_and_tables
from routers.chat import router as chat_router

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)

@app.get('/')
def test_app():
    return {'message': 'app is working'}
