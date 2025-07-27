from fastapi import APIRouter

router = APIRouter(prefix='/chat')

@router.get('/')
def test_chat_route():
    return {'message': 'get /chat ok !'}