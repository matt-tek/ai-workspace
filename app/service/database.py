from database.init_db import SessionDep

class DatabaseService:
    def __init__(self):
        self.db = SessionDep

    def query(self, model, id):
        return self.get(model, id)

    def create(self, data):
        self.db.add(data)
        self.db.commit()
        self.db.refresh(data)
        return data
