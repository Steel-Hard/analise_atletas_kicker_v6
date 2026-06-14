from app.core.database import get_database
from app.models.domain import SessaoFisica
from typing import List

class SessaoRepository:
    def __init__(self):
        self.collection_name = "sessoes"

    def get_collection(self):
        db = get_database()
        return db[self.collection_name]

    async def insert_many(self, sessoes: List[SessaoFisica]):
        if not sessoes:
            return
        collection = self.get_collection()
        docs = [s.model_dump() for s in sessoes]
        await collection.insert_many(docs)

    async def delete_by_athlete(self, athlete_id: str):
        collection = self.get_collection()
        await collection.delete_many({"athlete_id": athlete_id})

    async def get_by_athlete(self, athlete_id: str) -> List[SessaoFisica]:
        collection = self.get_collection()
        cursor = collection.find({"athlete_id": athlete_id}).sort("start_date", 1)
        sessoes = []
        async for doc in cursor:
            sessoes.append(SessaoFisica(**doc))
        return sessoes

    async def get_all(self) -> List[SessaoFisica]:
        collection = self.get_collection()
        cursor = collection.find({})
        sessoes = []
        async for doc in cursor:
            sessoes.append(SessaoFisica(**doc))
        return sessoes
