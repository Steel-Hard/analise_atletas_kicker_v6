from app.core.database import get_database
from app.models.domain import Atleta
from typing import List, Optional

class AtletaRepository:
    def __init__(self):
        self.collection_name = "atletas"

    def get_collection(self):
        db = get_database()
        return db[self.collection_name]

    async def create_or_update(self, atleta: Atleta):
        collection = self.get_collection()
        await collection.update_one(
            {"athlete_id": atleta.athlete_id},
            {"$set": atleta.model_dump()},
            upsert=True
        )
        return atleta

    async def get_by_id(self, athlete_id: str) -> Optional[Atleta]:
        collection = self.get_collection()
        doc = await collection.find_one({"athlete_id": athlete_id})
        if doc:
            return Atleta(**doc)
        return None

    async def get_all(self) -> List[Atleta]:
        collection = self.get_collection()
        cursor = collection.find({})
        atletas = []
        async for doc in cursor:
            atletas.append(Atleta(**doc))
        return atletas
