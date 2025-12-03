from pydantic import BaseModel, EmailStr
from typing import Optional
import datetime

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    is_admin: bool
    class Config:
        orm_mode = True

class UserLogin(BaseModel):
    username: str
    password: str

class ReviewCreate(BaseModel):
    summary: str
    category: Optional[str] = None

class ReviewOut(BaseModel):
    id: int
    user_id: Optional[int]
    summary: str
    sentiment: Optional[str]
    score: Optional[float]
    timestamp: datetime.datetime
    class Config:
        orm_mode = True
