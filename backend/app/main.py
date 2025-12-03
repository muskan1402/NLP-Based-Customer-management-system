from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from . import models, database, crud, schemas, auth, inference
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

# dev CORS for local Streamlit and React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:8501","http://localhost:8502"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/api/register", response_model=schemas.UserOut)
def register(user_in: schemas.UserCreate, db: Session = Depends(get_db)):
    if crud.get_user_by_username(db, user_in.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    if crud.get_user_by_email(db, user_in.email):
        raise HTTPException(status_code=400, detail="Email already exists")
    user = crud.create_user(db, user_in, is_admin=False)
    return user

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

@app.post("/api/login", response_model=LoginResponse)
def login(form: schemas.UserLogin, db: Session = Depends(get_db)):
    user = crud.get_user_by_username(db, form.username)
    if not user or not auth.verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = auth.create_access_token({"sub": user.username})
    return {"access_token": token}

@app.post("/api/reviews", response_model=schemas.ReviewOut)
def post_review(review_in: schemas.ReviewCreate, request: Request, db: Session = Depends(get_db)):
    # read Authorization header
    auth_header = request.headers.get("authorization")
    user = None
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
        user = auth.get_user_from_token(token, db)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required to submit a review")
    sentiment, score = inference.predict(review_in.summary)
    review = crud.create_review(db, user.id, review_in, sentiment, score)
    return review

@app.get("/api/reviews")
def list_reviews(limit:int = 100, db: Session = Depends(get_db)):
    reviews = crud.get_recent_reviews(db, limit=limit)
    return reviews

@app.get("/api/admin/analytics")
def analytics(request: Request, db: Session = Depends(get_db)):
    # token-based admin check
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Admin token required")
    token = auth_header.split(" ", 1)[1]
    user = auth.get_user_from_token(token, db)
    if user is None or not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return crud.get_analytics(db)
