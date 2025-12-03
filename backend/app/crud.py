from sqlalchemy.orm import Session
from . import models, schemas, auth
from sqlalchemy import func

def create_user(db: Session, user_in: schemas.UserCreate, is_admin: bool = False):
    hashed = auth.get_password_hash(user_in.password)
    db_user = models.User(username=user_in.username, email=user_in.email, password_hash=hashed, is_admin=is_admin)
    db.add(db_user); db.commit(); db.refresh(db_user)
    return db_user

def get_user_by_username(db: Session, username: str):
    return db.query(models.User).filter(models.User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(models.User).filter(models.User.email == email).first()

def create_review(db: Session, user_id: int, review_in: schemas.ReviewCreate, sentiment: str, score: float):
    db_review = models.Review(user_id=user_id, summary=review_in.summary, sentiment=sentiment, score=score, category=review_in.category)
    db.add(db_review); db.commit(); db.refresh(db_review)
    return db_review

def get_recent_reviews(db: Session, limit:int=100):
    return db.query(models.Review).order_by(models.Review.timestamp.desc()).limit(limit).all()

def get_analytics(db: Session):
    q = db.query(models.Review.sentiment, func.count(models.Review.id)).group_by(models.Review.sentiment).all()
    return {sent or "unknown": cnt for sent, cnt in q}
