from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.session import get_db
from app.db.models import User
from app.core.security import hash_password, verify_password, create_access_token


router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=72)

class LoginIn(BaseModel):
    email: EmailStr
    password: str

@router.post("/register")
def register_admin(payload: RegisterIn, db: Session = Depends(get_db)):

    try:
        pw_hash = hash_password(payload.password)
    except ValueError:
        # If any hashing algorithm throws (bcrypt, etc.), return clean message
        raise HTTPException(
            status_code=400,
            detail="Password is too long or contains unsupported characters. Try a shorter password."
        )

    user = User(
        name=payload.name,
        email=payload.email,
        password_hash=pw_hash,
        role="admin",
    )


    db.add(user)
    db.commit()

    return {"message": "Admin registered successfully. Please login."}


@router.post("/login")
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({
    "sub": user.email,
    "role": user.role,
    "name": user.name,
})

    return {"access_token": token, "token_type": "bearer", "role": user.role, "email": user.email}
