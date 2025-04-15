import hashlib

def to_int_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)

def hash_to_digest(i: int) -> str:
    return hex(abs(i))[2:].zfill(6)[:6]