# api/_helpers.py
from typing import Optional
from urllib.parse import urlparse
from fastapi import Request

def normalize_gender(g: Optional[str]) -> Optional[str]:
    if not g:
        return None
    g = g.strip().lower()
    if g in ("men", "male", "m"):
        return "Men"
    if g in ("women", "female", "w", "f"):
        return "Women"
    return g.title()

def normalize_category(c: Optional[str]) -> Optional[str]:
    """
    Map UI categories -> DB masterCategory.
    UI: garments, jewelry/jewelery/jewellery
    DB: Apparel, Accessories, Footwear, ...
    """
    if not c:
        return None
    c = c.strip().lower()
    mapping = {
        "garments": "Apparel",
        "garment": "Apparel",
        "apparel": "Apparel",
        "jewelry": "Accessories",
        "jewelery": "Accessories",   # common typo
        "jewellery": "Accessories",  # UK spelling
        "accessories": "Accessories",
        "footwear": "Footwear",
    }
    return mapping.get(c, c.title())

def img_url_from_product(request: Request, p) -> str:
    """
    Build absolute image URL.
    If Product.image_path is absolute -> return it.
    If relative -> mount under /images.
    Else fallback to /images/{id}.jpg
    """
    base = str(request.base_url).rstrip("/")
    path = getattr(p, "image_path", None)
    if path:
        s = path.strip().replace("\\", "/")
        try:
            parsed = urlparse(s)
            if parsed.scheme in ("http", "https"):
                return s
        except Exception:
            pass
        if s.startswith("/images/"):
            return f"{base}{s}"
        if s.startswith("images/"):
            return f"{base}/{s}"
        return f"{base}/images/{s}"
    return f"{base}/images/{p.id}.jpg"
