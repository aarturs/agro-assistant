import os, json, time
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from supabase import create_client
from bs4 import BeautifulSoup

load_dotenv(Path(__file__).parent.parent / "backend" / ".env")

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

print("Loading embedding model...")
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def parse_vaad_html():
    html_path = Path("vaad.html")
    if not html_path.exists():
        print("vaad.html not found. Copy it with:")
        print("cp ~/Desktop/Augu\\ auzsardzibas\\ lidzeklu\\ saraksts.html ~/projects/agro-assistant/scripts/vaad.html")
        exit(1)

    print("Parsing vaad.html...")
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    headers = [td.get_text(strip=True) for td in rows[0].find_all("td")]
    print(f"Columns: {headers}")

    records = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if cells and len(cells) >= 3:
            records.append({
                "product_name": cells[0] if len(cells) > 0 else "",
                "category":     cells[1] if len(cells) > 1 else "",
                "owner":        cells[2] if len(cells) > 2 else "",
                "reg_number":   cells[3] if len(cells) > 3 else "",
                "reg_class":    cells[4] if len(cells) > 4 else "",
                "active_subst": cells[5] if len(cells) > 5 else "",
                "valid":        cells[6] if len(cells) > 6 else "",
                "notes":        cells[7] if len(cells) > 7 else "",
            })

    print(f"Parsed {len(records)} products")
    return records

def build_chunk(p):
    return (
        "Preparats: " + p.get("product_name", "") + ". "
        "Kategorija: " + p.get("category", "") + ". "
        "Ipasnieks: " + p.get("owner", "") + ". "
        "Registracijas numurs: " + p.get("reg_number", "") + ". "
        "Darbiga viela: " + p.get("active_subst", "") + ". "
        "Derigs: " + p.get("valid", "") + ". "
        "Piez: " + p.get("notes", "") + "."
    )

def ingest():
    records = parse_vaad_html()
    if not records:
        print("No records found.")
        return

    # Delete cache so re-runs use fresh HTML
    Path("vaad_scraped.json").unlink(missing_ok=True)

    print(f"Embedding and uploading {len(records)} products...")
    batch, errors, count = [], 0, 0

    for i, p in enumerate(records):
        try:
            chunk = build_chunk(p)
            if len(chunk) < 20:
                continue
            embedding = model.encode(
                "search_document: " + chunk, normalize_embeddings=True
            ).tolist()
            reg = p.get("reg_number") or "VAAD_" + str(i)
            batch.append({
                "product_name":   p.get("product_name", "Nav")[:200],
                "active_subst":   p.get("active_subst", "")[:200],
                "culture":        p.get("category", ""),
                "pest_disease":   "",
                "dose":           "",
                "quarantine_days": None,
                "reg_number":     reg,
                "content":        chunk,
                "embedding":      embedding,
            })
            count += 1
            if len(batch) >= 50:
                supabase.table("vaad_products").upsert(
                    batch, on_conflict="reg_number"
                ).execute()
                print(f"  Upserted {count}/{len(records)}...")
                batch = []
                time.sleep(0.2)
        except Exception as e:
            errors += 1
            print(f"  Error row {i}: {e}")

    if batch:
        supabase.table("vaad_products").upsert(
            batch, on_conflict="reg_number"
        ).execute()

    print(f"Done! {count} products ingested, {errors} errors.")

if __name__ == "__main__":
    ingest()